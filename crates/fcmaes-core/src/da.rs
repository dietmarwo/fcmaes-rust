//! Dual Annealing — Rust port of the C++ `daoptimizer.cpp`.
//!
//! Generalized simulated annealing (derived from SciPy's `_dual_annealing`):
//! a distorted Cauchy-Lorentz visiting distribution, a Markov strategy chain
//! with generalized accept/reject, re-annealing, and an optional local search.
//! The C++ used LBFGSpp's L-BFGS-B for the local search; this port uses a
//! self-contained bounded limited-memory quasi-Newton (projected L-BFGS on the
//! `[0,1]` box with a finite-difference gradient), avoiding a Fortran
//! dependency. C++-only in the original (no pure-Python twin); validated by
//! convergence rather than a reference distribution.

use crate::fitness::Objective;
use crate::rng::Rng;

/// Outcome of a Dual Annealing run (mirrors the C++ `DaResult`).
#[derive(Clone, Debug)]
pub struct DaResult {
    pub x: Vec<f64>,
    pub y: f64,
    pub evaluations: u64,
    pub iterations: i32,
    pub stop: i32,
}

/// Tunable inputs for [`optimize_da`].
#[derive(Clone, Debug)]
pub struct DaParams {
    pub max_evaluations: u64,
    pub use_local_search: bool,
    pub seed: u64,
    pub runid: i64,
}

impl Default for DaParams {
    fn default() -> Self {
        Self {
            max_evaluations: 100_000,
            use_local_search: true,
            seed: 0,
            runid: 0,
        }
    }
}

const BIG_VALUE: f64 = 1e16;
const TAIL_LIMIT: f64 = 1e8;
const MIN_VISIT_BOUND: f64 = 1e-10;
const MAX_REINIT_COUNT: i32 = 1000;

const TEMPERATURE_START: f64 = 5230.0;
const QV: f64 = 2.62;
const QA: f64 = -5.0;
const MAXSTEPS: i32 = 1000;
const TEMPERATURE_RESTART: f64 = 0.1;

struct Da<'a, O: Objective> {
    obj: &'a O,
    dim: usize,
    has_bounds: bool,
    lower: Vec<f64>,
    scale: Vec<f64>,
    max_evals: u64,
    eval_counter: u64,
    use_local_search: bool,
    rng: Rng,

    // fitness best (reset per local search)
    fit_best_y: f64,
    fit_best_x: Vec<f64>,

    // visiting distribution factors
    factor4_p: f64,
    factor6: f64,

    // energy state
    ebest: f64,
    xbest: Vec<f64>,
    current_energy: f64,
    current_location: Vec<f64>,

    // strategy chain
    emin: f64,
    xmin: Vec<f64>,
    not_improved_idx: i32,
    not_improved_max_idx: i32,
    temperature_step: f64,
    k: f64,
    state_improved: bool,

    reinit_failed: bool,
}

impl<'a, O: Objective> Da<'a, O> {
    fn new(
        obj: &'a O,
        dim: usize,
        lower: Vec<f64>,
        upper: Vec<f64>,
        max_evals: u64,
        use_local_search: bool,
        seed: u64,
    ) -> Self {
        let has_bounds = !lower.is_empty();
        let scale: Vec<f64> = if has_bounds {
            upper.iter().zip(&lower).map(|(u, l)| u - l).collect()
        } else {
            vec![1.0; dim]
        };
        // visiting-distribution invariants
        let factor2 = ((4.0 - QV) * (QV - 1.0).ln()).exp();
        let factor3 = ((2.0 - QV) * 2.0_f64.ln() / (QV - 1.0)).exp();
        let factor4_p = std::f64::consts::PI.sqrt() * factor2 / (factor3 * (3.0 - QV));
        let factor5 = 1.0 / (QV - 1.0) - 0.5;
        let d1 = 2.0 - factor5;
        let factor6 = std::f64::consts::PI * (1.0 - factor5)
            / (std::f64::consts::PI * (1.0 - factor5)).sin()
            / libm::lgamma(d1).exp();
        Da {
            obj,
            dim,
            has_bounds,
            lower,
            scale,
            max_evals,
            eval_counter: 0,
            use_local_search,
            rng: Rng::new(seed),
            fit_best_y: f64::MAX,
            fit_best_x: vec![],
            factor4_p,
            factor6,
            ebest: f64::MAX,
            xbest: vec![],
            current_energy: f64::MAX,
            current_location: vec![],
            emin: f64::MAX,
            xmin: vec![],
            not_improved_idx: 0,
            not_improved_max_idx: 1000,
            temperature_step: 0.0,
            k: 100.0 * dim as f64,
            state_improved: false,
            reinit_failed: false,
        }
    }

    fn closest_feasible(&self, x: &[f64]) -> Vec<f64> {
        if self.has_bounds {
            x.iter().map(|&v| v.clamp(-1.0, 1.0)).collect()
        } else {
            x.to_vec()
        }
    }

    fn encode(&self, x: &[f64]) -> Vec<f64> {
        if self.has_bounds {
            (0..self.dim)
                .map(|i| (x[i] - self.lower[i]) / self.scale[i])
                .collect()
        } else {
            x.to_vec()
        }
    }

    fn decode(&self, x: &[f64]) -> Vec<f64> {
        if self.has_bounds {
            (0..self.dim)
                .map(|i| x[i] * self.scale[i] + self.lower[i])
                .collect()
        } else {
            x.to_vec()
        }
    }

    fn raw_eval(&mut self, x_decoded: &[f64]) -> f64 {
        self.eval_counter += 1;
        self.obj.eval_scalar(x_decoded)
    }

    /// Evaluate an *encoded* point, tracking the local-search best.
    fn value(&mut self, x: &[f64]) -> f64 {
        let res = if self.has_bounds {
            let feas = self.closest_feasible(x);
            let dec = self.decode(&feas);
            self.raw_eval(&dec)
        } else {
            self.raw_eval(x)
        };
        if res < self.fit_best_y {
            self.fit_best_y = res;
            self.fit_best_x = x.to_vec();
        }
        res
    }

    fn max_eval_reached(&self) -> bool {
        self.eval_counter >= self.max_evals
    }

    fn normal_vec(&mut self) -> Vec<f64> {
        (0..self.dim).map(|_| self.rng.gaussian()).collect()
    }
    fn uniform_vec(&mut self) -> Vec<f64> {
        (0..self.dim).map(|_| self.rng.uniform01()).collect()
    }

    // ---- visiting distribution ----

    fn visit_fn(&mut self, temperature: f64, n: usize) -> Vec<f64> {
        let x: Vec<f64> = (0..n).map(|_| self.rng.gaussian()).collect();
        let y: Vec<f64> = (0..n).map(|_| self.rng.gaussian()).collect();
        let factor1 = (temperature.ln() / (QV - 1.0)).exp();
        let factor4 = self.factor4_p * factor1;
        let sigmax = (-(QV - 1.0) * (self.factor6 / factor4).ln() / (3.0 - QV)).exp();
        (0..n)
            .map(|i| {
                let xi = x[i] * sigmax;
                let den = ((y[i].abs() * (QV - 1.0)).ln() / (3.0 - QV)).exp();
                xi / den
            })
            .collect()
    }

    fn visiting(&mut self, x: &[f64], step: usize, temperature: f64) -> Vec<f64> {
        if step < self.dim {
            let upper_sample = self.rng.uniform01();
            let lower_sample = self.rng.uniform01();
            let mut visits = self.visit_fn(temperature, self.dim);
            for v in visits.iter_mut() {
                if *v > TAIL_LIMIT {
                    *v = TAIL_LIMIT * upper_sample;
                } else if *v < -TAIL_LIMIT {
                    *v = -TAIL_LIMIT * lower_sample;
                }
            }
            let mut x_visit: Vec<f64> = (0..self.dim).map(|i| visits[i] + x[i]).collect();
            for xv in x_visit.iter_mut() {
                let b = (*xv % 1.0) + 1.0;
                *xv = b % 1.0;
                if xv.abs() < MIN_VISIT_BOUND {
                    *xv += 1e-10;
                }
            }
            x_visit
        } else {
            let mut x_visit = x.to_vec();
            let mut visit = self.visit_fn(temperature, 1)[0];
            if visit > TAIL_LIMIT {
                visit = TAIL_LIMIT * self.rng.uniform01();
            } else if visit < -TAIL_LIMIT {
                visit = -TAIL_LIMIT * self.rng.uniform01();
            }
            let index = step - self.dim;
            x_visit[index] = visit + x[index];
            let b = (x_visit[index] % 1.0) + 1.0;
            x_visit[index] = b % 1.0;
            if x_visit[index].abs() < MIN_VISIT_BOUND {
                x_visit[index] += MIN_VISIT_BOUND;
            }
            x_visit
        }
    }

    // ---- energy state ----

    fn reset_energy(&mut self, x0: &[f64]) {
        self.current_location = if x0.is_empty() {
            self.normal_vec()
        } else {
            x0.to_vec()
        };
        let mut reinit_counter = 0;
        loop {
            self.current_energy = self.value(&self.current_location.clone());
            if self.current_energy >= BIG_VALUE || self.current_energy.is_nan() {
                if reinit_counter >= MAX_REINIT_COUNT {
                    self.reinit_failed = true;
                    return;
                }
                self.current_location = self.uniform_vec();
                reinit_counter += 1;
            } else {
                if self.ebest == f64::MAX && self.xbest.is_empty() {
                    self.ebest = self.current_energy;
                    self.xbest = self.current_location.clone();
                }
                return;
            }
        }
    }

    // ---- strategy chain ----

    fn accept_reject(&mut self, j: usize, e: f64, x_visit: &[f64]) {
        let r = self.rng.uniform01();
        let pqv_temp = (QA - 1.0) * (e - self.current_energy) / (self.temperature_step + 1.0);
        let pqv = if pqv_temp < 0.0 {
            0.0
        } else {
            (pqv_temp.ln() / (1.0 - QA)).exp()
        };
        if r <= pqv {
            self.current_energy = e;
            self.current_location = x_visit.to_vec();
            self.xmin = self.current_location.clone();
        }
        if self.not_improved_idx >= self.not_improved_max_idx
            && (j == 0 || self.current_energy < self.emin)
        {
            self.emin = self.current_energy;
            self.xmin = self.current_location.clone();
        }
    }

    fn run_chain(&mut self, step: usize, temperature: f64) {
        self.temperature_step = temperature / (step as f64 + 1.0);
        self.not_improved_idx += 1;
        let iters = self.current_location.len() * 2;
        for j in 0..iters {
            if j == 0 {
                self.state_improved = false;
            }
            if step == 0 && j == 0 {
                self.state_improved = true;
            }
            let x_visit = self.visiting(&self.current_location.clone(), j, temperature);
            let e = self.value(&x_visit);
            if e < self.current_energy {
                self.current_energy = e;
                self.current_location = x_visit.clone();
                if e < self.ebest {
                    self.ebest = e;
                    self.xbest = x_visit.clone();
                    self.state_improved = true;
                    self.not_improved_idx = 0;
                }
            } else {
                self.accept_reject(j, e, &x_visit);
            }
            if self.max_eval_reached() {
                return;
            }
        }
    }

    fn chain_local_search(&mut self) {
        if self.state_improved {
            let (e, x) = self.local_search(&self.xbest.clone());
            if e < self.ebest {
                self.not_improved_idx = 0;
                self.ebest = e;
                self.xbest = x.clone();
                self.current_energy = e;
                self.current_location = x;
                if self.max_eval_reached() {
                    return;
                }
            }
        }
        let mut do_ls = false;
        if self.k < 90.0 * self.dim as f64 {
            let pls = (self.k * (self.ebest - self.current_energy) / self.temperature_step).exp();
            if pls >= self.rng.uniform01() {
                do_ls = true;
            }
        }
        if self.not_improved_idx >= self.not_improved_max_idx {
            do_ls = true;
        }
        if do_ls {
            let (e, x) = self.local_search(&self.xmin.clone());
            self.xmin = x.clone();
            self.emin = e;
            self.not_improved_idx = 0;
            self.not_improved_max_idx = self.current_location.len() as i32;
            if e < self.ebest {
                self.ebest = e;
                self.xbest = x.clone();
                self.current_energy = e;
                self.current_location = x;
            }
        }
    }

    // ---- bounded L-BFGS local search on [0,1]^dim ----

    /// Finite-difference gradient matching the C++ `LBFGSFunc` (per-coordinate
    /// forward/backward difference with boundary handling); returns `(grad, f)`.
    fn fd_grad(&mut self, arg: &[f64]) -> (Vec<f64>, f64) {
        let eps = 1e-6;
        let mut grad = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut x1 = arg.to_vec();
            let mut x2 = arg.to_vec();
            let mut e1 = eps;
            let mut e2 = eps;
            x1[i] += eps;
            if x1[i] > 1.0 {
                x1[i] = 1.0;
                e1 = 1.0 - arg[i];
            }
            x2[i] -= eps;
            if x2[i] < 0.0 {
                x2[i] = 0.0;
                e2 = arg[i];
            }
            let f1 = self.value(&x1);
            let f2 = self.value(&x2);
            grad[i] = (f1 - f2) / (e1 + e2);
        }
        let f = self.value(arg);
        (grad, f)
    }

    fn local_search(&mut self, x0: &[f64]) -> (f64, Vec<f64>) {
        // reset the fitness-best tracker; the best point seen during the search
        // (through `value`) is the returned result — matching the C++ contract.
        self.fit_best_y = f64::MAX;
        let mut max_iter = (6 * self.dim) as i32;
        max_iter = max_iter.clamp(100, 1000);

        let clamp01 = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.clamp(0.0, 1.0)).collect() };
        let mut x = clamp01(&self.closest_feasible(x0));
        let m = 6usize;
        let mut s_hist: Vec<Vec<f64>> = Vec::new();
        let mut y_hist: Vec<Vec<f64>> = Vec::new();
        let mut rho: Vec<f64> = Vec::new();

        let (mut g, mut f) = self.fd_grad(&x);
        for _ in 0..max_iter {
            // projected-gradient stopping test
            let pg_norm: f64 = (0..self.dim)
                .map(|i| {
                    let step = (x[i] - g[i]).clamp(0.0, 1.0) - x[i];
                    step * step
                })
                .sum::<f64>()
                .sqrt();
            if pg_norm < 1e-10 {
                break;
            }
            // two-loop recursion for the direction d = -H g
            let mut q = g.clone();
            let kh = s_hist.len();
            let mut alpha = vec![0.0; kh];
            for i in (0..kh).rev() {
                let a = rho[i] * dot(&s_hist[i], &q);
                alpha[i] = a;
                for j in 0..self.dim {
                    q[j] -= a * y_hist[i][j];
                }
            }
            let gamma = if kh > 0 {
                let last = kh - 1;
                dot(&s_hist[last], &y_hist[last]) / dot(&y_hist[last], &y_hist[last])
            } else {
                1.0
            };
            for qi in q.iter_mut() {
                *qi *= gamma;
            }
            for i in 0..kh {
                let beta = rho[i] * dot(&y_hist[i], &q);
                for j in 0..self.dim {
                    q[j] += (alpha[i] - beta) * s_hist[i][j];
                }
            }
            let d: Vec<f64> = q.iter().map(|v| -v).collect();

            // projected backtracking line search (Armijo)
            let gd = dot(&g, &d);
            let mut step = 1.0;
            let mut x_new = x.clone();
            let mut f_new = f;
            let mut ok = false;
            for _ in 0..20 {
                let cand: Vec<f64> = (0..self.dim)
                    .map(|i| (x[i] + step * d[i]).clamp(0.0, 1.0))
                    .collect();
                let fc = self.value(&cand);
                if fc.is_finite() && fc <= f + 1e-4 * step * gd {
                    x_new = cand;
                    f_new = fc;
                    ok = true;
                    break;
                }
                step *= 0.5;
                if self.max_eval_reached() {
                    break;
                }
            }
            if !ok || self.max_eval_reached() {
                break;
            }

            let (g_new, _) = self.fd_grad(&x_new);
            let s: Vec<f64> = (0..self.dim).map(|i| x_new[i] - x[i]).collect();
            let yv: Vec<f64> = (0..self.dim).map(|i| g_new[i] - g[i]).collect();
            let sy = dot(&s, &yv);
            if sy > 1e-12 {
                if s_hist.len() == m {
                    s_hist.remove(0);
                    y_hist.remove(0);
                    rho.remove(0);
                }
                s_hist.push(s);
                y_hist.push(yv);
                rho.push(1.0 / sy);
            }
            x = x_new;
            g = g_new;
            f = f_new;
            if self.max_eval_reached() {
                break;
            }
        }
        (self.fit_best_y, self.fit_best_x.clone())
    }

    // ---- main annealing loop ----

    fn search(&mut self) {
        let mut iter = 0i32;
        let t1 = ((QV - 1.0) * 2.0_f64.ln()).exp() - 1.0;
        loop {
            for i in 0..MAXSTEPS {
                let s = i as f64 + 2.0;
                let t2 = ((QV - 1.0) * s.ln()).exp() - 1.0;
                let temperature = TEMPERATURE_START * t1 / t2;
                iter += 1;
                if iter >= MAXSTEPS {
                    return;
                }
                if temperature < TEMPERATURE_RESTART {
                    self.reset_energy(&[]);
                    if self.reinit_failed {
                        return;
                    }
                    break;
                }
                self.run_chain(i as usize, temperature);
                if self.max_eval_reached() {
                    return;
                }
                if self.use_local_search {
                    self.chain_local_search();
                    if self.max_eval_reached() {
                        return;
                    }
                }
            }
        }
    }

    fn optimize(&mut self, guess: &[f64]) -> DaResult {
        let enc = self.encode(guess);
        self.reset_energy(&enc);
        self.emin = self.current_energy;
        self.xmin = self.current_location.clone();
        self.not_improved_max_idx = 1000;
        if !self.reinit_failed {
            self.search();
        }
        DaResult {
            x: self.decode(&self.xbest),
            y: self.ebest,
            evaluations: self.eval_counter,
            iterations: 0,
            stop: if self.reinit_failed { -1 } else { 0 },
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Run Dual Annealing. `lower`/`upper` empty ⇒ unbounded.
pub fn optimize_da(
    obj: &impl Objective,
    guess: &[f64],
    lower: Vec<f64>,
    upper: Vec<f64>,
    p: &DaParams,
) -> DaResult {
    let dim = guess.len();
    let max_evals = if p.max_evaluations == 0 {
        10_000_000
    } else {
        p.max_evaluations
    };
    let mut da = Da::new(
        obj,
        dim,
        lower,
        upper,
        max_evals,
        p.use_local_search,
        p.seed.wrapping_add(p.runid as u64),
    );
    da.optimize(guess)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
    fn rosen(x: &[f64]) -> f64 {
        (0..x.len() - 1)
            .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
            .sum()
    }

    fn run(obj: impl Objective, dim: usize, seed: u64, ls: bool) -> DaResult {
        let params = DaParams {
            max_evaluations: 40_000,
            use_local_search: ls,
            seed,
            ..Default::default()
        };
        optimize_da(
            &obj,
            &vec![0.0; dim],
            vec![-5.0; dim],
            vec![5.0; dim],
            &params,
        )
    }

    #[test]
    fn minimizes_sphere_with_local_search() {
        let r = run(sphere as fn(&[f64]) -> f64, 4, 1, true);
        assert!(r.y < 1e-6, "sphere not solved: {}", r.y);
        assert!((sphere(&r.x) - r.y).abs() < 1e-6);
    }

    #[test]
    fn minimizes_sphere_without_local_search() {
        let r = run(sphere as fn(&[f64]) -> f64, 4, 2, false);
        assert!(r.y < 1e-2, "sphere (no ls) too large: {}", r.y);
    }

    #[test]
    fn minimizes_rosenbrock() {
        let r = run(rosen as fn(&[f64]) -> f64, 3, 3, true);
        assert!(r.y < 1e-2, "rosenbrock not solved: {}", r.y);
    }

    #[test]
    fn unbounded_nonfinite_objective_reports_reinitialization_failure() {
        let result = optimize_da(
            &(|_: &[f64]| f64::NAN),
            &[0.0],
            Vec::new(),
            Vec::new(),
            &DaParams {
                max_evaluations: 0,
                use_local_search: false,
                seed: 4,
                runid: -1,
            },
        );
        assert_eq!(result.stop, -1);
        assert!(result.evaluations > 1_000);
    }

    #[test]
    fn finite_difference_handles_both_box_boundaries() {
        let objective = sphere as fn(&[f64]) -> f64;
        let mut optimizer = Da::new(&objective, 2, vec![0.0; 2], vec![1.0; 2], 100, false, 5);
        let (gradient, value) = optimizer.fd_grad(&[0.0, 1.0]);
        assert!(gradient.iter().all(|component| component.is_finite()));
        assert_eq!(value, 1.0);
    }
}
