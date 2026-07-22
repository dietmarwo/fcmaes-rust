//! CR-FM-NES — Rust port of the C++ `crfmnes.cpp`.
//!
//! Fast Moving Natural Evolution Strategy for high-dimensional problems
//! (<https://arxiv.org/abs/2201.11422>, derived from
//! <https://github.com/nomuramasahir0/crfmnes>). Faithful translation of the
//! Eigen implementation; the dense per-generation natural-gradient update on
//! `(v, D, sigma)` is expressed with column-vector algebra via `nalgebra`.
//!
//! The population is evaluated as a whole (the C++ backend only had a parallel
//! `func_par` callback), so the driver takes a batch closure. Replaces both the
//! C++ optimizer and the pure-Python `fcmaes/crfmnes.py`; parity is statistical.

use nalgebra::DVector;

use crate::fitness::Fitness;
use crate::rng::Rng;

/// Outcome of a CR-FM-NES run (mirrors the C++ `CrfmnesResult`).
#[derive(Clone, Debug)]
pub struct CrfmnesResult {
    pub x: Vec<f64>,
    pub y: f64,
    pub evaluations: u64,
    pub iterations: i32,
    pub stop: i32,
}

/// Tunable inputs for [`Crfmnes::new`].
#[derive(Clone, Debug)]
pub struct CrfmnesParams {
    pub popsize: i32,
    pub max_evaluations: u64,
    pub stop_fitness: f64,
    pub penalty_coef: f64,
    pub use_constraint_violation: bool,
    pub seed: u64,
    pub runid: i64,
}

impl Default for CrfmnesParams {
    fn default() -> Self {
        Self {
            popsize: 32,
            max_evaluations: 100_000,
            stop_fitness: f64::NEG_INFINITY,
            penalty_coef: 1e5,
            use_constraint_violation: true,
            seed: 0,
            runid: 0,
        }
    }
}

fn sort_index(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

pub struct Crfmnes {
    fitfun: Fitness,
    rng: Rng,
    dim: usize,
    lamb: usize,
    mu: usize,
    max_evaluations: u64,
    stopfitness: f64,
    penalty_coef: f64,
    use_constraint_violation: bool,

    m: DVector<f64>,
    sigma: f64,
    v: DVector<f64>,
    d: DVector<f64>,
    pc: DVector<f64>,
    ps: DVector<f64>,

    w_rank_hat: DVector<f64>,
    w_rank: DVector<f64>,
    mueff: f64,
    cs: f64,
    cc: f64,
    c1_cma: f64,
    chi_n: f64,
    h_inv: f64,
    eta_m: f64,
    eta_move_sigma: f64,

    // per-generation state (columns held as vectors)
    z: Vec<DVector<f64>>,
    y: Vec<DVector<f64>>,
    x: Vec<DVector<f64>>,
    xs_no_sort: Vec<DVector<f64>>,

    normv: f64,
    normv2: f64,
    vbar: DVector<f64>,

    iterations: i32,
    stop: i32,
    f_best: f64,
    x_best: DVector<f64>,
    external_evaluations: u64,
}

impl Crfmnes {
    pub fn new(mut fitfun: Fitness, guess: &[f64], sigma: f64, p: &CrfmnesParams) -> Self {
        let dim = fitfun.dim();
        fitfun.reset_evaluations();
        let lamb = p.popsize.max(2) as usize;
        let mu = lamb / 2;
        let dimf = dim as f64;

        let m = DVector::from_vec(fitfun.encode(guess));
        let mut rng = Rng::new(p.seed.wrapping_add(p.runid as u64));
        let v = DVector::from_iterator(dim, (0..dim).map(|_| rng.gaussian())) / dimf.sqrt();
        let d = DVector::from_element(dim, 1.0);

        // w_rank_hat[k] = max(0, ln(mu+1) - ln(k+1))
        let w_rank_hat = DVector::from_iterator(
            lamb,
            (1..=lamb).map(|k| ((mu as f64 + 1.0).ln() - (k as f64).ln()).max(0.0)),
        );
        let sum_hat = w_rank_hat.sum();
        let w_rank = w_rank_hat.map(|w| w / sum_hat - 1.0 / lamb as f64);
        let wlamb = w_rank.map(|w| w + 1.0 / lamb as f64);
        let mueff = 1.0 / wlamb.dot(&wlamb);
        let cs = (mueff + 2.0) / (dimf + mueff + 5.0);
        let cc = (4.0 + mueff / dimf) / (dimf + 4.0 + 2.0 * mueff / dimf);
        let c1_cma = 2.0 / ((dimf + 1.3).powi(2) + mueff);
        let chi_n = dimf.sqrt() * (1.0 - 1.0 / (4.0 * dimf) + 1.0 / (21.0 * dimf * dimf));
        let h_inv = get_h_inv(dimf);

        Crfmnes {
            dim,
            lamb,
            mu,
            max_evaluations: p.max_evaluations,
            stopfitness: p.stop_fitness,
            penalty_coef: if p.penalty_coef > 0.0 {
                p.penalty_coef
            } else {
                1e5
            },
            use_constraint_violation: p.use_constraint_violation,
            m,
            sigma,
            v,
            d,
            pc: DVector::zeros(dim),
            ps: DVector::zeros(dim),
            w_rank_hat,
            w_rank,
            mueff,
            cs,
            cc,
            c1_cma,
            chi_n,
            h_inv,
            eta_m: 1.0,
            eta_move_sigma: 1.0,
            z: vec![],
            y: vec![],
            x: vec![],
            xs_no_sort: vec![],
            normv: 0.0,
            normv2: 0.0,
            vbar: DVector::zeros(dim),
            iterations: 0,
            stop: 0,
            f_best: f64::INFINITY,
            x_best: DVector::zeros(dim),
            external_evaluations: 0,
            rng,
            fitfun,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn popsize(&self) -> usize {
        self.lamb
    }
    pub fn stop(&self) -> i32 {
        self.stop
    }

    fn cexp(a: f64) -> f64 {
        a.min(100.0).exp()
    }
    fn c1(&self, lamb_f: f64) -> f64 {
        self.c1_cma * (self.dim as f64 - 5.0) / 6.0 * (lamb_f / self.lamb as f64)
    }
    fn eta_b(&self, lamb_f: f64) -> f64 {
        let dimf = self.dim as f64;
        ((0.02 * lamb_f).min(3.0 * dimf.ln()) + 5.0) / (0.23 * dimf + 25.0)
    }
    fn eta_b_tanh(&self, lamb_f: f64) -> f64 {
        self.eta_b(lamb_f).tanh()
    }
    fn alpha_dist(&self, lamb_f: f64) -> f64 {
        let dimf = self.dim as f64;
        self.h_inv * (self.lamb as f64 / dimf).sqrt().min(1.0) * (lamb_f / self.lamb as f64).sqrt()
    }
    fn w_dist_hat(&self, z_norm: f64, lamb_f: f64) -> f64 {
        Self::cexp(self.alpha_dist(lamb_f) * z_norm)
    }
    fn eta_stag_sigma(&self, lamb_f: f64) -> f64 {
        let dimf = self.dim as f64;
        ((0.024 * lamb_f + 0.7 * dimf + 20.0) / (dimf + 12.0)).tanh()
    }
    fn eta_conv_sigma(&self, lamb_f: f64) -> f64 {
        let dimf = self.dim as f64;
        2.0 * ((0.025 * lamb_f + 0.75 * dimf + 10.0) / (dimf + 4.0)).tanh()
    }

    /// Sample and store the generation, returning the *encoded* candidates.
    fn ask(&mut self) {
        let dim = self.dim;
        // z: mu gaussian columns mirrored to lamb
        let zhalf: Vec<DVector<f64>> = (0..self.mu)
            .map(|_| DVector::from_iterator(dim, (0..dim).map(|_| self.rng.gaussian())))
            .collect();
        self.z = (0..self.lamb)
            .map(|i| {
                if i < self.mu {
                    zhalf[i].clone()
                } else {
                    -&zhalf[i - self.mu]
                }
            })
            .collect();
        self.normv = self.v.norm();
        self.normv2 = self.normv * self.normv;
        self.vbar = &self.v / self.normv;
        let coef = (1.0 + self.normv2).sqrt() - 1.0;
        self.y = self
            .z
            .iter()
            .map(|zk| zk + &self.vbar * (coef * self.vbar.dot(zk)))
            .collect();
        self.x = self
            .y
            .iter()
            .map(|yk| &self.m + self.sigma * yk.component_mul(&self.d))
            .collect();
    }

    /// Decoded, in-bounds population rows (for evaluation / reporting).
    fn decode_columns(&self, cols: &[DVector<f64>]) -> Vec<Vec<f64>> {
        cols.iter()
            .map(|c| {
                self.fitfun
                    .closest_feasible(&self.fitfun.decode(c.as_slice()))
            })
            .collect()
    }

    fn sort_indices_by(&self, evals: &[f64]) -> Vec<usize> {
        let mut sorted = sort_index(evals);
        // Feasible-first is already guaranteed here (evals are finite); the C++
        // z-distance tie-break for +inf entries only matters for genuinely
        // infinite evals, which the sanitized pipeline never produces.
        let n_inf = evals.iter().filter(|&&e| e == f64::INFINITY).count();
        if n_inf > 0 {
            let n_feasible = evals.len() - n_inf;
            let mut dist: Vec<(f64, usize)> = evals
                .iter()
                .enumerate()
                .filter(|&(_, &e)| e == f64::INFINITY)
                .map(|(i, _)| (self.z[i].dot(&self.z[i]), i))
                .collect();
            dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (j, (_, _)) in dist.iter().enumerate() {
                sorted[n_feasible + j] = dist[j].1;
            }
        }
        sorted
    }

    fn tell(&mut self, evs: &[f64]) -> i32 {
        let dim = self.dim;
        let lamb = self.lamb;
        let mut evals: Vec<f64> = evs
            .iter()
            .map(|&e| if e.is_finite() { e } else { f64::MAX })
            .collect();
        self.xs_no_sort = self.x.clone();

        if self.use_constraint_violation {
            for (k, e) in evals.iter_mut().enumerate() {
                *e += self
                    .fitfun
                    .violation(self.x[k].as_slice(), self.penalty_coef);
            }
        }
        let sorted = self.sort_indices_by(&evals);
        let best_eval_id = sorted[0];
        let f_best_ = evals[best_eval_id];

        // reorder z, y, x
        self.z = sorted.iter().map(|&i| self.z[i].clone()).collect();
        self.y = sorted.iter().map(|&i| self.y[i].clone()).collect();
        self.x = sorted.iter().map(|&i| self.x[i].clone()).collect();

        self.external_evaluations += lamb as u64;
        if f_best_ < self.f_best {
            self.f_best = f_best_;
            self.x_best =
                DVector::from_vec(self.fitfun.decode(self.xs_no_sort[best_eval_id].as_slice()));
            if self.f_best < self.stopfitness {
                self.stop = 1;
            }
        }

        let lamb_f = evals.iter().filter(|&&e| e < f64::MAX).count() as f64;

        // evolution path p_sigma = (1-cs) ps + sqrt(cs(2-cs)mueff) * (z w_rank)
        let z_wrank = weighted_sum(&self.z, self.w_rank.as_slice());
        self.ps =
            &self.ps * (1.0 - self.cs) + z_wrank * (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        let ps_norm = self.ps.norm();

        // distance weights
        let mut w_tmp = DVector::zeros(lamb);
        for k in 0..lamb {
            w_tmp[k] = self.w_rank_hat[k] * self.w_dist_hat(self.z[k].norm(), lamb_f);
        }
        let w_tmp_sum = w_tmp.sum();
        let weights_dist = w_tmp.map(|w| w / w_tmp_sum - 1.0 / lamb as f64);

        let weights = if ps_norm >= self.chi_n {
            weights_dist
        } else {
            self.w_rank.clone()
        };
        let eta_sigma = if ps_norm >= self.chi_n {
            self.eta_move_sigma
        } else if ps_norm >= 0.1 * self.chi_n {
            self.eta_stag_sigma(lamb_f)
        } else {
            self.eta_conv_sigma(lamb_f)
        };

        // update pc, m
        let wxm = weighted_sum(
            &self.x.iter().map(|xk| xk - &self.m).collect::<Vec<_>>(),
            weights.as_slice(),
        );
        self.pc = &self.pc * (1.0 - self.cc)
            + &wxm * ((self.cc * (2.0 - self.cc) * self.mueff).sqrt() / self.sigma);
        self.m += self.eta_m * &wxm;

        // ---- natural gradient step for (v, D) ----
        let normv2 = self.normv2;
        let normv4 = normv2 * normv2;
        let gammav = 1.0 + normv2;
        let vbar = self.vbar.clone();
        let vbarbar = vbar.component_mul(&vbar);

        // exY columns: y (lamb of them) then pc./D
        let mut ex_y: Vec<DVector<f64>> = self.y.clone();
        ex_y.push(self.pc.component_div(&self.d));
        let lp1 = lamb + 1;

        let ip_yvbar: Vec<f64> = ex_y.iter().map(|c| vbar.dot(c)).collect();
        let yy: Vec<DVector<f64>> = ex_y.iter().map(|c| c.component_mul(c)).collect();
        let yvbar: Vec<DVector<f64>> = ex_y.iter().map(|c| c.component_mul(&vbar)).collect();

        let alphavd =
            (normv4 + (2.0 * gammav - gammav.sqrt()) / vbarbar.max()).sqrt() / (2.0 + normv2);
        let alphavd = alphavd.min(1.0);

        let ibg: Vec<f64> = ip_yvbar.iter().map(|&p| p * p + gammav).collect();
        // t columns
        let mut t: Vec<DVector<f64>> = (0..lp1)
            .map(|k| &ex_y[k] * ip_yvbar[k] - &vbar * (ibg[k] / 2.0))
            .collect();

        let a2 = alphavd * alphavd;
        let b = -(1.0 - a2) * normv4 / gammav + 2.0 * a2;
        let h = vbarbar.map(|vb| 2.0 - (b + 2.0 * a2) * vb);
        let inv_h = h.map(|v| 1.0 / v);

        // s_step1
        let s_step1: Vec<DVector<f64>> = (0..lp1)
            .map(|k| {
                &yy[k]
                    - &yvbar[k] * (normv2 / gammav * ip_yvbar[k])
                    - DVector::from_element(dim, 1.0)
            })
            .collect();
        let ip_vbart: Vec<f64> = t.iter().map(|c| vbar.dot(c)).collect();
        let s_step2: Vec<DVector<f64>> = (0..lp1)
            .map(|k| {
                &s_step1[k]
                    - (t[k].component_mul(&vbar) * (2.0 + normv2)
                        - &vbarbar * (normv2 * ip_vbart[k]))
                        * (alphavd / gammav)
            })
            .collect();

        let inv_h_vbarbar = inv_h.component_mul(&vbarbar);
        let ip_s2: Vec<f64> = s_step2.iter().map(|c| inv_h_vbarbar.dot(c)).collect();
        let div = 1.0 + b * vbarbar.dot(&inv_h_vbarbar);
        if div == 0.0 {
            self.stop = -1;
            return self.stop;
        }
        let s: Vec<DVector<f64>> = (0..lp1)
            .map(|k| s_step2[k].component_mul(&inv_h) - &inv_h_vbarbar * (b / div * ip_s2[k]))
            .collect();
        let ip_svbarbar: Vec<f64> = s.iter().map(|c| vbarbar.dot(c)).collect();
        for k in 0..lp1 {
            t[k] = &t[k]
                - (s[k].component_mul(&vbar) * (2.0 + normv2) - &vbar * ip_svbarbar[k]) * alphavd;
        }

        // exw
        let mut exw = vec![0.0; lp1];
        let eta_b = self.eta_b_tanh(lamb_f);
        for k in 0..lamb {
            exw[k] = eta_b * weights[k];
        }
        exw[lamb] = self.c1(lamb_f);

        self.v = &self.v + weighted_sum(&t, &exw) / self.normv;
        let sd = weighted_sum(&s, &exw);
        self.d = &self.d + sd.component_mul(&self.d);

        if self.d.min() < 0.0 {
            self.stop = -1;
            return self.stop;
        }
        let nthrootdeta = Self::cexp(
            self.d.map(|v| v.ln()).sum() / dim as f64
                + (1.0 + self.v.dot(&self.v)).ln() / (2.0 * dim as f64),
        );
        self.d /= nthrootdeta;

        // update sigma
        let mut g_s = 0.0;
        for k in 0..lamb {
            let zz = self.z[k].component_mul(&self.z[k]);
            g_s += (zz.sum() - dim as f64) * weights[k];
        }
        g_s /= dim as f64;
        self.sigma *= Self::cexp(eta_sigma / 2.0 * g_s);

        self.stop
    }

    fn make_result(&self, evaluations: u64) -> CrfmnesResult {
        CrfmnesResult {
            x: self.x_best.as_slice().to_vec(),
            y: self.f_best,
            evaluations,
            iterations: self.iterations,
            stop: self.stop,
        }
    }

    /// Run the generational loop, evaluating each population through the batch
    /// closure `eval_batch(decoded_rows) -> ys` (the C++ `func_par` path).
    pub fn optimize_batch<F>(&mut self, mut eval_batch: F) -> CrfmnesResult
    where
        F: FnMut(&[Vec<f64>]) -> Vec<f64>,
    {
        self.iterations = 1;
        self.fitfun.reset_evaluations();
        while self.fitfun.evaluations() < self.max_evaluations && !self.fitfun.terminate() {
            self.ask();
            let rows = self.decode_columns(&self.x);
            let mut ys = eval_batch(&rows);
            for v in ys.iter_mut() {
                if !v.is_finite() {
                    *v = crate::fitness::NAN_REPLACEMENT;
                }
            }
            self.fitfun.incr_evaluations(self.lamb as u64);
            self.tell(&ys);
            if self.stop != 0 {
                break;
            }
            self.iterations += 1;
        }
        self.make_result(self.fitfun.evaluations())
    }

    // ---- ask/tell interface (mirrors CrfmnesState::Impl) ----

    /// Ask for a decoded, in-bounds population (rows = individuals).
    pub fn ask_pop(&mut self) -> Vec<Vec<f64>> {
        self.ask();
        self.decode_columns(&self.x)
    }

    /// Tell fitness values for the population from [`ask_pop`](Crfmnes::ask_pop).
    pub fn tell_pop(&mut self, ys: &[f64]) -> i32 {
        self.iterations += 1;
        self.tell(ys)
    }

    pub fn population(&self) -> Vec<Vec<f64>> {
        self.decode_columns(&self.xs_no_sort)
    }

    pub fn result(&self) -> CrfmnesResult {
        self.make_result(self.external_evaluations)
    }
}

/// Weighted sum of column vectors: `sum_k cols[k] * w[k]`.
fn weighted_sum(cols: &[DVector<f64>], w: &[f64]) -> DVector<f64> {
    let dim = cols[0].len();
    let mut acc = DVector::zeros(dim);
    for (c, &wk) in cols.iter().zip(w) {
        acc += c * wk;
    }
    acc
}

fn get_h_inv(dimf: f64) -> f64 {
    let f = |a: f64| ((1.0 + a * a) * (a * a / 2.0).min(100.0).exp() / 0.24) - 10.0 - dimf;
    let f_prime = |a: f64| (1.0 / 0.24) * a * (a * a / 2.0).min(100.0).exp() * (3.0 + a * a);
    let mut h = 1.0;
    while f(h).abs() > 1e-10 {
        h -= 0.5 * (f(h) / f_prime(h));
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn run(dim: usize, seed: u64, evals: u64) -> f64 {
        let fit = Fitness::bounded(dim, 1, &vec![-5.0; dim], &vec![5.0; dim]);
        let params = CrfmnesParams {
            popsize: 32,
            max_evaluations: evals,
            seed,
            ..Default::default()
        };
        let mut opt = Crfmnes::new(fit, &vec![0.0; dim], 2.0, &params);
        let r = opt.optimize_batch(|rows| rows.iter().map(|x| sphere(x)).collect());
        r.y
    }

    #[test]
    fn minimizes_sphere() {
        let y = run(5, 1, 8000);
        assert!(y < 1e-6, "sphere not solved: {y}");
    }

    #[test]
    fn ask_tell_converges() {
        let fit = Fitness::bounded(5, 1, &[-5.0; 5], &[5.0; 5]);
        let params = CrfmnesParams {
            popsize: 16,
            max_evaluations: 8000,
            seed: 3,
            ..Default::default()
        };
        let mut opt = Crfmnes::new(fit, &[0.0; 5], 2.0, &params);
        for _ in 0..600 {
            let pop = opt.ask_pop();
            let ys: Vec<f64> = pop.iter().map(|x| sphere(x)).collect();
            if opt.tell_pop(&ys) != 0 {
                break;
            }
            if opt.result().evaluations >= 8000 {
                break;
            }
        }
        assert!(
            opt.result().y < 1e-3,
            "ask/tell did not converge: {}",
            opt.result().y
        );
    }

    #[test]
    fn getters_infinite_sort_default_penalty_and_stop() {
        let fit = Fitness::bounded(3, 1, &[-1.0; 3], &[1.0; 3]);
        let params = CrfmnesParams {
            popsize: 1,
            max_evaluations: 2,
            stop_fitness: 1.0,
            penalty_coef: 0.0,
            use_constraint_violation: false,
            seed: 8,
            runid: -1,
        };
        let mut optimizer = Crfmnes::new(fit, &[0.0; 3], 0.5, &params);
        assert_eq!(optimizer.dim(), 3);
        assert_eq!(optimizer.popsize(), 2);
        assert_eq!(optimizer.stop(), 0);
        let population = optimizer.ask_pop();
        assert_eq!(population.len(), 2);
        let order = optimizer.sort_indices_by(&[f64::INFINITY, 0.0]);
        assert_eq!(order[0], 1);
        assert_eq!(optimizer.tell_pop(&[0.0, f64::INFINITY]), 1);
        assert_eq!(optimizer.stop(), 1);
        assert_eq!(optimizer.population().len(), 2);
    }

    #[test]
    fn batch_sanitizes_nonfinite_scores() {
        let fit = Fitness::bounded(2, 1, &[-1.0; 2], &[1.0; 2]);
        let mut optimizer = Crfmnes::new(
            fit,
            &[0.0; 2],
            0.5,
            &CrfmnesParams {
                popsize: 4,
                max_evaluations: 4,
                ..Default::default()
            },
        );
        let result = optimizer.optimize_batch(|rows| vec![f64::NAN; rows.len()]);
        assert_eq!(result.evaluations, 4);
        assert!(result.y.is_finite());
    }
}
