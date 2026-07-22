//! Differential Evolution — Rust port of the C++ `deoptimizer.cpp`.
//!
//! DE/best/1 with the fcmaes extensions: temporal locality (an extra
//! improvement trial along the previous move), age-based reinitialization of
//! stale individuals, oscillating `F`/`CR` between generations, optional
//! normal-distributed sampling around a guess, and mixed-integer "modify"
//! resampling. Replaces both the C++ optimizer and the pure-Python
//! `fcmaes/de.py`. Cross-implementation parity is statistical.

use std::collections::VecDeque;

use crate::fitness::{Fitness, Objective};
use crate::rng::Rng;

/// Outcome of a DE run (mirrors the C++ `DeResult`).
#[derive(Clone, Debug)]
pub struct DeResult {
    pub x: Vec<f64>,
    pub y: f64,
    pub evaluations: u64,
    pub iterations: i32,
    pub stop: i32,
}

/// Tunable inputs for [`De::new`]. Non-positive values select DE defaults.
#[derive(Clone, Debug)]
pub struct DeParams {
    pub popsize: i32,
    pub max_evaluations: u64,
    pub keep: f64,
    pub stop_fitness: f64,
    pub f: f64,
    pub cr: f64,
    pub min_mutate: f64,
    pub max_mutate: f64,
    pub min_sigma: f64,
    pub seed: u64,
    pub runid: i64,
}

impl Default for DeParams {
    fn default() -> Self {
        Self {
            popsize: 31,
            max_evaluations: 100_000,
            keep: 200.0,
            stop_fitness: f64::NEG_INFINITY,
            f: 0.5,
            cr: 0.9,
            min_mutate: 0.1,
            max_mutate: 0.5,
            min_sigma: 0.0,
            seed: 0,
            runid: 0,
        }
    }
}

pub struct De {
    fitfun: Fitness,
    rng: Rng,
    dim: usize,
    popsize: usize,
    max_evaluations: u64,
    keep: f64,
    stopfitness: f64,
    f0: f64,
    cr0: f64,
    f: f64,
    cr: f64,
    min_mutate: f64,
    max_mutate: f64,
    is_int: Option<Vec<bool>>,

    // normal-sampling around a guess
    use_normal: bool,
    mean: Vec<f64>,
    sigma: Vec<f64>,
    max_sigma: Vec<f64>,
    min_sigma_vec: Vec<f64>,
    min_sigma_val: f64,
    mean_hist: Vec<Vec<f64>>, // 10 columns, each dim long
    mean_hist_index: usize,

    // population
    pop_x: Vec<Vec<f64>>,
    pop_x0: Vec<Vec<f64>>,
    pop_y: Vec<f64>,
    pop_iter: Vec<i32>,
    best_i: usize,
    best_x: Vec<f64>,
    best_y: f64,

    iterations: i32,
    stop: i32,
    pos: usize,

    // ask/tell bookkeeping
    improves_x: VecDeque<Vec<f64>>,
    improves_p: VecDeque<usize>,
    asked_x: Vec<Vec<f64>>,
    asked_p: Vec<usize>,
    external_evaluations: u64,
}

impl De {
    /// Build a DE optimizer. `guess`/`sigma` empty ⇒ uniform sampling in the
    /// box; non-empty ⇒ normal sampling around `guess`. `ints` marks discrete
    /// coordinates (length `dim`) or is `None`.
    pub fn new(
        mut fitfun: Fitness,
        guess: &[f64],
        sigma: &[f64],
        ints: Option<Vec<bool>>,
        p: &DeParams,
    ) -> Self {
        let dim = fitfun.dim();
        fitfun.reset_evaluations();
        let popsize = if p.popsize > 0 {
            p.popsize as usize
        } else {
            15 * dim
        };
        let keep = if p.keep > 0.0 { p.keep } else { 30.0 };
        let f0 = if p.f > 0.0 { p.f } else { 0.5 };
        let cr0 = if p.cr > 0.0 { p.cr } else { 0.9 };
        let min_mutate = if p.min_mutate > 0.0 {
            p.min_mutate
        } else {
            0.1
        };
        let max_mutate = if p.max_mutate > 0.0 {
            p.max_mutate
        } else {
            0.5
        };

        let use_normal = !guess.is_empty();
        let mean = guess.to_vec();
        let sigma_v = sigma.to_vec();

        let mut de = De {
            dim,
            popsize,
            max_evaluations: if p.max_evaluations > 0 {
                p.max_evaluations
            } else {
                50_000
            },
            keep,
            stopfitness: p.stop_fitness,
            f0,
            cr0,
            f: f0,
            cr: cr0,
            min_mutate,
            max_mutate,
            is_int: ints,
            use_normal,
            mean,
            sigma: sigma_v,
            max_sigma: vec![],
            min_sigma_vec: vec![],
            min_sigma_val: p.min_sigma,
            mean_hist: vec![],
            mean_hist_index: 0,
            pop_x: vec![],
            pop_x0: vec![],
            pop_y: vec![],
            pop_iter: vec![],
            best_i: 0,
            best_x: vec![],
            best_y: f64::MAX,
            iterations: 0,
            stop: 0,
            pos: 0,
            improves_x: VecDeque::new(),
            improves_p: VecDeque::new(),
            asked_x: vec![],
            asked_p: vec![],
            external_evaluations: 0,
            rng: Rng::new(p.seed.wrapping_add(p.runid as u64)),
            fitfun,
        };
        de.init();
        de
    }

    fn init(&mut self) {
        let dim = self.dim;
        self.mean_hist = (0..10).map(|_| self.mean.clone()).collect();
        self.mean_hist_index = 0;
        if self.use_normal {
            self.max_sigma = self
                .sigma
                .iter()
                .map(|s| s / (0.1 + self.min_sigma_val))
                .collect();
            self.min_sigma_vec = self.sigma.iter().map(|s| self.min_sigma_val * s).collect();
        }
        self.pop_x = Vec::with_capacity(self.popsize);
        self.pop_x0 = Vec::with_capacity(self.popsize);
        self.pop_y = vec![f64::MAX; self.popsize];
        for _ in 0..self.popsize {
            let s = self.sample();
            self.pop_x0.push(s.clone());
            self.pop_x.push(s);
        }
        self.best_i = 0;
        self.best_x = self.pop_x[0].clone();
        self.pop_iter = vec![0; self.popsize];
        self.asked_x = vec![vec![0.0; dim]; self.popsize];
        self.asked_p = vec![0; self.popsize];
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn popsize(&self) -> usize {
        self.popsize
    }
    pub fn stop(&self) -> i32 {
        self.stop
    }

    fn sample(&mut self) -> Vec<f64> {
        if self.use_normal {
            let raw: Vec<f64> = (0..self.dim)
                .map(|i| self.mean[i] + self.rng.gaussian() * self.sigma[i])
                .collect();
            self.fitfun.closest_feasible(&raw)
        } else {
            self.fitfun.sample(&mut self.rng)
        }
    }

    fn sample_i(&mut self, i: usize) -> f64 {
        if self.use_normal {
            let v = self.rng.normreal(self.mean[i], self.sigma[i]);
            self.fitfun.closest_feasible_i(i, v)
        } else {
            self.fitfun.sample_i(i, &mut self.rng)
        }
    }

    fn update_mean(&mut self) {
        if !self.use_normal {
            return;
        }
        self.mean_hist[self.mean_hist_index] = self.pop_x[self.best_i].clone();
        self.mean_hist_index = (self.mean_hist_index + 1) % self.mean_hist.len();
        // delta = rowwise (max - min) over the history columns; clamped per
        // coordinate against parallel max/min-sigma arrays.
        let mut sigma_new = vec![0.0; self.dim];
        for (i, sn) in sigma_new.iter_mut().enumerate() {
            let mut lo = f64::MAX;
            let mut hi = f64::MIN;
            for col in &self.mean_hist {
                lo = lo.min(col[i]);
                hi = hi.max(col[i]);
            }
            *sn = (hi - lo).min(self.max_sigma[i]).max(self.min_sigma_vec[i]);
        }
        let mean_new: f64 = sigma_new.iter().sum::<f64>() / self.dim as f64;
        let mean_old: f64 = self.sigma.iter().sum::<f64>() / self.dim as f64;
        if mean_new > mean_old {
            self.sigma = sigma_new;
        } else {
            for (s, sn) in self.sigma.iter_mut().zip(&sigma_new) {
                *s = 0.9 * *s + 0.1 * sn;
            }
        }
        let best = self.pop_x[self.best_i].clone();
        for (m, b) in self.mean.iter_mut().zip(&best) {
            *m = 0.9 * *m + 0.1 * b;
        }
    }

    fn modify(&mut self, x: &mut [f64]) {
        let Some(is_int) = self.is_int.clone() else {
            return;
        };
        let n_ints = is_int.iter().filter(|&&b| b).count() as f64;
        if n_ints == 0.0 {
            return;
        }
        let to_mutate =
            self.min_mutate + self.rng.uniform01() * (self.max_mutate - self.min_mutate);
        for i in 0..self.dim {
            if is_int[i] && self.rng.uniform01() < to_mutate / n_ints {
                x[i] = self.sample_i(i).trunc();
            }
        }
    }

    fn next_improve(&mut self, xb: &[f64], x: &[f64], xi: &[f64]) -> Vec<f64> {
        let raw: Vec<f64> = (0..self.dim)
            .map(|j| xb[j] + (x[j] - xi[j]) * self.f0)
            .collect();
        let mut nextx = self.fitfun.closest_feasible(&raw);
        self.modify(&mut nextx);
        nextx
    }

    fn oscillate(&mut self) {
        self.cr = if self.iterations % 2 == 0 {
            0.5 * self.cr0
        } else {
            self.cr0
        };
        self.f = if self.iterations % 2 == 0 {
            0.5 * self.f0
        } else {
            self.f0
        };
    }

    fn pick_two(&mut self, p: usize) -> (usize, usize) {
        let mut r1;
        loop {
            r1 = self.rng.int_below(self.popsize as i64) as usize;
            if r1 != p && r1 != self.best_i {
                break;
            }
        }
        let mut r2;
        loop {
            r2 = self.rng.int_below(self.popsize as i64) as usize;
            if r2 != p && r2 != self.best_i && r2 != r1 {
                break;
            }
        }
        (r1, r2)
    }

    /// Ask-path donor construction (the C++ `nextX`).
    fn next_x(&mut self, p: usize, xp: &[f64], xb: &[f64]) -> Vec<f64> {
        if p == 0 {
            self.iterations += 1;
            self.oscillate();
            if self.iterations > 2 {
                self.update_mean();
            }
        }
        let (r1, r2) = self.pick_two(p);
        let x1 = self.pop_x[r1].clone();
        let x2 = self.pop_x[r2].clone();
        let mut x: Vec<f64> = (0..self.dim)
            .map(|j| xb[j] + (x1[j] - x2[j]) * self.f)
            .collect();
        let r = self.rng.int_below(self.dim as i64) as usize;
        for j in 0..self.dim {
            if j != r && self.rng.uniform01() > self.cr {
                x[j] = xp[j];
            }
        }
        let mut nextx = self.fitfun.closest_feasible(&x);
        self.modify(&mut nextx);
        nextx
    }

    fn ask_one(&mut self) -> (usize, Vec<f64>) {
        if self.improves_x.is_empty() {
            let p = self.pos;
            let xp = self.pop_x[p].clone();
            let xb = self.pop_x[self.best_i].clone();
            let x = self.next_x(p, &xp, &xb);
            self.pos = (self.pos + 1) % self.popsize;
            (p, x)
        } else {
            let p = self.improves_p.pop_front().unwrap();
            let x = self.improves_x.pop_front().unwrap();
            (p, x)
        }
    }

    fn tell_one(&mut self, y: f64, x: &[f64], p: usize) -> i32 {
        if y.is_finite() && y < self.pop_y[p] {
            if self.iterations > 1 {
                let xb = self.pop_x[self.best_i].clone();
                let xi = self.pop_x0[p].clone();
                let improved = self.next_improve(&xb, x, &xi);
                self.improves_p.push_back(p);
                self.improves_x.push_back(improved);
            }
            self.pop_x0[p] = self.pop_x[p].clone();
            self.pop_x[p] = x.to_vec();
            self.pop_y[p] = y;
            self.pop_iter[p] = self.iterations;
            if y < self.pop_y[self.best_i] {
                self.best_i = p;
                if y < self.best_y {
                    self.best_y = y;
                    self.best_x = x.to_vec();
                    if self.stopfitness.is_finite() && self.best_y < self.stopfitness {
                        self.stop = 1;
                    }
                }
            }
        } else if self.keep * self.rng.uniform01() < (self.iterations - self.pop_iter[p]) as f64 {
            self.pop_x[p] = self.sample();
            self.pop_y[p] = f64::MAX;
        }
        self.stop
    }

    /// Serial generational loop (the C++ `doOptimize`), the driver behind
    /// `optimize_de` for `workers <= 1`.
    pub fn optimize(&mut self, obj: &impl Objective) -> DeResult {
        self.iterations = 1;
        self.fitfun.reset_evaluations();
        while self.fitfun.evaluations() < self.max_evaluations && !self.fitfun.terminate() {
            if self.iterations > 2 {
                self.update_mean();
            }
            self.oscillate();
            for p in 0..self.popsize {
                let xp = self.pop_x[p].clone();
                let xb = self.pop_x[self.best_i].clone();
                let (r1, r2) = self.pick_two(p);
                let x1 = self.pop_x[r1].clone();
                let x2 = self.pop_x[r2].clone();
                let r = self.rng.int_below(self.dim as i64) as usize;
                let mut x = xp.clone();
                for j in 0..self.dim {
                    if j == r || self.rng.uniform01() < self.cr {
                        x[j] = xb[j] + self.f * (x1[j] - x2[j]);
                        if !self.fitfun.feasible_i(j, x[j]) {
                            x[j] = self.sample_i(j);
                        }
                    }
                }
                self.modify(&mut x);
                let mut y = self.fitfun.eval_encoded_scalar(&x, obj);
                if y.is_finite() && y < self.pop_y[p] {
                    // temporal locality: an extra trial along the last move
                    let x2t = self.next_improve(&xb, &x, &xp);
                    let y2 = self.fitfun.eval_encoded_scalar(&x2t, obj);
                    if y2.is_finite() && y2 < y {
                        y = y2;
                        x = x2t;
                    }
                    self.pop_x[p] = x.clone();
                    self.pop_y[p] = y;
                    self.pop_iter[p] = self.iterations;
                    if y < self.pop_y[self.best_i] {
                        self.best_i = p;
                        if y < self.best_y {
                            self.best_y = y;
                            self.best_x = x.clone();
                            if self.stopfitness.is_finite() && self.best_y < self.stopfitness {
                                self.stop = 1;
                                return self.make_result(self.fitfun.evaluations());
                            }
                        }
                    }
                } else if self.keep * self.rng.uniform01()
                    < (self.iterations - self.pop_iter[p]) as f64
                {
                    self.pop_x[p] = self.sample();
                    self.pop_y[p] = f64::MAX;
                }
            }
            self.iterations += 1;
        }
        self.make_result(self.fitfun.evaluations())
    }

    fn make_result(&self, evaluations: u64) -> DeResult {
        DeResult {
            x: self.best_x.clone(),
            y: self.best_y,
            evaluations,
            iterations: self.iterations,
            stop: self.stop,
        }
    }

    // ---- ask/tell interface (mirrors DeState::Impl) ----

    /// Ask for a full population of candidate rows.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        for i in 0..self.popsize {
            let (p, x) = self.ask_one();
            self.asked_p[i] = p;
            self.asked_x[i] = x;
        }
        self.asked_x.clone()
    }

    /// Tell fitness values for the population returned by [`ask`](De::ask).
    pub fn tell(&mut self, ys: &[f64]) -> i32 {
        for (i, &y) in ys.iter().enumerate() {
            let x = self.asked_x[i].clone();
            self.tell_one(y, &x, self.asked_p[i]);
        }
        self.external_evaluations += ys.len() as u64;
        self.stop
    }

    pub fn population(&self) -> Vec<Vec<f64>> {
        self.pop_x.clone()
    }

    pub fn result(&self) -> DeResult {
        self.make_result(self.external_evaluations)
    }
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

    fn optimize(obj: impl Objective, dim: usize, seed: u64, evals: u64) -> f64 {
        let fit = Fitness::bounded(dim, 1, &vec![-5.0; dim], &vec![5.0; dim]);
        let params = DeParams {
            popsize: 31,
            max_evaluations: evals,
            seed,
            ..Default::default()
        };
        let mut de = De::new(fit, &[], &[], None, &params);
        de.optimize(&obj).y
    }

    #[test]
    fn minimizes_sphere() {
        assert!(optimize(sphere as fn(&[f64]) -> f64, 5, 1, 8000) < 1e-6);
    }

    #[test]
    fn minimizes_rosenbrock() {
        let mut v: Vec<f64> = (0..5)
            .map(|s| optimize(rosen as fn(&[f64]) -> f64, 5, s, 12000))
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(v[2] < 1.0, "rosen median too large: {v:?}");
    }

    #[test]
    fn ask_tell_converges() {
        let fit = Fitness::bounded(5, 1, &[-5.0; 5], &[5.0; 5]);
        let params = DeParams {
            popsize: 24,
            max_evaluations: 8000,
            seed: 3,
            ..Default::default()
        };
        let mut de = De::new(fit, &[], &[], None, &params);
        for _ in 0..400 {
            let pop = de.ask();
            let ys: Vec<f64> = pop.iter().map(|x| sphere(x)).collect();
            if de.tell(&ys) != 0 {
                break;
            }
        }
        assert!(de.result().y < 1e-3, "ask/tell did not converge");
    }

    #[test]
    fn integer_modify_runs() {
        let fit = Fitness::bounded(4, 1, &[-5.0; 4], &[5.0; 4]);
        let params = DeParams {
            popsize: 20,
            max_evaluations: 4000,
            seed: 5,
            ..Default::default()
        };
        let ints = Some(vec![true, false, true, false]);
        let mut de = De::new(fit, &[], &[], ints, &params);
        let r = de.optimize(&(sphere as fn(&[f64]) -> f64));
        assert!(r.evaluations > 0);
    }

    #[test]
    fn normal_sampling_default_parameters_getters_and_early_stop() {
        let fit = Fitness::bounded(2, 1, &[-1.0; 2], &[1.0; 2]);
        let params = DeParams {
            popsize: 0,
            max_evaluations: 10,
            keep: 0.0,
            stop_fitness: 1.0,
            f: 0.0,
            cr: 0.0,
            min_mutate: 0.0,
            max_mutate: 0.0,
            min_sigma: 0.1,
            seed: 11,
            runid: -2,
        };
        let mut optimizer = De::new(fit, &[0.0, 0.0], &[0.2, 0.2], None, &params);
        assert_eq!(optimizer.dim(), 2);
        assert_eq!(optimizer.popsize(), 30);
        assert_eq!(optimizer.population().len(), 30);
        assert_eq!(optimizer.stop(), 0);
        let result = optimizer.optimize(&(sphere as fn(&[f64]) -> f64));
        assert_eq!(result.stop, 1);
        assert!(result.evaluations > 0);
    }
}
