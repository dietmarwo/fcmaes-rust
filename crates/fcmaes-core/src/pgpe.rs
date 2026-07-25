//! Parameter-Exploring Policy Gradients (PGPE).
//!
//! PGPE searches the parameters of a sampling distribution instead of
//! perturbing actions. This implementation combines symmetric (“mirrored”)
//! sampling with an Adam update of the distribution center.
//!
//! # Reference
//!
//! F. Sehnke, C. Osendorfer, T. Rückstieß, A. Graves, J. Peters, and
//! J. Schmidhuber, [“Parameter-Exploring Policy
//! Gradients”](https://doi.org/10.1016/j.neunet.2009.12.004), *Neural
//! Networks* 23(4), 551–559 (2010).
//!
//! # Example
//!
//! ```
//! use fcmaes_core::{Fitness, Pgpe, PgpeParams};
//!
//! let fit = Fitness::bounded(6, 1, &[-3.0; 6], &[3.0; 6]);
//! let params = PgpeParams {
//!     max_evaluations: 1_024,
//!     seed: 19,
//!     ..Default::default()
//! };
//! let mut pgpe = Pgpe::new(fit, &[1.0; 6], &[0.4; 6], &params);
//! let result = pgpe.optimize_batch(|population| {
//!     population
//!         .iter()
//!         .map(|x| x.iter().map(|v| v * v).sum())
//!         .collect()
//! });
//! assert!(result.y.is_finite());
//! ```

use nalgebra::DVector;

use crate::fitness::Fitness;
use crate::rng::Rng;

/// Outcome of a PGPE run.
#[derive(Clone, Debug)]
pub struct PgpeResult {
    /// Best decoded decision vector found.
    pub x: Vec<f64>,
    /// Objective value at [`x`](Self::x).
    pub y: f64,
    /// Number of objective evaluations charged to the run.
    pub evaluations: u64,
    /// Number of completed distribution updates.
    pub iterations: i32,
    /// Termination code; `1` means `stop_fitness` was reached.
    pub stop: i32,
}

/// Tunable inputs for [`Pgpe::new`].
#[derive(Clone, Debug)]
pub struct PgpeParams {
    /// Population size; the implementation rounds odd values up for mirrored pairs.
    pub popsize: i32,
    /// Maximum number of objective evaluations.
    pub max_evaluations: u64,
    /// Stop after finding an objective value strictly below this threshold.
    pub stop_fitness: f64,
    /// Number of updates over which the center learning rate decays.
    pub lr_decay_steps: i32,
    /// Use rank-normalized utilities instead of raw objective differences.
    pub use_ranking: bool,
    /// Initial Adam learning rate for the distribution center.
    pub center_learning_rate: f64,
    /// Learning rate for the coordinate-wise standard deviations.
    pub stdev_learning_rate: f64,
    /// Maximum relative standard-deviation change per update.
    pub stdev_max_change: f64,
    /// Adam first-moment decay coefficient.
    pub b1: f64,
    /// Adam second-moment decay coefficient.
    pub b2: f64,
    /// Adam numerical-stability constant.
    pub eps: f64,
    /// Multiplicative learning-rate decay coefficient.
    pub decay_coef: f64,
    /// Seed for the optimizer's independent random stream.
    pub seed: u64,
    /// Additional run identifier mixed into [`seed`](Self::seed).
    pub runid: i64,
}

impl Default for PgpeParams {
    fn default() -> Self {
        Self {
            popsize: 32,
            max_evaluations: 100_000,
            stop_fitness: f64::NEG_INFINITY,
            lr_decay_steps: 1000,
            use_ranking: true,
            center_learning_rate: 0.15,
            stdev_learning_rate: 0.1,
            stdev_max_change: 0.2,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            decay_coef: 1.0,
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

/// ADAM optimizer for the distribution center (the C++ `ADAM`).
struct Adam {
    x: DVector<f64>,
    m: DVector<f64>,
    v: DVector<f64>,
    b1: f64,
    b2: f64,
    eps: f64,
    center_lr: f64,
    decay_coef: f64,
}

impl Adam {
    fn new(x0: &DVector<f64>, b1: f64, b2: f64, eps: f64, center_lr: f64, decay_coef: f64) -> Self {
        let dim = x0.len();
        Adam {
            x: x0.clone(),
            m: DVector::zeros(dim),
            v: DVector::zeros(dim),
            b1,
            b2,
            eps,
            center_lr,
            decay_coef,
        }
    }

    fn step_size(&self, i: i32) -> f64 {
        self.center_lr * self.decay_coef.powi(i)
    }

    fn update(&mut self, i: i32, g: &DVector<f64>) {
        self.m = g * (1.0 - self.b1) + &self.m * self.b1;
        self.v = g.map(|v| v * v) * (1.0 - self.b2) + &self.v * self.b2;
        let bc1 = 1.0 / (1.0 - self.b1.powi(i + 1));
        let bc2 = 1.0 / (1.0 - self.b2.powi(i + 1));
        let mhat = &self.m * bc1;
        let vhat = &self.v * bc2;
        let step = self.step_size(i);
        let delta = DVector::from_iterator(
            self.x.len(),
            (0..self.x.len()).map(|k| step * mhat[k] / (vhat[k].sqrt() + self.eps)),
        );
        self.x -= delta;
    }
}

/// Stateful PGPE optimizer with batch and ask/tell evaluation interfaces.
pub struct Pgpe {
    fitfun: Fitness,
    rng: Rng,
    dim: usize,
    popsize: usize,
    max_evaluations: u64,
    stopfitness: f64,
    lr_decay_steps: i32,
    use_ranking: bool,
    stdev_learning_rate: f64,
    stdev_max_change: f64,

    adam: Adam,
    center: DVector<f64>,
    stdev: DVector<f64>,
    scaled_noises: Vec<DVector<f64>>, // n columns
    pop_x: Vec<DVector<f64>>,         // decoded population (popsize)

    best_x: DVector<f64>,
    best_y: f64,
    iterations: i32,
    stop: i32,
    external_evaluations: u64,
}

impl Pgpe {
    /// Create a PGPE distribution centered on `guess`.
    ///
    /// `input_sigma` supplies one initial standard deviation per dimension.
    pub fn new(mut fitfun: Fitness, guess: &[f64], input_sigma: &[f64], p: &PgpeParams) -> Self {
        let dim = fitfun.dim();
        fitfun.reset_evaluations();
        let mut popsize = if p.popsize > 0 {
            p.popsize as usize
        } else {
            4 * dim
        };
        if popsize % 2 == 1 {
            popsize += 1;
        }
        let center = DVector::from_vec(fitfun.encode(guess));
        let stdev = if input_sigma.len() == 1 {
            DVector::from_element(dim, input_sigma[0])
        } else {
            DVector::from_row_slice(input_sigma)
        };
        // ADAM optimizes the center in encoded space. (The C++ seeded ADAM with
        // the raw guess while `center` was encoded, so after the first tell the
        // center jumped to raw coordinates in normalized space — a latent
        // inconsistency; seeding ADAM with the encoded center fixes it.)
        let adam = Adam::new(
            &center,
            p.b1,
            p.b2,
            p.eps,
            p.center_learning_rate,
            p.decay_coef,
        );
        Pgpe {
            dim,
            popsize,
            max_evaluations: if p.max_evaluations > 0 {
                p.max_evaluations
            } else {
                50_000
            },
            stopfitness: p.stop_fitness,
            lr_decay_steps: p.lr_decay_steps.max(1),
            use_ranking: p.use_ranking,
            stdev_learning_rate: p.stdev_learning_rate.abs(),
            stdev_max_change: p.stdev_max_change.abs(),
            adam,
            center,
            stdev,
            scaled_noises: vec![],
            pop_x: vec![DVector::zeros(dim); popsize],
            best_x: DVector::zeros(dim),
            best_y: f64::MAX,
            iterations: 0,
            stop: 0,
            external_evaluations: 0,
            rng: Rng::new(p.seed.wrapping_add(p.runid as u64)),
            fitfun,
        }
    }

    /// Number of decision variables.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Number of mirrored samples in each population.
    pub fn popsize(&self) -> usize {
        self.popsize
    }
    /// Current termination code, or zero while the optimizer can continue.
    pub fn stop(&self) -> i32 {
        self.stop
    }

    /// Symmetric sampling: returns `popsize` *encoded* candidates, interleaved
    /// `[center+n0, center-n0, center+n1, center-n1, ...]`, storing the noises.
    fn ask_encoded(&mut self) -> Vec<DVector<f64>> {
        let n = self.popsize / 2;
        self.scaled_noises = (0..n)
            .map(|_| {
                let noise =
                    DVector::from_iterator(self.dim, (0..self.dim).map(|_| self.rng.gaussian()));
                noise.component_mul(&self.stdev)
            })
            .collect();
        let mut xs = Vec::with_capacity(self.popsize);
        for p in 0..n {
            xs.push(&self.center + &self.scaled_noises[p]);
            xs.push(&self.center - &self.scaled_noises[p]);
        }
        xs
    }

    /// Decoded, in-bounds population (rows), stored for the reinforce update.
    fn ask_pop_internal(&mut self) -> Vec<Vec<f64>> {
        let xs = self.ask_encoded();
        self.pop_x = xs
            .iter()
            .map(|c| {
                let feasible = self.fitfun.closest_feasible_normed(c.as_slice());
                DVector::from_vec(self.fitfun.decode(&feasible))
            })
            .collect();
        self.pop_x.iter().map(|c| c.as_slice().to_vec()).collect()
    }

    fn process_scores(&self, ys: &[f64]) -> DVector<f64> {
        if self.use_ranking {
            let n = ys.len();
            let order = sort_index(ys);
            let mut ranks = DVector::zeros(n);
            for (i, &idx) in order.iter().enumerate() {
                ranks[idx] = i as f64 / n as f64 - 0.5;
            }
            ranks
        } else {
            DVector::from_row_slice(ys)
        }
    }

    /// grad_center, grad_stdev from the REINFORCE estimator (the C++
    /// `compute_reinforce_update`).
    fn reinforce(&self, pop_y: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let n = self.popsize / 2;
        let mean_all = pop_y.mean();
        let mut grad_center = DVector::zeros(self.dim);
        let mut grad_stdev = DVector::zeros(self.dim);
        for i in 0..self.dim {
            let mut gc = 0.0;
            let mut gs = 0.0;
            for p in 0..n {
                let fit1 = pop_y[2 * p];
                let fit2 = pop_y[2 * p + 1];
                let score = fit1 - fit2;
                let avg = 0.5 * (fit1 + fit2);
                let sn = self.scaled_noises[p][i];
                gc += sn * score * 0.5;
                gs += (avg - mean_all) * (sn * sn - self.stdev[i] * self.stdev[i]) / self.stdev[i];
            }
            grad_center[i] = gc / n as f64;
            grad_stdev[i] = gs / n as f64;
        }
        (grad_center, grad_stdev)
    }

    fn update_stdev(&self, grad: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(
            self.dim,
            (0..self.dim).map(|i| {
                let allowed = self.stdev[i].abs() * self.stdev_max_change;
                let lo = self.stdev[i] - allowed;
                let hi = self.stdev[i] + allowed;
                (self.stdev[i] + self.stdev_learning_rate * grad[i]).clamp(lo, hi)
            }),
        )
    }

    fn tell(&mut self, ys: &[f64]) -> i32 {
        let neg: Vec<f64> = ys.iter().map(|y| -y).collect();
        let pop_y = self.process_scores(&neg);
        // Track the *true* best fitness/point. (The C++ tracked `-max(process_
        // scores(-ys))`, which under ranking is a rank value, not a fitness, so
        // its reported best was unusable; using the raw ys is correct and keeps
        // the result meaningful for retry/comparison.)
        let mut best_p = 0;
        for p in 1..self.popsize {
            if ys[p] < ys[best_p] {
                best_p = p;
            }
        }
        if ys[best_p] < self.best_y {
            self.best_y = ys[best_p];
            self.best_x = self.pop_x[best_p].clone();
            if self.best_y < self.stopfitness {
                self.stop = 1;
            }
        }
        let (grad_center, grad_stdev) = self.reinforce(&pop_y);
        self.adam
            .update(self.iterations / self.lr_decay_steps, &(-grad_center));
        self.iterations += 1;
        self.center = self.adam.x.clone();
        self.stdev = self.update_stdev(&grad_stdev);
        self.stop
    }

    fn make_result(&self, evaluations: u64) -> PgpeResult {
        PgpeResult {
            x: self.best_x.as_slice().to_vec(),
            y: self.best_y,
            evaluations,
            iterations: self.iterations,
            stop: self.stop,
        }
    }

    /// Generational loop evaluating each population through a batch closure.
    pub fn optimize_batch<F>(&mut self, mut eval_batch: F) -> PgpeResult
    where
        F: FnMut(&[Vec<f64>]) -> Vec<f64>,
    {
        self.iterations = 0;
        self.fitfun.reset_evaluations();
        while self.fitfun.evaluations() < self.max_evaluations
            && !self.fitfun.terminate()
            && self.stop == 0
        {
            let rows = self.ask_pop_internal();
            let mut ys = eval_batch(&rows);
            for v in ys.iter_mut() {
                if !v.is_finite() {
                    *v = crate::fitness::NAN_REPLACEMENT;
                }
            }
            self.fitfun.incr_evaluations(self.popsize as u64);
            self.tell(&ys);
        }
        self.make_result(self.fitfun.evaluations())
    }

    // ---- ask/tell interface (mirrors PgpeState::Impl) ----

    /// Sample and return one decoded population.
    pub fn ask_pop(&mut self) -> Vec<Vec<f64>> {
        self.ask_pop_internal()
    }

    /// Update the distribution from objective values corresponding to
    /// [`ask_pop`](Self::ask_pop), returning the current stop code.
    pub fn tell_pop(&mut self, ys: &[f64]) -> i32 {
        let stop = self.tell(ys);
        self.external_evaluations += ys.len() as u64;
        stop
    }

    /// Return the most recently sampled decoded population.
    pub fn population(&self) -> Vec<Vec<f64>> {
        self.pop_x.iter().map(|c| c.as_slice().to_vec()).collect()
    }

    /// Return the current best result for an ask/tell run.
    pub fn result(&self) -> PgpeResult {
        self.make_result(self.external_evaluations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    #[test]
    fn converges_on_sphere() {
        // use_ranking=false so best_y is the true fitness.
        let mut fit = Fitness::bounded(5, 1, &[-5.0; 5], &[5.0; 5]);
        fit.set_normalize(true);
        let params = PgpeParams {
            popsize: 40,
            max_evaluations: 20000,
            use_ranking: false,
            seed: 1,
            ..Default::default()
        };
        let mut opt = Pgpe::new(fit, &[3.0; 5], &[0.5; 5], &params);
        let r = opt.optimize_batch(|rows| rows.iter().map(|x| sphere(x)).collect());
        assert!(r.y < 1e-2, "pgpe did not converge: {}", r.y);
    }

    #[test]
    fn ask_tell_best_x_is_real_point() {
        let mut fit = Fitness::bounded(4, 1, &[-5.0; 4], &[5.0; 4]);
        fit.set_normalize(true);
        let params = PgpeParams {
            popsize: 32,
            use_ranking: false,
            seed: 2,
            ..Default::default()
        };
        let mut opt = Pgpe::new(fit, &[2.0; 4], &[0.5; 4], &params);
        for _ in 0..400 {
            let pop = opt.ask_pop();
            let ys: Vec<f64> = pop.iter().map(|x| sphere(x)).collect();
            opt.tell_pop(&ys);
        }
        let r = opt.result();
        // best_x must evaluate close to the reported best value.
        assert!((sphere(&r.x) - r.y).abs() < 1e-6 || r.y < 1e-2);
        assert!(sphere(&r.x) < 1e-1, "best_x not good: {}", sphere(&r.x));
    }

    #[test]
    fn ranking_defaults_odd_population_getters_and_nonfinite_scores() {
        let mut fit = Fitness::bounded(2, 1, &[-1.0; 2], &[1.0; 2]);
        fit.set_normalize(true);
        let params = PgpeParams {
            popsize: 3,
            max_evaluations: 0,
            stop_fitness: 1.0e100,
            lr_decay_steps: 0,
            use_ranking: true,
            stdev_learning_rate: -0.1,
            stdev_max_change: -0.2,
            seed: 14,
            ..Default::default()
        };
        let mut optimizer = Pgpe::new(fit, &[0.0; 2], &[0.25], &params);
        assert_eq!(optimizer.dim(), 2);
        assert_eq!(optimizer.popsize(), 4);
        assert_eq!(optimizer.stop(), 0);
        let result = optimizer.optimize_batch(|rows| vec![f64::NAN; rows.len()]);
        assert_eq!(result.evaluations, 4);
        assert_eq!(result.stop, 1);
        assert_eq!(optimizer.population().len(), 4);
        assert_eq!(optimizer.stop(), 1);
    }
}
