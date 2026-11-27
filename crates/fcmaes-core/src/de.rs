//! Differential Evolution (DE) for bounded, derivative-free optimization.
//!
//! This implementation uses DE/best/1 with fcmaes extensions: temporal locality
//! (an extra
//! improvement trial along the previous move), age-based reinitialization of
//! stale individuals, oscillating `F`/`CR` between generations, optional
//! normal-distributed sampling around a guess, and mixed-integer "modify"
//! resampling. It is implemented entirely in Rust; historical implementations
//! informed compatibility testing, not the public API.
//!
//! # Reference
//!
//! R. Storn and K. Price, [“Differential Evolution – A Simple and Efficient
//! Heuristic for Global Optimization over Continuous
//! Spaces”](https://doi.org/10.1023/A:1008202821328), *Journal of Global
//! Optimization* 11, 341–359 (1997).
//!
//! # Example
//!
//! ```
//! use fcmaes_core::{De, DeParams, Fitness};
//!
//! let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
//! let fit = Fitness::bounded(3, 1, &[-5.0; 3], &[5.0; 3]);
//! let params = DeParams {
//!     max_evaluations: 2_000,
//!     seed: 42,
//!     ..Default::default()
//! };
//! let mut de = De::new(fit, &[], &[], None, &params);
//! let result = de.optimize(&sphere);
//! assert!(result.y.is_finite());
//! ```

use std::collections::VecDeque;

use crate::fitness::{Fitness, Objective};
use crate::rng::Rng;

/// Outcome of a Differential Evolution run.
#[derive(Clone, Debug)]
pub struct DeResult {
    /// The cooperative stop predicate fired; every dispatched point is counted.
    pub stopped: bool,
    /// Best evaluated decision vector. For external ask/tell, this is the
    /// corresponding row from `ask`.
    pub x: Vec<f64>,
    /// Best finite value observed, or `f64::MAX` until there is one.
    pub y: f64,
    /// Objective evaluations actually made; the serial
    /// [`optimize`](De::optimize) can exceed [`DeParams::max_evaluations`]
    /// (see there).
    pub evaluations: u64,
    /// Number of completed generations.
    pub iterations: i32,
    /// Termination code; `1` means `stop_fitness` was reached.
    pub stop: i32,
}

/// Tunable inputs for [`De::new`]. Non-positive values select DE defaults.
#[derive(Clone, Debug)]
pub struct DeParams {
    /// Population size; non-positive values select the dimension-dependent default.
    pub popsize: i32,
    /// Evaluation budget. [`optimize_delayed`](De::optimize_delayed) never
    /// exceeds it. The serial [`optimize`](De::optimize) checks it before each
    /// generation, and a generation costs one or two evaluations per member,
    /// so a run can make up to `2 * popsize - 1` more evaluations;
    /// [`DeResult::evaluations`] reports the actual count.
    pub max_evaluations: u64,
    /// Expected lifetime controlling age-based population reinitialization.
    pub keep: f64,
    /// Stop after finding an objective value strictly below this threshold.
    pub stop_fitness: f64,
    /// Base differential weight `F`.
    pub f: f64,
    /// Base crossover probability `CR`.
    pub cr: f64,
    /// Minimum probability used by mixed-integer mutation.
    pub min_mutate: f64,
    /// Maximum probability used by mixed-integer mutation.
    pub max_mutate: f64,
    /// Lower limit for normal-sampling step sizes.
    pub min_sigma: f64,
    /// Seed for the optimizer's independent random stream.
    pub seed: u64,
    /// Additional run identifier mixed into [`seed`](Self::seed).
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

/// Stateful Differential Evolution optimizer.
///
/// Use [`optimize`](Self::optimize) for an in-process objective or
/// [`ask`](Self::ask)/[`tell`](Self::tell) when another system evaluates a
/// population.
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
    /// Whether `asked_x` holds a batch from [`De::ask`] not told yet.
    pending: bool,
    external_evaluations: u64,
}

const DE_NOT_PENDING: &str = "DE tell called without a pending ask batch";
const DE_BATCH_LENGTH: &str = "DE tell batch length must equal popsize";

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
            best_y: f64::INFINITY,
            iterations: 0,
            stop: 0,
            pos: 0,
            improves_x: VecDeque::new(),
            improves_p: VecDeque::new(),
            asked_x: vec![],
            asked_p: vec![],
            pending: false,
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

    /// Number of decision variables.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Number of individuals evaluated per generation.
    pub fn popsize(&self) -> usize {
        self.popsize
    }
    /// Current termination code, or zero while the optimizer can continue.
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

    fn record_best(&mut self, y: f64, x: &[f64], decode: bool) -> bool {
        if y.is_finite() && y < self.best_y {
            self.best_y = y;
            if decode {
                self.best_x.copy_from_slice(&self.fitfun.decode_clamped(x));
            } else {
                self.best_x.copy_from_slice(x);
            }
            if self.stopfitness.is_finite() && y < self.stopfitness {
                self.stop = 1;
                return true;
            }
        }
        false
    }

    fn tell_one(&mut self, y: f64, x: Vec<f64>, p: usize, decode: bool) -> i32 {
        self.record_best(y, &x, decode);
        if y.is_finite() && y < self.pop_y[p] {
            if self.iterations > 1 {
                let xb = self.pop_x[self.best_i].clone();
                let xi = self.pop_x0[p].clone();
                let improved = self.next_improve(&xb, &x, &xi);
                self.improves_p.push_back(p);
                self.improves_x.push_back(improved);
            }
            self.pop_x0[p] = std::mem::replace(&mut self.pop_x[p], x);
            self.pop_y[p] = y;
            self.pop_iter[p] = self.iterations;
            if y < self.pop_y[self.best_i] {
                self.best_i = p;
            }
        } else if self.keep * self.rng.uniform01() < (self.iterations - self.pop_iter[p]) as f64 {
            self.pop_x[p] = self.sample();
            self.pop_y[p] = f64::MAX;
        }
        self.stop
    }

    fn evaluate(&mut self, x: &[f64], obj: &impl Objective) -> f64 {
        let y = obj.eval_scalar(&self.fitfun.decode_clamped(x));
        self.fitfun.incr_evaluations(1);
        y
    }

    /// Run the serial generational loop against an in-process objective.
    ///
    /// The budget is checked before each generation, so the run can overshoot
    /// [`DeParams::max_evaluations`] by up to `2 * popsize - 1` evaluations;
    /// use [`optimize_delayed`](De::optimize_delayed) for an exact limit.
    pub fn optimize(&mut self, obj: &impl Objective) -> DeResult {
        self.optimize_until(obj, &|| false)
    }

    /// Serial DE with a cooperative stop before each evaluation, including
    /// temporal-locality trials. The usual generational budget is unchanged.
    pub fn optimize_until(&mut self, obj: &impl Objective, stop: &dyn Fn() -> bool) -> DeResult {
        self.iterations = 1;
        self.fitfun.reset_evaluations();
        while self.fitfun.evaluations() < self.max_evaluations && !self.fitfun.terminate() {
            if self.iterations > 2 {
                self.update_mean();
            }
            self.oscillate();
            for p in 0..self.popsize {
                if stop() {
                    let mut result = self.make_result(self.fitfun.evaluations());
                    result.stopped = true;
                    return result;
                }
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
                let mut y = self.evaluate(&x, obj);
                let mut was_stopped = false;
                let accepted = y.is_finite() && y < self.pop_y[p];
                if accepted {
                    // Temporal locality: an extra trial along the last move.
                    let trial = self.next_improve(&xb, &x, &xp);
                    was_stopped = stop();
                    if !was_stopped {
                        let score = self.evaluate(&trial, obj);
                        if score.is_finite() && score < y {
                            y = score;
                            x = trial;
                        }
                    }
                }
                let hit_target = self.record_best(y, &x, true);
                if accepted {
                    self.pop_x[p] = x;
                    self.pop_y[p] = y;
                    self.pop_iter[p] = self.iterations;
                    if y < self.pop_y[self.best_i] {
                        self.best_i = p;
                    }
                } else if self.keep * self.rng.uniform01()
                    < (self.iterations - self.pop_iter[p]) as f64
                {
                    self.pop_x[p] = self.sample();
                    self.pop_y[p] = f64::MAX;
                }
                if was_stopped || hit_target {
                    let mut result = self.make_result(self.fitfun.evaluations());
                    result.stopped = was_stopped;
                    return result;
                }
            }
            self.iterations += 1;
        }
        self.make_result(self.fitfun.evaluations())
    }

    /// Delayed-update DE with `workers` candidates in flight, the parallel
    /// mode of the Python fcmaes `de.minimize(..., workers=...)`
    /// (`do_optimize_delayed_update`): `workers` candidates are asked and
    /// evaluated concurrently on the current Rayon pool; whenever the oldest
    /// one is evaluated, it is told and one new candidate is asked.
    ///
    /// Unlike the Python version, results are told in the order the
    /// candidates were asked, not in completion order, so the run depends on
    /// `workers` and the seed only, never on thread timing; candidates still
    /// in flight when the budget is spent are told too. At most
    /// `max_evaluations` are spent. With one worker this is the ask/tell
    /// variant of the serial [`optimize`](De::optimize), which updates the
    /// population after every evaluation and evaluates the temporal-locality
    /// trial immediately.
    ///
    /// Run it inside the Rayon pool that should evaluate (`ThreadPool::install`);
    /// outside any pool, evaluations run on Rayon's global pool.
    ///
    /// # Panics
    ///
    /// Resumes a panic of the objective in the caller.
    pub fn optimize_delayed(&mut self, obj: &impl Objective, workers: usize) -> DeResult {
        self.optimize_delayed_until(obj, workers, &|| false)
    }

    /// Delayed DE with a cooperative stop before dispatch. At most `workers`
    /// evaluations are in flight; all are drained and told in ask order.
    ///
    /// # Panics
    /// Resumes an objective panic in the caller.
    pub fn optimize_delayed_until(
        &mut self,
        obj: &impl Objective,
        workers: usize,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> DeResult {
        use std::sync::mpsc;
        type Evaluation = (usize, usize, Vec<f64>, std::thread::Result<f64>);

        let budget = self.max_evaluations;
        let window = budget.min(workers.max(1) as u64) as usize;
        // Workers own their candidate until they return it. The normal
        // (already feasible, unnormalized) objective can borrow that row.
        let fitness = self.fitfun.clone();
        let (sender, receiver) = mpsc::channel::<Evaluation>();
        let mut finished: Vec<Option<(usize, Vec<f64>, f64)>> = vec![None; window];
        let (mut dispatched, mut told) = (0u64, 0u64);
        rayon::in_place_scope(|scope| {
            let dispatch = |de: &mut Self, dispatched: &mut u64| {
                let (p, x) = de.ask_one();
                let slot = (*dispatched % window as u64) as usize;
                let sender = sender.clone();
                let fitness = &fitness;
                scope.spawn(move |_| {
                    let y = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        obj.eval_scalar(&fitness.decode_clamped(&x))
                    }));
                    let _ = sender.send((slot, p, x, y));
                });
                *dispatched += 1;
            };

            while dispatched < window as u64 && !stop() {
                dispatch(self, &mut dispatched);
            }
            while told < dispatched {
                let oldest = (told % window as u64) as usize;
                let (p, x, y) = loop {
                    if let Some(result) = finished[oldest].take() {
                        break result;
                    }
                    // Let this worker run a queued objective if the caller
                    // occupied a slot in the same pool.
                    let (slot, p, x, result) = match receiver.try_recv() {
                        Ok(result) => result,
                        Err(_) => match rayon::yield_now() {
                            Some(rayon::Yield::Executed) => continue,
                            _ => receiver.recv().expect("an evaluation is in flight"),
                        },
                    };
                    let y = match result {
                        Ok(y) => y,
                        Err(panic) => std::panic::resume_unwind(panic),
                    };
                    debug_assert!(finished[slot].is_none());
                    finished[slot] = Some((p, x, y));
                };
                self.tell_one(y, x, p, true);
                told += 1;
                if self.stop == 0 && dispatched < budget && !self.fitfun.terminate() && !stop() {
                    dispatch(self, &mut dispatched);
                }
            }
        });
        self.external_evaluations += told;
        let mut result = self.make_result(told);
        result.stopped = stop();
        result
    }

    fn make_result(&self, evaluations: u64) -> DeResult {
        DeResult {
            stopped: false,
            x: self.best_x.clone(),
            y: self.best_y.min(f64::MAX),
            evaluations,
            iterations: self.iterations,
            stop: self.stop,
        }
    }

    // ---- ask/tell interface (mirrors DeState::Impl) ----

    /// Ask for a full population of `popsize` candidate rows.
    ///
    /// The batch stays pending until it is told. Calling `ask` again before
    /// that draws a new batch that replaces the pending one; the replaced
    /// batch can no longer be told.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        for i in 0..self.popsize {
            let (p, x) = self.ask_one();
            self.asked_p[i] = p;
            self.asked_x[i] = x;
        }
        self.pending = true;
        self.asked_x.clone()
    }

    /// Tell one fitness value per row of the batch returned by
    /// [`ask`](De::ask), in the same order. Returns the termination code.
    ///
    /// # Panics
    ///
    /// Panics if no batch is pending or if `ys` does not have `popsize`
    /// values. Use [`De::try_tell`] to receive these as errors instead.
    pub fn tell(&mut self, ys: &[f64]) -> i32 {
        self.try_tell(ys).expect("invalid DE tell call")
    }

    /// Fallible tell variant validating call order and batch length. A
    /// rejected call changes no optimizer state.
    ///
    /// # Errors
    ///
    /// Returns an error if no batch is pending (no [`ask`](De::ask) since the
    /// last successful tell) or if `ys` does not have `popsize` values.
    pub fn try_tell(&mut self, ys: &[f64]) -> Result<i32, &'static str> {
        if !self.pending {
            return Err(DE_NOT_PENDING);
        }
        if ys.len() != self.popsize {
            return Err(DE_BATCH_LENGTH);
        }
        self.pending = false;
        for (i, &y) in ys.iter().enumerate() {
            let x = std::mem::take(&mut self.asked_x[i]);
            self.tell_one(y, x, self.asked_p[i], false);
        }
        self.external_evaluations += ys.len() as u64;
        Ok(self.stop)
    }

    /// Return a decoded snapshot of the current population.
    pub fn population(&self) -> Vec<Vec<f64>> {
        self.pop_x.clone()
    }

    /// Return the current best result for an ask/tell run.
    pub fn result(&self) -> DeResult {
        self.make_result(self.external_evaluations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_driver_records_the_first_finite_result_even_on_a_plateau() {
        let costs = |_: &[f64]| 7.0;
        let mut serial = de(5);
        serial.stopfitness = 8.0;
        let direct = serial.optimize(&costs);
        assert_eq!((direct.y, direct.stop), (7.0, 1));
        assert_eq!(costs(&direct.x), direct.y);

        let mut batch = de(5);
        batch.stopfitness = 8.0;
        let rows = batch.ask();
        assert_eq!(batch.tell(&vec![7.0; rows.len()]), 1);
        assert_eq!(batch.result().x, rows[0]);
        assert_eq!(batch.result().y, 7.0);

        let mut delayed = de(5);
        delayed.stopfitness = 8.0;
        let run = delayed.optimize_delayed(&costs, 8);
        assert_eq!((run.y, run.evaluations, run.stop), (7.0, 8, 1));
        assert_eq!(costs(&run.x), run.y);
    }

    #[test]
    fn failed_callbacks_cannot_replace_a_finite_incumbent() {
        let failed = |_: &[f64]| f64::NAN;
        let mut serial = de(9);
        serial.max_evaluations = 31;
        serial.stopfitness = f64::MAX;
        let run = serial.optimize(&failed);
        assert_eq!(run.y, f64::MAX);
        assert_eq!(run.stop, 0);

        let mut delayed = de(9);
        delayed.max_evaluations = 31;
        delayed.stopfitness = f64::MAX;
        let run = delayed.optimize_delayed(&failed, 8);
        assert_eq!((run.y, run.evaluations, run.stop), (f64::MAX, 31, 0));

        let mut batch = de(9);
        let rows = batch.ask();
        let mut costs = vec![f64::INFINITY; rows.len()];
        costs[0] = f64::MAX;
        batch.tell(&costs);
        assert_eq!(batch.result().x, rows[0]);
        let _ = batch.ask();
        batch.tell(&vec![f64::NEG_INFINITY; rows.len()]);
        assert_eq!(batch.result().x, rows[0]);
    }

    #[test]
    fn cancellation_before_the_locality_trial_returns_the_observed_point() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let objective = |x: &[f64]| {
            calls.fetch_add(1, Ordering::Relaxed);
            sphere(x)
        };
        let mut optimizer = de(2);
        optimizer.fitfun.set_normalize(true);
        let r = optimizer.optimize_until(&objective, &|| calls.load(Ordering::Relaxed) == 1);
        assert!(r.stopped);
        assert_eq!((r.evaluations, calls.load(Ordering::Relaxed)), (1, 1));
        assert_eq!(r.y, sphere(&r.x));
    }

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

    fn de(seed: u64) -> De {
        let fit = Fitness::new(6, 1, vec![-5.0; 6], vec![5.0; 6]);
        let params = DeParams {
            popsize: 16,
            max_evaluations: 20_000,
            seed,
            ..Default::default()
        };
        De::new(fit, &[], &[], None, &params)
    }

    fn delayed(seed: u64, workers: usize, threads: usize) -> DeResult {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| de(seed).optimize_delayed(&rosen, workers))
    }

    /// The delayed update without threads: ask `workers`, then tell the
    /// oldest and ask one more, until the budget is dispatched; tell the rest.
    fn delayed_reference(seed: u64, workers: usize) -> DeResult {
        let mut de = de(seed);
        let mut asked = VecDeque::new();
        let mut dispatched = 0;
        while dispatched < workers.min(20_000) {
            asked.push_back(de.ask_one());
            dispatched += 1;
        }
        let mut told = 0;
        while let Some((p, x)) = asked.pop_front() {
            de.tell_one(rosen(&x), x, p, true);
            told += 1;
            if de.stop == 0 && dispatched < 20_000 {
                asked.push_back(de.ask_one());
                dispatched += 1;
            }
        }
        de.make_result(told)
    }

    #[test]
    fn delayed_updates_tell_in_ask_order_whatever_the_threads() {
        for workers in [1, 7, 32] {
            let reference = delayed_reference(3, workers);
            for threads in [1, 4] {
                let r = delayed(3, workers, threads);
                assert_eq!(r.x, reference.x, "{workers} workers, {threads} threads");
                assert_eq!(r.y, reference.y);
                assert_eq!(r.evaluations, 20_000);
            }
            assert_eq!(rosen(&reference.x), reference.y);
        }
        // Delayed feedback still converges: the median of five seeds.
        for workers in [8, 32] {
            let mut ys: Vec<f64> = (0..5).map(|seed| delayed(seed, workers, 4).y).collect();
            ys.sort_by(f64::total_cmp);
            assert!(ys[2] < 1e-3, "{workers} workers: {ys:?}");
        }
    }

    #[test]
    #[should_panic(expected = "objective failed")]
    fn a_panicking_objective_fails_the_delayed_run() {
        let failing = |_: &[f64]| -> f64 { panic!("objective failed") };
        de(1).optimize_delayed(&failing, 4);
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

    fn summary(r: &DeResult) -> (Vec<f64>, f64, u64, i32, i32) {
        (r.x.clone(), r.y, r.evaluations, r.iterations, r.stop)
    }

    /// One ask/tell pair is one batch of the `ask_one`/`tell_one` sequence the
    /// interface is built from (the serial `optimize` evaluates its
    /// temporal-locality trials inline and so follows a different path).
    #[test]
    fn ask_tell_equals_the_generation_reference() {
        let mut reference = de(4);
        let mut driven = de(4);
        for _ in 0..40 {
            let batch: Vec<(usize, Vec<f64>)> = (0..reference.popsize)
                .map(|_| reference.ask_one())
                .collect();
            let xs = driven.ask();
            assert_eq!(xs, batch.iter().map(|(_, x)| x.clone()).collect::<Vec<_>>());
            let ys: Vec<f64> = xs.iter().map(|x| rosen(x)).collect();
            for ((p, x), &y) in batch.iter().zip(&ys) {
                reference.tell_one(y, x.clone(), *p, false);
            }
            assert_eq!(driven.tell(&ys), reference.stop);
        }
        assert_eq!(driven.population(), reference.population());
        assert_eq!(
            summary(&driven.result()),
            summary(&reference.make_result(40 * 16))
        );
    }

    /// Rejected tells (none pending, short, long, repeated) change nothing:
    /// the run continues exactly as one without them.
    #[test]
    fn rejected_tells_leave_the_state_unchanged() {
        let run = |reject: bool| {
            let mut de = de(5);
            if reject {
                assert_eq!(de.try_tell(&[]), Err(DE_NOT_PENDING));
                assert_eq!(de.try_tell(&[1.0; 16]), Err(DE_NOT_PENDING));
            }
            let mut asked = Vec::new();
            for _ in 0..5 {
                let xs = de.ask();
                let ys: Vec<f64> = xs.iter().map(|x| rosen(x)).collect();
                if reject {
                    assert_eq!(de.try_tell(&ys[..15]), Err(DE_BATCH_LENGTH));
                    let long = [ys.as_slice(), &[0.0]].concat();
                    assert_eq!(de.try_tell(&long), Err(DE_BATCH_LENGTH));
                    assert_eq!(de.try_tell(&[]), Err(DE_BATCH_LENGTH));
                }
                assert_eq!(de.try_tell(&ys), Ok(0));
                if reject {
                    assert_eq!(de.try_tell(&ys), Err(DE_NOT_PENDING));
                }
                asked.push(xs);
            }
            asked.push(de.ask());
            (asked, de.population(), summary(&de.result()))
        };
        assert_eq!(run(true), run(false));
    }

    #[test]
    #[should_panic(expected = "DE tell batch length must equal popsize")]
    fn tell_panics_on_a_short_batch() {
        let mut de = de(1);
        let xs = de.ask();
        de.tell(&vec![0.0; xs.len() - 1]);
    }

    #[test]
    #[should_panic(expected = "DE tell batch length must equal popsize")]
    fn tell_panics_on_a_long_batch() {
        let mut de = de(1);
        let xs = de.ask();
        de.tell(&vec![0.0; xs.len() + 1]);
    }

    #[test]
    #[should_panic(expected = "DE tell called without a pending ask batch")]
    fn tell_panics_without_a_pending_ask() {
        let mut de = de(1);
        let xs = de.ask();
        de.tell(&vec![0.0; xs.len()]);
        de.tell(&vec![0.0; xs.len()]);
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
