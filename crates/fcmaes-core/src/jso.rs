//! jSO: adaptive Differential Evolution with linear population size
//! reduction, a descendant of L-SHADE and iL-SHADE.
//!
//! jSO keeps a memory of successful scaling factors `F` and crossover rates
//! `CR`, mutates with a weighted current-to-pbest strategy and an archive of
//! replaced parents, and shrinks its population linearly from
//! `round(25 · ln(D) · √D)` to four individuals over the evaluation budget.
//! Because the population schedule, the p-best fraction and the parameter
//! stages are all tied to the budget, a run needs a finite evaluation budget
//! from the start, and its initial population is uniform in a finite box.
//!
//! This is an independent implementation written from the published
//! description. Three details follow the observable behavior of the authors'
//! published CEC 2017 reference program, which was consulted for facts only
//! and from which no code is derived: the memory index cycles over all five
//! cells with the last cell fixed at 0.9; the p-best index is redrawn while
//! it equals the target in the first half of the run; and a terminal
//! crossover-rate cell is not permanent.
//!
//! # When to use it
//!
//! As a first stage before CMA-ES, alongside DE. In the candidate-optimizer
//! benchmark of this repository
//! ([decision](https://github.com/dietmarwo/fcmaes-rust/blob/main/benchmarks/candidate-optimizers/DECISION.md)), the
//! jSO→CMA sequence ranked about as well as the DE→CMA sequence over ten
//! problems, with a different profile: better on GTOC1 and Tandem, worse on
//! Cassini1, Sagas and Messenger full. jSO alone lost most cells against the
//! existing optimizers in both rounds of the benchmark.
//!
//! # Reference
//!
//! J. Brest, M. S. Maučec and B. Bošković, [“Single objective real-parameter
//! optimization: algorithm jSO”](https://doi.org/10.1109/CEC.2017.7969456),
//! *Proc. IEEE Congress on Evolutionary Computation (CEC 2017)*, 1311–1318.
//!
//! # Algorithm (differences from L-SHADE)
//!
//! - `N_init = round(25 · ln(D) · √D)` ([`JsoParams::popsize`]),
//!   `N_min = 4` ([`JsoParams::min_popsize`]), linear reduction over the
//!   budget.
//! - Memory of `H = 5` cells ([`JsoParams::memory_size`]), initialized to
//!   `M_F = 0.3`, `M_CR = 0.8`. Sampling the last cell always yields
//!   `M_F = M_CR = 0.9`. `F` is drawn from a Cauchy distribution with scale
//!   0.1 (redrawn while not positive, truncated to 1), `CR` from a normal
//!   distribution with standard deviation 0.1, clipped to `[0, 1]`.
//! - `p = 0.25 · (1 - 0.5 · nfes / max_nfes)`, decreasing linearly from
//!   `p_max = 0.25` to `p_min = 0.125` ([`JsoParams::p_max`],
//!   [`JsoParams::p_min`]); at least two individuals form the p-best set.
//!   Archive capacity `round(1.0 · N)` ([`JsoParams::archive_rate`]).
//! - Mutation current-to-pbest-w/1 with archive:
//!   `v = x_i + Fw (x_pbest - x_i) + F (x_r1 - x_r2)`, with `Fw = 0.7 F` while
//!   `nfes < 0.2 max_nfes`, `0.8 F` while `nfes < 0.4 max_nfes`, else `1.2 F`.
//!   While `nfes < 0.5 max_nfes`, `pbest` is redrawn until it differs from `i`.
//! - `F` is capped at 0.7 while `nfes < 0.6 max_nfes`; `CR = max(CR, 0.7)`
//!   while `nfes < 0.25 max_nfes`, `CR = max(CR, 0.6)` while
//!   `nfes < 0.5 max_nfes`.
//! - Memory update of cell `k` (cyclic over all `H` cells):
//!   `M_F[k] = (M_F[k] + mean_WL(S_F)) / 2` and
//!   `M_CR[k] = (M_CR[k] + c) / 2` with `c = -1` if `max(S_CR) = 0`, otherwise
//!   `c = mean_WL(S_CR)`. A negative `M_CR` cell samples `CR = 0`.
//!
//! # Deviations and unspecified details
//!
//! - **Fixed cell inside `H`.** The paper sets `M_F,H = M_CR,H = 0.9` for the
//!   last of the `H = 5` cells; it is not an additional sixth cell. As in the
//!   reference program, the cyclic update index runs over all `H` cells, so
//!   an update that lands on the last cell has no effect.
//! - **Initial `M_F`.** Section IV-B of the paper gives 0.3, Algorithm 1
//!   line 3 gives 0.5; 0.3 (the stated parameter setting) is used.
//! - **Direction of `p`.** The prose and the reference program decrease `p`
//!   from `p_max` to `p_min`; only the printed Eq. (1) increases. The
//!   decreasing schedule is used.
//! - **`pbest` different from `i`.** Not in the paper; from the reference
//!   program.
//! - **Terminal `CR`.** Algorithm 1 sets `CR = 0` for a negative memory value.
//!   The terminal value `-1` is averaged into the cell like any other mean
//!   (as in the reference program), so it is not permanent, unlike L-SHADE.
//! - **Stage clock.** Algorithm 1 phrases the `F` and `CR` stages with the
//!   generation ratio `g / G_max`, which is undefined under population size
//!   reduction; the evaluation ratio `nfes / max_nfes` is used, as for `Fw`.
//!   All stage decisions use the evaluation count at the start of the
//!   generation.
//! - **Bound handling.** The paper only says a repair is applied; the
//!   midpoint-target repair of L-SHADE is used: a coordinate outside the box
//!   moves halfway between the violated bound and the parent's coordinate.
//! - **Archive rate and `N_min`.** Not stated in the paper ("kept unchanged"
//!   from L-SHADE/iL-SHADE); 1.0 and 4. When the archive is full, a new
//!   parent overwrites a random member.
//! - **Recorded values.** `S_F` stores the capped `F` (not `Fw`), `S_CR` the
//!   floored `CR`. Successes are weighted by their fitness improvement
//!   (weighted Lehmer mean); an infinite improvement (a finite trial
//!   replacing a parent without a finite value) takes all the weight.
//! - **Selection.** A trial replaces its parent when it is not worse; only a
//!   strict improvement records `F` and `CR` and archives the parent.
//! - **Non-finite values.** NaN and infinite objective values count as
//!   `+inf`, worse than every finite value.
//! - **Budget.** The last generation is cut at the budget: trials past it are
//!   not evaluated, and the memory, population and archive updates use the
//!   evaluated trials only. If the budget is smaller than the initial
//!   population, the run ends after evaluating part of it.
//!
//! # Example
//!
//! ```
//! use fcmaes_core::{Fitness, Jso, JsoParams};
//!
//! let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
//! let fit = Fitness::bounded(3, 1, &[-5.0; 3], &[5.0; 3]);
//! let params = JsoParams {
//!     max_evaluations: 2_000,
//!     seed: 42,
//!     ..Default::default()
//! };
//! let mut jso = Jso::new(fit, &params);
//! let result = jso.optimize(&sphere);
//! assert!(result.y.is_finite());
//! assert!(result.evaluations <= 2_000);
//! ```

use crate::fitness::{Fitness, Objective, parallel_batch};
use crate::rng::Rng;

/// Initial value of the `M_F` memory cells.
const MEMORY_F_INIT: f64 = 0.3;
/// Initial value of the `M_CR` memory cells.
const MEMORY_CR_INIT: f64 = 0.8;
/// Value the last memory cell yields for both `F` and `CR`.
const MEMORY_FIXED: f64 = 0.9;
/// Scale of the Cauchy distribution of `F`.
const F_SCALE: f64 = 0.1;
/// Standard deviation of the normal distribution of `CR`.
const CR_SIGMA: f64 = 0.1;
/// Factor of the published initial population size `round(25 · ln(D) · √D)`.
const POPSIZE_FACTOR: f64 = 25.0;
/// Value averaged into `M_CR` when every recorded `CR` is zero.
const TERMINAL: f64 = -1.0;

/// Outcome of a jSO run.
#[derive(Clone, Debug)]
pub struct JsoResult {
    /// The cooperative stop predicate fired.
    pub stopped: bool,
    /// Best decision vector evaluated; the box center if nothing was
    /// evaluated.
    pub x: Vec<f64>,
    /// Objective value at [`x`](Self::x); `+inf` if nothing was evaluated or
    /// every value was NaN or infinite.
    pub y: f64,
    /// Objective evaluations actually made; never more than
    /// [`JsoParams::max_evaluations`].
    pub evaluations: u64,
    /// Number of completed generations after the initial population.
    pub iterations: i32,
    /// Termination code; `1` means `stop_fitness` was reached.
    pub stop: i32,
}

/// Tunable inputs for [`Jso::new`]. The defaults are the published values.
#[derive(Clone, Debug)]
pub struct JsoParams {
    /// Initial population size; non-positive values select the published
    /// `round(25 · ln(D) · √D)`. Raised to [`min_popsize`](Self::min_popsize)
    /// if smaller.
    pub popsize: i32,
    /// Evaluation budget; must be positive. It is a hard limit: [`Jso`]
    /// never evaluates more. The population size, the p-best fraction and
    /// the parameter stages follow the fraction of it already spent.
    pub max_evaluations: u64,
    /// Stop after finding an objective value strictly below this threshold.
    pub stop_fitness: f64,
    /// Final population size `N_min` of the linear reduction (4); values
    /// below four are raised to four.
    pub min_popsize: usize,
    /// Number of memory cells `H` including the fixed last cell (5); values
    /// below two are raised to two.
    pub memory_size: usize,
    /// Initial p-best fraction `p_max` (0.25).
    pub p_max: f64,
    /// Final p-best fraction `p_min` (0.125).
    pub p_min: f64,
    /// Archive capacity relative to the current population size (1.0).
    pub archive_rate: f64,
    /// Seed for the optimizer's independent random stream.
    pub seed: u64,
    /// Additional run identifier mixed into [`seed`](Self::seed).
    pub runid: i64,
}

impl Default for JsoParams {
    fn default() -> Self {
        Self {
            popsize: 0,
            max_evaluations: 100_000,
            stop_fitness: f64::NEG_INFINITY,
            min_popsize: 4,
            memory_size: 5,
            p_max: 0.25,
            p_min: 0.125,
            archive_rate: 1.0,
            seed: 0,
            runid: 0,
        }
    }
}

/// The batch handed out by the last [`Jso::ask`] that has not been told yet.
enum Pending {
    /// Initial population members `start..start + len`.
    Init { start: usize, len: usize },
    /// Trials `(u, F, CR)` of the current generation, one per member in
    /// member order, cut at the budget.
    Generation(Vec<(Vec<f64>, f64, f64)>),
}

/// Stateful jSO optimizer.
///
/// Use [`optimize`](Self::optimize) for an in-process objective,
/// [`optimize_parallel`](Self::optimize_parallel) to evaluate each
/// generation on several threads, or [`ask`](Self::ask)/[`tell`](Self::tell)
/// when another system evaluates the population. One `ask`/`tell` pair is one
/// generation: the first [`ask`](Self::ask) returns the initial population,
/// later ones one trial per current member. The population shrinks over the
/// run, so the batch length varies; [`popsize`](Self::popsize) gives the
/// current population size.
pub struct Jso {
    rng: Rng,
    dim: usize,
    lower: Vec<f64>,
    upper: Vec<f64>,
    max_evaluations: u64,
    stop_fitness: f64,
    n_init: usize,
    n_min: usize,
    memory: usize,
    p_max: f64,
    p_min: f64,
    archive_rate: f64,

    /// Current population; before initialization, the initial sample.
    pop: Vec<Vec<f64>>,
    /// Values of `pop`; during initialization, of its first members only.
    fit: Vec<f64>,
    initialized: bool,
    m_f: Vec<f64>,
    m_cr: Vec<f64>,
    k: usize,
    archive: ParentArchive,

    pending: Option<Pending>,
    evaluations: u64,
    iterations: i32,
    stop: i32,
    best_x: Vec<f64>,
    best_y: f64,
}

/// Check the configuration [`Jso::try_new`] accepts.
fn validate_jso_inputs(fitfun: &Fitness, p: &JsoParams) -> Result<(), &'static str> {
    if fitfun.dim() == 0 {
        return Err("jSO needs at least one decision variable");
    }
    if !fitfun.has_bounds() {
        return Err("jSO needs finite bounds");
    }
    if fitfun
        .lower()
        .iter()
        .zip(fitfun.upper())
        .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi || !(hi - lo).is_finite())
    {
        return Err("jSO bounds must be finite with lower <= upper");
    }
    if p.max_evaluations == 0 {
        return Err("jSO needs a positive max_evaluations");
    }
    if !(p.p_min.is_finite() && p.p_max.is_finite() && 0.0 < p.p_min && p.p_min <= p.p_max)
        || p.p_max > 1.0
    {
        return Err("jSO p-best fractions must satisfy 0 < p_min <= p_max <= 1");
    }
    if !p.archive_rate.is_finite() || p.archive_rate < 0.0 {
        return Err("jSO archive_rate must be finite and non-negative");
    }
    Ok(())
}

impl Jso {
    /// Build a jSO optimizer for the box of `fitfun`.
    ///
    /// The initial population of [`popsize`](Self::popsize) points is drawn
    /// uniformly from the box at once. jSO works in the decoded coordinates of
    /// `fitfun`; [`Fitness::set_normalize`] has no effect on it.
    ///
    /// # Errors
    ///
    /// Returns an error if `fitfun` has no decision variables or no bounds,
    /// if a bound is not finite, a lower bound exceeds its upper bound or a
    /// box width overflows, if `max_evaluations` is zero, if the p-best
    /// fractions do not satisfy `0 < p_min <= p_max <= 1`, or if
    /// `archive_rate` is negative or not finite.
    pub fn try_new(fitfun: Fitness, p: &JsoParams) -> Result<Self, &'static str> {
        validate_jso_inputs(&fitfun, p)?;
        Ok(Self::new_unchecked(fitfun, p))
    }

    /// Build a jSO optimizer, panicking on an invalid configuration.
    /// Applications that accept user input should prefer [`Jso::try_new`].
    ///
    /// # Panics
    ///
    /// Panics on any configuration [`Jso::try_new`] rejects, in particular
    /// without finite bounds or without a positive `max_evaluations`.
    pub fn new(fitfun: Fitness, p: &JsoParams) -> Self {
        Self::try_new(fitfun, p).expect("invalid jSO configuration")
    }

    fn new_unchecked(fitfun: Fitness, p: &JsoParams) -> Self {
        let dim = fitfun.dim();
        let lower = fitfun.lower().to_vec();
        let upper = fitfun.upper().to_vec();
        let n_min = p.min_popsize.max(4);
        let d = dim as f64;
        let n_init = if p.popsize > 0 {
            p.popsize as usize
        } else {
            (POPSIZE_FACTOR * d.ln() * d.sqrt()).round() as usize
        }
        .max(n_min);
        let memory = p.memory_size.max(2);
        let mut rng = Rng::new(p.seed.wrapping_add(p.runid as u64));
        let pop = (0..n_init)
            .map(|_| uniform_point(&mut rng, &lower, &upper))
            .collect();
        Self {
            rng,
            dim,
            lower,
            upper,
            max_evaluations: p.max_evaluations,
            stop_fitness: p.stop_fitness,
            n_init,
            n_min,
            memory,
            p_max: p.p_max,
            p_min: p.p_min,
            archive_rate: p.archive_rate,
            pop,
            fit: Vec::with_capacity(n_init),
            initialized: false,
            m_f: vec![MEMORY_F_INIT; memory],
            m_cr: vec![MEMORY_CR_INIT; memory],
            k: 0,
            archive: ParentArchive::default(),
            pending: None,
            evaluations: 0,
            iterations: 0,
            stop: 0,
            best_x: Vec::new(),
            best_y: f64::INFINITY,
        }
    }

    /// Number of decision variables.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Current population size: the initial size until the initial
    /// population is told, then the size after the latest reduction. It is
    /// the length of the next [`ask`](Self::ask) batch unless the remaining
    /// budget is smaller.
    pub fn popsize(&self) -> usize {
        self.pop.len()
    }

    /// Current termination code, or zero while the optimizer can continue.
    pub fn stop(&self) -> i32 {
        self.stop
    }

    fn remaining(&self) -> u64 {
        self.max_evaluations.saturating_sub(self.evaluations)
    }

    /// Generate the next batch without the call-order check.
    fn prepare_batch(&mut self) {
        if self.stop != 0 || self.remaining() == 0 {
            self.pending = None;
            return;
        }
        let remaining = usize::try_from(self.remaining()).unwrap_or(usize::MAX);
        if !self.initialized {
            let start = self.fit.len();
            let len = (self.pop.len() - start).min(remaining);
            self.pending = Some(Pending::Init { start, len });
            return;
        }
        let mut trials = self.generate_trials();
        trials.truncate(remaining);
        self.pending = Some(Pending::Generation(trials));
    }

    fn next_batch(&mut self) -> Vec<Vec<f64>> {
        self.prepare_batch();
        match &self.pending {
            Some(Pending::Init { start, len }) => self.pop[*start..*start + *len].to_vec(),
            Some(Pending::Generation(trials)) => {
                trials.iter().map(|(row, _, _)| row.clone()).collect()
            }
            None => Vec::new(),
        }
    }

    fn pending_rows_inner(&self) -> impl Iterator<Item = &[f64]> {
        let (initial, trials) = match &self.pending {
            Some(Pending::Init { start, len }) => (&self.pop[*start..*start + *len], &[][..]),
            Some(Pending::Generation(trials)) => (&[][..], trials.as_slice()),
            None => (&[][..], &[][..]),
        };
        initial
            .iter()
            .map(Vec::as_slice)
            .chain(trials.iter().map(|(x, _, _)| x.as_slice()))
    }

    /// One trial per member of the current population.
    fn generate_trials(&mut self) -> Vec<(Vec<f64>, f64, f64)> {
        let n = self.pop.len();
        let h = self.memory;
        let progress = (self.evaluations as f64 / self.max_evaluations as f64).min(1.0);
        let p = self.p_max - (self.p_max - self.p_min) * progress;
        let order = sorted_indices(&self.fit);
        let n_pbest = pbest_count(p, n, 2);
        let fw_factor = if progress < 0.2 {
            0.7
        } else if progress < 0.4 {
            0.8
        } else {
            1.2
        };

        let rng = &mut self.rng;
        let mut trials = Vec::with_capacity(n);
        for i in 0..n {
            let r = rand_index(rng, h);
            let (mean_f, mean_cr) = memory_cell(&self.m_f, &self.m_cr, r, MEMORY_FIXED);
            let mut cr = if mean_cr < 0.0 {
                0.0
            } else {
                sample_clipped_normal(rng, mean_cr, CR_SIGMA)
            };
            if progress < 0.25 {
                cr = cr.max(0.7);
            } else if progress < 0.5 {
                cr = cr.max(0.6);
            }
            let mut f = sample_f_cauchy(rng, mean_f, F_SCALE);
            if progress < 0.6 && f > 0.7 {
                f = 0.7;
            }
            let fw = fw_factor * f;

            let pbest = draw_pbest(rng, &order, n_pbest, i, progress < 0.5);
            let r1 = loop {
                let c = rand_index(rng, n);
                if c != i {
                    break c;
                }
            };
            let r2 = loop {
                let c = rand_index(rng, n + self.archive.len());
                if c != i && c != r1 {
                    break c;
                }
            };
            let x_r2: &[f64] = if r2 < n {
                &self.pop[r2]
            } else {
                self.archive.get(r2 - n)
            };
            let xi = &self.pop[i];
            let mut v: Vec<f64> = (0..self.dim)
                .map(|j| {
                    xi[j] + fw * (self.pop[pbest][j] - xi[j]) + f * (self.pop[r1][j] - x_r2[j])
                })
                .collect();
            repair_midpoint(&mut v, xi, &self.lower, &self.upper);
            let mut u = vec![0.0; self.dim];
            binomial_crossover(rng, &v, xi, cr, &mut u);
            trials.push((u, f, cr));
        }
        trials
    }

    /// Record an evaluated point; returns its sanitized value.
    fn record(&mut self, x: &[f64], raw: f64) -> f64 {
        let y = if raw.is_finite() { raw } else { f64::INFINITY };
        self.evaluations += 1;
        if self.best_x.is_empty() || y < self.best_y {
            self.best_y = y;
            self.best_x = x.to_vec();
        }
        y
    }

    /// Incorporate the values of the first `ys.len()` rows of the pending
    /// batch (all of them for [`tell`](Self::tell); a prefix when a run is
    /// stopped or the stop fitness is reached part way).
    fn tell_values(&mut self, ys: &[f64]) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        match pending {
            Pending::Init { start, len } => {
                debug_assert!(ys.len() <= len);
                for (offset, &raw) in ys.iter().enumerate() {
                    let x = self.pop[start + offset].clone();
                    let y = self.record(&x, raw);
                    self.fit.push(y);
                }
                if self.fit.len() == self.pop.len() {
                    self.initialized = true;
                } else if self.remaining() == 0 {
                    // The budget ends the run inside the initial population.
                    self.pop.truncate(self.fit.len());
                }
            }
            Pending::Generation(trials) => {
                debug_assert!(ys.len() <= trials.len());
                let n = self.pop.len();
                let capacity = (self.archive_rate * n as f64).round() as usize;
                let (mut s_f, mut s_cr, mut s_df) = (Vec::new(), Vec::new(), Vec::new());
                for (i, ((u, f, cr), &raw)) in trials.into_iter().zip(ys).enumerate() {
                    let fu = self.record(&u, raw);
                    if fu <= self.fit[i] {
                        if fu < self.fit[i] {
                            s_f.push(f);
                            s_cr.push(cr);
                            s_df.push(self.fit[i] - fu);
                            let parent = std::mem::replace(&mut self.pop[i], u);
                            self.archive.push(&mut self.rng, parent, capacity);
                        } else {
                            self.pop[i] = u;
                        }
                        self.fit[i] = fu;
                    }
                }

                if !s_f.is_empty() {
                    let k = self.k;
                    if let Some(mean) = weighted_lehmer_mean(&s_f, &s_df) {
                        self.m_f[k] = 0.5 * (self.m_f[k] + mean);
                    }
                    self.m_cr[k] = updated_cr_cell(self.m_cr[k], &s_cr, &s_df);
                    self.k = (k + 1) % self.memory;
                }

                let n_next = lpsr_size(
                    self.n_init,
                    self.n_min,
                    self.evaluations,
                    self.max_evaluations,
                );
                if n_next < self.pop.len() {
                    let order = sorted_indices(&self.fit);
                    let keep = &order[..n_next];
                    self.pop = keep
                        .iter()
                        .map(|&j| std::mem::take(&mut self.pop[j]))
                        .collect();
                    self.fit = keep.iter().map(|&j| self.fit[j]).collect();
                }
                let capacity = (self.archive_rate * self.pop.len() as f64).round() as usize;
                self.archive.shrink(&mut self.rng, capacity);
                self.iterations += 1;
            }
        }
        if self.best_y < self.stop_fitness {
            self.stop = 1;
        }
    }

    /// Run serially against an in-process objective until the budget is
    /// spent or `stop_fitness` is reached.
    ///
    /// The evaluation budget is a hard limit: the run never makes more than
    /// [`JsoParams::max_evaluations`] calls. (DE's serial `optimize` checks
    /// its budget only per generation.)
    pub fn optimize(&mut self, obj: &impl Objective) -> JsoResult {
        self.optimize_until(obj, &|| false)
    }

    /// Serial jSO with a cooperative stop checked before every evaluation.
    /// A generation cut by the stop is completed with the trials evaluated
    /// so far, as at the end of the budget.
    pub fn optimize_until(&mut self, obj: &impl Objective, stop: &dyn Fn() -> bool) -> JsoResult {
        let mut ys = Vec::new();
        loop {
            self.prepare_batch();
            if self.pending.is_none() {
                return self.make_result(false);
            }
            ys.clear();
            let mut stopped = false;
            for x in self.pending_rows_inner() {
                if stop() {
                    stopped = true;
                    break;
                }
                let y = obj.eval_scalar(x);
                ys.push(y);
                if y.is_finite() && y < self.stop_fitness {
                    break;
                }
            }
            self.tell_values(&ys);
            if stopped {
                return self.make_result(true);
            }
        }
    }

    /// Run with each generation evaluated as one batch on `workers` threads:
    /// `1` evaluates serially, values above one use that many cached Rayon
    /// threads, and values at or below zero use Rayon's global pool (as
    /// [`parallel_batch`]).
    ///
    /// The result does not depend on `workers` or on thread timing. With the
    /// default `stop_fitness` it equals the serial [`optimize`](Self::optimize);
    /// when `stop_fitness` is reached, the rest of that generation is still
    /// evaluated. The evaluation budget stays a hard limit.
    ///
    /// # Panics
    ///
    /// Resumes a panic of the objective in the caller.
    pub fn optimize_parallel(&mut self, obj: &impl Objective, workers: i32) -> JsoResult {
        self.optimize_parallel_until(obj, workers, &|| false)
    }

    /// Like [`optimize_parallel`](Self::optimize_parallel), with a cooperative
    /// stop checked before every generation; a generation in flight is told
    /// in full.
    ///
    /// # Panics
    ///
    /// Resumes a panic of the objective in the caller.
    pub fn optimize_parallel_until(
        &mut self,
        obj: &impl Objective,
        workers: i32,
        stop: &dyn Fn() -> bool,
    ) -> JsoResult {
        loop {
            if stop() {
                return self.make_result(true);
            }
            let rows = self.next_batch();
            if rows.is_empty() {
                return self.make_result(false);
            }
            let ys = parallel_batch(&rows, workers, |x| obj.eval_scalar(x));
            self.tell_values(&ys);
        }
    }

    fn make_result(&self, stopped: bool) -> JsoResult {
        let x = if self.best_x.is_empty() {
            self.lower
                .iter()
                .zip(&self.upper)
                .map(|(lo, hi)| 0.5 * (lo + hi))
                .collect()
        } else {
            self.best_x.clone()
        };
        JsoResult {
            stopped,
            x,
            y: self.best_y,
            evaluations: self.evaluations,
            iterations: self.iterations,
            stop: self.stop,
        }
    }

    // ---- ask/tell interface ----

    /// Ask for the next generation: the initial population first, then one
    /// trial per current population member.
    ///
    /// The batch length varies: the population shrinks over the run, and the
    /// last batch is cut so that the rows asked never exceed
    /// [`JsoParams::max_evaluations`] in total. Telling that shorter batch
    /// completes the last generation with the evaluated trials only, as
    /// [`optimize`](Self::optimize) does, so ask/tell with serial evaluation
    /// reproduces `optimize` exactly. Once the budget is spent or
    /// `stop_fitness` is reached, the batch is empty and no `tell` is due.
    ///
    /// # Panics
    ///
    /// Panics if a previous batch is still pending, that is if `ask` is called
    /// twice without an intervening `tell`. Use [`Jso::try_ask`] to receive
    /// that as an error instead.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        self.try_ask().expect("invalid jSO ask call")
    }

    /// Fallible ask variant for interfaces that need to report call-order
    /// errors rather than panic.
    ///
    /// # Errors
    ///
    /// Returns an error if the previously asked batch has not been told yet.
    pub fn try_ask(&mut self) -> Result<Vec<Vec<f64>>, &'static str> {
        if self.pending.is_some() {
            return Err("jSO ask called before telling the pending batch");
        }
        Ok(self.next_batch())
    }

    /// Tell one objective value per row of the batch returned by
    /// [`ask`](Self::ask), in the same order. NaN and infinite values count
    /// as `+inf`. Returns the termination code.
    ///
    /// # Panics
    ///
    /// Panics if no batch is pending or if `ys` does not have one value per
    /// asked row. Use [`Jso::try_tell`] to receive these as errors instead.
    pub fn tell(&mut self, ys: &[f64]) -> i32 {
        self.try_tell(ys).expect("invalid jSO tell call")
    }

    /// Fallible tell variant validating call order and batch length.
    ///
    /// # Errors
    ///
    /// Returns an error if no batch is pending or if `ys` does not have one
    /// value per row of the pending batch.
    pub fn try_tell(&mut self, ys: &[f64]) -> Result<i32, &'static str> {
        let expected = match &self.pending {
            None => return Err("jSO tell called without a pending ask batch"),
            Some(Pending::Init { len, .. }) => *len,
            Some(Pending::Generation(trials)) => trials.len(),
        };
        if ys.len() != expected {
            return Err("jSO tell batch length must equal the asked batch length");
        }
        self.tell_values(ys);
        Ok(self.stop)
    }

    /// Return a snapshot of the current population. Before the initial
    /// population is told, this is the initial sample.
    pub fn population(&self) -> Vec<Vec<f64>> {
        self.pop.clone()
    }

    /// Return the current best result.
    pub fn result(&self) -> JsoResult {
        self.make_result(false)
    }
}

// ---- operators ----

/// Uniform random point in `[lower, upper]`.
fn uniform_point(rng: &mut Rng, lower: &[f64], upper: &[f64]) -> Vec<f64> {
    lower
        .iter()
        .zip(upper)
        .map(|(lo, hi)| (lo + rng.uniform01() * (hi - lo)).clamp(*lo, *hi))
        .collect()
}

/// Uniform index in `0..n` (`n > 0`).
fn rand_index(rng: &mut Rng, n: usize) -> usize {
    debug_assert!(n > 0);
    (rng.int_below(n as i64) as usize).min(n - 1)
}

/// Cauchy sample with location `loc` and scale `scale`.
fn cauchy(rng: &mut Rng, loc: f64, scale: f64) -> f64 {
    loc + scale * (std::f64::consts::PI * (rng.uniform01() - 0.5)).tan()
}

/// SHADE scaling-factor sample: Cauchy(`loc`, `scale`), redrawn while it is
/// not positive, truncated to 1.
fn sample_f_cauchy(rng: &mut Rng, loc: f64, scale: f64) -> f64 {
    loop {
        let f = cauchy(rng, loc, scale);
        if f > 0.0 {
            return f.min(1.0);
        }
    }
}

/// Normal sample clipped to `[0, 1]`.
fn sample_clipped_normal(rng: &mut Rng, mean: f64, sigma: f64) -> f64 {
    rng.normreal(mean, sigma).clamp(0.0, 1.0)
}

/// Weighted Lehmer mean `sum(w v^2) / sum(w v)` with weights proportional to
/// `weights` (fitness improvements).
///
/// Infinite improvements dominate: the weight is shared equally among them.
/// If all weights are zero, equal weights are used. Returns `None` for an
/// empty set or a zero denominator.
fn weighted_lehmer_mean(values: &[f64], weights: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let any_infinite = weights.iter().any(|w| w.is_infinite());
    let total: f64 = weights.iter().filter(|w| w.is_finite()).sum();
    let weight = |w: f64| -> f64 {
        if any_infinite {
            if w.is_infinite() { 1.0 } else { 0.0 }
        } else if total > 0.0 {
            w.max(0.0)
        } else {
            1.0
        }
    };
    let (mut num, mut den) = (0.0, 0.0);
    for (&v, &w) in values.iter().zip(weights) {
        let w = weight(w);
        num += w * v * v;
        den += w * v;
    }
    (den > 0.0).then(|| num / den)
}

/// Midpoint-target repair: a coordinate below (above) its bound moves halfway
/// between the bound and the reference (parent) coordinate.
fn repair_midpoint(v: &mut [f64], reference: &[f64], lower: &[f64], upper: &[f64]) {
    for j in 0..v.len() {
        if v[j] < lower[j] {
            v[j] = 0.5 * (lower[j] + reference[j]);
        } else if v[j] > upper[j] {
            v[j] = 0.5 * (upper[j] + reference[j]);
        }
        // Guards against rounding only; the midpoint of two in-box values is
        // already inside the box.
        v[j] = v[j].clamp(lower[j], upper[j]);
    }
}

/// Binomial crossover with a forced coordinate `j_rand`: `trial[j] = mutant[j]`
/// if `rand <= cr` or `j == j_rand`, otherwise `target[j]`.
fn binomial_crossover(rng: &mut Rng, mutant: &[f64], target: &[f64], cr: f64, trial: &mut [f64]) {
    let j_rand = rand_index(rng, mutant.len());
    for j in 0..mutant.len() {
        trial[j] = if j == j_rand || rng.uniform01() <= cr {
            mutant[j]
        } else {
            target[j]
        };
    }
}

/// Linear population size reduction:
/// `round(n_init + (n_min - n_init) * fes / max_fes)`, never below `n_min`.
fn lpsr_size(n_init: usize, n_min: usize, fes: u64, max_fes: u64) -> usize {
    if max_fes == 0 {
        return n_min;
    }
    let ratio = (fes as f64 / max_fes as f64).min(1.0);
    let n = (n_init as f64 + (n_min as f64 - n_init as f64) * ratio).round();
    (n.max(n_min as f64) as usize).min(n_init.max(n_min))
}

/// Indices `0..fitness.len()` sorted by ascending fitness (best first).
fn sorted_indices(fitness: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..fitness.len()).collect();
    order.sort_by(|&a, &b| fitness[a].total_cmp(&fitness[b]).then(a.cmp(&b)));
    order
}

/// Size of the p-best set: `round(p * n)`, at least `min_count`, at most `n`.
fn pbest_count(p: f64, n: usize, min_count: usize) -> usize {
    ((p * n as f64).round() as usize)
        .max(min_count)
        .min(n)
        .max(1)
}

/// Memory values `(M_F, M_CR)` for sampled cell `r`; the last cell is fixed.
fn memory_cell(m_f: &[f64], m_cr: &[f64], r: usize, fixed: f64) -> (f64, f64) {
    if r + 1 == m_f.len() {
        (fixed, fixed)
    } else {
        (m_f[r], m_cr[r])
    }
}

/// New `M_CR` cell value: the old value averaged with the weighted Lehmer
/// mean of `s_cr`, or with the terminal value `-1` if `max(S_CR) = 0`.
fn updated_cr_cell(old: f64, s_cr: &[f64], s_df: &[f64]) -> f64 {
    let max_cr = s_cr.iter().copied().fold(0.0_f64, f64::max);
    let target = if max_cr == 0.0 {
        TERMINAL
    } else {
        match weighted_lehmer_mean(s_cr, s_df) {
            Some(mean) => mean,
            None => return old,
        }
    };
    0.5 * (old + target)
}

/// Uniform index among the `n_pbest` best (`order` is best first); while
/// `redraw_self` holds it is redrawn until it differs from `i`.
fn draw_pbest(
    rng: &mut Rng,
    order: &[usize],
    n_pbest: usize,
    i: usize,
    redraw_self: bool,
) -> usize {
    loop {
        let c = order[rand_index(rng, n_pbest)];
        if !(redraw_self && c == i && n_pbest > 1) {
            return c;
        }
    }
}

/// External archive of replaced parents with random removal.
#[derive(Default)]
struct ParentArchive {
    members: Vec<Vec<f64>>,
}

impl ParentArchive {
    fn len(&self) -> usize {
        self.members.len()
    }

    fn get(&self, i: usize) -> &[f64] {
        &self.members[i]
    }

    /// Add `x`. If the archive already holds `capacity` members, `x`
    /// overwrites a random member, so a new parent always enters. With
    /// `capacity == 0` nothing is stored.
    fn push(&mut self, rng: &mut Rng, x: Vec<f64>, capacity: usize) {
        self.shrink(rng, capacity);
        if capacity == 0 {
            return;
        }
        if self.members.len() < capacity {
            self.members.push(x);
        } else {
            let r = rand_index(rng, self.members.len());
            self.members[r] = x;
        }
    }

    /// Remove random members until at most `capacity` remain.
    fn shrink(&mut self, rng: &mut Rng, capacity: usize) {
        while self.members.len() > capacity {
            let r = rand_index(rng, self.members.len());
            self.members.swap_remove(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    const DIM: usize = 10;

    /// Optimum off the box center and the origin.
    fn shift(i: usize) -> f64 {
        1.7 - 0.31 * i as f64
    }

    fn sphere(x: &[f64]) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, v)| (v - shift(i)).powi(2))
            .sum()
    }

    fn rosen(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    }

    fn jso(dim: usize, budget: u64, seed: u64) -> Jso {
        let fit = Fitness::bounded(dim, 1, &vec![-5.0; dim], &vec![5.0; dim]);
        Jso::new(
            fit,
            &JsoParams {
                max_evaluations: budget,
                seed,
                ..Default::default()
            },
        )
    }

    /// Every point the objective sees, in call order.
    fn recorded(
        run: impl FnOnce(&(dyn Fn(&[f64]) -> f64 + Sync)) -> JsoResult,
    ) -> (JsoResult, Vec<Vec<f64>>) {
        let points = Mutex::new(Vec::new());
        let objective = |x: &[f64]| {
            points.lock().unwrap().push(x.to_vec());
            sphere(x)
        };
        let result = run(&objective);
        (result, points.into_inner().unwrap())
    }

    #[test]
    fn defaults_are_the_published_values() {
        let p = JsoParams::default();
        assert_eq!(p.popsize, 0);
        assert_eq!(p.max_evaluations, 100_000);
        assert_eq!(p.stop_fitness, f64::NEG_INFINITY);
        assert_eq!((p.min_popsize, p.memory_size), (4, 5));
        assert_eq!((p.p_max, p.p_min, p.archive_rate), (0.25, 0.125, 1.0));
        assert_eq!((p.seed, p.runid), (0, 0));
        // round(25 ln(D) sqrt(D)), never below the final size of four.
        assert_eq!(jso(10, 1_000, 1).popsize(), 182);
        assert_eq!(jso(5, 1_000, 1).popsize(), 90);
        assert_eq!(jso(1, 1_000, 1).popsize(), 4);
        let fit = Fitness::bounded(3, 1, &[-1.0; 3], &[1.0; 3]);
        let small = Jso::new(
            fit,
            &JsoParams {
                popsize: 2,
                ..Default::default()
            },
        );
        assert_eq!((small.dim(), small.popsize(), small.stop()), (3, 4, 0));
        assert_eq!(small.population().len(), 4);
    }

    #[test]
    fn same_seed_reproduces_the_run() {
        let a = jso(5, 3_000, 7).optimize(&sphere);
        let b = jso(5, 3_000, 7).optimize(&sphere);
        assert_eq!((a.x.clone(), a.y, a.evaluations), (b.x, b.y, b.evaluations));
        let c = jso(5, 3_000, 8).optimize(&sphere);
        assert_ne!(a.x, c.x);
        // runid is added to the seed.
        let fit = Fitness::bounded(5, 1, &[-5.0; 5], &[5.0; 5]);
        let mixed = Jso::new(
            fit,
            &JsoParams {
                max_evaluations: 3_000,
                seed: 5,
                runid: 2,
                ..Default::default()
            },
        )
        .optimize(&sphere);
        assert_eq!(mixed.x, a.x);
    }

    #[test]
    fn budget_is_never_exceeded() {
        // The initial population of D = 5 has 90 members; budgets below it
        // end the run inside the initial population.
        for budget in [1, 3, 89, 90, 91, 500, 2_003] {
            for workers in [0, 1, 3] {
                let calls = AtomicU64::new(0);
                let objective = |x: &[f64]| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    sphere(x)
                };
                let mut optimizer = jso(5, budget, 3);
                let result = if workers == 0 {
                    optimizer.optimize(&objective)
                } else {
                    optimizer.optimize_parallel(&objective, workers)
                };
                let calls = calls.load(Ordering::Relaxed);
                assert_eq!(calls, budget, "budget {budget}, workers {workers}");
                assert_eq!(result.evaluations, calls);
                assert_eq!(sphere(&result.x), result.y);
                assert!(optimizer.ask().is_empty());
            }
        }
    }

    #[test]
    fn bounds_are_respected() {
        let (lower, upper) = ([-1.0, 2.0, -0.5, 0.0], [0.5, 2.5, 3.0, 1e-3]);
        let fit = Fitness::bounded(4, 1, &lower, &upper);
        let mut optimizer = Jso::new(
            fit,
            &JsoParams {
                max_evaluations: 5_000,
                seed: 2,
                ..Default::default()
            },
        );
        let (_, points) = recorded(|f| optimizer.optimize(&f));
        assert_eq!(points.len(), 5_000);
        for x in &points {
            for ((v, lo), hi) in x.iter().zip(&lower).zip(&upper) {
                assert!(lo <= v && v <= hi, "{v} outside [{lo}, {hi}]");
            }
        }
    }

    #[test]
    fn ask_tell_and_parallel_equal_the_serial_run() {
        // 2,003 is not a whole number of generations.
        let (serial, serial_points) = recorded(|f| jso(6, 2_003, 4).optimize(&f));
        let mut optimizer = jso(6, 2_003, 4);
        let mut asked = Vec::new();
        let mut generations = 0;
        loop {
            let rows = optimizer.ask();
            if rows.is_empty() {
                break;
            }
            let expected = optimizer.popsize();
            assert!(rows.len() <= expected);
            let ys: Vec<f64> = rows.iter().map(|x| sphere(x)).collect();
            asked.extend(rows);
            assert_eq!(optimizer.tell(&ys), 0);
            generations += 1;
        }
        assert_eq!(asked, serial_points);
        let told = optimizer.result();
        assert_eq!((told.x.clone(), told.y), (serial.x.clone(), serial.y));
        assert_eq!(
            (told.evaluations, told.iterations),
            (2_003, generations - 1)
        );
        assert_eq!(serial.iterations, told.iterations);
        // The population shrank.
        assert!(optimizer.popsize() < jso(6, 2_003, 4).popsize());

        for workers in [2, 4, -1] {
            let parallel = jso(6, 2_003, 4).optimize_parallel(&sphere, workers);
            assert_eq!((parallel.x, parallel.y), (serial.x.clone(), serial.y));
        }
    }

    #[test]
    fn wrong_calls_are_rejected() {
        let mut optimizer = jso(3, 1_000, 1);
        assert!(optimizer.try_tell(&[1.0]).is_err());
        let rows = optimizer.ask();
        assert_eq!(rows.len(), optimizer.popsize());
        assert!(optimizer.try_ask().is_err());
        let ys: Vec<f64> = rows.iter().map(|x| sphere(x)).collect();
        assert!(optimizer.try_tell(&ys[1..]).is_err());
        let mut longer = ys.clone();
        longer.push(0.0);
        assert!(optimizer.try_tell(&longer).is_err());
        // A rejected tell leaves the batch pending.
        assert_eq!(optimizer.try_tell(&ys), Ok(0));
        assert_eq!(optimizer.result().evaluations, rows.len() as u64);
    }

    #[test]
    #[should_panic(expected = "invalid jSO tell call")]
    fn tell_with_the_wrong_length_panics() {
        let mut optimizer = jso(3, 1_000, 1);
        let rows = optimizer.ask();
        optimizer.tell(&vec![0.0; rows.len() + 1]);
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        let ok = JsoParams::default();
        let bounded = || Fitness::bounded(2, 1, &[-1.0; 2], &[1.0; 2]);
        assert!(Jso::try_new(bounded(), &ok).is_ok());
        let unbounded = Fitness::new(2, 1, Vec::new(), Vec::new());
        assert!(Jso::try_new(unbounded, &ok).is_err());
        let infinite = Fitness::bounded(2, 1, &[f64::NEG_INFINITY, -1.0], &[1.0; 2]);
        assert!(Jso::try_new(infinite, &ok).is_err());
        let inverted = Fitness::bounded(2, 1, &[1.0, -1.0], &[-1.0, 1.0]);
        assert!(Jso::try_new(inverted, &ok).is_err());
        let overflowing = Fitness::bounded(1, 1, &[-f64::MAX], &[f64::MAX]);
        assert!(Jso::try_new(overflowing, &ok).is_err());
        let empty = Fitness::bounded(0, 1, &[], &[]);
        assert!(Jso::try_new(empty, &ok).is_err());
        for params in [
            JsoParams {
                max_evaluations: 0,
                ..Default::default()
            },
            JsoParams {
                p_min: 0.0,
                ..Default::default()
            },
            JsoParams {
                p_max: 0.1,
                ..Default::default()
            },
            JsoParams {
                p_max: 1.5,
                ..Default::default()
            },
            JsoParams {
                archive_rate: f64::NAN,
                ..Default::default()
            },
        ] {
            assert!(Jso::try_new(bounded(), &params).is_err());
        }
    }

    #[test]
    #[should_panic(expected = "invalid jSO configuration")]
    fn new_panics_without_bounds() {
        Jso::new(
            Fitness::new(2, 1, Vec::new(), Vec::new()),
            &JsoParams::default(),
        );
    }

    #[test]
    fn stop_fitness_ends_the_run() {
        let fit = Fitness::bounded(DIM, 1, &[-5.0; DIM], &[5.0; DIM]);
        let mut optimizer = Jso::new(
            fit,
            &JsoParams {
                max_evaluations: 50_000,
                stop_fitness: 1e-3,
                seed: 1,
                ..Default::default()
            },
        );
        let result = optimizer.optimize(&sphere);
        assert_eq!(result.stop, 1);
        assert!(result.y < 1e-3);
        assert!(result.evaluations < 50_000);
        // The run stops at the evaluation that reached the threshold.
        assert!(optimizer.ask().is_empty());
        assert!(optimizer.try_tell(&[]).is_err());
    }

    #[test]
    fn cooperative_stop_ends_the_run() {
        let calls = AtomicU64::new(0);
        let objective = |x: &[f64]| {
            calls.fetch_add(1, Ordering::Relaxed);
            sphere(x)
        };
        let result =
            jso(4, 10_000, 1).optimize_until(&objective, &|| calls.load(Ordering::Relaxed) >= 333);
        assert!(result.stopped);
        assert_eq!(result.evaluations, 333);
        assert_eq!(calls.load(Ordering::Relaxed), 333);
    }

    #[test]
    fn non_finite_values_do_not_panic() {
        // NaN on half of the box, +inf and -inf elsewhere.
        let partly = |x: &[f64]| {
            if x[0] < 0.0 {
                f64::NAN
            } else if x[1] < -4.0 {
                f64::NEG_INFINITY
            } else {
                sphere(x)
            }
        };
        let result = jso(4, 3_000, 2).optimize(&partly);
        assert!(result.y.is_finite() && result.x[0] >= 0.0);
        assert_eq!(result.evaluations, 3_000);

        let never = |_: &[f64]| f64::NAN;
        let result = jso(4, 500, 2).optimize(&never);
        assert_eq!(result.y, f64::INFINITY);
        assert_eq!(result.evaluations, 500);
        assert!(result.x.iter().all(|v| (-5.0..=5.0).contains(v)));

        let mut asked = jso(2, 100, 3);
        let rows = asked.ask();
        asked.tell(&vec![f64::NAN; rows.len()]);
        let rows = asked.ask();
        asked.tell(&vec![f64::INFINITY; rows.len()]);
        assert_eq!(asked.result().y, f64::INFINITY);
    }

    #[test]
    fn minimizes_sphere() {
        let result = jso(DIM, 50_000, 1).optimize(&sphere);
        assert!(result.y < 1e-8, "sphere not solved: {}", result.y);
    }

    #[test]
    fn minimizes_rosenbrock() {
        let values: Vec<f64> = (1..=5)
            .map(|seed| jso(DIM, 200_000, seed).optimize(&rosen).y)
            .collect();
        let passed = values.iter().filter(|&&y| y < 1e-6).count();
        assert!(
            passed >= 3,
            "rosenbrock below 1e-6 in {passed}/5: {values:?}"
        );
    }

    #[test]
    fn operators_match_their_definitions() {
        let m_f = [0.1, 0.2, 0.3, 0.4, 0.5];
        let m_cr = [0.6, 0.7, 0.8, 0.9, -1.0];
        assert_eq!(memory_cell(&m_f, &m_cr, 4, 0.9), (0.9, 0.9));
        assert_eq!(memory_cell(&m_f, &m_cr, 1, 0.9), (0.2, 0.7));
        // The terminal value is averaged in, so a negative cell recovers.
        assert!((updated_cr_cell(0.6, &[0.0, 0.0], &[1.0, 2.0]) - (-0.2)).abs() < 1e-12);
        let once = updated_cr_cell(-1.0, &[0.8], &[1.0]);
        assert!((once - (-0.1)).abs() < 1e-12);
        assert!((updated_cr_cell(once, &[0.8], &[1.0]) - 0.35).abs() < 1e-12);
        let mean = weighted_lehmer_mean(&[0.2, 0.6], &[1.0, 3.0]).unwrap();
        let expected = (0.25 * 0.04 + 0.75 * 0.36) / (0.25 * 0.2 + 0.75 * 0.6);
        assert!((mean - expected).abs() < 1e-12);
        assert_eq!(weighted_lehmer_mean(&[], &[]), None);
        let infinite = weighted_lehmer_mean(&[0.3, 0.9], &[f64::INFINITY, 2.0]).unwrap();
        assert!((infinite - 0.3).abs() < 1e-12);
        assert_eq!(lpsr_size(180, 4, 500, 1000), 92);
        assert_eq!(lpsr_size(180, 4, 5000, 1000), 4);
        let mut v = [-3.0, 5.0];
        repair_midpoint(&mut v, &[0.5, 0.5], &[-1.0, -1.0], &[1.0, 1.0]);
        assert_eq!(v, [-0.25, 0.75]);
        let mut rng = Rng::new(4);
        let order = [7, 3, 0, 1];
        assert!((0..1000).all(|_| draw_pbest(&mut rng, &order, 2, 7, true) != 7));
        let mut archive = ParentArchive::default();
        for v in 0..10 {
            archive.push(&mut rng, vec![f64::from(v)], 3);
            assert!(archive.len() <= 3);
            assert!((0..archive.len()).any(|i| archive.get(i)[0] == f64::from(v)));
        }
    }
}
