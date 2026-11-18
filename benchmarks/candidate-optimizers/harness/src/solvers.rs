//! Solver registry: incumbent `fcmaes-core` adapters and the candidates.
//!
//! # Incumbent configuration
//!
//! Every incumbent runs single-threaded (`workers = 1`) with the library
//! defaults that the repository's own GTOP benchmark binaries and job worker
//! use; the README lists every value.
//!
//! # Exact budgets
//!
//! Several core optimizers check their budget once per generation and may
//! overshoot it. The adapters never let that happen, so the guard never has
//! to refuse a call:
//!
//! - BiteOpt and DE run their library `optimize_until`, whose cooperative stop
//!   is checked before *every* evaluation (DE: also before its
//!   temporal-locality trial). The stop fires when the budget is spent, so
//!   the run is the exact prefix of the library run and uses the budget
//!   exactly.
//! - CMA-ES and CR-FM-NES run their library `optimize_until` /
//!   `optimize_batch_until`, whose stop is checked before every generation.
//!   It fires when the remaining budget is smaller than one population, so
//!   the run is the library run truncated after the last generation that
//!   fits; up to `popsize - 1` evaluations stay unused.
//! - Dual Annealing can overshoot its limit inside the local search and has
//!   no cooperative stop; the adapter answers its calls after the `B`-th
//!   evaluation with `+inf` without evaluating them, so the evaluated
//!   sequence is the exact `B`-prefix of the library run (see [`Da`]).
//! - DE→CMA stops its DE phase exactly at the DE share of the budget and its
//!   CMA phase like CMA-ES above.
//!
//! Each point is also projected onto the box before it is evaluated. The core
//! optimizers clamp their samples already; the projection only removes
//! last-ulp rounding of their normalized-to-real decoding (for example
//! `0.5 * 1.0 * scale + typx` can exceed `upper` by one ulp).

use std::sync::atomic::{AtomicU64, Ordering};

use fcmaes_core::{
    BiteParams, Cmaes, CmaesParams, Crfmnes, CrfmnesParams, DaParams, De, DeParams, DeepBiteOpt,
    Fitness, Rng, optimize_da,
};
use solver_api::{Objective, Outcome, Solver};

use crate::bipop::Bipop;

/// Initial step size of CMA-ES and CR-FM-NES in the normalized `[-1, 1]` box
/// (`CMAES_SIGMA` and `CRFMNES_SIGMA` of the fcmaes job worker).
pub const ES_SIGMA: f64 = 0.3;

/// Objective wrapper that counts calls and projects points onto the box.
///
/// It does not refuse calls: the adapters stop the core optimizers before
/// the budget is exceeded, and the harness guard checks that they did.
pub struct Budgeted<'a> {
    objective: Objective<'a>,
    lower: &'a [f64],
    upper: &'a [f64],
    budget: u64,
    calls: AtomicU64,
    truncated: AtomicU64,
    best: std::sync::Mutex<(f64, Vec<f64>)>,
}

impl<'a> Budgeted<'a> {
    /// Wrap `objective` for a run with `budget` evaluations.
    pub fn new(objective: Objective<'a>, lower: &'a [f64], upper: &'a [f64], budget: u64) -> Self {
        Self {
            objective,
            lower,
            upper,
            budget,
            calls: AtomicU64::new(0),
            truncated: AtomicU64::new(0),
            best: std::sync::Mutex::new((f64::INFINITY, Vec::new())),
        }
    }

    /// Evaluate the box projection of `x` and count the call.
    pub fn eval(&self, x: &[f64]) -> f64 {
        let projected: Vec<f64> = x
            .iter()
            .zip(self.lower.iter().zip(self.upper))
            .map(|(v, (lo, hi))| v.clamp(*lo, *hi))
            .collect();
        self.calls.fetch_add(1, Ordering::Relaxed);
        let y = (self.objective)(&projected);
        let mut best = self
            .best
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if y < best.0 {
            *best = (y, projected);
        }
        y
    }

    /// Calls made so far.
    pub fn used(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    /// Calls still allowed.
    pub fn remaining(&self) -> u64 {
        self.budget.saturating_sub(self.used())
    }

    /// Whether the budget is spent.
    pub fn exhausted(&self) -> bool {
        self.remaining() == 0
    }

    /// Like [`eval`](Self::eval), but once the budget is spent the call is
    /// answered with `+inf` without evaluating anything. Used only for Dual
    /// Annealing, which cannot be stopped from outside; see [`Da`].
    pub fn eval_truncated(&self, x: &[f64]) -> f64 {
        if self.exhausted() {
            self.truncated.fetch_add(1, Ordering::Relaxed);
            return f64::INFINITY;
        }
        self.eval(x)
    }

    /// Calls answered by [`eval_truncated`](Self::eval_truncated) without
    /// evaluation.
    pub fn truncated(&self) -> u64 {
        self.truncated.load(Ordering::Relaxed)
    }

    /// Outcome from the wrapper's own record of the best evaluated point.
    pub fn outcome(&self) -> Outcome {
        let best = self
            .best
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Outcome {
            x: best.1.clone(),
            y: best.0,
            evaluations: self.used(),
        }
    }
}

/// Uniform random point in the box.
pub fn uniform_point(lower: &[f64], upper: &[f64], rng: &mut Rng) -> Vec<f64> {
    lower
        .iter()
        .zip(upper)
        .map(|(lo, hi)| lo + rng.uniform01() * (hi - lo))
        .collect()
}

fn normalized_fitness(lower: &[f64], upper: &[f64]) -> Fitness {
    let mut fitness = Fitness::bounded(lower.len(), 1, lower, upper);
    fitness.set_normalize(true);
    fitness
}

/// BiteOpt (`optimize_bite` defaults, depth 1) from a uniform random guess,
/// seeded as in the GTOP benchmark's `bite_run`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bite;

impl Solver for Bite {
    fn name(&self) -> &'static str {
        "bite"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let wrapped = Budgeted::new(objective, lower, upper, max_evaluations);
        let mut rng = Rng::new(seed);
        let guess = uniform_point(lower, upper, &mut rng);
        let bite_seed = (rng.uniform01() * f64::from(u32::MAX)) as u64;
        let params = BiteParams {
            max_evaluations,
            seed: bite_seed,
            ..Default::default()
        };
        let mut optimizer = DeepBiteOpt::new(lower, upper, Some(&guess), &params, 1);
        let f = |x: &[f64]| wrapped.eval(x);
        optimizer.optimize_until(&f, &|| wrapped.exhausted());
        wrapped.outcome()
    }
}

/// Differential Evolution (`DeParams` defaults: population 31, F 0.5,
/// CR 0.9, keep 200) with its own uniform initial population.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeSolver;

impl Solver for DeSolver {
    fn name(&self) -> &'static str {
        "de"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let wrapped = Budgeted::new(objective, lower, upper, max_evaluations);
        let params = DeParams {
            max_evaluations,
            seed,
            ..Default::default()
        };
        let fitness = Fitness::bounded(lower.len(), 1, lower, upper);
        let mut de = De::new(fitness, &[], &[], None, &params);
        let f = |x: &[f64]| wrapped.eval(x);
        de.optimize_until(&f, &|| wrapped.exhausted());
        wrapped.outcome()
    }
}

/// Active CMA-ES (`CmaesParams` defaults: population 31) in the normalized
/// box, from a uniform random guess with sigma [`ES_SIGMA`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Cma;

impl Solver for Cma {
    fn name(&self) -> &'static str {
        "cma"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let wrapped = Budgeted::new(objective, lower, upper, max_evaluations);
        let mut rng = Rng::new(seed);
        let guess = uniform_point(lower, upper, &mut rng);
        let params = CmaesParams {
            max_evaluations,
            seed: rng.next_u64(),
            ..Default::default()
        };
        let mut cma = Cmaes::new(
            normalized_fitness(lower, upper),
            &guess,
            &[ES_SIGMA],
            &params,
        );
        let popsize = cma.popsize() as u64;
        let f = |x: &[f64]| wrapped.eval(x);
        cma.optimize_until(&f, 1, &|| wrapped.remaining() < popsize);
        wrapped.outcome()
    }
}

/// CR-FM-NES (`CrfmnesParams` defaults: population 32, bound penalty 1e5)
/// in the normalized box, from a uniform random guess with sigma
/// [`ES_SIGMA`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CrfmnesSolver;

impl Solver for CrfmnesSolver {
    fn name(&self) -> &'static str {
        "crfmnes"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let wrapped = Budgeted::new(objective, lower, upper, max_evaluations);
        let mut rng = Rng::new(seed);
        let guess = uniform_point(lower, upper, &mut rng);
        let params = CrfmnesParams {
            max_evaluations,
            seed: rng.next_u64(),
            ..Default::default()
        };
        let mut nes = Crfmnes::new(normalized_fitness(lower, upper), &guess, ES_SIGMA, &params);
        let popsize = nes.popsize() as u64;
        nes.optimize_batch_until(
            |rows: &[Vec<f64>]| rows.iter().map(|x| wrapped.eval(x)).collect(),
            &|| wrapped.remaining() < popsize,
        );
        wrapped.outcome()
    }
}

/// Dual Annealing (`DaParams` defaults: local search on) from a uniform
/// random guess.
///
/// Despite its documentation, core `optimize_da` can exceed
/// `max_evaluations`: its finite-difference gradient (`2D + 1` calls) and the
/// first line-search step of the local search evaluate without checking the
/// budget, and a local search can follow the global step that spent it. It
/// has no cooperative stop, so the adapter cuts the run instead: after the
/// `B`-th evaluation every further call of the core is answered with `+inf`
/// without evaluating anything (the guard never sees these calls). All
/// decisions before the `B`-th evaluation depend only on earlier values, so
/// the evaluated sequence is exactly the first `B` evaluations of the library
/// run with `max_evaluations = B`, and the full budget is available, as for
/// the other incumbents. (A reduced internal limit would have to subtract a
/// margin of several `2D + 1` blocks.)
#[derive(Clone, Copy, Debug, Default)]
pub struct Da;

impl Solver for Da {
    fn name(&self) -> &'static str {
        "da"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let wrapped = Budgeted::new(objective, lower, upper, max_evaluations);
        let mut rng = Rng::new(seed);
        let guess = uniform_point(lower, upper, &mut rng);
        let params = DaParams {
            max_evaluations,
            seed: rng.next_u64(),
            ..Default::default()
        };
        let f = |x: &[f64]| wrapped.eval_truncated(x);
        optimize_da(&f, &guess, lower.to_vec(), upper.to_vec(), &params);
        wrapped.outcome()
    }
}

/// DE→CMA sequence of the GTOP benchmark (`de_cma_run` in
/// `examples/src/runner.rs`): DE with `max(2B/5, 31)` evaluations and a
/// uniform initial population, then CMA-ES (population 31, normalized box)
/// from the DE optimum with the retry coordinator's random step size
/// `0.05 + 0.05 U[0,1]` and `max(B - DE share, 31)` evaluations.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeCma;

impl Solver for DeCma {
    fn name(&self) -> &'static str {
        "de_cma"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let wrapped = Budgeted::new(objective, lower, upper, max_evaluations);
        let dim = lower.len();
        // The retry coordinator draws the step size first, then the run seed.
        let mut rng = Rng::new(seed);
        let sdev = 0.05 + 0.05 * rng.uniform01();
        let run_seed = rng.next_u64();
        let de_budget = (max_evaluations * 2 / 5).max(31);
        let cma_budget = max_evaluations.saturating_sub(de_budget).max(31);
        let f = |x: &[f64]| wrapped.eval(x);

        let mut de = De::new(
            Fitness::bounded(dim, 1, lower, upper),
            &[],
            &[],
            None,
            &DeParams {
                max_evaluations: de_budget,
                seed: run_seed,
                ..Default::default()
            },
        );
        // Below 31 evaluations the DE share exceeds the whole budget.
        let de_stop = de_budget.min(max_evaluations);
        let de_result = de.optimize_until(&f, &|| wrapped.used() >= de_stop);
        if wrapped.exhausted() {
            return wrapped.outcome();
        }

        let mut cma = Cmaes::new(
            normalized_fitness(lower, upper),
            &de_result.x,
            &vec![sdev; dim],
            &CmaesParams {
                max_evaluations: cma_budget,
                seed: run_seed ^ 0xA076_1D64_78BD_642F,
                ..Default::default()
            },
        );
        let popsize = cma.popsize() as u64;
        cma.optimize_until(&f, 1, &|| wrapped.remaining() < popsize);
        wrapped.outcome()
    }
}

/// Names of the incumbents, in registry order.
pub const INCUMBENTS: [&str; 6] = ["bite", "de", "cma", "crfmnes", "da", "de_cma"];

/// Names of the candidates, in registry order.
pub const CANDIDATES: [&str; 6] = ["lshade", "jso", "lsrtde", "lsrtde_b", "bipop", "amalgam"];

/// All solvers available in this build: the six incumbents, BIPOP and, with
/// the `candidates` feature, L-SHADE, jSO, both L-SRTDE forms and AMaLGaM.
pub fn registry() -> Vec<Box<dyn Solver>> {
    #[allow(unused_mut)]
    let mut solvers: Vec<Box<dyn Solver>> = vec![
        Box::new(Bite),
        Box::new(DeSolver),
        Box::new(Cma),
        Box::new(CrfmnesSolver),
        Box::new(Da),
        Box::new(DeCma),
        Box::new(Bipop),
    ];
    #[cfg(feature = "candidates")]
    {
        solvers.push(Box::new(cand_de::LShade::default()));
        solvers.push(Box::new(cand_de::Jso::default()));
        solvers.push(Box::new(cand_de::LSrtde::default()));
        solvers.push(Box::new(LSrtdeBase));
        solvers.push(Box::new(cand_eda::Amalgam::default()));
    }
    solvers
}

/// L-SRTDE with the p-best difference taken from the base vector
/// (`pbest_difference_from_base`), the form of the predecessor LL-NADE.
///
/// The open descriptions of L-SRTDE print the difference from the current
/// vector, which is what `"lsrtde"` runs; that form does not converge on a
/// 30-variable rotated ellipsoid, while this one does. Both forms are
/// benchmarked because open sources cannot settle which one the authors ran.
#[cfg(feature = "candidates")]
struct LSrtdeBase;

#[cfg(feature = "candidates")]
impl Solver for LSrtdeBase {
    fn name(&self) -> &'static str {
        "lsrtde_b"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let solver = cand_de::LSrtde {
            pbest_difference_from_base: true,
            ..cand_de::LSrtde::default()
        };
        solver.minimize(objective, lower, upper, max_evaluations, seed)
    }
}

/// Look up a solver by name.
pub fn by_name(name: &str) -> Option<Box<dyn Solver>> {
    registry().into_iter().find(|solver| solver.name() == name)
}
