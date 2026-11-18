//! BIPOP-aCMA-ES: a bi-population restart schedule over core `Cmaes`.
//!
//! Source: N. Hansen, "Benchmarking a BI-population CMA-ES on the BBOB-2009
//! function testbed", GECCO 2009 companion. The primary paper could not be
//! retrieved here; the schedule was checked against two secondary sources:
//! the description of Hansen's BIPOP in I. Loshchilov, M. Schoenauer and
//! M. Sebag, "Alternative restart strategies for CMA-ES" (PPSN 2012,
//! arXiv:1207.0206), and the BIPOP example of the DEAP library
//! (`examples/es/cma_bipop.py`, commit `1509d0b`). Hansen's reference
//! implementation pycma (`cma.fmin(..., bipop=True)`) was compared as well;
//! where it differs, this is stated below.
//!
//! # Schedule
//!
//! With `λ_def = 4 + ⌊3 ln D⌋` and the default step size `σ_0`:
//!
//! 1. The first run is the large regime with multiplier `2^0`: `λ_def`, `σ_0`.
//!    Its evaluations are booked to neither regime (DEAP: "First run does
//!    not count into the budget").
//! 2. The `k`-th large restart (`k ≥ 1`) uses `λ = 2^k λ_def` and `σ_0`.
//! 3. A small restart uses `λ_s = ⌊λ_def (½ λ_next / λ_def)^(U₁²)⌋` and
//!    `σ_s = σ_0 · 10^(−2 U₂)`, `U₁, U₂ ~ U[0,1]`, where `λ_next = 2^(k+1)
//!    λ_def` is the population the next large restart would use after `k`
//!    large restarts; hence `λ_s ∈ [λ_def, 2^k λ_def]`, up to the latest
//!    large population. Both DEAP and pycma index `λ_large` this way.
//! 4. A small run stops after at most `max(1, ⌈½ L / λ_s⌉)` generations,
//!    where `L` is the number of evaluations of the most recent large
//!    restart (DEAP: `MAXITER = 0.5 * largebudget[-1] / lambda_`, checked
//!    after each generation).
//! 5. The next restart is small if the small regime has used strictly fewer
//!    evaluations so far than the large regime, otherwise large (DEAP:
//!    `sum(smallbudget) < sum(largebudget)`; pycma uses the same strict
//!    comparison). A tie, including `0 = 0` after the first run, goes to the
//!    large regime, so the first restart is large.
//! 6. Every run starts at a uniform random point of the box.
//! 7. A run also ends when core `Cmaes` reports a stop condition or when the
//!    remaining budget is smaller than one generation of that run; the
//!    schedule ends when the next run's population does not fit into the
//!    remaining budget.
//!
//! # Verified against the sources
//!
//! - Doubling of the large population, the `λ_s` and `σ_s` formulas and the
//!   choice of the regime with fewer evaluations (Loshchilov et al.; DEAP;
//!   pycma, whose `0.01**U` equals `10^(−2U)`).
//! - `σ_0 = 0.2` of the box width: Hansen's BBOB setting `σ_0 = 2` on
//!   `[−5, 5]` (DEAP: `SIGMA0 = 2.0  # 1/5th of the domain [-5 5]`). `Cmaes`
//!   works in normalized coordinates where the box is `[−1, 1]` (width 2), so
//!   the step size passed to it is `0.4`.
//! - First run in the large regime with multiplier 1 and excluded from the
//!   budget accounting; small-run cap of half the most recent large run;
//!   strict comparison with ties to the large regime (DEAP).
//!
//! # Differences between the sources, and choices
//!
//! - pycma books the first run to the *small* budget ("its budget accounting
//!   is the same as in case of small population") and caps a small run at
//!   half of the *cumulated* large-regime evaluations (`0.5 * sum(large_i) /
//!   popsize` generations, at most its default `maxiter`). The repository's
//!   job-worker BIPOP also uses the cumulated large budget. This module
//!   follows DEAP (first run uncounted, most recent large run).
//! - DEAP and pycma cap the generations of large runs with Hansen's default
//!   `MaxIter` (DEAP `100 + 50 (N+3)² / √λ`). Core `Cmaes` has no generation
//!   limit; large runs end through the fcmaes default termination criteria
//!   (`CmaesParams` defaults) or the budget.
//! - DEAP runs a fixed number of restarts and forces the last one into the
//!   large regime; here the evaluation budget ends the schedule instead, and
//!   the number of large restarts is limited only by the budget.
//! - The core `Cmaes` is fcmaes's active CMA-ES with its own strategy
//!   parameters (`max_evaluations` = remaining budget at the start of each
//!   run). Runs use the ask/tell interface; asked points are projected onto
//!   the box (a no-op apart from last-ulp rounding) before evaluation.

use fcmaes_core::{Cmaes, CmaesParams, Fitness, Rng};
use solver_api::{Objective, Outcome, Solver};

use crate::solvers::{Budgeted, uniform_point};

/// Initial step size as a fraction of the box width (Hansen's BBOB setting,
/// `σ_0 = 2` on `[−5, 5]`).
pub const SIGMA0_BOX_FRACTION: f64 = 0.2;

/// Width of the normalized `Cmaes` box `[-1, 1]`.
const NORMALIZED_WIDTH: f64 = 2.0;

/// Initial step size passed to `Cmaes` (normalized coordinates).
pub const SIGMA0_NORMALIZED: f64 = SIGMA0_BOX_FRACTION * NORMALIZED_WIDTH;

/// Default population `4 + ⌊3 ln D⌋`.
pub fn default_population(dim: usize) -> usize {
    4 + (3.0 * (dim as f64).ln()).floor() as usize
}

/// Restart regime of one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    /// The first run: large regime with multiplier 1, not booked.
    First,
    /// A large-population restart.
    Large,
    /// A small-population restart.
    Small,
}

/// Record of one run of the schedule (for tests and diagnostics).
#[derive(Clone, Debug)]
pub struct RunRecord {
    /// Regime of the run.
    pub regime: Regime,
    /// Population size.
    pub population: usize,
    /// Initial step size in normalized coordinates.
    pub sigma: f64,
    /// Generation cap of a small run (`None` for the large regime).
    pub max_generations: Option<u64>,
    /// Evaluations the run used.
    pub evaluations: u64,
    /// Core stop code at the end of the run (0: ended by a cap or the budget).
    pub stop: i32,
}

/// BIPOP-aCMA-ES solver.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bipop;

impl Bipop {
    /// Run the schedule and return the outcome with the per-run records.
    pub fn minimize_with_log(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> (Outcome, Vec<RunRecord>) {
        let dim = lower.len();
        let wrapped = Budgeted::new(objective, lower, upper, max_evaluations);
        let mut rng = Rng::new(seed);
        let lambda_def = default_population(dim);
        let (mut small_used, mut large_used) = (0u64, 0u64);
        let mut last_large = 0u64;
        let mut large_restarts = 0u32;
        let mut log: Vec<RunRecord> = Vec::new();
        loop {
            let (regime, population, sigma, max_generations) = if log.is_empty() {
                (Regime::First, lambda_def, SIGMA0_NORMALIZED, None)
            } else if small_used < large_used {
                let lambda_next = (lambda_def as f64) * 2f64.powi(large_restarts as i32 + 1);
                let u1 = rng.uniform01();
                let u2 = rng.uniform01();
                let population = ((lambda_def as f64)
                    * (0.5 * lambda_next / lambda_def as f64).powf(u1 * u1))
                .floor() as usize;
                let population = population.max(lambda_def);
                let sigma = SIGMA0_NORMALIZED * 10f64.powf(-2.0 * u2);
                let cap = (0.5 * last_large as f64 / population as f64).ceil() as u64;
                (Regime::Small, population, sigma, Some(cap.max(1)))
            } else {
                large_restarts += 1;
                let population = lambda_def.saturating_mul(1usize << large_restarts.min(60));
                (Regime::Large, population, SIGMA0_NORMALIZED, None)
            };
            if (population as u64) > wrapped.remaining() {
                break;
            }
            let guess = uniform_point(lower, upper, &mut rng);
            let mut fitness = Fitness::bounded(dim, 1, lower, upper);
            fitness.set_normalize(true);
            let mut cma = Cmaes::new(
                fitness,
                &guess,
                &[sigma],
                &CmaesParams {
                    popsize: population as i32,
                    max_evaluations: wrapped.remaining(),
                    seed: rng.next_u64(),
                    ..Default::default()
                },
            );
            let start = wrapped.used();
            let mut stop = 0;
            let mut generations = 0u64;
            while wrapped.remaining() >= population as u64
                && max_generations.is_none_or(|cap| generations < cap)
            {
                let xs = cma.ask();
                let ys: Vec<f64> = xs.iter().map(|x| wrapped.eval(x)).collect();
                stop = cma.tell(&ys);
                generations += 1;
                if stop != 0 {
                    break;
                }
            }
            let evaluations = wrapped.used() - start;
            match regime {
                Regime::First => {}
                Regime::Large => {
                    large_used += evaluations;
                    last_large = evaluations;
                }
                Regime::Small => small_used += evaluations,
            }
            log.push(RunRecord {
                regime,
                population,
                sigma,
                max_generations,
                evaluations,
                stop,
            });
        }
        (wrapped.outcome(), log)
    }
}

impl Solver for Bipop {
    fn name(&self) -> &'static str {
        "bipop"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        self.minimize_with_log(objective, lower, upper, max_evaluations, seed)
            .0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_population_and_sigma() {
        assert_eq!(default_population(2), 6);
        assert_eq!(default_population(10), 10);
        assert_eq!(default_population(114), 18);
        assert!((SIGMA0_NORMALIZED - 0.4).abs() < 1e-15);
    }

    #[test]
    fn schedule_follows_the_published_rules() {
        let f = |x: &[f64]| {
            // Rastrigin: multimodal, so runs stop early and restarts happen.
            10.0 * x.len() as f64
                + x.iter()
                    .map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos())
                    .sum::<f64>()
        };
        let (lo, hi) = (vec![-5.0; 5], vec![5.0; 5]);
        let (outcome, log) = Bipop.minimize_with_log(&f, &lo, &hi, 200_000, 3);
        assert!(outcome.evaluations <= 200_000);
        let lambda_def = default_population(5);
        assert_eq!(log[0].regime, Regime::First);
        assert_eq!(log[0].population, lambda_def);
        assert!((log[0].sigma - 0.4).abs() < 1e-15);
        // The first run is not booked, so the first restart is large.
        assert_eq!(log[1].regime, Regime::Large);
        assert_eq!(log[1].population, 2 * lambda_def);
        assert!(log.iter().any(|run| run.regime == Regime::Small));
        let mut large = 0u32;
        let (mut small_used, mut large_used, mut last_large) = (0u64, 0u64, 0u64);
        for (index, run) in log.iter().enumerate() {
            if index > 0 {
                let expected = if small_used < large_used {
                    Regime::Small
                } else {
                    Regime::Large
                };
                assert_eq!(run.regime, expected, "run {index}");
            }
            match run.regime {
                Regime::First => assert_eq!(run.max_generations, None),
                Regime::Large => {
                    large += 1;
                    assert_eq!(run.population, lambda_def << large);
                    assert!((run.sigma - 0.4).abs() < 1e-15);
                    assert_eq!(run.max_generations, None);
                    large_used += run.evaluations;
                    last_large = run.evaluations;
                }
                Regime::Small => {
                    assert!(run.population >= lambda_def);
                    assert!(run.population <= lambda_def << large);
                    assert!(run.sigma <= 0.4 && run.sigma >= 0.004);
                    let cap = (0.5 * last_large as f64 / run.population as f64)
                        .ceil()
                        .max(1.0) as u64;
                    assert_eq!(run.max_generations, Some(cap));
                    assert!(run.evaluations <= cap * run.population as u64);
                    small_used += run.evaluations;
                }
            }
        }
    }
}
