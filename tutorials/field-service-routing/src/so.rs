//! Equal-budget scalar optimizer comparison on a discontinuous decoder.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    BiteParams, Cmaes, CmaesParams, De, DeParams, Fitness, RetryBounds, RetryConfig,
    RetryImprovement, RetryRunResult, Rng, optimize_bite, retry,
};

use crate::INVALID_OBJECTIVE;
use crate::instance::{DIMENSION, Instance};
use crate::scenarios::{RobustEvaluation, evaluate_training, robust_seed_controls};

/// Scalar optimizer identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoOptimizer {
    /// Active CMA-ES retry.
    Cma,
    /// Differential evolution retry.
    De,
    /// BiteOpt retry.
    Bite,
}

impl SoOptimizer {
    /// All equal-budget arms.
    pub const ALL: [Self; 3] = [Self::Cma, Self::De, Self::Bite];

    /// Stable artifact label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cma => "cma",
            Self::De => "de",
            Self::Bite => "bite",
        }
    }
}

/// One scalar run protocol.
#[derive(Clone, Debug)]
pub struct SoConfig {
    /// Total calls requested.
    pub evaluations: u64,
    /// Parallel retry count.
    pub retries: usize,
    /// Candidate workers, zero uses available CPUs.
    pub workers: usize,
    /// Root seed.
    pub seed: u64,
}

/// Scalar arm result.
#[derive(Clone, Debug)]
pub struct SoResult {
    /// Optimizer.
    pub optimizer: SoOptimizer,
    /// Requested calls.
    pub requested_evaluations: u64,
    /// Actual objective calls.
    pub actual_evaluations: u64,
    /// Completed retries.
    pub completed_retries: usize,
    /// Wall duration.
    pub elapsed: Duration,
    /// Best replayed plan, possibly infeasible and labelled as such.
    pub best: RobustEvaluation,
    /// Best candidate returned by the search before the seed fallback.
    pub search_best: RobustEvaluation,
    /// Whether search found a feasible plan cheaper than the seed.
    pub search_found_feasible_improvement: bool,
    /// Monotone retry improvements.
    pub improvements: Vec<RetryImprovement>,
}

fn jittered_witness(instance: &Instance, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    robust_seed_controls(instance)
        .into_iter()
        .map(|value| (value + 0.12 * (rng.uniform01() - 0.5)).clamp(0.0, 1.0))
        .collect()
}

fn retry_seed(root: u64, run_id: usize) -> u64 {
    let mut value = root ^ (run_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Run one scalar optimization arm.
pub fn optimize(
    optimizer: SoOptimizer,
    instance: &Instance,
    config: &SoConfig,
) -> Result<SoResult, Box<dyn Error>> {
    if config.evaluations == 0 || config.retries == 0 {
        return Err("evaluations and retries must be positive".into());
    }
    let bounds = RetryBounds::new(vec![0.0; DIMENSION], vec![1.0; DIMENSION])?;
    let per_retry = config.evaluations.div_ceil(config.retries as u64);
    let calls = Arc::new(AtomicU64::new(0));
    let objective_calls = Arc::clone(&calls);
    let physical = Arc::new(instance.clone());
    let objective_instance = Arc::clone(&physical);
    let objective = move |controls: &[f64]| {
        objective_calls.fetch_add(1, Ordering::Relaxed);
        evaluate_training(controls, &objective_instance)
            .map_or(INVALID_OBJECTIVE, |evaluation| evaluation.objective)
    };
    let started = Instant::now();
    let arm_seed = config.seed.wrapping_add(match optimizer {
        SoOptimizer::Cma => 0,
        SoOptimizer::De => 10_000,
        SoOptimizer::Bite => 20_000,
    });
    let retry_config = RetryConfig {
        num_retries: config.retries,
        workers: config.workers,
        capacity: config.retries,
        max_evaluations: per_retry,
        seed: arm_seed,
        statistic_num: 250,
        ..Default::default()
    };
    let result = retry(&objective, &bounds, &retry_config, |objective, context| {
        // `fcmaes-core` assigns context seeds from worker-local streams. Key
        // the tutorial's runs by run_id so changing the worker count does not
        // change the numerical experiment.
        let seed = retry_seed(arm_seed, context.run_id);
        let guess = if context.run_id == 0 {
            robust_seed_controls(&physical)
        } else {
            jittered_witness(&physical, seed)
        };
        match optimizer {
            SoOptimizer::Cma => {
                let fitness =
                    Fitness::bounded(DIMENSION, 1, context.bounds.lower(), context.bounds.upper());
                let mut solver = Cmaes::new(
                    fitness,
                    &guess,
                    &[0.18],
                    &CmaesParams {
                        max_evaluations: context.max_evaluations,
                        seed,
                        stop_tol_hist_fun: 0.0,
                        ..Default::default()
                    },
                );
                let result = solver.optimize(objective, 1);
                RetryRunResult {
                    x: result.x,
                    y: result.y,
                    evaluations: result.evaluations,
                }
            }
            SoOptimizer::De => {
                let fitness =
                    Fitness::bounded(DIMENSION, 1, context.bounds.lower(), context.bounds.upper());
                let mut solver = De::new(
                    fitness,
                    &guess,
                    &[0.2; DIMENSION],
                    None,
                    &DeParams {
                        popsize: 15,
                        max_evaluations: context.max_evaluations,
                        seed,
                        ..Default::default()
                    },
                );
                let result = solver.optimize(objective);
                RetryRunResult {
                    x: result.x,
                    y: result.y,
                    evaluations: result.evaluations,
                }
            }
            SoOptimizer::Bite => {
                let result = optimize_bite(
                    objective,
                    context.bounds.lower(),
                    context.bounds.upper(),
                    Some(&guess),
                    &BiteParams {
                        max_evaluations: context.max_evaluations,
                        seed,
                        ..Default::default()
                    },
                    1,
                );
                RetryRunResult {
                    x: result.x,
                    y: result.y,
                    evaluations: result.evaluations,
                }
            }
        }
    });
    if !result.success {
        return Err(format!("{} retained no finite candidate", optimizer.name()).into());
    }
    let search_best =
        evaluate_training(&result.x, instance).ok_or("best scalar candidate cannot replay")?;
    let witness = evaluate_training(&robust_seed_controls(instance), instance)
        .ok_or("witness candidate cannot replay")?;
    let search_found_feasible_improvement =
        search_best.feasible() && search_best.worst_cost < witness.worst_cost - 1.0e-9;
    let best = if witness.objective < search_best.objective {
        witness.clone()
    } else {
        search_best.clone()
    };
    Ok(SoResult {
        optimizer,
        requested_evaluations: config.evaluations,
        actual_evaluations: calls.load(Ordering::Relaxed),
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        best,
        search_best,
        search_found_feasible_improvement,
        improvements: result.improvements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::load_primary;

    #[test]
    fn tiny_bite_retry_replays() {
        let result = optimize(
            SoOptimizer::Bite,
            &load_primary().unwrap(),
            &SoConfig {
                evaluations: 80,
                retries: 2,
                workers: 2,
                seed: 42,
            },
        )
        .unwrap();
        assert!(result.best.objective.is_finite());
        assert!(result.actual_evaluations > 0);
    }

    #[test]
    fn retry_runs_are_worker_count_invariant() {
        let instance = load_primary().unwrap();
        let config = |workers| SoConfig {
            evaluations: 600,
            retries: 3,
            workers,
            seed: 42,
        };
        let serial = optimize(SoOptimizer::De, &instance, &config(1)).unwrap();
        let parallel = optimize(SoOptimizer::De, &instance, &config(4)).unwrap();
        assert_eq!(serial.actual_evaluations, parallel.actual_evaluations);
        assert_eq!(serial.best.objective, parallel.best.objective);
        assert_eq!(serial.search_best.objective, parallel.search_best.objective);
        assert_eq!(
            serial.search_found_feasible_improvement,
            parallel.search_found_feasible_improvement
        );
    }
}
