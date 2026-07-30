//! Equal-budget scalar optimizer comparison.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use epanet_rs::model::network::Network;
use fcmaes_core::{
    BiteParams, Cmaes, CmaesParams, De, DeParams, Fitness, RetryBounds, RetryConfig,
    RetryImprovement, RetryRunResult, Rng, optimize_bite, retry,
};

use crate::decode::seed_controls;
use crate::evaluate::{RobustEvaluation, evaluate_training};
use crate::{DIMENSION, INVALID_OBJECTIVE};

/// Scalar optimizer arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoOptimizer {
    Cma,
    De,
    Bite,
}

impl SoOptimizer {
    pub const ALL: [Self; 3] = [Self::Cma, Self::De, Self::Bite];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cma => "cma",
            Self::De => "de",
            Self::Bite => "bite",
        }
    }
}

/// One equal-budget arm configuration.
#[derive(Clone, Debug)]
pub struct SoConfig {
    pub evaluations: u64,
    pub retries: usize,
    pub workers: usize,
    pub seed: u64,
}

/// Replayed scalar arm.
#[derive(Clone, Debug)]
pub struct SoResult {
    pub optimizer: SoOptimizer,
    pub requested_evaluations: u64,
    pub actual_evaluations: u64,
    pub completed_retries: usize,
    pub elapsed: Duration,
    pub best: RobustEvaluation,
    pub improvements: Vec<RetryImprovement>,
}

fn jittered_seed(seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    seed_controls()
        .into_iter()
        .map(|value| (value + 0.18 * (rng.uniform01() - 0.5)).clamp(0.0, 1.0))
        .collect()
}

/// Optimize one arm and replay its result.
pub fn optimize(
    optimizer: SoOptimizer,
    network: &Network,
    config: &SoConfig,
) -> Result<SoResult, Box<dyn Error>> {
    if config.evaluations == 0 || config.retries == 0 {
        return Err("evaluations and retries must be positive".into());
    }
    let bounds = RetryBounds::new(vec![0.0; DIMENSION], vec![1.0; DIMENSION])?;
    let per_retry = config.evaluations.div_ceil(config.retries as u64);
    let calls = Arc::new(AtomicU64::new(0));
    let objective_calls = Arc::clone(&calls);
    let physical = Arc::new(network.clone());
    let objective_network = Arc::clone(&physical);
    let objective = move |controls: &[f64]| {
        objective_calls.fetch_add(1, Ordering::Relaxed);
        evaluate_training(controls, &objective_network)
            .map_or(INVALID_OBJECTIVE, |evaluation| evaluation.objective)
    };
    let started = Instant::now();
    let retry_config = RetryConfig {
        num_retries: config.retries,
        workers: config.workers,
        capacity: config.retries,
        max_evaluations: per_retry,
        seed: config.seed.wrapping_add(match optimizer {
            SoOptimizer::Cma => 0,
            SoOptimizer::De => 10_000,
            SoOptimizer::Bite => 20_000,
        }),
        statistic_num: 50,
        ..Default::default()
    };
    let result = retry(&objective, &bounds, &retry_config, |objective, context| {
        let guess = jittered_seed(context.seed);
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
                        seed: context.seed,
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
                        popsize: 12,
                        max_evaluations: context.max_evaluations,
                        seed: context.seed,
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
                        seed: context.seed,
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
    let optimized = evaluate_training(&result.x, &physical)?;
    let witness = evaluate_training(&seed_controls(), &physical)?;
    let best = if witness.objective < optimized.objective {
        witness
    } else {
        optimized
    };
    Ok(SoResult {
        optimizer,
        requested_evaluations: config.evaluations,
        actual_evaluations: calls.load(Ordering::Relaxed),
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        best,
        improvements: result.improvements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network;

    #[test]
    fn tiny_bite_campaign_returns_finite_replay() {
        let network = network::load().unwrap();
        let result = optimize(
            SoOptimizer::Bite,
            &network,
            &SoConfig {
                evaluations: 24,
                retries: 1,
                workers: 1,
                seed: 42,
            },
        )
        .unwrap();
        assert!(result.best.objective.is_finite());
    }
}
