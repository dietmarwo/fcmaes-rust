//! Equal-budget scalar topology-and-sizing comparison.

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fcmaes_core::{
    BiteParams, Cmaes, CmaesParams, De, DeParams, Fitness, RetryBounds, RetryConfig,
    RetryImprovement, RetryRunResult, Rng, optimize_bite, retry,
};

use crate::INVALID_COST;
use crate::decode::{baseline_controls, dimension};
use crate::evaluate::{Evaluation, evaluate};
use crate::fem::{Scenario, WorkCounter, WorkSnapshot};
use crate::ground::GroundStructure;

/// Scalar comparison arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoOptimizer {
    /// Seeded active CMA-ES retry.
    Cma,
    /// Seeded differential-evolution retry.
    De,
    /// Seeded BiteOpt retry.
    Bite,
    /// Uniform-random-start BiteOpt control.
    BiteUnseeded,
}

impl SoOptimizer {
    /// Frozen arm order.
    pub const ALL: [Self; 4] = [Self::Cma, Self::De, Self::Bite, Self::BiteUnseeded];

    /// Artifact label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cma => "cma-seeded",
            Self::De => "de-seeded",
            Self::Bite => "bite-seeded",
            Self::BiteUnseeded => "bite-unseeded",
        }
    }

    /// Whether the deterministic braced design is supplied.
    #[must_use]
    pub const fn seeded(self) -> bool {
        !matches!(self, Self::BiteUnseeded)
    }
}

/// Common scalar budget.
#[derive(Clone, Debug)]
pub struct SoConfig {
    /// Requested candidate calls per arm.
    pub evaluations_per_arm: u64,
    /// Independent retries.
    pub retries: usize,
    /// Worker threads; zero uses available parallelism.
    pub workers: usize,
    /// Root seed.
    pub seed: u64,
}

/// One completed arm.
#[derive(Clone, Debug)]
pub struct SoArmResult {
    /// Optimizer identity.
    pub optimizer: SoOptimizer,
    /// Requested objective calls.
    pub requested_evaluations: u64,
    /// Optimizer-reported objective calls.
    pub actual_evaluations: u64,
    /// Completed retry count.
    pub completed_retries: usize,
    /// Wall duration.
    pub elapsed: Duration,
    /// Replayed incumbent.
    pub best: Evaluation,
    /// Monotone retry improvements.
    pub improvements: Vec<RetryImprovement>,
    /// Physical-work accounting including replay.
    pub work: WorkSnapshot,
}

#[inline]
const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn run_seed(optimizer: SoOptimizer, root: u64) -> u64 {
    root.wrapping_add(match optimizer {
        SoOptimizer::Cma => 0,
        SoOptimizer::De => 10_000,
        SoOptimizer::Bite => 20_000,
        SoOptimizer::BiteUnseeded => 30_000,
    })
}

/// Run one scalar arm.
pub fn optimize_arm(
    optimizer: SoOptimizer,
    config: &SoConfig,
) -> Result<SoArmResult, Box<dyn Error>> {
    if config.evaluations_per_arm == 0 || config.retries == 0 {
        return Err("SO budget and retry count must be positive".into());
    }
    let ground = Arc::new(GroundStructure::reference());
    let decision_dimension = dimension(&ground);
    let baseline = baseline_controls(&ground);
    let counter = Arc::new(WorkCounter::default());
    let objective_ground = Arc::clone(&ground);
    let objective_counter = Arc::clone(&counter);
    let objective = move |controls: &[f64]| {
        evaluate(
            controls,
            &objective_ground,
            Scenario::TRAINING,
            false,
            &objective_counter,
        )
        .map_or(INVALID_COST, |evaluation| evaluation.objective)
    };
    let per_retry = config.evaluations_per_arm.div_ceil(config.retries as u64);
    let bounds = RetryBounds::new(vec![0.0; decision_dimension], vec![1.0; decision_dimension])?;
    let arm_seed = run_seed(optimizer, config.seed);
    let retry_config = RetryConfig {
        num_retries: config.retries,
        workers: config.workers,
        capacity: config.retries,
        max_evaluations: per_retry,
        seed: arm_seed,
        statistic_num: 200,
        ..Default::default()
    };
    let started = Instant::now();
    let result = retry(&objective, &bounds, &retry_config, |objective, context| {
        let seed = splitmix64(arm_seed ^ context.run_id as u64);
        let mut rng = Rng::new(seed);
        let mut guess = if optimizer.seeded() {
            baseline.clone()
        } else {
            (0..decision_dimension).map(|_| rng.uniform01()).collect()
        };
        if optimizer.seeded() {
            for value in &mut guess {
                *value = (*value + 0.02 * (rng.uniform01() - 0.5)).clamp(0.0, 1.0);
            }
        }
        match optimizer {
            SoOptimizer::Cma => {
                let fitness = Fitness::bounded(
                    decision_dimension,
                    1,
                    context.bounds.lower(),
                    context.bounds.upper(),
                );
                let mut cma = Cmaes::new(
                    fitness,
                    &guess,
                    &[0.10],
                    &CmaesParams {
                        max_evaluations: context.max_evaluations,
                        seed,
                        stop_tol_hist_fun: 0.0,
                        ..Default::default()
                    },
                );
                let optimized = cma.optimize(objective, 1);
                RetryRunResult {
                    x: optimized.x,
                    y: optimized.y,
                    evaluations: optimized.evaluations,
                }
            }
            SoOptimizer::De => {
                let fitness = Fitness::bounded(
                    decision_dimension,
                    1,
                    context.bounds.lower(),
                    context.bounds.upper(),
                );
                let sigma = vec![0.12; decision_dimension];
                let mut de = De::new(
                    fitness,
                    &guess,
                    &sigma,
                    None,
                    &DeParams {
                        popsize: 31,
                        max_evaluations: context.max_evaluations,
                        seed,
                        ..Default::default()
                    },
                );
                let optimized = de.optimize(objective);
                RetryRunResult {
                    x: optimized.x,
                    y: optimized.y,
                    evaluations: optimized.evaluations,
                }
            }
            SoOptimizer::Bite | SoOptimizer::BiteUnseeded => {
                let optimized = optimize_bite(
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
                    x: optimized.x,
                    y: optimized.y,
                    evaluations: optimized.evaluations,
                }
            }
        }
    });
    if !result.success {
        return Err(format!("{} retained no finite candidate", optimizer.name()).into());
    }
    let optimized = evaluate(&result.x, &ground, Scenario::TRAINING, false, &counter)
        .ok_or("optimized controls failed replay")?;
    let best = if optimizer.seeded() {
        let seed = evaluate(&baseline, &ground, Scenario::TRAINING, false, &counter)
            .ok_or("baseline failed replay")?;
        if seed.objective <= optimized.objective {
            seed
        } else {
            optimized
        }
    } else {
        optimized
    };
    Ok(SoArmResult {
        optimizer,
        requested_evaluations: config.evaluations_per_arm,
        actual_evaluations: result.evaluations,
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        best,
        improvements: result.improvements,
        work: counter.snapshot(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_seeded_bite_run_is_replayable() {
        let result = optimize_arm(
            SoOptimizer::Bite,
            &SoConfig {
                evaluations_per_arm: 64,
                retries: 1,
                workers: 1,
                seed: 42,
            },
        )
        .unwrap();
        assert!(result.best.objective.is_finite());
        assert!(
            (crate::decode::MIN_ACTIVE..=crate::decode::MAX_ACTIVE)
                .contains(&result.best.active_count)
        );
        assert!(result.work.factorizations > 0);
        assert!(result.work.fem_solves <= 2 * result.work.candidate_evaluations);
    }

    #[test]
    fn retry_streams_do_not_depend_on_worker_scheduling() {
        let run = |workers| {
            optimize_arm(
                SoOptimizer::Bite,
                &SoConfig {
                    evaluations_per_arm: 128,
                    retries: 2,
                    workers,
                    seed: 91,
                },
            )
            .unwrap()
        };
        let serial = run(1);
        let parallel = run(2);
        assert_eq!(serial.actual_evaluations, parallel.actual_evaluations);
        assert_eq!(serial.best.objective, parallel.best.objective);
        assert_eq!(serial.best.controls, parallel.best.controls);
    }
}
