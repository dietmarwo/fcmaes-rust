//! Equal-budget scalar optimizer comparison.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    BiteParams, Cmaes, CmaesParams, De, DeParams, Fitness, RetryBounds, RetryConfig,
    RetryImprovement, RetryRunResult, Rng, optimize_bite, retry,
};

use crate::config::Preset;
use crate::decode::DIMENSION;
use crate::evaluate::{
    INVALID_OBJECTIVE, OuterEvaluation, analytic_seed, evaluate_training, feasible,
};

/// Scalar optimizer arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoOptimizer {
    /// Active CMA-ES retry.
    Cma,
    /// Differential-evolution retry.
    De,
    /// BiteOpt retry.
    Bite,
}

impl SoOptimizer {
    /// All equal-budget comparison arms.
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

/// Shared settings for one equal-budget arm.
#[derive(Clone, Copy, Debug)]
pub struct SoConfig {
    /// Dispatch horizon preset.
    pub preset: Preset,
    /// Requested candidate calls per arm.
    pub evaluations_per_arm: u64,
    /// Parallel retry count.
    pub retries: usize,
    /// Candidate worker threads; zero uses available parallelism.
    pub workers: usize,
    /// Root seed.
    pub seed: u64,
}

/// Work performed by nested LP solves.
#[derive(Clone, Copy, Debug, Default)]
pub struct Work {
    /// Outer candidate calls.
    pub candidate_evaluations: u64,
    /// Inner LP solves.
    pub lp_solves: u64,
    /// Cumulative simplex pivots.
    pub simplex_iterations: u64,
    /// Failed candidate replays.
    pub solver_failures: u64,
}

/// Result of one optimizer arm.
#[derive(Clone, Debug)]
pub struct SoArmResult {
    /// Arm identity.
    pub optimizer: SoOptimizer,
    /// Requested outer calls.
    pub requested_evaluations: u64,
    /// Actual nested work.
    pub work: Work,
    /// Completed retry runs.
    pub completed_retries: usize,
    /// Wall duration.
    pub elapsed: Duration,
    /// Best replayed sizing design.
    pub best: OuterEvaluation,
    /// Monotone retry improvements.
    pub improvements: Vec<RetryImprovement>,
}

#[derive(Default)]
struct Counters {
    candidates: AtomicU64,
    pivots: AtomicU64,
    failures: AtomicU64,
}

impl Counters {
    fn evaluate(&self, controls: &[f64], preset: Preset) -> f64 {
        self.candidates.fetch_add(1, Ordering::Relaxed);
        match evaluate_training(controls, preset) {
            Some(evaluation) => {
                self.pivots
                    .fetch_add(evaluation.simplex_iterations, Ordering::Relaxed);
                evaluation.objective
            }
            None => {
                self.failures.fetch_add(1, Ordering::Relaxed);
                INVALID_OBJECTIVE
            }
        }
    }

    fn work(&self) -> Work {
        let candidate_evaluations = self.candidates.load(Ordering::Relaxed);
        Work {
            candidate_evaluations,
            lp_solves: 5 * candidate_evaluations,
            simplex_iterations: self.pivots.load(Ordering::Relaxed),
            solver_failures: self.failures.load(Ordering::Relaxed),
        }
    }
}

fn jittered_seed(seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut guess = analytic_seed();
    for (index, value) in guess.iter_mut().enumerate() {
        let scale = if matches!(index, 6..=9) { 0.08 } else { 0.04 };
        *value = (*value + scale * (rng.uniform01() - 0.5)).clamp(0.0, 1.0);
    }
    guess
}

#[inline]
const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Run one scalar optimizer using the frozen retry protocol.
pub fn optimize_arm(
    optimizer: SoOptimizer,
    config: &SoConfig,
) -> Result<SoArmResult, Box<dyn Error>> {
    if config.evaluations_per_arm == 0 || config.retries == 0 {
        return Err("SO evaluation budget and retries must be positive".into());
    }
    let per_retry = config.evaluations_per_arm.div_ceil(config.retries as u64);
    let bounds = RetryBounds::new(vec![0.0; DIMENSION], vec![1.0; DIMENSION])?;
    let counters = Arc::new(Counters::default());
    let objective_counters = Arc::clone(&counters);
    let objective = move |controls: &[f64]| objective_counters.evaluate(controls, config.preset);
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
        statistic_num: 500,
        ..Default::default()
    };
    let arm_seed = retry_config.seed;
    let started = Instant::now();
    let result = retry(
        &objective,
        &bounds,
        &retry_config,
        |objective, retry_context| {
            // Bind stochastic state to a stable retry id, not to whichever
            // worker happens to claim the run.
            let run_seed = splitmix64(arm_seed ^ retry_context.run_id as u64);
            let guess = jittered_seed(run_seed);
            match optimizer {
                SoOptimizer::Cma => {
                    let fitness = Fitness::bounded(
                        DIMENSION,
                        1,
                        retry_context.bounds.lower(),
                        retry_context.bounds.upper(),
                    );
                    let mut cma = Cmaes::new(
                        fitness,
                        &guess,
                        &[0.12],
                        &CmaesParams {
                            max_evaluations: retry_context.max_evaluations,
                            seed: run_seed,
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
                        DIMENSION,
                        1,
                        retry_context.bounds.lower(),
                        retry_context.bounds.upper(),
                    );
                    let mut de = De::new(
                        fitness,
                        &guess,
                        &[0.15; DIMENSION],
                        None,
                        &DeParams {
                            popsize: 15,
                            max_evaluations: retry_context.max_evaluations,
                            seed: run_seed,
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
                SoOptimizer::Bite => {
                    let optimized = optimize_bite(
                        objective,
                        retry_context.bounds.lower(),
                        retry_context.bounds.upper(),
                        Some(&guess),
                        &BiteParams {
                            max_evaluations: retry_context.max_evaluations,
                            seed: run_seed,
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
        },
    );
    if !result.success {
        return Err(format!("{} retained no finite result", optimizer.name()).into());
    }
    let analytic = evaluate_training(&analytic_seed(), config.preset)
        .ok_or("analytic scalar seed could not be replayed")?;
    let optimized = evaluate_training(&result.x, config.preset)
        .ok_or("best scalar candidate could not be replayed")?;
    let best = if feasible(&optimized)
        && (!feasible(&analytic) || optimized.objective < analytic.objective)
    {
        optimized
    } else {
        analytic
    };
    Ok(SoArmResult {
        optimizer,
        requested_evaluations: config.evaluations_per_arm,
        work: counters.work(),
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        best,
        improvements: result.improvements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_bite_retry_retains_a_feasible_replay() {
        let result = optimize_arm(
            SoOptimizer::Bite,
            &SoConfig {
                preset: Preset::Smoke,
                evaluations_per_arm: 48,
                retries: 2,
                workers: 2,
                seed: 42,
            },
        )
        .unwrap();
        assert!(feasible(&result.best));
        assert!(result.work.candidate_evaluations > 0);
        assert_eq!(result.work.lp_solves, 5 * result.work.candidate_evaluations);
        assert!(result.work.simplex_iterations > 0);
    }

    #[test]
    fn retry_trajectories_do_not_depend_on_worker_assignment() {
        let run = |workers| {
            optimize_arm(
                SoOptimizer::Bite,
                &SoConfig {
                    preset: Preset::Smoke,
                    evaluations_per_arm: 48,
                    retries: 2,
                    workers,
                    seed: 42,
                },
            )
            .unwrap()
        };
        let serial = run(1);
        let parallel = run(2);
        assert_eq!(serial.completed_retries, parallel.completed_retries);
        assert_eq!(
            serial.work.candidate_evaluations,
            parallel.work.candidate_evaluations
        );
        assert_eq!(
            serial.best.objective.to_bits(),
            parallel.best.objective.to_bits()
        );
        assert_eq!(
            serial
                .best
                .controls
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            parallel
                .best
                .controls
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
