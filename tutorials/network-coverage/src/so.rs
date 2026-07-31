//! Equal-budget classic vertex-cover comparisons.

use std::error::Error;
use std::time::{Duration, Instant};

use fcmaes_core::{
    De, DeParams, Fitness, RetryBounds, RetryConfig, RetryImprovement, RetryRunResult, Rng, retry,
};

use crate::coverage::{CoverageMetrics, GROUP_WEIGHT_EXPONENT, evaluate};
use crate::decode::{UPPER, controls, decode};
use crate::instance::Instance;
use crate::oracle::{matching_cover, verify_cover, weighted_primal_dual};

/// Scalar objective optimized by an arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoObjective {
    /// Minimize the number of vertices in an ordinary-edge cover.
    Cardinality,
    /// Minimize the node cost of an ordinary-edge cover.
    Weighted,
}

impl SoObjective {
    /// Stable artifact label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cardinality => "de-cardinality",
            Self::Weighted => "de-weighted",
        }
    }
}

/// Common scalar budget.
#[derive(Clone, Debug)]
pub struct SoConfig {
    /// Candidate calls requested for each optimized arm.
    pub evaluations_per_arm: u64,
    /// Independent parallel retries.
    pub retries: usize,
    /// Worker threads; zero uses available parallelism.
    pub workers: usize,
    /// Root seed.
    pub seed: u64,
}

/// One completed optimized arm.
#[derive(Clone, Debug)]
pub struct SoResult {
    /// Objective identity.
    pub objective: SoObjective,
    /// Requested calls.
    pub requested_evaluations: u64,
    /// Optimizer-reported calls.
    pub actual_evaluations: u64,
    /// Completed retries.
    pub completed_retries: usize,
    /// Wall time.
    pub elapsed: Duration,
    /// Replayed certified construction supplied before optimization.
    pub seed: CoverageMetrics,
    /// Replayed raw optimizer incumbent before certified-seed fallback.
    pub optimizer_incumbent: Option<CoverageMetrics>,
    /// Raw optimizer objective, even when its controls cannot be replayed.
    pub optimizer_objective: f64,
    /// Replayed retained incumbent after the explicit fallback comparison.
    pub best: CoverageMetrics,
    /// Machine-readable origin of the retained incumbent.
    pub retained_source: &'static str,
    /// Retained objective minus certified-seed objective.
    pub delta_vs_seed: f64,
    /// Whether the ordinary-edge cover passed independent replay.
    pub verified: bool,
    /// Monotone retry improvements.
    pub improvements: Vec<RetryImprovement>,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn objective_value(instance: &Instance, selected: &[bool], objective: SoObjective) -> f64 {
    let uncovered = instance
        .edges
        .iter()
        .filter(|edge| !selected[edge.u] && !selected[edge.v])
        .count();
    match objective {
        SoObjective::Cardinality => {
            selected.iter().filter(|value| **value).count() as f64 + 2.0 * uncovered as f64
        }
        SoObjective::Weighted => {
            let cost = selected
                .iter()
                .zip(&instance.costs)
                .filter(|(chosen, _)| **chosen)
                .map(|(_, cost)| cost)
                .sum::<f64>();
            cost + (instance.costs.iter().sum::<f64>() + 1.0) * uncovered as f64
        }
    }
}

fn seed(instance: &Instance, objective: SoObjective) -> Vec<bool> {
    match objective {
        SoObjective::Cardinality => matching_cover(instance).selected,
        SoObjective::Weighted => weighted_primal_dual(instance).selected,
    }
}

/// Optimize one classic vertex-cover objective with integer-aware DE retry.
pub fn optimize(
    instance: &Instance,
    objective: SoObjective,
    config: &SoConfig,
) -> Result<SoResult, Box<dyn Error>> {
    if config.evaluations_per_arm == 0 || config.retries == 0 {
        return Err("SO budget and retry count must be positive".into());
    }
    let dimension = instance.nodes();
    let seed_mask = seed(instance, objective);
    let seed_controls = controls(&seed_mask);
    let per_retry = config.evaluations_per_arm.div_ceil(config.retries as u64);
    let bounds = RetryBounds::new(vec![0.0; dimension], vec![UPPER; dimension])?;
    let retry_config = RetryConfig {
        num_retries: config.retries,
        workers: config.workers,
        capacity: config.retries,
        max_evaluations: per_retry,
        seed: config.seed.wrapping_add(match objective {
            SoObjective::Cardinality => 10_000,
            SoObjective::Weighted => 20_000,
        }),
        statistic_num: 200,
        ..Default::default()
    };
    let objective_fn = |candidate: &[f64]| {
        decode(candidate).map_or(f64::INFINITY, |selected| {
            objective_value(instance, &selected, objective)
        })
    };
    let started = Instant::now();
    let root_seed = retry_config.seed;
    let result = retry(
        &objective_fn,
        &bounds,
        &retry_config,
        |function, context| {
            let run_seed = splitmix64(root_seed ^ context.run_id as u64);
            let mut rng = Rng::new(run_seed);
            let mut guess = seed_controls.clone();
            for value in &mut guess {
                if rng.uniform01() < 0.08 {
                    *value = if *value < 1.0 { 1.5 } else { 0.5 };
                }
            }
            let fitness =
                Fitness::bounded(dimension, 1, context.bounds.lower(), context.bounds.upper());
            let mut de = De::new(
                fitness,
                &guess,
                &vec![0.45; dimension],
                Some(vec![true; dimension]),
                &DeParams {
                    popsize: 31,
                    max_evaluations: context.max_evaluations,
                    seed: run_seed,
                    ..Default::default()
                },
            );
            let optimized = de.optimize(function);
            RetryRunResult {
                x: optimized.x,
                y: optimized.y,
                evaluations: optimized.evaluations,
            }
        },
    );
    let optimizer_incumbent = result
        .success
        .then(|| decode(&result.x))
        .flatten()
        .and_then(|selected| evaluate(instance, &selected, GROUP_WEIGHT_EXPONENT));
    let seed_metrics =
        evaluate(instance, &seed_mask, GROUP_WEIGHT_EXPONENT).ok_or("SO seed failed replay")?;
    let seed_objective = objective_value(instance, &seed_mask, objective);
    let optimizer_objective = optimizer_incumbent.as_ref().map_or(result.y, |metrics| {
        objective_value(instance, &metrics.selected, objective)
    });
    let (best_mask, retained_source) = if optimizer_incumbent.as_ref().is_some_and(|metrics| {
        objective_value(instance, &metrics.selected, objective) < seed_objective
    }) {
        (
            optimizer_incumbent
                .as_ref()
                .expect("checked optimizer incumbent")
                .selected
                .clone(),
            "optimizer",
        )
    } else {
        (seed_mask, "seed")
    };
    let best = evaluate(instance, &best_mask, GROUP_WEIGHT_EXPONENT)
        .ok_or("SO incumbent failed replay")?;
    let retained_objective = objective_value(instance, &best.selected, objective);
    let verified = verify_cover(instance, &best.selected);
    if !verified {
        return Err("SO retained an ordinary-edge-infeasible incumbent".into());
    }
    Ok(SoResult {
        objective,
        requested_evaluations: config.evaluations_per_arm,
        actual_evaluations: result.evaluations,
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        seed: seed_metrics,
        optimizer_incumbent,
        optimizer_objective,
        best,
        retained_source,
        delta_vs_seed: retained_objective - seed_objective,
        verified,
        improvements: result.improvements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{FIXTURES, generate};

    #[test]
    fn smoke_arms_keep_verified_seed_or_better() {
        let instance = generate(&FIXTURES[0]).unwrap();
        for objective in [SoObjective::Cardinality, SoObjective::Weighted] {
            let result = optimize(
                &instance,
                objective,
                &SoConfig {
                    evaluations_per_arm: 64,
                    retries: 1,
                    workers: 1,
                    seed: 42,
                },
            )
            .unwrap();
            assert!(result.verified);
            assert_eq!(result.best.uncovered_edges, 0);
        }
    }

    #[test]
    fn retry_incumbent_does_not_depend_on_worker_scheduling() {
        let instance = generate(&FIXTURES[0]).unwrap();
        let run = |workers| -> (u64, u64, Vec<bool>) {
            let result = optimize(
                &instance,
                SoObjective::Weighted,
                &SoConfig {
                    evaluations_per_arm: 128,
                    retries: 2,
                    workers,
                    seed: 91,
                },
            )
            .unwrap();
            (
                result.actual_evaluations,
                result.optimizer_objective.to_bits(),
                result
                    .optimizer_incumbent
                    .expect("retry produced a decodable incumbent")
                    .selected,
            )
        };
        assert_eq!(run(1), run(2));
    }
}
