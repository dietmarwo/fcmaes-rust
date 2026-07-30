//! Constrained multi-objective scheduling.

use std::error::Error;
use std::time::{Duration, Instant};

use epanet_rs::model::network::Network;
use fcmaes_core::{Fitness, Mode, ModeParams, Rng, parallel_batch, pareto_indices};

use crate::decode::seed_controls;
use crate::evaluate::{RobustEvaluation, evaluate_training};
use crate::{DIMENSION, INVALID_OBJECTIVE};

pub const OBJECTIVES: usize = 4;
pub const CONSTRAINTS: usize = 4;
const WIDTH: usize = OBJECTIVES + CONSTRAINTS;

/// Replayed multi-objective candidate.
#[derive(Clone, Debug)]
pub struct MoEvaluation {
    pub robust: RobustEvaluation,
    pub objectives: [f64; OBJECTIVES],
    pub constraints: [f64; CONSTRAINTS],
}

fn evaluate_mo(controls: &[f64], network: &Network) -> Option<MoEvaluation> {
    let robust = evaluate_training(controls, network).ok()?;
    let objectives = [
        robust
            .scenarios
            .iter()
            .map(|item| item.energy_cost)
            .fold(0.0, f64::max),
        robust
            .scenarios
            .iter()
            .map(|item| (20.0 - item.min_pressure_m).max(0.0))
            .fold(0.0, f64::max),
        robust
            .scenarios
            .iter()
            .map(|item| item.switching_cost)
            .fold(0.0, f64::max),
        robust
            .scenarios
            .iter()
            .map(|item| (item.max_pressure_m - 35.0).max(0.0))
            .sum::<f64>()
            / robust.scenarios.len() as f64,
    ];
    let constraints = [
        robust
            .scenarios
            .iter()
            .map(|item| item.constraints[2])
            .fold(f64::NEG_INFINITY, f64::max),
        robust
            .scenarios
            .iter()
            .map(|item| item.constraints[3])
            .fold(f64::NEG_INFINITY, f64::max),
        robust
            .scenarios
            .iter()
            .map(|item| item.constraints[4])
            .fold(f64::NEG_INFINITY, f64::max),
        robust
            .scenarios
            .iter()
            .map(|item| item.constraints[6])
            .fold(f64::NEG_INFINITY, f64::max),
    ];
    Some(MoEvaluation {
        robust,
        objectives,
        constraints,
    })
}

fn values(controls: &[f64], network: &Network) -> Vec<f64> {
    evaluate_mo(controls, network).map_or_else(
        || vec![INVALID_OBJECTIVE; WIDTH],
        |evaluation| {
            evaluation
                .objectives
                .into_iter()
                .chain(evaluation.constraints)
                .collect()
        },
    )
}

/// MODE protocol.
#[derive(Clone, Debug)]
pub struct MoConfig {
    pub evaluations: usize,
    pub population: usize,
    pub workers: i32,
    pub seed: u64,
}

/// Retained nondominated point.
#[derive(Clone, Debug)]
pub struct ParetoPoint {
    pub evaluation: MoEvaluation,
    pub selected: bool,
}

/// MODE campaign.
#[derive(Clone, Debug)]
pub struct MoResult {
    pub requested_evaluations: usize,
    pub actual_evaluations: usize,
    pub elapsed: Duration,
    pub pareto: Vec<ParetoPoint>,
}

fn population(count: usize, seed: u64) -> Vec<Vec<f64>> {
    let witness = seed_controls();
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|index| {
            if index == 0 {
                witness.clone()
            } else {
                let scale = 0.1 + 0.7 * (index % 11) as f64 / 10.0;
                witness
                    .iter()
                    .map(|value| (value + scale * (rng.uniform01() - 0.5)).clamp(0.0, 1.0))
                    .collect()
            }
        })
        .collect()
}

/// Run constrained MODE with continuous random-key coordinates.
pub fn optimize(network: &Network, config: &MoConfig) -> Result<MoResult, Box<dyn Error>> {
    if config.population < 4
        || !config.population.is_multiple_of(2)
        || config.evaluations < config.population
    {
        return Err("invalid MODE budget or population".into());
    }
    let fitness = Fitness::bounded(DIMENSION, WIDTH, &[0.0; DIMENSION], &[1.0; DIMENSION]);
    let mut mode = Mode::try_new(
        fitness,
        OBJECTIVES,
        CONSTRAINTS,
        None,
        &ModeParams {
            popsize: config.population as i32,
            seed: config.seed,
            nsga_update: true,
            ..Default::default()
        },
    )?;
    let started = Instant::now();
    let initial_x = population(config.population, config.seed);
    let initial_y = parallel_batch(&initial_x, config.workers, |x| values(x, network));
    mode.set_population(&initial_x, &initial_y);
    let mut actual = initial_y.len();
    let generations = config
        .evaluations
        .saturating_sub(config.population)
        .div_ceil(config.population);
    for _ in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, config.workers, |x| values(x, network));
        actual += ys.len();
        mode.tell(&ys);
    }
    let result = mode.result();
    let feasible = result
        .y
        .iter()
        .enumerate()
        .filter(|(_, row)| row[OBJECTIVES..].iter().all(|value| *value <= 1e-9))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let objective_rows = feasible
        .iter()
        .map(|index| result.y[*index][..OBJECTIVES].to_vec())
        .collect::<Vec<_>>();
    let front = pareto_indices(&objective_rows, OBJECTIVES)?;
    let mut pareto = front
        .into_iter()
        .filter_map(|index| evaluate_mo(&result.x[feasible[index]], network))
        .map(|evaluation| ParetoPoint {
            evaluation,
            selected: false,
        })
        .collect::<Vec<_>>();
    for objective in 0..OBJECTIVES {
        if let Some(index) = (0..pareto.len()).min_by(|left, right| {
            pareto[*left].evaluation.objectives[objective]
                .total_cmp(&pareto[*right].evaluation.objectives[objective])
        }) {
            pareto[index].selected = true;
        }
    }
    Ok(MoResult {
        requested_evaluations: config.evaluations,
        actual_evaluations: actual,
        elapsed: started.elapsed(),
        pareto,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network;

    #[test]
    fn small_mode_front_is_feasible_and_nondominated() {
        let result = optimize(
            &network::load().unwrap(),
            &MoConfig {
                evaluations: 80,
                population: 40,
                workers: 2,
                seed: 42,
            },
        )
        .unwrap();
        assert!(!result.pareto.is_empty());
        assert!(result.pareto.iter().all(|point| {
            point
                .evaluation
                .constraints
                .iter()
                .all(|value| *value <= 1e-9)
        }));
        for (i, left) in result.pareto.iter().enumerate() {
            for (j, right) in result.pareto.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dominates = left
                    .evaluation
                    .objectives
                    .iter()
                    .zip(right.evaluation.objectives)
                    .all(|(a, b)| *a <= b)
                    && left
                        .evaluation
                        .objectives
                        .iter()
                        .zip(right.evaluation.objectives)
                        .any(|(a, b)| *a < b);
                assert!(!dominates);
            }
        }
    }
}
