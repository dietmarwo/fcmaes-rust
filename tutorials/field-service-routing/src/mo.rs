//! Constrained soft-window MODE formulation.

use std::error::Error;
use std::time::{Duration, Instant};

use fcmaes_core::{Fitness, Mode, ModeParams, Rng, parallel_batch, pareto_indices};

use crate::INVALID_OBJECTIVE;
use crate::decode::{decode, witness_controls};
use crate::evaluate::{EvalConfig, SolutionMetrics, constraints, evaluate};
use crate::instance::{DIMENSION, Instance};

/// Four objectives.
pub const OBJECTIVES: usize = 4;
/// Capacity and shift constraints.
pub const CONSTRAINTS: usize = 2;
const WIDTH: usize = OBJECTIVES + CONSTRAINTS;

/// Replayed soft-window point.
#[derive(Clone, Debug)]
pub struct MoEvaluation {
    /// Controls.
    pub controls: Vec<f64>,
    /// Nominal physical metrics.
    pub metrics: SolutionMetrics,
    /// Distance, vehicles, makespan and lateness.
    pub objectives: [f64; OBJECTIVES],
    /// Capacity and shift violation, feasible at zero.
    pub constraints: [f64; CONSTRAINTS],
}

fn evaluate_mo(controls: &[f64], instance: &Instance) -> Option<MoEvaluation> {
    let decoded = decode(controls, instance).ok()?;
    let metrics = evaluate(&decoded, instance, EvalConfig::default());
    let hard = constraints(&metrics);
    let objectives = [
        metrics.distance_km,
        metrics.used_vehicles as f64,
        metrics.makespan_s,
        metrics.total_lateness_s,
    ];
    let constraints = [hard[0], hard[2]];
    Some(MoEvaluation {
        controls: controls.to_vec(),
        metrics,
        objectives,
        constraints,
    })
}

fn values(controls: &[f64], instance: &Instance) -> Vec<f64> {
    evaluate_mo(controls, instance).map_or_else(
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

/// MODE settings.
#[derive(Clone, Debug)]
pub struct MoConfig {
    /// Candidate budget.
    pub evaluations: usize,
    /// Population.
    pub population: usize,
    /// Candidate workers.
    pub workers: i32,
    /// Root seed.
    pub seed: u64,
}

/// MODE progress.
#[derive(Clone, Debug)]
pub struct MoProgress {
    /// Calls.
    pub evaluations: usize,
    /// Wall seconds.
    pub elapsed_seconds: f64,
    /// Feasible population members.
    pub feasible: usize,
    /// Nondominated feasible members.
    pub pareto: usize,
}

/// One retained point.
#[derive(Clone, Debug)]
pub struct ParetoPoint {
    /// Replayed evaluation.
    pub evaluation: MoEvaluation,
    /// Documentation representative.
    pub selected: bool,
}

/// Final MODE result.
#[derive(Clone, Debug)]
pub struct MoResult {
    /// Requested calls.
    pub requested_evaluations: usize,
    /// Actual calls.
    pub actual_evaluations: usize,
    /// Wall duration.
    pub elapsed: Duration,
    /// Feasible nondominated front.
    pub pareto: Vec<ParetoPoint>,
    /// Progress.
    pub progress: Vec<MoProgress>,
}

fn seeded_population(instance: &Instance, population: usize, seed: u64) -> Vec<Vec<f64>> {
    let witness = witness_controls(instance);
    let mut rng = Rng::new(seed);
    (0..population)
        .map(|index| {
            if index == 0 {
                witness.clone()
            } else {
                let scale = 0.04 + 0.5 * (index % 9) as f64 / 8.0;
                witness
                    .iter()
                    .map(|value| (value + scale * (rng.uniform01() - 0.5)).clamp(0.0, 1.0))
                    .collect()
            }
        })
        .collect()
}

fn summary(values: &[Vec<f64>]) -> (usize, usize) {
    let feasible = values
        .iter()
        .enumerate()
        .filter(|(_, row)| row[OBJECTIVES..].iter().all(|value| *value <= 1.0e-9))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let objectives = feasible
        .iter()
        .map(|index| values[*index][..OBJECTIVES].to_vec())
        .collect::<Vec<_>>();
    let pareto = pareto_indices(&objectives, OBJECTIVES).map_or(0, |front| front.len());
    (feasible.len(), pareto)
}

/// Run constrained MODE. Assignment keys remain continuous.
pub fn optimize(instance: &Instance, config: &MoConfig) -> Result<MoResult, Box<dyn Error>> {
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
    let initial_x = seeded_population(instance, config.population, config.seed);
    let initial_y = parallel_batch(&initial_x, config.workers, |x| values(x, instance));
    mode.set_population(&initial_x, &initial_y);
    let mut actual = initial_y.len();
    let (feasible, pareto) = summary(&initial_y);
    let mut progress = vec![MoProgress {
        evaluations: actual,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        feasible,
        pareto,
    }];
    let generations = config
        .evaluations
        .saturating_sub(config.population)
        .div_ceil(config.population);
    for generation in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, config.workers, |x| values(x, instance));
        actual += ys.len();
        mode.tell(&ys);
        if generation == 0 || (generation + 1).is_multiple_of(5) || generation + 1 == generations {
            let current = mode.result();
            let (feasible, pareto) = summary(&current.y);
            progress.push(MoProgress {
                evaluations: actual,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                feasible,
                pareto,
            });
        }
    }
    let result = mode.result();
    let feasible = result
        .y
        .iter()
        .enumerate()
        .filter(|(_, row)| row[OBJECTIVES..].iter().all(|value| *value <= 1.0e-9))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let objectives = feasible
        .iter()
        .map(|index| result.y[*index][..OBJECTIVES].to_vec())
        .collect::<Vec<_>>();
    let front = pareto_indices(&objectives, OBJECTIVES)?;
    let mut pareto = front
        .iter()
        .filter_map(|front_index| {
            let population_index = feasible[*front_index];
            evaluate_mo(&result.x[population_index], instance).map(|evaluation| ParetoPoint {
                evaluation,
                selected: false,
            })
        })
        .collect::<Vec<_>>();
    for objective in 0..OBJECTIVES {
        if let Some(index) = (0..pareto.len()).min_by(|a, b| {
            pareto[*a].evaluation.objectives[objective]
                .total_cmp(&pareto[*b].evaluation.objectives[objective])
        }) {
            pareto[index].selected = true;
        }
    }
    Ok(MoResult {
        requested_evaluations: config.evaluations,
        actual_evaluations: actual,
        elapsed: started.elapsed(),
        pareto,
        progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::load_primary;

    #[test]
    fn small_mode_keeps_feasible_nondominated_points() {
        let instance = load_primary().unwrap();
        let result = optimize(
            &instance,
            &MoConfig {
                evaluations: 128,
                population: 32,
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
                .all(|constraint| *constraint <= 1.0e-9)
                && point.evaluation.metrics.used_vehicles as f64 == point.evaluation.objectives[1]
        }));
        let objectives = result
            .pareto
            .iter()
            .map(|point| point.evaluation.objectives.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(
            pareto_indices(&objectives, OBJECTIVES).unwrap().len(),
            objectives.len()
        );
    }
}
