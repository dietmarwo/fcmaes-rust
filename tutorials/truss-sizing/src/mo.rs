//! Constrained multi-objective truss design.

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fcmaes_core::{Fitness, Mode, ModeParams, Rng, parallel_batch, pareto_indices};

use crate::INVALID_COST;
use crate::decode::{baseline_controls, dimension};
use crate::evaluate::{Evaluation, evaluate};
use crate::fem::{Scenario, WorkCounter, WorkSnapshot};
use crate::ground::GroundStructure;

/// Number of minimized objectives.
pub const OBJECTIVES: usize = 4;
/// Number of explicit MO constraints.
pub const CONSTRAINTS: usize = 5;
const VALUE_WIDTH: usize = OBJECTIVES + CONSTRAINTS;

/// MODE configuration.
#[derive(Clone, Debug)]
pub struct MoConfig {
    /// Candidate-call budget.
    pub evaluations: usize,
    /// Even population size.
    pub population: usize,
    /// Candidate worker threads.
    pub workers: i32,
    /// Root seed.
    pub seed: u64,
}

/// Convergence observation.
#[derive(Clone, Debug)]
pub struct MoProgress {
    /// Candidate calls.
    pub evaluations: usize,
    /// Wall duration.
    pub elapsed_seconds: f64,
    /// Feasible population members.
    pub feasible_population: usize,
    /// Feasible nondominated members.
    pub pareto_population: usize,
    /// Scalar diagnostic convergence summary.
    pub best_quality: f64,
}

/// One retained feasible nondominated point.
#[derive(Clone, Debug)]
pub struct ParetoPoint {
    /// Replayed physical evaluation.
    pub evaluation: Evaluation,
    /// Four minimized objectives.
    pub objectives: [f64; OBJECTIVES],
    /// Selected documentation representative.
    pub selected: bool,
}

/// Completed MODE campaign.
#[derive(Clone, Debug)]
pub struct MoResult {
    /// Requested candidate calls.
    pub requested_evaluations: usize,
    /// Actual candidate calls.
    pub actual_evaluations: usize,
    /// Wall duration.
    pub elapsed: Duration,
    /// Feasible nondominated points.
    pub pareto: Vec<ParetoPoint>,
    /// Progress trace.
    pub progress: Vec<MoProgress>,
    /// Physical-work accounting including replay.
    pub work: WorkSnapshot,
}

fn objectives(evaluation: &Evaluation) -> [f64; OBJECTIVES] {
    [
        evaluation.mass_kg,
        evaluation
            .metrics
            .as_ref()
            .map_or(1.0, |metrics| metrics.max_displacement_m),
        evaluation
            .redundancy
            .map_or(100.0, |redundancy| redundancy.degradation),
        evaluation.active_count as f64,
    ]
}

fn constraints(evaluation: &Evaluation) -> [f64; CONSTRAINTS] {
    let values = evaluation.constraints.optimizer_values();
    [values[0], values[1], values[2], values[3], values[4]]
}

fn values(evaluation: &Evaluation) -> Vec<f64> {
    let mut result = objectives(evaluation).to_vec();
    result.extend(constraints(evaluation));
    result
}

fn values_or_penalty(
    controls: &[f64],
    ground: &GroundStructure,
    counter: &WorkCounter,
) -> Vec<f64> {
    evaluate(controls, ground, Scenario::TRAINING, true, counter)
        .map_or_else(|| vec![INVALID_COST; VALUE_WIDTH], |result| values(&result))
}

fn seeded_population(ground: &GroundStructure, population: usize, seed: u64) -> Vec<Vec<f64>> {
    let baseline = baseline_controls(ground);
    let member_count = ground.members.len();
    (0..population)
        .map(|index| {
            if index == 0 {
                return baseline.clone();
            }
            let mut rng = Rng::new(seed.wrapping_add(index as u64));
            let mut controls = baseline.clone();
            let section = 8 + index % 4;
            for member in 0..member_count {
                controls[1 + member_count + member] = (section as f64 + 0.5) / 12.0;
            }
            let offsets = 1 + 2 * member_count;
            for value in &mut controls[offsets..] {
                *value = (0.5 + 0.25 * (rng.uniform01() - 0.5)).clamp(0.0, 1.0);
            }
            controls
        })
        .collect()
}

fn feasible_indices(values: &[Vec<f64>]) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter(|(_, row)| row[OBJECTIVES..].iter().all(|value| *value <= 0.0))
        .map(|(index, _)| index)
        .collect()
}

fn summary(values: &[Vec<f64>]) -> (usize, usize, f64) {
    let feasible = feasible_indices(values);
    if feasible.is_empty() {
        return (0, 0, f64::INFINITY);
    }
    let objective_rows = feasible
        .iter()
        .map(|index| values[*index][..OBJECTIVES].to_vec())
        .collect::<Vec<_>>();
    let pareto = pareto_indices(&objective_rows, OBJECTIVES).map_or(0, |front| front.len());
    let best = objective_rows
        .iter()
        .map(|row| row[0] / 5_000.0 + row[1] / 0.05 + row[2] / 100.0 + row[3] / 40.0)
        .fold(f64::INFINITY, f64::min);
    (feasible.len(), pareto, best)
}

/// Run constrained MODE with expensive removal robustness.
pub fn optimize_mode(config: &MoConfig) -> Result<MoResult, Box<dyn Error>> {
    if config.population < 4
        || !config.population.is_multiple_of(2)
        || config.evaluations < config.population
    {
        return Err("MODE requires an even population and at least one population budget".into());
    }
    let ground = Arc::new(GroundStructure::reference());
    let counter = Arc::new(WorkCounter::default());
    let dim = dimension(&ground);
    let fitness = Fitness::bounded(dim, VALUE_WIDTH, &vec![0.0; dim], &vec![1.0; dim]);
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
    let initial_x = seeded_population(&ground, config.population, config.seed);
    let eval_ground = Arc::clone(&ground);
    let eval_counter = Arc::clone(&counter);
    let initial_y = parallel_batch(&initial_x, config.workers, move |controls| {
        values_or_penalty(controls, &eval_ground, &eval_counter)
    });
    mode.set_population(&initial_x, &initial_y);
    let mut actual_evaluations = initial_y.len();
    let mut progress = Vec::new();
    let (feasible, pareto, best_quality) = summary(&initial_y);
    progress.push(MoProgress {
        evaluations: actual_evaluations,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        feasible_population: feasible,
        pareto_population: pareto,
        best_quality,
    });
    let generations = config
        .evaluations
        .saturating_sub(config.population)
        .div_ceil(config.population);
    for generation in 0..generations {
        let xs = mode.ask();
        let eval_ground = Arc::clone(&ground);
        let eval_counter = Arc::clone(&counter);
        let ys = parallel_batch(&xs, config.workers, move |controls| {
            values_or_penalty(controls, &eval_ground, &eval_counter)
        });
        actual_evaluations += ys.len();
        mode.tell(&ys);
        if generation == 0 || (generation + 1).is_multiple_of(4) || generation + 1 == generations {
            let result = mode.result();
            let (feasible, pareto, best_quality) = summary(&result.y);
            progress.push(MoProgress {
                evaluations: actual_evaluations,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                feasible_population: feasible,
                pareto_population: pareto,
                best_quality,
            });
        }
    }
    let result = mode.result();
    let feasible = feasible_indices(&result.y);
    if feasible.is_empty() {
        return Err("MODE retained no stable stress/buckling-feasible truss".into());
    }
    let objective_rows = feasible
        .iter()
        .map(|index| result.y[*index][..OBJECTIVES].to_vec())
        .collect::<Vec<_>>();
    let front = pareto_indices(&objective_rows, OBJECTIVES)?;
    let mut pareto = front
        .into_iter()
        .filter_map(|front_index| {
            let population_index = feasible[front_index];
            evaluate(
                &result.x[population_index],
                &ground,
                Scenario::TRAINING,
                true,
                &counter,
            )
            .map(|evaluation| ParetoPoint {
                objectives: objectives(&evaluation),
                evaluation,
                selected: false,
            })
        })
        .collect::<Vec<_>>();
    for objective in 0..OBJECTIVES {
        if let Some(index) = (0..pareto.len()).min_by(|left, right| {
            pareto[*left].objectives[objective].total_cmp(&pareto[*right].objectives[objective])
        }) {
            pareto[index].selected = true;
        }
    }
    Ok(MoResult {
        requested_evaluations: config.evaluations,
        actual_evaluations,
        elapsed: started.elapsed(),
        pareto,
        progress,
        work: counter.snapshot(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_mode_run_retains_a_replayable_point() {
        let result = optimize_mode(&MoConfig {
            evaluations: 16,
            population: 8,
            workers: 1,
            seed: 42,
        })
        .unwrap();
        assert!(!result.pareto.is_empty());
        assert!(result.pareto.iter().all(|point| {
            point.evaluation.constraints.optimizer_values()[..5]
                .iter()
                .all(|value| *value <= 0.0)
        }));
        assert!(result.work.factorizations > result.work.candidate_evaluations);
    }
}
