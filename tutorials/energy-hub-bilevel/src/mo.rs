//! Constrained multi-objective energy-hub sizing.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{Fitness, Mode, ModeParams, parallel_batch, pareto_indices};

use crate::config::Preset;
use crate::decode::DIMENSION;
use crate::evaluate::{INVALID_OBJECTIVE, OuterEvaluation, evaluate_training};
use crate::pilot::structured_candidate;

/// Objective count.
pub const OBJECTIVES: usize = 4;
/// Constraint count.
pub const CONSTRAINTS: usize = 3;
const VALUE_WIDTH: usize = OBJECTIVES + CONSTRAINTS;

/// One decoded MODE candidate.
#[derive(Clone, Debug)]
pub struct MoEvaluation {
    /// Replayed robust evaluation.
    pub outer: OuterEvaluation,
    /// Annualized CAPEX, unserved energy, grid CO₂, and curtailed energy.
    pub objectives: [f64; OBJECTIVES],
    /// Self-sufficiency, cycle-budget, and LP-status residuals.
    pub constraints: [f64; CONSTRAINTS],
}

impl MoEvaluation {
    fn values(&self) -> Vec<f64> {
        let mut values = self.objectives.to_vec();
        values.extend(self.constraints);
        values
    }
}

/// Evaluate one constrained trade-off candidate.
#[must_use]
pub fn evaluate_mo(controls: &[f64], preset: Preset) -> Option<MoEvaluation> {
    let outer = evaluate_training(controls, preset)?;
    let annual_unserved_kwh = outer
        .scenarios
        .iter()
        .map(|scenario| scenario.annualization * scenario.dispatch.unserved_kwh)
        .fold(0.0, f64::max);
    let objectives = [
        outer.capex.annualized,
        annual_unserved_kwh,
        outer.mean_co2_kg,
        outer.mean_curtailed_kwh,
    ];
    let constraints = [
        outer.constraint_self_sufficiency,
        outer.constraint_cycles,
        outer.constraint_lp_status,
    ];
    objectives
        .iter()
        .chain(&constraints)
        .all(|value| value.is_finite())
        .then_some(MoEvaluation {
            outer,
            objectives,
            constraints,
        })
}

fn values_or_penalty(controls: &[f64], preset: Preset, pivots: &AtomicU64) -> Vec<f64> {
    evaluate_mo(controls, preset)
        .map(|evaluation| {
            pivots.fetch_add(evaluation.outer.simplex_iterations, Ordering::Relaxed);
            evaluation.values()
        })
        .unwrap_or_else(|| vec![INVALID_OBJECTIVE; VALUE_WIDTH])
}

/// MODE settings.
#[derive(Clone, Copy, Debug)]
pub struct MoConfig {
    /// Horizon preset.
    pub preset: Preset,
    /// Candidate budget.
    pub evaluations: usize,
    /// Population size.
    pub population: usize,
    /// Candidate workers.
    pub workers: i32,
    /// Root seed.
    pub seed: u64,
}

/// One feasible nondominated point.
#[derive(Clone, Debug)]
pub struct ParetoPoint {
    /// Replayed evaluation.
    pub evaluation: MoEvaluation,
    /// Representative extreme selected for documentation.
    pub selected: bool,
}

/// MODE progress observation.
#[derive(Clone, Copy, Debug)]
pub struct MoProgress {
    /// Candidate calls.
    pub evaluations: usize,
    /// Wall seconds.
    pub elapsed_seconds: f64,
    /// Feasible population members.
    pub feasible_population: usize,
    /// Nondominated feasible members.
    pub pareto_population: usize,
    /// Best normalized compromise score.
    pub best_compromise: f64,
}

/// Completed MODE result.
#[derive(Clone, Debug)]
pub struct MoResult {
    /// Requested budget.
    pub requested_evaluations: usize,
    /// Actual candidate calls.
    pub actual_evaluations: usize,
    /// Inner LP solves.
    pub lp_solves: usize,
    /// Cumulative simplex pivots.
    pub simplex_iterations: u64,
    /// Wall duration.
    pub elapsed: Duration,
    /// Feasible nondominated front.
    pub pareto: Vec<ParetoPoint>,
    /// Progress trace.
    pub progress: Vec<MoProgress>,
}

fn population_summary(ys: &[Vec<f64>]) -> (usize, usize, f64) {
    let feasible = ys
        .iter()
        .enumerate()
        .filter(|(_, values)| values[OBJECTIVES..].iter().all(|value| *value <= 0.0))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if feasible.is_empty() {
        return (0, 0, f64::INFINITY);
    }
    let objectives = feasible
        .iter()
        .map(|index| ys[*index][..OBJECTIVES].to_vec())
        .collect::<Vec<_>>();
    let pareto = pareto_indices(&objectives, OBJECTIVES).map_or(0, |front| front.len());
    let scale = [1.0e6, 1.0e5, 1.0e6, 1.0e6];
    let best = feasible
        .iter()
        .map(|index| {
            (0..OBJECTIVES)
                .map(|objective| ys[*index][objective] / scale[objective])
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    (feasible.len(), pareto, best)
}

/// Run constrained MODE with continuous normalized optimizer coordinates.
pub fn optimize_mode(config: &MoConfig) -> Result<MoResult, Box<dyn Error>> {
    if config.population < 4
        || !config.population.is_multiple_of(2)
        || config.evaluations < config.population
    {
        return Err("invalid MODE population or budget".into());
    }
    let fitness = Fitness::bounded(DIMENSION, VALUE_WIDTH, &[0.0; DIMENSION], &[1.0; DIMENSION]);
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
    let pivots = Arc::new(AtomicU64::new(0));
    let started = Instant::now();
    let initial_x = (0..config.population)
        .map(|sample| structured_candidate(config.seed, sample))
        .collect::<Vec<_>>();
    let initial_pivots = Arc::clone(&pivots);
    let initial_y = parallel_batch(&initial_x, config.workers, move |x| {
        values_or_penalty(x, config.preset, &initial_pivots)
    });
    mode.set_population(&initial_x, &initial_y);
    let mut actual_evaluations = initial_y.len();
    let mut progress = Vec::new();
    let (feasible, pareto, best_compromise) = population_summary(&initial_y);
    progress.push(MoProgress {
        evaluations: actual_evaluations,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        feasible_population: feasible,
        pareto_population: pareto,
        best_compromise,
    });
    let generations = config
        .evaluations
        .saturating_sub(config.population)
        .div_ceil(config.population);
    for generation in 0..generations {
        let xs = mode.ask();
        let generation_pivots = Arc::clone(&pivots);
        let ys = parallel_batch(&xs, config.workers, move |x| {
            values_or_penalty(x, config.preset, &generation_pivots)
        });
        actual_evaluations += ys.len();
        mode.tell(&ys);
        if generation == 0 || (generation + 1).is_multiple_of(5) || generation + 1 == generations {
            let result = mode.result();
            let (feasible, pareto, best_compromise) = population_summary(&result.y);
            progress.push(MoProgress {
                evaluations: actual_evaluations,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                feasible_population: feasible,
                pareto_population: pareto,
                best_compromise,
            });
        }
    }
    let result = mode.result();
    let feasible_indices = result
        .y
        .iter()
        .enumerate()
        .filter(|(_, values)| values[OBJECTIVES..].iter().all(|value| *value <= 0.0))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if feasible_indices.is_empty() {
        return Err("MODE retained no feasible design".into());
    }
    let objectives = feasible_indices
        .iter()
        .map(|index| result.y[*index][..OBJECTIVES].to_vec())
        .collect::<Vec<_>>();
    let front = pareto_indices(&objectives, OBJECTIVES)?;
    let mut pareto = front
        .iter()
        .filter_map(|front_index| {
            let population_index = feasible_indices[*front_index];
            evaluate_mo(&result.x[population_index], config.preset).map(|evaluation| ParetoPoint {
                evaluation,
                selected: false,
            })
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
        actual_evaluations,
        lp_solves: 5 * actual_evaluations,
        simplex_iterations: pivots.load(Ordering::Relaxed),
        elapsed: started.elapsed(),
        pareto,
        progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_mode_returns_a_feasible_nondominated_set() {
        let result = optimize_mode(&MoConfig {
            preset: Preset::Smoke,
            evaluations: 64,
            population: 16,
            workers: 2,
            seed: 42,
        })
        .unwrap();
        assert!(!result.pareto.is_empty());
        assert!(result.pareto.iter().all(|point| {
            point
                .evaluation
                .constraints
                .iter()
                .all(|constraint| *constraint <= 0.0)
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
        assert!(result.simplex_iterations > 0);
    }
}
