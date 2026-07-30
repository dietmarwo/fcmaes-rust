//! Constrained multi-objective beam synthesis.

use std::error::Error;
use std::time::{Duration, Instant};

use fcmaes_core::{Fitness, Mode, ModeParams, parallel_batch, pareto_indices};

use crate::INVALID_COST;
use crate::decode::{Excitation, decode_with_activation};
use crate::metrics::{Sector, analyse_linear};
use crate::scenarios::{RobustMetrics, evaluate_training_with_field};
use crate::so::{BeamContext, ELEMENTS, analytic_seed};

/// MODE decision dimension: count + keys + phase + attenuation.
pub const DIMENSION: usize = 1 + 3 * ELEMENTS;
/// Objective count.
pub const OBJECTIVES: usize = 4;
/// Constraint count.
pub const CONSTRAINTS: usize = 2;
const VALUE_WIDTH: usize = OBJECTIVES + CONSTRAINTS;
const MIN_ACTIVE: usize = 8;
const MAX_ACTIVE: usize = 14;
const NULL_LIMIT_DB: f64 = -6.0;

/// Decoded multi-objective evaluation.
#[derive(Clone, Debug)]
pub struct MoEvaluation {
    /// Normalized MODE controls.
    pub controls: Vec<f64>,
    /// Decoded register state.
    pub excitation: Excitation,
    /// Exact activation cardinality.
    pub active_count: usize,
    /// Robust metrics.
    pub robust: RobustMetrics,
    /// Four minimized objectives.
    pub objectives: [f64; OBJECTIVES],
    /// Two constraints, feasible at `≤0`.
    pub constraints: [f64; CONSTRAINTS],
}

impl MoEvaluation {
    fn values(&self) -> Vec<f64> {
        let mut values = self.objectives.to_vec();
        values.extend(self.constraints);
        values
    }
}

fn decode_active_count(value: f64) -> Option<usize> {
    value.is_finite().then(|| {
        MIN_ACTIVE
            + ((value.clamp(0.0, 1.0) * (MAX_ACTIVE - MIN_ACTIVE + 1) as f64).floor() as usize)
                .min(MAX_ACTIVE - MIN_ACTIVE)
    })
}

fn activation_layout(controls: &[f64]) -> Option<(usize, Vec<f64>)> {
    if controls.len() != DIMENSION || controls.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let active_count = decode_active_count(controls[0])?;
    let mut decoded = Vec::with_capacity(3 * ELEMENTS);
    decoded.extend_from_slice(&controls[1 + ELEMENTS..1 + 2 * ELEMENTS]);
    decoded.extend_from_slice(&controls[1 + 2 * ELEMENTS..]);
    decoded.extend_from_slice(&controls[1..1 + ELEMENTS]);
    Some((active_count, decoded))
}

/// Evaluate one constrained trade-off candidate.
#[must_use]
pub fn evaluate_mo(controls: &[f64], context: &BeamContext) -> Option<MoEvaluation> {
    let (active_count, decoded) = activation_layout(controls)?;
    let excitation =
        decode_with_activation(&decoded, ELEMENTS, active_count, &context.quantization).ok()?;
    let (robust, field) =
        evaluate_training_with_field(&excitation.weights, &context.steering, &context.grid).ok()?;
    let sector_metrics = analyse_linear(
        &field,
        &context.grid,
        &excitation.weights,
        Some(Sector {
            lower_deg: -5.0,
            upper_deg: 5.0,
        }),
    );
    let peak_gain_db = 20.0 * robust.nominal.peak_linear.max(1.0e-15).log10();
    let objectives = [
        -peak_gain_db,
        robust.nominal.psll_db,
        active_count as f64,
        robust.worst_psll_db - robust.nominal.psll_db,
    ];
    let constraints = [
        sector_metrics.null_depth_db.unwrap_or(0.0) - NULL_LIMIT_DB,
        if robust.kernel_failures == 0 {
            -1.0
        } else {
            crate::KERNEL_FAILURE_SCALE
        },
    ];
    objectives
        .iter()
        .chain(&constraints)
        .all(|value| value.is_finite())
        .then_some(MoEvaluation {
            controls: controls.to_vec(),
            excitation,
            active_count,
            robust,
            objectives,
            constraints,
        })
}

fn values_or_penalty(controls: &[f64], context: &BeamContext) -> Vec<f64> {
    evaluate_mo(controls, context)
        .map(|evaluation| evaluation.values())
        .unwrap_or_else(|| vec![INVALID_COST; VALUE_WIDTH])
}

fn seeded_population(population: usize) -> Vec<Vec<f64>> {
    (0..population)
        .map(|index| {
            let requested =
                -45.0 + 90.0 * index as f64 / population.saturating_sub(1).max(1) as f64;
            let taper = 0.55 + 0.45 * ((index * 5) % 11) as f64 / 10.0;
            let active_count = MIN_ACTIVE + index % (MAX_ACTIVE - MIN_ACTIVE + 1);
            let beam = analytic_seed(requested, taper);
            let mut controls = vec![0.0; DIMENSION];
            controls[0] = (active_count - MIN_ACTIVE) as f64 / (MAX_ACTIVE - MIN_ACTIVE) as f64;
            let center = (ELEMENTS - 1) as f64 / 2.0;
            for element in 0..ELEMENTS {
                controls[1 + element] = (element as f64 - center).abs() / center;
                controls[1 + ELEMENTS + element] = beam[element];
                controls[1 + 2 * ELEMENTS + element] = beam[ELEMENTS + element];
            }
            controls
        })
        .collect()
}

/// MODE settings.
#[derive(Clone, Debug)]
pub struct MoConfig {
    /// Candidate budget.
    pub evaluations: usize,
    /// Population size.
    pub population: usize,
    /// Candidate workers.
    pub workers: i32,
    /// Root seed.
    pub seed: u64,
    /// Cut samples.
    pub points: usize,
}

/// One feasible nondominated point.
#[derive(Clone, Debug)]
pub struct ParetoPoint {
    /// Replayed evaluation.
    pub evaluation: MoEvaluation,
    /// Representative selected for documentation.
    pub selected: bool,
}

/// MODE progress.
#[derive(Clone, Debug)]
pub struct MoProgress {
    /// Candidate calls.
    pub evaluations: usize,
    /// Wall time.
    pub elapsed_seconds: f64,
    /// Feasible population members.
    pub feasible_population: usize,
    /// Nondominated feasible members.
    pub pareto_population: usize,
    /// Scalar convergence summary.
    pub best_quality: f64,
}

/// Final MODE result.
#[derive(Clone, Debug)]
pub struct MoResult {
    /// Requested budget.
    pub requested_evaluations: usize,
    /// Actual calls.
    pub actual_evaluations: usize,
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
    let best = feasible
        .iter()
        .map(|index| {
            ys[*index][0] / 30.0
                + ys[*index][1] / 20.0
                + ys[*index][2] / 16.0
                + ys[*index][3] / 10.0
        })
        .fold(f64::INFINITY, f64::min);
    (feasible.len(), pareto, best)
}

/// Run constrained MODE.
pub fn optimize_mode(config: &MoConfig) -> Result<MoResult, Box<dyn Error>> {
    if config.population < 4
        || !config.population.is_multiple_of(2)
        || config.evaluations < config.population
        || config.points < 101
    {
        return Err("invalid MODE population, budget, or grid".into());
    }
    let context = BeamContext::stage_a(config.points);
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
    let started = Instant::now();
    let initial_x = seeded_population(config.population);
    let initial_y = parallel_batch(&initial_x, config.workers, |x| {
        values_or_penalty(x, &context)
    });
    mode.set_population(&initial_x, &initial_y);
    let mut actual_evaluations = config.population;
    let mut progress = Vec::new();
    let (feasible, pareto, best_quality) = population_summary(&initial_y);
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
        let ys = parallel_batch(&xs, config.workers, |x| values_or_penalty(x, &context));
        actual_evaluations += ys.len();
        mode.tell(&ys);
        if generation == 0 || (generation + 1).is_multiple_of(5) || generation + 1 == generations {
            let result = mode.result();
            let (feasible, pareto, best_quality) = population_summary(&result.y);
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
    let feasible = result
        .y
        .iter()
        .enumerate()
        .filter(|(_, values)| values[OBJECTIVES..].iter().all(|value| *value <= 0.0))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if feasible.is_empty() {
        return Err("MODE retained no feasible beam".into());
    }
    let objectives = feasible
        .iter()
        .map(|index| result.y[*index][..OBJECTIVES].to_vec())
        .collect::<Vec<_>>();
    let front = pareto_indices(&objectives, OBJECTIVES)?;
    let mut pareto = front
        .iter()
        .filter_map(|front_index| {
            let population_index = feasible[*front_index];
            evaluate_mo(&result.x[population_index], &context).map(|evaluation| ParetoPoint {
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
        elapsed: started.elapsed(),
        pareto,
        progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_mode_returns_exact_cardinality_feasible_points() {
        let result = optimize_mode(&MoConfig {
            evaluations: 128,
            population: 32,
            workers: 2,
            seed: 42,
            points: 361,
        })
        .unwrap();
        assert!(!result.pareto.is_empty());
        assert!(result.pareto.iter().all(|point| {
            point
                .evaluation
                .excitation
                .active
                .iter()
                .filter(|value| **value)
                .count()
                == point.evaluation.active_count
                && point
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
    }

    #[test]
    fn steering_matrix_type_is_shared_with_scalar_context() {
        let context = BeamContext::stage_a(361);
        let _: &crate::kernel::SteeringMatrix = &context.steering;
    }
}
