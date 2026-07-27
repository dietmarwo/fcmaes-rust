//! Constrained multi-objective optimization of a fourth-order low-pass filter.

use std::error::Error;
use std::time::{Duration, Instant};

use fcmaes_core::{Fitness, Mode, ModeParams, parallel_batch, pareto_indices};

use crate::INVALID_COST;
use crate::decode::decode_lowpass;
use crate::features::{LowpassFeatures, gain_curve, lowpass_features};
use crate::netlist::sallen_key_lowpass4;

pub const DIMENSION: usize = 8;
pub const OBJECTIVES: usize = 3;
pub const CONSTRAINTS: usize = 1;
pub const VALUE_WIDTH: usize = OBJECTIVES + CONSTRAINTS;
const TARGET_CUTOFF_HZ: f64 = 100_000.0;

/// One decoded low-pass evaluation.
#[derive(Clone, Debug)]
pub struct LowpassEvaluation {
    pub controls: Vec<f64>,
    pub components: [f64; 8],
    pub features: LowpassFeatures,
    pub objectives: [f64; OBJECTIVES],
    pub constraint: f64,
}

impl LowpassEvaluation {
    pub fn values(&self) -> Vec<f64> {
        let mut values = self.objectives.to_vec();
        values.push(self.constraint);
        values
    }
}

/// MODE search settings.
#[derive(Clone, Debug)]
pub struct MoConfig {
    pub evaluations: usize,
    pub popsize: usize,
    pub workers: i32,
    pub seed: u64,
    pub points: usize,
}

/// Recorded MODE progress.
#[derive(Clone, Debug)]
pub struct MoProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub feasible_population: usize,
    pub pareto_population: usize,
    pub best_quality: f64,
}

/// One feasible nondominated design.
#[derive(Clone, Debug)]
pub struct ParetoPoint {
    pub evaluation: LowpassEvaluation,
    pub selected: bool,
}

/// Final multi-objective result.
#[derive(Clone, Debug)]
pub struct MoResult {
    pub requested_evaluations: usize,
    pub actual_evaluations: usize,
    pub generations: usize,
    pub elapsed: Duration,
    pub pareto: Vec<ParetoPoint>,
    pub progress: Vec<MoProgress>,
}

/// Evaluate one normalized fourth-order filter design.
pub fn evaluate_lowpass(u: &[f64], points: usize) -> Option<LowpassEvaluation> {
    if u.len() != DIMENSION {
        return None;
    }
    let components = decode_lowpass(u);
    let curve = gain_curve(
        &sallen_key_lowpass4(&components),
        "out",
        TARGET_CUTOFF_HZ / 100.0,
        TARGET_CUTOFF_HZ * 100.0,
        points,
    )?;
    let features = lowpass_features(&curve, 0.8 * TARGET_CUTOFF_HZ)?;
    let objectives = [
        (features.cutoff_hz / TARGET_CUTOFF_HZ).log10().abs(),
        features.passband_ripple_db,
        components[4..].iter().sum::<f64>() * 1e9,
    ];
    let constraint = features.peak_above_dc_db - 3.0;
    objectives
        .iter()
        .chain(std::iter::once(&constraint))
        .all(|value| value.is_finite())
        .then_some(LowpassEvaluation {
            controls: u.to_vec(),
            components,
            features,
            objectives,
            constraint,
        })
}

fn values_or_penalty(u: &[f64], points: usize) -> Vec<f64> {
    evaluate_lowpass(u, points)
        .map(|evaluation| evaluation.values())
        .unwrap_or_else(|| vec![INVALID_COST; VALUE_WIDTH])
}

fn population_summary(
    xs: &[Vec<f64>],
    ys: &[Vec<f64>],
) -> Result<(usize, usize, f64), Box<dyn Error>> {
    let feasible: Vec<usize> = ys
        .iter()
        .enumerate()
        .filter(|(_, values)| values[OBJECTIVES..].iter().all(|value| *value <= 0.0))
        .map(|(index, _)| index)
        .collect();
    if feasible.is_empty() {
        return Ok((0, 0, f64::INFINITY));
    }
    let objectives: Vec<Vec<f64>> = feasible
        .iter()
        .map(|index| ys[*index][..OBJECTIVES].to_vec())
        .collect();
    let front = pareto_indices(&objectives, OBJECTIVES)?;
    let best_quality = feasible
        .iter()
        .map(|index| {
            let values = &ys[*index];
            values[0] + values[1] / 3.0 + values[2] / 100.0
        })
        .fold(f64::INFINITY, f64::min);
    let _ = xs;
    Ok((feasible.len(), front.len(), best_quality))
}

fn mark_representatives(pareto: &mut [ParetoPoint]) {
    if pareto.is_empty() {
        return;
    }
    for objective in 0..OBJECTIVES {
        let index = (0..pareto.len())
            .min_by(|&left, &right| {
                pareto[left].evaluation.objectives[objective]
                    .total_cmp(&pareto[right].evaluation.objectives[objective])
            })
            .expect("non-empty Pareto set");
        pareto[index].selected = true;
    }
    let compromise = (0..pareto.len())
        .min_by(|&left, &right| {
            let score = |point: &ParetoPoint| {
                point.evaluation.objectives[0]
                    + point.evaluation.objectives[1] / 3.0
                    + point.evaluation.objectives[2] / 100.0
            };
            score(&pareto[left]).total_cmp(&score(&pareto[right]))
        })
        .expect("non-empty Pareto set");
    pareto[compromise].selected = true;
}

/// Run constrained MODE and retain its feasible nondominated population.
pub fn optimize_mode(config: &MoConfig) -> Result<MoResult, Box<dyn Error>> {
    if config.evaluations == 0 {
        return Err("MODE evaluations must be positive".into());
    }
    if config.popsize < 4 || !config.popsize.is_multiple_of(2) {
        return Err("MODE popsize must be an even integer of at least four".into());
    }
    if config.points < 9 {
        return Err("MODE AC sweep requires at least nine points".into());
    }
    let fitness = Fitness::bounded(DIMENSION, VALUE_WIDTH, &[0.0; DIMENSION], &[1.0; DIMENSION]);
    let mut mode = Mode::try_new(
        fitness,
        OBJECTIVES,
        CONSTRAINTS,
        None,
        &ModeParams {
            popsize: config.popsize as i32,
            seed: config.seed,
            nsga_update: true,
            ..Default::default()
        },
    )?;
    let generations = config.evaluations.div_ceil(config.popsize);
    let started = Instant::now();
    let mut progress = Vec::new();
    let mut actual_evaluations = 0;
    for generation in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, config.workers, |x| values_or_penalty(x, config.points));
        actual_evaluations += ys.len();
        mode.tell(&ys);
        if generation == 0 || (generation + 1) % 5 == 0 || generation + 1 == generations {
            let result = mode.result();
            let (feasible, pareto, quality) = population_summary(&result.x, &result.y)?;
            progress.push(MoProgress {
                evaluations: actual_evaluations,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                feasible_population: feasible,
                pareto_population: pareto,
                best_quality: quality,
            });
        }
    }
    let result = mode.result();
    let feasible_indices: Vec<usize> = result
        .y
        .iter()
        .enumerate()
        .filter(|(_, values)| values[OBJECTIVES..].iter().all(|value| *value <= 0.0))
        .map(|(index, _)| index)
        .collect();
    if feasible_indices.is_empty() {
        return Err("MODE retained no feasible low-pass design".into());
    }
    let objective_rows: Vec<Vec<f64>> = feasible_indices
        .iter()
        .map(|index| result.y[*index][..OBJECTIVES].to_vec())
        .collect();
    let front = pareto_indices(&objective_rows, OBJECTIVES)?;
    let mut pareto = front
        .iter()
        .filter_map(|front_index| {
            let population_index = feasible_indices[*front_index];
            evaluate_lowpass(&result.x[population_index], config.points).map(|evaluation| {
                ParetoPoint {
                    evaluation,
                    selected: false,
                }
            })
        })
        .collect::<Vec<_>>();
    mark_representatives(&mut pareto);
    Ok(MoResult {
        requested_evaluations: config.evaluations,
        actual_evaluations,
        generations,
        elapsed: started.elapsed(),
        pareto,
        progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_design_has_finite_lowpass_features() {
        let evaluation = evaluate_lowpass(&[0.5; DIMENSION], 41).unwrap();
        assert!(evaluation.features.cutoff_hz.is_finite());
        assert!(evaluation.objectives.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn smoke_mode_returns_a_feasible_front() {
        let result = optimize_mode(&MoConfig {
            evaluations: 256,
            popsize: 32,
            workers: 2,
            seed: 11,
            points: 21,
        })
        .unwrap();
        assert!(!result.pareto.is_empty());
        assert!(
            result
                .pareto
                .iter()
                .all(|point| point.evaluation.constraint <= 0.0)
        );
    }
}
