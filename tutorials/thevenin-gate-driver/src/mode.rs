//! Constrained MODE search over gate resistance and snubber resistance.

use std::error::Error;
use std::time::{Duration, Instant};

use fcmaes_core::{Fitness, Mode, ModeParams, parallel_batch, pareto_indices};

use crate::{CONSTRAINTS, DIMENSION, GateEvaluation, OBJECTIVES, VALUE_WIDTH, evaluate};

const INVALID_COST: f64 = 1.0e12;

#[derive(Clone, Debug)]
/// Configuration of one deterministic constrained-MODE run.
pub struct ModeConfig {
    /// Requested objective evaluations; rounded up to a whole population.
    pub evaluations: usize,
    /// Even MODE population size.
    pub popsize: usize,
    /// Number of parallel simulation workers.
    pub workers: i32,
    /// Reproducible random seed.
    pub seed: u64,
}

#[derive(Clone, Debug)]
/// Periodic progress sample from the retained MODE population.
pub struct ModeProgress {
    /// Number of completed objective evaluations.
    pub evaluations: usize,
    /// Wall-clock seconds since the search started.
    pub elapsed_seconds: f64,
    /// Number of feasible members in the current population.
    pub feasible_population: usize,
    /// Number of feasible nondominated population members.
    pub pareto_population: usize,
    /// Small scalar diagnostic used only to track search progress.
    pub best_quality: f64,
}

#[derive(Clone, Debug)]
/// One replayed feasible point on the final Pareto front.
pub struct ParetoPoint {
    /// Physical simulation and measured objectives for the point.
    pub evaluation: GateEvaluation,
    /// Whether this point was selected as a plotted representative.
    pub selected: bool,
}

#[derive(Clone, Debug)]
/// Replayed Pareto front and run diagnostics returned by [`optimize_mode`].
pub struct ModeResult {
    /// Evaluation budget requested by the caller.
    pub requested_evaluations: usize,
    /// Actual evaluations after rounding to whole generations.
    pub actual_evaluations: usize,
    /// Number of MODE generations.
    pub generations: usize,
    /// Total search and replay wall time.
    pub elapsed: Duration,
    /// Final feasible nondominated points.
    pub pareto: Vec<ParetoPoint>,
    /// Periodic convergence samples.
    pub progress: Vec<ModeProgress>,
}

fn values_or_penalty(u: &[f64]) -> Vec<f64> {
    evaluate(u)
        .map(|evaluation| evaluation.values())
        .unwrap_or_else(|| vec![INVALID_COST; VALUE_WIDTH])
}

fn population_summary(ys: &[Vec<f64>]) -> Result<(usize, usize, f64), Box<dyn Error>> {
    let feasible: Vec<usize> = ys
        .iter()
        .enumerate()
        .filter(|(_, values)| {
            values[OBJECTIVES..]
                .iter()
                .all(|value| value.is_finite() && *value <= 0.0)
        })
        .map(|(index, _)| index)
        .collect();
    if feasible.is_empty() {
        return Ok((0, 0, f64::INFINITY));
    }
    let objectives: Vec<Vec<f64>> = feasible
        .iter()
        .map(|index| ys[*index][..OBJECTIVES].to_vec())
        .collect();
    let pareto = pareto_indices(&objectives, OBJECTIVES)?.len();
    let best_quality = feasible
        .iter()
        .map(|index| ys[*index][0] / 20.0 + ys[*index][1] / 25.0)
        .fold(f64::INFINITY, f64::min);
    Ok((feasible.len(), pareto, best_quality))
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
    let minima = std::array::from_fn::<_, OBJECTIVES, _>(|objective| {
        pareto
            .iter()
            .map(|point| point.evaluation.objectives[objective])
            .fold(f64::INFINITY, f64::min)
    });
    let maxima = std::array::from_fn::<_, OBJECTIVES, _>(|objective| {
        pareto
            .iter()
            .map(|point| point.evaluation.objectives[objective])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let compromise = (0..pareto.len())
        .min_by(|&left, &right| {
            let score = |index: usize| {
                (0..OBJECTIVES)
                    .map(|objective| {
                        let width = (maxima[objective] - minima[objective]).max(1.0e-12);
                        (pareto[index].evaluation.objectives[objective] - minima[objective]) / width
                    })
                    .sum::<f64>()
            };
            score(left).total_cmp(&score(right))
        })
        .expect("non-empty Pareto set");
    pareto[compromise].selected = true;
}

/// Run constrained MODE and replay every feasible nondominated point.
pub fn optimize_mode(config: &ModeConfig) -> Result<ModeResult, Box<dyn Error>> {
    if config.evaluations == 0 {
        return Err("MODE evaluations must be positive".into());
    }
    if config.popsize < 4 || !config.popsize.is_multiple_of(2) {
        return Err("MODE popsize must be an even integer of at least four".into());
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
        let ys = parallel_batch(&xs, config.workers, |x| values_or_penalty(x));
        actual_evaluations += ys.len();
        mode.tell(&ys);
        if generation == 0 || (generation + 1) % 5 == 0 || generation + 1 == generations {
            let result = mode.result();
            let (feasible, pareto, best_quality) = population_summary(&result.y)?;
            progress.push(ModeProgress {
                evaluations: actual_evaluations,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                feasible_population: feasible,
                pareto_population: pareto,
                best_quality,
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
        return Err("MODE retained no feasible gate-driver design".into());
    }
    let objective_rows: Vec<Vec<f64>> = feasible_indices
        .iter()
        .map(|index| result.y[*index][..OBJECTIVES].to_vec())
        .collect();
    let front = pareto_indices(&objective_rows, OBJECTIVES)?;
    let mut pareto = front
        .iter()
        .filter_map(|front_index| {
            evaluate(&result.x[feasible_indices[*front_index]]).map(|evaluation| ParetoPoint {
                evaluation,
                selected: false,
            })
        })
        .collect::<Vec<_>>();
    mark_representatives(&mut pareto);
    Ok(ModeResult {
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
    fn smoke_mode_returns_a_feasible_front() {
        let result = optimize_mode(&ModeConfig {
            evaluations: 128,
            popsize: 32,
            workers: 2,
            seed: 11,
        })
        .unwrap();
        assert!(!result.pareto.is_empty());
        assert!(
            result
                .pareto
                .iter()
                .all(|point| point.evaluation.is_feasible())
        );
    }
}
