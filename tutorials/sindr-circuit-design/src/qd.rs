//! E12 catalogue search with deterministic tolerance robustness and MAP-Elites.

use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    Archive, MapElitesParams, QdBatchFitness, Rng, map_elites_batch_with_progress, parallel_batch,
};

use crate::decode::{decode_bandpass_e12, e12_values};
use crate::features::{BandpassFeatures, bandpass_features, gain_curve};
use crate::netlist::mfb_bandpass;
use crate::{BANDPASS_DIMENSION, INVALID_COST};

/// Frozen after the range study; descriptors are `[log10(f0/Hz), peak gain dB]`.
pub const DESCRIPTOR_LOWER: [f64; 2] = [2.0, -60.0];
pub const DESCRIPTOR_UPPER: [f64; 2] = [6.5, 40.0];

/// Precomputed common tolerance perturbations shared by all candidates.
#[derive(Clone, Debug)]
pub struct ToleranceDraws {
    pub multipliers: Vec<[f64; BANDPASS_DIMENSION]>,
}

impl ToleranceDraws {
    pub fn new(seed: u64, draws: usize) -> Self {
        let mut rng = Rng::new(seed);
        let multipliers = (0..draws)
            .map(|_| {
                let mut values = [1.0; BANDPASS_DIMENSION];
                for value in &mut values {
                    *value = 0.95 + 0.10 * rng.uniform01();
                }
                values
            })
            .collect();
        Self { multipliers }
    }
}

/// Nominal catalogue design plus tolerance quality.
#[derive(Clone, Debug)]
pub struct QdEvaluation {
    pub coordinates: Vec<f64>,
    pub indices: [usize; BANDPASS_DIMENSION],
    pub components: [f64; BANDPASS_DIMENSION],
    pub features: BandpassFeatures,
    pub quality: f64,
    pub descriptors: [f64; 2],
}

/// One candidate from the descriptor range study.
#[derive(Clone, Debug)]
pub struct RangeStudyRow {
    pub sample: usize,
    pub indices: [usize; BANDPASS_DIMENSION],
    pub components: [f64; BANDPASS_DIMENSION],
    pub descriptors: [f64; 2],
}

/// MAP-Elites search settings.
#[derive(Clone, Debug)]
pub struct QdConfig {
    pub evaluations: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    pub workers: i32,
    pub seed: u64,
    pub points: usize,
    pub mc_draws: usize,
}

/// One archive progress record.
#[derive(Clone, Debug)]
pub struct QdProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub coverage: f64,
    pub qd_score: f64,
    pub best_quality: f64,
    pub invalid_fraction: f64,
}

/// One occupied archive niche.
#[derive(Clone, Debug)]
pub struct Elite {
    pub niche: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub visits: u64,
    pub evaluation: QdEvaluation,
    pub selected: bool,
    pub curve: Vec<(f64, f64)>,
}

/// Final QD outcome.
#[derive(Clone, Debug)]
pub struct QdResult {
    pub requested_evaluations: usize,
    pub actual_evaluations: usize,
    pub ac_solves: usize,
    pub invalid_evaluations: usize,
    pub out_of_range_descriptors: usize,
    pub distinct_elite_designs: usize,
    pub capacity: usize,
    pub elapsed: Duration,
    pub elites: Vec<Elite>,
    pub progress: Vec<QdProgress>,
}

type NominalResult = ([f64; 5], [usize; 5], BandpassFeatures, Vec<(f64, f64)>);

fn catalogues() -> (Vec<f64>, Vec<f64>) {
    (e12_values(100.0, 100_000.0), e12_values(10e-12, 1e-6))
}

fn nominal(
    x: &[f64],
    resistor_values: &[f64],
    capacitor_values: &[f64],
    points: usize,
) -> Option<NominalResult> {
    let (components, indices) = decode_bandpass_e12(x, resistor_values, capacitor_values);
    let curve = gain_curve(
        &mfb_bandpass(&components),
        "out",
        10.0,
        10_000_000.0,
        points,
    )?;
    let features = bandpass_features(&curve)?;
    Some((components, indices, features, curve))
}

/// Evaluate one catalogue design with common-random-number tolerance draws.
pub fn evaluate_qd(
    x: &[f64],
    resistor_values: &[f64],
    capacitor_values: &[f64],
    tolerance: &ToleranceDraws,
    points: usize,
) -> Option<QdEvaluation> {
    evaluate_qd_counted(x, resistor_values, capacitor_values, tolerance, points).0
}

fn evaluate_qd_counted(
    x: &[f64],
    resistor_values: &[f64],
    capacitor_values: &[f64],
    tolerance: &ToleranceDraws,
    points: usize,
) -> (Option<QdEvaluation>, usize) {
    let Some((components, indices, features, _)) =
        nominal(x, resistor_values, capacitor_values, points)
    else {
        return (None, 1);
    };
    if tolerance.multipliers.is_empty() {
        return (None, 1);
    }
    let mut ac_solves = 1;
    let offsets = tolerance
        .multipliers
        .iter()
        .filter_map(|multipliers| {
            ac_solves += 1;
            let mut perturbed = components;
            for (value, multiplier) in perturbed.iter_mut().zip(multipliers) {
                *value *= multiplier;
            }
            let curve = gain_curve(&mfb_bandpass(&perturbed), "out", 10.0, 10_000_000.0, points)?;
            let perturbed_features = bandpass_features(&curve)?;
            Some(20.0 * (perturbed_features.peak_hz / features.peak_hz).log10())
        })
        .collect::<Vec<_>>();
    if offsets.len() != tolerance.multipliers.len() {
        return (None, ac_solves);
    }
    let mean = offsets.iter().sum::<f64>() / offsets.len() as f64;
    let quality = (offsets
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / offsets.len() as f64)
        .sqrt();
    let descriptors = [features.peak_hz.log10(), features.peak_db];
    (
        (quality.is_finite() && descriptors.iter().all(|value| value.is_finite())).then_some(
            QdEvaluation {
                coordinates: x.to_vec(),
                indices,
                components,
                features,
                quality,
                descriptors,
            },
        ),
        ac_solves,
    )
}

/// Uniformly sample the decoded catalogue before descriptor bounds are used.
pub fn range_study(
    samples: usize,
    seed: u64,
    points: usize,
) -> Result<Vec<RangeStudyRow>, Box<dyn Error>> {
    let (resistors, capacitors) = catalogues();
    let mut rng = Rng::new(seed);
    let mut rows = Vec::new();
    for sample in 0..samples {
        let x = vec![
            rng.uniform01() * (resistors.len() - 1) as f64,
            rng.uniform01() * (resistors.len() - 1) as f64,
            rng.uniform01() * (resistors.len() - 1) as f64,
            rng.uniform01() * (capacitors.len() - 1) as f64,
            rng.uniform01() * (capacitors.len() - 1) as f64,
        ];
        if let Some((components, indices, features, _)) =
            nominal(&x, &resistors, &capacitors, points)
        {
            rows.push(RangeStudyRow {
                sample,
                indices,
                components,
                descriptors: [features.peak_hz.log10(), features.peak_db],
            });
        }
    }
    if rows.is_empty() {
        return Err("descriptor range study produced no valid band-pass response".into());
    }
    Ok(rows)
}

struct CircuitQdBatch<'a> {
    resistors: &'a [f64],
    capacitors: &'a [f64],
    tolerance: &'a ToleranceDraws,
    workers: i32,
    points: usize,
    evaluations: Arc<AtomicUsize>,
    ac_solves: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    outside: Arc<AtomicUsize>,
}

impl QdBatchFitness for CircuitQdBatch<'_> {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let evaluated = parallel_batch(xs, self.workers, |x| {
            evaluate_qd_counted(
                x,
                self.resistors,
                self.capacitors,
                self.tolerance,
                self.points,
            )
        });
        self.evaluations
            .fetch_add(evaluated.len(), Ordering::Relaxed);
        self.ac_solves.fetch_add(
            evaluated.iter().map(|(_, solves)| solves).sum(),
            Ordering::Relaxed,
        );
        evaluated
            .into_iter()
            .map(|(evaluation, _)| match evaluation {
                Some(value) => {
                    let outside = value
                        .descriptors
                        .iter()
                        .zip(DESCRIPTOR_LOWER.iter().zip(DESCRIPTOR_UPPER))
                        .any(|(descriptor, (lower, upper))| {
                            descriptor < lower || *descriptor > upper
                        });
                    if outside {
                        self.outside.fetch_add(1, Ordering::Relaxed);
                        (f64::INFINITY, value.descriptors.to_vec())
                    } else {
                        (value.quality, value.descriptors.to_vec())
                    }
                }
                None => {
                    self.invalid.fetch_add(1, Ordering::Relaxed);
                    (INVALID_COST, vec![f64::INFINITY; 2])
                }
            })
            .collect()
    }
}

fn grid_coordinate(value: f64, lower: f64, upper: f64, side: usize) -> usize {
    (((value - lower) / (upper - lower) * side as f64).floor() as isize).clamp(0, side as isize - 1)
        as usize
}

fn select_catalogue_examples(elites: &mut [Elite]) {
    if elites.is_empty() {
        return;
    }
    let mut order: Vec<usize> = (0..elites.len()).collect();
    order.sort_by(|left, right| {
        elites[*left].evaluation.descriptors[0].total_cmp(&elites[*right].evaluation.descriptors[0])
    });
    let count = 6.min(order.len());
    for selection in 0..count {
        let rank = if count == 1 {
            0
        } else {
            selection * (order.len() - 1) / (count - 1)
        };
        elites[order[rank]].selected = true;
    }
}

/// Run deterministic batch MAP-Elites over the E12 catalogue.
pub fn optimize_qd(config: &QdConfig) -> Result<QdResult, Box<dyn Error>> {
    if config.evaluations == 0 || config.mc_draws == 0 {
        return Err("QD evaluations and Monte Carlo draws must be positive".into());
    }
    if config.chunk_size < 2 || !config.chunk_size.is_multiple_of(2) {
        return Err("QD chunk size must be an even number of at least two".into());
    }
    let side = (config.capacity as f64).sqrt() as usize;
    if side < 2 || side * side != config.capacity {
        return Err("QD capacity must be a perfect square of at least four".into());
    }
    let (resistors, capacitors) = catalogues();
    let lower = [0.0, 0.0, 0.0, 0.0, 0.0];
    let upper = [
        (resistors.len() - 1) as f64,
        (resistors.len() - 1) as f64,
        (resistors.len() - 1) as f64,
        (capacitors.len() - 1) as f64,
        (capacitors.len() - 1) as f64,
    ];
    let generations = config.evaluations.div_ceil(config.chunk_size);
    let tolerance = ToleranceDraws::new(config.seed.wrapping_add(91_337), config.mc_draws);
    let mut rng = Rng::new(config.seed);
    let mut archive = Archive::try_new(
        BANDPASS_DIMENSION,
        &DESCRIPTOR_LOWER,
        &DESCRIPTOR_UPPER,
        config.capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(&lower, &upper, &mut rng);
    let evaluations = Arc::new(AtomicUsize::new(0));
    let ac_solves = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let outside = Arc::new(AtomicUsize::new(0));
    let mut evaluator = CircuitQdBatch {
        resistors: &resistors,
        capacitors: &capacitors,
        tolerance: &tolerance,
        workers: config.workers,
        points: config.points,
        evaluations: Arc::clone(&evaluations),
        ac_solves: Arc::clone(&ac_solves),
        invalid: Arc::clone(&invalid),
        outside: Arc::clone(&outside),
    };
    let params = MapElitesParams {
        generations,
        chunk_size: config.chunk_size,
        use_sbx: false,
        iso_sigma: 0.02,
        line_sigma: 0.2,
        cma_generations: 0,
        ..Default::default()
    };
    let started = Instant::now();
    let mut progress = Vec::new();
    let mut callback = |generation: usize, current: &Archive| {
        if generation == 1 || generation.is_multiple_of(5) || generation == generations {
            progress.push(QdProgress {
                evaluations: generation * config.chunk_size,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                coverage: current.occupied() as f64 / current.capacity() as f64,
                qd_score: current.qd_score(),
                best_quality: current.best_y(),
                invalid_fraction: invalid.load(Ordering::Relaxed) as f64
                    / evaluations.load(Ordering::Relaxed).max(1) as f64,
            });
        }
    };
    map_elites_batch_with_progress(
        &mut archive,
        &mut evaluator,
        &lower,
        &upper,
        &params,
        &mut rng,
        &mut callback,
    )?;
    let actual_evaluations = evaluations.load(Ordering::Relaxed);
    let invalid_evaluations = invalid.load(Ordering::Relaxed);
    let out_of_range_descriptors = outside.load(Ordering::Relaxed);
    let mut elites = Vec::new();
    for niche in 0..archive.capacity() {
        if !archive.ys()[niche].is_finite() {
            continue;
        }
        let x = &archive.xs()[niche];
        let Some(mut evaluation) =
            evaluate_qd(x, &resistors, &capacitors, &tolerance, config.points)
        else {
            continue;
        };
        let Some((_, _, _, curve)) = nominal(x, &resistors, &capacitors, config.points) else {
            continue;
        };
        evaluation.coordinates = x.clone();
        elites.push(Elite {
            niche,
            grid_x: grid_coordinate(
                evaluation.descriptors[0],
                DESCRIPTOR_LOWER[0],
                DESCRIPTOR_UPPER[0],
                side,
            ),
            grid_y: grid_coordinate(
                evaluation.descriptors[1],
                DESCRIPTOR_LOWER[1],
                DESCRIPTOR_UPPER[1],
                side,
            ),
            visits: archive.counts()[niche],
            evaluation,
            selected: false,
            curve,
        });
    }
    select_catalogue_examples(&mut elites);
    let distinct_elite_designs = elites
        .iter()
        .map(|elite| elite.evaluation.indices)
        .collect::<HashSet<_>>()
        .len();
    Ok(QdResult {
        requested_evaluations: config.evaluations,
        actual_evaluations,
        ac_solves: ac_solves.load(Ordering::Relaxed),
        invalid_evaluations,
        out_of_range_descriptors,
        distinct_elite_designs,
        capacity: config.capacity,
        elapsed: started.elapsed(),
        elites,
        progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_random_numbers_make_repeated_quality_identical() {
        let (resistors, capacitors) = catalogues();
        let tolerance = ToleranceDraws::new(42, 4);
        let coordinates = [18.0, 12.0, 20.0, 30.0, 30.0];
        let first = evaluate_qd(&coordinates, &resistors, &capacitors, &tolerance, 31).unwrap();
        let second = evaluate_qd(&coordinates, &resistors, &capacitors, &tolerance, 31).unwrap();
        assert_eq!(first.indices, second.indices);
        assert_eq!(first.quality, second.quality);
    }

    #[test]
    fn range_study_reports_finite_descriptors() {
        let rows = range_study(32, 4, 31).unwrap();
        assert!(!rows.is_empty());
        assert!(
            rows.iter()
                .all(|row| row.descriptors.iter().all(|value| value.is_finite()))
        );
    }

    #[test]
    fn tiny_archive_contains_replayable_elites() {
        let result = optimize_qd(&QdConfig {
            evaluations: 64,
            capacity: 16,
            chunk_size: 16,
            workers: 2,
            seed: 9,
            points: 31,
            mc_draws: 2,
        })
        .unwrap();
        assert!(!result.elites.is_empty());
        assert_eq!(result.actual_evaluations, 64);
        assert!((64..=192).contains(&result.ac_solves));
    }
}
