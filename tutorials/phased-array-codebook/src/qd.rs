//! MAP-Elites beam-codebook synthesis.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    Archive, MapElitesParams, QdBatchFitness, Rng, map_elites_batch_with_progress, parallel_batch,
};

use crate::archive_grid::ArchiveGrid;
use crate::decode::{Excitation, decode_beam, from_codes};
use crate::pilot::{DESCRIPTOR_LOWER, DESCRIPTOR_UPPER, PUBLICATION_CAPACITY};
use crate::scenarios::{
    HoldoutContext, RobustMetrics, evaluate_holdout, evaluate_training,
    representative_holdout_metrics,
};
use crate::so::{BeamContext, DIMENSION, ELEMENTS, ROBUST_PSLL_LIMIT_DB, analytic_seed};

/// QD campaign settings.
#[derive(Clone, Debug)]
pub struct QdConfig {
    /// Total candidate budget, including analytic seeds.
    pub evaluations: usize,
    /// Regular-grid archive capacity.
    pub capacity: usize,
    /// Batch size.
    pub chunk_size: usize,
    /// Candidate workers.
    pub workers: i32,
    /// Root seed.
    pub seed: u64,
    /// Training cut samples.
    pub points: usize,
}

/// MAP-Elites progress.
#[derive(Clone, Debug)]
pub struct QdProgress {
    /// Candidate evaluations.
    pub evaluations: usize,
    /// Wall time.
    pub elapsed_seconds: f64,
    /// Occupied fraction.
    pub coverage: f64,
    /// Archive QD score.
    pub qd_score: f64,
    /// Best minimized linear sidelobe ratio.
    pub best_quality: f64,
    /// Invalid candidate fraction.
    pub invalid_fraction: f64,
    /// Physically evaluable but constraint-infeasible candidate fraction.
    pub infeasible_fraction: f64,
}

/// Replayed occupied niche.
#[derive(Clone, Debug)]
pub struct CodebookEntry {
    /// Archive niche index.
    pub niche: usize,
    /// Grid x coordinate.
    pub grid_x: usize,
    /// Grid y coordinate.
    pub grid_y: usize,
    /// Archive visit count.
    pub visits: u64,
    /// Minimized linear sidelobe ratio.
    pub quality: f64,
    /// Training descriptor.
    pub descriptors: [f64; 2],
    /// Representative holdout descriptor.
    pub holdout_descriptors: [f64; 2],
    /// Training robust metrics.
    pub robust: RobustMetrics,
    /// Holdout robust metrics.
    pub holdout: RobustMetrics,
    /// Machine register state.
    pub excitation: Excitation,
    /// Normalized controls retained for exact replay.
    pub controls: Vec<f64>,
}

/// Final codebook outcome.
#[derive(Clone, Debug)]
pub struct QdResult {
    /// Requested budget.
    pub requested_evaluations: usize,
    /// Actual candidate calls.
    pub actual_evaluations: usize,
    /// Invalid candidates.
    pub invalid_evaluations: usize,
    /// Physically evaluable candidates rejected by robust feasibility.
    pub infeasible_evaluations: usize,
    /// Descriptor clamping events.
    pub clamped_descriptors: usize,
    /// Archive capacity.
    pub capacity: usize,
    /// Wall duration.
    pub elapsed: Duration,
    /// Occupied entries.
    pub entries: Vec<CodebookEntry>,
    /// Progress trace.
    pub progress: Vec<QdProgress>,
}

#[derive(Clone)]
struct Evaluation {
    excitation: Excitation,
    robust: RobustMetrics,
    descriptors: [f64; 2],
    quality: f64,
}

enum EvaluationFailure {
    Invalid,
    Infeasible,
}

fn evaluate(controls: &[f64], context: &BeamContext) -> Result<Evaluation, EvaluationFailure> {
    let excitation = decode_beam(controls, ELEMENTS, &context.quantization)
        .map_err(|_| EvaluationFailure::Invalid)?;
    let robust = evaluate_training(&excitation.weights, &context.steering, &context.grid)
        .map_err(|_| EvaluationFailure::Invalid)?;
    if robust.nominal.degenerate
        || robust.kernel_failures > 0
        || !robust.nominal.hpbw_deg.is_finite()
    {
        return Err(EvaluationFailure::Invalid);
    }
    let descriptors = [robust.nominal.peak_theta_deg, robust.nominal.hpbw_deg];
    let quality = 10_f64.powf(robust.worst_psll_db / 20.0);
    if !quality.is_finite() || descriptors.iter().any(|value| !value.is_finite()) {
        return Err(EvaluationFailure::Invalid);
    }
    if robust.degenerate_scenarios > 0 || robust.worst_psll_db > ROBUST_PSLL_LIMIT_DB {
        return Err(EvaluationFailure::Infeasible);
    }
    Ok(Evaluation {
        excitation,
        robust,
        descriptors,
        quality,
    })
}

fn clamp_descriptors(descriptors: [f64; 2]) -> ([f64; 2], bool) {
    let clamped = [
        descriptors[0].clamp(DESCRIPTOR_LOWER[0], DESCRIPTOR_UPPER[0]),
        descriptors[1].clamp(DESCRIPTOR_LOWER[1], DESCRIPTOR_UPPER[1]),
    ];
    (clamped, clamped != descriptors)
}

struct BeamQdBatch {
    context: Arc<BeamContext>,
    workers: i32,
    evaluations: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    infeasible: Arc<AtomicUsize>,
    clamped: Arc<AtomicUsize>,
}

impl QdBatchFitness for BeamQdBatch {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let context = Arc::clone(&self.context);
        let evaluated = parallel_batch(xs, self.workers, move |x| evaluate(x, &context));
        self.evaluations
            .fetch_add(evaluated.len(), Ordering::Relaxed);
        evaluated
            .into_iter()
            .map(|evaluation| match evaluation {
                Ok(evaluation) => {
                    let (descriptors, was_clamped) = clamp_descriptors(evaluation.descriptors);
                    if was_clamped {
                        self.clamped.fetch_add(1, Ordering::Relaxed);
                    }
                    (evaluation.quality, descriptors.to_vec())
                }
                Err(EvaluationFailure::Invalid) => {
                    self.invalid.fetch_add(1, Ordering::Relaxed);
                    (f64::INFINITY, vec![0.0, 0.0])
                }
                Err(EvaluationFailure::Infeasible) => {
                    self.infeasible.fetch_add(1, Ordering::Relaxed);
                    (f64::INFINITY, vec![0.0, 0.0])
                }
            })
            .collect()
    }
}

fn seed_candidates(limit: usize) -> Vec<Vec<f64>> {
    let mut seeds = Vec::new();
    // Cover the full steering range first, beginning with the taper most
    // likely to satisfy the robust sidelobe gate.
    for taper_index in (0..8).rev() {
        for steer_index in 0..31 {
            let steer = -50.0 + 100.0 * steer_index as f64 / 30.0;
            seeds.push(analytic_seed(steer, taper_index as f64 / 7.0));
            if seeds.len() == limit {
                return seeds;
            }
        }
    }
    seeds
}

/// Build the quantized beam codebook.
pub fn optimize_qd(config: &QdConfig) -> Result<QdResult, Box<dyn Error>> {
    if config.evaluations < 32
        || config.chunk_size < 2
        || !config.chunk_size.is_multiple_of(2)
        || config.points < 101
    {
        return Err("QD budget/grid settings are too small".into());
    }
    if config.capacity != PUBLICATION_CAPACITY && config.capacity != 60 {
        return Err("QD capacity must be 60 (smoke) or the 120-cell publication grid".into());
    }
    let context = Arc::new(BeamContext::stage_a(config.points));
    let holdout_beam = BeamContext::stage_a(2 * config.points - 1);
    let holdout_context = HoldoutContext::new(holdout_beam.array, holdout_beam.grid)?;
    let layout = ArchiveGrid::new(config.capacity);
    let mut rng = Rng::new(config.seed);
    let mut archive = Archive::try_new(
        DIMENSION,
        &DESCRIPTOR_LOWER,
        &DESCRIPTOR_UPPER,
        config.capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(&[0.0; DIMENSION], &[1.0; DIMENSION], &mut rng);
    let evaluations = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let infeasible = Arc::new(AtomicUsize::new(0));
    let clamped = Arc::new(AtomicUsize::new(0));
    let mut evaluator = BeamQdBatch {
        context: Arc::clone(&context),
        workers: config.workers,
        evaluations: Arc::clone(&evaluations),
        invalid: Arc::clone(&invalid),
        infeasible: Arc::clone(&infeasible),
        clamped: Arc::clone(&clamped),
    };
    let seed_count = config.evaluations.min(248);
    let seeds = seed_candidates(seed_count);
    let seeded = evaluator.eval_batch(&seeds);
    for (controls, (quality, descriptors)) in seeds.iter().zip(seeded) {
        if quality.is_finite() {
            let niche = archive.index_of_niche(&descriptors);
            archive.set(niche, quality, &descriptors, controls);
        }
    }
    let remaining = config.evaluations.saturating_sub(seed_count);
    let generations = remaining.div_ceil(config.chunk_size);
    let started = Instant::now();
    let mut progress = vec![QdProgress {
        evaluations: evaluations.load(Ordering::Relaxed),
        elapsed_seconds: 0.0,
        coverage: archive.occupied() as f64 / archive.capacity() as f64,
        qd_score: archive.qd_score(),
        best_quality: archive.best_y(),
        invalid_fraction: invalid.load(Ordering::Relaxed) as f64
            / evaluations.load(Ordering::Relaxed).max(1) as f64,
        infeasible_fraction: infeasible.load(Ordering::Relaxed) as f64
            / evaluations.load(Ordering::Relaxed).max(1) as f64,
    }];
    if generations > 0 {
        let params = MapElitesParams {
            generations,
            chunk_size: config.chunk_size,
            use_sbx: true,
            iso_sigma: 0.02,
            line_sigma: 0.2,
            cma_generations: 0,
            ..Default::default()
        };
        let mut callback = |generation: usize, current: &Archive| {
            if generation == 1 || generation.is_multiple_of(10) || generation == generations {
                let actual = evaluations.load(Ordering::Relaxed);
                progress.push(QdProgress {
                    evaluations: actual,
                    elapsed_seconds: started.elapsed().as_secs_f64(),
                    coverage: current.occupied() as f64 / current.capacity() as f64,
                    qd_score: current.qd_score(),
                    best_quality: current.best_y(),
                    invalid_fraction: invalid.load(Ordering::Relaxed) as f64 / actual.max(1) as f64,
                    infeasible_fraction: infeasible.load(Ordering::Relaxed) as f64
                        / actual.max(1) as f64,
                });
            }
        };
        map_elites_batch_with_progress(
            &mut archive,
            &mut evaluator,
            &[0.0; DIMENSION],
            &[1.0; DIMENSION],
            &params,
            &mut rng,
            &mut callback,
        )?;
    }
    let mut entries = Vec::new();
    for niche in 0..archive.capacity() {
        if !archive.ys()[niche].is_finite() {
            continue;
        }
        let controls = archive.xs()[niche].clone();
        let Ok(evaluation) = evaluate(&controls, &context) else {
            continue;
        };
        let holdout = evaluate_holdout(&evaluation.excitation.weights, &holdout_context)?;
        let holdout_metrics =
            representative_holdout_metrics(&evaluation.excitation.weights, &holdout_context)?;
        let holdout_descriptors = [holdout_metrics.peak_theta_deg, holdout_metrics.hpbw_deg];
        let (grid_x, grid_y) = layout
            .coordinate(niche)
            .ok_or("archive returned an out-of-range niche")?;
        entries.push(CodebookEntry {
            niche,
            grid_x,
            grid_y,
            visits: archive.counts()[niche],
            quality: evaluation.quality,
            descriptors: evaluation.descriptors,
            holdout_descriptors,
            robust: evaluation.robust,
            holdout,
            excitation: evaluation.excitation,
            controls,
        });
    }
    Ok(QdResult {
        requested_evaluations: config.evaluations,
        actual_evaluations: evaluations.load(Ordering::Relaxed),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        infeasible_evaluations: infeasible.load(Ordering::Relaxed),
        clamped_descriptors: clamped.load(Ordering::Relaxed),
        capacity: config.capacity,
        elapsed: started.elapsed(),
        entries,
        progress,
    })
}

/// Replay one codebook entry from register values.
#[must_use]
pub fn replay_codes(entry: &CodebookEntry, context: &BeamContext) -> Option<RobustMetrics> {
    let decoded = from_codes(
        &entry.excitation.phase_codes,
        &entry.excitation.attenuator_codes,
        &entry.excitation.active,
        &context.quantization,
    )
    .ok()?;
    evaluate_training(&decoded.weights, &context.steering, &context.grid).ok()
}

/// Return the closest codebook beam to a requested descriptor.
#[must_use]
pub fn nearest_entry(entries: &[CodebookEntry], target: [f64; 2]) -> Option<&CodebookEntry> {
    entries.iter().min_by(|left, right| {
        let distance = |entry: &CodebookEntry| {
            ((entry.descriptors[0] - target[0]) / (DESCRIPTOR_UPPER[0] - DESCRIPTOR_LOWER[0]))
                .powi(2)
                + ((entry.descriptors[1] - target[1]) / (DESCRIPTOR_UPPER[1] - DESCRIPTOR_LOWER[1]))
                    .powi(2)
        };
        distance(left).total_cmp(&distance(right))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_archive_contains_machine_replayable_register_codes() {
        let result = optimize_qd(&QdConfig {
            evaluations: 128,
            capacity: 60,
            chunk_size: 32,
            workers: 2,
            seed: 42,
            points: 361,
        })
        .unwrap();
        assert!(result.entries.len() >= 8, "{}", result.entries.len());
        let context = BeamContext::stage_a(361);
        let layout = ArchiveGrid::new(result.capacity);
        for entry in &result.entries {
            let replayed = replay_codes(entry, &context).unwrap();
            assert!((replayed.nominal.psll_db - entry.robust.nominal.psll_db).abs() < 1.0e-12);
            assert!(entry.descriptors[0] >= DESCRIPTOR_LOWER[0]);
            assert!(entry.descriptors[0] <= DESCRIPTOR_UPPER[0]);
            assert_eq!(
                layout.coordinate(entry.niche),
                Some((entry.grid_x, entry.grid_y))
            );
        }
        let requested = nearest_entry(&result.entries, [20.0, 8.0]).unwrap();
        assert!((requested.descriptors[0] - 20.0).abs() < 12.0);
    }

    #[test]
    fn checked_in_codebook_replays_from_integer_registers() {
        let context = BeamContext::stage_a(4_001);
        let codebook = include_str!("../results/publication/qd/codebook.csv");
        let mut rows = 0;
        for line in codebook.lines().skip(1).filter(|line| !line.is_empty()) {
            let columns = line.split(',').collect::<Vec<_>>();
            assert_eq!(columns.len(), 10);
            let expected_psll = columns[3].parse::<f64>().unwrap();
            let codes = |column: &str| {
                column
                    .trim_matches('"')
                    .split(';')
                    .map(|code| code.parse::<u32>().unwrap())
                    .collect::<Vec<_>>()
            };
            let phase = codes(columns[8]);
            let attenuation = codes(columns[9]);
            let excitation = from_codes(
                &phase,
                &attenuation,
                &vec![true; phase.len()],
                &context.quantization,
            )
            .unwrap();
            let replayed =
                evaluate_training(&excitation.weights, &context.steering, &context.grid).unwrap();
            assert!((replayed.nominal.psll_db - expected_psll).abs() < 1.0e-9);
            rows += 1;
        }
        assert_eq!(rows, 43);
    }
}
