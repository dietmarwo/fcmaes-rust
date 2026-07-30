//! MAP-Elites repertoire over emergent fleet size and route imbalance.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    Archive, MapElitesParams, QdBatchFitness, Rng, map_elites_batch_with_progress, parallel_batch,
};

use crate::archive_grid::ArchiveGrid;
use crate::instance::{DIMENSION, Instance};
use crate::pilot::{DESCRIPTOR_LOWER, DESCRIPTOR_UPPER};
use crate::scenarios::{
    RobustEvaluation, evaluate_holdout, evaluate_training, robust_seed_controls,
};

/// QD settings.
#[derive(Clone, Debug)]
pub struct QdConfig {
    /// Candidate calls.
    pub evaluations: usize,
    /// Archive capacity, 60 or 120.
    pub capacity: usize,
    /// Batch size.
    pub chunk_size: usize,
    /// Candidate workers.
    pub workers: i32,
    /// Root seed.
    pub seed: u64,
}

/// MAP-Elites progress.
#[derive(Clone, Debug)]
pub struct QdProgress {
    /// Calls.
    pub evaluations: usize,
    /// Wall seconds.
    pub elapsed_seconds: f64,
    /// Occupied fraction.
    pub coverage: f64,
    /// Archive QD score.
    pub qd_score: f64,
    /// Best robust cost.
    pub best_quality: f64,
    /// Infeasible fraction.
    pub invalid_fraction: f64,
}

/// Replayed occupied niche.
#[derive(Clone, Debug)]
pub struct RepertoireEntry {
    /// Archive index.
    pub niche: usize,
    /// Grid column.
    pub grid_x: usize,
    /// Grid row.
    pub grid_y: usize,
    /// Archive visits.
    pub visits: u64,
    /// Robust cost.
    pub quality: f64,
    /// Vehicles × imbalance.
    pub descriptors: [f64; 2],
    /// Training evaluation.
    pub training: RobustEvaluation,
    /// Holdout evaluation.
    pub holdout: Option<RobustEvaluation>,
    /// Replayable normalized controls.
    pub controls: Vec<f64>,
}

/// QD campaign output.
#[derive(Clone, Debug)]
pub struct QdResult {
    /// Requested calls.
    pub requested_evaluations: usize,
    /// Actual calls.
    pub actual_evaluations: usize,
    /// Infeasible calls.
    pub invalid_evaluations: usize,
    /// Descriptor clamping events.
    pub clamped_descriptors: usize,
    /// Archive capacity.
    pub capacity: usize,
    /// Wall time.
    pub elapsed: Duration,
    /// Occupied repertoire.
    pub entries: Vec<RepertoireEntry>,
    /// Progress.
    pub progress: Vec<QdProgress>,
}

fn evaluate_candidate(controls: &[f64], instance: &Instance) -> Option<RobustEvaluation> {
    evaluate_training(controls, instance).filter(RobustEvaluation::feasible)
}

fn descriptor(evaluation: &RobustEvaluation) -> [f64; 2] {
    [
        evaluation.nominal().metrics.used_vehicles as f64,
        evaluation.nominal().metrics.imbalance_cv,
    ]
}

fn clamp_descriptor(values: [f64; 2]) -> ([f64; 2], bool) {
    let clamped = [
        values[0].clamp(DESCRIPTOR_LOWER[0], DESCRIPTOR_UPPER[0]),
        values[1].clamp(DESCRIPTOR_LOWER[1], DESCRIPTOR_UPPER[1]),
    ];
    (clamped, clamped != values)
}

struct RoutingBatch {
    instance: Arc<Instance>,
    workers: i32,
    calls: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    clamped: Arc<AtomicUsize>,
}

impl QdBatchFitness for RoutingBatch {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let instance = Arc::clone(&self.instance);
        let evaluated = parallel_batch(xs, self.workers, move |x| evaluate_candidate(x, &instance));
        self.calls.fetch_add(evaluated.len(), Ordering::Relaxed);
        evaluated
            .into_iter()
            .map(|result| match result {
                Some(evaluation) => {
                    let (descriptors, clamped) = clamp_descriptor(descriptor(&evaluation));
                    if clamped {
                        self.clamped.fetch_add(1, Ordering::Relaxed);
                    }
                    (evaluation.worst_cost, descriptors.to_vec())
                }
                None => {
                    self.invalid.fetch_add(1, Ordering::Relaxed);
                    (f64::INFINITY, vec![0.0, 0.0])
                }
            })
            .collect()
    }
}

fn seeds(instance: &Instance, count: usize, seed: u64) -> Vec<Vec<f64>> {
    let witness = robust_seed_controls(instance);
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|index| {
            if index == 0 {
                witness.clone()
            } else {
                let scale = 0.02 + 0.65 * (index % 17) as f64 / 16.0;
                witness
                    .iter()
                    .map(|value| (value + scale * (rng.uniform01() - 0.5)).clamp(0.0, 1.0))
                    .collect()
            }
        })
        .collect()
}

/// Build the robust dispatch repertoire.
pub fn optimize(instance: &Instance, config: &QdConfig) -> Result<QdResult, Box<dyn Error>> {
    if config.evaluations < 32
        || config.chunk_size < 2
        || !config.chunk_size.is_multiple_of(2)
        || !matches!(config.capacity, 60 | 120)
    {
        return Err("invalid QD settings".into());
    }
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
    let calls = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clamped = Arc::new(AtomicUsize::new(0));
    let mut evaluator = RoutingBatch {
        instance: Arc::new(instance.clone()),
        workers: config.workers,
        calls: Arc::clone(&calls),
        invalid: Arc::clone(&invalid),
        clamped: Arc::clone(&clamped),
    };
    let seed_count = config.evaluations.min(256);
    let seeded_x = seeds(instance, seed_count, config.seed);
    let seeded_y = evaluator.eval_batch(&seeded_x);
    for (controls, (quality, descriptors)) in seeded_x.iter().zip(seeded_y) {
        if quality.is_finite() {
            let niche = archive.index_of_niche(&descriptors);
            archive.set(niche, quality, &descriptors, controls);
        }
    }
    if archive.occupied() == 0 {
        return Err("descriptor seed set contained no robust-feasible plan".into());
    }
    let started = Instant::now();
    let mut progress = vec![QdProgress {
        evaluations: calls.load(Ordering::Relaxed),
        elapsed_seconds: 0.0,
        coverage: archive.occupied() as f64 / archive.capacity() as f64,
        qd_score: archive.qd_score(),
        best_quality: archive.best_y(),
        invalid_fraction: invalid.load(Ordering::Relaxed) as f64
            / calls.load(Ordering::Relaxed).max(1) as f64,
    }];
    let remaining = config.evaluations.saturating_sub(seed_count);
    let generations = remaining.div_ceil(config.chunk_size);
    if generations > 0 {
        let params = MapElitesParams {
            generations,
            chunk_size: config.chunk_size,
            use_sbx: true,
            iso_sigma: 0.03,
            line_sigma: 0.2,
            cma_generations: 0,
            ..Default::default()
        };
        let mut callback = |generation: usize, current: &Archive| {
            if generation == 1 || generation.is_multiple_of(10) || generation == generations {
                let actual = calls.load(Ordering::Relaxed);
                progress.push(QdProgress {
                    evaluations: actual,
                    elapsed_seconds: started.elapsed().as_secs_f64(),
                    coverage: current.occupied() as f64 / current.capacity() as f64,
                    qd_score: current.qd_score(),
                    best_quality: current.best_y(),
                    invalid_fraction: invalid.load(Ordering::Relaxed) as f64 / actual.max(1) as f64,
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
    let layout = ArchiveGrid::new(config.capacity);
    let mut entries = Vec::new();
    for niche in 0..archive.capacity() {
        if !archive.ys()[niche].is_finite() {
            continue;
        }
        let controls = archive.xs()[niche].clone();
        let Some(training) = evaluate_candidate(&controls, instance) else {
            continue;
        };
        let descriptors = descriptor(&training);
        let coordinates = layout
            .coordinates(descriptors, DESCRIPTOR_LOWER, DESCRIPTOR_UPPER)
            .expect("replayed descriptors are finite");
        debug_assert_eq!(
            layout.niche(descriptors, DESCRIPTOR_LOWER, DESCRIPTOR_UPPER),
            Some(niche)
        );
        entries.push(RepertoireEntry {
            niche,
            grid_x: coordinates[0],
            grid_y: coordinates[1],
            visits: archive.counts()[niche],
            quality: training.worst_cost,
            descriptors,
            holdout: evaluate_holdout(&controls, instance),
            training,
            controls,
        });
    }
    Ok(QdResult {
        requested_evaluations: config.evaluations,
        actual_evaluations: calls.load(Ordering::Relaxed),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clamped_descriptors: clamped.load(Ordering::Relaxed),
        capacity: config.capacity,
        elapsed: started.elapsed(),
        entries,
        progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::load_primary;

    #[test]
    fn small_archive_contains_replayable_hard_feasible_plans() {
        let instance = load_primary().unwrap();
        let result = optimize(
            &instance,
            &QdConfig {
                evaluations: 320,
                capacity: 60,
                chunk_size: 32,
                workers: 2,
                seed: 42,
            },
        )
        .unwrap();
        assert!(!result.entries.is_empty());
        assert!(result.entries.iter().all(|entry| {
            entry.training.feasible()
                && (evaluate_training(&entry.controls, &instance)
                    .unwrap()
                    .worst_cost
                    - entry.quality)
                    .abs()
                    < 1.0e-9
        }));
    }
}
