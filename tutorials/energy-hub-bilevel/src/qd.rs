//! MAP-Elites portfolio of robust energy-hub designs.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    Archive, MapElitesParams, QdBatchFitness, Rng, map_elites_batch_with_progress, parallel_batch,
};

use crate::archive_grid::ArchiveGrid;
use crate::config::Preset;
use crate::decode::DIMENSION;
use crate::evaluate::{
    OuterEvaluation, ScenarioEvaluation, behavior, behavior_for_scenario, evaluate_holdout,
    evaluate_training, feasible,
};
use crate::pilot::{DescriptorPair, structured_candidate};
use crate::scenarios::holdout;

/// QD campaign settings.
#[derive(Clone, Copy, Debug)]
pub struct QdConfig {
    /// Horizon preset.
    pub preset: Preset,
    /// Registered pair selected by the pilot.
    pub descriptor_pair: DescriptorPair,
    /// Total candidate budget, including structured seeds.
    pub evaluations: usize,
    /// Archive capacity.
    pub capacity: usize,
    /// Candidate batch size.
    pub chunk_size: usize,
    /// Candidate workers.
    pub workers: i32,
    /// Root seed.
    pub seed: u64,
}

/// One archive progress observation.
#[derive(Clone, Copy, Debug)]
pub struct QdProgress {
    /// Candidate calls.
    pub evaluations: usize,
    /// Wall seconds.
    pub elapsed_seconds: f64,
    /// Occupied archive fraction.
    pub coverage: f64,
    /// Archive QD score.
    pub qd_score: f64,
    /// Best minimized LCOE.
    pub best_quality: f64,
    /// Invalid candidate fraction.
    pub invalid_fraction: f64,
}

/// Replayed archive elite.
#[derive(Clone, Debug)]
pub struct PortfolioEntry {
    /// Archive niche.
    pub niche: usize,
    /// Display-grid x coordinate.
    pub grid_x: usize,
    /// Display-grid y coordinate.
    pub grid_y: usize,
    /// Number of candidates mapped to this niche.
    pub visits: u64,
    /// Stored robust mean LCOE.
    pub quality: f64,
    /// Clamped training descriptors.
    pub descriptors: [f64; 2],
    /// Representative battery-derating holdout descriptors.
    pub holdout_descriptors: [f64; 2],
    /// Exact training replay.
    pub training: OuterEvaluation,
    /// All structurally distinct holdout replays.
    pub holdout: Vec<ScenarioEvaluation>,
    /// Normalized controls retained for replay.
    pub controls: Vec<f64>,
}

/// Completed QD campaign.
#[derive(Clone, Debug)]
pub struct QdResult {
    /// Selected descriptor pair.
    pub descriptor_pair: DescriptorPair,
    /// Requested budget.
    pub requested_evaluations: usize,
    /// Actual candidate calls.
    pub actual_evaluations: usize,
    /// Invalid or infeasible candidates.
    pub invalid_evaluations: usize,
    /// Descriptor clamping events.
    pub clamped_descriptors: usize,
    /// Inner LP solves.
    pub lp_solves: usize,
    /// Cumulative simplex pivots.
    pub simplex_iterations: u64,
    /// Archive capacity.
    pub capacity: usize,
    /// Wall duration.
    pub elapsed: Duration,
    /// Occupied archive entries.
    pub entries: Vec<PortfolioEntry>,
    /// Progress trace.
    pub progress: Vec<QdProgress>,
}

#[derive(Clone)]
struct CandidateEvaluation {
    outer: OuterEvaluation,
    descriptors: [f64; 2],
    quality: f64,
}

fn clamp_pair(pair: DescriptorPair, values: [f64; 2]) -> ([f64; 2], bool) {
    let lower = pair.lower();
    let upper = pair.upper();
    let clamped = [
        values[0].clamp(lower[0], upper[0]),
        values[1].clamp(lower[1], upper[1]),
    ];
    (clamped, clamped != values)
}

fn evaluate(controls: &[f64], preset: Preset, pair: DescriptorPair) -> Option<CandidateEvaluation> {
    let outer = evaluate_training(controls, preset)?;
    if !feasible(&outer) {
        return None;
    }
    let quality = outer.mean_lcoe;
    let descriptors = pair.values(behavior(&outer));
    (quality.is_finite() && descriptors.iter().all(|value| value.is_finite())).then_some(
        CandidateEvaluation {
            outer,
            descriptors,
            quality,
        },
    )
}

struct HubQdBatch {
    preset: Preset,
    pair: DescriptorPair,
    workers: i32,
    evaluations: Arc<AtomicUsize>,
    pivots: Arc<AtomicU64>,
    invalid: Arc<AtomicUsize>,
    clamped: Arc<AtomicUsize>,
}

impl QdBatchFitness for HubQdBatch {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let preset = self.preset;
        let pair = self.pair;
        let evaluated = parallel_batch(xs, self.workers, move |x| evaluate(x, preset, pair));
        self.evaluations
            .fetch_add(evaluated.len(), Ordering::Relaxed);
        evaluated
            .into_iter()
            .map(|evaluation| match evaluation {
                Some(evaluation) => {
                    self.pivots
                        .fetch_add(evaluation.outer.simplex_iterations, Ordering::Relaxed);
                    let (descriptors, was_clamped) = clamp_pair(self.pair, evaluation.descriptors);
                    if was_clamped {
                        self.clamped.fetch_add(1, Ordering::Relaxed);
                    }
                    (evaluation.quality, descriptors.to_vec())
                }
                None => {
                    self.invalid.fetch_add(1, Ordering::Relaxed);
                    (f64::INFINITY, vec![0.0, 0.0])
                }
            })
            .collect()
    }
}

/// Build the robust sizing portfolio with MAP-Elites.
pub fn optimize_qd(config: &QdConfig) -> Result<QdResult, Box<dyn Error>> {
    if config.evaluations < 32
        || config.chunk_size < 2
        || !config.chunk_size.is_multiple_of(2)
        || !matches!(config.capacity, 60 | 120)
    {
        return Err("invalid QD budget, chunk size, or capacity".into());
    }
    let lower = config.descriptor_pair.lower();
    let upper = config.descriptor_pair.upper();
    let layout = ArchiveGrid::new(config.capacity);
    let mut rng = Rng::new(config.seed);
    let mut archive = Archive::try_new(DIMENSION, &lower, &upper, config.capacity, 0, &mut rng)?;
    archive.seed_uniform(&[0.0; DIMENSION], &[1.0; DIMENSION], &mut rng);
    let evaluations = Arc::new(AtomicUsize::new(0));
    let pivots = Arc::new(AtomicU64::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clamped = Arc::new(AtomicUsize::new(0));
    let mut evaluator = HubQdBatch {
        preset: config.preset,
        pair: config.descriptor_pair,
        workers: config.workers,
        evaluations: Arc::clone(&evaluations),
        pivots: Arc::clone(&pivots),
        invalid: Arc::clone(&invalid),
        clamped: Arc::clone(&clamped),
    };
    let seed_count = (config.evaluations / 2).max(24);
    let seeds = (0..seed_count.min(config.evaluations))
        .map(|sample| structured_candidate(config.seed, sample))
        .collect::<Vec<_>>();
    let seeded = evaluator.eval_batch(&seeds);
    for (controls, (quality, descriptors)) in seeds.iter().zip(seeded) {
        if quality.is_finite() {
            let niche = archive.index_of_niche(&descriptors);
            archive.set(niche, quality, &descriptors, controls);
        }
    }
    let remaining = config.evaluations.saturating_sub(seeds.len());
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
    }];
    if generations > 0 {
        let params = MapElitesParams {
            generations,
            chunk_size: config.chunk_size,
            use_sbx: true,
            iso_sigma: 0.03,
            line_sigma: 0.25,
            cma_generations: 0,
            ..Default::default()
        };
        let mut callback = |generation: usize, current: &Archive| {
            if generation == 1 || generation.is_multiple_of(5) || generation == generations {
                let actual = evaluations.load(Ordering::Relaxed);
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

    let battery_scenario = holdout()
        .into_iter()
        .find(|scenario| scenario.name == "battery_derated_80pct")
        .ok_or("missing battery derating scenario")?;
    let mut entries = Vec::new();
    for niche in 0..archive.capacity() {
        if !archive.ys()[niche].is_finite() {
            continue;
        }
        let controls = archive.xs()[niche].clone();
        let Some(training) = evaluate_training(&controls, config.preset) else {
            continue;
        };
        if !feasible(&training) {
            continue;
        }
        let holdout = evaluate_holdout(&controls, config.preset)?;
        let battery_holdout = holdout
            .iter()
            .find(|scenario| scenario.name == "battery_derated_80pct")
            .ok_or("battery derating replay missing")?;
        let mut effective_design = training.design.clone();
        effective_design.capacities = battery_scenario.capacities(effective_design.capacities);
        let raw_descriptors = config.descriptor_pair.values(behavior(&training));
        let (descriptors, _) = clamp_pair(config.descriptor_pair, raw_descriptors);
        let raw_holdout = config
            .descriptor_pair
            .values(behavior_for_scenario(&effective_design, battery_holdout));
        let (holdout_descriptors, _) = clamp_pair(config.descriptor_pair, raw_holdout);
        let (grid_x, grid_y) = layout
            .coordinate(niche)
            .ok_or("archive returned an out-of-range niche")?;
        entries.push(PortfolioEntry {
            niche,
            grid_x,
            grid_y,
            visits: archive.counts()[niche],
            quality: training.mean_lcoe,
            descriptors,
            holdout_descriptors,
            training,
            holdout,
            controls,
        });
    }
    let actual_evaluations = evaluations.load(Ordering::Relaxed);
    Ok(QdResult {
        descriptor_pair: config.descriptor_pair,
        requested_evaluations: config.evaluations,
        actual_evaluations,
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clamped_descriptors: clamped.load(Ordering::Relaxed),
        lp_solves: 5 * actual_evaluations,
        simplex_iterations: pivots.load(Ordering::Relaxed),
        capacity: config.capacity,
        elapsed: started.elapsed(),
        entries,
        progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_archive_elites_replay_exactly() {
        let result = optimize_qd(&QdConfig {
            preset: Preset::Smoke,
            descriptor_pair: DescriptorPair::D1,
            evaluations: 96,
            capacity: 60,
            chunk_size: 16,
            workers: 2,
            seed: 42,
        })
        .unwrap();
        assert!(result.entries.len() >= 3, "{}", result.entries.len());
        let layout = ArchiveGrid::new(result.capacity);
        for entry in &result.entries {
            let replay = evaluate_training(&entry.controls, Preset::Smoke).unwrap();
            assert!((replay.mean_lcoe - entry.quality).abs() < 1.0e-12);
            let lower = result.descriptor_pair.lower();
            let upper = result.descriptor_pair.upper();
            assert!(
                entry
                    .descriptors
                    .iter()
                    .enumerate()
                    .all(|(axis, value)| (lower[axis]..=upper[axis]).contains(value))
            );
            assert_eq!(
                layout.coordinate(entry.niche),
                Some((entry.grid_x, entry.grid_y))
            );
            assert_eq!(
                layout.niche(
                    entry.descriptors,
                    result.descriptor_pair.lower(),
                    result.descriptor_pair.upper()
                ),
                Some(entry.niche)
            );
        }
        assert!(result.simplex_iterations > 0);
    }
}
