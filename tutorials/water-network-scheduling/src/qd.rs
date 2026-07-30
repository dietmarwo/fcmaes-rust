//! Quality-diversity strategy catalogue.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use epanet_rs::model::network::Network;
use fcmaes_core::{
    Archive, MapElitesParams, QdBatchFitness, Rng, map_elites_batch, parallel_batch,
};

use crate::DIMENSION;
use crate::decode::seed_controls;
use crate::evaluate::{RobustEvaluation, evaluate_training};
use crate::pilot::{DESCRIPTOR_LOWER, DESCRIPTOR_UPPER, QdDecision};

/// MAP-Elites protocol.
#[derive(Clone, Debug)]
pub struct QdConfig {
    pub evaluations: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    pub workers: i32,
    pub seed: u64,
}

/// Replayed occupied niche.
#[derive(Clone, Debug)]
pub struct RepertoireEntry {
    pub niche: usize,
    pub visits: u64,
    pub quality: f64,
    pub descriptors: [f64; 2],
    pub controls: Vec<f64>,
    pub training: RobustEvaluation,
}

/// QD campaign.
#[derive(Clone, Debug)]
pub struct QdResult {
    pub requested_evaluations: usize,
    pub actual_evaluations: usize,
    pub invalid_evaluations: usize,
    pub clamped_evaluations: usize,
    pub capacity: usize,
    pub elapsed: Duration,
    pub entries: Vec<RepertoireEntry>,
}

struct HydraulicBatch {
    network: Arc<Network>,
    workers: i32,
    decision: QdDecision,
    calls: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    clamped: Arc<AtomicUsize>,
}

fn descriptors(evaluation: &RobustEvaluation, decision: QdDecision) -> [f64; 2] {
    match decision {
        QdDecision::AcceptedD1 => evaluation.descriptors,
        QdDecision::AcceptedD2 => {
            let nominal = &evaluation.scenarios[0];
            [
                ((nominal.max_pressure_m - nominal.min_pressure_m) / 50.0).clamp(0.30, 0.35),
                evaluation.descriptors[1],
            ]
        }
        QdDecision::Rejected => [0.0, 0.0],
    }
}

fn descriptor_bounds(decision: QdDecision) -> ([f64; 2], [f64; 2]) {
    match decision {
        QdDecision::AcceptedD1 => (DESCRIPTOR_LOWER, DESCRIPTOR_UPPER),
        QdDecision::AcceptedD2 => ([0.30, 0.08], [0.35, 0.23]),
        QdDecision::Rejected => ([0.0; 2], [1.0; 2]),
    }
}

impl QdBatchFitness for HydraulicBatch {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let network = Arc::clone(&self.network);
        let evaluated = parallel_batch(xs, self.workers, move |controls| {
            evaluate_training(controls, &network).ok()
        });
        self.calls.fetch_add(evaluated.len(), Ordering::Relaxed);
        evaluated
            .into_iter()
            .map(|result| match result {
                Some(evaluation) if evaluation.feasible => (evaluation.operating_cost, {
                    let raw = descriptors(&evaluation, self.decision);
                    let (lower, upper) = descriptor_bounds(self.decision);
                    let clamped = [
                        raw[0].clamp(lower[0], upper[0]),
                        raw[1].clamp(lower[1], upper[1]),
                    ];
                    if clamped != raw {
                        self.clamped.fetch_add(1, Ordering::Relaxed);
                    }
                    clamped.to_vec()
                }),
                _ => {
                    self.invalid.fetch_add(1, Ordering::Relaxed);
                    (f64::INFINITY, vec![0.0, 0.0])
                }
            })
            .collect()
    }
}

/// Build an operating-strategy catalogue after the pilot gate.
pub fn optimize(
    network: &Network,
    decision: QdDecision,
    config: &QdConfig,
) -> Result<QdResult, Box<dyn Error>> {
    if decision == QdDecision::Rejected {
        return Err("descriptor pilot rejected QD".into());
    }
    let mut rng = Rng::new(config.seed);
    let (lower, upper) = descriptor_bounds(decision);
    let mut archive = Archive::try_new(DIMENSION, &lower, &upper, config.capacity, 0, &mut rng)?;
    archive.seed_uniform(&[0.0; DIMENSION], &[1.0; DIMENSION], &mut rng);
    let calls = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clamped = Arc::new(AtomicUsize::new(0));
    let mut evaluator = HydraulicBatch {
        network: Arc::new(network.clone()),
        workers: config.workers,
        decision,
        calls: Arc::clone(&calls),
        invalid: Arc::clone(&invalid),
        clamped: Arc::clone(&clamped),
    };
    let mut seed_rng = Rng::new(config.seed.wrapping_add(91));
    let witness = seed_controls();
    let seed_count = config.evaluations.min(64);
    let seed_x = (0..seed_count)
        .map(|index| {
            if index == 0 {
                witness.clone()
            } else {
                witness
                    .iter()
                    .map(|value| (value + 0.55 * (seed_rng.uniform01() - 0.5)).clamp(0.0, 1.0))
                    .collect()
            }
        })
        .collect::<Vec<Vec<f64>>>();
    let seed_y = evaluator.eval_batch(&seed_x);
    for (controls, (quality, descriptor)) in seed_x.iter().zip(seed_y) {
        if quality.is_finite() {
            let niche = archive.index_of_niche(&descriptor);
            archive.set(niche, quality, &descriptor, controls);
        }
    }
    if archive.occupied() == 0 {
        return Err("no robust-feasible QD seeds".into());
    }
    let started = Instant::now();
    let remaining = config.evaluations.saturating_sub(seed_count);
    if remaining > 0 {
        map_elites_batch(
            &mut archive,
            &mut evaluator,
            &[0.0; DIMENSION],
            &[1.0; DIMENSION],
            &MapElitesParams {
                generations: remaining.div_ceil(config.chunk_size),
                chunk_size: config.chunk_size,
                use_sbx: true,
                iso_sigma: 0.03,
                line_sigma: 0.2,
                ..Default::default()
            },
            &mut rng,
        )?;
    }
    let mut entries = Vec::new();
    for niche in 0..archive.capacity() {
        if !archive.ys()[niche].is_finite() {
            continue;
        }
        let controls = archive.xs()[niche].clone();
        let Ok(training) = evaluate_training(&controls, network) else {
            continue;
        };
        entries.push(RepertoireEntry {
            niche,
            visits: archive.counts()[niche],
            quality: training.operating_cost,
            descriptors: {
                let raw = descriptors(&training, decision);
                [
                    raw[0].clamp(lower[0], upper[0]),
                    raw[1].clamp(lower[1], upper[1]),
                ]
            },
            controls,
            training,
        });
    }
    Ok(QdResult {
        requested_evaluations: config.evaluations,
        actual_evaluations: calls.load(Ordering::Relaxed),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clamped_evaluations: clamped.load(Ordering::Relaxed),
        capacity: config.capacity,
        elapsed: started.elapsed(),
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network;

    #[test]
    fn accepted_gate_builds_replayable_archive() {
        let network = network::load().unwrap();
        let result = optimize(
            &network,
            QdDecision::AcceptedD1,
            &QdConfig {
                evaluations: 64,
                capacity: 20,
                chunk_size: 16,
                workers: 2,
                seed: 42,
            },
        )
        .unwrap();
        assert!(!result.entries.is_empty());
        assert!(
            result
                .entries
                .iter()
                .all(|entry| entry.training.feasible && entry.quality.is_finite())
        );
        assert!(result.entries.iter().all(|entry| {
            entry.descriptors[0] >= DESCRIPTOR_LOWER[0]
                && entry.descriptors[0] <= DESCRIPTOR_UPPER[0]
                && entry.descriptors[1] >= DESCRIPTOR_LOWER[1]
                && entry.descriptors[1] <= DESCRIPTOR_UPPER[1]
        }));
    }
}
