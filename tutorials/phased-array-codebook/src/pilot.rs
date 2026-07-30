//! Pre-registered descriptor pilot.

use fcmaes_core::Rng;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::archive_grid::ArchiveGrid;
use crate::decode::decode_with_activation;
use crate::scenarios::{
    HoldoutContext, evaluate_holdout, evaluate_training, representative_holdout_metrics,
};
use crate::so::{BeamContext, ELEMENTS, ROBUST_PSLL_LIMIT_DB, analytic_seed};

/// Primary QD descriptor lower bounds.
pub const DESCRIPTOR_LOWER: [f64; 2] = [-52.0, 6.0];
/// Primary QD descriptor upper bounds.
pub const DESCRIPTOR_UPPER: [f64; 2] = [52.0, 14.0];
/// Publication archive capacity used by the descriptor gate.
pub const PUBLICATION_CAPACITY: usize = 120;
/// Secondary `(peak direction, taper efficiency)` bounds.
pub const D2_DESCRIPTOR_LOWER: [f64; 2] = [-52.0, 0.65];
/// Secondary `(peak direction, taper efficiency)` bounds.
pub const D2_DESCRIPTOR_UPPER: [f64; 2] = [52.0, 1.0];

/// Pre-registered QD verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QdDecision {
    /// D1 clears every gate.
    Accepted,
    /// D1 fails but the efficiency fallback remains useful.
    PrimarySecondary,
    /// No descriptor pair clears the gate.
    Rejected,
}

impl QdDecision {
    /// Stable human- and machine-facing label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::PrimarySecondary => "primary-secondary",
            Self::Rejected => "rejected",
        }
    }
}

/// One feasible descriptor observation.
#[derive(Clone, Debug)]
pub struct PilotRow {
    /// Root seed arm.
    pub seed: u64,
    /// Sample within the arm.
    pub sample: usize,
    /// Measured training descriptors `[peak angle, HPBW]`.
    pub descriptors: [f64; 2],
    /// Representative holdout descriptors.
    pub holdout_descriptors: [f64; 2],
    /// Taper efficiency.
    pub taper_efficiency: f64,
    /// Taper efficiency under the representative holdout perturbation.
    pub holdout_taper_efficiency: f64,
    /// Exact active-element count.
    pub active_count: usize,
    /// Worst training PSLL.
    pub worst_psll_db: f64,
    /// Worst holdout PSLL.
    pub holdout_worst_psll_db: f64,
}

/// Attempt and feasibility accounting for one independent root-seed arm.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PilotSeedSummary {
    /// Root seed.
    pub seed: u64,
    /// Attempted candidates.
    pub attempted: usize,
    /// Candidates clearing the frozen robust feasibility filter.
    pub feasible: usize,
    /// Feasible fraction.
    pub feasible_fraction: f64,
}

/// Pilot decision and all pre-registered diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PilotSummary {
    /// Wall time for the complete structured pilot.
    pub elapsed_seconds: f64,
    /// Mixed analytic/uniform candidate designs attempted across all seed arms.
    pub attempted_candidates: usize,
    /// Feasible rows retained.
    pub feasible_candidates: usize,
    /// Overall feasible fraction.
    pub feasible_fraction: f64,
    /// Per-seed feasibility accounting.
    pub seed_summaries: Vec<PilotSeedSummary>,
    /// Spearman correlation of D1.
    pub d1_rank_correlation: f64,
    /// Spearman correlation of D2.
    pub d2_rank_correlation: f64,
    /// Spearman correlation of decision-led D3.
    pub d3_rank_correlation: f64,
    /// Intended-grid coverage.
    pub coverage: f64,
    /// D2 archive coverage.
    pub d2_coverage: f64,
    /// Lower/upper clipping fraction for peak direction.
    pub peak_clipping: f64,
    /// Lower/upper clipping fraction for HPBW.
    pub hpbw_clipping: f64,
    /// Lower/upper clipping fraction for taper efficiency.
    pub efficiency_clipping: f64,
    /// Same-niche retention under representative holdout.
    pub holdout_niche_retention: f64,
    /// Same-niche D2 retention under representative holdout.
    pub d2_holdout_niche_retention: f64,
    /// Retention on the same factorization at a coarser 30-niche capacity.
    pub coarse_holdout_niche_retention: f64,
    /// Pre-registered decision.
    pub decision: QdDecision,
    /// Human-readable rule outcome.
    pub reason: String,
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut result = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[start]].to_bits() == values[order[end]].to_bits() {
            end += 1;
        }
        let rank = 0.5 * (start + end - 1) as f64;
        for index in &order[start..end] {
            result[*index] = rank;
        }
        start = end;
    }
    result
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    if left.len() < 2 || left.len() != right.len() {
        return f64::NAN;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let covariance = left
        .iter()
        .zip(right)
        .map(|(l, r)| (l - left_mean) * (r - right_mean))
        .sum::<f64>();
    let left_scale = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>();
    let right_scale = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>();
    if left_scale <= 0.0 || right_scale <= 0.0 {
        f64::NAN
    } else {
        covariance / (left_scale * right_scale).sqrt()
    }
}

fn spearman(left: &[f64], right: &[f64]) -> f64 {
    pearson(&ranks(left), &ranks(right))
}

fn activation_controls(
    base: &[f64],
    active_count: usize,
    rng: &mut Rng,
    random_keys: bool,
) -> Vec<f64> {
    let mut controls = Vec::with_capacity(3 * ELEMENTS);
    controls.extend_from_slice(&base[..ELEMENTS]);
    controls.extend_from_slice(&base[ELEMENTS..]);
    let center = (ELEMENTS - 1) as f64 / 2.0;
    controls.extend((0..ELEMENTS).map(|index| {
        if random_keys {
            rng.uniform01()
        } else {
            let distance = (index as f64 - center).abs() / center;
            (distance + 0.08 * (rng.uniform01() - 0.5)).clamp(0.0, 1.0)
        }
    }));
    debug_assert_eq!(controls.len(), 3 * ELEMENTS);
    debug_assert!(active_count <= ELEMENTS);
    controls
}

fn jitter_hardware_controls(base: &mut [f64], rng: &mut Rng) {
    for value in &mut base[..ELEMENTS] {
        *value = (*value + 2.0 * (rng.uniform01() - 0.5) / 64.0).clamp(0.0, 1.0);
    }
    for value in &mut base[ELEMENTS..] {
        *value = (*value + 2.0 * (rng.uniform01() - 0.5) / 32.0).clamp(0.0, 1.0);
    }
}

/// Execute the descriptor pilot over three deterministic seed arms.
#[must_use]
pub fn run_pilot(samples_per_seed: usize, points: usize) -> (Vec<PilotRow>, PilotSummary) {
    let started = Instant::now();
    let context = BeamContext::stage_a(points);
    let holdout_beam = BeamContext::stage_a(2 * points - 1);
    let holdout_context = HoldoutContext::new(holdout_beam.array, holdout_beam.grid)
        .expect("stage A has 16 elements");
    let mut rows = Vec::new();
    let mut seed_summaries = Vec::new();
    for (seed_index, seed) in [42_u64, 4242, 424_242].into_iter().enumerate() {
        let rows_before = rows.len();
        let mut rng = Rng::new(seed);
        for sample in 0..samples_per_seed {
            let steer_index = sample % 31;
            // Five is coprime to eight, so even a short smoke run exercises
            // the complete taper range instead of only its untapered prefix.
            let taper_index = (sample * 5) % 8;
            let requested_deg = (-50.0
                + 100.0 * steer_index as f64 / 30.0
                + (rng.uniform01() - 0.5) * 100.0 / 30.0)
                .clamp(-50.0, 50.0);
            let taper = (taper_index as f64 / 7.0 + (rng.uniform01() - 0.5) / 7.0).clamp(0.0, 1.0);
            let active_count = if sample.is_multiple_of(3) {
                10 + (sample / 3 + 2 * seed_index) % 7
            } else {
                ELEMENTS
            };
            // Each seed arm mixes a perturbed analytic manifold with 20%
            // uniform controls. This samples what MAP-Elites can actually
            // emit while retaining enough feasible points to test descriptors.
            let uniform_sample = (sample + seed_index).is_multiple_of(5);
            let base = if uniform_sample {
                (0..2 * ELEMENTS)
                    .map(|_| rng.uniform01())
                    .collect::<Vec<_>>()
            } else {
                let mut seeded = analytic_seed(requested_deg, taper);
                jitter_hardware_controls(&mut seeded, &mut rng);
                seeded
            };
            debug_assert_eq!(base.len(), 2 * ELEMENTS);
            let controls = activation_controls(&base, active_count, &mut rng, uniform_sample);
            let Ok(excitation) =
                decode_with_activation(&controls, ELEMENTS, active_count, &context.quantization)
            else {
                continue;
            };
            let Ok(training) =
                evaluate_training(&excitation.weights, &context.steering, &context.grid)
            else {
                continue;
            };
            if training.nominal.degenerate
                || training.worst_psll_db > ROBUST_PSLL_LIMIT_DB
                || !training.nominal.hpbw_deg.is_finite()
            {
                continue;
            }
            let Ok(holdout) = evaluate_holdout(&excitation.weights, &holdout_context) else {
                continue;
            };
            let Ok(holdout_descriptor) =
                representative_holdout_metrics(&excitation.weights, &holdout_context)
            else {
                continue;
            };
            if holdout_descriptor.degenerate || !holdout_descriptor.hpbw_deg.is_finite() {
                continue;
            }
            rows.push(PilotRow {
                seed,
                sample,
                descriptors: [training.nominal.peak_theta_deg, training.nominal.hpbw_deg],
                holdout_descriptors: [
                    holdout_descriptor.peak_theta_deg,
                    holdout_descriptor.hpbw_deg,
                ],
                taper_efficiency: training.nominal.taper_efficiency,
                holdout_taper_efficiency: holdout_descriptor.taper_efficiency,
                active_count,
                worst_psll_db: training.worst_psll_db,
                holdout_worst_psll_db: holdout.worst_psll_db,
            });
        }
        let feasible = rows.len() - rows_before;
        seed_summaries.push(PilotSeedSummary {
            seed,
            attempted: samples_per_seed,
            feasible,
            feasible_fraction: feasible as f64 / samples_per_seed.max(1) as f64,
        });
    }
    let peak = rows
        .iter()
        .map(|row| row.descriptors[0])
        .collect::<Vec<_>>();
    let width = rows
        .iter()
        .map(|row| row.descriptors[1])
        .collect::<Vec<_>>();
    let efficiency = rows
        .iter()
        .map(|row| row.taper_efficiency)
        .collect::<Vec<_>>();
    let active = rows
        .iter()
        .map(|row| row.active_count as f64)
        .collect::<Vec<_>>();
    let d1_rank_correlation = spearman(&peak, &width);
    let d2_rank_correlation = spearman(&peak, &efficiency);
    let d3_rank_correlation = spearman(&active, &width);
    let layout = ArchiveGrid::new(PUBLICATION_CAPACITY);
    let occupied = rows
        .iter()
        .filter_map(|row| layout.niche(row.descriptors, DESCRIPTOR_LOWER, DESCRIPTOR_UPPER))
        .collect::<std::collections::HashSet<_>>()
        .len();
    let coverage = occupied as f64 / PUBLICATION_CAPACITY as f64;
    let d2_occupied = rows
        .iter()
        .filter_map(|row| {
            layout.niche(
                [row.descriptors[0], row.taper_efficiency],
                D2_DESCRIPTOR_LOWER,
                D2_DESCRIPTOR_UPPER,
            )
        })
        .collect::<std::collections::HashSet<_>>()
        .len();
    let d2_coverage = d2_occupied as f64 / PUBLICATION_CAPACITY as f64;
    let clipping = |index: usize, lower: [f64; 2], upper: [f64; 2], d2: bool| {
        rows.iter()
            .filter(|row| {
                let descriptors = if d2 {
                    [row.descriptors[0], row.taper_efficiency]
                } else {
                    row.descriptors
                };
                !(lower[index]..=upper[index]).contains(&descriptors[index])
            })
            .count() as f64
            / rows.len().max(1) as f64
    };
    let peak_clipping = clipping(0, DESCRIPTOR_LOWER, DESCRIPTOR_UPPER, false);
    let hpbw_clipping = clipping(1, DESCRIPTOR_LOWER, DESCRIPTOR_UPPER, false);
    let efficiency_clipping = clipping(1, D2_DESCRIPTOR_LOWER, D2_DESCRIPTOR_UPPER, true);
    let retention = |capacity, lower, upper, d2| {
        let grid = ArchiveGrid::new(capacity);
        let comparable = rows
            .iter()
            .filter_map(|row| {
                let training = if d2 {
                    [row.descriptors[0], row.taper_efficiency]
                } else {
                    row.descriptors
                };
                let holdout = if d2 {
                    [row.holdout_descriptors[0], row.holdout_taper_efficiency]
                } else {
                    row.holdout_descriptors
                };
                Some((
                    grid.niche(training, lower, upper)?,
                    grid.niche(holdout, lower, upper)?,
                ))
            })
            .collect::<Vec<_>>();
        comparable
            .iter()
            .filter(|(training, holdout)| training == holdout)
            .count() as f64
            / comparable.len().max(1) as f64
    };
    let holdout_niche_retention = retention(
        PUBLICATION_CAPACITY,
        DESCRIPTOR_LOWER,
        DESCRIPTOR_UPPER,
        false,
    );
    let d2_holdout_niche_retention = retention(
        PUBLICATION_CAPACITY,
        D2_DESCRIPTOR_LOWER,
        D2_DESCRIPTOR_UPPER,
        true,
    );
    let coarse_holdout_niche_retention = retention(30, DESCRIPTOR_LOWER, DESCRIPTOR_UPPER, false);
    let d1_passes = d1_rank_correlation.abs() < 0.7
        && peak_clipping < 0.1
        && hpbw_clipping < 0.1
        && coverage > 0.4
        && holdout_niche_retention > 0.6;
    let d2_passes = d2_rank_correlation.abs() < 0.7
        && peak_clipping < 0.1
        && efficiency_clipping < 0.1
        && d2_coverage > 0.4
        && d2_holdout_niche_retention > 0.6;
    let (decision, reason) = if d1_passes {
        (
            QdDecision::Accepted,
            "D1 clears correlation, clipping, coverage, and holdout-retention gates".to_owned(),
        )
    } else if d2_passes {
        (
            QdDecision::PrimarySecondary,
            "D1 misses a pre-registered gate; D2 clears the same correlation, clipping, coverage, and holdout-retention gates"
                .to_owned(),
        )
    } else {
        (
            QdDecision::Rejected,
            "neither D1 nor D2 clears the pre-registered descriptor gates".to_owned(),
        )
    };
    let summary = PilotSummary {
        elapsed_seconds: started.elapsed().as_secs_f64(),
        attempted_candidates: samples_per_seed.saturating_mul(3),
        feasible_candidates: rows.len(),
        feasible_fraction: rows.len() as f64 / samples_per_seed.saturating_mul(3).max(1) as f64,
        seed_summaries,
        d1_rank_correlation,
        d2_rank_correlation,
        d3_rank_correlation,
        coverage,
        d2_coverage,
        peak_clipping,
        hpbw_clipping,
        efficiency_clipping,
        holdout_niche_retention,
        d2_holdout_niche_retention,
        coarse_holdout_niche_retention,
        decision,
        reason,
    };
    (rows, summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pilot_is_finite_and_reports_the_pre_registered_rules() {
        let (rows, summary) = run_pilot(36, 361);
        assert!(rows.len() > 8);
        assert!(summary.d1_rank_correlation.is_finite());
        assert!(summary.coverage > 0.05);
        assert!((0.0..=1.0).contains(&summary.holdout_niche_retention));
        assert_eq!(summary.seed_summaries.len(), 3);
        assert!(
            summary
                .seed_summaries
                .iter()
                .all(|seed| (0.0..=1.0).contains(&seed.feasible_fraction))
        );
    }
}
