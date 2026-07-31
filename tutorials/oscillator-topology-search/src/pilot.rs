//! Fresh period/amplitude descriptor gate for topology search.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::config::TRAINING_SEEDS;
use crate::outer::Campaign;
use crate::score;

pub const GRID_SIDE: usize = 12;
pub const COARSE_GRID_SIDE: usize = 6;
const PERIOD_RANGE: (f64, f64) = (8.0, 64.0);
const AMPLITUDE_RANGE: (f64, f64) = (0.0, 200.0);
const REQUIRED_ARM_COUNT: usize = 3;

/// Recorded pre-QD gate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DescriptorPilot {
    pub schema_version: u32,
    pub status: String,
    pub descriptor_pair: [String; 2],
    pub structural_control: String,
    pub candidate_count: usize,
    pub observed_arm_count: usize,
    pub required_arm_count: usize,
    pub arm_limit: Option<String>,
    pub training_replications: usize,
    pub validation_replications: usize,
    pub high_replication_training_replications: usize,
    pub grid_side: usize,
    pub coarse_grid_side: usize,
    pub observed_period_range: [f64; 2],
    pub observed_amplitude_range: [f64; 2],
    pub arm_coverage: BTreeMap<String, f64>,
    pub minimum_arm_coverage: f64,
    pub period_below_fraction: f64,
    pub period_above_fraction: f64,
    pub amplitude_below_fraction: f64,
    pub amplitude_above_fraction: f64,
    pub out_of_range_fraction: f64,
    pub descriptor_correlation: f64,
    pub holdout_niche_retention: f64,
    pub coarse_holdout_niche_retention: f64,
    pub high_replication_holdout_niche_retention: f64,
    pub rejection_reasons: Vec<String>,
}

/// Native row/column index; `None` marks out-of-range descriptors.
pub fn cell(period: f64, amplitude: f64) -> Option<usize> {
    cell_at_resolution(period, amplitude, GRID_SIDE)
}

fn cell_at_resolution(period: f64, amplitude: f64, grid_side: usize) -> Option<usize> {
    assert!(grid_side > 0);
    if !(PERIOD_RANGE.0..=PERIOD_RANGE.1).contains(&period)
        || !(AMPLITUDE_RANGE.0..=AMPLITUDE_RANGE.1).contains(&amplitude)
    {
        return None;
    }
    let x = (((period - PERIOD_RANGE.0) / (PERIOD_RANGE.1 - PERIOD_RANGE.0) * grid_side as f64)
        .floor() as usize)
        .min(grid_side - 1);
    let y = (((amplitude - AMPLITUDE_RANGE.0) / (AMPLITUDE_RANGE.1 - AMPLITUDE_RANGE.0)
        * grid_side as f64)
        .floor() as usize)
        .min(grid_side - 1);
    Some(y * grid_side + x)
}

fn correlation(pairs: &[(f64, f64)]) -> f64 {
    if pairs.len() < 3 {
        return 1.0;
    }
    let mean_x = pairs.iter().map(|pair| pair.0).sum::<f64>() / pairs.len() as f64;
    let mean_y = pairs.iter().map(|pair| pair.1).sum::<f64>() / pairs.len() as f64;
    let numerator = pairs
        .iter()
        .map(|pair| (pair.0 - mean_x) * (pair.1 - mean_y))
        .sum::<f64>();
    let left = pairs
        .iter()
        .map(|pair| (pair.0 - mean_x).powi(2))
        .sum::<f64>();
    let right = pairs
        .iter()
        .map(|pair| (pair.1 - mean_y).powi(2))
        .sum::<f64>();
    if left <= 0.0 || right <= 0.0 {
        1.0
    } else {
        numerator / (left * right).sqrt()
    }
}

fn observed_range(values: impl Iterator<Item = f64>) -> [f64; 2] {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for value in values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    if minimum.is_finite() && maximum.is_finite() {
        [minimum, maximum]
    } else {
        [0.0, 0.0]
    }
}

/// Evaluate the pre-registered gate on the available independent arms.
pub fn evaluate(campaigns: &[&Campaign]) -> DescriptorPilot {
    let candidates: Vec<_> = campaigns
        .iter()
        .flat_map(|campaign| &campaign.archive.candidates)
        .collect();
    let mut arm_coverage = BTreeMap::new();
    for campaign in campaigns {
        let cells: BTreeSet<_> = campaign
            .archive
            .candidates
            .iter()
            .filter_map(|candidate| cell(candidate.training.period, candidate.training.amplitude))
            .collect();
        arm_coverage.insert(
            campaign.strategy.label().to_owned(),
            cells.len() as f64 / (GRID_SIDE * GRID_SIDE) as f64,
        );
    }
    let minimum_arm_coverage = arm_coverage
        .values()
        .copied()
        .reduce(f64::min)
        .unwrap_or(0.0);
    let period_below = candidates
        .iter()
        .filter(|candidate| candidate.training.period < PERIOD_RANGE.0)
        .count();
    let period_above = candidates
        .iter()
        .filter(|candidate| candidate.training.period > PERIOD_RANGE.1)
        .count();
    let amplitude_below = candidates
        .iter()
        .filter(|candidate| candidate.training.amplitude < AMPLITUDE_RANGE.0)
        .count();
    let amplitude_above = candidates
        .iter()
        .filter(|candidate| candidate.training.amplitude > AMPLITUDE_RANGE.1)
        .count();
    let out_of_range = candidates
        .iter()
        .filter(|candidate| cell(candidate.training.period, candidate.training.amplitude).is_none())
        .count();
    let descriptor_pairs: Vec<_> = candidates
        .iter()
        .map(|candidate| (candidate.training.period, candidate.training.amplitude))
        .collect();
    let retained = candidates
        .iter()
        .filter(|candidate| {
            let training_cell = cell(candidate.training.period, candidate.training.amplitude);
            training_cell.is_some()
                && training_cell
                    == cell(candidate.validation.period, candidate.validation.amplitude)
        })
        .count();
    let retained_coarse = candidates
        .iter()
        .filter(|candidate| {
            let training_cell = cell_at_resolution(
                candidate.training.period,
                candidate.training.amplitude,
                COARSE_GRID_SIDE,
            );
            training_cell.is_some()
                && training_cell
                    == cell_at_resolution(
                        candidate.validation.period,
                        candidate.validation.amplitude,
                        COARSE_GRID_SIDE,
                    )
        })
        .count();
    let retained_high_replication = candidates
        .iter()
        .filter(|candidate| {
            let training = score::training(
                &candidate.topology,
                &candidate.parameters,
                TRAINING_SEEDS.len(),
            );
            let training_cell = cell(training.period, training.amplitude);
            training_cell.is_some()
                && training_cell
                    == cell(candidate.validation.period, candidate.validation.amplitude)
        })
        .count();
    let candidate_count = candidates.len();
    let denominator = candidate_count.max(1) as f64;
    let period_below_fraction = period_below as f64 / denominator;
    let period_above_fraction = period_above as f64 / denominator;
    let amplitude_below_fraction = amplitude_below as f64 / denominator;
    let amplitude_above_fraction = amplitude_above as f64 / denominator;
    let out_of_range_fraction = out_of_range as f64 / denominator;
    let holdout_niche_retention = retained as f64 / candidate_count.max(1) as f64;
    let coarse_holdout_niche_retention = retained_coarse as f64 / candidate_count.max(1) as f64;
    let high_replication_holdout_niche_retention =
        retained_high_replication as f64 / candidate_count.max(1) as f64;
    let descriptor_correlation = correlation(&descriptor_pairs);
    let observed_arm_count = campaigns.len();
    let arm_limit = (observed_arm_count < REQUIRED_ARM_COUNT).then(|| {
        format!(
            "{observed_arm_count} of {REQUIRED_ARM_COUNT} required arms; publication agent is not-run"
        )
    });
    let mut rejection_reasons = Vec::new();
    if candidate_count < 24 {
        rejection_reasons.push("fewer than 24 independent control candidates".to_owned());
    }
    if observed_arm_count < REQUIRED_ARM_COUNT {
        rejection_reasons.push("fewer than three descriptor-pilot arms".to_owned());
    }
    if minimum_arm_coverage < 0.05 {
        rejection_reasons.push("minimum per-arm native-grid coverage is below 5%".to_owned());
    }
    if out_of_range_fraction > 0.05 {
        rejection_reasons.push("descriptor out-of-range fraction exceeds 5%".to_owned());
    }
    if descriptor_correlation.abs() > 0.90 {
        rejection_reasons.push("absolute descriptor correlation exceeds 0.90".to_owned());
    }
    if holdout_niche_retention < 0.25 {
        rejection_reasons.push("same-niche holdout retention is below 25%".to_owned());
    }
    DescriptorPilot {
        schema_version: 1,
        status: if rejection_reasons.is_empty() {
            "accepted"
        } else {
            "rejected"
        }
        .to_owned(),
        descriptor_pair: [
            "measured_period".to_owned(),
            "measured_amplitude".to_owned(),
        ],
        structural_control: "E-A-I-S-motif is decision-derived and never a QD descriptor"
            .to_owned(),
        candidate_count,
        observed_arm_count,
        required_arm_count: REQUIRED_ARM_COUNT,
        arm_limit,
        training_replications: candidates
            .first()
            .map_or(0, |candidate| candidate.training.replicates.len()),
        validation_replications: candidates
            .first()
            .map_or(0, |candidate| candidate.validation.replicates.len()),
        high_replication_training_replications: TRAINING_SEEDS.len(),
        grid_side: GRID_SIDE,
        coarse_grid_side: COARSE_GRID_SIDE,
        observed_period_range: observed_range(
            candidates.iter().map(|candidate| candidate.training.period),
        ),
        observed_amplitude_range: observed_range(
            candidates
                .iter()
                .map(|candidate| candidate.training.amplitude),
        ),
        arm_coverage,
        minimum_arm_coverage,
        period_below_fraction,
        period_above_fraction,
        amplitude_below_fraction,
        amplitude_above_fraction,
        out_of_range_fraction,
        descriptor_correlation,
        holdout_niche_retention,
        coarse_holdout_niche_retention,
        high_replication_holdout_niche_retention,
        rejection_reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_bounds_are_native_and_not_clamped() {
        assert_eq!(cell(8.0, 0.0), Some(0));
        assert_eq!(cell(64.0, 200.0), Some(GRID_SIDE * GRID_SIDE - 1));
        assert_eq!(cell(7.9, 20.0), None);
        assert_eq!(cell(20.0, 201.0), None);
        assert_eq!(
            cell_at_resolution(64.0, 200.0, COARSE_GRID_SIDE),
            Some(COARSE_GRID_SIDE * COARSE_GRID_SIDE - 1)
        );
    }
}
