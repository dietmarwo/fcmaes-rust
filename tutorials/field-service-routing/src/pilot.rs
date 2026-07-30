//! Pre-registered descriptor pilot.

use std::collections::HashSet;

use fcmaes_core::Rng;
use serde::Serialize;

use crate::archive_grid::ArchiveGrid;
use crate::instance::Instance;
use crate::scenarios::{evaluate_holdout, evaluate_training, robust_seed_controls};

/// D1 descriptor lower bounds.
pub const DESCRIPTOR_LOWER: [f64; 2] = [3.5, 0.0];
/// D1 descriptor upper bounds.
pub const DESCRIPTOR_UPPER: [f64; 2] = [8.5, 1.0];

/// Candidate-generator stratum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleSource {
    /// Perturbation around the constructed robust witness.
    Local,
    /// Uniform sample over the complete normalized decision box.
    Uniform,
}

impl SampleSource {
    /// Stable artifact label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Uniform => "uniform",
        }
    }
}

/// One training-feasible pilot observation.
#[derive(Clone, Debug)]
pub struct PilotRow {
    /// Deterministic seed arm.
    pub seed: u64,
    /// Sample number within the seed arm.
    pub sample: usize,
    /// Candidate-generator stratum.
    pub source: SampleSource,
    /// Training vehicles used.
    pub vehicles: f64,
    /// Training route-distance CV.
    pub imbalance: f64,
    /// Training mean waiting per used route.
    pub mean_waiting_s: f64,
    /// Training total nominal distance.
    pub distance_km: f64,
    /// Holdout vehicles used.
    pub holdout_vehicles: f64,
    /// Holdout route-distance CV.
    pub holdout_imbalance: f64,
    /// Holdout mean waiting per used route.
    pub holdout_mean_waiting_s: f64,
    /// Holdout total distance.
    pub holdout_distance_km: f64,
    /// Remained hard-feasible on all holdouts.
    pub holdout_feasible: bool,
}

#[derive(Clone, Copy)]
enum Pair {
    D1,
    D2,
    D3,
}

impl Pair {
    const fn bounds(self) -> [[f64; 2]; 2] {
        match self {
            Self::D1 => [
                [DESCRIPTOR_LOWER[0], DESCRIPTOR_UPPER[0]],
                [DESCRIPTOR_LOWER[1], DESCRIPTOR_UPPER[1]],
            ],
            Self::D2 => [[3.5, 8.5], [0.0, 10_000.0]],
            Self::D3 => [[50.0, 800.0], [3.5, 8.5]],
        }
    }

    const fn training(self, row: &PilotRow) -> [f64; 2] {
        match self {
            Self::D1 => [row.vehicles, row.imbalance],
            Self::D2 => [row.vehicles, row.mean_waiting_s],
            Self::D3 => [row.distance_km, row.vehicles],
        }
    }

    const fn holdout(self, row: &PilotRow) -> [f64; 2] {
        match self {
            Self::D1 => [row.holdout_vehicles, row.holdout_imbalance],
            Self::D2 => [row.holdout_vehicles, row.holdout_mean_waiting_s],
            Self::D3 => [row.holdout_distance_km, row.holdout_vehicles],
        }
    }
}

/// Descriptor-pair evidence.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct PairSummary {
    /// Spearman rank correlation with average tied ranks.
    pub rank_correlation: f64,
    /// Occupied archive-native fraction over every feasible observation.
    pub coverage: f64,
    /// Minimum coverage over the three deterministic seed arms.
    pub minimum_seed_coverage: f64,
    /// Coverage of locally perturbed feasible observations.
    pub local_coverage: f64,
    /// Coverage of uniformly sampled feasible observations.
    pub uniform_coverage: f64,
    /// Fraction outside the first declared bound.
    pub clipping_axis_1: f64,
    /// Fraction outside the second declared bound.
    pub clipping_axis_2: f64,
    /// Fraction remaining hard-feasible over every holdout case.
    pub holdout_feasible_fraction: f64,
    /// Training-to-holdout same-niche retention at archive capacity.
    pub holdout_niche_retention: f64,
    /// Same retention on an archive with one quarter the capacity.
    pub coarse_holdout_niche_retention: f64,
    /// Minimum observed training descriptors.
    pub reachable_minimum: [f64; 2],
    /// Maximum observed training descriptors.
    pub reachable_maximum: [f64; 2],
}

/// Gate outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QdDecision {
    /// D1 passed every pre-registered threshold.
    Accepted,
    /// The registered fallback passed.
    PrimarySecondary,
    /// No defensible descriptor pair.
    Rejected,
}

impl QdDecision {
    /// Stable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::PrimarySecondary => "primary-secondary",
            Self::Rejected => "rejected",
        }
    }
}

/// Full pilot result.
#[derive(Clone, Debug)]
pub struct PilotSummary {
    /// Feasible candidate rows.
    pub rows: Vec<PilotRow>,
    /// Attempted candidates.
    pub attempted: usize,
    /// Locally perturbed candidates attempted.
    pub local_attempted: usize,
    /// Uniform candidates attempted.
    pub uniform_attempted: usize,
    /// Archive capacity whose QD arm is gated.
    pub archive_capacity: usize,
    /// Exact archive row lengths.
    pub archive_row_lengths: Vec<usize>,
    /// D1 vehicles × imbalance.
    pub d1: PairSummary,
    /// D2 vehicles × waiting.
    pub d2: PairSummary,
    /// D3 distance × vehicles control.
    pub d3: PairSummary,
    /// Pre-registered gate outcome.
    pub decision: QdDecision,
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

fn spearman(left: &[f64], right: &[f64]) -> f64 {
    if left.len() < 2 || left.len() != right.len() {
        return f64::NAN;
    }
    let left = ranks(left);
    let right = ranks(right);
    let mean_left = left.iter().sum::<f64>() / left.len() as f64;
    let mean_right = right.iter().sum::<f64>() / right.len() as f64;
    let numerator = left
        .iter()
        .zip(&right)
        .map(|(left, right)| (left - mean_left) * (right - mean_right))
        .sum::<f64>();
    let denominator = (left
        .iter()
        .map(|value| (value - mean_left).powi(2))
        .sum::<f64>()
        * right
            .iter()
            .map(|value| (value - mean_right).powi(2))
            .sum::<f64>())
    .sqrt();
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

fn summarize(
    rows: &[PilotRow],
    pair: Pair,
    capacity: usize,
    expected_seeds: &[u64],
) -> PairSummary {
    let bounds = pair.bounds();
    let lower = [bounds[0][0], bounds[1][0]];
    let upper = [bounds[0][1], bounds[1][1]];
    let layout = ArchiveGrid::new(capacity);
    let coarse = ArchiveGrid::new(capacity.div_ceil(4));
    let training = rows
        .iter()
        .map(|row| pair.training(row))
        .collect::<Vec<_>>();
    let holdout = rows.iter().map(|row| pair.holdout(row)).collect::<Vec<_>>();
    let coverage = |seed: Option<u64>, source: Option<SampleSource>| {
        rows.iter()
            .filter(|row| seed.is_none_or(|wanted| row.seed == wanted))
            .filter(|row| source.is_none_or(|wanted| row.source == wanted))
            .filter_map(|row| layout.niche(pair.training(row), lower, upper))
            .collect::<HashSet<_>>()
            .len() as f64
            / layout.capacity() as f64
    };
    let retention = |grid: &ArchiveGrid| {
        let comparable = training
            .iter()
            .zip(&holdout)
            .filter_map(|(training, holdout)| {
                Some((
                    grid.niche(*training, lower, upper)?,
                    grid.niche(*holdout, lower, upper)?,
                ))
            })
            .collect::<Vec<_>>();
        comparable
            .iter()
            .filter(|(training, holdout)| training == holdout)
            .count() as f64
            / comparable.len().max(1) as f64
    };
    let clipping = |axis: usize| {
        training
            .iter()
            .filter(|value| !(lower[axis]..=upper[axis]).contains(&value[axis]))
            .count() as f64
            / training.len().max(1) as f64
    };
    PairSummary {
        rank_correlation: spearman(
            &training.iter().map(|value| value[0]).collect::<Vec<_>>(),
            &training.iter().map(|value| value[1]).collect::<Vec<_>>(),
        ),
        coverage: coverage(None, None),
        minimum_seed_coverage: expected_seeds
            .iter()
            .map(|seed| coverage(Some(*seed), None))
            .fold(f64::INFINITY, f64::min),
        local_coverage: coverage(None, Some(SampleSource::Local)),
        uniform_coverage: coverage(None, Some(SampleSource::Uniform)),
        clipping_axis_1: clipping(0),
        clipping_axis_2: clipping(1),
        holdout_feasible_fraction: rows.iter().filter(|row| row.holdout_feasible).count() as f64
            / rows.len().max(1) as f64,
        holdout_niche_retention: retention(&layout),
        coarse_holdout_niche_retention: retention(&coarse),
        reachable_minimum: [
            training
                .iter()
                .map(|value| value[0])
                .fold(f64::INFINITY, f64::min),
            training
                .iter()
                .map(|value| value[1])
                .fold(f64::INFINITY, f64::min),
        ],
        reachable_maximum: [
            training
                .iter()
                .map(|value| value[0])
                .fold(f64::NEG_INFINITY, f64::max),
            training
                .iter()
                .map(|value| value[1])
                .fold(f64::NEG_INFINITY, f64::max),
        ],
    }
}

fn passes(pair: PairSummary) -> bool {
    pair.rank_correlation.is_finite()
        && pair.rank_correlation.abs() < 0.7
        && pair.clipping_axis_1 < 0.1
        && pair.clipping_axis_2 < 0.1
        && pair.coverage > 0.4
        && pair.holdout_feasible_fraction > 0.6
        && pair.holdout_niche_retention > 0.6
}

/// Run a frozen three-seed, half-local/half-uniform pilot.
#[must_use]
pub fn run(instance: &Instance, samples: usize, seed: u64, capacity: usize) -> PilotSummary {
    let witness = robust_seed_controls(instance);
    let seed_arms = [seed, seed.wrapping_add(101), seed.wrapping_add(211)];
    let mut rows = Vec::new();
    let mut rngs = seed_arms.map(Rng::new);
    let mut local_attempted = 0;
    let mut uniform_attempted = 0;
    for index in 0..samples {
        let arm = index % seed_arms.len();
        let sample = index / seed_arms.len();
        let source = if sample.is_multiple_of(2) {
            SampleSource::Uniform
        } else {
            SampleSource::Local
        };
        match source {
            SampleSource::Local => local_attempted += 1,
            SampleSource::Uniform => uniform_attempted += 1,
        }
        let rng = &mut rngs[arm];
        let scale = 0.08 + 0.7 * (sample % 11) as f64 / 10.0;
        let controls = witness
            .iter()
            .map(|value| match source {
                SampleSource::Local => (value + scale * (rng.uniform01() - 0.5)).clamp(0.0, 1.0),
                SampleSource::Uniform => rng.uniform01(),
            })
            .collect::<Vec<_>>();
        let Some(training) = evaluate_training(&controls, instance) else {
            continue;
        };
        if !training.feasible() {
            continue;
        }
        let nominal = &training.nominal().metrics;
        let holdout = evaluate_holdout(&controls, instance);
        let held = holdout
            .as_ref()
            .map(|evaluation| &evaluation.nominal().metrics);
        rows.push(PilotRow {
            seed: seed_arms[arm],
            sample,
            source,
            vehicles: nominal.used_vehicles as f64,
            imbalance: nominal.imbalance_cv,
            mean_waiting_s: nominal.mean_waiting_s,
            distance_km: nominal.distance_km,
            holdout_vehicles: held.map_or(f64::NAN, |metrics| metrics.used_vehicles as f64),
            holdout_imbalance: held.map_or(f64::NAN, |metrics| metrics.imbalance_cv),
            holdout_mean_waiting_s: held.map_or(f64::NAN, |metrics| metrics.mean_waiting_s),
            holdout_distance_km: held.map_or(f64::NAN, |metrics| metrics.distance_km),
            holdout_feasible: holdout.is_some_and(|evaluation| evaluation.feasible()),
        });
    }
    let d1 = summarize(&rows, Pair::D1, capacity, &seed_arms);
    let d2 = summarize(&rows, Pair::D2, capacity, &seed_arms);
    let d3 = summarize(&rows, Pair::D3, capacity, &seed_arms);
    let decision = if passes(d1) {
        QdDecision::Accepted
    } else if passes(d2) {
        QdDecision::PrimarySecondary
    } else {
        QdDecision::Rejected
    };
    PilotSummary {
        rows,
        attempted: samples,
        local_attempted,
        uniform_attempted,
        archive_capacity: capacity,
        archive_row_lengths: ArchiveGrid::new(capacity).row_lengths(),
        d1,
        d2,
        d3,
        decision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::load_primary;

    #[test]
    fn tied_ranks_are_averaged() {
        assert_eq!(ranks(&[1.0, 1.0, 2.0, 3.0, 3.0]), [0.5, 0.5, 2.0, 3.5, 3.5]);
    }

    #[test]
    fn pilot_is_deterministic_and_reports_all_pairs() {
        let instance = load_primary().unwrap();
        let left = run(&instance, 90, 42, 120);
        let right = run(&instance, 90, 42, 120);
        assert_eq!(left.rows.len(), right.rows.len());
        assert_eq!(
            left.d1.rank_correlation.to_bits(),
            right.d1.rank_correlation.to_bits()
        );
        assert_eq!(left.local_attempted, 45);
        assert_eq!(left.uniform_attempted, 45);
        assert_eq!(left.archive_row_lengths, vec![12; 10]);
        assert!((0.0..=1.0).contains(&left.d1.holdout_niche_retention));
        assert!((0.0..=1.0).contains(&left.d3.coarse_holdout_niche_retention));
    }
}
