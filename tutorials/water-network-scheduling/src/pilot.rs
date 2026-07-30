//! Pre-registered descriptor-pilot gate.

use std::collections::HashSet;

use epanet_rs::model::network::Network;
use fcmaes_core::Rng;
use serde::Serialize;

use crate::DIMENSION;
use crate::archive_grid::ArchiveGrid;
use crate::decode::seed_controls;
use crate::evaluate::{RobustEvaluation, evaluate_scenarios, evaluate_training};
use crate::scenarios::{holdout, training};

/// Lower bounds of the primary descriptor pair.
pub const DESCRIPTOR_LOWER: [f64; 2] = [0.15, 0.08];
/// Upper bounds of the primary descriptor pair.
pub const DESCRIPTOR_UPPER: [f64; 2] = [0.35, 0.23];

#[derive(Clone, Copy)]
enum Pair {
    D1,
    D2,
    D3,
}

impl Pair {
    const fn lower(self) -> [f64; 2] {
        match self {
            Self::D1 => DESCRIPTOR_LOWER,
            Self::D2 => [0.30, 0.08],
            Self::D3 => [0.50, 0.15],
        }
    }

    const fn upper(self) -> [f64; 2] {
        match self {
            Self::D1 => DESCRIPTOR_UPPER,
            Self::D2 => [0.35, 0.23],
            Self::D3 => [0.75, 0.35],
        }
    }

    const fn values(self, behavior: Behavior) -> [f64; 2] {
        match self {
            Self::D1 => [behavior.off_peak_fraction, behavior.tank_turnover],
            Self::D2 => [behavior.pressure_spread, behavior.tank_turnover],
            Self::D3 => [behavior.mean_pump_speed, behavior.off_peak_fraction],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Behavior {
    off_peak_fraction: f64,
    tank_turnover: f64,
    pressure_spread: f64,
    mean_pump_speed: f64,
}

/// Descriptor observation from one robust-feasible candidate.
#[derive(Clone, Debug)]
pub struct PilotRow {
    /// Deterministic seed arm.
    pub seed: u64,
    /// Sample number within the seed arm.
    pub sample: usize,
    /// Robust six-scenario training behavior.
    training: Behavior,
    /// Unseen-demand holdout behavior.
    holdout: Behavior,
    /// Nominal one-hour behavior used as the resolution baseline.
    resolution_baseline: Behavior,
    /// Nominal half-hour behavior.
    resolution_fine: Behavior,
    /// Robust operating cost.
    pub operating_cost: f64,
}

impl PilotRow {
    /// Training descriptor pair by stable pair label.
    #[must_use]
    pub fn training_pair(&self, pair: &str) -> [f64; 2] {
        named_pair(pair).values(self.training)
    }

    /// Holdout descriptor pair by stable pair label.
    #[must_use]
    pub fn holdout_pair(&self, pair: &str) -> [f64; 2] {
        named_pair(pair).values(self.holdout)
    }

    /// One-hour resolution-baseline descriptor pair.
    #[must_use]
    pub fn resolution_baseline_pair(&self, pair: &str) -> [f64; 2] {
        named_pair(pair).values(self.resolution_baseline)
    }

    /// Half-hour resolution descriptor pair.
    #[must_use]
    pub fn resolution_fine_pair(&self, pair: &str) -> [f64; 2] {
        named_pair(pair).values(self.resolution_fine)
    }
}

fn named_pair(pair: &str) -> Pair {
    match pair {
        "D1" => Pair::D1,
        "D2" => Pair::D2,
        "D3" => Pair::D3,
        _ => panic!("unknown pilot pair {pair}"),
    }
}

/// Quantitative diagnostics for one descriptor pair.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct PairSummary {
    /// Spearman rank correlation with average tied ranks.
    pub rank_correlation: f64,
    /// Archive-native aggregate coverage.
    pub coverage: f64,
    /// Minimum coverage across the three seed arms.
    pub minimum_seed_coverage: f64,
    /// Fraction outside the first axis bounds.
    pub clipping_axis_1: f64,
    /// Fraction outside the second axis bounds.
    pub clipping_axis_2: f64,
    /// Same-niche retention under the unseen-demand holdout.
    pub holdout_retention: f64,
    /// Holdout retention on an archive with one quarter the capacity.
    pub coarse_holdout_retention: f64,
    /// Same-niche retention after halving the hydraulic timestep.
    pub timestep_retention: f64,
    /// Minimum observed training descriptors.
    pub reachable_minimum: [f64; 2],
    /// Maximum observed training descriptors.
    pub reachable_maximum: [f64; 2],
}

/// Gate outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QdDecision {
    /// Primary D1 passed every frozen gate.
    AcceptedD1,
    /// D1 failed and the emergent D2 fallback passed.
    AcceptedD2,
    /// Neither emergent pair passed.
    Rejected,
}

impl QdDecision {
    /// Stable artifact label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::AcceptedD1 => "accepted-d1",
            Self::AcceptedD2 => "accepted-d2",
            Self::Rejected => "rejected",
        }
    }
}

/// Full pilot evidence.
#[derive(Clone, Debug)]
pub struct PilotSummary {
    /// Structured candidates attempted across all seed arms.
    pub attempted: usize,
    /// Archive capacity whose QD arm is being gated.
    pub archive_capacity: usize,
    /// Exact archive row lengths.
    pub archive_row_lengths: Vec<usize>,
    /// Robust-feasible observations.
    pub rows: Vec<PilotRow>,
    /// Primary-pair diagnostics.
    pub d1: PairSummary,
    /// Emergent fallback diagnostics.
    pub d2: PairSummary,
    /// Decision-led negative-control diagnostics.
    pub d3: PairSummary,
    /// Frozen rule outcome.
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

fn behavior(evaluation: &RobustEvaluation) -> Behavior {
    let nominal = &evaluation.scenarios[0];
    Behavior {
        off_peak_fraction: evaluation.descriptors[0],
        tank_turnover: evaluation.descriptors[1],
        pressure_spread: ((nominal.max_pressure_m - nominal.min_pressure_m) / 50.0).clamp(0.0, 1.0),
        mean_pump_speed: evaluation.plan.levels.iter().flatten().sum::<f64>() / 24.0,
    }
}

fn diagnostics(
    rows: &[PilotRow],
    pair: Pair,
    capacity: usize,
    expected_seeds: &[u64],
) -> PairSummary {
    let lower = pair.lower();
    let upper = pair.upper();
    let training = rows
        .iter()
        .map(|row| pair.values(row.training))
        .collect::<Vec<_>>();
    let holdout = rows
        .iter()
        .map(|row| pair.values(row.holdout))
        .collect::<Vec<_>>();
    let resolution_baseline = rows
        .iter()
        .map(|row| pair.values(row.resolution_baseline))
        .collect::<Vec<_>>();
    let resolution_fine = rows
        .iter()
        .map(|row| pair.values(row.resolution_fine))
        .collect::<Vec<_>>();
    let layout = ArchiveGrid::new(capacity);
    let coarse = ArchiveGrid::new(capacity.div_ceil(4));
    let clipping = |axis: usize| {
        training
            .iter()
            .filter(|values| !(lower[axis]..=upper[axis]).contains(&values[axis]))
            .count() as f64
            / training.len().max(1) as f64
    };
    let coverage = |seed: Option<u64>| {
        rows.iter()
            .filter(|row| seed.is_none_or(|wanted| row.seed == wanted))
            .filter_map(|row| layout.niche(pair.values(row.training), lower, upper))
            .collect::<HashSet<_>>()
            .len() as f64
            / layout.capacity() as f64
    };
    let retention = |left: &[[f64; 2]], right: &[[f64; 2]], grid: &ArchiveGrid| {
        let comparable = left
            .iter()
            .zip(right)
            .filter_map(|(left, right)| {
                Some((
                    grid.niche(*left, lower, upper)?,
                    grid.niche(*right, lower, upper)?,
                ))
            })
            .collect::<Vec<_>>();
        comparable
            .iter()
            .filter(|(left, right)| left == right)
            .count() as f64
            / comparable.len().max(1) as f64
    };
    PairSummary {
        rank_correlation: spearman(
            &training.iter().map(|values| values[0]).collect::<Vec<_>>(),
            &training.iter().map(|values| values[1]).collect::<Vec<_>>(),
        ),
        coverage: coverage(None),
        minimum_seed_coverage: expected_seeds
            .iter()
            .map(|seed| coverage(Some(*seed)))
            .fold(f64::INFINITY, f64::min),
        clipping_axis_1: clipping(0),
        clipping_axis_2: clipping(1),
        holdout_retention: retention(&training, &holdout, &layout),
        coarse_holdout_retention: retention(&training, &holdout, &coarse),
        timestep_retention: retention(&resolution_baseline, &resolution_fine, &layout),
        reachable_minimum: [
            training
                .iter()
                .map(|values| values[0])
                .fold(f64::INFINITY, f64::min),
            training
                .iter()
                .map(|values| values[1])
                .fold(f64::INFINITY, f64::min),
        ],
        reachable_maximum: [
            training
                .iter()
                .map(|values| values[0])
                .fold(f64::NEG_INFINITY, f64::max),
            training
                .iter()
                .map(|values| values[1])
                .fold(f64::NEG_INFINITY, f64::max),
        ],
    }
}

fn passes(summary: PairSummary) -> bool {
    summary.rank_correlation.is_finite()
        && summary.rank_correlation.abs() < 0.7
        && summary.clipping_axis_1 < 0.1
        && summary.clipping_axis_2 < 0.1
        && summary.coverage > 0.4
        && summary.holdout_retention > 0.6
}

/// Run three deterministic perturbation streams around the structured seed.
#[must_use]
pub fn run(network: &Network, samples: usize, seed: u64, capacity: usize) -> PilotSummary {
    let witness = seed_controls();
    let seed_arms = [seed, seed.wrapping_add(4_200), seed.wrapping_add(424_200)];
    let unseen = holdout()
        .into_iter()
        .find(|scenario| scenario.name == "unseen_demand_profile")
        .expect("checked-in unseen-demand holdout");
    let nominal = training()
        .into_iter()
        .next()
        .expect("checked-in nominal scenario");
    let mut rows = Vec::new();
    for (arm, arm_seed) in seed_arms.into_iter().enumerate() {
        let arm_samples = samples / seed_arms.len() + usize::from(arm < samples % seed_arms.len());
        let mut rng = Rng::new(arm_seed);
        for sample in 0..arm_samples {
            let scale = if sample.is_multiple_of(11) {
                2.0
            } else {
                0.08 + 0.42 * (sample % 13) as f64 / 12.0
            };
            let controls = (0..DIMENSION)
                .map(|coordinate| {
                    if scale > 1.0 {
                        rng.uniform01()
                    } else {
                        (witness[coordinate] + scale * (rng.uniform01() - 0.5)).clamp(0.0, 1.0)
                    }
                })
                .collect::<Vec<_>>();
            let Ok(evaluation) = evaluate_training(&controls, network) else {
                continue;
            };
            if !evaluation.feasible {
                continue;
            }
            let Ok(held) =
                evaluate_scenarios(&controls, network, std::slice::from_ref(&unseen), 3_600)
            else {
                continue;
            };
            let Ok(resolution_baseline) =
                evaluate_scenarios(&controls, network, std::slice::from_ref(&nominal), 3_600)
            else {
                continue;
            };
            let Ok(resolution_fine) =
                evaluate_scenarios(&controls, network, std::slice::from_ref(&nominal), 1_800)
            else {
                continue;
            };
            rows.push(PilotRow {
                seed: arm_seed,
                sample,
                training: behavior(&evaluation),
                holdout: behavior(&held),
                resolution_baseline: behavior(&resolution_baseline),
                resolution_fine: behavior(&resolution_fine),
                operating_cost: evaluation.operating_cost,
            });
        }
    }
    let d1 = diagnostics(&rows, Pair::D1, capacity, &seed_arms);
    let d2 = diagnostics(&rows, Pair::D2, capacity, &seed_arms);
    let d3 = diagnostics(&rows, Pair::D3, capacity, &seed_arms);
    let decision = if passes(d1) {
        QdDecision::AcceptedD1
    } else if passes(d2) {
        QdDecision::AcceptedD2
    } else {
        QdDecision::Rejected
    };
    PilotSummary {
        attempted: samples,
        archive_capacity: capacity,
        archive_row_lengths: ArchiveGrid::new(capacity).row_lengths(),
        rows,
        d1,
        d2,
        d3,
        decision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network;

    #[test]
    fn tied_ranks_are_averaged() {
        assert_eq!(ranks(&[1.0, 1.0, 2.0, 3.0, 3.0]), [0.5, 0.5, 2.0, 3.5, 3.5]);
    }

    #[test]
    fn pilot_is_deterministic_and_reports_full_protocol() {
        let network = network::load().unwrap();
        let left = run(&network, 12, 42, 40);
        let right = run(&network, 12, 42, 40);
        assert_eq!(left.rows.len(), right.rows.len());
        assert_eq!(left.d1.rank_correlation, right.d1.rank_correlation);
        assert_eq!(left.archive_row_lengths, vec![7, 7, 7, 7, 6, 6]);
        assert!((0.0..=1.0).contains(&left.d1.holdout_retention));
        assert!((0.0..=1.0).contains(&left.d1.timestep_retention));
    }
}
