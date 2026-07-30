//! Pre-registered descriptor pilot for the MAP-Elites arm.

use std::collections::HashSet;
use std::time::Instant;

use fcmaes_core::Rng;
use serde::{Deserialize, Serialize};

use crate::archive_grid::ArchiveGrid;
use crate::config::Preset;
use crate::decode::DIMENSION;
use crate::evaluate::{
    Behavior, analytic_seed, behavior, behavior_for_scenario, evaluate_custom_profile,
    evaluate_holdout, evaluate_training, feasible,
};
use crate::profiles::{ProfileModifiers, hourly_validation_day};
use crate::scenarios::holdout;

/// Registered descriptor pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DescriptorPair {
    /// Daily battery throughput per installed kWh and peak-import ratio.
    D1,
    /// Self-sufficiency and curtailed-renewable fraction.
    D2,
    /// PV-to-battery capacity ratio and self-sufficiency control pair.
    D3,
}

impl DescriptorPair {
    /// Stable artifact label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::D1 => "d1",
            Self::D2 => "d2",
            Self::D3 => "d3",
        }
    }

    /// Frozen descriptor lower bounds.
    #[must_use]
    pub const fn lower(self) -> [f64; 2] {
        match self {
            Self::D1 => [0.0, 0.0],
            Self::D2 => [0.55, 0.0],
            Self::D3 => [0.0, 0.55],
        }
    }

    /// Frozen descriptor upper bounds.
    #[must_use]
    pub const fn upper(self) -> [f64; 2] {
        match self {
            Self::D1 => [2.2, 1.0],
            Self::D2 => [1.0, 0.65],
            Self::D3 => [12.0, 1.0],
        }
    }

    /// Extract this pair from an evaluated behavior.
    #[must_use]
    pub const fn values(self, behavior: Behavior) -> [f64; 2] {
        match self {
            Self::D1 => [behavior.throughput_per_kwh, behavior.peak_import_ratio],
            Self::D2 => [behavior.self_sufficiency, behavior.curtailed_fraction],
            Self::D3 => [behavior.pv_battery_ratio, behavior.self_sufficiency],
        }
    }
}

/// Pre-registered QD verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QdDecision {
    /// D1 cleared all gates.
    Accepted,
    /// D1 failed, but the emergent D2 fallback cleared the same gates.
    PrimarySecondary,
    /// Neither emergent pair cleared every gate.
    Rejected,
}

impl QdDecision {
    /// Stable artifact label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::PrimarySecondary => "primary-secondary",
            Self::Rejected => "rejected",
        }
    }
}

/// One feasible structured pilot candidate.
#[derive(Clone, Debug)]
pub struct PilotRow {
    /// Seed arm.
    pub seed: u64,
    /// Sample within the arm.
    pub sample: usize,
    /// Retained normalized controls.
    pub controls: Vec<f64>,
    /// Training behavior.
    pub training: Behavior,
    /// Battery-derating holdout behavior.
    pub holdout: Behavior,
    /// Quarter-hour discretization behavior.
    pub quarter_hour: Behavior,
    /// Hourly replay of the exact same validation day.
    pub hourly_day: Behavior,
    /// Robust training LCOE.
    pub quality: f64,
}

/// Diagnostics for one registered pair.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PairDiagnostics {
    /// Spearman rank correlation.
    pub rank_correlation: f64,
    /// Fraction outside axis-one bounds.
    pub clipping_axis_1: f64,
    /// Fraction outside axis-two bounds.
    pub clipping_axis_2: f64,
    /// Occupied fraction of the exact `fcmaes-core` archive grid.
    pub coverage: f64,
    /// Minimum coverage across the three seed arms.
    pub minimum_seed_coverage: f64,
    /// Same-niche battery-derating holdout retention.
    pub holdout_retention: f64,
    /// Same-niche retention on an archive with one quarter as many cells.
    pub coarse_holdout_retention: f64,
}

/// Complete pilot outcome and frozen rule verdict.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PilotSummary {
    /// Wall duration.
    pub elapsed_seconds: f64,
    /// Structured candidates attempted.
    pub attempted_candidates: usize,
    /// Feasible candidates retained.
    pub feasible_candidates: usize,
    /// D1 diagnostics.
    pub d1: PairDiagnostics,
    /// D2 diagnostics.
    pub d2: PairDiagnostics,
    /// Decision-led D3 control diagnostics.
    pub d3: PairDiagnostics,
    /// Mean D1 movement under the quarter-hour replay, normalized by axis spans.
    pub timestep_mean_normalized_shift: f64,
    /// Rule outcome.
    pub decision: QdDecision,
    /// Pair selected for an allowed QD run.
    pub selected_pair: Option<DescriptorPair>,
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

fn spearman(left: &[f64], right: &[f64]) -> f64 {
    if left.len() < 2 || left.len() != right.len() {
        return f64::NAN;
    }
    let left = ranks(left);
    let right = ranks(right);
    let mean_left = left.iter().sum::<f64>() / left.len() as f64;
    let mean_right = right.iter().sum::<f64>() / right.len() as f64;
    let covariance = left
        .iter()
        .zip(&right)
        .map(|(l, r)| (l - mean_left) * (r - mean_right))
        .sum::<f64>();
    let scale_left = left
        .iter()
        .map(|value| (value - mean_left).powi(2))
        .sum::<f64>();
    let scale_right = right
        .iter()
        .map(|value| (value - mean_right).powi(2))
        .sum::<f64>();
    if scale_left <= 0.0 || scale_right <= 0.0 {
        f64::NAN
    } else {
        covariance / (scale_left * scale_right).sqrt()
    }
}

fn diagnostics(rows: &[PilotRow], pair: DescriptorPair, capacity: usize) -> PairDiagnostics {
    let training = rows
        .iter()
        .map(|row| pair.values(row.training))
        .collect::<Vec<_>>();
    let held = rows
        .iter()
        .map(|row| pair.values(row.holdout))
        .collect::<Vec<_>>();
    let lower = pair.lower();
    let upper = pair.upper();
    let clipping = |axis: usize| {
        training
            .iter()
            .filter(|values| !(lower[axis]..=upper[axis]).contains(&values[axis]))
            .count() as f64
            / training.len().max(1) as f64
    };
    let coverage_for = |seed: Option<u64>, layout: &ArchiveGrid| {
        rows.iter()
            .filter(|row| seed.is_none_or(|wanted| row.seed == wanted))
            .filter_map(|row| layout.niche(pair.values(row.training), pair.lower(), pair.upper()))
            .collect::<HashSet<_>>()
            .len() as f64
            / layout.capacity() as f64
    };
    let retention = |layout: &ArchiveGrid| {
        let comparable = training
            .iter()
            .zip(&held)
            .filter_map(|(left, right)| {
                Some((
                    layout.niche(*left, pair.lower(), pair.upper())?,
                    layout.niche(*right, pair.lower(), pair.upper())?,
                ))
            })
            .collect::<Vec<_>>();
        comparable
            .iter()
            .filter(|(left, right)| left == right)
            .count() as f64
            / comparable.len().max(1) as f64
    };
    let layout = ArchiveGrid::new(capacity);
    let coarse_layout = ArchiveGrid::new(capacity.div_ceil(4));
    PairDiagnostics {
        rank_correlation: spearman(
            &training.iter().map(|values| values[0]).collect::<Vec<_>>(),
            &training.iter().map(|values| values[1]).collect::<Vec<_>>(),
        ),
        clipping_axis_1: clipping(0),
        clipping_axis_2: clipping(1),
        coverage: coverage_for(None, &layout),
        minimum_seed_coverage: [42_u64, 4_242, 424_242]
            .into_iter()
            .map(|seed| coverage_for(Some(seed), &layout))
            .fold(f64::INFINITY, f64::min),
        holdout_retention: retention(&layout),
        coarse_holdout_retention: retention(&coarse_layout),
    }
}

/// Deterministic capacity/architecture candidate shared with QD initialization.
#[must_use]
pub fn structured_candidate(seed: u64, sample: usize) -> Vec<f64> {
    let mut rng = Rng::new(seed.wrapping_add((sample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    let mut values = analytic_seed();
    values[0] = 0.24 + 0.72 * rng.uniform01();
    values[1] = 0.12 + 0.84 * rng.uniform01();
    values[2] = 0.04 + 0.90 * rng.uniform01();
    values[3] = 0.05 + 0.88 * rng.uniform01();
    values[6] = ((2 + sample % 4) as f64 + 0.5) / 6.0;
    values[7] = if sample.is_multiple_of(7) { 0.25 } else { 0.75 };
    values[8] = if sample.is_multiple_of(9) { 0.25 } else { 0.75 };
    values[9] = 0.25;
    debug_assert_eq!(values.len(), DIMENSION);
    values
}

fn passes(row: PairDiagnostics) -> bool {
    row.rank_correlation.is_finite()
        && row.rank_correlation.abs() < 0.7
        && row.clipping_axis_1 < 0.1
        && row.clipping_axis_2 < 0.1
        && row.coverage > 0.4
        && row.holdout_retention > 0.6
}

/// Execute the structured pilot across three deterministic seed arms.
#[must_use]
pub fn run_pilot(
    samples_per_seed: usize,
    preset: Preset,
    capacity: usize,
) -> (Vec<PilotRow>, PilotSummary) {
    let started = Instant::now();
    let battery_scenario = holdout()
        .into_iter()
        .find(|scenario| scenario.name == "battery_derated_80pct")
        .expect("checked-in battery derating scenario");
    let mut rows = Vec::new();
    for seed in [42_u64, 4_242, 424_242] {
        for sample in 0..samples_per_seed {
            let controls = structured_candidate(seed, sample);
            let Some(training) = evaluate_training(&controls, preset) else {
                continue;
            };
            if !feasible(&training) {
                continue;
            }
            let Ok(held) = evaluate_holdout(&controls, preset) else {
                continue;
            };
            let Some(battery_holdout) = held
                .iter()
                .find(|scenario| scenario.name == "battery_derated_80pct")
            else {
                continue;
            };
            let Some(quarter_hour) = held
                .iter()
                .find(|scenario| scenario.name == "quarter_hour_replay")
            else {
                continue;
            };
            let mut held_design = training.design.clone();
            held_design.capacities = battery_scenario.capacities(held_design.capacities);
            let Ok(hourly_day) = evaluate_custom_profile(
                &controls,
                hourly_validation_day(ProfileModifiers::default()),
                "hourly_validation_day",
            ) else {
                continue;
            };
            rows.push(PilotRow {
                seed,
                sample,
                controls,
                training: behavior(&training),
                holdout: behavior_for_scenario(&held_design, battery_holdout),
                quarter_hour: behavior_for_scenario(&training.design, quarter_hour),
                hourly_day: behavior_for_scenario(&training.design, &hourly_day),
                quality: training.mean_lcoe,
            });
        }
    }
    let d1 = diagnostics(&rows, DescriptorPair::D1, capacity);
    let d2 = diagnostics(&rows, DescriptorPair::D2, capacity);
    let d3 = diagnostics(&rows, DescriptorPair::D3, capacity);
    let d1_span = [
        DescriptorPair::D1.upper()[0] - DescriptorPair::D1.lower()[0],
        DescriptorPair::D1.upper()[1] - DescriptorPair::D1.lower()[1],
    ];
    let timestep_mean_normalized_shift = rows
        .iter()
        .map(|row| {
            let base = DescriptorPair::D1.values(row.hourly_day);
            let fine = DescriptorPair::D1.values(row.quarter_hour);
            (((fine[0] - base[0]) / d1_span[0]).powi(2)
                + ((fine[1] - base[1]) / d1_span[1]).powi(2))
            .sqrt()
        })
        .sum::<f64>()
        / rows.len().max(1) as f64;
    let (decision, selected_pair, reason) = if passes(d1) {
        (
            QdDecision::Accepted,
            Some(DescriptorPair::D1),
            "D1 clears correlation, clipping, coverage, and holdout-retention gates".to_owned(),
        )
    } else if passes(d2) {
        (
            QdDecision::PrimarySecondary,
            Some(DescriptorPair::D2),
            "D1 misses a pre-registered gate; emergent D2 clears the same rule".to_owned(),
        )
    } else {
        (
            QdDecision::Rejected,
            None,
            "neither emergent descriptor pair clears every pre-registered gate".to_owned(),
        )
    };
    let summary = PilotSummary {
        elapsed_seconds: started.elapsed().as_secs_f64(),
        attempted_candidates: 3 * samples_per_seed,
        feasible_candidates: rows.len(),
        d1,
        d2,
        d3,
        timestep_mean_normalized_shift,
        decision,
        selected_pair,
        reason,
    };
    (rows, summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pilot_reports_all_pre_registered_diagnostics() {
        let (rows, summary) = run_pilot(12, Preset::Smoke, 60);
        assert!(rows.len() >= 10, "retained only {} rows", rows.len());
        assert!(summary.d1.rank_correlation.is_finite());
        assert!((0.0..=1.0).contains(&summary.d1.holdout_retention));
        assert!((0.0..=1.0).contains(&summary.d2.coverage));
        assert!(summary.timestep_mean_normalized_shift.is_finite());
    }
}
