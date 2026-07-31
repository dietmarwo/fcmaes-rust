//! Pre-registered behavior-descriptor pilot.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use fcmaes_core::Rng;

use crate::archive_grid::ArchiveGrid;
use crate::decode::{MAX_ACTIVE, MIN_ACTIVE, baseline_controls, dimension};
use crate::evaluate::evaluate;
use crate::fem::{Scenario, WorkCounter, WorkSnapshot};
use crate::ground::GroundStructure;

/// Frozen publication archive capacity.
pub const PUBLICATION_CAPACITY: usize = 120;
/// Revised protocol after the first pilot exposed an under-scaled survival axis
/// and an overly local feasible-candidate generator.
pub const PILOT_PROTOCOL_REVISION: usize = 2;
/// Every fourth attempt uses the broad generator; this 25% mixture is frozen.
pub const BROAD_UNIFORM_STRIDE: usize = 4;
/// Primary depth/survival bounds.
pub const D1_LOWER: [f64; 2] = [0.28, 0.0];
/// Primary depth/survival bounds. Protocol v1 used a survival upper bound of
/// 1.0; the revised broad-generator calibration uses a round 0.30 bound.
pub const D1_UPPER: [f64; 2] = [0.39, 0.30];
/// Fallback utilization-spread/survival bounds.
pub const D2_LOWER: [f64; 2] = [0.0, 0.0];
/// Fallback utilization-spread/survival bounds, revised after the same
/// calibration so neither axis is compressed into a small part of the grid.
pub const D2_UPPER: [f64; 2] = [0.30, 0.30];
/// Decision-led negative-control bounds.
pub const D3_LOWER: [f64; 2] = [MIN_ACTIVE as f64, 0.0];
/// Decision-led negative-control bounds.
pub const D3_UPPER: [f64; 2] = [MAX_ACTIVE as f64, 5_000.0];

/// Frozen candidate-generator component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PilotGenerator {
    /// Local perturbations around the known feasible triangulated design.
    StructuredLocal,
    /// Uniform topology ranks and node offsets at maximum cardinality, with
    /// conservative sections to make feasible observations plausible.
    BroadUniform,
}

impl PilotGenerator {
    /// Stable artifact label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::StructuredLocal => "structured-local",
            Self::BroadUniform => "broad-uniform",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::StructuredLocal => 0,
            Self::BroadUniform => 1,
        }
    }
}

/// Descriptor-pilot decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QdDecision {
    /// Primary D1 clears every gate.
    Accepted,
    /// Only fallback D2 clears every gate.
    PrimarySecondary,
    /// Neither emergent pair clears the frozen gates.
    Rejected,
}

impl QdDecision {
    /// Schema label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::PrimarySecondary => "primary-secondary",
            Self::Rejected => "rejected",
        }
    }
}

/// One feasible pilot observation.
#[derive(Clone, Debug)]
pub struct PilotRow {
    /// Deterministic seed arm.
    pub arm: usize,
    /// Stable observation index.
    pub observation: usize,
    /// Candidate-generator component.
    pub generator: PilotGenerator,
    /// Training mass.
    pub mass_kg: f64,
    /// Active-member count.
    pub active_count: usize,
    /// Training depth/span.
    pub depth_to_span_train: f64,
    /// Holdout depth/span.
    pub depth_to_span_holdout: f64,
    /// Training utilization spread.
    pub utilization_spread_train: f64,
    /// Holdout utilization spread.
    pub utilization_spread_holdout: f64,
    /// Training removal-survival descriptor.
    pub survival_train: f64,
    /// Holdout removal-survival descriptor.
    pub survival_holdout: f64,
    /// Exact normalized controls for replay.
    pub controls: Vec<f64>,
}

impl PilotRow {
    fn train(&self, pair: usize) -> [f64; 2] {
        match pair {
            0 => [self.depth_to_span_train, self.survival_train],
            1 => [self.utilization_spread_train, self.survival_train],
            _ => [self.active_count as f64, self.mass_kg],
        }
    }

    fn holdout(&self, pair: usize) -> [f64; 2] {
        match pair {
            0 => [self.depth_to_span_holdout, self.survival_holdout],
            1 => [self.utilization_spread_holdout, self.survival_holdout],
            _ => [self.active_count as f64, self.mass_kg],
        }
    }
}

/// Measured gates for one pair.
#[derive(Clone, Debug)]
pub struct PairSummary {
    /// Stable pair label.
    pub name: &'static str,
    /// Registered lower descriptor bounds.
    pub lower_bound: [f64; 2],
    /// Registered upper descriptor bounds.
    pub upper_bound: [f64; 2],
    /// Reachable minima.
    pub reachable_min: [f64; 2],
    /// Reachable maxima.
    pub reachable_max: [f64; 2],
    /// Spearman rank correlation.
    pub spearman: f64,
    /// Lower clipping fractions.
    pub lower_clipping: [f64; 2],
    /// Upper clipping fractions.
    pub upper_clipping: [f64; 2],
    /// Per-arm occupied coverage.
    pub arm_coverage: [f64; 3],
    /// Minimum arm coverage.
    pub minimum_arm_coverage: f64,
    /// Holdout same-niche retention.
    pub holdout_niche_retention: f64,
    /// Coarser-grid holdout retention.
    pub coarse_holdout_niche_retention: f64,
    /// Complete gate outcome.
    pub passed: bool,
}

/// Attempt and feasibility counts for one generator component.
#[derive(Clone, Debug)]
pub struct GeneratorSummary {
    /// Stable component label.
    pub name: &'static str,
    /// Attempts in each deterministic seed arm.
    pub attempted_by_arm: [usize; 3],
    /// Feasible observations in each deterministic seed arm.
    pub feasible_by_arm: [usize; 3],
}

impl GeneratorSummary {
    /// Total attempted candidates.
    #[must_use]
    pub fn attempted(&self) -> usize {
        self.attempted_by_arm.iter().sum()
    }

    /// Total feasible observations.
    #[must_use]
    pub fn feasible(&self) -> usize {
        self.feasible_by_arm.iter().sum()
    }
}

/// Complete pilot result.
#[derive(Clone, Debug)]
pub struct PilotResult {
    /// Attempted structured candidates.
    pub attempted: usize,
    /// Training-feasible candidates with physical holdout metrics.
    pub feasible: usize,
    /// Full-precision observations.
    pub rows: Vec<PilotRow>,
    /// Separately reported generator-mixture outcomes.
    pub generators: [GeneratorSummary; 2],
    /// D1, D2, and negative-control summaries.
    pub pairs: [PairSummary; 3],
    /// Registered verdict.
    pub decision: QdDecision,
    /// Wall duration.
    pub elapsed: Duration,
    /// Physical-work accounting.
    pub work: WorkSnapshot,
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut ordered = (0..values.len()).collect::<Vec<_>>();
    ordered.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut result = vec![0.0; values.len()];
    let mut start = 0;
    while start < ordered.len() {
        let mut end = start + 1;
        while end < ordered.len() && values[ordered[end]] == values[ordered[start]] {
            end += 1;
        }
        let rank = 0.5 * (start + end - 1) as f64;
        for index in &ordered[start..end] {
            result[*index] = rank;
        }
        start = end;
    }
    result
}

fn spearman(left: &[f64], right: &[f64]) -> f64 {
    if left.len() < 2 {
        return 1.0;
    }
    let left = ranks(left);
    let right = ranks(right);
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let covariance = left
        .iter()
        .zip(&right)
        .map(|(a, b)| (a - left_mean) * (b - right_mean))
        .sum::<f64>();
    let left_scale = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_scale = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>()
        .sqrt();
    if left_scale == 0.0 || right_scale == 0.0 {
        1.0
    } else {
        covariance / (left_scale * right_scale)
    }
}

fn summarize_pair(
    rows: &[PilotRow],
    pair: usize,
    name: &'static str,
    lower: [f64; 2],
    upper: [f64; 2],
) -> PairSummary {
    if rows.is_empty() {
        return PairSummary {
            name,
            lower_bound: lower,
            upper_bound: upper,
            reachable_min: [0.0; 2],
            reachable_max: [0.0; 2],
            spearman: 1.0,
            lower_clipping: [1.0; 2],
            upper_clipping: [0.0; 2],
            arm_coverage: [0.0; 3],
            minimum_arm_coverage: 0.0,
            holdout_niche_retention: 0.0,
            coarse_holdout_niche_retention: 0.0,
            passed: false,
        };
    }
    let grid = ArchiveGrid::from_capacity(PUBLICATION_CAPACITY).unwrap();
    let coarse = ArchiveGrid {
        columns: grid.columns / 2,
        rows: grid.rows / 2,
    };
    let train = rows.iter().map(|row| row.train(pair)).collect::<Vec<_>>();
    let holdout = rows.iter().map(|row| row.holdout(pair)).collect::<Vec<_>>();
    let reachable_min = [
        train
            .iter()
            .map(|value| value[0])
            .fold(f64::INFINITY, f64::min),
        train
            .iter()
            .map(|value| value[1])
            .fold(f64::INFINITY, f64::min),
    ];
    let reachable_max = [
        train
            .iter()
            .map(|value| value[0])
            .fold(f64::NEG_INFINITY, f64::max),
        train
            .iter()
            .map(|value| value[1])
            .fold(f64::NEG_INFINITY, f64::max),
    ];
    let count = rows.len().max(1) as f64;
    let lower_clipping = [
        train.iter().filter(|value| value[0] <= lower[0]).count() as f64 / count,
        train.iter().filter(|value| value[1] <= lower[1]).count() as f64 / count,
    ];
    let upper_clipping = [
        train.iter().filter(|value| value[0] >= upper[0]).count() as f64 / count,
        train.iter().filter(|value| value[1] >= upper[1]).count() as f64 / count,
    ];
    let mut arm_coverage = [0.0; 3];
    for (arm, coverage) in arm_coverage.iter_mut().enumerate() {
        let occupied = rows
            .iter()
            .filter(|row| row.arm == arm)
            .filter_map(|row| grid.niche(row.train(pair), lower, upper))
            .collect::<HashSet<_>>();
        *coverage = occupied.len() as f64 / grid.capacity() as f64;
    }
    let retained = train
        .iter()
        .zip(&holdout)
        .filter(|(a, b)| grid.niche(**a, lower, upper) == grid.niche(**b, lower, upper))
        .count() as f64
        / count;
    let coarse_retained = train
        .iter()
        .zip(&holdout)
        .filter(|(a, b)| coarse.niche(**a, lower, upper) == coarse.niche(**b, lower, upper))
        .count() as f64
        / count;
    let correlation = spearman(
        &train.iter().map(|value| value[0]).collect::<Vec<_>>(),
        &train.iter().map(|value| value[1]).collect::<Vec<_>>(),
    );
    let minimum_arm_coverage = arm_coverage.iter().copied().fold(f64::INFINITY, f64::min);
    let passed = correlation.abs() < 0.7
        && lower_clipping.iter().all(|value| *value < 0.1)
        && upper_clipping.iter().all(|value| *value < 0.1)
        && minimum_arm_coverage > 0.4
        && retained > 0.6
        && coarse_retained > 0.6;
    PairSummary {
        name,
        lower_bound: lower,
        upper_bound: upper,
        reachable_min,
        reachable_max,
        spearman: correlation,
        lower_clipping,
        upper_clipping,
        arm_coverage,
        minimum_arm_coverage,
        holdout_niche_retention: retained,
        coarse_holdout_niche_retention: coarse_retained,
        passed,
    }
}

fn structured_controls(
    ground: &GroundStructure,
    arm: usize,
    observation: usize,
    rng: &mut Rng,
) -> Vec<f64> {
    let mut controls = baseline_controls(ground);
    let member_count = ground.members.len();
    let active_count = 34 + (observation + 2 * arm) % 7;
    controls[0] = (active_count - MIN_ACTIVE) as f64 / (MAX_ACTIVE - MIN_ACTIVE) as f64;
    let baseline = ground.baseline_members();
    let inactive = (0..member_count)
        .filter(|member| baseline.binary_search(member).is_err())
        .collect::<Vec<_>>();
    for _ in 0..(observation % 5) {
        let remove = baseline[(rng.next_u64() as usize) % baseline.len()];
        let add = inactive[(rng.next_u64() as usize) % inactive.len()];
        controls[1 + remove] = 0.85 + 0.1 * rng.uniform01();
        controls[1 + add] = 0.05 + 0.1 * rng.uniform01();
    }
    for member in 0..member_count {
        let section = 7 + (rng.next_u64() as usize % 5);
        controls[1 + member_count + member] = (section as f64 + 0.5) / 12.0;
    }
    let offset_start = 1 + 2 * member_count;
    for value in &mut controls[offset_start..] {
        *value = (0.18 + 0.64 * rng.uniform01()).clamp(0.0, 1.0);
    }
    controls
}

fn broad_uniform_controls(ground: &GroundStructure, rng: &mut Rng) -> Vec<f64> {
    let member_count = ground.members.len();
    let mut controls = (0..dimension(ground))
        .map(|_| rng.uniform01())
        .collect::<Vec<_>>();
    // The topology ranks and all movable-node offsets cover their complete
    // boxes. Maximum cardinality and the two largest catalogue sections keep
    // this breadth diagnostic from becoming only a mechanism detector.
    controls[0] = 1.0;
    for member in 0..member_count {
        controls[1 + member_count + member] = 0.84 + 0.16 * rng.uniform01();
    }
    controls
}

/// Run the three-arm registered pilot.
#[must_use]
pub fn run_pilot(per_arm: usize, seed: u64) -> PilotResult {
    let ground = GroundStructure::reference();
    let counter = WorkCounter::default();
    let started = Instant::now();
    let mut rows = Vec::new();
    let mut generators = [
        GeneratorSummary {
            name: PilotGenerator::StructuredLocal.name(),
            attempted_by_arm: [0; 3],
            feasible_by_arm: [0; 3],
        },
        GeneratorSummary {
            name: PilotGenerator::BroadUniform.name(),
            attempted_by_arm: [0; 3],
            feasible_by_arm: [0; 3],
        },
    ];
    for arm in 0..3 {
        let mut rng = Rng::new(seed.wrapping_add(10_000 * arm as u64));
        for observation in 0..per_arm {
            let generator = if observation % BROAD_UNIFORM_STRIDE == 0 {
                PilotGenerator::BroadUniform
            } else {
                PilotGenerator::StructuredLocal
            };
            generators[generator.index()].attempted_by_arm[arm] += 1;
            let controls = match generator {
                PilotGenerator::StructuredLocal => {
                    structured_controls(&ground, arm, observation, &mut rng)
                }
                PilotGenerator::BroadUniform => broad_uniform_controls(&ground, &mut rng),
            };
            let Some(training) = evaluate(&controls, &ground, Scenario::TRAINING, true, &counter)
            else {
                continue;
            };
            if !training.feasible() {
                continue;
            }
            let Some(holdout) = evaluate(&controls, &ground, Scenario::HOLDOUT, true, &counter)
            else {
                continue;
            };
            let (Some(train_metrics), Some(holdout_metrics)) =
                (&training.redundancy, &holdout.redundancy)
            else {
                continue;
            };
            generators[generator.index()].feasible_by_arm[arm] += 1;
            rows.push(PilotRow {
                arm,
                observation,
                generator,
                mass_kg: training.mass_kg,
                active_count: training.active_count,
                depth_to_span_train: training.depth_to_span,
                depth_to_span_holdout: holdout.depth_to_span,
                utilization_spread_train: training.utilization_spread().unwrap_or(0.0),
                utilization_spread_holdout: holdout.utilization_spread().unwrap_or(0.0),
                survival_train: train_metrics.survival,
                survival_holdout: holdout_metrics.survival,
                controls,
            });
        }
    }
    let pairs = [
        summarize_pair(&rows, 0, "D1 depth/survival", D1_LOWER, D1_UPPER),
        summarize_pair(
            &rows,
            1,
            "D2 utilization-spread/survival",
            D2_LOWER,
            D2_UPPER,
        ),
        summarize_pair(&rows, 2, "D3 active-count/mass control", D3_LOWER, D3_UPPER),
    ];
    let decision = if pairs[0].passed {
        QdDecision::Accepted
    } else if pairs[1].passed {
        QdDecision::PrimarySecondary
    } else {
        QdDecision::Rejected
    };
    PilotResult {
        attempted: 3 * per_arm,
        feasible: rows.len(),
        rows,
        generators,
        pairs,
        decision,
        elapsed: started.elapsed(),
        work: counter.snapshot(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_grid_is_derived_from_capacity() {
        assert_eq!(
            ArchiveGrid::from_capacity(PUBLICATION_CAPACITY).unwrap(),
            ArchiveGrid {
                columns: 12,
                rows: 10
            }
        );
    }

    #[test]
    fn tiny_pilot_is_deterministic_and_reports_a_verdict() {
        let left = run_pilot(3, 42);
        let right = run_pilot(3, 42);
        assert_eq!(left.attempted, 9);
        assert_eq!(left.feasible, right.feasible);
        assert_eq!(left.decision, right.decision);
        assert_eq!(
            left.rows.iter().map(|row| row.mass_kg).collect::<Vec<_>>(),
            right.rows.iter().map(|row| row.mass_kg).collect::<Vec<_>>()
        );
    }

    #[test]
    fn generated_candidates_keep_exact_cardinality() {
        let ground = GroundStructure::reference();
        let mut rng = Rng::new(42);
        let controls = structured_controls(&ground, 0, 6, &mut rng);
        let design = crate::decode::decode(&controls, &ground).unwrap();
        assert_eq!(design.active.len(), 40);
    }

    #[test]
    fn broad_candidates_cover_the_box_and_keep_maximum_cardinality() {
        let ground = GroundStructure::reference();
        let mut rng = Rng::new(42);
        let controls = broad_uniform_controls(&ground, &mut rng);
        assert_eq!(controls.len(), dimension(&ground));
        assert!(controls.iter().all(|value| (0.0..=1.0).contains(value)));
        assert_eq!(
            crate::decode::decode(&controls, &ground)
                .unwrap()
                .active
                .len(),
            MAX_ACTIVE
        );
    }

    #[test]
    fn mixed_generator_fraction_is_frozen_and_reported_separately() {
        let result = run_pilot(4, 42);
        assert_eq!(result.generators[0].attempted_by_arm, [3; 3]);
        assert_eq!(result.generators[1].attempted_by_arm, [1; 3]);
        assert_eq!(
            result
                .generators
                .iter()
                .map(GeneratorSummary::attempted)
                .sum::<usize>(),
            result.attempted
        );
        assert_eq!(
            result
                .generators
                .iter()
                .map(GeneratorSummary::feasible)
                .sum::<usize>(),
            result.feasible
        );
    }

    #[test]
    fn empty_pilot_has_finite_rejected_evidence() {
        let result = run_pilot(0, 42);
        assert_eq!(result.decision, QdDecision::Rejected);
        assert!(result.pairs.iter().all(|pair| {
            pair.reachable_min.iter().all(|value| value.is_finite())
                && pair.reachable_max.iter().all(|value| value.is_finite())
        }));
    }
}
