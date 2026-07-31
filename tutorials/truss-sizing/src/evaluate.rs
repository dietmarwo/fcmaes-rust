//! Physical objectives, explicit constraints, and removal robustness.

use crate::INVALID_COST;
use crate::catalogue::sections;
use crate::decode::{DecodedDesign, decode};
use crate::fem::{AnalysisFailure, AnalysisMetrics, RCOND_MIN, Scenario, WorkCounter, analyze};
use crate::ground::GroundStructure;

/// Allowed nodal displacement magnitude.
pub const DISPLACEMENT_LIMIT_M: f64 = 0.050;
/// Capped degradation used only for optimizer transport after a failed removal.
pub const FAILED_REMOVAL_DEGRADATION: f64 = 100.0;

/// Constraint values; optional physics fields are absent after a typed failure.
#[derive(Clone, Debug)]
pub struct Constraints {
    /// Connectivity failure flag, feasible at `≤ 0`.
    pub disconnected: f64,
    /// Rank-deficiency flag, feasible at `≤ 0`.
    pub mechanism: f64,
    /// Log-scaled conditioning constraint.
    pub conditioning: f64,
    /// Yield utilization minus one.
    pub stress: Option<f64>,
    /// Euler-buckling utilization minus one.
    pub buckling: Option<f64>,
    /// Displacement utilization minus one.
    pub displacement: Option<f64>,
}

impl Constraints {
    /// True only when all required physical constraints exist and pass.
    #[must_use]
    pub fn feasible(&self) -> bool {
        self.disconnected <= 0.0
            && self.mechanism <= 0.0
            && self.conditioning <= 0.0
            && self.stress.is_some_and(|value| value <= 0.0)
            && self.buckling.is_some_and(|value| value <= 0.0)
            && self.displacement.is_some_and(|value| value <= 0.0)
    }

    /// Numeric optimizer vector. Missing response fields receive an explicit
    /// failure sentinel; artifacts retain them as absent values.
    #[must_use]
    pub fn optimizer_values(&self) -> [f64; 6] {
        [
            self.disconnected,
            self.mechanism,
            self.conditioning,
            self.stress.unwrap_or(1.0),
            self.buckling.unwrap_or(1.0),
            self.displacement.unwrap_or(1.0),
        ]
    }

    fn penalty(&self) -> f64 {
        self.optimizer_values()
            .iter()
            .map(|value| value.max(0.0).powi(2))
            .sum()
    }
}

/// Single-member-removal robustness.
#[derive(Clone, Copy, Debug)]
pub struct RedundancyMetrics {
    /// Minimized worst-compliance degradation.
    pub degradation: f64,
    /// Bounded intact/worst compliance descriptor.
    pub survival: f64,
    /// Number of removals that failed before physical metrics existed.
    pub failed_removals: usize,
}

/// Fully replayable candidate evaluation.
#[derive(Clone, Debug)]
pub struct Evaluation {
    /// Normalized optimizer controls, empty for directly constructed tests.
    pub controls: Vec<f64>,
    /// Decoded topology and sections.
    pub design: DecodedDesign,
    /// Structural mass.
    pub mass_kg: f64,
    /// Indicative embodied carbon.
    pub carbon_kg_co2e: f64,
    /// Active member count.
    pub active_count: usize,
    /// Structural depth divided by span.
    pub depth_to_span: f64,
    /// Explicit constraints.
    pub constraints: Constraints,
    /// Typed failure, if response metrics are unavailable.
    pub failure: Option<AnalysisFailure>,
    /// Physical response only after the stability gate.
    pub metrics: Option<AnalysisMetrics>,
    /// Optional expensive removal study.
    pub redundancy: Option<RedundancyMetrics>,
    /// Penalized scalar mass objective.
    pub objective: f64,
}

impl Evaluation {
    /// Feasibility under the complete scalar contract.
    #[must_use]
    pub fn feasible(&self) -> bool {
        self.constraints.feasible()
    }

    /// Standard deviation of member utilization, available only after analysis.
    #[must_use]
    pub fn utilization_spread(&self) -> Option<f64> {
        let utilizations = &self.metrics.as_ref()?.member_utilizations;
        if utilizations.is_empty() {
            return Some(0.0);
        }
        let mean = utilizations.iter().sum::<f64>() / utilizations.len() as f64;
        Some(
            (utilizations
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / utilizations.len() as f64)
                .sqrt(),
        )
    }
}

fn mass_and_carbon(ground: &GroundStructure, design: &DecodedDesign) -> (f64, f64) {
    let catalogue = sections();
    design.active.iter().fold((0.0, 0.0), |sum, active| {
        let member = ground.members[active.member_index];
        let dx = design.nodes[member.b].x - design.nodes[member.a].x;
        let dy = design.nodes[member.b].y - design.nodes[member.a].y;
        let length = dx.hypot(dy);
        let section = catalogue[active.section_index];
        let mass = length * section.mass_kg_m;
        (sum.0 + mass, sum.1 + mass * section.carbon_kg_co2e_per_kg)
    })
}

fn constraints_from(
    result: &Result<AnalysisMetrics, AnalysisFailure>,
) -> (
    Constraints,
    Option<AnalysisFailure>,
    Option<AnalysisMetrics>,
) {
    match result {
        Ok(metrics) => (
            Constraints {
                disconnected: -1.0,
                mechanism: -1.0,
                conditioning: (RCOND_MIN / metrics.rcond).log10(),
                stress: Some(metrics.max_stress_ratio - 1.0),
                buckling: Some(metrics.max_buckling_ratio - 1.0),
                displacement: Some(metrics.max_displacement_m / DISPLACEMENT_LIMIT_M - 1.0),
            },
            None,
            Some(metrics.clone()),
        ),
        Err(failure) => {
            let (disconnected, mechanism, conditioning) = match failure {
                AnalysisFailure::Disconnected => (1.0, -1.0, -1.0),
                AnalysisFailure::Singular { .. } | AnalysisFailure::SolveFailure => {
                    (-1.0, 1.0, 1.0)
                }
                AnalysisFailure::IllConditioned { rcond } => {
                    (-1.0, -1.0, (RCOND_MIN / rcond).log10())
                }
            };
            (
                Constraints {
                    disconnected,
                    mechanism,
                    conditioning,
                    stress: None,
                    buckling: None,
                    displacement: None,
                },
                Some(failure.clone()),
                None,
            )
        }
    }
}

fn removal_robustness(
    ground: &GroundStructure,
    design: &DecodedDesign,
    intact: &AnalysisMetrics,
    scenario: Scenario,
    counter: &WorkCounter,
) -> RedundancyMetrics {
    let intact_compliance = intact.compliance_j.max(1.0e-12);
    let mut worst = intact_compliance;
    let mut failed = 0;
    for omitted in 0..design.active.len() {
        match analyze(ground, design, scenario, Some(omitted), counter) {
            Ok(metrics) => worst = worst.max(metrics.compliance_j),
            Err(_) => failed += 1,
        }
    }
    if failed > 0 {
        RedundancyMetrics {
            degradation: FAILED_REMOVAL_DEGRADATION,
            survival: 0.0,
            failed_removals: failed,
        }
    } else {
        RedundancyMetrics {
            degradation: (worst / intact_compliance - 1.0).max(0.0),
            survival: (intact_compliance / worst).clamp(0.0, 1.0),
            failed_removals: 0,
        }
    }
}

/// Evaluate an already decoded design.
#[must_use]
pub fn evaluate_decoded(
    controls: Vec<f64>,
    design: DecodedDesign,
    ground: &GroundStructure,
    scenario: Scenario,
    with_redundancy: bool,
    counter: &WorkCounter,
) -> Evaluation {
    counter.candidate();
    let (mass_kg, carbon_kg_co2e) = mass_and_carbon(ground, &design);
    let active_count = design.active.len();
    let min_y = design
        .nodes
        .iter()
        .map(|node| node.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = design
        .nodes
        .iter()
        .map(|node| node.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let depth_to_span = (max_y - min_y) / ground.span_m;
    let result = analyze(ground, &design, scenario, None, counter);
    let (constraints, failure, metrics) = constraints_from(&result);
    let redundancy = if with_redundancy {
        metrics
            .as_ref()
            .map(|intact| removal_robustness(ground, &design, intact, scenario, counter))
    } else {
        None
    };
    let objective = if mass_kg.is_finite() {
        mass_kg + 1.0e8 * constraints.penalty()
    } else {
        INVALID_COST
    };
    Evaluation {
        controls,
        design,
        mass_kg,
        carbon_kg_co2e,
        active_count,
        depth_to_span,
        constraints,
        failure,
        metrics,
        redundancy,
        objective,
    }
}

/// Decode and evaluate one normalized candidate.
#[must_use]
pub fn evaluate(
    controls: &[f64],
    ground: &GroundStructure,
    scenario: Scenario,
    with_redundancy: bool,
    counter: &WorkCounter,
) -> Option<Evaluation> {
    let design = decode(controls, ground).ok()?;
    Some(evaluate_decoded(
        controls.to_vec(),
        design,
        ground,
        scenario,
        with_redundancy,
        counter,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use fcmaes_core::parallel_batch;

    use crate::decode::{DecodedDesign, baseline_controls};

    #[test]
    fn empty_design_is_evaluable_without_fabricated_response() {
        let ground = GroundStructure::reference();
        let evaluation = evaluate_decoded(
            vec![],
            DecodedDesign {
                nodes: ground.nodes.clone(),
                active: vec![],
            },
            &ground,
            Scenario::TRAINING,
            false,
            &WorkCounter::default(),
        );
        assert!(matches!(
            evaluation.failure,
            Some(AnalysisFailure::Disconnected)
        ));
        assert!(evaluation.metrics.is_none());
        assert!(evaluation.constraints.stress.is_none());
        assert!(evaluation.objective.is_finite());
        assert!(!evaluation.feasible());
    }

    #[test]
    fn baseline_produces_counted_physics() {
        let ground = GroundStructure::reference();
        let counter = WorkCounter::default();
        let evaluation = evaluate(
            &baseline_controls(&ground),
            &ground,
            Scenario::TRAINING,
            false,
            &counter,
        )
        .unwrap();
        assert!(evaluation.metrics.is_some(), "{:?}", evaluation.failure);
        let work = counter.snapshot();
        assert_eq!(work.candidate_evaluations, 1);
        assert_eq!(work.factorizations, 1);
        assert_eq!(work.fem_solves, ground.load_cases.len() as u64);
    }

    #[test]
    fn holdout_changes_physical_metrics() {
        let ground = GroundStructure::reference();
        let controls = baseline_controls(&ground);
        let training = evaluate(
            &controls,
            &ground,
            Scenario::TRAINING,
            false,
            &WorkCounter::default(),
        )
        .unwrap();
        let holdout = evaluate(
            &controls,
            &ground,
            Scenario::HOLDOUT,
            false,
            &WorkCounter::default(),
        )
        .unwrap();
        assert_ne!(
            training.metrics.as_ref().unwrap().max_displacement_m,
            holdout.metrics.as_ref().unwrap().max_displacement_m
        );
        assert_ne!(
            training.metrics.as_ref().unwrap().max_stress_pa,
            holdout.metrics.as_ref().unwrap().max_stress_pa
        );
    }

    #[test]
    fn serial_and_parallel_batches_are_identical() {
        let ground = Arc::new(GroundStructure::reference());
        let candidates = vec![baseline_controls(&ground); 6];
        let batch = |workers| {
            let ground = Arc::clone(&ground);
            parallel_batch(&candidates, workers, move |controls| {
                evaluate(
                    controls,
                    &ground,
                    Scenario::TRAINING,
                    false,
                    &WorkCounter::default(),
                )
                .unwrap()
                .objective
            })
        };
        assert_eq!(batch(1), batch(3));
    }
}
