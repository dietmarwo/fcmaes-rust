// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Budgeted L1 Sims–Flanagan refinement of archived Lambert routes.

use std::time::Instant;

use fcmaes_core::{Cmaes, CmaesParams, Fitness, RetryConfig, RetryContext, RetryRunResult, retry};
use serde::{Deserialize, Serialize};

use crate::low_thrust_sequences::{
    INITIAL_MASS_KG, LowThrustLegEvaluation, LowThrustLegProblem, MINIMUM_HELIOCENTRIC_DISTANCE_AU,
    SequenceScaffold,
};
use crate::real::MAXIMUM_FLIGHT_DAYS;
use crate::route_archive::{RefinementResult, SearchResult};
use crate::route_search::{
    FailureCode, FailureObservation, RouteCase, RouteDerivationConfig, RouteSearchError, route_seed,
};

/// One penalty/mesh stage in the L1 continuation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefinementStage {
    /// Cartesian Sims–Flanagan impulses per leg.
    pub segments: usize,
    /// Squared endpoint/throttle constraint penalty.
    pub penalty: f64,
    /// Independent CMA-ES retries per leg.
    pub retries: usize,
    /// Objective evaluations per retry.
    pub evaluations: u64,
    /// Optional normalized CMA-ES standard deviation.
    pub sigma: Option<f64>,
}

impl RefinementStage {
    fn validate(&self) -> Result<(), RouteSearchError> {
        if self.segments == 0
            || self.retries == 0
            || self.evaluations == 0
            || !self.penalty.is_finite()
            || self.penalty <= 0.0
            || self
                .sigma
                .is_some_and(|sigma| !sigma.is_finite() || sigma <= 0.0 || sigma > 0.5)
        {
            return Err(RouteSearchError::Grammar(
                "invalid L1 continuation stage".to_owned(),
            ));
        }
        Ok(())
    }

    fn requested_evaluations_per_leg(&self) -> u64 {
        self.evaluations
            .saturating_mul(u64::try_from(self.retries).unwrap_or(u64::MAX))
    }
}

/// Reproducible L1 continuation and threshold configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefinementConfig {
    /// Ordered low-to-high fidelity continuation.
    pub stages: Vec<RefinementStage>,
    /// Retry workers; zero resolves to available logical CPUs.
    pub workers: usize,
    /// Kepler coast samples per Sims–Flanagan interval.
    pub solar_samples_per_coast: usize,
    /// Threshold on the normalized seven-component endpoint mismatch.
    pub maximum_mismatch_norm: f64,
    /// Numerical tolerance above the unit throttle ball.
    pub throttle_tolerance: f64,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            stages: vec![
                RefinementStage {
                    segments: 12,
                    penalty: 1.0e9,
                    retries: 8,
                    evaluations: 100_000,
                    sigma: None,
                },
                RefinementStage {
                    segments: 12,
                    penalty: 1.0e12,
                    retries: 8,
                    evaluations: 200_000,
                    sigma: None,
                },
                RefinementStage {
                    segments: 25,
                    penalty: 1.0e15,
                    retries: 32,
                    evaluations: 1_200_000,
                    sigma: None,
                },
            ],
            workers: 0,
            solar_samples_per_coast: 128,
            maximum_mismatch_norm: 1.0e-7,
            throttle_tolerance: 1.0e-8,
        }
    }
}

impl RefinementConfig {
    /// Tiny L1 configuration for offline integration tests.
    #[must_use]
    pub fn smoke() -> Self {
        Self {
            stages: vec![RefinementStage {
                segments: 5,
                penalty: 1.0e9,
                retries: 1,
                evaluations: 100,
                sigma: Some(0.2),
            }],
            workers: 1,
            solar_samples_per_coast: 2,
            maximum_mismatch_norm: 1.0e-7,
            throttle_tolerance: 1.0e-8,
        }
    }

    /// Validates all numerical and termination settings.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty continuation or non-positive/non-finite
    /// settings.
    pub fn validate(&self) -> Result<(), RouteSearchError> {
        if self.stages.is_empty()
            || self.solar_samples_per_coast == 0
            || !self.maximum_mismatch_norm.is_finite()
            || self.maximum_mismatch_norm <= 0.0
            || !self.throttle_tolerance.is_finite()
            || self.throttle_tolerance < 0.0
        {
            return Err(RouteSearchError::Grammar(
                "invalid L1 refinement configuration".to_owned(),
            ));
        }
        for stage in &self.stages {
            stage.validate()?;
        }
        Ok(())
    }

    /// Deterministic declared objective cap for a route with `legs` legs.
    #[must_use]
    pub fn requested_evaluations(&self, legs: usize) -> u64 {
        self.stages
            .iter()
            .map(RefinementStage::requested_evaluations_per_leg)
            .sum::<u64>()
            .saturating_mul(u64::try_from(legs).unwrap_or(u64::MAX))
    }
}

/// Refines one archived L0 route through the configured Sims–Flanagan stages.
///
/// A failed threshold gate returns a complete [`RefinementResult`] with
/// `threshold_passed=false`; budget exhaustion is not returned as physical
/// infeasibility.
///
/// # Errors
///
/// Returns an error for invalid configuration, missing/inconsistent L0 data,
/// or numerical setup failures that prevent constructing the L1 problem.
#[allow(clippy::too_many_lines)]
pub fn refine_route(
    result: &SearchResult,
    derivation: &RouteDerivationConfig,
    config: &RefinementConfig,
    root_seed: u64,
) -> Result<RefinementResult, RouteSearchError> {
    config.validate()?;
    if !result.l0.evaluation_found {
        return Err(RouteSearchError::Grammar(
            "cannot promote an L0 route for which no Lambert chain was found".to_owned(),
        ));
    }
    let route = RouteCase::derive(result.variant.clone(), derivation.clone())?;
    let scaffold = SequenceScaffold::from_selected_route(
        &route,
        &result.l0.physical_decision,
        &result.l0.branches,
    )?;
    let started = Instant::now();
    let mut controls: Vec<Vec<f64>> = Vec::new();
    let mut previous_segments = 0;
    let mut total_evaluations = 0_u64;
    let mut worker_seconds = 0.0;
    let mut final_evaluations = Vec::new();

    for (stage_index, stage) in config.stages.iter().enumerate() {
        let mut stage_controls = Vec::with_capacity(scaffold.leg_count());
        let mut stage_evaluations = Vec::with_capacity(scaffold.leg_count());
        let mut mass = INITIAL_MASS_KG;
        for leg_index in 0..scaffold.leg_count() {
            let problem = LowThrustLegProblem::new(&scaffold, leg_index, mass, stage.segments)?;
            let guess = controls.get(leg_index).map_or_else(
                || Ok(problem.initial_guess(None)),
                |source| {
                    resample_controls(source, previous_segments, stage.segments, leg_index == 0)
                },
            )?;
            let initial = problem.evaluate(&guess)?;
            let (decision, evaluation, evaluations, leg_worker_seconds) =
                if clears_leg_threshold(&initial, config) {
                    (guess, initial, 1, 0.0)
                } else {
                    solve_leg(
                        &problem,
                        &guess,
                        stage,
                        config.workers,
                        refinement_seed(root_seed, result, stage_index, leg_index),
                    )?
                };
            mass = evaluation.final_mass_kg;
            total_evaluations = total_evaluations.saturating_add(evaluations);
            worker_seconds += leg_worker_seconds;
            stage_controls.push(decision);
            stage_evaluations.push(evaluation);
        }
        controls = stage_controls;
        final_evaluations = stage_evaluations;
        previous_segments = stage.segments;
    }

    let mut minimum_solar_distance_au = f64::INFINITY;
    let mut mass = INITIAL_MASS_KG;
    let mut maximum_mismatch = 0.0_f64;
    let mut maximum_throttle = 0.0_f64;
    let mut leg_fuel_kg = Vec::with_capacity(scaffold.leg_count());
    for (leg_index, decision) in controls.iter().enumerate() {
        let problem = LowThrustLegProblem::new(&scaffold, leg_index, mass, previous_segments)?;
        let evaluation = problem.evaluate(decision)?;
        minimum_solar_distance_au = minimum_solar_distance_au.min(
            problem.minimum_heliocentric_distance_au(decision, config.solar_samples_per_coast)?,
        );
        maximum_mismatch = maximum_mismatch.max(evaluation.mismatch_norm);
        maximum_throttle = maximum_throttle.max(evaluation.maximum_throttle);
        leg_fuel_kg.push(evaluation.fuel_kg);
        mass = evaluation.final_mass_kg;
    }
    debug_assert_eq!(final_evaluations.len(), scaffold.leg_count());

    let threshold_passed = maximum_mismatch <= config.maximum_mismatch_norm
        && maximum_throttle <= 1.0 + config.throttle_tolerance
        && scaffold.powered_delta_v_km_s() < 1.0e-7
        && scaffold.minimum_periapsis_margin_km() >= -1.0e-5
        && scaffold.flight_days() <= MAXIMUM_FLIGHT_DAYS
        && minimum_solar_distance_au >= MINIMUM_HELIOCENTRIC_DISTANCE_AU;
    let score = scaffold.impact_score(mass);
    let outcome = (!threshold_passed).then(|| FailureObservation {
        code: FailureCode::RefinementNotClosed,
        leg: final_evaluations
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.mismatch_norm.total_cmp(&right.1.mismatch_norm))
            .map(|(index, _)| index),
        value: Some(maximum_mismatch),
        message: Some("L1 threshold not found within the declared continuation budget".to_owned()),
    });
    let final_stage = config.stages.last().ok_or_else(|| {
        RouteSearchError::Grammar("L1 continuation unexpectedly has no final stage".to_owned())
    })?;
    Ok(RefinementResult {
        threshold_passed,
        final_mass_kg: Some(mass),
        score: Some(score),
        maximum_normalized_mismatch: Some(maximum_mismatch),
        maximum_throttle_norm: Some(maximum_throttle),
        powered_delta_v_km_s: Some(scaffold.powered_delta_v_km_s()),
        minimum_periapsis_margin_km: Some(scaffold.minimum_periapsis_margin_km()),
        minimum_solar_distance_au: Some(minimum_solar_distance_au),
        leg_fuel_kg,
        controls,
        segments: final_stage.segments,
        requested_evaluations: config.requested_evaluations(scaffold.leg_count()),
        actual_evaluations: total_evaluations,
        resolved_workers: resolved_workers(config.workers, final_stage.retries),
        worker_seconds,
        wall_seconds: started.elapsed().as_secs_f64(),
        outcome,
    })
}

fn clears_leg_threshold(evaluation: &LowThrustLegEvaluation, config: &RefinementConfig) -> bool {
    evaluation.mismatch_norm <= config.maximum_mismatch_norm
        && evaluation.maximum_throttle <= 1.0 + config.throttle_tolerance
}

#[allow(clippy::cast_precision_loss)]
fn solve_leg(
    problem: &LowThrustLegProblem,
    initial_guess: &[f64],
    stage: &RefinementStage,
    workers: usize,
    seed: u64,
) -> Result<(Vec<f64>, LowThrustLegEvaluation, u64, f64), RouteSearchError> {
    let bounds = problem.bounds();
    let config = RetryConfig {
        num_retries: stage.retries,
        workers,
        max_evaluations: stage.evaluations,
        seed,
        value_limit: f64::INFINITY,
        stop_fitness: f64::NEG_INFINITY,
        statistic_num: 100,
        ..Default::default()
    };
    let started = Instant::now();
    let retry_result = retry(
        &|decision: &[f64]| problem.objective_with_penalty(decision, stage.penalty),
        &bounds,
        &config,
        |objective, context| {
            let mut adjusted = context.clone();
            if let Some(sigma) = stage.sigma {
                adjusted.sdev.fill(sigma);
            }
            cma_run(objective, &adjusted, initial_guess)
        },
    );
    let wall_seconds = started.elapsed().as_secs_f64();
    let evaluation = problem.evaluate(&retry_result.x)?;
    Ok((
        retry_result.x,
        evaluation,
        retry_result.evaluations,
        wall_seconds * resolved_workers(workers, stage.retries) as f64,
    ))
}

fn cma_run<O>(objective: &O, context: &RetryContext, initial_guess: &[f64]) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let mut fitness = Fitness::bounded(
        context.bounds.dim(),
        1,
        context.bounds.lower(),
        context.bounds.upper(),
    );
    fitness.set_normalize(true);
    let guess = context.guess.as_deref().unwrap_or(initial_guess);
    let mut cma = Cmaes::new(
        fitness,
        guess,
        &context.sdev,
        &CmaesParams {
            max_evaluations: context.max_evaluations,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed,
            runid: i64::try_from(context.run_id).expect("retry identifier fits i64"),
            ..Default::default()
        },
    );
    let result = cma.optimize(objective, 1);
    RetryRunResult {
        x: result.x,
        y: result.y,
        evaluations: result.evaluations,
    }
}

fn resample_controls(
    source: &[f64],
    source_segments: usize,
    target_segments: usize,
    first_leg: bool,
) -> Result<Vec<f64>, RouteSearchError> {
    let offset = if first_leg { 4 } else { 1 };
    if source_segments == 0 || source.len() != offset + 3 * source_segments {
        return Err(RouteSearchError::Dimension {
            name: "L1 warm-start controls",
            expected: offset + 3 * source_segments,
            actual: source.len(),
        });
    }
    let mut target = Vec::with_capacity(offset + 3 * target_segments);
    target.extend_from_slice(&source[..offset]);
    for target_segment in 0..target_segments {
        let source_segment = ((2 * target_segment + 1) * source_segments / (2 * target_segments))
            .min(source_segments - 1);
        let source_offset = offset + 3 * source_segment;
        target.extend_from_slice(&source[source_offset..source_offset + 3]);
    }
    Ok(target)
}

fn refinement_seed(root_seed: u64, result: &SearchResult, stage: usize, leg: usize) -> u64 {
    route_seed(root_seed, &result.variant)
        ^ u64::try_from(stage)
            .unwrap_or(u64::MAX)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::try_from(leg)
            .unwrap_or(u64::MAX)
            .wrapping_mul(0xD1B5_4A32_D192_ED03)
}

fn resolved_workers(requested: usize, retries: usize) -> usize {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let requested = if requested == 0 { available } else { requested };
    requested.min(available).min(retries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route_archive::{BranchChoice, L0Result, SearchResult, Strategy};
    use crate::route_search::{PhysicalDecision, RouteVariant};
    use crate::sequences::{JPL2, JPL2_HISTORICAL_DECISION};
    use std::collections::BTreeMap;

    fn stored_jpl2_controls() -> Vec<Vec<f64>> {
        let mut lines = include_str!("../results/jpl2-low-thrust-25-final.txt").lines();
        assert_eq!(lines.next(), Some("model endpoint-mass-v1"));
        assert_eq!(lines.next(), Some("case Jpl2"));
        assert_eq!(lines.next(), Some("segments 25"));
        lines
            .enumerate()
            .map(|(expected_leg, line)| {
                let mut fields = line.split_whitespace();
                assert_eq!(fields.next(), Some("leg"));
                assert_eq!(
                    fields.next().unwrap().parse::<usize>().unwrap(),
                    expected_leg
                );
                fields.map(|field| field.parse::<f64>().unwrap()).collect()
            })
            .collect()
    }

    fn historical_l0_result() -> SearchResult {
        let variant = RouteVariant::from_sequence_case(JPL2);
        let route = RouteCase::derive(variant.clone(), RouteDerivationConfig::default()).unwrap();
        let physical = PhysicalDecision {
            launch_mjd2000: JPL2_HISTORICAL_DECISION[0],
            leg_days: JPL2_HISTORICAL_DECISION[1..].to_vec(),
        };
        let coordinates = route.codec().encode(&physical).unwrap();
        let evaluation = route.evaluate(&coordinates).unwrap();
        SearchResult::new(
            variant,
            0,
            1,
            Strategy::Seed,
            None,
            L0Result {
                evaluation_found: true,
                objective: evaluation.sequence.objective,
                estimated_score: evaluation.sequence.estimated_score,
                fixed_mass_score: evaluation.sequence.score,
                constraint: evaluation.sequence.constraint,
                launch_v_infinity_km_s: evaluation.sequence.launch_v_infinity_km_s,
                powered_delta_v_km_s: evaluation.sequence.powered_delta_v_km_s,
                endpoint_repair_delta_v_km_s: evaluation.sequence.endpoint_repair_delta_v_km_s,
                minimum_periapsis_margin_km: evaluation.sequence.minimum_periapsis_margin_km,
                flight_days: physical.total_flight_days(),
                branches: evaluation
                    .sequence
                    .branches
                    .iter()
                    .map(|&(revolutions, path)| BranchChoice {
                        revolutions,
                        path: path.into(),
                    })
                    .collect(),
                epochs_mjd2000: evaluation.sequence.epochs_mjd2000,
                optimizer_decision: coordinates,
                physical_decision: physical,
                requested_evaluations: 0,
                actual_evaluations: 0,
                resolved_workers: 1,
                worker_seconds: 0.0,
                wall_seconds: 0.0,
                failures: BTreeMap::new(),
                failure_examples: Vec::new(),
            },
            "test".to_owned(),
            0,
        )
        .unwrap()
    }

    #[test]
    fn smoke_refinement_is_finite_and_retains_l2_warm_start() {
        let result = refine_route(
            &historical_l0_result(),
            &RouteDerivationConfig::default(),
            &RefinementConfig::smoke(),
            42,
        )
        .unwrap();
        assert_eq!(result.controls.len(), JPL2.bodies.len() - 1);
        assert!(result.controls.iter().all(|control| !control.is_empty()));
        assert!(result.final_mass_kg.unwrap().is_finite());
        assert!(result.score.unwrap().is_finite());
        assert_eq!(result.requested_evaluations, 1_000);
        assert!(result.actual_evaluations >= 1_000);
        assert!(!result.threshold_passed);
        assert_eq!(
            result.outcome.as_ref().unwrap().code,
            FailureCode::RefinementNotClosed
        );
    }

    #[test]
    fn stored_jpl2_controls_pin_the_l1_regression() {
        use pykep_core::astro::lambert::LambertPath;

        let route = RouteCase::derive(
            RouteVariant::from_sequence_case(JPL2),
            RouteDerivationConfig::default(),
        )
        .unwrap();
        let physical = PhysicalDecision {
            launch_mjd2000: JPL2_HISTORICAL_DECISION[0],
            leg_days: JPL2_HISTORICAL_DECISION[1..].to_vec(),
        };
        let branches = [
            (3, LambertPath::Left),
            (2, LambertPath::Left),
            (0, LambertPath::ZeroRevolution),
            (3, LambertPath::Right),
            (0, LambertPath::ZeroRevolution),
            (1, LambertPath::Right),
            (0, LambertPath::ZeroRevolution),
            (0, LambertPath::ZeroRevolution),
            (0, LambertPath::ZeroRevolution),
            (0, LambertPath::ZeroRevolution),
        ]
        .map(|(revolutions, path)| BranchChoice {
            revolutions,
            path: path.into(),
        });
        let scaffold = SequenceScaffold::from_selected_route(&route, &physical, &branches).unwrap();
        let controls = stored_jpl2_controls();
        assert_eq!(controls.len(), scaffold.leg_count());
        let mut mass = INITIAL_MASS_KG;
        let mut maximum_mismatch = 0.0_f64;
        let mut maximum_mismatch_leg = 0;
        let mut maximum_throttle = 0.0_f64;
        let mut minimum_solar = f64::INFINITY;
        for (leg, decision) in controls.iter().enumerate() {
            let problem = LowThrustLegProblem::new(&scaffold, leg, mass, 25).unwrap();
            let evaluation = problem.evaluate(decision).unwrap();
            if evaluation.mismatch_norm > maximum_mismatch {
                maximum_mismatch = evaluation.mismatch_norm;
                maximum_mismatch_leg = leg;
            }
            maximum_throttle = maximum_throttle.max(evaluation.maximum_throttle);
            minimum_solar = minimum_solar.min(
                problem
                    .minimum_heliocentric_distance_au(decision, 128)
                    .unwrap(),
            );
            mass = evaluation.final_mass_kg;
        }
        assert!((mass - 1_424.093_608_744).abs() < 1.0e-9);
        assert!(
            (maximum_mismatch - 3.128_450_089_626e-8).abs() < 1.0e-14,
            "maximum mismatch changed to {maximum_mismatch:.17e} on leg {maximum_mismatch_leg}"
        );
        assert!((maximum_throttle - 0.999_975_870_154).abs() < 1.0e-12);
        assert!((minimum_solar - 0.654_921_189_476).abs() < 1.0e-12);
        assert!(scaffold.powered_delta_v_km_s() < 1.0e-12);
        assert!(scaffold.minimum_periapsis_margin_km() >= -1.0e-5);
    }
}
