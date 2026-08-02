// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Impulsive multiple-gravity-assist screening for GTOC1 route variants.
//!
//! This model follows pykep's `mga` construction: independent Lambert arcs
//! are connected by the minimum powered-flyby impulse at the body's safe
//! radius. GTOC1 differs at the terminal body, where relative arrival velocity
//! creates the desired asteroid impact instead of an arrival-burn cost.

use std::time::Instant;

use fcmaes_core::{
    AdvancedRetryConfig, Cmaes, CmaesParams, De, DeParams, Fitness, RetryBounds, RetryConfig,
    RetryContext, RetryRunResult, advanced_retry,
};
use pykep_core::astro::flyby::flyby_delta_v;
use pykep_core::astro::lambert::{LambertPath, LambertProblem};
use pykep_core::{CartesianState, Vector3};
use serde::Serialize;

use crate::real::{MAXIMUM_FLIGHT_DAYS, competition_state};
use crate::route_archive::BranchChoice;
use crate::route_search::{InnerBudget, PhysicalDecision, RouteCase, RouteSearchError, route_seed};
use crate::sequences::JPL;
use crate::{
    BODY_MU_KM3_S2, DAY_SECONDS, Gtoc1Error, LEGACY_MU_SUN, distance, dot, split_state, subtract,
};

const INITIAL_MASS_KG: f64 = 1_500.0;
const FIXED_REFERENCE_MASS_KG: f64 = 1_442.9;
const EXHAUST_VELOCITY_KM_S: f64 = 2_500.0 * 9.806_65 / 1_000.0;
const FREE_LAUNCH_V_INFINITY_KM_S: f64 = 2.5;
const SAFE_PERIAPSIS_KM: [f64; 9] = [
    0.0, 2_740.0, 6_351.0, 6_678.0, 3_689.0, 600_000.0, 70_000.0, 0.0, 0.0,
];

#[derive(Clone, Copy, Debug)]
struct Arc {
    departure_velocity: Vector3,
    arrival_velocity: Vector3,
    revolutions: usize,
    path: LambertPath,
}

#[derive(Clone, Debug)]
struct PartialPath {
    charged_delta_v_km_s: f64,
    flyby_delta_v_km_s: Vec<f64>,
    branches: Vec<(usize, LambertPath)>,
}

/// Diagnostics from one GTOC1-adapted impulsive MGA evaluation.
#[derive(Clone, Debug, Serialize)]
pub struct MgaEvaluation {
    /// Minimized scalar, equal to the negative mass-adjusted impact score.
    pub objective: f64,
    /// Mass-adjusted asteroid-impact score maximized by the model.
    pub score: f64,
    /// Impact score at JPL's reported 1442.9 kg reference mass.
    pub fixed_mass_score: f64,
    /// Impact score per kilogram of retained spacecraft mass.
    pub impact_factor_km2_s2: f64,
    /// Magnitude of the asteroid-relative arrival velocity.
    pub impact_relative_speed_km_s: f64,
    /// Earth-departure hyperbolic excess magnitude.
    pub launch_v_infinity_km_s: f64,
    /// Departure excess above the free 2.5 km/s launcher capability.
    pub launch_delta_v_km_s: f64,
    /// Minimum powered impulse charged at each intermediate encounter.
    pub flyby_delta_v_km_s: Vec<f64>,
    /// Launch charge plus every powered-flyby impulse.
    pub charged_delta_v_km_s: f64,
    /// Rocket-equation mass after the charged impulsive delta-v.
    pub final_mass_kg: f64,
    /// Selected multi-revolution Lambert branch on every leg.
    pub branches: Vec<BranchChoice>,
    /// Absolute encounter epochs in MJD2000 days.
    pub epochs_mjd2000: Vec<f64>,
}

/// Complete optimizer result for one fixed route variant.
#[derive(Clone, Debug, Serialize)]
pub struct MgaOptimizationResult {
    /// Best finite MGA evaluation.
    pub evaluation: MgaEvaluation,
    /// Direct launch epoch followed by one duration per leg.
    pub optimizer_decision: Vec<f64>,
    /// Named box-bound profile used for the direct MGA decision.
    pub bounds_profile: String,
    /// Decoded launch epoch and physical leg durations.
    pub physical_decision: PhysicalDecision,
    /// Deterministic sum of requested retry caps.
    pub requested_evaluations: u64,
    /// Actual optimizer objective calls.
    pub actual_evaluations: u64,
    /// Number of retry workers used.
    pub resolved_workers: usize,
    /// Retry workers multiplied by measured wall time.
    pub worker_seconds: f64,
    /// Measured optimization wall time.
    pub wall_seconds: f64,
}

/// Evaluates one decoded route schedule with the GTOC1-adapted MGA model.
///
/// # Errors
///
/// Returns an error for inconsistent route dimensions, failed ephemerides or
/// Lambert solves, singular flybys, or a non-finite derived quantity.
pub fn evaluate_mga(
    route: &RouteCase,
    physical: &PhysicalDecision,
) -> Result<MgaEvaluation, RouteSearchError> {
    let decision = physical.as_sequence_decision();
    evaluate_runtime_mga(
        &route.variant().structure.bodies,
        &route.variant().clockwise,
        &route.maximum_revolutions(),
        &decision,
    )
    .map_err(RouteSearchError::from)
}

/// Evaluates a runtime body order and physical schedule with impulsive MGA.
///
/// The first decision is launch MJD2000 and every remaining decision is one
/// leg duration in days. Multi-revolution Lambert branches are enumerated;
/// dynamic programming retains the minimum charged delta-v reaching each arc.
///
/// # Errors
///
/// Returns an error for inconsistent dimensions, invalid dates, unsupported
/// bodies, or unavailable numerical trajectory components.
#[allow(clippy::too_many_lines)]
pub fn evaluate_runtime_mga(
    bodies: &[usize],
    clockwise: &[bool],
    maximum_revolutions: &[usize],
    decision: &[f64],
) -> Result<MgaEvaluation, Gtoc1Error> {
    let dimension = bodies.len();
    if dimension < 2
        || clockwise.len() + 1 != dimension
        || maximum_revolutions.len() + 1 != dimension
        || decision.len() != dimension
    {
        return Err(Gtoc1Error::Dimension {
            actual: decision.len(),
        });
    }
    let epochs = encounter_epochs(decision)?;
    let states = bodies
        .iter()
        .zip(&epochs)
        .map(|(&body, &epoch)| competition_state(body, epoch))
        .collect::<Result<Vec<CartesianState>, _>>()?;
    let leg_count = dimension - 1;
    let mut arc_families = Vec::with_capacity(leg_count);
    for leg in 0..leg_count {
        let (initial_position, _) = split_state(states[leg]);
        let (final_position, _) = split_state(states[leg + 1]);
        let problem = LambertProblem::new(
            initial_position,
            final_position,
            decision[leg + 1] * DAY_SECONDS,
            LEGACY_MU_SUN,
            clockwise[leg],
            maximum_revolutions[leg],
        )?;
        let family = problem
            .solutions()
            .iter()
            .map(|solution| Arc {
                departure_velocity: solution.departure_velocity,
                arrival_velocity: solution.arrival_velocity,
                revolutions: solution.revolutions,
                path: solution.path,
            })
            .collect::<Vec<_>>();
        if family.is_empty() {
            return Err(Gtoc1Error::Numerical("empty MGA Lambert family"));
        }
        arc_families.push(family);
    }

    let (_, launch_body_velocity) = split_state(states[0]);
    let mut paths = arc_families[0]
        .iter()
        .map(|arc| {
            let launch_v_infinity_km_s =
                distance(arc.departure_velocity, launch_body_velocity) / 1_000.0;
            PartialPath {
                charged_delta_v_km_s: (launch_v_infinity_km_s - FREE_LAUNCH_V_INFINITY_KM_S)
                    .max(0.0),
                flyby_delta_v_km_s: Vec::new(),
                branches: vec![(arc.revolutions, arc.path)],
            }
        })
        .collect::<Vec<_>>();

    for leg in 1..leg_count {
        let (_, planet_velocity) = split_state(states[leg]);
        let body = bodies[leg];
        let safe_radius_m = SAFE_PERIAPSIS_KM
            .get(body)
            .copied()
            .ok_or(Gtoc1Error::Numerical("unsupported MGA flyby body"))?
            * 1_000.0;
        let mu_m3_s2 = BODY_MU_KM3_S2
            .get(body)
            .copied()
            .ok_or(Gtoc1Error::Numerical("unsupported MGA flyby body"))?
            * 1.0e9;
        if safe_radius_m <= 0.0 || mu_m3_s2 <= 0.0 {
            return Err(Gtoc1Error::Numerical("unsupported MGA flyby body"));
        }
        let mut next_paths = Vec::with_capacity(arc_families[leg].len());
        for current in &arc_families[leg] {
            let mut best: Option<PartialPath> = None;
            for (previous, path) in arc_families[leg - 1].iter().zip(&paths) {
                let incoming = subtract(previous.arrival_velocity, planet_velocity);
                let outgoing = subtract(current.departure_velocity, planet_velocity);
                let delta_v_km_s =
                    flyby_delta_v(&incoming, &outgoing, mu_m3_s2, safe_radius_m)? / 1_000.0;
                let charged_delta_v_km_s = path.charged_delta_v_km_s + delta_v_km_s;
                if best
                    .as_ref()
                    .is_none_or(|candidate| charged_delta_v_km_s < candidate.charged_delta_v_km_s)
                {
                    let mut flyby_delta_v_km_s = path.flyby_delta_v_km_s.clone();
                    flyby_delta_v_km_s.push(delta_v_km_s);
                    let mut branches = path.branches.clone();
                    branches.push((current.revolutions, current.path));
                    best = Some(PartialPath {
                        charged_delta_v_km_s,
                        flyby_delta_v_km_s,
                        branches,
                    });
                }
            }
            next_paths.push(best.ok_or(Gtoc1Error::Numerical("no connected MGA branch"))?);
        }
        paths = next_paths;
    }

    let (_, asteroid_velocity) = split_state(states[dimension - 1]);
    let mut best: Option<MgaEvaluation> = None;
    for (arc, path) in arc_families[leg_count - 1].iter().zip(paths) {
        let relative = subtract(asteroid_velocity, arc.arrival_velocity);
        let impact_factor_km2_s2 = (dot(relative, asteroid_velocity) / 1.0e6).abs();
        let impact_relative_speed_km_s =
            distance(asteroid_velocity, arc.arrival_velocity) / 1_000.0;
        let final_mass_kg =
            INITIAL_MASS_KG * (-path.charged_delta_v_km_s / EXHAUST_VELOCITY_KM_S).exp();
        let score = final_mass_kg * impact_factor_km2_s2;
        let launch_v_infinity_km_s = distance(
            arc_families[0]
                .iter()
                .find(|candidate| {
                    candidate.revolutions == path.branches[0].0
                        && candidate.path == path.branches[0].1
                })
                .ok_or(Gtoc1Error::Numerical("selected MGA launch branch missing"))?
                .departure_velocity,
            launch_body_velocity,
        ) / 1_000.0;
        let candidate = MgaEvaluation {
            objective: -score,
            score,
            fixed_mass_score: FIXED_REFERENCE_MASS_KG * impact_factor_km2_s2,
            impact_factor_km2_s2,
            impact_relative_speed_km_s,
            launch_v_infinity_km_s,
            launch_delta_v_km_s: (launch_v_infinity_km_s - FREE_LAUNCH_V_INFINITY_KM_S).max(0.0),
            flyby_delta_v_km_s: path.flyby_delta_v_km_s,
            charged_delta_v_km_s: path.charged_delta_v_km_s,
            final_mass_kg,
            branches: path
                .branches
                .into_iter()
                .map(|(revolutions, path)| BranchChoice {
                    revolutions,
                    path: path.into(),
                })
                .collect(),
            epochs_mjd2000: epochs.clone(),
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate.objective < current.objective)
        {
            best = Some(candidate);
        }
    }
    best.ok_or(Gtoc1Error::Numerical("no complete MGA path"))
}

/// Optimizes one fixed route using coordinated DE followed by CMA-ES retries.
///
/// The optimizer uses pykep-compatible direct leg times. The published JPL
/// variant receives its historical per-leg box bounds; other runtime routes
/// receive their derived leg-profile bounds. `initial_physical` can preserve a
/// known schedule basin, such as the published JPL encounter dates.
///
/// # Errors
///
/// Returns an error if the initial schedule cannot be encoded or neither the
/// optimized nor initial decision has a finite MGA evaluation.
#[allow(clippy::cast_precision_loss)]
pub fn optimize_mga(
    route: &RouteCase,
    budget: &InnerBudget,
    root_seed: u64,
    initial_physical: Option<&PhysicalDecision>,
) -> Result<MgaOptimizationResult, RouteSearchError> {
    optimize_mga_with_profile(route, budget, root_seed, initial_physical, true)
}

/// Optimizes one route under the campaign's history-blind MGA protocol.
///
/// Unlike [`optimize_mga`], this entry point never recognizes the historical
/// JPL body order and never injects its published timing bounds or incumbent.
/// Every random, evolutionary, and agent-proposed route therefore receives
/// the same route-derived bound policy.
///
/// # Errors
///
/// Returns an error if neither the optimized nor midpoint decision has a
/// finite MGA evaluation.
pub fn optimize_mga_campaign(
    route: &RouteCase,
    budget: &InnerBudget,
    root_seed: u64,
) -> Result<MgaOptimizationResult, RouteSearchError> {
    optimize_mga_with_profile(route, budget, root_seed, None, false)
}

#[allow(clippy::cast_precision_loss)]
fn optimize_mga_with_profile(
    route: &RouteCase,
    budget: &InnerBudget,
    root_seed: u64,
    initial_physical: Option<&PhysicalDecision>,
    allow_historical_jpl_profile: bool,
) -> Result<MgaOptimizationResult, RouteSearchError> {
    let jpl_variant = crate::route_search::RouteVariant::from_sequence_case(JPL);
    let historical_jpl = allow_historical_jpl_profile && route.variant() == &jpl_variant;
    let (bounds, bounds_profile) = if historical_jpl {
        (JPL.bounds(), "historical-jpl")
    } else {
        let mut lower = Vec::with_capacity(route.profiles().len() + 1);
        let mut upper = Vec::with_capacity(route.profiles().len() + 1);
        lower.push(3_653.0);
        upper.push(10_958.0);
        lower.extend(route.profiles().iter().map(|profile| profile.lower_days));
        upper.extend(route.profiles().iter().map(|profile| profile.upper_days));
        let bounds = RetryBounds::new(lower, upper).map_err(|_| {
            RouteSearchError::Grammar("route profiles define invalid direct MGA bounds".to_owned())
        })?;
        (bounds, "route-derived-direct")
    };
    let initial_guess = if let Some(physical) = initial_physical {
        physical.as_sequence_decision()
    } else if historical_jpl {
        JPL.guess.to_vec()
    } else {
        let codec_bounds = route.codec().optimizer_bounds();
        let neutral_coordinates = codec_bounds
            .lower()
            .iter()
            .zip(codec_bounds.upper())
            .map(|(&lower, &upper)| 0.5 * (lower + upper))
            .collect::<Vec<_>>();
        route
            .codec()
            .decode(&neutral_coordinates)?
            .as_sequence_decision()
    };
    validate_inside_bounds(&initial_guess, &bounds)?;
    let objective = |coordinates: &[f64]| {
        direct_physical(coordinates, route.profiles().len())
            .and_then(|physical| evaluate_mga(route, &physical))
            .map_or(fcmaes_core::NAN_REPLACEMENT, |evaluation| {
                evaluation.objective
            })
    };
    let resolved_workers = resolved_workers(budget.workers, budget.retries);
    let config = AdvancedRetryConfig {
        retry: RetryConfig {
            num_retries: budget.retries,
            workers: resolved_workers,
            max_evaluations: budget.initial_evaluations,
            seed: route_seed(root_seed, route.variant()),
            value_limit: f64::INFINITY,
            stop_fitness: f64::NEG_INFINITY,
            statistic_num: 100,
            ..Default::default()
        },
        check_interval: 100,
        max_eval_fac: budget.maximum_evaluation_factor,
        ..Default::default()
    };
    let started = Instant::now();
    let retry = advanced_retry(&objective, &bounds, &config, |function, context| {
        de_cma_run(function, context, &initial_guess)
    });
    let wall_seconds = started.elapsed().as_secs_f64();
    let optimized = direct_physical(&retry.x, route.profiles().len()).and_then(|physical| {
        let evaluation = evaluate_mga(route, &physical)?;
        Ok::<_, RouteSearchError>((retry.x, physical, evaluation))
    });
    let initial = direct_physical(&initial_guess, route.profiles().len()).and_then(|physical| {
        let evaluation = evaluate_mga(route, &physical)?;
        Ok::<_, RouteSearchError>((initial_guess, physical, evaluation))
    });
    let (optimizer_decision, physical_decision, evaluation) = match (optimized, initial) {
        (Ok(optimized), Ok(initial)) => {
            if optimized.2.objective < initial.2.objective {
                optimized
            } else {
                initial
            }
        }
        (Ok(optimized), Err(_)) => optimized,
        (Err(_), Ok(initial)) => initial,
        (Err(error), Err(_)) => return Err(error),
    };
    Ok(MgaOptimizationResult {
        evaluation,
        optimizer_decision,
        bounds_profile: bounds_profile.to_owned(),
        physical_decision,
        requested_evaluations: budget.requested_evaluations(),
        actual_evaluations: retry.evaluations,
        resolved_workers,
        worker_seconds: wall_seconds * resolved_workers as f64,
        wall_seconds,
    })
}

fn direct_physical(
    decision: &[f64],
    leg_count: usize,
) -> Result<PhysicalDecision, RouteSearchError> {
    if decision.len() != leg_count + 1 {
        return Err(RouteSearchError::Dimension {
            name: "direct MGA decision",
            expected: leg_count + 1,
            actual: decision.len(),
        });
    }
    Ok(PhysicalDecision {
        launch_mjd2000: decision[0],
        leg_days: decision[1..].to_vec(),
    })
}

fn validate_inside_bounds(decision: &[f64], bounds: &RetryBounds) -> Result<(), RouteSearchError> {
    if decision.len() != bounds.dim() {
        return Err(RouteSearchError::Dimension {
            name: "initial direct MGA decision",
            expected: bounds.dim(),
            actual: decision.len(),
        });
    }
    for (index, (&value, (&lower, &upper))) in decision
        .iter()
        .zip(bounds.lower().iter().zip(bounds.upper()))
        .enumerate()
    {
        if !value.is_finite() || value < lower || value > upper {
            return Err(RouteSearchError::Coordinate {
                index,
                value,
                reason: "outside direct MGA bounds",
            });
        }
    }
    Ok(())
}

fn encounter_epochs(decision: &[f64]) -> Result<Vec<f64>, Gtoc1Error> {
    if decision.is_empty()
        || !decision[0].is_finite()
        || !(3_653.0..=10_958.0).contains(&decision[0])
    {
        return Err(Gtoc1Error::InvalidDecision {
            index: 0,
            value: decision.first().copied().unwrap_or(f64::NAN),
        });
    }
    let mut epochs = Vec::with_capacity(decision.len());
    epochs.push(decision[0]);
    for (index, &duration) in decision.iter().enumerate().skip(1) {
        if !duration.is_finite() || duration <= 0.0 {
            return Err(Gtoc1Error::InvalidDecision {
                index,
                value: duration,
            });
        }
        epochs.push(epochs[index - 1] + duration);
    }
    let flight_days = epochs[epochs.len() - 1] - epochs[0];
    if flight_days > MAXIMUM_FLIGHT_DAYS {
        return Err(Gtoc1Error::InvalidDecision {
            index: decision.len() - 1,
            value: flight_days,
        });
    }
    Ok(epochs)
}

fn de_cma_run<O>(objective: &O, context: &RetryContext, initial_guess: &[f64]) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let dimension = context.bounds.dim();
    let de_budget = (context.max_evaluations * 2 / 5).max(31);
    let cma_budget = context.max_evaluations.saturating_sub(de_budget).max(31);
    let de_fitness = Fitness::bounded(dimension, 1, context.bounds.lower(), context.bounds.upper());
    let de_sigma = context
        .sdev
        .iter()
        .zip(context.bounds.lower().iter().zip(context.bounds.upper()))
        .map(|(&sigma, (&lower, &upper))| sigma * (upper - lower))
        .collect::<Vec<_>>();
    let guess = context.guess.as_deref().unwrap_or(initial_guess);
    let mut de = De::new(
        de_fitness,
        guess,
        &de_sigma,
        None,
        &DeParams {
            max_evaluations: de_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed,
            runid: i64::try_from(context.run_id).expect("retry identifier fits i64"),
            ..Default::default()
        },
    );
    let de_result = de.optimize(objective);
    let mut cma_fitness =
        Fitness::bounded(dimension, 1, context.bounds.lower(), context.bounds.upper());
    cma_fitness.set_normalize(true);
    let mut cma = Cmaes::new(
        cma_fitness,
        &de_result.x,
        &context.sdev,
        &CmaesParams {
            max_evaluations: cma_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed ^ 0xA076_1D64_78BD_642F,
            runid: i64::try_from(context.run_id).expect("retry identifier fits i64"),
            ..Default::default()
        },
    );
    let cma_result = cma.optimize(objective, 1);
    let (x, y) = if cma_result.y < de_result.y {
        (cma_result.x, cma_result.y)
    } else {
        (de_result.x, de_result.y)
    };
    RetryRunResult {
        x,
        y,
        evaluations: de_result.evaluations + cma_result.evaluations,
    }
}

fn resolved_workers(requested: usize, retries: usize) -> usize {
    let available = num_cpus::get_physical().max(1).min(num_cpus::get().max(1));
    let requested = if requested == 0 { available } else { requested };
    requested.min(available).min(retries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real::JPL_DECISION;
    use crate::route_search::{RouteDerivationConfig, RouteVariant};
    use crate::sequences::JPL;

    fn jpl_route() -> RouteCase {
        RouteCase::derive(
            RouteVariant::from_sequence_case(JPL),
            RouteDerivationConfig::default(),
        )
        .unwrap()
    }

    fn jpl_physical() -> PhysicalDecision {
        PhysicalDecision {
            launch_mjd2000: JPL_DECISION[0],
            leg_days: JPL_DECISION[1..].to_vec(),
        }
    }

    #[test]
    fn published_jpl_schedule_has_a_finite_multi_revolution_mga_path() {
        let evaluation = evaluate_mga(&jpl_route(), &jpl_physical()).unwrap();
        assert_eq!(evaluation.branches.len(), JPL_DECISION.len() - 1);
        assert!(
            evaluation
                .branches
                .iter()
                .any(|branch| branch.revolutions > 0)
        );
        assert!(evaluation.score.is_finite() && evaluation.score > 0.0);
        assert!(evaluation.charged_delta_v_km_s > 0.0);
        assert!(evaluation.final_mass_kg > 0.0 && evaluation.final_mass_kg < INITIAL_MASS_KG);
        assert!((evaluation.score - 1_683_472.314_190_621_3).abs() < 1.0e-6);
        assert!((evaluation.charged_delta_v_km_s - 3.293_617_970_282_626).abs() < 1.0e-12);
    }

    #[test]
    fn objective_and_mass_accounting_are_exact() {
        let evaluation = evaluate_mga(&jpl_route(), &jpl_physical()).unwrap();
        assert_eq!(
            evaluation.objective.to_bits(),
            (-evaluation.score).to_bits()
        );
        let expected_mass =
            INITIAL_MASS_KG * (-evaluation.charged_delta_v_km_s / EXHAUST_VELOCITY_KM_S).exp();
        assert!((evaluation.final_mass_kg - expected_mass).abs() < 1.0e-12);
        assert!(
            (evaluation.charged_delta_v_km_s
                - evaluation.launch_delta_v_km_s
                - evaluation.flyby_delta_v_km_s.iter().sum::<f64>())
            .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn optimization_never_discards_the_supplied_incumbent() {
        let route = jpl_route();
        let physical = jpl_physical();
        let incumbent = evaluate_mga(&route, &physical).unwrap();
        let result = optimize_mga(
            &route,
            &InnerBudget {
                retries: 1,
                initial_evaluations: 31,
                maximum_evaluation_factor: 1.0,
                workers: 1,
            },
            43,
            Some(&physical),
        )
        .unwrap();
        assert!(result.evaluation.score >= incumbent.score);
        assert_eq!(result.bounds_profile, "historical-jpl");
    }

    #[test]
    fn campaign_does_not_receive_the_historical_jpl_incumbent() {
        let result = optimize_mga_campaign(
            &jpl_route(),
            &InnerBudget {
                retries: 1,
                initial_evaluations: 31,
                maximum_evaluation_factor: 1.0,
                workers: 1,
            },
            43,
        )
        .unwrap();
        assert_eq!(result.bounds_profile, "route-derived-direct");
    }

    #[test]
    fn neutral_campaign_schedules_are_finite_across_sampled_routes() {
        use crate::route_grammar::{GrammarConfig, GrammarRng, sample_route};

        let grammar = GrammarConfig::default();
        let mut rng = GrammarRng::new(7);
        for _ in 0..50 {
            let route = RouteCase::derive(
                sample_route(&grammar, &mut rng).unwrap(),
                RouteDerivationConfig::default(),
            )
            .unwrap();
            let codec_bounds = route.codec().optimizer_bounds();
            let midpoint = codec_bounds
                .lower()
                .iter()
                .zip(codec_bounds.upper())
                .map(|(&lower, &upper)| 0.5 * (lower + upper))
                .collect::<Vec<_>>();
            let physical = route.codec().decode(&midpoint).unwrap();
            assert!(physical.total_flight_days() <= MAXIMUM_FLIGHT_DAYS);
            evaluate_mga(&route, &physical).unwrap();
        }
    }
}
