// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! GTOC1 EVEEEJSJA competition trajectory support.
//!
//! This module uses VSOP2013 planetary states and the multi-revolution Lambert
//! solver from `pykep-core`. It first supplies the ballistic backbone needed
//! to identify the Lambert branches around JPL's published encounter dates.

use std::sync::OnceLock;

use fcmaes_core::RetryBounds;
use pykep_core::astro::lambert::{LambertPath, LambertProblem, LambertSolution};
use pykep_core::astro::propagation::propagate_lagrangian;
use pykep_core::ephemeris::{Ephemeris, Vsop2013};
use pykep_core::leg::{SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint};
use pykep_core::{CartesianState, Vector3};

use super::{
    BODY_MU_KM3_S2, DAY_SECONDS, Gtoc1Error, LEGACY_MU_SUN, asteroid, distance, dot,
    powered_swingby_inverse, split_state, subtract,
};

/// Number of encounter epochs: launch plus eight legs.
pub const REAL_DIMENSION: usize = 9;

/// Winning JPL sequence: Earth, Venus, Earth, Earth, Earth, Jupiter, Saturn,
/// Jupiter, asteroid.
pub const REAL_SEQUENCE: [usize; REAL_DIMENSION] = [3, 2, 3, 3, 3, 5, 6, 5, 10];

/// JPL's published encounter epochs expressed as MJD2000 days.
pub const JPL_ENCOUNTER_EPOCHS: [f64; REAL_DIMENSION] = [
    8_998.0, 10_276.0, 11_226.0, 12_415.0, 14_171.0, 14_657.0, 15_139.0, 18_414.0, 18_957.0,
];

/// JPL's published launch epoch followed by leg durations in days.
pub const JPL_DECISION: [f64; REAL_DIMENSION] = [
    8_998.0, 1_278.0, 950.0, 1_189.0, 1_756.0, 486.0, 482.0, 3_275.0, 543.0,
];

/// Competition launch-window lower bound.
pub const REAL_LOWER_BOUNDS: [f64; REAL_DIMENSION] = [
    3_653.0, 700.0, 650.0, 850.0, 1_300.0, 300.0, 300.0, 2_400.0, 300.0,
];

/// Search bounds around the published EVEEEJSJA basin.
pub const REAL_UPPER_BOUNDS: [f64; REAL_DIMENSION] = [
    10_958.0, 1_700.0, 1_250.0, 1_550.0, 2_200.0, 750.0, 750.0, 4_100.0, 850.0,
];

/// Competition flight-time limit in days.
pub const MAXIMUM_FLIGHT_DAYS: f64 = 30.0 * 365.25;

/// JPL's winning score reported by ESA.
pub const JPL_WINNING_SCORE: f64 = 1_850_000.0;

/// Number of Sims-Flanagan impulses on the propelled Earth-Venus leg.
pub const LOW_THRUST_SEGMENTS: usize = 24;
const LOW_THRUST_SEGMENTS_F64: f64 = 24.0;

/// Full optimization dimension: dates, endpoint geometry, final mass, and
/// three spherical control coordinates per low-thrust segment.
pub const FULL_DIMENSION: usize = 15 + 3 * LOW_THRUST_SEGMENTS;

/// Feasible warm start used for score-focused refinement.
///
/// Each throttle from the optimized twelve-segment solution is duplicated
/// over two half-duration segments. The resulting 24-segment transcription
/// starts close to the same continuous thrust history while allowing finer
/// control refinement.
#[allow(clippy::unreadable_literal)]
pub const FEASIBLE_DECISION: [f64; FULL_DIMENSION] = [
    8997.959874875154,
    1278.1880791382803,
    950.13307713091,
    1189.105625049305,
    1755.6608780273382,
    486.28635911258954,
    482.45107984979853,
    3274.2079609228713,
    543.5311811234445,
    2.4999999906842376,
    1.8350983838195205,
    -2.149714309250336,
    2.277600016966431,
    -2.047409741100634,
    1442.4542878628104,
    0.2574689168974119,
    1.959984047393032,
    -1.9838320228090482,
    3.414857003445834e-7,
    2.342869202765804,
    -1.8259989753012584,
    0.9999992421282609,
    1.9693130519043889,
    -2.3934700971592666,
    4.3502455994784417e-7,
    1.6823502929173688,
    -1.9249160439377893,
    2.3080195455348063e-7,
    0.27898625474272915,
    2.679105153513149,
    1.109597482758587e-7,
    0.3218914454591354,
    3.1413083401655983,
    5.77229687191011e-8,
    0.8426892366915403,
    2.301666136394472,
    1.7780841999291697e-7,
    1.1474329903098448,
    2.0749067257596665,
    0.4053935133911089,
    1.9706149677308122,
    -2.0098724646551456,
    8.784333675650874e-8,
    1.8285393997451223,
    -3.1241404053948596,
    0.9999991605916151,
    1.699262786833367,
    -2.3691205499730748,
    0.9999989388936122,
    1.9368193970165015,
    -1.9846063536244964,
    0.0,
    1.9955036981187648,
    3.112535342582514,
    5.626778031269514e-8,
    2.2605934245318386,
    2.3813067138255835,
    1.157315663181252e-7,
    0.36952550213327673,
    3.090360292666451,
    3.550072624628267e-7,
    0.2971374983189397,
    3.022392536455803,
    0.9999996753427126,
    1.8266104985103728,
    -2.1949117999648666,
    0.999999469838649,
    2.049292832614568,
    -1.7361175621650506,
    7.59224980193679e-9,
    1.9353346399672924,
    -1.145525819619634,
    4.655264953298127e-8,
    1.998011006171835,
    -2.6825834674487985,
    1.3352118612402347e-7,
    1.4985071591187984,
    -2.7067613278423743,
    0.9999995102568943,
    1.7338149843487736,
    -2.3654526985585966,
    0.9999998636271755,
    1.9624240355393552,
    -1.901448445805965,
    1.0045489569343591e-7,
    2.1351804290033645,
    -0.9738205759044608,
];

/// Competition wet mass in kilograms.
pub const INITIAL_MASS_KG: f64 = 1_500.0;
const JPL_FINAL_MASS_KG: f64 = 1_442.9;
const LAUNCH_V_INFINITY_KM_S: f64 = 2.5;
const CONSTRAINT_PENALTY: f64 = 1.0e10;
const FULL_CONSTRAINT_PENALTY: f64 = 1.0e15;
const MAXIMUM_THRUST_NEWTONS: f64 = 0.04;
const EXHAUST_VELOCITY_M_S: f64 = 2_500.0 * 9.806_65;
const ASTRONOMICAL_UNIT_METRES: f64 = 149_597_870_660.0;
const MINIMUM_HELIOCENTRIC_DISTANCE_AU: f64 = 0.2;
const VELOCITY_SCALE_M_S: f64 = 29_784.7;
const MAXIMUM_REVOLUTIONS: [usize; REAL_DIMENSION - 1] = [3, 3, 4, 5, 1, 1, 2, 1];
const CLOCKWISE: [bool; REAL_DIMENSION - 1] =
    [false, false, false, false, false, false, true, true];
const SELECTED_BRANCHES: [(usize, LambertPath); REAL_DIMENSION - 1] = [
    (3, LambertPath::Right),
    (1, LambertPath::Left),
    (1, LambertPath::Right),
    (0, LambertPath::ZeroRevolution),
    (0, LambertPath::ZeroRevolution),
    (0, LambertPath::ZeroRevolution),
    (0, LambertPath::ZeroRevolution),
    (0, LambertPath::ZeroRevolution),
];
const MINIMUM_PERIAPSIS_KM: [f64; 9] = [
    0.0, 2_740.0, 6_351.0, 6_678.0, 3_689.0, 600_000.0, 70_000.0, 0.0, 0.0,
];

#[derive(Clone, Debug)]
struct Arc {
    departure_velocity: Vector3,
    arrival_velocity: Vector3,
    revolutions: usize,
    path: LambertPath,
}

#[derive(Clone, Debug)]
struct PartialPath {
    constraint: f64,
    launch_v_infinity_km_s: f64,
    powered_delta_v_km_s: f64,
    minimum_periapsis_margin_km: f64,
    branches: Vec<(usize, LambertPath)>,
}

/// Ballistic-backbone diagnostics at one encounter schedule.
#[derive(Clone, Debug)]
pub struct BallisticEvaluation {
    /// Penalized minimization objective.
    pub objective: f64,
    /// Impact score using JPL's reported final mass.
    pub score: f64,
    /// Earth departure hyperbolic excess.
    pub launch_v_infinity_km_s: f64,
    /// Sum of powered impulses required to repair gravity-assist energy
    /// mismatches. A competition-feasible ballistic backbone drives this to
    /// zero.
    pub powered_delta_v_km_s: f64,
    /// Smallest flyby periapsis margin relative to the competition limits.
    pub minimum_periapsis_margin_km: f64,
    /// Selected `(revolutions, path)` for every leg.
    pub branches: Vec<(usize, LambertPath)>,
    /// Encounter epochs in MJD2000 days.
    pub epochs_mjd2000: [f64; REAL_DIMENSION],
}

/// Diagnostics for the competition low-thrust EVEEEJSJA transcription.
#[derive(Clone, Debug)]
pub struct FullEvaluation {
    /// Penalized minimization objective used by coordinated retry.
    pub objective: f64,
    /// Unpenalized competition impact score.
    pub score: f64,
    /// Sims-Flanagan normalized seven-component mismatch norm.
    pub low_thrust_mismatch_norm: f64,
    /// Earth launch hyperbolic excess in kilometres per second.
    pub launch_v_infinity_km_s: f64,
    /// Optimized arrival mass in kilograms.
    pub final_mass_kg: f64,
    /// Sum of equivalent powered impulses at nominally unpowered flybys.
    pub powered_delta_v_km_s: f64,
    /// Smallest flyby periapsis margin in kilometres.
    pub minimum_periapsis_margin_km: f64,
    /// Encounter epochs in MJD2000 days.
    pub epochs_mjd2000: [f64; REAL_DIMENSION],
    /// Raw Sims-Flanagan cut mismatch `[r,v,m]`.
    pub low_thrust_mismatch: [f64; 7],
}

/// Constructs the box used by the full competition optimization.
///
/// The encounter-date box surrounds the winning EVEEEJSJA basin disclosed in
/// JPL's workshop presentation. Direction angles and throttle vectors retain
/// their complete domains.
///
/// # Panics
///
/// Panics only if the compile-time bounds are inconsistent.
#[must_use]
pub fn full_bounds() -> RetryBounds {
    let mut lower = vec![
        8_800.0,
        1_100.0,
        850.0,
        1_050.0,
        1_600.0,
        430.0,
        430.0,
        3_100.0,
        480.0,
        0.0,
        0.0,
        -core::f64::consts::PI,
        0.0,
        -core::f64::consts::PI,
        1_350.0,
    ];
    let mut upper = vec![
        9_200.0,
        1_450.0,
        1_050.0,
        1_330.0,
        1_900.0,
        550.0,
        550.0,
        3_450.0,
        610.0,
        LAUNCH_V_INFINITY_KM_S,
        core::f64::consts::PI,
        core::f64::consts::PI,
        core::f64::consts::PI,
        core::f64::consts::PI,
        INITIAL_MASS_KG,
    ];
    for _ in 0..LOW_THRUST_SEGMENTS {
        lower.extend([0.0, 0.0, -core::f64::consts::PI]);
        upper.extend([1.0, core::f64::consts::PI, core::f64::consts::PI]);
    }
    RetryBounds::new(lower, upper).expect("competition bounds are valid")
}

/// Constructs a bounded neighborhood around an incumbent.
///
/// `fraction` is measured against each complete search-box width.
///
/// # Errors
///
/// Returns an error for the wrong incumbent dimension or a non-positive,
/// non-finite fraction.
pub fn refinement_bounds(incumbent: &[f64], fraction: f64) -> Result<RetryBounds, Gtoc1Error> {
    if incumbent.len() != FULL_DIMENSION {
        return Err(Gtoc1Error::Dimension {
            actual: incumbent.len(),
        });
    }
    if !fraction.is_finite() || fraction <= 0.0 {
        return Err(Gtoc1Error::InvalidDecision {
            index: FULL_DIMENSION,
            value: fraction,
        });
    }
    let global = full_bounds();
    let mut lower = Vec::with_capacity(FULL_DIMENSION);
    let mut upper = Vec::with_capacity(FULL_DIMENSION);
    for (index, &value) in incumbent.iter().enumerate() {
        let radius = fraction * (global.upper()[index] - global.lower()[index]);
        lower.push(global.lower()[index].max(value - radius));
        upper.push(global.upper()[index].min(value + radius));
    }
    RetryBounds::new(lower, upper).map_err(|_| Gtoc1Error::Numerical("refinement bounds"))
}

/// Converts a launch epoch plus eight durations to encounter epochs.
///
/// # Errors
///
/// Returns an error for an invalid dimension, non-finite/non-positive
/// duration, an out-of-window launch, or a flight longer than 30 years.
pub fn encounter_epochs(x: &[f64]) -> Result<[f64; REAL_DIMENSION], Gtoc1Error> {
    if x.len() != REAL_DIMENSION {
        return Err(Gtoc1Error::Dimension { actual: x.len() });
    }
    if !x[0].is_finite() || !(3_653.0..=10_958.0).contains(&x[0]) {
        return Err(Gtoc1Error::InvalidDecision {
            index: 0,
            value: x[0],
        });
    }
    let mut epochs = [0.0; REAL_DIMENSION];
    epochs[0] = x[0];
    for index in 1..REAL_DIMENSION {
        if !x[index].is_finite() || x[index] <= 0.0 {
            return Err(Gtoc1Error::InvalidDecision {
                index,
                value: x[index],
            });
        }
        epochs[index] = epochs[index - 1] + x[index];
    }
    if epochs[REAL_DIMENSION - 1] - epochs[0] > MAXIMUM_FLIGHT_DAYS {
        return Err(Gtoc1Error::InvalidDecision {
            index: REAL_DIMENSION - 1,
            value: epochs[REAL_DIMENSION - 1] - epochs[0],
        });
    }
    Ok(epochs)
}

/// Evaluates all feasible Lambert branches of the EVEEEJSJA ballistic
/// backbone.
///
/// This scouting objective uses JPL's reported 1442.9 kg arrival mass. The
/// full low-thrust transcription replaces this fixed mass after branch
/// identification.
///
/// # Errors
///
/// Returns an error for invalid dates, ephemeris failures, or an empty
/// Lambert branch family.
#[allow(clippy::too_many_lines)]
pub fn evaluate_ballistic_backbone(x: &[f64]) -> Result<BallisticEvaluation, Gtoc1Error> {
    let epochs = encounter_epochs(x)?;
    let states = competition_states(&epochs)?;
    let mut arc_families = Vec::with_capacity(REAL_DIMENSION - 1);
    for leg in 0..REAL_DIMENSION - 1 {
        let (initial_position, _) = split_state(states[leg]);
        let (final_position, _) = split_state(states[leg + 1]);
        let problem = LambertProblem::new(
            initial_position,
            final_position,
            x[leg + 1] * DAY_SECONDS,
            LEGACY_MU_SUN,
            CLOCKWISE[leg],
            MAXIMUM_REVOLUTIONS[leg],
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
            return Err(Gtoc1Error::Numerical("empty Lambert branch family"));
        }
        arc_families.push(family);
    }

    let (_, earth_velocity) = split_state(states[0]);
    let mut paths = arc_families[0]
        .iter()
        .map(|arc| {
            let launch_v_infinity_km_s = distance(arc.departure_velocity, earth_velocity) / 1_000.0;
            let excess = (launch_v_infinity_km_s - LAUNCH_V_INFINITY_KM_S).max(0.0);
            PartialPath {
                constraint: excess * excess,
                launch_v_infinity_km_s,
                powered_delta_v_km_s: 0.0,
                minimum_periapsis_margin_km: f64::INFINITY,
                branches: vec![(arc.revolutions, arc.path)],
            }
        })
        .collect::<Vec<_>>();

    for leg in 1..REAL_DIMENSION - 1 {
        let (_, planet_velocity) = split_state(states[leg]);
        let body = REAL_SEQUENCE[leg];
        let mut next_paths = Vec::with_capacity(arc_families[leg].len());
        for current in &arc_families[leg] {
            let mut best: Option<PartialPath> = None;
            for (previous, path) in arc_families[leg - 1].iter().zip(&paths) {
                let incoming = subtract(previous.arrival_velocity, planet_velocity);
                let outgoing = subtract(current.departure_velocity, planet_velocity);
                let incoming_speed = vector_norm(incoming) / 1_000.0;
                let outgoing_speed = vector_norm(outgoing) / 1_000.0;
                if incoming_speed == 0.0 || outgoing_speed == 0.0 {
                    continue;
                }
                let angle = (dot(incoming, outgoing)
                    / (vector_norm(incoming) * vector_norm(outgoing)))
                .clamp(-1.0, 1.0)
                .acos();
                let Ok((delta_v, nondimensional_periapsis)) =
                    powered_swingby_inverse(incoming_speed, outgoing_speed, angle)
                else {
                    continue;
                };
                let periapsis = nondimensional_periapsis * BODY_MU_KM3_S2[body];
                let margin = periapsis - MINIMUM_PERIAPSIS_KM[body];
                let normalized_shortfall = (-margin).max(0.0) / MINIMUM_PERIAPSIS_KM[body].max(1.0);
                let constraint = path.constraint + delta_v * delta_v + normalized_shortfall.powi(2);
                if best
                    .as_ref()
                    .is_none_or(|candidate| constraint < candidate.constraint)
                {
                    let mut branches = path.branches.clone();
                    branches.push((current.revolutions, current.path));
                    best = Some(PartialPath {
                        constraint,
                        launch_v_infinity_km_s: path.launch_v_infinity_km_s,
                        powered_delta_v_km_s: path.powered_delta_v_km_s + delta_v,
                        minimum_periapsis_margin_km: path.minimum_periapsis_margin_km.min(margin),
                        branches,
                    });
                }
            }
            next_paths.push(best.ok_or(Gtoc1Error::Numerical("no connected Lambert branch"))?);
        }
        paths = next_paths;
    }

    let (_, asteroid_velocity) = split_state(states[REAL_DIMENSION - 1]);
    let mut best: Option<BallisticEvaluation> = None;
    for (arc, path) in arc_families[REAL_DIMENSION - 2].iter().zip(paths) {
        let relative = subtract(asteroid_velocity, arc.arrival_velocity);
        let score = JPL_FINAL_MASS_KG * dot(relative, asteroid_velocity) / 1.0e6;
        let objective = CONSTRAINT_PENALTY * path.constraint - score;
        let candidate = BallisticEvaluation {
            objective,
            score,
            launch_v_infinity_km_s: path.launch_v_infinity_km_s,
            powered_delta_v_km_s: path.powered_delta_v_km_s,
            minimum_periapsis_margin_km: path.minimum_periapsis_margin_km,
            branches: path.branches,
            epochs_mjd2000: epochs,
        };
        if best
            .as_ref()
            .is_none_or(|current| objective < current.objective)
        {
            best = Some(candidate);
        }
    }
    best.ok_or(Gtoc1Error::Numerical("no complete Lambert path"))
}

/// Scalar ballistic scouting callback.
#[must_use]
pub fn ballistic_backbone_objective(x: &[f64]) -> f64 {
    evaluate_ballistic_backbone(x).map_or(fcmaes_core::NAN_REPLACEMENT, |value| value.objective)
}

/// Evaluates the complete low-thrust and ballistic EVEEEJSJA transcription.
///
/// The first leg is a 24-segment Sims-Flanagan transcription with the
/// specified 0.04 N thrust and 2500 s specific impulse. The remaining seven
/// legs use the Lambert branches identified at the published JPL schedule.
/// Gravity assists are unpowered: velocity mismatch and periapsis shortfall
/// enter the feasibility penalty.
///
/// # Errors
///
/// Returns an error for an invalid decision, ephemeris/Lambert failure, or
/// low-thrust propagation failure.
pub fn evaluate_full(x: &[f64]) -> Result<FullEvaluation, Gtoc1Error> {
    validate_full_decision(x)?;
    let epochs = encounter_epochs(&x[..REAL_DIMENSION])?;
    let states = competition_states(&epochs)?;
    let arcs = selected_ballistic_arcs(x, &states)?;

    let (earth_position, earth_velocity) = split_state(states[0]);
    let (venus_position, venus_velocity) = split_state(states[1]);

    let departure_direction = spherical_direction(x[10], x[11]);
    let departure_v_infinity = departure_direction.map(|value| value * x[9] * 1_000.0);
    let departure_velocity = add(earth_velocity, departure_v_infinity);

    let venus_outgoing = subtract(arcs[0].departure_velocity, venus_velocity);
    let venus_outgoing_speed = vector_norm(venus_outgoing);
    let venus_incoming_direction = spherical_direction(x[12], x[13]);
    let venus_incoming = venus_incoming_direction.map(|value| value * venus_outgoing_speed);
    let arrival_velocity = add(venus_velocity, venus_incoming);

    let throttles = (0..LOW_THRUST_SEGMENTS)
        .map(|segment| {
            let offset = 15 + 3 * segment;
            let direction = spherical_direction(x[offset + 1], x[offset + 2]);
            direction.map(|value| value * x[offset])
        })
        .collect::<Vec<_>>();
    let leg = SimsFlanaganLeg::new(
        SpacecraftEndpoint::new(
            join_state(earth_position, departure_velocity),
            INITIAL_MASS_KG,
        )?,
        throttles,
        SpacecraftEndpoint::new(join_state(venus_position, arrival_velocity), x[14])?,
        SimsFlanaganSettings::new(
            x[1] * DAY_SECONDS,
            MAXIMUM_THRUST_NEWTONS,
            EXHAUST_VELOCITY_M_S,
            LEGACY_MU_SUN,
            0.5,
        )?,
    )?;
    let mismatch = leg.mismatch_constraints()?;
    let normalized_mismatch = [
        mismatch[0] / ASTRONOMICAL_UNIT_METRES,
        mismatch[1] / ASTRONOMICAL_UNIT_METRES,
        mismatch[2] / ASTRONOMICAL_UNIT_METRES,
        mismatch[3] / VELOCITY_SCALE_M_S,
        mismatch[4] / VELOCITY_SCALE_M_S,
        mismatch[5] / VELOCITY_SCALE_M_S,
        mismatch[6] / INITIAL_MASS_KG,
    ];
    let low_thrust_constraint = normalized_mismatch
        .iter()
        .map(|value| value * value)
        .sum::<f64>();

    let (venus_constraint, venus_delta_v, venus_margin) =
        gravity_assist_constraint(venus_incoming, venus_outgoing, 2)?;
    let mut gravity_constraint = venus_constraint;
    let mut powered_delta_v_km_s = venus_delta_v;
    let mut minimum_periapsis_margin_km = venus_margin;
    for leg_index in 1..arcs.len() {
        let (_, planet_velocity) = split_state(states[leg_index + 1]);
        let incoming = subtract(arcs[leg_index - 1].arrival_velocity, planet_velocity);
        let outgoing = subtract(arcs[leg_index].departure_velocity, planet_velocity);
        let body = REAL_SEQUENCE[leg_index + 1];
        let (constraint, delta_v, margin) = gravity_assist_constraint(incoming, outgoing, body)?;
        gravity_constraint += constraint;
        powered_delta_v_km_s += delta_v;
        minimum_periapsis_margin_km = minimum_periapsis_margin_km.min(margin);
    }

    let (_, asteroid_velocity) = split_state(states[REAL_DIMENSION - 1]);
    let arrival_relative = subtract(asteroid_velocity, arcs[arcs.len() - 1].arrival_velocity);
    let score = x[14] * dot(arrival_relative, asteroid_velocity) / 1.0e6;
    let constraint = low_thrust_constraint + gravity_constraint;
    Ok(FullEvaluation {
        objective: FULL_CONSTRAINT_PENALTY * constraint - score,
        score,
        low_thrust_mismatch_norm: low_thrust_constraint.sqrt(),
        launch_v_infinity_km_s: x[9],
        final_mass_kg: x[14],
        powered_delta_v_km_s,
        minimum_periapsis_margin_km,
        epochs_mjd2000: epochs,
        low_thrust_mismatch: mismatch,
    })
}

/// Scalar callback for coordinated retry.
#[must_use]
pub fn full_objective(x: &[f64]) -> f64 {
    evaluate_full(x).map_or(fcmaes_core::NAN_REPLACEMENT, |value| value.objective)
}

/// Samples the complete trajectory to check the competition's 0.2 AU solar
/// exclusion constraint.
///
/// The low-thrust leg is reconstructed with the same impulse-centred
/// Sims-Flanagan convention used by [`SimsFlanaganLeg`]. Each coast is sampled
/// at 32 uniform intervals and every Lambert arc at 512 uniform intervals.
/// This diagnostic is intentionally separate from [`full_objective`] so it
/// does not add thousands of propagations to every optimizer evaluation.
///
/// # Errors
///
/// Returns an error for an invalid decision or a failed two-body propagation.
pub fn minimum_heliocentric_distance_au(x: &[f64]) -> Result<f64, Gtoc1Error> {
    validate_full_decision(x)?;
    let epochs = encounter_epochs(&x[..REAL_DIMENSION])?;
    let states = competition_states(&epochs)?;
    let arcs = selected_ballistic_arcs(x, &states)?;

    let (earth_position, earth_velocity) = split_state(states[0]);
    let departure_direction = spherical_direction(x[10], x[11]);
    let departure_v_infinity = departure_direction.map(|value| value * x[9] * 1_000.0);
    let mut state = join_state(earth_position, add(earth_velocity, departure_v_infinity));
    let mut mass = INITIAL_MASS_KG;
    let segment_duration = x[1] * DAY_SECONDS / LOW_THRUST_SEGMENTS_F64;
    let mut minimum_radius = position_norm(&state);

    for segment in 0..LOW_THRUST_SEGMENTS {
        let coast = if segment == 0 {
            segment_duration / 2.0
        } else {
            segment_duration
        };
        minimum_radius = minimum_radius.min(sample_coast(&mut state, coast, 32)?);

        let offset = 15 + 3 * segment;
        let throttle =
            spherical_direction(x[offset + 1], x[offset + 2]).map(|value| value * x[offset]);
        let scale = MAXIMUM_THRUST_NEWTONS * segment_duration / mass;
        let impulse = throttle.map(|value| scale * value);
        for component in 0..3 {
            state[component + 3] += impulse[component];
        }
        mass *= (-vector_norm(impulse) / EXHAUST_VELOCITY_M_S).exp();
    }
    minimum_radius = minimum_radius.min(sample_coast(&mut state, segment_duration / 2.0, 32)?);

    for (index, arc) in arcs.iter().enumerate() {
        let (position, _) = split_state(states[index + 1]);
        let mut arc_state = join_state(position, arc.departure_velocity);
        minimum_radius = minimum_radius.min(sample_coast(
            &mut arc_state,
            x[index + 2] * DAY_SECONDS,
            512,
        )?);
    }
    Ok(minimum_radius / ASTRONOMICAL_UNIT_METRES)
}

/// Returns whether the sampled trajectory clears the 0.2 AU solar exclusion
/// distance.
///
/// # Errors
///
/// Returns the same errors as [`minimum_heliocentric_distance_au`].
pub fn clears_solar_exclusion(x: &[f64]) -> Result<bool, Gtoc1Error> {
    Ok(minimum_heliocentric_distance_au(x)? >= MINIMUM_HELIOCENTRIC_DISTANCE_AU)
}

fn validate_full_decision(x: &[f64]) -> Result<(), Gtoc1Error> {
    if x.len() != FULL_DIMENSION {
        return Err(Gtoc1Error::Dimension { actual: x.len() });
    }
    for (index, &value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(Gtoc1Error::InvalidDecision { index, value });
        }
    }
    let bounds = full_bounds();
    for (index, ((&value, &lower), &upper)) in
        x.iter().zip(bounds.lower()).zip(bounds.upper()).enumerate()
    {
        if !(lower..=upper).contains(&value) {
            return Err(Gtoc1Error::InvalidDecision { index, value });
        }
    }
    Ok(())
}

fn selected_ballistic_arcs(
    x: &[f64],
    states: &[CartesianState; REAL_DIMENSION],
) -> Result<Vec<Arc>, Gtoc1Error> {
    let mut arcs = Vec::with_capacity(REAL_DIMENSION - 2);
    for leg in 1..REAL_DIMENSION - 1 {
        let (initial_position, _) = split_state(states[leg]);
        let (final_position, _) = split_state(states[leg + 1]);
        let branch = SELECTED_BRANCHES[leg];
        let problem = LambertProblem::new(
            initial_position,
            final_position,
            x[leg + 1] * DAY_SECONDS,
            LEGACY_MU_SUN,
            CLOCKWISE[leg],
            branch.0,
        )?;
        let solution = select_solution(problem.solutions(), branch).ok_or(
            Gtoc1Error::Numerical("selected Lambert branch is unavailable"),
        )?;
        arcs.push(Arc {
            departure_velocity: solution.departure_velocity,
            arrival_velocity: solution.arrival_velocity,
            revolutions: solution.revolutions,
            path: solution.path,
        });
    }
    Ok(arcs)
}

fn select_solution(
    solutions: &[LambertSolution],
    branch: (usize, LambertPath),
) -> Option<&LambertSolution> {
    solutions
        .iter()
        .find(|solution| solution.revolutions == branch.0 && solution.path == branch.1)
}

fn gravity_assist_constraint(
    incoming: Vector3,
    outgoing: Vector3,
    body: usize,
) -> Result<(f64, f64, f64), Gtoc1Error> {
    let incoming_norm = vector_norm(incoming);
    let outgoing_norm = vector_norm(outgoing);
    if incoming_norm == 0.0 || outgoing_norm == 0.0 {
        return Err(Gtoc1Error::Numerical("gravity-assist relative velocity"));
    }
    let angle = (dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
        .clamp(-1.0, 1.0)
        .acos();
    let (delta_v, nondimensional_periapsis) =
        powered_swingby_inverse(incoming_norm / 1_000.0, outgoing_norm / 1_000.0, angle)?;
    let periapsis = nondimensional_periapsis * BODY_MU_KM3_S2[body];
    let margin = periapsis - MINIMUM_PERIAPSIS_KM[body];
    let normalized_shortfall = (-margin).max(0.0) / MINIMUM_PERIAPSIS_KM[body].max(1.0);
    Ok((
        delta_v * delta_v + normalized_shortfall.powi(2),
        delta_v,
        margin,
    ))
}

fn spherical_direction(theta: f64, phi: f64) -> Vector3 {
    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();
    [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
}

fn add(left: Vector3, right: Vector3) -> Vector3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn join_state(position: Vector3, velocity: Vector3) -> CartesianState {
    [
        position[0],
        position[1],
        position[2],
        velocity[0],
        velocity[1],
        velocity[2],
    ]
}

fn competition_states(
    epochs: &[f64; REAL_DIMENSION],
) -> Result<[CartesianState; REAL_DIMENSION], Gtoc1Error> {
    let mut states = [[0.0; 6]; REAL_DIMENSION];
    for index in 0..REAL_DIMENSION {
        states[index] = competition_state(REAL_SEQUENCE[index], epochs[index])?;
    }
    Ok(states)
}

/// Returns the VSOP2013/asteroid state used by the competition model.
///
/// `body` follows the restricted route-search grammar (Venus 2, Earth 3,
/// Jupiter 5, Saturn 6, or asteroid 10) and `epoch` is MJD2000.
///
/// # Errors
///
/// Returns an error for an unsupported body or ephemeris failure.
pub fn competition_state(body: usize, epoch: f64) -> Result<CartesianState, Gtoc1Error> {
    Ok(match body {
        2 => venus().state(epoch)?,
        3 => earth().state(epoch)?,
        5 => jupiter().state(epoch)?,
        6 => saturn().state(epoch)?,
        10 => rotate_ecliptic_to_icrf(asteroid().state(epoch)?),
        _ => return Err(Gtoc1Error::Numerical("unsupported competition body")),
    })
}

fn venus() -> &'static Vsop2013 {
    static VALUE: OnceLock<Vsop2013> = OnceLock::new();
    VALUE.get_or_init(|| {
        Vsop2013::with_threshold("venus", 1.0e-9).expect("VSOP2013 Venus is available")
    })
}

fn earth() -> &'static Vsop2013 {
    static VALUE: OnceLock<Vsop2013> = OnceLock::new();
    VALUE.get_or_init(|| {
        Vsop2013::with_threshold("earth_moon", 1.0e-9).expect("VSOP2013 Earth-Moon is available")
    })
}

fn jupiter() -> &'static Vsop2013 {
    static VALUE: OnceLock<Vsop2013> = OnceLock::new();
    VALUE.get_or_init(|| {
        Vsop2013::with_threshold("jupiter", 1.0e-9).expect("VSOP2013 Jupiter is available")
    })
}

fn saturn() -> &'static Vsop2013 {
    static VALUE: OnceLock<Vsop2013> = OnceLock::new();
    VALUE.get_or_init(|| {
        Vsop2013::with_threshold("saturn", 1.0e-9).expect("VSOP2013 Saturn is available")
    })
}

fn rotate_ecliptic_to_icrf(state: CartesianState) -> CartesianState {
    let epsilon: f64 = 0.409_092_626_586_596_2;
    let phi: f64 = -2.515_213_377_596_228_5e-7;
    let (sin_epsilon, cos_epsilon) = epsilon.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();
    let rotate = |vector: [f64; 3]| {
        [
            cos_phi * vector[0] - sin_phi * cos_epsilon * vector[1]
                + sin_phi * sin_epsilon * vector[2],
            sin_phi * vector[0] + cos_phi * cos_epsilon * vector[1]
                - cos_phi * sin_epsilon * vector[2],
            sin_epsilon * vector[1] + cos_epsilon * vector[2],
        ]
    };
    let position = rotate([state[0], state[1], state[2]]);
    let velocity = rotate([state[3], state[4], state[5]]);
    [
        position[0],
        position[1],
        position[2],
        velocity[0],
        velocity[1],
        velocity[2],
    ]
}

fn vector_norm(vector: Vector3) -> f64 {
    dot(vector, vector).sqrt()
}

fn position_norm(state: &CartesianState) -> f64 {
    (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt()
}

fn sample_coast(
    state: &mut CartesianState,
    duration: f64,
    intervals: usize,
) -> Result<f64, Gtoc1Error> {
    let interval_count = f64::from(
        u32::try_from(intervals).map_err(|_| Gtoc1Error::Numerical("solar-distance sampling"))?,
    );
    let step = duration / interval_count;
    let mut minimum_radius = position_norm(state);
    for _ in 0..intervals {
        *state = propagate_lagrangian(state, step, LEGACY_MU_SUN)?;
        minimum_radius = minimum_radius.min(position_norm(state));
    }
    Ok(minimum_radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_route_grammar_body_has_a_finite_competition_state() {
        for body in [2, 3, 5, 6, 10] {
            let state = competition_state(body, 8_998.0)
                .unwrap_or_else(|error| panic!("body {body} has no competition state: {error}"));
            assert!(
                state.iter().all(|value| value.is_finite()),
                "body {body} returned a non-finite state"
            );
            assert!(position_norm(&state) > 0.0, "body {body}");
        }
        assert!(competition_state(1, 8_998.0).is_err());
        assert!(competition_state(4, 8_998.0).is_err());
        assert!(competition_state(7, 8_998.0).is_err());
    }

    #[test]
    fn jpl_dates_respect_the_competition_window() {
        let epochs = encounter_epochs(&JPL_DECISION).unwrap();
        assert!(
            epochs
                .iter()
                .zip(JPL_ENCOUNTER_EPOCHS)
                .all(|(&actual, expected)| actual.to_bits() == expected.to_bits())
        );
        assert!(epochs[REAL_DIMENSION - 1] - epochs[0] <= MAXIMUM_FLIGHT_DAYS);
    }

    #[test]
    fn published_schedule_has_a_finite_ballistic_backbone() {
        let evaluation = evaluate_ballistic_backbone(&JPL_DECISION).unwrap();
        assert!(evaluation.objective.is_finite());
        assert!(evaluation.score > 0.0);
        assert_eq!(evaluation.branches.len(), REAL_DIMENSION - 1);
    }

    #[test]
    fn feasible_incumbent_clears_sampled_solar_exclusion() {
        let minimum = minimum_heliocentric_distance_au(&FEASIBLE_DECISION).unwrap();
        assert!(minimum >= MINIMUM_HELIOCENTRIC_DISTANCE_AU, "{minimum}");
    }

    #[test]
    fn stored_incumbent_beats_the_reference_inside_the_model() {
        let evaluation = evaluate_full(&FEASIBLE_DECISION).unwrap();
        assert!(evaluation.score > JPL_WINNING_SCORE);
        assert!(evaluation.objective < -JPL_WINNING_SCORE);
        assert!(evaluation.launch_v_infinity_km_s <= LAUNCH_V_INFINITY_KM_S);
        assert!(evaluation.low_thrust_mismatch_norm < 1.0e-8);
        assert!(evaluation.powered_delta_v_km_s < 1.0e-7);
        assert!(evaluation.minimum_periapsis_margin_km >= 0.0);
        assert!(
            evaluation.epochs_mjd2000[REAL_DIMENSION - 1] - evaluation.epochs_mjd2000[0]
                <= MAXIMUM_FLIGHT_DAYS
        );
    }
}
