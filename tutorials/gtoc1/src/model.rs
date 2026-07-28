// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Real GTOC1 EVEEEJSJA trajectory transcription.

pub mod tour;

use std::sync::OnceLock;

use fcmaes_core::{NAN_REPLACEMENT, RetryBounds};
use pykep_core::astro::elements::ClassicalElements;
use pykep_core::astro::lambert::{LambertPath, LambertProblem, LambertSolution};
use pykep_core::astro::propagation::propagate_lagrangian;
use pykep_core::dynamics::zoh::ZohKeplerDynamics;
use pykep_core::ephemeris::{Ephemeris, KeplerianEphemeris, Vsop2013};
use pykep_core::integration::{IntegrationMethod, IntegratorOptions};
use pykep_core::leg::{ZohKeplerLeg, ZohLegHistory};
use pykep_core::time::epoch::Epoch;
use pykep_core::{CartesianState, PykepError, Vector3};

pub const ENCOUNTERS: usize = 9;
pub const SEGMENTS: usize = 24;
pub const DIMENSION: usize = 15 + 3 * SEGMENTS;
pub const JPL_SCORE: f64 = 1_850_000.0;

const DAY_SECONDS: f64 = 86_400.0;
const MU_SUN: f64 = 1.327_124_28e20;
const AU_METRES: f64 = 149_597_870_660.0;
const INITIAL_MASS_KG: f64 = 1_500.0;
const MAX_THRUST_NEWTONS: f64 = 0.04;
const EXHAUST_VELOCITY_M_S: f64 = 2_500.0 * 9.806_65;
const PENALTY: f64 = 1.0e15;
const LAUNCH_START_MJD2000: f64 = 3_653.0;
const LAUNCH_END_MJD2000: f64 = 10_958.0;
const MAX_FLIGHT_DAYS: f64 = 30.0 * 365.25;
const VSOP_THRESHOLD: f64 = 1.0e-9;
const ZOH_OPTIONS: IntegratorOptions = IntegratorOptions {
    relative_tolerance: 1.0e-12,
    absolute_tolerance: 1.0e-12,
    initial_step: None,
    maximum_step: None,
    maximum_steps: 100_000,
    maximum_rejections: 100,
};
const SEQUENCE: [usize; ENCOUNTERS] = [3, 2, 3, 3, 3, 5, 6, 5, 10];
const CLOCKWISE: [bool; ENCOUNTERS - 2] = [false, false, false, false, false, true, true];
const SELECTED_BRANCHES: [(usize, LambertPath); ENCOUNTERS - 2] = [
    (1, LambertPath::Left),
    (1, LambertPath::Right),
    (0, LambertPath::ZeroRevolution),
    (0, LambertPath::ZeroRevolution),
    (0, LambertPath::ZeroRevolution),
    (0, LambertPath::ZeroRevolution),
    (0, LambertPath::ZeroRevolution),
];

/// DOP853-validated continuous-thrust solution found by the recorded campaign.
#[allow(clippy::unreadable_literal)]
pub const VALIDATED_DECISION: [f64; DIMENSION] = [
    8997.959874875181,
    1278.1880791382773,
    950.1330771309185,
    1189.1056250493066,
    1755.6608780273355,
    486.28635911258914,
    482.45107984979796,
    3274.207960922852,
    543.5311811234408,
    2.49999999068422,
    1.8350983838195136,
    -2.149714309250333,
    2.2776000169664417,
    -2.0474097411006342,
    1436.6632590373858,
    0.33864396042052225,
    2.1227632850910587,
    -2.1426103604107336,
    0.06164988819486977,
    2.1244402680036085,
    -1.7944072679794727,
    0.9983038052308129,
    1.92072598491896,
    -2.460087741007464,
    0.06446621492461554,
    1.3940762700094296,
    -1.6541734807542525,
    0.0016341617927808955,
    0.27810055698843433,
    2.2090402308134,
    0.0104985500736876,
    0.10902804742518929,
    3.033348437986179,
    0.030773627961343898,
    0.7524834026329543,
    2.9004079607690803,
    0.005222001240443204,
    1.1153324901965644,
    2.0104614913901564,
    0.5040277333885848,
    1.970230968046777,
    -2.088221283009241,
    0.042493391354915235,
    1.6243158035978942,
    -2.799270488385496,
    0.9919618057176668,
    1.66441201480348,
    -2.4132721153399,
    0.9923693237616142,
    1.9358025447159763,
    -2.0679871378819743,
    0.029775920975839757,
    2.0520157345444976,
    2.954360034318847,
    0.055960958724143875,
    1.9607364454679337,
    2.676311671292894,
    0.0312762491083611,
    0.22583917160529193,
    2.927912603464237,
    0.026782106693429286,
    0.2911216799005454,
    2.9345178827135485,
    0.9958655113938998,
    1.8207496793446227,
    -2.197442655012731,
    0.9933325235674658,
    1.9729790017014326,
    -1.5749297191855995,
    0.08548090766245617,
    1.8211721336212048,
    -1.4531022795877964,
    0.043514189277955945,
    2.2730434866381826,
    -2.5779076103727507,
    0.04798643491786598,
    1.7136896076766812,
    -2.911098823519931,
    0.9969022351174172,
    1.7670466842739272,
    -2.476970233704083,
    0.9963360498941604,
    1.925946123033742,
    -1.7580274230098731,
    0.09118089137622148,
    2.0354845054896638,
    -0.6905077534416454,
];

#[derive(Debug)]
pub enum ModelError {
    Invalid(&'static str),
    Pykep(PykepError),
}

impl From<PykepError> for ModelError {
    fn from(value: PykepError) -> Self {
        Self::Pykep(value)
    }
}

#[derive(Clone, Debug)]
struct Arc {
    departure_velocity: Vector3,
    arrival_velocity: Vector3,
}

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub objective: f64,
    pub score: f64,
    pub final_mass_kg: f64,
    pub mismatch_norm: f64,
    pub powered_delta_v_km_s: f64,
    pub minimum_periapsis_margin_km: f64,
    pub epochs_mjd2000: [f64; ENCOUNTERS],
}

#[derive(Clone, Debug)]
pub struct Dop853Validation {
    pub taylor_mismatch_norm: f64,
    pub dop853_mismatch_norm: f64,
    pub maximum_backend_difference: f64,
    pub minimum_solar_distance_au: f64,
}

pub fn bounds() -> RetryBounds {
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
        2.5,
        core::f64::consts::PI,
        core::f64::consts::PI,
        core::f64::consts::PI,
        core::f64::consts::PI,
        INITIAL_MASS_KG,
    ];
    for _ in 0..SEGMENTS {
        lower.extend([0.0, 0.0, -core::f64::consts::PI]);
        upper.extend([1.0, core::f64::consts::PI, core::f64::consts::PI]);
    }
    RetryBounds::new(lower, upper).expect("constant bounds are valid")
}

pub fn refinement_bounds(incumbent: &[f64], fraction: f64) -> RetryBounds {
    let global = bounds();
    let mut lower = Vec::with_capacity(DIMENSION);
    let mut upper = Vec::with_capacity(DIMENSION);
    for (index, &value) in incumbent.iter().enumerate() {
        let radius = if index < 9 || index == 14 {
            value.abs().max(1.0) * 1.0e-14
        } else {
            fraction * (global.upper()[index] - global.lower()[index])
        };
        lower.push(global.lower()[index].max(value - radius));
        upper.push(global.upper()[index].min(value + radius));
    }
    RetryBounds::new(lower, upper).expect("refinement bounds are valid")
}

pub fn objective(x: &[f64]) -> f64 {
    evaluate(x).map_or(NAN_REPLACEMENT, |value| value.objective)
}

pub fn evaluate(x: &[f64]) -> Result<Evaluation, ModelError> {
    evaluate_with_method(x, IntegrationMethod::Taylor).map(|(evaluation, _)| evaluation)
}

fn evaluate_with_method(
    x: &[f64],
    method: IntegrationMethod,
) -> Result<(Evaluation, [f64; 7]), ModelError> {
    validate(x)?;
    let epochs = encounter_epochs(x)?;
    let states = competition_states(&epochs)?;
    let arcs = selected_arcs(x, &states)?;

    let (leg, venus_incoming, venus_outgoing) = zoh_leg(x, &states, &arcs, 0.5)?;
    let mismatch = leg.mismatch_constraints_with_method(method)?;
    let low_thrust_constraint = mismatch.iter().map(|value| value * value).sum::<f64>();

    let (venus_constraint, venus_delta_v, venus_margin) =
        gravity_assist_constraint(venus_incoming, venus_outgoing, 2)?;
    let mut gravity_constraint = venus_constraint;
    let mut powered_delta_v_km_s = venus_delta_v;
    let mut minimum_periapsis_margin_km = venus_margin;
    for leg_index in 1..arcs.len() {
        let (_, planet_velocity) = split_state(states[leg_index + 1]);
        let incoming = subtract(arcs[leg_index - 1].arrival_velocity, planet_velocity);
        let outgoing = subtract(arcs[leg_index].departure_velocity, planet_velocity);
        let body = SEQUENCE[leg_index + 1];
        let (constraint, delta_v, margin) = gravity_assist_constraint(incoming, outgoing, body)?;
        gravity_constraint += constraint;
        powered_delta_v_km_s += delta_v;
        minimum_periapsis_margin_km = minimum_periapsis_margin_km.min(margin);
    }

    let (_, asteroid_velocity) = split_state(states[ENCOUNTERS - 1]);
    let arrival_relative = subtract(asteroid_velocity, arcs[arcs.len() - 1].arrival_velocity);
    let score = x[14] * dot(arrival_relative, asteroid_velocity) / 1.0e6;
    Ok((
        Evaluation {
            objective: PENALTY * (low_thrust_constraint + gravity_constraint) - score,
            score,
            final_mass_kg: x[14],
            mismatch_norm: low_thrust_constraint.sqrt(),
            powered_delta_v_km_s,
            minimum_periapsis_margin_km,
            epochs_mjd2000: epochs,
        },
        mismatch,
    ))
}

pub fn dop853_validation(x: &[f64]) -> Result<Dop853Validation, ModelError> {
    let (taylor, taylor_mismatch) = evaluate_with_method(x, IntegrationMethod::Taylor)?;
    let (dop853, dop853_mismatch) = evaluate_with_method(x, IntegrationMethod::Dop853)?;
    let maximum_backend_difference = taylor_mismatch
        .iter()
        .zip(dop853_mismatch)
        .map(|(&left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    Ok(Dop853Validation {
        taylor_mismatch_norm: taylor.mismatch_norm,
        dop853_mismatch_norm: dop853.mismatch_norm,
        maximum_backend_difference,
        minimum_solar_distance_au: minimum_solar_distance_au(x)?,
    })
}

pub fn repair_zoh(initial: &[f64], iterations: usize) -> Result<Vec<f64>, ModelError> {
    let mut decision = initial.to_vec();
    let global = bounds();
    for _ in 0..iterations {
        let (_, residual) = evaluate_with_method(&decision, IntegrationMethod::Taylor)?;
        if vector_norm7(residual) <= 1.0e-10 {
            break;
        }
        let jacobian = zoh_spherical_jacobian(&decision)?;
        let mut normal = [[0.0; 7]; 7];
        for row in 0..7 {
            for column in 0..7 {
                normal[row][column] = jacobian
                    .iter()
                    .map(|derivative| derivative[row] * derivative[column])
                    .sum();
            }
        }
        let scale = (0..7).map(|index| normal[index][index]).sum::<f64>() / 7.0;
        for (index, row) in normal.iter_mut().enumerate() {
            row[index] += scale.max(1.0e-16) * 1.0e-10;
        }
        let multipliers =
            solve_seven(normal, residual).ok_or(ModelError::Invalid("ZOH repair Jacobian"))?;
        let correction = jacobian
            .iter()
            .map(|derivative| {
                -derivative
                    .iter()
                    .zip(multipliers)
                    .map(|(&left, right)| left * right)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let previous_norm = vector_norm7(residual);
        let mut accepted = None;
        let mut step = 1.0;
        for _ in 0..16 {
            let mut trial = decision.clone();
            for (column, &delta) in correction.iter().enumerate() {
                let index = 15 + column;
                let span = global.upper()[index] - global.lower()[index];
                trial[index] = (trial[index] + step * delta * span)
                    .clamp(global.lower()[index], global.upper()[index]);
            }
            let (_, trial_residual) = evaluate_with_method(&trial, IntegrationMethod::Taylor)?;
            if vector_norm7(trial_residual) < previous_norm {
                accepted = Some(trial);
                break;
            }
            step *= 0.5;
        }
        if let Some(trial) = accepted {
            decision = trial;
        } else {
            break;
        }
    }
    Ok(decision)
}

pub fn minimum_solar_distance_au(x: &[f64]) -> Result<f64, ModelError> {
    validate(x)?;
    let epochs = encounter_epochs(x)?;
    let states = competition_states(&epochs)?;
    let arcs = selected_arcs(x, &states)?;
    let (leg, _, _) = zoh_leg(x, &states, &arcs, 1.0)?;
    let samples_per_segment = ((x[1] / SEGMENTS as f64).ceil() as usize).saturating_add(1);
    let history = leg.state_history_with_method(samples_per_segment, IntegrationMethod::Dop853)?;
    let mut minimum = minimum_history_radius(&history);

    for (index, arc) in arcs.iter().enumerate() {
        let (position, _) = split_state(states[index + 1]);
        let mut arc_state = join_state(position, arc.departure_velocity);
        let daily_intervals = x[index + 2].ceil() as u32;
        minimum = minimum.min(
            sample_coast(&mut arc_state, x[index + 2] * DAY_SECONDS, daily_intervals)? / AU_METRES,
        );
    }
    Ok(minimum)
}

fn validate(x: &[f64]) -> Result<(), ModelError> {
    if x.len() != DIMENSION || x.iter().any(|value| !value.is_finite()) {
        return Err(ModelError::Invalid("decision vector"));
    }
    let global = bounds();
    if x.iter()
        .zip(global.lower().iter().zip(global.upper()))
        .any(|(&value, (&lower, &upper))| !(lower..=upper).contains(&value))
    {
        return Err(ModelError::Invalid("decision bounds"));
    }
    if !(LAUNCH_START_MJD2000..=LAUNCH_END_MJD2000).contains(&x[0]) {
        return Err(ModelError::Invalid("competition launch window"));
    }
    Ok(())
}

fn encounter_epochs(x: &[f64]) -> Result<[f64; ENCOUNTERS], ModelError> {
    let mut epochs = [0.0; ENCOUNTERS];
    epochs[0] = x[0];
    for index in 1..ENCOUNTERS {
        if x[index] <= 0.0 {
            return Err(ModelError::Invalid("leg duration"));
        }
        epochs[index] = epochs[index - 1] + x[index];
    }
    if epochs[ENCOUNTERS - 1] - epochs[0] > MAX_FLIGHT_DAYS {
        return Err(ModelError::Invalid("30-year flight limit"));
    }
    Ok(epochs)
}

fn selected_arcs(x: &[f64], states: &[CartesianState; ENCOUNTERS]) -> Result<Vec<Arc>, ModelError> {
    let mut arcs = Vec::with_capacity(ENCOUNTERS - 2);
    for leg in 1..ENCOUNTERS - 1 {
        let ballistic_leg = leg - 1;
        let (initial_position, _) = split_state(states[leg]);
        let (final_position, _) = split_state(states[leg + 1]);
        let branch = SELECTED_BRANCHES[ballistic_leg];
        let problem = LambertProblem::new(
            initial_position,
            final_position,
            x[leg + 1] * DAY_SECONDS,
            MU_SUN,
            CLOCKWISE[ballistic_leg],
            branch.0,
        )?;
        let solution = select_solution(problem.solutions(), branch)
            .ok_or(ModelError::Invalid("Lambert branch"))?;
        arcs.push(Arc {
            departure_velocity: solution.departure_velocity,
            arrival_velocity: solution.arrival_velocity,
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
) -> Result<(f64, f64, f64), ModelError> {
    let incoming_norm = vector_norm(incoming);
    let outgoing_norm = vector_norm(outgoing);
    if incoming_norm == 0.0 || outgoing_norm == 0.0 {
        return Err(ModelError::Invalid("gravity-assist velocity"));
    }
    let angle = (dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
        .clamp(-1.0, 1.0)
        .acos();
    let (delta_v, nondimensional_periapsis) =
        powered_swingby_inverse(incoming_norm / 1_000.0, outgoing_norm / 1_000.0, angle)?;
    let (body_mu, minimum_periapsis) = flyby_body(body)?;
    let periapsis = nondimensional_periapsis * body_mu;
    let margin = periapsis - minimum_periapsis;
    let shortfall = (-margin).max(0.0) / minimum_periapsis;
    Ok((delta_v * delta_v + shortfall * shortfall, delta_v, margin))
}

fn flyby_body(body: usize) -> Result<(f64, f64), ModelError> {
    match body {
        1 => Ok((22_321.0, 2_740.0)),
        2 => Ok((324_860.0, 6_351.0)),
        3 => Ok((398_601.19, 6_678.0)),
        4 => Ok((42_828.3, 3_689.0)),
        5 => Ok((126.7e6, 600_000.0)),
        6 => Ok((37.9e6, 70_000.0)),
        _ => Err(ModelError::Invalid("flyby body")),
    }
}

fn powered_swingby_inverse(
    incoming: f64,
    outgoing: f64,
    angle: f64,
) -> Result<(f64, f64), ModelError> {
    let incoming_axis = 1.0 / incoming.powi(2);
    let outgoing_axis = 1.0 / outgoing.powi(2);
    let mut periapsis = 1.0;
    for _ in 0..30 {
        let function = (incoming_axis / (incoming_axis + periapsis)).asin()
            + (outgoing_axis / (outgoing_axis + periapsis)).asin()
            - angle;
        let derivative = -incoming_axis
            / ((periapsis + 2.0 * incoming_axis) * periapsis).sqrt()
            / (incoming_axis + periapsis)
            - outgoing_axis
                / ((periapsis + 2.0 * outgoing_axis) * periapsis).sqrt()
                / (outgoing_axis + periapsis);
        let next = periapsis - function / derivative;
        if next > 0.0 {
            let error = (next - periapsis).abs();
            periapsis = next;
            if error <= 1.0e-8 {
                break;
            }
        } else {
            periapsis /= 2.0;
        }
    }
    let delta_v = ((outgoing.powi(2) + 2.0 / periapsis).sqrt()
        - (incoming.powi(2) + 2.0 / periapsis).sqrt())
    .abs();
    if delta_v.is_finite() && periapsis.is_finite() && periapsis > 0.0 {
        Ok((delta_v, periapsis))
    } else {
        Err(ModelError::Invalid("gravity-assist inverse"))
    }
}

fn zoh_leg(
    x: &[f64],
    states: &[CartesianState; ENCOUNTERS],
    arcs: &[Arc],
    cut: f64,
) -> Result<(ZohKeplerLeg, Vector3, Vector3), ModelError> {
    let (earth_position, earth_velocity) = split_state(states[0]);
    let (venus_position, venus_velocity) = split_state(states[1]);
    let departure_direction = spherical_direction(x[10], x[11]);
    let departure_v_infinity = departure_direction.map(|value| value * x[9] * 1_000.0);
    let departure_velocity = add(earth_velocity, departure_v_infinity);

    let venus_outgoing = subtract(arcs[0].departure_velocity, venus_velocity);
    let venus_incoming_direction = spherical_direction(x[12], x[13]);
    let venus_incoming = venus_incoming_direction.map(|value| value * vector_norm(venus_outgoing));
    let arrival_velocity = add(venus_velocity, venus_incoming);

    let time_scale = canonical_time_seconds();
    let duration = x[1] * DAY_SECONDS / time_scale;
    let time_grid = (0..=SEGMENTS)
        .map(|index| duration * index as f64 / SEGMENTS as f64)
        .collect();
    let leg = ZohKeplerLeg::new(
        ZohKeplerDynamics,
        normalized_endpoint(
            join_state(earth_position, departure_velocity),
            INITIAL_MASS_KG,
        ),
        zoh_controls(x),
        normalized_endpoint(join_state(venus_position, arrival_velocity), x[14]),
        time_grid,
        [AU_METRES / time_scale / EXHAUST_VELOCITY_M_S],
        cut,
        ZOH_OPTIONS,
    )?;
    Ok((leg, venus_incoming, venus_outgoing))
}

fn zoh_controls(x: &[f64]) -> Vec<[f64; 4]> {
    let time_scale = canonical_time_seconds();
    let maximum_thrust = MAX_THRUST_NEWTONS * time_scale * time_scale / INITIAL_MASS_KG / AU_METRES;
    (0..SEGMENTS)
        .map(|segment| {
            let offset = 15 + 3 * segment;
            let direction = spherical_direction(x[offset + 1], x[offset + 2]);
            [
                maximum_thrust * x[offset],
                direction[0],
                direction[1],
                direction[2],
            ]
        })
        .collect()
}

fn zoh_spherical_jacobian(x: &[f64]) -> Result<Vec<[f64; 7]>, ModelError> {
    let epochs = encounter_epochs(x)?;
    let states = competition_states(&epochs)?;
    let arcs = selected_arcs(x, &states)?;
    let (leg, _, _) = zoh_leg(x, &states, &arcs, 0.5)?;
    let cartesian = leg.mismatch_jacobian()?;
    let time_scale = canonical_time_seconds();
    let maximum_thrust = MAX_THRUST_NEWTONS * time_scale * time_scale / INITIAL_MASS_KG / AU_METRES;
    let global = bounds();
    let mut spherical = vec![[0.0; 7]; 3 * SEGMENTS];
    for segment in 0..SEGMENTS {
        let offset = 15 + 3 * segment;
        let theta = x[offset + 1];
        let phi = x[offset + 2];
        let (sin_theta, cos_theta) = theta.sin_cos();
        let (sin_phi, cos_phi) = phi.sin_cos();
        let direction_theta = [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta];
        let direction_phi = [-sin_theta * sin_phi, sin_theta * cos_phi, 0.0];
        for (row, controls) in cartesian.controls.iter().enumerate() {
            let control = &controls[4 * segment..4 * segment + 4];
            spherical[3 * segment][row] = control[0] * maximum_thrust;
            spherical[3 * segment + 1][row] = control[1] * direction_theta[0]
                + control[2] * direction_theta[1]
                + control[3] * direction_theta[2];
            spherical[3 * segment + 2][row] = control[1] * direction_phi[0]
                + control[2] * direction_phi[1]
                + control[3] * direction_phi[2];
        }
        for parameter in 0..3 {
            let index = offset + parameter;
            let span = global.upper()[index] - global.lower()[index];
            for derivative in &mut spherical[3 * segment + parameter] {
                *derivative *= span;
            }
        }
    }
    Ok(spherical)
}

fn solve_seven(mut matrix: [[f64; 7]; 7], mut rhs: [f64; 7]) -> Option<[f64; 7]> {
    for column in 0..7 {
        let pivot = (column..7).max_by(|&left, &right| {
            matrix[left][column]
                .abs()
                .total_cmp(&matrix[right][column].abs())
        })?;
        if matrix[pivot][column].abs() <= 1.0e-20 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        for row in column + 1..7 {
            let factor = matrix[row][column] / matrix[column][column];
            let pivot_row = matrix[column];
            for (target, &source) in matrix[row][column..].iter_mut().zip(&pivot_row[column..]) {
                *target -= factor * source;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = [0.0; 7];
    for row in (0..7).rev() {
        let tail = (row + 1..7)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

fn vector_norm7(vector: [f64; 7]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn normalized_endpoint(state: CartesianState, mass_kg: f64) -> [f64; 7] {
    let velocity_scale = AU_METRES / canonical_time_seconds();
    [
        state[0] / AU_METRES,
        state[1] / AU_METRES,
        state[2] / AU_METRES,
        state[3] / velocity_scale,
        state[4] / velocity_scale,
        state[5] / velocity_scale,
        mass_kg / INITIAL_MASS_KG,
    ]
}

fn canonical_time_seconds() -> f64 {
    (AU_METRES.powi(3) / MU_SUN).sqrt()
}

fn minimum_history_radius(history: &ZohLegHistory<7>) -> f64 {
    history
        .forward
        .iter()
        .chain(&history.backward)
        .flatten()
        .map(|state| (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt())
        .fold(f64::INFINITY, f64::min)
}

fn competition_states(
    epochs: &[f64; ENCOUNTERS],
) -> Result<[CartesianState; ENCOUNTERS], ModelError> {
    let mut states = [[0.0; 6]; ENCOUNTERS];
    for index in 0..ENCOUNTERS {
        states[index] = match SEQUENCE[index] {
            2 => venus().state(epochs[index])?,
            3 => earth().state(epochs[index])?,
            5 => jupiter().state(epochs[index])?,
            6 => saturn().state(epochs[index])?,
            10 => rotate_ecliptic_to_icrf(asteroid().state(epochs[index])?),
            _ => return Err(ModelError::Invalid("body sequence")),
        };
    }
    Ok(states)
}

fn venus() -> &'static Vsop2013 {
    static VALUE: OnceLock<Vsop2013> = OnceLock::new();
    VALUE.get_or_init(|| {
        Vsop2013::with_threshold("venus", VSOP_THRESHOLD).expect("VSOP2013 Venus is available")
    })
}

fn earth() -> &'static Vsop2013 {
    static VALUE: OnceLock<Vsop2013> = OnceLock::new();
    VALUE.get_or_init(|| {
        Vsop2013::with_threshold("earth_moon", VSOP_THRESHOLD)
            .expect("VSOP2013 Earth-Moon is available")
    })
}

fn jupiter() -> &'static Vsop2013 {
    static VALUE: OnceLock<Vsop2013> = OnceLock::new();
    VALUE.get_or_init(|| {
        Vsop2013::with_threshold("jupiter", VSOP_THRESHOLD).expect("VSOP2013 Jupiter is available")
    })
}

fn saturn() -> &'static Vsop2013 {
    static VALUE: OnceLock<Vsop2013> = OnceLock::new();
    VALUE.get_or_init(|| {
        Vsop2013::with_threshold("saturn", VSOP_THRESHOLD).expect("VSOP2013 Saturn is available")
    })
}

fn asteroid() -> &'static KeplerianEphemeris {
    static VALUE: OnceLock<KeplerianEphemeris> = OnceLock::new();
    VALUE.get_or_init(|| {
        KeplerianEphemeris::from_classical_mean(
            Epoch::from_mjd2000(2_056.0).expect("constant epoch is valid"),
            ClassicalElements::new(
                2.589_726_1 * AU_METRES,
                0.273_462_5,
                6.407_34_f64.to_radians(),
                128.347_11_f64.to_radians(),
                264.786_91_f64.to_radians(),
                320.479_555_f64.to_radians(),
            ),
            MU_SUN,
            "GTOC1 asteroid",
            None,
            None,
            None,
        )
        .expect("constant asteroid elements are valid")
    })
}

fn rotate_ecliptic_to_icrf(state: CartesianState) -> CartesianState {
    let epsilon: f64 = 0.409_092_626_586_596_2;
    let phi: f64 = -2.515_213_377_596_228_5e-7;
    let (sin_epsilon, cos_epsilon) = epsilon.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();
    let rotate = |vector: Vector3| {
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

fn sample_coast(
    state: &mut CartesianState,
    duration: f64,
    intervals: u32,
) -> Result<f64, ModelError> {
    let step = duration / f64::from(intervals);
    let mut minimum = position_norm(state);
    for _ in 0..intervals {
        *state = propagate_lagrangian(state, step, MU_SUN)?;
        minimum = minimum.min(position_norm(state));
    }
    Ok(minimum)
}

fn spherical_direction(theta: f64, phi: f64) -> Vector3 {
    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();
    [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
}

fn split_state(state: CartesianState) -> (Vector3, Vector3) {
    (
        [state[0], state[1], state[2]],
        [state[3], state[4], state[5]],
    )
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

fn add(left: Vector3, right: Vector3) -> Vector3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn subtract(left: Vector3, right: Vector3) -> Vector3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: Vector3, right: Vector3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn vector_norm(vector: Vector3) -> f64 {
    dot(vector, vector).sqrt()
}

fn position_norm(state: &CartesianState) -> f64 {
    (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt()
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(operation) => write!(formatter, "invalid {operation}"),
            Self::Pykep(error) => write!(formatter, "pykep evaluation failed: {error}"),
        }
    }
}

impl std::error::Error for ModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_decision_reproduces_the_continuous_model_score() {
        let evaluation = evaluate(&VALIDATED_DECISION).unwrap();
        assert!((evaluation.score - 1_843_300.529_365).abs() < 1.0e-6);
        assert!(evaluation.score < JPL_SCORE);
        assert!(evaluation.mismatch_norm < 1.0e-10);
        assert!(evaluation.powered_delta_v_km_s < 1.0e-7);
        assert!(evaluation.minimum_periapsis_margin_km >= 0.0);
        assert!(
            (LAUNCH_START_MJD2000..=LAUNCH_END_MJD2000).contains(&evaluation.epochs_mjd2000[0])
        );
        assert!(
            evaluation.epochs_mjd2000[ENCOUNTERS - 1] - evaluation.epochs_mjd2000[0]
                <= MAX_FLIGHT_DAYS
        );
        let validation = dop853_validation(&VALIDATED_DECISION).unwrap();
        assert!(validation.dop853_mismatch_norm < 1.0e-9);
        assert!(validation.maximum_backend_difference < 1.0e-9);
        assert!(validation.minimum_solar_distance_au >= 0.2);
    }

    #[test]
    fn invalid_decision_maps_to_a_finite_optimizer_penalty() {
        assert!(evaluate(&[]).is_err());
        assert_eq!(objective(&[]).to_bits(), NAN_REPLACEMENT.to_bits());
    }

    #[test]
    fn competition_limits_and_flyby_tables_fail_safely() {
        let mut long_flight = VALIDATED_DECISION;
        long_flight[1..ENCOUNTERS].fill(1_400.0);
        assert!(encounter_epochs(&long_flight).is_err());
        assert!(flyby_body(10).is_err());
    }
}
