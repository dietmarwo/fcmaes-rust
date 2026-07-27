// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Real GTOC1 EVEEEJSJA trajectory transcription.

use std::sync::OnceLock;

use fcmaes_core::{NAN_REPLACEMENT, RetryBounds};
use pykep_core::astro::elements::ClassicalElements;
use pykep_core::astro::lambert::{LambertPath, LambertProblem, LambertSolution};
use pykep_core::astro::propagation::propagate_lagrangian;
use pykep_core::ephemeris::{Ephemeris, KeplerianEphemeris, Vsop2013};
use pykep_core::leg::{SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint};
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
const VELOCITY_SCALE_M_S: f64 = 29_784.7;
const PENALTY: f64 = 1.0e15;
const SEGMENTS_F64: f64 = 24.0;
const LAUNCH_START_MJD2000: f64 = 3_653.0;
const LAUNCH_END_MJD2000: f64 = 10_958.0;
const MAX_FLIGHT_DAYS: f64 = 30.0 * 365.25;
const VSOP_THRESHOLD: f64 = 1.0e-9;
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

/// Highest-precision VSOP2013 solution found by the recorded campaign.
#[allow(clippy::unreadable_literal)]
pub const WINNING_DECISION: [f64; DIMENSION] = [
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
        let radius = fraction * (global.upper()[index] - global.lower()[index]);
        lower.push(global.lower()[index].max(value - radius));
        upper.push(global.upper()[index].min(value + radius));
    }
    RetryBounds::new(lower, upper).expect("refinement bounds are valid")
}

pub fn objective(x: &[f64]) -> f64 {
    evaluate(x).map_or(NAN_REPLACEMENT, |value| value.objective)
}

pub fn evaluate(x: &[f64]) -> Result<Evaluation, ModelError> {
    validate(x)?;
    let epochs = encounter_epochs(x)?;
    let states = competition_states(&epochs)?;
    let arcs = selected_arcs(x, &states)?;

    let (earth_position, earth_velocity) = split_state(states[0]);
    let (venus_position, venus_velocity) = split_state(states[1]);
    let departure_direction = spherical_direction(x[10], x[11]);
    let departure_v_infinity = departure_direction.map(|value| value * x[9] * 1_000.0);
    let departure_velocity = add(earth_velocity, departure_v_infinity);

    let venus_outgoing = subtract(arcs[0].departure_velocity, venus_velocity);
    let venus_incoming_direction = spherical_direction(x[12], x[13]);
    let venus_incoming = venus_incoming_direction.map(|value| value * vector_norm(venus_outgoing));
    let arrival_velocity = add(venus_velocity, venus_incoming);

    let throttles = throttles(x);
    let leg = SimsFlanaganLeg::new(
        SpacecraftEndpoint::new(
            join_state(earth_position, departure_velocity),
            INITIAL_MASS_KG,
        )?,
        throttles,
        SpacecraftEndpoint::new(join_state(venus_position, arrival_velocity), x[14])?,
        SimsFlanaganSettings::new(
            x[1] * DAY_SECONDS,
            MAX_THRUST_NEWTONS,
            EXHAUST_VELOCITY_M_S,
            MU_SUN,
            0.5,
        )?,
    )?;
    let mismatch = leg.mismatch_constraints()?;
    let normalized = [
        mismatch[0] / AU_METRES,
        mismatch[1] / AU_METRES,
        mismatch[2] / AU_METRES,
        mismatch[3] / VELOCITY_SCALE_M_S,
        mismatch[4] / VELOCITY_SCALE_M_S,
        mismatch[5] / VELOCITY_SCALE_M_S,
        mismatch[6] / INITIAL_MASS_KG,
    ];
    let low_thrust_constraint = normalized.iter().map(|value| value * value).sum::<f64>();

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
    Ok(Evaluation {
        objective: PENALTY * (low_thrust_constraint + gravity_constraint) - score,
        score,
        final_mass_kg: x[14],
        mismatch_norm: low_thrust_constraint.sqrt(),
        powered_delta_v_km_s,
        minimum_periapsis_margin_km,
        epochs_mjd2000: epochs,
    })
}

pub fn minimum_solar_distance_au(x: &[f64]) -> Result<f64, ModelError> {
    validate(x)?;
    let epochs = encounter_epochs(x)?;
    let states = competition_states(&epochs)?;
    let arcs = selected_arcs(x, &states)?;
    let (earth_position, earth_velocity) = split_state(states[0]);
    let direction = spherical_direction(x[10], x[11]);
    let departure_v_infinity = direction.map(|value| value * x[9] * 1_000.0);
    let mut state = join_state(earth_position, add(earth_velocity, departure_v_infinity));
    let mut mass = INITIAL_MASS_KG;
    let duration = x[1] * DAY_SECONDS / SEGMENTS_F64;
    let controls = throttles(x);
    let mut minimum = position_norm(&state);

    for (index, throttle) in controls.into_iter().enumerate() {
        let coast = if index == 0 { duration / 2.0 } else { duration };
        minimum = minimum.min(sample_coast(&mut state, coast, 32)?);
        let scale = MAX_THRUST_NEWTONS * duration / mass;
        let impulse = throttle.map(|value| value * scale);
        for component in 0..3 {
            state[component + 3] += impulse[component];
        }
        mass *= (-vector_norm(impulse) / EXHAUST_VELOCITY_M_S).exp();
    }
    minimum = minimum.min(sample_coast(&mut state, duration / 2.0, 32)?);

    for (index, arc) in arcs.iter().enumerate() {
        let (position, _) = split_state(states[index + 1]);
        let mut arc_state = join_state(position, arc.departure_velocity);
        minimum = minimum.min(sample_coast(
            &mut arc_state,
            x[index + 2] * DAY_SECONDS,
            512,
        )?);
    }
    Ok(minimum / AU_METRES)
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

fn throttles(x: &[f64]) -> Vec<Vector3> {
    (0..SEGMENTS)
        .map(|segment| {
            let offset = 15 + 3 * segment;
            spherical_direction(x[offset + 1], x[offset + 2]).map(|value| value * x[offset])
        })
        .collect()
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
    fn stored_decision_reproduces_the_model_score() {
        let evaluation = evaluate(&WINNING_DECISION).unwrap();
        assert!((evaluation.score - 1_850_730.667_522).abs() < 1.0e-6);
        assert!(evaluation.objective < -JPL_SCORE);
        assert!(evaluation.mismatch_norm < 1.0e-8);
        assert!(evaluation.powered_delta_v_km_s < 1.0e-7);
        assert!(evaluation.minimum_periapsis_margin_km >= 0.0);
        assert!(
            (LAUNCH_START_MJD2000..=LAUNCH_END_MJD2000).contains(&evaluation.epochs_mjd2000[0])
        );
        assert!(
            evaluation.epochs_mjd2000[ENCOUNTERS - 1] - evaluation.epochs_mjd2000[0]
                <= MAX_FLIGHT_DAYS
        );
        assert!(minimum_solar_distance_au(&WINNING_DECISION).unwrap() >= 0.2);
    }

    #[test]
    fn invalid_decision_maps_to_a_finite_optimizer_penalty() {
        assert!(evaluate(&[]).is_err());
        assert_eq!(objective(&[]).to_bits(), NAN_REPLACEMENT.to_bits());
    }

    #[test]
    fn competition_limits_and_flyby_tables_fail_safely() {
        let mut long_flight = WINNING_DECISION;
        long_flight[1..ENCOUNTERS].fill(1_400.0);
        assert!(encounter_epochs(&long_flight).is_err());
        assert!(flyby_body(10).is_err());
    }
}
