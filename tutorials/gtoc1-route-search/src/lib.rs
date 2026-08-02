// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! GTOC1 asteroid-impact objective backed by the published `pykep-core` crate.
//!
//! This is a safe Rust reimplementation of `GTOPtoolbox/trajobjfuns.cpp`
//! `gtoc1()` and the MGA path it calls. The legacy GTOP ephemeris coefficients
//! and objective units are retained so the optimization problem does not
//! change. `pykep-core` performs element conversion, asteroid propagation, and
//! Lambert solution.

#![allow(clippy::excessive_precision)] // Preserve the GTOP toolbox constants.

/// Full low-thrust validation for alternate competition planet sequences.
pub mod low_thrust_sequences;
/// Impulsive MGA screening with the GTOC1 asteroid-impact terminal objective.
pub mod mga;
/// Competition-faithful EVEEEJSJA trajectory model.
pub mod real;
/// Bounded agent subprocess, mock, replay, and JSON protocol.
pub mod route_agent;
/// Persistent route archive, niche elites, and proposal-event records.
pub mod route_archive;
/// Equal-budget MGA campaign, random baseline, and route evolutionary search.
pub mod route_campaign;
/// Variable-length route grammar, sampling, mutation, and edit distance.
pub mod route_grammar;
/// Budgeted L1 Sims–Flanagan promotion of archived Lambert routes.
pub mod route_refine;
/// Phase-0 runtime route model, identities, diagnostics, and persistence.
pub mod route_search;
/// Fast multi-revolution Lambert scouts for alternate GTOC1 sequences.
pub mod sequences;
/// Whole-tour Taylor/DOP853 ZOH experiments for alternate planet sequences.
pub mod zoh_tour_sequences;

use std::fmt;
use std::sync::OnceLock;

use fcmaes_core::{NAN_REPLACEMENT, RetryBounds};
use pykep_core::astro::anomalies::mean_to_true_anomaly;
use pykep_core::astro::elements::{ClassicalElements, classical_to_cartesian};
use pykep_core::astro::lambert::LambertProblem;
use pykep_core::ephemeris::{Ephemeris, KeplerianEphemeris};
use pykep_core::time::epoch::Epoch;
use pykep_core::{CartesianState, PykepError, Vector3};

/// Number of GTOC1 decision variables.
pub const DIMENSION: usize = 8;

/// Lower decision bounds from the original GTOP benchmark.
pub const LOWER_BOUNDS: [f64; DIMENSION] = [3_000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0];

/// Upper decision bounds from the original GTOP benchmark.
pub const UPPER_BOUNDS: [f64; DIMENSION] = [
    10_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 9_000.0, 9_000.0, 9_000.0,
];

/// Published best-known objective for the original benchmark.
pub const BEST_KNOWN_OBJECTIVE: f64 = -1_581_950.0;

/// Relaxed stopping value used by the fcmaes GTOP benchmarks.
pub const STOP_OBJECTIVE: f64 = -1_574_080.0;

const SEQUENCE: [usize; DIMENSION] = [3, 2, 3, 2, 3, 5, 6, 10];
const CLOCKWISE: [bool; DIMENSION] = [false, false, false, false, false, false, true, false];
const DAY_SECONDS: f64 = 86_400.0;
const LEGACY_AU_METRES: f64 = 149_597_870_660.0;
const LEGACY_MU_SUN: f64 = 1.327_124_28e20;
const INITIAL_MASS_KG: f64 = 1_500.0;
const SPECIFIC_IMPULSE_SECONDS: f64 = 2_500.0;
const STANDARD_GRAVITY_KM_S2: f64 = 9.806_65 / 1_000.0;
const LAUNCH_DELTA_V_KM_S: f64 = 2.5;

// The original MGA routine deliberately used these rounded km³/s² values.
const BODY_MU_KM3_S2: [f64; 9] = [
    1.327_124_28e11,
    22_321.0,
    324_860.0,
    398_601.19,
    42_828.3,
    126.7e6,
    37.9e6,
    5.78e6,
    6.8e6,
];
const MINIMUM_PERIAPSIS_KM: [f64; 9] = [
    0.0, 0.0, 6_351.8, 6_778.1, 6_000.0, 600_000.0, 70_000.0, 0.0, 0.0,
];
const PERIAPSIS_PENALTY: [f64; 9] = [0.0, 0.0, 0.01, 0.01, 0.01, 0.001, 0.01, 0.0, 0.0];

/// Error returned when a decision cannot be evaluated.
#[derive(Debug)]
pub enum Gtoc1Error {
    /// Decision vector has the wrong dimension.
    Dimension {
        /// Actual decision-vector length.
        actual: usize,
    },
    /// A decision coordinate is non-finite or outside its box bound.
    InvalidDecision {
        /// Invalid coordinate index.
        index: usize,
        /// Invalid coordinate value.
        value: f64,
    },
    /// An underlying pykep computation failed.
    Pykep(PykepError),
    /// A derived trajectory quantity was non-finite.
    Numerical(&'static str),
}

impl fmt::Display for Gtoc1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimension { actual } => {
                write!(
                    formatter,
                    "expected {DIMENSION} decisions, received {actual}"
                )
            }
            Self::InvalidDecision { index, value } => {
                write!(formatter, "invalid decision x[{index}]={value}")
            }
            Self::Pykep(error) => write!(formatter, "pykep evaluation failed: {error}"),
            Self::Numerical(operation) => {
                write!(formatter, "non-finite result in {operation}")
            }
        }
    }
}

impl std::error::Error for Gtoc1Error {}

impl From<PykepError> for Gtoc1Error {
    fn from(error: PykepError) -> Self {
        Self::Pykep(error)
    }
}

/// Complete diagnostics corresponding to the C++ objective and `rp` output.
#[derive(Clone, Debug, PartialEq)]
pub struct Gtoc1Evaluation {
    /// Minimized asteroid-impact objective.
    pub objective: f64,
    /// Six powered-flyby periapsis radii in kilometres.
    pub periapsis_radii_km: [f64; DIMENSION - 2],
    /// Six powered-flyby impulses in kilometres per second.
    pub flyby_delta_v_km_s: [f64; DIMENSION - 2],
    /// Hyperbolic launch excess above Earth in kilometres per second.
    pub launch_delta_v_km_s: f64,
    /// Penalized delta-v charged to the spacecraft mass model.
    pub penalized_delta_v_km_s: f64,
    /// Spacecraft mass at asteroid impact in kilograms.
    pub final_mass_kg: f64,
}

/// Constructs the validated box bounds used by both retry strategies.
///
/// # Panics
///
/// Panics only if the compile-time bound constants are invalid.
#[must_use]
pub fn bounds() -> RetryBounds {
    RetryBounds::new(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec())
        .expect("GTOC1 constants define valid bounds")
}

/// Evaluates GTOC1 and writes the six periapsis radii into `rp`.
///
/// This mirrors the original C++ signature. Invalid trajectories return
/// [`NAN_REPLACEMENT`] so minimizers can continue, and fill `rp` with NaNs.
#[must_use]
pub fn gtoc1(x: &[f64], rp: &mut Vec<f64>) -> f64 {
    if let Ok(evaluation) = evaluate_gtoc1(x) {
        rp.clear();
        rp.extend(evaluation.periapsis_radii_km);
        evaluation.objective
    } else {
        rp.clear();
        rp.resize(DIMENSION - 2, f64::NAN);
        NAN_REPLACEMENT
    }
}

/// Evaluates only the scalar objective for optimizer callbacks.
#[must_use]
pub fn gtoc1_objective(x: &[f64]) -> f64 {
    evaluate_gtoc1(x).map_or(NAN_REPLACEMENT, |evaluation| evaluation.objective)
}

/// Evaluates the full GTOC1 trajectory and diagnostics.
///
/// # Errors
///
/// Returns an error for invalid decisions, failed ephemeris/Lambert
/// calculations, singular flyby geometry, or non-finite derived values.
pub fn evaluate_gtoc1(x: &[f64]) -> Result<Gtoc1Evaluation, Gtoc1Error> {
    validate_decision(x)?;

    let mut positions = [[0.0; 3]; DIMENSION];
    let mut velocities = [[0.0; 3]; DIMENSION];
    let mut epoch = 0.0;
    for index in 0..DIMENSION {
        epoch += x[index];
        let state = if SEQUENCE[index] == 10 {
            asteroid().state(epoch)?
        } else {
            legacy_planet_state(epoch, SEQUENCE[index])?
        };
        (positions[index], velocities[index]) = split_state(state);
    }

    let mut previous = solve_leg(positions[0], positions[1], x[1] * DAY_SECONDS, CLOCKWISE[0])?;
    let launch_delta_v_km_s = distance(previous.0, velocities[0]) / 1_000.0;
    let mut periapsis_radii_km = [0.0; DIMENSION - 2];
    let mut flyby_delta_v_km_s = [0.0; DIMENSION - 2];

    for index in 1..=DIMENSION - 2 {
        let next = solve_leg(
            positions[index],
            positions[index + 1],
            x[index + 1] * DAY_SECONDS,
            CLOCKWISE[index],
        )?;
        let incoming_relative = subtract(previous.1, velocities[index]);
        let outgoing_relative = subtract(next.0, velocities[index]);
        let incoming_speed = norm(incoming_relative) / 1_000.0;
        let outgoing_speed = norm(outgoing_relative) / 1_000.0;
        if incoming_speed == 0.0 || outgoing_speed == 0.0 {
            return Err(Gtoc1Error::Numerical("powered flyby relative speed"));
        }
        let cosine =
            (dot(incoming_relative, outgoing_relative) / 1.0e6 / (incoming_speed * outgoing_speed))
                .clamp(-1.0, 1.0);
        let angle = cosine.acos();
        let (delta_v, nondimensional_periapsis) =
            powered_swingby_inverse(incoming_speed, outgoing_speed, angle)?;
        flyby_delta_v_km_s[index - 1] = delta_v;
        periapsis_radii_km[index - 1] = nondimensional_periapsis * BODY_MU_KM3_S2[SEQUENCE[index]];
        previous = next;
    }

    let mut penalized_delta_v_km_s = flyby_delta_v_km_s.iter().sum::<f64>();
    for (index, &periapsis) in periapsis_radii_km.iter().enumerate() {
        let body = SEQUENCE[index + 1];
        if periapsis < MINIMUM_PERIAPSIS_KM[body] {
            penalized_delta_v_km_s +=
                PERIAPSIS_PENALTY[body] * (periapsis - MINIMUM_PERIAPSIS_KM[body]).abs();
        }
    }
    if launch_delta_v_km_s > LAUNCH_DELTA_V_KM_S {
        penalized_delta_v_km_s += launch_delta_v_km_s - LAUNCH_DELTA_V_KM_S;
    }

    let final_mass_kg = INITIAL_MASS_KG
        * (-penalized_delta_v_km_s / (SPECIFIC_IMPULSE_SECONDS * STANDARD_GRAVITY_KM_S2)).exp();
    let relative_at_impact_km_s = scale(
        subtract(velocities[DIMENSION - 1], previous.1),
        1.0 / 1_000.0,
    );
    let asteroid_velocity_km_s = scale(velocities[DIMENSION - 1], 1.0 / 1_000.0);
    let objective = -final_mass_kg * dot(relative_at_impact_km_s, asteroid_velocity_km_s).abs();

    let evaluation = Gtoc1Evaluation {
        objective,
        periapsis_radii_km,
        flyby_delta_v_km_s,
        launch_delta_v_km_s,
        penalized_delta_v_km_s,
        final_mass_kg,
    };
    if [
        evaluation.objective,
        evaluation.launch_delta_v_km_s,
        evaluation.penalized_delta_v_km_s,
        evaluation.final_mass_kg,
    ]
    .into_iter()
    .chain(evaluation.periapsis_radii_km)
    .chain(evaluation.flyby_delta_v_km_s)
    .all(f64::is_finite)
    {
        Ok(evaluation)
    } else {
        Err(Gtoc1Error::Numerical("GTOC1 objective"))
    }
}

fn validate_decision(x: &[f64]) -> Result<(), Gtoc1Error> {
    if x.len() != DIMENSION {
        return Err(Gtoc1Error::Dimension { actual: x.len() });
    }
    for (index, ((&value, &lower), &upper)) in x
        .iter()
        .zip(LOWER_BOUNDS.iter())
        .zip(UPPER_BOUNDS.iter())
        .enumerate()
    {
        if !value.is_finite() || !(lower..=upper).contains(&value) {
            return Err(Gtoc1Error::InvalidDecision { index, value });
        }
    }
    Ok(())
}

fn asteroid() -> &'static KeplerianEphemeris {
    static ASTEROID: OnceLock<KeplerianEphemeris> = OnceLock::new();
    ASTEROID.get_or_init(|| {
        KeplerianEphemeris::from_classical_mean(
            Epoch::from_mjd2000(2_056.0).expect("constant epoch is valid"),
            ClassicalElements::new(
                2.589_726_1 * LEGACY_AU_METRES,
                0.273_462_5,
                6.407_34_f64.to_radians(),
                128.347_11_f64.to_radians(),
                264.786_91_f64.to_radians(),
                320.479_555_f64.to_radians(),
            ),
            LEGACY_MU_SUN,
            "GTOC1 asteroid",
            None,
            None,
            None,
        )
        .expect("constant asteroid elements are valid")
    })
}

fn legacy_planet_state(epoch_mjd2000: f64, planet: usize) -> Result<CartesianState, Gtoc1Error> {
    let century = (epoch_mjd2000 + 36_525.0) / 36_525.0;
    let t2 = century * century;
    let t3 = t2 * century;
    let (axis_au, eccentricity, inclination, node, periapsis, mean) = match planet {
        2 => {
            let rate = 58_517.803_875 + 1.286_055_555_555_555_5e-3 * century;
            (
                0.723_331_60,
                0.006_820_690 - 0.000_047_740 * century + 0.000_000_091 * t2,
                3.393_630_555_555_555_6 + 1.005_833_333_333_333_4e-3 * century
                    - 9.722_222_222_222_222e-7 * t2,
                75.779_647_222_222_22 + 0.899_85 * century + 4.1e-4 * t2,
                54.384_186_111_111_11 + 0.508_186_111_111_111_1 * century
                    - 1.386_388_888_888_889e-3 * t2,
                212.603_219_444_444_44 + rate * century,
            )
        }
        3 => {
            let rate = 35_999.049_75
                - 1.502_777_777_777_777_8e-4 * century
                - 3.333_333_333_333_333_3e-6 * t2;
            (
                1.000_000_230,
                0.016_751_040 - 0.000_041_800 * century - 0.000_000_126 * t2,
                0.0,
                0.0,
                101.220_833_333_333_33
                    + 1.719_175 * century
                    + 4.527_777_777_777_778e-4 * t2
                    + 3.333_333_333_333_333_3e-6 * t3,
                358.475_844_444_444_44 + rate * century,
            )
        }
        5 => {
            let rate = 3_034.692_023_888_889 - 7.215_888_888_888_889e-4 * century
                + 1.784_444_444_444_444_4e-6 * t2;
            (
                5.202_561,
                0.048_334_750 + 0.000_164_180 * century
                    - 0.000_000_467_60 * t2
                    - 0.000_000_001_70 * t3,
                1.308_736_111_111_111 - 5.696_111_111_111_111e-3 * century
                    + 3.888_888_888_889e-6 * t2,
                99.443_386_111_111_11 + 1.010_530 * century + 3.522_222_222_222_222e-4 * t2
                    - 8.511_111_111_111_111e-6 * t3,
                273.277_541_666_666_67
                    + 0.599_431_666_666_666_7 * century
                    + 7.040_5e-4 * t2
                    + 5.077_777_777_778e-6 * t3,
                225.328_327_777_777_78 + rate * century,
            )
        }
        6 => {
            let rate = 1_221.551_467_777_777_8
                - 5.018_194_444_444_444e-4 * century
                - 5.194_444_444_445e-6 * t2;
            (
                9.554_747,
                0.055_892_320 - 0.000_345_50 * century - 0.000_000_728 * t2
                    + 0.000_000_000_740 * t3,
                2.492_519_444_444_444_4
                    - 3.918_888_888_889e-3 * century
                    - 1.548_888_888_888_888_9e-5 * t2
                    + 4.444_444_444_444_444_4e-8 * t3,
                112.790_388_888_888_89 + 0.873_195_138_888_888_9 * century
                    - 1.521_805_555_555_555_6e-4 * t2
                    - 5.305_555_555_556e-6 * t3,
                338.307_772_222_222_2
                    + 1.085_220_694_444_444_4 * century
                    + 9.785_416_666_666_667e-4 * t2
                    + 9.916_666_666_667e-6 * t3,
                175.466_216_666_666_67 + rate * century,
            )
        }
        _ => return Err(Gtoc1Error::Numerical("unsupported legacy planet")),
    };
    let mean_anomaly = mean.to_radians().rem_euclid(std::f64::consts::TAU);
    let true_anomaly = mean_to_true_anomaly(mean_anomaly, eccentricity)?;
    Ok(classical_to_cartesian(
        ClassicalElements::new(
            axis_au * LEGACY_AU_METRES,
            eccentricity,
            inclination.to_radians(),
            node.to_radians(),
            periapsis.to_radians(),
            true_anomaly,
        ),
        LEGACY_MU_SUN,
    )?)
}

fn solve_leg(
    initial_position: Vector3,
    final_position: Vector3,
    time_seconds: f64,
    clockwise: bool,
) -> Result<(Vector3, Vector3), Gtoc1Error> {
    let problem = LambertProblem::new(
        initial_position,
        final_position,
        time_seconds,
        LEGACY_MU_SUN,
        clockwise,
        0,
    )?;
    let solution = problem
        .solutions()
        .first()
        .ok_or(Gtoc1Error::Numerical("empty Lambert solution"))?;
    Ok((solution.departure_velocity, solution.arrival_velocity))
}

fn powered_swingby_inverse(
    incoming: f64,
    outgoing: f64,
    angle: f64,
) -> Result<(f64, f64), Gtoc1Error> {
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
        Err(Gtoc1Error::Numerical("powered swingby inverse"))
    }
}

#[inline]
fn split_state(state: CartesianState) -> (Vector3, Vector3) {
    (
        [state[0], state[1], state[2]],
        [state[3], state[4], state[5]],
    )
}

#[inline]
fn dot(left: Vector3, right: Vector3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[inline]
fn norm(vector: Vector3) -> f64 {
    dot(vector, vector).sqrt()
}

#[inline]
fn distance(left: Vector3, right: Vector3) -> f64 {
    norm(subtract(left, right))
}

#[inline]
fn subtract(left: Vector3, right: Vector3) -> Vector3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

#[inline]
fn scale(vector: Vector3, factor: f64) -> Vector3 {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn midpoint() -> [f64; DIMENSION] {
        std::array::from_fn(|index| LOWER_BOUNDS[index].midpoint(UPPER_BOUNDS[index]))
    }

    #[test]
    fn midpoint_is_finite_and_matches_wrapper() {
        let decision = midpoint();
        let evaluation = evaluate_gtoc1(&decision).unwrap();
        let mut periapses = Vec::new();
        assert_eq!(
            gtoc1(&decision, &mut periapses).to_bits(),
            evaluation.objective.to_bits()
        );
        assert_eq!(periapses, evaluation.periapsis_radii_km);
        assert!(evaluation.objective.is_finite() && evaluation.objective <= 0.0);
        assert!(
            evaluation
                .periapsis_radii_km
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
    }

    #[test]
    fn representative_value_matches_the_cpp_objective() {
        let decision = [
            6_882.093_509_110_844,
            140.741_070_622_892_2,
            1_657.675_507_577_268,
            1_268.485_496_656_420_8,
            1_519.562_251_809_552_4,
            3_255.281_116_355_828_5,
            8_747.006_742_625_596,
            8_070.153_755_503_12,
        ];
        let expected_cpp = -2_307.571_478_895_144_5;
        let actual = gtoc1_objective(&decision);
        let relative_error = (actual - expected_cpp).abs() / expected_cpp.abs();
        assert!(
            relative_error < 1.0e-6,
            "Rust={actual}, C++={expected_cpp}, relative error={relative_error}"
        );
    }

    #[test]
    fn invalid_decisions_return_a_finite_optimizer_penalty() {
        assert!(matches!(
            evaluate_gtoc1(&[]),
            Err(Gtoc1Error::Dimension { actual: 0 })
        ));
        let mut decision = midpoint();
        decision[3] = f64::NAN;
        assert!(matches!(
            evaluate_gtoc1(&decision),
            Err(Gtoc1Error::InvalidDecision { index: 3, .. })
        ));
        assert_eq!(
            gtoc1_objective(&decision).to_bits(),
            NAN_REPLACEMENT.to_bits()
        );
    }
}
