// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Whole-tour continuous-thrust transcription with a coarse ZOH mesh per leg.

use super::*;
use pykep_core::astro::flyby::flyby_outgoing_velocity;
use pykep_core::dynamics::zoh::{
    ControlSchedule, ZohKeplerDynamics, propagate_schedule_with_method,
};

const LEGS: usize = ENCOUNTERS - 1;
const FLYBYS: usize = ENCOUNTERS - 2;
const HEADER: usize = 9 + 3 + 2 * FLYBYS;
const MAX_PERIAPSIS_FACTOR: f64 = 200.0;

#[derive(Clone, Debug)]
pub struct TourEvaluation {
    pub objective: f64,
    pub score: f64,
    pub final_mass_kg: f64,
    pub position_mismatch_norm: f64,
    pub maximum_position_mismatch: f64,
}

#[derive(Clone, Debug)]
pub struct TourValidation {
    pub taylor: TourEvaluation,
    pub dop853: TourEvaluation,
    pub maximum_backend_difference: f64,
}

pub const fn dimension(segments_per_leg: usize) -> usize {
    HEADER + 3 * LEGS * segments_per_leg
}

pub fn bounds(segments_per_leg: usize) -> RetryBounds {
    assert!((5..=8).contains(&segments_per_leg));
    let reference = super::bounds();
    let mut lower = reference.lower()[..9].to_vec();
    let mut upper = reference.upper()[..9].to_vec();
    lower.extend([0.0, 0.0, -core::f64::consts::PI]);
    upper.extend([2.5, core::f64::consts::PI, core::f64::consts::PI]);
    for _ in 0..FLYBYS {
        lower.extend([0.0, -core::f64::consts::PI]);
        upper.extend([1.0, core::f64::consts::PI]);
    }
    for _ in 0..LEGS * segments_per_leg {
        lower.extend([0.0, 0.0, -core::f64::consts::PI]);
        upper.extend([1.0, core::f64::consts::PI, core::f64::consts::PI]);
    }
    RetryBounds::new(lower, upper).expect("whole-tour bounds are valid")
}

pub fn objective(x: &[f64], segments_per_leg: usize) -> f64 {
    evaluate(x, segments_per_leg).map_or(NAN_REPLACEMENT, |value| value.objective)
}

pub fn evaluate(x: &[f64], segments_per_leg: usize) -> Result<TourEvaluation, ModelError> {
    evaluate_with_method(x, segments_per_leg, IntegrationMethod::Taylor)
        .map(|(evaluation, _)| evaluation)
}

pub fn validate_backends(x: &[f64], segments_per_leg: usize) -> Result<TourValidation, ModelError> {
    let (taylor, taylor_residual) =
        evaluate_with_method(x, segments_per_leg, IntegrationMethod::Taylor)?;
    let (dop853, dop853_residual) =
        evaluate_with_method(x, segments_per_leg, IntegrationMethod::Dop853)?;
    let maximum_backend_difference = taylor_residual
        .iter()
        .zip(dop853_residual)
        .map(|(&left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    Ok(TourValidation {
        taylor,
        dop853,
        maximum_backend_difference,
    })
}

pub fn seed(segments_per_leg: usize) -> Result<Vec<f64>, ModelError> {
    if !(5..=8).contains(&segments_per_leg) {
        return Err(ModelError::Invalid("segments per leg"));
    }
    let reference = &VALIDATED_DECISION;
    let epochs = encounter_epochs(reference)?;
    let states = competition_states(&epochs)?;
    let arcs = selected_arcs(reference, &states)?;
    let (reference_leg, _, _) = zoh_leg(reference, &states, &arcs, 1.0)?;
    let first_mismatch =
        reference_leg.mismatch_constraints_with_method(IntegrationMethod::Taylor)?;
    let first_target = reference_leg.final_state();
    let first_arrival_normalized: [f64; 7] =
        core::array::from_fn(|index| first_target[index] + first_mismatch[index]);
    let velocity_scale = AU_METRES / canonical_time_seconds();
    let mut incoming = [
        first_arrival_normalized[3] * velocity_scale,
        first_arrival_normalized[4] * velocity_scale,
        first_arrival_normalized[5] * velocity_scale,
    ];

    let mut result = Vec::with_capacity(dimension(segments_per_leg));
    result.extend_from_slice(&reference[..9]);
    result.extend_from_slice(&reference[9..12]);
    for flyby in 0..FLYBYS {
        let (_, planet_velocity) = split_state(states[flyby + 1]);
        let outgoing = arcs[flyby].departure_velocity;
        let body = SEQUENCE[flyby + 1];
        let (safe_radius_km, mu_km) = {
            let (mu, radius) = flyby_body(body)?;
            (radius, mu)
        };
        let (radius_fraction, beta) = inverse_flyby_parameters(
            incoming,
            outgoing,
            planet_velocity,
            mu_km * 1.0e9,
            safe_radius_km * 1_000.0,
        )?;
        result.extend([radius_fraction, beta]);
        incoming = arcs[flyby].arrival_velocity;
    }

    let coarse_first = resample_first_leg_controls(segments_per_leg);
    for leg in 0..LEGS {
        if leg == 0 {
            for &control in &coarse_first {
                result.extend(control);
            }
        } else {
            for _ in 0..segments_per_leg {
                result.extend([0.0, 0.5 * core::f64::consts::PI, 0.0]);
            }
        }
    }
    debug_assert_eq!(result.len(), dimension(segments_per_leg));
    Ok(result)
}

pub fn resample(
    decision: &[f64],
    source_segments: usize,
    target_segments: usize,
) -> Result<Vec<f64>, ModelError> {
    validate_input(decision, source_segments)?;
    if !(5..=8).contains(&target_segments) {
        return Err(ModelError::Invalid("target segments per leg"));
    }
    let mut result = Vec::with_capacity(dimension(target_segments));
    result.extend_from_slice(&decision[..HEADER]);
    for leg in 0..LEGS {
        let source = (0..source_segments)
            .map(|segment| {
                let offset = HEADER + 3 * (leg * source_segments + segment);
                spherical_direction(decision[offset + 1], decision[offset + 2])
                    .map(|value| value * decision[offset])
            })
            .collect::<Vec<_>>();
        for target in 0..target_segments {
            let start = target as f64 / target_segments as f64;
            let end = (target + 1) as f64 / target_segments as f64;
            let mut vector = [0.0; 3];
            for (index, control) in source.iter().enumerate() {
                let source_start = index as f64 / source_segments as f64;
                let source_end = (index + 1) as f64 / source_segments as f64;
                let overlap = end.min(source_end) - start.max(source_start);
                if overlap > 0.0 {
                    for axis in 0..3 {
                        vector[axis] += control[axis] * overlap * target_segments as f64;
                    }
                }
            }
            result.extend(spherical_control(vector));
        }
    }
    debug_assert_eq!(result.len(), dimension(target_segments));
    Ok(result)
}

fn evaluate_with_method(
    x: &[f64],
    segments_per_leg: usize,
    method: IntegrationMethod,
) -> Result<(TourEvaluation, Vec<f64>), ModelError> {
    validate_input(x, segments_per_leg)?;
    let epochs = encounter_epochs(x)?;
    let states = competition_states(&epochs)?;
    let (_, earth_velocity) = split_state(states[0]);
    let departure_direction = spherical_direction(x[10], x[11]);
    let departure_velocity = add(
        earth_velocity,
        departure_direction.map(|value| value * x[9] * 1_000.0),
    );
    let mut spacecraft = normalized_endpoint(
        join_state(split_state(states[0]).0, departure_velocity),
        INITIAL_MASS_KG,
    );
    let time_scale = canonical_time_seconds();
    let velocity_scale = AU_METRES / time_scale;
    let maximum_thrust = MAX_THRUST_NEWTONS * time_scale * time_scale / INITIAL_MASS_KG / AU_METRES;
    let mass_flow = AU_METRES / time_scale / EXHAUST_VELOCITY_M_S;
    let mut residual = Vec::with_capacity(3 * LEGS);

    for leg in 0..LEGS {
        let duration = x[leg + 1] * DAY_SECONDS / time_scale;
        let boundaries = (0..=segments_per_leg)
            .map(|index| duration * index as f64 / segments_per_leg as f64)
            .collect();
        let controls = (0..segments_per_leg)
            .map(|segment| {
                let offset = HEADER + 3 * (leg * segments_per_leg + segment);
                let direction = spherical_direction(x[offset + 1], x[offset + 2]);
                [
                    maximum_thrust * x[offset],
                    direction[0],
                    direction[1],
                    direction[2],
                ]
            })
            .collect();
        let schedule = ControlSchedule::new(boundaries, controls)?;
        spacecraft = propagate_schedule_with_method(
            &ZohKeplerDynamics,
            &schedule,
            spacecraft,
            [mass_flow],
            ZOH_OPTIONS,
            method,
        )?
        .state;
        if !spacecraft.iter().all(|value| value.is_finite()) || spacecraft[6] <= 0.0 {
            return Err(ModelError::Invalid("whole-tour mass"));
        }
        let (target_position, planet_velocity) = split_state(states[leg + 1]);
        for axis in 0..3 {
            residual.push(spacecraft[axis] - target_position[axis] / AU_METRES);
        }
        if leg < FLYBYS {
            let body = SEQUENCE[leg + 1];
            let (mu_km, safe_radius_km) = flyby_body(body)?;
            let radius_fraction = x[12 + 2 * leg];
            let beta = x[13 + 2 * leg];
            let periapsis = safe_radius_km * MAX_PERIAPSIS_FACTOR.powf(radius_fraction) * 1_000.0;
            let incoming = [
                spacecraft[3] * velocity_scale,
                spacecraft[4] * velocity_scale,
                spacecraft[5] * velocity_scale,
            ];
            let outgoing = flyby_outgoing_velocity(
                &incoming,
                &planet_velocity,
                periapsis,
                beta,
                mu_km * 1.0e9,
            )?;
            spacecraft[0] = target_position[0] / AU_METRES;
            spacecraft[1] = target_position[1] / AU_METRES;
            spacecraft[2] = target_position[2] / AU_METRES;
            spacecraft[3] = outgoing[0] / velocity_scale;
            spacecraft[4] = outgoing[1] / velocity_scale;
            spacecraft[5] = outgoing[2] / velocity_scale;
        }
    }

    let (_, asteroid_velocity) = split_state(states[ENCOUNTERS - 1]);
    let arrival_velocity = [
        spacecraft[3] * velocity_scale,
        spacecraft[4] * velocity_scale,
        spacecraft[5] * velocity_scale,
    ];
    let arrival_relative = subtract(asteroid_velocity, arrival_velocity);
    let final_mass_kg = spacecraft[6] * INITIAL_MASS_KG;
    let score = final_mass_kg * dot(arrival_relative, asteroid_velocity) / 1.0e6;
    let constraint = residual.iter().map(|value| value * value).sum::<f64>();
    let maximum_position_mismatch = residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
    Ok((
        TourEvaluation {
            objective: PENALTY * constraint - score,
            score,
            final_mass_kg,
            position_mismatch_norm: constraint.sqrt(),
            maximum_position_mismatch,
        },
        residual,
    ))
}

fn validate_input(x: &[f64], segments_per_leg: usize) -> Result<(), ModelError> {
    if !(5..=8).contains(&segments_per_leg) || x.len() != dimension(segments_per_leg) {
        return Err(ModelError::Invalid("whole-tour decision dimension"));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(ModelError::Invalid("whole-tour decision"));
    }
    let limits = bounds(segments_per_leg);
    if x.iter()
        .zip(limits.lower().iter().zip(limits.upper()))
        .any(|(&value, (&lower, &upper))| !(lower..=upper).contains(&value))
    {
        return Err(ModelError::Invalid("whole-tour decision bounds"));
    }
    encounter_epochs(x)?;
    Ok(())
}

fn resample_first_leg_controls(segments_per_leg: usize) -> Vec<[f64; 3]> {
    let fine = (0..SEGMENTS)
        .map(|segment| {
            let offset = 15 + 3 * segment;
            spherical_direction(
                VALIDATED_DECISION[offset + 1],
                VALIDATED_DECISION[offset + 2],
            )
            .map(|value| value * VALIDATED_DECISION[offset])
        })
        .collect::<Vec<_>>();
    (0..segments_per_leg)
        .map(|coarse| {
            let start = coarse as f64 / segments_per_leg as f64;
            let end = (coarse + 1) as f64 / segments_per_leg as f64;
            let mut vector = [0.0; 3];
            for (index, control) in fine.iter().enumerate() {
                let fine_start = index as f64 / SEGMENTS as f64;
                let fine_end = (index + 1) as f64 / SEGMENTS as f64;
                let overlap = end.min(fine_end) - start.max(fine_start);
                if overlap > 0.0 {
                    for axis in 0..3 {
                        vector[axis] += control[axis] * overlap * segments_per_leg as f64;
                    }
                }
            }
            spherical_control(vector)
        })
        .collect()
}

fn spherical_control(vector: Vector3) -> [f64; 3] {
    let magnitude = vector_norm(vector).clamp(0.0, 1.0);
    if magnitude == 0.0 {
        [0.0, 0.5 * core::f64::consts::PI, 0.0]
    } else {
        [
            magnitude,
            (vector[2] / magnitude).clamp(-1.0, 1.0).acos(),
            vector[1].atan2(vector[0]),
        ]
    }
}

fn inverse_flyby_parameters(
    incoming: Vector3,
    outgoing: Vector3,
    planet_velocity: Vector3,
    mu: f64,
    safe_radius: f64,
) -> Result<(f64, f64), ModelError> {
    let incoming_relative = subtract(incoming, planet_velocity);
    let outgoing_relative = subtract(outgoing, planet_velocity);
    let speed = vector_norm(incoming_relative);
    if speed == 0.0 || vector_norm(outgoing_relative) == 0.0 {
        return Err(ModelError::Invalid("whole-tour flyby seed"));
    }
    let i_hat = incoming_relative.map(|value| value / speed);
    let j_hat = normalize(cross(i_hat, planet_velocity))?;
    let k_hat = cross(i_hat, j_hat);
    let outgoing_hat = outgoing_relative.map(|value| value / vector_norm(outgoing_relative));
    let turn = dot(i_hat, outgoing_hat).clamp(-1.0, 1.0).acos();
    let sine = turn.sin();
    if sine.abs() < 1.0e-12 {
        return Ok((1.0, 0.0));
    }
    let eccentricity = 1.0 / (0.5 * turn).sin().max(1.0e-12);
    let periapsis = mu * (eccentricity - 1.0) / speed.powi(2);
    let fraction = (periapsis.max(safe_radius) / safe_radius).ln() / MAX_PERIAPSIS_FACTOR.ln();
    let beta = (dot(outgoing_hat, k_hat) / sine).atan2(dot(outgoing_hat, j_hat) / sine);
    Ok((fraction.clamp(0.0, 1.0), beta))
}

fn cross(left: Vector3, right: Vector3) -> Vector3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(vector: Vector3) -> Result<Vector3, ModelError> {
    let norm = vector_norm(vector);
    if norm == 0.0 {
        Err(ModelError::Invalid("whole-tour flyby plane"))
    } else {
        Ok(vector.map(|value| value / norm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_bounds_and_seeds_cover_five_through_eight_segments_per_leg() {
        for segments in 5..=8 {
            let limits = bounds(segments);
            let initial = seed(segments).unwrap();
            assert_eq!(limits.dim(), dimension(segments));
            assert_eq!(initial.len(), dimension(segments));
            assert!(evaluate(&initial, segments).unwrap().objective.is_finite());
        }
    }

    #[test]
    fn invalid_segment_counts_and_decisions_fail_safely() {
        assert!(seed(4).is_err());
        assert!(evaluate(&[], 5).is_err());
        assert_eq!(objective(&[], 5).to_bits(), NAN_REPLACEMENT.to_bits());
    }

    #[test]
    fn coarse_meshes_resample_without_changing_headers_or_bounds() {
        let five = seed(5).unwrap();
        let eight = resample(&five, 5, 8).unwrap();
        assert_eq!(&eight[..HEADER], &five[..HEADER]);
        assert_eq!(eight.len(), dimension(8));
        assert!(evaluate(&eight, 8).unwrap().objective.is_finite());
        assert!(resample(&five, 5, 9).is_err());
    }

    #[test]
    fn whole_tour_seed_cross_checks_taylor_and_dop853() {
        let initial = seed(5).unwrap();
        let validation = validate_backends(&initial, 5).unwrap();
        assert!(validation.taylor.position_mismatch_norm.is_finite());
        assert!(validation.dop853.position_mismatch_norm.is_finite());
        assert!(validation.maximum_backend_difference < 1.0e-3);
    }
}
