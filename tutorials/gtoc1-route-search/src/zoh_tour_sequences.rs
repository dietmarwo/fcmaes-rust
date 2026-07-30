// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Direct whole-tour finite-thrust experiments for alternate GTOC1 routes.
//!
//! Every interplanetary leg is propagated with bounded piecewise-constant
//! thrust. Planet encounters are multiple-shooting nodes: position mismatch is
//! penalized, while the incoming velocity is mapped through an exact unpowered
//! flyby to initialize the next leg.

use fcmaes_core::{NAN_REPLACEMENT, RetryBounds};
use pykep_core::astro::flyby::flyby_outgoing_velocity;
use pykep_core::dynamics::zoh::{
    ControlSchedule, ZohKeplerDynamics, propagate_schedule_with_method,
};
use pykep_core::integration::{IntegrationMethod, IntegratorOptions};
use pykep_core::{CartesianState, Vector3};

use crate::low_thrust_sequences::{
    EXHAUST_VELOCITY_M_S, INITIAL_MASS_KG, MAXIMUM_THRUST_NEWTONS, SequenceScaffold,
};
use crate::real::{MAXIMUM_FLIGHT_DAYS, competition_state};
use crate::sequences::SequenceCase;
use crate::{BODY_MU_KM3_S2, DAY_SECONDS, Gtoc1Error, LEGACY_MU_SUN};

const ASTRONOMICAL_UNIT_METRES: f64 = 149_597_870_660.0;
const CONSTRAINT_PENALTY: f64 = 1.0e15;
const MAXIMUM_PERIAPSIS_FACTOR: f64 = 200.0;
const MINIMUM_PERIAPSIS_KM: [f64; 9] = [
    0.0, 2_740.0, 6_351.0, 6_678.0, 3_689.0, 600_000.0, 70_000.0, 0.0, 0.0,
];
const INTEGRATOR_OPTIONS: IntegratorOptions = IntegratorOptions {
    relative_tolerance: 1.0e-12,
    absolute_tolerance: 1.0e-12,
    initial_step: None,
    maximum_step: None,
    maximum_steps: 100_000,
    maximum_rejections: 100,
};

/// Parsed controls from an earlier per-leg Sims–Flanagan checkpoint.
#[derive(Clone, Debug)]
pub struct SimsFlanaganWarmStart {
    /// Number of impulses in every checkpoint leg.
    pub segments: usize,
    /// Complete checkpoint decision vector for every leg.
    pub legs: Vec<Vec<f64>>,
}

/// One direct whole-tour finite-thrust evaluation.
#[derive(Clone, Debug)]
pub struct ZohTourEvaluation {
    /// Penalized minimization objective.
    pub objective: f64,
    /// Unpenalized competition impact score.
    pub score: f64,
    /// Propagated spacecraft mass at asteroid impact.
    pub final_mass_kg: f64,
    /// Euclidean norm of all canonical planet-position residuals.
    pub position_mismatch_norm: f64,
    /// Largest absolute canonical position-residual component.
    pub maximum_position_mismatch: f64,
    /// Earth-departure hyperbolic excess in kilometres per second.
    pub launch_v_infinity_km_s: f64,
    /// Launch-to-impact duration in days.
    pub flight_days: f64,
}

/// Taylor/DOP853 cross-check of the same whole-tour decision.
#[derive(Clone, Debug)]
pub struct ZohTourValidation {
    /// Evaluation produced by accelerated Taylor propagation.
    pub taylor: ZohTourEvaluation,
    /// Evaluation produced by DOP853 propagation.
    pub dop853: ZohTourEvaluation,
    /// Largest absolute difference between corresponding residual components.
    pub maximum_backend_difference: f64,
}

/// Fixed-sequence direct transcription with a uniform ZOH mesh on every leg.
#[derive(Clone, Debug)]
pub struct ZohTourProblem {
    case: SequenceCase,
    segments_per_leg: usize,
    seed_scaffold: SequenceScaffold,
}

impl ZohTourProblem {
    /// Builds a problem around one encounter schedule.
    ///
    /// The schedule initializes the direct transcription. Encounter dates
    /// remain decision variables inside the sequence case bounds.
    ///
    /// # Errors
    ///
    /// Returns an error unless there are 5–8 segments per leg and the
    /// Lambert/flyby seed scaffold can be constructed.
    pub fn new(
        case: SequenceCase,
        schedule: &[f64],
        segments_per_leg: usize,
    ) -> Result<Self, Gtoc1Error> {
        if !(5..=8).contains(&segments_per_leg) {
            return Err(Gtoc1Error::Numerical(
                "ZOH segments per leg must be in 5..=8",
            ));
        }
        if schedule.len() != case.bodies.len() {
            return Err(Gtoc1Error::Dimension {
                actual: schedule.len(),
            });
        }
        let seed_scaffold = SequenceScaffold::new(case, schedule)?;
        Ok(Self {
            case,
            segments_per_leg,
            seed_scaffold,
        })
    }

    /// Selected alternate sequence.
    #[must_use]
    pub const fn case(&self) -> SequenceCase {
        self.case
    }

    /// Number of propagated finite-thrust intervals on every leg.
    #[must_use]
    pub const fn segments_per_leg(&self) -> usize {
        self.segments_per_leg
    }

    /// Number of interplanetary legs.
    #[must_use]
    pub fn leg_count(&self) -> usize {
        self.case.bodies.len() - 1
    }

    /// Number of decision variables.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.header_dimension() + 3 * self.leg_count() * self.segments_per_leg
    }

    /// Complete global bounds for dates, flybys, launch, and thrust controls.
    ///
    /// # Panics
    ///
    /// Panics only if the static sequence bounds are inconsistent.
    #[must_use]
    pub fn bounds(&self) -> RetryBounds {
        let mut lower = self.case.lower.to_vec();
        let mut upper = self.case.upper.to_vec();
        lower.extend([0.0, 0.0, -core::f64::consts::PI]);
        upper.extend([2.5, core::f64::consts::PI, core::f64::consts::PI]);
        for _ in 0..self.flyby_count() {
            lower.extend([0.0, -core::f64::consts::PI]);
            upper.extend([1.0, core::f64::consts::PI]);
        }
        for _ in 0..self.leg_count() * self.segments_per_leg {
            lower.extend([0.0, 0.0, -core::f64::consts::PI]);
            upper.extend([1.0, core::f64::consts::PI, core::f64::consts::PI]);
        }
        RetryBounds::new(lower, upper).expect("ZOH tour bounds are valid")
    }

    /// Constructs a bounded neighborhood around an incumbent.
    ///
    /// A fraction of one reproduces the complete global box.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid incumbent or a fraction outside
    /// `(0, 1]`.
    pub fn refinement_bounds(
        &self,
        incumbent: &[f64],
        fraction: f64,
    ) -> Result<RetryBounds, Gtoc1Error> {
        self.validate_decision(incumbent)?;
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) || fraction == 0.0 {
            return Err(Gtoc1Error::Numerical("invalid ZOH refinement fraction"));
        }
        if fraction >= 1.0 {
            return Ok(self.bounds());
        }
        let global = self.bounds();
        let mut lower = Vec::with_capacity(self.dimension());
        let mut upper = Vec::with_capacity(self.dimension());
        for (index, &value) in incumbent.iter().enumerate() {
            let radius = fraction * (global.upper()[index] - global.lower()[index]);
            lower.push(global.lower()[index].max(value - radius));
            upper.push(global.upper()[index].min(value + radius));
        }
        RetryBounds::new(lower, upper)
            .map_err(|_| Gtoc1Error::Numerical("invalid ZOH refinement bounds"))
    }

    /// Builds a warm decision from the repaired Lambert scaffold.
    ///
    /// When supplied, Sims–Flanagan Cartesian throttles are conservatively
    /// averaged onto the requested ZOH mesh. They initialize the search but
    /// are never used by the propagated objective.
    ///
    /// # Errors
    ///
    /// Returns an error for an incompatible warm checkpoint or singular flyby
    /// seed geometry.
    pub fn seed(&self, warm: Option<&SimsFlanaganWarmStart>) -> Result<Vec<f64>, Gtoc1Error> {
        let schedule = self.seed_scaffold.schedule();
        let states = self.seed_scaffold.states();
        let departures = self.seed_scaffold.departure_velocities();
        let arrivals = self.seed_scaffold.arrival_velocities();
        let mut result = Vec::with_capacity(self.dimension());
        result.extend_from_slice(schedule);

        if let Some(checkpoint) = warm {
            let first = checkpoint
                .legs
                .first()
                .ok_or(Gtoc1Error::Numerical("empty Sims-Flanagan warm start"))?;
            if first.len() < 4 {
                return Err(Gtoc1Error::Numerical(
                    "invalid first Sims-Flanagan warm leg",
                ));
            }
            result.extend_from_slice(&first[..3]);
        } else {
            let (_, earth_velocity) = split_state(states[0]);
            let launch = subtract(departures[0], earth_velocity);
            result.extend(spherical_velocity(launch, 1.0e-3, 2.5));
        }

        for node in 1..self.case.bodies.len() - 1 {
            let (_, planet_velocity) = split_state(states[node]);
            let body = self.case.bodies[node];
            let (fraction, beta) = inverse_flyby_parameters(
                arrivals[node - 1],
                departures[node],
                planet_velocity,
                BODY_MU_KM3_S2[body] * 1.0e9,
                MINIMUM_PERIAPSIS_KM[body] * 1_000.0,
            )?;
            result.extend([fraction, beta]);
        }

        for leg in 0..self.leg_count() {
            let controls = if let Some(checkpoint) = warm {
                resample_sims_controls(checkpoint, leg, self.segments_per_leg)?
            } else {
                vec![[0.0; 3]; self.segments_per_leg]
            };
            for control in controls {
                result.extend(spherical_control(control));
            }
        }
        if result.len() != self.dimension() {
            return Err(Gtoc1Error::Numerical("invalid ZOH seed dimension"));
        }
        self.validate_decision(&result)?;
        Ok(result)
    }

    /// Evaluates the accelerated Taylor objective.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid decision, ephemeris/flyby failure, or
    /// failed finite-thrust propagation.
    pub fn evaluate(&self, x: &[f64]) -> Result<ZohTourEvaluation, Gtoc1Error> {
        self.evaluate_with_method(x, IntegrationMethod::Taylor)
            .map(|(evaluation, _)| evaluation)
    }

    /// Scalar optimizer callback with a finite invalid-point replacement.
    #[must_use]
    pub fn objective(&self, x: &[f64]) -> f64 {
        self.evaluate(x)
            .map_or(NAN_REPLACEMENT, |evaluation| evaluation.objective)
    }

    /// Repropagates a decision independently with Taylor and DOP853.
    ///
    /// # Errors
    ///
    /// Returns an error if either propagation backend rejects the trajectory.
    pub fn validate_backends(&self, x: &[f64]) -> Result<ZohTourValidation, Gtoc1Error> {
        let (taylor, taylor_residual) = self.evaluate_with_method(x, IntegrationMethod::Taylor)?;
        let (dop853, dop853_residual) = self.evaluate_with_method(x, IntegrationMethod::Dop853)?;
        let maximum_backend_difference = taylor_residual
            .iter()
            .zip(dop853_residual)
            .map(|(&left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        Ok(ZohTourValidation {
            taylor,
            dop853,
            maximum_backend_difference,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn evaluate_with_method(
        &self,
        x: &[f64],
        method: IntegrationMethod,
    ) -> Result<(ZohTourEvaluation, Vec<f64>), Gtoc1Error> {
        self.validate_decision(x)?;
        let (epochs, flight_days) = self.encounter_epochs(x)?;
        let states = self
            .case
            .bodies
            .iter()
            .zip(epochs)
            .map(|(&body, epoch)| competition_state(body, epoch))
            .collect::<Result<Vec<_>, _>>()?;
        let schedule_dimension = self.case.bodies.len();
        let launch_index = schedule_dimension;
        let (_, earth_velocity) = split_state(states[0]);
        let departure_direction = spherical_direction(x[launch_index + 1], x[launch_index + 2]);
        let departure_velocity = add(
            earth_velocity,
            scale(departure_direction, x[launch_index] * 1_000.0),
        );
        let (earth_position, _) = split_state(states[0]);
        let time_scale = canonical_time_seconds();
        let velocity_scale = ASTRONOMICAL_UNIT_METRES / time_scale;
        let maximum_thrust = MAXIMUM_THRUST_NEWTONS * time_scale * time_scale
            / INITIAL_MASS_KG
            / ASTRONOMICAL_UNIT_METRES;
        let mass_flow = ASTRONOMICAL_UNIT_METRES / time_scale / EXHAUST_VELOCITY_M_S;
        let mut spacecraft = normalized_state(earth_position, departure_velocity);
        let mut residual = Vec::with_capacity(3 * self.leg_count());
        let segment_count = f64::from(
            u32::try_from(self.segments_per_leg)
                .map_err(|_| Gtoc1Error::Numerical("segment count too large"))?,
        );

        for leg in 0..self.leg_count() {
            let duration = x[leg + 1] * DAY_SECONDS / time_scale;
            let boundaries = (0..=self.segments_per_leg)
                .map(|index| {
                    let index = f64::from(
                        u32::try_from(index)
                            .map_err(|_| Gtoc1Error::Numerical("segment index too large"))?,
                    );
                    Ok(duration * index / segment_count)
                })
                .collect::<Result<Vec<_>, Gtoc1Error>>()?;
            let controls = (0..self.segments_per_leg)
                .map(|segment| {
                    let offset =
                        self.header_dimension() + 3 * (leg * self.segments_per_leg + segment);
                    let direction = spherical_direction(x[offset + 1], x[offset + 2]);
                    [
                        maximum_thrust * x[offset],
                        direction[0],
                        direction[1],
                        direction[2],
                    ]
                })
                .collect();
            let controls = ControlSchedule::new(boundaries, controls)?;
            spacecraft = propagate_schedule_with_method(
                &ZohKeplerDynamics,
                &controls,
                spacecraft,
                [mass_flow],
                INTEGRATOR_OPTIONS,
                method,
            )?
            .state;
            if !spacecraft.iter().all(|value| value.is_finite()) || spacecraft[6] <= 0.0 {
                return Err(Gtoc1Error::Numerical("invalid propagated ZOH mass"));
            }
            let (target_position, planet_velocity) = split_state(states[leg + 1]);
            for axis in 0..3 {
                residual.push(spacecraft[axis] - target_position[axis] / ASTRONOMICAL_UNIT_METRES);
            }
            if leg < self.flyby_count() {
                let body = self.case.bodies[leg + 1];
                let flyby_offset = schedule_dimension + 3 + 2 * leg;
                let periapsis = MINIMUM_PERIAPSIS_KM[body]
                    * MAXIMUM_PERIAPSIS_FACTOR.powf(x[flyby_offset])
                    * 1_000.0;
                let incoming = [
                    spacecraft[3] * velocity_scale,
                    spacecraft[4] * velocity_scale,
                    spacecraft[5] * velocity_scale,
                ];
                let outgoing = flyby_outgoing_velocity(
                    &incoming,
                    &planet_velocity,
                    periapsis,
                    x[flyby_offset + 1],
                    BODY_MU_KM3_S2[body] * 1.0e9,
                )?;
                for axis in 0..3 {
                    spacecraft[axis] = target_position[axis] / ASTRONOMICAL_UNIT_METRES;
                    spacecraft[axis + 3] = outgoing[axis] / velocity_scale;
                }
            }
        }

        let (_, asteroid_velocity) = split_state(states[states.len() - 1]);
        let arrival_velocity = [
            spacecraft[3] * velocity_scale,
            spacecraft[4] * velocity_scale,
            spacecraft[5] * velocity_scale,
        ];
        let final_mass_kg = spacecraft[6] * INITIAL_MASS_KG;
        let score = final_mass_kg
            * dot(
                subtract(asteroid_velocity, arrival_velocity),
                asteroid_velocity,
            )
            / 1.0e6;
        let mismatch_squared = residual.iter().map(|value| value * value).sum::<f64>();
        let maximum_position_mismatch =
            residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
        Ok((
            ZohTourEvaluation {
                objective: CONSTRAINT_PENALTY * mismatch_squared - score,
                score,
                final_mass_kg,
                position_mismatch_norm: mismatch_squared.sqrt(),
                maximum_position_mismatch,
                launch_v_infinity_km_s: x[launch_index],
                flight_days,
            },
            residual,
        ))
    }

    fn validate_decision(&self, x: &[f64]) -> Result<(), Gtoc1Error> {
        if x.len() != self.dimension() {
            return Err(Gtoc1Error::Dimension { actual: x.len() });
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(Gtoc1Error::Numerical("non-finite ZOH tour decision"));
        }
        let bounds = self.bounds();
        if x.iter()
            .zip(bounds.lower().iter().zip(bounds.upper()))
            .any(|(&value, (&lower, &upper))| !(lower..=upper).contains(&value))
        {
            return Err(Gtoc1Error::Numerical("out-of-bounds ZOH tour decision"));
        }
        self.encounter_epochs(x)?;
        Ok(())
    }

    fn encounter_epochs(&self, x: &[f64]) -> Result<(Vec<f64>, f64), Gtoc1Error> {
        let schedule_dimension = self.case.bodies.len();
        let mut epochs = Vec::with_capacity(schedule_dimension);
        epochs.push(x[0]);
        for duration in &x[1..schedule_dimension] {
            let next = epochs[epochs.len() - 1] + duration;
            epochs.push(next);
        }
        let flight_days = epochs[epochs.len() - 1] - epochs[0];
        if flight_days > MAXIMUM_FLIGHT_DAYS {
            return Err(Gtoc1Error::Numerical("ZOH tour exceeds 30 years"));
        }
        Ok((epochs, flight_days))
    }

    fn flyby_count(&self) -> usize {
        self.leg_count() - 1
    }

    fn header_dimension(&self) -> usize {
        self.case.bodies.len() + 3 + 2 * self.flyby_count()
    }
}

fn resample_sims_controls(
    checkpoint: &SimsFlanaganWarmStart,
    leg: usize,
    target_segments: usize,
) -> Result<Vec<Vector3>, Gtoc1Error> {
    if checkpoint.segments == 0 {
        return Err(Gtoc1Error::Numerical(
            "zero-segment Sims-Flanagan warm start",
        ));
    }
    let source = checkpoint
        .legs
        .get(leg)
        .ok_or(Gtoc1Error::Numerical("missing Sims-Flanagan warm leg"))?;
    let offset = if leg == 0 { 4 } else { 1 };
    if source.len() != offset + 3 * checkpoint.segments {
        return Err(Gtoc1Error::Numerical(
            "invalid Sims-Flanagan warm-leg dimension",
        ));
    }
    let source = source[offset..]
        .chunks_exact(3)
        .map(|values| [values[0], values[1], values[2]])
        .collect::<Vec<_>>();
    let source_segments = f64::from(
        u32::try_from(checkpoint.segments)
            .map_err(|_| Gtoc1Error::Numerical("warm segment count too large"))?,
    );
    let target_segments_float = f64::from(
        u32::try_from(target_segments)
            .map_err(|_| Gtoc1Error::Numerical("target segment count too large"))?,
    );
    (0..target_segments)
        .map(|target| {
            let start = f64::from(
                u32::try_from(target)
                    .map_err(|_| Gtoc1Error::Numerical("target index too large"))?,
            ) / target_segments_float;
            let end = f64::from(
                u32::try_from(target + 1)
                    .map_err(|_| Gtoc1Error::Numerical("target index too large"))?,
            ) / target_segments_float;
            let mut vector = [0.0; 3];
            for (index, &control) in source.iter().enumerate() {
                let source_index = f64::from(
                    u32::try_from(index)
                        .map_err(|_| Gtoc1Error::Numerical("source index too large"))?,
                );
                let source_start = source_index / source_segments;
                let source_end = (source_index + 1.0) / source_segments;
                let overlap = end.min(source_end) - start.max(source_start);
                if overlap > 0.0 {
                    for axis in 0..3 {
                        vector[axis] += control[axis] * overlap * target_segments_float;
                    }
                }
            }
            Ok(vector)
        })
        .collect()
}

fn inverse_flyby_parameters(
    incoming: Vector3,
    outgoing: Vector3,
    planet_velocity: Vector3,
    mu: f64,
    safe_radius: f64,
) -> Result<(f64, f64), Gtoc1Error> {
    let incoming_relative = subtract(incoming, planet_velocity);
    let outgoing_relative = subtract(outgoing, planet_velocity);
    let speed = norm(incoming_relative);
    let outgoing_speed = norm(outgoing_relative);
    if speed == 0.0 || outgoing_speed == 0.0 {
        return Err(Gtoc1Error::Numerical("zero ZOH flyby seed velocity"));
    }
    let i_hat = scale(incoming_relative, 1.0 / speed);
    let j_hat = normalize(cross(i_hat, planet_velocity))?;
    let k_hat = cross(i_hat, j_hat);
    let outgoing_hat = scale(outgoing_relative, 1.0 / outgoing_speed);
    let turn = dot(i_hat, outgoing_hat).clamp(-1.0, 1.0).acos();
    if turn.sin().abs() < 1.0e-12 {
        return Ok((1.0, 0.0));
    }
    let eccentricity = 1.0 / (0.5 * turn).sin().max(1.0e-12);
    let periapsis = mu * (eccentricity - 1.0) / speed.powi(2);
    let fraction = (periapsis.max(safe_radius) / safe_radius).ln() / MAXIMUM_PERIAPSIS_FACTOR.ln();
    let sine = turn.sin();
    let beta = (dot(outgoing_hat, k_hat) / sine).atan2(dot(outgoing_hat, j_hat) / sine);
    Ok((fraction.clamp(0.0, 1.0), beta))
}

fn spherical_velocity(vector: Vector3, scale_factor: f64, maximum: f64) -> [f64; 3] {
    let magnitude = norm(vector);
    if magnitude == 0.0 {
        [0.0, 0.5 * core::f64::consts::PI, 0.0]
    } else {
        [
            (magnitude * scale_factor).min(maximum),
            (vector[2] / magnitude).clamp(-1.0, 1.0).acos(),
            vector[1].atan2(vector[0]),
        ]
    }
}

fn spherical_control(vector: Vector3) -> [f64; 3] {
    let magnitude = norm(vector).clamp(0.0, 1.0);
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

fn spherical_direction(theta: f64, phi: f64) -> Vector3 {
    [
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    ]
}

fn normalized_state(position: Vector3, velocity: Vector3) -> [f64; 7] {
    let velocity_scale = ASTRONOMICAL_UNIT_METRES / canonical_time_seconds();
    [
        position[0] / ASTRONOMICAL_UNIT_METRES,
        position[1] / ASTRONOMICAL_UNIT_METRES,
        position[2] / ASTRONOMICAL_UNIT_METRES,
        velocity[0] / velocity_scale,
        velocity[1] / velocity_scale,
        velocity[2] / velocity_scale,
        1.0,
    ]
}

fn canonical_time_seconds() -> f64 {
    (ASTRONOMICAL_UNIT_METRES.powi(3) / LEGACY_MU_SUN).sqrt()
}

fn split_state(state: CartesianState) -> (Vector3, Vector3) {
    (
        [state[0], state[1], state[2]],
        [state[3], state[4], state[5]],
    )
}

fn add(left: Vector3, right: Vector3) -> Vector3 {
    core::array::from_fn(|index| left[index] + right[index])
}

fn subtract(left: Vector3, right: Vector3) -> Vector3 {
    core::array::from_fn(|index| left[index] - right[index])
}

fn scale(vector: Vector3, factor: f64) -> Vector3 {
    vector.map(|value| value * factor)
}

fn dot(left: Vector3, right: Vector3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm(vector: Vector3) -> f64 {
    dot(vector, vector).sqrt()
}

fn cross(left: Vector3, right: Vector3) -> Vector3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(vector: Vector3) -> Result<Vector3, Gtoc1Error> {
    let magnitude = norm(vector);
    if magnitude == 0.0 {
        Err(Gtoc1Error::Numerical("singular ZOH flyby plane"))
    } else {
        Ok(scale(vector, 1.0 / magnitude))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequences::{DEIMOS, JENA, JPL2};

    #[test]
    fn alternate_cases_cover_all_requested_meshes() {
        for case in [JPL2, JENA, DEIMOS] {
            for segments in 5..=8 {
                let problem = ZohTourProblem::new(case, case.guess, segments).unwrap();
                let seed = problem.seed(None).unwrap();
                assert_eq!(seed.len(), problem.dimension());
                assert_eq!(problem.bounds().dim(), problem.dimension());
                assert!(problem.evaluate(&seed).unwrap().objective.is_finite());
            }
        }
    }

    #[test]
    fn whole_tour_seed_cross_checks_taylor_and_dop853() {
        let problem = ZohTourProblem::new(JPL2, JPL2.guess, 5).unwrap();
        let seed = problem.seed(None).unwrap();
        let validation = problem.validate_backends(&seed).unwrap();
        assert!(validation.taylor.position_mismatch_norm.is_finite());
        assert!(validation.dop853.position_mismatch_norm.is_finite());
        assert!(validation.maximum_backend_difference < 1.0e-2);
    }

    #[test]
    fn invalid_mesh_and_decision_fail_safely() {
        assert!(ZohTourProblem::new(JPL2, JPL2.guess, 4).is_err());
        let problem = ZohTourProblem::new(JPL2, JPL2.guess, 5).unwrap();
        assert!(problem.evaluate(&[]).is_err());
        assert_eq!(problem.objective(&[]).to_bits(), NAN_REPLACEMENT.to_bits());
    }
}
