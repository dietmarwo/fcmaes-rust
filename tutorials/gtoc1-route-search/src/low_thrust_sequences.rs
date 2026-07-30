// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Full low-thrust validation of alternate GTOC1 sequences.
//!
//! The fast sequence scout identifies a multi-revolution Lambert chain. At
//! every intermediate planet this module replaces the two hyperbolic-excess
//! vectors by equal-speed vectors whose turn does not exceed the competition
//! limit at minimum periapsis. Each resulting fixed-endpoint leg is then a
//! separate Sims-Flanagan low-thrust problem. Solving the legs in chronological
//! order carries the optimized spacecraft mass through the complete mission.

use fcmaes_core::RetryBounds;
use pykep_core::astro::lambert::{LambertPath, LambertProblem};
use pykep_core::astro::propagation::propagate_lagrangian;
use pykep_core::leg::{SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint};
use pykep_core::{CartesianState, Vector3};

use crate::real::competition_state;
use crate::route_archive::BranchChoice;
use crate::route_search::{PhysicalDecision, RouteCase, RouteSearchError, RouteVariant};
use crate::sequences::SequenceCase;
use crate::sequences::{
    DEIMOS, DEIMOS_HISTORICAL_DECISIONS, JENA, JENA_HISTORICAL_DECISIONS, JPL2,
    JPL2_HISTORICAL_DECISION,
};
use crate::{
    BODY_MU_KM3_S2, DAY_SECONDS, Gtoc1Error, LEGACY_MU_SUN, distance, dot, powered_swingby_inverse,
    split_state, subtract,
};

/// Competition wet mass in kilograms.
pub const INITIAL_MASS_KG: f64 = 1_500.0;
/// Maximum electric-propulsion thrust in newtons.
pub const MAXIMUM_THRUST_NEWTONS: f64 = 0.04;
/// Effective exhaust velocity for 2500 s specific impulse in metres per second.
pub const EXHAUST_VELOCITY_M_S: f64 = 2_500.0 * 9.806_65;
/// Earth-departure hyperbolic-excess limit in kilometres per second.
pub const MAXIMUM_LAUNCH_V_INFINITY_KM_S: f64 = 2.5;
/// Competition solar-exclusion radius in astronomical units.
pub const MINIMUM_HELIOCENTRIC_DISTANCE_AU: f64 = 0.2;

const ASTRONOMICAL_UNIT_METRES: f64 = 149_597_870_660.0;
const VELOCITY_SCALE_M_S: f64 = 29_784.7;
const CONSTRAINT_PENALTY: f64 = 1.0e15;
const MINIMUM_PERIAPSIS_KM: [f64; 9] = [
    0.0, 2_740.0, 6_351.0, 6_678.0, 3_689.0, 600_000.0, 70_000.0, 0.0, 0.0,
];

/// One selected Lambert arc before its endpoint velocities are repaired.
#[derive(Clone, Debug)]
struct Arc {
    departure_velocity: Vector3,
    arrival_velocity: Vector3,
    revolutions: usize,
    path: LambertPath,
}

/// One repaired, competition-feasible gravity assist.
#[derive(Clone, Debug)]
pub struct RepairedFlyby {
    /// Encounter-node index in the complete sequence.
    pub node: usize,
    /// GTOP body identifier.
    pub body: usize,
    /// Common incoming and outgoing hyperbolic-excess speed.
    pub v_infinity_km_s: f64,
    /// Turn angle after repair.
    pub turn_angle_rad: f64,
    /// Maximum turn angle at the minimum permitted periapsis.
    pub maximum_turn_angle_rad: f64,
    /// Powered impulse implied by an inverse swingby check.
    pub powered_delta_v_km_s: f64,
    /// Inverse-swingby periapsis margin over the competition limit.
    pub periapsis_margin_km: f64,
}

/// Lambert branch chain with endpoint velocities repaired for unpowered flybys.
#[derive(Clone, Debug)]
pub struct SequenceScaffold {
    case: Option<SequenceCase>,
    name: String,
    variant: RouteVariant,
    schedule: Vec<f64>,
    epochs: Vec<f64>,
    states: Vec<CartesianState>,
    departure_velocities: Vec<Vector3>,
    arrival_velocities: Vec<Vector3>,
    branches: Vec<(usize, LambertPath)>,
    flybys: Vec<RepairedFlyby>,
    endpoint_repair_delta_v_km_s: f64,
}

impl SequenceScaffold {
    /// Builds the selected Lambert chain and repairs every intermediate flyby.
    ///
    /// The branch chain is selected by the propelled-first-leg scout. The
    /// repair preserves the mean incoming/outgoing excess speed and changes
    /// directions symmetrically until the minimum-periapsis turn limit is met.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid schedules, ephemeris or Lambert failures,
    /// missing selected branches, or singular flyby geometry.
    #[allow(clippy::too_many_lines)]
    pub fn new(case: SequenceCase, schedule: &[f64]) -> Result<Self, Gtoc1Error> {
        let scout = case.evaluate_endpoint_repair_scout(schedule)?;
        let states = case
            .bodies
            .iter()
            .zip(&scout.epochs_mjd2000)
            .map(|(&body, &epoch)| competition_state(body, epoch))
            .collect::<Result<Vec<_>, _>>()?;
        let mut arc_families = Vec::with_capacity(case.bodies.len() - 1);
        for leg in 0..case.bodies.len() - 1 {
            let (initial_position, _) = split_state(states[leg]);
            let (final_position, _) = split_state(states[leg + 1]);
            let problem = LambertProblem::new(
                initial_position,
                final_position,
                schedule[leg + 1] * DAY_SECONDS,
                LEGACY_MU_SUN,
                case.rev_flags[leg],
                case.maximum_revolutions[leg],
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

        let arcs = if case.name == DEIMOS.name
            && schedule
                .iter()
                .zip(DEIMOS_HISTORICAL_DECISIONS[0])
                .all(|(&actual, expected)| (actual - expected).abs() < 1.0e-6)
        {
            published_deimos_arcs(&arc_families, &states, schedule)?
        } else if case.name == JPL2.name && schedule_matches(schedule, &JPL2_HISTORICAL_DECISION) {
            select_arcs(
                &arc_families,
                &[
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
                ],
            )?
        } else if case.name == JENA.name
            && schedule_matches(schedule, &JENA_HISTORICAL_DECISIONS[0])
        {
            select_arcs(
                &arc_families,
                &[
                    (2, LambertPath::Left),
                    (1, LambertPath::Left),
                    (2, LambertPath::Left),
                    (0, LambertPath::ZeroRevolution),
                    (1, LambertPath::Left),
                    (0, LambertPath::ZeroRevolution),
                    (1, LambertPath::Right),
                    (0, LambertPath::ZeroRevolution),
                    (0, LambertPath::ZeroRevolution),
                    (0, LambertPath::ZeroRevolution),
                ],
            )?
        } else {
            arc_families
                .iter()
                .zip(&scout.branches)
                .map(|(family, &(revolutions, path))| {
                    family
                        .iter()
                        .find(|arc| arc.revolutions == revolutions && arc.path == path)
                        .cloned()
                        .ok_or(Gtoc1Error::Numerical(
                            "scouted Lambert branch missing from low-thrust family",
                        ))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        Self::from_arcs(
            Some(case),
            case.name,
            RouteVariant::from_sequence_case(case),
            schedule,
            scout.epochs_mjd2000,
            states,
            &arcs,
        )
    }

    /// Builds a low-thrust scaffold from an archived runtime L0 solution.
    ///
    /// This constructor consumes the exact physical schedule and Lambert
    /// branch chain selected by L0. It does not rerun the branch dynamic
    /// program, so L1 refines the same numerical candidate that was promoted.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent route dimensions, unavailable
    /// selected Lambert branches, ephemeris/Lambert failures, or singular
    /// flyby geometry.
    pub fn from_selected_route(
        route: &RouteCase,
        physical: &PhysicalDecision,
        branches: &[BranchChoice],
    ) -> Result<Self, RouteSearchError> {
        let variant = route.variant().clone();
        let bodies = &variant.structure.bodies;
        let schedule = physical.as_sequence_decision();
        let maximum_revolutions = route.maximum_revolutions();
        if physical.leg_days.len() + 1 != bodies.len()
            || branches.len() + 1 != bodies.len()
            || maximum_revolutions.len() + 1 != bodies.len()
        {
            return Err(RouteSearchError::Dimension {
                name: "runtime low-thrust scaffold",
                expected: bodies.len(),
                actual: branches.len() + 1,
            });
        }
        let mut epochs = Vec::with_capacity(bodies.len());
        let mut epoch = physical.launch_mjd2000;
        epochs.push(epoch);
        for &duration in &physical.leg_days {
            epoch += duration;
            epochs.push(epoch);
        }
        let states = bodies
            .iter()
            .zip(&epochs)
            .map(|(&body, &encounter_epoch)| competition_state(body, encounter_epoch))
            .collect::<Result<Vec<_>, _>>()?;
        let mut arcs = Vec::with_capacity(branches.len());
        for (leg, branch) in branches.iter().enumerate() {
            let (initial_position, _) = split_state(states[leg]);
            let (final_position, _) = split_state(states[leg + 1]);
            let problem = LambertProblem::new(
                initial_position,
                final_position,
                physical.leg_days[leg] * DAY_SECONDS,
                LEGACY_MU_SUN,
                variant.clockwise[leg],
                maximum_revolutions[leg],
            )
            .map_err(Gtoc1Error::from)?;
            let path: LambertPath = branch.path.into();
            let selected = problem
                .solutions()
                .iter()
                .find(|solution| {
                    solution.revolutions == branch.revolutions && solution.path == path
                })
                .ok_or(Gtoc1Error::Numerical(
                    "archived Lambert branch missing from runtime low-thrust family",
                ))?;
            arcs.push(Arc {
                departure_velocity: selected.departure_velocity,
                arrival_velocity: selected.arrival_velocity,
                revolutions: selected.revolutions,
                path: selected.path,
            });
        }
        Self::from_arcs(
            None,
            "runtime-route",
            variant,
            &schedule,
            epochs,
            states,
            &arcs,
        )
        .map_err(RouteSearchError::from)
    }

    fn from_arcs(
        case: Option<SequenceCase>,
        name: &str,
        variant: RouteVariant,
        schedule: &[f64],
        epochs: Vec<f64>,
        states: Vec<CartesianState>,
        arcs: &[Arc],
    ) -> Result<Self, Gtoc1Error> {
        let bodies = &variant.structure.bodies;
        let mut departure_velocities = arcs
            .iter()
            .map(|arc| arc.departure_velocity)
            .collect::<Vec<_>>();
        let mut arrival_velocities = arcs
            .iter()
            .map(|arc| arc.arrival_velocity)
            .collect::<Vec<_>>();
        let mut flybys = Vec::with_capacity(bodies.len() - 2);
        let mut endpoint_repair_delta_v = 0.0;
        for node in 1..bodies.len() - 1 {
            let (_, planet_velocity) = split_state(states[node]);
            let incoming = subtract(arrival_velocities[node - 1], planet_velocity);
            let outgoing = subtract(departure_velocities[node], planet_velocity);
            let body = bodies[node];
            let (repaired_incoming, repaired_outgoing, speed_km_s, maximum_angle) =
                repair_relative_velocities(incoming, outgoing, body)?;
            endpoint_repair_delta_v += distance(incoming, repaired_incoming);
            endpoint_repair_delta_v += distance(outgoing, repaired_outgoing);
            arrival_velocities[node - 1] = add(planet_velocity, repaired_incoming);
            departure_velocities[node] = add(planet_velocity, repaired_outgoing);

            let turn_angle = dot(repaired_incoming, repaired_outgoing)
                .mul_add(1.0 / norm(repaired_incoming) / norm(repaired_outgoing), 0.0)
                .clamp(-1.0, 1.0)
                .acos();
            let mu = BODY_MU_KM3_S2[body];
            let periapsis = MINIMUM_PERIAPSIS_KM[body];
            let (powered_delta_v, nondimensional_periapsis) =
                powered_swingby_inverse(speed_km_s, speed_km_s, turn_angle)?;
            flybys.push(RepairedFlyby {
                node,
                body,
                v_infinity_km_s: speed_km_s,
                turn_angle_rad: turn_angle,
                maximum_turn_angle_rad: maximum_angle,
                powered_delta_v_km_s: powered_delta_v,
                periapsis_margin_km: nondimensional_periapsis * mu - periapsis,
            });
        }

        Ok(Self {
            case,
            name: name.to_owned(),
            variant,
            schedule: schedule.to_vec(),
            epochs,
            states,
            departure_velocities,
            arrival_velocities,
            branches: arcs.iter().map(|arc| (arc.revolutions, arc.path)).collect(),
            flybys,
            endpoint_repair_delta_v_km_s: endpoint_repair_delta_v / 1_000.0,
        })
    }

    /// Historical sequence case, or `None` for an archived runtime route.
    #[must_use]
    pub const fn case(&self) -> Option<SequenceCase> {
        self.case
    }

    /// Human-readable scaffold identity.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Exact body order and Lambert direction flags.
    #[must_use]
    pub const fn variant(&self) -> &RouteVariant {
        &self.variant
    }

    /// Launch epoch followed by leg durations in days.
    #[must_use]
    pub fn schedule(&self) -> &[f64] {
        &self.schedule
    }

    /// Absolute encounter epochs in MJD2000 days.
    #[must_use]
    pub fn epochs_mjd2000(&self) -> &[f64] {
        &self.epochs
    }

    /// Selected `(revolutions, path)` Lambert branch per leg.
    #[must_use]
    pub fn branches(&self) -> &[(usize, LambertPath)] {
        &self.branches
    }

    /// Repaired intermediate gravity-assist diagnostics.
    #[must_use]
    pub fn flybys(&self) -> &[RepairedFlyby] {
        &self.flybys
    }

    /// Number of low-thrust legs.
    #[must_use]
    pub fn leg_count(&self) -> usize {
        self.departure_velocities.len()
    }

    /// Competition flight time in days.
    #[must_use]
    pub fn flight_days(&self) -> f64 {
        self.epochs[self.epochs.len() - 1] - self.epochs[0]
    }

    /// Smallest repaired-flyby periapsis margin in kilometres.
    #[must_use]
    pub fn minimum_periapsis_margin_km(&self) -> f64 {
        self.flybys
            .iter()
            .map(|flyby| flyby.periapsis_margin_km)
            .fold(f64::INFINITY, f64::min)
    }

    /// Sum of powered impulses reported by independent inverse-swingby checks.
    #[must_use]
    pub fn powered_delta_v_km_s(&self) -> f64 {
        self.flybys
            .iter()
            .map(|flyby| flyby.powered_delta_v_km_s)
            .sum()
    }

    /// Sum of incoming and outgoing velocity changes made by flyby repair.
    #[must_use]
    pub const fn endpoint_repair_delta_v_km_s(&self) -> f64 {
        self.endpoint_repair_delta_v_km_s
    }

    pub(crate) fn states(&self) -> &[CartesianState] {
        &self.states
    }

    pub(crate) fn departure_velocities(&self) -> &[Vector3] {
        &self.departure_velocities
    }

    pub(crate) fn arrival_velocities(&self) -> &[Vector3] {
        &self.arrival_velocities
    }

    /// Impact score for a specified final mass.
    #[must_use]
    pub fn impact_score(&self, final_mass_kg: f64) -> f64 {
        let (_, asteroid_velocity) = split_state(self.states[self.states.len() - 1]);
        let spacecraft_velocity = self.arrival_velocities[self.arrival_velocities.len() - 1];
        final_mass_kg
            * dot(
                subtract(asteroid_velocity, spacecraft_velocity),
                asteroid_velocity,
            )
            / 1.0e6
    }
}

fn schedule_matches(actual: &[f64], expected: &[f64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&left, &right)| (left - right).abs() < 1.0e-6)
}

fn select_arcs(
    families: &[Vec<Arc>],
    selected: &[(usize, LambertPath)],
) -> Result<Vec<Arc>, Gtoc1Error> {
    if families.len() != selected.len() {
        return Err(Gtoc1Error::Numerical(
            "stored Lambert branch chain has the wrong length",
        ));
    }
    families
        .iter()
        .zip(selected)
        .map(|(family, &(revolutions, path))| {
            family
                .iter()
                .find(|arc| arc.revolutions == revolutions && arc.path == path)
                .cloned()
                .ok_or(Gtoc1Error::Numerical(
                    "stored Lambert branch missing from family",
                ))
        })
        .collect()
}

fn published_deimos_arcs(
    families: &[Vec<Arc>],
    states: &[CartesianState],
    schedule: &[f64],
) -> Result<Vec<Arc>, Gtoc1Error> {
    const SELECTED: [(usize, LambertPath); 13] = [
        (3, LambertPath::Right),
        (3, LambertPath::Right),
        (0, LambertPath::ZeroRevolution),
        (1, LambertPath::Left),
        (1, LambertPath::Left),
        (1, LambertPath::Left),
        (1, LambertPath::Right),
        (1, LambertPath::Left),
        (1, LambertPath::Right),
        (0, LambertPath::ZeroRevolution),
        (0, LambertPath::ZeroRevolution),
        (0, LambertPath::ZeroRevolution),
        (0, LambertPath::ZeroRevolution),
    ];
    const PUBLISHED_ARRIVAL_V_INFINITY_KM_S: [(usize, Vector3); 4] = [
        (0, [-2.599_89, -1.720_14, -3.763_57]),
        (1, [-0.063_23, -2.513_65, 4.174_42]),
        (3, [7.666_88, -0.488_58, -0.657_98]),
        (5, [5.516_72, 12.237_61, 3.019_19]),
    ];
    let mut arcs = families
        .iter()
        .zip(SELECTED)
        .map(|(family, (revolutions, path))| {
            family
                .iter()
                .find(|arc| arc.revolutions == revolutions && arc.path == path)
                .cloned()
                .ok_or(Gtoc1Error::Numerical(
                    "published Deimos Lambert branch missing",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    for (leg, relative_velocity_km_s) in PUBLISHED_ARRIVAL_V_INFINITY_KM_S {
        let (arrival_position, arrival_planet_velocity) = split_state(states[leg + 1]);
        let arrival_velocity = add(
            arrival_planet_velocity,
            scale(relative_velocity_km_s, 1_000.0),
        );
        if leg == 0 {
            arcs[leg].arrival_velocity = arrival_velocity;
            continue;
        }
        let arrival_state = join_state(arrival_position, arrival_velocity);
        let departure_state = propagate_lagrangian(
            &arrival_state,
            -schedule[leg + 1] * DAY_SECONDS,
            LEGACY_MU_SUN,
        )?;
        arcs[leg].departure_velocity = split_state(departure_state).1;
        arcs[leg].arrival_velocity = arrival_velocity;
    }
    Ok(arcs)
}

fn repair_relative_velocities(
    incoming: Vector3,
    outgoing: Vector3,
    body: usize,
) -> Result<(Vector3, Vector3, f64, f64), Gtoc1Error> {
    let incoming_norm = norm(incoming);
    let outgoing_norm = norm(outgoing);
    if incoming_norm == 0.0 || outgoing_norm == 0.0 {
        return Err(Gtoc1Error::Numerical("zero flyby excess speed"));
    }
    let common_norm = 0.5 * (incoming_norm + outgoing_norm);
    let incoming_direction = scale(incoming, 1.0 / incoming_norm);
    let outgoing_direction = scale(outgoing, 1.0 / outgoing_norm);
    let original_angle = dot(incoming_direction, outgoing_direction)
        .clamp(-1.0, 1.0)
        .acos();
    let periapsis = MINIMUM_PERIAPSIS_KM[body];
    let mu = BODY_MU_KM3_S2[body];
    let speed_km_s = common_norm / 1_000.0;
    let maximum_angle = 2.0 * (mu / periapsis / (speed_km_s * speed_km_s + mu / periapsis)).asin();
    let (repaired_incoming_direction, repaired_outgoing_direction) =
        if original_angle > maximum_angle {
            let fraction = 0.5 * (original_angle - maximum_angle) / original_angle;
            (
                spherical_interpolate(incoming_direction, outgoing_direction, fraction),
                spherical_interpolate(incoming_direction, outgoing_direction, 1.0 - fraction),
            )
        } else {
            (incoming_direction, outgoing_direction)
        };
    Ok((
        scale(repaired_incoming_direction, common_norm),
        scale(repaired_outgoing_direction, common_norm),
        speed_km_s,
        maximum_angle,
    ))
}

/// Diagnostics for one fixed-endpoint Sims-Flanagan leg.
#[derive(Clone, Debug)]
pub struct LowThrustLegEvaluation {
    /// Penalized scalar optimization objective.
    pub objective: f64,
    /// Normalized seven-component cut-mismatch norm.
    pub mismatch_norm: f64,
    /// Raw cut mismatch `[position metres, velocity m/s, mass kg]`.
    pub mismatch: [f64; 7],
    /// Initial mass in kilograms.
    pub initial_mass_kg: f64,
    /// Arrival mass in kilograms implied by the impulse sequence.
    pub final_mass_kg: f64,
    /// Propellant consumed in kilograms.
    pub fuel_kg: f64,
    /// Largest throttle-vector norm.
    pub maximum_throttle: f64,
    /// Sum of squared unit-ball violations.
    pub throttle_violation: f64,
    /// Launch hyperbolic excess in kilometres per second; zero on later legs.
    pub launch_v_infinity_km_s: f64,
}

/// Fixed-endpoint low-thrust problem for one repaired scaffold leg.
#[derive(Clone, Debug)]
pub struct LowThrustLegProblem {
    leg_index: usize,
    segment_count: usize,
    duration_seconds: f64,
    initial_mass_kg: f64,
    initial_position: Vector3,
    final_position: Vector3,
    planet_departure_velocity: Vector3,
    fixed_departure_velocity: Vector3,
    final_velocity: Vector3,
}

impl LowThrustLegProblem {
    /// Constructs one leg problem at a specified incoming mass and segment count.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid leg index, mass, or segment count.
    pub fn new(
        scaffold: &SequenceScaffold,
        leg_index: usize,
        initial_mass_kg: f64,
        segment_count: usize,
    ) -> Result<Self, Gtoc1Error> {
        if leg_index >= scaffold.leg_count() {
            return Err(Gtoc1Error::InvalidDecision {
                index: leg_index,
                value: f64::NAN,
            });
        }
        if !initial_mass_kg.is_finite() || initial_mass_kg <= 0.0 || segment_count == 0 {
            return Err(Gtoc1Error::Numerical(
                "invalid low-thrust leg configuration",
            ));
        }
        let (initial_position, planet_departure_velocity) = split_state(scaffold.states[leg_index]);
        let (final_position, _) = split_state(scaffold.states[leg_index + 1]);
        Ok(Self {
            leg_index,
            segment_count,
            duration_seconds: scaffold.schedule[leg_index + 1] * DAY_SECONDS,
            initial_mass_kg,
            initial_position,
            final_position,
            planet_departure_velocity,
            fixed_departure_velocity: scaffold.departure_velocities[leg_index],
            final_velocity: scaffold.arrival_velocities[leg_index],
        })
    }

    /// Zero-based leg index.
    #[must_use]
    pub const fn leg_index(&self) -> usize {
        self.leg_index
    }

    /// Number of Sims-Flanagan impulses.
    #[must_use]
    pub const fn segment_count(&self) -> usize {
        self.segment_count
    }

    /// Decision-vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        3 * self.segment_count + if self.leg_index == 0 { 4 } else { 1 }
    }

    /// Complete leg duration in days.
    #[must_use]
    pub fn duration_days(&self) -> f64 {
        self.duration_seconds / DAY_SECONDS
    }

    /// Full box bounds for launch geometry and Cartesian throttles.
    ///
    /// # Panics
    ///
    /// Panics only if the compile-time bounds are inconsistent.
    #[must_use]
    pub fn bounds(&self) -> RetryBounds {
        let mut lower = Vec::with_capacity(self.dimension());
        let mut upper = Vec::with_capacity(self.dimension());
        if self.leg_index == 0 {
            lower.extend([0.0, 0.0, -core::f64::consts::PI]);
            upper.extend([
                MAXIMUM_LAUNCH_V_INFINITY_KM_S,
                core::f64::consts::PI,
                core::f64::consts::PI,
            ]);
        }
        lower.push(500.0_f64.min(0.5 * self.initial_mass_kg));
        upper.push(self.initial_mass_kg);
        for _ in 0..self.segment_count {
            lower.extend([-1.0; 3]);
            upper.extend([1.0; 3]);
        }
        RetryBounds::new(lower, upper).expect("low-thrust bounds are valid")
    }

    /// Zero-throttle starting point with an optional launch-vector seed.
    #[must_use]
    pub fn initial_guess(&self, launch_v_infinity_m_s: Option<Vector3>) -> Vec<f64> {
        let mut guess = vec![0.0; self.dimension()];
        if self.leg_index == 0 {
            let launch = launch_v_infinity_m_s.unwrap_or_else(|| {
                subtract(
                    self.fixed_departure_velocity,
                    self.planet_departure_velocity,
                )
            });
            let launch_norm = norm(launch);
            if launch_norm > 0.0 {
                guess[0] = (launch_norm / 1_000.0).min(MAXIMUM_LAUNCH_V_INFINITY_KM_S);
                guess[1] = (launch[2] / launch_norm).clamp(-1.0, 1.0).acos();
                guess[2] = launch[1].atan2(launch[0]);
            }
            guess[3] = self.initial_mass_kg;
        } else {
            guess[0] = self.initial_mass_kg;
        }
        guess
    }

    /// Evaluates the scalar objective for an optimizer callback.
    #[must_use]
    pub fn objective(&self, x: &[f64]) -> f64 {
        self.evaluate(x)
            .map_or(fcmaes_core::NAN_REPLACEMENT, |value| value.objective)
    }

    /// Evaluates the objective with a caller-selected constraint penalty.
    ///
    /// This is useful for continuation: a moderate penalty first discovers a
    /// low-propellant control structure, then a larger penalty tightens the
    /// endpoint match.
    #[must_use]
    pub fn objective_with_penalty(&self, x: &[f64], penalty: f64) -> f64 {
        if !penalty.is_finite() || penalty <= 0.0 {
            return fcmaes_core::NAN_REPLACEMENT;
        }
        self.evaluate(x)
            .map_or(fcmaes_core::NAN_REPLACEMENT, |value| {
                penalty * (value.mismatch_norm.powi(2) + value.throttle_violation) + value.fuel_kg
            })
    }

    /// Evaluates complete mismatch, throttle, mass, and launch diagnostics.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong decision dimension or a failed
    /// Sims-Flanagan propagation.
    pub fn evaluate(&self, x: &[f64]) -> Result<LowThrustLegEvaluation, Gtoc1Error> {
        if x.len() != self.dimension() {
            return Err(Gtoc1Error::Dimension { actual: x.len() });
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(Gtoc1Error::Numerical("non-finite low-thrust decision"));
        }
        let (departure_velocity, launch_v_infinity_km_s, mass_index, offset) =
            if self.leg_index == 0 {
                let direction = spherical_direction(x[1], x[2]);
                let launch = scale(direction, x[0] * 1_000.0);
                (
                    add(self.planet_departure_velocity, launch),
                    x[0],
                    3_usize,
                    4_usize,
                )
            } else {
                (self.fixed_departure_velocity, 0.0, 0_usize, 1_usize)
            };
        let throttles = x[offset..]
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();
        let maximum_throttle = throttles
            .iter()
            .map(|control| norm(*control))
            .fold(0.0, f64::max);
        let throttle_violation = throttles
            .iter()
            .map(|control| (dot(*control, *control) - 1.0).max(0.0).powi(2))
            .sum::<f64>();
        let final_mass_kg = x[mass_index];
        let leg = SimsFlanaganLeg::new(
            SpacecraftEndpoint::new(
                join_state(self.initial_position, departure_velocity),
                self.initial_mass_kg,
            )?,
            throttles,
            SpacecraftEndpoint::new(
                join_state(self.final_position, self.final_velocity),
                final_mass_kg,
            )?,
            SimsFlanaganSettings::new(
                self.duration_seconds,
                MAXIMUM_THRUST_NEWTONS,
                EXHAUST_VELOCITY_M_S,
                LEGACY_MU_SUN,
                0.5,
            )?,
        )?;
        let mismatch = leg.mismatch_constraints()?;
        let normalized = normalize_mismatch(mismatch);
        let mismatch_squared = dot_extended(normalized, normalized);
        let fuel_kg = self.initial_mass_kg - final_mass_kg;
        Ok(LowThrustLegEvaluation {
            objective: CONSTRAINT_PENALTY * (mismatch_squared + throttle_violation) + fuel_kg,
            mismatch_norm: mismatch_squared.sqrt(),
            mismatch,
            initial_mass_kg: self.initial_mass_kg,
            final_mass_kg,
            fuel_kg,
            maximum_throttle,
            throttle_violation,
            launch_v_infinity_km_s,
        })
    }

    /// Samples the impulsive Sims-Flanagan path and returns its smallest solar
    /// distance in astronomical units.
    ///
    /// `samples_per_coast` controls the subdivision of every coast interval.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid controls, zero sampling, or a failed
    /// Kepler propagation.
    pub fn minimum_heliocentric_distance_au(
        &self,
        x: &[f64],
        samples_per_coast: usize,
    ) -> Result<f64, Gtoc1Error> {
        if samples_per_coast == 0 {
            return Err(Gtoc1Error::Numerical("zero solar-distance samples"));
        }
        let evaluation = self.evaluate(x)?;
        let (departure_velocity, offset) = if self.leg_index == 0 {
            let launch = scale(spherical_direction(x[1], x[2]), x[0] * 1_000.0);
            (add(self.planet_departure_velocity, launch), 4)
        } else {
            (self.fixed_departure_velocity, 1)
        };
        let throttles = x[offset..]
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();
        let mut state = join_state(self.initial_position, departure_velocity);
        let mut mass = evaluation.initial_mass_kg;
        let segment_duration = self.duration_seconds
            / f64::from(
                u32::try_from(self.segment_count)
                    .map_err(|_| Gtoc1Error::Numerical("segment count too large"))?,
            );
        let mut minimum_radius = norm(self.initial_position);
        for (segment, &throttle) in throttles.iter().enumerate() {
            let coast = if segment == 0 {
                0.5 * segment_duration
            } else {
                segment_duration
            };
            minimum_radius =
                minimum_radius.min(sample_coast(&mut state, coast, samples_per_coast)?);
            let impulse_scale = MAXIMUM_THRUST_NEWTONS * segment_duration / mass;
            for component in 0..3 {
                state[component + 3] += impulse_scale * throttle[component];
            }
            let impulse_norm = impulse_scale * norm(throttle);
            mass *= (-impulse_norm / EXHAUST_VELOCITY_M_S).exp();
        }
        minimum_radius = minimum_radius.min(sample_coast(
            &mut state,
            0.5 * segment_duration,
            samples_per_coast,
        )?);
        Ok(minimum_radius / ASTRONOMICAL_UNIT_METRES)
    }
}

fn sample_coast(
    state: &mut CartesianState,
    duration: f64,
    samples: usize,
) -> Result<f64, Gtoc1Error> {
    let interval = duration
        / f64::from(
            u32::try_from(samples)
                .map_err(|_| Gtoc1Error::Numerical("solar sample count too large"))?,
        );
    let mut minimum = norm([state[0], state[1], state[2]]);
    for _ in 0..samples {
        *state = propagate_lagrangian(state, interval, LEGACY_MU_SUN)?;
        minimum = minimum.min(norm([state[0], state[1], state[2]]));
    }
    Ok(minimum)
}

fn normalize_mismatch(mismatch: [f64; 7]) -> [f64; 7] {
    [
        mismatch[0] / ASTRONOMICAL_UNIT_METRES,
        mismatch[1] / ASTRONOMICAL_UNIT_METRES,
        mismatch[2] / ASTRONOMICAL_UNIT_METRES,
        mismatch[3] / VELOCITY_SCALE_M_S,
        mismatch[4] / VELOCITY_SCALE_M_S,
        mismatch[5] / VELOCITY_SCALE_M_S,
        mismatch[6] / INITIAL_MASS_KG,
    ]
}

fn spherical_interpolate(start: Vector3, end: Vector3, fraction: f64) -> Vector3 {
    let angle = dot(start, end).clamp(-1.0, 1.0).acos();
    if angle < 1.0e-14 {
        return start;
    }
    let sine = angle.sin();
    if sine.abs() < 1.0e-14 {
        return normalize(add(scale(start, 1.0 - fraction), scale(end, fraction)));
    }
    add(
        scale(start, ((1.0 - fraction) * angle).sin() / sine),
        scale(end, (fraction * angle).sin() / sine),
    )
}

fn spherical_direction(theta: f64, phi: f64) -> Vector3 {
    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();
    [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
}

fn normalize(vector: Vector3) -> Vector3 {
    let length = norm(vector);
    if length == 0.0 {
        vector
    } else {
        scale(vector, 1.0 / length)
    }
}

fn norm(vector: Vector3) -> f64 {
    dot(vector, vector).sqrt()
}

fn add(left: Vector3, right: Vector3) -> Vector3 {
    core::array::from_fn(|index| left[index] + right[index])
}

fn scale(vector: Vector3, factor: f64) -> Vector3 {
    vector.map(|value| factor * value)
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

fn dot_extended(left: [f64; 7], right: [f64; 7]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route_archive::BranchChoice;
    use crate::route_search::{PhysicalDecision, RouteDerivationConfig};
    use crate::sequences::{
        DEIMOS, DEIMOS_HISTORICAL_DECISIONS, JENA, JENA_HISTORICAL_DECISIONS, JPL2,
        JPL2_HISTORICAL_DECISION,
    };

    #[test]
    fn repaired_scaffolds_have_unpowered_legal_flybys() {
        for (case, schedule) in [
            (JPL2, JPL2_HISTORICAL_DECISION.as_slice()),
            (JENA, JENA_HISTORICAL_DECISIONS[0].as_slice()),
            (DEIMOS, DEIMOS_HISTORICAL_DECISIONS[0].as_slice()),
        ] {
            let scout = case.evaluate_endpoint_repair_scout(schedule).unwrap();
            let scaffold = SequenceScaffold::new(case, schedule).unwrap();
            assert!(scaffold.flight_days() <= crate::real::MAXIMUM_FLIGHT_DAYS);
            assert!(scaffold.powered_delta_v_km_s() < 1.0e-7);
            assert!(scaffold.minimum_periapsis_margin_km() > -1.0e-5);
            assert_eq!(scaffold.leg_count(), case.bodies.len() - 1);
            if case.name == DEIMOS.name {
                assert_eq!(
                    scaffold.branches(),
                    [
                        (3, LambertPath::Right),
                        (3, LambertPath::Right),
                        (0, LambertPath::ZeroRevolution),
                        (1, LambertPath::Left),
                        (1, LambertPath::Left),
                        (1, LambertPath::Left),
                        (1, LambertPath::Right),
                        (1, LambertPath::Left),
                        (1, LambertPath::Right),
                        (0, LambertPath::ZeroRevolution),
                        (0, LambertPath::ZeroRevolution),
                        (0, LambertPath::ZeroRevolution),
                        (0, LambertPath::ZeroRevolution),
                    ]
                );
            } else if case.name == JPL2.name {
                assert_eq!(scaffold.branches()[5], (1, LambertPath::Right));
                assert!(
                    (scaffold.endpoint_repair_delta_v_km_s() - 0.275_592_619_188).abs() < 1.0e-9
                );
            } else if case.name == JENA.name {
                assert_eq!(scaffold.branches()[0], (2, LambertPath::Left));
                assert!((scaffold.endpoint_repair_delta_v_km_s() - 1.341_292).abs() < 1.0e-5);
            } else {
                assert_eq!(scaffold.branches(), scout.branches);
                assert!(
                    (scaffold.endpoint_repair_delta_v_km_s() - scout.endpoint_repair_delta_v_km_s)
                        .abs()
                        < 1.0e-12
                );
            }
        }
    }

    #[test]
    fn archived_runtime_route_reuses_the_exact_l0_branches() {
        let route = RouteCase::derive(
            RouteVariant::from_sequence_case(JPL2),
            RouteDerivationConfig::default(),
        )
        .unwrap();
        let physical = PhysicalDecision {
            launch_mjd2000: JPL2_HISTORICAL_DECISION[0],
            leg_days: JPL2_HISTORICAL_DECISION[1..].to_vec(),
        };
        let coordinates = route.codec().encode(&physical).unwrap();
        let evaluation = route.evaluate(&coordinates).unwrap();
        let branches = evaluation
            .sequence
            .branches
            .iter()
            .map(|&(revolutions, path)| BranchChoice {
                revolutions,
                path: path.into(),
            })
            .collect::<Vec<_>>();
        let scaffold = SequenceScaffold::from_selected_route(&route, &physical, &branches).unwrap();
        assert_eq!(scaffold.variant(), route.variant());
        assert_eq!(scaffold.schedule(), JPL2_HISTORICAL_DECISION);
        assert_eq!(scaffold.branches(), evaluation.sequence.branches);
        assert!(scaffold.case().is_none());
        assert!(scaffold.powered_delta_v_km_s() < 1.0e-7);
    }

    #[test]
    fn zero_controls_preserve_a_ballistic_fixed_endpoint_leg() {
        let scaffold = SequenceScaffold::new(JPL2, &JPL2_HISTORICAL_DECISION).unwrap();
        let problem = LowThrustLegProblem::new(&scaffold, 9, INITIAL_MASS_KG, 12).unwrap();
        let guess = problem.initial_guess(None);
        let evaluation = problem.evaluate(&guess).unwrap();
        assert!(evaluation.mismatch_norm.is_finite());
        assert!(evaluation.fuel_kg.abs() < f64::EPSILON);
        assert!(evaluation.maximum_throttle.abs() < f64::EPSILON);
    }
}
