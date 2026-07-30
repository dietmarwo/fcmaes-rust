// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Fast ballistic scouting of alternate GTOC1 planet sequences.
//!
//! The scout deliberately matches [`crate::real::evaluate_ballistic_backbone`]:
//! VSOP2013 planetary states, all configured multi-revolution Lambert
//! solutions, dynamic programming across flybys, and a fixed impact mass of
//! 1442.9 kg. It is a sequence-ranking model, not a replacement for the
//! propelled-leg and low-thrust feasibility transcription.

use fcmaes_core::RetryBounds;
use pykep_core::astro::lambert::{LambertPath, LambertProblem};
use pykep_core::{CartesianState, Vector3};

use crate::real::{JPL_DECISION, MAXIMUM_FLIGHT_DAYS, competition_state};
use crate::{
    BODY_MU_KM3_S2, DAY_SECONDS, Gtoc1Error, LEGACY_MU_SUN, distance, dot, powered_swingby_inverse,
    split_state, subtract,
};

const FIXED_FINAL_MASS_KG: f64 = 1_442.9;
const INITIAL_MASS_KG: f64 = 1_500.0;
const EXHAUST_VELOCITY_KM_S: f64 = 2_500.0 * 9.806_65 / 1_000.0;
const LAUNCH_V_INFINITY_KM_S: f64 = 2.5;
const CONSTRAINT_PENALTY: f64 = 1.0e10;
const MINIMUM_PERIAPSIS_KM: [f64; 9] = [
    0.0, 2_740.0, 6_351.0, 6_678.0, 3_689.0, 600_000.0, 70_000.0, 0.0, 0.0,
];

/// JPL's winning EVEEEJSJA sequence, included as the published-date baseline.
pub const JPL_SEQUENCE: [usize; 9] = [3, 2, 3, 3, 3, 5, 6, 5, 10];
/// Direction flags for the outgoing legs of [`JPL_SEQUENCE`].
pub const JPL_REV_FLAGS: [bool; 9] = [false, false, false, false, false, false, true, true, false];
/// EVVEEEEJSJA sequence labelled `JPL2` by the historical local Java model.
pub const JPL2_SEQUENCE: [usize; 11] = [3, 2, 2, 3, 3, 3, 3, 5, 6, 5, 10];
/// Direction flags for the outgoing legs of [`JPL2_SEQUENCE`].
pub const JPL2_REV_FLAGS: [bool; 11] = [
    false, false, false, false, false, false, false, false, true, true, false,
];
/// Jena's EVVEVVEESJA sequence.
pub const JENA_SEQUENCE: [usize; 11] = [3, 2, 2, 3, 2, 2, 3, 3, 6, 5, 10];
/// Direction flags for the outgoing legs of [`JENA_SEQUENCE`].
pub const JENA_REV_FLAGS: [bool; 11] = [
    false, false, false, false, false, false, false, false, true, true, false,
];
/// Deimos Space's EVVEEVVEVEJSJA competition sequence.
pub const DEIMOS_SEQUENCE: [usize; 14] = [3, 2, 2, 3, 3, 2, 2, 3, 2, 3, 5, 6, 5, 10];
/// Historical direction flags for the outgoing legs of [`DEIMOS_SEQUENCE`].
pub const DEIMOS_REV_FLAGS: [bool; 14] = [
    false, false, false, false, false, false, false, false, false, false, false, true, true, false,
];

const JPL_MAXIMUM_REVOLUTIONS: [usize; 8] = [3, 3, 4, 5, 1, 1, 2, 1];
const JPL2_MAXIMUM_REVOLUTIONS: [usize; 10] = [3, 5, 3, 5, 5, 5, 1, 1, 2, 1];
const JENA_MAXIMUM_REVOLUTIONS: [usize; 10] = [3, 5, 3, 3, 5, 3, 5, 1, 2, 1];
const DEIMOS_MAXIMUM_REVOLUTIONS: [usize; 13] = [3, 5, 3, 5, 3, 5, 3, 3, 3, 1, 1, 2, 1];

const JPL_LOWER: [f64; 9] = [
    3_653.0, 700.0, 650.0, 850.0, 1_300.0, 300.0, 300.0, 2_400.0, 300.0,
];
const JPL_UPPER: [f64; 9] = [
    10_958.0, 1_700.0, 1_250.0, 1_550.0, 2_200.0, 750.0, 750.0, 4_100.0, 850.0,
];
const JPL2_LOWER: [f64; 11] = [
    3_653.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 300.0, 300.0, 300.0, 300.0,
];
const JPL2_UPPER: [f64; 11] = [
    10_958.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 1_000.0, 1_000.0, 4_000.0,
    1_000.0,
];
const JENA_LOWER: [f64; 11] = [
    3_653.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 300.0, 300.0, 300.0, 300.0,
];
const JENA_UPPER: [f64; 11] = [
    10_958.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 4_000.0, 4_000.0, 4_000.0,
    4_000.0,
];
const DEIMOS_LOWER: [f64; 14] = [
    3_653.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 300.0, 300.0, 2_400.0, 300.0,
];
const DEIMOS_UPPER: [f64; 14] = [
    10_958.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0, 2_000.0,
    1_000.0, 1_000.0, 4_000.0, 1_000.0,
];

/// Historical JPL2 EVVEEEEJSJA schedule used to seed Rust refinement.
pub const JPL2_HISTORICAL_DECISION: [f64; 11] = [
    8_140.888_828_37,
    1_011.795_383_4,
    834.812_473_05,
    398.974_238_85,
    1_309.917_833_71,
    602.193_607_46,
    1_891.862_240_17,
    472.042_841_97,
    482.291_305_17,
    3_271.493_982_36,
    544.049_024_39,
];
/// Three historical Jena EVVEVVEESJA schedules used as independent seeds.
pub const JENA_HISTORICAL_DECISIONS: [[f64; 11]; 3] = [
    [
        8_168.992_985_33,
        751.285_123_02,
        666.992_828_89,
        797.342_089_69,
        156.089_725_78,
        848.754_326_48,
        572.387_480_53,
        2_645.595_737_3,
        836.153_585_73,
        2_992.221_036_98,
        524.850_496_97,
    ],
    [
        8_538.862_622_1,
        154.154_040_13,
        1_119.915_640_12,
        632.061_107_51,
        911.387_653_97,
        504.231_188_19,
        833.479_246_31,
        2_300.185_417_42,
        819.063_146_26,
        2_652.763_512_3,
        499.831_903_94,
    ],
    [
        8_537.314_214_060_178,
        155.907_019_665_345_73,
        1_119.970_407_582_337_7,
        630.878_208_892_383_1,
        911.844_834_133_636_5,
        504.047_723_519_743_8,
        833.366_426_518_142_1,
        2_300.059_103_481_938,
        841.216_823_349_245_3,
        2_633.219_865_144_814_3,
        498.415_203_998_790_5,
    ],
];

/// Calendar-derived Deimos schedule followed by historical Java refinements.
pub const DEIMOS_HISTORICAL_DECISIONS: [[f64; 14]; 8] = [
    [
        8_608.103_578_518_52,
        996.875,
        786.401_388_888_887_6,
        268.454_861_111_111_3,
        365.256_944_444_445_25,
        369.813_888_888_887_96,
        224.686_111_111_112_04,
        451.831_249_999_999_3,
        1_075.199_305_555_555_8,
        1_077.900_694_444_444_5,
        443.196_527_777_778_1,
        480.100_694_444_443_43,
        3_264.543_749_999_999,
        543.768_750_000_002_9,
    ],
    [
        8_215.437_354_44,
        690.890_671_6,
        946.545_805_63,
        289.785_762_86,
        546.889_718_71,
        519.973_499_17,
        585.328_299_21,
        647.458_669_53,
        471.122_849_72,
        1_314.895_971_1,
        440.375_570_84,
        480.608_859_52,
        3_263.367_701_76,
        545.121_580_41,
    ],
    [
        8_390.498_208_723_5,
        405.839_649_652_7,
        323.950_123_433_4,
        524.174_415_007_7,
        703.572_366_019_7,
        400.135_755_607_3,
        1_047.924_367_903_5,
        646.294_345_316_3,
        472.354_642_577_2,
        1_313.181_303_076_2,
        440.687_473_301_3,
        480.610_594_015_8,
        3_263.299_388_085_2,
        545.181_036_527_9,
    ],
    [
        8_241.749_993_52,
        674.288_762_6,
        454.888_683_48,
        715.084_294_26,
        551.546_459_36,
        543.648_183_1,
        379.390_742_99,
        854.380_666_76,
        717.620_523_14,
        1_096.486_958_05,
        440.196_986,
        481.316_588_62,
        3_269.699_642_98,
        542.381_238_73,
    ],
    [
        8_411.878_623_78,
        385.941_055_1,
        329.308_955_66,
        505.558_612_14,
        683.759_400_98,
        439.272_132_9,
        1_036.520_749_05,
        473.219_841_9,
        656.143_501_96,
        1_304.868_063_57,
        441.945_726_97,
        480.837_224_6,
        3_264.988_662_66,
        544.574_337_22,
    ],
    [
        8_245.567_432_803_487,
        672.075_365_484_18,
        468.853_081_081_561_64,
        396.303_330_645_667_4,
        557.912_809_499_464_5,
        656.708_260_466_137_1,
        797.717_099_023_269_8,
        647.178_666_100_305_5,
        471.420_513_655_216_04,
        1_314.494_420_821_405_6,
        440.432_231_888_369_64,
        480.576_341_992_805_5,
        3_263.062_741_489_001,
        545.257_853_502_095_7,
    ],
    [
        8_256.981_314_86,
        665.469_153_11,
        466.120_144_69,
        394.659_372_43,
        557.699_144_27,
        656.420_192_59,
        798.167_669_45,
        646.818_893_42,
        471.801_696_71,
        1_313.969_071_17,
        440.538_050_81,
        480.594_004_12,
        3_263.191_605_45,
        545.212_855_1,
    ],
    [
        8_266.374_604_21,
        661.166_561_33,
        470.975_675_72,
        567.609_884_48,
        465.379_516_95,
        519.770_067_4,
        841.208_560_63,
        640.415_517_95,
        526.446_607_45,
        1_164.955_474_59,
        522.302_271_03,
        505.098_643_69,
        3_269.172_711_44,
        542.130_546_11,
    ],
];

/// Static definition of one sequence-scouting problem.
#[derive(Clone, Copy, Debug)]
pub struct SequenceCase {
    /// Short identifier used by the command-line driver.
    pub name: &'static str,
    /// Encounter bodies, using the GTOP convention (Venus 2, Earth 3,
    /// Jupiter 5, Saturn 6, asteroid 10).
    pub bodies: &'static [usize],
    /// Per-node direction flags; `true` requests a clockwise Lambert solution
    /// for the outgoing leg. The last node has no outgoing leg.
    pub rev_flags: &'static [bool],
    /// Largest Lambert revolution count considered on each leg.
    pub maximum_revolutions: &'static [usize],
    /// Lower bounds for launch MJD2000 followed by leg durations in days.
    pub lower: &'static [f64],
    /// Upper bounds for launch MJD2000 followed by leg durations in days.
    pub upper: &'static [f64],
    /// Resonance-informed starting point for optimization.
    pub guess: &'static [f64],
}

/// Published JPL EVEEEJSJA baseline.
pub const JPL: SequenceCase = SequenceCase {
    name: "JPL",
    bodies: &JPL_SEQUENCE,
    rev_flags: &JPL_REV_FLAGS,
    maximum_revolutions: &JPL_MAXIMUM_REVOLUTIONS,
    lower: &JPL_LOWER,
    upper: &JPL_UPPER,
    guess: &JPL_DECISION,
};

/// Historical local `JPL2` EVVEEEEJSJA case.
pub const JPL2: SequenceCase = SequenceCase {
    name: "JPL2",
    bodies: &JPL2_SEQUENCE,
    rev_flags: &JPL2_REV_FLAGS,
    maximum_revolutions: &JPL2_MAXIMUM_REVOLUTIONS,
    lower: &JPL2_LOWER,
    upper: &JPL2_UPPER,
    guess: &JPL2_HISTORICAL_DECISION,
};

/// Jena EVVEVVEESJA sequence.
pub const JENA: SequenceCase = SequenceCase {
    name: "JENA",
    bodies: &JENA_SEQUENCE,
    rev_flags: &JENA_REV_FLAGS,
    maximum_revolutions: &JENA_MAXIMUM_REVOLUTIONS,
    lower: &JENA_LOWER,
    upper: &JENA_UPPER,
    guess: &JENA_HISTORICAL_DECISIONS[0],
};

/// Deimos Space EVVEEVVEVEJSJA case later reoptimized by JPL.
pub const DEIMOS: SequenceCase = SequenceCase {
    name: "DEIMOS",
    bodies: &DEIMOS_SEQUENCE,
    rev_flags: &DEIMOS_REV_FLAGS,
    maximum_revolutions: &DEIMOS_MAXIMUM_REVOLUTIONS,
    lower: &DEIMOS_LOWER,
    upper: &DEIMOS_UPPER,
    guess: &DEIMOS_HISTORICAL_DECISIONS[0],
};

/// Result of one fast sequence evaluation.
#[derive(Clone, Debug)]
pub struct SequenceEvaluation {
    /// Penalized minimization objective.
    pub objective: f64,
    /// Unpenalized impact score at the fixed 1442.9 kg mass.
    pub score: f64,
    /// Sum of squared feasibility residuals used by the objective.
    pub constraint: f64,
    /// Earth-departure hyperbolic excess in kilometres per second.
    pub launch_v_infinity_km_s: f64,
    /// Sum of powered impulses needed to connect the ballistic flybys.
    pub powered_delta_v_km_s: f64,
    /// Sum of the endpoint-velocity changes made by the low-thrust scaffold
    /// repair, including the first Venus encounter.
    pub endpoint_repair_delta_v_km_s: f64,
    /// Final mass estimated from launch excess plus endpoint repair.
    pub estimated_final_mass_kg: f64,
    /// Impact score scaled by [`Self::estimated_final_mass_kg`].
    pub estimated_score: f64,
    /// Smallest periapsis margin relative to the competition limits.
    pub minimum_periapsis_margin_km: f64,
    /// Selected `(revolutions, path)` for every leg.
    pub branches: Vec<(usize, LambertPath)>,
    /// Encounter epochs in MJD2000 days.
    pub epochs_mjd2000: Vec<f64>,
}

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
    endpoint_repair_delta_v_km_s: f64,
    minimum_periapsis_margin_km: f64,
    branches: Vec<(usize, LambertPath)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScoutMode {
    AllBallistic,
    Propulsive,
    EndpointRepair,
}

impl SequenceCase {
    /// Returns validated fcmaes bounds for this case.
    ///
    /// # Panics
    ///
    /// Panics only when a compile-time case definition is inconsistent.
    #[must_use]
    pub fn bounds(self) -> RetryBounds {
        RetryBounds::new(self.lower.to_vec(), self.upper.to_vec())
            .expect("sequence bounds are valid")
    }

    /// Constructs a bounded neighborhood around a schedule.
    ///
    /// `fraction` is measured against each coordinate's global box width.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong center dimension or a non-positive,
    /// non-finite fraction.
    pub fn refinement_bounds(
        self,
        center: &[f64],
        fraction: f64,
    ) -> Result<RetryBounds, Gtoc1Error> {
        if center.len() != self.bodies.len() {
            return Err(Gtoc1Error::Dimension {
                actual: center.len(),
            });
        }
        if !fraction.is_finite() || fraction <= 0.0 {
            return Err(Gtoc1Error::InvalidDecision {
                index: center.len(),
                value: fraction,
            });
        }
        let mut lower = Vec::with_capacity(center.len());
        let mut upper = Vec::with_capacity(center.len());
        for (index, &value) in center.iter().enumerate() {
            let radius = fraction * (self.upper[index] - self.lower[index]);
            lower.push(self.lower[index].max(value - radius));
            upper.push(self.upper[index].min(value + radius));
        }
        RetryBounds::new(lower, upper)
            .map_err(|_| Gtoc1Error::Numerical("sequence refinement bounds"))
    }

    /// Evaluates all configured Lambert branches at a proposed schedule.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dates, unsupported bodies, ephemeris
    /// failures, or an empty/disconnected Lambert family.
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(self, x: &[f64]) -> Result<SequenceEvaluation, Gtoc1Error> {
        self.evaluate_internal(x, ScoutMode::AllBallistic)
    }

    /// Evaluates the fast propelled-first-leg sequence proxy.
    ///
    /// Unlike [`Self::evaluate`], this does not penalize the ballistic launch
    /// excess or force the first Lambert arc to connect at Venus. Those
    /// quantities are controlled by the propelled Earth-Venus transcription
    /// in the real competition model. Flyby matching begins with the
    /// Venus-departure Lambert arc.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::evaluate`].
    pub fn evaluate_propulsive_scout(self, x: &[f64]) -> Result<SequenceEvaluation, Gtoc1Error> {
        self.evaluate_internal(x, ScoutMode::Propulsive)
    }

    /// Evaluates a fuel-correlated cheap proxy for the complete sequence.
    ///
    /// The Lambert branches are selected as in [`Self::evaluate_propulsive_scout`],
    /// but the objective scales the impact score by a rocket-equation mass
    /// estimate based on launch excess and the endpoint changes needed by the
    /// low-thrust flyby repair. This remains a screening model, but unlike the
    /// propelled-first-leg scout it cannot silently accept a multi-km/s first
    /// Venus endpoint change.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::evaluate`].
    pub fn evaluate_endpoint_repair_scout(
        self,
        x: &[f64],
    ) -> Result<SequenceEvaluation, Gtoc1Error> {
        self.evaluate_internal(x, ScoutMode::EndpointRepair)
    }

    /// Scalar propelled-first-leg callback for fcmaes optimizers.
    #[must_use]
    pub fn propulsive_objective(self, x: &[f64]) -> f64 {
        self.evaluate_propulsive_scout(x)
            .map_or(fcmaes_core::NAN_REPLACEMENT, |value| value.objective)
    }

    /// Scalar endpoint-repair proxy callback for fcmaes optimizers.
    #[must_use]
    pub fn endpoint_repair_objective(self, x: &[f64]) -> f64 {
        self.evaluate_endpoint_repair_scout(x)
            .map_or(fcmaes_core::NAN_REPLACEMENT, |value| value.objective)
    }

    #[allow(clippy::too_many_lines)]
    fn evaluate_internal(
        self,
        x: &[f64],
        mode: ScoutMode,
    ) -> Result<SequenceEvaluation, Gtoc1Error> {
        self.validate()?;
        evaluate_sequence(
            self.bodies,
            &self.rev_flags[..self.bodies.len() - 1],
            self.maximum_revolutions,
            x,
            mode,
        )
    }

    /// Scalar callback for fcmaes optimizers.
    #[must_use]
    pub fn objective(self, x: &[f64]) -> f64 {
        self.evaluate(x)
            .map_or(fcmaes_core::NAN_REPLACEMENT, |value| value.objective)
    }

    fn validate(self) -> Result<(), Gtoc1Error> {
        let dimension = self.bodies.len();
        if dimension < 2
            || self.rev_flags.len() != dimension
            || self.maximum_revolutions.len() + 1 != dimension
            || self.lower.len() != dimension
            || self.upper.len() != dimension
            || self.guess.len() != dimension
        {
            return Err(Gtoc1Error::Numerical("invalid sequence definition"));
        }
        Ok(())
    }
}

/// Evaluates the endpoint-repair L0 scout for a runtime route definition.
///
/// `physical_decision` is the launch epoch followed by one duration per leg.
/// Clockwise direction and Lambert path are independent: the former is fixed
/// here, while the dynamic program selects zero-, left-, or right-path
/// solutions from every configured revolution family.
///
/// # Errors
///
/// Returns an error for inconsistent dimensions, invalid epochs, ephemeris
/// failures, or an unavailable/disconnected Lambert family.
pub fn evaluate_runtime_endpoint_repair(
    bodies: &[usize],
    clockwise: &[bool],
    maximum_revolutions: &[usize],
    physical_decision: &[f64],
) -> Result<SequenceEvaluation, Gtoc1Error> {
    evaluate_sequence(
        bodies,
        clockwise,
        maximum_revolutions,
        physical_decision,
        ScoutMode::EndpointRepair,
    )
}

#[allow(clippy::too_many_lines)]
fn evaluate_sequence(
    bodies: &[usize],
    clockwise: &[bool],
    maximum_revolutions: &[usize],
    x: &[f64],
    mode: ScoutMode,
) -> Result<SequenceEvaluation, Gtoc1Error> {
    let dimension = bodies.len();
    if dimension < 2
        || clockwise.len() + 1 != dimension
        || maximum_revolutions.len() + 1 != dimension
        || x.len() != dimension
    {
        return Err(Gtoc1Error::Dimension { actual: x.len() });
    }
    let epochs = encounter_epochs(dimension, x)?;
    let states = bodies
        .iter()
        .zip(&epochs)
        .map(|(&body, &epoch)| competition_state(body, epoch))
        .collect::<Result<Vec<CartesianState>, _>>()?;
    let leg_count = bodies.len() - 1;
    let mut arc_families = Vec::with_capacity(leg_count);
    for leg in 0..leg_count {
        let (initial_position, _) = split_state(states[leg]);
        let (final_position, _) = split_state(states[leg + 1]);
        let problem = LambertProblem::new(
            initial_position,
            final_position,
            x[leg + 1] * DAY_SECONDS,
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
                constraint: if matches!(mode, ScoutMode::AllBallistic | ScoutMode::EndpointRepair) {
                    excess * excess
                } else {
                    0.0
                },
                launch_v_infinity_km_s,
                powered_delta_v_km_s: 0.0,
                endpoint_repair_delta_v_km_s: 0.0,
                minimum_periapsis_margin_km: f64::INFINITY,
                branches: vec![(arc.revolutions, arc.path)],
            }
        })
        .collect::<Vec<_>>();

    for leg in 1..leg_count {
        let (_, planet_velocity) = split_state(states[leg]);
        let body = bodies[leg];
        let mut next_paths = Vec::with_capacity(arc_families[leg].len());
        for current in &arc_families[leg] {
            if leg == 1 && mode == ScoutMode::Propulsive {
                let path = paths
                    .iter()
                    .min_by(|left, right| {
                        left.launch_v_infinity_km_s
                            .total_cmp(&right.launch_v_infinity_km_s)
                    })
                    .ok_or(Gtoc1Error::Numerical("empty propelled Lambert family"))?;
                let mut branches = path.branches.clone();
                branches.push((current.revolutions, current.path));
                next_paths.push(PartialPath {
                    constraint: 0.0,
                    launch_v_infinity_km_s: path.launch_v_infinity_km_s,
                    powered_delta_v_km_s: 0.0,
                    endpoint_repair_delta_v_km_s: 0.0,
                    minimum_periapsis_margin_km: f64::INFINITY,
                    branches,
                });
                continue;
            }
            let mut best: Option<PartialPath> = None;
            for (previous, path) in arc_families[leg - 1].iter().zip(&paths) {
                let incoming = subtract(previous.arrival_velocity, planet_velocity);
                let outgoing = subtract(current.departure_velocity, planet_velocity);
                let incoming_norm = vector_norm(incoming);
                let outgoing_norm = vector_norm(outgoing);
                if incoming_norm == 0.0 || outgoing_norm == 0.0 {
                    continue;
                }
                let angle = (dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
                    .clamp(-1.0, 1.0)
                    .acos();
                let Ok((delta_v, nondimensional_periapsis)) = powered_swingby_inverse(
                    incoming_norm / 1_000.0,
                    outgoing_norm / 1_000.0,
                    angle,
                ) else {
                    continue;
                };
                let periapsis = nondimensional_periapsis * BODY_MU_KM3_S2[body];
                let margin = periapsis - MINIMUM_PERIAPSIS_KM[body];
                let normalized_shortfall = (-margin).max(0.0) / MINIMUM_PERIAPSIS_KM[body].max(1.0);
                let constraint = if mode == ScoutMode::EndpointRepair {
                    path.constraint + normalized_shortfall.powi(2)
                } else {
                    path.constraint + delta_v * delta_v + normalized_shortfall.powi(2)
                };
                let (repaired_incoming, repaired_outgoing) =
                    repair_relative_velocities(incoming, outgoing, body)?;
                let endpoint_repair_delta_v_km_s = path.endpoint_repair_delta_v_km_s
                    + (distance(incoming, repaired_incoming)
                        + distance(outgoing, repaired_outgoing))
                        / 1_000.0;
                let improves = best.as_ref().is_none_or(|candidate| {
                    if mode == ScoutMode::EndpointRepair {
                        constraint < candidate.constraint
                            || (constraint.total_cmp(&candidate.constraint).is_eq()
                                && endpoint_repair_delta_v_km_s
                                    < candidate.endpoint_repair_delta_v_km_s)
                    } else {
                        constraint < candidate.constraint
                    }
                });
                if improves {
                    let mut branches = path.branches.clone();
                    branches.push((current.revolutions, current.path));
                    best = Some(PartialPath {
                        constraint,
                        launch_v_infinity_km_s: path.launch_v_infinity_km_s,
                        powered_delta_v_km_s: path.powered_delta_v_km_s + delta_v,
                        endpoint_repair_delta_v_km_s,
                        minimum_periapsis_margin_km: path.minimum_periapsis_margin_km.min(margin),
                        branches,
                    });
                }
            }
            next_paths.push(best.ok_or(Gtoc1Error::Numerical("no connected Lambert branch"))?);
        }
        paths = next_paths;
    }

    let (_, asteroid_velocity) = split_state(states[states.len() - 1]);
    let mut best: Option<SequenceEvaluation> = None;
    for (arc, path) in arc_families[leg_count - 1].iter().zip(paths) {
        let relative = subtract(asteroid_velocity, arc.arrival_velocity);
        let score = FIXED_FINAL_MASS_KG * dot(relative, asteroid_velocity) / 1.0e6;
        let endpoint_repair_delta_v_km_s =
            endpoint_repair_delta_v(&path.branches, &arc_families, &states, bodies)?;
        let estimated_delta_v_km_s = endpoint_repair_delta_v_km_s
            + (path.launch_v_infinity_km_s - LAUNCH_V_INFINITY_KM_S).max(0.0);
        let estimated_final_mass_kg =
            INITIAL_MASS_KG * (-estimated_delta_v_km_s / EXHAUST_VELOCITY_KM_S).exp();
        let estimated_score = score * estimated_final_mass_kg / FIXED_FINAL_MASS_KG;
        let objective = if mode == ScoutMode::EndpointRepair {
            CONSTRAINT_PENALTY * path.constraint - estimated_score
        } else {
            CONSTRAINT_PENALTY * path.constraint - score
        };
        let candidate = SequenceEvaluation {
            objective,
            score,
            constraint: path.constraint,
            launch_v_infinity_km_s: path.launch_v_infinity_km_s,
            powered_delta_v_km_s: path.powered_delta_v_km_s,
            endpoint_repair_delta_v_km_s,
            estimated_final_mass_kg,
            estimated_score,
            minimum_periapsis_margin_km: path.minimum_periapsis_margin_km,
            branches: path.branches,
            epochs_mjd2000: epochs.clone(),
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

fn encounter_epochs(dimension: usize, x: &[f64]) -> Result<Vec<f64>, Gtoc1Error> {
    if x.len() != dimension {
        return Err(Gtoc1Error::Dimension { actual: x.len() });
    }
    if !x[0].is_finite() || !(3_653.0..=10_958.0).contains(&x[0]) {
        return Err(Gtoc1Error::InvalidDecision {
            index: 0,
            value: x[0],
        });
    }
    let mut epochs = Vec::with_capacity(x.len());
    epochs.push(x[0]);
    for (index, &duration) in x.iter().enumerate().skip(1) {
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
            index: x.len() - 1,
            value: flight_days,
        });
    }
    Ok(epochs)
}

fn endpoint_repair_delta_v(
    branches: &[(usize, LambertPath)],
    arc_families: &[Vec<Arc>],
    states: &[CartesianState],
    bodies: &[usize],
) -> Result<f64, Gtoc1Error> {
    let arcs = arc_families
        .iter()
        .zip(branches)
        .map(|(family, &(revolutions, path))| {
            family
                .iter()
                .find(|arc| arc.revolutions == revolutions && arc.path == path)
                .ok_or(Gtoc1Error::Numerical(
                    "selected branch missing while estimating endpoint repair",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut delta_v_m_s = 0.0;
    for node in 1..bodies.len() - 1 {
        let (_, planet_velocity) = split_state(states[node]);
        let incoming = subtract(arcs[node - 1].arrival_velocity, planet_velocity);
        let outgoing = subtract(arcs[node].departure_velocity, planet_velocity);
        let (repaired_incoming, repaired_outgoing) =
            repair_relative_velocities(incoming, outgoing, bodies[node])?;
        delta_v_m_s += distance(incoming, repaired_incoming);
        delta_v_m_s += distance(outgoing, repaired_outgoing);
    }
    Ok(delta_v_m_s / 1_000.0)
}

fn repair_relative_velocities(
    incoming: Vector3,
    outgoing: Vector3,
    body: usize,
) -> Result<(Vector3, Vector3), Gtoc1Error> {
    let incoming_norm = vector_norm(incoming);
    let outgoing_norm = vector_norm(outgoing);
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
    ))
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

fn normalize(vector: Vector3) -> Vector3 {
    let length = vector_norm(vector);
    if length == 0.0 {
        vector
    } else {
        scale(vector, 1.0 / length)
    }
}

fn add(left: Vector3, right: Vector3) -> Vector3 {
    core::array::from_fn(|index| left[index] + right[index])
}

fn scale(vector: Vector3, factor: f64) -> Vector3 {
    vector.map(|value| factor * value)
}

fn vector_norm(vector: Vector3) -> f64 {
    dot(vector, vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_supplied_sequence_definitions_are_consistent() {
        JPL2.validate().unwrap();
        JENA.validate().unwrap();
        DEIMOS.validate().unwrap();
        assert_eq!(JPL2.bodies, [3, 2, 2, 3, 3, 3, 3, 5, 6, 5, 10]);
        assert_eq!(JENA.bodies, [3, 2, 2, 3, 2, 2, 3, 3, 6, 5, 10]);
        assert_eq!(DEIMOS.bodies, [3, 2, 2, 3, 3, 2, 2, 3, 2, 3, 5, 6, 5, 10]);
        assert!(JPL2.rev_flags[8] && JPL2.rev_flags[9]);
        assert!(JENA.rev_flags[8] && JENA.rev_flags[9]);
        assert!(DEIMOS.rev_flags[11] && DEIMOS.rev_flags[12]);
    }

    #[test]
    fn generalized_baseline_matches_specialized_evaluator() {
        let generalized = JPL.evaluate(&JPL_DECISION).unwrap();
        let specialized = crate::real::evaluate_ballistic_backbone(&JPL_DECISION).unwrap();
        assert!((generalized.objective - specialized.objective).abs() < 1.0e-3);
        assert!((generalized.score - specialized.score).abs() < 1.0e-6);
        assert_eq!(generalized.branches, specialized.branches);
    }

    #[test]
    fn alternate_sequence_guesses_have_finite_evaluations() {
        let cases_and_guesses = [
            (JPL2, JPL2_HISTORICAL_DECISION.as_slice()),
            (JENA, JENA_HISTORICAL_DECISIONS[0].as_slice()),
            (JENA, JENA_HISTORICAL_DECISIONS[1].as_slice()),
            (JENA, JENA_HISTORICAL_DECISIONS[2].as_slice()),
        ];
        for (case, guess) in cases_and_guesses {
            let evaluation = case.evaluate(guess).unwrap();
            assert!(evaluation.objective.is_finite());
            assert!(evaluation.score.is_finite());
            assert_eq!(evaluation.branches.len(), case.bodies.len() - 1);
            case.refinement_bounds(guess, 0.01).unwrap();
        }
        for guess in &DEIMOS_HISTORICAL_DECISIONS {
            let evaluation = DEIMOS.evaluate(guess).unwrap();
            assert!(evaluation.objective.is_finite());
            assert!(evaluation.score.is_finite());
            assert_eq!(evaluation.branches.len(), DEIMOS.bodies.len() - 1);
            DEIMOS.refinement_bounds(guess, 0.01).unwrap();
        }
    }
}
