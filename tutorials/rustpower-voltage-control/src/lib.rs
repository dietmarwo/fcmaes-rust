//! Robust mixed-integer voltage-control design on RustPower's IEEE-39 network.
//!
//! The objective is intentionally evaluated entirely in Rust. Each candidate
//! is applied to a fresh network for six deterministic load, renewable, battery
//! and outage scenarios. RustPower's serial RSparse Newton-Raphson solver runs
//! inside each evaluation; fcmaes owns candidate-level parallelism.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use fcmaes_core::{
    Archive, Fitness, MapElitesParams, Mode, ModeParams, QdBatchFitness, Rng,
    map_elites_batch_with_progress, parallel_batch, pareto_indices,
};
use rustpower::io::pandapower::{Network, SGen, Shunt};
use rustpower::prelude::ecs::elements::BusID;
use rustpower::prelude::ecs::post_processing::{LineResultData, VBusResult};
use rustpower::prelude::ecs::powerflow::prelude::PowerFlowMat;
use rustpower::prelude::{PPNetwork, PostProcessing, PowerFlowResult, default_app};
use rustpower::testcases::case_ieee39::IEEE_39;

pub const DIMENSION: usize = 20;
pub const OBJECTIVES: usize = 4;
pub const CONSTRAINTS: usize = 3;
pub const VALUE_WIDTH: usize = OBJECTIVES + CONSTRAINTS;
pub const QD_DESCRIPTOR_LOWER: [f64; 2] = [0.0, 0.0];
pub const QD_DESCRIPTOR_UPPER: [f64; 2] = [300.0, 300.0];

pub const VARIABLE_NAMES: [&str; DIMENSION] = [
    "generator_vm_1_pu",
    "generator_vm_2_pu",
    "generator_vm_3_pu",
    "generator_vm_4_pu",
    "generator_vm_5_pu",
    "generator_vm_6_pu",
    "generator_vm_7_pu",
    "generator_vm_8_pu",
    "generator_vm_9_pu",
    "tap_offset_1",
    "tap_offset_2",
    "tap_offset_3",
    "capacitor_bus_index_1",
    "capacitor_steps_1",
    "capacitor_bus_index_2",
    "capacitor_steps_2",
    "battery_bus_index",
    "battery_capacity_mw",
    "renewable_curtailment_1",
    "renewable_curtailment_2",
];

const GENERATOR_COUNT: usize = 9;
const TAP_TRANSFORMERS: [usize; 3] = [3, 4, 7];
const CONTROL_BUSES: [i64; 6] = [3, 7, 14, 19, 24, 27];
const RENEWABLE_BUSES: [i64; 2] = [3, 24];
const RENEWABLE_CAPACITY_MW: [f64; 2] = [300.0, 220.0];
const CAPACITOR_STEP_MVAR: f64 = 25.0;
/// The embedded conversion's raw current limits are lower than its own
/// reference loading. Treat 1.6 times those values as the planning rating,
/// then apply each scenario's normal/emergency multiplier.
const CASE_RATING_CALIBRATION: f64 = 1.60;

pub const LOWER_BOUNDS: [f64; DIMENSION] = [
    0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, // generator V
    -4.0, -4.0, -4.0, // transformer tap offsets
    0.0, 0.0, 0.0, 0.0, // capacitor location/steps, twice
    0.0, 0.0, // battery location/capacity
    0.0, 0.0, // renewable curtailment
];

pub const UPPER_BOUNDS: [f64; DIMENSION] = [
    1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, // generator V
    4.0, 4.0, 4.0, // transformer tap offsets
    5.0, 6.0, 5.0, 6.0, // capacitor location/steps, twice
    5.0, 300.0, // battery location/capacity
    0.50, 0.50, // renewable curtailment
];

pub const INTEGER_MASK: [bool; DIMENSION] = [
    false, false, false, false, false, false, false, false, false, // generator V
    true, true, true, // transformer taps
    true, true, true, true, // capacitor location/steps
    true, false, // battery location/capacity
    false, false, // renewable curtailment
];

/// A reasonable in-bounds reference, not an optimized solution.
pub const BASELINE_DESIGN: [f64; DIMENSION] = [
    1.0499, 0.9841, 0.9972, 1.0123, 1.0494, 1.0600, 1.0275, 1.0265, 1.0300, 0.0, 0.0, 0.0, 0.0,
    0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0,
];

/// Feasible MODE representative from the documented seed-42 run. The
/// feasible volume is too small for an unseeded MAP-Elites archive, so QD
/// evaluates and counts this point as its disclosed warm start.
pub const QD_SEED_DESIGN: [f64; DIMENSION] = [
    1.017061883580,
    1.009570994316,
    0.989964604406,
    1.015448682041,
    1.011000465018,
    0.993151149326,
    1.000223266469,
    0.976329882384,
    0.998677540454,
    0.0,
    1.0,
    -2.0,
    1.0,
    1.0,
    1.0,
    6.0,
    2.0,
    270.830943232590,
    0.433666553196,
    0.466846708170,
];

#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    pub name: &'static str,
    pub load_scale: f64,
    pub renewable_scale: f64,
    /// Positive means discharge/generation; negative means charging/load.
    pub battery_dispatch: f64,
    pub line_outage: Option<usize>,
    /// Emergency line-loading multiplier. Normal operation uses 1.0.
    pub line_limit_multiplier: f64,
    pub weight: f64,
}

pub const SCENARIOS: [Scenario; 6] = [
    Scenario {
        name: "base",
        load_scale: 1.00,
        renewable_scale: 0.65,
        battery_dispatch: 0.00,
        line_outage: None,
        line_limit_multiplier: 1.00,
        weight: 1.0,
    },
    Scenario {
        name: "evening_peak",
        load_scale: 1.12,
        renewable_scale: 0.25,
        battery_dispatch: 1.00,
        line_outage: None,
        line_limit_multiplier: 1.00,
        weight: 1.0,
    },
    Scenario {
        name: "high_renewable_low_load",
        load_scale: 0.90,
        renewable_scale: 0.90,
        battery_dispatch: -1.00,
        line_outage: None,
        line_limit_multiplier: 1.00,
        weight: 1.0,
    },
    Scenario {
        name: "hot_peak",
        load_scale: 1.20,
        renewable_scale: 0.45,
        battery_dispatch: 0.80,
        line_outage: None,
        line_limit_multiplier: 1.00,
        weight: 1.0,
    },
    Scenario {
        name: "n_minus_1_line_23",
        load_scale: 1.08,
        renewable_scale: 0.40,
        battery_dispatch: 0.80,
        line_outage: Some(22),
        line_limit_multiplier: 1.20,
        weight: 1.0,
    },
    Scenario {
        name: "renewable_trip",
        load_scale: 1.05,
        renewable_scale: 0.00,
        battery_dispatch: 0.80,
        line_outage: None,
        line_limit_multiplier: 1.00,
        weight: 1.0,
    },
];

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedDesign {
    pub generator_vm_pu: [f64; GENERATOR_COUNT],
    pub tap_offsets: [i32; 3],
    pub capacitor_bus_indices: [usize; 2],
    pub capacitor_steps: [i32; 2],
    pub battery_bus_index: usize,
    pub battery_capacity_mw: f64,
    pub curtailment: [f64; 2],
}

impl DecodedDesign {
    pub fn from_slice(x: &[f64]) -> Result<Self, &'static str> {
        if x.len() != DIMENSION || x.iter().any(|value| !value.is_finite()) {
            return Err("design must contain 20 finite values");
        }
        let mut clamped = [0.0; DIMENSION];
        for i in 0..DIMENSION {
            clamped[i] = x[i].clamp(LOWER_BOUNDS[i], UPPER_BOUNDS[i]);
        }
        let mut generator_vm_pu = [0.0; GENERATOR_COUNT];
        generator_vm_pu.copy_from_slice(&clamped[..GENERATOR_COUNT]);
        Ok(Self {
            generator_vm_pu,
            tap_offsets: [
                rounded_i32(clamped[9], -4, 4),
                rounded_i32(clamped[10], -4, 4),
                rounded_i32(clamped[11], -4, 4),
            ],
            capacitor_bus_indices: [
                rounded_usize(clamped[12], CONTROL_BUSES.len() - 1),
                rounded_usize(clamped[14], CONTROL_BUSES.len() - 1),
            ],
            capacitor_steps: [
                rounded_i32(clamped[13], 0, 6),
                rounded_i32(clamped[15], 0, 6),
            ],
            battery_bus_index: rounded_usize(clamped[16], CONTROL_BUSES.len() - 1),
            battery_capacity_mw: clamped[17],
            curtailment: [clamped[18], clamped[19]],
        })
    }

    pub fn capacitor_buses(&self) -> [i64; 2] {
        [
            CONTROL_BUSES[self.capacitor_bus_indices[0]],
            CONTROL_BUSES[self.capacitor_bus_indices[1]],
        ]
    }

    pub fn battery_bus(&self) -> i64 {
        CONTROL_BUSES[self.battery_bus_index]
    }

    pub fn total_capacitor_mvar(&self) -> f64 {
        self.capacitor_steps
            .iter()
            .map(|&steps| f64::from(steps) * CAPACITOR_STEP_MVAR)
            .sum()
    }

    pub fn as_vector(&self) -> Vec<f64> {
        let mut x = Vec::with_capacity(DIMENSION);
        x.extend(self.generator_vm_pu);
        x.extend(self.tap_offsets.map(f64::from));
        x.push(self.capacitor_bus_indices[0] as f64);
        x.push(self.capacitor_steps[0] as f64);
        x.push(self.capacitor_bus_indices[1] as f64);
        x.push(self.capacitor_steps[1] as f64);
        x.push(self.battery_bus_index as f64);
        x.push(self.battery_capacity_mw);
        x.extend(self.curtailment);
        x
    }
}

fn rounded_i32(value: f64, lower: i32, upper: i32) -> i32 {
    (value.round() as i32).clamp(lower, upper)
}

fn rounded_usize(value: f64, upper: usize) -> usize {
    (value.round() as isize).clamp(0, upper as isize) as usize
}

#[derive(Clone, Debug)]
pub struct ScenarioResult {
    pub name: &'static str,
    pub converged: bool,
    pub iterations: usize,
    pub line_loss_mw: f64,
    pub rms_voltage_deviation_pu: f64,
    pub minimum_voltage_pu: f64,
    pub maximum_voltage_pu: f64,
    pub maximum_line_loading_percent: f64,
    pub voltage_constraint_pu: f64,
    pub thermal_constraint: f64,
    pub security_index: f64,
    pub mismatch_pu: f64,
}

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub design: DecodedDesign,
    /// `[mean line loss MW, RMS voltage deviation mpu, lifecycle cost M$,
    /// worst security/violation index]`.
    pub objectives: [f64; OBJECTIVES],
    /// `[voltage violation pu, thermal overload fraction, PF failure severity]`.
    /// Feasible values are all `<= 0`.
    pub constraints: [f64; CONSTRAINTS],
    pub scenarios: Vec<ScenarioResult>,
}

impl Evaluation {
    pub fn values(&self) -> Vec<f64> {
        self.objectives
            .iter()
            .chain(&self.constraints)
            .copied()
            .collect()
    }

    pub fn is_feasible(&self) -> bool {
        self.objectives.iter().all(|value| value.is_finite())
            && self.constraints.iter().all(|&value| value <= 0.0)
    }

    /// Higher is better; only used to select and compare representatives.
    pub fn quality(&self) -> f64 {
        let [loss, voltage_mpu, cost, security] = self.objectives;
        let violation = self
            .constraints
            .iter()
            .map(|&value| value.max(0.0))
            .sum::<f64>();
        1.0 / ((1.0 + loss.max(0.0) / 60.0)
            * (1.0 + voltage_mpu.max(0.0) / 30.0)
            * (1.0 + cost.max(0.0) / 50.0)
            * (1.0 + security.max(0.0))
            * (1.0 + 20.0 * violation))
    }
}

#[derive(Clone)]
pub struct VoltageControlModel {
    base: Network,
}

impl VoltageControlModel {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let base: Network = serde_json::from_str(IEEE_39)?;
        validate_network(&base)?;
        Ok(Self { base })
    }

    pub fn evaluate(&self, x: &[f64]) -> Evaluation {
        let design = DecodedDesign::from_slice(x)
            .unwrap_or_else(|_| DecodedDesign::from_slice(&BASELINE_DESIGN).unwrap());
        let mut scenarios = Vec::with_capacity(SCENARIOS.len());
        for scenario in SCENARIOS {
            scenarios.push(self.evaluate_scenario(&design, scenario));
        }

        let weight_sum = SCENARIOS
            .iter()
            .map(|scenario| scenario.weight)
            .sum::<f64>();
        let mean_loss_mw = scenarios
            .iter()
            .zip(SCENARIOS)
            .map(|(result, scenario)| result.line_loss_mw * scenario.weight)
            .sum::<f64>()
            / weight_sum;
        let rms_voltage_mpu = 1_000.0
            * (scenarios
                .iter()
                .zip(SCENARIOS)
                .map(|(result, scenario)| result.rms_voltage_deviation_pu.powi(2) * scenario.weight)
                .sum::<f64>()
                / weight_sum)
                .sqrt();
        let lifecycle_cost_musd = lifecycle_cost(&design);
        let worst_security_index = scenarios
            .iter()
            .map(|result| result.security_index)
            .fold(0.0_f64, f64::max);
        let voltage_constraint = scenarios
            .iter()
            .map(|result| result.voltage_constraint_pu)
            .fold(f64::NEG_INFINITY, f64::max);
        let thermal_constraint = scenarios
            .iter()
            .map(|result| result.thermal_constraint)
            .fold(f64::NEG_INFINITY, f64::max);
        let failures: Vec<_> = scenarios
            .iter()
            .filter(|result| !result.converged)
            .collect();
        let failure_constraint = if failures.is_empty() {
            -1.0
        } else {
            failures.len() as f64 / scenarios.len() as f64
                + failures
                    .iter()
                    .map(|result| result.mismatch_pu.min(1.0))
                    .sum::<f64>()
                    / failures.len() as f64
        };

        Evaluation {
            design,
            objectives: [
                mean_loss_mw,
                rms_voltage_mpu,
                lifecycle_cost_musd,
                worst_security_index,
            ],
            constraints: [voltage_constraint, thermal_constraint, failure_constraint],
            scenarios,
        }
    }

    fn evaluate_scenario(&self, design: &DecodedDesign, scenario: Scenario) -> ScenarioResult {
        let mut network = self.base.clone();
        apply_design(&mut network, design, scenario);

        let mut grid = default_app();
        grid.world_mut().insert_resource(PPNetwork(network));
        grid.update();

        let result = grid
            .world()
            .get_resource::<PowerFlowResult>()
            .expect("RustPower did not produce PowerFlowResult")
            .clone();
        let mismatch_pu = {
            let mat = grid
                .world()
                .get_resource::<PowerFlowMat>()
                .expect("RustPower did not produce PowerFlowMat");
            let calculated = result
                .v
                .component_mul(&(&mat.y_bus * &result.v).conjugate());
            let mismatch = calculated - &mat.s_bus;
            let pq = mismatch
                .iter()
                .take(mat.npq)
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max);
            let pv_active = mismatch
                .iter()
                .skip(mat.npq)
                .take(mat.npv)
                .map(|value| value.re.abs())
                .fold(0.0_f64, f64::max);
            pq.max(pv_active)
        };

        grid.post_process();
        let world = grid.world_mut();
        let mut voltages = vec![f64::NAN; self.base.bus.len()];
        let mut bus_query = world.query::<(&BusID, &VBusResult)>();
        for (bus, voltage) in bus_query.iter(world) {
            if bus.0 >= 0 && (bus.0 as usize) < voltages.len() {
                voltages[bus.0 as usize] = voltage.0.norm();
            }
        }
        let finite_voltages: Vec<_> = voltages
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        let minimum_voltage_pu = finite_voltages
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let maximum_voltage_pu = finite_voltages
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let rms_voltage_deviation_pu = if finite_voltages.is_empty() {
            1.0
        } else {
            (finite_voltages
                .iter()
                .map(|voltage| (voltage - 1.0).powi(2))
                .sum::<f64>()
                / finite_voltages.len() as f64)
                .sqrt()
        };

        let mut voltage_constraint_pu = -f64::INFINITY;
        let mut voltage_utilization = 0.0_f64;
        for (bus, &voltage) in self.base.bus.iter().zip(&voltages) {
            if !voltage.is_finite() {
                voltage_constraint_pu = voltage_constraint_pu.max(1.0);
                voltage_utilization = voltage_utilization.max(2.0);
                continue;
            }
            let lower = bus.min_vm_pu.unwrap_or(0.94);
            let upper = bus.max_vm_pu.unwrap_or(1.06);
            voltage_constraint_pu =
                voltage_constraint_pu.max((lower - voltage).max(voltage - upper));
            let utilization = if voltage >= 1.0 {
                (voltage - 1.0) / (upper - 1.0).max(1.0e-6)
            } else {
                (1.0 - voltage) / (1.0 - lower).max(1.0e-6)
            };
            voltage_utilization = voltage_utilization.max(utilization);
        }

        let mut line_loss_mw = 0.0;
        let mut maximum_line_loading_percent = 0.0_f64;
        let mut line_query = world.query::<&LineResultData>();
        for line in line_query.iter(world) {
            if line.pl_mw.is_finite() {
                line_loss_mw += line.pl_mw.max(0.0);
            }
            if line.loading_percent.is_finite() {
                maximum_line_loading_percent =
                    maximum_line_loading_percent.max(line.loading_percent);
            }
        }
        let thermal_utilization = maximum_line_loading_percent
            / (100.0 * CASE_RATING_CALIBRATION * scenario.line_limit_multiplier);
        let thermal_constraint = thermal_utilization - 1.0;
        let failure_stress = if result.converged {
            0.0
        } else {
            1.0 + mismatch_pu.min(10.0)
        };

        ScenarioResult {
            name: scenario.name,
            converged: result.converged,
            iterations: result.iterations,
            line_loss_mw,
            rms_voltage_deviation_pu,
            minimum_voltage_pu,
            maximum_voltage_pu,
            maximum_line_loading_percent,
            voltage_constraint_pu,
            thermal_constraint,
            security_index: voltage_utilization
                .max(thermal_utilization)
                .max(failure_stress),
            mismatch_pu,
        }
    }
}

fn validate_network(network: &Network) -> Result<(), Box<dyn Error>> {
    if network.bus.len() != 39
        || network.r#gen.as_ref().map(Vec::len) != Some(GENERATOR_COUNT)
        || network
            .trafo
            .as_ref()
            .is_none_or(|values| values.len() <= TAP_TRANSFORMERS[2])
        || network
            .line
            .as_ref()
            .is_none_or(|values| values.len() <= 22)
    {
        return Err("embedded IEEE-39 network does not match the expected RustPower case".into());
    }
    Ok(())
}

fn apply_design(network: &mut Network, design: &DecodedDesign, scenario: Scenario) {
    for (generator, &vm_pu) in network
        .r#gen
        .as_mut()
        .expect("validated generators")
        .iter_mut()
        .zip(&design.generator_vm_pu)
    {
        generator.vm_pu = vm_pu;
    }
    for (&transformer_index, &offset) in TAP_TRANSFORMERS.iter().zip(&design.tap_offsets) {
        let transformer =
            &mut network.trafo.as_mut().expect("validated transformers")[transformer_index];
        let reference = transformer
            .tap_pos
            .unwrap_or(transformer.tap_neutral.unwrap_or(0.0));
        transformer.tap_pos = Some(reference + f64::from(offset));
    }
    for load in network.load.as_mut().expect("IEEE-39 loads") {
        load.p_mw *= scenario.load_scale;
        load.q_mvar *= scenario.load_scale;
    }
    if let Some(line_index) = scenario.line_outage {
        network.line.as_mut().expect("validated lines")[line_index].in_service = false;
    }

    let mut shunts = network.shunt.take().unwrap_or_default();
    for bank in 0..2 {
        let steps = design.capacitor_steps[bank];
        if steps == 0 {
            continue;
        }
        let bus = design.capacitor_buses()[bank];
        shunts.push(Shunt {
            bus,
            // pandapower/RustPower shunts use positive Q for inductive
            // consumption, hence a capacitor is negative.
            q_mvar: -CAPACITOR_STEP_MVAR,
            p_mw: 0.0,
            vn_kv: network.bus[bus as usize].vn_kv,
            step: steps,
            max_step: 6,
            in_service: true,
            name: Some(format!("optimized_capacitor_{}", bank + 1)),
        });
    }
    network.shunt = Some(shunts);

    let mut static_generators = network.sgen.take().unwrap_or_default();
    for renewable in 0..2 {
        static_generators.push(SGen {
            name: Some(format!("renewable_{}", renewable + 1)),
            bus: RENEWABLE_BUSES[renewable],
            p_mw: RENEWABLE_CAPACITY_MW[renewable]
                * scenario.renewable_scale
                * (1.0 - design.curtailment[renewable]),
            q_mvar: 0.0,
            sn_mva: Some(RENEWABLE_CAPACITY_MW[renewable]),
            scaling: 1.0,
            in_service: true,
            type_: Some("renewable".to_string()),
            current_source: false,
            controllable: Some(false),
        });
    }
    if design.battery_capacity_mw > 0.0 {
        static_generators.push(SGen {
            name: Some("optimized_battery".to_string()),
            bus: design.battery_bus(),
            p_mw: design.battery_capacity_mw * scenario.battery_dispatch,
            q_mvar: 0.0,
            sn_mva: Some(design.battery_capacity_mw),
            scaling: 1.0,
            in_service: true,
            type_: Some("battery".to_string()),
            current_source: false,
            controllable: Some(false),
        });
    }
    network.sgen = Some(static_generators);
}

fn lifecycle_cost(design: &DecodedDesign) -> f64 {
    let installed_capacitor_mvar = design
        .capacitor_steps
        .iter()
        .map(|&steps| f64::from(steps) * CAPACITOR_STEP_MVAR)
        .sum::<f64>();
    let capacitor_fixed = design
        .capacitor_steps
        .iter()
        .filter(|&&steps| steps > 0)
        .count() as f64
        * 0.25;
    let capacitor_cost = 0.018 * installed_capacitor_mvar + capacitor_fixed;
    let battery_cost = if design.battery_capacity_mw > 0.5 {
        2.0 + 0.22 * design.battery_capacity_mw
    } else {
        0.0
    };
    let tap_control_cost = design
        .tap_offsets
        .iter()
        .map(|offset| f64::from(offset.abs()) * 0.08)
        .sum::<f64>();
    let mean_renewable_scale = SCENARIOS
        .iter()
        .map(|scenario| scenario.renewable_scale)
        .sum::<f64>()
        / SCENARIOS.len() as f64;
    let curtailed_mwh = RENEWABLE_CAPACITY_MW
        .iter()
        .zip(design.curtailment)
        .map(|(&capacity, curtailment)| capacity * mean_renewable_scale * curtailment * 8_760.0)
        .sum::<f64>();
    let curtailment_cost = curtailed_mwh * 45.0 / 1_000_000.0;
    capacitor_cost + battery_cost + tap_control_cost + curtailment_cost
}

#[derive(Clone, Debug)]
pub struct Progress {
    pub generation: usize,
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub feasible_population: usize,
    pub pareto_population: usize,
    pub best_quality: f64,
}

#[derive(Clone, Debug)]
pub struct ParetoPoint {
    pub x: Vec<f64>,
    pub evaluation: Evaluation,
}

#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub evaluations: usize,
    pub generations: usize,
    pub elapsed: Duration,
    pub pareto: Vec<ParetoPoint>,
    pub representative: ParetoPoint,
    pub progress: Vec<Progress>,
}

#[derive(Clone, Debug)]
pub struct QdOptions {
    pub evaluations: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    pub workers: i32,
    pub seed: u64,
}

impl Default for QdOptions {
    fn default() -> Self {
        Self {
            evaluations: 8_192,
            capacity: 400,
            chunk_size: 128,
            workers: 0,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QdPoint {
    pub niche_id: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub x: Vec<f64>,
    pub evaluation: Evaluation,
    pub quality: f64,
    pub descriptors: [f64; 2],
    pub visit_count: u64,
}

#[derive(Clone, Debug)]
pub struct QdProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    pub coverage: f64,
    pub qd_score: f64,
    pub best_quality: f64,
    pub invalid_fraction: f64,
}

#[derive(Clone, Debug)]
pub struct QdOutcome {
    pub elites: Vec<QdPoint>,
    pub representative: QdPoint,
    pub evaluations: usize,
    pub occupied: usize,
    pub capacity: usize,
    pub qd_score: f64,
    pub invalid_evaluations: usize,
    pub clipped_descriptors: usize,
    pub elapsed: Duration,
    pub progress: Vec<QdProgress>,
}

pub fn optimize_mode(
    model: &VoltageControlModel,
    evaluations: usize,
    popsize: usize,
    workers: i32,
    seed: u64,
) -> Result<OptimizationResult, Box<dyn Error>> {
    if popsize < 4 {
        return Err("MODE population size must be at least four".into());
    }
    if evaluations == 0 {
        return Err("MODE evaluation budget must be positive".into());
    }
    if workers < 0 {
        return Err("worker count must be non-negative".into());
    }
    let fitness = Fitness::bounded(DIMENSION, VALUE_WIDTH, &LOWER_BOUNDS, &UPPER_BOUNDS);
    let params = ModeParams {
        popsize: popsize as i32,
        seed,
        ..Default::default()
    };
    let mut mode = Mode::try_new(
        fitness,
        OBJECTIVES,
        CONSTRAINTS,
        Some(INTEGER_MASK.to_vec()),
        &params,
    )?;
    let generations = evaluations.div_ceil(popsize);
    let started = Instant::now();
    let mut evaluated = 0;
    let mut progress = Vec::with_capacity(generations);
    for generation in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, workers, |x| model.evaluate(x).values());
        evaluated += ys.len();
        mode.tell(&ys);
        if generation == 0 || (generation + 1) % 10 == 0 || generation + 1 == generations {
            let result = mode.result();
            let (feasible, pareto, quality) = population_summary(model, &result.x, &result.y)?;
            progress.push(Progress {
                generation: generation + 1,
                evaluations: evaluated,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                feasible_population: feasible,
                pareto_population: pareto,
                best_quality: quality,
            });
        }
    }
    let elapsed = started.elapsed();
    let result = mode.result();
    let feasible_indices: Vec<_> = result
        .y
        .iter()
        .enumerate()
        .filter(|(_, values)| values[OBJECTIVES..].iter().all(|&value| value <= 0.0))
        .map(|(index, _)| index)
        .collect();
    if feasible_indices.is_empty() {
        let least_violation = result
            .y
            .iter()
            .map(|values| {
                values[OBJECTIVES..]
                    .iter()
                    .copied()
                    .map(|value| value.max(0.0))
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        return Err(format!(
            "MODE found no feasible design; least summed violation={least_violation:.6}"
        )
        .into());
    }
    let feasible_objectives: Vec<Vec<f64>> = feasible_indices
        .iter()
        .map(|&index| result.y[index][..OBJECTIVES].to_vec())
        .collect();
    let front = pareto_indices(&feasible_objectives, OBJECTIVES)?;
    let mut pareto: Vec<_> = front
        .iter()
        .map(|&front_index| {
            let population_index = feasible_indices[front_index];
            let evaluation = model.evaluate(&result.x[population_index]);
            // Store the evaluated mixed-integer controls, not MODE's
            // continuous genotype before categorical/integer decoding.
            let x = evaluation.design.as_vector();
            ParetoPoint { evaluation, x }
        })
        .collect();
    pareto.sort_by(|left, right| {
        right
            .evaluation
            .quality()
            .total_cmp(&left.evaluation.quality())
    });
    let representative = pareto
        .first()
        .cloned()
        .ok_or("MODE returned an empty Pareto front")?;
    Ok(OptimizationResult {
        evaluations: evaluated,
        generations,
        elapsed,
        pareto,
        representative,
        progress,
    })
}

/// Return the minimized archive quality and two continuous physical
/// descriptors. Categorical installation buses are deliberately not treated
/// as Euclidean descriptors; they are retained as archive metadata.
pub fn qd_objective(x: &[f64], model: &VoltageControlModel) -> (f64, [f64; 2]) {
    let evaluation = model.evaluate(x);
    if !evaluation.is_feasible() {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    }
    let descriptors = [
        evaluation.design.battery_capacity_mw,
        evaluation.design.total_capacitor_mvar(),
    ];
    let quality = 1.0 / evaluation.quality();
    if !quality.is_finite() || descriptors.iter().any(|value| !value.is_finite()) {
        (f64::INFINITY, [f64::INFINITY; 2])
    } else {
        (quality, descriptors)
    }
}

fn normalize_qd(x: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(LOWER_BOUNDS.iter().zip(UPPER_BOUNDS))
        .map(|(&value, (&lower, upper))| (value - lower) / (upper - lower))
        .collect()
}

fn denormalize_qd(x: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(LOWER_BOUNDS.iter().zip(UPPER_BOUNDS))
        .map(|(&value, (&lower, upper))| lower + value * (upper - lower))
        .collect()
}

struct VoltageControlQdBatch<'a> {
    model: &'a VoltageControlModel,
    workers: i32,
    evaluations: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    clipped: Arc<AtomicUsize>,
}

impl QdBatchFitness for VoltageControlQdBatch<'_> {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let evaluated = parallel_batch(xs, self.workers, |x| {
            qd_objective(&denormalize_qd(x), self.model)
        });
        self.evaluations
            .fetch_add(evaluated.len(), Ordering::Relaxed);
        let mut output = Vec::with_capacity(evaluated.len());
        for (quality, descriptors) in evaluated {
            if !quality.is_finite() || descriptors.iter().any(|value| !value.is_finite()) {
                self.invalid.fetch_add(1, Ordering::Relaxed);
            } else if descriptors
                .iter()
                .zip(QD_DESCRIPTOR_LOWER.iter().zip(QD_DESCRIPTOR_UPPER))
                .any(|(&value, (&lower, upper))| value < lower || value > upper)
            {
                self.clipped.fetch_add(1, Ordering::Relaxed);
            }
            output.push((quality, descriptors.to_vec()));
        }
        output
    }
}

pub fn optimize_qd(
    model: &VoltageControlModel,
    options: &QdOptions,
) -> Result<QdOutcome, Box<dyn Error>> {
    if options.evaluations == 0 {
        return Err("QD evaluation budget must be positive".into());
    }
    if options.workers < 0 {
        return Err("worker count must be non-negative".into());
    }
    if options.chunk_size < 2 || !options.chunk_size.is_multiple_of(2) {
        return Err("QD chunk size must be an even number of at least two".into());
    }
    let side = (options.capacity as f64).sqrt() as usize;
    if side < 2 || side * side != options.capacity {
        return Err("QD capacity must be a perfect square of at least four".into());
    }
    let generations = options
        .evaluations
        .saturating_sub(1)
        .div_ceil(options.chunk_size);
    let actual_evaluations = generations * options.chunk_size + 1;
    let mut rng = Rng::new(options.seed);
    let mut archive = Archive::try_new(
        DIMENSION,
        &QD_DESCRIPTOR_LOWER,
        &QD_DESCRIPTOR_UPPER,
        options.capacity,
        0,
        &mut rng,
    )?;
    let normalized_lower = [0.0; DIMENSION];
    let normalized_upper = [1.0; DIMENSION];
    archive.seed_uniform(&normalized_lower, &normalized_upper, &mut rng);
    let (seed_quality, seed_descriptors) = qd_objective(&QD_SEED_DESIGN, model);
    if !seed_quality.is_finite() {
        return Err("the disclosed QD warm-start design is no longer feasible".into());
    }
    let seed_x = normalize_qd(&QD_SEED_DESIGN);
    let seed_niche = archive.index_of_niche(&seed_descriptors);
    archive.set(seed_niche, seed_quality, &seed_descriptors, &seed_x);
    let evaluations = Arc::new(AtomicUsize::new(1));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clipped = Arc::new(AtomicUsize::new(0));
    let mut batch = VoltageControlQdBatch {
        model,
        workers: options.workers,
        evaluations: Arc::clone(&evaluations),
        invalid: Arc::clone(&invalid),
        clipped: Arc::clone(&clipped),
    };
    let parameters = MapElitesParams {
        generations,
        chunk_size: options.chunk_size,
        use_sbx: true,
        ..Default::default()
    };
    let started = Instant::now();
    let mut progress = vec![QdProgress {
        evaluations: 1,
        elapsed_seconds: 0.0,
        coverage: archive.occupied() as f64 / archive.capacity() as f64,
        qd_score: archive.qd_score(),
        best_quality: archive.best_y(),
        invalid_fraction: 0.0,
    }];
    map_elites_batch_with_progress(
        &mut archive,
        &mut batch,
        &normalized_lower,
        &normalized_upper,
        &parameters,
        &mut rng,
        &mut |_, archive| {
            let count = evaluations.load(Ordering::Relaxed);
            progress.push(QdProgress {
                evaluations: count,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                coverage: archive.occupied() as f64 / archive.capacity() as f64,
                qd_score: archive.qd_score(),
                best_quality: archive.best_y(),
                invalid_fraction: invalid.load(Ordering::Relaxed) as f64 / count.max(1) as f64,
            });
        },
    )?;
    debug_assert_eq!(evaluations.load(Ordering::Relaxed), actual_evaluations);

    let candidates = (0..archive.capacity())
        .filter(|&index| archive.ys()[index].is_finite())
        .map(|index| (index, denormalize_qd(&archive.xs()[index])))
        .collect::<Vec<_>>();
    let reevaluated = parallel_batch(
        &candidates
            .iter()
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>(),
        options.workers,
        |x| model.evaluate(x),
    );
    let mut elites = Vec::with_capacity(candidates.len());
    for ((niche_id, _), evaluation) in candidates.into_iter().zip(reevaluated) {
        if !evaluation.is_feasible() {
            return Err("a retained MAP-Elites design failed deterministic re-evaluation".into());
        }
        elites.push(QdPoint {
            niche_id,
            grid_x: niche_id % side,
            grid_y: niche_id / side,
            quality: archive.ys()[niche_id],
            descriptors: [
                archive.descriptors()[niche_id][0],
                archive.descriptors()[niche_id][1],
            ],
            visit_count: archive.counts()[niche_id],
            x: evaluation.design.as_vector(),
            evaluation,
        });
    }
    elites.sort_by(|left, right| left.quality.total_cmp(&right.quality));
    let representative = elites
        .first()
        .cloned()
        .ok_or("MAP-Elites found no feasible voltage-control design")?;
    Ok(QdOutcome {
        representative,
        evaluations: actual_evaluations,
        occupied: archive.occupied(),
        capacity: archive.capacity(),
        qd_score: archive.qd_score(),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clipped_descriptors: clipped.load(Ordering::Relaxed),
        elapsed: started.elapsed(),
        progress,
        elites,
    })
}

fn population_summary(
    model: &VoltageControlModel,
    x: &[Vec<f64>],
    y: &[Vec<f64>],
) -> Result<(usize, usize, f64), Box<dyn Error>> {
    let feasible_indices: Vec<_> = y
        .iter()
        .enumerate()
        .filter(|(_, values)| values[OBJECTIVES..].iter().all(|&value| value <= 0.0))
        .map(|(index, _)| index)
        .collect();
    if feasible_indices.is_empty() {
        return Ok((0, 0, 0.0));
    }
    let values: Vec<_> = feasible_indices
        .iter()
        .map(|&index| y[index][..OBJECTIVES].to_vec())
        .collect();
    let front = pareto_indices(&values, OBJECTIVES)?;
    let quality = front
        .iter()
        .map(|&index| model.evaluate(&x[feasible_indices[index]]).quality())
        .fold(0.0_f64, f64::max);
    Ok((feasible_indices.len(), front.len(), quality))
}

pub fn write_artifacts(
    directory: &Path,
    baseline: &Evaluation,
    result: &OptimizationResult,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    write_pareto_csv(directory, result)?;
    write_scenarios_csv(directory, baseline, &result.representative.evaluation)?;
    write_progress_csv(directory, result)?;
    write_report(directory, baseline, result)?;
    Ok(())
}

pub fn write_qd_artifacts(
    directory: &Path,
    baseline: &Evaluation,
    outcome: &QdOutcome,
    options: &QdOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    write_scenarios_csv(directory, baseline, &outcome.representative.evaluation)?;

    let mut archive = String::from(
        "niche_id,grid_x,grid_y,quality_train,descriptor_battery_capacity_mw_train,descriptor_capacitor_mvar_train,category_battery_bus,category_capacitor_bus_1,category_capacitor_bus_2,visit_count,objective_loss_mw,objective_voltage_deviation_mpu,objective_lifecycle_cost_musd,objective_security_index,constraint_voltage,constraint_thermal,constraint_power_flow",
    );
    for name in VARIABLE_NAMES {
        let _ = write!(archive, ",decision_{name}");
    }
    archive.push('\n');
    for point in &outcome.elites {
        let evaluation = &point.evaluation;
        let design = &evaluation.design;
        let capacitor_buses = design.capacitor_buses();
        let _ = write!(
            archive,
            "{},{},{},{:.12},{:.12},{:.12},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            point.niche_id,
            point.grid_x,
            point.grid_y,
            point.quality,
            point.descriptors[0],
            point.descriptors[1],
            design.battery_bus(),
            capacitor_buses[0],
            capacitor_buses[1],
            point.visit_count,
            evaluation.objectives[0],
            evaluation.objectives[1],
            evaluation.objectives[2],
            evaluation.objectives[3],
            evaluation.constraints[0],
            evaluation.constraints[1],
            evaluation.constraints[2],
        );
        for value in &point.x {
            let _ = write!(archive, ",{value:.12}");
        }
        archive.push('\n');
    }
    fs::write(directory.join("qd_archive.csv"), archive)?;

    let mut convergence = String::from(
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction\n",
    );
    for sample in &outcome.progress {
        let _ = writeln!(
            convergence,
            "{},{:.12},{:.12},{:.12},{:.12},{:.12}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.coverage,
            sample.qd_score,
            sample.best_quality,
            sample.invalid_fraction,
        );
    }
    fs::write(directory.join("convergence.csv"), convergence)?;

    let representative = &outcome.representative;
    let mut representatives = String::from(
        "label,niche_id,quality_train,descriptor_battery_capacity_mw_train,descriptor_capacitor_mvar_train,category_battery_bus,objective_loss_mw,objective_voltage_deviation_mpu,objective_lifecycle_cost_musd,objective_security_index\n",
    );
    let _ = writeln!(
        representatives,
        "best,{},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12},{:.12}",
        representative.niche_id,
        representative.quality,
        representative.descriptors[0],
        representative.descriptors[1],
        representative.evaluation.design.battery_bus(),
        representative.evaluation.objectives[0],
        representative.evaluation.objectives[1],
        representative.evaluation.objectives[2],
        representative.evaluation.objectives[3],
    );
    fs::write(directory.join("representatives.csv"), representatives)?;

    let side = (outcome.capacity as f64).sqrt() as usize;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "rustpower-voltage-control",
        "formulation": "qd",
        "command": command,
        "seed": options.seed,
        "workers": if options.workers == 0 {
            std::thread::available_parallelism().map(usize::from).unwrap_or(1)
        } else {
            options.workers as usize
        },
        "requested_evaluations": options.evaluations,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "simulation": {
            "network": "IEEE-39",
            "scenarios": SCENARIOS.iter().map(|scenario| scenario.name).collect::<Vec<_>>(),
            "solver": "RustPower RSparse Newton-Raphson"
        },
        "warm_start": {
            "source": "documented seed-42 MODE representative",
            "counted_evaluations": 1
        },
        "decision_normalization": "[0,1] per variable before MAP-Elites variation",
        "descriptors": [
            {
                "column": "descriptor_battery_capacity_mw",
                "label": "Installed battery capacity",
                "unit": "MW",
                "bounds": [QD_DESCRIPTOR_LOWER[0], QD_DESCRIPTOR_UPPER[0]]
            },
            {
                "column": "descriptor_capacitor_mvar",
                "label": "Installed capacitor support",
                "unit": "MVAr",
                "bounds": [QD_DESCRIPTOR_LOWER[1], QD_DESCRIPTOR_UPPER[1]]
            }
        ],
        "categories": [
            {
                "column": "category_battery_bus",
                "label": "Battery installation bus",
                "values": CONTROL_BUSES
            }
        ],
        "qd": {
            "capacity": outcome.capacity,
            "grid_shape": [side, side],
            "chunk_size": options.chunk_size,
            "quality_train_column": "quality_train",
            "quality_label": "Balanced feasible design cost (lower is better)",
            "occupied": outcome.occupied,
            "coverage": outcome.occupied as f64 / outcome.capacity as f64,
            "qd_score": outcome.qd_score,
            "best_quality": outcome.representative.quality,
            "invalid_evaluations": outcome.invalid_evaluations,
            "clipped_descriptors": outcome.clipped_descriptors
        },
        "convergence_metrics": [
            "coverage", "qd_score", "best_quality", "invalid_fraction"
        ],
        "artifacts": {
            "qd_archive": "qd_archive.csv",
            "convergence": "convergence.csv",
            "representatives": "representatives.csv",
            "scenarios": "scenarios.csv"
        }
    });
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

fn write_pareto_csv(directory: &Path, result: &OptimizationResult) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "point_id,feasible,selected,objective_loss_mw,objective_voltage_deviation_mpu,objective_lifecycle_cost_musd,objective_security_index,quality",
    );
    for name in VARIABLE_NAMES {
        let _ = write!(csv, ",decision_{name}");
    }
    csv.push('\n');
    for (rank, point) in result.pareto.iter().enumerate() {
        let values = point.evaluation.objectives;
        let _ = write!(
            csv,
            "{rank},1,{},{:.12},{:.12},{:.12},{:.12},{:.12}",
            u8::from(rank == 0),
            values[0],
            values[1],
            values[2],
            values[3],
            point.evaluation.quality(),
        );
        for value in &point.x {
            let _ = write!(csv, ",{value:.12}");
        }
        csv.push('\n');
    }
    fs::write(directory.join("pareto.csv"), csv)?;
    Ok(())
}

fn write_scenarios_csv(
    directory: &Path,
    baseline: &Evaluation,
    selected: &Evaluation,
) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "design,scenario,converged,iterations,loss_mw,voltage_rms_pu,min_vm_pu,max_vm_pu,max_line_loading_percent,voltage_constraint_pu,thermal_constraint,security_index,mismatch_pu\n",
    );
    for (name, evaluation) in [("baseline", baseline), ("selected", selected)] {
        for scenario in &evaluation.scenarios {
            csv.push_str(&format!(
                "{name},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}\n",
                scenario.name,
                scenario.converged,
                scenario.iterations,
                scenario.line_loss_mw,
                scenario.rms_voltage_deviation_pu,
                scenario.minimum_voltage_pu,
                scenario.maximum_voltage_pu,
                scenario.maximum_line_loading_percent,
                scenario.voltage_constraint_pu,
                scenario.thermal_constraint,
                scenario.security_index,
                scenario.mismatch_pu,
            ));
        }
    }
    fs::write(directory.join("scenarios.csv"), csv)?;
    Ok(())
}

fn write_progress_csv(directory: &Path, result: &OptimizationResult) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "evaluations,elapsed_seconds,best_quality,generation,feasible_population,pareto_population\n",
    );
    for sample in &result.progress {
        let _ = writeln!(
            csv,
            "{},{:.12},{:.12},{},{},{}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.best_quality,
            sample.generation,
            sample.feasible_population,
            sample.pareto_population,
        );
    }
    fs::write(directory.join("convergence.csv"), csv)?;
    Ok(())
}

fn write_report(
    directory: &Path,
    baseline: &Evaluation,
    result: &OptimizationResult,
) -> Result<(), Box<dyn Error>> {
    let selected = &result.representative.evaluation;
    let mut points = String::new();
    let min_loss = result
        .pareto
        .iter()
        .map(|point| point.evaluation.objectives[0])
        .fold(f64::INFINITY, f64::min);
    let max_loss = result
        .pareto
        .iter()
        .map(|point| point.evaluation.objectives[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_voltage = result
        .pareto
        .iter()
        .map(|point| point.evaluation.objectives[1])
        .fold(f64::INFINITY, f64::min);
    let max_voltage = result
        .pareto
        .iter()
        .map(|point| point.evaluation.objectives[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let max_cost = result
        .pareto
        .iter()
        .map(|point| point.evaluation.objectives[2])
        .fold(0.0_f64, f64::max)
        .max(1.0);
    for point in &result.pareto {
        let evaluation = &point.evaluation;
        let x = scale(evaluation.objectives[0], min_loss, max_loss, 55.0, 745.0);
        let y = scale(
            evaluation.objectives[1],
            min_voltage,
            max_voltage,
            445.0,
            35.0,
        );
        let hue = 125.0 * (1.0 - evaluation.objectives[2] / max_cost).clamp(0.0, 1.0);
        points.push_str(&format!(
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"5\" fill=\"hsl({hue:.1} 70% 45%)\"><title>loss {:.3} MW; voltage {:.3} mpu; cost {:.3} M$; security {:.3}</title></circle>",
            evaluation.objectives[0],
            evaluation.objectives[1],
            evaluation.objectives[2],
            evaluation.objectives[3],
        ));
    }
    let mut scenario_rows = String::new();
    for scenario in &selected.scenarios {
        scenario_rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.4}</td><td>{:.4}</td><td>{:.1}%</td><td>{:.3}</td></tr>",
            scenario.name,
            scenario.converged,
            scenario.line_loss_mw,
            scenario.minimum_voltage_pu,
            scenario.maximum_voltage_pu,
            scenario.maximum_line_loading_percent,
            scenario.security_index,
        ));
    }
    let design = &selected.design;
    let design_text = format!(
        "generator V: {:?}\ntap offsets: {:?}\ncapacitors: buses {:?}, steps {:?}\nbattery: bus {}, {:.3} MW\ncurtailment: {:?}",
        design.generator_vm_pu,
        design.tap_offsets,
        design.capacitor_buses(),
        design.capacitor_steps,
        design.battery_bus(),
        design.battery_capacity_mw,
        design.curtailment,
    );
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>RustPower voltage control</title><style>
body{{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#17202a}}
h1,h2{{color:#154360}} .cards{{display:flex;gap:1rem;flex-wrap:wrap}} .card{{padding:1rem;border:1px solid #ccd;border-radius:8px;min-width:270px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}
svg{{width:100%;height:auto;border:1px solid #ddd;background:#fafcff}}pre{{background:#f5f7f9;padding:1rem;overflow:auto}}
</style></head><body>
<h1>Robust IEEE-39 voltage-control design</h1>
<p>RustPower 0.5.0 RSparse power flow × six deterministic scenarios × fcmaes constrained MODE.</p>
<div class=\"cards\"><div class=\"card\"><h2>Reference</h2><p>Loss <b>{:.3} MW</b><br>Voltage RMS <b>{:.3} mpu</b><br>Cost <b>{:.3} M$</b><br>Security <b>{:.3}</b><br>Quality <b>{:.6}</b></p></div>
<div class=\"card\"><h2>Selected Pareto point</h2><p>Loss <b>{:.3} MW</b><br>Voltage RMS <b>{:.3} mpu</b><br>Cost <b>{:.3} M$</b><br>Security <b>{:.3}</b><br>Quality <b>{:.6}</b></p></div></div>
<h2>Pareto projection</h2><p>Line loss versus voltage deviation; greener points have lower lifecycle cost.</p>
<svg viewBox=\"0 0 780 480\" role=\"img\"><line x1=\"55\" y1=\"445\" x2=\"750\" y2=\"445\" stroke=\"#333\"/><line x1=\"55\" y1=\"445\" x2=\"55\" y2=\"30\" stroke=\"#333\"/>
<text x=\"330\" y=\"475\">mean line loss (MW)</text><text x=\"18\" y=\"300\" transform=\"rotate(-90 18 300)\">RMS voltage deviation (mpu)</text>{points}</svg>
<h2>Selected scenario audit</h2><table><thead><tr><th>Scenario</th><th>Converged</th><th>Loss MW</th><th>Min V</th><th>Max V</th><th>Max line</th><th>Security</th></tr></thead><tbody>{scenario_rows}</tbody></table>
<h2>Selected controls</h2><pre>{design_text}</pre>
<p>MODE evaluations: {}; Pareto points: {}; wall time: {:.3} s. This experimental model is not an operational power-system study.</p>
</body></html>",
        baseline.objectives[0],
        baseline.objectives[1],
        baseline.objectives[2],
        baseline.objectives[3],
        baseline.quality(),
        selected.objectives[0],
        selected.objectives[1],
        selected.objectives[2],
        selected.objectives[3],
        selected.quality(),
        result.evaluations,
        result.pareto.len(),
        result.elapsed.as_secs_f64(),
    );
    fs::write(directory.join("report.html"), html)?;
    Ok(())
}

fn scale(value: f64, min: f64, max: f64, out_min: f64, out_max: f64) -> f64 {
    if (max - min).abs() < 1.0e-12 {
        0.5 * (out_min + out_max)
    } else {
        out_min + (value - min) * (out_max - out_min) / (max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_network_and_baseline_evaluate() {
        let model = VoltageControlModel::new().unwrap();
        let evaluation = model.evaluate(&BASELINE_DESIGN);
        assert_eq!(evaluation.scenarios.len(), SCENARIOS.len());
        assert!(evaluation.objectives.iter().all(|value| value.is_finite()));
        assert!(evaluation.constraints.iter().all(|value| value.is_finite()));
        assert!(
            evaluation
                .scenarios
                .iter()
                .all(|scenario| scenario.iterations > 0)
        );
    }

    #[test]
    fn decoder_rounds_and_clamps_discrete_controls() {
        let mut x = BASELINE_DESIGN;
        x[9] = 2.6;
        x[12] = 99.0;
        x[13] = 2.4;
        x[16] = -5.0;
        x[17] = 999.0;
        let decoded = DecodedDesign::from_slice(&x).unwrap();
        assert_eq!(decoded.tap_offsets[0], 3);
        assert_eq!(decoded.capacitor_bus_indices[0], 5);
        assert_eq!(decoded.capacitor_steps[0], 2);
        assert_eq!(decoded.battery_bus_index, 0);
        assert_eq!(decoded.battery_capacity_mw, 300.0);
        assert_eq!(decoded.as_vector().len(), DIMENSION);
    }

    #[test]
    fn decoder_rejects_wrong_width_and_non_finite_values() {
        assert!(DecodedDesign::from_slice(&[]).is_err());
        let mut x = BASELINE_DESIGN;
        x[0] = f64::NAN;
        assert!(DecodedDesign::from_slice(&x).is_err());
    }

    #[test]
    fn active_assets_increase_cost() {
        let baseline = DecodedDesign::from_slice(&BASELINE_DESIGN).unwrap();
        let mut x = BASELINE_DESIGN;
        x[13] = 3.0;
        x[15] = 2.0;
        x[17] = 100.0;
        x[18] = 0.2;
        let assets = DecodedDesign::from_slice(&x).unwrap();
        assert!(lifecycle_cost(&assets) > lifecycle_cost(&baseline));
    }

    #[test]
    fn qd_uses_continuous_capacity_descriptors_and_strict_feasibility() {
        let model = VoltageControlModel::new().unwrap();
        let (quality, descriptors) = qd_objective(&QD_SEED_DESIGN, &model);
        assert!(quality.is_finite());
        assert!((descriptors[0] - QD_SEED_DESIGN[17]).abs() < 1.0e-12);
        assert_eq!(descriptors[1], 175.0);
        // Battery bus index 2 decodes to bus 14, but category identity is not
        // embedded in either continuous MAP-Elites axis.
        assert_eq!(model.evaluate(&QD_SEED_DESIGN).design.battery_bus(), 14);
        let normalized = normalize_qd(&QD_SEED_DESIGN);
        let restored = denormalize_qd(&normalized);
        assert!(
            restored
                .iter()
                .zip(QD_SEED_DESIGN)
                .all(|(&left, right)| (left - right).abs() < 1.0e-12)
        );

        let (quality, descriptors) = qd_objective(&BASELINE_DESIGN, &model);
        assert!(!quality.is_finite());
        assert!(descriptors.iter().all(|value| !value.is_finite()));
        assert!(
            optimize_qd(
                &model,
                &QdOptions {
                    capacity: 15,
                    ..Default::default()
                }
            )
            .is_err()
        );
        let outcome = optimize_qd(
            &model,
            &QdOptions {
                evaluations: 8,
                capacity: 4,
                chunk_size: 4,
                workers: 2,
                seed: 9,
            },
        )
        .unwrap();
        assert_eq!(outcome.evaluations, 9);
        assert!(!outcome.elites.is_empty());
    }

    #[test]
    fn tiny_mode_run_returns_a_consistent_front_or_clear_error() {
        let model = VoltageControlModel::new().unwrap();
        match optimize_mode(&model, 8, 4, 1, 7) {
            Ok(result) => {
                assert_eq!(result.evaluations, 8);
                assert!(!result.pareto.is_empty());
                assert!(result.representative.evaluation.is_feasible());
            }
            Err(error) => assert!(error.to_string().contains("no feasible design")),
        }
    }

    #[test]
    fn artifact_scale_handles_degenerate_range() {
        assert_eq!(scale(1.0, 1.0, 1.0, 10.0, 20.0), 15.0);
    }
}
