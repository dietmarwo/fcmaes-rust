//! Experimental room-ventilation CFD objective.
//!
//! The flow field uses a D2Q9 lattice-Boltzmann pressure/velocity solver and
//! the passive pollutant uses a D2Q5 advection-diffusion lattice. This compact
//! implementation keeps arbitrary rasterized baffles and variable wall vents
//! cheap enough for optimization. See the project README for its educational
//! scope, custom-backend rationale, and verification evidence.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const FLOW_DIRECTIONS: usize = 9;
const FLOW_EX: [isize; FLOW_DIRECTIONS] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const FLOW_EY: [isize; FLOW_DIRECTIONS] = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const FLOW_WEIGHTS: [f64; FLOW_DIRECTIONS] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];
const FLOW_OPPOSITE: [usize; FLOW_DIRECTIONS] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

const SCALAR_DIRECTIONS: usize = 5;
const SCALAR_EX: [isize; SCALAR_DIRECTIONS] = [0, 1, 0, -1, 0];
const SCALAR_EY: [isize; SCALAR_DIRECTIONS] = [0, 0, 1, 0, -1];
const SCALAR_WEIGHTS: [f64; SCALAR_DIRECTIONS] =
    [1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];
const SCALAR_OPPOSITE: [usize; SCALAR_DIRECTIONS] = [0, 3, 4, 1, 2];

/// Nine physical design variables.
pub const DIMENSION: usize = 9;

/// Pollutant releases used by the optimization objective, as fractions of room
/// width and height.
pub const TRAINING_SOURCES: [[f64; 2]; 3] = [[0.72, 0.30], [0.30, 0.25], [0.58, 0.55]];

/// Releases reserved for post-optimization robustness checks.
pub const VALIDATION_SOURCES: [[f64; 2]; 3] = [[0.22, 0.48], [0.48, 0.36], [0.78, 0.52]];

/// Bounds for `[inlet_y, inlet_width, outlet_y, outlet_width, velocity,
/// baffle_x, baffle_y, baffle_length, baffle_angle]`.
pub const LOWER_BOUNDS: [f64; DIMENSION] = [0.12, 0.12, 0.12, 0.12, 0.25, 0.20, 0.15, 0.15, -1.40];
pub const UPPER_BOUNDS: [f64; DIMENSION] = [0.88, 0.45, 0.88, 0.45, 1.50, 0.80, 0.85, 0.65, 1.40];

/// Decoded room design. Positions and vent widths are fractions of room
/// height/width as documented in the README; velocity is in m/s and angle in
/// radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Design {
    pub inlet_y: f64,
    pub inlet_width: f64,
    pub outlet_y: f64,
    pub outlet_width: f64,
    pub inlet_velocity: f64,
    pub baffle_x: f64,
    pub baffle_y: f64,
    pub baffle_length: f64,
    pub baffle_angle: f64,
}

impl Design {
    pub fn decode(x: &[f64]) -> Option<Self> {
        if x.len() != DIMENSION || x.iter().any(|value| !value.is_finite()) {
            return None;
        }
        Some(Self {
            inlet_y: x[0],
            inlet_width: x[1],
            outlet_y: x[2],
            outlet_width: x[3],
            inlet_velocity: x[4],
            baffle_x: x[5],
            baffle_y: x[6],
            baffle_length: x[7],
            baffle_angle: x[8],
        })
    }

    pub fn as_array(self) -> [f64; DIMENSION] {
        [
            self.inlet_y,
            self.inlet_width,
            self.outlet_y,
            self.outlet_width,
            self.inlet_velocity,
            self.baffle_x,
            self.baffle_y,
            self.baffle_length,
            self.baffle_angle,
        ]
    }
}

impl Default for Design {
    fn default() -> Self {
        Self {
            inlet_y: 0.72,
            inlet_width: 0.24,
            outlet_y: 0.30,
            outlet_width: 0.28,
            inlet_velocity: 0.75,
            baffle_x: 0.52,
            baffle_y: 0.52,
            baffle_length: 0.35,
            baffle_angle: 0.15,
        }
    }
}

/// Numerical and physical controls for one objective evaluation.
#[derive(Clone, Debug)]
pub struct RoomConfig {
    pub nx: usize,
    pub ny: usize,
    pub room_width_m: f64,
    pub room_height_m: f64,
    pub flow_steps: usize,
    pub scalar_steps: usize,
    pub flow_tolerance: f64,
    pub minimum_fresh_air_m2_s: f64,
    pub maximum_mass_imbalance: f64,
    /// Pollutant source locations as fractions of room width and height.
    pub pollutant_sources: Vec<[f64; 2]>,
}

impl Default for RoomConfig {
    fn default() -> Self {
        Self {
            nx: 40,
            ny: 24,
            room_width_m: 5.0,
            room_height_m: 3.0,
            flow_steps: 500,
            scalar_steps: 300,
            flow_tolerance: 5.0e-4,
            minimum_fresh_air_m2_s: 0.18,
            maximum_mass_imbalance: 0.05,
            pollutant_sources: TRAINING_SOURCES.to_vec(),
        }
    }
}

impl RoomConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.nx < 12 || self.ny < 10 {
            return Err("the CFD grid must be at least 12 by 10");
        }
        if self.room_width_m <= 0.0
            || self.room_height_m <= 0.0
            || !self.room_width_m.is_finite()
            || !self.room_height_m.is_finite()
        {
            return Err("room dimensions must be finite and positive");
        }
        if self.flow_steps < 20 || self.scalar_steps < 20 {
            return Err("flow and scalar step counts must be at least 20");
        }
        if !self.flow_tolerance.is_finite()
            || self.flow_tolerance <= 0.0
            || !self.minimum_fresh_air_m2_s.is_finite()
            || self.minimum_fresh_air_m2_s < 0.0
            || !self.maximum_mass_imbalance.is_finite()
            || self.maximum_mass_imbalance <= 0.0
        {
            return Err("solver tolerances and limits must be positive");
        }
        if self.pollutant_sources.is_empty()
            || self.pollutant_sources.iter().any(|source| {
                source
                    .iter()
                    .any(|value| !value.is_finite() || !(0.05..=0.95).contains(value))
            })
        {
            return Err("pollutant sources must lie inside normalized room bounds");
        }
        Ok(())
    }
}

/// Four objectives, two quality-diversity descriptors, and four `<= 0`
/// constraints returned by the simulation.
#[derive(Clone, Debug)]
pub struct Evaluation {
    /// Mean occupied-zone concentration integrated over the scalar horizon and
    /// normalized by its initial occupied-zone mean.
    pub exposure: f64,
    /// Maximum concentration seen by the fixed receptor network, relative to
    /// the initial field maximum.
    pub maximum_receptor: f64,
    /// Normalized `flow_rate * inlet_velocity^2` fan-power proxy.
    pub fan_power: f64,
    /// Diagnostic fraction of the scalar horizon needed to clear 90% of
    /// initial mass. This is deliberately not an objective because it is
    /// right-censored when the threshold is not reached.
    pub clearance_time: f64,
    pub fresh_air_constraint: f64,
    pub baffle_constraint: f64,
    pub mass_balance_constraint: f64,
    pub convergence_constraint: f64,
    pub flow_rate_m2_s: f64,
    pub mass_imbalance: f64,
    pub pressure_drop_lattice: f64,
    pub flow_residual: f64,
    pub flow_iterations: usize,
    /// Fraction of occupied-zone fluid cells below 0.1 m/s. This and
    /// [`Self::flow_rate_m2_s`] are the MAP-Elites behavior descriptors.
    pub low_velocity_fraction: f64,
    pub final_mass_fraction: f64,
    /// Number of pollutant releases aggregated by this evaluation.
    pub source_count: usize,
    /// Source producing the largest occupied-zone exposure.
    pub worst_exposure_source: [f64; 2],
    pub valid: bool,
}

impl Evaluation {
    pub fn objectives(&self) -> [f64; 4] {
        [
            self.exposure,
            self.maximum_receptor,
            self.fan_power,
            self.final_mass_fraction,
        ]
    }

    pub fn constraints(&self) -> [f64; 4] {
        [
            self.fresh_air_constraint,
            self.baffle_constraint,
            self.mass_balance_constraint,
            self.convergence_constraint,
        ]
    }

    pub fn mode_values(&self) -> Vec<f64> {
        if !self.valid {
            return vec![1.0e6; 8];
        }
        let mut values = self.objectives().to_vec();
        values.extend(self.constraints());
        values
    }

    pub fn scalar_objective(&self) -> f64 {
        if !self.valid {
            return 1.0e12;
        }
        let penalty = self
            .constraints()
            .iter()
            .map(|value| value.max(0.0))
            .sum::<f64>();
        self.exposure
            + 0.5 * self.maximum_receptor
            + 0.20 * self.fan_power
            + 0.5 * self.final_mass_fraction
            + 100.0 * penalty
    }

    pub fn feasible(&self) -> bool {
        self.valid && self.constraints().iter().all(|value| *value <= 0.0)
    }

    fn invalid() -> Self {
        Self {
            exposure: 1.0e6,
            maximum_receptor: 1.0e6,
            fan_power: 1.0e6,
            clearance_time: 1.0e6,
            fresh_air_constraint: 1.0e6,
            baffle_constraint: 1.0e6,
            mass_balance_constraint: 1.0e6,
            convergence_constraint: 1.0e6,
            flow_rate_m2_s: 0.0,
            mass_imbalance: 1.0e6,
            pressure_drop_lattice: 1.0e6,
            flow_residual: 1.0e6,
            flow_iterations: 0,
            low_velocity_fraction: 1.0,
            final_mass_fraction: 1.0,
            source_count: 0,
            worst_exposure_source: [f64::NAN; 2],
            valid: false,
        }
    }
}

/// Optional final fields used for CSV output, never allocated by the normal
/// optimizer objective.
#[derive(Clone, Debug)]
pub struct FieldSnapshot {
    pub nx: usize,
    pub ny: usize,
    pub solid: Vec<bool>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub pressure: Vec<f64>,
    pub concentration: Vec<f64>,
    pub pollutant_source: [f64; 2],
}

#[derive(Clone, Debug)]
pub struct DetailedEvaluation {
    pub metrics: Evaluation,
    pub field: FieldSnapshot,
}

/// Immutable, thread-safe CFD objective. Every call allocates isolated solver
/// state, so independent retries or MODE batches can evaluate it concurrently.
#[derive(Clone, Debug)]
pub struct RoomProblem {
    config: RoomConfig,
}

impl RoomProblem {
    pub fn new(config: RoomConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &RoomConfig {
        &self.config
    }

    pub fn evaluate(&self, x: &[f64]) -> Evaluation {
        let Some(design) = Design::decode(x) else {
            return Evaluation::invalid();
        };
        self.simulate(design, false)
            .map(|result| result.metrics)
            .unwrap_or_else(Evaluation::invalid)
    }

    pub fn evaluate_design(&self, design: Design) -> Evaluation {
        self.simulate(design, false)
            .map(|result| result.metrics)
            .unwrap_or_else(Evaluation::invalid)
    }

    pub fn evaluate_detailed(&self, design: Design) -> Option<DetailedEvaluation> {
        self.simulate(design, true)
    }

    /// Construct a problem with the same numerical settings and held-out
    /// pollutant releases.
    pub fn validation_problem(&self) -> Result<Self, &'static str> {
        let mut config = self.config.clone();
        config.pollutant_sources = VALIDATION_SOURCES.to_vec();
        Self::new(config)
    }

    fn simulate(&self, design: Design, capture: bool) -> Option<DetailedEvaluation> {
        if design
            .as_array()
            .iter()
            .zip(LOWER_BOUNDS.iter().zip(UPPER_BOUNDS.iter()))
            .any(|(&value, (&lower, &upper))| value < lower || value > upper)
        {
            return None;
        }
        let geometry = Geometry::new(&self.config, design);
        let flow = solve_flow(&self.config, design, &geometry)?;
        let scalar_results: Vec<ScalarResult> = self
            .config
            .pollutant_sources
            .iter()
            .map(|&source| solve_scalar(&self.config, &geometry, &flow, source))
            .collect();
        let worst_exposure_index = scalar_results
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.exposure.total_cmp(&right.exposure))
            .map(|(index, _)| index)?;
        let exposure = scalar_results
            .iter()
            .map(|scalar| scalar.exposure)
            .fold(f64::NEG_INFINITY, f64::max);
        let maximum_receptor = scalar_results
            .iter()
            .map(|scalar| scalar.maximum_receptor)
            .fold(f64::NEG_INFINITY, f64::max);
        let clearance_time = scalar_results
            .iter()
            .map(|scalar| scalar.clearance_time)
            .fold(f64::NEG_INFINITY, f64::max);
        let final_mass_fraction = scalar_results
            .iter()
            .map(|scalar| scalar.final_mass_fraction)
            .fold(f64::NEG_INFINITY, f64::max);

        let flow_rate = design.inlet_velocity * design.inlet_width * self.config.room_height_m;
        let fan_power =
            (flow_rate * design.inlet_velocity * design.inlet_velocity / 4.6).clamp(0.0, 10.0);
        let metrics = Evaluation {
            exposure,
            maximum_receptor,
            fan_power,
            clearance_time,
            fresh_air_constraint: self.config.minimum_fresh_air_m2_s - flow_rate,
            baffle_constraint: geometry.baffle_violation_m,
            mass_balance_constraint: flow.mass_imbalance - self.config.maximum_mass_imbalance,
            convergence_constraint: flow.residual - self.config.flow_tolerance,
            flow_rate_m2_s: flow_rate,
            mass_imbalance: flow.mass_imbalance,
            pressure_drop_lattice: flow.pressure_drop,
            flow_residual: flow.residual,
            flow_iterations: flow.iterations,
            low_velocity_fraction: flow.low_velocity_fraction,
            final_mass_fraction,
            source_count: scalar_results.len(),
            worst_exposure_source: self.config.pollutant_sources[worst_exposure_index],
            valid: flow.valid
                && scalar_results.iter().all(|scalar| scalar.valid)
                && [
                    exposure,
                    maximum_receptor,
                    fan_power,
                    clearance_time,
                    final_mass_fraction,
                    flow.mass_imbalance,
                    flow.residual,
                    flow.low_velocity_fraction,
                ]
                .iter()
                .all(|value| value.is_finite()),
        };
        let field = if capture {
            FieldSnapshot {
                nx: self.config.nx,
                ny: self.config.ny,
                solid: geometry.solid,
                u: flow.u,
                v: flow.v,
                pressure: flow.pressure,
                concentration: scalar_results[worst_exposure_index].concentration.clone(),
                pollutant_source: self.config.pollutant_sources[worst_exposure_index],
            }
        } else {
            FieldSnapshot {
                nx: 0,
                ny: 0,
                solid: Vec::new(),
                u: Vec::new(),
                v: Vec::new(),
                pressure: Vec::new(),
                concentration: Vec::new(),
                pollutant_source: [f64::NAN; 2],
            }
        };
        Some(DetailedEvaluation { metrics, field })
    }
}

impl Default for RoomProblem {
    fn default() -> Self {
        Self::new(RoomConfig::default()).expect("default room configuration is valid")
    }
}

impl FieldSnapshot {
    /// Write cell-centered final fields for external Python/matplotlib or
    /// ParaView-oriented conversion without doing visualization in the
    /// objective.
    pub fn write_csv(&self, path: impl AsRef<Path>, config: &RoomConfig) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut output = BufWriter::new(file);
        writeln!(
            output,
            "i,j,x_m,y_m,solid,u_lattice,v_lattice,pressure_lattice,concentration,source_x_fraction,source_y_fraction"
        )?;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let index = cell(i, j, self.nx);
                let x = (i as f64 + 0.5) * config.room_width_m / self.nx as f64;
                let y = (j as f64 + 0.5) * config.room_height_m / self.ny as f64;
                writeln!(
                    output,
                    "{i},{j},{x:.9},{y:.9},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
                    u8::from(self.solid[index]),
                    self.u[index],
                    self.v[index],
                    self.pressure[index],
                    self.concentration[index],
                    self.pollutant_source[0],
                    self.pollutant_source[1]
                )?;
            }
        }
        output.flush()
    }
}

/// Diagnostics from a symmetric straight-channel reference case.
#[derive(Clone, Copy, Debug)]
pub struct ChannelReference {
    pub nx: usize,
    pub ny: usize,
    pub symmetry_relative_l2: f64,
    pub maximum_transverse_velocity: f64,
    pub maximum_to_mean_axial_velocity: f64,
    pub mass_imbalance: f64,
    pub residual: f64,
    pub iterations: usize,
}

/// Run a full-height, baffle-free straight channel used as a numerical
/// reference check for symmetry, transverse velocity, and flux conservation.
///
/// This is a property-based solver verification rather than a claim of
/// agreement with a calibrated physical experiment.
pub fn straight_channel_reference(
    nx: usize,
    ny: usize,
    flow_steps: usize,
) -> Result<ChannelReference, &'static str> {
    let config = RoomConfig {
        nx,
        ny,
        flow_steps,
        scalar_steps: 20,
        flow_tolerance: 1.0e-6,
        maximum_mass_imbalance: 1.0,
        ..Default::default()
    };
    config.validate()?;
    let mut solid = vec![false; nx * ny];
    for i in 0..nx {
        solid[cell(i, 0, nx)] = true;
        solid[cell(i, ny - 1, nx)] = true;
    }
    let inlet: Vec<usize> = (1..ny - 1).map(|j| cell(0, j, nx)).collect();
    let outlet: Vec<usize> = (1..ny - 1).map(|j| cell(nx - 1, j, nx)).collect();
    let geometry = Geometry {
        solid,
        inlet,
        outlet,
        baffle_violation_m: -1.0,
    };
    let design = Design {
        inlet_velocity: 0.5,
        ..Default::default()
    };
    let flow = solve_flow(&config, design, &geometry).ok_or("straight-channel flow diverged")?;
    let mid = nx / 2;
    let axial: Vec<f64> = (1..ny - 1).map(|j| flow.u[cell(mid, j, nx)]).collect();
    let transverse: Vec<f64> = (1..ny - 1).map(|j| flow.v[cell(mid, j, nx)]).collect();
    let mean_axial = axial.iter().sum::<f64>() / axial.len() as f64;
    let symmetry_relative_l2 = (axial
        .iter()
        .zip(axial.iter().rev())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        / axial.len() as f64)
        .sqrt()
        / mean_axial.abs().max(1.0e-12);
    let maximum_transverse_velocity = transverse
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let maximum_axial_velocity = axial.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(ChannelReference {
        nx,
        ny,
        symmetry_relative_l2,
        maximum_transverse_velocity,
        maximum_to_mean_axial_velocity: maximum_axial_velocity / mean_axial.max(1.0e-12),
        mass_imbalance: flow.mass_imbalance,
        residual: flow.residual,
        iterations: flow.iterations,
    })
}

#[derive(Clone, Debug)]
struct Geometry {
    solid: Vec<bool>,
    inlet: Vec<usize>,
    outlet: Vec<usize>,
    baffle_violation_m: f64,
}

impl Geometry {
    fn new(config: &RoomConfig, design: Design) -> Self {
        let mut solid = vec![false; config.nx * config.ny];
        for i in 0..config.nx {
            solid[cell(i, 0, config.nx)] = true;
            solid[cell(i, config.ny - 1, config.nx)] = true;
        }
        for j in 0..config.ny {
            solid[cell(0, j, config.nx)] = true;
            solid[cell(config.nx - 1, j, config.nx)] = true;
        }

        let inlet = vent_cells(config, design.inlet_y, design.inlet_width, 0);
        let outlet = vent_cells(config, design.outlet_y, design.outlet_width, config.nx - 1);
        for &index in inlet.iter().chain(&outlet) {
            solid[index] = false;
        }

        let center_x = design.baffle_x * config.room_width_m;
        let center_y = design.baffle_y * config.room_height_m;
        let length = design.baffle_length * config.room_height_m;
        let half_dx = 0.5 * length * design.baffle_angle.cos();
        let half_dy = 0.5 * length * design.baffle_angle.sin();
        let x0 = center_x - half_dx;
        let y0 = center_y - half_dy;
        let x1 = center_x + half_dx;
        let y1 = center_y + half_dy;
        let dx = config.room_width_m / config.nx as f64;
        let dy = config.room_height_m / config.ny as f64;
        let thickness = 0.60 * dx.hypot(dy);
        for j in 1..config.ny - 1 {
            for i in 1..config.nx - 1 {
                let x = (i as f64 + 0.5) * dx;
                let y = (j as f64 + 0.5) * dy;
                if segment_distance(x, y, x0, y0, x1, y1) <= thickness {
                    solid[cell(i, j, config.nx)] = true;
                }
            }
        }
        let margin = dx.max(dy);
        let baffle_violation_m = [
            margin - x0,
            x0 - (config.room_width_m - margin),
            margin - x1,
            x1 - (config.room_width_m - margin),
            margin - y0,
            y0 - (config.room_height_m - margin),
            margin - y1,
            y1 - (config.room_height_m - margin),
        ]
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
        Self {
            solid,
            inlet,
            outlet,
            baffle_violation_m,
        }
    }
}

fn vent_cells(config: &RoomConfig, center_y: f64, width: f64, i: usize) -> Vec<usize> {
    let lower = center_y - 0.5 * width;
    let upper = center_y + 0.5 * width;
    let mut cells = Vec::new();
    for j in 1..config.ny - 1 {
        let y = (j as f64 + 0.5) / config.ny as f64;
        if y >= lower && y <= upper {
            cells.push(cell(i, j, config.nx));
        }
    }
    if cells.is_empty() {
        let j = ((center_y * config.ny as f64).floor() as usize).clamp(1, config.ny - 2);
        cells.push(cell(i, j, config.nx));
    }
    cells
}

fn segment_distance(x: f64, y: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let length_squared = dx * dx + dy * dy;
    let t = if length_squared <= 1.0e-20 {
        0.0
    } else {
        ((x - x0) * dx + (y - y0) * dy) / length_squared
    }
    .clamp(0.0, 1.0);
    (x - (x0 + t * dx)).hypot(y - (y0 + t * dy))
}

#[derive(Clone, Debug)]
struct FlowResult {
    u: Vec<f64>,
    v: Vec<f64>,
    pressure: Vec<f64>,
    residual: f64,
    mass_imbalance: f64,
    pressure_drop: f64,
    iterations: usize,
    low_velocity_fraction: f64,
    valid: bool,
}

fn solve_flow(config: &RoomConfig, design: Design, geometry: &Geometry) -> Option<FlowResult> {
    let count = config.nx * config.ny;
    let tau = 0.58;
    let omega = 1.0 / tau;
    let inlet_lattice_velocity = (design.inlet_velocity * 0.05).clamp(0.0125, 0.075);
    let mut populations = vec![0.0; FLOW_DIRECTIONS * count];
    let mut next = vec![0.0; populations.len()];
    let mut u = vec![0.0; count];
    let mut v = vec![0.0; count];
    let mut rho = vec![1.0; count];
    for index in 0..count {
        if !geometry.solid[index] {
            for direction in 0..FLOW_DIRECTIONS {
                populations[distribution(direction, index, count)] =
                    flow_equilibrium(direction, 1.0, 0.0, 0.0);
            }
        }
    }
    impose_flow_boundaries(
        &mut populations,
        &mut rho,
        &mut u,
        &mut v,
        config,
        geometry,
        inlet_lattice_velocity,
    );

    let mut old_u = u.clone();
    let mut old_v = v.clone();
    let mut residual = f64::INFINITY;
    let mut iterations = config.flow_steps;
    for step in 0..config.flow_steps {
        for index in 0..count {
            if geometry.solid[index] {
                rho[index] = 1.0;
                u[index] = 0.0;
                v[index] = 0.0;
                continue;
            }
            let mut density = 0.0;
            let mut momentum_x = 0.0;
            let mut momentum_y = 0.0;
            for direction in 0..FLOW_DIRECTIONS {
                let value = populations[distribution(direction, index, count)];
                density += value;
                momentum_x += value * FLOW_EX[direction] as f64;
                momentum_y += value * FLOW_EY[direction] as f64;
            }
            if !density.is_finite() || density <= 1.0e-12 {
                return None;
            }
            rho[index] = density;
            u[index] = momentum_x / density;
            v[index] = momentum_y / density;
            if !u[index].is_finite() || !v[index].is_finite() || u[index].hypot(v[index]) > 0.35 {
                return None;
            }
        }

        next.fill(0.0);
        for j in 0..config.ny {
            for i in 0..config.nx {
                let index = cell(i, j, config.nx);
                if geometry.solid[index] {
                    continue;
                }
                for direction in 0..FLOW_DIRECTIONS {
                    let slot = distribution(direction, index, count);
                    let equilibrium = flow_equilibrium(direction, rho[index], u[index], v[index]);
                    let post_collision =
                        populations[slot] - omega * (populations[slot] - equilibrium);
                    let ni = i as isize + FLOW_EX[direction];
                    let nj = j as isize + FLOW_EY[direction];
                    if let Some(neighbor) = neighbor_index(ni, nj, config.nx, config.ny)
                        && !geometry.solid[neighbor]
                    {
                        next[distribution(direction, neighbor, count)] += post_collision;
                    } else {
                        next[distribution(FLOW_OPPOSITE[direction], index, count)] +=
                            post_collision;
                    }
                }
            }
        }
        std::mem::swap(&mut populations, &mut next);
        impose_flow_boundaries(
            &mut populations,
            &mut rho,
            &mut u,
            &mut v,
            config,
            geometry,
            inlet_lattice_velocity,
        );

        if (step + 1) % 20 == 0 {
            update_macroscopic(
                &populations,
                &geometry.solid,
                &mut rho,
                &mut u,
                &mut v,
                count,
            )?;
            let mut difference = 0.0;
            let mut active = 0usize;
            for index in 0..count {
                if !geometry.solid[index] {
                    difference +=
                        (u[index] - old_u[index]).powi(2) + (v[index] - old_v[index]).powi(2);
                    active += 1;
                }
            }
            residual = (difference / active.max(1) as f64).sqrt();
            old_u.copy_from_slice(&u);
            old_v.copy_from_slice(&v);
            if step >= 99 && residual <= config.flow_tolerance {
                iterations = step + 1;
                break;
            }
        }
    }
    update_macroscopic(
        &populations,
        &geometry.solid,
        &mut rho,
        &mut u,
        &mut v,
        count,
    )?;
    if !residual.is_finite() {
        residual = 1.0;
    }
    let inlet_flux = geometry
        .inlet
        .iter()
        .map(|&index| u[index + 1].max(0.0))
        .sum::<f64>();
    let outlet_flux = geometry
        .outlet
        .iter()
        .map(|&index| u[index - 1].max(0.0))
        .sum::<f64>();
    let mass_imbalance = (inlet_flux - outlet_flux).abs() / inlet_flux.max(1.0e-12);
    let inlet_density = geometry
        .inlet
        .iter()
        .map(|&index| rho[index + 1])
        .sum::<f64>()
        / geometry.inlet.len() as f64;
    let outlet_density = geometry
        .outlet
        .iter()
        .map(|&index| rho[index - 1])
        .sum::<f64>()
        / geometry.outlet.len() as f64;
    let pressure_drop = ((inlet_density - outlet_density) / 3.0).abs();
    let pressure: Vec<f64> = rho.iter().map(|density| (density - 1.0) / 3.0).collect();
    let low_velocity_fraction = occupied_low_velocity_fraction(config, geometry, &u, &v);
    Some(FlowResult {
        u,
        v,
        pressure,
        residual,
        mass_imbalance,
        pressure_drop,
        iterations,
        low_velocity_fraction,
        valid: mass_imbalance.is_finite()
            && residual.is_finite()
            && low_velocity_fraction.is_finite(),
    })
}

fn occupied_low_velocity_fraction(
    config: &RoomConfig,
    geometry: &Geometry,
    u: &[f64],
    v: &[f64],
) -> f64 {
    // The lattice conversion is u_lattice = 0.05 * u_m/s, so 0.005
    // corresponds to the documented 0.1 m/s occupied-zone threshold.
    const LOW_SPEED_LATTICE: f64 = 0.005;
    let mut low_speed = 0usize;
    let mut cells = 0usize;
    for j in 1..config.ny - 1 {
        let y = (j as f64 + 0.5) / config.ny as f64;
        if !(0.08..=0.65).contains(&y) {
            continue;
        }
        for i in 1..config.nx - 1 {
            let x = (i as f64 + 0.5) / config.nx as f64;
            let index = cell(i, j, config.nx);
            if (0.10..=0.90).contains(&x) && !geometry.solid[index] {
                low_speed += usize::from(u[index].hypot(v[index]) < LOW_SPEED_LATTICE);
                cells += 1;
            }
        }
    }
    low_speed as f64 / cells.max(1) as f64
}

fn update_macroscopic(
    populations: &[f64],
    solid: &[bool],
    rho: &mut [f64],
    u: &mut [f64],
    v: &mut [f64],
    count: usize,
) -> Option<()> {
    for index in 0..count {
        if solid[index] {
            rho[index] = 1.0;
            u[index] = 0.0;
            v[index] = 0.0;
            continue;
        }
        let mut density = 0.0;
        let mut momentum_x = 0.0;
        let mut momentum_y = 0.0;
        for direction in 0..FLOW_DIRECTIONS {
            let value = populations[distribution(direction, index, count)];
            density += value;
            momentum_x += value * FLOW_EX[direction] as f64;
            momentum_y += value * FLOW_EY[direction] as f64;
        }
        if !density.is_finite() || density <= 1.0e-12 {
            return None;
        }
        rho[index] = density;
        u[index] = momentum_x / density;
        v[index] = momentum_y / density;
        if !u[index].is_finite() || !v[index].is_finite() || u[index].hypot(v[index]) > 0.35 {
            return None;
        }
    }
    Some(())
}

fn impose_flow_boundaries(
    populations: &mut [f64],
    rho: &mut [f64],
    u: &mut [f64],
    v: &mut [f64],
    config: &RoomConfig,
    geometry: &Geometry,
    inlet_velocity: f64,
) {
    let count = config.nx * config.ny;
    for &index in &geometry.inlet {
        rho[index] = 1.0;
        u[index] = inlet_velocity;
        v[index] = 0.0;
        for direction in 0..FLOW_DIRECTIONS {
            populations[distribution(direction, index, count)] =
                flow_equilibrium(direction, 1.0, inlet_velocity, 0.0);
        }
    }
    for &index in &geometry.outlet {
        let neighbor = index - 1;
        let outlet_u = (inlet_velocity * geometry.inlet.len() as f64
            / geometry.outlet.len() as f64)
            .clamp(0.0, 0.15);
        let outlet_v = v[neighbor].clamp(-0.10, 0.10);
        rho[index] = 1.0;
        u[index] = outlet_u;
        v[index] = outlet_v;
        for direction in 0..FLOW_DIRECTIONS {
            populations[distribution(direction, index, count)] =
                flow_equilibrium(direction, 1.0, outlet_u, outlet_v);
        }
    }
}

fn flow_equilibrium(direction: usize, rho: f64, u: f64, v: f64) -> f64 {
    let eu = FLOW_EX[direction] as f64 * u + FLOW_EY[direction] as f64 * v;
    let velocity_squared = u * u + v * v;
    FLOW_WEIGHTS[direction] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * velocity_squared)
}

#[derive(Clone, Debug)]
struct ScalarResult {
    concentration: Vec<f64>,
    exposure: f64,
    maximum_receptor: f64,
    clearance_time: f64,
    final_mass_fraction: f64,
    valid: bool,
}

fn solve_scalar(
    config: &RoomConfig,
    geometry: &Geometry,
    flow: &FlowResult,
    source: [f64; 2],
) -> ScalarResult {
    let count = config.nx * config.ny;
    let tau = 0.68;
    let omega = 1.0 / tau;
    let mut concentration = vec![0.0; count];
    let source_x = source[0] * config.nx as f64;
    let source_y = source[1] * config.ny as f64;
    let sigma = 0.075 * config.nx.min(config.ny) as f64;
    for j in 1..config.ny - 1 {
        for i in 1..config.nx - 1 {
            let index = cell(i, j, config.nx);
            if geometry.solid[index] {
                continue;
            }
            let radius_squared =
                (i as f64 + 0.5 - source_x).powi(2) + (j as f64 + 0.5 - source_y).powi(2);
            concentration[index] = (-0.5 * radius_squared / (sigma * sigma)).exp();
        }
    }
    let initial_mass = concentration.iter().sum::<f64>().max(1.0e-12);
    let initial_max = concentration.iter().copied().fold(0.0_f64, f64::max);
    let initial_occupied = occupied_mean(config, geometry, &concentration).max(1.0e-12);
    let receptors = receptor_indices(config, geometry);
    let mut populations = vec![0.0; SCALAR_DIRECTIONS * count];
    let mut next = vec![0.0; populations.len()];
    for index in 0..count {
        if geometry.solid[index] {
            continue;
        }
        for direction in 0..SCALAR_DIRECTIONS {
            populations[distribution(direction, index, count)] = scalar_equilibrium(
                direction,
                concentration[index],
                flow.u[index],
                flow.v[index],
            );
        }
    }
    let mut exposure_sum = 0.0;
    let mut maximum_receptor = 0.0_f64;
    let mut clearance_step = None;
    let mut valid = true;
    for step in 0..config.scalar_steps {
        next.fill(0.0);
        for j in 0..config.ny {
            for i in 0..config.nx {
                let index = cell(i, j, config.nx);
                if geometry.solid[index] {
                    continue;
                }
                let scalar = (0..SCALAR_DIRECTIONS)
                    .map(|direction| populations[distribution(direction, index, count)])
                    .sum::<f64>();
                if !scalar.is_finite() {
                    valid = false;
                    continue;
                }
                for direction in 0..SCALAR_DIRECTIONS {
                    let slot = distribution(direction, index, count);
                    let equilibrium = scalar_equilibrium(
                        direction,
                        scalar.max(0.0),
                        flow.u[index],
                        flow.v[index],
                    );
                    let post_collision =
                        populations[slot] - omega * (populations[slot] - equilibrium);
                    let ni = i as isize + SCALAR_EX[direction];
                    let nj = j as isize + SCALAR_EY[direction];
                    if let Some(neighbor) = neighbor_index(ni, nj, config.nx, config.ny)
                        && !geometry.solid[neighbor]
                    {
                        next[distribution(direction, neighbor, count)] += post_collision;
                    } else {
                        next[distribution(SCALAR_OPPOSITE[direction], index, count)] +=
                            post_collision;
                    }
                }
            }
        }
        std::mem::swap(&mut populations, &mut next);
        for &index in &geometry.inlet {
            for direction in 0..SCALAR_DIRECTIONS {
                populations[distribution(direction, index, count)] = 0.0;
            }
        }
        for &index in &geometry.outlet {
            for direction in 0..SCALAR_DIRECTIONS {
                populations[distribution(direction, index, count)] = 0.0;
            }
        }
        for index in 0..count {
            concentration[index] = if geometry.solid[index] {
                0.0
            } else {
                (0..SCALAR_DIRECTIONS)
                    .map(|direction| populations[distribution(direction, index, count)])
                    .sum::<f64>()
                    .max(0.0)
            };
        }
        exposure_sum += occupied_mean(config, geometry, &concentration) / initial_occupied;
        for &index in &receptors {
            maximum_receptor =
                maximum_receptor.max(concentration[index] / initial_max.max(1.0e-12));
        }
        let mass_fraction = concentration.iter().sum::<f64>() / initial_mass;
        if clearance_step.is_none() && mass_fraction <= 0.10 {
            clearance_step = Some(step + 1);
        }
    }
    let final_mass_fraction = concentration.iter().sum::<f64>() / initial_mass;
    ScalarResult {
        concentration,
        exposure: exposure_sum / config.scalar_steps as f64,
        maximum_receptor,
        clearance_time: clearance_step.unwrap_or(config.scalar_steps) as f64
            / config.scalar_steps as f64,
        final_mass_fraction,
        valid: valid
            && exposure_sum.is_finite()
            && maximum_receptor.is_finite()
            && final_mass_fraction.is_finite(),
    }
}

fn scalar_equilibrium(direction: usize, concentration: f64, u: f64, v: f64) -> f64 {
    let eu = SCALAR_EX[direction] as f64 * u + SCALAR_EY[direction] as f64 * v;
    SCALAR_WEIGHTS[direction] * concentration * (1.0 + 3.0 * eu)
}

fn occupied_mean(config: &RoomConfig, geometry: &Geometry, concentration: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut cells = 0usize;
    for j in 1..config.ny - 1 {
        let y = (j as f64 + 0.5) / config.ny as f64;
        if !(0.08..=0.65).contains(&y) {
            continue;
        }
        for i in 1..config.nx - 1 {
            let x = (i as f64 + 0.5) / config.nx as f64;
            let index = cell(i, j, config.nx);
            if (0.10..=0.90).contains(&x) && !geometry.solid[index] {
                total += concentration[index];
                cells += 1;
            }
        }
    }
    total / cells.max(1) as f64
}

fn receptor_indices(config: &RoomConfig, geometry: &Geometry) -> Vec<usize> {
    [
        (0.25, 0.20),
        (0.50, 0.20),
        (0.75, 0.20),
        (0.25, 0.50),
        (0.50, 0.50),
        (0.75, 0.50),
    ]
    .into_iter()
    .filter_map(|(x, y)| {
        let i = (x * config.nx as f64).floor() as usize;
        let j = (y * config.ny as f64).floor() as usize;
        let index = cell(i.min(config.nx - 1), j.min(config.ny - 1), config.nx);
        (!geometry.solid[index]).then_some(index)
    })
    .collect()
}

#[inline]
fn distribution(direction: usize, index: usize, count: usize) -> usize {
    direction * count + index
}

#[inline]
fn cell(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}

fn neighbor_index(i: isize, j: isize, nx: usize, ny: usize) -> Option<usize> {
    if i < 0 || j < 0 || i >= nx as isize || j >= ny as isize {
        None
    } else {
        Some(cell(i as usize, j as usize, nx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_problem() -> RoomProblem {
        RoomProblem::new(RoomConfig {
            nx: 20,
            ny: 12,
            flow_steps: 120,
            scalar_steps: 80,
            flow_tolerance: 5.0e-3,
            maximum_mass_imbalance: 1.0,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn default_design_is_inside_bounds_and_roundtrips() {
        let design = Design::default();
        let values = design.as_array();
        assert_eq!(Design::decode(&values), Some(design));
        for ((value, lower), upper) in values.iter().zip(LOWER_BOUNDS).zip(UPPER_BOUNDS) {
            assert!((lower..=upper).contains(value));
        }
    }

    #[test]
    fn geometry_opens_vents_and_rasterizes_baffle() {
        let config = RoomConfig {
            nx: 24,
            ny: 16,
            ..Default::default()
        };
        let geometry = Geometry::new(&config, Design::default());
        assert!(!geometry.inlet.is_empty());
        assert!(!geometry.outlet.is_empty());
        assert!(geometry.inlet.iter().all(|&index| !geometry.solid[index]));
        assert!(geometry.outlet.iter().all(|&index| !geometry.solid[index]));
        let interior_solids = (1..config.ny - 1)
            .flat_map(|j| (1..config.nx - 1).map(move |i| cell(i, j, config.nx)))
            .filter(|&index| geometry.solid[index])
            .count();
        assert!(interior_solids > 0);
        assert!(geometry.baffle_violation_m <= 0.0);
    }

    #[test]
    fn simulation_is_deterministic_finite_and_zero_on_solids() {
        let problem = small_problem();
        let first = problem.evaluate_detailed(Design::default()).unwrap();
        let second = problem.evaluate_detailed(Design::default()).unwrap();
        assert!(first.metrics.valid);
        assert_eq!(
            first.metrics.scalar_objective(),
            second.metrics.scalar_objective()
        );
        assert_eq!(first.field.u, second.field.u);
        assert_eq!(first.field.concentration, second.field.concentration);
        for index in 0..first.field.solid.len() {
            if first.field.solid[index] {
                assert_eq!(first.field.u[index], 0.0);
                assert_eq!(first.field.v[index], 0.0);
                assert_eq!(first.field.concentration[index], 0.0);
            }
        }
    }

    #[test]
    fn scalar_transport_metrics_and_descriptors_are_bounded() {
        let evaluation = small_problem().evaluate_design(Design::default());
        assert!(evaluation.valid);
        assert!((0.0..=1.01).contains(&evaluation.final_mass_fraction));
        assert!(evaluation.exposure > 0.0);
        assert!((0.0..=1.0).contains(&evaluation.clearance_time));
        assert!((0.0..=1.0).contains(&evaluation.low_velocity_fraction));
        assert_eq!(evaluation.objectives()[3], evaluation.final_mass_fraction);
        assert_eq!(evaluation.source_count, TRAINING_SOURCES.len());
    }

    #[test]
    fn robust_objectives_are_worst_case_and_validation_sources_are_held_out() {
        let problem = small_problem();
        let robust = problem.evaluate_design(Design::default());
        let individual: Vec<Evaluation> = TRAINING_SOURCES
            .iter()
            .map(|&source| {
                let mut config = problem.config().clone();
                config.pollutant_sources = vec![source];
                RoomProblem::new(config)
                    .unwrap()
                    .evaluate_design(Design::default())
            })
            .collect();
        for objective in 0..4 {
            let expected = individual
                .iter()
                .map(|evaluation| evaluation.objectives()[objective])
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(robust.objectives()[objective], expected);
        }

        let validation = problem.validation_problem().unwrap();
        assert_eq!(validation.config().pollutant_sources, VALIDATION_SOURCES);
        let held_out = validation.evaluate_design(Design::default());
        assert_eq!(held_out.source_count, VALIDATION_SOURCES.len());
        assert_ne!(robust.scalar_objective(), held_out.scalar_objective());
    }

    #[test]
    fn malformed_or_out_of_bounds_decisions_are_rejected() {
        let problem = small_problem();
        assert!(!problem.evaluate(&[0.0]).valid);
        let mut values = Design::default().as_array();
        values[4] = 100.0;
        assert!(!problem.evaluate(&values).valid);
        values[4] = f64::NAN;
        assert!(!problem.evaluate(&values).valid);
    }

    #[test]
    fn configuration_validation_rejects_bad_grids_and_steps() {
        let config = RoomConfig {
            nx: 4,
            ..Default::default()
        };
        assert!(RoomProblem::new(config).is_err());
        let config = RoomConfig {
            scalar_steps: 1,
            ..Default::default()
        };
        assert!(RoomProblem::new(config).is_err());
        let config = RoomConfig {
            pollutant_sources: vec![],
            ..Default::default()
        };
        assert!(RoomProblem::new(config).is_err());
        let config = RoomConfig {
            flow_tolerance: f64::NAN,
            ..Default::default()
        };
        assert!(RoomProblem::new(config).is_err());
    }

    #[test]
    fn straight_channel_reference_is_symmetric_and_flux_balanced() {
        let reference = straight_channel_reference(48, 20, 1_200).unwrap();
        assert!(reference.symmetry_relative_l2 < 1.0e-10, "{reference:?}");
        assert!(
            reference.maximum_transverse_velocity < 5.0e-5,
            "{reference:?}"
        );
        assert!(
            (1.0..=2.5).contains(&reference.maximum_to_mean_axial_velocity),
            "{reference:?}"
        );
        assert!(reference.mass_imbalance < 0.02, "{reference:?}");
        assert!(reference.residual.is_finite());
    }
}
