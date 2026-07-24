//! Rapier-based trebuchet simulation and fcmaes optimization helpers.
//!
//! Each objective call constructs and advances an independent Rapier world.
//! Rapier's optional `parallel` feature is intentionally disabled: parallelism
//! belongs at the outer objective-evaluation level where fcmaes can keep every
//! worker busy on a separate candidate.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    Archive, BiteParams, Fitness, MapElitesParams, Mode, ModeParams, QdBatchFitness, RetryBounds,
    RetryConfig, RetryImprovement, RetryRunResult, Rng, map_elites_batch_with_progress,
    optimize_bite, parallel_batch, pareto_indices, retry,
};
use rapier2d_f64::prelude::*;

pub const DIMENSION: usize = 8;
pub const OBJECTIVES: usize = 3;
pub const GRAVITY: f64 = 9.81;
pub const PIVOT_HEIGHT: f64 = 7.0;
pub const PROJECTILE_RADIUS: f64 = 0.12;
pub const QD_DESCRIPTOR_LOWER: [f64; 2] = [0.0, 0.0];
pub const QD_DESCRIPTOR_UPPER: [f64; 2] = [60.0, 8.0];

pub const LOWER_BOUNDS: [f64; DIMENSION] = [2.0, 20.0, 0.5, 1.0, -1.25, -0.25, 0.0, 0.0];
pub const UPPER_BOUNDS: [f64; DIMENSION] = [6.0, 220.0, 10.0, 6.0, -0.35, 0.55, 12.0, 20.0];

/// A readable, deliberately non-optimal starting point.
pub const INITIAL_DESIGN: [f64; DIMENSION] = [3.5, 80.0, 4.0, 2.5, -0.75, -0.05, 2.0, 2.0];

/// The eight optimizer controls in physical units.
#[derive(Clone, Debug, PartialEq)]
pub struct Design {
    pub arm_length: f64,
    pub counterweight_mass: f64,
    pub projectile_mass: f64,
    pub sling_length: f64,
    pub initial_arm_angle: f64,
    pub release_angle: f64,
    pub joint_damping: f64,
    pub pivot_friction: f64,
}

impl Design {
    pub fn from_slice(x: &[f64]) -> Result<Self, &'static str> {
        if x.len() != DIMENSION {
            return Err("a trebuchet design must contain exactly eight values");
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err("all trebuchet design values must be finite");
        }
        if x.iter()
            .zip(LOWER_BOUNDS.iter().zip(UPPER_BOUNDS))
            .any(|(&value, (&lower, upper))| value < lower || value > upper)
        {
            return Err("trebuchet design lies outside the supported bounds");
        }
        Ok(Self {
            arm_length: x[0],
            counterweight_mass: x[1],
            projectile_mass: x[2],
            sling_length: x[3],
            initial_arm_angle: x[4],
            release_angle: x[5],
            joint_damping: x[6],
            pivot_friction: x[7],
        })
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.arm_length,
            self.counterweight_mass,
            self.projectile_mass,
            self.sling_length,
            self.initial_arm_angle,
            self.release_angle,
            self.joint_damping,
            self.pivot_friction,
        ]
    }
}

impl Default for Design {
    fn default() -> Self {
        Self::from_slice(&INITIAL_DESIGN).expect("the built-in design is valid")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulationStatus {
    Landed,
    NoRelease,
    NoLanding,
    Invalid,
}

impl SimulationStatus {
    pub fn is_valid(self) -> bool {
        self == Self::Landed
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Landed => "landed",
            Self::NoRelease => "no-release",
            Self::NoLanding => "no-landing",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub target_position: f64,
    pub time_step: f64,
    pub max_time: f64,
    pub record_trajectory: bool,
    pub record_stride: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            target_position: 35.0,
            time_step: 1.0 / 180.0,
            max_time: 8.0,
            record_trajectory: false,
            record_stride: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrajectoryPoint {
    pub time: f64,
    pub arm_angle: f64,
    pub counterweight: [f64; 2],
    pub arm_tip: [f64; 2],
    pub projectile: [f64; 2],
    pub released: bool,
}

#[derive(Clone, Debug)]
pub struct SimulationResult {
    pub status: SimulationStatus,
    pub landing_position: f64,
    pub target_error: f64,
    pub input_energy: f64,
    pub peak_joint_force: f64,
    pub release_time: Option<f64>,
    pub flight_time: f64,
    pub apex_height: f64,
    pub scalar_score: f64,
    pub trajectory: Vec<TrajectoryPoint>,
}

impl SimulationResult {
    pub fn objectives(&self) -> [f64; OBJECTIVES] {
        if self.status.is_valid() {
            [self.target_error, self.input_energy, self.peak_joint_force]
        } else {
            // An invalid low-energy design must not survive as a Pareto point.
            let penalty = invalid_penalty(self.status);
            [
                self.target_error + penalty,
                self.input_energy + 10.0 * penalty,
                self.peak_joint_force + 10.0 * penalty,
            ]
        }
    }
}

fn invalid_penalty(status: SimulationStatus) -> f64 {
    match status {
        SimulationStatus::Landed => 0.0,
        SimulationStatus::NoLanding => 500.0,
        SimulationStatus::NoRelease => 1_000.0,
        SimulationStatus::Invalid => 2_000.0,
    }
}

fn world_point(angle: f64, local_x: f64) -> [f64; 2] {
    [
        local_x.mul_add(angle.cos(), 0.0),
        PIVOT_HEIGHT + local_x * angle.sin(),
    ]
}

struct ReplayGeometry {
    arm_handle: RigidBodyHandle,
    projectile_handle: RigidBodyHandle,
    arm_length: f64,
    short_arm: f64,
}

fn record_point(
    trajectory: &mut Vec<TrajectoryPoint>,
    world: &PhysicsWorld,
    geometry: &ReplayGeometry,
    time: f64,
    released: bool,
) {
    let arm = &world.bodies[geometry.arm_handle];
    let angle = arm.rotation().angle();
    let position = arm.position();
    let counterweight = position * Vector::new(-geometry.short_arm, 0.0);
    let arm_tip = position * Vector::new(geometry.arm_length, 0.0);
    let projectile = world.bodies[geometry.projectile_handle].translation();
    trajectory.push(TrajectoryPoint {
        time,
        arm_angle: angle,
        counterweight: [counterweight.x, counterweight.y],
        arm_tip: [arm_tip.x, arm_tip.y],
        projectile: [projectile.x, projectile.y],
        released,
    });
}

/// Run one deterministic f64 Rapier simulation.
pub fn simulate(design: &Design, config: &SimulationConfig) -> SimulationResult {
    if !config.target_position.is_finite()
        || config.target_position <= 0.0
        || !config.time_step.is_finite()
        || config.time_step <= 0.0
        || !config.max_time.is_finite()
        || config.max_time <= config.time_step
    {
        return SimulationResult {
            status: SimulationStatus::Invalid,
            landing_position: 0.0,
            target_error: config.target_position.abs(),
            input_energy: 0.0,
            peak_joint_force: 0.0,
            release_time: None,
            flight_time: 0.0,
            apex_height: 0.0,
            scalar_score: 2_000.0,
            trajectory: Vec::new(),
        };
    }

    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -GRAVITY);
    world.integration_parameters.dt = config.time_step;

    // A broad ground slab catches the projectile even for poor candidates.
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.15)),
        ColliderBuilder::cuboid(250.0, 0.15)
            .friction(0.85)
            .restitution(0.02),
    );
    let pivot_handle =
        world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(0.0, PIVOT_HEIGHT)));

    let short_arm = 0.32 * design.arm_length;
    let total_arm = design.arm_length + short_arm;
    let arm_center = 0.5 * (design.arm_length - short_arm);
    let arm_handle = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, PIVOT_HEIGHT))
            .rotation(design.initial_arm_angle)
            .can_sleep(false),
    );
    world.insert_collider(
        ColliderBuilder::cuboid(0.5 * total_arm, 0.075)
            .translation(Vector::new(arm_center, 0.0))
            .mass(3.0 + 0.8 * total_arm)
            .friction(0.6),
        Some(arm_handle),
    );
    world.insert_collider(
        ColliderBuilder::ball(0.24 + 0.00045 * design.counterweight_mass)
            .translation(Vector::new(-short_arm, 0.0))
            .mass(design.counterweight_mass)
            .friction(0.7),
        Some(arm_handle),
    );

    let pivot_joint = world.insert_impulse_joint(
        pivot_handle,
        arm_handle,
        RevoluteJointBuilder::new()
            .local_anchor1(Vector::new(0.0, 0.0))
            .local_anchor2(Vector::new(0.0, 0.0))
            .limits([-1.50, 1.70]),
    );

    let tip = world_point(design.initial_arm_angle, design.arm_length);
    let counterweight_drop = short_arm * (1.0 - design.initial_arm_angle.sin()).max(0.0);
    let input_energy = design.counterweight_mass * GRAVITY * counterweight_drop;
    // Start with the taut sling trailing the arm's counter-clockwise motion.
    // Putting the projectile ahead of the tip makes the rope pull it backward
    // during take-up and reverses the intended throw direction.
    let projectile_position = Vector::new(
        tip[0] + design.sling_length * design.initial_arm_angle.sin(),
        tip[1] - design.sling_length * design.initial_arm_angle.cos(),
    );
    if projectile_position.y <= PROJECTILE_RADIUS {
        let target_error = config.target_position;
        let status = SimulationStatus::Invalid;
        return SimulationResult {
            status,
            landing_position: 0.0,
            target_error,
            input_energy,
            peak_joint_force: 0.0,
            release_time: None,
            flight_time: 0.0,
            apex_height: projectile_position.y,
            scalar_score: target_error + 0.002 * input_energy + invalid_penalty(status),
            trajectory: Vec::new(),
        };
    }
    let (projectile_handle, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(projectile_position)
            .ccd_enabled(true)
            .can_sleep(false),
        ColliderBuilder::ball(PROJECTILE_RADIUS)
            .mass(design.projectile_mass)
            .friction(0.65)
            .restitution(0.02),
    );
    let mut sling_joint = Some(
        world.insert_impulse_joint(
            arm_handle,
            projectile_handle,
            RopeJointBuilder::new(design.sling_length)
                .local_anchor1(Vector::new(design.arm_length, 0.0))
                .local_anchor2(Vector::new(0.0, 0.0))
                .contacts_enabled(false),
        ),
    );

    let max_steps = (config.max_time / config.time_step).ceil() as usize;
    let stride = config.record_stride.max(1);
    let mut peak_joint_force = 0.0_f64;
    let mut release_time = None;
    let mut landing_position = 0.0;
    let mut status = SimulationStatus::NoRelease;
    let mut trajectory = Vec::new();
    let mut last_x = projectile_position.x;
    let mut last_y = projectile_position.y;
    let mut apex_height = projectile_position.y;
    let mut last_time = 0.0;
    let replay_geometry = ReplayGeometry {
        arm_handle,
        projectile_handle,
        arm_length: design.arm_length,
        short_arm,
    };

    if config.record_trajectory {
        record_point(&mut trajectory, &world, &replay_geometry, 0.0, false);
    }

    for step in 1..=max_steps {
        let time = step as f64 * config.time_step;
        {
            let arm = &mut world.bodies[arm_handle];
            let angular_velocity = arm.angvel();
            let viscous = design.joint_damping * angular_velocity;
            let coulomb = if angular_velocity.abs() > 1.0e-8 {
                design.pivot_friction * angular_velocity.signum()
            } else {
                0.0
            };
            arm.add_torque(-(viscous + coulomb), true);
        }

        world.step();

        if let Some(joint) = world.impulse_joints.get(pivot_joint) {
            // The first two spatial-impulse components are linear. Excluding
            // the angular component avoids mixing force and torque units.
            let linear_impulse = joint.impulses.truncate().length();
            peak_joint_force = peak_joint_force.max(linear_impulse / config.time_step);
        }

        let arm_angle = world.bodies[arm_handle].rotation().angle();
        if sling_joint.is_some() && arm_angle >= design.release_angle {
            if let Some(handle) = sling_joint.take() {
                world.remove_impulse_joint(handle);
            }
            release_time = Some(time);
            status = SimulationStatus::NoLanding;
        }

        let projectile = &world.bodies[projectile_handle];
        let position = projectile.translation();
        apex_height = apex_height.max(position.y);
        if release_time.is_some()
            && position.y <= PROJECTILE_RADIUS + 0.015
            && projectile.linvel().y <= 0.0
        {
            // Linear interpolation gives a less timestep-sensitive first-impact x.
            let alpha = if last_y > PROJECTILE_RADIUS && last_y != position.y {
                ((last_y - PROJECTILE_RADIUS) / (last_y - position.y)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            landing_position = last_x + alpha * (position.x - last_x);
            status = SimulationStatus::Landed;
            last_time = time;
            if config.record_trajectory {
                record_point(&mut trajectory, &world, &replay_geometry, time, true);
            }
            break;
        }

        if config.record_trajectory && (step.is_multiple_of(stride) || step == max_steps) {
            record_point(
                &mut trajectory,
                &world,
                &replay_geometry,
                time,
                release_time.is_some(),
            );
        }
        last_x = position.x;
        last_y = position.y;
        last_time = time;
    }

    if status != SimulationStatus::Landed {
        landing_position = world.bodies[projectile_handle].translation().x;
    }
    if !landing_position.is_finite() || !input_energy.is_finite() || !peak_joint_force.is_finite() {
        status = SimulationStatus::Invalid;
        landing_position = 0.0;
        peak_joint_force = 0.0;
    }

    let target_error = (landing_position - config.target_position).abs();
    // This is a minimization objective: resource and invalidity terms are
    // positive penalties. Negative terms would incorrectly reward high energy
    // and high structural loads.
    let scalar_score =
        target_error + 0.002 * input_energy + 0.0002 * peak_joint_force + invalid_penalty(status);
    SimulationResult {
        status,
        landing_position,
        target_error,
        input_energy,
        peak_joint_force,
        release_time,
        flight_time: release_time.map_or(0.0, |release| (last_time - release).max(0.0)),
        apex_height,
        scalar_score,
        trajectory,
    }
}

pub fn scalar_objective(x: &[f64], config: &SimulationConfig) -> f64 {
    Design::from_slice(x)
        .map(|design| simulate(&design, config).scalar_score)
        .unwrap_or(1.0e99)
}

pub fn multi_objective(x: &[f64], config: &SimulationConfig) -> Vec<f64> {
    match Design::from_slice(x) {
        Ok(design) => simulate(&design, config).objectives().to_vec(),
        Err(_) => vec![1.0e99; OBJECTIVES],
    }
}

#[derive(Clone, Debug)]
pub struct ScalarOptions {
    pub evaluations_per_retry: u64,
    pub retries: usize,
    pub workers: usize,
    pub depth: i32,
    pub seed: u64,
}

impl Default for ScalarOptions {
    fn default() -> Self {
        Self {
            evaluations_per_retry: 5_000,
            retries: 8,
            workers: 0,
            depth: 6,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScalarOutcome {
    pub design: Design,
    pub simulation: SimulationResult,
    pub evaluations: u64,
    pub completed_retries: usize,
    pub elapsed: Duration,
    pub improvements: Vec<RetryImprovement>,
}

pub fn optimize_scalar(
    simulation_config: &SimulationConfig,
    options: &ScalarOptions,
) -> Result<ScalarOutcome, Box<dyn Error>> {
    if options.evaluations_per_retry == 0 || options.retries == 0 {
        return Err("scalar evaluations and retries must be positive".into());
    }
    if !(1..=36).contains(&options.depth) {
        return Err("BiteOpt depth must lie in 1..=36".into());
    }
    let bounds = RetryBounds::new(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec())?;
    let objective = |x: &[f64]| scalar_objective(x, simulation_config);
    let retry_config = RetryConfig {
        num_retries: options.retries,
        workers: options.workers,
        capacity: options.retries.min(500),
        max_evaluations: options.evaluations_per_retry,
        seed: options.seed,
        statistic_num: 1_000,
        ..Default::default()
    };
    let started = Instant::now();
    let result = retry(&objective, &bounds, &retry_config, |objective, context| {
        let mut rng = Rng::new(context.seed);
        let random_guess: Vec<f64> = context
            .bounds
            .lower()
            .iter()
            .zip(context.bounds.upper())
            .map(|(&lower, &upper)| lower + rng.uniform01() * (upper - lower))
            .collect();
        let guess = context.guess.as_deref().unwrap_or(&random_guess);
        let optimized = optimize_bite(
            objective,
            context.bounds.lower(),
            context.bounds.upper(),
            Some(guess),
            &BiteParams {
                max_evaluations: context.max_evaluations,
                seed: rng.next_u64(),
                runid: context.run_id as i64,
                ..Default::default()
            },
            options.depth,
        );
        RetryRunResult {
            x: optimized.x,
            y: optimized.y,
            evaluations: optimized.evaluations,
        }
    });
    if !result.success {
        return Err("BiteOpt retry returned no finite design".into());
    }
    let design = Design::from_slice(&result.x)?;
    let mut replay_config = simulation_config.clone();
    replay_config.record_trajectory = true;
    let simulation = simulate(&design, &replay_config);
    Ok(ScalarOutcome {
        design,
        simulation,
        evaluations: result.evaluations,
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        improvements: result.improvements,
    })
}

#[derive(Clone, Debug)]
pub struct MultiOptions {
    pub evaluations: usize,
    pub popsize: usize,
    pub workers: usize,
    pub seed: u64,
}

impl Default for MultiOptions {
    fn default() -> Self {
        Self {
            evaluations: 20_000,
            popsize: 128,
            workers: 0,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParetoPoint {
    pub design: Design,
    pub objectives: [f64; OBJECTIVES],
}

#[derive(Clone, Copy, Debug)]
pub struct MoProgress {
    pub evaluations: usize,
    pub elapsed_seconds: f64,
    /// Higher is better: the negative balanced MODE score.
    pub best_quality: f64,
}

#[derive(Clone, Debug)]
pub struct MultiOutcome {
    pub pareto: Vec<ParetoPoint>,
    pub representative: ParetoPoint,
    pub simulation: SimulationResult,
    pub evaluations: usize,
    pub generations: usize,
    pub elapsed: Duration,
    pub convergence: Vec<MoProgress>,
    /// Higher is better. This is the negative balanced score of the selected
    /// representative, not a substitute for inspecting the Pareto front.
    pub quality: f64,
}

fn balanced_objective(values: &[f64; OBJECTIVES], target: f64) -> f64 {
    values[0] / target.max(1.0) + values[1] / 10_000.0 + values[2] / 10_000.0
}

pub fn optimize_multi(
    simulation_config: &SimulationConfig,
    options: &MultiOptions,
) -> Result<MultiOutcome, Box<dyn Error>> {
    if options.evaluations == 0 {
        return Err("MODE evaluations must be positive".into());
    }
    if options.popsize < 4 {
        return Err("MODE population size must be at least four".into());
    }
    if options.popsize > i32::MAX as usize {
        return Err("MODE population size is too large".into());
    }
    let generations = options.evaluations.div_ceil(options.popsize);
    let evaluations = generations * options.popsize;
    let fitness = Fitness::bounded(DIMENSION, OBJECTIVES, &LOWER_BOUNDS, &UPPER_BOUNDS);
    let parameters = ModeParams {
        popsize: options.popsize as i32,
        nsga_update: true,
        seed: options.seed,
        ..Default::default()
    };
    let mut mode = Mode::try_new(fitness, OBJECTIVES, 0, None, &parameters)?;
    let mut convergence = Vec::with_capacity(generations);
    let mut best_balanced = f64::INFINITY;
    let started = Instant::now();
    for generation in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, options.workers as i32, |x| {
            multi_objective(x, simulation_config)
        });
        for values in &ys {
            let values = [values[0], values[1], values[2]];
            best_balanced = best_balanced.min(balanced_objective(
                &values,
                simulation_config.target_position,
            ));
        }
        mode.tell(&ys);
        convergence.push(MoProgress {
            evaluations: (generation + 1) * options.popsize,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            best_quality: -best_balanced,
        });
    }

    let population = mode.population();
    let values = parallel_batch(&population, options.workers as i32, |x| {
        multi_objective(x, simulation_config)
    });
    let indices = pareto_indices(&values, OBJECTIVES)?;
    let mut pareto = Vec::with_capacity(indices.len());
    for index in indices {
        pareto.push(ParetoPoint {
            design: Design::from_slice(&population[index])?,
            objectives: [values[index][0], values[index][1], values[index][2]],
        });
    }
    if pareto.is_empty() {
        return Err("MODE returned an empty Pareto front".into());
    }
    pareto.sort_by(|left, right| {
        balanced_objective(&left.objectives, simulation_config.target_position).total_cmp(
            &balanced_objective(&right.objectives, simulation_config.target_position),
        )
    });
    let representative = pareto[0].clone();
    let quality = -balanced_objective(
        &representative.objectives,
        simulation_config.target_position,
    );
    let mut replay_config = simulation_config.clone();
    replay_config.record_trajectory = true;
    let simulation = simulate(&representative.design, &replay_config);
    Ok(MultiOutcome {
        pareto,
        representative,
        simulation,
        evaluations,
        generations,
        elapsed: started.elapsed(),
        convergence,
        quality,
    })
}

#[derive(Clone, Debug)]
pub struct QdOptions {
    pub evaluations: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    pub workers: usize,
    pub seed: u64,
}

impl Default for QdOptions {
    fn default() -> Self {
        Self {
            evaluations: 20_000,
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
    pub design: Design,
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
    pub simulation: SimulationResult,
    pub evaluations: usize,
    pub occupied: usize,
    pub capacity: usize,
    pub qd_score: f64,
    pub invalid_evaluations: usize,
    pub clipped_descriptors: usize,
    pub elapsed: Duration,
    pub convergence: Vec<QdProgress>,
}

/// MAP-Elites quality and behavior descriptor.
///
/// Mechanically invalid throws return non-finite values and therefore cannot
/// occupy even an otherwise empty archive niche.
pub fn qd_objective(x: &[f64], config: &SimulationConfig) -> (f64, [f64; 2]) {
    let Ok(design) = Design::from_slice(x) else {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    };
    let simulation = simulate(&design, config);
    let Some(release_time) = simulation.release_time else {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    };
    if !simulation.status.is_valid()
        || !simulation.apex_height.is_finite()
        || !release_time.is_finite()
    {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    }
    let quality = balanced_objective(&simulation.objectives(), config.target_position);
    (quality, [simulation.apex_height, release_time])
}

struct TrebuchetQdBatch<'a> {
    config: &'a SimulationConfig,
    workers: usize,
    evaluations: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    clipped: Arc<AtomicUsize>,
}

impl QdBatchFitness for TrebuchetQdBatch<'_> {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let evaluated = parallel_batch(xs, self.workers as i32, |x| qd_objective(x, self.config));
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
    simulation_config: &SimulationConfig,
    options: &QdOptions,
) -> Result<QdOutcome, Box<dyn Error>> {
    if options.evaluations == 0 {
        return Err("QD evaluations must be positive".into());
    }
    if options.chunk_size < 2 || !options.chunk_size.is_multiple_of(2) {
        return Err("QD chunk size must be an even number of at least two".into());
    }
    let side = (options.capacity as f64).sqrt() as usize;
    if side < 2 || side * side != options.capacity {
        return Err("QD capacity must be a perfect square of at least four".into());
    }
    let generations = options.evaluations.div_ceil(options.chunk_size);
    let actual_evaluations = generations * options.chunk_size;
    let mut rng = Rng::new(options.seed);
    let mut archive = Archive::try_new(
        DIMENSION,
        &QD_DESCRIPTOR_LOWER,
        &QD_DESCRIPTOR_UPPER,
        options.capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(&LOWER_BOUNDS, &UPPER_BOUNDS, &mut rng);

    let evaluations = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clipped = Arc::new(AtomicUsize::new(0));
    let mut batch = TrebuchetQdBatch {
        config: simulation_config,
        workers: options.workers,
        evaluations: Arc::clone(&evaluations),
        invalid: Arc::clone(&invalid),
        clipped: Arc::clone(&clipped),
    };
    let parameters = MapElitesParams {
        generations,
        chunk_size: options.chunk_size,
        use_sbx: false,
        ..Default::default()
    };
    let started = Instant::now();
    let mut convergence = Vec::with_capacity(generations);
    map_elites_batch_with_progress(
        &mut archive,
        &mut batch,
        &LOWER_BOUNDS,
        &UPPER_BOUNDS,
        &parameters,
        &mut rng,
        &mut |_, archive| {
            let count = evaluations.load(Ordering::Relaxed);
            let invalid_count = invalid.load(Ordering::Relaxed);
            convergence.push(QdProgress {
                evaluations: count,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                coverage: archive.occupied() as f64 / archive.capacity() as f64,
                qd_score: archive.qd_score(),
                best_quality: archive.best_y(),
                invalid_fraction: invalid_count as f64 / count.max(1) as f64,
            });
        },
    )?;
    debug_assert_eq!(evaluations.load(Ordering::Relaxed), actual_evaluations);

    let mut elites = Vec::with_capacity(archive.occupied());
    for niche_id in 0..archive.capacity() {
        if !archive.ys()[niche_id].is_finite() {
            continue;
        }
        elites.push(QdPoint {
            niche_id,
            grid_x: niche_id % side,
            grid_y: niche_id / side,
            design: Design::from_slice(&archive.xs()[niche_id])?,
            quality: archive.ys()[niche_id],
            descriptors: [
                archive.descriptors()[niche_id][0],
                archive.descriptors()[niche_id][1],
            ],
            visit_count: archive.counts()[niche_id],
        });
    }
    elites.sort_by(|left, right| left.quality.total_cmp(&right.quality));
    let representative = elites
        .first()
        .cloned()
        .ok_or("MAP-Elites did not find a valid landing")?;
    let mut replay_config = simulation_config.clone();
    replay_config.record_trajectory = true;
    let simulation = simulate(&representative.design, &replay_config);
    Ok(QdOutcome {
        elites,
        representative,
        simulation,
        evaluations: actual_evaluations,
        occupied: archive.occupied(),
        capacity: archive.capacity(),
        qd_score: archive.qd_score(),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clipped_descriptors: clipped.load(Ordering::Relaxed),
        elapsed: started.elapsed(),
        convergence,
    })
}

fn effective_workers(requested: usize) -> usize {
    if requested == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        requested
    }
}

/// Write a complete schema-v1 MAP-Elites result directory.
pub fn write_qd_artifacts(
    directory: &Path,
    target: f64,
    initial: &SimulationResult,
    outcome: &QdOutcome,
    options: &QdOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    write_artifacts(directory, target, initial, &outcome.simulation, &[], &[])?;

    let mut archive_csv = String::from(
        "niche_id,grid_x,grid_y,quality_train,descriptor_apex_height_train,descriptor_release_time_train,visit_count",
    );
    for name in [
        "arm_length",
        "counterweight_mass",
        "projectile_mass",
        "sling_length",
        "initial_arm_angle",
        "release_angle",
        "joint_damping",
        "pivot_friction",
    ] {
        let _ = write!(archive_csv, ",decision_{name}");
    }
    archive_csv.push('\n');
    for point in &outcome.elites {
        let _ = write!(
            archive_csv,
            "{},{},{},{},{},{},{}",
            point.niche_id,
            point.grid_x,
            point.grid_y,
            point.quality,
            point.descriptors[0],
            point.descriptors[1],
            point.visit_count,
        );
        for value in point.design.to_vec() {
            let _ = write!(archive_csv, ",{value}");
        }
        archive_csv.push('\n');
    }
    fs::write(directory.join("qd_archive.csv"), archive_csv)?;

    let selections = [
        ("best_quality", outcome.elites.first()),
        (
            "lowest_apex",
            outcome
                .elites
                .iter()
                .min_by(|left, right| left.descriptors[0].total_cmp(&right.descriptors[0])),
        ),
        (
            "highest_apex",
            outcome
                .elites
                .iter()
                .max_by(|left, right| left.descriptors[0].total_cmp(&right.descriptors[0])),
        ),
        (
            "earliest_release",
            outcome
                .elites
                .iter()
                .min_by(|left, right| left.descriptors[1].total_cmp(&right.descriptors[1])),
        ),
        (
            "latest_release",
            outcome
                .elites
                .iter()
                .max_by(|left, right| left.descriptors[1].total_cmp(&right.descriptors[1])),
        ),
    ];
    let mut seen = Vec::new();
    let mut representatives_csv =
        String::from("role,niche_id,quality,apex_height,release_time,landing_position\n");
    for (role, point) in selections {
        let Some(point) = point else {
            continue;
        };
        if seen.contains(&point.niche_id) {
            continue;
        }
        seen.push(point.niche_id);
        let simulation = simulate(
            &point.design,
            &SimulationConfig {
                target_position: target,
                ..Default::default()
            },
        );
        let _ = writeln!(
            representatives_csv,
            "{role},{},{},{},{},{}",
            point.niche_id,
            point.quality,
            point.descriptors[0],
            point.descriptors[1],
            simulation.landing_position,
        );
    }
    fs::write(directory.join("representatives.csv"), representatives_csv)?;

    let mut convergence_csv = String::from(
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction\n",
    );
    for sample in &outcome.convergence {
        let _ = writeln!(
            convergence_csv,
            "{},{},{},{},{},{}",
            sample.evaluations,
            sample.elapsed_seconds,
            sample.coverage,
            sample.qd_score,
            sample.best_quality,
            sample.invalid_fraction,
        );
    }
    fs::write(directory.join("convergence.csv"), convergence_csv)?;

    let side = (outcome.capacity as f64).sqrt() as usize;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "rapier-trebuchet",
        "formulation": "qd",
        "command": command,
        "seed": options.seed,
        "workers": effective_workers(options.workers),
        "requested_evaluations": options.evaluations,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "simulation": {
            "target_position_m": target,
            "deterministic": true,
            "training_seeds": [],
            "validation_seeds": []
        },
        "descriptors": [
            {
                "column": "descriptor_apex_height",
                "label": "Trajectory apex",
                "unit": "m",
                "bounds": QD_DESCRIPTOR_LOWER[0..1]
                    .iter()
                    .chain(QD_DESCRIPTOR_UPPER[0..1].iter())
                    .copied()
                    .collect::<Vec<_>>()
            },
            {
                "column": "descriptor_release_time",
                "label": "Release time",
                "unit": "s",
                "bounds": [QD_DESCRIPTOR_LOWER[1], QD_DESCRIPTOR_UPPER[1]]
            }
        ],
        "qd": {
            "capacity": outcome.capacity,
            "grid_shape": [side, side],
            "chunk_size": options.chunk_size,
            "quality_train_column": "quality_train",
            "quality_label": "Balanced throw quality (lower is better)",
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
            "trajectory": "trajectory.csv",
            "replay": "replay.html"
        }
    });
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

fn push_js_trajectory(output: &mut String, name: &str, points: &[TrajectoryPoint]) {
    let _ = writeln!(output, "const {name} = [");
    for point in points {
        let _ = writeln!(
            output,
            "[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}],",
            point.time,
            point.arm_angle,
            point.counterweight[0],
            point.counterweight[1],
            point.arm_tip[0],
            point.arm_tip[1],
            point.projectile[0],
            point.projectile[1],
            u8::from(point.released),
        );
    }
    output.push_str("];\n");
}

fn write_replay_html(
    path: &Path,
    target: f64,
    initial: &SimulationResult,
    optimized: &SimulationResult,
    convergence: &[MoProgress],
) -> Result<(), Box<dyn Error>> {
    let mut data = String::new();
    push_js_trajectory(&mut data, "initial", &initial.trajectory);
    push_js_trajectory(&mut data, "optimized", &optimized.trajectory);
    let _ = writeln!(data, "const target = {target:.9};");
    data.push_str("const convergence = [");
    for sample in convergence {
        let _ = write!(
            data,
            "[{},{:.12}],",
            sample.evaluations, sample.best_quality
        );
    }
    data.push_str("];\n");
    let html = format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fcmaes + Rapier trebuchet replay</title>
<style>
body {{ margin:0; font:16px system-ui,sans-serif; color:#e8edf2; background:#111820; }}
main {{ max-width:1100px; margin:auto; padding:24px; }}
h1,h2 {{ color:#fff; }} canvas {{ width:100%; height:auto; background:#17232e; border-radius:8px; }}
button,input {{ margin:0 8px 12px 0; }} .key span {{ margin-right:18px; }}
.initial {{ color:#aab5c0; }} .optimized {{ color:#50d890; }} .target {{ color:#ffca58; }}
</style>
</head>
<body><main>
<h1>Trebuchet design replay</h1>
<p class="key"><span class="initial">● initial</span><span class="optimized">● optimized</span><span class="target">│ target</span></p>
<button id="initial">Replay initial</button><button id="optimized">Replay optimized</button>
<button id="pause">Pause</button><input id="frame" type="range" min="0" value="0">
<canvas id="scene" width="1050" height="560"></canvas>
<h2>Convergence history</h2>
<canvas id="plot" width="1050" height="260"></canvas>
<script>
{data}
const canvas=document.getElementById("scene"),ctx=canvas.getContext("2d"),slider=document.getElementById("frame");
let active=optimized,frame=0,playing=true,last=performance.now();
const all=initial.concat(optimized), xs=all.flatMap(p=>[p[2],p[4],p[6],target]), ys=all.flatMap(p=>[p[3],p[5],p[7],0]);
const xmin=Math.min(...xs)-2,xmax=Math.max(...xs)+2,ymax=Math.max(...ys)+1;
const sx=x=>45+(x-xmin)/(xmax-xmin)*(canvas.width-90), sy=y=>canvas.height-35-y/ymax*(canvas.height-70);
function path(points,color,dash){{ctx.save();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash);ctx.beginPath();points.forEach((p,i)=>{{const x=sx(p[6]),y=sy(p[7]);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});ctx.stroke();ctx.restore();}}
function draw(){{
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle="#657482";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(0,sy(0));ctx.lineTo(canvas.width,sy(0));ctx.stroke();
 ctx.strokeStyle="#ffca58";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(sx(target),sy(0));ctx.lineTo(sx(target),sy(2));ctx.stroke();
 path(initial,"#778491",[7,7]);path(optimized,"#50d890",[]);
 const p=active[Math.min(frame,active.length-1)];if(!p)return;
 ctx.strokeStyle="#d9e2ea";ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(sx(p[2]),sy(p[3]));ctx.lineTo(sx(p[4]),sy(p[5]));ctx.stroke();
 if(!p[8]){{ctx.strokeStyle="#c6a2ff";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx(p[4]),sy(p[5]));ctx.lineTo(sx(p[6]),sy(p[7]));ctx.stroke();}}
 ctx.fillStyle="#f17c67";ctx.beginPath();ctx.arc(sx(p[2]),sy(p[3]),10,0,Math.PI*2);ctx.fill();
 ctx.fillStyle="#50d890";ctx.beginPath();ctx.arc(sx(p[6]),sy(p[7]),7,0,Math.PI*2);ctx.fill();
 ctx.fillStyle="#d9e2ea";ctx.beginPath();ctx.arc(sx(0),sy(7),5,0,Math.PI*2);ctx.fill();
 ctx.fillStyle="#fff";ctx.fillText(`t=${{p[0].toFixed(2)}} s  x=${{p[6].toFixed(2)}} m`,15,24);
 slider.max=Math.max(0,active.length-1);slider.value=frame;
}}
function tick(now){{if(playing&&now-last>25){{frame=(frame+1)%active.length;last=now;draw();}}requestAnimationFrame(tick);}}
function select(points){{active=points;frame=0;playing=true;draw();}}
document.getElementById("initial").onclick=()=>select(initial);document.getElementById("optimized").onclick=()=>select(optimized);
document.getElementById("pause").onclick=()=>{{playing=!playing}};slider.oninput=()=>{{frame=Number(slider.value);playing=false;draw();}};
const pc=document.getElementById("plot"),px=pc.getContext("2d");px.clearRect(0,0,pc.width,pc.height);
if(convergence.length>1){{const maxx=convergence.at(-1)[0], vals=convergence.map(p=>p[1]),miny=Math.min(...vals),maxy=Math.max(...vals);
 const X=x=>55+x/maxx*(pc.width-80),Y=y=>25+(maxy-y)/Math.max(1e-12,maxy-miny)*(pc.height-60);
 px.strokeStyle="#50d890";px.lineWidth=2;px.beginPath();convergence.forEach((p,i)=>{{i?px.lineTo(X(p[0]),Y(p[1])):px.moveTo(X(p[0]),Y(p[1]))}});px.stroke();
 px.fillStyle="#d9e2ea";px.fillText("evaluations",pc.width-90,pc.height-10);px.fillText("best quality",8,18);}}
draw();requestAnimationFrame(tick);
</script></main></body></html>"##
    );
    fs::write(path, html)?;
    Ok(())
}

/// Write CSV data and a self-contained HTML replay.
pub fn write_artifacts(
    directory: &Path,
    target: f64,
    initial: &SimulationResult,
    optimized: &SimulationResult,
    convergence: &[MoProgress],
    pareto: &[ParetoPoint],
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;

    let mut trajectory = String::from(
        "design,time,arm_angle,counterweight_x,counterweight_y,arm_tip_x,arm_tip_y,projectile_x,projectile_y,released\n",
    );
    for (label, result) in [("initial", initial), ("optimized", optimized)] {
        for point in &result.trajectory {
            let _ = writeln!(
                trajectory,
                "{label},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{}",
                point.time,
                point.arm_angle,
                point.counterweight[0],
                point.counterweight[1],
                point.arm_tip[0],
                point.arm_tip[1],
                point.projectile[0],
                point.projectile[1],
                point.released,
            );
        }
    }
    fs::write(directory.join("trajectory.csv"), trajectory)?;

    let mut convergence_csv = String::from("evaluations,elapsed_seconds,best_quality\n");
    for sample in convergence {
        let _ = writeln!(
            convergence_csv,
            "{},{:.12},{:.12}",
            sample.evaluations, sample.elapsed_seconds, sample.best_quality
        );
    }
    fs::write(directory.join("convergence.csv"), convergence_csv)?;

    let mut pareto_csv = String::from(
        "point_id,feasible,selected,objective_target_error,objective_input_energy,objective_peak_joint_force,decision_arm_length,decision_counterweight_mass,decision_projectile_mass,decision_sling_length,decision_initial_arm_angle,decision_release_angle,decision_joint_damping,decision_pivot_friction\n",
    );
    for (index, point) in pareto.iter().enumerate() {
        let x = point.design.to_vec();
        let _ = writeln!(
            pareto_csv,
            "{index},1,{},{},{},{},{}",
            u8::from(index == 0),
            point.objectives[0],
            point.objectives[1],
            point.objectives[2],
            x.iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    fs::write(directory.join("pareto.csv"), pareto_csv)?;
    write_replay_html(
        &directory.join("replay.html"),
        target,
        initial,
        optimized,
        convergence,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fast_config() -> SimulationConfig {
        SimulationConfig {
            time_step: 1.0 / 120.0,
            max_time: 4.0,
            ..Default::default()
        }
    }

    #[test]
    fn design_round_trip_and_bounds() {
        let design = Design::default();
        assert_eq!(Design::from_slice(&design.to_vec()).unwrap(), design);
        assert!(Design::from_slice(&[1.0; 7]).is_err());
        assert!(Design::from_slice(&[f64::NAN; DIMENSION]).is_err());
        assert!(Design::from_slice(&[0.0; DIMENSION]).is_err());
    }

    #[test]
    fn simulation_is_finite_and_deterministic() {
        let design = Design::default();
        let first = simulate(&design, &fast_config());
        let second = simulate(&design, &fast_config());
        assert_eq!(first.status, second.status);
        assert_eq!(first.landing_position, second.landing_position);
        assert_eq!(first.peak_joint_force, second.peak_joint_force);
        assert!(first.scalar_score.is_finite());
        assert!(first.input_energy > 0.0);
        assert!(first.peak_joint_force > 0.0);
        assert!(first.apex_height.is_finite());
        assert_eq!(first.apex_height, second.apex_height);
    }

    #[test]
    fn trajectory_recording_contains_geometry() {
        let mut config = fast_config();
        config.record_trajectory = true;
        config.record_stride = 10;
        let result = simulate(&Design::default(), &config);
        assert!(result.trajectory.len() > 2);
        assert_eq!(result.trajectory[0].time, 0.0);
        assert!(
            result
                .trajectory
                .iter()
                .all(|point| point.projectile.iter().all(|value| value.is_finite()))
        );
    }

    #[test]
    fn invalid_config_is_penalized() {
        let result = simulate(
            &Design::default(),
            &SimulationConfig {
                target_position: -1.0,
                ..Default::default()
            },
        );
        assert_eq!(result.status, SimulationStatus::Invalid);
        assert!(result.objectives().iter().all(|value| *value > 1_000.0));
    }

    #[test]
    fn initial_ground_intersection_is_invalid() {
        let design = Design::from_slice(&[6.0, 100.0, 4.0, 6.0, -1.25, 0.0, 1.0, 1.0]).unwrap();
        let result = simulate(&design, &fast_config());
        assert_eq!(result.status, SimulationStatus::Invalid);
        assert!(result.scalar_score > 2_000.0);
    }

    #[test]
    fn objective_adapters_reject_bad_points() {
        let config = fast_config();
        assert_eq!(scalar_objective(&[0.0], &config), 1.0e99);
        assert_eq!(multi_objective(&[0.0], &config), vec![1.0e99; OBJECTIVES]);
        assert!(!qd_objective(&[0.0], &config).0.is_finite());
    }

    #[test]
    fn tiny_parallel_mode_run_returns_a_front() {
        let result = optimize_multi(
            &fast_config(),
            &MultiOptions {
                evaluations: 8,
                popsize: 4,
                workers: 2,
                seed: 7,
            },
        )
        .unwrap();
        assert_eq!(result.evaluations, 8);
        assert!(!result.pareto.is_empty());
        assert!(result.quality.is_finite());
    }

    #[test]
    fn optimization_options_are_validated() {
        assert!(
            optimize_scalar(
                &fast_config(),
                &ScalarOptions {
                    evaluations_per_retry: 0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            optimize_multi(
                &fast_config(),
                &MultiOptions {
                    popsize: 3,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            optimize_qd(
                &fast_config(),
                &QdOptions {
                    capacity: 15,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            optimize_qd(
                &fast_config(),
                &QdOptions {
                    chunk_size: 3,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn tiny_parallel_qd_run_is_consistent() {
        let result = optimize_qd(
            &fast_config(),
            &QdOptions {
                evaluations: 32,
                capacity: 16,
                chunk_size: 8,
                workers: 2,
                seed: 11,
            },
        )
        .unwrap();
        assert_eq!(result.evaluations, 32);
        assert_eq!(result.capacity, 16);
        assert_eq!(result.convergence.len(), 4);
        assert_eq!(result.occupied, result.elites.len());
        assert!(result.representative.quality.is_finite());
        assert!(result.simulation.status.is_valid());
    }
}
