//! Deterministic Rapier quadruped rollouts and fcmaes optimization adapters.
//!
//! Rapier's internal parallel feature is intentionally disabled. Each rollout
//! is serial and isolated; `fcmaes-core` owns parallel candidate evaluation.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    Archive, BiteParams, MapElitesParams, QdBatchFitness, RetryBounds, RetryConfig,
    RetryImprovement, RetryRunResult, Rng, map_elites_batch_with_progress, optimize_bite,
    parallel_batch, retry,
};
use rapier3d_f64::prelude::*;

pub const DIMENSION: usize = 25;
pub const JOINTS: usize = 8;
pub const FEET: usize = 4;
pub const DESCRIPTOR_LOWER: [f64; 2] = [0.0, 0.0];
pub const DESCRIPTOR_UPPER: [f64; 2] = [1.0, 200.0];
pub const INVALID_QUALITY: f64 = f64::INFINITY;

pub const LOWER_BOUNDS: [f64; DIMENSION] = {
    let mut bounds = [0.0; DIMENSION];
    bounds[0] = 0.5;
    let mut joint = 0;
    while joint < JOINTS {
        bounds[1 + 3 * joint] = 0.0;
        bounds[2 + 3 * joint] = -std::f64::consts::PI;
        bounds[3 + 3 * joint] = -0.6;
        joint += 1;
    }
    bounds
};

pub const UPPER_BOUNDS: [f64; DIMENSION] = {
    let mut bounds = [0.0; DIMENSION];
    bounds[0] = 3.0;
    let mut joint = 0;
    while joint < JOINTS {
        bounds[1 + 3 * joint] = 0.8;
        bounds[2 + 3 * joint] = std::f64::consts::PI;
        bounds[3 + 3 * joint] = 0.8;
        joint += 1;
    }
    bounds
};

/// Readable trot-like seed. Optimizers are not restricted to its symmetries.
pub const INITIAL_GAIT: [f64; DIMENSION] = [
    1.5, // frequency
    0.35,
    0.0,
    0.0, // front-left hip
    0.35,
    std::f64::consts::PI,
    0.0, // front-right hip
    0.35,
    std::f64::consts::PI,
    0.0, // rear-left hip
    0.35,
    0.0,
    0.0, // rear-right hip
    0.45,
    std::f64::consts::FRAC_PI_2,
    0.35, // front-left knee
    0.45,
    -std::f64::consts::FRAC_PI_2,
    0.35, // front-right knee
    0.45,
    -std::f64::consts::FRAC_PI_2,
    0.35, // rear-left knee
    0.45,
    std::f64::consts::FRAC_PI_2,
    0.35, // rear-right knee
];

#[derive(Clone, Debug, PartialEq)]
pub struct Gait {
    pub values: [f64; DIMENSION],
}

impl Gait {
    pub fn from_slice(values: &[f64]) -> Result<Self, &'static str> {
        let values: [f64; DIMENSION] = values
            .try_into()
            .map_err(|_| "a gait must contain 25 values")?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err("all gait values must be finite");
        }
        if values
            .iter()
            .zip(LOWER_BOUNDS.iter().zip(UPPER_BOUNDS))
            .any(|(&value, (&lower, upper))| value < lower || value > upper)
        {
            return Err("gait lies outside the supported bounds");
        }
        Ok(Self { values })
    }

    pub fn initial() -> Self {
        Self::from_slice(&INITIAL_GAIT).expect("built-in gait is valid")
    }

    pub fn target(&self, joint: usize, time: f64) -> f64 {
        let amplitude = self.values[1 + 3 * joint];
        let phase = self.values[2 + 3 * joint];
        let offset = self.values[3 + 3 * joint];
        let angle = std::f64::consts::TAU.mul_add(self.values[0] * time, phase);
        offset + amplitude * angle.sin()
    }
}

#[derive(Clone, Debug)]
pub struct RolloutConfig {
    pub duration_s: f64,
    pub settle_s: f64,
    pub time_step_s: f64,
    pub terrain_seed: u64,
    pub record: bool,
    pub record_stride: usize,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        Self {
            duration_s: 4.0,
            settle_s: 1.0,
            time_step_s: 1.0 / 240.0,
            terrain_seed: 17,
            record: false,
            record_stride: 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReplayPoint {
    pub time_s: f64,
    pub torso: [f64; 3],
    pub contacts: [bool; FEET],
}

#[derive(Clone, Debug)]
pub struct Rollout {
    pub feasible: bool,
    pub forward_distance_m: f64,
    pub lateral_drift_m: f64,
    pub mechanical_work_j: f64,
    pub duty_factor: f64,
    pub body_height_std_mm: f64,
    pub minimum_torso_height_m: f64,
    pub terrain_contact_steps: usize,
    pub fall_constraint_m: f64,
    pub drift_constraint_m: f64,
    pub score: f64,
    pub replay: Vec<ReplayPoint>,
}

impl Rollout {
    pub fn descriptors(&self) -> [f64; 2] {
        [self.duty_factor, self.body_height_std_mm]
    }

    pub fn qd_quality(&self) -> f64 {
        if self.feasible {
            // Positive minimized quality: a better/faster gait is smaller.
            10.0 - self.forward_distance_m + 0.002 * self.mechanical_work_j
        } else {
            INVALID_QUALITY
        }
    }
}

struct Robot {
    torso: RigidBodyHandle,
    joints: [ImpulseJointHandle; JOINTS],
    joint_bodies: [(RigidBodyHandle, RigidBodyHandle); JOINTS],
    feet: [ColliderHandle; FEET],
    terrain: Vec<ColliderHandle>,
}

fn terrain_random(state: &mut u64) -> f64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state).wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
}

fn build_world(config: &RolloutConfig) -> (PhysicsWorld, Robot) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -9.81, 0.0);
    world.integration_parameters.dt = config.time_step_s;
    world.integration_parameters.max_ccd_substeps = 1;

    let (_, ground) = world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(2.0, -0.06, 0.0)),
        ColliderBuilder::cuboid(8.0, 0.06, 1.2)
            .friction(1.0)
            .restitution(0.0),
    );
    let mut terrain = vec![ground];
    let mut state = config.terrain_seed.max(1);
    // A contiguous rough strip starts under the initial stance. Low boxes sit
    // on top of the base plane, so even slow candidates encounter variation.
    for index in 0..48 {
        let height = 0.008 + 0.035 * terrain_random(&mut state);
        let width = 0.075 + 0.025 * terrain_random(&mut state);
        let x = -0.65 + index as f64 * 0.16;
        let (_, collider) = world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(x, height * 0.5, 0.0)),
            ColliderBuilder::cuboid(width, height * 0.5, 0.65)
                .friction(1.05)
                .restitution(0.0),
        );
        terrain.push(collider);
    }

    let (torso, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.72, 0.0))
            .can_sleep(false)
            .ccd_enabled(true),
        ColliderBuilder::cuboid(0.34, 0.08, 0.18)
            .mass(8.0)
            .friction(0.8),
    );

    let mut upper_handles = Vec::new();
    let mut lower_handles = Vec::new();
    let mut foot_handles = Vec::new();
    let mut hip_joints = Vec::new();
    let mut knee_joints = Vec::new();
    let x_locations = [0.25, 0.25, -0.25, -0.25];
    let z_locations = [0.17, -0.17, 0.17, -0.17];
    for leg in 0..FEET {
        let x = x_locations[leg];
        let z = z_locations[leg];
        let (upper, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(x, 0.49, z))
                .can_sleep(false)
                .ccd_enabled(true),
            ColliderBuilder::cuboid(0.045, 0.15, 0.045)
                .mass(0.7)
                .friction(0.8),
        );
        let (lower, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(x, 0.19, z))
                .can_sleep(false)
                .ccd_enabled(true),
            ColliderBuilder::cuboid(0.04, 0.15, 0.04)
                .mass(0.5)
                .friction(0.85),
        );
        let foot = world.insert_collider(
            ColliderBuilder::ball(0.06)
                .translation(Vector::new(0.0, -0.17, 0.0))
                .mass(0.08)
                .friction(1.2)
                .restitution(0.0),
            Some(lower),
        );
        let hip = world.insert_impulse_joint(
            torso,
            upper,
            RevoluteJointBuilder::new(Vector::Z)
                .local_anchor1(Vector::new(x, -0.08, z))
                .local_anchor2(Vector::new(0.0, 0.15, 0.0))
                .limits([-0.9, 0.9])
                .motor_position(0.0, 45.0, 4.0)
                .motor_max_force(30.0)
                .contacts_enabled(false),
        );
        let knee = world.insert_impulse_joint(
            upper,
            lower,
            RevoluteJointBuilder::new(Vector::Z)
                .local_anchor1(Vector::new(0.0, -0.15, 0.0))
                .local_anchor2(Vector::new(0.0, 0.15, 0.0))
                .limits([-0.25, 1.35])
                .motor_position(0.35, 35.0, 3.5)
                .motor_max_force(22.0)
                .contacts_enabled(false),
        );
        upper_handles.push(upper);
        lower_handles.push(lower);
        foot_handles.push(foot);
        hip_joints.push(hip);
        knee_joints.push(knee);
    }
    let joints: [ImpulseJointHandle; JOINTS] = [
        hip_joints[0],
        hip_joints[1],
        hip_joints[2],
        hip_joints[3],
        knee_joints[0],
        knee_joints[1],
        knee_joints[2],
        knee_joints[3],
    ];
    let joint_bodies = [
        (torso, upper_handles[0]),
        (torso, upper_handles[1]),
        (torso, upper_handles[2]),
        (torso, upper_handles[3]),
        (upper_handles[0], lower_handles[0]),
        (upper_handles[1], lower_handles[1]),
        (upper_handles[2], lower_handles[2]),
        (upper_handles[3], lower_handles[3]),
    ];
    (
        world,
        Robot {
            torso,
            joints,
            joint_bodies,
            feet: foot_handles.try_into().expect("four feet"),
            terrain,
        },
    )
}

fn foot_contacts(world: &PhysicsWorld, robot: &Robot) -> [bool; FEET] {
    std::array::from_fn(|foot| {
        world
            .narrow_phase
            .contact_pairs_with(robot.feet[foot])
            .any(|pair| {
                pair.has_any_active_contact()
                    && robot.terrain.iter().any(|terrain| {
                        (pair.collider1 == robot.feet[foot] && pair.collider2 == *terrain)
                            || (pair.collider2 == robot.feet[foot] && pair.collider1 == *terrain)
                    })
            })
    })
}

/// Run one deterministic 9-body, 8-motor quadruped rollout.
pub fn rollout(gait: &Gait, config: &RolloutConfig) -> Rollout {
    if !config.duration_s.is_finite()
        || !config.settle_s.is_finite()
        || !config.time_step_s.is_finite()
        || config.duration_s <= config.settle_s
        || config.settle_s < 0.0
        || config.time_step_s <= 0.0
    {
        return invalid_rollout();
    }
    let (mut world, robot) = build_world(config);
    let steps = (config.duration_s / config.time_step_s).ceil() as usize;
    let settle_step = (config.settle_s / config.time_step_s).ceil() as usize;
    let start_position = world.bodies[robot.torso].translation();
    let mut metric_start = start_position;
    let mut contact_samples = [0_usize; FEET];
    let mut metric_samples = 0_usize;
    let mut height_mean = 0.0;
    let mut height_m2 = 0.0;
    let mut mechanical_work_j = 0.0;
    let mut minimum_torso_height_m = start_position.y;
    let mut terrain_contact_steps = 0;
    let mut replay = Vec::new();

    for step in 0..steps {
        let time = step as f64 * config.time_step_s;
        for joint_index in 0..JOINTS {
            if let Some(joint) = world
                .impulse_joints
                .get_mut(robot.joints[joint_index], true)
            {
                joint.data.set_motor_position(
                    JointAxis::AngX,
                    gait.target(joint_index, time),
                    if joint_index < 4 { 45.0 } else { 35.0 },
                    if joint_index < 4 { 4.0 } else { 3.5 },
                );
                joint.data.set_motor_max_force(
                    JointAxis::AngX,
                    if joint_index < 4 { 30.0 } else { 22.0 },
                );
            }
        }
        world.step();
        let contacts = foot_contacts(&world, &robot);
        let torso = world.bodies[robot.torso].translation();
        minimum_torso_height_m = minimum_torso_height_m.min(torso.y);
        if step == settle_step {
            metric_start = torso;
        }
        if step >= settle_step {
            metric_samples += 1;
            for (count, contact) in contact_samples.iter_mut().zip(contacts) {
                *count += usize::from(contact);
            }
            if contacts.iter().any(|contact| *contact) {
                terrain_contact_steps += 1;
            }
            let delta = torso.y - height_mean;
            height_mean += delta / metric_samples as f64;
            height_m2 += delta * (torso.y - height_mean);
            for joint_index in 0..JOINTS {
                if let Some(joint) = world.impulse_joints.get(robot.joints[joint_index])
                    && let Some(motor) = joint.data.motor(JointAxis::AngX)
                {
                    let (parent, child) = robot.joint_bodies[joint_index];
                    let relative_speed = (world.bodies[child].angvel()
                        - world.bodies[parent].angvel())
                    .dot(Vector::Z);
                    // Solver impulse [N m s] times angular speed [rad/s]
                    // is actual motor work for this integration step [J].
                    mechanical_work_j += (motor.impulse * relative_speed).abs();
                }
            }
        }
        if config.record && (step.is_multiple_of(config.record_stride.max(1)) || step + 1 == steps)
        {
            replay.push(ReplayPoint {
                time_s: time + config.time_step_s,
                torso: [torso.x, torso.y, torso.z],
                contacts,
            });
        }
    }
    let end = world.bodies[robot.torso].translation();
    let forward_distance_m = end.x - metric_start.x;
    let lateral_drift_m = (end.z - metric_start.z).abs();
    let duty_factor =
        contact_samples.iter().sum::<usize>() as f64 / (metric_samples.max(1) * FEET) as f64;
    let body_height_std_mm =
        (height_m2 / metric_samples.saturating_sub(1).max(1) as f64).sqrt() * 1_000.0;
    let fall_constraint_m = 0.15 - minimum_torso_height_m;
    let drift_constraint_m = lateral_drift_m - 0.5;
    let finite = [
        forward_distance_m,
        lateral_drift_m,
        mechanical_work_j,
        duty_factor,
        body_height_std_mm,
        minimum_torso_height_m,
    ]
    .iter()
    .all(|value| value.is_finite());
    let feasible = finite && fall_constraint_m <= 0.0 && drift_constraint_m <= 0.0;
    let score = if feasible {
        -forward_distance_m + 0.002 * mechanical_work_j
    } else {
        1_000.0 + 1_000.0 * fall_constraint_m.max(0.0) + 100.0 * drift_constraint_m.max(0.0)
    };
    Rollout {
        feasible,
        forward_distance_m,
        lateral_drift_m,
        mechanical_work_j,
        duty_factor,
        body_height_std_mm,
        minimum_torso_height_m,
        terrain_contact_steps,
        fall_constraint_m,
        drift_constraint_m,
        score,
        replay,
    }
}

fn invalid_rollout() -> Rollout {
    Rollout {
        feasible: false,
        forward_distance_m: 0.0,
        lateral_drift_m: 0.0,
        mechanical_work_j: 0.0,
        duty_factor: 0.0,
        body_height_std_mm: 0.0,
        minimum_torso_height_m: 0.0,
        terrain_contact_steps: 0,
        fall_constraint_m: 0.15,
        drift_constraint_m: -0.5,
        score: 1_150.0,
        replay: Vec::new(),
    }
}

#[derive(Clone, Debug)]
pub struct RangeRow {
    pub sample: usize,
    pub gait: Gait,
    pub rollout: Rollout,
}

pub fn range_study(
    samples: usize,
    workers: i32,
    seed: u64,
    config: &RolloutConfig,
) -> Vec<RangeRow> {
    let mut rng = Rng::new(seed);
    let candidates = (0..samples)
        .map(|_| {
            LOWER_BOUNDS
                .iter()
                .zip(UPPER_BOUNDS)
                .map(|(&lower, upper)| lower + rng.uniform01() * (upper - lower))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    parallel_batch(&candidates, workers, |values| {
        let gait = Gait::from_slice(values).expect("generated gait is in bounds");
        let result = rollout(&gait, config);
        (gait, result)
    })
    .into_iter()
    .enumerate()
    .map(|(sample, (gait, rollout))| RangeRow {
        sample,
        gait,
        rollout,
    })
    .collect()
}

#[derive(Clone, Debug)]
pub struct ScalarConfig {
    pub evaluations: u64,
    pub retries: usize,
    pub workers: usize,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct ScalarResult {
    pub gait: Gait,
    pub rollout: Rollout,
    pub requested_evaluations: u64,
    pub actual_evaluations: u64,
    pub elapsed: Duration,
    pub improvements: Vec<RetryImprovement>,
}

pub fn optimize_scalar(
    config: &ScalarConfig,
    rollout_config: &RolloutConfig,
) -> Result<ScalarResult, Box<dyn Error>> {
    if config.evaluations == 0 || config.retries == 0 {
        return Err("scalar evaluations and retries must be positive".into());
    }
    let bounds = RetryBounds::new(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec())?;
    let objective = |x: &[f64]| {
        Gait::from_slice(x)
            .map(|gait| rollout(&gait, rollout_config).score)
            .unwrap_or(1.0e12)
    };
    let per_retry = config.evaluations.div_ceil(config.retries as u64);
    let started = Instant::now();
    let result = retry(
        &objective,
        &bounds,
        &RetryConfig {
            num_retries: config.retries,
            workers: config.workers,
            capacity: config.retries,
            max_evaluations: per_retry,
            seed: config.seed,
            statistic_num: 100,
            ..Default::default()
        },
        |objective, context| {
            let mut rng = Rng::new(context.seed);
            let guess = context
                .bounds
                .lower()
                .iter()
                .zip(context.bounds.upper())
                .map(|(&lower, &upper)| lower + rng.uniform01() * (upper - lower))
                .collect::<Vec<_>>();
            let optimized = optimize_bite(
                objective,
                context.bounds.lower(),
                context.bounds.upper(),
                Some(&guess),
                &BiteParams {
                    max_evaluations: context.max_evaluations,
                    seed: context.seed,
                    ..Default::default()
                },
                3,
            );
            RetryRunResult {
                x: optimized.x,
                y: optimized.y,
                evaluations: optimized.evaluations,
            }
        },
    );
    if !result.success {
        return Err("BiteOpt retry returned no gait".into());
    }
    let gait = Gait::from_slice(&result.x)?;
    let mut replay_config = rollout_config.clone();
    replay_config.record = true;
    let simulation = rollout(&gait, &replay_config);
    Ok(ScalarResult {
        gait,
        rollout: simulation,
        requested_evaluations: config.evaluations,
        actual_evaluations: result.evaluations,
        elapsed: started.elapsed(),
        improvements: result.improvements,
    })
}

#[derive(Clone, Debug)]
pub struct QdConfig {
    pub evaluations: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    pub workers: i32,
    pub seed: u64,
    pub holdout_seeds: Vec<u64>,
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
pub struct QdElite {
    pub niche_id: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub gait: Gait,
    pub train: Rollout,
    pub validation: Rollout,
    pub validation_feasible_fraction: f64,
    pub quality: f64,
    pub visit_count: u64,
}

#[derive(Clone, Debug)]
pub struct QdResult {
    pub requested_evaluations: usize,
    pub actual_evaluations: usize,
    pub capacity: usize,
    pub occupied: usize,
    pub invalid_evaluations: usize,
    pub rejected_out_of_bounds: usize,
    pub elapsed: Duration,
    pub qd_score: f64,
    pub elites: Vec<QdElite>,
    pub progress: Vec<QdProgress>,
}

struct GaitBatch<'a> {
    config: &'a RolloutConfig,
    workers: i32,
    evaluations: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    rejected_out_of_bounds: Arc<AtomicUsize>,
}

impl QdBatchFitness for GaitBatch<'_> {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        let mut rows = parallel_batch(xs, self.workers, |x| {
            let Ok(gait) = Gait::from_slice(x) else {
                return (INVALID_QUALITY, vec![f64::INFINITY; 2]);
            };
            let result = rollout(&gait, self.config);
            (result.qd_quality(), result.descriptors().to_vec())
        });
        self.evaluations.fetch_add(rows.len(), Ordering::Relaxed);
        for (quality, descriptors) in &mut rows {
            if !quality.is_finite() || descriptors.iter().any(|value| !value.is_finite()) {
                self.invalid.fetch_add(1, Ordering::Relaxed);
            } else if descriptors
                .iter()
                .zip(DESCRIPTOR_LOWER.iter().zip(DESCRIPTOR_UPPER))
                .any(|(&value, (&lower, upper))| value < lower || value > upper)
            {
                self.rejected_out_of_bounds.fetch_add(1, Ordering::Relaxed);
                *quality = INVALID_QUALITY;
                descriptors.fill(f64::INFINITY);
            }
        }
        rows
    }
}

fn mean_holdout(gait: &Gait, base: &RolloutConfig, seeds: &[u64]) -> (Rollout, f64) {
    if seeds.is_empty() {
        return (rollout(gait, base), 1.0);
    }
    let rows = seeds
        .iter()
        .map(|seed| {
            let mut config = base.clone();
            config.terrain_seed = *seed;
            rollout(gait, &config)
        })
        .collect::<Vec<_>>();
    let count = rows.len() as f64;
    let feasible_fraction = rows.iter().filter(|row| row.feasible).count() as f64 / count;
    let mean = |extract: fn(&Rollout) -> f64| rows.iter().map(extract).sum::<f64>() / count;
    (
        Rollout {
            feasible: feasible_fraction == 1.0,
            forward_distance_m: mean(|row| row.forward_distance_m),
            lateral_drift_m: mean(|row| row.lateral_drift_m),
            mechanical_work_j: mean(|row| row.mechanical_work_j),
            duty_factor: mean(|row| row.duty_factor),
            body_height_std_mm: mean(|row| row.body_height_std_mm),
            minimum_torso_height_m: rows
                .iter()
                .map(|row| row.minimum_torso_height_m)
                .fold(f64::INFINITY, f64::min),
            terrain_contact_steps: rows
                .iter()
                .map(|row| row.terrain_contact_steps)
                .sum::<usize>()
                / rows.len(),
            fall_constraint_m: rows
                .iter()
                .map(|row| row.fall_constraint_m)
                .fold(f64::NEG_INFINITY, f64::max),
            drift_constraint_m: rows
                .iter()
                .map(|row| row.drift_constraint_m)
                .fold(f64::NEG_INFINITY, f64::max),
            score: mean(|row| row.score),
            replay: Vec::new(),
        },
        feasible_fraction,
    )
}

pub fn optimize_qd(
    config: &QdConfig,
    rollout_config: &RolloutConfig,
) -> Result<QdResult, Box<dyn Error>> {
    if config.evaluations == 0 || config.chunk_size < 2 || !config.chunk_size.is_multiple_of(2) {
        return Err("invalid QD evaluations or chunk size".into());
    }
    let side = (config.capacity as f64).sqrt() as usize;
    if side < 2 || side * side != config.capacity {
        return Err("QD capacity must be a perfect square".into());
    }
    let generations = config.evaluations.div_ceil(config.chunk_size);
    let actual_evaluations = generations * config.chunk_size;
    let mut rng = Rng::new(config.seed);
    let mut archive = Archive::try_new(
        DIMENSION,
        &DESCRIPTOR_LOWER,
        &DESCRIPTOR_UPPER,
        config.capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(&LOWER_BOUNDS, &UPPER_BOUNDS, &mut rng);
    let evaluations = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let rejected_out_of_bounds = Arc::new(AtomicUsize::new(0));
    let mut batch = GaitBatch {
        config: rollout_config,
        workers: config.workers,
        evaluations: Arc::clone(&evaluations),
        invalid: Arc::clone(&invalid),
        rejected_out_of_bounds: Arc::clone(&rejected_out_of_bounds),
    };
    let started = Instant::now();
    let mut progress = Vec::new();
    map_elites_batch_with_progress(
        &mut archive,
        &mut batch,
        &LOWER_BOUNDS,
        &UPPER_BOUNDS,
        &MapElitesParams {
            generations,
            chunk_size: config.chunk_size,
            use_sbx: false,
            ..Default::default()
        },
        &mut rng,
        &mut |_, archive| {
            let completed = evaluations.load(Ordering::Relaxed);
            progress.push(QdProgress {
                evaluations: completed,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                coverage: archive.occupied() as f64 / archive.capacity() as f64,
                qd_score: archive.qd_score(),
                best_quality: archive.best_y(),
                invalid_fraction: invalid.load(Ordering::Relaxed) as f64 / completed.max(1) as f64,
            });
        },
    )?;
    let mut elites = Vec::new();
    for niche_id in 0..archive.capacity() {
        if !archive.ys()[niche_id].is_finite() {
            continue;
        }
        let gait = Gait::from_slice(&archive.xs()[niche_id])?;
        let train = rollout(&gait, rollout_config);
        let (validation, validation_feasible_fraction) =
            mean_holdout(&gait, rollout_config, &config.holdout_seeds);
        elites.push(QdElite {
            niche_id,
            grid_x: niche_id % side,
            grid_y: niche_id / side,
            gait,
            train,
            validation,
            validation_feasible_fraction,
            quality: archive.ys()[niche_id],
            visit_count: archive.counts()[niche_id],
        });
    }
    elites.sort_by(|left, right| left.quality.total_cmp(&right.quality));
    Ok(QdResult {
        requested_evaluations: config.evaluations,
        actual_evaluations,
        capacity: archive.capacity(),
        occupied: archive.occupied(),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        rejected_out_of_bounds: rejected_out_of_bounds.load(Ordering::Relaxed),
        elapsed: started.elapsed(),
        qd_score: archive.qd_score(),
        elites,
        progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fast_config() -> RolloutConfig {
        RolloutConfig {
            duration_s: 0.5,
            settle_s: 0.1,
            time_step_s: 1.0 / 120.0,
            terrain_seed: 17,
            record: false,
            record_stride: 2,
        }
    }

    #[test]
    fn gait_decode_and_targets_are_finite() {
        let gait = Gait::initial();
        assert_eq!(gait.values.len(), DIMENSION);
        assert!((0..JOINTS).all(|joint| gait.target(joint, 0.4).is_finite()));
        assert!(Gait::from_slice(&[0.0]).is_err());
    }

    #[test]
    fn rollout_is_bit_replayable() {
        let gait = Gait::initial();
        let first = rollout(&gait, &fast_config());
        let second = rollout(&gait, &fast_config());
        assert_eq!(first.forward_distance_m, second.forward_distance_m);
        assert_eq!(first.mechanical_work_j, second.mechanical_work_j);
        assert_eq!(first.duty_factor, second.duty_factor);
    }

    #[test]
    fn solver_motor_work_and_contacts_are_measured() {
        let first = rollout(&Gait::initial(), &fast_config());
        assert!(first.mechanical_work_j > 0.0);
        assert!(first.terrain_contact_steps > 0);
        assert!((0.0..=1.0).contains(&first.duty_factor));
    }

    #[test]
    fn recording_contains_foot_contacts_and_torso_states() {
        let mut config = fast_config();
        config.record = true;
        let result = rollout(&Gait::initial(), &config);
        assert!(result.replay.len() > 2);
        assert!(
            result
                .replay
                .iter()
                .all(|row| row.torso.iter().all(|value| value.is_finite()))
        );
    }

    #[test]
    fn random_range_study_reaches_the_terrain() {
        let rows = range_study(8, 2, 9, &fast_config());
        assert_eq!(rows.len(), 8);
        assert!(rows.iter().any(|row| row.rollout.terrain_contact_steps > 0));
    }

    #[test]
    fn invalid_configuration_is_explicitly_infeasible() {
        let result = rollout(
            &Gait::initial(),
            &RolloutConfig {
                duration_s: 1.0,
                settle_s: 1.0,
                ..Default::default()
            },
        );
        assert!(!result.feasible);
        assert!(!result.qd_quality().is_finite());
    }
}
