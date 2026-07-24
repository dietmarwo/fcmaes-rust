//! Self-contained cart-pole policy-search tutorial.
//!
//! The simulator and fixed-topology neural controller deliberately have no
//! dependency beyond `std`. Optimizer populations are evaluated in parallel
//! with one explicitly sized Rayon pool.

use fcmaes_core::{
    BiteOpt, BiteParams, Cmaes, CmaesParams, Crfmnes, CrfmnesParams, Fitness, Pgpe, PgpeParams,
};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::fmt::{Display, Formatter};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

pub const INPUTS: usize = 5;
pub const HIDDEN: usize = 16;
pub const PARAMS: usize = HIDDEN * (INPUTS + 1) + HIDDEN + 1 + INPUTS;
pub const LOWER_WEIGHT: f64 = -3.0;
pub const UPPER_WEIGHT: f64 = 3.0;

const GRAVITY: f64 = 9.81;
const FORCE_LIMIT: f64 = 10.0;
const CART_LIMIT: f64 = 2.4;
const DT: f64 = 0.02;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Algorithm {
    Pgpe,
    Crfmnes,
    Cmaes,
    Biteopt,
}

impl Algorithm {
    pub const ALL: [Self; 4] = [Self::Pgpe, Self::Crfmnes, Self::Cmaes, Self::Biteopt];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pgpe => "pgpe",
            Self::Crfmnes => "crfmnes",
            Self::Cmaes => "cmaes",
            Self::Biteopt => "biteopt",
        }
    }
}

impl Display for Algorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Algorithm {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "pgpe" => Ok(Self::Pgpe),
            "crfmnes" | "cr-fm-nes" => Ok(Self::Crfmnes),
            "cmaes" | "cma-es" => Ok(Self::Cmaes),
            "biteopt" | "bite" => Ok(Self::Biteopt),
            _ => Err(format!("unknown algorithm '{value}'")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenarioMode {
    Fixed,
    Rotating,
}

impl ScenarioMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Rotating => "rotating",
        }
    }
}

impl Display for ScenarioMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ScenarioMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "fixed" => Ok(Self::Fixed),
            "rotating" | "rotate" => Ok(Self::Rotating),
            _ => Err(format!("unknown scenario mode '{value}'")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub evaluations: u64,
    pub popsize: usize,
    pub workers: usize,
    pub train_scenarios: usize,
    pub validation_scenarios: usize,
    pub horizon: usize,
    pub scenario_mode: ScenarioMode,
    pub monitor_interval: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            evaluations: 20_480,
            popsize: 64,
            workers: 16,
            train_scenarios: 4,
            validation_scenarios: 128,
            horizon: 300,
            scenario_mode: ScenarioMode::Fixed,
            monitor_interval: 2_048,
        }
    }
}

impl SearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.evaluations == 0 {
            return Err("evaluations must be positive".into());
        }
        if self.popsize < 4 || !self.popsize.is_multiple_of(2) {
            return Err("popsize must be an even integer of at least four".into());
        }
        if !self.evaluations.is_multiple_of(self.popsize as u64) {
            return Err("evaluations must be an exact multiple of popsize".into());
        }
        if self.workers == 0 {
            return Err("workers must be positive".into());
        }
        if self.train_scenarios == 0 || self.validation_scenarios == 0 {
            return Err("training and validation scenario counts must be positive".into());
        }
        if self.horizon < 50 {
            return Err("horizon must be at least 50 steps".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct State {
    pub position: f64,
    pub velocity: f64,
    pub angle: f64,
    pub angular_velocity: f64,
}

#[derive(Clone, Copy, Debug)]
struct Plant {
    cart_mass: f64,
    pole_mass: f64,
    half_length: f64,
    friction: f64,
    motor_scale: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct EpisodeMetrics {
    pub loss: f64,
    pub steps: usize,
    pub success: bool,
    pub rms_force: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ValidationMetrics {
    pub score: f64,
    pub mean_loss: f64,
    pub cvar_loss: f64,
    pub success_rate: f64,
    pub mean_steps: f64,
    pub rms_force: f64,
}

#[derive(Clone, Debug)]
pub struct ConvergencePoint {
    pub evaluations: u64,
    pub train_best: f64,
    pub monitor_score: f64,
}

#[derive(Clone, Debug)]
pub struct RunRecord {
    pub experiment: String,
    pub scenario_mode: ScenarioMode,
    pub algorithm: Algorithm,
    pub seed: u64,
    pub workers: usize,
    pub popsize: usize,
    pub evaluations: u64,
    pub train_scenarios: usize,
    pub validation_scenarios: usize,
    pub horizon: usize,
    pub train_best: f64,
    pub validation: ValidationMetrics,
    pub wall_seconds: f64,
    pub policy: Vec<f64>,
    pub convergence: Vec<ConvergencePoint>,
}

#[derive(Clone, Debug)]
pub struct BaselineRecord {
    pub name: &'static str,
    pub validation: ValidationMetrics,
}

#[derive(Clone, Copy, Debug)]
pub struct Summary {
    pub mean: f64,
    pub sdev: f64,
}

#[derive(Clone, Debug)]
struct FastRng {
    state: u64,
}

impl FastRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn signed(&mut self) -> f64 {
        2.0 * self.uniform01() - 1.0
    }

    fn range(&mut self, lower: f64, upper: f64) -> f64 {
        lower + (upper - lower) * self.uniform01()
    }
}

fn mix_seed(root: u64, stream: u64, index: u64) -> u64 {
    let mut rng = FastRng::new(
        root ^ stream.wrapping_mul(0xd2b7_4407_b1ce_6e93)
            ^ index.wrapping_mul(0xca5a_8263_9512_1157),
    );
    rng.next_u64()
}

fn scenario_seeds(root: u64, generation: u64, count: usize, mode: ScenarioMode) -> Vec<u64> {
    let stream = match mode {
        ScenarioMode::Fixed => 0,
        ScenarioMode::Rotating => generation + 1,
    };
    (0..count)
        .map(|index| mix_seed(root, stream, index as u64))
        .collect()
}

fn validation_seeds(root: u64, count: usize) -> Vec<u64> {
    (0..count)
        .map(|index| mix_seed(root ^ 0xa076_1d64_78bd_642f, 1_000_003, index as u64))
        .collect()
}

fn plant_and_state(seed: u64) -> (Plant, State, FastRng) {
    let mut rng = FastRng::new(seed);
    let plant = Plant {
        cart_mass: rng.range(0.82, 1.18),
        pole_mass: rng.range(0.075, 0.14),
        half_length: rng.range(0.42, 0.62),
        friction: rng.range(0.0, 0.075),
        motor_scale: rng.range(0.88, 1.12),
    };
    let state = State {
        position: rng.range(-0.18, 0.18),
        velocity: rng.range(-0.12, 0.12),
        angle: std::f64::consts::PI + rng.range(-0.2, 0.2),
        angular_velocity: rng.range(-0.25, 0.25),
    };
    (plant, state, rng)
}

fn normalized_inputs(state: State, rng: &mut FastRng) -> [f64; INPUTS] {
    let mut values = [
        state.position / CART_LIMIT,
        state.velocity / 3.0,
        state.angle.sin(),
        state.angle.cos(),
        state.angular_velocity / 6.0,
    ];
    for value in &mut values {
        *value = (*value + 0.012 * rng.signed()).clamp(-2.0, 2.0);
    }
    values
}

pub fn neural_action(params: &[f64], inputs: [f64; INPUTS]) -> f64 {
    assert_eq!(params.len(), PARAMS);
    let mut index = 0;
    let mut hidden = [0.0; HIDDEN];
    for activation in &mut hidden {
        let mut sum = 0.0;
        for input in inputs {
            sum += params[index] * input;
            index += 1;
        }
        sum += params[index];
        index += 1;
        *activation = sum.clamp(0.0, 4.0);
    }
    let mut raw = 0.0;
    for activation in hidden {
        raw += params[index] * activation;
        index += 1;
    }
    raw += params[index];
    index += 1;
    for input in inputs {
        raw += params[index] * input;
        index += 1;
    }
    debug_assert_eq!(index, PARAMS);
    raw.tanh()
}

fn hand_action(inputs: [f64; INPUTS]) -> f64 {
    if inputs[0].abs() > 0.55 {
        return (-3.0 * inputs[0] - 0.6 * inputs[1]).tanh();
    }
    let angle = inputs[2].atan2(inputs[3]);
    let angular_velocity = 6.0 * inputs[4];
    let raw = if angle.abs() < 0.45 {
        -0.45 * inputs[0] - 0.4 * inputs[1] + 1.8 * angle + 0.55 * angular_velocity
    } else {
        let energy_error = 0.5 * angular_velocity.powi(2) + GRAVITY * (angle.cos() - 1.0);
        -0.045 * energy_error * (angular_velocity * angle.cos()).signum()
            - 1.4 * inputs[0]
            - 0.35 * inputs[1]
    };
    raw.tanh()
}

fn step(state: &mut State, plant: Plant, action: f64, wind: f64) {
    let force = FORCE_LIMIT * plant.motor_scale * action + wind - plant.friction * state.velocity;
    let sin_theta = state.angle.sin();
    let cos_theta = state.angle.cos();
    let total_mass = plant.cart_mass + plant.pole_mass;
    let pole_mass_length = plant.pole_mass * plant.half_length;
    let temp = (force + pole_mass_length * state.angular_velocity.powi(2) * sin_theta) / total_mass;
    let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
        / (plant.half_length * (4.0 / 3.0 - plant.pole_mass * cos_theta.powi(2) / total_mass));
    let x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;

    state.position += DT * state.velocity;
    state.velocity += DT * x_acc;
    state.angle += DT * state.angular_velocity;
    state.angle = (state.angle + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    state.angular_velocity += DT * theta_acc;
}

fn simulate_with<F>(
    seed: u64,
    horizon: usize,
    mut controller: F,
    trajectory: Option<&mut Vec<(usize, State, f64)>>,
) -> EpisodeMetrics
where
    F: FnMut([f64; INPUTS]) -> f64,
{
    let (plant, mut state, mut rng) = plant_and_state(seed);
    let mut state_cost = 0.0;
    let mut force_sq = 0.0;
    let mut steps = 0;
    let mut upright_tail = 0;
    let mut trajectory = trajectory;

    for time in 0..horizon {
        let inputs = normalized_inputs(state, &mut rng);
        let action = controller(inputs).clamp(-1.0, 1.0);
        let wind = 0.45 * rng.signed();
        if let Some(rows) = trajectory.as_deref_mut() {
            rows.push((time, state, FORCE_LIMIT * action));
        }
        step(&mut state, plant, action, wind);
        steps = time + 1;
        let angle_cost = 0.5 * (1.0 - state.angle.cos());
        let position_n = state.position / CART_LIMIT;
        state_cost += 0.65 * angle_cost
            + 0.04 * position_n.powi(2)
            + 0.01 * (state.velocity / 3.0).powi(2)
            + 0.006 * (state.angular_velocity / 6.0).powi(2)
            + 0.002 * action.powi(2);
        force_sq += (FORCE_LIMIT * action).powi(2);
        if time >= 3 * horizon / 4 && state.angle.abs() < 0.25 && state.position.abs() < 1.8 {
            upright_tail += 1;
        }

        if !state.position.is_finite()
            || !state.angle.is_finite()
            || state.position.abs() > CART_LIMIT
        {
            break;
        }
    }

    let survival = steps as f64 / horizon as f64;
    let tail_steps = (horizon - 3 * horizon / 4).max(1);
    let upright_fraction = upright_tail as f64 / tail_steps as f64;
    let success = steps == horizon && upright_fraction >= 0.8;
    let running = state_cost / steps.max(1) as f64;
    let loss = 3.0 * (1.0 - survival) + running + 0.8 * (1.0 - upright_fraction);
    EpisodeMetrics {
        loss,
        steps,
        success,
        rms_force: (force_sq / steps.max(1) as f64).sqrt(),
    }
}

pub fn simulate_policy(params: &[f64], seed: u64, horizon: usize) -> EpisodeMetrics {
    simulate_with(seed, horizon, |inputs| neural_action(params, inputs), None)
}

fn simulate_zero(seed: u64, horizon: usize) -> EpisodeMetrics {
    simulate_with(seed, horizon, |_| 0.0, None)
}

fn simulate_hand(seed: u64, horizon: usize) -> EpisodeMetrics {
    simulate_with(seed, horizon, hand_action, None)
}

fn aggregate(mut episodes: Vec<EpisodeMetrics>) -> ValidationMetrics {
    if episodes.is_empty() {
        return ValidationMetrics::default();
    }
    let n = episodes.len() as f64;
    let mean_loss = episodes.iter().map(|episode| episode.loss).sum::<f64>() / n;
    episodes.sort_by(|a, b| {
        b.loss
            .partial_cmp(&a.loss)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let tail_count = episodes.len().div_ceil(5).max(1);
    let cvar_loss = episodes
        .iter()
        .take(tail_count)
        .map(|episode| episode.loss)
        .sum::<f64>()
        / tail_count as f64;
    let success_rate =
        episodes.iter().filter(|episode| episode.success).count() as f64 / episodes.len() as f64;
    let mean_steps = episodes
        .iter()
        .map(|episode| episode.steps as f64)
        .sum::<f64>()
        / n;
    let rms_force = episodes
        .iter()
        .map(|episode| episode.rms_force)
        .sum::<f64>()
        / n;
    ValidationMetrics {
        score: mean_loss + 0.35 * cvar_loss,
        mean_loss,
        cvar_loss,
        success_rate,
        mean_steps,
        rms_force,
    }
}

pub fn evaluate_policy(params: &[f64], seeds: &[u64], horizon: usize) -> ValidationMetrics {
    aggregate(
        seeds
            .iter()
            .map(|&seed| simulate_policy(params, seed, horizon))
            .collect(),
    )
}

/// Evaluate one selected policy on a named seed stream that is disjoint from
/// every optimizer-root validation stream.
pub fn evaluate_frozen_test(
    params: &[f64],
    scenario_count: usize,
    horizon: usize,
) -> ValidationMetrics {
    let seeds = validation_seeds(0x3c6e_f372_fe94_f82b, scenario_count);
    evaluate_policy(params, &seeds, horizon)
}

fn evaluate_zero(seeds: &[u64], horizon: usize) -> ValidationMetrics {
    aggregate(
        seeds
            .iter()
            .map(|&seed| simulate_zero(seed, horizon))
            .collect(),
    )
}

fn evaluate_hand(seeds: &[u64], horizon: usize) -> ValidationMetrics {
    aggregate(
        seeds
            .iter()
            .map(|&seed| simulate_hand(seed, horizon))
            .collect(),
    )
}

pub fn initial_policy() -> Vec<f64> {
    let mut rng = FastRng::new(0x243f_6a88_85a3_08d3);
    let mut params: Vec<f64> = (0..PARAMS).map(|_| 0.08 * rng.signed()).collect();
    let skip = PARAMS - INPUTS;
    params[skip] = -0.02;
    params[skip + 1] = -0.04;
    params[skip + 2] = 0.12;
    params[skip + 3] = 0.0;
    params[skip + 4] = 0.03;
    params
}

pub fn baselines(root_seed: u64, config: &SearchConfig) -> Vec<BaselineRecord> {
    let seeds = validation_seeds(root_seed, config.validation_scenarios);
    let initial = initial_policy();
    vec![
        BaselineRecord {
            name: "zero-action",
            validation: evaluate_zero(&seeds, config.horizon),
        },
        BaselineRecord {
            name: "initial-neural",
            validation: evaluate_policy(&initial, &seeds, config.horizon),
        },
        BaselineRecord {
            name: "hand-energy-heuristic",
            validation: evaluate_hand(&seeds, config.horizon),
        },
    ]
}

fn make_fitness() -> Fitness {
    let mut fitness = Fitness::bounded(
        PARAMS,
        1,
        &vec![LOWER_WEIGHT; PARAMS],
        &vec![UPPER_WEIGHT; PARAMS],
    );
    fitness.set_normalize(true);
    fitness
}

fn thread_pool(workers: usize) -> Result<ThreadPool, String> {
    ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("policy-eval-{index}"))
        .build()
        .map_err(|error| format!("failed to build {workers}-worker pool: {error}"))
}

fn evaluate_population(
    pool: &ThreadPool,
    candidates: &[Vec<f64>],
    seeds: &[u64],
    horizon: usize,
    workers: usize,
) -> Vec<f64> {
    if workers == 1 {
        candidates
            .iter()
            .map(|candidate| evaluate_policy(candidate, seeds, horizon).score)
            .collect()
    } else {
        pool.install(|| {
            candidates
                .par_iter()
                .map(|candidate| evaluate_policy(candidate, seeds, horizon).score)
                .collect()
        })
    }
}

fn monitor_if_due(
    convergence: &mut Vec<ConvergencePoint>,
    last_monitor: &mut u64,
    evaluations: u64,
    train_best: f64,
    policy: &[f64],
    monitor_seeds: &[u64],
    config: &SearchConfig,
) -> f64 {
    if convergence.is_empty()
        || evaluations >= last_monitor.saturating_add(config.monitor_interval)
        || evaluations >= config.evaluations
    {
        let start = Instant::now();
        let monitor = evaluate_policy(policy, monitor_seeds, config.horizon);
        convergence.push(ConvergencePoint {
            evaluations,
            train_best,
            monitor_score: monitor.score,
        });
        *last_monitor = evaluations;
        start.elapsed().as_secs_f64()
    } else {
        0.0
    }
}

pub fn run_search(
    experiment: &str,
    algorithm: Algorithm,
    root_seed: u64,
    config: &SearchConfig,
) -> Result<RunRecord, String> {
    config.validate()?;
    let pool = thread_pool(config.workers)?;
    let initial = initial_policy();
    let monitor_seeds = validation_seeds(root_seed ^ 0x6a09_e667_f3bc_c909, 24);
    let validation_seed_set = validation_seeds(root_seed, config.validation_scenarios);
    let lower = vec![LOWER_WEIGHT; PARAMS];
    let upper = vec![UPPER_WEIGHT; PARAMS];
    let mut convergence = Vec::new();
    let mut last_monitor = 0;
    let mut monitor_seconds = 0.0;
    let mut generation = 0_u64;
    let start = Instant::now();

    let (policy, train_best, evaluations) = match algorithm {
        Algorithm::Pgpe => {
            let params = PgpeParams {
                popsize: config.popsize as i32,
                max_evaluations: config.evaluations,
                seed: root_seed,
                ..Default::default()
            };
            let mut optimizer = Pgpe::new(make_fitness(), &initial, &[0.18], &params);
            while optimizer.result().evaluations < config.evaluations && optimizer.stop() == 0 {
                let candidates = optimizer.ask_pop();
                let seeds = scenario_seeds(
                    root_seed,
                    generation,
                    config.train_scenarios,
                    config.scenario_mode,
                );
                let values =
                    evaluate_population(&pool, &candidates, &seeds, config.horizon, config.workers);
                optimizer.tell_pop(&values);
                generation += 1;
                let result = optimizer.result();
                monitor_seconds += monitor_if_due(
                    &mut convergence,
                    &mut last_monitor,
                    result.evaluations,
                    result.y,
                    &result.x,
                    &monitor_seeds,
                    config,
                );
            }
            let result = optimizer.result();
            (result.x, result.y, result.evaluations)
        }
        Algorithm::Crfmnes => {
            let params = CrfmnesParams {
                popsize: config.popsize as i32,
                max_evaluations: config.evaluations,
                seed: root_seed,
                ..Default::default()
            };
            let mut optimizer = Crfmnes::new(make_fitness(), &initial, 0.18, &params);
            while optimizer.result().evaluations < config.evaluations && optimizer.stop() == 0 {
                let candidates = optimizer.ask_pop();
                let seeds = scenario_seeds(
                    root_seed,
                    generation,
                    config.train_scenarios,
                    config.scenario_mode,
                );
                let values =
                    evaluate_population(&pool, &candidates, &seeds, config.horizon, config.workers);
                optimizer.tell_pop(&values);
                generation += 1;
                let result = optimizer.result();
                monitor_seconds += monitor_if_due(
                    &mut convergence,
                    &mut last_monitor,
                    result.evaluations,
                    result.y,
                    &result.x,
                    &monitor_seeds,
                    config,
                );
            }
            let result = optimizer.result();
            (result.x, result.y, result.evaluations)
        }
        Algorithm::Cmaes => {
            let params = CmaesParams {
                popsize: config.popsize as i32,
                max_evaluations: config.evaluations,
                stop_tol_hist_fun: 0.0,
                seed: root_seed,
                ..Default::default()
            };
            let mut optimizer = Cmaes::new(make_fitness(), &initial, &[0.18], &params);
            while optimizer.result().evaluations < config.evaluations && optimizer.stop() == 0 {
                let candidates = optimizer.ask();
                let seeds = scenario_seeds(
                    root_seed,
                    generation,
                    config.train_scenarios,
                    config.scenario_mode,
                );
                let values =
                    evaluate_population(&pool, &candidates, &seeds, config.horizon, config.workers);
                optimizer.tell(&values);
                generation += 1;
                let result = optimizer.result();
                monitor_seconds += monitor_if_due(
                    &mut convergence,
                    &mut last_monitor,
                    result.evaluations,
                    result.y,
                    &result.x,
                    &monitor_seeds,
                    config,
                );
            }
            let result = optimizer.result();
            (result.x, result.y, result.evaluations)
        }
        Algorithm::Biteopt => {
            let params = BiteParams {
                popsize: config.popsize as i32,
                max_evaluations: config.evaluations,
                seed: root_seed,
                ..Default::default()
            };
            let mut optimizer = BiteOpt::new(&lower, &upper, Some(&initial), &params);
            while optimizer.result_public().evaluations < config.evaluations
                && optimizer.stop_code() == 0
            {
                let candidates = optimizer.ask(config.popsize);
                if candidates.is_empty() {
                    break;
                }
                let seeds = scenario_seeds(
                    root_seed,
                    generation,
                    config.train_scenarios,
                    config.scenario_mode,
                );
                let values =
                    evaluate_population(&pool, &candidates, &seeds, config.horizon, config.workers);
                optimizer.tell(&values);
                generation += 1;
                let result = optimizer.result_public();
                monitor_seconds += monitor_if_due(
                    &mut convergence,
                    &mut last_monitor,
                    result.evaluations,
                    result.y,
                    &result.x,
                    &monitor_seeds,
                    config,
                );
            }
            let result = optimizer.result_public();
            (result.x, result.y, result.evaluations)
        }
    };

    let wall_seconds = (start.elapsed().as_secs_f64() - monitor_seconds).max(0.0);
    let validation = evaluate_policy(&policy, &validation_seed_set, config.horizon);
    Ok(RunRecord {
        experiment: experiment.to_string(),
        scenario_mode: config.scenario_mode,
        algorithm,
        seed: root_seed,
        workers: config.workers,
        popsize: config.popsize,
        evaluations,
        train_scenarios: config.train_scenarios,
        validation_scenarios: config.validation_scenarios,
        horizon: config.horizon,
        train_best,
        validation,
        wall_seconds,
        policy,
        convergence,
    })
}

pub fn summarize(values: &[f64]) -> Summary {
    if values.is_empty() {
        return Summary {
            mean: f64::NAN,
            sdev: f64::NAN,
        };
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sdev = if values.len() > 1 {
        (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    Summary { mean, sdev }
}

pub fn write_records(path: &Path, records: &[RunRecord]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "experiment,scenario_mode,algorithm,seed,workers,popsize,evaluations,optimizer_rollouts,train_scenarios,validation_scenarios,horizon,train_best,validation_score,mean_loss,cvar_loss,success_rate,mean_steps,rms_force,wall_seconds"
    )
    .map_err(|error| error.to_string())?;
    for record in records {
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.6},{:.12},{:.9}",
            record.experiment,
            record.scenario_mode,
            record.algorithm,
            record.seed,
            record.workers,
            record.popsize,
            record.evaluations,
            record.evaluations * record.train_scenarios as u64,
            record.train_scenarios,
            record.validation_scenarios,
            record.horizon,
            record.train_best,
            record.validation.score,
            record.validation.mean_loss,
            record.validation.cvar_loss,
            record.validation.success_rate,
            record.validation.mean_steps,
            record.validation.rms_force,
            record.wall_seconds,
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub fn write_baselines(path: &Path, records: &[BaselineRecord]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "baseline,validation_score,mean_loss,cvar_loss,success_rate,mean_steps,rms_force"
    )
    .map_err(|error| error.to_string())?;
    for record in records {
        writeln!(
            out,
            "{},{:.12},{:.12},{:.12},{:.12},{:.6},{:.12}",
            record.name,
            record.validation.score,
            record.validation.mean_loss,
            record.validation.cvar_loss,
            record.validation.success_rate,
            record.validation.mean_steps,
            record.validation.rms_force,
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub fn write_frozen_test(
    path: &Path,
    scenario_count: usize,
    metrics: ValidationMetrics,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "scenario_set,scenarios,validation_score,mean_loss,cvar_loss,success_rate,mean_steps,rms_force"
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        out,
        "frozen-final,{scenario_count},{:.12},{:.12},{:.12},{:.12},{:.6},{:.12}",
        metrics.score,
        metrics.mean_loss,
        metrics.cvar_loss,
        metrics.success_rate,
        metrics.mean_steps,
        metrics.rms_force,
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

pub fn write_convergence(path: &Path, records: &[RunRecord]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "experiment,scenario_mode,algorithm,seed,workers,evaluations,train_best,monitor_score"
    )
    .map_err(|error| error.to_string())?;
    for record in records {
        for point in &record.convergence {
            writeln!(
                out,
                "{},{},{},{},{},{},{:.12},{:.12}",
                record.experiment,
                record.scenario_mode,
                record.algorithm,
                record.seed,
                record.workers,
                point.evaluations,
                point.train_best,
                point.monitor_score,
            )
            .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

pub fn write_policy(path: &Path, record: &RunRecord) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut out = BufWriter::new(file);
    writeln!(out, "index,weight").map_err(|error| error.to_string())?;
    for (index, value) in record.policy.iter().enumerate() {
        writeln!(out, "{index},{value:.17}").map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub fn write_trajectory(path: &Path, record: &RunRecord, seed: u64) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut rows = Vec::new();
    simulate_with(
        seed,
        record.horizon,
        |inputs| neural_action(&record.policy, inputs),
        Some(&mut rows),
    );
    let file = File::create(path).map_err(|error| error.to_string())?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "step,time,position,velocity,angle,angular_velocity,force"
    )
    .map_err(|error| error.to_string())?;
    for (step, state, force) in rows {
        writeln!(
            out,
            "{step},{:.6},{:.12},{:.12},{:.12},{:.12},{:.12}",
            step as f64 * DT,
            state.position,
            state.velocity,
            state.angle,
            state.angular_velocity,
            force,
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neural_parameter_layout_is_consumed_exactly() {
        assert_eq!(PARAMS, 118);
        let action = neural_action(&vec![0.0; PARAMS], [0.1, -0.2, 0.3, -0.4, 0.5]);
        assert_eq!(action, 0.0);
    }

    #[test]
    fn policy_evaluation_is_deterministic() {
        let policy = initial_policy();
        let seeds = validation_seeds(42, 8);
        let first = evaluate_policy(&policy, &seeds, 100);
        let second = evaluate_policy(&policy, &seeds, 100);
        assert_eq!(first.score.to_bits(), second.score.to_bits());
        assert_eq!(first.success_rate.to_bits(), second.success_rate.to_bits());
    }

    #[test]
    fn baselines_are_finite_and_zero_does_not_swing_up() {
        let seeds = validation_seeds(7, 32);
        let zero = evaluate_zero(&seeds, 200);
        let hand = evaluate_hand(&seeds, 200);
        assert!(zero.score.is_finite());
        assert!(hand.score.is_finite());
        assert_eq!(zero.success_rate, 0.0);
    }

    #[test]
    fn serial_and_parallel_batches_match() {
        let candidates = vec![initial_policy(), vec![0.0; PARAMS]];
        let seeds = validation_seeds(11, 4);
        let serial_pool = thread_pool(1).expect("serial pool");
        let parallel_pool = thread_pool(2).expect("parallel pool");
        let serial = evaluate_population(&serial_pool, &candidates, &seeds, 80, 1);
        let parallel = evaluate_population(&parallel_pool, &candidates, &seeds, 80, 2);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn tiny_pgpe_and_crfmnes_runs_are_finite() {
        let config = SearchConfig {
            evaluations: 64,
            popsize: 8,
            workers: 2,
            train_scenarios: 1,
            validation_scenarios: 4,
            horizon: 50,
            monitor_interval: 64,
            ..Default::default()
        };
        for algorithm in [Algorithm::Pgpe, Algorithm::Crfmnes] {
            let result = run_search("test", algorithm, 3, &config).expect("optimizer run");
            assert!(result.train_best.is_finite());
            assert!(result.validation.score.is_finite());
            assert_eq!(result.evaluations, 64);
        }
    }
}
