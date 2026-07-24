#![allow(clippy::too_many_arguments)] // ReBop generates a 15-rate constructor.

//! Robust stochastic oscillator design with ReBop and fcmaes.
//!
//! ReBop executes one seeded stochastic simulation at a time. Parallelism is
//! deliberately applied outside the simulator: every fcmaes worker evaluates
//! one complete candidate over the same fixed training seed set.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{
    Archive, BiteParams, Fitness, MapElitesParams, Mode, ModeParams, QdBatchFitness, RetryBounds,
    RetryConfig, RetryImprovement, RetryRunResult, Rng, map_elites_batch_with_progress,
    optimize_bite, parallel_batch, pareto_indices, retry,
};
use rebop::define_system;

pub const DIMENSION: usize = 15;
pub const OBJECTIVES: usize = 3;
pub const SAMPLE_COUNT: usize = 128;
pub const SAMPLE_INTERVAL: f64 = 1.0;
pub const BURN_IN: f64 = 64.0;
pub const MAX_MOLECULES: f64 = 100_000.0;
pub const LOG_RATE_RADIUS: f64 = 0.5;
pub const QD_DESCRIPTOR_LOWER: [f64; 2] = [8.0, 0.0];
pub const QD_DESCRIPTOR_UPPER: [f64; 2] = [64.0, 20_000.0];

pub const RATE_NAMES: [&str; DIMENSION] = [
    "alpha_a",
    "alpha_prime_a",
    "alpha_r",
    "alpha_prime_r",
    "beta_a",
    "beta_r",
    "delta_ma",
    "delta_mr",
    "delta_a",
    "delta_r",
    "gamma_a",
    "gamma_r",
    "gamma_c",
    "theta_a",
    "theta_r",
];

pub const BASE_RATES: [f64; DIMENSION] = [
    50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0,
];

/// Named common-random-number streams used by every optimizer candidate.
pub const TRAINING_SEEDS: [u64; 16] = [
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
    0x4528_21e6_38d0_1377,
    0xbe54_66cf_34e9_0c6c,
    0xc0ac_29b7_c97c_50dd,
    0x3f84_d5b5_b547_0917,
    0x9216_d5d9_8979_fb1b,
    0xd131_0ba6_98df_b5ac,
    0x2ffd_72db_d01a_dfb7,
    0xb8e1_afed_6a26_7e96,
    0xba7c_9045_f12c_7f99,
    0x24a1_9947_b391_6cf7,
    0x0801_f2e2_858e_fc16,
    0x6369_20d8_7157_4e69,
];

/// Disjoint streams reserved for post-optimization robustness validation.
pub const VALIDATION_SEEDS: [u64; 16] = [
    0xa458_fea3_f493_3d7e,
    0x0d95_748f_728e_b658,
    0x718b_cd58_8215_4aee,
    0x7b54_a41d_c25a_59b5,
    0x9c30_d539_2af2_6013,
    0xc5d1_b023_2860_85f0,
    0xca41_7918_b8db_38ef,
    0x8e79_dcb0_603a_180e,
    0x6c9e_0e8b_b01e_8a3e,
    0xd715_77c1_bd31_4b27,
    0x78af_2fda_5560_5c60,
    0xe655_25f3_aa55_ab94,
    0x5748_9862_63e8_1440,
    0x55ca_396a_2aab_10b6,
    0xb4cc_5c34_1141_e8ce,
    0xa154_86af_7c72_e993,
];

define_system! {
    alpha_a alpha_prime_a alpha_r alpha_prime_r beta_a beta_r delta_ma delta_mr
    delta_a delta_r gamma_a gamma_r gamma_c theta_a theta_r;
    Vilar { da, dr, dpa, dpr, ma, mr, a, r, c }
    translation_a       : ma      => ma + a  @ beta_a
    complexation        : a + r   => c       @ gamma_c
    decomplexation      : c       => r       @ delta_a
    decay_protein_a     : a       =>         @ delta_a
    decay_mrna_a        : ma      =>         @ delta_ma
    transcription_pa    : dpa     => dpa + ma @ alpha_prime_a
    translation_r       : mr      => mr + r  @ beta_r
    decay_protein_r     : r       =>         @ delta_r
    transcription_a     : da      => da + ma @ alpha_a
    activation_r        : dr + a  => dpr     @ gamma_r
    deactivation_r      : dpr     => dr + a  @ theta_r
    activation_a        : da + a  => dpa     @ gamma_a
    deactivation_a      : dpa     => da + a  @ theta_a
    transcription_pr    : dpr     => dpr + mr @ alpha_prime_r
    decay_mrna_r        : mr      =>         @ delta_mr
    transcription_r     : dr      => dr + mr @ alpha_r
}

pub fn base_log_rates() -> [f64; DIMENSION] {
    BASE_RATES.map(f64::log10)
}

pub fn lower_bounds() -> [f64; DIMENSION] {
    base_log_rates().map(|value| value - LOG_RATE_RADIUS)
}

pub fn upper_bounds() -> [f64; DIMENSION] {
    base_log_rates().map(|value| value + LOG_RATE_RADIUS)
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogRates {
    values: [f64; DIMENSION],
}

impl LogRates {
    pub fn from_slice(values: &[f64]) -> Result<Self, &'static str> {
        if values.len() != DIMENSION {
            return Err("an oscillator design must contain exactly fifteen log10 rates");
        }
        let lower = lower_bounds();
        let upper = upper_bounds();
        if values.iter().enumerate().any(|(index, value)| {
            !value.is_finite() || *value < lower[index] || *value > upper[index]
        }) {
            return Err("oscillator log rates lie outside the supported bounds");
        }
        let mut array = [0.0; DIMENSION];
        array.copy_from_slice(values);
        Ok(Self { values: array })
    }

    pub fn values(&self) -> &[f64; DIMENSION] {
        &self.values
    }

    pub fn rates(&self) -> [f64; DIMENSION] {
        self.values.map(|value| 10.0_f64.powf(value))
    }
}

impl Default for LogRates {
    fn default() -> Self {
        Self {
            values: base_log_rates(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvaluationConfig {
    pub target_period: f64,
    pub replications: usize,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            target_period: 20.0,
            replications: 4,
        }
    }
}

impl EvaluationConfig {
    pub fn validate(&self, available_seeds: usize) -> Result<(), &'static str> {
        if !self.target_period.is_finite() || !(8.0..=64.0).contains(&self.target_period) {
            return Err("target period must be finite and lie in 8..=64");
        }
        if self.replications == 0 || self.replications > available_seeds {
            return Err("replications must fit the selected named seed set");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TracePoint {
    pub time: f64,
    pub activator: f64,
    pub repressor: f64,
    pub complex: f64,
    pub total_molecules: f64,
}

#[derive(Clone, Debug)]
pub struct ReplicateMetrics {
    pub seed: u64,
    pub period: f64,
    pub amplitude: f64,
    pub spectral_concentration: f64,
    pub autocorrelation_decay: f64,
    pub mean_molecules: f64,
    pub failed: bool,
    pub trace: Vec<TracePoint>,
}

#[derive(Clone, Debug)]
pub struct RobustMetrics {
    pub period: f64,
    pub period_error: f64,
    pub amplitude: f64,
    pub spectral_concentration: f64,
    pub autocorrelation_decay: f64,
    pub mean_molecules: f64,
    pub failure_fraction: f64,
    pub period_cv: f64,
    pub amplitude_cv: f64,
    pub oscillation_error: f64,
    pub fragility: f64,
    pub scalar_score: f64,
    pub replicates: Vec<ReplicateMetrics>,
}

impl RobustMetrics {
    pub fn objectives(&self) -> [f64; OBJECTIVES] {
        [
            self.oscillation_error,
            self.mean_molecules + 5_000.0 * self.failure_fraction,
            self.fragility,
        ]
    }
}

#[derive(Debug)]
struct SpectrumPlan {
    bins: Vec<(usize, Vec<[f64; 2]>)>,
}

fn spectrum_plan() -> &'static SpectrumPlan {
    static PLAN: OnceLock<SpectrumPlan> = OnceLock::new();
    PLAN.get_or_init(|| {
        let bins = (2..=16)
            .map(|bin| {
                let weights = (0..SAMPLE_COUNT)
                    .map(|index| {
                        let phase =
                            std::f64::consts::TAU * bin as f64 * index as f64 / SAMPLE_COUNT as f64;
                        [phase.cos(), phase.sin()]
                    })
                    .collect();
                (bin, weights)
            })
            .collect();
        SpectrumPlan { bins }
    })
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 1.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean.abs() <= 1.0e-12 {
        return 1.0;
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (variance.sqrt() / mean.abs()).min(10.0)
}

fn correlation(centered: &[f64], lag: usize) -> f64 {
    if lag == 0 || lag >= centered.len() {
        return 0.0;
    }
    let mut cross = 0.0;
    let mut left = 0.0;
    let mut right = 0.0;
    for index in 0..centered.len() - lag {
        cross += centered[index] * centered[index + lag];
        left += centered[index].powi(2);
        right += centered[index + lag].powi(2);
    }
    if left <= 0.0 || right <= 0.0 {
        0.0
    } else {
        (cross / (left * right).sqrt()).clamp(-1.0, 1.0)
    }
}

fn analyze_trace(seed: u64, trace: Vec<TracePoint>) -> ReplicateMetrics {
    let signal: Vec<f64> = trace.iter().map(|point| point.repressor).collect();
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let centered: Vec<f64> = signal.iter().map(|value| value - mean).collect();
    let sum_squares = centered.iter().map(|value| value * value).sum::<f64>();
    let mut ordered = signal.clone();
    ordered.sort_unstable_by(f64::total_cmp);
    let low = ordered[ordered.len() / 10];
    let high = ordered[ordered.len() * 9 / 10];
    let amplitude = high - low;

    let powers: Vec<(usize, f64)> = spectrum_plan()
        .bins
        .iter()
        .map(|(bin, weights)| {
            let (real, imag) =
                centered
                    .iter()
                    .zip(weights)
                    .fold((0.0, 0.0), |(real, imag), (value, weight)| {
                        (real + value * weight[0], imag - value * weight[1])
                    });
            (*bin, real.mul_add(real, imag * imag))
        })
        .collect();
    let peak_index = powers
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.1.total_cmp(&right.1.1))
        .map_or(0, |(index, _)| index);
    let peak_bin = powers[peak_index].0 as f64;
    let interpolated_bin = if peak_index > 0 && peak_index + 1 < powers.len() {
        let left = powers[peak_index - 1].1.max(1.0e-300).ln();
        let center = powers[peak_index].1.max(1.0e-300).ln();
        let right = powers[peak_index + 1].1.max(1.0e-300).ln();
        let denominator = left - 2.0 * center + right;
        let offset = if denominator.abs() > 1.0e-12 {
            0.5 * (left - right) / denominator
        } else {
            0.0
        };
        peak_bin + offset.clamp(-0.5, 0.5)
    } else {
        peak_bin
    };
    let period = SAMPLE_COUNT as f64 * SAMPLE_INTERVAL / interpolated_bin;
    let band_power = powers[peak_index].1
        + peak_index
            .checked_sub(1)
            .map_or(0.0, |index| powers[index].1)
        + powers.get(peak_index + 1).map_or(0.0, |entry| entry.1);
    let spectral_concentration = if sum_squares > 0.0 {
        (2.0 * band_power / (SAMPLE_COUNT as f64 * sum_squares)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let lag = (period / SAMPLE_INTERVAL).round() as usize;
    let corr_one = correlation(&centered, lag);
    let corr_two = correlation(&centered, 2 * lag);
    let autocorrelation_decay =
        (1.0 - corr_two.max(0.0) + 0.25 * (corr_one - corr_two).max(0.0)).min(2.0);
    let mean_molecules =
        trace.iter().map(|point| point.total_molecules).sum::<f64>() / trace.len() as f64;
    let physically_invalid = trace.iter().any(|point| {
        !point.total_molecules.is_finite()
            || point.total_molecules < 0.0
            || point.total_molecules > MAX_MOLECULES
    });
    let failed = physically_invalid
        || !period.is_finite()
        || amplitude < 10.0
        || spectral_concentration < 0.05;
    ReplicateMetrics {
        seed,
        period,
        amplitude,
        spectral_concentration,
        autocorrelation_decay,
        mean_molecules,
        failed,
        trace,
    }
}

fn simulate_replicate(log_rates: &LogRates, seed: u64, record_trace: bool) -> ReplicateMetrics {
    let rates = log_rates.rates();
    let mut model = Vilar::with_parameters(
        rates[0], rates[1], rates[2], rates[3], rates[4], rates[5], rates[6], rates[7], rates[8],
        rates[9], rates[10], rates[11], rates[12], rates[13], rates[14],
    );
    model.seed(seed);
    model.da = 1;
    model.dr = 1;
    model.advance_until(BURN_IN);
    let mut trace = Vec::with_capacity(SAMPLE_COUNT);
    for sample in 1..=SAMPLE_COUNT {
        let time = BURN_IN + sample as f64 * SAMPLE_INTERVAL;
        model.advance_until(time);
        let total_molecules = (model.ma + model.mr + model.a + model.r + model.c) as f64;
        trace.push(TracePoint {
            time,
            activator: model.a as f64,
            repressor: model.r as f64,
            complex: model.c as f64,
            total_molecules,
        });
    }
    let mut metrics = analyze_trace(seed, trace);
    if !record_trace {
        metrics.trace.clear();
    }
    metrics
}

pub fn evaluate_with_seeds(
    log_rates: &LogRates,
    target_period: f64,
    seeds: &[u64],
    record_first_trace: bool,
) -> RobustMetrics {
    let replicates: Vec<ReplicateMetrics> = seeds
        .iter()
        .enumerate()
        .map(|(index, &seed)| simulate_replicate(log_rates, seed, record_first_trace && index == 0))
        .collect();
    let count = replicates.len().max(1) as f64;
    let period = replicates.iter().map(|run| run.period).sum::<f64>() / count;
    let period_error = replicates
        .iter()
        .map(|run| (run.period - target_period).abs() / target_period)
        .sum::<f64>()
        / count;
    let amplitude = replicates.iter().map(|run| run.amplitude).sum::<f64>() / count;
    let spectral_concentration = replicates
        .iter()
        .map(|run| run.spectral_concentration)
        .sum::<f64>()
        / count;
    let autocorrelation_decay = replicates
        .iter()
        .map(|run| run.autocorrelation_decay)
        .sum::<f64>()
        / count;
    let mean_molecules = replicates.iter().map(|run| run.mean_molecules).sum::<f64>() / count;
    let failure_fraction = replicates.iter().filter(|run| run.failed).count() as f64 / count;
    let periods: Vec<f64> = replicates.iter().map(|run| run.period).collect();
    let amplitudes: Vec<f64> = replicates.iter().map(|run| run.amplitude).collect();
    let period_cv = coefficient_of_variation(&periods);
    let amplitude_cv = coefficient_of_variation(&amplitudes);
    let amplitude_penalty = ((30.0 - amplitude) / 30.0).max(0.0);
    let spectral_impurity = 1.0 - spectral_concentration;
    let oscillation_error = period_error
        + 2.0 * spectral_impurity
        + amplitude_penalty
        + 0.5 * autocorrelation_decay
        + 5.0 * failure_fraction;
    let fragility = period_cv + 0.5 * amplitude_cv + autocorrelation_decay + 5.0 * failure_fraction;
    let scalar_score = oscillation_error + 0.001 * mean_molecules + 2.0 * fragility;
    RobustMetrics {
        period,
        period_error,
        amplitude,
        spectral_concentration,
        autocorrelation_decay,
        mean_molecules,
        failure_fraction,
        period_cv,
        amplitude_cv,
        oscillation_error,
        fragility,
        scalar_score,
        replicates,
    }
}

pub fn evaluate_training(
    values: &[f64],
    config: &EvaluationConfig,
    record_first_trace: bool,
) -> Result<RobustMetrics, &'static str> {
    config.validate(TRAINING_SEEDS.len())?;
    let rates = LogRates::from_slice(values)?;
    Ok(evaluate_with_seeds(
        &rates,
        config.target_period,
        &TRAINING_SEEDS[..config.replications],
        record_first_trace,
    ))
}

pub fn evaluate_validation(
    values: &[f64],
    config: &EvaluationConfig,
    record_first_trace: bool,
) -> Result<RobustMetrics, &'static str> {
    config.validate(VALIDATION_SEEDS.len())?;
    let rates = LogRates::from_slice(values)?;
    Ok(evaluate_with_seeds(
        &rates,
        config.target_period,
        &VALIDATION_SEEDS[..config.replications],
        record_first_trace,
    ))
}

pub fn scalar_objective(values: &[f64], config: &EvaluationConfig) -> f64 {
    evaluate_training(values, config, false).map_or(1.0e99, |metrics| metrics.scalar_score)
}

pub fn multi_objective(values: &[f64], config: &EvaluationConfig) -> Vec<f64> {
    evaluate_training(values, config, false).map_or_else(
        |_| vec![1.0e99; OBJECTIVES],
        |metrics| metrics.objectives().to_vec(),
    )
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
            evaluations_per_retry: 2_000,
            retries: 8,
            workers: 0,
            depth: 6,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScalarOutcome {
    pub design: LogRates,
    pub training: RobustMetrics,
    pub validation: RobustMetrics,
    pub evaluations: u64,
    pub completed_retries: usize,
    pub elapsed: Duration,
    pub improvements: Vec<RetryImprovement>,
}

pub fn optimize_scalar(
    training_config: &EvaluationConfig,
    validation_config: &EvaluationConfig,
    options: &ScalarOptions,
) -> Result<ScalarOutcome, Box<dyn Error>> {
    training_config.validate(TRAINING_SEEDS.len())?;
    validation_config.validate(VALIDATION_SEEDS.len())?;
    if options.evaluations_per_retry == 0 || options.retries == 0 {
        return Err("scalar evaluations and retries must be positive".into());
    }
    if !(1..=36).contains(&options.depth) {
        return Err("BiteOpt depth must lie in 1..=36".into());
    }
    let lower = lower_bounds();
    let upper = upper_bounds();
    let bounds = RetryBounds::new(lower.to_vec(), upper.to_vec())?;
    let objective = |values: &[f64]| scalar_objective(values, training_config);
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
        return Err("BiteOpt retry returned no finite oscillator design".into());
    }
    let design = LogRates::from_slice(&result.x)?;
    let training = evaluate_training(design.values(), training_config, false)?;
    let validation = evaluate_validation(design.values(), validation_config, true)?;
    Ok(ScalarOutcome {
        design,
        training,
        validation,
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
    pub design: LogRates,
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
    pub training: RobustMetrics,
    pub validation: RobustMetrics,
    pub evaluations: usize,
    pub generations: usize,
    pub elapsed: Duration,
    pub convergence: Vec<MoProgress>,
    /// Higher is better: negative balanced score for the final representative.
    pub quality: f64,
}

fn balanced_objective(values: &[f64; OBJECTIVES]) -> f64 {
    values[0] + 0.001 * values[1] + 2.0 * values[2]
}

pub fn optimize_multi(
    training_config: &EvaluationConfig,
    validation_config: &EvaluationConfig,
    options: &MultiOptions,
) -> Result<MultiOutcome, Box<dyn Error>> {
    training_config.validate(TRAINING_SEEDS.len())?;
    validation_config.validate(VALIDATION_SEEDS.len())?;
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
    let lower = lower_bounds();
    let upper = upper_bounds();
    let fitness = Fitness::bounded(DIMENSION, OBJECTIVES, &lower, &upper);
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
        let ys = parallel_batch(&xs, options.workers as i32, |values| {
            multi_objective(values, training_config)
        });
        for values in &ys {
            best_balanced =
                best_balanced.min(balanced_objective(&[values[0], values[1], values[2]]));
        }
        mode.tell(&ys);
        convergence.push(MoProgress {
            evaluations: (generation + 1) * options.popsize,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            best_quality: -best_balanced,
        });
    }

    let population = mode.population();
    let values = parallel_batch(&population, options.workers as i32, |candidate| {
        multi_objective(candidate, training_config)
    });
    let indices = pareto_indices(&values, OBJECTIVES)?;
    let mut pareto = Vec::with_capacity(indices.len());
    for index in indices {
        pareto.push(ParetoPoint {
            design: LogRates::from_slice(&population[index])?,
            objectives: [values[index][0], values[index][1], values[index][2]],
        });
    }
    if pareto.is_empty() {
        return Err("MODE returned an empty Pareto front".into());
    }
    pareto.sort_by(|left, right| {
        balanced_objective(&left.objectives).total_cmp(&balanced_objective(&right.objectives))
    });
    let representative = pareto[0].clone();
    let quality = -balanced_objective(&representative.objectives);
    let training = evaluate_training(representative.design.values(), training_config, false)?;
    let validation = evaluate_validation(representative.design.values(), validation_config, true)?;
    Ok(MultiOutcome {
        pareto,
        representative,
        training,
        validation,
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
    pub validation_niche_id: Option<usize>,
    pub grid_x: usize,
    pub grid_y: usize,
    pub design: LogRates,
    pub quality_train: f64,
    pub quality_validation: f64,
    pub descriptors_train: [f64; 2],
    pub descriptors_validation: [f64; 2],
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
    pub training: RobustMetrics,
    pub validation: RobustMetrics,
    pub evaluations: usize,
    pub validation_evaluations: usize,
    pub occupied: usize,
    pub capacity: usize,
    pub qd_score: f64,
    pub invalid_evaluations: usize,
    pub clipped_descriptors: usize,
    pub validation_same_niche_fraction: f64,
    pub elapsed: Duration,
    pub validation_elapsed: Duration,
    pub convergence: Vec<QdProgress>,
}

pub fn qd_objective(values: &[f64], config: &EvaluationConfig) -> (f64, [f64; 2]) {
    let Ok(metrics) = evaluate_training(values, config, false) else {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    };
    if !metrics.scalar_score.is_finite()
        || !metrics.period.is_finite()
        || !metrics.amplitude.is_finite()
        || metrics.failure_fraction >= 1.0
    {
        return (f64::INFINITY, [f64::INFINITY; 2]);
    }
    (
        metrics.scalar_score,
        [metrics.period, metrics.amplitude.max(0.0)],
    )
}

struct OscillatorQdBatch<'a> {
    config: &'a EvaluationConfig,
    workers: usize,
    evaluations: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    clipped: Arc<AtomicUsize>,
}

impl QdBatchFitness for OscillatorQdBatch<'_> {
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
    training_config: &EvaluationConfig,
    validation_config: &EvaluationConfig,
    options: &QdOptions,
) -> Result<QdOutcome, Box<dyn Error>> {
    training_config.validate(TRAINING_SEEDS.len())?;
    validation_config.validate(VALIDATION_SEEDS.len())?;
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
    let lower = lower_bounds();
    let upper = upper_bounds();
    let mut rng = Rng::new(options.seed);
    let mut archive = Archive::try_new(
        DIMENSION,
        &QD_DESCRIPTOR_LOWER,
        &QD_DESCRIPTOR_UPPER,
        options.capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(&lower, &upper, &mut rng);

    let evaluations = Arc::new(AtomicUsize::new(0));
    let invalid = Arc::new(AtomicUsize::new(0));
    let clipped = Arc::new(AtomicUsize::new(0));
    let mut batch = OscillatorQdBatch {
        config: training_config,
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
        &lower,
        &upper,
        &parameters,
        &mut rng,
        &mut |_, archive| {
            let count = evaluations.load(Ordering::Relaxed);
            convergence.push(QdProgress {
                evaluations: count,
                elapsed_seconds: started.elapsed().as_secs_f64(),
                coverage: archive.occupied() as f64 / archive.capacity() as f64,
                qd_score: archive.qd_score(),
                best_quality: archive.best_y(),
                invalid_fraction: invalid.load(Ordering::Relaxed) as f64 / count.max(1) as f64,
            });
        },
    )?;
    let elapsed = started.elapsed();
    debug_assert_eq!(evaluations.load(Ordering::Relaxed), actual_evaluations);

    let occupied_indices: Vec<usize> = (0..archive.capacity())
        .filter(|&index| archive.ys()[index].is_finite())
        .collect();
    let validation_started = Instant::now();
    let validation_metrics = parallel_batch(
        &occupied_indices
            .iter()
            .map(|&index| archive.xs()[index].clone())
            .collect::<Vec<_>>(),
        options.workers as i32,
        |values| evaluate_validation(values, validation_config, false),
    );
    let validation_elapsed = validation_started.elapsed();

    let mut same_niche = 0usize;
    let mut validation_finite = 0usize;
    let mut elites = Vec::with_capacity(occupied_indices.len());
    for (&niche_id, validation) in occupied_indices.iter().zip(validation_metrics) {
        let (quality_validation, descriptors_validation, validation_niche_id) = match validation {
            Ok(metrics)
                if metrics.scalar_score.is_finite()
                    && metrics.period.is_finite()
                    && metrics.amplitude.is_finite()
                    && metrics.failure_fraction < 1.0 =>
            {
                let descriptors = [metrics.period, metrics.amplitude.max(0.0)];
                let validation_niche = archive.index_of_niche(&descriptors);
                validation_finite += 1;
                if validation_niche == niche_id {
                    same_niche += 1;
                }
                (metrics.scalar_score, descriptors, Some(validation_niche))
            }
            _ => (f64::INFINITY, [f64::INFINITY; 2], None),
        };
        elites.push(QdPoint {
            niche_id,
            validation_niche_id,
            grid_x: niche_id % side,
            grid_y: niche_id / side,
            design: LogRates::from_slice(&archive.xs()[niche_id])?,
            quality_train: archive.ys()[niche_id],
            quality_validation,
            descriptors_train: [
                archive.descriptors()[niche_id][0],
                archive.descriptors()[niche_id][1],
            ],
            descriptors_validation,
            visit_count: archive.counts()[niche_id],
        });
    }
    elites.sort_by(|left, right| left.quality_train.total_cmp(&right.quality_train));
    let representative = elites
        .first()
        .cloned()
        .ok_or("MAP-Elites did not find a valid stochastic oscillator")?;
    let training = evaluate_training(representative.design.values(), training_config, false)?;
    let validation = evaluate_validation(representative.design.values(), validation_config, true)?;
    Ok(QdOutcome {
        representative,
        training,
        validation,
        evaluations: actual_evaluations,
        validation_evaluations: occupied_indices.len(),
        occupied: archive.occupied(),
        capacity: archive.capacity(),
        qd_score: archive.qd_score(),
        invalid_evaluations: invalid.load(Ordering::Relaxed),
        clipped_descriptors: clipped.load(Ordering::Relaxed),
        validation_same_niche_fraction: same_niche as f64 / validation_finite.max(1) as f64,
        elapsed,
        validation_elapsed,
        convergence,
        elites,
    })
}

fn effective_workers(requested: usize) -> usize {
    if requested == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        requested
    }
}

pub fn write_qd_artifacts(
    directory: &Path,
    initial_training: &RobustMetrics,
    initial_validation: &RobustMetrics,
    outcome: &QdOutcome,
    training_config: &EvaluationConfig,
    validation_config: &EvaluationConfig,
    options: &QdOptions,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    write_artifacts(
        directory,
        initial_training,
        initial_validation,
        &outcome.training,
        &outcome.validation,
        &[],
        &[],
    )?;

    let mut archive_csv = String::from(
        "niche_id,grid_x,grid_y,quality_train,quality_validation,descriptor_period_train,descriptor_amplitude_train,descriptor_period_validation,descriptor_amplitude_validation,validation_niche_id,same_niche,visit_count",
    );
    for name in RATE_NAMES {
        let _ = write!(archive_csv, ",decision_log10_{name}");
    }
    archive_csv.push('\n');
    for point in &outcome.elites {
        let _ = write!(
            archive_csv,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            point.niche_id,
            point.grid_x,
            point.grid_y,
            point.quality_train,
            point.quality_validation,
            point.descriptors_train[0],
            point.descriptors_train[1],
            point.descriptors_validation[0],
            point.descriptors_validation[1],
            point
                .validation_niche_id
                .map_or(-1_i64, |value| value as i64),
            u8::from(point.validation_niche_id == Some(point.niche_id)),
            point.visit_count,
        );
        for value in point.design.values() {
            let _ = write!(archive_csv, ",{value}");
        }
        archive_csv.push('\n');
    }
    fs::write(directory.join("qd_archive.csv"), archive_csv)?;

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
        "tutorial": "rebop-oscillator",
        "formulation": "qd",
        "command": command,
        "seed": options.seed,
        "workers": effective_workers(options.workers),
        "requested_evaluations": options.evaluations,
        "actual_evaluations": outcome.evaluations,
        "elapsed_seconds": outcome.elapsed.as_secs_f64(),
        "validation_elapsed_seconds": outcome.validation_elapsed.as_secs_f64(),
        "simulation": {
            "target_period": training_config.target_period,
            "replications": training_config.replications,
            "validation_replications": validation_config.replications,
            "training_seeds": TRAINING_SEEDS[..training_config.replications]
                .iter().map(|seed| format!("{seed:#018x}")).collect::<Vec<_>>(),
            "validation_seeds": VALIDATION_SEEDS[..validation_config.replications]
                .iter().map(|seed| format!("{seed:#018x}")).collect::<Vec<_>>()
        },
        "descriptors": [
            {
                "column": "descriptor_period",
                "label": "Mean period",
                "unit": "model time",
                "bounds": [QD_DESCRIPTOR_LOWER[0], QD_DESCRIPTOR_UPPER[0]]
            },
            {
                "column": "descriptor_amplitude",
                "label": "Mean amplitude",
                "unit": "molecules",
                "bounds": [QD_DESCRIPTOR_LOWER[1], QD_DESCRIPTOR_UPPER[1]]
            }
        ],
        "qd": {
            "capacity": outcome.capacity,
            "grid_shape": [side, side],
            "chunk_size": options.chunk_size,
            "quality_train_column": "quality_train",
            "quality_validation_column": "quality_validation",
            "quality_label": "Robust oscillator quality (lower is better)",
            "occupied": outcome.occupied,
            "coverage": outcome.occupied as f64 / outcome.capacity as f64,
            "qd_score": outcome.qd_score,
            "best_quality": outcome.representative.quality_train,
            "invalid_evaluations": outcome.invalid_evaluations,
            "clipped_descriptors": outcome.clipped_descriptors,
            "validation_evaluations": outcome.validation_evaluations,
            "validation_same_niche_fraction": outcome.validation_same_niche_fraction
        },
        "convergence_metrics": [
            "coverage", "qd_score", "best_quality", "invalid_fraction"
        ],
        "artifacts": {
            "qd_archive": "qd_archive.csv",
            "convergence": "convergence.csv",
            "replications": "replications.csv",
            "traces": "traces.csv",
            "report": "report.html"
        }
    });
    fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

fn push_trace_js(output: &mut String, name: &str, metrics: &RobustMetrics) {
    let _ = writeln!(output, "const {name} = [");
    if let Some(replication) = metrics.replicates.first() {
        for point in &replication.trace {
            let _ = writeln!(
                output,
                "[{:.6},{:.6},{:.6},{:.6}],",
                point.time, point.activator, point.repressor, point.complex,
            );
        }
    }
    output.push_str("];\n");
}

fn write_report_html(
    path: &Path,
    initial: &RobustMetrics,
    optimized: &RobustMetrics,
    convergence: &[MoProgress],
) -> Result<(), Box<dyn Error>> {
    let mut data = String::new();
    push_trace_js(&mut data, "initial", initial);
    push_trace_js(&mut data, "optimized", optimized);
    data.push_str("const convergence=[");
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
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fcmaes + ReBop oscillator report</title>
<style>
body{{margin:0;background:#111820;color:#e8edf2;font:16px system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:24px}} canvas{{width:100%;height:auto;background:#17232e;border-radius:8px}}
.initial{{color:#aab5c0}}.optimized{{color:#50d890}}
</style></head><body><main>
<h1>Robust stochastic oscillator</h1>
<p><span class="initial">initial repressor</span> · <span class="optimized">optimized repressor</span></p>
<canvas id="trace" width="1050" height="430"></canvas>
<h2>Optimization convergence</h2><canvas id="conv" width="1050" height="260"></canvas>
<script>{data}
function plot(canvas,series){{
 const c=document.getElementById(canvas),x=c.getContext("2d"),all=series.flatMap(s=>s[0]);
 if(!all.length)return;const xs=all.map(p=>p[0]),ys=all.map(p=>p[2]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const X=v=>45+(v-xmin)/Math.max(1e-12,xmax-xmin)*(c.width-70),Y=v=>20+(ymax-v)/Math.max(1e-12,ymax-ymin)*(c.height-55);
 series.forEach(([points,color,dash])=>{{x.strokeStyle=color;x.lineWidth=2;x.setLineDash(dash);x.beginPath();points.forEach((p,i)=>i?x.lineTo(X(p[0]),Y(p[2])):x.moveTo(X(p[0]),Y(p[2])));x.stroke();}});
 x.setLineDash([]);x.fillStyle="#d9e2ea";x.fillText("time",c.width-55,c.height-10);x.fillText("R",12,18);
}}
plot("trace",[[initial,"#8996a3",[7,7]],[optimized,"#50d890",[]]]);
if(convergence.length>1){{
 const c=document.getElementById("conv"),x=c.getContext("2d"),maxx=convergence.at(-1)[0],ys=convergence.map(p=>p[1]),miny=Math.min(...ys),maxy=Math.max(...ys);
 const X=v=>50+v/maxx*(c.width-75),Y=v=>20+(maxy-v)/Math.max(1e-12,maxy-miny)*(c.height-55);
 x.strokeStyle="#50d890";x.lineWidth=2;x.beginPath();convergence.forEach((p,i)=>i?x.lineTo(X(p[0]),Y(p[1])):x.moveTo(X(p[0]),Y(p[1])));x.stroke();
 x.fillStyle="#d9e2ea";x.fillText("evaluations",c.width-90,c.height-10);x.fillText("best quality",8,18);
}}
</script></main></body></html>"##
    );
    fs::write(path, html)?;
    Ok(())
}

fn write_replications(output: &mut String, design: &str, seed_set: &str, metrics: &RobustMetrics) {
    for run in &metrics.replicates {
        let _ = writeln!(
            output,
            "{design},{seed_set},{},{:.9},{:.9},{:.9},{:.9},{:.9},{}",
            run.seed,
            run.period,
            run.amplitude,
            run.spectral_concentration,
            run.autocorrelation_decay,
            run.mean_molecules,
            run.failed,
        );
    }
}

/// Write traces, per-replication metrics, convergence, Pareto data, and HTML.
pub fn write_artifacts(
    directory: &Path,
    initial_training: &RobustMetrics,
    initial_validation: &RobustMetrics,
    optimized_training: &RobustMetrics,
    optimized_validation: &RobustMetrics,
    convergence: &[MoProgress],
    pareto: &[ParetoPoint],
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut traces = String::from("design,seed,time,activator,repressor,complex,total_molecules\n");
    for (label, metrics) in [
        ("initial", initial_validation),
        ("optimized", optimized_validation),
    ] {
        if let Some(replication) = metrics.replicates.first() {
            for point in &replication.trace {
                let _ = writeln!(
                    traces,
                    "{label},{},{:.9},{:.9},{:.9},{:.9},{:.9}",
                    replication.seed,
                    point.time,
                    point.activator,
                    point.repressor,
                    point.complex,
                    point.total_molecules,
                );
            }
        }
    }
    fs::write(directory.join("traces.csv"), traces)?;

    let mut replications = String::from(
        "design,seed_set,seed,period,amplitude,spectral_concentration,autocorrelation_decay,mean_molecules,failed\n",
    );
    write_replications(&mut replications, "initial", "training", initial_training);
    write_replications(
        &mut replications,
        "initial",
        "validation",
        initial_validation,
    );
    write_replications(
        &mut replications,
        "optimized",
        "training",
        optimized_training,
    );
    write_replications(
        &mut replications,
        "optimized",
        "validation",
        optimized_validation,
    );
    fs::write(directory.join("replications.csv"), replications)?;

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
        "point_id,feasible,selected,objective_oscillation_error,objective_molecule_cost,objective_fragility",
    );
    for name in RATE_NAMES {
        let _ = write!(pareto_csv, ",decision_log10_{name}");
    }
    pareto_csv.push('\n');
    for (index, point) in pareto.iter().enumerate() {
        let _ = writeln!(
            pareto_csv,
            "{index},1,{},{},{},{},{}",
            u8::from(index == 0),
            point.objectives[0],
            point.objectives[1],
            point.objectives[2],
            point
                .design
                .values()
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    fs::write(directory.join("pareto.csv"), pareto_csv)?;
    write_report_html(
        &directory.join("report.html"),
        initial_validation,
        optimized_validation,
        convergence,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> EvaluationConfig {
        EvaluationConfig {
            target_period: 20.0,
            replications: 1,
        }
    }

    #[test]
    fn log_rates_round_trip_and_bounds() {
        let design = LogRates::default();
        assert_eq!(LogRates::from_slice(design.values()).unwrap(), design);
        assert!(LogRates::from_slice(&[0.0; DIMENSION - 1]).is_err());
        assert!(LogRates::from_slice(&[f64::NAN; DIMENSION]).is_err());
        let expected = BASE_RATES;
        for (actual, expected) in design.rates().iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12 * expected.max(1.0));
        }
    }

    #[test]
    fn named_seed_sets_are_disjoint() {
        assert!(
            TRAINING_SEEDS
                .iter()
                .all(|seed| !VALIDATION_SEEDS.contains(seed))
        );
    }

    #[test]
    fn seeded_simulation_is_reproducible() {
        let design = LogRates::default();
        let first = simulate_replicate(&design, TRAINING_SEEDS[0], true);
        let second = simulate_replicate(&design, TRAINING_SEEDS[0], true);
        assert_eq!(first.period, second.period);
        assert_eq!(first.amplitude, second.amplitude);
        assert_eq!(first.trace.len(), SAMPLE_COUNT);
        assert_eq!(first.trace[20].repressor, second.trace[20].repressor);
    }

    #[test]
    fn different_seeds_produce_different_paths() {
        let design = LogRates::default();
        let first = simulate_replicate(&design, TRAINING_SEEDS[0], true);
        let second = simulate_replicate(&design, TRAINING_SEEDS[1], true);
        assert!(
            first
                .trace
                .iter()
                .zip(&second.trace)
                .any(|(left, right)| left.repressor != right.repressor)
        );
    }

    #[test]
    fn baseline_metrics_are_finite() {
        let metrics =
            evaluate_training(LogRates::default().values(), &tiny_config(), true).unwrap();
        assert!(metrics.scalar_score.is_finite());
        assert!(metrics.period > 0.0);
        assert!(metrics.mean_molecules > 0.0);
        assert_eq!(metrics.replicates[0].trace.len(), SAMPLE_COUNT);
    }

    #[test]
    fn config_and_objective_adapters_validate_inputs() {
        assert!(
            EvaluationConfig {
                target_period: 1.0,
                replications: 1
            }
            .validate(TRAINING_SEEDS.len())
            .is_err()
        );
        assert_eq!(scalar_objective(&[0.0], &tiny_config()), 1.0e99);
        assert_eq!(
            multi_objective(&[0.0], &tiny_config()),
            vec![1.0e99; OBJECTIVES]
        );
        assert!(!qd_objective(&[0.0], &tiny_config()).0.is_finite());
    }

    #[test]
    fn tiny_parallel_mode_run_returns_a_front() {
        let outcome = optimize_multi(
            &tiny_config(),
            &tiny_config(),
            &MultiOptions {
                evaluations: 8,
                popsize: 4,
                workers: 2,
                seed: 7,
            },
        )
        .unwrap();
        assert_eq!(outcome.evaluations, 8);
        assert!(!outcome.pareto.is_empty());
        assert!(outcome.quality.is_finite());
    }

    #[test]
    fn optimization_options_are_validated() {
        assert!(
            optimize_scalar(
                &tiny_config(),
                &tiny_config(),
                &ScalarOptions {
                    evaluations_per_retry: 0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            optimize_multi(
                &tiny_config(),
                &tiny_config(),
                &MultiOptions {
                    popsize: 3,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            optimize_qd(
                &tiny_config(),
                &tiny_config(),
                &QdOptions {
                    capacity: 15,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn tiny_parallel_qd_run_validates_elites_on_holdout_seeds() {
        let outcome = optimize_qd(
            &tiny_config(),
            &tiny_config(),
            &QdOptions {
                evaluations: 32,
                capacity: 16,
                chunk_size: 8,
                workers: 2,
                seed: 17,
            },
        )
        .unwrap();
        assert_eq!(outcome.evaluations, 32);
        assert_eq!(outcome.occupied, outcome.elites.len());
        assert_eq!(outcome.validation_evaluations, outcome.occupied);
        assert!(outcome.representative.quality_train.is_finite());
        assert!(outcome.validation_same_niche_fraction.is_finite());
    }
}
