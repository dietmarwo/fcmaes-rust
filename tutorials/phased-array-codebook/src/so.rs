//! Equal-budget scalar optimizer comparison.

use std::error::Error;
use std::f64::consts::{PI, TAU};
use std::time::{Duration, Instant};

use fcmaes_core::{
    BiteParams, Cmaes, CmaesParams, De, DeParams, Fitness, RetryBounds, RetryConfig,
    RetryImprovement, RetryRunResult, Rng, optimize_bite, retry,
};

use crate::INVALID_COST;
use crate::array::{AngleGrid, Array};
use crate::config::Quantization;
use crate::decode::{Excitation, decode_beam};
use crate::kernel::SteeringMatrix;
use crate::scenarios::{RobustMetrics, evaluate_training};

/// Stage-A element count.
pub const ELEMENTS: usize = 16;
/// Stage-A decision dimension.
pub const DIMENSION: usize = 2 * ELEMENTS;
/// Pointing tolerance in degrees.
pub const POINTING_TOLERANCE_DEG: f64 = 0.25;
/// Worst-training-scenario PSLL feasibility limit.
pub const ROBUST_PSLL_LIMIT_DB: f64 = -10.0;

/// Scalar optimizer arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoOptimizer {
    /// Active CMA-ES retry.
    Cma,
    /// Differential-evolution retry.
    De,
    /// BiteOpt retry.
    Bite,
}

impl SoOptimizer {
    /// All equal-budget comparison arms.
    pub const ALL: [Self; 3] = [Self::Cma, Self::De, Self::Bite];

    /// Stable artifact label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cma => "cma",
            Self::De => "de",
            Self::Bite => "bite",
        }
    }
}

/// Immutable stage-A evaluator.
#[derive(Clone, Debug)]
pub struct BeamContext {
    /// ULA geometry.
    pub array: Array,
    /// Uniform-`u` cut.
    pub grid: AngleGrid,
    /// Precomputed steering matrix.
    pub steering: SteeringMatrix,
    /// Hardware quantization.
    pub quantization: Quantization,
}

impl BeamContext {
    /// Build the primary ULA context.
    #[must_use]
    pub fn stage_a(points: usize) -> Self {
        let array = Array::uniform_linear(ELEMENTS, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(points);
        let steering = SteeringMatrix::build(&array, &grid);
        Self {
            array,
            grid,
            steering,
            quantization: Quantization::default(),
        }
    }
}

/// Fully replayable robust scalar evaluation.
#[derive(Clone, Debug)]
pub struct SoEvaluation {
    /// Normalized optimizer controls.
    pub controls: Vec<f64>,
    /// Decoded register state.
    pub excitation: Excitation,
    /// Nominal and training-scenario metrics.
    pub robust: RobustMetrics,
    /// `|peak-target|-tolerance`, feasible at `≤0`.
    pub constraint_pointing: f64,
    /// `worst_psll-limit`, feasible at `≤0`.
    pub constraint_psll: f64,
    /// Calibrated kernel/scenario failure constraint, feasible at `≤0`.
    pub constraint_kernel: f64,
    /// Penalized scalar objective.
    pub objective: f64,
}

/// Evaluate one quantized beam.
#[must_use]
pub fn evaluate_beam(
    controls: &[f64],
    context: &BeamContext,
    requested_deg: f64,
) -> Option<SoEvaluation> {
    let excitation = decode_beam(controls, ELEMENTS, &context.quantization).ok()?;
    let robust = evaluate_training(&excitation.weights, &context.steering, &context.grid).ok()?;
    let pointing_error = (robust.nominal.peak_theta_deg - requested_deg).abs();
    let constraint_pointing = pointing_error - POINTING_TOLERANCE_DEG;
    let constraint_psll = robust.worst_psll_db - ROBUST_PSLL_LIMIT_DB;
    let constraint_kernel = if robust.kernel_failures == 0 {
        -1.0
    } else {
        crate::KERNEL_FAILURE_SCALE
    };
    let penalty = 50.0 * constraint_pointing.max(0.0)
        + 20.0 * constraint_psll.max(0.0)
        + constraint_kernel.max(0.0);
    let objective = robust.worst_psll_db + penalty;
    objective.is_finite().then_some(SoEvaluation {
        controls: controls.to_vec(),
        excitation,
        robust,
        constraint_pointing,
        constraint_psll,
        constraint_kernel,
        objective,
    })
}

#[inline]
const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Deterministic steered/tapered hardware seed.
#[must_use]
pub fn analytic_seed(requested_deg: f64, taper: f64) -> Vec<f64> {
    let quantization = Quantization::default();
    let phase_levels = 1_u32 << quantization.phase_bits;
    let attenuation_levels = 1_u32 << quantization.attenuator_bits;
    let target_u = requested_deg.to_radians().sin();
    let center = (ELEMENTS - 1) as f64 / 2.0;
    let mut controls = vec![0.0; DIMENSION];
    for index in 0..ELEMENTS {
        let phase = (-PI * (index as f64 - center) * target_u).rem_euclid(TAU);
        let phase_code = ((phase / TAU * f64::from(phase_levels)).round() as u32) % phase_levels;
        controls[index] = (f64::from(phase_code) + 0.5) / f64::from(phase_levels);

        let cosine = (2.0 * PI * index as f64 / (ELEMENTS - 1) as f64).cos();
        let amplitude = ((1.0 - taper) + taper * (0.54 - 0.46 * cosine)).clamp(0.168, 1.0);
        let attenuation_db = -20.0 * amplitude.log10();
        let attenuation_code = (attenuation_db / quantization.attenuator_step_db)
            .round()
            .clamp(0.0, f64::from(attenuation_levels - 1)) as u32;
        controls[ELEMENTS + index] =
            (f64::from(attenuation_code) + 0.5) / f64::from(attenuation_levels);
    }
    controls
}

/// Shared arm settings.
#[derive(Clone, Debug)]
pub struct SoConfig {
    /// Evaluation budget per arm.
    pub evaluations_per_arm: u64,
    /// Parallel retries.
    pub retries: usize,
    /// Candidate worker threads; zero means available parallelism.
    pub workers: usize,
    /// Root seed.
    pub seed: u64,
    /// Cut samples.
    pub points: usize,
    /// Requested signed steering angle.
    pub requested_deg: f64,
}

/// One optimizer-arm outcome.
#[derive(Clone, Debug)]
pub struct SoArmResult {
    /// Arm identity.
    pub optimizer: SoOptimizer,
    /// Requested objective calls.
    pub requested_evaluations: u64,
    /// Actual objective calls.
    pub actual_evaluations: u64,
    /// Completed retries.
    pub completed_retries: usize,
    /// Wall duration.
    pub elapsed: Duration,
    /// Best replayed design.
    pub best: SoEvaluation,
    /// Monotone retry improvements.
    pub improvements: Vec<RetryImprovement>,
}

/// Run one scalar optimizer under the common retry protocol.
pub fn optimize_arm(
    optimizer: SoOptimizer,
    config: &SoConfig,
) -> Result<SoArmResult, Box<dyn Error>> {
    if config.evaluations_per_arm == 0 || config.retries == 0 || config.points < 101 {
        return Err("SO budget/retries must be positive and cut needs at least 101 points".into());
    }
    let context = BeamContext::stage_a(config.points);
    let per_retry = config.evaluations_per_arm.div_ceil(config.retries as u64);
    let bounds = RetryBounds::new(vec![0.0; DIMENSION], vec![1.0; DIMENSION])?;
    let objective = |controls: &[f64]| {
        evaluate_beam(controls, &context, config.requested_deg)
            .map(|evaluation| evaluation.objective)
            .unwrap_or(INVALID_COST)
    };
    let retry_config = RetryConfig {
        num_retries: config.retries,
        workers: config.workers,
        capacity: config.retries,
        max_evaluations: per_retry,
        seed: config.seed.wrapping_add(match optimizer {
            SoOptimizer::Cma => 0,
            SoOptimizer::De => 10_000,
            SoOptimizer::Bite => 20_000,
        }),
        statistic_num: 500,
        ..Default::default()
    };
    let arm_seed = retry_config.seed;
    let started = Instant::now();
    let result = retry(
        &objective,
        &bounds,
        &retry_config,
        |objective, retry_context| {
            // A retry belongs to its stable run id, not to whichever worker
            // happened to claim it first.
            let run_seed = splitmix64(arm_seed ^ retry_context.run_id as u64);
            let mut rng = Rng::new(run_seed);
            let mut guess = analytic_seed(config.requested_deg, 0.35 + 0.6 * rng.uniform01());
            for value in &mut guess {
                *value = (*value + 0.01 * (rng.uniform01() - 0.5)).clamp(0.0, 1.0);
            }
            match optimizer {
                SoOptimizer::Cma => {
                    let fitness = Fitness::bounded(
                        DIMENSION,
                        1,
                        retry_context.bounds.lower(),
                        retry_context.bounds.upper(),
                    );
                    let mut cma = Cmaes::new(
                        fitness,
                        &guess,
                        &[0.12],
                        &CmaesParams {
                            max_evaluations: retry_context.max_evaluations,
                            seed: run_seed,
                            stop_tol_hist_fun: 0.0,
                            ..Default::default()
                        },
                    );
                    let optimized = cma.optimize(objective, 1);
                    RetryRunResult {
                        x: optimized.x,
                        y: optimized.y,
                        evaluations: optimized.evaluations,
                    }
                }
                SoOptimizer::De => {
                    let fitness = Fitness::bounded(
                        DIMENSION,
                        1,
                        retry_context.bounds.lower(),
                        retry_context.bounds.upper(),
                    );
                    let mut de = De::new(
                        fitness,
                        &guess,
                        &[0.15; DIMENSION],
                        None,
                        &DeParams {
                            popsize: 31,
                            max_evaluations: retry_context.max_evaluations,
                            seed: run_seed,
                            ..Default::default()
                        },
                    );
                    let optimized = de.optimize(objective);
                    RetryRunResult {
                        x: optimized.x,
                        y: optimized.y,
                        evaluations: optimized.evaluations,
                    }
                }
                SoOptimizer::Bite => {
                    let optimized = optimize_bite(
                        objective,
                        retry_context.bounds.lower(),
                        retry_context.bounds.upper(),
                        Some(&guess),
                        &BiteParams {
                            max_evaluations: retry_context.max_evaluations,
                            seed: run_seed,
                            ..Default::default()
                        },
                        1,
                    );
                    RetryRunResult {
                        x: optimized.x,
                        y: optimized.y,
                        evaluations: optimized.evaluations,
                    }
                }
            }
        },
    );
    if !result.success {
        return Err(format!("{} retained no finite result", optimizer.name()).into());
    }
    let best = evaluate_beam(&result.x, &context, config.requested_deg)
        .ok_or("best scalar candidate could not be replayed")?;
    Ok(SoArmResult {
        optimizer,
        requested_evaluations: config.evaluations_per_arm,
        actual_evaluations: result.evaluations,
        completed_retries: result.runs,
        elapsed: started.elapsed(),
        best,
        improvements: result.improvements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_quantized_seed_is_replayable_and_points_correctly() {
        let context = BeamContext::stage_a(2_001);
        let evaluation = evaluate_beam(&analytic_seed(30.0, 1.0), &context, 30.0).unwrap();
        assert!(evaluation.constraint_pointing <= 0.0);
        assert!(evaluation.robust.nominal.psll_db < -15.0);
    }

    #[test]
    fn tiny_bite_retry_retains_a_feasible_quantized_beam() {
        let config = SoConfig {
            evaluations_per_arm: 96,
            retries: 2,
            workers: 1,
            seed: 42,
            points: 361,
            requested_deg: 20.0,
        };
        let result = optimize_arm(SoOptimizer::Bite, &config).unwrap();
        assert!(result.best.constraint_pointing <= 0.0);
        assert!(result.best.constraint_kernel <= 0.0);
        assert!(result.best.objective.is_finite());
        let parallel = optimize_arm(
            SoOptimizer::Bite,
            &SoConfig {
                workers: 2,
                ..config
            },
        )
        .unwrap();
        assert_eq!(
            result.best.excitation.phase_codes,
            parallel.best.excitation.phase_codes
        );
        assert_eq!(
            result.best.excitation.attenuator_codes,
            parallel.best.excitation.attenuator_codes
        );
        assert_eq!(
            result.best.objective.to_bits(),
            parallel.best.objective.to_bits()
        );
    }
}
