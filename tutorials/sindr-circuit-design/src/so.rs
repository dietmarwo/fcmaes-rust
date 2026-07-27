//! Single-objective MFB band-pass tuning and equal-budget optimizer comparison.

use std::error::Error;
use std::time::{Duration, Instant};

use fcmaes_core::{
    BiteParams, Cmaes, CmaesParams, De, DeParams, Fitness, RetryBounds, RetryConfig,
    RetryImprovement, RetryRunResult, Rng, optimize_bite, retry,
};

use crate::decode::decode_bandpass_continuous;
use crate::features::{
    BandpassFeatures, bandpass_features, gain_curve, interpolated_peak, peak_index,
};
use crate::netlist::mfb_bandpass;
use crate::{BANDPASS_DIMENSION, INVALID_COST};

const TARGET_FREQUENCY_HZ: f64 = 10_000.0;
const TARGET_Q: f64 = 5.0;

/// Optimizer arm used by the scalar comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoOptimizer {
    Cma,
    De,
    Bite,
}

impl SoOptimizer {
    pub const ALL: [Self; 3] = [Self::Cma, Self::De, Self::Bite];

    pub fn name(self) -> &'static str {
        match self {
            Self::Cma => "cma",
            Self::De => "de",
            Self::Bite => "bite",
        }
    }
}

/// One fully decoded scalar evaluation.
#[derive(Clone, Debug)]
pub struct BandpassEvaluation {
    pub controls: Vec<f64>,
    pub components: [f64; 5],
    pub features: BandpassFeatures,
    pub objective: f64,
}

/// Configuration shared by all three SO comparison arms.
#[derive(Clone, Debug)]
pub struct SoConfig {
    pub evaluations_per_arm: u64,
    pub retries: usize,
    pub workers: usize,
    pub seed: u64,
    pub points: usize,
}

/// Outcome of one independent optimizer arm.
#[derive(Clone, Debug)]
pub struct SoArmResult {
    pub optimizer: SoOptimizer,
    pub requested_evaluations: u64,
    pub actual_evaluations: u64,
    pub completed_retries: usize,
    pub elapsed: Duration,
    pub best: BandpassEvaluation,
    pub improvements: Vec<RetryImprovement>,
}

/// Evaluate normalized MFB controls.
pub fn evaluate_bandpass(u: &[f64], points: usize) -> Option<BandpassEvaluation> {
    if u.len() != BANDPASS_DIMENSION {
        return None;
    }
    let components = decode_bandpass_continuous(u);
    let curve = gain_curve(
        &mfb_bandpass(&components),
        "out",
        TARGET_FREQUENCY_HZ / 31.622_776_601_683_793,
        TARGET_FREQUENCY_HZ * 31.622_776_601_683_793,
        points,
    )?;
    let features = bandpass_features(&curve)?;
    let amplitude = 10_f64.powf(features.peak_db / 20.0);
    let objective = (features.peak_hz / TARGET_FREQUENCY_HZ).log10().abs()
        + 0.2 * (features.q / TARGET_Q).log10().abs()
        + (1.0 - amplitude).max(0.0);
    objective.is_finite().then_some(BandpassEvaluation {
        controls: u.to_vec(),
        components,
        features,
        objective,
    })
}

fn random_guess(context: &fcmaes_core::RetryContext) -> Vec<f64> {
    let mut rng = Rng::new(context.seed);
    (0..context.bounds.dim()).map(|_| rng.uniform01()).collect()
}

/// Run one SO optimizer under the common parallel-retry protocol.
pub fn optimize_arm(
    optimizer: SoOptimizer,
    config: &SoConfig,
) -> Result<SoArmResult, Box<dyn Error>> {
    if config.evaluations_per_arm == 0 || config.retries == 0 {
        return Err("SO evaluations and retries must be positive".into());
    }
    if config.points < 9 {
        return Err("SO AC sweep requires at least nine points".into());
    }
    let per_retry = config.evaluations_per_arm.div_ceil(config.retries as u64);
    let bounds = RetryBounds::new(vec![0.0; BANDPASS_DIMENSION], vec![1.0; BANDPASS_DIMENSION])?;
    let objective = |u: &[f64]| {
        evaluate_bandpass(u, config.points)
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
        statistic_num: 1_000,
        ..Default::default()
    };
    let started = Instant::now();
    let result = retry(&objective, &bounds, &retry_config, |objective, context| {
        let guess = random_guess(context);
        match optimizer {
            SoOptimizer::Cma => {
                let fitness = Fitness::bounded(
                    BANDPASS_DIMENSION,
                    1,
                    context.bounds.lower(),
                    context.bounds.upper(),
                );
                let mut cma = Cmaes::new(
                    fitness,
                    &guess,
                    &[0.25],
                    &CmaesParams {
                        max_evaluations: context.max_evaluations,
                        seed: context.seed,
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
                    BANDPASS_DIMENSION,
                    1,
                    context.bounds.lower(),
                    context.bounds.upper(),
                );
                let mut de = De::new(
                    fitness,
                    &guess,
                    &[0.3; BANDPASS_DIMENSION],
                    None,
                    &DeParams {
                        popsize: 31,
                        max_evaluations: context.max_evaluations,
                        seed: context.seed,
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
                    context.bounds.lower(),
                    context.bounds.upper(),
                    Some(&guess),
                    &BiteParams {
                        max_evaluations: context.max_evaluations,
                        seed: context.seed,
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
    });
    if !result.success {
        return Err(format!("{} retry retained no finite result", optimizer.name()).into());
    }
    let best = evaluate_bandpass(&result.x, config.points)
        .ok_or_else(|| format!("{} best candidate could not be replayed", optimizer.name()))?;
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

/// One row of the arg-max versus interpolated-peak regression demonstration.
#[derive(Clone, Debug)]
pub struct SmoothnessRow {
    pub sample: usize,
    pub r1_ohm: f64,
    pub grid_peak_hz: f64,
    pub interpolated_peak_hz: f64,
}

/// AC curve and component sweep used by the feature-extraction figure.
pub type FeatureDemo = (Vec<(f64, f64)>, Vec<SmoothnessRow>);

/// Reproduce the feature-extraction demonstration used by the documentation.
pub fn feature_demo(points: usize) -> Result<FeatureDemo, Box<dyn Error>> {
    let base = [2_500.0, 10_000.0, 8_200.0, 2.2e-9, 2.2e-9];
    let base_curve = gain_curve(
        &mfb_bandpass(&base),
        "out",
        TARGET_FREQUENCY_HZ / 31.622_776_601_683_793,
        TARGET_FREQUENCY_HZ * 31.622_776_601_683_793,
        points,
    )
    .ok_or("feature-demo base circuit failed")?;
    let mut rows = Vec::new();
    for sample in 0..24 {
        let mut components = base;
        components[0] *= 0.954 + 0.004 * sample as f64;
        let curve = gain_curve(
            &mfb_bandpass(&components),
            "out",
            TARGET_FREQUENCY_HZ / 31.622_776_601_683_793,
            TARGET_FREQUENCY_HZ * 31.622_776_601_683_793,
            points,
        )
        .ok_or("feature-demo sweep circuit failed")?;
        let index = peak_index(&curve).ok_or("feature-demo curve has no peak")?;
        let smooth = interpolated_peak(&curve).ok_or("feature-demo interpolation failed")?;
        rows.push(SmoothnessRow {
            sample,
            r1_ohm: components[0],
            grid_peak_hz: curve[index].0,
            interpolated_peak_hz: smooth.0,
        });
    }
    Ok((base_curve, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_point_is_a_finite_replayable_objective() {
        let evaluation = evaluate_bandpass(&[0.5; BANDPASS_DIMENSION], 41).unwrap();
        assert!(evaluation.objective.is_finite());
        assert!(evaluation.features.peak_hz > 0.0);
        assert!(evaluation.features.q > 0.0);
    }

    #[test]
    fn tiny_retry_improves_or_matches_retained_candidate() {
        let run = optimize_arm(
            SoOptimizer::Cma,
            &SoConfig {
                evaluations_per_arm: 128,
                retries: 2,
                workers: 2,
                seed: 7,
                points: 21,
            },
        )
        .unwrap();
        assert!(run.best.objective.is_finite());
        assert!(run.actual_evaluations >= 64);
    }
}
