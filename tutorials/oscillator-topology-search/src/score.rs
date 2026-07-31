//! Three-trace oscillation scoring adapted from the fixed Vilar tutorial.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::config::{
    MAX_MOLECULES, SAMPLE_COUNT, SAMPLE_INTERVAL, TARGET_PERIOD, TRAINING_SEEDS, VALIDATION_SEEDS,
};
use crate::grammar::{GENES, Topology};
use crate::network::{Trace, simulate};

#[derive(Debug)]
struct SpectrumPlan {
    bins: Vec<(usize, Vec<[f64; 2]>)>,
}

fn spectrum_plan() -> &'static SpectrumPlan {
    static PLAN: OnceLock<SpectrumPlan> = OnceLock::new();
    PLAN.get_or_init(|| SpectrumPlan {
        bins: (2..=16)
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
            .collect(),
    })
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

fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return if values.is_empty() { 1.0 } else { 0.0 };
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

#[derive(Clone, Copy, Debug)]
struct SignalMetrics {
    period: f64,
    amplitude: f64,
    spectral_concentration: f64,
    autocorrelation_decay: f64,
    participates: bool,
}

fn analyze_signal(signal: &[f64]) -> SignalMetrics {
    if signal.len() != SAMPLE_COUNT || signal.iter().any(|value| !value.is_finite()) {
        return SignalMetrics {
            period: 64.0,
            amplitude: 0.0,
            spectral_concentration: 0.0,
            autocorrelation_decay: 2.0,
            participates: false,
        };
    }
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let centered: Vec<f64> = signal.iter().map(|value| value - mean).collect();
    let sum_squares = centered.iter().map(|value| value * value).sum::<f64>();
    let mut ordered = signal.to_vec();
    ordered.sort_unstable_by(f64::total_cmp);
    let amplitude = ordered[ordered.len() * 9 / 10] - ordered[ordered.len() / 10];

    let powers: Vec<(usize, f64)> = spectrum_plan()
        .bins
        .iter()
        .map(|(bin, weights)| {
            let (real, imaginary) = centered.iter().zip(weights).fold(
                (0.0, 0.0),
                |(real, imaginary), (value, weight)| {
                    (real + value * weight[0], imaginary - value * weight[1])
                },
            );
            (*bin, real.mul_add(real, imaginary * imaginary))
        })
        .collect();
    let peak = powers
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.1.total_cmp(&right.1.1))
        .map_or(0, |(index, _)| index);
    let peak_bin = powers[peak].0 as f64;
    let interpolated = if peak > 0 && peak + 1 < powers.len() {
        let left = powers[peak - 1].1.max(1.0e-300).ln();
        let center = powers[peak].1.max(1.0e-300).ln();
        let right = powers[peak + 1].1.max(1.0e-300).ln();
        let denominator = left - 2.0 * center + right;
        peak_bin
            + if denominator.abs() > 1.0e-12 {
                (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
            } else {
                0.0
            }
    } else {
        peak_bin
    };
    let period = SAMPLE_COUNT as f64 * SAMPLE_INTERVAL / interpolated;
    let band_power = powers[peak].1
        + peak.checked_sub(1).map_or(0.0, |index| powers[index].1)
        + powers.get(peak + 1).map_or(0.0, |entry| entry.1);
    let spectral_concentration = if sum_squares > 0.0 {
        (2.0 * band_power / (SAMPLE_COUNT as f64 * sum_squares)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let lag = (period / SAMPLE_INTERVAL).round() as usize;
    let one = correlation(&centered, lag);
    let two = correlation(&centered, 2 * lag);
    let autocorrelation_decay = (1.0 - two.max(0.0) + 0.25 * (one - two).max(0.0)).min(2.0);
    SignalMetrics {
        period,
        amplitude,
        spectral_concentration,
        autocorrelation_decay,
        participates: amplitude >= 5.0 && spectral_concentration >= 0.05,
    }
}

/// Metrics for one stochastic path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicateMetrics {
    pub seed: u64,
    pub period: f64,
    pub amplitude: f64,
    pub spectral_concentration: f64,
    pub autocorrelation_decay: f64,
    pub participation: f64,
    pub period_cv_across_genes: f64,
    pub mean_molecules: f64,
    pub failed: bool,
}

/// Aggregate objective and emergent descriptors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metrics {
    pub scalar_score: f64,
    pub period: f64,
    pub period_error: f64,
    pub amplitude: f64,
    pub spectral_concentration: f64,
    pub autocorrelation_decay: f64,
    pub participation: f64,
    pub period_cv: f64,
    pub mean_molecules: f64,
    pub failure_fraction: f64,
    pub replicates: Vec<ReplicateMetrics>,
}

/// Analyze a path without running a simulator.
pub fn analyze_trace(trace: &Trace) -> ReplicateMetrics {
    let signals: [Vec<f64>; GENES] =
        std::array::from_fn(|gene| trace.values.iter().map(|state| state[gene]).collect());
    let signal_metrics: [SignalMetrics; GENES] =
        std::array::from_fn(|gene| analyze_signal(&signals[gene]));
    let participating: Vec<_> = signal_metrics
        .iter()
        .filter(|metrics| metrics.participates)
        .collect();
    let count = participating.len().max(1) as f64;
    let period = if participating.is_empty() {
        64.0
    } else {
        participating
            .iter()
            .map(|metrics| metrics.period)
            .sum::<f64>()
            / count
    };
    let amplitude = signal_metrics
        .iter()
        .map(|metrics| metrics.amplitude)
        .sum::<f64>()
        / GENES as f64;
    let spectral_concentration = signal_metrics
        .iter()
        .map(|metrics| metrics.spectral_concentration)
        .sum::<f64>()
        / GENES as f64;
    let autocorrelation_decay = signal_metrics
        .iter()
        .map(|metrics| metrics.autocorrelation_decay)
        .sum::<f64>()
        / GENES as f64;
    let participation = participating.len() as f64 / GENES as f64;
    let periods: Vec<f64> = participating.iter().map(|metrics| metrics.period).collect();
    let mean_molecules = if trace.values.is_empty() {
        MAX_MOLECULES
    } else {
        trace
            .values
            .iter()
            .map(|state| state.iter().sum::<f64>())
            .sum::<f64>()
            / trace.values.len() as f64
    };
    ReplicateMetrics {
        seed: trace.seed,
        period,
        amplitude,
        spectral_concentration,
        autocorrelation_decay,
        participation,
        period_cv_across_genes: coefficient_of_variation(&periods),
        mean_molecules,
        failed: trace.runaway || trace.values.len() != SAMPLE_COUNT || participation == 0.0,
    }
}

/// Aggregate named replications into the minimized scalar objective.
pub fn aggregate(replicates: Vec<ReplicateMetrics>) -> Metrics {
    let count = replicates.len().max(1) as f64;
    let average = |selector: fn(&ReplicateMetrics) -> f64| {
        replicates.iter().map(selector).sum::<f64>() / count
    };
    let period = average(|run| run.period);
    let amplitude = average(|run| run.amplitude);
    let spectral_concentration = average(|run| run.spectral_concentration);
    let autocorrelation_decay = average(|run| run.autocorrelation_decay);
    let participation = average(|run| run.participation);
    let mean_molecules = average(|run| run.mean_molecules);
    let failure_fraction = replicates.iter().filter(|run| run.failed).count() as f64 / count;
    let periods: Vec<f64> = replicates.iter().map(|run| run.period).collect();
    let replicate_period_cv = coefficient_of_variation(&periods);
    let gene_period_cv = average(|run| run.period_cv_across_genes);
    let period_cv = replicate_period_cv + gene_period_cv;
    let period_error = (period - TARGET_PERIOD).abs() / TARGET_PERIOD;
    let amplitude_penalty = ((10.0 - amplitude) / 10.0).max(0.0);
    let scalar_score = period_error
        + 2.0 * (1.0 - spectral_concentration)
        + amplitude_penalty
        + 0.5 * autocorrelation_decay
        + 2.0 * (1.0 - participation)
        + period_cv
        + 5.0 * failure_fraction
        + 0.0002 * mean_molecules;
    Metrics {
        scalar_score,
        period,
        period_error,
        amplitude,
        spectral_concentration,
        autocorrelation_decay,
        participation,
        period_cv,
        mean_molecules,
        failure_fraction,
        replicates,
    }
}

/// Evaluate on an explicit seed slice.
pub fn evaluate(topology: &Topology, decision: &[f64], seeds: &[u64]) -> Metrics {
    let replicates = seeds
        .iter()
        .map(|&seed| {
            simulate(topology, decision, seed).map_or_else(
                |_| ReplicateMetrics {
                    seed,
                    period: 64.0,
                    amplitude: 0.0,
                    spectral_concentration: 0.0,
                    autocorrelation_decay: 2.0,
                    participation: 0.0,
                    period_cv_across_genes: 1.0,
                    mean_molecules: MAX_MOLECULES,
                    failed: true,
                },
                |trace| analyze_trace(&trace),
            )
        })
        .collect();
    aggregate(replicates)
}

/// Common-random-number training score.
pub fn training(topology: &Topology, decision: &[f64], replications: usize) -> Metrics {
    evaluate(topology, decision, &TRAINING_SEEDS[..replications])
}

/// Disjoint post-optimization validation score.
pub fn validation(topology: &Topology, decision: &[f64], replications: usize) -> Metrics {
    evaluate(topology, decision, &VALIDATION_SEEDS[..replications])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analytic_trace(periods: [f64; 3], amplitudes: [f64; 3]) -> Trace {
        let values = (0..SAMPLE_COUNT)
            .map(|index| {
                std::array::from_fn(|gene| {
                    30.0 + amplitudes[gene]
                        * (std::f64::consts::TAU * index as f64 / periods[gene]).sin()
                })
            })
            .collect();
        Trace {
            seed: 1,
            time: (0..SAMPLE_COUNT).map(|value| value as f64).collect(),
            values,
            runaway: false,
        }
    }

    #[test]
    fn analytic_period_and_amplitude_are_recovered() {
        let metrics = analyze_trace(&analytic_trace([32.0; 3], [10.0; 3]));
        assert!(
            (metrics.period - 32.0).abs() < 0.1,
            "period={}",
            metrics.period
        );
        assert!((metrics.amplitude - 19.0).abs() < 2.0);
        assert_eq!(metrics.participation, 1.0);
    }

    #[test]
    fn broad_participation_beats_one_active_trace() {
        let broad = aggregate(vec![analyze_trace(&analytic_trace([24.0; 3], [10.0; 3]))]);
        let narrow = aggregate(vec![analyze_trace(&analytic_trace(
            [24.0; 3],
            [10.0, 0.0, 0.0],
        ))]);
        assert!(broad.scalar_score < narrow.scalar_score);
    }

    #[test]
    fn target_period_beats_shifted_and_flat_traces() {
        let target = aggregate(vec![analyze_trace(&analytic_trace([24.0; 3], [10.0; 3]))]);
        let shifted = aggregate(vec![analyze_trace(&analytic_trace([48.0; 3], [10.0; 3]))]);
        let flat = aggregate(vec![analyze_trace(&analytic_trace([24.0; 3], [0.0; 3]))]);
        assert!(target.scalar_score < shifted.scalar_score);
        assert!(target.scalar_score < flat.scalar_score);
        assert_eq!(flat.failure_fraction, 1.0);
    }
}
