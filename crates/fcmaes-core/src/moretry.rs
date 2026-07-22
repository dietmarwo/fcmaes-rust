//! Parallel weighted-scalarization retry for multi-objective problems.
//!
//! This is the native counterpart of `fcmaes/moretry.py`. Every retry draws a
//! different weight vector, normalizes it with the configured p-norm, maps it
//! into the requested weight bounds, and runs an arbitrary scalar optimizer.
//! Objective values and the sampled weights are retained with each result.
//! As in [`mod@crate::retry`], workers own persistent independent PCG streams.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use crate::fitness::{NAN_REPLACEMENT, Objective};
use crate::retry::{
    RetryBounds, RetryConfig, RetryContext, RetryImprovement, RetryRunResult, run_parallel,
    spawned_worker_rng, worker_count,
};
use crate::rng::Rng;

/// A synchronized vector-valued objective.
pub trait MultiObjective: Sync {
    fn eval(&self, x: &[f64]) -> Vec<f64>;
}

impl<F> MultiObjective for F
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    fn eval(&self, x: &[f64]) -> Vec<f64> {
        self(x)
    }
}

/// Scalar view of a [`MultiObjective`] for one retry's sampled weights.
pub struct WeightedObjective<'a, O: MultiObjective> {
    objective: &'a O,
    weights: &'a [f64],
    ncon: usize,
    value_exp: f64,
}

impl<'a, O: MultiObjective> WeightedObjective<'a, O> {
    pub fn weights(&self) -> &[f64] {
        self.weights
    }

    pub fn ncon(&self) -> usize {
        self.ncon
    }

    pub fn value_exp(&self) -> f64 {
        self.value_exp
    }

    /// Evaluate the original, unscalarized objective.
    pub fn eval_multi(&self, x: &[f64]) -> Vec<f64> {
        self.objective.eval(x)
    }
}

impl<O: MultiObjective> Objective for WeightedObjective<'_, O> {
    fn nobj(&self) -> usize {
        1
    }

    fn eval(&self, x: &[f64]) -> Vec<f64> {
        vec![self.eval_scalar(x)]
    }

    #[inline]
    fn eval_scalar(&self, x: &[f64]) -> f64 {
        scalarize(
            &self.objective.eval(x),
            self.weights,
            self.ncon,
            self.value_exp,
        )
    }
}

/// Apply the `moretry.py` p-norm scalarization and its positive-constraint
/// penalty. Constraints are the final `ncon` values and are feasible at `<= 0`.
pub fn scalarize(values: &[f64], weights: &[f64], ncon: usize, value_exp: f64) -> f64 {
    if values.len() != weights.len()
        || ncon >= values.len()
        || !value_exp.is_finite()
        || value_exp <= 0.0
        || values.iter().any(|value| !value.is_finite())
        || weights.iter().any(|weight| !weight.is_finite())
    {
        return NAN_REPLACEMENT;
    }
    let powered = values
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| (value * weight).powf(value_exp))
        .sum::<f64>();
    let mut scalar = powered.powf(value_exp.recip());
    let nobj = values.len() - ncon;
    for index in nobj..values.len() {
        if values[index] > 0.0 {
            scalar += weights[index];
        }
    }
    if scalar.is_finite() {
        scalar
    } else {
        NAN_REPLACEMENT
    }
}

/// Multi-objective retry configuration.
#[derive(Clone, Debug)]
pub struct MoRetryConfig {
    pub retry: RetryConfig,
    pub weight_lower: Vec<f64>,
    pub weight_upper: Vec<f64>,
    pub ncon: usize,
    pub value_exp: f64,
    /// Optional strict upper bounds for every objective and constraint value.
    pub value_limits: Option<Vec<f64>>,
}

impl MoRetryConfig {
    pub fn new(weight_lower: Vec<f64>, weight_upper: Vec<f64>) -> Self {
        Self {
            retry: RetryConfig::default(),
            weight_lower,
            weight_upper,
            ncon: 0,
            value_exp: 2.0,
            value_limits: None,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.weight_lower.is_empty() || self.weight_lower.len() != self.weight_upper.len() {
            return Err("weight bounds must be non-empty and have equal lengths");
        }
        if self
            .weight_lower
            .iter()
            .zip(&self.weight_upper)
            .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
        {
            return Err("weight bounds must be finite and satisfy lower <= upper");
        }
        if self.ncon >= self.weight_lower.len() {
            return Err("ncon must leave at least one objective");
        }
        if !self.value_exp.is_finite() || self.value_exp <= 0.0 {
            return Err("value_exp must be finite and positive");
        }
        if self.value_limits.as_ref().is_some_and(|limits| {
            limits.len() != self.weight_lower.len() || limits.iter().any(|limit| limit.is_nan())
        }) {
            return Err("value_limits must match the objective width and not contain NaN");
        }
        Ok(())
    }
}

/// One retained weighted-scalarization result.
#[derive(Clone, Debug, PartialEq)]
pub struct MoRetryEntry {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub weights: Vec<f64>,
    pub scalar_value: f64,
}

/// Final result of [`moretry`].
#[derive(Clone, Debug)]
pub struct MoRetryResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub scalar_value: f64,
    pub evaluations: u64,
    pub runs: usize,
    pub success: bool,
    pub entries: Vec<MoRetryEntry>,
    pub improvements: Vec<RetryImprovement>,
}

struct MoStore {
    dim: usize,
    width: usize,
    capacity: usize,
    entries: Vec<MoRetryEntry>,
    evaluations: u64,
    runs: usize,
    best_scalar: f64,
    improvements: Vec<RetryImprovement>,
    statistic_num: usize,
    started: Instant,
}

impl MoStore {
    fn new(dim: usize, width: usize, capacity: usize, statistic_num: usize) -> Self {
        Self {
            dim,
            width,
            capacity: capacity.max(1),
            entries: Vec::with_capacity(capacity.max(1)),
            evaluations: 0,
            runs: 0,
            best_scalar: f64::INFINITY,
            improvements: Vec::with_capacity(statistic_num),
            statistic_num,
            started: Instant::now(),
        }
    }

    fn add(
        &mut self,
        result: RetryRunResult,
        values: Vec<f64>,
        weights: Vec<f64>,
        config: &MoRetryConfig,
    ) {
        self.runs += 1;
        self.evaluations = self.evaluations.saturating_add(result.evaluations);
        let within_limits = config.value_limits.as_ref().is_none_or(|limits| {
            values
                .iter()
                .zip(limits)
                .all(|(&value, &limit)| value < limit)
        });
        if result.x.len() != self.dim
            || values.len() != self.width
            || values.iter().any(|value| !value.is_finite())
            || !result.y.is_finite()
            || result.y >= config.retry.value_limit
            || !within_limits
        {
            return;
        }

        if result.y < self.best_scalar {
            self.best_scalar = result.y;
            if self.statistic_num > 0 {
                let sample = RetryImprovement {
                    elapsed_seconds: self.started.elapsed().as_secs_f64(),
                    evaluations: self.evaluations,
                    value: result.y,
                };
                if self.improvements.len() == self.statistic_num {
                    *self.improvements.last_mut().expect("non-empty statistics") = sample;
                } else {
                    self.improvements.push(sample);
                }
            }
        }

        if self.entries.len() >= self.capacity {
            self.entries
                .sort_unstable_by(|a, b| a.scalar_value.total_cmp(&b.scalar_value));
            let keep = ((self.capacity as f64) * 0.9).floor() as usize;
            self.entries
                .truncate(keep.max(1).min(self.capacity.saturating_sub(1)));
        }
        self.entries.push(MoRetryEntry {
            x: result.x,
            y: values,
            weights,
            scalar_value: result.y,
        });
    }

    fn into_result(mut self) -> MoRetryResult {
        self.entries
            .sort_unstable_by(|a, b| a.scalar_value.total_cmp(&b.scalar_value));
        let (x, y, scalar_value) = self.entries.first().map_or_else(
            || (Vec::new(), Vec::new(), f64::INFINITY),
            |entry| (entry.x.clone(), entry.y.clone(), entry.scalar_value),
        );
        MoRetryResult {
            x,
            y,
            scalar_value,
            evaluations: self.evaluations,
            runs: self.runs,
            success: !self.entries.is_empty(),
            entries: self.entries,
            improvements: self.improvements,
        }
    }
}

fn lock_store(store: &Mutex<MoStore>) -> MutexGuard<'_, MoStore> {
    store
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn sample_weights(config: &MoRetryConfig, rng: &mut Rng) -> Vec<f64> {
    let mut raw: Vec<f64> = (0..config.weight_lower.len())
        .map(|_| rng.uniform01())
        .collect();
    let mut norm = raw
        .iter()
        .map(|value| value.powf(config.value_exp))
        .sum::<f64>()
        .powf(config.value_exp.recip());
    if !norm.is_finite() || norm == 0.0 {
        raw.fill(0.0);
        raw[0] = 1.0;
        norm = 1.0;
    }
    raw.iter()
        .zip(&config.weight_lower)
        .zip(&config.weight_upper)
        .map(|((&value, &lo), &hi)| lo + value / norm * (hi - lo))
        .collect()
}

/// Run independent weighted-scalarization retries in parallel.
pub fn moretry<O, F>(
    objective: &O,
    bounds: &RetryBounds,
    config: &MoRetryConfig,
    optimize: F,
) -> Result<MoRetryResult, &'static str>
where
    O: MultiObjective,
    F: for<'a> Fn(&WeightedObjective<'a, O>, &RetryContext) -> RetryRunResult + Sync + Send,
{
    config.validate()?;
    let width = config.weight_lower.len();
    if config.retry.num_retries == 0 {
        return Ok(MoStore::new(
            bounds.dim(),
            width,
            config.retry.capacity,
            config.retry.statistic_num,
        )
        .into_result());
    }

    let workers = worker_count(config.retry.workers).min(config.retry.num_retries);
    let next_run = AtomicUsize::new(0);
    let stopped = AtomicBool::new(false);
    let store = Mutex::new(MoStore::new(
        bounds.dim(),
        width,
        config.retry.capacity,
        config.retry.statistic_num,
    ));

    run_parallel(workers, |worker_id| {
        let mut worker_rng = spawned_worker_rng(config.retry.seed, worker_id);
        loop {
            if stopped.load(AtomicOrdering::Relaxed) {
                break;
            }
            let run_id = next_run.fetch_add(1, AtomicOrdering::Relaxed);
            if run_id >= config.retry.num_retries {
                break;
            }
            let weights = sample_weights(config, &mut worker_rng);
            let sdev = vec![0.05 + 0.05 * worker_rng.uniform01(); bounds.dim()];
            let context = RetryContext {
                run_id,
                seed: worker_rng.next_u64(),
                bounds: bounds.clone(),
                guess: None,
                sdev,
                max_evaluations: config.retry.max_evaluations,
                value_limit: config.retry.value_limit,
                crossover: false,
            };
            let weighted = WeightedObjective {
                objective,
                weights: &weights,
                ncon: config.ncon,
                value_exp: config.value_exp,
            };
            let result = optimize(&weighted, &context);
            let values = if result.x.len() == bounds.dim() {
                objective.eval(&result.x)
            } else {
                Vec::new()
            };
            let mut shared = lock_store(&store);
            shared.add(result, values, weights, config);
            if shared.best_scalar <= config.retry.stop_fitness {
                stopped.store(true, AtomicOrdering::Relaxed);
            }
        }
    });

    Ok(store
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .into_result())
}

/// Indices of non-dominated rows considering the first `nobj` values.
pub fn pareto_indices(values: &[Vec<f64>], nobj: usize) -> Result<Vec<usize>, &'static str> {
    if nobj == 0 {
        return Err("nobj must be positive");
    }
    if values
        .iter()
        .any(|row| row.len() < nobj || row[..nobj].iter().any(|value| !value.is_finite()))
    {
        return Err("every value row must contain nobj finite values");
    }
    let mut front = Vec::new();
    for candidate in 0..values.len() {
        let dominated = (0..values.len()).any(|other| {
            other != candidate
                && (0..nobj).all(|j| values[other][j] <= values[candidate][j])
                && (0..nobj).any(|j| values[other][j] < values[candidate][j])
        });
        if !dominated {
            front.push(candidate);
        }
    }
    front.sort_by(|&left, &right| values[left][0].total_cmp(&values[right][0]));
    Ok(front)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> RetryBounds {
        RetryBounds::new(vec![-2.0, -2.0], vec![2.0, 2.0]).unwrap()
    }

    #[test]
    fn scalarization_matches_python_formula_and_penalty() {
        assert_eq!(scalarize(&[3.0, 4.0], &[1.0, 1.0], 0, 2.0), 5.0);
        let penalized = scalarize(&[3.0, 4.0, 0.5], &[1.0, 1.0, 2.0], 1, 2.0);
        assert!((penalized - (26.0_f64.sqrt() + 2.0)).abs() < 1e-12);
        assert_eq!(scalarize(&[1.0], &[1.0, 2.0], 0, 2.0), NAN_REPLACEMENT);
    }

    #[test]
    fn validates_configuration() {
        assert!(
            MoRetryConfig::new(Vec::new(), Vec::new())
                .validate()
                .is_err()
        );
        let mut config = MoRetryConfig::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        config.ncon = 2;
        assert!(config.validate().is_err());
        config.ncon = 0;
        config.value_exp = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn weighted_retry_is_deterministic_and_retains_vectors() {
        let objective = |x: &[f64]| vec![x[0] * x[0], (x[1] - 1.0).powi(2)];
        let mut config = MoRetryConfig::new(vec![0.5, 0.5], vec![1.5, 1.5]);
        config.retry = RetryConfig {
            num_retries: 12,
            workers: 1,
            capacity: 5,
            seed: 123,
            statistic_num: 3,
            ..Default::default()
        };
        let run = |weighted: &WeightedObjective<'_, _>, context: &RetryContext| {
            let mut rng = Rng::new(context.seed);
            let x = vec![-2.0 + 4.0 * rng.uniform01(), -2.0 + 4.0 * rng.uniform01()];
            RetryRunResult {
                y: weighted.eval_scalar(&x),
                x,
                evaluations: 1,
            }
        };
        let first = moretry(&objective, &bounds(), &config, run).unwrap();
        let second = moretry(&objective, &bounds(), &config, run).unwrap();
        assert_eq!(first.entries, second.entries);
        assert_eq!(first.runs, 12);
        assert_eq!(first.evaluations, 12);
        assert!(first.success);
        assert!(first.entries.len() <= 5);
        assert!(first.entries.iter().all(|entry| entry.y.len() == 2));
        assert!(first.improvements.len() <= 3);
    }

    #[test]
    fn value_limits_filter_and_stop_works() {
        let objective = |_: &[f64]| vec![0.0, 2.0];
        let mut config = MoRetryConfig::new(vec![1.0, 1.0], vec![1.0, 1.0]);
        config.value_limits = Some(vec![1.0, 1.0]);
        config.retry.num_retries = 3;
        config.retry.workers = 1;
        let filtered = moretry(&objective, &bounds(), &config, |weighted, _| {
            let x = vec![0.0, 0.0];
            RetryRunResult {
                y: weighted.eval_scalar(&x),
                x,
                evaluations: 1,
            }
        })
        .unwrap();
        assert!(!filtered.success);
        assert_eq!(filtered.runs, 3);

        config.value_limits = None;
        config.retry.stop_fitness = 3.0;
        config.retry.num_retries = 20;
        let stopped = moretry(&objective, &bounds(), &config, |weighted, _| {
            let x = vec![0.0, 0.0];
            RetryRunResult {
                y: weighted.eval_scalar(&x),
                x,
                evaluations: 1,
            }
        })
        .unwrap();
        assert_eq!(stopped.runs, 1);
    }

    #[test]
    fn pareto_indices_handles_tradeoffs_duplicates_and_dominance() {
        let values = vec![
            vec![0.0, 2.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 2.0],
            vec![1.0, 1.0],
        ];
        assert_eq!(pareto_indices(&values, 2).unwrap(), vec![0, 1, 4, 2]);
        assert!(pareto_indices(&values, 0).is_err());
    }
}
