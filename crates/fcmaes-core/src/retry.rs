//! Parallel optimization restart coordinators.
//!
//! This is the native coordination core of the former `retry.py` and
//! `advretry.py` implementations. A caller supplies one objective and a
//! restart closure; the coordinator owns scheduling, independently spawned
//! worker random streams, result retention, early stopping, and advanced
//! crossover.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use rayon::prelude::*;

use crate::rng::Rng;

/// Validated finite box bounds shared by all retries.
#[derive(Clone, Debug, PartialEq)]
pub struct RetryBounds {
    lower: Arc<[f64]>,
    upper: Arc<[f64]>,
}

impl RetryBounds {
    /// Construct bounds, rejecting empty, mismatched, non-finite, or reversed
    /// intervals.
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self, &'static str> {
        if lower.is_empty() || lower.len() != upper.len() {
            return Err("bounds must be non-empty and have equal lengths");
        }
        if lower
            .iter()
            .zip(&upper)
            .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo >= hi)
        {
            return Err("bounds must contain finite intervals with lower < upper");
        }
        Ok(Self {
            lower: lower.into(),
            upper: upper.into(),
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.lower.len()
    }

    #[inline]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    #[inline]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }
}

/// Inputs for one independent optimizer run.
#[derive(Clone, Debug)]
pub struct RetryContext {
    pub run_id: usize,
    pub seed: u64,
    pub bounds: RetryBounds,
    pub guess: Option<Vec<f64>>,
    pub sdev: Vec<f64>,
    pub max_evaluations: u64,
    /// A crossover result is retained only when it improves this parent.
    pub value_limit: f64,
    pub crossover: bool,
}

/// Result returned by a caller-provided restart optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct RetryRunResult {
    pub x: Vec<f64>,
    pub y: f64,
    pub evaluations: u64,
}

/// One retained retry result.
#[derive(Clone, Debug, PartialEq)]
pub struct RetryEntry {
    pub x: Vec<f64>,
    pub y: f64,
}

/// Progress sample registered whenever a completed retry improves the best
/// retained objective value.
#[derive(Clone, Debug, PartialEq)]
pub struct RetryImprovement {
    pub elapsed_seconds: f64,
    pub evaluations: u64,
    pub value: f64,
}

/// Final output common to basic and coordinated retry.
#[derive(Clone, Debug)]
pub struct RetryResult {
    pub x: Vec<f64>,
    pub y: f64,
    pub evaluations: u64,
    pub runs: usize,
    pub success: bool,
    pub entries: Vec<RetryEntry>,
    pub improvements: Vec<RetryImprovement>,
}

/// Configuration corresponding to the scheduling and store controls in
/// `retry.py`.
#[derive(Clone, Debug)]
pub struct RetryConfig {
    pub num_retries: usize,
    /// `0` uses all available CPUs.
    pub workers: usize,
    pub capacity: usize,
    pub value_limit: f64,
    pub stop_fitness: f64,
    pub max_evaluations: u64,
    pub seed: u64,
    pub statistic_num: usize,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            num_retries: 1_024,
            workers: 0,
            capacity: 500,
            value_limit: f64::INFINITY,
            stop_fitness: f64::NEG_INFINITY,
            max_evaluations: 50_000,
            seed: 0,
            statistic_num: 0,
        }
    }
}

/// Additional controls corresponding to `advretry.py`.
#[derive(Clone, Debug)]
pub struct AdvancedRetryConfig {
    pub retry: RetryConfig,
    pub check_interval: usize,
    pub max_eval_fac: f64,
    pub crossover_probability: f64,
    pub diversity_threshold: f64,
}

impl Default for AdvancedRetryConfig {
    fn default() -> Self {
        Self {
            retry: RetryConfig {
                num_retries: 5_000,
                max_evaluations: 1_500,
                ..Default::default()
            },
            check_interval: 100,
            max_eval_fac: 50.0,
            crossover_probability: 0.5,
            diversity_threshold: 0.15,
        }
    }
}

#[derive(Debug)]
struct RetryStore {
    dim: usize,
    capacity: usize,
    entries: Vec<RetryEntry>,
    best_x: Vec<f64>,
    best_y: f64,
    evaluations: u64,
    completed_runs: usize,
    improvements: Vec<RetryImprovement>,
    statistic_num: usize,
    started: Instant,
}

impl RetryStore {
    fn new(dim: usize, capacity: usize, statistic_num: usize) -> Self {
        Self {
            dim,
            capacity: capacity.max(1),
            entries: Vec::with_capacity(capacity.max(1)),
            best_x: vec![0.0; dim],
            best_y: f64::INFINITY,
            evaluations: 0,
            completed_runs: 0,
            improvements: Vec::with_capacity(statistic_num),
            statistic_num,
            started: Instant::now(),
        }
    }

    fn add(&mut self, result: RetryRunResult, limit: f64) -> bool {
        self.completed_runs += 1;
        self.evaluations = self.evaluations.saturating_add(result.evaluations);
        if result.x.len() != self.dim || !result.y.is_finite() || result.y >= limit {
            return false;
        }

        let improved = result.y < self.best_y;
        if improved {
            self.best_y = result.y;
            self.best_x.clone_from(&result.x);
            if self.statistic_num > 0 {
                let sample = RetryImprovement {
                    elapsed_seconds: self.started.elapsed().as_secs_f64(),
                    evaluations: self.evaluations,
                    value: result.y,
                };
                if self.improvements.len() == self.statistic_num {
                    if let Some(last) = self.improvements.last_mut() {
                        *last = sample;
                    }
                } else {
                    self.improvements.push(sample);
                }
            }
        }

        if self.entries.len() >= self.capacity {
            self.sort_basic();
            if self.entries.len() >= self.capacity {
                self.entries.pop();
            }
        }
        self.entries.push(RetryEntry {
            x: result.x,
            y: result.y,
        });
        improved
    }

    fn sort_basic(&mut self) {
        self.entries.sort_unstable_by(|a, b| a.y.total_cmp(&b.y));
        let keep = ((self.capacity as f64) * 0.9).floor() as usize;
        self.entries.truncate(keep.max(1).min(self.capacity));
    }

    #[cfg(test)]
    fn normalized_distance(&self, a: &[f64], b: &[f64], bounds: &RetryBounds) -> f64 {
        let squared = a
            .iter()
            .zip(b)
            .zip(bounds.lower().iter().zip(bounds.upper()))
            .map(|((&av, &bv), (&lo, &hi))| ((av - bv) / (hi - lo)).powi(2))
            .sum::<f64>();
        (squared / self.dim as f64).sqrt()
    }

    fn sort_diverse(&mut self, bounds: &RetryBounds, threshold: f64) {
        self.entries.sort_unstable_by(|a, b| a.y.total_cmp(&b.y));
        let mut diverse = Vec::with_capacity(self.entries.len());
        for entry in self.entries.drain(..) {
            let sufficiently_different =
                diverse.iter().rev().take(2).all(|previous: &RetryEntry| {
                    let squared = previous
                        .x
                        .iter()
                        .zip(&entry.x)
                        .zip(bounds.lower().iter().zip(bounds.upper()))
                        .map(|((&a, &b), (&lo, &hi))| ((a - b) / (hi - lo)).powi(2))
                        .sum::<f64>();
                    (squared / self.dim as f64).sqrt() > threshold
                });
            if sufficiently_different {
                diverse.push(entry);
            }
        }
        let keep = ((self.capacity as f64) * 0.9).floor() as usize;
        diverse.truncate(keep.max(1).min(self.capacity));
        self.entries = diverse;
    }

    fn into_result(mut self) -> RetryResult {
        self.entries.sort_unstable_by(|a, b| a.y.total_cmp(&b.y));
        RetryResult {
            x: self.best_x,
            y: self.best_y,
            evaluations: self.evaluations,
            runs: self.completed_runs,
            success: self.best_y.is_finite(),
            entries: self.entries,
            improvements: self.improvements,
        }
    }
}

fn lock_store(store: &Mutex<RetryStore>) -> MutexGuard<'_, RetryStore> {
    store
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) fn worker_count(requested: usize) -> usize {
    if requested > 0 {
        requested
    } else {
        std::thread::available_parallelism().map_or(1, usize::from)
    }
}

/// SplitMix64 finalizer used to initialize spawned worker streams.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Spawn one persistent PCG stream per retry worker. Distinct PCG stream
/// selectors provide the same independence property sought by NumPy's
/// `SeedSequence.spawn(workers)` without sharing mutable RNG state.
pub(crate) fn spawned_worker_rng(root_seed: u64, worker_id: usize) -> Rng {
    let worker = worker_id as u64;
    let state = ((splitmix64(root_seed ^ 0xD2B7_4407_B1CE_6E93 ^ worker) as u128) << 64)
        | splitmix64(root_seed ^ 0xCA5A_8263_9512_1157 ^ worker) as u128;
    // The low half makes stream selectors unique for every worker. The high
    // half separates equal worker indices under different root seeds.
    let stream = ((splitmix64(root_seed ^ 0x9E37_79B9_7F4A_7C15) as u128) << 64) | worker as u128;
    Rng::from_state_stream(state, stream)
}

fn initial_sdev(dim: usize, rng: &mut Rng) -> Vec<f64> {
    let value = 0.05 + 0.05 * rng.uniform01();
    vec![value; dim]
}

pub(crate) fn run_parallel<F>(workers: usize, task: F)
where
    F: Fn(usize) + Sync + Send,
{
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("fcmaes-retry-{index}"))
        .build()
        .expect("failed to build retry worker pool")
        .install(|| {
            (0..workers).into_par_iter().for_each(task);
        });
}

/// Run independent restarts in parallel. The optimizer closure is invoked
/// exactly once for every claimed run unless another worker already reached
/// `stop_fitness`.
pub fn retry<O, F>(
    objective: &O,
    bounds: &RetryBounds,
    config: &RetryConfig,
    optimize: F,
) -> RetryResult
where
    O: Fn(&[f64]) -> f64 + Sync,
    F: Fn(&O, &RetryContext) -> RetryRunResult + Sync + Send,
{
    if config.num_retries == 0 {
        return RetryStore::new(bounds.dim(), config.capacity, config.statistic_num).into_result();
    }
    let workers = worker_count(config.workers).min(config.num_retries);
    let next_run = AtomicUsize::new(0);
    let stopped = AtomicBool::new(false);
    let store = Mutex::new(RetryStore::new(
        bounds.dim(),
        config.capacity,
        config.statistic_num,
    ));

    run_parallel(workers, |worker_id| {
        let mut worker_rng = spawned_worker_rng(config.seed, worker_id);
        loop {
            if stopped.load(AtomicOrdering::Relaxed) {
                break;
            }
            let run_id = next_run.fetch_add(1, AtomicOrdering::Relaxed);
            if run_id >= config.num_retries {
                break;
            }
            let sdev = initial_sdev(bounds.dim(), &mut worker_rng);
            let context = RetryContext {
                run_id,
                seed: worker_rng.next_u64(),
                bounds: bounds.clone(),
                guess: None,
                sdev,
                max_evaluations: config.max_evaluations,
                value_limit: config.value_limit,
                crossover: false,
            };
            let result = optimize(objective, &context);
            let mut shared = lock_store(&store);
            shared.add(result, config.value_limit);
            if shared.best_y <= config.stop_fitness {
                stopped.store(true, AtomicOrdering::Relaxed);
            }
        }
    });

    store
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .into_result()
}

fn advanced_context(
    run_id: usize,
    bounds: &RetryBounds,
    config: &AdvancedRetryConfig,
    store: &mut RetryStore,
    worker_rng: &mut Rng,
) -> RetryContext {
    if config.check_interval > 0 && run_id > 0 && run_id.is_multiple_of(config.check_interval) {
        store.sort_diverse(bounds, config.diversity_threshold.max(0.0));
    }

    let progress = if config.retry.num_retries <= 1 {
        1.0
    } else {
        run_id as f64 / (config.retry.num_retries - 1) as f64
    };
    let factor = 1.0 + (config.max_eval_fac.max(1.0) - 1.0) * progress;
    let max_evaluations = ((config.retry.max_evaluations as f64) * factor)
        .round()
        .clamp(1.0, u64::MAX as f64) as u64;

    let try_crossover = worker_rng.uniform01() < config.crossover_probability.clamp(0.0, 1.0);
    let use_crossover = store.entries.len() >= 2 && try_crossover;
    if !use_crossover {
        let sdev = initial_sdev(bounds.dim(), worker_rng);
        return RetryContext {
            run_id,
            seed: worker_rng.next_u64(),
            bounds: bounds.clone(),
            guess: None,
            sdev,
            max_evaluations,
            value_limit: config.retry.value_limit,
            crossover: false,
        };
    }

    // The store is sorted at every checkpoint and on capacity pressure. Bias
    // both parents toward its best 20%, while keeping them distinct.
    let elite = ((store.entries.len() as f64 * 0.2).ceil() as usize)
        .max(2)
        .min(store.entries.len());
    let first = ((worker_rng.uniform01().powi(2) * elite as f64) as usize).min(elite - 1);
    let mut second = ((worker_rng.uniform01().powi(2) * elite as f64) as usize).min(elite - 1);
    if first == second {
        second = (second + 1) % elite;
    }
    let parent = &store.entries[first];
    let donor = &store.entries[second];
    let diff_fac = 0.5 + 0.5 * worker_rng.uniform01();
    let limit_fac = (2.0 + 2.0 * worker_rng.uniform01()) * diff_fac;
    let mut lower = Vec::with_capacity(bounds.dim());
    let mut upper = Vec::with_capacity(bounds.dim());
    let mut guess = Vec::with_capacity(bounds.dim());
    let mut sdev = Vec::with_capacity(bounds.dim());
    for i in 0..bounds.dim() {
        let global_delta = bounds.upper()[i] - bounds.lower()[i];
        let delta = (donor.x[i] - parent.x[i]).abs();
        let local_delta = (limit_fac * delta).max(0.0001);
        let lo = bounds.lower()[i].max(parent.x[i] - local_delta);
        let hi = bounds.upper()[i].min(parent.x[i] + local_delta);
        lower.push(lo);
        upper.push(hi);
        guess.push(donor.x[i].clamp(lo, hi));
        sdev.push((diff_fac * delta / global_delta).clamp(0.001, 0.5));
    }

    RetryContext {
        run_id,
        seed: worker_rng.next_u64(),
        bounds: RetryBounds::new(lower, upper).expect("crossover bounds are valid"),
        guess: Some(guess),
        sdev,
        max_evaluations,
        value_limit: parent.y.min(config.retry.value_limit),
        crossover: true,
    }
}

/// Run adaptive-budget retries with coordinated crossover guesses and
/// diversity-preserving result retention.
pub fn advanced_retry<O, F>(
    objective: &O,
    bounds: &RetryBounds,
    config: &AdvancedRetryConfig,
    optimize: F,
) -> RetryResult
where
    O: Fn(&[f64]) -> f64 + Sync,
    F: Fn(&O, &RetryContext) -> RetryRunResult + Sync + Send,
{
    if config.retry.num_retries == 0 {
        return RetryStore::new(
            bounds.dim(),
            config.retry.capacity,
            config.retry.statistic_num,
        )
        .into_result();
    }
    let workers = worker_count(config.retry.workers).min(config.retry.num_retries);
    let next_run = AtomicUsize::new(0);
    let stopped = AtomicBool::new(false);
    let store = Mutex::new(RetryStore::new(
        bounds.dim(),
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
            let context = {
                let mut shared = lock_store(&store);
                advanced_context(run_id, bounds, config, &mut shared, &mut worker_rng)
            };
            let limit = context.value_limit;
            let result = optimize(objective, &context);
            let mut shared = lock_store(&store);
            shared.add(result, limit);
            if shared.best_y <= config.retry.stop_fitness {
                stopped.store(true, AtomicOrdering::Relaxed);
            }
        }
    });

    let mut store = store
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    store.sort_diverse(bounds, config.diversity_threshold.max(0.0));
    store.into_result()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> RetryBounds {
        RetryBounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).unwrap()
    }

    fn sample_run<O: Fn(&[f64]) -> f64>(objective: &O, context: &RetryContext) -> RetryRunResult {
        let mut rng = Rng::new(context.seed);
        let x: Vec<f64> = (0..context.bounds.dim())
            .map(|i| {
                context.bounds.lower()[i]
                    + rng.uniform01() * (context.bounds.upper()[i] - context.bounds.lower()[i])
            })
            .collect();
        RetryRunResult {
            y: objective(&x),
            x,
            evaluations: 1,
        }
    }

    #[test]
    fn rejects_invalid_bounds() {
        assert!(RetryBounds::new(vec![], vec![]).is_err());
        assert!(RetryBounds::new(vec![0.0], vec![1.0, 2.0]).is_err());
        assert!(RetryBounds::new(vec![1.0], vec![1.0]).is_err());
        assert!(RetryBounds::new(vec![f64::NAN], vec![1.0]).is_err());
    }

    #[test]
    fn single_worker_retry_is_deterministic_and_counts() {
        let config = RetryConfig {
            num_retries: 40,
            workers: 1,
            capacity: 8,
            seed: 123,
            statistic_num: 3,
            ..Default::default()
        };
        let objective = |x: &[f64]| x.iter().map(|v| v * v).sum();
        let first = retry(&objective, &bounds(), &config, sample_run);
        let second = retry(&objective, &bounds(), &config, sample_run);
        assert!(first.success);
        assert_eq!(first.y, second.y);
        assert_eq!(first.x, second.x);
        assert_eq!(first.runs, 40);
        assert_eq!(first.evaluations, 40);
        assert!(first.entries.len() <= config.capacity);
        assert!(first.improvements.len() <= 3);
    }

    #[test]
    fn spawned_worker_streams_are_independent_and_reproducible() {
        let sample = |root_seed| {
            (0..8)
                .map(|worker_id| {
                    let mut rng = spawned_worker_rng(root_seed, worker_id);
                    (0..16).map(|_| rng.next_u64()).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let first = sample(123);
        assert_eq!(first, sample(123));
        assert_ne!(first, sample(124));
        for left in 0..first.len() {
            for right in left + 1..first.len() {
                assert_ne!(first[left], first[right]);
            }
        }
    }

    #[test]
    fn basic_retry_draws_contexts_from_the_persistent_worker_stream() {
        let observed = Mutex::new(Vec::new());
        let config = RetryConfig {
            num_retries: 3,
            workers: 1,
            seed: 321,
            ..Default::default()
        };
        retry(&|_: &[f64]| 0.0, &bounds(), &config, |_, context| {
            observed
                .lock()
                .unwrap()
                .push((context.sdev[0], context.seed));
            RetryRunResult {
                x: vec![0.0; context.bounds.dim()],
                y: 0.0,
                evaluations: 1,
            }
        });

        let mut worker_rng = spawned_worker_rng(config.seed, 0);
        let expected: Vec<(f64, u64)> = (0..config.num_retries)
            .map(|_| {
                let sdev = initial_sdev(bounds().dim(), &mut worker_rng)[0];
                (sdev, worker_rng.next_u64())
            })
            .collect();
        assert_eq!(observed.into_inner().unwrap(), expected);
    }

    #[test]
    fn advanced_retry_draws_contexts_from_the_persistent_worker_stream() {
        let observed = Mutex::new(Vec::new());
        let config = AdvancedRetryConfig {
            retry: RetryConfig {
                num_retries: 3,
                workers: 1,
                seed: 654,
                ..Default::default()
            },
            crossover_probability: 0.0,
            ..Default::default()
        };
        advanced_retry(&|_: &[f64]| 0.0, &bounds(), &config, |_, context| {
            observed
                .lock()
                .unwrap()
                .push((context.sdev[0], context.seed));
            RetryRunResult {
                x: vec![0.0; context.bounds.dim()],
                y: 0.0,
                evaluations: 1,
            }
        });

        let mut worker_rng = spawned_worker_rng(config.retry.seed, 0);
        let expected: Vec<(f64, u64)> = (0..config.retry.num_retries)
            .map(|_| {
                let _crossover_draw = worker_rng.uniform01();
                let sdev = initial_sdev(bounds().dim(), &mut worker_rng)[0];
                (sdev, worker_rng.next_u64())
            })
            .collect();
        assert_eq!(observed.into_inner().unwrap(), expected);
    }

    #[test]
    fn retry_filters_bad_results_and_empty_runs() {
        let empty = retry(
            &|_: &[f64]| 0.0,
            &bounds(),
            &RetryConfig {
                num_retries: 0,
                ..Default::default()
            },
            sample_run,
        );
        assert!(!empty.success);
        assert!(empty.y.is_infinite());

        let filtered = retry(
            &|_: &[f64]| 2.0,
            &bounds(),
            &RetryConfig {
                num_retries: 3,
                workers: 1,
                value_limit: 1.0,
                ..Default::default()
            },
            sample_run,
        );
        assert!(!filtered.success);
        assert!(filtered.entries.is_empty());
        assert_eq!(filtered.runs, 3);
    }

    #[test]
    fn stop_fitness_stops_early() {
        let result = retry(
            &|_: &[f64]| -1.0,
            &bounds(),
            &RetryConfig {
                num_retries: 100,
                workers: 1,
                stop_fitness: 0.0,
                ..Default::default()
            },
            sample_run,
        );
        assert_eq!(result.runs, 1);
        assert_eq!(result.y, -1.0);
    }

    #[test]
    fn advanced_retry_increases_budget_and_crosses_over() {
        let contexts = Mutex::new(Vec::new());
        let config = AdvancedRetryConfig {
            retry: RetryConfig {
                num_retries: 12,
                workers: 1,
                capacity: 10,
                max_evaluations: 100,
                seed: 99,
                ..Default::default()
            },
            check_interval: 2,
            max_eval_fac: 4.0,
            crossover_probability: 1.0,
            diversity_threshold: 0.0,
        };
        let result = advanced_retry(&|x: &[f64]| x[0], &bounds(), &config, |objective, ctx| {
            contexts.lock().unwrap().push(ctx.clone());
            sample_run(objective, ctx)
        });
        let contexts = contexts.into_inner().unwrap();
        assert_eq!(contexts.first().unwrap().max_evaluations, 100);
        assert_eq!(contexts.last().unwrap().max_evaluations, 400);
        assert!(contexts.iter().skip(2).any(|context| context.crossover));
        assert!(
            contexts
                .iter()
                .filter(|context| context.crossover)
                .all(|context| context.guess.is_some())
        );
        assert!(result.success);
    }

    #[test]
    fn advanced_retry_handles_single_run_and_filters_dimension_mismatch() {
        let config = AdvancedRetryConfig {
            retry: RetryConfig {
                num_retries: 1,
                workers: 0,
                max_evaluations: 7,
                ..Default::default()
            },
            check_interval: 0,
            max_eval_fac: 3.0,
            ..Default::default()
        };
        let result = advanced_retry(&|_: &[f64]| 0.0, &bounds(), &config, |_, context| {
            assert_eq!(context.max_evaluations, 21);
            RetryRunResult {
                x: vec![0.0],
                y: 0.0,
                evaluations: 5,
            }
        });
        assert!(!result.success);
        assert_eq!(result.runs, 1);
        assert_eq!(result.evaluations, 5);

        let empty = advanced_retry(
            &|_: &[f64]| 0.0,
            &bounds(),
            &AdvancedRetryConfig {
                retry: RetryConfig {
                    num_retries: 0,
                    ..Default::default()
                },
                ..Default::default()
            },
            sample_run,
        );
        assert!(!empty.success);
        assert_eq!(empty.runs, 0);
    }

    #[test]
    fn store_diversity_and_distance() {
        let bounds = bounds();
        let mut store = RetryStore::new(2, 10, 0);
        for (x, y) in [
            (vec![0.0, 0.0], 0.0),
            (vec![0.01, 0.01], 1.0),
            (vec![4.0, 4.0], 2.0),
        ] {
            store.add(
                RetryRunResult {
                    x,
                    y,
                    evaluations: 1,
                },
                f64::INFINITY,
            );
        }
        assert!(store.normalized_distance(&[0.0, 0.0], &[5.0, 5.0], &bounds) > 0.0);
        store.sort_diverse(&bounds, 0.15);
        assert_eq!(store.entries.len(), 2);
        assert_eq!(store.entries[0].y, 0.0);

        let mut tiny = RetryStore::new(2, 1, 0);
        for value in [3.0, 2.0, 1.0] {
            tiny.add(
                RetryRunResult {
                    x: vec![value; 2],
                    y: value,
                    evaluations: 1,
                },
                f64::INFINITY,
            );
        }
        assert_eq!(tiny.entries.len(), 1);
        assert_eq!(tiny.best_y, 1.0);
    }

    #[test]
    fn advanced_stop_fitness_stops_early() {
        let result = advanced_retry(
            &|_: &[f64]| -2.0,
            &bounds(),
            &AdvancedRetryConfig {
                retry: RetryConfig {
                    num_retries: 20,
                    workers: 1,
                    stop_fitness: -1.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            sample_run,
        );
        assert_eq!(result.runs, 1);
        assert_eq!(result.y, -2.0);
    }
}
