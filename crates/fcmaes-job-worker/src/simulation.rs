//! Cooperative, in-process Rust simulation context. A callback must check
//! `checkpoint` between steps; ordinary Rust code cannot be forcibly preempted.
use crate::{Control, registry::Scalar};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Clone)]
pub struct SimContext {
    cancelled: Arc<AtomicBool>,
    deadline: Instant,
    threads: usize,
    seed: u64,
}
impl SimContext {
    pub(crate) fn new(
        cancelled: Arc<AtomicBool>,
        deadline: Instant,
        threads: usize,
        seed: u64,
    ) -> Self {
        Self {
            cancelled,
            deadline,
            threads,
            seed,
        }
    }
    /// True on job cancellation or at the absolute deadline.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || Instant::now() >= self.deadline
    }
    pub fn checkpoint(&self) -> Result<(), String> {
        if self.is_cancelled() {
            Err("cancelled".into())
        } else {
            Ok(())
        }
    }
    pub fn remaining_seconds(&self) -> f64 {
        self.deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64()
    }
    /// Per-evaluation allowance; nested Rayon work shares the task's pool.
    pub fn threads(&self) -> usize {
        self.threads
    }
    /// Stable for the same point and run, including its independent recheck.
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub(crate) fn at_point(&self, point: &[f64]) -> Self {
        let mut seed = self.seed;
        for (i, &x) in point.iter().enumerate() {
            let bits = if x == 0.0 { 0 } else { x.to_bits() };
            seed = fcmaes_surrogate::stream_seed(seed, bits, i as u64);
        }
        Self {
            seed,
            ..self.clone()
        }
    }
}

/// The instant `remaining` seconds from now (at most about 30 years ahead).
pub(crate) fn deadline(remaining: f64) -> Instant {
    Instant::now() + Duration::from_secs_f64(remaining.clamp(0.0, 1e9))
}

/// Why a simulation gave no value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    /// The job was cancelled or reached its deadline before or during the
    /// evaluation.
    Cancelled,
    /// The callback returned this error.
    Failed(String),
}

/// Distinct failure messages a result reports, with their counts.
const FAILURE_MESSAGES: usize = 3;
/// Longest failure message a result reports.
const FAILURE_MESSAGE_CHARS: usize = 200;

/// A scalar model as `fcmaes.retry/1` and `fcmaes.evaluate/1` call it. A
/// `Scalar::new` model is called directly. A `Scalar::simulation` model
/// gets a [`SimContext`] with one thread, a seed derived from `seed` and the
/// point, the task's remaining time and the job's cancellation; for an
/// optimizer ([`Evaluator::value`]), an evaluation that fails or is cancelled
/// is `+inf` (the worst value) and counted.
pub(crate) struct Evaluator<'a> {
    model: &'a Scalar,
    context: Option<SimContext>,
    failed: AtomicU64,
    cancelled: AtomicU64,
    /// The smallest distinct failure messages with their counts, which do
    /// not depend on the order in which parallel evaluations fail.
    messages: Mutex<BTreeMap<String, u64>>,
}

impl<'a> Evaluator<'a> {
    pub(crate) fn new(model: &'a Scalar, control: &Control, remaining: f64, seed: u64) -> Self {
        let context = model
            .is_simulation()
            .then(|| SimContext::new(control.token(), deadline(remaining), 1, seed));
        Self {
            model,
            context,
            failed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            messages: Mutex::new(BTreeMap::new()),
        }
    }
    pub(crate) fn is_simulation(&self) -> bool {
        self.context.is_some()
    }
    /// True once the job of a simulation model is cancelled or at its
    /// deadline; always false for a plain model.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.context.as_ref().is_some_and(SimContext::is_cancelled)
    }
    /// The model's value at `x`, without counting.
    pub(crate) fn try_value(&self, x: &[f64]) -> Result<f64, Failure> {
        let Some(context) = &self.context else {
            return Ok((self.model.evaluate)(x));
        };
        if context.is_cancelled() {
            return Err(Failure::Cancelled);
        }
        match self.model.value(x, &context.at_point(x)) {
            Ok(value) => Ok(value),
            // An error while the job stops is cancelled work, not a failure.
            Err(_) if context.is_cancelled() => Err(Failure::Cancelled),
            Err(message) => Err(Failure::Failed(message)),
        }
    }
    /// The value an optimizer ranks: the model's value, or `+inf` for a
    /// failed or cancelled evaluation, which is counted.
    pub(crate) fn value(&self, x: &[f64]) -> f64 {
        match self.try_value(x) {
            Ok(value) => value,
            Err(Failure::Cancelled) => {
                self.cancelled.fetch_add(1, Ordering::Relaxed);
                f64::INFINITY
            }
            Err(Failure::Failed(message)) => {
                self.failed.fetch_add(1, Ordering::Relaxed);
                self.record(message);
                f64::INFINITY
            }
        }
    }
    fn record(&self, message: String) {
        let message: String = message.chars().take(FAILURE_MESSAGE_CHARS).collect();
        let mut messages = self.messages.lock().unwrap_or_else(PoisonError::into_inner);
        // Keeping the smallest messages keeps exact counts for them: one of
        // the final messages is never the largest of a full map.
        *messages.entry(message).or_insert(0) += 1;
        if messages.len() > FAILURE_MESSAGES {
            messages.pop_last();
        }
    }
    pub(crate) fn failed(&self) -> u64 {
        self.failed.load(Ordering::Relaxed)
    }
    pub(crate) fn cancelled(&self) -> u64 {
        self.cancelled.load(Ordering::Relaxed)
    }
    /// Up to three distinct failure messages (the smallest) with counts.
    pub(crate) fn messages(&self) -> BTreeMap<String, u64> {
        self.messages
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}
