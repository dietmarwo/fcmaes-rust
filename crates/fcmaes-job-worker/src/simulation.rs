//! Cooperative, in-process Rust simulation context. A callback must check
//! `checkpoint` between steps; ordinary Rust code cannot be forcibly preempted.
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
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
