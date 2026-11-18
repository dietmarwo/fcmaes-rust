//! Shared contract between the benchmark harness and every optimizer it runs.
//!
//! A [`Solver`] minimizes a scalar box-bounded objective under a hard
//! evaluation budget. The contract is deliberately minimal so that incumbent
//! `fcmaes-core` optimizers and candidate algorithms are compared through the
//! same call.
//!
//! Contract every implementation must satisfy:
//!
//! - **Budget:** `objective` is called at most `max_evaluations` times.
//! - **Bounds:** every evaluated point satisfies `lower[i] <= x[i] <= upper[i]`.
//! - **Determinism:** the same `seed` and inputs give the same evaluation
//!   sequence.
//! - **Serial:** a solver evaluates on the calling thread only. The harness
//!   owns all parallelism (independent runs on separate threads).
//! - **Non-finite values:** `NaN` and infinite objective values are treated
//!   as worse than every finite value; they never panic the solver.
//!
//! [`Guard`] enforces the first two points and records the best value seen,
//! so results never depend on a solver's own bookkeeping.

#![forbid(unsafe_code)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Scalar objective passed to a solver. It is `Sync` because the incumbent
/// `fcmaes-core` optimizers require it; solvers still evaluate serially.
pub type Objective<'a> = &'a (dyn Fn(&[f64]) -> f64 + Sync);

/// What a solver reports back. The harness trusts [`Guard`], not these fields.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// Best point found.
    pub x: Vec<f64>,
    /// Objective value at [`x`](Self::x).
    pub y: f64,
    /// Objective calls the solver made.
    pub evaluations: u64,
}

/// A box-bounded scalar minimizer with a hard evaluation budget.
pub trait Solver: Sync {
    /// Stable identifier used in result files.
    fn name(&self) -> &'static str;

    /// Minimize `objective` inside `[lower, upper]`.
    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome;
}

/// Budget and bounds guard around an objective.
///
/// Calls beyond the budget and calls outside the box are counted as contract
/// violations and answered with `f64::INFINITY` without evaluating the
/// objective, so a violating solver can never gain from them.
pub struct Guard<'a> {
    inner: Objective<'a>,
    lower: &'a [f64],
    upper: &'a [f64],
    budget: u64,
    calls: AtomicU64,
    over_budget: AtomicU64,
    out_of_bounds: AtomicU64,
    best: Mutex<(f64, Vec<f64>)>,
}

impl<'a> Guard<'a> {
    /// Wrap `inner` with a budget of `budget` calls inside `[lower, upper]`.
    pub fn new(inner: Objective<'a>, lower: &'a [f64], upper: &'a [f64], budget: u64) -> Self {
        Self {
            inner,
            lower,
            upper,
            budget,
            calls: AtomicU64::new(0),
            over_budget: AtomicU64::new(0),
            out_of_bounds: AtomicU64::new(0),
            best: Mutex::new((f64::INFINITY, Vec::new())),
        }
    }

    /// Guarded evaluation; pass `&|x| guard.eval(x)` to the solver.
    pub fn eval(&self, x: &[f64]) -> f64 {
        let inside = x.len() == self.lower.len()
            && x.iter()
                .zip(self.lower.iter().zip(self.upper.iter()))
                .all(|(v, (lo, hi))| *v >= *lo && *v <= *hi);
        if !inside {
            self.out_of_bounds.fetch_add(1, Ordering::Relaxed);
            return f64::INFINITY;
        }
        // Reserve a budget slot first so concurrent callers cannot overshoot.
        let reserved = self
            .calls
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| {
                (c < self.budget).then_some(c + 1)
            });
        if reserved.is_err() {
            self.over_budget.fetch_add(1, Ordering::Relaxed);
            return f64::INFINITY;
        }
        let y = (self.inner)(x);
        let mut best = self
            .best
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if y < best.0 {
            best.0 = y;
            best.1 = x.to_vec();
        }
        y
    }

    /// Objective calls actually made (never above the budget).
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    /// Calls refused because the budget was exhausted.
    pub fn over_budget(&self) -> u64 {
        self.over_budget.load(Ordering::Relaxed)
    }

    /// Calls refused because the point was outside the box.
    pub fn out_of_bounds(&self) -> u64 {
        self.out_of_bounds.load(Ordering::Relaxed)
    }

    /// Best finite objective value seen, or `f64::INFINITY`.
    pub fn best_y(&self) -> f64 {
        self.best
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .0
    }

    /// Point at which [`best_y`](Self::best_y) was seen.
    pub fn best_x(&self) -> Vec<f64> {
        self.best
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .1
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_refuses_over_budget_and_out_of_bounds() {
        let f = |x: &[f64]| x[0] * x[0];
        let (lo, hi) = ([-1.0], [1.0]);
        let g = Guard::new(&f, &lo, &hi, 2);
        assert_eq!(g.eval(&[0.5]), 0.25);
        assert_eq!(g.eval(&[2.0]), f64::INFINITY);
        assert_eq!(g.eval(&[0.1]), 0.1 * 0.1);
        assert_eq!(g.eval(&[0.0]), f64::INFINITY);
        assert_eq!((g.calls(), g.over_budget(), g.out_of_bounds()), (2, 1, 1));
        assert_eq!(g.best_y(), 0.1 * 0.1);
        assert_eq!(g.best_x(), vec![0.1]);
    }
}
