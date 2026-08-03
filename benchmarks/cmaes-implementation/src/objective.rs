use std::hint::spin_loop;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::problems::Problem;

#[derive(Clone)]
pub struct SharedObjective {
    state: Arc<State>,
}

struct State {
    problem: Problem,
    calls: AtomicU64,
    best_bits: AtomicU64,
    injected_cost: Duration,
    start: Instant,
}

impl SharedObjective {
    pub fn new(problem: &Problem, injected_cost_ns: u64) -> Self {
        Self {
            state: Arc::new(State {
                problem: problem.clone(),
                calls: AtomicU64::new(0),
                best_bits: AtomicU64::new(f64::INFINITY.to_bits()),
                injected_cost: Duration::from_nanos(injected_cost_ns),
                start: Instant::now(),
            }),
        }
    }

    #[inline]
    pub fn evaluate(&self, normalized: &[f64]) -> f64 {
        self.state.calls.fetch_add(1, Ordering::Relaxed);
        let start = Instant::now();
        let point: Vec<f64> = normalized
            .iter()
            .zip(
                self.state
                    .problem
                    .lower
                    .iter()
                    .zip(&self.state.problem.upper),
            )
            .map(|(&value, (&lower, &upper))| {
                let phase = ((value + 1.0) * 0.5).rem_euclid(2.0);
                let unit = if phase <= 1.0 { phase } else { 2.0 - phase };
                lower + unit * (upper - lower)
            })
            .collect();
        let value = self.state.problem.evaluate(&point);
        while start.elapsed() < self.state.injected_cost {
            spin_loop();
        }
        self.record_best(value);
        value
    }

    fn record_best(&self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let mut current = self.state.best_bits.load(Ordering::Relaxed);
        loop {
            let current_value = f64::from_bits(current);
            if value >= current_value {
                break;
            }
            match self.state.best_bits.compare_exchange_weak(
                current,
                value.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    pub fn calls(&self) -> u64 {
        self.state.calls.load(Ordering::Relaxed)
    }

    pub fn best(&self) -> f64 {
        f64::from_bits(self.state.best_bits.load(Ordering::Relaxed))
    }

    pub fn elapsed(&self) -> Duration {
        self.state.start.elapsed()
    }
}

pub fn calibrate(problem: &Problem, injected_cost_ns: u64) -> f64 {
    let samples = match injected_cost_ns {
        0 => 20_000,
        1..=9_999 => 5_000,
        10_000..=999_999 => 200,
        _ => 5,
    };
    let objective = SharedObjective::new(problem, injected_cost_ns);
    let point = problem.initial_mean();
    for _ in 0..samples {
        std::hint::black_box(objective.evaluate(std::hint::black_box(&point)));
    }
    objective.elapsed().as_nanos() as f64 / samples as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::problems;

    #[test]
    fn reflection_maps_to_the_same_point_in_each_period() {
        let problem = problems().remove(0);
        let objective = SharedObjective::new(&problem, 0);
        let center = objective.evaluate(&[0.25; 10]);
        let reflected = objective.evaluate(&[3.75; 10]);
        assert!((center - reflected).abs() < 1e-12);
        assert_eq!(objective.calls(), 2);
    }

    #[test]
    fn best_is_shared_between_clones() {
        let problem = problems().remove(0);
        let first = SharedObjective::new(&problem, 0);
        let second = first.clone();
        first.evaluate(&[0.5; 10]);
        second.evaluate(&[0.0; 10]);
        assert_eq!(first.best(), 0.0);
        assert_eq!(first.calls(), 2);
    }
}
