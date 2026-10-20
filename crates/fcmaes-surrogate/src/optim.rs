//! Deterministic bound-constrained L-BFGS with a projected Armijo search.
//!
//! This is a compact projected quasi-Newton method rather than full L-BFGS-B:
//! variables held at a bound by their gradient are frozen for the iteration,
//! the two-loop recursion runs on the free subspace, and the step is projected
//! back into the box. Values are evaluated before gradients, so rejected
//! line-search trials never pay for a gradient.
use crate::{Error, Result, linalg::dot};
use std::collections::VecDeque;

const MEMORY: usize = 8;
const ARMIJO: f64 = 1e-4;
const BACKTRACKS: usize = 30;

pub(crate) trait Objective {
    type State;
    /// Value at `x`; `None` marks a numerical failure, treated as `+inf`.
    fn value(&self, x: &[f64]) -> Option<(f64, Self::State)>;
    fn gradient(&self, x: &[f64], state: &Self::State) -> Vec<f64>;
}

pub(crate) struct Settings<'a> {
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    pub iterations: usize,
    /// Largest coordinate change of the first, unscaled step.
    pub first_step: f64,
    /// Relative decrease below which the search stops.
    pub tolerance: f64,
}

/// The state at the last accepted point, which callers keep (the GP fit keeps
/// the factorization), and the number of value evaluations.
pub(crate) struct Minimum<S> {
    pub state: S,
    pub evaluations: usize,
}

fn project(x: &mut [f64], lower: &[f64], upper: &[f64]) {
    for ((v, &l), &u) in x.iter_mut().zip(lower).zip(upper) {
        *v = v.clamp(l, u);
    }
}

pub(crate) fn minimize<O: Objective>(
    objective: &O,
    start: &[f64],
    settings: &Settings<'_>,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Minimum<O::State>> {
    let (lower, upper) = (settings.lower, settings.upper);
    let d = start.len();
    let mut x = start.to_vec();
    project(&mut x, lower, upper);
    let (mut f, mut state) = objective
        .value(&x)
        .filter(|(f, _)| f.is_finite())
        .ok_or_else(|| Error::Numerical("nonfinite value at the start point".into()))?;
    let mut evaluations = 1;
    let mut g = objective.gradient(&x, &state);
    let mut memory: VecDeque<(Vec<f64>, Vec<f64>, f64)> = VecDeque::with_capacity(MEMORY);
    let mut free = vec![true; d];
    for _ in 0..settings.iterations {
        if stop() {
            return Err(Error::Stopped);
        }
        if g.iter().any(|v| !v.is_finite()) {
            break;
        }
        let mut largest = 0.0_f64;
        for i in 0..d {
            free[i] = !((x[i] <= lower[i] && g[i] > 0.0) || (x[i] >= upper[i] && g[i] < 0.0));
            if free[i] {
                largest = largest.max(g[i].abs());
            }
        }
        if largest == 0.0 {
            break;
        }
        let masked = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(&free)
                .map(|(&v, &f)| if f { v } else { 0.0 })
                .collect()
        };
        // Two-loop recursion on the free subspace.
        let mut q = masked(&g);
        let mut alphas = Vec::with_capacity(memory.len());
        for (s, y, rho) in memory.iter().rev() {
            let a = rho * dot(&masked(s), &q);
            let y = masked(y);
            for (q, y) in q.iter_mut().zip(&y) {
                *q -= a * y;
            }
            alphas.push(a);
        }
        let mut step_limit = f64::INFINITY;
        if let Some((s, y, _)) = memory.back() {
            let (s, y) = (masked(s), masked(y));
            let yy = dot(&y, &y);
            let sy = dot(&s, &y);
            if yy > 0.0 && sy > 0.0 {
                for v in &mut q {
                    *v *= sy / yy;
                }
            }
        } else {
            step_limit = settings.first_step;
        }
        for ((s, y, rho), a) in memory.iter().zip(alphas.iter().rev()) {
            let b = rho * dot(&masked(y), &q);
            let s = masked(s);
            for (q, s) in q.iter_mut().zip(&s) {
                *q += (a - b) * s;
            }
        }
        let mut p: Vec<f64> = q.iter().map(|v| -v).collect();
        let slope = dot(&p, &g);
        if slope.is_nan() || slope >= 0.0 || p.iter().any(|v| !v.is_finite()) {
            memory.clear();
            p = masked(&g).iter().map(|v| -v).collect();
            step_limit = settings.first_step;
        }
        let longest = p.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut t = if longest > step_limit {
            step_limit / longest
        } else {
            1.0
        };
        let mut accepted = None;
        for _ in 0..BACKTRACKS {
            let mut trial: Vec<f64> = x.iter().zip(&p).map(|(x, p)| x + t * p).collect();
            project(&mut trial, lower, upper);
            let change: Vec<f64> = trial.iter().zip(&x).map(|(a, b)| a - b).collect();
            if change.iter().all(|v| *v == 0.0) {
                break;
            }
            evaluations += 1;
            if let Some((value, trial_state)) = objective.value(&trial)
                && value.is_finite()
                && value <= f + ARMIJO * dot(&g, &change)
            {
                accepted = Some((trial, value, trial_state, change));
                break;
            }
            t *= 0.5;
        }
        let Some((trial, value, trial_state, s)) = accepted else {
            break;
        };
        let gradient = objective.gradient(&trial, &trial_state);
        let y: Vec<f64> = gradient.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-10 * libm::sqrt(dot(&s, &s) * dot(&y, &y)) && sy.is_finite() {
            if memory.len() == MEMORY {
                memory.pop_front();
            }
            memory.push_back((s, y, 1.0 / sy));
        }
        let decrease = f - value;
        x = trial;
        f = value;
        state = trial_state;
        g = gradient;
        if decrease <= settings.tolerance * f.abs().max(1.0) {
            break;
        }
    }
    Ok(Minimum { state, evaluations })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rosenbrock;
    impl Objective for Rosenbrock {
        type State = (Vec<f64>, f64);
        fn value(&self, x: &[f64]) -> Option<(f64, Self::State)> {
            let f = x
                .windows(2)
                .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
                .sum();
            Some((f, (x.to_vec(), f)))
        }
        fn gradient(&self, x: &[f64], _: &Self::State) -> Vec<f64> {
            let mut g = vec![0.0; x.len()];
            for i in 0..x.len() - 1 {
                let t = x[i + 1] - x[i] * x[i];
                g[i] += -400.0 * t * x[i] - 2.0 * (1.0 - x[i]);
                g[i + 1] += 200.0 * t;
            }
            g
        }
    }

    #[test]
    fn converges_inside_and_stops_at_active_bounds() {
        let free = minimize(
            &Rosenbrock,
            &[-1.2, 1.0, -0.5, 0.3],
            &Settings {
                lower: &[-2.0; 4],
                upper: &[2.0; 4],
                iterations: 500,
                first_step: 0.5,
                tolerance: 0.0,
            },
            &|| false,
        )
        .unwrap()
        .state;
        assert!(free.1 < 1e-10, "{}", free.1);
        let bounded = minimize(
            &Rosenbrock,
            &[0.0; 4],
            &Settings {
                lower: &[-2.0; 4],
                upper: &[0.5; 4],
                iterations: 500,
                first_step: 0.5,
                tolerance: 0.0,
            },
            &|| false,
        )
        .unwrap()
        .state
        .0;
        assert!(bounded.iter().all(|v| *v <= 0.5));
        assert!(bounded[0] > 0.45, "{bounded:?}");
        assert!(matches!(
            minimize(
                &Rosenbrock,
                &[0.0; 2],
                &Settings {
                    lower: &[-1.0; 2],
                    upper: &[1.0; 2],
                    iterations: 5,
                    first_step: 0.5,
                    tolerance: 0.0
                },
                &|| true
            ),
            Err(Error::Stopped)
        ));
    }
}
