//! Experiment-local bounded Nelder--Mead implementation.
//!
//! This deliberately lives in the benchmark crate rather than `fcmaes-core`:
//! the experiment is deciding whether a native public refiner is justified.

use fcmaes_core::{NAN_REPLACEMENT, Objective};

#[derive(Clone, Debug)]
pub struct NmResult {
    pub x: Vec<f64>,
    pub y: f64,
    pub evaluations: u64,
}

struct Budget<'a, O: Objective> {
    objective: &'a O,
    lower: &'a [f64],
    upper: &'a [f64],
    used: u64,
    limit: u64,
}

impl<O: Objective> Budget<'_, O> {
    fn eval(&mut self, point: &mut [f64]) -> Option<f64> {
        if self.used >= self.limit {
            return None;
        }
        for ((value, low), high) in point.iter_mut().zip(self.lower).zip(self.upper) {
            *value = value.clamp(*low, *high);
        }
        self.used += 1;
        let value = self.objective.eval_scalar(point);
        Some(if value.is_finite() {
            value
        } else {
            NAN_REPLACEMENT
        })
    }
}

fn centroid(simplex: &[(Vec<f64>, f64)], dimension: usize) -> Vec<f64> {
    let mut center = vec![0.0; dimension];
    for (point, _) in &simplex[..dimension] {
        for (slot, value) in center.iter_mut().zip(point) {
            *slot += *value;
        }
    }
    for value in &mut center {
        *value /= dimension as f64;
    }
    center
}

fn along(center: &[f64], target: &[f64], factor: f64) -> Vec<f64> {
    center
        .iter()
        .zip(target)
        .map(|(base, value)| base + factor * (value - base))
        .collect()
}

fn descend<O: Objective>(
    budget: &mut Budget<'_, O>,
    origin: &[f64],
    step: &[f64],
) -> Option<(Vec<f64>, f64)> {
    let dimension = origin.len();
    if budget.limit.saturating_sub(budget.used) < dimension as u64 + 1 {
        return None;
    }

    let mut simplex = Vec::with_capacity(dimension + 1);
    let mut first = origin.to_vec();
    let first_value = budget.eval(&mut first)?;
    simplex.push((first.clone(), first_value));
    for axis in 0..dimension {
        let mut vertex = first.clone();
        vertex[axis] += step[axis];
        let mut value = budget.eval(&mut vertex)?;
        if vertex[axis].to_bits() == first[axis].to_bits() {
            vertex[axis] = first[axis] - step[axis];
            value = budget.eval(&mut vertex)?;
        }
        simplex.push((vertex, value));
    }

    // Gao--Han dimension-adaptive coefficients.
    let n = dimension as f64;
    let reflection = 1.0;
    let expansion = 1.0 + 2.0 / n;
    let contraction = 0.75 - 1.0 / (2.0 * n);
    let shrink = 1.0 - 1.0 / n;

    while budget.used < budget.limit {
        simplex.sort_by(|left, right| left.1.total_cmp(&right.1));
        let center = centroid(&simplex, dimension);
        let worst = simplex[dimension].0.clone();
        let worst_value = simplex[dimension].1;
        let second_worst = simplex[dimension - 1].1;
        let best = simplex[0].1;

        let mut reflected = along(&center, &worst, -reflection);
        let Some(reflected_value) = budget.eval(&mut reflected) else {
            break;
        };
        if reflected_value < best {
            let mut expanded = along(&center, &worst, -expansion);
            match budget.eval(&mut expanded) {
                Some(expanded_value) if expanded_value < reflected_value => {
                    simplex[dimension] = (expanded, expanded_value);
                }
                _ => simplex[dimension] = (reflected, reflected_value),
            }
            continue;
        }
        if reflected_value < second_worst {
            simplex[dimension] = (reflected, reflected_value);
            continue;
        }

        let outside = reflected_value < worst_value;
        let mut contracted = if outside {
            along(&center, &reflected, contraction)
        } else {
            along(&center, &worst, contraction)
        };
        let Some(contracted_value) = budget.eval(&mut contracted) else {
            break;
        };
        let accepted = if outside {
            contracted_value <= reflected_value
        } else {
            contracted_value < worst_value
        };
        if accepted {
            simplex[dimension] = (contracted, contracted_value);
            continue;
        }

        let anchor = simplex[0].0.clone();
        for entry in &mut simplex[1..] {
            let mut point = along(&anchor, &entry.0, shrink);
            let Some(value) = budget.eval(&mut point) else {
                simplex.sort_by(|left, right| left.1.total_cmp(&right.1));
                return simplex.first().cloned();
            };
            *entry = (point, value);
        }
    }

    simplex.sort_by(|left, right| left.1.total_cmp(&right.1));
    simplex.first().cloned()
}

/// Run bounded Nelder--Mead and deterministic shrinking restarts until the
/// exact objective-call budget is exhausted.
pub fn optimize<O: Objective>(
    objective: &O,
    guess: &[f64],
    initial_step: &[f64],
    lower: &[f64],
    upper: &[f64],
    max_evaluations: u64,
) -> NmResult {
    assert_eq!(guess.len(), initial_step.len());
    assert_eq!(guess.len(), lower.len());
    assert_eq!(guess.len(), upper.len());
    let mut budget = Budget {
        objective,
        lower,
        upper,
        used: 0,
        limit: max_evaluations,
    };
    let mut incumbent: Vec<f64> = guess
        .iter()
        .zip(lower)
        .zip(upper)
        .map(|((value, low), high)| value.clamp(*low, *high))
        .collect();
    let mut best = f64::INFINITY;
    let mut scale = 1.0;

    while budget.used < budget.limit {
        let remaining = budget.limit - budget.used;
        if remaining < incumbent.len() as u64 + 1 {
            while budget.used < budget.limit {
                let mut point = incumbent.clone();
                if let Some(value) = budget.eval(&mut point)
                    && value < best
                {
                    best = value;
                    incumbent = point;
                }
            }
            break;
        }
        let step: Vec<f64> = initial_step.iter().map(|value| value * scale).collect();
        if let Some((point, value)) = descend(&mut budget, &incumbent, &step)
            && value <= best
        {
            incumbent = point;
            best = value;
        }
        scale *= 0.3;
    }

    NmResult {
        x: incumbent,
        y: best,
        evaluations: budget.used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consumes_exact_budget_and_refines() {
        let sphere = |x: &[f64]| x.iter().map(|value| (value - 0.2).powi(2)).sum();
        let result = optimize(&sphere, &[1.0; 6], &[0.2; 6], &[-2.0; 6], &[2.0; 6], 200);
        assert_eq!(result.evaluations, 200);
        assert!(result.y < 1.0e-3, "{}", result.y);
        assert!(result.x.iter().all(|value| (-2.0..=2.0).contains(value)));
    }
}
