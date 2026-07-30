//! Deterministic routing-aware witness improvement.

use std::time::{Duration, Instant};

use crate::decode::{Decoded, Route};
use crate::evaluate::{EvalConfig, SolutionMetrics, evaluate, feasible};
use crate::instance::Instance;

/// Result of one frozen-operation structural baseline.
#[derive(Clone, Debug)]
pub struct BaselineResult {
    /// Improved explicit routes.
    pub decoded: Decoded,
    /// Final nominal metrics.
    pub metrics: SolutionMetrics,
    /// Attempted insertion and 2-opt moves.
    pub operations: usize,
    /// Whether greedy construction had to fall back to the feasible witness.
    pub construction_fallback: bool,
    /// Wall time, reported but not budgeted.
    pub elapsed: Duration,
}

fn witness(instance: &Instance) -> Decoded {
    let routes = instance
        .witness_routes
        .iter()
        .enumerate()
        .map(|(vehicle, tasks)| Route {
            vehicle,
            tasks: tasks.clone(),
        })
        .collect::<Vec<_>>();
    Decoded {
        used_vehicles: routes
            .iter()
            .filter(|route| !route.tasks.is_empty())
            .count(),
        routes,
    }
}

/// Improve the known feasible construction with best-accept 2-opt.
#[must_use]
pub fn optimize(instance: &Instance, max_operations: usize) -> BaselineResult {
    let started = Instant::now();
    let mut decoded = Decoded {
        routes: (0..instance.vehicles.len())
            .map(|vehicle| Route {
                vehicle,
                tasks: Vec::new(),
            })
            .collect(),
        used_vehicles: 0,
    };
    let mut operations = 0;
    let mut construction_fallback = false;
    let mut tasks = instance
        .tasks
        .iter()
        .filter(|task| task.base)
        .map(|task| task.id)
        .collect::<Vec<_>>();
    tasks.sort_by(|left, right| {
        instance.tasks[*left]
            .latest_s
            .total_cmp(&instance.tasks[*right].latest_s)
            .then_with(|| left.cmp(right))
    });
    for task in tasks {
        let mut best: Option<(Decoded, SolutionMetrics)> = None;
        for vehicle in &instance.vehicles {
            if vehicle.skills & instance.tasks[task].skill == 0 {
                continue;
            }
            for position in 0..=decoded.routes[vehicle.id].tasks.len() {
                if operations >= max_operations {
                    break;
                }
                operations += 1;
                let mut candidate = decoded.clone();
                candidate.routes[vehicle.id].tasks.insert(position, task);
                candidate.used_vehicles = candidate
                    .routes
                    .iter()
                    .filter(|route| !route.tasks.is_empty())
                    .count();
                let metrics = evaluate(&candidate, instance, EvalConfig::default());
                if feasible(&metrics)
                    && best
                        .as_ref()
                        .is_none_or(|(_, incumbent): &(Decoded, SolutionMetrics)| {
                            metrics.cost < incumbent.cost
                        })
                {
                    best = Some((candidate, metrics));
                }
            }
        }
        if let Some((candidate, _)) = best {
            decoded = candidate;
        } else {
            decoded = witness(instance);
            construction_fallback = true;
            break;
        }
    }
    let mut current = evaluate(&decoded, instance, EvalConfig::default());
    'search: loop {
        let mut best = None;
        for vehicle in 0..decoded.routes.len() {
            let route_len = decoded.routes[vehicle].tasks.len();
            for left in 0..route_len.saturating_sub(1) {
                for right in left + 1..route_len {
                    if operations >= max_operations {
                        break 'search;
                    }
                    operations += 1;
                    let mut candidate = decoded.clone();
                    candidate.routes[vehicle].tasks[left..=right].reverse();
                    let metrics = evaluate(&candidate, instance, EvalConfig::default());
                    if feasible(&metrics)
                        && metrics.cost + 1.0e-12 < current.cost
                        && best.as_ref().is_none_or(
                            |(_, incumbent): &(Decoded, SolutionMetrics)| {
                                metrics.cost < incumbent.cost
                            },
                        )
                    {
                        best = Some((candidate, metrics));
                    }
                }
            }
        }
        match best {
            Some((candidate, metrics)) => {
                decoded = candidate;
                current = metrics;
            }
            None => break,
        }
    }
    BaselineResult {
        decoded,
        metrics: current,
        operations,
        construction_fallback,
        elapsed: started.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{SEEDS, generate};

    #[test]
    fn baseline_is_feasible_and_never_worsens_witness() {
        let instance = generate(SEEDS[0], 0);
        let original = evaluate(&witness(&instance), &instance, EvalConfig::default());
        let improved = optimize(&instance, 10_000);
        assert!(feasible(&improved.metrics));
        assert!(improved.metrics.cost.is_finite());
        assert!(improved.operations <= 10_000);
        if improved.construction_fallback {
            assert!((improved.metrics.cost - original.cost).abs() < 1.0e-9);
        }
    }
}
