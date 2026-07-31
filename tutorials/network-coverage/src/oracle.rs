//! Independently verifiable classic vertex-cover certificates.

use std::error::Error;

use microlp::{ComparisonOp, OptimizationDirection, Problem, SolutionStatus};

use crate::instance::Instance;

fn can_remove(instance: &Instance, selected: &[bool], node: usize) -> bool {
    instance.incident_edges[node].iter().all(|edge_index| {
        let edge = instance.edges[*edge_index];
        let other = if edge.u == node { edge.v } else { edge.u };
        selected[other]
    })
}

fn reverse_delete(instance: &Instance, selected: &mut [bool], weighted: bool) {
    let mut order = (0..instance.nodes())
        .filter(|node| selected[*node])
        .collect::<Vec<_>>();
    order.sort_by(|left, right| {
        if weighted {
            instance.costs[*right]
                .total_cmp(&instance.costs[*left])
                .then_with(|| left.cmp(right))
        } else {
            instance.incident_edges[*left]
                .len()
                .cmp(&instance.incident_edges[*right].len())
                .then_with(|| left.cmp(right))
        }
    });
    for node in order {
        if can_remove(instance, selected, node) {
            selected[node] = false;
        }
    }
}

/// Independently verify that every ordinary edge has a selected endpoint.
#[must_use]
pub fn verify_cover(instance: &Instance, selected: &[bool]) -> bool {
    selected.len() == instance.nodes()
        && instance
            .edges
            .iter()
            .all(|edge| selected[edge.u] || selected[edge.v])
}

/// Cardinality 2-approximation and matching lower-bound certificate.
#[derive(Clone, Debug)]
pub struct CardinalityCertificate {
    /// Verified cover.
    pub selected: Vec<bool>,
    /// Maximal-matching size and certified lower bound.
    pub lower_bound: usize,
}

/// Deterministic maximal-matching endpoint cover followed by reverse deletion.
#[must_use]
pub fn matching_cover(instance: &Instance) -> CardinalityCertificate {
    let mut matched = vec![false; instance.nodes()];
    let mut selected = vec![false; instance.nodes()];
    let mut lower_bound = 0;
    for edge in &instance.edges {
        if !matched[edge.u] && !matched[edge.v] {
            matched[edge.u] = true;
            matched[edge.v] = true;
            selected[edge.u] = true;
            selected[edge.v] = true;
            lower_bound += 1;
        }
    }
    reverse_delete(instance, &mut selected, false);
    CardinalityCertificate {
        selected,
        lower_bound,
    }
}

/// Weighted primal-dual 2-approximation and dual lower-bound certificate.
#[derive(Clone, Debug)]
pub struct WeightedCertificate {
    /// Verified weighted cover.
    pub selected: Vec<bool>,
    /// Feasible dual objective and certified lower bound.
    pub lower_bound: f64,
    /// Selected cover cost.
    pub cost: f64,
}

/// Standard deterministic primal-dual weighted cover followed by reverse deletion.
#[must_use]
pub fn weighted_primal_dual(instance: &Instance) -> WeightedCertificate {
    let mut residual = instance.costs.clone();
    let mut selected = vec![false; instance.nodes()];
    let mut lower_bound = 0.0;
    let tolerance = 1.0e-12;
    for edge in &instance.edges {
        if selected[edge.u] || selected[edge.v] {
            continue;
        }
        let delta = residual[edge.u].min(residual[edge.v]).max(0.0);
        residual[edge.u] -= delta;
        residual[edge.v] -= delta;
        lower_bound += delta;
        if residual[edge.u] <= tolerance {
            selected[edge.u] = true;
        }
        if residual[edge.v] <= tolerance {
            selected[edge.v] = true;
        }
    }
    reverse_delete(instance, &mut selected, true);
    let cost = selected
        .iter()
        .zip(&instance.costs)
        .filter(|(chosen, _)| **chosen)
        .map(|(_, cost)| cost)
        .sum();
    WeightedCertificate {
        selected,
        lower_bound,
        cost,
    }
}

/// Exact binary-program result.
#[derive(Clone, Debug)]
pub struct ExactCover {
    /// Selected nodes.
    pub selected: Vec<bool>,
    /// Proven objective.
    pub objective: f64,
}

/// Solve cardinality or weighted vertex cover with `microlp`.
pub fn exact_cover(instance: &Instance, weighted: bool) -> Result<ExactCover, Box<dyn Error>> {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let variables = instance
        .costs
        .iter()
        .map(|cost| problem.add_binary_var(if weighted { *cost } else { 1.0 }))
        .collect::<Vec<_>>();
    for edge in &instance.edges {
        problem.add_constraint(
            [(variables[edge.u], 1.0), (variables[edge.v], 1.0)],
            ComparisonOp::Ge,
            1.0,
        );
    }
    let solution = problem
        .solve()?
        .into_solution()
        .map_err(|error| format!("microlp solve was interrupted: {error:?}"))?;
    if solution.status() != SolutionStatus::Optimal {
        return Err("microlp returned a feasible but unproven cover".into());
    }
    let selected = variables
        .iter()
        .map(|variable| solution.var_value(*variable) > 0.5)
        .collect::<Vec<_>>();
    if !verify_cover(instance, &selected) {
        return Err("microlp assignment failed independent verification".into());
    }
    Ok(ExactCover {
        selected,
        objective: solution.objective(),
    })
}

/// Exhaustive tiny-instance optimum for an independent ILP test.
#[must_use]
pub fn brute_force(instance: &Instance, weighted: bool) -> ExactCover {
    assert!(instance.nodes() <= 20);
    let mut best = ExactCover {
        selected: vec![true; instance.nodes()],
        objective: f64::INFINITY,
    };
    for pattern in 0_u64..1 << instance.nodes() {
        let selected = (0..instance.nodes())
            .map(|node| pattern & (1 << node) != 0)
            .collect::<Vec<_>>();
        if !verify_cover(instance, &selected) {
            continue;
        }
        let objective = selected
            .iter()
            .zip(&instance.costs)
            .filter(|(chosen, _)| **chosen)
            .map(|(_, cost)| if weighted { *cost } else { 1.0 })
            .sum::<f64>();
        if objective < best.objective {
            best = ExactCover {
                selected,
                objective,
            };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{FIXTURES, generate};

    #[test]
    fn certificates_are_feasible_and_respect_their_own_bounds() {
        for config in &FIXTURES[..3] {
            let instance = generate(config).unwrap();
            let cardinality = matching_cover(&instance);
            assert!(verify_cover(&instance, &cardinality.selected));
            assert!(
                cardinality.selected.iter().filter(|value| **value).count()
                    <= 2 * cardinality.lower_bound
            );
            let weighted = weighted_primal_dual(&instance);
            assert!(verify_cover(&instance, &weighted.selected));
            assert!(weighted.cost <= 2.0 * weighted.lower_bound + 1.0e-9);
        }
    }

    #[test]
    fn binary_programs_match_exhaustive_tiny_oracles() {
        let instance = generate(&FIXTURES[0]).unwrap();
        for weighted in [false, true] {
            let exact = exact_cover(&instance, weighted).unwrap();
            let brute = brute_force(&instance, weighted);
            assert!((exact.objective - brute.objective).abs() < 1.0e-9);
        }
    }
}
