//! Linear-storage coverage kernel and marginal-greedy frontier.

use crate::instance::Instance;

/// Frozen group-pair exponent.
pub const GROUP_WEIGHT_EXPONENT: f64 = 0.5;

/// One fully replayed subset.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageMetrics {
    /// Selected-node mask.
    pub selected: Vec<bool>,
    /// Selected-node count.
    pub selected_count: usize,
    /// Sum of selected node costs.
    pub cost: f64,
    /// Covered ordinary-edge count.
    pub edge_coverage: usize,
    /// Weighted native group-pair coverage.
    pub group_coverage: f64,
    /// Ordinary plus group coverage.
    pub coverage: f64,
    /// Coverage divided by the all-selected value.
    pub roi: f64,
    /// Ordinary edges with neither endpoint selected.
    pub uncovered_edges: usize,
}

/// One prefix from deterministic marginal-gain greedy.
#[derive(Clone, Debug)]
pub struct GreedyPoint {
    /// Prefix length.
    pub step: usize,
    /// Replayed physical metrics.
    pub metrics: CoverageMetrics,
}

/// Weight of one implicit pair relation in a group.
#[must_use]
pub fn group_pair_weight(size: usize, exponent: f64) -> f64 {
    (size as f64).powf(-exponent)
}

fn pairs(count: usize) -> usize {
    count.saturating_mul(count.saturating_sub(1)) / 2
}

/// Analytic coverage obtained when every node is selected.
#[must_use]
pub fn full_coverage(instance: &Instance, exponent: f64) -> f64 {
    instance.edges.len() as f64
        + instance
            .groups
            .iter()
            .map(|group| group_pair_weight(group.len(), exponent) * pairs(group.len()) as f64)
            .sum::<f64>()
}

/// Evaluate ordinary edges and native groups without clique materialization.
#[must_use]
pub fn evaluate(instance: &Instance, selected: &[bool], exponent: f64) -> Option<CoverageMetrics> {
    if selected.len() != instance.nodes() {
        return None;
    }
    let selected_count = selected.iter().filter(|value| **value).count();
    let cost = selected
        .iter()
        .zip(&instance.costs)
        .filter(|(chosen, _)| **chosen)
        .map(|(_, cost)| cost)
        .sum::<f64>()
        + 0.0;
    let edge_coverage = instance
        .edges
        .iter()
        .filter(|edge| selected[edge.u] || selected[edge.v])
        .count();
    let group_coverage = instance
        .groups
        .iter()
        .map(|group| {
            let chosen = group.iter().filter(|node| selected[**node]).count();
            let covered_pairs = pairs(group.len()) - pairs(group.len() - chosen);
            group_pair_weight(group.len(), exponent) * covered_pairs as f64
        })
        .sum::<f64>()
        + 0.0;
    let coverage = edge_coverage as f64 + group_coverage + 0.0;
    Some(CoverageMetrics {
        selected: selected.to_vec(),
        selected_count,
        cost,
        edge_coverage,
        group_coverage,
        coverage,
        roi: coverage / full_coverage(instance, exponent).max(1.0e-30),
        uncovered_edges: instance.edges.len() - edge_coverage,
    })
}

/// Literal weighted clique-edge coverage used only as a tiny independent oracle.
#[must_use]
pub fn expanded_group_coverage(instance: &Instance, selected: &[bool], exponent: f64) -> f64 {
    instance
        .groups
        .iter()
        .map(|group| {
            let weight = group_pair_weight(group.len(), exponent);
            let mut covered = 0.0;
            for left in 0..group.len() {
                for right in left + 1..group.len() {
                    if selected[group[left]] || selected[group[right]] {
                        covered += weight;
                    }
                }
            }
            covered
        })
        .sum()
}

fn marginal_gain(
    instance: &Instance,
    selected: &[bool],
    group_counts: &[usize],
    node: usize,
    exponent: f64,
) -> f64 {
    let ordinary = instance.incident_edges[node]
        .iter()
        .filter(|edge_index| {
            let edge = instance.edges[**edge_index];
            !selected[edge.u] && !selected[edge.v]
        })
        .count() as f64;
    let group = instance.node_groups[node]
        .iter()
        .map(|group_index| {
            let size = instance.groups[*group_index].len();
            let unselected_after = size - group_counts[*group_index] - 1;
            group_pair_weight(size, exponent) * unselected_after as f64
        })
        .sum::<f64>();
    ordinary + group
}

/// Build the complete deterministic marginal-gain-per-cost prefix frontier.
#[must_use]
pub fn marginal_greedy(instance: &Instance, exponent: f64) -> Vec<GreedyPoint> {
    let mut selected = vec![false; instance.nodes()];
    let mut group_counts = vec![0; instance.groups.len()];
    let mut result = Vec::with_capacity(instance.nodes() + 1);
    result.push(GreedyPoint {
        step: 0,
        metrics: evaluate(instance, &selected, exponent).unwrap(),
    });
    for step in 1..=instance.nodes() {
        let node = (0..instance.nodes())
            .filter(|node| !selected[*node])
            .max_by(|left, right| {
                let left_gain = marginal_gain(instance, &selected, &group_counts, *left, exponent)
                    / instance.costs[*left];
                let right_gain =
                    marginal_gain(instance, &selected, &group_counts, *right, exponent)
                        / instance.costs[*right];
                left_gain
                    .total_cmp(&right_gain)
                    .then_with(|| right.cmp(left))
            })
            .expect("one unselected node remains");
        selected[node] = true;
        for group in &instance.node_groups[node] {
            group_counts[*group] += 1;
        }
        result.push(GreedyPoint {
            step,
            metrics: evaluate(instance, &selected, exponent).unwrap(),
        });
    }
    result
}

/// Return the nondominated cost/coverage points from an ordered candidate list.
#[must_use]
pub fn nondominated(points: &[CoverageMetrics]) -> Vec<usize> {
    points
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            !points.iter().enumerate().any(|(other_index, other)| {
                other_index != *index
                    && other.cost <= candidate.cost
                    && other.roi >= candidate.roi
                    && (other.cost < candidate.cost || other.roi > candidate.roi)
            })
        })
        .map(|(index, _)| index)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{FIXTURES, generate};

    #[test]
    fn native_groups_equal_expanded_weighted_cliques() {
        let instance = generate(&FIXTURES[0]).unwrap();
        for pattern in 0_u32..1 << instance.nodes() {
            let selected = (0..instance.nodes())
                .map(|node| pattern & (1 << node) != 0)
                .collect::<Vec<_>>();
            let native = evaluate(&instance, &selected, GROUP_WEIGHT_EXPONENT)
                .unwrap()
                .group_coverage;
            let expanded = expanded_group_coverage(&instance, &selected, GROUP_WEIGHT_EXPONENT);
            assert!((native - expanded).abs() < 1.0e-10);
        }
    }

    #[test]
    fn empty_and_full_sets_match_analytic_endpoints() {
        let instance = generate(&FIXTURES[1]).unwrap();
        let empty = evaluate(
            &instance,
            &vec![false; instance.nodes()],
            GROUP_WEIGHT_EXPONENT,
        )
        .unwrap();
        let full = evaluate(
            &instance,
            &vec![true; instance.nodes()],
            GROUP_WEIGHT_EXPONENT,
        )
        .unwrap();
        assert_eq!(empty.coverage, 0.0);
        assert_eq!(empty.cost.to_bits(), 0.0_f64.to_bits());
        assert_eq!(empty.group_coverage.to_bits(), 0.0_f64.to_bits());
        assert_eq!(empty.roi, 0.0);
        assert!((full.coverage - full_coverage(&instance, GROUP_WEIGHT_EXPONENT)).abs() < 1.0e-9);
        assert!((full.roi - 1.0).abs() < 1.0e-12);
    }
}
