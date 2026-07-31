//! Bounded three-gene topology grammar and motif classification.

use std::collections::BTreeSet;

use fcmaes_core::Rng;
use serde::{Deserialize, Serialize};

/// Number of genes and runtime species.
pub const GENES: usize = 3;
/// One signed decision for every ordered pair.
pub const EDGE_SLOTS: usize = GENES * GENES;
/// Active-edge grammar bounds.
pub const MIN_ACTIVE_EDGES: usize = 2;
pub const MAX_ACTIVE_EDGES: usize = 6;
/// Self edges first, then ordered cross edges.
pub const EDGE_INDEX: [(usize, usize); EDGE_SLOTS] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
];

/// A signed regulatory edge.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(u8)]
pub enum Edge {
    Absent = 0,
    Activation = 1,
    Inhibition = 2,
}

impl Edge {
    fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Absent),
            1 => Some(Self::Activation),
            2 => Some(Self::Inhibition),
            _ => None,
        }
    }
}

/// Canonical nine-slot topology.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Topology {
    pub edges: [u8; EDGE_SLOTS],
}

impl Topology {
    /// Build and validate a topology.
    pub fn try_new(edges: [u8; EDGE_SLOTS]) -> Result<Self, &'static str> {
        let topology = Self { edges };
        topology.validate()?;
        Ok(topology)
    }

    /// Parse `000200220` or a comma-separated nine-value vector.
    pub fn parse(value: &str) -> Result<Self, &'static str> {
        let digits: Vec<u8> = if value.contains(',') {
            value
                .split(',')
                .map(|part| part.trim().parse::<u8>().map_err(|_| "invalid edge value"))
                .collect::<Result<_, _>>()?
        } else {
            value
                .chars()
                .map(|character| {
                    character
                        .to_digit(10)
                        .map(|digit| digit as u8)
                        .ok_or("invalid topology digit")
                })
                .collect::<Result<_, _>>()?
        };
        let edges: [u8; EDGE_SLOTS] = digits.try_into().map_err(|_| "topology needs nine edges")?;
        Self::try_new(edges)
    }

    /// Validate edge values, edge count and connectivity.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self
            .edges
            .iter()
            .any(|&value| Edge::from_u8(value).is_none())
        {
            return Err("edges must be 0, 1 or 2");
        }
        if !(MIN_ACTIVE_EDGES..=MAX_ACTIVE_EDGES).contains(&self.active_edges()) {
            return Err("active edge count must lie in 2..=6");
        }
        if self.has_isolated_gene() {
            return Err("topology contains an isolated gene");
        }
        Ok(())
    }

    /// Number of active edges.
    pub fn active_edges(&self) -> usize {
        self.edges.iter().filter(|&&value| value != 0).count()
    }

    /// Variable inner dimension.
    pub fn parameter_dimension(&self) -> usize {
        2 * GENES + 2 * self.active_edges()
    }

    /// Does any gene have neither an incoming nor outgoing edge?
    pub fn has_isolated_gene(&self) -> bool {
        (0..GENES).any(|gene| {
            !self.edges.iter().enumerate().any(|(index, &value)| {
                let (source, target) = EDGE_INDEX[index];
                value != 0 && (source == gene || target == gene)
            })
        })
    }

    /// Compact canonical key.
    pub fn key(&self) -> String {
        self.edges
            .iter()
            .map(|value| char::from(b'0' + *value))
            .collect()
    }

    /// Hamming distance between slot vectors.
    pub fn edit_distance(&self, other: &Self) -> usize {
        self.edges
            .iter()
            .zip(other.edges)
            .filter(|(left, right)| **left != *right)
            .count()
    }

    /// Decision-derived structural labels.
    pub fn motif_flags(&self) -> Vec<String> {
        let mut flags = BTreeSet::new();
        for cycle in [[0, 1, 2], [0, 2, 1]] {
            let values = [
                self.edge(cycle[0], cycle[1]),
                self.edge(cycle[1], cycle[2]),
                self.edge(cycle[2], cycle[0]),
            ];
            match values {
                [2, 2, 2] => {
                    flags.insert("repressilator");
                }
                [1, 1, 2] | [1, 2, 1] | [2, 1, 1] => {
                    flags.insert("goodwin-like");
                }
                [1, 1, 1] => {
                    flags.insert("positive-cycle");
                }
                [a, b, c] if a != 0 && b != 0 && c != 0 => {
                    flags.insert("mixed-cycle");
                }
                _ => {}
            }
        }
        for left in 0..GENES {
            for right in left + 1..GENES {
                if self.edge(left, right) == 2 && self.edge(right, left) == 2 {
                    flags.insert("toggle-core");
                }
            }
        }
        if flags.is_empty() {
            flags.insert("other");
        }
        flags.into_iter().map(str::to_owned).collect()
    }

    /// Structural archive key; this is a control, not an emergent descriptor.
    pub fn niche_key(&self) -> String {
        let activations = self.edges.iter().filter(|&&value| value == 1).count();
        let inhibitions = self.edges.iter().filter(|&&value| value == 2).count();
        let self_edges = self.edges[..GENES]
            .iter()
            .filter(|&&value| value != 0)
            .count();
        format!(
            "E{}-A{activations}-I{inhibitions}-S{self_edges}-C{}",
            self.active_edges(),
            self.motif_flags().join("+")
        )
    }

    fn edge(&self, source: usize, target: usize) -> u8 {
        let index = EDGE_INDEX
            .iter()
            .position(|&pair| pair == (source, target))
            .expect("all ordered pairs exist");
        self.edges[index]
    }
}

/// Literature/reference motifs. The toggle includes `A -> C` only to satisfy
/// the no-isolated-gene grammar; its `A <-> B` inhibition pair is the control.
pub const REFERENCES: [(&str, Topology); 4] = [
    (
        "repressilator",
        Topology {
            edges: [0, 0, 0, 2, 0, 0, 2, 2, 0],
        },
    ),
    (
        "goodwin-like",
        Topology {
            edges: [0, 0, 0, 1, 0, 0, 1, 2, 0],
        },
    ),
    (
        "positive-cycle",
        Topology {
            edges: [0, 0, 0, 1, 0, 0, 1, 1, 0],
        },
    ),
    (
        "toggle-control",
        Topology {
            edges: [0, 0, 0, 2, 1, 2, 0, 0, 0],
        },
    ),
];

/// Exact reference name, if any. Reference optimization is kept separate from
/// proposal archives so these rows cannot count as rediscovery.
pub fn exact_reference(topology: &Topology) -> Option<&'static str> {
    REFERENCES
        .iter()
        .find_map(|(name, reference)| (topology == reference).then_some(*name))
}

/// Sample uniformly over active-edge count, active slots and signs.
pub fn random_valid(rng: &mut Rng) -> Topology {
    for _ in 0..10_000 {
        let active = MIN_ACTIVE_EDGES
            + rng.int_below((MAX_ACTIVE_EDGES - MIN_ACTIVE_EDGES + 1) as i64) as usize;
        let mut slots: Vec<usize> = (0..EDGE_SLOTS).collect();
        for index in 0..active {
            let swap = index + rng.int_below((EDGE_SLOTS - index) as i64) as usize;
            slots.swap(index, swap);
        }
        let mut edges = [0; EDGE_SLOTS];
        for &slot in &slots[..active] {
            edges[slot] = 1 + rng.int_below(2) as u8;
        }
        if let Ok(topology) = Topology::try_new(edges) {
            return topology;
        }
    }
    panic!("bounded topology sampler exhausted its retry guard")
}

/// One grammar-preserving edit, with random fallback.
pub fn mutate(parent: &Topology, rng: &mut Rng) -> Topology {
    for _ in 0..1_000 {
        let mut edges = parent.edges;
        let slot = rng.int_below(EDGE_SLOTS as i64) as usize;
        edges[slot] = match edges[slot] {
            0 => 1 + rng.int_below(2) as u8,
            1 if rng.uniform01() < 0.5 => 0,
            2 if rng.uniform01() < 0.5 => 0,
            1 => 2,
            2 => 1,
            _ => unreachable!(),
        };
        if let Ok(topology) = Topology::try_new(edges) {
            return topology;
        }
    }
    random_valid(rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_are_valid_and_dimensions_are_ten_to_eighteen() {
        for (_, topology) in REFERENCES {
            topology.validate().unwrap();
            assert!((10..=18).contains(&topology.parameter_dimension()));
        }
    }

    #[test]
    fn invalid_topologies_are_rejected_individually() {
        assert!(Topology::try_new([0; 9]).is_err());
        assert!(Topology::try_new([3, 0, 0, 1, 0, 0, 1, 0, 0]).is_err());
        assert!(Topology::try_new([1, 1, 0, 0, 0, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn motif_classifier_distinguishes_near_miss() {
        let repressilator = REFERENCES[0].1;
        assert!(
            repressilator
                .motif_flags()
                .contains(&"repressilator".to_owned())
        );
        let near = Topology::try_new([0, 0, 0, 1, 0, 0, 2, 2, 0]).unwrap();
        assert!(!near.motif_flags().contains(&"repressilator".to_owned()));
        assert_eq!(repressilator.edit_distance(&near), 1);
    }

    #[test]
    fn seeded_sampling_and_mutation_stay_inside_grammar() {
        let mut first = Rng::new(42);
        let mut second = Rng::new(42);
        let mut mutation_rng = Rng::new(7);
        for _ in 0..100 {
            let left = random_valid(&mut first);
            let right = random_valid(&mut second);
            assert_eq!(left, right);
            mutate(&left, &mut mutation_rng).validate().unwrap();
        }
    }

    #[test]
    fn documented_random_reference_prior_matches_the_grammar() {
        let mut total_by_edges = [0_usize; MAX_ACTIVE_EDGES + 1];
        let mut valid_by_edges = [0_usize; MAX_ACTIVE_EDGES + 1];
        for encoded in 0..3_u32.pow(EDGE_SLOTS as u32) {
            let mut value = encoded;
            let mut edges = [0_u8; EDGE_SLOTS];
            for edge in &mut edges {
                *edge = (value % 3) as u8;
                value /= 3;
            }
            let active = edges.iter().filter(|&&edge| edge != 0).count();
            if (MIN_ACTIVE_EDGES..=MAX_ACTIVE_EDGES).contains(&active) {
                total_by_edges[active] += 1;
                if Topology::try_new(edges).is_ok() {
                    valid_by_edges[active] += 1;
                }
            }
        }
        let valid_attempt = (MIN_ACTIVE_EDGES..=MAX_ACTIVE_EDGES)
            .map(|active| valid_by_edges[active] as f64 / total_by_edges[active] as f64)
            .sum::<f64>()
            / (MAX_ACTIVE_EDGES - MIN_ACTIVE_EDGES + 1) as f64;
        let specified_three_edge_attempt = 1.0 / 5.0 / total_by_edges[3] as f64;
        let specified_reference = specified_three_edge_attempt / valid_attempt;
        assert!((specified_reference - 1.0 / 2912.0).abs() < 1.0e-15);
        assert!((4.0 * specified_reference - 1.0 / 728.0).abs() < 1.0e-15);
    }
}
