//! Deterministic ground structure, supports, and named load cases.

/// One planar node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    /// Horizontal coordinate in metres.
    pub x: f64,
    /// Vertical coordinate in metres.
    pub y: f64,
}

/// One candidate pin-jointed member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Member {
    /// First node index.
    pub a: usize,
    /// Second node index.
    pub b: usize,
}

/// One nodal force.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodalLoad {
    /// Node index.
    pub node: usize,
    /// Horizontal force in newtons.
    pub fx: f64,
    /// Vertical force in newtons.
    pub fy: f64,
}

/// One simultaneous load case.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadCase {
    /// Stable case label.
    pub name: &'static str,
    /// Applied nodal forces.
    pub loads: Vec<NodalLoad>,
}

/// Fixed geometry and candidate connectivity.
#[derive(Clone, Debug)]
pub struct GroundStructure {
    /// Nominal lattice nodes.
    pub nodes: Vec<Node>,
    /// Candidate members.
    pub members: Vec<Member>,
    /// Nodes whose coordinates may move.
    pub movable_nodes: Vec<usize>,
    /// Nodes receiving service loads.
    pub load_nodes: [usize; 2],
    /// Pinned support node.
    pub pinned_node: usize,
    /// Vertical roller support node.
    pub roller_node: usize,
    /// Training load cases.
    pub load_cases: Vec<LoadCase>,
    /// Nominal span.
    pub span_m: f64,
    /// Horizontal bay width.
    pub bay_m: f64,
    /// Vertical lattice spacing.
    pub level_m: f64,
}

impl GroundStructure {
    /// Build the frozen 6 × 3 reference lattice.
    #[must_use]
    pub fn reference() -> Self {
        let bay_m = 2.4;
        let level_m = 2.0;
        let mut nodes = Vec::with_capacity(18);
        for column in 0..6 {
            for row in 0..3 {
                nodes.push(Node {
                    x: column as f64 * bay_m,
                    y: row as f64 * level_m,
                });
            }
        }
        let mut members = Vec::new();
        for a in 0..nodes.len() {
            for b in a + 1..nodes.len() {
                let dx = nodes[b].x - nodes[a].x;
                let dy = nodes[b].y - nodes[a].y;
                if dx.hypot(dy) <= 5.0 + 1.0e-12 {
                    members.push(Member { a, b });
                }
            }
        }
        let pinned_node = 0;
        let roller_node = 15;
        let load_nodes = [8, 11];
        let movable_nodes = (1..5)
            .flat_map(|column| (0..3).map(move |row| column * 3 + row))
            .filter(|node| !load_nodes.contains(node))
            .collect();
        let load_cases = vec![
            LoadCase {
                name: "vertical-service",
                loads: load_nodes
                    .iter()
                    .map(|node| NodalLoad {
                        node: *node,
                        fx: 0.0,
                        fy: -180_000.0,
                    })
                    .collect(),
            },
            LoadCase {
                name: "combined-service",
                loads: vec![
                    NodalLoad {
                        node: load_nodes[0],
                        fx: 45_000.0,
                        fy: -135_000.0,
                    },
                    NodalLoad {
                        node: load_nodes[1],
                        fx: 45_000.0,
                        fy: -135_000.0,
                    },
                ],
            },
        ];
        Self {
            nodes,
            members,
            movable_nodes,
            load_nodes,
            pinned_node,
            roller_node,
            load_cases,
            span_m: 12.0,
            bay_m,
            level_m,
        }
    }

    /// Find the candidate index for an unordered node pair.
    #[must_use]
    pub fn member_index(&self, a: usize, b: usize) -> Option<usize> {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        self.members
            .iter()
            .position(|member| member.a == a && member.b == b)
    }

    /// Deterministic triangulated maximum-cardinality baseline.
    #[must_use]
    pub fn baseline_members(&self) -> Vec<usize> {
        let mut pairs = Vec::new();
        for column in 0..5 {
            for row in 0..3 {
                pairs.push((column * 3 + row, (column + 1) * 3 + row));
            }
        }
        for column in 0..6 {
            for row in 0..2 {
                pairs.push((column * 3 + row, column * 3 + row + 1));
            }
        }
        for column in 0..5 {
            for row in 0..2 {
                let lower_left = column * 3 + row;
                let upper_left = lower_left + 1;
                let lower_right = (column + 1) * 3 + row;
                let upper_right = lower_right + 1;
                if (column + row) % 2 == 0 {
                    pairs.push((lower_left, upper_right));
                } else {
                    pairs.push((upper_left, lower_right));
                }
            }
        }
        for pair in [(1, 3), (7, 9), (13, 15)] {
            pairs.push(pair);
        }
        let mut indices = pairs
            .into_iter()
            .map(|(a, b)| {
                self.member_index(a, b)
                    .expect("baseline member must exist in the ground structure")
            })
            .collect::<Vec<_>>();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), 40);
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_ground_structure_is_frozen() {
        let ground = GroundStructure::reference();
        assert_eq!(ground.nodes.len(), 18);
        assert_eq!(ground.members.len(), 75);
        assert_eq!(ground.movable_nodes.len(), 10);
        assert_eq!(ground.baseline_members().len(), 40);
        assert!(
            ground
                .movable_nodes
                .iter()
                .all(|node| !ground.load_nodes.contains(node))
        );
    }

    #[test]
    fn generation_is_deterministic() {
        let left = GroundStructure::reference();
        let right = GroundStructure::reference();
        assert_eq!(left.nodes, right.nodes);
        assert_eq!(left.members, right.members);
        assert_eq!(left.baseline_members(), right.baseline_members());
    }
}
