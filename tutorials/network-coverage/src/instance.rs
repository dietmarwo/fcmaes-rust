//! Synthetic stochastic-block instances and portable CSV input.

use std::collections::HashSet;
use std::error::Error;
use std::f64::consts::TAU;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use fcmaes_core::Rng;
use petgraph::algo::connected_components;
use petgraph::graph::UnGraph;
use serde::{Deserialize, Serialize};

/// One undirected ordinary edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Edge {
    /// First endpoint, always smaller than `v`.
    pub u: usize,
    /// Second endpoint.
    pub v: usize,
}

/// Generator and provenance metadata.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InstanceMetadata {
    /// Stable fixture name.
    pub name: String,
    /// Deterministic generator seed.
    pub seed: u64,
    /// Number of stochastic blocks.
    pub blocks: usize,
    /// Whether the graph came from an external edge list.
    pub external_edges: bool,
    /// Self-loops dropped while importing an external edge list.
    #[serde(default)]
    pub dropped_self_loops: usize,
}

/// Complete weighted network-coverage instance.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    /// Per-node positive selection costs.
    pub costs: Vec<f64>,
    /// Per-node block labels used only for visualization.
    pub blocks: Vec<usize>,
    /// Sorted unique undirected ordinary edges.
    pub edges: Vec<Edge>,
    /// Native overlapping groups.
    pub groups: Vec<Vec<usize>>,
    /// Reproducibility metadata.
    pub metadata: InstanceMetadata,
    /// Incident ordinary-edge indices per node.
    pub incident_edges: Vec<Vec<usize>>,
    /// Group indices per node.
    pub node_groups: Vec<Vec<usize>>,
}

impl Instance {
    /// Build derived adjacency indices and validate the raw arrays.
    pub fn from_parts(
        costs: Vec<f64>,
        blocks: Vec<usize>,
        mut edges: Vec<Edge>,
        mut groups: Vec<Vec<usize>>,
        metadata: InstanceMetadata,
    ) -> Result<Self, Box<dyn Error>> {
        let nodes = costs.len();
        if nodes == 0 || blocks.len() != nodes {
            return Err("cost and block arrays must have the same positive length".into());
        }
        if costs
            .iter()
            .any(|cost| !cost.is_finite() || *cost <= 0.0 || *cost > 1.0)
        {
            return Err("node costs must be finite and in (0, 1]".into());
        }
        for edge in &mut edges {
            if edge.u > edge.v {
                std::mem::swap(&mut edge.u, &mut edge.v);
            }
            if edge.u == edge.v || edge.v >= nodes {
                return Err("ordinary edges must be valid and loop-free".into());
            }
        }
        edges.sort_by_key(|edge| (edge.u, edge.v));
        edges.dedup();
        for group in &mut groups {
            group.sort_unstable();
            group.dedup();
            if group.len() < 2 || group.iter().any(|node| *node >= nodes) {
                return Err("groups need at least two valid distinct nodes".into());
            }
        }
        let mut incident_edges = vec![Vec::new(); nodes];
        for (index, edge) in edges.iter().enumerate() {
            incident_edges[edge.u].push(index);
            incident_edges[edge.v].push(index);
        }
        let mut node_groups = vec![Vec::new(); nodes];
        for (index, group) in groups.iter().enumerate() {
            for node in group {
                node_groups[*node].push(index);
            }
        }
        Ok(Self {
            costs,
            blocks,
            edges,
            groups,
            metadata,
            incident_edges,
            node_groups,
        })
    }

    /// Number of nodes and decision variables.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.costs.len()
    }

    /// True when the ordinary graph has one connected component.
    #[must_use]
    pub fn connected(&self) -> bool {
        let mut graph = UnGraph::<(), ()>::default();
        let nodes = (0..self.nodes())
            .map(|_| graph.add_node(()))
            .collect::<Vec<_>>();
        for edge in &self.edges {
            graph.add_edge(nodes[edge.u], nodes[edge.v], ());
        }
        connected_components(&graph) == 1
    }

    /// Write the portable fixture directory.
    pub fn write_csv(&self, directory: &Path) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(directory)?;
        let mut nodes = String::from("node,cost,block\n");
        for (node, (cost, block)) in self.costs.iter().zip(&self.blocks).enumerate() {
            writeln!(nodes, "{node},{cost:.17},{block}")?;
        }
        let mut edges = String::from("source,target\n");
        for edge in &self.edges {
            writeln!(edges, "{},{}", edge.u, edge.v)?;
        }
        let mut groups = String::from("group,node\n");
        for (group_index, group) in self.groups.iter().enumerate() {
            for node in group {
                writeln!(groups, "{group_index},{node}")?;
            }
        }
        fs::write(directory.join("nodes.csv"), nodes)?;
        fs::write(directory.join("edges.csv"), edges)?;
        fs::write(directory.join("groups.csv"), groups)?;
        fs::write(
            directory.join("metadata.json"),
            serde_json::to_string_pretty(&self.metadata)? + "\n",
        )?;
        Ok(())
    }

    /// Read a fixture written by [`write_csv`](Self::write_csv).
    pub fn read_csv(directory: &Path) -> Result<Self, Box<dyn Error>> {
        let mut costs = Vec::new();
        let mut blocks = Vec::new();
        for (line_index, line) in fs::read_to_string(directory.join("nodes.csv"))?
            .lines()
            .enumerate()
            .skip(1)
        {
            let fields = line.split(',').collect::<Vec<_>>();
            if fields.len() != 3 || fields[0].parse::<usize>()? != line_index - 1 {
                return Err("nodes.csv must use contiguous ordered node indices".into());
            }
            costs.push(fields[1].parse()?);
            blocks.push(fields[2].parse()?);
        }
        let mut edges = Vec::new();
        for line in fs::read_to_string(directory.join("edges.csv"))?
            .lines()
            .skip(1)
        {
            let fields = line.split(',').collect::<Vec<_>>();
            if fields.len() != 2 {
                return Err("edges.csv rows need two columns".into());
            }
            edges.push(Edge {
                u: fields[0].parse()?,
                v: fields[1].parse()?,
            });
        }
        let mut groups = Vec::<Vec<usize>>::new();
        for line in fs::read_to_string(directory.join("groups.csv"))?
            .lines()
            .skip(1)
        {
            let fields = line.split(',').collect::<Vec<_>>();
            if fields.len() != 2 {
                return Err("groups.csv rows need two columns".into());
            }
            let group: usize = fields[0].parse()?;
            groups.resize_with(group + 1, Vec::new);
            groups[group].push(fields[1].parse()?);
        }
        let metadata = serde_json::from_str(&fs::read_to_string(directory.join("metadata.json"))?)?;
        Self::from_parts(costs, blocks, edges, groups, metadata)
    }
}

/// Deterministic synthetic-generator configuration.
#[derive(Clone, Debug)]
pub struct GeneratorConfig {
    /// Stable fixture name.
    pub name: &'static str,
    /// Node count.
    pub nodes: usize,
    /// Stochastic block count.
    pub blocks: usize,
    /// Target ordinary-edge count.
    pub edges: usize,
    /// Overlapping-group count.
    pub groups: usize,
    /// Minimum group size.
    pub min_group: usize,
    /// Maximum group size.
    pub max_group: usize,
    /// Root seed.
    pub seed: u64,
}

/// Frozen fixture configurations.
pub const FIXTURES: [GeneratorConfig; 4] = [
    GeneratorConfig {
        name: "tiny",
        nodes: 14,
        blocks: 2,
        edges: 28,
        groups: 4,
        min_group: 3,
        max_group: 6,
        seed: 11,
    },
    GeneratorConfig {
        name: "small",
        nodes: 60,
        blocks: 4,
        edges: 240,
        groups: 12,
        min_group: 3,
        max_group: 18,
        seed: 23,
    },
    GeneratorConfig {
        name: "reference-1k",
        nodes: 1_000,
        blocks: 10,
        edges: 7_500,
        groups: 80,
        min_group: 5,
        max_group: 100,
        seed: 42,
    },
    GeneratorConfig {
        name: "reference-4k",
        nodes: 4_000,
        blocks: 20,
        edges: 30_025,
        groups: 200,
        min_group: 5,
        max_group: 300,
        seed: 43,
    },
];

fn choose_in_block(rng: &mut Rng, block: usize, blocks: usize, nodes: usize) -> usize {
    let count = (nodes + blocks - block - 1) / blocks;
    block + blocks * (rng.next_u64() as usize % count)
}

/// Generate one connected stochastic-block fixture.
pub fn generate(config: &GeneratorConfig) -> Result<Instance, Box<dyn Error>> {
    if config.nodes < 2
        || config.blocks == 0
        || config.edges < config.nodes - 1
        || config.edges > config.nodes * (config.nodes - 1) / 2
        || config.min_group < 2
        || config.max_group > config.nodes
        || config.min_group > config.max_group
    {
        return Err("invalid generator configuration".into());
    }
    let mut rng = Rng::new(config.seed);
    let mut raw_costs = Vec::with_capacity(config.nodes);
    for _ in 0..config.nodes {
        let u1 = rng.uniform01().max(1.0e-12);
        let u2 = rng.uniform01();
        let normal = (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos();
        raw_costs.push((-0.6 + 0.65 * normal).exp());
    }
    let scale = raw_costs.iter().copied().fold(0.0_f64, f64::max);
    let costs = raw_costs
        .into_iter()
        .map(|value| 0.05 + 0.95 * value / scale)
        .collect::<Vec<_>>();
    let blocks = (0..config.nodes)
        .map(|node| node % config.blocks)
        .collect::<Vec<_>>();
    let mut edge_set = HashSet::with_capacity(config.edges);
    for node in 1..config.nodes {
        edge_set.insert(Edge {
            u: node - 1,
            v: node,
        });
    }
    while edge_set.len() < config.edges {
        let u = rng.next_u64() as usize % config.nodes;
        let v = if rng.uniform01() < 0.82 {
            choose_in_block(&mut rng, blocks[u], config.blocks, config.nodes)
        } else {
            rng.next_u64() as usize % config.nodes
        };
        if u != v {
            edge_set.insert(Edge {
                u: u.min(v),
                v: u.max(v),
            });
        }
    }
    let mut edges = edge_set.into_iter().collect::<Vec<_>>();
    edges.sort_by_key(|edge| (edge.u, edge.v));
    let mut groups = Vec::with_capacity(config.groups);
    for group_index in 0..config.groups {
        let span = config.max_group - config.min_group;
        let size = if config.groups <= 1 {
            config.min_group
        } else {
            config.min_group + span * group_index / (config.groups - 1)
        };
        let base_block = rng.next_u64() as usize % config.blocks;
        let mut members = HashSet::with_capacity(size);
        while members.len() < size {
            let block = if rng.uniform01() < 0.78 {
                base_block
            } else {
                rng.next_u64() as usize % config.blocks
            };
            members.insert(choose_in_block(
                &mut rng,
                block,
                config.blocks,
                config.nodes,
            ));
        }
        let mut members = members.into_iter().collect::<Vec<_>>();
        members.sort_unstable();
        groups.push(members);
    }
    Instance::from_parts(
        costs,
        blocks,
        edges,
        groups,
        InstanceMetadata {
            name: config.name.to_owned(),
            seed: config.seed,
            blocks: config.blocks,
            external_edges: false,
            dropped_self_loops: 0,
        },
    )
}

/// Read a local whitespace edge list. No external source is redistributed.
pub fn read_edge_list(path: &Path, seed: u64) -> Result<Instance, Box<dyn Error>> {
    let mut edges = Vec::new();
    let mut maximum = None;
    let mut dropped_self_loops = 0;
    for line in fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 2 {
            return Err("edge-list rows need two integer node identifiers".into());
        }
        let u: usize = fields[0].parse()?;
        let v: usize = fields[1].parse()?;
        maximum = Some(maximum.map_or(u.max(v), |value: usize| value.max(u).max(v)));
        if u == v {
            dropped_self_loops += 1;
            continue;
        }
        edges.push(Edge { u, v });
    }
    let nodes = maximum.ok_or("edge list is empty")? + 1;
    if dropped_self_loops > 0 {
        eprintln!(
            "warning: dropped {dropped_self_loops} self-loop(s) from {}",
            path.display()
        );
    }
    let mut rng = Rng::new(seed);
    let costs = (0..nodes)
        .map(|_| 0.05 + 0.95 * rng.uniform01())
        .collect::<Vec<_>>();
    Instance::from_parts(
        costs,
        vec![0; nodes],
        edges,
        vec![],
        InstanceMetadata {
            name: path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("external")
                .to_owned(),
            seed,
            blocks: 1,
            external_edges: true,
            dropped_self_loops,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn frozen_generators_are_connected_and_deterministic() {
        for config in &FIXTURES {
            let left = generate(config).unwrap();
            let right = generate(config).unwrap();
            assert_eq!(left, right);
            assert!(left.connected());
            assert_eq!(left.nodes(), config.nodes);
            assert_eq!(left.edges.len(), config.edges);
            assert_eq!(left.groups.len(), config.groups);
            assert_eq!(left.groups.first().unwrap().len(), config.min_group);
            assert_eq!(left.groups.last().unwrap().len(), config.max_group);
        }
    }

    #[test]
    fn checked_in_csv_fixtures_match_the_frozen_generator() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("instances");
        for config in &FIXTURES {
            let checked_in = Instance::read_csv(&root.join(config.name)).unwrap();
            assert_eq!(checked_in, generate(config).unwrap());
        }
    }

    #[test]
    fn external_import_drops_and_counts_self_loops() {
        let path = std::env::temp_dir().join(format!(
            "network-coverage-self-loops-{}.txt",
            std::process::id()
        ));
        fs::write(&path, "0 0\n0 1\n1 0\n2 2\n").unwrap();
        let instance = read_edge_list(&path, 42).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(instance.nodes(), 3);
        assert_eq!(instance.edges, vec![Edge { u: 0, v: 1 }]);
        assert_eq!(instance.metadata.dropped_self_loops, 2);
    }
}
