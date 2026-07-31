//! Linear 2-D truss assembly, spectral stability gate, and response recovery.

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::catalogue::{ALLOWABLE_STRESS_PA, STEEL_E_PA, sections};
use crate::decode::{ActiveMember, DecodedDesign};
use crate::ground::{GroundStructure, LoadCase, Member, NodalLoad, Node};

/// Reciprocal-condition feasibility threshold.
pub const RCOND_MIN: f64 = 1.0e-10;

/// Scenario modifiers applied without changing the ground structure.
#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    /// Stable label.
    pub name: &'static str,
    /// Multiplier on every applied force.
    pub load_scale: f64,
    /// Multiplier on Young's modulus.
    pub modulus_scale: f64,
    /// Prescribed pinned-support vertical displacement.
    pub pinned_settlement_y_m: f64,
    /// Prescribed roller vertical displacement.
    pub roller_settlement_y_m: f64,
}

impl Scenario {
    /// Nominal training scenario.
    pub const TRAINING: Self = Self {
        name: "training",
        load_scale: 1.0,
        modulus_scale: 1.0,
        pinned_settlement_y_m: 0.0,
        roller_settlement_y_m: 0.0,
    };
    /// Kind-changing holdout scenario.
    pub const HOLDOUT: Self = Self {
        name: "holdout-settlement-reduced-e",
        load_scale: 1.10,
        modulus_scale: 0.90,
        pinned_settlement_y_m: 0.0,
        roller_settlement_y_m: -0.005,
    };
}

/// Physical-work counters shared by parallel objectives.
#[derive(Default, Debug)]
pub struct WorkCounter {
    candidates: AtomicU64,
    fem_solves: AtomicU64,
    factorizations: AtomicU64,
}

/// Snapshot of physical work.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkSnapshot {
    /// Logical candidate evaluations.
    pub candidate_evaluations: u64,
    /// Load-case linear solves.
    pub fem_solves: u64,
    /// Attempted matrix factorizations.
    pub factorizations: u64,
}

impl WorkCounter {
    /// Record one logical candidate.
    pub fn candidate(&self) {
        self.candidates.fetch_add(1, Ordering::Relaxed);
    }

    fn solve(&self) {
        self.fem_solves.fetch_add(1, Ordering::Relaxed);
    }

    fn factorization(&self) {
        self.factorizations.fetch_add(1, Ordering::Relaxed);
    }

    /// Read a consistent-enough accounting snapshot.
    #[must_use]
    pub fn snapshot(&self) -> WorkSnapshot {
        WorkSnapshot {
            candidate_evaluations: self.candidates.load(Ordering::Relaxed),
            fem_solves: self.fem_solves.load(Ordering::Relaxed),
            factorizations: self.factorizations.load(Ordering::Relaxed),
        }
    }
}

/// Typed reason why physical response is unavailable.
#[derive(Clone, Debug, PartialEq)]
pub enum AnalysisFailure {
    /// A service-load node has no active path to either support.
    Disconnected,
    /// The reduced stiffness matrix is rank deficient.
    Singular {
        /// Smallest eigenvalue.
        lambda_min: f64,
        /// Scale-aware rank threshold.
        rank_tolerance: f64,
    },
    /// The matrix is positive definite but below the declared conditioning gate.
    IllConditioned {
        /// Measured reciprocal condition.
        rcond: f64,
    },
    /// A Cholesky solve unexpectedly failed after the spectral gate.
    SolveFailure,
}

impl AnalysisFailure {
    /// Stable artifact label.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Disconnected => "disconnected",
            Self::Singular { .. } => "singular",
            Self::IllConditioned { .. } => "ill-conditioned",
            Self::SolveFailure => "solve-failure",
        }
    }
}

/// Recovered intact-truss response across all named load cases.
#[derive(Clone, Debug)]
pub struct AnalysisMetrics {
    /// Spectral reciprocal condition.
    pub rcond: f64,
    /// Worst absolute axial stress.
    pub max_stress_pa: f64,
    /// Worst yield utilization.
    pub max_stress_ratio: f64,
    /// Worst Euler-buckling utilization over compression members.
    pub max_buckling_ratio: f64,
    /// Worst nodal displacement magnitude.
    pub max_displacement_m: f64,
    /// Worst absolute external compliance.
    pub compliance_j: f64,
    /// Maximum stress/buckling utilization per active member.
    pub member_utilizations: Vec<f64>,
    /// Signed force associated with each member's maximum utilization.
    pub member_forces_n: Vec<f64>,
    /// Full displacement vector for every load case.
    pub case_displacements: Vec<Vec<f64>>,
}

/// Independent equilibrium and virtual-work evidence for the three-bar oracle.
#[derive(Clone, Debug)]
pub struct OracleEvidence {
    /// Applied vertical load.
    pub load_n: f64,
    /// Closed-form member forces in member order.
    pub analytic_forces_n: [f64; 3],
    /// FEM member forces in member order.
    pub fem_forces_n: [f64; 3],
    /// Closed-form vertical displacement at the loaded node.
    pub analytic_displacement_m: f64,
    /// FEM vertical displacement at the loaded node.
    pub fem_displacement_m: f64,
    /// Physical-work accounting for the oracle solve.
    pub work: WorkSnapshot,
}

fn connected(
    ground: &GroundStructure,
    design: &DecodedDesign,
    omitted_active: Option<usize>,
) -> bool {
    let mut adjacency = vec![Vec::new(); design.nodes.len()];
    for (active_index, active) in design.active.iter().enumerate() {
        if Some(active_index) == omitted_active {
            continue;
        }
        let member = ground.members[active.member_index];
        adjacency[member.a].push(member.b);
        adjacency[member.b].push(member.a);
    }
    let mut visited = vec![false; design.nodes.len()];
    let mut queue = VecDeque::from([ground.pinned_node, ground.roller_node]);
    while let Some(node) = queue.pop_front() {
        if visited[node] {
            continue;
        }
        visited[node] = true;
        queue.extend(adjacency[node].iter().copied());
    }
    ground.load_nodes.iter().all(|node| visited[*node])
}

fn element_geometry(nodes: &[Node], member: Member) -> Option<(f64, f64, f64)> {
    let dx = nodes[member.b].x - nodes[member.a].x;
    let dy = nodes[member.b].y - nodes[member.a].y;
    let length = dx.hypot(dy);
    (length > 1.0e-9).then_some((length, dx / length, dy / length))
}

/// Analyze a design, optionally removing one active member.
pub fn analyze(
    ground: &GroundStructure,
    design: &DecodedDesign,
    scenario: Scenario,
    omitted_active: Option<usize>,
    counter: &WorkCounter,
) -> Result<AnalysisMetrics, AnalysisFailure> {
    if !connected(ground, design, omitted_active) {
        return Err(AnalysisFailure::Disconnected);
    }
    let dofs = 2 * design.nodes.len();
    let modulus = STEEL_E_PA * scenario.modulus_scale;
    let catalogue = sections();
    let mut stiffness = DMatrix::<f64>::zeros(dofs, dofs);
    let mut geometries = Vec::with_capacity(design.active.len());
    for (active_index, active) in design.active.iter().enumerate() {
        let member = ground.members[active.member_index];
        let Some((length, c, s)) = element_geometry(&design.nodes, member) else {
            return Err(AnalysisFailure::Singular {
                lambda_min: 0.0,
                rank_tolerance: 0.0,
            });
        };
        geometries.push((member, length, c, s));
        if Some(active_index) == omitted_active {
            continue;
        }
        let coefficient = modulus * catalogue[active.section_index].area_m2 / length;
        let vector = [c, s, -c, -s];
        let indices = [
            2 * member.a,
            2 * member.a + 1,
            2 * member.b,
            2 * member.b + 1,
        ];
        for row in 0..4 {
            for column in 0..4 {
                stiffness[(indices[row], indices[column])] +=
                    coefficient * vector[row] * vector[column];
            }
        }
    }
    let constrained = [
        (2 * ground.pinned_node, 0.0),
        (2 * ground.pinned_node + 1, scenario.pinned_settlement_y_m),
        (2 * ground.roller_node + 1, scenario.roller_settlement_y_m),
    ];
    let free = (0..dofs)
        .filter(|dof| !constrained.iter().any(|(fixed, _)| fixed == dof))
        .collect::<Vec<_>>();
    let mut reduced = DMatrix::<f64>::zeros(free.len(), free.len());
    for (row, global_row) in free.iter().enumerate() {
        for (column, global_column) in free.iter().enumerate() {
            reduced[(row, column)] = stiffness[(*global_row, *global_column)];
        }
    }
    counter.factorization();
    let eigenvalues = SymmetricEigen::new(reduced.clone()).eigenvalues;
    let lambda_min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let lambda_max = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let rank_tolerance = free.len().max(1) as f64 * f64::EPSILON * lambda_max.max(1.0);
    if !lambda_min.is_finite() || lambda_min <= rank_tolerance {
        return Err(AnalysisFailure::Singular {
            lambda_min,
            rank_tolerance,
        });
    }
    let rcond = lambda_min / lambda_max;
    if rcond < RCOND_MIN {
        return Err(AnalysisFailure::IllConditioned { rcond });
    }
    let cholesky = reduced.cholesky().ok_or(AnalysisFailure::SolveFailure)?;
    let mut max_stress_pa: f64 = 0.0;
    let mut max_stress_ratio: f64 = 0.0;
    let mut max_buckling_ratio: f64 = 0.0;
    let mut max_displacement_m: f64 = 0.0;
    let mut compliance_j: f64 = 0.0;
    let mut member_utilizations = vec![0.0_f64; design.active.len()];
    let mut member_forces_n = vec![0.0_f64; design.active.len()];
    let mut case_displacements = Vec::with_capacity(ground.load_cases.len());
    for load_case in &ground.load_cases {
        let mut force = DVector::<f64>::zeros(dofs);
        for load in &load_case.loads {
            force[2 * load.node] += scenario.load_scale * load.fx;
            force[2 * load.node + 1] += scenario.load_scale * load.fy;
        }
        let mut rhs = DVector::<f64>::zeros(free.len());
        for (row, global_row) in free.iter().enumerate() {
            rhs[row] = force[*global_row];
            for (fixed, value) in constrained {
                rhs[row] -= stiffness[(*global_row, fixed)] * value;
            }
        }
        let solved = cholesky.solve(&rhs);
        if solved.iter().any(|value| !value.is_finite()) {
            return Err(AnalysisFailure::SolveFailure);
        }
        counter.solve();
        let mut displacement = DVector::<f64>::zeros(dofs);
        for (row, global) in free.iter().enumerate() {
            displacement[*global] = solved[row];
        }
        for (fixed, value) in constrained {
            displacement[fixed] = value;
        }
        for node in 0..design.nodes.len() {
            max_displacement_m =
                max_displacement_m.max(displacement[2 * node].hypot(displacement[2 * node + 1]));
        }
        compliance_j = compliance_j.max(force.dot(&displacement).abs());
        for (active_index, active) in design.active.iter().enumerate() {
            if Some(active_index) == omitted_active {
                continue;
            }
            let (member, length, c, s) = geometries[active_index];
            let axial_extension = (displacement[2 * member.b] - displacement[2 * member.a]) * c
                + (displacement[2 * member.b + 1] - displacement[2 * member.a + 1]) * s;
            let section = catalogue[active.section_index];
            let axial_force = modulus * section.area_m2 / length * axial_extension;
            let stress = axial_force / section.area_m2;
            let stress_ratio = stress.abs() / ALLOWABLE_STRESS_PA;
            let buckling_ratio = if axial_force < 0.0 {
                let critical = PI.powi(2) * modulus * section.inertia_m4 / length.powi(2);
                axial_force.abs() / critical
            } else {
                0.0
            };
            let utilization = stress_ratio.max(buckling_ratio);
            max_stress_pa = max_stress_pa.max(stress.abs());
            max_stress_ratio = max_stress_ratio.max(stress_ratio);
            max_buckling_ratio = max_buckling_ratio.max(buckling_ratio);
            if utilization > member_utilizations[active_index] {
                member_utilizations[active_index] = utilization;
                member_forces_n[active_index] = axial_force;
            }
        }
        case_displacements.push(displacement.as_slice().to_vec());
    }
    Ok(AnalysisMetrics {
        rcond,
        max_stress_pa,
        max_stress_ratio,
        max_buckling_ratio,
        max_displacement_m,
        compliance_j,
        member_utilizations,
        member_forces_n,
        case_displacements,
    })
}

/// Solve the frozen three-bar validation oracle and return both derivations.
///
/// The two diagonal forces follow directly from joint equilibrium. The base
/// force follows horizontal equilibrium, and the vertical displacement is an
/// independent unit-load/virtual-work sum.
pub fn triangular_oracle() -> Result<OracleEvidence, AnalysisFailure> {
    let load_n = 100_000.0;
    let ground = GroundStructure {
        nodes: vec![
            Node { x: -2.0, y: 0.0 },
            Node { x: 2.0, y: 0.0 },
            Node { x: 0.0, y: 3.0 },
        ],
        members: vec![
            Member { a: 0, b: 1 },
            Member { a: 0, b: 2 },
            Member { a: 1, b: 2 },
        ],
        movable_nodes: vec![],
        load_nodes: [2, 2],
        pinned_node: 0,
        roller_node: 1,
        load_cases: vec![LoadCase {
            name: "oracle",
            loads: vec![NodalLoad {
                node: 2,
                fx: 0.0,
                fy: -load_n,
            }],
        }],
        span_m: 4.0,
        bay_m: 2.0,
        level_m: 3.0,
    };
    let design = DecodedDesign {
        nodes: ground.nodes.clone(),
        active: (0..3)
            .map(|member_index| ActiveMember {
                member_index,
                section_index: 8,
            })
            .collect(),
    };
    let counter = WorkCounter::default();
    let metrics = analyze(&ground, &design, Scenario::TRAINING, None, &counter)?;
    let diagonal_length = 13.0_f64.sqrt();
    let analytic_forces_n = [
        load_n * 2.0 / (2.0 * 3.0),
        -load_n * diagonal_length / (2.0 * 3.0),
        -load_n * diagonal_length / (2.0 * 3.0),
    ];
    let section = sections()[8];
    let lengths = [4.0, diagonal_length, diagonal_length];
    let analytic_displacement_m = analytic_forces_n
        .iter()
        .zip(lengths)
        .map(|(force, length)| force.powi(2) * length / (STEEL_E_PA * section.area_m2 * load_n))
        .sum::<f64>();
    Ok(OracleEvidence {
        load_n,
        analytic_forces_n,
        fem_forces_n: metrics
            .member_forces_n
            .try_into()
            .expect("the oracle has exactly three members"),
        analytic_displacement_m,
        fem_displacement_m: -metrics.case_displacements[0][5],
        work: counter.snapshot(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle(load: f64) -> (GroundStructure, DecodedDesign) {
        let ground = GroundStructure {
            nodes: vec![
                Node { x: -2.0, y: 0.0 },
                Node { x: 2.0, y: 0.0 },
                Node { x: 0.0, y: 3.0 },
            ],
            members: vec![
                Member { a: 0, b: 1 },
                Member { a: 0, b: 2 },
                Member { a: 1, b: 2 },
            ],
            movable_nodes: vec![],
            load_nodes: [2, 2],
            pinned_node: 0,
            roller_node: 1,
            load_cases: vec![LoadCase {
                name: "oracle",
                loads: vec![NodalLoad {
                    node: 2,
                    fx: 0.0,
                    fy: -load,
                }],
            }],
            span_m: 4.0,
            bay_m: 2.0,
            level_m: 3.0,
        };
        let design = DecodedDesign {
            nodes: ground.nodes.clone(),
            active: (0..3)
                .map(|member_index| ActiveMember {
                    member_index,
                    section_index: 8,
                })
                .collect(),
        };
        (ground, design)
    }

    #[test]
    fn triangular_oracle_matches_equilibrium_and_virtual_work() {
        let evidence = triangular_oracle().unwrap();
        for (actual, expected) in evidence.fem_forces_n.iter().zip(evidence.analytic_forces_n) {
            assert!((actual - expected).abs() < 1.0e-8 * evidence.load_n);
        }
        assert!(
            (evidence.fem_forces_n[1] - evidence.fem_forces_n[2]).abs() < 1.0e-10 * evidence.load_n
        );
        assert!((evidence.fem_displacement_m - evidence.analytic_displacement_m).abs() < 1.0e-10);
    }

    #[test]
    fn mechanism_is_typed_and_has_no_metrics() {
        let (ground, design) = triangle(100_000.0);
        let failure = analyze(
            &ground,
            &design,
            Scenario::TRAINING,
            Some(0),
            &WorkCounter::default(),
        )
        .unwrap_err();
        assert!(matches!(failure, AnalysisFailure::Singular { .. }));
    }

    #[test]
    fn equal_support_settlement_is_rigid_translation() {
        let (ground, design) = triangle(0.0);
        let settlement = -0.004;
        let metrics = analyze(
            &ground,
            &design,
            Scenario {
                name: "rigid-settlement",
                load_scale: 1.0,
                modulus_scale: 1.0,
                pinned_settlement_y_m: settlement,
                roller_settlement_y_m: settlement,
            },
            None,
            &WorkCounter::default(),
        )
        .unwrap();
        let displacement = &metrics.case_displacements[0];
        assert!((displacement[5] - settlement).abs() < 1.0e-12);
        assert!(
            metrics
                .member_forces_n
                .iter()
                .all(|force| force.abs() < 1.0e-6)
        );
    }
}
