//! Lennard-Jones cluster energy, gradients, encodings, and reference metadata.
//!
//! The scalar targets are source-cited putative minima. Coordinates are not
//! distributed; [`load_coordinates`] lets a reader audit a structure obtained
//! separately.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

use fcmaes_core::Rng;
use sha2::{Digest, Sha256};

use super::{KnownOptimum, Suite, SuiteError, validate_decision};

/// Cambridge table containing the source-cited scalar targets.
pub const REFERENCE_SOURCE: &str =
    "https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html";
/// Distance below which the finite soft-core continuation is used.
pub const MINIMUM_DISTANCE: f64 = 0.75;
/// A result is successful when it is no more than this far above the target.
pub const SUCCESS_TOLERANCE: f64 = 1.0e-3;
/// Coordination-number neighbor cutoff used by the descriptor pilot.
pub const COORDINATION_CUTOFF: f64 = 1.35;

/// A source-cited putative minimum energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceTarget {
    /// Number of atoms.
    pub atoms: usize,
    /// Putative minimum energy in reduced units.
    pub energy: f64,
    /// Point-group label from the Cambridge table.
    pub point_group: &'static str,
}

/// Targets selected for the scaling study.
pub const REFERENCE_TARGETS: [ReferenceTarget; 5] = [
    ReferenceTarget {
        atoms: 13,
        energy: -44.326_801,
        point_group: "Ih",
    },
    ReferenceTarget {
        atoms: 38,
        energy: -173.928_427,
        point_group: "Oh",
    },
    ReferenceTarget {
        atoms: 55,
        energy: -279.248_470,
        point_group: "Ih",
    },
    ReferenceTarget {
        atoms: 75,
        energy: -397.492_331,
        point_group: "D5h",
    },
    ReferenceTarget {
        atoms: 98,
        energy: -543.665_361,
        point_group: "Td",
    },
];

/// Coordinate representation used by the optimizer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Parameterization {
    /// Optimize all `3N` Cartesian coordinates.
    Free,
    /// Remove translation and rotation by fixing the first three atoms.
    FixedFrame,
}

impl Parameterization {
    /// Stable artifact label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::FixedFrame => "fixed-frame",
        }
    }
}

/// One potential evaluation and its diagnostic count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyEvaluation {
    /// Cluster energy in reduced units.
    pub energy: f64,
    /// Number of pairs evaluated through the finite soft-core continuation.
    pub overlap_pairs: usize,
}

/// Optional audit of a separately obtained reference structure.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceAudit {
    /// Re-evaluated structure energy.
    pub measured_energy: f64,
    /// Source-cited target energy.
    pub target_energy: f64,
    /// Absolute difference between measured and target energy.
    pub absolute_error: f64,
    /// Whether the audit meets the frozen `1e-6` tolerance.
    pub matches: bool,
    /// SHA-256 digest of the exact external coordinate file that was audited.
    pub coordinate_sha256: String,
}

/// Loader or canonical-frame failure.
#[derive(Debug)]
pub enum LjError {
    /// File-system failure.
    Io(std::io::Error),
    /// Coordinate text was malformed.
    Format(String),
    /// Atom count differs from the requested problem.
    AtomCount { expected: usize, actual: usize },
    /// The first three atoms cannot define a unique frame.
    DegenerateAnchors,
    /// The selected atom count has no source-cited target.
    MissingTarget(usize),
}

impl fmt::Display for LjError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "failed to read coordinate file: {error}"),
            Self::Format(message) => write!(formatter, "invalid coordinate file: {message}"),
            Self::AtomCount { expected, actual } => {
                write!(formatter, "expected {expected} atoms, found {actual}")
            }
            Self::DegenerateAnchors => formatter.write_str(
                "atoms 0, 1, and 2 must be distinct and non-collinear for fixed-frame encoding",
            ),
            Self::MissingTarget(atoms) => {
                write!(
                    formatter,
                    "no source-cited Lennard-Jones target for N={atoms}"
                )
            }
        }
    }
}

impl Error for LjError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LjError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Lennard-Jones cluster benchmark in reduced units (`epsilon = sigma = 1`).
#[derive(Clone, Debug)]
pub struct LennardJones {
    atoms: usize,
    parameterization: Parameterization,
}

impl LennardJones {
    /// Construct a problem with at least three atoms.
    pub fn new(atoms: usize, parameterization: Parameterization) -> Result<Self, SuiteError> {
        if atoms < 3 {
            return Err(SuiteError::InvalidConfiguration);
        }
        Ok(Self {
            atoms,
            parameterization,
        })
    }

    /// Number of atoms.
    pub const fn atoms(&self) -> usize {
        self.atoms
    }

    /// Active coordinate representation.
    pub const fn parameterization(&self) -> Parameterization {
        self.parameterization
    }

    /// Half-width of the frozen Cartesian box.
    pub fn half_width(&self) -> f64 {
        1.5 * (self.atoms as f64).cbrt()
    }

    /// Source-cited target for this size, if it is part of the study.
    pub fn target(&self) -> Option<ReferenceTarget> {
        REFERENCE_TARGETS
            .iter()
            .copied()
            .find(|target| target.atoms == self.atoms)
    }

    /// Decode an optimizer vector into Cartesian coordinates.
    pub fn decode(&self, decision: &[f64]) -> Result<Vec<[f64; 3]>, SuiteError> {
        let (lower, upper) = self.bounds();
        validate_decision(decision, &lower, &upper)?;
        match self.parameterization {
            Parameterization::Free => Ok(decision
                .chunks_exact(3)
                .map(|point| [point[0], point[1], point[2]])
                .collect()),
            Parameterization::FixedFrame => {
                let mut points = Vec::with_capacity(self.atoms);
                points.push([0.0, 0.0, 0.0]);
                points.push([decision[0], 0.0, 0.0]);
                points.push([decision[1], decision[2], 0.0]);
                points.extend(
                    decision[3..]
                        .chunks_exact(3)
                        .map(|point| [point[0], point[1], point[2]]),
                );
                Ok(points)
            }
        }
    }

    /// Encode coordinates, canonicalizing them when the fixed frame is used.
    pub fn encode(&self, points: &[[f64; 3]]) -> Result<Vec<f64>, LjError> {
        if points.len() != self.atoms {
            return Err(LjError::AtomCount {
                expected: self.atoms,
                actual: points.len(),
            });
        }
        if points.iter().flatten().any(|value| !value.is_finite()) {
            return Err(LjError::Format("coordinates must be finite".to_owned()));
        }
        if self.parameterization == Parameterization::Free {
            return Ok(points.iter().flatten().copied().collect());
        }
        let origin = points[0];
        let first = sub(points[1], origin);
        let first_norm = norm(first);
        if first_norm <= 1.0e-12 {
            return Err(LjError::DegenerateAnchors);
        }
        let axis_x = scale(first, first_norm.recip());
        let second = sub(points[2], origin);
        let second_x = dot(second, axis_x);
        let perpendicular = sub(second, scale(axis_x, second_x));
        let second_y = norm(perpendicular);
        if second_y <= 1.0e-12 {
            return Err(LjError::DegenerateAnchors);
        }
        let axis_y = scale(perpendicular, second_y.recip());
        let axis_z = cross(axis_x, axis_y);
        let mut encoded = Vec::with_capacity(3 * self.atoms - 6);
        encoded.extend([first_norm, second_x, second_y]);
        for point in &points[3..] {
            let relative = sub(*point, origin);
            encoded.extend([
                dot(relative, axis_x),
                dot(relative, axis_y),
                dot(relative, axis_z),
            ]);
        }
        Ok(encoded)
    }

    /// Evaluate energy and overlap count from an optimizer vector.
    pub fn energy(&self, decision: &[f64]) -> Result<EnergyEvaluation, SuiteError> {
        Ok(energy_and_cartesian_gradient(&self.decode(decision)?).0)
    }

    /// Evaluate energy, analytic decision-space gradient, and overlap count in
    /// one pair traversal.
    pub fn value_gradient(
        &self,
        decision: &[f64],
    ) -> Result<(EnergyEvaluation, Vec<f64>), SuiteError> {
        let points = self.decode(decision)?;
        let (evaluation, cartesian) = energy_and_cartesian_gradient(&points);
        let gradient = match self.parameterization {
            Parameterization::Free => cartesian.iter().flatten().copied().collect(),
            Parameterization::FixedFrame => {
                let mut result = Vec::with_capacity(self.dimension());
                result.extend([cartesian[1][0], cartesian[2][0], cartesian[2][1]]);
                result.extend(cartesian[3..].iter().flatten().copied());
                result
            }
        };
        Ok((evaluation, gradient))
    }

    /// Deterministic compact candidate shared by every optimizer arm.
    pub fn initial_decision(&self, seed: u64) -> Result<Vec<f64>, LjError> {
        let mut rng = Rng::new(seed);
        let radius = 0.8 * (self.atoms as f64).cbrt();
        let mut points: Vec<[f64; 3]> = Vec::with_capacity(self.atoms);
        let max_attempts = self.atoms * 100_000;
        for _ in 0..max_attempts {
            let candidate = [
                radius * (2.0 * rng.uniform01() - 1.0),
                radius * (2.0 * rng.uniform01() - 1.0),
                radius * (2.0 * rng.uniform01() - 1.0),
            ];
            if dot(candidate, candidate) > radius * radius
                || points.iter().any(|point| {
                    norm_squared(sub(candidate, *point)) < MINIMUM_DISTANCE * MINIMUM_DISTANCE
                })
            {
                continue;
            }
            points.push(candidate);
            if points.len() == self.atoms {
                let center = centroid(&points);
                for point in &mut points {
                    *point = sub(*point, center);
                }
                if self.parameterization == Parameterization::FixedFrame {
                    let origin = (0..points.len())
                        .min_by(|&left, &right| {
                            norm_squared(points[left]).total_cmp(&norm_squared(points[right]))
                        })
                        .unwrap_or(0);
                    points.swap(0, origin);
                    let first = (1..points.len())
                        .min_by(|&left, &right| {
                            norm_squared(sub(points[left], points[0]))
                                .total_cmp(&norm_squared(sub(points[right], points[0])))
                        })
                        .unwrap_or(1);
                    points.swap(1, first);
                    let axis = sub(points[1], points[0]);
                    let axis_squared = norm_squared(axis);
                    let third = (2..points.len())
                        .max_by(|&left, &right| {
                            let perpendicular = |index| {
                                let offset = sub(points[index], points[0]);
                                norm_squared(offset) - dot(offset, axis).powi(2) / axis_squared
                            };
                            perpendicular(left).total_cmp(&perpendicular(right))
                        })
                        .unwrap_or(2);
                    points.swap(2, third);
                }
                return self.encode(&points);
            }
        }
        Err(LjError::Format(format!(
            "compact initializer failed after {max_attempts} attempts"
        )))
    }

    /// Radius-of-gyration and coordination descriptors used by the QD pilot.
    pub fn descriptors(&self, decision: &[f64]) -> Result<[f64; 2], SuiteError> {
        self.descriptors_at_cutoff(decision, COORDINATION_CUTOFF)
    }

    /// Descriptor evaluation at an explicit coordination cutoff, used only
    /// for the pilot's numerical-sensitivity check.
    pub fn descriptors_at_cutoff(
        &self,
        decision: &[f64],
        coordination_cutoff: f64,
    ) -> Result<[f64; 2], SuiteError> {
        let points = self.decode(decision)?;
        let center = centroid(&points);
        let radius = (points
            .iter()
            .map(|point| norm_squared(sub(*point, center)))
            .sum::<f64>()
            / self.atoms as f64)
            .sqrt()
            / (self.atoms as f64).cbrt();
        let cutoff_squared = coordination_cutoff * coordination_cutoff;
        let mut neighbors = 0;
        for i in 0..self.atoms {
            for j in (i + 1)..self.atoms {
                if norm_squared(sub(points[i], points[j])) <= cutoff_squared {
                    neighbors += 2;
                }
            }
        }
        Ok([radius, neighbors as f64 / self.atoms as f64])
    }

    /// Audit a separately obtained structure against the source-cited target.
    pub fn audit_reference(&self, path: &Path) -> Result<ReferenceAudit, LjError> {
        let target = self.target().ok_or(LjError::MissingTarget(self.atoms))?;
        let bytes = fs::read(path)?;
        let points = load_coordinates(path, self.atoms)?;
        let measured = energy_and_cartesian_gradient(&points).0.energy;
        let absolute_error = (measured - target.energy).abs();
        Ok(ReferenceAudit {
            measured_energy: measured,
            target_energy: target.energy,
            absolute_error,
            matches: absolute_error <= 1.0e-6,
            coordinate_sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }
}

impl Suite for LennardJones {
    fn name(&self) -> &'static str {
        "lennard-jones"
    }

    fn dimension(&self) -> usize {
        match self.parameterization {
            Parameterization::Free => 3 * self.atoms,
            Parameterization::FixedFrame => 3 * self.atoms - 6,
        }
    }

    fn objectives(&self) -> usize {
        1
    }

    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let half_width = self.half_width();
        match self.parameterization {
            Parameterization::Free => (
                vec![-half_width; self.dimension()],
                vec![half_width; self.dimension()],
            ),
            Parameterization::FixedFrame => {
                let mut lower = vec![-half_width; self.dimension()];
                let mut upper = vec![half_width; self.dimension()];
                lower[0] = 0.0;
                lower[2] = 0.0;
                upper[0] = half_width;
                upper[2] = half_width;
                (lower, upper)
            }
        }
    }

    fn evaluate(&self, decision: &[f64]) -> Result<Vec<f64>, SuiteError> {
        Ok(vec![self.energy(decision)?.energy])
    }

    fn known_optimum(&self) -> Option<KnownOptimum> {
        None
    }

    fn known_optimum_value(&self) -> Option<f64> {
        self.target().map(|target| target.energy)
    }

    fn reference_front(&self, _points: usize) -> Option<Vec<Vec<f64>>> {
        None
    }

    fn reference_point(&self) -> Option<fcmaes_core::ReferencePoint> {
        None
    }
}

/// Load plain `x y z`, `symbol x y z`, or conventional XYZ coordinates.
pub fn load_coordinates(path: &Path, expected_atoms: usize) -> Result<Vec<[f64; 3]>, LjError> {
    let text = fs::read_to_string(path)?;
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let first = lines.iter().position(|line| !line.is_empty());
    let Some(first) = first else {
        return Err(LjError::Format("file is empty".to_owned()));
    };
    let xyz_atoms = lines[first].parse::<usize>().ok();
    let coordinate_lines: Vec<&str> = if let Some(atoms) = xyz_atoms {
        if atoms != expected_atoms {
            return Err(LjError::AtomCount {
                expected: expected_atoms,
                actual: atoms,
            });
        }
        if lines.len() < first + atoms + 2 {
            return Err(LjError::Format(
                "XYZ file needs an atom-count line, comment line, and all coordinates".to_owned(),
            ));
        }
        lines[(first + 2)..]
            .iter()
            .copied()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect()
    } else {
        lines[first..]
            .iter()
            .copied()
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect()
    };
    let mut points = Vec::with_capacity(expected_atoms);
    for (index, line) in coordinate_lines.iter().enumerate() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let start = match fields.len() {
            3 => 0,
            4 => 1,
            _ => {
                return Err(LjError::Format(format!(
                    "coordinate line {} must contain x y z or symbol x y z",
                    index + 1
                )));
            }
        };
        let mut point = [0.0; 3];
        for axis in 0..3 {
            point[axis] = fields[start + axis].parse::<f64>().map_err(|_| {
                LjError::Format(format!("coordinate line {} is not numeric", index + 1))
            })?;
        }
        if point.iter().any(|value| !value.is_finite()) {
            return Err(LjError::Format(format!(
                "coordinate line {} is not finite",
                index + 1
            )));
        }
        points.push(point);
    }
    if points.len() != expected_atoms {
        return Err(LjError::AtomCount {
            expected: expected_atoms,
            actual: points.len(),
        });
    }
    Ok(points)
}

fn energy_and_cartesian_gradient(points: &[[f64; 3]]) -> (EnergyEvaluation, Vec<[f64; 3]>) {
    let mut energy = 0.0;
    let mut overlap_pairs = 0;
    let mut gradient = vec![[0.0; 3]; points.len()];
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let delta = sub(points[i], points[j]);
            let squared_distance = norm_squared(delta);
            let (pair_energy, derivative_s, guarded) = pair_value_derivative(squared_distance);
            energy += pair_energy;
            overlap_pairs += usize::from(guarded);
            let factor = 2.0 * derivative_s;
            for axis in 0..3 {
                let component = factor * delta[axis];
                gradient[i][axis] += component;
                gradient[j][axis] -= component;
            }
        }
    }
    (
        EnergyEvaluation {
            energy,
            overlap_pairs,
        },
        gradient,
    )
}

fn pair_value_derivative(squared_distance: f64) -> (f64, f64, bool) {
    let minimum_squared = MINIMUM_DISTANCE * MINIMUM_DISTANCE;
    if squared_distance >= minimum_squared {
        let inverse = squared_distance.recip();
        let inverse3 = inverse * inverse * inverse;
        let inverse6 = inverse3 * inverse3;
        let value = 4.0 * (inverse6 - inverse3);
        let derivative = -24.0 * inverse6 * inverse + 12.0 * inverse3 * inverse;
        return (value, derivative, false);
    }
    let inverse = minimum_squared.recip();
    let inverse3 = inverse * inverse * inverse;
    let inverse6 = inverse3 * inverse3;
    let boundary_value = 4.0 * (inverse6 - inverse3);
    let boundary_derivative = -24.0 * inverse6 * inverse + 12.0 * inverse3 * inverse;
    let curvature = boundary_derivative.abs() / minimum_squared;
    let offset = squared_distance - minimum_squared;
    (
        boundary_value + boundary_derivative * offset + 0.5 * curvature * offset * offset,
        boundary_derivative + curvature * offset,
        true,
    )
}

fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let mut center = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            center[axis] += point[axis];
        }
    }
    scale(center, (points.len() as f64).recip())
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm_squared(vector: [f64; 3]) -> f64 {
    dot(vector, vector)
}

fn norm(vector: [f64; 3]) -> f64 {
    norm_squared(vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn free(points: &[[f64; 3]]) -> LennardJones {
        LennardJones::new(points.len(), Parameterization::Free).unwrap()
    }

    #[test]
    fn pair_curve_matches_closed_form_and_minimum() {
        let minimum = 2.0_f64.powf(1.0 / 6.0);
        let points = [[0.0, 0.0, 0.0], [minimum, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let pair = pair_value_derivative(minimum * minimum);
        assert!((pair.0 + 1.0).abs() < 1.0e-12);
        assert!(pair.1.abs() < 1.0e-12);
        let distance = 1.5_f64;
        let expected = 4.0 * (distance.powi(-12) - distance.powi(-6));
        assert!((pair_value_derivative(distance * distance).0 - expected).abs() < 1.0e-14);
        assert!(
            free(&points)
                .energy(&points.iter().flatten().copied().collect::<Vec<_>>())
                .is_ok()
        );
    }

    #[test]
    fn soft_core_is_finite_continuous_monotone_and_counted() {
        let boundary = MINIMUM_DISTANCE * MINIMUM_DISTANCE;
        let below = pair_value_derivative(boundary - 1.0e-10);
        let at = pair_value_derivative(boundary);
        assert!((below.0 - at.0).abs() < 1.0e-6);
        assert!(below.1.is_finite());
        assert!(pair_value_derivative(0.0).0 > pair_value_derivative(boundary / 2.0).0);
        let problem = LennardJones::new(3, Parameterization::Free).unwrap();
        let evaluation = problem
            .energy(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(evaluation.overlap_pairs, 1);
        assert!(evaluation.energy.is_finite());
    }

    #[test]
    fn analytic_gradient_matches_central_difference() {
        for parameterization in [Parameterization::Free, Parameterization::FixedFrame] {
            let problem = LennardJones::new(13, parameterization).unwrap();
            let decision = problem.initial_decision(17).unwrap();
            let (_, gradient) = problem.value_gradient(&decision).unwrap();
            let step = 1.0e-6;
            for axis in 0..decision.len() {
                let mut low = decision.clone();
                let mut high = decision.clone();
                low[axis] -= step;
                high[axis] += step;
                let (lower, upper) = problem.bounds();
                if low[axis] < lower[axis] || high[axis] > upper[axis] {
                    continue;
                }
                let numerical = (problem.energy(&high).unwrap().energy
                    - problem.energy(&low).unwrap().energy)
                    / (2.0 * step);
                let scale = 1.0_f64.max(numerical.abs()).max(gradient[axis].abs());
                assert!(
                    (numerical - gradient[axis]).abs() / scale < 2.0e-6,
                    "parameterization={parameterization:?} axis={axis} numerical={numerical} analytic={}",
                    gradient[axis]
                );
            }
        }
    }

    #[test]
    fn fixed_frame_removes_six_dimensions_and_preserves_energy() {
        let free_problem = LennardJones::new(13, Parameterization::Free).unwrap();
        let fixed_problem = LennardJones::new(13, Parameterization::FixedFrame).unwrap();
        assert_eq!(free_problem.dimension(), 39);
        assert_eq!(fixed_problem.dimension(), 33);
        let free_decision = free_problem.initial_decision(19).unwrap();
        let points = free_problem.decode(&free_decision).unwrap();
        let fixed_decision = fixed_problem.encode(&points).unwrap();
        let fixed_points = fixed_problem.decode(&fixed_decision).unwrap();
        let free_energy = energy_and_cartesian_gradient(&points).0.energy;
        let fixed_energy = energy_and_cartesian_gradient(&fixed_points).0.energy;
        assert!((free_energy - fixed_energy).abs() < 1.0e-10);
        assert!(fixed_points[1][0] > 0.0);
        assert!(fixed_points[2][1] > 0.0);
    }

    #[test]
    fn rigid_transform_leaves_energy_unchanged() {
        let problem = LennardJones::new(13, Parameterization::Free).unwrap();
        let decision = problem.initial_decision(23).unwrap();
        let points = problem.decode(&decision).unwrap();
        let angle = 0.731_f64;
        let transformed: Vec<[f64; 3]> = points
            .iter()
            .map(|point| {
                [
                    angle.cos() * point[0] - angle.sin() * point[1] + 2.0,
                    angle.sin() * point[0] + angle.cos() * point[1] - 1.0,
                    point[2] + 0.5,
                ]
            })
            .collect();
        let first = energy_and_cartesian_gradient(&points).0.energy;
        let second = energy_and_cartesian_gradient(&transformed).0.energy;
        assert!((first - second).abs() < 1.0e-10);
    }

    #[test]
    fn compact_initializer_respects_bounds_and_separation() {
        for atoms in [13, 38, 55, 75, 98] {
            for parameterization in [Parameterization::Free, Parameterization::FixedFrame] {
                let problem = LennardJones::new(atoms, parameterization).unwrap();
                for seed in 0..32 {
                    let decision = problem.initial_decision(atoms as u64 + seed).unwrap();
                    let points = problem.decode(&decision).unwrap();
                    let evaluation = problem.energy(&decision).unwrap();
                    assert_eq!(evaluation.overlap_pairs, 0);
                    assert_eq!(points.len(), atoms);
                }
            }
        }
    }

    #[test]
    fn coordinate_loader_has_no_silent_fallback() {
        let directory = std::env::temp_dir();
        let path = directory.join(format!("fcmaes-lj-{}.xyz", std::process::id()));
        let mut file = fs::File::create(&path).unwrap();
        writeln!(file, "3\nlocal fixture\nX 0 0 0\nX 1.1 0 0\nX 0 1.1 0").unwrap();
        drop(file);
        let points = load_coordinates(&path, 3).unwrap();
        assert_eq!(points.len(), 3);
        let mut file = fs::File::create(&path).unwrap();
        writeln!(file, "3\n\nX 0 0 0\nX 1.1 0 0\nX 0 1.1 0").unwrap();
        drop(file);
        assert_eq!(load_coordinates(&path, 3).unwrap().len(), 3);
        fs::remove_file(&path).unwrap();
        assert!(matches!(load_coordinates(&path, 3), Err(LjError::Io(_))));
    }

    #[test]
    fn audit_records_an_unmatched_local_file_without_fallback() {
        let path =
            std::env::temp_dir().join(format!("fcmaes-lj-audit-{}.points", std::process::id()));
        let mut file = fs::File::create(&path).unwrap();
        for atom in 0..13 {
            writeln!(file, "{} 0 0", 2 * atom).unwrap();
        }
        drop(file);
        let problem = LennardJones::new(13, Parameterization::Free).unwrap();
        let audit = problem.audit_reference(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert!(!audit.matches);
        assert!(audit.measured_energy.is_finite());
        assert!(audit.absolute_error > 1.0);
        assert_eq!(audit.coordinate_sha256.len(), 64);
        assert!(
            audit
                .coordinate_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        );
    }

    #[test]
    fn target_metadata_is_complete_and_well_formed() {
        assert_eq!(
            REFERENCE_TARGETS
                .iter()
                .map(|target| target.atoms)
                .collect::<Vec<_>>(),
            [13, 38, 55, 75, 98]
        );
        assert!(
            REFERENCE_TARGETS
                .windows(2)
                .all(|pair| pair[0].atoms < pair[1].atoms && pair[0].energy > pair[1].energy)
        );
        assert!(REFERENCE_TARGETS.iter().all(|target| {
            target.energy.is_finite() && target.energy < 0.0 && !target.point_group.is_empty()
        }));
        assert!(REFERENCE_SOURCE.starts_with("https://"));
    }
}
