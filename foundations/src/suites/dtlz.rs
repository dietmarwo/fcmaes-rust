//! DTLZ1–7 with objective-count-parametric deterministic reference fronts.

use std::f64::consts::PI;

use fcmaes_core::ReferencePoint;

use super::fronts::{halton, positive_sphere, simplex};
use super::{KnownOptimum, Suite, SuiteError, validate_decision};

/// DTLZ function family member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DtlzKind {
    /// Linear simplex front with a multimodal distance function.
    Dtlz1,
    /// Positive unit-hypersphere front.
    Dtlz2,
    /// DTLZ2 geometry with a multimodal distance function.
    Dtlz3,
    /// Biased density on the DTLZ2 front.
    Dtlz4,
    /// Degenerate manifold controlled by the distance function.
    Dtlz5,
    /// DTLZ5 geometry with a nonuniform distance function.
    Dtlz6,
    /// Disconnected front.
    Dtlz7,
}

/// Configured DTLZ problem.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dtlz {
    kind: DtlzKind,
    dimension: usize,
    objectives: usize,
}

impl Dtlz {
    /// Construct a problem using the conventional family-specific `k`.
    ///
    /// # Errors
    ///
    /// Returns [`SuiteError::InvalidConfiguration`] when `objectives < 2`.
    pub fn new(kind: DtlzKind, objectives: usize) -> Result<Self, SuiteError> {
        if objectives < 2 {
            return Err(SuiteError::InvalidConfiguration);
        }
        let k = match kind {
            DtlzKind::Dtlz1 => 5,
            DtlzKind::Dtlz7 => 20,
            _ => 10,
        };
        Ok(Self {
            kind,
            dimension: objectives + k - 1,
            objectives,
        })
    }

    /// Family member.
    pub fn kind(&self) -> DtlzKind {
        self.kind
    }

    /// All seven conventional problems for one objective count.
    pub fn all(objectives: usize) -> Result<[Self; 7], SuiteError> {
        Ok([
            Self::new(DtlzKind::Dtlz1, objectives)?,
            Self::new(DtlzKind::Dtlz2, objectives)?,
            Self::new(DtlzKind::Dtlz3, objectives)?,
            Self::new(DtlzKind::Dtlz4, objectives)?,
            Self::new(DtlzKind::Dtlz5, objectives)?,
            Self::new(DtlzKind::Dtlz6, objectives)?,
            Self::new(DtlzKind::Dtlz7, objectives)?,
        ])
    }

    fn distance(&self, decision: &[f64]) -> f64 {
        let tail = &decision[self.objectives - 1..];
        match self.kind {
            DtlzKind::Dtlz1 | DtlzKind::Dtlz3 => {
                100.0
                    * (tail.len() as f64
                        + tail
                            .iter()
                            .map(|value| (value - 0.5).powi(2) - (20.0 * PI * (value - 0.5)).cos())
                            .sum::<f64>())
            }
            DtlzKind::Dtlz2 | DtlzKind::Dtlz4 | DtlzKind::Dtlz5 => {
                tail.iter().map(|value| (value - 0.5).powi(2)).sum()
            }
            DtlzKind::Dtlz6 => tail.iter().map(|value| value.powf(0.1)).sum(),
            DtlzKind::Dtlz7 => 1.0 + 9.0 * tail.iter().sum::<f64>() / tail.len() as f64,
        }
    }

    fn sphere_from_angles(&self, angles: &[f64], radius: f64) -> Vec<f64> {
        (0..self.objectives)
            .map(|objective| {
                let cosines = self.objectives - objective - 1;
                let mut value = radius;
                for angle in &angles[..cosines] {
                    value *= angle.cos();
                }
                if objective > 0 {
                    value *= angles[cosines].sin();
                }
                value
            })
            .collect()
    }

    fn disconnected_front(&self, points: usize) -> Vec<Vec<f64>> {
        const INTERVALS: [(f64, f64); 2] =
            [(0.0, 0.251_411_836_1), (0.631_626_530_7, 0.859_400_856_6)];
        const PRIMES: [usize; 7] = [2, 3, 5, 7, 11, 13, 17];
        (1..=points)
            .map(|index| {
                let mut front: Vec<f64> = (0..self.objectives - 1)
                    .map(|axis| {
                        let interval = INTERVALS[(index >> axis) & 1];
                        interval.0 + (interval.1 - interval.0) * halton(index, PRIMES[axis])
                    })
                    .collect();
                let last = 2.0 * self.objectives as f64
                    - front
                        .iter()
                        .map(|value| value * (1.0 + (3.0 * PI * value).sin()))
                        .sum::<f64>();
                front.push(last);
                front
            })
            .collect()
    }
}

impl Suite for Dtlz {
    fn name(&self) -> &'static str {
        match self.kind {
            DtlzKind::Dtlz1 => "dtlz1",
            DtlzKind::Dtlz2 => "dtlz2",
            DtlzKind::Dtlz3 => "dtlz3",
            DtlzKind::Dtlz4 => "dtlz4",
            DtlzKind::Dtlz5 => "dtlz5",
            DtlzKind::Dtlz6 => "dtlz6",
            DtlzKind::Dtlz7 => "dtlz7",
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn objectives(&self) -> usize {
        self.objectives
    }

    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0; self.dimension], vec![1.0; self.dimension])
    }

    fn evaluate(&self, decision: &[f64]) -> Result<Vec<f64>, SuiteError> {
        let (lower, upper) = self.bounds();
        validate_decision(decision, &lower, &upper)?;
        let g = self.distance(decision);
        match self.kind {
            DtlzKind::Dtlz1 => Ok((0..self.objectives)
                .map(|objective| {
                    let factors = self.objectives - objective - 1;
                    let mut value = 0.5 * (1.0 + g);
                    for factor in &decision[..factors] {
                        value *= factor;
                    }
                    if objective > 0 {
                        value *= 1.0 - decision[factors];
                    }
                    value
                })
                .collect()),
            DtlzKind::Dtlz2 | DtlzKind::Dtlz3 | DtlzKind::Dtlz4 => {
                let angles: Vec<f64> = decision[..self.objectives - 1]
                    .iter()
                    .map(|value| {
                        let coordinate = if self.kind == DtlzKind::Dtlz4 {
                            value.powi(100)
                        } else {
                            *value
                        };
                        coordinate * PI / 2.0
                    })
                    .collect();
                Ok(self.sphere_from_angles(&angles, 1.0 + g))
            }
            DtlzKind::Dtlz5 | DtlzKind::Dtlz6 => {
                let mut angles = Vec::with_capacity(self.objectives - 1);
                angles.push(decision[0] * PI / 2.0);
                angles.extend(
                    decision[1..self.objectives - 1]
                        .iter()
                        .map(|value| PI * (1.0 + 2.0 * g * value) / (4.0 * (1.0 + g))),
                );
                Ok(self.sphere_from_angles(&angles, 1.0 + g))
            }
            DtlzKind::Dtlz7 => {
                let mut values = decision[..self.objectives - 1].to_vec();
                let h = self.objectives as f64
                    - values
                        .iter()
                        .map(|value| value / (1.0 + g) * (1.0 + (3.0 * PI * value).sin()))
                        .sum::<f64>();
                values.push((1.0 + g) * h);
                Ok(values)
            }
        }
    }

    fn known_optimum(&self) -> Option<KnownOptimum> {
        None
    }

    fn reference_front(&self, points: usize) -> Option<Vec<Vec<f64>>> {
        Some(match self.kind {
            DtlzKind::Dtlz1 => simplex(points, self.objectives, 0.5),
            DtlzKind::Dtlz2 | DtlzKind::Dtlz3 | DtlzKind::Dtlz4 => {
                positive_sphere(points, self.objectives)
            }
            DtlzKind::Dtlz5 | DtlzKind::Dtlz6 => (0..points)
                .map(|index| {
                    let fraction = if points == 1 {
                        0.5
                    } else {
                        index as f64 / (points - 1) as f64
                    };
                    let mut angles = vec![PI / 4.0; self.objectives - 1];
                    angles[0] = fraction * PI / 2.0;
                    self.sphere_from_angles(&angles, 1.0)
                })
                .collect(),
            DtlzKind::Dtlz7 => self.disconnected_front(points),
        })
    }

    fn reference_point(&self) -> Option<ReferencePoint> {
        let mut point = vec![1.1; self.objectives];
        if self.kind == DtlzKind::Dtlz7 {
            point[self.objectives - 1] = 2.2 * self.objectives as f64;
        }
        ReferencePoint::new(point).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcmaes_core::pareto_indices;

    #[test]
    fn reference_fronts_have_the_expected_geometry() {
        for objectives in [2, 3, 5] {
            for problem in Dtlz::all(objectives).unwrap() {
                let front = problem.reference_front(257).unwrap();
                assert_eq!(front.len(), 257);
                assert!(front.iter().all(|point| point.len() == objectives));
                match problem.kind() {
                    DtlzKind::Dtlz1 => assert!(
                        front
                            .iter()
                            .all(|point| (point.iter().sum::<f64>() - 0.5).abs() < 1.0e-12)
                    ),
                    DtlzKind::Dtlz2
                    | DtlzKind::Dtlz3
                    | DtlzKind::Dtlz4
                    | DtlzKind::Dtlz5
                    | DtlzKind::Dtlz6 => assert!(front.iter().all(|point| {
                        (point.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() < 1.0e-12
                    })),
                    DtlzKind::Dtlz7 => {}
                }
                assert_eq!(
                    pareto_indices(&front, objectives).unwrap().len(),
                    front.len(),
                    "{} m={objectives}",
                    problem.name()
                );
                assert_eq!(front, problem.reference_front(257).unwrap());
            }
        }
    }

    #[test]
    fn minimum_distance_decisions_land_on_the_reference_geometry() {
        for problem in Dtlz::all(3).unwrap() {
            let tail = match problem.kind() {
                DtlzKind::Dtlz6 | DtlzKind::Dtlz7 => 0.0,
                _ => 0.5,
            };
            let mut decision = vec![tail; problem.dimension()];
            decision[..2].fill(0.3);
            if problem.kind() == DtlzKind::Dtlz7 {
                decision[2..].fill(0.0);
            }
            let values = problem.evaluate(&decision).unwrap();
            match problem.kind() {
                DtlzKind::Dtlz1 => {
                    assert!((values.iter().sum::<f64>() - 0.5).abs() < 1.0e-12)
                }
                DtlzKind::Dtlz2
                | DtlzKind::Dtlz3
                | DtlzKind::Dtlz4
                | DtlzKind::Dtlz5
                | DtlzKind::Dtlz6 => assert!(
                    (values.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() < 1.0e-12
                ),
                DtlzKind::Dtlz7 => assert!(values.iter().all(|value| value.is_finite())),
            }
        }
    }
}
