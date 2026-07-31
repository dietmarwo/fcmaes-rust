//! Eight dimension-parametric single-objective functions.

use std::f64::consts::{E, PI};

use super::{KnownOptimum, Suite, SuiteError, validate_decision};

/// Classic unshifted single-objective problem.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Classic {
    /// Convex sum of squares.
    Sphere(usize),
    /// Curved Rosenbrock valley.
    Rosenbrock(usize),
    /// Separable periodic Rastrigin landscape.
    Rastrigin(usize),
    /// Nearly flat, periodic Ackley landscape.
    Ackley(usize),
    /// Product-coupled Griewank landscape.
    Griewank(usize),
    /// Deceptive Schwefel landscape.
    Schwefel(usize),
    /// Levy landscape.
    Levy(usize),
    /// Polynomial Zakharov landscape.
    Zakharov(usize),
}

impl Classic {
    fn raw(self, x: &[f64]) -> f64 {
        match self {
            Self::Sphere(_) => x.iter().map(|value| value * value).sum(),
            Self::Rosenbrock(_) => x
                .windows(2)
                .map(|pair| 100.0 * (pair[1] - pair[0] * pair[0]).powi(2) + (pair[0] - 1.0).powi(2))
                .sum(),
            Self::Rastrigin(_) => {
                10.0 * x.len() as f64
                    + x.iter()
                        .map(|value| value * value - 10.0 * (2.0 * PI * value).cos())
                        .sum::<f64>()
            }
            Self::Ackley(_) => {
                let n = x.len() as f64;
                let squares = x.iter().map(|value| value * value).sum::<f64>() / n;
                let cosines = x.iter().map(|value| (2.0 * PI * value).cos()).sum::<f64>() / n;
                -20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + 20.0 + E
            }
            Self::Griewank(_) => {
                x.iter().map(|value| value * value).sum::<f64>() / 4000.0
                    - x.iter()
                        .enumerate()
                        .map(|(index, value)| (value / ((index + 1) as f64).sqrt()).cos())
                        .product::<f64>()
                    + 1.0
            }
            Self::Schwefel(_) => {
                418.982_887_272_433_8 * x.len() as f64
                    - x.iter()
                        .map(|value| value * value.abs().sqrt().sin())
                        .sum::<f64>()
            }
            Self::Levy(_) => {
                let w: Vec<f64> = x.iter().map(|value| 1.0 + (value - 1.0) / 4.0).collect();
                (PI * w[0]).sin().powi(2)
                    + w.windows(2)
                        .map(|pair| {
                            (pair[0] - 1.0).powi(2)
                                * (1.0 + 10.0 * (PI * pair[0] + 1.0).sin().powi(2))
                        })
                        .sum::<f64>()
                    + (w[w.len() - 1] - 1.0).powi(2)
                        * (1.0 + (2.0 * PI * w[w.len() - 1]).sin().powi(2))
            }
            Self::Zakharov(_) => {
                let squares = x.iter().map(|value| value * value).sum::<f64>();
                let weighted = x
                    .iter()
                    .enumerate()
                    .map(|(index, value)| 0.5 * (index + 1) as f64 * value)
                    .sum::<f64>();
                squares + weighted.powi(2) + weighted.powi(4)
            }
        }
    }

    /// All eight problems at dimension `dimension`.
    pub fn all(dimension: usize) -> [Self; 8] {
        [
            Self::Sphere(dimension),
            Self::Rosenbrock(dimension),
            Self::Rastrigin(dimension),
            Self::Ackley(dimension),
            Self::Griewank(dimension),
            Self::Schwefel(dimension),
            Self::Levy(dimension),
            Self::Zakharov(dimension),
        ]
    }
}

impl Suite for Classic {
    fn name(&self) -> &'static str {
        match self {
            Self::Sphere(_) => "sphere",
            Self::Rosenbrock(_) => "rosenbrock",
            Self::Rastrigin(_) => "rastrigin",
            Self::Ackley(_) => "ackley",
            Self::Griewank(_) => "griewank",
            Self::Schwefel(_) => "schwefel",
            Self::Levy(_) => "levy",
            Self::Zakharov(_) => "zakharov",
        }
    }

    fn dimension(&self) -> usize {
        match self {
            Self::Sphere(value)
            | Self::Rosenbrock(value)
            | Self::Rastrigin(value)
            | Self::Ackley(value)
            | Self::Griewank(value)
            | Self::Schwefel(value)
            | Self::Levy(value)
            | Self::Zakharov(value) => *value,
        }
    }

    fn objectives(&self) -> usize {
        1
    }

    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let (lower, upper) = match self {
            Self::Sphere(_) | Self::Rastrigin(_) => (-5.12, 5.12),
            Self::Rosenbrock(_) | Self::Zakharov(_) => (-5.0, 10.0),
            Self::Ackley(_) => (-32.768, 32.768),
            Self::Griewank(_) => (-600.0, 600.0),
            Self::Schwefel(_) => (-500.0, 500.0),
            Self::Levy(_) => (-10.0, 10.0),
        };
        (vec![lower; self.dimension()], vec![upper; self.dimension()])
    }

    fn evaluate(&self, decision: &[f64]) -> Result<Vec<f64>, SuiteError> {
        if self.dimension() == 0 {
            return Err(SuiteError::InvalidConfiguration);
        }
        let (lower, upper) = self.bounds();
        validate_decision(decision, &lower, &upper)?;
        Ok(vec![self.raw(decision)])
    }

    fn known_optimum(&self) -> Option<KnownOptimum> {
        if self.dimension() == 0 {
            return None;
        }
        let decision = match self {
            Self::Rosenbrock(_) | Self::Levy(_) => vec![1.0; self.dimension()],
            Self::Schwefel(_) => vec![420.968_746_227_503_6; self.dimension()],
            _ => vec![0.0; self.dimension()],
        };
        Some(KnownOptimum {
            objectives: vec![self.raw(&decision)],
            decision,
        })
    }

    fn reference_front(&self, _points: usize) -> Option<Vec<Vec<f64>>> {
        None
    }

    fn reference_point(&self) -> Option<fcmaes_core::ReferencePoint> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_solutions_and_shapes_are_consistent() {
        for dimension in [2, 10, 40] {
            for problem in Classic::all(dimension) {
                let optimum = problem.known_optimum().unwrap();
                let evaluated = problem.evaluate(&optimum.decision).unwrap();
                assert_eq!(evaluated, optimum.objectives, "{}", problem.name());
                assert!(
                    evaluated[0].abs() < 1.0e-12,
                    "{}={}",
                    problem.name(),
                    evaluated[0]
                );
                let (lower, upper) = problem.bounds();
                assert_eq!(lower.len(), dimension);
                assert_eq!(upper.len(), dimension);
                assert!(
                    optimum
                        .decision
                        .iter()
                        .zip(&lower)
                        .zip(&upper)
                        .all(|((&value, &lo), &hi)| value >= lo && value <= hi)
                );
            }
        }
    }

    #[test]
    fn invalid_decisions_fail_closed() {
        let sphere = Classic::Sphere(2);
        assert_eq!(sphere.evaluate(&[0.0]), Err(SuiteError::DimensionMismatch));
        assert_eq!(
            sphere.evaluate(&[0.0, f64::NAN]),
            Err(SuiteError::NonFiniteDecision)
        );
        assert_eq!(sphere.evaluate(&[6.0, 0.0]), Err(SuiteError::OutOfBounds));
    }
}
