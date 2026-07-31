//! Standard single- and multi-objective benchmark suites.

pub mod bbob;
pub mod cec;
pub mod classic;
pub mod dtlz;
pub mod fronts;
pub mod lennard_jones;
pub mod wfg;
pub mod zdt;

use std::error::Error;
use std::fmt;

use fcmaes_core::ReferencePoint;

use self::classic::Classic;
use self::dtlz::{Dtlz, DtlzKind};
use self::zdt::Zdt;

/// A known decision and its corresponding objective vector.
#[derive(Clone, Debug, PartialEq)]
pub struct KnownOptimum {
    /// Decision vector attaining the optimum.
    pub decision: Vec<f64>,
    /// Objective vector at that decision.
    pub objectives: Vec<f64>,
}

/// Validation failures from benchmark evaluators.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SuiteError {
    /// Decision dimension differs from the problem definition.
    DimensionMismatch,
    /// Decision contains NaN or infinity.
    NonFiniteDecision,
    /// Decision lies outside a declared bound.
    OutOfBounds,
    /// Problem dimensions/objective count are not defined by the suite.
    InvalidConfiguration,
}

impl fmt::Display for SuiteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DimensionMismatch => "decision dimension does not match the problem",
            Self::NonFiniteDecision => "decision values must be finite",
            Self::OutOfBounds => "decision lies outside the benchmark bounds",
            Self::InvalidConfiguration => "invalid benchmark configuration",
        })
    }
}

impl Error for SuiteError {}

/// Uniform interface shared by all foundation benchmark problems.
pub trait Suite {
    /// Stable problem identifier.
    fn name(&self) -> &'static str;
    /// Decision-space dimension.
    fn dimension(&self) -> usize;
    /// Number of minimized objectives.
    fn objectives(&self) -> usize;
    /// Per-coordinate lower and upper bounds.
    fn bounds(&self) -> (Vec<f64>, Vec<f64>);
    /// Evaluate a validated decision vector.
    fn evaluate(&self, decision: &[f64]) -> Result<Vec<f64>, SuiteError>;
    /// Known single-objective solution, when available.
    fn known_optimum(&self) -> Option<KnownOptimum>;
    /// Known single-objective value when the decision itself is not shipped.
    fn known_optimum_value(&self) -> Option<f64> {
        self.known_optimum()
            .and_then(|optimum| optimum.objectives.first().copied())
    }
    /// Deterministic analytic or constructed objective-space reference set.
    fn reference_front(&self, points: usize) -> Option<Vec<Vec<f64>>>;
    /// Explicit hypervolume reference point in native objective coordinates.
    fn reference_point(&self) -> Option<ReferencePoint>;
}

/// Construct a conventional teaching configuration from its stable name.
///
/// Classic problems use ten variables, ZDT problems use their publication
/// dimensions, and DTLZ problems use three objectives.
pub fn by_name(name: &str) -> Result<Box<dyn Suite>, SuiteError> {
    let suite: Box<dyn Suite> = match name.to_ascii_lowercase().as_str() {
        "sphere" => Box::new(Classic::Sphere(10)),
        "rosenbrock" => Box::new(Classic::Rosenbrock(10)),
        "rastrigin" => Box::new(Classic::Rastrigin(10)),
        "ackley" => Box::new(Classic::Ackley(10)),
        "griewank" => Box::new(Classic::Griewank(10)),
        "schwefel" => Box::new(Classic::Schwefel(10)),
        "levy" => Box::new(Classic::Levy(10)),
        "zakharov" => Box::new(Classic::Zakharov(10)),
        "zdt1" => Box::new(Zdt::Zdt1(30)),
        "zdt2" => Box::new(Zdt::Zdt2(30)),
        "zdt3" => Box::new(Zdt::Zdt3(30)),
        "zdt4" => Box::new(Zdt::Zdt4(10)),
        "zdt6" => Box::new(Zdt::Zdt6(30)),
        "dtlz1" => Box::new(Dtlz::new(DtlzKind::Dtlz1, 3)?),
        "dtlz2" => Box::new(Dtlz::new(DtlzKind::Dtlz2, 3)?),
        "dtlz3" => Box::new(Dtlz::new(DtlzKind::Dtlz3, 3)?),
        "dtlz4" => Box::new(Dtlz::new(DtlzKind::Dtlz4, 3)?),
        "dtlz5" => Box::new(Dtlz::new(DtlzKind::Dtlz5, 3)?),
        "dtlz6" => Box::new(Dtlz::new(DtlzKind::Dtlz6, 3)?),
        "dtlz7" => Box::new(Dtlz::new(DtlzKind::Dtlz7, 3)?),
        "lennard-jones" | "lj13" => Box::new(lennard_jones::LennardJones::new(
            13,
            lennard_jones::Parameterization::Free,
        )?),
        _ => return Err(SuiteError::InvalidConfiguration),
    };
    Ok(suite)
}

pub(crate) fn validate_decision(
    decision: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Result<(), SuiteError> {
    if decision.len() != lower.len() || lower.len() != upper.len() {
        return Err(SuiteError::DimensionMismatch);
    }
    if decision.iter().any(|value| !value.is_finite()) {
        return Err(SuiteError::NonFiniteDecision);
    }
    if decision
        .iter()
        .zip(lower)
        .zip(upper)
        .any(|((&value, &lo), &hi)| value < lo || value > hi)
    {
        return Err(SuiteError::OutOfBounds);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_teaching_configurations_are_complete_and_valid() {
        let names = [
            "sphere",
            "rosenbrock",
            "rastrigin",
            "ackley",
            "griewank",
            "schwefel",
            "levy",
            "zakharov",
            "zdt1",
            "zdt2",
            "zdt3",
            "zdt4",
            "zdt6",
            "dtlz1",
            "dtlz2",
            "dtlz3",
            "dtlz4",
            "dtlz5",
            "dtlz6",
            "dtlz7",
            "lennard-jones",
        ];
        for name in names {
            let problem = by_name(name).unwrap();
            assert_eq!(problem.name(), name);
            assert_eq!(problem.bounds().0.len(), problem.dimension());
            if let Some(optimum) = problem.known_optimum() {
                assert_eq!(
                    problem.known_optimum_value(),
                    optimum.objectives.first().copied()
                );
            }
        }
        assert!(matches!(
            by_name("not-a-suite"),
            Err(SuiteError::InvalidConfiguration)
        ));
    }
}
