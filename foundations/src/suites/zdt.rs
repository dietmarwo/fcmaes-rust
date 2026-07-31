//! ZDT1–4 and ZDT6 with deterministic analytic reference fronts.

use std::f64::consts::PI;

use fcmaes_core::ReferencePoint;

use super::{KnownOptimum, Suite, SuiteError, validate_decision};

/// Supported ZDT problem.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Zdt {
    /// Convex front.
    Zdt1(usize),
    /// Concave front.
    Zdt2(usize),
    /// Five-part disconnected front.
    Zdt3(usize),
    /// Multimodal decision space.
    Zdt4(usize),
    /// Nonuniform concave front.
    Zdt6(usize),
}

impl Zdt {
    /// Publication defaults: 30 variables, except ZDT4 with 10.
    pub fn publication() -> [Self; 5] {
        [
            Self::Zdt1(30),
            Self::Zdt2(30),
            Self::Zdt3(30),
            Self::Zdt4(10),
            Self::Zdt6(30),
        ]
    }

    fn relation(self, f1: f64) -> f64 {
        match self {
            Self::Zdt1(_) | Self::Zdt4(_) => 1.0 - f1.sqrt(),
            Self::Zdt2(_) | Self::Zdt6(_) => 1.0 - f1 * f1,
            Self::Zdt3(_) => 1.0 - f1.sqrt() - f1 * (10.0 * PI * f1).sin(),
        }
    }
}

impl Suite for Zdt {
    fn name(&self) -> &'static str {
        match self {
            Self::Zdt1(_) => "zdt1",
            Self::Zdt2(_) => "zdt2",
            Self::Zdt3(_) => "zdt3",
            Self::Zdt4(_) => "zdt4",
            Self::Zdt6(_) => "zdt6",
        }
    }

    fn dimension(&self) -> usize {
        match self {
            Self::Zdt1(value)
            | Self::Zdt2(value)
            | Self::Zdt3(value)
            | Self::Zdt4(value)
            | Self::Zdt6(value) => *value,
        }
    }

    fn objectives(&self) -> usize {
        2
    }

    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let mut lower = vec![0.0; self.dimension()];
        let mut upper = vec![1.0; self.dimension()];
        if matches!(self, Self::Zdt4(_)) {
            lower[1..].fill(-5.0);
            upper[1..].fill(5.0);
        }
        (lower, upper)
    }

    fn evaluate(&self, decision: &[f64]) -> Result<Vec<f64>, SuiteError> {
        if self.dimension() < 2 {
            return Err(SuiteError::InvalidConfiguration);
        }
        let (lower, upper) = self.bounds();
        validate_decision(decision, &lower, &upper)?;
        let f1 = match self {
            Self::Zdt6(_) => {
                1.0 - (-4.0 * decision[0]).exp() * (6.0 * PI * decision[0]).sin().powi(6)
            }
            _ => decision[0],
        };
        let tail = &decision[1..];
        let g = match self {
            Self::Zdt4(_) => {
                1.0 + 10.0 * tail.len() as f64
                    + tail
                        .iter()
                        .map(|value| value * value - 10.0 * (4.0 * PI * value).cos())
                        .sum::<f64>()
            }
            Self::Zdt6(_) => 1.0 + 9.0 * (tail.iter().sum::<f64>() / tail.len() as f64).powf(0.25),
            _ => 1.0 + 9.0 * tail.iter().sum::<f64>() / tail.len() as f64,
        };
        let ratio = f1 / g;
        let h = match self {
            Self::Zdt1(_) | Self::Zdt4(_) => 1.0 - ratio.sqrt(),
            Self::Zdt2(_) | Self::Zdt6(_) => 1.0 - ratio * ratio,
            Self::Zdt3(_) => 1.0 - ratio.sqrt() - ratio * (10.0 * PI * f1).sin(),
        };
        Ok(vec![f1, g * h])
    }

    fn known_optimum(&self) -> Option<KnownOptimum> {
        None
    }

    fn reference_front(&self, points: usize) -> Option<Vec<Vec<f64>>> {
        if points == 0 {
            return Some(Vec::new());
        }
        if matches!(self, Self::Zdt3(_)) {
            const INTERVALS: [(f64, f64); 5] = [
                (0.0, 0.083_001_534_9),
                (0.182_228_728_0, 0.257_762_363_4),
                (0.409_313_674_8, 0.453_882_104_1),
                (0.618_396_794_4, 0.652_511_703_8),
                (0.823_331_798_3, 0.851_832_865_4),
            ];
            let lengths: Vec<f64> = INTERVALS.iter().map(|(lo, hi)| hi - lo).collect();
            let total = lengths.iter().sum::<f64>();
            let mut front = Vec::with_capacity(points);
            for index in 0..points {
                let target = (index as f64 + 0.5) * total / points as f64;
                let mut cumulative = 0.0;
                for ((lo, hi), length) in INTERVALS.iter().zip(&lengths) {
                    if target <= cumulative + length {
                        let f1 = lo + (target - cumulative).clamp(0.0, *length);
                        front.push(vec![f1, self.relation(f1)]);
                        break;
                    }
                    cumulative += hi - lo;
                }
            }
            return Some(front);
        }
        let minimum = if matches!(self, Self::Zdt6(_)) {
            0.280_775_319_1
        } else {
            0.0
        };
        Some(
            (0..points)
                .map(|index| {
                    let fraction = if points == 1 {
                        0.5
                    } else {
                        index as f64 / (points - 1) as f64
                    };
                    let f1 = minimum + (1.0 - minimum) * fraction;
                    vec![f1, self.relation(f1)]
                })
                .collect(),
        )
    }

    fn reference_point(&self) -> Option<ReferencePoint> {
        ReferencePoint::new(vec![1.1, 1.1]).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcmaes_core::pareto_indices;

    #[test]
    fn analytic_fronts_satisfy_relations_and_are_nondominated() {
        for problem in Zdt::publication() {
            let front = problem.reference_front(501).unwrap();
            for point in &front {
                assert!((point[1] - problem.relation(point[0])).abs() < 1.0e-12);
            }
            assert_eq!(
                pareto_indices(&front, 2).unwrap().len(),
                front.len(),
                "{}",
                problem.name()
            );
            let mut with_perturbation = front.clone();
            let mut dominated = front[front.len() / 2].clone();
            dominated.iter_mut().for_each(|value| *value += 0.01);
            with_perturbation.push(dominated);
            assert_eq!(
                pareto_indices(&with_perturbation, 2).unwrap().len(),
                front.len(),
                "{} perturbed point was not dominated",
                problem.name()
            );
            assert_eq!(front, problem.reference_front(501).unwrap());
        }
    }

    #[test]
    fn pareto_decisions_evaluate_to_the_analytic_relation() {
        for problem in Zdt::publication() {
            let mut decision = vec![0.0; problem.dimension()];
            decision[0] = if matches!(problem, Zdt::Zdt6(_)) {
                0.0
            } else {
                0.3
            };
            let value = problem.evaluate(&decision).unwrap();
            assert!((value[1] - problem.relation(value[0])).abs() < 1.0e-12);
        }
    }
}
