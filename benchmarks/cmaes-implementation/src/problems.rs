use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Problem {
    pub key: &'static str,
    pub label: &'static str,
    pub dimension: usize,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub initial_normalized: f64,
    pub optimum: f64,
    pub natural_cost_only: bool,
    pub(crate) evaluator: fn(&[f64]) -> f64,
}

impl Problem {
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        (self.evaluator)(point)
    }

    pub fn initial_mean(&self) -> Vec<f64> {
        vec![self.initial_normalized; self.dimension]
    }

    pub fn default_population(&self) -> usize {
        4 + (3.0 * (self.dimension as f64).ln()).floor() as usize
    }
}

pub fn problems() -> Vec<Problem> {
    vec![
        analytic("sphere10", "Sphere n=10", 10, -5.0, 5.0, 0.4, sphere),
        analytic("sphere100", "Sphere n=100", 100, -5.0, 5.0, 0.4, sphere),
        analytic(
            "rosenbrock10",
            "Rosenbrock n=10",
            10,
            -5.0,
            5.0,
            -0.4,
            rosenbrock,
        ),
        analytic(
            "rosenbrock40",
            "Rosenbrock n=40",
            40,
            -5.0,
            5.0,
            -0.4,
            rosenbrock,
        ),
        analytic(
            "rastrigin10",
            "Rastrigin n=10",
            10,
            -5.12,
            5.12,
            0.4,
            rastrigin,
        ),
        analytic(
            "rastrigin40",
            "Rastrigin n=40",
            40,
            -5.12,
            5.12,
            0.4,
            rastrigin,
        ),
        analytic(
            "ellipsoid100",
            "Ellipsoid 1e6 n=100",
            100,
            -5.0,
            5.0,
            0.4,
            ellipsoid,
        ),
        Problem {
            key: "cassini1",
            label: "Cassini1",
            dimension: 6,
            lower: vec![-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
            upper: vec![0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
            initial_normalized: 0.0,
            optimum: 4.9307,
            natural_cost_only: true,
            evaluator: fcmaes_gtop::cassini1,
        },
    ]
}

fn analytic(
    key: &'static str,
    label: &'static str,
    dimension: usize,
    lower: f64,
    upper: f64,
    initial_normalized: f64,
    evaluator: fn(&[f64]) -> f64,
) -> Problem {
    Problem {
        key,
        label,
        dimension,
        lower: vec![lower; dimension],
        upper: vec![upper; dimension],
        initial_normalized,
        optimum: 0.0,
        natural_cost_only: false,
        evaluator,
    }
}

fn sphere(point: &[f64]) -> f64 {
    point.iter().map(|value| value * value).sum()
}

fn rosenbrock(point: &[f64]) -> f64 {
    point
        .windows(2)
        .map(|pair| 100.0 * (pair[1] - pair[0] * pair[0]).powi(2) + (1.0 - pair[0]).powi(2))
        .sum()
}

fn rastrigin(point: &[f64]) -> f64 {
    10.0 * point.len() as f64
        + point
            .iter()
            .map(|value| value * value - 10.0 * (2.0 * PI * value).cos())
            .sum::<f64>()
}

fn ellipsoid(point: &[f64]) -> f64 {
    let denominator = (point.len() - 1).max(1) as f64;
    point
        .iter()
        .enumerate()
        .map(|(index, value)| 1.0e6_f64.powf(index as f64 / denominator) * value * value)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_optima_are_zero() {
        assert_eq!(sphere(&[0.0; 10]), 0.0);
        assert_eq!(rosenbrock(&[1.0; 10]), 0.0);
        assert_eq!(rastrigin(&[0.0; 10]), 0.0);
        assert_eq!(ellipsoid(&[0.0; 10]), 0.0);
    }

    #[test]
    fn registry_keys_are_unique() {
        let mut keys: Vec<_> = problems().into_iter().map(|problem| problem.key).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), problems().len());
    }
}
