//! Deterministic, seed-free objective-space reference-set helpers.

/// Radical-inverse Halton coordinate for one-based `index`.
pub fn halton(mut index: usize, base: usize) -> f64 {
    let mut factor = 1.0;
    let mut value = 0.0;
    while index > 0 {
        factor /= base as f64;
        value += factor * (index % base) as f64;
        index /= base;
    }
    value
}

/// Deterministic points on the positive unit simplex.
pub fn simplex(points: usize, dimension: usize, total: f64) -> Vec<Vec<f64>> {
    const PRIMES: [usize; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
    (1..=points)
        .map(|index| {
            let mut weights: Vec<f64> = (0..dimension)
                .map(|axis| -halton(index, PRIMES[axis]).max(f64::MIN_POSITIVE).ln())
                .collect();
            let sum = weights.iter().sum::<f64>();
            for value in &mut weights {
                *value *= total / sum;
            }
            weights
        })
        .collect()
}

/// Deterministic points on the positive unit hypersphere.
pub fn positive_sphere(points: usize, dimension: usize) -> Vec<Vec<f64>> {
    simplex(points, dimension, 1.0)
        .into_iter()
        .map(|mut point| {
            let norm = point.iter().map(|value| value * value).sum::<f64>().sqrt();
            for value in &mut point {
                *value /= norm;
            }
            point
        })
        .collect()
}

fn dominates(left: &[f64], right: &[f64]) -> bool {
    left.iter().zip(right).all(|(a, b)| a <= b) && left.iter().zip(right).any(|(a, b)| a < b)
}

/// Stable nondominated filter for minimized objective vectors.
pub fn nondominated(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    points
        .iter()
        .enumerate()
        .filter(|(index, point)| {
            !points
                .iter()
                .enumerate()
                .any(|(other, candidate)| other != *index && dominates(candidate, point))
        })
        .map(|(_, point)| point.clone())
        .collect()
}
