//! Weighted spherical t-design objective from `examples/tdesign.py`.

use std::f64::consts::PI;

pub fn normalize_weights(weights: &[f64]) -> Vec<f64> {
    let sum: f64 = weights.iter().sum();
    if sum == 0.0 {
        vec![1.0; weights.len()]
    } else {
        let scale = weights.len() as f64 / sum;
        weights.iter().map(|weight| scale * weight).collect()
    }
}

/// Golden-angle seed in `(theta, phi)` coordinates.
pub fn fibonacci_sphere(n: usize) -> Vec<[f64; 2]> {
    assert!(n >= 2);
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - i as f64 / (n - 1) as f64 * 2.0;
            let radius = (1.0 - y * y).sqrt();
            let angle = golden_angle * i as f64;
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;
            let theta = z.acos();
            let mut phi = y.atan2(x);
            if phi < 0.0 {
                phi += 2.0 * PI;
            }
            [theta, phi]
        })
        .collect()
}

/// Returns the per-degree symmetry measures, including the degree-zero term.
pub fn symmetry(points: &[[f64; 2]], l_max: usize, weights: Option<&[f64]>) -> Vec<f64> {
    assert!(!points.is_empty());
    if let Some(weights) = weights {
        assert_eq!(weights.len(), points.len());
    }
    let normalized;
    let weights = if let Some(weights) = weights {
        normalized = normalize_weights(weights);
        normalized.as_slice()
    } else {
        &[]
    };
    let n = points.len() as f64;
    let mut result = vec![0.0; l_max + 1];
    for (degree, degree_result) in result.iter_mut().enumerate() {
        let mut degree_sum = 0.0;
        for order in -(degree as isize)..=degree as isize {
            let harmonic_sum: f64 = points
                .iter()
                .enumerate()
                .map(|(i, point)| {
                    let weight = if weights.is_empty() { 1.0 } else { weights[i] };
                    weight * real_spherical_harmonic(degree, order, point[0], point[1])
                })
                .sum();
            degree_sum += harmonic_sum * harmonic_sum;
        }
        if degree_sum.abs() < 1e-20 {
            degree_sum = 0.0;
        }
        *degree_result = degree_sum * 4.0 * PI / (n * n) / (2 * degree + 1) as f64;
    }
    result
}

pub fn t_design_error(points: &[[f64; 2]], l_max: usize, weights: Option<&[f64]>) -> f64 {
    symmetry(points, l_max, weights)[1..].iter().sum()
}

/// Decode `[theta..., phi..., weight...]` like `optimize_weighted`.
pub fn weighted_objective(x: &[f64], l_max: usize) -> f64 {
    assert_eq!(x.len() % 3, 0);
    let n = x.len() / 3;
    let points: Vec<[f64; 2]> = (0..n).map(|i| [x[i], x[n + i]]).collect();
    t_design_error(&points, l_max, Some(&x[2 * n..]))
}

fn real_spherical_harmonic(degree: usize, order: isize, theta: f64, phi: f64) -> f64 {
    let m = order.unsigned_abs();
    let p = associated_legendre(degree, m, theta.cos());
    let ratio = if m == 0 {
        1.0
    } else {
        ((degree - m + 1)..=(degree + m)).fold(1.0, |product, value| product / value as f64)
    };
    let norm = (((2 * degree + 1) as f64 / (4.0 * PI)) * ratio).sqrt();
    if order == 0 {
        norm * p
    } else if order > 0 {
        2.0_f64.sqrt() * norm * p * (m as f64 * phi).cos()
    } else {
        2.0_f64.sqrt() * norm * p * (m as f64 * phi).sin()
    }
}

/// Associated Legendre P_l^m, including SciPy's Condon-Shortley phase.
fn associated_legendre(degree: usize, order: usize, x: f64) -> f64 {
    debug_assert!(order <= degree);
    let mut p_mm = 1.0;
    if order > 0 {
        let root = (1.0 - x * x).max(0.0).sqrt();
        let mut factor = 1.0;
        for _ in 1..=order {
            p_mm *= -factor * root;
            factor += 2.0;
        }
    }
    if degree == order {
        return p_mm;
    }
    let p_m1m = x * (2 * order + 1) as f64 * p_mm;
    if degree == order + 1 {
        return p_m1m;
    }
    let (mut previous, mut current) = (p_mm, p_m1m);
    for l in (order + 2)..=degree {
        let next = ((2 * l - 1) as f64 * x * current - (l + order - 1) as f64 * previous)
            / (l - order) as f64;
        previous = current;
        current = next;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_and_known_harmonics() {
        assert_eq!(normalize_weights(&[0.0, 0.0]), [1.0, 1.0]);
        assert_eq!(normalize_weights(&[1.0, 3.0]), [0.5, 1.5]);
        let y00 = real_spherical_harmonic(0, 0, 0.4, 0.7);
        assert!((y00 - 0.28209479177387814).abs() < 1e-15);
        let y10 = real_spherical_harmonic(1, 0, 0.4, 0.7);
        assert!((y10 - 0.4500327152856099).abs() < 1e-15);
    }

    #[test]
    fn fibonacci_reference_matches_python() {
        let points = fibonacci_sphere(10);
        let error = t_design_error(&points, 4, None);
        assert!((error - 0.11708262542419427).abs() < 1e-13, "{error}");
        let weights: Vec<f64> = (0..10).map(|i| 0.5 + i as f64 / 10.0).collect();
        let weighted = t_design_error(&points, 4, Some(&weights));
        assert!((weighted - 0.16962586391160267).abs() < 1e-13, "{weighted}");
    }
}
