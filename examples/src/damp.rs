//! Controlled damped-spring example from `examples/damp.py`.
//!
//! The Python/C++ example numerically integrates a linear, constant-control
//! segment. Rust uses its closed-form solution, removing integrator overhead
//! without changing the mathematical objective.

pub const MAX_ALPHA: f64 = 0.1;
pub const MAX_TIME: f64 = 40.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DampResult {
    pub amplitude: f64,
    pub elapsed_time: f64,
    pub energy: f64,
    pub state: [f64; 2],
}

pub fn evaluate(x: &[f64]) -> DampResult {
    assert!(!x.is_empty() && x.len().is_multiple_of(2));
    let n = x.len() / 2;
    let duration_scale = 2.0 * MAX_TIME / n as f64;
    let mut state = [1.0_f64, 0.0_f64];
    let mut elapsed_time = 0.0;
    let mut energy = 0.0;
    for i in 0..n {
        let duration = x[i] * duration_scale;
        let alpha = x[n + i] * 2.0 * MAX_ALPHA - MAX_ALPHA;
        state = spring_segment(state, duration, alpha);
        elapsed_time += duration;
        energy += duration * alpha.abs();
    }
    DampResult {
        amplitude: state[0].abs() + state[1].abs(),
        elapsed_time,
        energy,
        state,
    }
}

pub fn fitness(x: &[f64]) -> f64 {
    evaluate(x).amplitude
}

pub fn descriptor(x: &[f64]) -> [f64; 2] {
    let result = evaluate(x);
    [result.elapsed_time, result.energy]
}

pub fn spring_segment([position, velocity]: [f64; 2], duration: f64, alpha: f64) -> [f64; 2] {
    let (sin, cos) = duration.sin_cos();
    let displaced = position - alpha;
    [
        alpha + displaced * cos + velocity * sin,
        -displaced * sin + velocity * cos,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_matches_known_oscillator() {
        let state = spring_segment([1.0, 0.0], std::f64::consts::FRAC_PI_2, 0.0);
        assert!(state[0].abs() < 1e-15);
        assert!((state[1] + 1.0).abs() < 1e-15);
    }

    #[test]
    fn objective_matches_python_reference() {
        let x = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.1, 0.9];
        let result = evaluate(&x);
        assert!((result.amplitude - 1.4984988964913732).abs() < 1e-12);
        assert!((result.elapsed_time - 40.0).abs() < 1e-12);
        assert!((result.energy - 2.72).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn odd_dimension_is_rejected() {
        evaluate(&[0.5]);
    }
}
