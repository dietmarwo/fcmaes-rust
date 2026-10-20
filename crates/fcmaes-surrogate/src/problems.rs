//! Explicit-domain small benchmark functions, not a BBOB implementation.
pub fn branin(x: &[f64]) -> f64 {
    let a = x[1] - 5.1 * x[0] * x[0] / (4.0 * std::f64::consts::PI.powi(2))
        + 5.0 * x[0] / std::f64::consts::PI
        - 6.0;
    a * a + 10.0 * (1.0 - 1.0 / (8.0 * std::f64::consts::PI)) * libm::cos(x[0]) + 10.0
}
pub fn hartmann3(x: &[f64]) -> f64 {
    let a = [
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ];
    let p = [
        [0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0381, 0.5743, 0.8828],
    ];
    let alpha = [1.0, 1.2, 3.0, 3.2];
    -(0..4)
        .map(|i| {
            alpha[i]
                * libm::exp(
                    -(0..3)
                        .map(|j| a[i][j] * (x[j] - p[i][j]).powi(2))
                        .sum::<f64>(),
                )
        })
        .sum::<f64>()
}
