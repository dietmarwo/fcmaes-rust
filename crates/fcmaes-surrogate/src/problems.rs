//! Explicit-domain small benchmark functions, not a BBOB implementation.
/// Branin on `[-5, 10] x [0, 15]`; minimum 0.39788735772973816.
pub fn branin(x: &[f64]) -> f64 {
    let a = x[1] - 5.1 * x[0] * x[0] / (4.0 * std::f64::consts::PI.powi(2))
        + 5.0 * x[0] / std::f64::consts::PI
        - 6.0;
    a * a + 10.0 * (1.0 - 1.0 / (8.0 * std::f64::consts::PI)) * libm::cos(x[0]) + 10.0
}
/// Hartmann-3 on the unit cube; minimum -3.86278214782076.
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
/// Hartmann-6 on the unit cube; minimum -3.322368011415515.
pub fn hartmann6(x: &[f64]) -> f64 {
    let a = [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ];
    let p = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ];
    let alpha = [1.0, 1.2, 3.0, 3.2];
    -(0..4)
        .map(|i| {
            alpha[i]
                * libm::exp(
                    -(0..6)
                        .map(|j| a[i][j] * (x[j] - p[i][j]).powi(2))
                        .sum::<f64>(),
                )
        })
        .sum::<f64>()
}
/// Styblinski-Tang in any dimension on `[-5, 5]^d`; minimum
/// `-39.16616570377142 d` at `x_i = -2.903534...`, away from the centre.
pub fn styblinski_tang(x: &[f64]) -> f64 {
    0.5 * x
        .iter()
        .map(|v| v.powi(4) - 16.0 * v * v + 5.0 * v)
        .sum::<f64>()
}
