//! Analytic LogEI in the minimization convention. No probability clipping.
const LOG_SQRT_2_PI: f64 = 0.9189385332046727;

pub fn normal_pdf(z: f64) -> f64 {
    libm::exp(-0.5 * z * z - LOG_SQRT_2_PI)
}
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// `log(phi(z) + z Phi(z))`. The negative tail uses the optimally truncated
/// Mills expansion, rather than subtraction of two underflowed numbers.
pub fn log_h(z: f64) -> f64 {
    if z > 8.0 {
        return libm::log(z);
    }
    if z >= -10.0 {
        return libm::log(normal_pdf(z) + z * normal_cdf(z));
    }
    let inv = 1.0 / (z * z);
    let mut term: f64 = 1.0;
    let mut sum: f64 = 1.0;
    for k in 1..200 {
        let next = -term * (2 * k + 1) as f64 * inv;
        if next.abs() >= term.abs() {
            break;
        }
        sum += next;
        term = next;
        if next.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    -0.5 * z * z - LOG_SQRT_2_PI - 2.0 * libm::log(-z) + libm::log(sum)
}

/// Log expected improvement for a Gaussian latent function value.
pub fn log_ei(best: f64, mean: f64, variance: f64) -> f64 {
    if variance <= 0.0 {
        return if best > mean {
            libm::log(best - mean)
        } else {
            f64::NEG_INFINITY
        };
    }
    let sigma = libm::sqrt(variance);
    log_h((best - mean) / sigma) + libm::log(sigma)
}
