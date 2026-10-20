//! Exact latent GP with fixed signal variance and a numerical nugget.
//!
//! Inputs are scaled once per lengthscale vector (`z = x / l`), the kernel
//! matrix is filled symmetrically in packed storage, and the MAP fit uses the
//! analytic gradient of the negative log posterior with respect to the log
//! lengthscales in a bounded L-BFGS search.
use crate::{
    Error, Result,
    linalg::{self, Cholesky, dot},
    optim::{self, Objective},
    stream_seed,
};
use fcmaes_core::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const NUGGET: f64 = 1e-8;
const LOG_2PI: f64 = 1.8378770664093453;
/// Below this size the gradient rows are processed serially.
const PARALLEL_ROWS: usize = 96;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kernel {
    #[default]
    Matern52,
    Rbf,
}

impl Kernel {
    /// Kernel value and `g = -2 dk/dr2` for a squared scaled distance `r2`,
    /// so that `dk/du_j = g dz_j^2` for `u = log l`.
    #[inline]
    fn value_and_slope(self, r2: f64) -> (f64, f64) {
        match self {
            Kernel::Rbf => {
                let k = libm::exp(-0.5 * r2);
                (k, k)
            }
            Kernel::Matern52 => {
                let s = libm::sqrt(5.0 * r2);
                let e = libm::exp(-s);
                ((1.0 + s + 5.0 * r2 / 3.0) * e, 5.0 / 3.0 * (1.0 + s) * e)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hyper {
    pub lengthscales: Vec<f64>,
}

fn scale(x: &[f64], inverse: &[f64], out: &mut [f64]) {
    for ((o, x), i) in out.iter_mut().zip(x).zip(inverse) {
        *o = x * i;
    }
}
#[inline]
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (a, b) in a.iter().zip(b) {
        let t = a - b;
        s += t * t;
    }
    s
}

pub fn covariance(kernel: Kernel, a: &[f64], b: &[f64], hyper: &Hyper) -> f64 {
    let inverse: Vec<f64> = hyper.lengthscales.iter().map(|l| 1.0 / l).collect();
    let mut za = vec![0.0; a.len()];
    let mut zb = vec![0.0; b.len()];
    scale(a, &inverse, &mut za);
    scale(b, &inverse, &mut zb);
    kernel.value_and_slope(squared_distance(&za, &zb)).0
}

fn prior_centre(d: usize) -> f64 {
    std::f64::consts::SQRT_2 + 0.5 * libm::log(d as f64)
}

/// Negative log density on lengthscales, without parameter-independent terms.
/// Optimization in log coordinates intentionally does not add a Jacobian.
pub fn lengthscale_prior(hyper: &Hyper) -> f64 {
    let mu = prior_centre(hyper.lengthscales.len());
    hyper
        .lengthscales
        .iter()
        .map(|&l| {
            let u = libm::log(l);
            u + (u - mu).powi(2) / 6.0
        })
        .sum()
}

fn lengthscale_bounds(d: usize) -> (f64, f64) {
    (0.025, 1000.0 * libm::sqrt(d as f64))
}

fn check_inputs(xs: &[Vec<f64>], ys: &[f64], d: usize) -> Result<()> {
    let n = xs.len();
    if n == 0
        || n > crate::MAX_OBSERVATIONS
        || d == 0
        || d > crate::MAX_DIMENSION
        || ys.len() != n
        || ys.iter().any(|v| !v.is_finite())
        || xs
            .iter()
            .any(|x| x.len() != d || x.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)))
    {
        return Err(Error::Invalid(
            "GP shape, finite unit data, lengthscales or size limit".into(),
        ));
    }
    Ok(())
}

/// A factored kernel matrix for one lengthscale vector.
struct Factor {
    lengthscales: Vec<f64>,
    z: Vec<f64>,
    /// Packed `g = -2 dk/dr2` for every pair, used by the gradient.
    slope: Vec<f64>,
    cholesky: Cholesky,
    alpha: Vec<f64>,
    jitter: f64,
    negative_log_posterior: f64,
}

impl Factor {
    fn new(xs: &[Vec<f64>], ys: &[f64], kernel: Kernel, lengthscales: Vec<f64>) -> Result<Self> {
        let n = xs.len();
        let d = lengthscales.len();
        let inverse: Vec<f64> = lengthscales.iter().map(|l| 1.0 / l).collect();
        let mut z = vec![0.0; n * d];
        for (x, z) in xs.iter().zip(z.chunks_exact_mut(d)) {
            scale(x, &inverse, z);
        }
        let mut matrix = vec![0.0; linalg::row(n)];
        let mut slope = vec![0.0; linalg::row(n)];
        for a in 0..n {
            let za = &z[a * d..(a + 1) * d];
            let offset = linalg::row(a);
            for b in 0..a {
                let (k, g) = kernel.value_and_slope(squared_distance(za, &z[b * d..(b + 1) * d]));
                matrix[offset + b] = k;
                slope[offset + b] = g;
            }
            matrix[offset + a] = 1.0;
        }
        let prior = lengthscale_prior(&Hyper {
            lengthscales: lengthscales.clone(),
        });
        let mut jitter = 1e-10;
        for _ in 0..8 {
            if let Some(cholesky) = Cholesky::factor(&matrix, n, NUGGET + jitter) {
                let alpha = cholesky.solve(ys);
                if alpha.iter().all(|v| v.is_finite()) {
                    let value = 0.5
                        * (dot(ys, &alpha) + cholesky.log_determinant() + n as f64 * LOG_2PI)
                        + prior;
                    if value.is_finite() {
                        return Ok(Self {
                            lengthscales,
                            z,
                            slope,
                            cholesky,
                            alpha,
                            jitter,
                            negative_log_posterior: value,
                        });
                    }
                }
            }
            jitter *= 10.0;
        }
        Err(Error::Numerical(
            "GP Cholesky/solve failed after bounded jitter escalation".into(),
        ))
    }

    /// Gradient of the negative log posterior with respect to `log l`:
    /// `sum_{a>b} W_ab g_ab dz_ab^2 + 1 + (u - mu) / 3`, `W = K^-1 - alpha alpha^T`.
    fn gradient(&self) -> Vec<f64> {
        let n = self.cholesky.dimension();
        let d = self.lengthscales.len();
        let inverse = self.cholesky.inverse_lower();
        let row = |a: usize| -> Vec<f64> {
            let mut kinv = vec![0.0; a + 1];
            linalg::inverse_row(&inverse, n, a, &mut kinv);
            let mut partial = vec![0.0; d];
            let za = &self.z[a * d..(a + 1) * d];
            let slope = &self.slope[linalg::row(a)..linalg::row(a) + a];
            for b in 0..a {
                let c = (kinv[b] - self.alpha[a] * self.alpha[b]) * slope[b];
                for ((p, za), zb) in partial.iter_mut().zip(za).zip(&self.z[b * d..(b + 1) * d]) {
                    let t = za - zb;
                    *p += c * t * t;
                }
            }
            partial
        };
        let rows: Vec<Vec<f64>> = if n >= PARALLEL_ROWS {
            (1..n).into_par_iter().map(row).collect()
        } else {
            (1..n).map(row).collect()
        };
        let mu = prior_centre(d);
        let mut gradient: Vec<f64> = self
            .lengthscales
            .iter()
            .map(|&l| 1.0 + (libm::log(l) - mu) / 3.0)
            .collect();
        for partial in rows {
            for (g, p) in gradient.iter_mut().zip(partial) {
                *g += p;
            }
        }
        gradient
    }
}

#[derive(Clone, Debug)]
pub struct ExactGp {
    xs: Vec<Vec<f64>>,
    z: Vec<f64>,
    inverse_lengthscales: Vec<f64>,
    ys: Vec<f64>,
    cholesky: Cholesky,
    alpha: Vec<f64>,
    pub hyper: Hyper,
    pub kernel: Kernel,
    pub jitter: f64,
    pub negative_log_posterior: f64,
}

impl ExactGp {
    pub fn with_hyper(xs: &[Vec<f64>], ys: &[f64], kernel: Kernel, hyper: Hyper) -> Result<Self> {
        let d = hyper.lengthscales.len();
        check_inputs(xs, ys, d)?;
        let (low, high) = lengthscale_bounds(d);
        if hyper
            .lengthscales
            .iter()
            .any(|v| !v.is_finite() || *v < low || *v > high)
        {
            return Err(Error::Invalid(
                "GP shape, finite unit data, lengthscales or size limit".into(),
            ));
        }
        Ok(Self::from_factor(
            Factor::new(xs, ys, kernel, hyper.lengthscales)?,
            xs,
            ys,
            kernel,
        ))
    }

    fn from_factor(f: Factor, xs: &[Vec<f64>], ys: &[f64], kernel: Kernel) -> Self {
        Self {
            xs: xs.to_vec(),
            inverse_lengthscales: f.lengthscales.iter().map(|l| 1.0 / l).collect(),
            z: f.z,
            ys: ys.to_vec(),
            cholesky: f.cholesky,
            alpha: f.alpha,
            hyper: Hyper {
                lengthscales: f.lengthscales,
            },
            kernel,
            jitter: f.jitter,
            negative_log_posterior: f.negative_log_posterior,
        }
    }

    /// Scaled point `x / l` and cross-covariances `k(x, x_a)`.
    fn cross(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let d = self.inverse_lengthscales.len();
        if x.len() != d || x.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
            return Err(Error::Invalid(
                "GP prediction point outside unit cube".into(),
            ));
        }
        let mut z = vec![0.0; d];
        scale(x, &self.inverse_lengthscales, &mut z);
        let k = self
            .z
            .chunks_exact(d)
            .map(|za| self.kernel.value_and_slope(squared_distance(&z, za)).0)
            .collect();
        Ok((z, k))
    }

    /// Posterior moments and `L^-1 k` (reused by fantasies).
    fn moments(&self, k: &[f64]) -> Result<(f64, f64, Vec<f64>)> {
        let v = self.cholesky.forward(k);
        let mean = dot(k, &self.alpha);
        let variance = 1.0 - dot(&v, &v);
        if !mean.is_finite() || !variance.is_finite() || variance < -1e-7 {
            return Err(Error::Numerical(
                "nonfinite posterior or materially negative variance".into(),
            ));
        }
        Ok((mean, variance.max(0.0), v))
    }

    pub fn posterior(&self, x: &[f64]) -> Result<(f64, f64)> {
        let (_, k) = self.cross(x)?;
        let (mean, variance, _) = self.moments(&k)?;
        Ok((mean, variance))
    }

    /// Fixed-transform, fixed-hyperparameter O(n²) Cholesky append.
    pub fn fantasize(&self, x: Vec<f64>, value: f64) -> Result<Self> {
        let (z, k) = self.cross(&x)?;
        let (_, _, v) = self.moments(&k)?;
        let n = self.ys.len();
        if !value.is_finite() || n >= crate::MAX_OBSERVATIONS {
            return Err(Error::Invalid("fantasy value or GP size limit".into()));
        }
        let mut ys = self.ys.clone();
        ys.push(value);
        let diagonal = 1.0 + NUGGET + self.jitter - dot(&v, &v);
        let mut xs = self.xs.clone();
        xs.push(x);
        if diagonal <= 0.0 || !diagonal.is_finite() {
            return Self::with_hyper(&xs, &ys, self.kernel, self.hyper.clone());
        }
        let mut cholesky = self.cholesky.clone();
        cholesky.append(&v, libm::sqrt(diagonal));
        let alpha = cholesky.solve(&ys);
        if alpha.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numerical("nonfinite fantasy solve".into()));
        }
        let negative_log_posterior = 0.5
            * (dot(&ys, &alpha) + cholesky.log_determinant() + (n + 1) as f64 * LOG_2PI)
            + lengthscale_prior(&self.hyper);
        let mut scaled = self.z.clone();
        scaled.extend_from_slice(&z);
        Ok(Self {
            xs,
            z: scaled,
            inverse_lengthscales: self.inverse_lengthscales.clone(),
            ys,
            cholesky,
            alpha,
            hyper: self.hyper.clone(),
            kernel: self.kernel,
            jitter: self.jitter,
            negative_log_posterior,
        })
    }
}

pub struct Fit<'a> {
    pub xs: &'a [Vec<f64>],
    pub ys: &'a [f64],
    pub kernel: Kernel,
    pub warm: Option<&'a Hyper>,
    pub seed: u64,
    pub restarts: usize,
    /// L-BFGS iterations per restart.
    pub iterations: usize,
}

pub const MAX_FIT_ITERATIONS: usize = 200;

struct Posterior<'a> {
    xs: &'a [Vec<f64>],
    ys: &'a [f64],
    kernel: Kernel,
    bounds: (f64, f64),
}

impl Objective for Posterior<'_> {
    type State = Factor;
    fn value(&self, u: &[f64]) -> Option<(f64, Factor)> {
        let lengthscales = u
            .iter()
            .map(|&u| libm::exp(u).clamp(self.bounds.0, self.bounds.1))
            .collect();
        let factor = Factor::new(self.xs, self.ys, self.kernel, lengthscales).ok()?;
        Some((factor.negative_log_posterior, factor))
    }
    fn gradient(&self, _: &[f64], factor: &Factor) -> Vec<f64> {
        factor.gradient()
    }
}

/// Deterministic bounded MAP fit of the log lengthscales: L-BFGS with the
/// analytic gradient from the warm (or prior-mode) start and seeded restarts.
/// Restarts run in parallel and are reduced in index order.
pub fn fit(input: Fit<'_>, stop: &(dyn Fn() -> bool + Sync)) -> Result<(ExactGp, usize)> {
    let d = input
        .xs
        .first()
        .ok_or_else(|| Error::Invalid("empty GP data".into()))?
        .len();
    if d == 0
        || d > crate::MAX_DIMENSION
        || input.restarts == 0
        || input.restarts > 4
        || input.iterations == 0
        || input.iterations > MAX_FIT_ITERATIONS
    {
        return Err(Error::Invalid(
            "GP fit dimension/restart/iteration limit".into(),
        ));
    }
    check_inputs(input.xs, input.ys, d)?;
    let bounds = lengthscale_bounds(d);
    let lower = vec![libm::log(bounds.0); d];
    let upper = vec![libm::log(bounds.1); d];
    let mode = prior_centre(d) - 3.0;
    let posterior = Posterior {
        xs: input.xs,
        ys: input.ys,
        kernel: input.kernel,
        bounds,
    };
    let settings = optim::Settings {
        lower: &lower,
        upper: &upper,
        iterations: input.iterations,
        first_step: 0.8,
        tolerance: 1e-9,
    };
    let results: Vec<Result<(Factor, usize)>> = (0..input.restarts)
        .into_par_iter()
        .map(|r| {
            if stop() {
                return Err(Error::Stopped);
            }
            let start: Vec<f64> = if r == 0 {
                match input.warm {
                    Some(h) if h.lengthscales.len() == d => {
                        h.lengthscales.iter().map(|&l| libm::log(l)).collect()
                    }
                    _ => vec![mode; d],
                }
            } else {
                let mut rng = Rng::new(stream_seed(input.seed, 31, r as u64));
                (0..d)
                    .map(|_| (mode + 1.2 * rng.gaussian()).clamp(lower[0], upper[0]))
                    .collect()
            };
            match optim::minimize(&posterior, &start, &settings, stop) {
                Ok(m) => Ok((m.state, m.evaluations)),
                Err(Error::Numerical(_)) => Err(Error::Numerical(
                    "GP Cholesky/solve failed after bounded jitter escalation".into(),
                )),
                Err(e) => Err(e),
            }
        })
        .collect();
    let mut best: Option<Factor> = None;
    let mut evaluations = 0;
    let mut failure = None;
    for result in results {
        match result {
            Ok((factor, count)) => {
                evaluations += count;
                if best
                    .as_ref()
                    .is_none_or(|b| factor.negative_log_posterior < b.negative_log_posterior)
                {
                    best = Some(factor);
                }
            }
            Err(Error::Numerical(s)) => failure = Some(Error::Numerical(s)),
            Err(e) => return Err(e),
        }
    }
    let best = best.ok_or_else(|| failure.unwrap_or(Error::Numerical("no GP fit".into())))?;
    Ok((
        ExactGp::from_factor(best, input.xs, input.ys, input.kernel),
        evaluations,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, d: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let xs = crate::design::latin_hypercube(n, d, 3);
        let ys = xs
            .iter()
            .map(|x| {
                x.iter()
                    .enumerate()
                    .map(|(j, v)| (3.0 * v + j as f64).sin())
                    .sum()
            })
            .collect();
        (xs, ys)
    }

    #[test]
    fn hyperparameter_gradient_matches_central_differences() {
        for kernel in [Kernel::Matern52, Kernel::Rbf] {
            for (n, d) in [(12, 1), (30, 3), (120, 5)] {
                let (xs, ys) = data(n, d);
                let u: Vec<f64> = (0..d).map(|j| -1.0 + 0.3 * j as f64).collect();
                let at = |u: &[f64]| {
                    Factor::new(&xs, &ys, kernel, u.iter().map(|u| libm::exp(*u)).collect())
                        .unwrap()
                };
                let factor = at(&u);
                assert_eq!(factor.jitter, 1e-10);
                let gradient = factor.gradient();
                for j in 0..d {
                    let h = 1e-5;
                    let mut plus = u.clone();
                    plus[j] += h;
                    let mut minus = u.clone();
                    minus[j] -= h;
                    let numeric = (at(&plus).negative_log_posterior
                        - at(&minus).negative_log_posterior)
                        / (2.0 * h);
                    assert!(
                        (numeric - gradient[j]).abs() <= 1e-5 * numeric.abs().max(1.0),
                        "{kernel:?} n={n} j={j}: {numeric} vs {}",
                        gradient[j]
                    );
                }
            }
        }
    }
}
