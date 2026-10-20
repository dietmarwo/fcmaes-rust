//! Exact latent GP with fixed signal variance and a numerical nugget.
use crate::{Error, Result, stream_seed};
use fcmaes_core::Rng;
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const NUGGET: f64 = 1e-8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kernel {
    #[default]
    Matern52,
    Rbf,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hyper {
    pub lengthscales: Vec<f64>,
}

pub fn covariance(kernel: Kernel, a: &[f64], b: &[f64], hyper: &Hyper) -> f64 {
    let r2: f64 = a
        .iter()
        .zip(b)
        .zip(&hyper.lengthscales)
        .map(|((&a, &b), &l)| ((a - b) / l).powi(2))
        .sum();
    match kernel {
        Kernel::Rbf => libm::exp(-0.5 * r2),
        Kernel::Matern52 => {
            let r = libm::sqrt(5.0 * r2);
            (1.0 + r + 5.0 * r2 / 3.0) * libm::exp(-r)
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExactGp {
    xs: Vec<Vec<f64>>,
    ys: Vec<f64>,
    lower: DMatrix<f64>,
    alpha: Vec<f64>,
    pub hyper: Hyper,
    pub kernel: Kernel,
    pub jitter: f64,
    pub negative_log_posterior: f64,
}

fn forward(lower: &DMatrix<f64>, y: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0; y.len()];
    for i in 0..y.len() {
        let mut sum = 0.0;
        for j in 0..i {
            sum += lower[(i, j)] * v[j];
        }
        v[i] = (y[i] - sum) / lower[(i, i)];
    }
    v
}
fn solve(lower: &DMatrix<f64>, y: &[f64]) -> Vec<f64> {
    let mut x = forward(lower, y);
    for i in (0..y.len()).rev() {
        let mut sum = 0.0;
        for j in i + 1..y.len() {
            sum += lower[(j, i)] * x[j];
        }
        x[i] = (x[i] - sum) / lower[(i, i)];
    }
    x
}

/// Negative log density on lengthscales, without parameter-independent terms.
/// Optimization in log coordinates intentionally does not add a Jacobian.
pub fn lengthscale_prior(hyper: &Hyper) -> f64 {
    let mu = std::f64::consts::SQRT_2 + 0.5 * libm::log(hyper.lengthscales.len() as f64);
    hyper
        .lengthscales
        .iter()
        .map(|&l| {
            let u = libm::log(l);
            u + (u - mu).powi(2) / 6.0
        })
        .sum()
}

impl ExactGp {
    pub fn with_hyper(xs: &[Vec<f64>], ys: &[f64], kernel: Kernel, hyper: Hyper) -> Result<Self> {
        let n = xs.len();
        let d = hyper.lengthscales.len();
        if n == 0
            || n > crate::MAX_OBSERVATIONS
            || d == 0
            || d > crate::MAX_DIMENSION
            || ys.len() != n
            || ys.iter().any(|v| !v.is_finite())
            || xs.iter().any(|x| {
                x.len() != d || x.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            })
            || hyper
                .lengthscales
                .iter()
                .any(|v| !v.is_finite() || *v < 0.025 || *v > 1000.0 * libm::sqrt(d as f64))
        {
            return Err(Error::Invalid(
                "GP shape, finite unit data, lengthscales or size limit".into(),
            ));
        }
        let base = DMatrix::from_fn(n, n, |i, j| covariance(kernel, &xs[i], &xs[j], &hyper));
        let mut jitter = 1e-10;
        for _ in 0..8 {
            let mut matrix = base.clone();
            for i in 0..n {
                matrix[(i, i)] += NUGGET + jitter;
            }
            if let Some(chol) = matrix.cholesky() {
                let lower = chol.l();
                let alpha = solve(&lower, ys);
                if alpha.iter().all(|v| v.is_finite()) {
                    let quadratic = ys.iter().zip(&alpha).map(|(a, b)| a * b).sum::<f64>();
                    let logdet = (0..n).map(|i| 2.0 * libm::log(lower[(i, i)])).sum::<f64>();
                    let value = 0.5
                        * (quadratic + logdet + n as f64 * libm::log(2.0 * std::f64::consts::PI))
                        + lengthscale_prior(&hyper);
                    if value.is_finite() {
                        return Ok(Self {
                            xs: xs.to_vec(),
                            ys: ys.to_vec(),
                            lower,
                            alpha,
                            hyper,
                            kernel,
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
    pub fn posterior(&self, x: &[f64]) -> Result<(f64, f64)> {
        if x.len() != self.hyper.lengthscales.len()
            || x.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(Error::Invalid(
                "GP prediction point outside unit cube".into(),
            ));
        }
        let k: Vec<f64> = self
            .xs
            .iter()
            .map(|xi| covariance(self.kernel, xi, x, &self.hyper))
            .collect();
        let v = forward(&self.lower, &k);
        let mean = k.iter().zip(&self.alpha).map(|(a, b)| a * b).sum::<f64>();
        let variance = 1.0 - v.iter().map(|v| v * v).sum::<f64>();
        if !mean.is_finite() || !variance.is_finite() || variance < -1e-7 {
            return Err(Error::Numerical(
                "nonfinite posterior or materially negative variance".into(),
            ));
        }
        Ok((mean, variance.max(0.0)))
    }
    /// Fixed-transform, fixed-hyperparameter O(n²) Cholesky append.
    pub fn fantasize(&self, x: Vec<f64>, value: f64) -> Result<Self> {
        self.posterior(&x)?;
        if !value.is_finite() || self.xs.len() >= crate::MAX_OBSERVATIONS {
            return Err(Error::Invalid("fantasy value or GP size limit".into()));
        }
        let n = self.xs.len();
        let k: Vec<f64> = self
            .xs
            .iter()
            .map(|xi| covariance(self.kernel, xi, &x, &self.hyper))
            .collect();
        let v = forward(&self.lower, &k);
        let diagonal = 1.0 + NUGGET + self.jitter - v.iter().map(|v| v * v).sum::<f64>();
        let mut xs = self.xs.clone();
        xs.push(x);
        let mut ys = self.ys.clone();
        ys.push(value);
        if diagonal <= 0.0 || !diagonal.is_finite() {
            return Self::with_hyper(&xs, &ys, self.kernel, self.hyper.clone());
        }
        let mut lower = DMatrix::zeros(n + 1, n + 1);
        for i in 0..n {
            for j in 0..=i {
                lower[(i, j)] = self.lower[(i, j)];
            }
            lower[(n, i)] = v[i];
        }
        lower[(n, n)] = libm::sqrt(diagonal);
        let alpha = solve(&lower, &ys);
        if alpha.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numerical("nonfinite fantasy solve".into()));
        }
        let quadratic = ys.iter().zip(&alpha).map(|(a, b)| a * b).sum::<f64>();
        let logdet = (0..n + 1)
            .map(|i| 2.0 * libm::log(lower[(i, i)]))
            .sum::<f64>();
        let negative_log_posterior = 0.5
            * (quadratic + logdet + (n + 1) as f64 * libm::log(2.0 * std::f64::consts::PI))
            + lengthscale_prior(&self.hyper);
        Ok(Self {
            xs,
            ys,
            lower,
            alpha,
            negative_log_posterior,
            ..self.clone()
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
    pub sweeps: usize,
}

/// Deterministic bounded coordinate-pattern MAP optimization. This is not
/// L-BFGS-B; analytic-gradient fitting is deliberately a later milestone.
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
        || input.sweeps == 0
        || input.sweeps > 24
    {
        return Err(Error::Invalid(
            "GP fit dimension/restart/sweep limit".into(),
        ));
    }
    let lower = libm::log(0.025);
    let upper = libm::log(1000.0 * libm::sqrt(d as f64));
    let mode = std::f64::consts::SQRT_2 + 0.5 * libm::log(d as f64) - 3.0;
    let results: Vec<Result<(ExactGp, usize)>> = (0..input.restarts)
        .into_par_iter()
        .map(|r| {
            if stop() {
                return Err(Error::Stopped);
            }
            let mut rng = Rng::new(stream_seed(input.seed, 31, r as u64));
            let start = if r == 0 {
                input.warm.cloned().unwrap_or(Hyper {
                    lengthscales: vec![libm::exp(mode); d],
                })
            } else {
                Hyper {
                    lengthscales: (0..d)
                        .map(|_| libm::exp((mode + 1.2 * rng.gaussian()).clamp(lower, upper)))
                        .collect(),
                }
            };
            let mut best = ExactGp::with_hyper(input.xs, input.ys, input.kernel, start)?;
            let mut fits = 1;
            let mut step = 0.8;
            for _ in 0..input.sweeps {
                let mut improved = false;
                for j in 0..d {
                    let u = libm::log(best.hyper.lengthscales[j]);
                    for direction in [-1.0, 1.0] {
                        if stop() {
                            return Err(Error::Stopped);
                        }
                        let mut hyper = best.hyper.clone();
                        hyper.lengthscales[j] =
                            libm::exp((u + direction * step).clamp(lower, upper));
                        fits += 1;
                        if let Ok(candidate) =
                            ExactGp::with_hyper(input.xs, input.ys, input.kernel, hyper)
                            && candidate.negative_log_posterior < best.negative_log_posterior
                        {
                            best = candidate;
                            improved = true;
                        }
                    }
                }
                if !improved {
                    step *= 0.5;
                }
                if step < 0.025 {
                    break;
                }
            }
            Ok((best, fits))
        })
        .collect();
    let mut best: Option<ExactGp> = None;
    let mut fits = 0;
    for result in results {
        let (candidate, count) = result?;
        fits += count;
        if best
            .as_ref()
            .is_none_or(|b| candidate.negative_log_posterior < b.negative_log_posterior)
        {
            best = Some(candidate);
        }
    }
    Ok((
        best.ok_or_else(|| Error::Numerical("no GP fit".into()))?,
        fits,
    ))
}
