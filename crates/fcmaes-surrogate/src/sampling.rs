//! Correlated finite-candidate oracle and approximate decoupled posterior paths.
use crate::{
    Error, Result,
    gp::{ExactGp, Kernel, covariance},
    linalg::{Cholesky, row},
};
use fcmaes_core::Rng;

/// Exact finite-set posterior draw, with <=1e-10 numerical diagonal shift.
pub fn exact_sample(gp: &ExactGp, points: &[Vec<f64>], seed: u64) -> Result<Vec<f64>> {
    if points.is_empty() || points.len() > 512 {
        return Err(Error::Invalid("sample count 1..512".into()));
    }
    let n = points.len();
    let means: Vec<_> = points
        .iter()
        .map(|x| gp.posterior(x).map(|v| v.0))
        .collect::<Result<_>>()?;
    let mut cov = vec![0.0; row(n)];
    for i in 0..n {
        for j in 0..=i {
            cov[row(i) + j] = gp.posterior_covariance(&points[i], &points[j])?;
        }
    }
    let factor = [0.0, 1e-14, 1e-12, 1e-10]
        .into_iter()
        .find_map(|s| Cholesky::factor(&cov, n, s))
        .ok_or_else(|| Error::Numerical("posterior sample covariance".into()))?;
    let mut rng = Rng::new(seed);
    let normal: Vec<_> = (0..n).map(|_| rng.gaussian()).collect();
    Ok((0..n)
        .map(|i| {
            means[i]
                + factor
                    .row(i)
                    .iter()
                    .zip(&normal)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
        })
        .collect())
}

/// A fixed approximate prior path plus exact-kernel conditioning correction.
/// Approximation quality depends on feature count; this is not an exact draw.
pub struct Path {
    frequencies: Vec<Vec<f64>>,
    phases: Vec<f64>,
    weights: Vec<f64>,
    points: Vec<Vec<f64>>,
    correction: Vec<f64>,
    gp: ExactGp,
}
impl Path {
    pub fn new(gp: &ExactGp, features: usize, seed: u64) -> Result<Self> {
        if !(32..=4096).contains(&features) {
            return Err(Error::Invalid("features 32..4096".into()));
        }
        let d = gp.hyper.lengthscales.len();
        let mut rng = Rng::new(seed);
        let mut frequencies = Vec::with_capacity(features);
        let mut phases = Vec::with_capacity(features);
        let mut weights = Vec::with_capacity(features);
        for _ in 0..features {
            let scale = match gp.kernel {
                Kernel::Rbf => 1.0,
                Kernel::Matern52 => {
                    let chi = (0..5).map(|_| rng.gaussian().powi(2)).sum::<f64>();
                    (5.0 / chi.max(1e-300)).sqrt()
                }
            };
            frequencies.push(
                (0..d)
                    .map(|j| rng.gaussian() * scale / gp.hyper.lengthscales[j])
                    .collect(),
            );
            phases.push(rng.uniform01() * std::f64::consts::TAU);
            weights.push(rng.gaussian() * (2.0 / features as f64).sqrt());
        }
        let mut path = Self {
            frequencies,
            phases,
            weights,
            points: gp.training_points().to_vec(),
            correction: Vec::new(),
            gp: gp.clone(),
        };
        // Include the regularized model's observation noise in the prior draw.
        let prior: Vec<_> = path
            .points
            .iter()
            .map(|x| path.prior(x) + rng.gaussian() * (gp.nugget + gp.jitter).sqrt())
            .collect();
        path.correction = gp.condition_path(&prior)?;
        Ok(path)
    }
    fn prior(&self, x: &[f64]) -> f64 {
        self.frequencies
            .iter()
            .zip(&self.phases)
            .zip(&self.weights)
            .map(|((w, p), a)| a * (w.iter().zip(x).map(|(w, x)| w * x).sum::<f64>() + p).cos())
            .sum()
    }
    pub fn value(&self, x: &[f64]) -> Result<f64> {
        self.gp.posterior(x)?; // validate the public input
        let value = self.prior(x)
            + self
                .points
                .iter()
                .zip(&self.correction)
                .map(|(p, a)| a * covariance(self.gp.kernel, p, x, &self.gp.hyper))
                .sum::<f64>();
        if !value.is_finite() {
            return Err(Error::Numerical("path evaluation".into()));
        }
        Ok(value)
    }
}
