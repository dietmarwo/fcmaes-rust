use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputTransform {
    #[default]
    Standardize,
    Bilog,
    Auto,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub bilog: bool,
    normalizer: f64,
    mean: f64,
    deviation: f64,
}
fn bilog(y: f64) -> f64 {
    y.signum() * libm::log1p(y.abs())
}

impl Transform {
    pub fn fit(ys: &[f64], kind: OutputTransform) -> Result<Self> {
        if ys.is_empty() || ys.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid(
                "transform needs finite nonempty data".into(),
            ));
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for y in ys.iter().filter(|y| **y != 0.0) {
            let l = libm::log(y.abs());
            lo = lo.min(l);
            hi = hi.max(l);
        }
        let use_bilog = kind == OutputTransform::Bilog
            || (kind == OutputTransform::Auto && hi - lo > libm::log(1e6));
        let values: Vec<f64> = ys
            .iter()
            .map(|&y| if use_bilog { bilog(y) } else { y })
            .collect();
        let normalizer = values
            .iter()
            .fold(0.0_f64, |a, b| a.max(b.abs()))
            .max(1e-300);
        let mean = values.iter().map(|v| v / normalizer).sum::<f64>() / values.len() as f64;
        let mut deviation = libm::sqrt(
            values
                .iter()
                .map(|v| (v / normalizer - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64,
        );
        if deviation < 1e-14 {
            deviation = 1.0;
        }
        Ok(Self {
            bilog: use_bilog,
            normalizer,
            mean,
            deviation,
        })
    }
    pub fn apply(&self, y: f64) -> f64 {
        let value = if self.bilog { bilog(y) } else { y };
        (value / self.normalizer - self.mean) / self.deviation
    }
    pub fn inverse(&self, z: f64) -> Result<f64> {
        let t = (z * self.deviation + self.mean) * self.normalizer;
        let value = if self.bilog {
            t.signum() * libm::expm1(t.abs())
        } else {
            t
        };
        if !value.is_finite() {
            return Err(Error::Numerical("inverse output transform overflow".into()));
        }
        Ok(value)
    }
}
