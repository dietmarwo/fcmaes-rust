use crate::{Error, MAX_DIMENSION, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Space {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub log_scale: Vec<bool>,
}
impl Space {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>, log_scale: Vec<bool>) -> Result<Self> {
        let s = Self {
            lower,
            upper,
            log_scale,
        };
        s.validate()?;
        Ok(s)
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.lower.len();
        if !(1..=MAX_DIMENSION).contains(&n) || self.upper.len() != n || self.log_scale.len() != n {
            return Err(Error::Invalid(
                "space needs 1..32 matching bounds and log_scale entries".into(),
            ));
        }
        for (i, ((&lo, &hi), &log)) in self
            .lower
            .iter()
            .zip(&self.upper)
            .zip(&self.log_scale)
            .enumerate()
        {
            if !lo.is_finite()
                || !hi.is_finite()
                || lo >= hi
                || !(hi - lo).is_finite()
                || (log && (lo <= 0.0 || libm::log(hi) <= libm::log(lo)))
            {
                return Err(Error::Invalid(format!(
                    "invalid bounds or log scale at coordinate {i}"
                )));
            }
        }
        Ok(())
    }
    pub fn dimension(&self) -> usize {
        self.lower.len()
    }
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dimension()
            && x.iter()
                .zip(self.lower.iter().zip(&self.upper))
                .all(|(&v, (&lo, &hi))| v.is_finite() && v >= lo && v <= hi)
    }
    pub fn to_unit(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.validate()?;
        if !self.contains(x) {
            return Err(Error::Invalid("point outside the space".into()));
        }
        Ok(x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let (lo, hi) = (self.lower[i], self.upper[i]);
                if self.log_scale[i] {
                    ((libm::log(v) - libm::log(lo)) / (libm::log(hi) - libm::log(lo)))
                        .clamp(0.0, 1.0)
                } else {
                    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
                }
            })
            .collect())
    }
    pub fn from_unit(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.validate()?;
        if x.len() != self.dimension()
            || x.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(Error::Invalid(
                "unit point needs matching entries in [0,1]".into(),
            ));
        }
        Ok(x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let (lo, hi) = (self.lower[i], self.upper[i]);
                let value = if self.log_scale[i] {
                    libm::exp(libm::log(lo) + v * (libm::log(hi) - libm::log(lo)))
                } else {
                    lo + v * (hi - lo)
                };
                value.clamp(lo, hi)
            })
            .collect())
    }
}
