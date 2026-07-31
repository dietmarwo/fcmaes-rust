//! Strict loader for locally obtained CEC shift and rotation data.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

/// CEC loader failure with an actionable message.
#[derive(Debug)]
pub struct CecError(String);

impl fmt::Display for CecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CecError {}

/// Locally loaded shift vector and orthogonal rotation matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CecTransform {
    /// Shift vector.
    pub shift: Vec<f64>,
    /// Row-major rotation matrix.
    pub rotation: Vec<Vec<f64>>,
}

impl CecTransform {
    /// Load the documented text format without any identity fallback.
    ///
    /// # Errors
    ///
    /// Returns [`CecError`] for missing files, malformed dimensions, non-finite
    /// entries, or a matrix that is not orthogonal to `1e-9`.
    pub fn load(path: &Path) -> Result<Self, CecError> {
        let source = fs::read_to_string(path).map_err(|error| {
            CecError(format!("cannot read CEC data {}: {error}", path.display()))
        })?;
        let rows: Vec<Vec<&str>> = source
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| line.split_whitespace().collect())
            .collect();
        let dimension = rows
            .iter()
            .find(|row| row.first() == Some(&"dimension"))
            .and_then(|row| row.get(1))
            .ok_or_else(|| CecError("CEC data needs `dimension N`".to_owned()))?
            .parse::<usize>()
            .map_err(|error| CecError(format!("invalid CEC dimension: {error}")))?;
        if dimension == 0 {
            return Err(CecError("CEC dimension must be positive".to_owned()));
        }
        let shift_row = rows
            .iter()
            .find(|row| row.first() == Some(&"shift"))
            .ok_or_else(|| CecError("CEC data needs one shift row".to_owned()))?;
        let parse = |values: &[&str]| -> Result<Vec<f64>, CecError> {
            let parsed: Result<Vec<f64>, _> = values.iter().map(|value| value.parse()).collect();
            let parsed =
                parsed.map_err(|error| CecError(format!("invalid CEC number: {error}")))?;
            if parsed.len() != dimension || parsed.iter().any(|value| !value.is_finite()) {
                return Err(CecError(format!("CEC rows need {dimension} finite values")));
            }
            Ok(parsed)
        };
        let shift = parse(&shift_row[1..])?;
        let rotation: Result<Vec<_>, _> = rows
            .iter()
            .filter(|row| row.first() == Some(&"rotation"))
            .map(|row| parse(&row[1..]))
            .collect();
        let rotation = rotation?;
        if rotation.len() != dimension {
            return Err(CecError(format!(
                "CEC data needs {dimension} rotation rows"
            )));
        }
        for i in 0..dimension {
            for j in 0..dimension {
                let dot = rotation[i]
                    .iter()
                    .zip(&rotation[j])
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
                let expected = f64::from(i == j);
                if (dot - expected).abs() > 1.0e-9 {
                    return Err(CecError("CEC rotation is not orthogonal".to_owned()));
                }
            }
        }
        Ok(Self { shift, rotation })
    }

    /// Apply `rotation × (decision - shift)`.
    ///
    /// # Errors
    ///
    /// Returns [`CecError`] for a dimension mismatch or non-finite decision.
    pub fn transform(&self, decision: &[f64]) -> Result<Vec<f64>, CecError> {
        if decision.len() != self.shift.len() {
            return Err(CecError("CEC decision dimension mismatch".to_owned()));
        }
        if decision.iter().any(|value| !value.is_finite()) {
            return Err(CecError("CEC decision must be finite".to_owned()));
        }
        let shifted: Vec<f64> = decision
            .iter()
            .zip(&self.shift)
            .map(|(value, shift)| value - shift)
            .collect();
        Ok(self
            .rotation
            .iter()
            .map(|row| row.iter().zip(&shifted).map(|(a, b)| a * b).sum())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_file_round_trips_and_missing_file_is_actionable() {
        let path =
            std::env::temp_dir().join(format!("fcmaes-foundations-cec-{}.txt", std::process::id()));
        fs::write(
            &path,
            "dimension 2\nshift 1.5 -2\nrotation 0 -1\nrotation 1 0\n",
        )
        .unwrap();
        let transform = CecTransform::load(&path).unwrap();
        assert_eq!(transform.transform(&[2.5, 0.0]).unwrap(), vec![-2.0, 1.0]);
        fs::remove_file(&path).unwrap();
        let error = CecTransform::load(&path).unwrap_err().to_string();
        assert!(error.contains("cannot read CEC data"));
        assert!(error.contains(path.to_str().unwrap()));
    }
}
