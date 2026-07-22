//! Native Rust Mazda three-car factory-design benchmark.
//!
//! The discrete decision table, three mass regressions, and 42 generated
//! radial-basis response surfaces are embedded in the example crate. Shared
//! training matrices are stored once and their distances are reused across
//! related constraints, so no C++ compiler, dynamic library, Python source,
//! or runtime data path is required.

use crate::mazda_model;

pub const MAZDA_DIM: usize = 222;
pub const MAZDA_OBJECTIVES: usize = 2;
pub const MAZDA_CONSTRAINTS: usize = 54;
pub const MAZDA_VALUE_WIDTH: usize = MAZDA_OBJECTIVES + MAZDA_CONSTRAINTS;
pub const MAZDA_QD_LOWER: [f64; 2] = [2.0, -74.0];
pub const MAZDA_QD_UPPER: [f64; 2] = [3.5, 0.0];

/// Embedded discrete thickness choices for the three cars.
#[derive(Clone, Debug)]
pub struct MazdaDecisionSpace {
    choices: Vec<Vec<f64>>,
}

impl MazdaDecisionSpace {
    /// Load the decision choices embedded in this crate.
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            choices: mazda_model::decision_choices()?,
        })
    }

    pub fn dim(&self) -> usize {
        self.choices.len()
    }

    pub fn lower(&self) -> Vec<f64> {
        vec![0.0; self.dim()]
    }

    pub fn upper(&self) -> Vec<f64> {
        self.choices
            .iter()
            .map(|values| values.len() as f64 - 1e-9)
            .collect()
    }

    /// Convert MODE/MAP-Elites coordinates into physical thicknesses.
    /// Coordinates follow Python's `int(xi)` convention.
    pub fn decode(&self, indices: &[f64]) -> Result<Vec<f64>, String> {
        let mut physical = vec![0.0; self.dim()];
        self.decode_into(indices, &mut physical)?;
        Ok(physical)
    }

    /// Decode into a caller-provided buffer, avoiding an allocation in hot
    /// objective-function loops.
    pub fn decode_into(&self, indices: &[f64], physical: &mut [f64]) -> Result<(), String> {
        if indices.len() != self.dim() {
            return Err(format!(
                "Mazda decision vector has length {}, expected {}",
                indices.len(),
                self.dim()
            ));
        }
        if physical.len() != self.dim() {
            return Err(format!(
                "Mazda physical output has length {}, expected {}",
                physical.len(),
                self.dim()
            ));
        }
        for ((output, &index), values) in physical.iter_mut().zip(indices).zip(&self.choices) {
            if !index.is_finite() {
                return Err("Mazda decision index is not finite".to_string());
            }
            let selected = index.trunc().clamp(0.0, (values.len() - 1) as f64) as usize;
            *output = values[selected];
        }
        Ok(())
    }

    pub fn choices(&self) -> &[Vec<f64>] {
        &self.choices
    }
}

/// Native three-car response-surface evaluator.
#[derive(Clone, Copy, Debug, Default)]
pub struct MazdaEvaluator;

impl MazdaEvaluator {
    pub fn new() -> Result<Self, String> {
        mazda_model::validate()?;
        Ok(Self)
    }

    /// Return `[mass, -common_parts, constraints...]`.
    ///
    /// The published Mazda model uses feasible `constraint >= 0`; MODE uses
    /// feasible `constraint <= 0`, so the 54 response values are negated here.
    pub fn evaluate_physical(&self, physical: &[f64]) -> Result<Vec<f64>, String> {
        let raw = mazda_model::evaluate(physical)?;
        let mut values = Vec::with_capacity(MAZDA_VALUE_WIDTH);
        values.extend_from_slice(&raw.objectives[..MAZDA_OBJECTIVES]);
        values.extend(raw.constraints.iter().map(|&constraint| -constraint));
        Ok(values)
    }

    pub fn evaluate_indices(
        &self,
        space: &MazdaDecisionSpace,
        indices: &[f64],
    ) -> Result<Vec<f64>, String> {
        let mut physical = [0.0; MAZDA_DIM];
        space.decode_into(indices, &mut physical)?;
        self.evaluate_physical(&physical)
    }
}

/// Mazda QD fitness and behavior descriptor used by the Python sample.
pub fn qd_value(values: &[f64]) -> Result<(f64, Vec<f64>), String> {
    if values.len() != MAZDA_VALUE_WIDTH || values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "Mazda QD input must contain {MAZDA_VALUE_WIDTH} finite values"
        ));
    }
    let constraint_penalty: f64 = values[MAZDA_OBJECTIVES..]
        .iter()
        .filter(|&&constraint| constraint > 0.0)
        .map(|&constraint| 1_000.0 * (constraint + 1.0))
        .sum();
    let mass =
        ((values[0] - MAZDA_QD_LOWER[0]) / (MAZDA_QD_UPPER[0] - MAZDA_QD_LOWER[0])).min(100_000.0);
    let common_parts =
        ((values[1] - MAZDA_QD_LOWER[1]) / (MAZDA_QD_UPPER[1] - MAZDA_QD_LOWER[1])).min(100_000.0);
    Ok((
        mass + common_parts + constraint_penalty,
        values[..MAZDA_OBJECTIVES].to_vec(),
    ))
}

pub fn is_feasible(values: &[f64]) -> bool {
    values.len() == MAZDA_VALUE_WIDTH
        && values[MAZDA_OBJECTIVES..]
            .iter()
            .all(|&constraint| constraint <= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_decision_table_has_expected_shape() {
        let space = MazdaDecisionSpace::new().unwrap();
        assert_eq!(space.dim(), MAZDA_DIM);
        assert!(space.choices().iter().all(|choices| choices.len() >= 5));
        assert_eq!(space.lower().len(), MAZDA_DIM);
        assert_eq!(space.upper().len(), MAZDA_DIM);
    }

    #[test]
    fn decoding_clamps_and_rejects_invalid_coordinates() {
        let space = MazdaDecisionSpace::new().unwrap();
        let low = space.decode(&vec![-10.0; MAZDA_DIM]).unwrap();
        let high = space.decode(&vec![f64::MAX; MAZDA_DIM]).unwrap();
        assert_eq!(low[0], space.choices()[0][0]);
        assert_eq!(high[0], *space.choices()[0].last().unwrap());
        assert!(space.decode(&[0.0; 2]).is_err());
        let mut invalid = vec![0.0; MAZDA_DIM];
        invalid[17] = f64::NAN;
        assert!(space.decode(&invalid).is_err());
    }

    #[test]
    fn evaluator_rejects_bad_inputs() {
        let evaluator = MazdaEvaluator::new().unwrap();
        assert!(evaluator.evaluate_physical(&[1.0; 4]).is_err());
        let mut invalid = vec![1.0; MAZDA_DIM];
        invalid[0] = f64::INFINITY;
        assert!(evaluator.evaluate_physical(&invalid).is_err());
    }

    #[test]
    fn qd_conversion_penalizes_violations() {
        let mut values = vec![0.0; MAZDA_VALUE_WIDTH];
        values[0] = 2.75;
        values[1] = -37.0;
        let (valid, descriptor) = qd_value(&values).unwrap();
        assert_eq!(valid, 1.0);
        assert_eq!(descriptor, vec![2.75, -37.0]);
        values[2] = 0.25;
        assert_eq!(qd_value(&values).unwrap().0, 1_251.0);
        assert!(!is_feasible(&values));
    }

    #[test]
    fn rejects_bad_qd_input() {
        assert!(qd_value(&[1.0, 2.0]).is_err());
        let mut values = vec![0.0; MAZDA_VALUE_WIDTH];
        values[0] = f64::NAN;
        assert!(qd_value(&values).is_err());
    }
}
