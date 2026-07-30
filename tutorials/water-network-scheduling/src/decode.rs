//! Authoritative normalized-coordinate decoder.

use crate::DIMENSION;

/// Quantized relative pump speeds.
pub const LEVELS: [f64; 4] = [0.0, 0.8, 0.9, 1.0];

/// Pump priority under the low-tank safety override.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Priority {
    Pump1,
    Pump2,
}

/// Physical control plan decoded from 28 normalized coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlPlan {
    pub levels: [[f64; 12]; 2],
    pub low_threshold_m: f64,
    pub high_threshold_m: f64,
    pub prv_setpoint_m: f64,
    pub priority: Priority,
}

fn finite_unit(value: f64, index: usize) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!("coordinate {index} is not finite"));
    }
    Ok(value.clamp(0.0, 1.0))
}

fn category(value: f64, count: usize) -> usize {
    ((value * count as f64).floor() as usize).min(count - 1)
}

/// Decode normalized coordinates with equal-width categorical bins.
pub fn decode(values: &[f64]) -> Result<ControlPlan, String> {
    if values.len() != DIMENSION {
        return Err(format!(
            "expected {DIMENSION} coordinates, got {}",
            values.len()
        ));
    }
    let mut unit = [0.0; DIMENSION];
    for (index, value) in values.iter().copied().enumerate() {
        unit[index] = finite_unit(value, index)?;
    }
    let mut levels = [[0.0; 12]; 2];
    for pump in 0..2 {
        for period in 0..12 {
            levels[pump][period] = LEVELS[category(unit[pump * 12 + period], LEVELS.len())];
        }
    }
    let low = 1.2 + unit[24] * (5.5 - 1.2);
    let high_min = low + 0.5;
    let high = high_min + unit[25] * (9.8 - high_min);
    Ok(ControlPlan {
        levels,
        low_threshold_m: low,
        high_threshold_m: high,
        prv_setpoint_m: 25.0 + 25.0 * unit[26],
        priority: if category(unit[27], 2) == 0 {
            Priority::Pump1
        } else {
            Priority::Pump2
        },
    })
}

/// Feasible deterministic optimizer seed.
#[must_use]
pub fn seed_controls() -> Vec<f64> {
    let mut values = vec![0.62; DIMENSION]; // 0.9 speed
    for period in 0..12 {
        values[12 + period] = if (3..=8).contains(&period) { 0.38 } else { 0.0 };
    }
    values[24] = 0.35;
    values[25] = 0.65;
    values[26] = 0.4;
    values[27] = 0.0;
    values
}

/// Deterministic schedule whose upper threshold trips inside a two-hour period.
#[must_use]
pub fn override_witness_plan() -> ControlPlan {
    let mut plan = decode(&seed_controls()).expect("the structured seed is valid");
    plan.low_threshold_m = 1.2;
    plan.high_threshold_m = 6.05;
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_reach_all_categories_and_order_thresholds() {
        let low = decode(&[0.0; DIMENSION]).unwrap();
        let high = decode(&[1.0; DIMENSION]).unwrap();
        assert_eq!(low.levels[0][0], 0.0);
        assert_eq!(high.levels[0][0], 1.0);
        assert!(low.low_threshold_m + 0.5 <= low.high_threshold_m);
        assert!(high.low_threshold_m + 0.5 <= high.high_threshold_m);
        assert_eq!(low.priority, Priority::Pump1);
        assert_eq!(high.priority, Priority::Pump2);
    }

    #[test]
    fn non_finite_coordinates_are_rejected() {
        let mut values = vec![0.5; DIMENSION];
        values[7] = f64::NAN;
        assert!(decode(&values).is_err());
    }

    #[test]
    fn bins_have_equal_width() {
        let mut counts = [0_usize; 4];
        for i in 0..10_000 {
            counts[category((i as f64 + 0.5) / 10_000.0, 4)] += 1;
        }
        assert_eq!(counts, [2_500; 4]);
    }

    #[test]
    fn override_witness_keeps_ordered_thresholds() {
        let plan = override_witness_plan();
        assert!(plan.low_threshold_m + 0.5 <= plan.high_threshold_m);
    }
}
