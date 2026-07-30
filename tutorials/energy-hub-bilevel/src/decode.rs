//! Authoritative normalized outer-decision decoder.

use serde::{Deserialize, Serialize};

use crate::dispatch::HubCapacities;

/// Outer decision dimension.
pub const DIMENSION: usize = 10;
/// Available grid-connection tiers in kW.
pub const GRID_TIERS_KW: [f64; 6] = [500.0, 850.0, 1_250.0, 1_800.0, 2_600.0, 4_000.0];

/// Decoded architecture and capacities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OuterDesign {
    /// Physical capacities.
    pub capacities: HubCapacities,
    /// Grid-catalogue index.
    pub grid_tier: usize,
    /// Wind architecture flag.
    pub include_wind: bool,
    /// Battery architecture flag.
    pub include_battery: bool,
    /// Hydrogen architecture flag.
    pub include_hydrogen: bool,
}

/// Decoder error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// Decision vector has the wrong length.
    Dimension,
    /// At least one coordinate is non-finite.
    NonFinite,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dimension => "outer decision vector has the wrong dimension".fmt(formatter),
            Self::NonFinite => {
                "outer decision vector contains a non-finite coordinate".fmt(formatter)
            }
        }
    }
}

impl std::error::Error for DecodeError {}

fn coordinate(value: f64) -> Result<f64, DecodeError> {
    value
        .is_finite()
        .then(|| value.clamp(0.0, 1.0))
        .ok_or(DecodeError::NonFinite)
}

/// Decode an equal-width categorical coordinate, including both endpoints.
pub fn categorical_index(value: f64, choices: usize) -> Result<usize, DecodeError> {
    let value = coordinate(value)?;
    if choices == 0 {
        return Err(DecodeError::Dimension);
    }
    Ok(((value * choices as f64).floor() as usize).min(choices - 1))
}

fn linear(value: f64, lower: f64, upper: f64) -> Result<f64, DecodeError> {
    Ok(lower + coordinate(value)? * (upper - lower))
}

fn logarithmic(value: f64, lower: f64, upper: f64) -> Result<f64, DecodeError> {
    Ok((lower.ln() + coordinate(value)? * (upper.ln() - lower.ln())).exp())
}

/// Decode capacities and architecture flags.
pub fn decode_outer(values: &[f64], annual_hydrogen: bool) -> Result<OuterDesign, DecodeError> {
    if values.len() != DIMENSION {
        return Err(DecodeError::Dimension);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DecodeError::NonFinite);
    }
    let grid_tier = categorical_index(values[6], GRID_TIERS_KW.len())?;
    let include_wind = categorical_index(values[7], 2)? == 1;
    let include_battery = categorical_index(values[8], 2)? == 1;
    let include_hydrogen = annual_hydrogen && categorical_index(values[9], 2)? == 1;
    Ok(OuterDesign {
        capacities: HubCapacities {
            pv_kwp: linear(values[0], 0.0, 5_000.0)?,
            wind_kw: if include_wind {
                linear(values[1], 0.0, 3_000.0)?
            } else {
                0.0
            },
            battery_kwh: if include_battery {
                logarithmic(values[2], 100.0, 20_000.0)?
            } else {
                0.0
            },
            battery_kw: if include_battery {
                logarithmic(values[3], 50.0, 5_000.0)?
            } else {
                0.0
            },
            electrolyser_kw: if include_hydrogen {
                linear(values[4], 0.0, 2_500.0)?
            } else {
                0.0
            },
            hydrogen_kwh: if include_hydrogen {
                linear(values[5], 0.0, 120_000.0)?
            } else {
                0.0
            },
            grid_kw: GRID_TIERS_KW[grid_tier],
        },
        grid_tier,
        include_wind,
        include_battery,
        include_hydrogen,
    })
}

#[cfg(test)]
mod tests {
    use fcmaes_core::parallel_batch;

    use super::*;

    #[test]
    fn tier_bins_are_flat_and_cover_endpoints() {
        assert_eq!(categorical_index(0.0, 6), Ok(0));
        assert_eq!(categorical_index(1.0, 6), Ok(5));
        let mut histogram = [0_usize; 6];
        for sample in 0..1_000_000 {
            let value = (sample as f64 + 0.5) / 1_000_000.0;
            histogram[categorical_index(value, 6).unwrap()] += 1;
        }
        let expected = 1_000_000.0 / 6.0;
        assert!(
            histogram
                .iter()
                .all(|count| (*count as f64 - expected).abs() / expected < 0.01)
        );
    }

    #[test]
    fn exclusions_zero_both_capacity_blocks() {
        let values = vec![0.49; DIMENSION];
        let main = decode_outer(&values, false).unwrap();
        assert!(!main.include_wind);
        assert!(!main.include_battery);
        assert!(!main.include_hydrogen);
        assert_eq!(main.capacities.wind_kw, 0.0);
        assert_eq!(main.capacities.battery_kwh, 0.0);
        assert_eq!(main.capacities.electrolyser_kw, 0.0);
    }

    #[test]
    fn serial_and_parallel_decoding_match() {
        let candidates = (0..10_000)
            .map(|row| {
                (0..DIMENSION)
                    .map(|column| ((row * 17 + column * 31) % 10_001) as f64 / 10_000.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let serial = candidates
            .iter()
            .map(|candidate| decode_outer(candidate, true).unwrap())
            .collect::<Vec<_>>();
        let parallel = parallel_batch(&candidates, 4, |candidate| {
            decode_outer(candidate, true).unwrap()
        });
        for (left, right) in serial.iter().zip(parallel) {
            assert_eq!(
                serde_json::to_string(left).unwrap(),
                serde_json::to_string(&right).unwrap()
            );
        }
    }
}
