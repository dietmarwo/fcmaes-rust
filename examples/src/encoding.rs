//! Small, deterministic decoders for fixed-length real optimizer vectors.
//!
//! These helpers are example code, not native heterogeneous chromosome
//! operators. They turn bounded `f64` coordinates into common discrete
//! structures while making bounds, tie handling, and repair rules explicit.

use std::fmt::{Display, Formatter};

/// Invalid input to a combinatorial decoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodingError {
    /// A decoder received the wrong number of coordinates.
    InvalidDimension { expected: usize, actual: usize },
    /// A decision coordinate or numeric bound is not finite.
    NonFinite,
    /// Bounds are empty, reversed, or invalid for the requested transform.
    InvalidBounds,
    /// A categorical decoder was asked to choose from no alternatives.
    EmptyChoices,
    /// A subset or repaired selection asks for more items than are available.
    InvalidCardinality { requested: usize, available: usize },
}

impl Display for EncodingError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimension { expected, actual } => {
                write!(
                    formatter,
                    "decision vector has length {actual}, expected {expected}"
                )
            }
            Self::NonFinite => formatter.write_str("decision coordinates must be finite"),
            Self::InvalidBounds => formatter.write_str("encoding bounds are invalid"),
            Self::EmptyChoices => formatter.write_str("categorical choices must be non-empty"),
            Self::InvalidCardinality {
                requested,
                available,
            } => write!(
                formatter,
                "requested cardinality {requested} exceeds {available} available items"
            ),
        }
    }
}

impl std::error::Error for EncodingError {}

fn unit_coordinate(value: f64) -> Result<f64, EncodingError> {
    if value.is_finite() {
        Ok(value.clamp(0.0, 1.0))
    } else {
        Err(EncodingError::NonFinite)
    }
}

/// Decode a normalized coordinate into an integer with equal-width bins.
///
/// Both endpoints are reachable: `0.0` maps to `lower` and `1.0` maps to
/// `upper`. Equal-width bins avoid the half-width endpoint bins produced by
/// rounding a linearly interpolated coordinate.
pub fn linear_integer(value: f64, lower: usize, upper: usize) -> Result<usize, EncodingError> {
    let count = upper
        .checked_sub(lower)
        .and_then(|width| width.checked_add(1))
        .ok_or(EncodingError::InvalidBounds)?;
    let index = categorical_index(value, count)?;
    lower.checked_add(index).ok_or(EncodingError::InvalidBounds)
}

/// Decode a normalized coordinate onto a positive logarithmic integer scale.
///
/// This is appropriate for quantities such as population size or tree count
/// whose useful scale is multiplicative. Unlike [`linear_integer`], decoded
/// integers do not occupy equal-width coordinate bins.
pub fn logarithmic_integer(value: f64, lower: usize, upper: usize) -> Result<usize, EncodingError> {
    if lower == 0 || lower > upper {
        return Err(EncodingError::InvalidBounds);
    }
    let unit = unit_coordinate(value)?;
    let log_lower = (lower as f64).ln();
    let log_upper = (upper as f64).ln();
    let decoded = (log_lower + unit * (log_upper - log_lower)).exp().round();
    Ok(decoded.clamp(lower as f64, upper as f64) as usize)
}

/// Decode a normalized coordinate into one of `choices` equal-width bins.
pub fn categorical_index(value: f64, choices: usize) -> Result<usize, EncodingError> {
    if choices == 0 {
        return Err(EncodingError::EmptyChoices);
    }
    let unit = unit_coordinate(value)?;
    Ok(((unit * choices as f64).floor() as usize).min(choices - 1))
}

/// Decode a normalized coordinate as a Boolean using a `0.5` threshold.
pub fn boolean(value: f64) -> Result<bool, EncodingError> {
    Ok(unit_coordinate(value)? >= 0.5)
}

/// Return item indices ordered by ascending random key.
///
/// Equal keys are broken by original item index, so decoding is deterministic
/// across serial and parallel evaluation.
pub fn permutation_from_keys(keys: &[f64]) -> Result<Vec<usize>, EncodingError> {
    if keys.iter().any(|value| !value.is_finite()) {
        return Err(EncodingError::NonFinite);
    }
    let mut indices: Vec<usize> = (0..keys.len()).collect();
    indices.sort_by(|&left, &right| {
        keys[left]
            .total_cmp(&keys[right])
            .then_with(|| left.cmp(&right))
    });
    Ok(indices)
}

/// Select exactly `count` items using the smallest random keys.
pub fn select_k_from_keys(keys: &[f64], count: usize) -> Result<Vec<usize>, EncodingError> {
    if count > keys.len() {
        return Err(EncodingError::InvalidCardinality {
            requested: count,
            available: keys.len(),
        });
    }
    let mut selected = permutation_from_keys(keys)?;
    selected.truncate(count);
    Ok(selected)
}

/// Decode categorical choices and repair collisions by advancing cyclically.
///
/// This always returns distinct indices, but the repair is biased toward the
/// next free index. Prefer [`select_k_from_keys`] when that bias is unwanted.
pub fn unique_indices_with_repair(
    values: &[f64],
    available: usize,
) -> Result<Vec<usize>, EncodingError> {
    if values.len() > available {
        return Err(EncodingError::InvalidCardinality {
            requested: values.len(),
            available,
        });
    }
    if values.is_empty() {
        return Ok(Vec::new());
    }
    if available == 0 {
        return Err(EncodingError::EmptyChoices);
    }
    let mut used = vec![false; available];
    let mut result = Vec::with_capacity(values.len());
    for &value in values {
        let mut candidate = categorical_index(value, available)?;
        while used[candidate] {
            candidate = (candidate + 1) % available;
        }
        used[candidate] = true;
        result.push(candidate);
    }
    Ok(result)
}

/// Decode an item permutation and partition it at normalized separator keys.
///
/// `separator_keys.len() + 1` groups are returned. Separators map to the
/// `0..=items` cut positions and are sorted. Repeated cuts deliberately create
/// empty groups; reject or repair them in the application when emptiness is
/// forbidden.
pub fn partition_from_keys(
    item_keys: &[f64],
    separator_keys: &[f64],
) -> Result<Vec<Vec<usize>>, EncodingError> {
    let permutation = permutation_from_keys(item_keys)?;
    let cut_slots = item_keys
        .len()
        .checked_add(1)
        .ok_or(EncodingError::InvalidBounds)?;
    let mut cuts = Vec::with_capacity(separator_keys.len());
    for &value in separator_keys {
        cuts.push(categorical_index(value, cut_slots)?);
    }
    cuts.sort_unstable();

    let mut groups = Vec::with_capacity(cuts.len() + 1);
    let mut previous = 0;
    for cut in cuts {
        groups.push(permutation[previous..cut].to_vec());
        previous = cut;
    }
    groups.push(permutation[previous..].to_vec());
    Ok(groups)
}

/// Map normalized coordinates into an interval and return them in time order.
pub fn sorted_breakpoints(
    values: &[f64],
    lower: f64,
    upper: f64,
) -> Result<Vec<f64>, EncodingError> {
    if !lower.is_finite() || !upper.is_finite() {
        return Err(EncodingError::NonFinite);
    }
    if lower >= upper {
        return Err(EncodingError::InvalidBounds);
    }
    let mut decoded = Vec::with_capacity(values.len());
    for &value in values {
        decoded.push(lower + unit_coordinate(value)? * (upper - lower));
    }
    decoded.sort_by(f64::total_cmp);
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_decoders_cover_bounds_and_reject_bad_inputs() {
        assert_eq!(
            EncodingError::InvalidDimension {
                expected: 4,
                actual: 3
            }
            .to_string(),
            "decision vector has length 3, expected 4"
        );
        assert_eq!(linear_integer(0.0, 3, 7), Ok(3));
        assert_eq!(linear_integer(0.2, 3, 7), Ok(4));
        assert_eq!(linear_integer(1.0, 3, 7), Ok(7));
        assert_eq!(linear_integer(-1.0, 3, 7), Ok(3));
        assert_eq!(linear_integer(2.0, 3, 7), Ok(7));
        assert_eq!(linear_integer(0.5, 7, 3), Err(EncodingError::InvalidBounds));

        assert_eq!(logarithmic_integer(0.0, 8, 512), Ok(8));
        assert_eq!(logarithmic_integer(1.0, 8, 512), Ok(512));
        assert_eq!(
            logarithmic_integer(0.5, 0, 512),
            Err(EncodingError::InvalidBounds)
        );

        assert_eq!(categorical_index(0.0, 4), Ok(0));
        assert_eq!(categorical_index(0.25, 4), Ok(1));
        assert_eq!(categorical_index(1.0, 4), Ok(3));
        assert_eq!(categorical_index(0.5, 0), Err(EncodingError::EmptyChoices));
        assert_eq!(boolean(0.499), Ok(false));
        assert_eq!(boolean(0.5), Ok(true));
        assert_eq!(boolean(f64::NAN), Err(EncodingError::NonFinite));
    }

    #[test]
    fn random_keys_are_deterministic_for_ties() {
        assert_eq!(
            permutation_from_keys(&[0.8, 0.1, 0.1, 0.4]),
            Ok(vec![1, 2, 3, 0])
        );
        assert_eq!(
            permutation_from_keys(&[0.0, f64::INFINITY]),
            Err(EncodingError::NonFinite)
        );
    }

    #[test]
    fn top_k_has_exact_cardinality() {
        assert_eq!(select_k_from_keys(&[0.7, 0.2, 0.9, 0.1], 2), Ok(vec![3, 1]));
        assert_eq!(
            select_k_from_keys(&[0.0], 2),
            Err(EncodingError::InvalidCardinality {
                requested: 2,
                available: 1
            })
        );
    }

    #[test]
    fn collision_repair_returns_distinct_indices() {
        assert_eq!(
            unique_indices_with_repair(&[0.0, 0.0, 0.0], 4),
            Ok(vec![0, 1, 2])
        );
        assert_eq!(unique_indices_with_repair(&[], 0), Ok(Vec::new()));
        assert_eq!(
            unique_indices_with_repair(&[0.0, 0.5], 1),
            Err(EncodingError::InvalidCardinality {
                requested: 2,
                available: 1
            })
        );
    }

    #[test]
    fn separators_partition_one_permutation() {
        let groups = partition_from_keys(&[0.4, 0.1, 0.3, 0.2], &[0.2, 0.8]).unwrap();
        assert_eq!(groups, [vec![1], vec![3, 2, 0], vec![]]);
        let flattened: Vec<_> = groups.into_iter().flatten().collect();
        assert_eq!(flattened, [1, 3, 2, 0]);

        let empty_groups = partition_from_keys(&[0.2, 0.1], &[0.0, 0.0]).unwrap();
        assert_eq!(empty_groups, [vec![], vec![], vec![1, 0]]);
    }

    #[test]
    fn breakpoints_are_bounded_and_sorted() {
        assert_eq!(
            sorted_breakpoints(&[0.8, 0.0, 0.4, 1.0], 10.0, 20.0),
            Ok(vec![10.0, 14.0, 18.0, 20.0])
        );
        assert_eq!(
            sorted_breakpoints(&[0.5], 2.0, 2.0),
            Err(EncodingError::InvalidBounds)
        );
        assert_eq!(
            sorted_breakpoints(&[f64::NAN], 0.0, 1.0),
            Err(EncodingError::NonFinite)
        );
    }
}
