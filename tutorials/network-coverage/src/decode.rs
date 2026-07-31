//! Two-level decoding over the deliberate `[0, 2)` search box.

/// Strict upper bound used instead of the open endpoint two.
pub const UPPER: f64 = 1.999_999_999_999;

/// Decode one coordinate per node.
#[must_use]
pub fn decode(controls: &[f64]) -> Option<Vec<bool>> {
    controls
        .iter()
        .map(|value| value.is_finite().then_some(*value >= 1.0))
        .collect()
}

/// Convert a bit mask into exact optimizer coordinates.
#[must_use]
pub fn controls(selected: &[bool]) -> Vec<f64> {
    selected
        .iter()
        .map(|chosen| if *chosen { 1.5 } else { 0.5 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_and_non_finite_values_are_explicit() {
        assert_eq!(
            decode(&[0.0, 0.999_999, 1.0, UPPER]),
            Some(vec![false, false, true, true])
        );
        assert!(decode(&[f64::NAN]).is_none());
        assert!(decode(&[f64::INFINITY]).is_none());
    }

    #[test]
    fn million_midpoints_split_into_equal_binary_bins() {
        let mut counts = [0_usize; 2];
        let samples = 1_000_000;
        for index in 0..samples {
            let value = UPPER * (index as f64 + 0.5) / samples as f64;
            counts[usize::from(decode(&[value]).unwrap()[0])] += 1;
        }
        assert!((counts[0] as isize - counts[1] as isize).abs() <= 1);
    }

    #[test]
    fn one_coordinate_sweep_flips_one_node_once() {
        let mut states = Vec::new();
        for index in 0..=1_000 {
            let mut controls = vec![0.5; 6];
            controls[3] = UPPER * index as f64 / 1_000.0;
            let decoded = decode(&controls).unwrap();
            if states.last() != Some(&decoded) {
                states.push(decoded);
            }
        }
        assert_eq!(states.len(), 2);
        assert_eq!(
            states[0]
                .iter()
                .zip(&states[1])
                .filter(|(left, right)| left != right)
                .count(),
            1
        );
    }
}
