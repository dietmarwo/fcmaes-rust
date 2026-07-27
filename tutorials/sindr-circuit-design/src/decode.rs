//! Logarithmic continuous decoding and discrete E12 component catalogues.

const E12_BASE: [f64; 12] = [
    10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 39.0, 47.0, 56.0, 68.0, 82.0,
];

/// Decode `u` from `[0, 1]` into the positive interval `[lower, upper]`.
pub fn log_decode(u: f64, lower: f64, upper: f64) -> f64 {
    lower * (upper / lower).powf(u.clamp(0.0, 1.0))
}

/// Build an inclusive E12 table over a positive numeric interval.
pub fn e12_values(lower: f64, upper: f64) -> Vec<f64> {
    assert!(lower.is_finite() && upper.is_finite() && 0.0 < lower && lower <= upper);
    let first_power = lower.log10().floor() as i32 - 2;
    let last_power = upper.log10().ceil() as i32;
    let mut values = Vec::new();
    for power in first_power..=last_power {
        let scale = 10_f64.powi(power);
        for base in E12_BASE {
            let value = base * scale;
            if value >= lower * (1.0 - 1e-12) && value <= upper * (1.0 + 1e-12) {
                values.push(value.clamp(lower, upper));
            }
        }
    }
    values.sort_by(f64::total_cmp);
    values.dedup_by(|left, right| (*left - *right).abs() <= 1e-12 * left.abs().max(1.0));
    values
}

/// Round and clamp a continuous catalogue coordinate, then return its index.
pub fn catalogue_index(coordinate: f64, length: usize) -> usize {
    assert!(length > 0);
    coordinate.round().clamp(0.0, (length - 1) as f64) as usize
}

/// Decode normalized MFB band-pass controls into `[R1, R2, R3, C1, C2]`.
pub fn decode_bandpass_continuous(u: &[f64]) -> [f64; 5] {
    assert_eq!(u.len(), 5);
    [
        log_decode(u[0], 100.0, 100_000.0),
        log_decode(u[1], 100.0, 100_000.0),
        log_decode(u[2], 100.0, 100_000.0),
        log_decode(u[3], 10e-12, 1e-6),
        log_decode(u[4], 10e-12, 1e-6),
    ]
}

/// Decode continuous index coordinates into an E12 MFB component tuple.
pub fn decode_bandpass_e12(
    x: &[f64],
    resistor_values: &[f64],
    capacitor_values: &[f64],
) -> ([f64; 5], [usize; 5]) {
    assert_eq!(x.len(), 5);
    let indices = [
        catalogue_index(x[0], resistor_values.len()),
        catalogue_index(x[1], resistor_values.len()),
        catalogue_index(x[2], resistor_values.len()),
        catalogue_index(x[3], capacitor_values.len()),
        catalogue_index(x[4], capacitor_values.len()),
    ];
    (
        [
            resistor_values[indices[0]],
            resistor_values[indices[1]],
            resistor_values[indices[2]],
            capacitor_values[indices[3]],
            capacitor_values[indices[4]],
        ],
        indices,
    )
}

/// Decode normalized low-pass controls into `[R1, R2, R3, R4, C1, C2, C3, C4]`.
pub fn decode_lowpass(u: &[f64]) -> [f64; 8] {
    assert_eq!(u.len(), 8);
    [
        log_decode(u[0], 100.0, 100_000.0),
        log_decode(u[1], 100.0, 100_000.0),
        log_decode(u[2], 100.0, 100_000.0),
        log_decode(u[3], 100.0, 100_000.0),
        log_decode(u[4], 10e-12, 100e-9),
        log_decode(u[5], 10e-12, 100e-9),
        log_decode(u[6], 10e-12, 100e-9),
        log_decode(u[7], 10e-12, 100e-9),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logarithmic_decode_hits_bounds_and_geometric_midpoint() {
        assert_eq!(log_decode(0.0, 100.0, 100_000.0), 100.0);
        assert_eq!(log_decode(1.0, 100.0, 100_000.0), 100_000.0);
        assert!((log_decode(0.5, 100.0, 100_000.0) - 3162.2776601683795).abs() < 1e-9);
    }

    #[test]
    fn e12_tables_include_requested_endpoints_without_duplicates() {
        let resistors = e12_values(100.0, 100_000.0);
        let capacitors = e12_values(10e-12, 1e-6);
        assert_eq!(resistors.len(), 37);
        assert_eq!(capacitors.len(), 61);
        assert_eq!(resistors[0], 100.0);
        assert_eq!(resistors[36], 100_000.0);
        assert_eq!(capacitors[0], 10e-12);
        assert_eq!(capacitors[60], 1e-6);
        assert!(resistors.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(capacitors.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn catalogue_decoding_rounds_and_clamps() {
        assert_eq!(catalogue_index(-4.0, 12), 0);
        assert_eq!(catalogue_index(2.49, 12), 2);
        assert_eq!(catalogue_index(2.51, 12), 3);
        assert_eq!(catalogue_index(99.0, 12), 11);
    }
}
