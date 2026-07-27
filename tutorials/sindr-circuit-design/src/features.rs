//! Smooth features extracted from sampled AC curves.

use sindr::Circuit;
use sindr::ac_analysis::{AcConfig, FrequencySpacing, solve_ac};

/// Interpolated band-pass response features.
#[derive(Clone, Copy, Debug)]
pub struct BandpassFeatures {
    pub peak_hz: f64,
    pub peak_db: f64,
    pub lower_3db_hz: f64,
    pub upper_3db_hz: f64,
    pub q: f64,
}

/// Interpolated low-pass response features.
#[derive(Clone, Copy, Debug)]
pub struct LowpassFeatures {
    pub cutoff_hz: f64,
    pub passband_ripple_db: f64,
    pub peak_above_dc_db: f64,
}

/// Run a logarithmic AC sweep and return `(frequency_hz, gain_db)`.
pub fn gain_curve(
    circuit: &Circuit,
    output_node: &str,
    f_start: f64,
    f_stop: f64,
    points: usize,
) -> Option<Vec<(f64, f64)>> {
    if points < 3 || !(0.0 < f_start && f_start < f_stop) {
        return None;
    }
    let result = solve_ac(
        circuit,
        &AcConfig {
            f_start,
            f_stop,
            num_points: points,
            spacing: FrequencySpacing::Logarithmic,
            source_id: "V1".into(),
            ac_magnitude: 1.0,
        },
    )
    .ok()?;
    let curve = result.gain_curve(output_node);
    valid_curve(&curve).then_some(curve)
}

fn valid_curve(curve: &[(f64, f64)]) -> bool {
    curve.len() >= 3
        && curve
            .iter()
            .all(|(f, y)| f.is_finite() && *f > 0.0 && y.is_finite())
        && curve.windows(2).all(|pair| pair[0].0 < pair[1].0)
}

/// Index of the largest sampled gain.
pub fn peak_index(curve: &[(f64, f64)]) -> Option<usize> {
    valid_curve(curve).then(|| {
        (0..curve.len())
            .max_by(|&left, &right| curve[left].1.total_cmp(&curve[right].1))
            .expect("validated curve is not empty")
    })
}

fn interpolated_extreme(curve: &[(f64, f64)], maximum: bool) -> Option<(f64, f64)> {
    if !valid_curve(curve) {
        return None;
    }
    let index = (0..curve.len())
        .min_by(|&left, &right| {
            let ordering = curve[left].1.total_cmp(&curve[right].1);
            if maximum {
                ordering.reverse()
            } else {
                ordering
            }
        })
        .expect("validated curve is not empty");
    if index == 0 || index + 1 == curve.len() {
        return Some(curve[index]);
    }
    let x0 = curve[index - 1].0.ln();
    let x1 = curve[index].0.ln();
    let x2 = curve[index + 1].0.ln();
    let y0 = curve[index - 1].1;
    let y1 = curve[index].1;
    let y2 = curve[index + 1].1;
    let slope01 = (y1 - y0) / (x1 - x0);
    let curvature = ((y2 - y1) / (x2 - x1) - slope01) / (x2 - x0);
    if !curvature.is_finite()
        || curvature.abs() <= f64::EPSILON
        || (maximum && curvature >= 0.0)
        || (!maximum && curvature <= 0.0)
    {
        return Some(curve[index]);
    }
    let linear = slope01 - curvature * (x0 + x1);
    let vertex = -linear / (2.0 * curvature);
    if !vertex.is_finite() || !(x0..=x2).contains(&vertex) {
        return Some(curve[index]);
    }
    let value = y0 + slope01 * (vertex - x0) + curvature * (vertex - x0) * (vertex - x1);
    value.is_finite().then_some((vertex.exp(), value))
}

/// Peak of a `(frequency, dB)` curve, parabolically interpolated in log-frequency.
pub fn interpolated_peak(curve: &[(f64, f64)]) -> Option<(f64, f64)> {
    let index = peak_index(curve)?;
    if index == 0 || index + 1 == curve.len() {
        return Some(curve[index]);
    }
    let x0 = curve[index - 1].0.ln();
    let x1 = curve[index].0.ln();
    let x2 = curve[index + 1].0.ln();
    let y0 = curve[index - 1].1;
    let y1 = curve[index].1;
    let y2 = curve[index + 1].1;
    let denominator = y0 - 2.0 * y1 + y2;
    if denominator.abs() <= f64::EPSILON || (x2 - x0).abs() <= f64::EPSILON {
        return Some(curve[index]);
    }
    let shift = (0.5 * (y0 - y2) / denominator).clamp(-1.0, 1.0);
    let peak_ln_f = x1 + 0.5 * shift * (x2 - x0);
    let peak_db = y1 - 0.25 * (y0 - y2) * shift;
    Some((peak_ln_f.exp(), peak_db))
}

fn segment_crossing(left: (f64, f64), right: (f64, f64), level_db: f64) -> Option<f64> {
    let dy = right.1 - left.1;
    if dy.abs() <= f64::EPSILON {
        return None;
    }
    let fraction = (level_db - left.1) / dy;
    if !(0.0..=1.0).contains(&fraction) {
        return None;
    }
    Some((left.0.ln() + fraction * (right.0.ln() - left.0.ln())).exp())
}

/// Find a level crossing left or right of a supplied grid index.
pub fn crossing(
    curve: &[(f64, f64)],
    from: usize,
    level_db: f64,
    search_right: bool,
) -> Option<f64> {
    if !valid_curve(curve) || from >= curve.len() {
        return None;
    }
    if search_right {
        (from..curve.len() - 1)
            .find_map(|index| segment_crossing(curve[index], curve[index + 1], level_db))
    } else {
        (1..=from)
            .rev()
            .find_map(|index| segment_crossing(curve[index - 1], curve[index], level_db))
    }
}

fn gain_at(curve: &[(f64, f64)], frequency_hz: f64) -> Option<f64> {
    if !valid_curve(curve)
        || !frequency_hz.is_finite()
        || frequency_hz < curve.first()?.0
        || frequency_hz > curve.last()?.0
    {
        return None;
    }
    if let Some((_, gain)) = curve
        .iter()
        .find(|(frequency, _)| *frequency == frequency_hz)
    {
        return Some(*gain);
    }
    curve.windows(2).find_map(|pair| {
        let left = pair[0];
        let right = pair[1];
        if !(left.0 < frequency_hz && frequency_hz < right.0) {
            return None;
        }
        let fraction = (frequency_hz.ln() - left.0.ln()) / (right.0.ln() - left.0.ln());
        Some(left.1 + fraction * (right.1 - left.1))
    })
}

/// Extract centre frequency, peak gain, -3 dB crossings, and Q.
pub fn bandpass_features(curve: &[(f64, f64)]) -> Option<BandpassFeatures> {
    let index = peak_index(curve)?;
    if index == 0 || index + 1 == curve.len() {
        return None;
    }
    let (peak_hz, peak_db) = interpolated_peak(curve)?;
    let level = peak_db - 3.010_299_956_639_812;
    let lower = crossing(curve, index, level, false)?;
    let upper = crossing(curve, index, level, true)?;
    let bandwidth = upper - lower;
    if !(bandwidth.is_finite() && bandwidth > 0.0) {
        return None;
    }
    Some(BandpassFeatures {
        peak_hz,
        peak_db,
        lower_3db_hz: lower,
        upper_3db_hz: upper,
        q: peak_hz / bandwidth,
    })
}

/// Extract cutoff and pass-band shape from a low-pass gain curve.
pub fn lowpass_features(curve: &[(f64, f64)], passband_stop_hz: f64) -> Option<LowpassFeatures> {
    if !valid_curve(curve)
        || !passband_stop_hz.is_finite()
        || passband_stop_hz <= curve[0].0
        || passband_stop_hz > curve.last()?.0
    {
        return None;
    }
    let dc_db = curve[0].1;
    let level = dc_db - 3.010_299_956_639_812;
    let cutoff_hz = crossing(curve, 0, level, true)?;
    let mut passband: Vec<(f64, f64)> = curve
        .iter()
        .take_while(|(frequency, _)| *frequency < passband_stop_hz)
        .copied()
        .collect();
    passband.push((passband_stop_hz, gain_at(curve, passband_stop_hz)?));
    if passband.len() < 2 {
        return None;
    }
    let minimum = interpolated_extreme(&passband, false)?.1;
    let maximum = interpolated_extreme(&passband, true)?.1;
    let overall_peak = interpolated_peak(curve)?.1;
    Some(LowpassFeatures {
        cutoff_hz,
        passband_ripple_db: maximum - minimum,
        peak_above_dc_db: overall_peak - dc_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::{mfb_bandpass, rc_lowpass};
    use crate::{PUBLICATION_MO_POINTS, PUBLICATION_SO_POINTS};

    #[test]
    fn analytic_rc_cutoff_is_recovered_within_half_percent() {
        let circuit = rc_lowpass(1_000.0, 159.154_943_091_895_35e-9);
        let curve = gain_curve(&circuit, "out", 10.0, 100_000.0, 81).unwrap();
        let extracted = lowpass_features(&curve, 500.0).unwrap();
        assert!((extracted.cutoff_hz / 1_000.0 - 1.0).abs() < 0.005);
    }

    #[test]
    fn interpolation_is_smooth_while_argmax_is_quantized() {
        let base = [2_500.0, 10_000.0, 8_200.0, 2.2e-9, 2.2e-9];
        let mut grid_peaks = Vec::new();
        let mut smooth_peaks = Vec::new();
        for step in 0..24 {
            let mut values = base;
            values[0] *= 0.954 + 0.004 * step as f64;
            let curve = gain_curve(&mfb_bandpass(&values), "out", 316.227, 316_227.0, 41).unwrap();
            let index = peak_index(&curve).unwrap();
            grid_peaks.push(curve[index].0);
            smooth_peaks.push(interpolated_peak(&curve).unwrap().0);
        }
        let distinct_grid = grid_peaks
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count();
        let distinct_smooth = smooth_peaks
            .windows(2)
            .filter(|pair| (pair[0] - pair[1]).abs() > 1e-8)
            .count();
        assert!(distinct_grid < distinct_smooth);
        assert_eq!(distinct_smooth, 23);
    }

    #[test]
    fn publication_bandpass_grid_converges_for_peak_and_q() {
        let values = [2_500.0, 10_000.0, 8_200.0, 2.2e-9, 2.2e-9];
        let circuit = mfb_bandpass(&values);
        let publication =
            gain_curve(&circuit, "out", 316.227, 316_227.0, PUBLICATION_SO_POINTS).unwrap();
        let reference = gain_curve(&circuit, "out", 316.227, 316_227.0, 801).unwrap();
        let publication_features = bandpass_features(&publication).unwrap();
        let reference_features = bandpass_features(&reference).unwrap();
        assert!((publication_features.peak_hz / reference_features.peak_hz - 1.0).abs() < 0.001);
        assert!((publication_features.q / reference_features.q - 1.0).abs() < 0.005);
    }

    #[test]
    fn publication_lowpass_grid_converges_for_cutoff_and_ripple() {
        let values = [
            1_910.509_197_440_457,
            102.947_740_312_304_29,
            9_022.762_971_635_404,
            2_774.396_437_672_846,
            101.410_467_309_628_02e-12,
            911.731_479_218_979_3e-12,
            973.275_963_408_539_3e-12,
            122.888_085_154_103_26e-12,
        ];
        let circuit = crate::netlist::sallen_key_lowpass4(&values);
        let publication = gain_curve(
            &circuit,
            "out",
            1_000.0,
            10_000_000.0,
            PUBLICATION_MO_POINTS,
        )
        .unwrap();
        let reference = gain_curve(&circuit, "out", 1_000.0, 10_000_000.0, 801).unwrap();
        let publication_features = lowpass_features(&publication, 80_000.0).unwrap();
        let reference_features = lowpass_features(&reference, 80_000.0).unwrap();
        assert!(
            (publication_features.cutoff_hz / reference_features.cutoff_hz - 1.0).abs() < 0.001
        );
        assert!(
            (publication_features.passband_ripple_db - reference_features.passband_ripple_db).abs()
                < 0.01
        );
    }

    #[test]
    fn monotone_curve_returns_endpoint_without_panicking() {
        let curve = vec![(1.0, 0.0), (10.0, -1.0), (100.0, -2.0)];
        assert_eq!(interpolated_peak(&curve), Some((1.0, 0.0)));
        assert!(bandpass_features(&curve).is_none());
    }
}
