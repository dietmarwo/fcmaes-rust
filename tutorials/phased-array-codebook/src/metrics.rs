//! Sub-grid pattern metrics for linear cuts and planar directivity.

use std::f64::consts::PI;

use num_complex::Complex64;

use crate::array::{AngleGrid, GridKind};

const HALF_POWER_DB: f64 = -3.010_299_956_639_812;
const FLOOR_DB: f64 = -300.0;

/// Optional angular sector on a signed linear cut.
#[derive(Clone, Copy, Debug)]
pub struct Sector {
    /// Lower signed angle in degrees.
    pub lower_deg: f64,
    /// Upper signed angle in degrees.
    pub upper_deg: f64,
}

/// Measured pattern properties.
#[derive(Clone, Debug)]
pub struct PatternMetrics {
    /// Peak field magnitude.
    pub peak_linear: f64,
    /// Signed peak angle in the x-z cut, or polar angle for a planar grid.
    pub peak_theta_deg: f64,
    /// Peak azimuth for a planar grid.
    pub peak_phi_deg: Option<f64>,
    /// Half-power beamwidth for a linear cut.
    pub hpbw_deg: f64,
    /// Peak sidelobe level relative to the main peak.
    pub psll_db: f64,
    /// Strongest level in the requested null sector.
    pub null_depth_db: Option<f64>,
    /// Excitation taper efficiency.
    pub taper_efficiency: f64,
    /// One-sided planar directivity.
    pub directivity_dbi: Option<f64>,
    /// Signed main-lobe angular bounds.
    pub main_lobe_bounds_deg: (f64, f64),
    /// True when no unique/resolvable main lobe exists.
    pub degenerate: bool,
}

fn relative_db(magnitude: f64, peak: f64) -> f64 {
    if magnitude <= 0.0 || peak <= 0.0 {
        FLOOR_DB
    } else {
        (20.0 * (magnitude / peak).log10()).max(FLOOR_DB)
    }
}

fn signed_theta_deg(u: f64) -> f64 {
    u.clamp(-1.0, 1.0).asin().to_degrees()
}

fn interpolate_x(x0: f64, y0: f64, x1: f64, y1: f64, target: f64) -> f64 {
    if (y1 - y0).abs() < 1.0e-15 {
        0.5 * (x0 + x1)
    } else {
        x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    }
}

fn crossing(
    values: &[f64],
    u: &[f64],
    peak: usize,
    bound: usize,
    target: f64,
    left: bool,
) -> Option<f64> {
    if left {
        for right in (bound + 1..=peak).rev() {
            let left_index = right - 1;
            if values[left_index] <= target && values[right] >= target {
                return Some(interpolate_x(
                    u[left_index],
                    values[left_index],
                    u[right],
                    values[right],
                    target,
                ));
            }
        }
    } else {
        for left_index in peak..bound {
            let right = left_index + 1;
            if values[left_index] >= target && values[right] <= target {
                return Some(interpolate_x(
                    u[left_index],
                    values[left_index],
                    u[right],
                    values[right],
                    target,
                ));
            }
        }
    }
    None
}

fn nearest_minimum(magnitudes: &[f64], peak: usize, left: bool) -> Option<usize> {
    if left {
        (1..peak).rev().find(|index| {
            magnitudes[*index] <= magnitudes[*index - 1]
                && magnitudes[*index] <= magnitudes[*index + 1]
        })
    } else {
        (peak + 1..magnitudes.len() - 1).find(|index| {
            magnitudes[*index] <= magnitudes[*index - 1]
                && magnitudes[*index] <= magnitudes[*index + 1]
        })
    }
}

fn nearest_angle_index(
    u: &[f64],
    target_deg: f64,
    range: std::ops::RangeInclusive<usize>,
) -> usize {
    range
        .min_by(|left, right| {
            (signed_theta_deg(u[*left]) - target_deg)
                .abs()
                .total_cmp(&(signed_theta_deg(u[*right]) - target_deg).abs())
        })
        .expect("fallback range is non-empty")
}

fn analyse_linear_impl(
    field: &[Complex64],
    grid: &AngleGrid,
    excitation: &[Complex64],
    sector: Option<Sector>,
    force_hpbw_fallback: bool,
) -> PatternMetrics {
    let u = grid
        .linear_u()
        .expect("analyse_linear requires a linear cut");
    if field.len() != u.len() || field.len() < 5 {
        return degenerate_metrics();
    }
    let magnitudes = field.iter().map(|value| value.norm()).collect::<Vec<_>>();
    let Some((peak_index, &peak)) = magnitudes
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
    else {
        return degenerate_metrics();
    };
    if !peak.is_finite() || peak <= 1.0e-14 {
        return degenerate_metrics();
    }
    let db = magnitudes
        .iter()
        .map(|magnitude| relative_db(*magnitude, peak))
        .collect::<Vec<_>>();

    let refined_u = if peak_index > 0 && peak_index + 1 < u.len() {
        let y0 = db[peak_index - 1];
        let y1 = db[peak_index];
        let y2 = db[peak_index + 1];
        let denominator = y0 - 2.0 * y1 + y2;
        if denominator.abs() > 1.0e-14 {
            let shift = (0.5 * (y0 - y2) / denominator).clamp(-1.0, 1.0);
            u[peak_index] + shift * (u[peak_index + 1] - u[peak_index])
        } else {
            u[peak_index]
        }
    } else {
        u[peak_index]
    };

    let left_half = crossing(&db, &u, peak_index, 0, HALF_POWER_DB, true);
    let right_half = crossing(&db, &u, peak_index, u.len() - 1, HALF_POWER_DB, false);
    let hpbw_deg = match (left_half, right_half) {
        (Some(left), Some(right)) => signed_theta_deg(right) - signed_theta_deg(left),
        _ => f64::NAN,
    };
    let left_null = (!force_hpbw_fallback)
        .then(|| nearest_minimum(&magnitudes, peak_index, true))
        .flatten();
    let right_null = (!force_hpbw_fallback)
        .then(|| nearest_minimum(&magnitudes, peak_index, false))
        .flatten();
    let (left_bound, right_bound, mut degenerate) =
        if let (Some(left), Some(right)) = (left_null, right_null) {
            (left, right, false)
        } else if hpbw_deg.is_finite() && hpbw_deg > 0.0 {
            // The fallback exclusion is 1.5 × the measured HPBW in total,
            // centered on the refined peak. Angle-based bounds adapt to
            // steering and to non-uniform/thinned aperture width.
            let peak_deg = signed_theta_deg(refined_u);
            let half_span_deg = 0.75 * hpbw_deg;
            (
                nearest_angle_index(&u, peak_deg - half_span_deg, 0..=peak_index),
                nearest_angle_index(&u, peak_deg + half_span_deg, peak_index..=u.len() - 1),
                true,
            )
        } else {
            (peak_index, peak_index, true)
        };
    if !hpbw_deg.is_finite() {
        degenerate = true;
    }
    let psll_db = db
        .iter()
        .enumerate()
        .filter(|(index, _)| *index < left_bound || *index > right_bound)
        .map(|(_, value)| *value)
        .fold(FLOOR_DB, f64::max);
    if psll_db > -1.0 {
        degenerate = true;
    }
    let null_depth_db = sector.map(|sector| {
        db.iter()
            .zip(&u)
            .filter(|(_, value)| {
                let theta = signed_theta_deg(**value);
                theta >= sector.lower_deg && theta <= sector.upper_deg
            })
            .map(|(value, _)| *value)
            .fold(FLOOR_DB, f64::max)
    });
    let amplitude_sum = excitation.iter().map(|weight| weight.norm()).sum::<f64>();
    let amplitude_energy = excitation.iter().map(Complex64::norm_sqr).sum::<f64>();
    let taper_efficiency = if amplitude_energy > 0.0 {
        amplitude_sum * amplitude_sum / (excitation.len() as f64 * amplitude_energy)
    } else {
        0.0
    };
    PatternMetrics {
        peak_linear: peak,
        peak_theta_deg: signed_theta_deg(refined_u),
        peak_phi_deg: None,
        hpbw_deg,
        psll_db,
        null_depth_db,
        taper_efficiency,
        directivity_dbi: None,
        main_lobe_bounds_deg: (
            signed_theta_deg(u[left_bound]),
            signed_theta_deg(u[right_bound]),
        ),
        degenerate,
    }
}

/// Analyse a signed uniform-`u` cut.
#[must_use]
pub fn analyse_linear(
    field: &[Complex64],
    grid: &AngleGrid,
    excitation: &[Complex64],
    sector: Option<Sector>,
) -> PatternMetrics {
    analyse_linear_impl(field, grid, excitation, sector, false)
}

/// Compute one-sided planar directivity from upper-hemisphere quadrature.
#[must_use]
pub fn planar_directivity(field: &[Complex64], grid: &AngleGrid) -> PatternMetrics {
    let GridKind::UpperHemisphere { .. } = grid.kind else {
        return degenerate_metrics();
    };
    if field.len() != grid.len() || field.is_empty() {
        return degenerate_metrics();
    }
    let Some((peak_index, peak)) = field
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.norm_sqr().total_cmp(&right.1.norm_sqr()))
    else {
        return degenerate_metrics();
    };
    let peak_power = peak.norm_sqr();
    let integral = field
        .iter()
        .zip(&grid.directions)
        .map(|(value, direction)| value.norm_sqr() * direction.solid_angle_weight)
        .sum::<f64>();
    if peak_power <= 0.0 || integral <= 0.0 || !integral.is_finite() {
        return degenerate_metrics();
    }
    let direction = grid.directions[peak_index];
    PatternMetrics {
        peak_linear: peak.norm(),
        peak_theta_deg: direction.theta_rad.to_degrees(),
        peak_phi_deg: Some(direction.phi_rad.to_degrees()),
        hpbw_deg: f64::NAN,
        psll_db: f64::NAN,
        null_depth_db: None,
        taper_efficiency: f64::NAN,
        directivity_dbi: Some(10.0 * (4.0 * PI * peak_power / integral).log10()),
        main_lobe_bounds_deg: (f64::NAN, f64::NAN),
        degenerate: false,
    }
}

pub(crate) fn degenerate_metrics() -> PatternMetrics {
    PatternMetrics {
        peak_linear: 0.0,
        peak_theta_deg: f64::NAN,
        peak_phi_deg: None,
        hpbw_deg: f64::NAN,
        psll_db: 0.0,
        null_depth_db: None,
        taper_efficiency: 0.0,
        directivity_dbi: None,
        main_lobe_bounds_deg: (f64::NAN, f64::NAN),
        degenerate: true,
    }
}

#[cfg(test)]
mod tests {
    use crate::array::{Array, ArrayLayout};
    use crate::kernel::field_direct;

    use super::*;

    const CHEB_30: [f64; 16] = [
        0.290_988_871_257_774_8,
        0.317_296_191_540_353_9,
        0.455_688_938_631_810_3,
        0.601_756_006_455_069_8,
        0.742_386_845_754_509_2,
        0.863_659_696_720_422_3,
        0.952_789_152_816_859_2,
        1.0,
        1.0,
        0.952_789_152_816_859_2,
        0.863_659_696_720_422_3,
        0.742_386_845_754_509_2,
        0.601_756_006_455_069_8,
        0.455_688_938_631_810_3,
        0.317_296_191_540_353_9,
        0.290_988_871_257_774_8,
    ];

    fn weights(values: &[f64]) -> Vec<Complex64> {
        values
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect()
    }

    #[test]
    fn uniform_ula_matches_textbook_psll_and_hpbw() {
        let array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(20_001);
        let excitation = weights(&[1.0; 16]);
        let field = field_direct(&array, &grid, &excitation).unwrap();
        let metrics = analyse_linear(&field, &grid, &excitation, None);
        assert!((metrics.psll_db + 13.26).abs() < 0.2, "{}", metrics.psll_db);
        assert!(
            (metrics.hpbw_deg - 6.35).abs() / 6.35 < 0.02,
            "{}",
            metrics.hpbw_deg
        );
        assert!(!metrics.degenerate);
    }

    #[test]
    fn chebyshev_known_answer_validates_main_lobe_exclusion() {
        let array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(20_001);
        let excitation = weights(&CHEB_30);
        let field = field_direct(&array, &grid, &excitation).unwrap();
        let metrics = analyse_linear(&field, &grid, &excitation, None);
        assert!((metrics.psll_db + 30.0).abs() < 0.1, "{}", metrics.psll_db);
    }

    #[test]
    fn interpolation_resolves_a_between_grid_peak() {
        let array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(721);
        let target_u = 0.237_41;
        let excitation = (0..16)
            .map(|index| Complex64::from_polar(1.0, -PI * index as f64 * target_u))
            .collect::<Vec<_>>();
        let field = field_direct(&array, &grid, &excitation).unwrap();
        let metrics = analyse_linear(&field, &grid, &excitation, None);
        let expected = target_u.asin().to_degrees();
        let grid_step_deg = (2.0 / 720.0) / (1.0 - target_u * target_u).sqrt() * 180.0 / PI;
        assert!((metrics.peak_theta_deg - expected).abs() < 0.1 * grid_step_deg);
    }

    #[test]
    fn measured_hpbw_fallback_preserves_thinned_aperture_psll() {
        let slots = [0, 1, 2, 4, 5, 7, 8, 10, 13, 15, 16, 18, 19, 21, 22, 23];
        let array = Array {
            positions: slots
                .iter()
                .map(|slot| [(*slot as f64 - 11.5) * 0.5, 0.0])
                .collect(),
            wavelength: 1.0,
            element_exponent: 0.0,
            layout: ArrayLayout::General,
        };
        let grid = AngleGrid::linear_cut(20_001);
        let excitation = weights(&CHEB_30);
        let field = field_direct(&array, &grid, &excitation).unwrap();
        let null_bounded = analyse_linear(&field, &grid, &excitation, None);
        let fallback = analyse_linear_impl(&field, &grid, &excitation, None, true);
        assert!(null_bounded.hpbw_deg < 5.0);
        assert!(
            (null_bounded.psll_db - fallback.psll_db).abs() < 1.0e-10,
            "{} {}",
            null_bounded.psll_db,
            fallback.psll_db
        );
    }

    #[test]
    fn degenerate_patterns_do_not_panic() {
        let array = Array::uniform_linear(16, 0.5, 1.0).with_element_exponent(0.0);
        let grid = AngleGrid::linear_cut(721);
        for excitation in [vec![Complex64::new(0.0, 0.0); 16], {
            let mut values = vec![Complex64::new(0.0, 0.0); 16];
            values[7] = Complex64::new(1.0, 0.0);
            values
        }] {
            let field = field_direct(&array, &grid, &excitation).unwrap();
            assert!(analyse_linear(&field, &grid, &excitation, None).degenerate);
        }
        let grating = Array::uniform_linear(16, 1.0, 1.0).with_element_exponent(0.0);
        let excitation = weights(&[1.0; 16]);
        let field = field_direct(&grating, &grid, &excitation).unwrap();
        assert!(analyse_linear(&field, &grid, &excitation, None).degenerate);
    }

    #[test]
    fn planar_directivity_converges_with_grid_resolution() {
        let array = Array::uniform_rectangular(8, 8, 0.5, 0.5, 1.0);
        let excitation = weights(&[1.0; 64]);
        let coarse_grid = AngleGrid::upper_hemisphere(90, 180);
        let fine_grid = AngleGrid::upper_hemisphere(180, 360);
        let coarse = planar_directivity(
            &field_direct(&array, &coarse_grid, &excitation).unwrap(),
            &coarse_grid,
        )
        .directivity_dbi
        .unwrap();
        let fine = planar_directivity(
            &field_direct(&array, &fine_grid, &excitation).unwrap(),
            &fine_grid,
        )
        .directivity_dbi
        .unwrap();
        assert!((coarse - fine).abs() < 0.05, "{coarse} {fine}");
        assert!(fine > 20.0 && fine < 25.0);
    }
}
