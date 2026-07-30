//! Non-uniform thinned-array experiment.

use std::error::Error;
use std::f64::consts::TAU;
use std::time::{Duration, Instant};

use fcmaes_core::{BiteParams, RetryBounds, RetryConfig, RetryRunResult, optimize_bite, retry};
use num_complex::Complex64;

use crate::INVALID_COST;
use crate::array::{AngleGrid, Array, ArrayLayout};
use crate::config::Quantization;
use crate::decode::{Excitation, equal_width_code};
use crate::kernel::field_direct;
use crate::metrics::{PatternMetrics, analyse_linear};

/// Available lattice slots.
pub const SLOTS: usize = 24;
/// Exact selected element count.
pub const ACTIVE: usize = 16;
/// Keys + positions + phase + attenuation.
pub const DIMENSION: usize = 4 * SLOTS;
/// Minimum allowed physical spacing.
pub const MINIMUM_SPACING_LAMBDA: f64 = 0.25;

/// Decoded/evaluated non-uniform design.
#[derive(Clone, Debug)]
pub struct GeometryEvaluation {
    /// Optimizer controls.
    pub controls: Vec<f64>,
    /// Active geometry.
    pub array: Array,
    /// Active register state.
    pub excitation: Excitation,
    /// Cut metrics.
    pub metrics: PatternMetrics,
    /// `required-realized`, feasible at `≤0`.
    pub constraint_spacing: f64,
    /// Penalized objective.
    pub objective: f64,
}

fn selected_indices(keys: &[f64]) -> Option<Vec<usize>> {
    if keys.len() != SLOTS || keys.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut order = (0..SLOTS).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        keys[*left]
            .total_cmp(&keys[*right])
            .then_with(|| left.cmp(right))
    });
    order.truncate(ACTIVE);
    order.sort_unstable();
    Some(order)
}

/// Decode and evaluate one 24-slot thinned array.
#[must_use]
pub fn evaluate_geometry(controls: &[f64], points: usize) -> Option<GeometryEvaluation> {
    if controls.len() != DIMENSION
        || controls.iter().any(|value| !value.is_finite())
        || points < 101
    {
        return None;
    }
    let selected = selected_indices(&controls[..SLOTS])?;
    let quantization = Quantization::default();
    let phase_levels = 1_u32 << quantization.phase_bits;
    let attenuation_levels = 1_u32 << quantization.attenuator_bits;
    let center = (SLOTS - 1) as f64 / 2.0;
    let mut positions = Vec::with_capacity(ACTIVE);
    let mut weights = Vec::with_capacity(ACTIVE);
    let mut phase_codes = Vec::with_capacity(ACTIVE);
    let mut attenuator_codes = Vec::with_capacity(ACTIVE);
    for &slot in &selected {
        let perturbation = (controls[SLOTS + slot].clamp(0.0, 1.0) - 0.5) * 0.5;
        positions.push([(slot as f64 - center) * 0.5 + perturbation, 0.0]);
        let phase_code =
            equal_width_code(controls[2 * SLOTS + slot], quantization.phase_bits).ok()?;
        let attenuation_code =
            equal_width_code(controls[3 * SLOTS + slot], quantization.attenuator_bits).ok()?;
        phase_codes.push(phase_code);
        attenuator_codes.push(attenuation_code);
        weights.push(Complex64::from_polar(
            10_f64.powf(-f64::from(attenuation_code) * quantization.attenuator_step_db / 20.0),
            TAU * f64::from(phase_code) / f64::from(phase_levels),
        ));
    }
    debug_assert!(
        attenuator_codes
            .iter()
            .all(|code| *code < attenuation_levels)
    );
    let array = Array {
        positions,
        wavelength: 1.0,
        element_exponent: 0.0,
        layout: ArrayLayout::General,
    };
    let excitation = Excitation {
        weights,
        active: vec![true; ACTIVE],
        phase_codes,
        attenuator_codes,
    };
    let grid = AngleGrid::linear_cut(points);
    let field = field_direct(&array, &grid, &excitation.weights).ok()?;
    let metrics = analyse_linear(&field, &grid, &excitation.weights, None);
    let constraint_spacing = MINIMUM_SPACING_LAMBDA - array.minimum_spacing_lambda();
    let pointing = metrics.peak_theta_deg.abs() - 0.5;
    let objective = metrics.psll_db
        + 100.0 * constraint_spacing.max(0.0)
        + 50.0 * pointing.max(0.0)
        + if metrics.degenerate { 100.0 } else { 0.0 };
    objective.is_finite().then_some(GeometryEvaluation {
        controls: controls.to_vec(),
        array,
        excitation,
        metrics,
        constraint_spacing,
        objective,
    })
}

/// Feasible broadside seed on the 24-slot lattice.
#[must_use]
pub fn geometry_seed() -> Vec<f64> {
    let mut controls = vec![1.0; DIMENSION];
    for slot in 0..SLOTS {
        let selected = (4..20).contains(&slot);
        controls[slot] = if selected { 0.0 } else { 1.0 };
        controls[SLOTS + slot] = 0.5;
        controls[2 * SLOTS + slot] = 0.5 / 64.0;
        let local = slot.saturating_sub(4).min(15);
        let amplitude = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * local as f64 / 15.0).cos();
        let code = ((-20.0 * amplitude.log10()) / 0.5).round().clamp(0.0, 31.0);
        controls[3 * SLOTS + slot] = (code + 0.5) / 32.0;
    }
    controls
}

/// Geometry-arm settings.
#[derive(Clone, Debug)]
pub struct GeometryConfig {
    /// Total objective calls.
    pub evaluations: u64,
    /// Parallel retry count.
    pub retries: usize,
    /// Candidate workers.
    pub workers: usize,
    /// Root seed.
    pub seed: u64,
    /// Cut points.
    pub points: usize,
}

/// Geometry-arm outcome.
#[derive(Clone, Debug)]
pub struct GeometryResult {
    /// Requested calls.
    pub requested_evaluations: u64,
    /// Actual calls.
    pub actual_evaluations: u64,
    /// Wall duration.
    pub elapsed: Duration,
    /// Best replayed geometry.
    pub best: GeometryEvaluation,
}

/// Optimize the non-uniform geometry with BiteOpt retry.
pub fn optimize_geometry(config: &GeometryConfig) -> Result<GeometryResult, Box<dyn Error>> {
    if config.evaluations == 0 || config.retries == 0 {
        return Err("geometry budget and retries must be positive".into());
    }
    let bounds = RetryBounds::new(vec![0.0; DIMENSION], vec![1.0; DIMENSION])?;
    let objective = |controls: &[f64]| {
        evaluate_geometry(controls, config.points)
            .map(|evaluation| evaluation.objective)
            .unwrap_or(INVALID_COST)
    };
    let retry_config = RetryConfig {
        num_retries: config.retries,
        workers: config.workers,
        capacity: config.retries,
        max_evaluations: config.evaluations.div_ceil(config.retries as u64),
        seed: config.seed,
        ..Default::default()
    };
    let initial = geometry_seed();
    let started = Instant::now();
    let result = retry(&objective, &bounds, &retry_config, |objective, context| {
        let optimized = optimize_bite(
            objective,
            context.bounds.lower(),
            context.bounds.upper(),
            Some(&initial),
            &BiteParams {
                max_evaluations: context.max_evaluations,
                seed: context.seed,
                ..Default::default()
            },
            1,
        );
        RetryRunResult {
            x: optimized.x,
            y: optimized.y,
            evaluations: optimized.evaluations,
        }
    });
    if !result.success {
        return Err("geometry retry retained no result".into());
    }
    let best =
        evaluate_geometry(&result.x, config.points).ok_or("geometry result did not replay")?;
    Ok(GeometryResult {
        requested_evaluations: config.evaluations,
        actual_evaluations: result.evaluations,
        elapsed: started.elapsed(),
        best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_seed_is_feasible_and_nonuniform_kernel_is_explicit() {
        let evaluation = evaluate_geometry(&geometry_seed(), 721).unwrap();
        assert!(evaluation.constraint_spacing <= 0.0);
        assert_eq!(evaluation.excitation.weights.len(), ACTIVE);
        let uniform = {
            let array = Array::uniform_linear(ACTIVE, 0.5, 1.0);
            let grid = AngleGrid::linear_cut(721);
            let excitation = vec![Complex64::new(1.0, 0.0); ACTIVE];
            let field = field_direct(&array, &grid, &excitation).unwrap();
            analyse_linear(&field, &grid, &excitation, None)
        };
        assert!(evaluation.metrics.psll_db < uniform.psll_db);
        #[cfg(feature = "fft")]
        assert!(matches!(
            crate::kernel::FftKernel::linear(&evaluation.array, 256),
            Err(crate::kernel::KernelError::NonUniformSpacing)
        ));
    }

    #[test]
    fn tiny_geometry_retry_replays() {
        let result = optimize_geometry(&GeometryConfig {
            evaluations: 64,
            retries: 2,
            workers: 2,
            seed: 42,
            points: 361,
        })
        .unwrap();
        assert!(result.best.constraint_spacing <= 0.0);
        assert!(result.best.objective.is_finite());
    }
}
