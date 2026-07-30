//! Measured convex baseline and delivered outer landscape.

use std::time::{Duration, Instant};

use fcmaes_core::Rng;

use crate::capex::{CapexBreakdown, annualized_capex, annualized_capex_linear_grid, lcoe};
use crate::config::Preset;
use crate::decode::{DIMENSION, GRID_TIERS_KW, OuterDesign, decode_outer};
use crate::dispatch::{DispatchConfig, HubCapacities, solve_dispatch};
use crate::profiles::{ProfileModifiers, representative_days};

/// One point on all four registered landscape curves.
#[derive(Clone, Copy, Debug)]
pub struct LandscapeRow {
    /// Straight-line coordinate.
    pub coordinate: f64,
    /// Convex total-cost baseline with linear grid CAPEX.
    pub convex_total_cost: f64,
    /// Total cost after grid-tier snapping.
    pub tiered_total_cost: f64,
    /// Ratio objective with linear grid CAPEX.
    pub ratio_lcoe: f64,
    /// Delivered ratio plus tiers and inclusion switches.
    pub delivered_lcoe: f64,
    /// Delivered tier.
    pub delivered_grid_tier: usize,
    /// Delivered architecture bit count.
    pub delivered_inclusions: usize,
}

/// Landscape curve and finite-difference diagnostics.
#[derive(Clone, Debug)]
pub struct LandscapeResult {
    /// Curve samples.
    pub rows: Vec<LandscapeRow>,
    /// Random finite-difference probes.
    pub derivative_probes: usize,
    /// Sign disagreements between two step sizes.
    pub derivative_disagreements: usize,
    /// Probes deliberately placed beside a tier or inclusion boundary.
    pub boundary_probes: usize,
    /// Boundary probes whose two derivative scales disagree in sign.
    pub boundary_disagreements: usize,
    /// LP solves used.
    pub lp_solves: usize,
    /// Simplex pivots used.
    pub simplex_iterations: u64,
    /// Wall time.
    pub elapsed: Duration,
}

fn interpolated_capacity(value: f64) -> HubCapacities {
    HubCapacities {
        pv_kwp: 500.0 + 3_500.0 * value,
        wind_kw: 300.0 + 1_700.0 * value,
        battery_kwh: 500.0 + 6_000.0 * value,
        battery_kw: 200.0 + 1_500.0 * value,
        grid_kw: 500.0 + 2_500.0 * value,
        ..HubCapacities::default()
    }
}

fn tier_for(grid_kw: f64) -> usize {
    GRID_TIERS_KW
        .iter()
        .position(|tier| *tier >= grid_kw)
        .unwrap_or(GRID_TIERS_KW.len() - 1)
}

fn design(capacities: HubCapacities, tier: usize) -> OuterDesign {
    OuterDesign {
        capacities,
        grid_tier: tier,
        include_wind: true,
        include_battery: true,
        include_hydrogen: false,
    }
}

fn annual_values(
    design: &OuterDesign,
    capex: CapexBreakdown,
    profile: &crate::profiles::Profile,
) -> Option<(f64, f64, u64)> {
    let dispatch = solve_dispatch(&design.capacities, profile, &DispatchConfig::default()).ok()?;
    let factor = 8_760.0 / (profile.len() as f64 * profile.dt_hours);
    let total = capex.annualized + factor * dispatch.operating_cost;
    let served = factor * dispatch.served_energy_kwh;
    Some((total, lcoe(total, served), dispatch.simplex_iterations))
}

fn delivered_controls(value: f64) -> [f64; DIMENSION] {
    [
        0.12 + 0.76 * value,
        0.18 + 0.70 * value,
        0.15 + 0.75 * value,
        0.22 + 0.68 * value,
        0.2,
        0.2,
        value,
        0.30 + 0.45 * value,
        0.25 + 0.55 * value,
        0.25,
    ]
}

fn delivered_value(controls: &[f64], profile: &crate::profiles::Profile) -> Option<(f64, u64)> {
    let design = decode_outer(controls, false).ok()?;
    let (_, ratio, iterations) = annual_values(&design, annualized_capex(&design), profile)?;
    Some((ratio, iterations))
}

/// Measure all landscape curves and finite-difference instability.
pub fn measure_landscape(preset: Preset, seed: u64) -> Result<LandscapeResult, String> {
    let points = match preset {
        Preset::Smoke => 41,
        Preset::Publication => 101,
    };
    let derivative_probes = match preset {
        Preset::Smoke => 20,
        Preset::Publication => 100,
    };
    let profile = representative_days(
        preset.protocol().representative_days,
        ProfileModifiers::default(),
    );
    let started = Instant::now();
    let mut simplex_iterations = 0_u64;
    let mut lp_solves = 0;
    let mut rows = Vec::with_capacity(points);
    for index in 0..points {
        let coordinate = index as f64 / (points - 1) as f64;
        let continuous_capacity = interpolated_capacity(coordinate);
        let continuous_tier = tier_for(continuous_capacity.grid_kw);
        let continuous = design(continuous_capacity, continuous_tier);
        let (convex_total_cost, ratio_lcoe, iterations) = annual_values(
            &continuous,
            annualized_capex_linear_grid(&continuous),
            &profile,
        )
        .ok_or("convex baseline dispatch failed")?;
        simplex_iterations += iterations;
        lp_solves += 1;

        let tier = tier_for(continuous_capacity.grid_kw);
        let mut snapped_capacity = continuous_capacity;
        snapped_capacity.grid_kw = GRID_TIERS_KW[tier];
        let snapped = design(snapped_capacity, tier);
        let (tiered_total_cost, _, iterations) =
            annual_values(&snapped, annualized_capex(&snapped), &profile)
                .ok_or("tiered dispatch failed")?;
        simplex_iterations += iterations;
        lp_solves += 1;

        let controls = delivered_controls(coordinate);
        let delivered = decode_outer(&controls, false).map_err(|error| error.to_string())?;
        let (delivered_lcoe, iterations) =
            delivered_value(&controls, &profile).ok_or("delivered landscape dispatch failed")?;
        simplex_iterations += iterations;
        lp_solves += 1;
        rows.push(LandscapeRow {
            coordinate,
            convex_total_cost,
            tiered_total_cost,
            ratio_lcoe,
            delivered_lcoe,
            delivered_grid_tier: delivered.grid_tier,
            delivered_inclusions: usize::from(delivered.include_wind)
                + usize::from(delivered.include_battery),
        });
    }

    let mut rng = Rng::new(seed);
    let boundary_probes = derivative_probes / 2;
    let mut derivative_disagreements = 0;
    let mut boundary_disagreements = 0;
    for probe in 0..derivative_probes {
        let mut controls = std::array::from_fn(|_| 0.05 + 0.9 * rng.uniform01());
        controls[9] = 0.25;
        let coordinate = if probe < boundary_probes {
            let coordinate = [6_usize, 7, 8][probe % 3];
            let boundary = if coordinate == 6 {
                (1 + (probe / 3) % 5) as f64 / 6.0
            } else {
                0.5
            };
            controls[coordinate] = boundary + 2.0e-4;
            coordinate
        } else {
            (rng.uniform01() * 6.0).floor() as usize % 6
        };
        let derivative = |step: f64, controls: &mut [f64; DIMENSION]| -> Option<(f64, u64)> {
            let center = controls[coordinate];
            controls[coordinate] = (center + step).min(1.0);
            let (upper, upper_iterations) = delivered_value(controls, &profile)?;
            controls[coordinate] = (center - step).max(0.0);
            let (lower, lower_iterations) = delivered_value(controls, &profile)?;
            controls[coordinate] = center;
            Some((
                (upper - lower) / (2.0 * step),
                upper_iterations + lower_iterations,
            ))
        };
        let (fine, fine_iterations) =
            derivative(1.0e-4, &mut controls).ok_or("fine derivative probe failed")?;
        let (coarse, coarse_iterations) =
            derivative(5.0e-3, &mut controls).ok_or("coarse derivative probe failed")?;
        simplex_iterations += fine_iterations + coarse_iterations;
        lp_solves += 4;
        let sign = |value: f64| {
            if value.abs() < 1.0e-10 {
                0_i8
            } else if value > 0.0 {
                1
            } else {
                -1
            }
        };
        let disagrees = sign(fine) != sign(coarse);
        derivative_disagreements += usize::from(disagrees);
        if probe < boundary_probes {
            boundary_disagreements += usize::from(disagrees);
        }
    }
    Ok(LandscapeResult {
        rows,
        derivative_probes,
        derivative_disagreements,
        boundary_probes,
        boundary_disagreements,
        lp_solves,
        simplex_iterations,
        elapsed: started.elapsed(),
    })
}

/// Scale-aware midpoint convexity gate for the smooth baseline.
#[must_use]
pub fn convexity_violation(rows: &[LandscapeRow]) -> f64 {
    rows.windows(3)
        .map(|window| {
            let chord = 0.5 * (window[0].convex_total_cost + window[2].convex_total_cost);
            (window[1].convex_total_cost - chord) / chord.abs().max(1.0)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_capex_total_cost_baseline_is_convex() {
        let result = measure_landscape(Preset::Smoke, 42).unwrap();
        assert!(convexity_violation(&result.rows) < 1.0e-8);
        assert!(result.simplex_iterations > 0);
        assert_eq!(result.lp_solves, 3 * 41 + 4 * 20);
    }
}
