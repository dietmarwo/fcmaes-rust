//! Focused chronological annual electricity-and-hydrogen extension.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use fcmaes_core::{BiteParams, Rng, optimize_bite};

use crate::capex::{CapexBreakdown, annualized_capex, lcoe};
use crate::decode::{DIMENSION, OuterDesign, decode_outer};
use crate::dispatch::{DispatchConfig, DispatchResult, solve_dispatch};
use crate::evaluate::{MAX_UNSERVED_FRACTION, analytic_seed};
use crate::profiles::{ProfileModifiers, chronological_year};

/// Annual electricity self-sufficiency target.
pub const ANNUAL_MIN_SELF_SUFFICIENCY: f64 = 0.45;
const INVALID: f64 = 1.0e6;

/// One chronological annual evaluation.
#[derive(Clone, Debug)]
pub struct AnnualEvaluation {
    /// Normalized outer controls.
    pub controls: Vec<f64>,
    /// Decoded capacities and architecture.
    pub design: OuterDesign,
    /// Annualized capital and fixed cost.
    pub capex: CapexBreakdown,
    /// Proven-optimal chronological dispatch.
    pub dispatch: DispatchResult,
    /// Resolution of this replay.
    pub dt_hours: usize,
    /// Combined electricity-and-hydrogen delivered-energy cost.
    pub delivered_energy_cost: f64,
    /// Electricity self-sufficiency.
    pub electricity_self_sufficiency: f64,
    /// Electrical unserved fraction.
    pub unserved_fraction: f64,
    /// Hydrogen supplied on site rather than purchased.
    pub onsite_hydrogen_fraction: f64,
    /// Self-sufficiency constraint, feasible at `<= 0`.
    pub constraint_self_sufficiency: f64,
    /// Unserved-energy constraint, feasible at `<= 0`.
    pub constraint_unserved: f64,
    /// Penalized scalar objective.
    pub objective: f64,
}

/// Evaluate one annual candidate at hourly or six-hour resolution.
#[must_use]
pub fn evaluate_annual(controls: &[f64], dt_hours: usize) -> Option<AnnualEvaluation> {
    let design = decode_outer(controls, true).ok()?;
    let capex = annualized_capex(&design);
    let profile = chronological_year(dt_hours, ProfileModifiers::default());
    let dispatch = solve_dispatch(
        &design.capacities,
        &profile,
        &DispatchConfig {
            hydrogen_enabled: true,
            ..DispatchConfig::default()
        },
    )
    .ok()?;
    let electrical_load = profile.load_energy_kwh();
    let hydrogen_demand = profile.hydrogen_energy_kwh();
    let electricity_self_sufficiency =
        (1.0 - dispatch.imported_kwh / electrical_load.max(1.0)).clamp(0.0, 1.0);
    let unserved_fraction = dispatch.unserved_kwh / electrical_load.max(1.0);
    let onsite_hydrogen_fraction =
        (1.0 - dispatch.purchased_hydrogen_kwh / hydrogen_demand.max(1.0)).clamp(0.0, 1.0);
    let delivered_energy = dispatch.served_energy_kwh + hydrogen_demand;
    let delivered_energy_cost = lcoe(capex.annualized + dispatch.operating_cost, delivered_energy);
    let constraint_self_sufficiency = ANNUAL_MIN_SELF_SUFFICIENCY - electricity_self_sufficiency;
    let constraint_unserved = unserved_fraction - MAX_UNSERVED_FRACTION;
    let objective = delivered_energy_cost
        + 5.0 * constraint_self_sufficiency.max(0.0)
        + 50.0 * constraint_unserved.max(0.0);
    objective.is_finite().then_some(AnnualEvaluation {
        controls: controls.to_vec(),
        design,
        capex,
        dispatch,
        dt_hours,
        delivered_energy_cost,
        electricity_self_sufficiency,
        unserved_fraction,
        onsite_hydrogen_fraction,
        constraint_self_sufficiency,
        constraint_unserved,
        objective,
    })
}

/// Deterministic seasonal-storage seed.
#[must_use]
pub fn annual_seed() -> Vec<f64> {
    let mut controls = analytic_seed();
    controls[0] = 0.76;
    controls[1] = 0.72;
    controls[4] = 0.56;
    controls[5] = 0.32;
    controls[6] = 5.5 / 6.0;
    controls[7] = 0.75;
    controls[8] = 0.75;
    controls[9] = 0.75;
    controls
}

/// Annual optimizer settings.
#[derive(Clone, Copy, Debug)]
pub struct AnnualConfig {
    /// Coarse six-hour candidate budget.
    pub evaluations: u64,
    /// Root seed.
    pub seed: u64,
}

/// Coarse sizing plus one final hourly replay.
#[derive(Clone, Debug)]
pub struct AnnualResult {
    /// Requested coarse candidate calls.
    pub requested_evaluations: u64,
    /// Actual coarse candidate calls.
    pub actual_evaluations: u64,
    /// Coarse inner LP solves.
    pub lp_solves: u64,
    /// Deterministic coarse-selection replays outside the optimizer budget.
    pub selection_replays: u64,
    /// Independent hourly validation replays.
    pub validation_replays: u64,
    /// Cumulative simplex pivots across optimization, selection, and validation.
    pub simplex_iterations: u64,
    /// Coarse optimization wall duration.
    pub elapsed: Duration,
    /// Selected coarse six-hour evaluation.
    pub coarse: AnnualEvaluation,
    /// Independent 8,760-step hourly validation.
    pub hourly: AnnualEvaluation,
}

/// Size at six-hour resolution with BiteOpt and replay the selection hourly.
pub fn optimize_annual(config: &AnnualConfig) -> Result<AnnualResult, Box<dyn Error>> {
    if config.evaluations == 0 {
        return Err("annual evaluation budget must be positive".into());
    }
    let calls = Arc::new(AtomicU64::new(0));
    let pivots = Arc::new(AtomicU64::new(0));
    let objective_calls = Arc::clone(&calls);
    let objective_pivots = Arc::clone(&pivots);
    let objective = move |controls: &[f64]| {
        objective_calls.fetch_add(1, Ordering::Relaxed);
        evaluate_annual(controls, 6).map_or(INVALID, |evaluation| {
            objective_pivots.fetch_add(evaluation.dispatch.simplex_iterations, Ordering::Relaxed);
            evaluation.objective
        })
    };
    let mut guess = annual_seed();
    let mut rng = Rng::new(config.seed);
    for value in &mut guess {
        *value = (*value + 0.01 * (rng.uniform01() - 0.5)).clamp(0.0, 1.0);
    }
    let started = Instant::now();
    let optimized = optimize_bite(
        &objective,
        &[0.0; DIMENSION],
        &[1.0; DIMENSION],
        Some(&guess),
        &BiteParams {
            max_evaluations: config.evaluations,
            seed: config.seed,
            ..Default::default()
        },
        1,
    );
    let mut candidates = vec![annual_seed(), optimized.x];
    let mut purchased_only = annual_seed();
    purchased_only[4] = 0.0;
    purchased_only[5] = 0.0;
    purchased_only[9] = 0.25;
    candidates.push(purchased_only);
    let mut large_store = annual_seed();
    large_store[0] = 0.90;
    large_store[1] = 0.86;
    large_store[4] = 0.72;
    large_store[5] = 0.60;
    candidates.push(large_store);
    let candidate_replays = candidates
        .iter()
        .filter_map(|controls| evaluate_annual(controls, 6))
        .collect::<Vec<_>>();
    let selection_pivots = candidate_replays
        .iter()
        .map(|evaluation| evaluation.dispatch.simplex_iterations)
        .sum::<u64>();
    let coarse = candidate_replays
        .into_iter()
        .min_by(|left, right| left.objective.total_cmp(&right.objective))
        .ok_or("annual arm retained no valid coarse candidate")?;
    let hourly = evaluate_annual(&coarse.controls, 1)
        .ok_or("selected annual design failed the hourly replay")?;
    Ok(AnnualResult {
        requested_evaluations: config.evaluations,
        actual_evaluations: calls.load(Ordering::Relaxed),
        lp_solves: calls.load(Ordering::Relaxed) + 5,
        selection_replays: 4,
        validation_replays: 1,
        simplex_iterations: pivots.load(Ordering::Relaxed)
            + selection_pivots
            + hourly.dispatch.simplex_iterations,
        elapsed: started.elapsed(),
        coarse,
        hourly,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_annual_seed_closes_both_storage_balances() {
        let evaluation = evaluate_annual(&annual_seed(), 6).unwrap();
        assert!(evaluation.dispatch.max_balance_residual_kw < 1.0e-6);
        assert!(evaluation.dispatch.max_storage_residual_kwh < 1.0e-6);
        assert!(evaluation.dispatch.purchased_hydrogen_kwh >= 0.0);
        assert!(evaluation.dispatch.hydrogen_amplitude_kwh > 1.0);
        assert_eq!(evaluation.dt_hours, 6);
    }
}
