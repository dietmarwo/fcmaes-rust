//! Outer bilevel evaluation and robust scenario aggregation.

use serde::{Deserialize, Serialize};

use crate::capex::{CapexBreakdown, annualized_capex, lcoe};
use crate::config::Preset;
use crate::decode::{DIMENSION, OuterDesign, decode_outer};
use crate::dispatch::{DispatchConfig, DispatchResult, solve_dispatch};
use crate::profiles::Profile;
use crate::scenarios::{Scenario, holdout, training};

/// Required minimum grid self-sufficiency.
pub const MIN_SELF_SUFFICIENCY: f64 = 0.55;
/// Maximum permitted unserved electrical-energy fraction.
pub const MAX_UNSERVED_FRACTION: f64 = 1.0e-4;
/// Annual equivalent-full-cycle limit.
pub const MAX_ANNUAL_CYCLES: f64 = 365.0;
/// Calibrated positive LP failure constraint.
pub const LP_FAILURE_CONSTRAINT: f64 = 1.0;
/// Invalid scalar fitness.
pub const INVALID_OBJECTIVE: f64 = 1.0e6;

/// One named scenario replay.
#[derive(Clone, Debug)]
pub struct ScenarioEvaluation {
    /// Scenario name.
    pub name: &'static str,
    /// Annualization factor from sampled horizon to one year.
    pub annualization: f64,
    /// Proven-optimal inner dispatch.
    pub dispatch: DispatchResult,
    /// Annualized LCOE for this scenario.
    pub lcoe: f64,
    /// Fraction of load supplied without grid import.
    pub self_sufficiency: f64,
    /// Unserved electrical-energy fraction.
    pub unserved_fraction: f64,
    /// Annual equivalent battery cycles.
    pub annual_cycles: f64,
    /// Annual grid-import emissions.
    pub co2_kg: f64,
    /// Renewable energy available before curtailment over the sampled horizon.
    pub renewable_available_kwh: f64,
}

/// Robust outer evaluation.
#[derive(Clone, Debug)]
pub struct OuterEvaluation {
    /// Authoritative normalized controls.
    pub controls: Vec<f64>,
    /// Decoded architecture and capacities.
    pub design: OuterDesign,
    /// Annualized capital and fixed cost.
    pub capex: CapexBreakdown,
    /// Training scenario replays.
    pub scenarios: Vec<ScenarioEvaluation>,
    /// Mean training LCOE.
    pub mean_lcoe: f64,
    /// Worst training LCOE.
    pub worst_lcoe: f64,
    /// Minimum training self-sufficiency.
    pub min_self_sufficiency: f64,
    /// Maximum training unserved fraction.
    pub max_unserved_fraction: f64,
    /// Maximum annual cycles.
    pub max_annual_cycles: f64,
    /// Mean annual CO₂.
    pub mean_co2_kg: f64,
    /// Mean annual curtailment.
    pub mean_curtailed_kwh: f64,
    /// Total simplex pivots across scenario LPs.
    pub simplex_iterations: u64,
    /// Self-sufficiency constraint, feasible at `<= 0`.
    pub constraint_self_sufficiency: f64,
    /// Unserved-energy constraint, feasible at `<= 0`.
    pub constraint_unserved: f64,
    /// Battery-cycle constraint, feasible at `<= 0`.
    pub constraint_cycles: f64,
    /// LP-status constraint, feasible at `<= 0`.
    pub constraint_lp_status: f64,
    /// Penalized scalar objective.
    pub objective: f64,
}

/// Compact metrics used by pilot and QD.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Behavior {
    /// Daily battery throughput per installed kWh.
    pub throughput_per_kwh: f64,
    /// Peak import divided by installed grid capacity.
    pub peak_import_ratio: f64,
    /// Base-scenario self-sufficiency.
    pub self_sufficiency: f64,
    /// Base-scenario curtailment fraction.
    pub curtailed_fraction: f64,
    /// PV capacity divided by battery energy, with zero when no battery exists.
    pub pv_battery_ratio: f64,
}

/// Whether all explicit constraints pass.
#[must_use]
pub fn feasible(evaluation: &OuterEvaluation) -> bool {
    [
        evaluation.constraint_self_sufficiency,
        evaluation.constraint_unserved,
        evaluation.constraint_cycles,
        evaluation.constraint_lp_status,
    ]
    .iter()
    .all(|value| *value <= 0.0)
}

fn evaluate_profile(
    capex: CapexBreakdown,
    capacities: crate::dispatch::HubCapacities,
    profile: Profile,
    name: &'static str,
) -> Result<ScenarioEvaluation, String> {
    let dispatch = solve_dispatch(
        &capacities,
        &profile,
        &DispatchConfig {
            hydrogen_enabled: false,
            ..DispatchConfig::default()
        },
    )
    .map_err(|error| format!("{name}: {error}"))?;
    let horizon_hours = profile.dt_hours * profile.len() as f64;
    let factor = 8_760.0 / horizon_hours;
    let annual_operating_cost = factor * dispatch.operating_cost;
    let annual_served = factor * dispatch.served_energy_kwh;
    let annual_import = factor * dispatch.imported_kwh;
    let annual_cycles = factor * dispatch.realized_cycles;
    let load = profile.load_energy_kwh();
    let renewable_available_kwh = profile.dt_hours
        * profile
            .solar_cf
            .iter()
            .zip(&profile.wind_cf)
            .map(|(solar, wind)| solar * capacities.pv_kwp + wind * capacities.wind_kw)
            .sum::<f64>();
    let self_sufficiency = (1.0 - dispatch.imported_kwh / load.max(1.0)).clamp(0.0, 1.0);
    let unserved_fraction = dispatch.unserved_kwh / load.max(1.0);
    Ok(ScenarioEvaluation {
        name,
        annualization: factor,
        lcoe: lcoe(capex.annualized + annual_operating_cost, annual_served),
        self_sufficiency,
        unserved_fraction,
        annual_cycles,
        co2_kg: 0.32 * annual_import,
        renewable_available_kwh,
        dispatch,
    })
}

fn evaluate_one(
    design: &OuterDesign,
    capex: CapexBreakdown,
    scenario: &Scenario,
    preset: Preset,
) -> Result<ScenarioEvaluation, String> {
    evaluate_profile(
        capex,
        scenario.capacities(design.capacities),
        scenario.profile(preset),
        scenario.name,
    )
}

/// Evaluate a decoded design on one custom deterministic profile.
pub fn evaluate_custom_profile(
    controls: &[f64],
    profile: Profile,
    name: &'static str,
) -> Result<ScenarioEvaluation, String> {
    let design = decode_outer(controls, false).map_err(|error| error.to_string())?;
    evaluate_profile(annualized_capex(&design), design.capacities, profile, name)
}

/// Evaluate the five frozen training scenarios.
#[must_use]
pub fn evaluate_training(controls: &[f64], preset: Preset) -> Option<OuterEvaluation> {
    let design = decode_outer(controls, false).ok()?;
    let capex = annualized_capex(&design);
    let scenarios = training()
        .into_iter()
        .map(|scenario| evaluate_one(&design, capex, scenario, preset))
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    aggregate(controls, design, capex, scenarios)
}

fn aggregate(
    controls: &[f64],
    design: OuterDesign,
    capex: CapexBreakdown,
    scenarios: Vec<ScenarioEvaluation>,
) -> Option<OuterEvaluation> {
    let count = scenarios.len() as f64;
    let mean_lcoe = scenarios.iter().map(|row| row.lcoe).sum::<f64>() / count;
    let worst_lcoe = scenarios
        .iter()
        .map(|row| row.lcoe)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_self_sufficiency = scenarios
        .iter()
        .map(|row| row.self_sufficiency)
        .fold(f64::INFINITY, f64::min);
    let max_unserved_fraction = scenarios
        .iter()
        .map(|row| row.unserved_fraction)
        .fold(0.0, f64::max);
    let max_annual_cycles = scenarios
        .iter()
        .map(|row| row.annual_cycles)
        .fold(0.0, f64::max);
    let mean_co2_kg = scenarios.iter().map(|row| row.co2_kg).sum::<f64>() / count;
    let mean_curtailed_kwh = scenarios
        .iter()
        .map(|row| row.annualization * row.dispatch.curtailed_kwh)
        .sum::<f64>()
        / count;
    let simplex_iterations = scenarios
        .iter()
        .map(|row| row.dispatch.simplex_iterations)
        .sum();
    let constraint_self_sufficiency = MIN_SELF_SUFFICIENCY - min_self_sufficiency;
    let constraint_unserved = max_unserved_fraction - MAX_UNSERVED_FRACTION;
    let constraint_cycles = max_annual_cycles - MAX_ANNUAL_CYCLES;
    let constraint_lp_status = -1.0;
    let objective = mean_lcoe
        + 5.0 * constraint_self_sufficiency.max(0.0)
        + 50.0 * constraint_unserved.max(0.0)
        + constraint_cycles.max(0.0) / MAX_ANNUAL_CYCLES;
    objective.is_finite().then_some(OuterEvaluation {
        controls: controls.to_vec(),
        design,
        capex,
        scenarios,
        mean_lcoe,
        worst_lcoe,
        min_self_sufficiency,
        max_unserved_fraction,
        max_annual_cycles,
        mean_co2_kg,
        mean_curtailed_kwh,
        simplex_iterations,
        constraint_self_sufficiency,
        constraint_unserved,
        constraint_cycles,
        constraint_lp_status,
        objective,
    })
}

/// Evaluate structurally different holdout scenarios for a selected design.
pub fn evaluate_holdout(
    controls: &[f64],
    preset: Preset,
) -> Result<Vec<ScenarioEvaluation>, String> {
    let design = decode_outer(controls, false).map_err(|error| error.to_string())?;
    let capex = annualized_capex(&design);
    holdout()
        .into_iter()
        .map(|scenario| evaluate_one(&design, capex, scenario, preset))
        .collect()
}

/// Extract the three registered descriptor pairs.
#[must_use]
pub fn behavior(evaluation: &OuterEvaluation) -> Behavior {
    let base = &evaluation.scenarios[0];
    behavior_for_scenario(&evaluation.design, base)
}

/// Extract descriptors from one scenario replay and its effective capacities.
#[must_use]
pub fn behavior_for_scenario(design: &OuterDesign, scenario: &ScenarioEvaluation) -> Behavior {
    let capacity = design.capacities;
    let sampled_days = 365.0 / scenario.annualization;
    Behavior {
        throughput_per_kwh: if capacity.battery_kwh > 0.0 {
            scenario.dispatch.battery_throughput_kwh
                / capacity.battery_kwh
                / sampled_days.max(1.0 / 24.0)
        } else {
            0.0
        },
        peak_import_ratio: scenario.dispatch.peak_import_kw / capacity.grid_kw.max(1.0),
        self_sufficiency: scenario.self_sufficiency,
        curtailed_fraction: scenario.dispatch.curtailed_kwh
            / scenario.renewable_available_kwh.max(1.0),
        pv_battery_ratio: if capacity.battery_kwh > 0.0 {
            capacity.pv_kwp / capacity.battery_kwh
        } else {
            0.0
        },
    }
}

/// Feasible analytic sizing seed.
#[must_use]
pub fn analytic_seed() -> Vec<f64> {
    let log_coordinate =
        |value: f64, lower: f64, upper: f64| (value.ln() - lower.ln()) / (upper.ln() - lower.ln());
    vec![
        0.48,
        0.42,
        log_coordinate(4_000.0, 100.0, 20_000.0),
        log_coordinate(1_000.0, 50.0, 5_000.0),
        0.4,
        0.3,
        3.5 / 6.0,
        0.75,
        0.75,
        0.25,
    ]
}

const _: () = assert!(DIMENSION == 10);

#[cfg(test)]
mod tests {
    use fcmaes_core::parallel_batch;

    use super::*;

    #[test]
    fn analytic_seed_is_replayable_and_reports_work() {
        let evaluation = evaluate_training(&analytic_seed(), Preset::Smoke).unwrap();
        assert!(evaluation.simplex_iterations > 0);
        assert_eq!(evaluation.scenarios.len(), 5);
        assert!(evaluation.constraint_lp_status <= 0.0);
        assert!(evaluation.max_unserved_fraction <= MAX_UNSERVED_FRACTION + 1.0e-9);
    }

    #[test]
    fn every_named_scenario_changes_a_metric() {
        let training = evaluate_training(&analytic_seed(), Preset::Smoke).unwrap();
        let base = &training.scenarios[0];
        for scenario in training.scenarios.iter().skip(1) {
            assert!(
                (scenario.lcoe - base.lcoe).abs() > 1.0e-8
                    || (scenario.self_sufficiency - base.self_sufficiency).abs() > 1.0e-8
            );
        }
        let held = evaluate_holdout(&analytic_seed(), Preset::Smoke).unwrap();
        for scenario in held {
            assert!(
                (scenario.lcoe - base.lcoe).abs() > 1.0e-8
                    || (scenario.self_sufficiency - base.self_sufficiency).abs() > 1.0e-8
                    || scenario.name == "quarter_hour_replay"
            );
            assert!(scenario.dispatch.max_balance_residual_kw < 1.0e-6);
            assert!(scenario.dispatch.max_storage_residual_kwh < 1.0e-6);
        }
    }

    #[test]
    fn repeated_serial_and_parallel_scores_are_identical() {
        let controls = analytic_seed();
        let expected = evaluate_training(&controls, Preset::Smoke)
            .unwrap()
            .objective
            .to_bits();
        for _ in 0..20 {
            assert_eq!(
                evaluate_training(&controls, Preset::Smoke)
                    .unwrap()
                    .objective
                    .to_bits(),
                expected
            );
        }
        let candidates = vec![controls; 20];
        let parallel = parallel_batch(&candidates, 4, |candidate| {
            evaluate_training(candidate, Preset::Smoke)
                .unwrap()
                .objective
                .to_bits()
        });
        assert!(parallel.into_iter().all(|value| value == expected));
    }
}
