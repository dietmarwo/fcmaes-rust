//! Inner linear-program dispatch model.

use microlp::{
    ComparisonOp, OptimizationDirection, Problem, SolutionStatus, SolveOutcome, TerminationReason,
    Variable,
};
use serde::{Deserialize, Serialize};

use crate::config::{CURTAILMENT_COST, ELECTRICITY_VOLL};
use crate::profiles::Profile;

/// Small per-kWh cost on charge, discharge, and electrolysis flows.
///
/// This selects a reproducible optimum when the physical objective is
/// degenerate. It is never applied to curtailment.
pub const FLOW_TIE_BREAK: f64 = 1.0e-9;

/// Fixed capacities supplied by the outer sizing problem.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct HubCapacities {
    /// Installed PV capacity.
    pub pv_kwp: f64,
    /// Installed wind capacity.
    pub wind_kw: f64,
    /// Battery energy capacity.
    pub battery_kwh: f64,
    /// Battery charge/discharge power.
    pub battery_kw: f64,
    /// Electrolyser electrical input power.
    pub electrolyser_kw: f64,
    /// Hydrogen-store energy capacity.
    pub hydrogen_kwh: f64,
    /// Symmetric electricity import/export connection.
    pub grid_kw: f64,
}

/// Physical dispatch constants.
#[derive(Clone, Copy, Debug)]
pub struct DispatchConfig {
    /// Battery charge efficiency.
    pub battery_charge_efficiency: f64,
    /// Battery discharge efficiency.
    pub battery_discharge_efficiency: f64,
    /// Electricity-to-hydrogen efficiency.
    pub electrolyser_efficiency: f64,
    /// Whether the annual hydrogen subsystem is enabled.
    pub hydrogen_enabled: bool,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            battery_charge_efficiency: 0.95,
            battery_discharge_efficiency: 0.95,
            electrolyser_efficiency: 0.68,
            hydrogen_enabled: false,
        }
    }
}

/// One trajectory retained for plotting and invariant checks.
#[derive(Clone, Debug, Default)]
pub struct DispatchTrace {
    /// Grid imports in kW.
    pub import_kw: Vec<f64>,
    /// Grid exports in kW.
    pub export_kw: Vec<f64>,
    /// Battery charge in kW.
    pub charge_kw: Vec<f64>,
    /// Battery discharge in kW.
    pub discharge_kw: Vec<f64>,
    /// Battery state in kWh.
    pub soc_kwh: Vec<f64>,
    /// Curtailed renewable power in kW.
    pub curtail_kw: Vec<f64>,
    /// Unserved electrical load in kW.
    pub unserved_kw: Vec<f64>,
    /// Electrolyser electrical input in kW.
    pub electrolyser_kw: Vec<f64>,
    /// Hydrogen-store state in kWh-H₂.
    pub hydrogen_kwh: Vec<f64>,
    /// Purchased hydrogen in kW-H₂.
    pub hydrogen_buy_kw: Vec<f64>,
}

/// Proven-optimal dispatch summary.
#[derive(Clone, Debug)]
pub struct DispatchResult {
    /// Inner LP operating cost.
    pub operating_cost: f64,
    /// Served electrical energy.
    pub served_energy_kwh: f64,
    /// Unserved electrical energy.
    pub unserved_kwh: f64,
    /// Curtailed renewable energy.
    pub curtailed_kwh: f64,
    /// Imported grid energy.
    pub imported_kwh: f64,
    /// Exported grid energy.
    pub exported_kwh: f64,
    /// Peak import.
    pub peak_import_kw: f64,
    /// Battery charge plus discharge energy.
    pub battery_throughput_kwh: f64,
    /// Equivalent full battery cycles.
    pub realized_cycles: f64,
    /// Electricity used by the electrolyser.
    pub electrolyser_energy_kwh: f64,
    /// Purchased hydrogen energy.
    pub purchased_hydrogen_kwh: f64,
    /// Seasonal hydrogen-store amplitude.
    pub hydrogen_amplitude_kwh: f64,
    /// Cumulative simplex pivots.
    pub simplex_iterations: u64,
    /// Maximum absolute electrical balance residual.
    pub max_balance_residual_kw: f64,
    /// Maximum storage-dynamics residual.
    pub max_storage_residual_kwh: f64,
    /// Full retained dispatch.
    pub trace: DispatchTrace,
}

/// Dispatch construction or solver failure.
#[derive(Clone, Debug)]
pub enum DispatchError {
    /// Invalid horizon or capacity.
    InvalidInput(String),
    /// Solver did not return a proven optimum.
    Solver(String),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Solver(message) => message.fmt(formatter),
        }
    }
}

impl std::error::Error for DispatchError {}

#[derive(Clone, Copy)]
struct StepVars {
    import: Variable,
    export: Variable,
    charge: Variable,
    discharge: Variable,
    soc: Variable,
    curtail: Variable,
    unserved: Variable,
    electrolyser: Option<Variable>,
    hydrogen: Option<Variable>,
    hydrogen_buy: Option<Variable>,
}

fn validate(
    capacities: HubCapacities,
    profile: &Profile,
    config: DispatchConfig,
) -> Result<(), DispatchError> {
    if profile.is_empty()
        || profile.cyclic_block_steps == 0
        || !profile.len().is_multiple_of(profile.cyclic_block_steps)
        || !profile.dt_hours.is_finite()
        || profile.dt_hours <= 0.0
    {
        return Err(DispatchError::InvalidInput(
            "invalid dispatch horizon or cyclic block".to_owned(),
        ));
    }
    let values = [
        capacities.pv_kwp,
        capacities.wind_kw,
        capacities.battery_kwh,
        capacities.battery_kw,
        capacities.electrolyser_kw,
        capacities.hydrogen_kwh,
        capacities.grid_kw,
        config.battery_charge_efficiency,
        config.battery_discharge_efficiency,
        config.electrolyser_efficiency,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || !(0.0..=1.0).contains(&config.battery_charge_efficiency)
        || !(0.0..=1.0).contains(&config.battery_discharge_efficiency)
        || !(0.0..=1.0).contains(&config.electrolyser_efficiency)
    {
        return Err(DispatchError::InvalidInput(
            "capacities and efficiencies must be finite and non-negative".to_owned(),
        ));
    }
    Ok(())
}

/// Build and solve one dispatch LP to proven optimality.
pub fn solve_dispatch(
    capacities: &HubCapacities,
    profile: &Profile,
    config: &DispatchConfig,
) -> Result<DispatchResult, DispatchError> {
    validate(*capacities, profile, *config)?;
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let dt = profile.dt_hours;
    let mut steps = Vec::with_capacity(profile.len());
    for timestep in 0..profile.len() {
        let renewable = profile.solar_cf[timestep] * capacities.pv_kwp
            + profile.wind_cf[timestep] * capacities.wind_kw;
        steps.push(StepVars {
            import: problem.add_var(
                dt * profile.import_price[timestep],
                (0.0, capacities.grid_kw),
            ),
            export: problem.add_var(
                -dt * profile.export_price[timestep],
                (0.0, capacities.grid_kw),
            ),
            charge: problem.add_var(dt * FLOW_TIE_BREAK, (0.0, capacities.battery_kw)),
            discharge: problem.add_var(dt * FLOW_TIE_BREAK, (0.0, capacities.battery_kw)),
            soc: problem.add_var(0.0, (0.0, capacities.battery_kwh)),
            curtail: problem.add_var(dt * CURTAILMENT_COST, (0.0, renewable)),
            unserved: problem.add_var(dt * ELECTRICITY_VOLL, (0.0, profile.load_kw[timestep])),
            electrolyser: config
                .hydrogen_enabled
                .then(|| problem.add_var(dt * FLOW_TIE_BREAK, (0.0, capacities.electrolyser_kw))),
            hydrogen: config
                .hydrogen_enabled
                .then(|| problem.add_var(0.0, (0.0, capacities.hydrogen_kwh))),
            hydrogen_buy: config.hydrogen_enabled.then(|| {
                problem.add_var(dt * profile.hydrogen_price[timestep], (0.0, f64::INFINITY))
            }),
        });
    }
    for timestep in 0..profile.len() {
        let current = steps[timestep];
        let block_start = timestep / profile.cyclic_block_steps * profile.cyclic_block_steps;
        let previous_index = if timestep == block_start {
            block_start + profile.cyclic_block_steps - 1
        } else {
            timestep - 1
        };
        let previous = steps[previous_index];
        let renewable = profile.solar_cf[timestep] * capacities.pv_kwp
            + profile.wind_cf[timestep] * capacities.wind_kw;
        let mut balance = vec![
            (current.import, 1.0),
            (current.export, -1.0),
            (current.charge, -1.0),
            (current.discharge, 1.0),
            (current.curtail, -1.0),
            (current.unserved, 1.0),
        ];
        if let Some(electrolyser) = current.electrolyser {
            balance.push((electrolyser, -1.0));
        }
        problem.add_constraint(
            balance,
            ComparisonOp::Eq,
            profile.load_kw[timestep] - renewable,
        );
        problem.add_constraint(
            [
                (current.soc, 1.0),
                (previous.soc, -1.0),
                (current.charge, -dt * config.battery_charge_efficiency),
                (
                    current.discharge,
                    dt / config.battery_discharge_efficiency.max(1.0e-12),
                ),
            ],
            ComparisonOp::Eq,
            0.0,
        );
        if config.hydrogen_enabled {
            let hydrogen = current.hydrogen.expect("annual step owns hydrogen state");
            let previous_hydrogen = previous
                .hydrogen
                .expect("annual previous step owns hydrogen state");
            problem.add_constraint(
                [
                    (hydrogen, 1.0),
                    (previous_hydrogen, -1.0),
                    (
                        current.electrolyser.expect("annual step owns electrolyser"),
                        -dt * config.electrolyser_efficiency,
                    ),
                    (
                        current
                            .hydrogen_buy
                            .expect("annual step owns hydrogen purchase"),
                        -dt,
                    ),
                ],
                ComparisonOp::Eq,
                -dt * profile.hydrogen_demand_kw[timestep],
            );
        }
    }

    let outcome = problem
        .solve()
        .map_err(|error| DispatchError::Solver(error.to_string()))?;
    let solution = match outcome {
        SolveOutcome::Solution(solution)
            if solution.status() == SolutionStatus::Optimal
                && solution.termination_reason() == TerminationReason::ProvenOptimal =>
        {
            solution
        }
        other => {
            return Err(DispatchError::Solver(format!(
                "dispatch did not reach proven optimum: {:?}",
                other.termination_reason()
            )));
        }
    };
    let mut trace = DispatchTrace::default();
    let mut max_balance_residual_kw = 0.0_f64;
    let mut max_storage_residual_kwh = 0.0_f64;
    for &step in &steps {
        trace.import_kw.push(solution.var_value(step.import));
        trace.export_kw.push(solution.var_value(step.export));
        trace.charge_kw.push(solution.var_value(step.charge));
        trace.discharge_kw.push(solution.var_value(step.discharge));
        trace.soc_kwh.push(solution.var_value(step.soc));
        trace.curtail_kw.push(solution.var_value(step.curtail));
        trace.unserved_kw.push(solution.var_value(step.unserved));
        trace.electrolyser_kw.push(
            step.electrolyser
                .map_or(0.0, |variable| solution.var_value(variable)),
        );
        trace.hydrogen_kwh.push(
            step.hydrogen
                .map_or(0.0, |variable| solution.var_value(variable)),
        );
        trace.hydrogen_buy_kw.push(
            step.hydrogen_buy
                .map_or(0.0, |variable| solution.var_value(variable)),
        );
    }
    for timestep in 0..profile.len() {
        let block_start = timestep / profile.cyclic_block_steps * profile.cyclic_block_steps;
        let previous = if timestep == block_start {
            block_start + profile.cyclic_block_steps - 1
        } else {
            timestep - 1
        };
        let renewable = profile.solar_cf[timestep] * capacities.pv_kwp
            + profile.wind_cf[timestep] * capacities.wind_kw;
        let balance = renewable - trace.curtail_kw[timestep]
            + trace.import_kw[timestep]
            + trace.discharge_kw[timestep]
            - profile.load_kw[timestep]
            - trace.export_kw[timestep]
            - trace.charge_kw[timestep]
            - trace.electrolyser_kw[timestep]
            + trace.unserved_kw[timestep];
        max_balance_residual_kw = max_balance_residual_kw.max(balance.abs());
        let battery_residual = trace.soc_kwh[timestep]
            - trace.soc_kwh[previous]
            - dt * (config.battery_charge_efficiency * trace.charge_kw[timestep]
                - trace.discharge_kw[timestep] / config.battery_discharge_efficiency.max(1.0e-12));
        max_storage_residual_kwh = max_storage_residual_kwh.max(battery_residual.abs());
        if config.hydrogen_enabled {
            let hydrogen_residual = trace.hydrogen_kwh[timestep]
                - trace.hydrogen_kwh[previous]
                - dt * (config.electrolyser_efficiency * trace.electrolyser_kw[timestep]
                    + trace.hydrogen_buy_kw[timestep]
                    - profile.hydrogen_demand_kw[timestep]);
            max_storage_residual_kwh = max_storage_residual_kwh.max(hydrogen_residual.abs());
        }
    }
    let sum_energy = |values: &[f64]| dt * values.iter().sum::<f64>();
    let imported_kwh = sum_energy(&trace.import_kw);
    let exported_kwh = sum_energy(&trace.export_kw);
    let unserved_kwh = sum_energy(&trace.unserved_kw);
    let curtailed_kwh = sum_energy(&trace.curtail_kw);
    let battery_throughput_kwh = sum_energy(&trace.charge_kw) + sum_energy(&trace.discharge_kw);
    let realized_cycles = if capacities.battery_kwh > 0.0 {
        battery_throughput_kwh / (2.0 * capacities.battery_kwh)
    } else {
        0.0
    };
    let hydrogen_min = trace
        .hydrogen_kwh
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hydrogen_max = trace
        .hydrogen_kwh
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(DispatchResult {
        operating_cost: solution.objective(),
        served_energy_kwh: profile.load_energy_kwh() - unserved_kwh,
        unserved_kwh,
        curtailed_kwh,
        imported_kwh,
        exported_kwh,
        peak_import_kw: trace.import_kw.iter().copied().fold(0.0, f64::max),
        battery_throughput_kwh,
        realized_cycles,
        electrolyser_energy_kwh: sum_energy(&trace.electrolyser_kw),
        purchased_hydrogen_kwh: sum_energy(&trace.hydrogen_buy_kw),
        hydrogen_amplitude_kwh: if config.hydrogen_enabled {
            hydrogen_max - hydrogen_min
        } else {
            0.0
        },
        simplex_iterations: solution.stats().lp_iterations,
        max_balance_residual_kw,
        max_storage_residual_kwh,
        trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::{ProfileModifiers, representative_days};

    fn two_hour(prices: [f64; 2]) -> Profile {
        Profile {
            dt_hours: 1.0,
            cyclic_block_steps: 2,
            solar_cf: vec![0.0; 2],
            wind_cf: vec![0.0; 2],
            load_kw: vec![1.0; 2],
            import_price: prices.to_vec(),
            export_price: vec![0.0; 2],
            hydrogen_demand_kw: vec![0.0; 2],
            hydrogen_price: vec![0.2; 2],
        }
    }

    #[test]
    fn two_step_arbitrage_has_analytic_optimum() {
        let capacities = HubCapacities {
            battery_kwh: 1.0,
            battery_kw: 1.0,
            grid_kw: 2.0,
            ..HubCapacities::default()
        };
        let config = DispatchConfig {
            battery_charge_efficiency: 1.0,
            battery_discharge_efficiency: 1.0,
            ..DispatchConfig::default()
        };
        let result = solve_dispatch(&capacities, &two_hour([0.1, 0.3]), &config).unwrap();
        assert!((result.operating_cost - (0.2 + 2.0 * FLOW_TIE_BREAK)).abs() < 1.0e-9);
        assert!(result.max_balance_residual_kw < 1.0e-9);
        assert!(result.max_storage_residual_kwh < 1.0e-9);
    }

    #[test]
    fn zero_capacity_serves_every_kwh_with_slack() {
        let profile = representative_days(1, ProfileModifiers::default());
        let result = solve_dispatch(
            &HubCapacities::default(),
            &profile,
            &DispatchConfig::default(),
        )
        .unwrap();
        assert!((result.unserved_kwh - profile.load_energy_kwh()).abs() < 1.0e-7);
        assert!(
            (result.operating_cost - ELECTRICITY_VOLL * profile.load_energy_kwh()).abs() < 1.0e-5
        );
    }

    #[test]
    fn more_capacity_never_increases_operating_cost() {
        let profile = representative_days(1, ProfileModifiers::default());
        for index in 0..200 {
            let scale = 0.05 + 1.15 * index as f64 / 199.0;
            let small = HubCapacities {
                pv_kwp: 500.0 * scale,
                wind_kw: 350.0 * scale,
                battery_kwh: 600.0 * scale,
                battery_kw: 250.0 * scale,
                grid_kw: 500.0 * scale,
                ..HubCapacities::default()
            };
            let large = HubCapacities {
                pv_kwp: small.pv_kwp + 100.0,
                wind_kw: small.wind_kw + 80.0,
                battery_kwh: small.battery_kwh + 120.0,
                battery_kw: small.battery_kw + 50.0,
                grid_kw: small.grid_kw + 100.0,
                ..HubCapacities::default()
            };
            let small_cost = solve_dispatch(&small, &profile, &DispatchConfig::default()).unwrap();
            let large_cost = solve_dispatch(&large, &profile, &DispatchConfig::default()).unwrap();
            assert!(large_cost.operating_cost <= small_cost.operating_cost + 1.0e-7);
        }
    }

    #[test]
    fn representative_dispatch_respects_bounds_and_daily_closure() {
        let profile = representative_days(4, ProfileModifiers::default());
        let capacities = HubCapacities {
            pv_kwp: 2_000.0,
            wind_kw: 1_000.0,
            battery_kwh: 2_500.0,
            battery_kw: 800.0,
            grid_kw: 1_800.0,
            ..HubCapacities::default()
        };
        let result = solve_dispatch(&capacities, &profile, &DispatchConfig::default()).unwrap();
        for step in 0..profile.len() {
            assert!(result.trace.import_kw[step] <= capacities.grid_kw + 1.0e-9);
            assert!(result.trace.export_kw[step] <= capacities.grid_kw + 1.0e-9);
            assert!(result.trace.charge_kw[step] <= capacities.battery_kw + 1.0e-9);
            assert!(result.trace.discharge_kw[step] <= capacities.battery_kw + 1.0e-9);
            assert!(result.trace.soc_kwh[step] <= capacities.battery_kwh + 1.0e-9);
            assert!(result.trace.soc_kwh[step] >= -1.0e-9);
        }
        assert!(result.max_balance_residual_kw < 1.0e-9);
        assert!(result.max_storage_residual_kwh < 1.0e-9);
    }

    #[test]
    fn flat_tariff_and_unit_efficiency_do_not_cycle() {
        let capacities = HubCapacities {
            battery_kwh: 1.0,
            battery_kw: 1.0,
            grid_kw: 2.0,
            ..HubCapacities::default()
        };
        let config = DispatchConfig {
            battery_charge_efficiency: 1.0,
            battery_discharge_efficiency: 1.0,
            ..DispatchConfig::default()
        };
        let result = solve_dispatch(&capacities, &two_hour([0.2, 0.2]), &config).unwrap();
        assert!(result.battery_throughput_kwh < 1.0e-8);
    }
}
