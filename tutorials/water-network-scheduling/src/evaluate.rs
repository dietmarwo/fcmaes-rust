//! Robust objectives and normalized constraints.

use std::error::Error;

use epanet_rs::model::network::Network;

use crate::decode::{ControlPlan, decode};
use crate::driver::{Trace, simulate};
use crate::scenarios::{AnalysisType, Scenario, training};

/// Per-scenario physical and economic metrics.
#[derive(Clone, Debug)]
pub struct ScenarioEvaluation {
    pub name: &'static str,
    pub analysis: AnalysisType,
    pub operating_cost: f64,
    pub energy_cost: f64,
    pub peak_charge: f64,
    pub switching_cost: f64,
    pub min_pressure_m: f64,
    pub max_pressure_m: f64,
    pub max_velocity_m_s: f64,
    pub tank_recovery_m: f64,
    pub unserved_fraction: Option<f64>,
    pub constraints: [f64; 7],
    pub failed: bool,
}

impl ScenarioEvaluation {
    #[must_use]
    pub fn violation(&self) -> f64 {
        self.constraints
            .iter()
            .copied()
            .map(|value| value.max(0.0))
            .sum()
    }
}

/// Robust aggregate over scenarios of one analysis type.
#[derive(Clone, Debug)]
pub struct RobustEvaluation {
    pub controls: Vec<f64>,
    pub plan: ControlPlan,
    pub scenarios: Vec<ScenarioEvaluation>,
    pub objective: f64,
    pub operating_cost: f64,
    pub violation: f64,
    pub feasible: bool,
    pub descriptors: [f64; 2],
}

/// Convert one trace to metrics. DDA and PDA fields remain distinct.
#[must_use]
pub fn summarize(trace: &Trace) -> ScenarioEvaluation {
    let min_pressure = trace
        .steps
        .iter()
        .map(|step| step.min_pressure_m)
        .fold(f64::INFINITY, f64::min);
    let max_pressure = trace
        .steps
        .iter()
        .map(|step| step.max_pressure_m)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_velocity = trace
        .steps
        .iter()
        .map(|step| step.max_velocity_m_s)
        .fold(0.0, f64::max);
    let min_tank = trace
        .steps
        .iter()
        .map(|step| step.tank_level_m)
        .fold(f64::INFINITY, f64::min);
    let max_tank = trace
        .steps
        .iter()
        .map(|step| step.tank_level_m)
        .fold(f64::NEG_INFINITY, f64::max);
    let requested = trace
        .steps
        .iter()
        .map(|step| step.requested_m3_s * step.interval_s as f64)
        .sum::<f64>();
    let delivered = trace
        .steps
        .iter()
        .map(|step| step.delivered_m3_s * step.interval_s as f64)
        .sum::<f64>();
    let unserved = (trace.analysis == AnalysisType::Pda && requested > 0.0)
        .then_some((1.0 - delivered / requested).clamp(0.0, 1.0));
    let peak_charge = 3.8 * trace.peak_kw_hourly;
    let switching_cost = 2.5 * trace.starts.iter().sum::<usize>() as f64;
    let operating_cost = trace.energy_cost + peak_charge + switching_cost;
    let failed = trace.failed_at_step.is_some() || trace.steps.is_empty();
    let pressure_constraint = if trace.analysis == AnalysisType::Dda {
        (20.0 - min_pressure) / 20.0
    } else {
        unserved.unwrap_or(1.0) / 0.01
    };
    let constraints = [
        pressure_constraint,
        (max_pressure - 70.0) / 20.0,
        ((1.0 - min_tank).max(max_tank - 10.0)) / 2.0,
        (trace.initial_tank_level_m - trace.final_tank_level_m - 0.25) / 2.0,
        (max_velocity - 2.5) / 1.0,
        (trace.starts.iter().sum::<usize>() as f64 - 8.0) / 8.0,
        if failed { 1.0 } else { -1.0 },
    ];
    ScenarioEvaluation {
        name: trace.scenario,
        analysis: trace.analysis,
        operating_cost,
        energy_cost: trace.energy_cost,
        peak_charge,
        switching_cost,
        min_pressure_m: min_pressure,
        max_pressure_m: max_pressure,
        max_velocity_m_s: max_velocity,
        tank_recovery_m: trace.final_tank_level_m - trace.initial_tank_level_m,
        unserved_fraction: unserved,
        constraints,
        failed,
    }
}

/// Evaluate a coordinate vector over a same-analysis scenario set.
pub fn evaluate_scenarios(
    controls: &[f64],
    base: &Network,
    scenarios: &[Scenario],
    timestep_s: usize,
) -> Result<RobustEvaluation, Box<dyn Error>> {
    if scenarios.is_empty() {
        return Err("scenario set is empty".into());
    }
    let analysis = scenarios[0].analysis;
    if scenarios
        .iter()
        .any(|scenario| scenario.analysis != analysis)
    {
        return Err("DDA and PDA objectives cannot be aggregated".into());
    }
    let plan = decode(controls)?;
    let mut evaluations = Vec::with_capacity(scenarios.len());
    let mut off_peak_energy = 0.0;
    let mut total_energy = 0.0;
    let mut turnover = 0.0;
    for scenario in scenarios {
        let trace = simulate(base, &plan, scenario, timestep_s)?;
        for step in &trace.steps {
            let energy = step.pump_power_kw.iter().sum::<f64>() * step.interval_s as f64 / 3_600.0;
            total_energy += energy;
            if step.time_s / 3_600 <= 5 {
                off_peak_energy += energy;
            }
        }
        turnover += trace
            .steps
            .windows(2)
            .map(|pair| (pair[1].tank_level_m - pair[0].tank_level_m).abs())
            .sum::<f64>();
        evaluations.push(summarize(&trace));
    }
    let operating_cost = evaluations
        .iter()
        .map(|evaluation| evaluation.operating_cost)
        .fold(0.0, f64::max);
    let violation = evaluations
        .iter()
        .map(ScenarioEvaluation::violation)
        .fold(0.0, f64::max);
    Ok(RobustEvaluation {
        controls: controls.to_vec(),
        plan,
        objective: operating_cost + 10_000.0 * violation,
        operating_cost,
        violation,
        feasible: violation <= 1e-10,
        scenarios: evaluations,
        descriptors: [
            if total_energy > 0.0 {
                off_peak_energy / total_energy
            } else {
                0.0
            },
            (turnover / scenarios.len() as f64 / 12.0).clamp(0.0, 1.0),
        ],
    })
}

/// Evaluate the six DDA training scenarios at one-hour hydraulics.
pub fn evaluate_training(
    controls: &[f64],
    base: &Network,
) -> Result<RobustEvaluation, Box<dyn Error>> {
    evaluate_scenarios(controls, base, &training(), 3_600)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::seed_controls;
    use crate::network;
    use crate::scenarios::{holdout, training};
    use fcmaes_core::{Rng, parallel_batch};

    #[test]
    fn robust_seed_is_finite() {
        let base = network::load().unwrap();
        let result = evaluate_training(&seed_controls(), &base).unwrap();
        assert!(result.objective.is_finite());
        assert_eq!(result.scenarios.len(), 6);
    }

    #[test]
    fn mixed_analysis_aggregation_is_rejected() {
        let base = network::load().unwrap();
        let mut scenarios = vec![training()[0].clone(), holdout()[1].clone()];
        assert!(evaluate_scenarios(&seed_controls(), &base, &scenarios, 3_600).is_err());
        scenarios.clear();
    }

    #[test]
    fn all_pumps_off_is_finite_and_constraint_violating_or_overridden() {
        let base = network::load().unwrap();
        let result = evaluate_training(&[0.0; crate::DIMENSION], &base).unwrap();
        assert!(result.objective.is_finite());
        assert!(result.violation >= 0.0);
    }

    #[test]
    fn serial_and_parallel_candidates_are_bit_identical() {
        let base = network::load().unwrap();
        let mut rng = Rng::new(42);
        let candidates = (0..8)
            .map(|_| {
                (0..crate::DIMENSION)
                    .map(|_| rng.uniform01())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let serial = parallel_batch(&candidates, 1, |controls| {
            evaluate_training(controls, &base).unwrap().objective
        });
        let parallel = parallel_batch(&candidates, 2, |controls| {
            evaluate_training(controls, &base).unwrap().objective
        });
        assert_eq!(serial, parallel);
    }

    #[test]
    fn robust_scalar_aggregation_is_scenario_order_independent() {
        let base = network::load().unwrap();
        let controls = seed_controls();
        let forward = training();
        let mut reverse = forward.clone();
        reverse.reverse();
        let left = evaluate_scenarios(&controls, &base, &forward, 3_600).unwrap();
        let right = evaluate_scenarios(&controls, &base, &reverse, 3_600).unwrap();
        assert_eq!(left.objective, right.objective);
        assert_eq!(left.operating_cost, right.operating_cost);
        assert_eq!(left.violation, right.violation);
    }

    #[test]
    fn pda_outage_reports_unserved_demand() {
        let base = network::load().unwrap();
        let outage = holdout()[1].clone();
        let result = evaluate_scenarios(
            &seed_controls(),
            &base,
            std::slice::from_ref(&outage),
            3_600,
        )
        .unwrap();
        let unserved = result.scenarios[0].unserved_fraction.unwrap();
        assert!(unserved > 0.0 && unserved < 1.0);
        assert_eq!(result.scenarios[0].analysis, AnalysisType::Pda);
    }
}
