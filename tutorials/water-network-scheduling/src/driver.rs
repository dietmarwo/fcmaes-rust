//! Stepwise extended-period hydraulic driver and control precedence.

use std::error::Error;

use epanet_rs::model::link::{LinkStatus, LinkType};
use epanet_rs::model::network::Network;
use epanet_rs::model::node::NodeType;
use epanet_rs::simulation::Simulation;
use serde::Serialize;

use crate::decode::{ControlPlan, Priority};
use crate::energy::{interval_energy_kwh, pump_power_kw};
use crate::scenarios::{AnalysisType, Scenario, tariff};

const M_PER_FT: f64 = 0.3048;
const M3_PER_CFS: f64 = 0.028_316_846_592;

/// Applied pump controls for one step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AppliedControl {
    pub speed: [f64; 2],
    pub safety_override: bool,
}

/// One successful hydraulic step, in SI units.
#[derive(Clone, Debug, Serialize)]
pub struct StepRecord {
    pub time_s: usize,
    pub interval_s: usize,
    pub min_pressure_m: f64,
    pub max_pressure_m: f64,
    pub max_velocity_m_s: f64,
    pub tank_level_m: f64,
    pub pump_flow_m3_s: [f64; 2],
    pub pump_head_m: [f64; 2],
    pub pump_power_kw: [f64; 2],
    pub pump_speed: [f64; 2],
    pub safety_override: bool,
    pub requested_m3_s: f64,
    pub delivered_m3_s: f64,
    pub continuity_residual_m3_s: f64,
}

/// Complete trace, including typed failure accounting.
#[derive(Clone, Debug)]
pub struct Trace {
    pub scenario: &'static str,
    pub analysis: AnalysisType,
    pub steps: Vec<StepRecord>,
    /// Index of the first hydraulic step that failed.
    pub failed_at_step: Option<usize>,
    pub failure: Option<String>,
    pub starts: [usize; 2],
    pub energy_kwh: f64,
    pub energy_cost: f64,
    pub peak_kw_native: f64,
    pub peak_kw_hourly: f64,
    pub initial_tank_level_m: f64,
    pub final_tank_level_m: f64,
}

/// Apply the documented safety-override precedence without a solver.
#[must_use]
pub fn control_for(plan: &ControlPlan, time_s: usize, tank_level_m: f64) -> AppliedControl {
    let period = (time_s / 7_200).min(11);
    let mut speed = [plan.levels[0][period], plan.levels[1][period]];
    let mut safety_override = false;
    if tank_level_m >= plan.high_threshold_m {
        speed = [0.0, 0.0];
        safety_override = true;
    } else if tank_level_m <= plan.low_threshold_m {
        let priority = match plan.priority {
            Priority::Pump1 => 0,
            Priority::Pump2 => 1,
        };
        speed = [0.0, 0.0];
        speed[priority] = plan.levels[priority][period].max(0.8);
        safety_override = true;
    }
    AppliedControl {
        speed,
        safety_override,
    }
}

fn junction_net_demand(state_values: &[f64], network: &Network) -> f64 {
    network
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.node_type, NodeType::Junction(_)))
        // Negative junction demand is a physical inflow and must not vanish
        // silently from requested/delivered volume accounting.
        .map(|(index, _)| state_values[index])
        .sum::<f64>()
        * M3_PER_CFS
}

fn hourly_peak(steps: &[StepRecord]) -> f64 {
    (0..24)
        .map(|hour| {
            let start = hour * 3_600;
            let end = start + 3_600;
            steps
                .iter()
                .filter_map(|step| {
                    let overlap_start = start.max(step.time_s);
                    let overlap_end = end.min(step.time_s + step.interval_s);
                    (overlap_end > overlap_start).then(|| {
                        step.pump_power_kw.iter().sum::<f64>()
                            * (overlap_end - overlap_start) as f64
                            / 3_600.0
                    })
                })
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
}

/// Simulate one scenario with an independent solver instance.
pub fn simulate(
    base: &Network,
    plan: &ControlPlan,
    scenario: &Scenario,
    timestep_s: usize,
) -> Result<Trace, Box<dyn Error>> {
    let mut network = base.clone();
    scenario.configure(&mut network, timestep_s)?;
    let pump_indices = [network.link_map["PU1"], network.link_map["PU2"]];
    let prv_index = network.link_map["PRV1"];
    let tank_index = network.node_map["T1"];
    let mut simulation = Simulation::new(network);
    simulation.skip_timesteps = false;
    simulation.initialize_hydraulics()?;
    let initial_tank_level_m = {
        let state = simulation.state.as_ref().ok_or("solver state missing")?;
        (state.heads[tank_index] - simulation.network.nodes[tank_index].elevation) * M_PER_FT
    };
    let mut steps = Vec::new();
    let mut failed_at_step = None;
    let mut failure = None;
    let mut starts = [0_usize; 2];
    let mut prior_on = [false; 2];
    let mut energy_kwh = 0.0;
    let mut energy_cost = 0.0;
    loop {
        let time = simulation.time;
        let state = simulation.state.as_mut().ok_or("solver state missing")?;
        state.apply_patterns(&simulation.network, time);
        let requested_m3_s = junction_net_demand(&state.demands, &simulation.network);
        let tank_level_m =
            (state.heads[tank_index] - simulation.network.nodes[tank_index].elevation) * M_PER_FT;
        let applied = control_for(plan, time, tank_level_m);
        for pump in 0..2 {
            let outage = scenario.pump_outage == Some(pump);
            let on = applied.speed[pump] > 0.0 && !outage;
            if on && !prior_on[pump] {
                starts[pump] += 1;
            }
            prior_on[pump] = on;
            state.statuses[pump_indices[pump]] = if on {
                LinkStatus::Open
            } else {
                LinkStatus::Closed
            };
            state.settings[pump_indices[pump]] = if on { applied.speed[pump] } else { 0.0 };
        }
        state.statuses[prv_index] = LinkStatus::Active;
        state.settings[prv_index] = plan.prv_setpoint_m / M_PER_FT;
        if let Some(pipe) = scenario.pipe_outage {
            state.statuses[simulation.network.link_map[pipe]] = LinkStatus::Closed;
        }

        if let Err(error) = simulation.run_hydraulics() {
            failed_at_step = Some(steps.len());
            failure = Some(error.to_string());
            break;
        }
        let state = simulation.state.as_ref().ok_or("solver state missing")?;
        let mut min_pressure_m = f64::INFINITY;
        let mut max_pressure_m = f64::NEG_INFINITY;
        for (index, node) in simulation.network.nodes.iter().enumerate() {
            if matches!(node.node_type, NodeType::Junction(_)) {
                let pressure = (state.heads[index] - node.elevation) * M_PER_FT;
                min_pressure_m = min_pressure_m.min(pressure);
                max_pressure_m = max_pressure_m.max(pressure);
            }
        }
        let mut max_velocity_m_s: f64 = 0.0;
        for (index, link) in simulation.network.links.iter().enumerate() {
            if let LinkType::Pipe(pipe) = &link.link_type {
                let diameter_m = pipe.diameter * M_PER_FT;
                let area = std::f64::consts::PI * diameter_m.powi(2) / 4.0;
                max_velocity_m_s =
                    max_velocity_m_s.max(state.flows[index].abs() * M3_PER_CFS / area);
            }
        }
        let mut balances = state
            .demands
            .iter()
            .map(|demand| -*demand)
            .collect::<Vec<_>>();
        for (index, link) in simulation.network.links.iter().enumerate() {
            balances[link.start_node] -= state.flows[index];
            balances[link.end_node] += state.flows[index];
        }
        let continuity_residual_m3_s = simulation
            .network
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node.node_type, NodeType::Junction(_)))
            .map(|(index, _)| balances[index].abs() * M3_PER_CFS)
            .fold(0.0, f64::max);
        let mut pump_flow_m3_s = [0.0; 2];
        let mut pump_head_m = [0.0; 2];
        let mut pump_power = [0.0; 2];
        for pump in 0..2 {
            let index = pump_indices[pump];
            let link = &simulation.network.links[index];
            pump_flow_m3_s[pump] = state.flows[index].max(0.0) * M3_PER_CFS;
            pump_head_m[pump] =
                (state.heads[link.end_node] - state.heads[link.start_node]).max(0.0) * M_PER_FT;
            pump_power[pump] = pump_power_kw(pump, pump_flow_m3_s[pump], pump_head_m[pump]);
        }
        let interval_s = simulation
            .network
            .options
            .time_options
            .hydraulic_timestep
            .min(24 * 3_600 - time);
        let power_kw = pump_power.iter().sum::<f64>();
        let interval_kwh = interval_energy_kwh(power_kw, interval_s);
        energy_kwh += interval_kwh;
        energy_cost += interval_kwh * tariff(time / 3_600, scenario.tariff_shift_h);
        let delivered_m3_s = junction_net_demand(&state.demands, &simulation.network);
        steps.push(StepRecord {
            time_s: time,
            interval_s,
            min_pressure_m,
            max_pressure_m,
            max_velocity_m_s,
            tank_level_m: (state.heads[tank_index]
                - simulation.network.nodes[tank_index].elevation)
                * M_PER_FT,
            pump_flow_m3_s,
            pump_head_m,
            pump_power_kw: pump_power,
            pump_speed: applied.speed,
            safety_override: applied.safety_override,
            requested_m3_s,
            delivered_m3_s,
            continuity_residual_m3_s,
        });
        if simulation.next_hydraulic_timestep() == 0 {
            break;
        }
    }
    let final_tank_level_m = simulation
        .state
        .as_ref()
        .map_or(initial_tank_level_m, |state| {
            (state.heads[tank_index] - simulation.network.nodes[tank_index].elevation) * M_PER_FT
        });
    let peak_kw_native = steps
        .iter()
        .map(|step| step.pump_power_kw.iter().sum::<f64>())
        .fold(0.0, f64::max);
    Ok(Trace {
        scenario: scenario.name,
        analysis: scenario.analysis,
        peak_kw_hourly: hourly_peak(&steps),
        steps,
        failed_at_step,
        failure,
        starts,
        energy_kwh,
        energy_cost,
        peak_kw_native,
        initial_tank_level_m,
        final_tank_level_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{Priority, override_witness_plan, seed_controls};
    use crate::network;
    use crate::scenarios::training;

    #[test]
    fn precedence_truth_table() {
        let plan = ControlPlan {
            levels: [[0.9; 12], [0.8; 12]],
            low_threshold_m: 3.0,
            high_threshold_m: 8.0,
            prv_setpoint_m: 35.0,
            priority: Priority::Pump2,
        };
        assert_eq!(control_for(&plan, 0, 9.0).speed, [0.0, 0.0]);
        assert_eq!(control_for(&plan, 0, 2.0).speed, [0.0, 0.8]);
        assert_eq!(control_for(&plan, 0, 5.0).speed, [0.9, 0.8]);
    }

    #[test]
    fn nominal_run_converges_and_is_finite() {
        let base = network::load().unwrap();
        let plan = crate::decode::decode(&seed_controls()).unwrap();
        let trace = simulate(&base, &plan, &training()[0], 3_600).unwrap();
        assert_eq!(trace.failed_at_step, None, "{:?}", trace.failure);
        assert!(!trace.steps.is_empty());
        assert!(trace.energy_kwh.is_finite() && trace.energy_kwh >= 0.0);
        assert!(
            trace
                .steps
                .iter()
                .all(|step| step.continuity_residual_m3_s < 1e-6)
        );
        let replay = trace
            .steps
            .iter()
            .map(|step| step.pump_power_kw.iter().sum::<f64>() * step.interval_s as f64 / 3_600.0)
            .sum::<f64>();
        assert!((trace.energy_kwh - replay).abs() < 1e-10);
        let repeated = simulate(&base, &plan, &training()[0], 3_600).unwrap();
        assert_eq!(trace.starts, repeated.starts);
        assert_eq!(trace.energy_kwh, repeated.energy_kwh);
    }

    #[test]
    fn energy_integral_converges_with_timestep() {
        let base = network::load().unwrap();
        let plan = crate::decode::decode(&seed_controls()).unwrap();
        let coarse = simulate(&base, &plan, &training()[0], 3_600).unwrap();
        let fine = simulate(&base, &plan, &training()[0], 900).unwrap();
        assert!((coarse.energy_kwh - fine.energy_kwh).abs() / fine.energy_kwh < 0.01);
    }

    #[test]
    fn threshold_witness_exercises_the_override_in_a_hydraulic_run() {
        let base = network::load().unwrap();
        let trace = simulate(&base, &override_witness_plan(), &training()[0], 1_800).unwrap();
        assert_eq!(trace.failed_at_step, None);
        assert!(trace.steps.iter().any(|step| step.safety_override));
        assert!(
            trace
                .steps
                .windows(2)
                .any(|steps| steps[0].safety_override != steps[1].safety_override)
        );
    }
}
