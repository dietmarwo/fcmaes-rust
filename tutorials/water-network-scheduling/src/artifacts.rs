//! Versioned CSV and JSON artifact writers.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use epanet_rs::model::network::Network;
use serde_json::json;

use crate::bench::BenchmarkRow;
use crate::decode::seed_controls;
use crate::driver::{StepRecord, Trace, simulate};
use crate::energy::EnergyOracleCheck;
use crate::evaluate::{ScenarioEvaluation, evaluate_training};
use crate::mo::MoResult;
use crate::pilot::PilotSummary;
use crate::qd::QdResult;
use crate::scenarios::training;
use crate::so::SoResult;

/// Common artifact metadata.
pub struct Metadata<'a> {
    pub directory: &'a Path,
    pub command: &'a str,
    pub preset: &'a str,
    pub seed: u64,
    pub workers: i32,
}

/// Resolution-study observation.
#[derive(Clone, Debug)]
pub struct ResolutionRow {
    pub case: &'static str,
    pub timestep_s: usize,
    pub energy_kwh: f64,
    pub energy_cost: f64,
    pub peak_kw_hourly: f64,
    pub peak_kw_native: f64,
    pub starts: usize,
    pub min_pressure_m: f64,
    pub max_velocity_m_s: f64,
    pub failed_at_step: Option<usize>,
    pub override_steps: usize,
    pub wall_seconds: f64,
}

fn prepare(path: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(path)?;
    Ok(())
}

fn controls(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17}"))
        .collect::<Vec<_>>()
        .join(";")
}

/// Write named training and holdout scenario metrics.
pub fn write_scenarios(
    metadata: &Metadata<'_>,
    rows: &[(&str, ScenarioEvaluation)],
) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    let mut csv = String::from(
        "set,scenario,analysis_type,failed,operating_cost,energy_cost,peak_charge,switching_cost,min_pressure_m,max_pressure_m,max_velocity_m_s,tank_recovery_m,unserved_fraction,violation\n",
    );
    for (set, row) in rows {
        writeln!(
            csv,
            "{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            set,
            row.name,
            row.analysis.name(),
            row.failed,
            row.operating_cost,
            row.energy_cost,
            row.peak_charge,
            row.switching_cost,
            row.min_pressure_m,
            row.max_pressure_m,
            row.max_velocity_m_s,
            row.tank_recovery_m,
            row.unserved_fraction.unwrap_or(f64::NAN),
            row.violation()
        )?;
    }
    fs::write(metadata.directory.join("scenario_metrics.csv"), csv)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"water-network-scheduling",
            "formulation":"named-scenarios",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "dda_pda_objectives_aggregated":false,
            "artifacts":{"metrics":"scenario_metrics.csv"}
        }))?,
    )?;
    Ok(())
}

/// Write validation summary and nominal trace.
pub fn write_validation(
    metadata: &Metadata<'_>,
    trace: &Trace,
    override_trace: &Trace,
    energy_replay_kwh: f64,
    oracle: &[EnergyOracleCheck],
    pipe_relative_error: f64,
) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    let max_continuity = trace
        .steps
        .iter()
        .map(|step| step.continuity_residual_m3_s)
        .fold(0.0, f64::max);
    let energy_relative_error =
        (trace.energy_kwh - energy_replay_kwh).abs() / trace.energy_kwh.abs().max(1e-12);
    let oracle_max_relative_error = oracle
        .iter()
        .map(|check| check.relative_error)
        .fold(0.0_f64, f64::max);
    let override_steps = override_trace
        .steps
        .iter()
        .filter(|step| step.safety_override)
        .count();
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"water-network-scheduling",
            "formulation":"hydraulic-validation",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "analysis_type":trace.analysis.name(),
            "energy_source":"tutorial_flow_head_integration",
            "external_validation":false,
            "checks":{
                "failed_at_step":trace.failed_at_step,
                "max_continuity_residual_m3_s":max_continuity,
                "energy_accumulation_replay_relative_error":energy_relative_error,
                "energy_oracle_max_relative_error":oracle_max_relative_error,
                "override_witness_steps":override_steps,
                "analytic_pipe_relative_error":pipe_relative_error
            },
            "artifacts":{
                "trace":"trace.csv",
                "override_trace":"override_trace.csv",
                "energy_accumulation":"energy_crosscheck.csv",
                "energy_oracle":"energy_oracle_check.csv"
            }
        }))?,
    )?;

    fn trace_csv(steps: &[StepRecord]) -> Result<String, std::fmt::Error> {
        let mut csv = String::from(
            "time_s,interval_s,min_pressure_m,max_pressure_m,max_velocity_m_s,tank_level_m,pump1_flow_m3_s,pump2_flow_m3_s,pump1_power_kw,pump2_power_kw,safety_override,requested_m3_s,delivered_m3_s,continuity_residual_m3_s\n",
        );
        for step in steps {
            writeln!(
                csv,
                "{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},{:.17},{:.17},{:.17}",
                step.time_s,
                step.interval_s,
                step.min_pressure_m,
                step.max_pressure_m,
                step.max_velocity_m_s,
                step.tank_level_m,
                step.pump_flow_m3_s[0],
                step.pump_flow_m3_s[1],
                step.pump_power_kw[0],
                step.pump_power_kw[1],
                step.safety_override,
                step.requested_m3_s,
                step.delivered_m3_s,
                step.continuity_residual_m3_s
            )?;
        }
        Ok(csv)
    }
    fs::write(
        metadata.directory.join("trace.csv"),
        trace_csv(&trace.steps)?,
    )?;
    fs::write(
        metadata.directory.join("override_trace.csv"),
        trace_csv(&override_trace.steps)?,
    )?;
    fs::write(
        metadata.directory.join("energy_crosscheck.csv"),
        format!(
            "production_kwh,stored_power_replay_kwh,relative_error\n{:.17},{:.17},{:.17}\n",
            trace.energy_kwh, energy_replay_kwh, energy_relative_error
        ),
    )?;
    let mut oracle_csv = String::from(
        "pump,flow_m3_s,head_gain_m,expected_efficiency,observed_efficiency,expected_power_kw,observed_power_kw,relative_error\n",
    );
    for check in oracle {
        writeln!(
            oracle_csv,
            "{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            check.point.pump + 1,
            check.point.flow_m3_s,
            check.point.head_gain_m,
            check.point.expected_efficiency,
            check.observed_efficiency,
            check.point.expected_power_kw,
            check.observed_power_kw,
            check.relative_error
        )?;
    }
    fs::write(
        metadata.directory.join("energy_oracle_check.csv"),
        oracle_csv,
    )?;
    Ok(())
}

/// Write scalar comparison, convergence and replayable schedules/traces.
pub fn write_so(
    metadata: &Metadata<'_>,
    network: &Network,
    arms: &[SoResult],
) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    let baseline = evaluate_training(&seed_controls(), network)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"water-network-scheduling",
            "formulation":"robust-dda-so",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "analysis_type":"DDA",
            "hydraulic_timestep_s":3600,
            "billing_resolution_s":3600,
            "start_count_resolution_s":3600,
            "energy_source":"tutorial_flow_head_integration",
            "requested_evaluations":arms.iter().map(|arm| arm.requested_evaluations).sum::<u64>(),
            "actual_evaluations":1 + arms.iter().map(|arm| arm.actual_evaluations).sum::<u64>(),
            "baseline_evaluations":1,
            "artifacts":{"arms":"arms.csv","convergence":"convergence.csv","schedule":"schedule.csv","tank":"tank_trace.csv","pressure":"pressure_envelope.csv"}
        }))?,
    )?;
    let mut arm_csv = String::from(
        "arm,feasible,objective,operating_cost,violation,actual_evaluations,elapsed_seconds\n",
    );
    let mut convergence = String::from("arm,evaluations,elapsed_seconds,best_objective\n");
    let mut schedule = String::from("arm,pump,period,start_hour,relative_speed,controls\n");
    let mut tank = String::from("arm,time_s,tank_level_m,safety_override\n");
    let mut pressure = String::from("arm,time_s,min_pressure_m,max_pressure_m,max_velocity_m_s\n");
    writeln!(
        arm_csv,
        "seed,{},{:.17},{:.17},{:.17},1,0.00000000000000000",
        baseline.feasible, baseline.objective, baseline.operating_cost, baseline.violation
    )?;
    for arm in arms {
        writeln!(
            arm_csv,
            "{},{},{:.17},{:.17},{:.17},{},{:.17}",
            arm.optimizer.name(),
            arm.best.feasible,
            arm.best.objective,
            arm.best.operating_cost,
            arm.best.violation,
            arm.actual_evaluations,
            arm.elapsed.as_secs_f64()
        )?;
        for improvement in &arm.improvements {
            writeln!(
                convergence,
                "{},{},{:.17},{:.17}",
                arm.optimizer.name(),
                improvement.evaluations,
                improvement.elapsed_seconds,
                improvement.value
            )?;
        }
        for pump in 0..2 {
            for period in 0..12 {
                writeln!(
                    schedule,
                    "{},{},{},{},{:.17},\"{}\"",
                    arm.optimizer.name(),
                    pump + 1,
                    period,
                    period * 2,
                    arm.best.plan.levels[pump][period],
                    controls(&arm.best.controls)
                )?;
            }
        }
        let trace = simulate(network, &arm.best.plan, &training()[0], 3_600)?;
        for step in trace.steps {
            writeln!(
                tank,
                "{},{},{:.17},{}",
                arm.optimizer.name(),
                step.time_s,
                step.tank_level_m,
                step.safety_override
            )?;
            writeln!(
                pressure,
                "{},{},{:.17},{:.17},{:.17}",
                arm.optimizer.name(),
                step.time_s,
                step.min_pressure_m,
                step.max_pressure_m,
                step.max_velocity_m_s
            )?;
        }
    }
    fs::write(metadata.directory.join("arms.csv"), arm_csv)?;
    fs::write(metadata.directory.join("convergence.csv"), convergence)?;
    fs::write(metadata.directory.join("schedule.csv"), schedule)?;
    fs::write(metadata.directory.join("tank_trace.csv"), tank)?;
    fs::write(metadata.directory.join("pressure_envelope.csv"), pressure)?;
    Ok(())
}

/// Write descriptor-gate evidence.
pub fn write_pilot(metadata: &Metadata<'_>, pilot: &PilotSummary) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"water-network-scheduling",
            "formulation":"descriptor-pilot",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "attempted":pilot.attempted,
            "feasible_candidates":pilot.rows.len(),
            "qd_decision":pilot.decision.label(),
            "archive":{
                "capacity":pilot.archive_capacity,
                "row_lengths":pilot.archive_row_lengths
            },
            "pairs":{
                "D1":pilot.d1,
                "D2":pilot.d2,
                "D3":pilot.d3
            },
            "artifacts":{"samples":"pilot.csv"}
        }))?,
    )?;
    let mut csv = String::from(
        "seed,sample,operating_cost,d1_axis1_train,d1_axis2_train,d1_axis1_holdout,d1_axis2_holdout,d1_axis1_hourly,d1_axis2_hourly,d1_axis1_half_hour,d1_axis2_half_hour,d2_axis1_train,d2_axis2_train,d2_axis1_holdout,d2_axis2_holdout,d2_axis1_hourly,d2_axis2_hourly,d2_axis1_half_hour,d2_axis2_half_hour,d3_axis1_train,d3_axis2_train,d3_axis1_holdout,d3_axis2_holdout\n",
    );
    for row in &pilot.rows {
        let d1 = row.training_pair("D1");
        let d1_holdout = row.holdout_pair("D1");
        let d1_hourly = row.resolution_baseline_pair("D1");
        let d1_half_hour = row.resolution_fine_pair("D1");
        let d2 = row.training_pair("D2");
        let d2_holdout = row.holdout_pair("D2");
        let d2_hourly = row.resolution_baseline_pair("D2");
        let d2_half_hour = row.resolution_fine_pair("D2");
        let d3 = row.training_pair("D3");
        let d3_holdout = row.holdout_pair("D3");
        writeln!(
            csv,
            "{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            row.seed,
            row.sample,
            row.operating_cost,
            d1[0],
            d1[1],
            d1_holdout[0],
            d1_holdout[1],
            d1_hourly[0],
            d1_hourly[1],
            d1_half_hour[0],
            d1_half_hour[1],
            d2[0],
            d2[1],
            d2_holdout[0],
            d2_holdout[1],
            d2_hourly[0],
            d2_hourly[1],
            d2_half_hour[0],
            d2_half_hour[1],
            d3[0],
            d3[1],
            d3_holdout[0],
            d3_holdout[1]
        )?;
    }
    fs::write(metadata.directory.join("pilot.csv"), csv)?;
    Ok(())
}

/// Write QD archive or a measured skip record.
pub fn write_qd(metadata: &Metadata<'_>, result: &QdResult) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"water-network-scheduling",
            "formulation":"robust-dda-qd",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "requested_evaluations":result.requested_evaluations,
            "actual_evaluations":result.actual_evaluations,
            "elapsed_seconds":result.elapsed.as_secs_f64(),
            "capacity":result.capacity,
            "occupied":result.entries.len(),
            "invalid_evaluations":result.invalid_evaluations,
            "clamped_evaluations":result.clamped_evaluations,
            "artifacts":{"archive":"qd_archive.csv","catalogue":"strategy_catalogue.csv"}
        }))?,
    )?;
    let mut csv = String::from("niche,visits,quality,descriptor_1,descriptor_2,controls\n");
    let mut catalogue =
        String::from("niche,operating_cost,off_peak_fraction,tank_turnover,controls\n");
    for entry in &result.entries {
        writeln!(
            csv,
            "{},{},{:.17},{:.17},{:.17},\"{}\"",
            entry.niche,
            entry.visits,
            entry.quality,
            entry.descriptors[0],
            entry.descriptors[1],
            controls(&entry.controls)
        )?;
        writeln!(
            catalogue,
            "{},{:.17},{:.17},{:.17},\"{}\"",
            entry.niche,
            entry.training.operating_cost,
            entry.descriptors[0],
            entry.descriptors[1],
            controls(&entry.controls)
        )?;
    }
    fs::write(metadata.directory.join("qd_archive.csv"), csv)?;
    fs::write(metadata.directory.join("strategy_catalogue.csv"), catalogue)?;
    Ok(())
}

pub fn write_qd_skipped(metadata: &Metadata<'_>, reason: &str) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    for stale in ["qd_archive.csv", "strategy_catalogue.csv"] {
        let path = metadata.directory.join(stale);
        if path.is_file() {
            fs::remove_file(path)?;
        }
    }
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"water-network-scheduling",
            "formulation":"robust-dda-qd",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "status":"skipped",
            "reason":reason,
            "requested_evaluations":0,
            "actual_evaluations":null,
            "artifacts":{}
        }))?,
    )?;
    Ok(())
}

/// Write constrained MODE front.
pub fn write_mo(metadata: &Metadata<'_>, result: &MoResult) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"water-network-scheduling",
            "formulation":"robust-dda-mode",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "requested_evaluations":result.requested_evaluations,
            "actual_evaluations":result.actual_evaluations,
            "elapsed_seconds":result.elapsed.as_secs_f64(),
            "pareto_points":result.pareto.len(),
            "artifacts":{"pareto":"mo_pareto.csv"}
        }))?,
    )?;
    let mut csv = String::from(
        "energy_cost,reliability_risk,switching_cost,excess_pressure,tank_bounds,tank_recovery,velocity,simulation_failure_constraint,selected,controls\n",
    );
    for point in &result.pareto {
        writeln!(
            csv,
            "{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},\"{}\"",
            point.evaluation.objectives[0],
            point.evaluation.objectives[1],
            point.evaluation.objectives[2],
            point.evaluation.objectives[3],
            point.evaluation.constraints[0],
            point.evaluation.constraints[1],
            point.evaluation.constraints[2],
            point.evaluation.constraints[3],
            point.selected,
            controls(&point.evaluation.robust.controls)
        )?;
    }
    fs::write(metadata.directory.join("mo_pareto.csv"), csv)?;
    Ok(())
}

/// Write resolution sensitivity.
pub fn write_resolution(
    metadata: &Metadata<'_>,
    rows: &[ResolutionRow],
) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    let mut csv = String::from(
        "case,hydraulic_timestep_s,billing_resolution_s,energy_kwh,energy_cost,peak_kw_hourly,peak_kw_native,starts,min_pressure_m,max_velocity_m_s,failed_at_step,override_steps,wall_seconds\n",
    );
    for row in rows {
        writeln!(
            csv,
            "{},{},3600,{:.17},{:.17},{:.17},{:.17},{},{:.17},{:.17},{},{},{:.17}",
            row.case,
            row.timestep_s,
            row.energy_kwh,
            row.energy_cost,
            row.peak_kw_hourly,
            row.peak_kw_native,
            row.starts,
            row.min_pressure_m,
            row.max_velocity_m_s,
            row.failed_at_step
                .map_or_else(String::new, |step| step.to_string()),
            row.override_steps,
            row.wall_seconds
        )?;
    }
    fs::write(metadata.directory.join("resolution_study.csv"), csv)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"water-network-scheduling",
            "formulation":"resolution-study",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "billing_resolution_s":3600,
            "hydraulic_timesteps_s":[3600,1800,900,300],
            "cases":["baseline","override-witness"],
            "artifacts":{"study":"resolution_study.csv"}
        }))?,
    )?;
    Ok(())
}

/// Write parallelism benchmark.
pub fn write_benchmark(
    metadata: &Metadata<'_>,
    rows: &[BenchmarkRow],
) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    let mut csv = String::from(
        "arrangement,candidates,workers,internal_parallel,wall_seconds,candidates_per_second,checksum\n",
    );
    for row in rows {
        writeln!(
            csv,
            "{},{},{},{},{:.17},{:.17},{:.17}",
            row.arrangement,
            row.candidates,
            row.workers,
            row.internal_parallel,
            row.wall_seconds,
            row.candidates_per_second,
            row.checksum
        )?;
    }
    fs::write(metadata.directory.join("parallelism_benchmark.csv"), csv)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"water-network-scheduling",
            "formulation":"parallelism-ownership",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "network_variant":"tank-free-control-free",
            "real_network_rule":"candidate parallelism only",
            "artifacts":{"benchmark":"parallelism_benchmark.csv"}
        }))?,
    )?;
    Ok(())
}
