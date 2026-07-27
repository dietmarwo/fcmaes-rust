//! Versioned JSON/CSV writers for optimization and validation evidence.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde_json::json;

use crate::mode::ModeResult;
use crate::studies::{ScalingRow, TimestepRow, ValidationRow};
use crate::{DEFAULT_STEP_S, DEFAULT_STOP_S, simulate_thevenin};

fn write(path: &Path, contents: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    Ok(())
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), Box<dyn Error>> {
    write(path, &(serde_json::to_string_pretty(value)? + "\n"))
}

/// Reproducibility metadata stored beside a MODE result.
pub struct RunMetadata<'a> {
    /// Output directory for the run artifacts.
    pub directory: &'a Path,
    /// Exact command used to launch the run.
    pub command: &'a str,
    /// Optimizer random seed.
    pub seed: u64,
    /// Number of parallel simulation workers.
    pub workers: i32,
}

/// Write MODE metadata, Pareto points, convergence, and selected waveforms.
pub fn write_mode(metadata: &RunMetadata<'_>, result: &ModeResult) -> Result<(), Box<dyn Error>> {
    let directory = metadata.directory;
    fs::create_dir_all(directory)?;
    let mut pareto = String::from(
        "point_id,feasible,selected,objective_rise_time_ns,objective_overshoot_percent,constraint_peak_current_a,constraint_settling_time_ns,u_resistance,u_snubber,resistance_ohm,snubber_resistance_ohm,peak_driver_current_a,settling_time_ns,final_gate_voltage_v\n",
    );
    for (point_id, point) in result.pareto.iter().enumerate() {
        let evaluation = &point.evaluation;
        writeln!(
            pareto,
            "{point_id},1,{},{},{},{},{},{},{},{},{},{},{},{}",
            usize::from(point.selected),
            evaluation.objectives[0],
            evaluation.objectives[1],
            evaluation.constraints[0],
            evaluation.constraints[1],
            evaluation.controls[0],
            evaluation.controls[1],
            evaluation.design.resistance_ohm,
            evaluation.design.snubber_resistance_ohm,
            evaluation.metrics.peak_driver_current_a,
            evaluation.metrics.settling_time_ns,
            evaluation.metrics.final_gate_voltage_v,
        )?;
    }
    let mut convergence = String::from(
        "evaluations,elapsed_seconds,best_quality,feasible_population,pareto_population\n",
    );
    for row in &result.progress {
        writeln!(
            convergence,
            "{},{},{},{},{}",
            row.evaluations,
            row.elapsed_seconds,
            row.best_quality,
            row.feasible_population,
            row.pareto_population
        )?;
    }
    let mut waveforms = String::from("point_id,time_ns,drive_v,trace_v,gate_v\n");
    for (point_id, point) in result.pareto.iter().enumerate() {
        if !point.selected {
            continue;
        }
        let waveform = simulate_thevenin(point.evaluation.design, DEFAULT_STEP_S, DEFAULT_STOP_S)?;
        for index in 0..waveform.time_s.len() {
            writeln!(
                waveforms,
                "{point_id},{},{},{},{}",
                waveform.time_s[index] * 1.0e9,
                waveform.drive_v[index],
                waveform.trace_v[index],
                waveform.gate_v[index],
            )?;
        }
    }
    write(&directory.join("pareto.csv"), &pareto)?;
    write(&directory.join("convergence.csv"), &convergence)?;
    write(&directory.join("waveforms.csv"), &waveforms)?;
    write_json(
        &directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "thevenin-gate-driver",
            "formulation": "constrained-mo",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [
                {"column": "objective_rise_time_ns", "label": "10–90% rise time", "unit": "ns"},
                {"column": "objective_overshoot_percent", "label": "Gate overshoot", "unit": "%"}
            ],
            "constraints": [
                {"column": "constraint_peak_current_a", "label": "Peak current above 5 A", "unit": "A", "feasible": "<= 0"},
                {"column": "constraint_settling_time_ns", "label": "Settling time above 75 ns", "unit": "ns", "feasible": "<= 0"}
            ],
            "descriptors": [],
            "pareto_points": result.pareto.len(),
            "selected_points": result.pareto.iter().filter(|point| point.selected).count(),
            "transient": {
                "method": "trapezoidal",
                "maximum_step_s": DEFAULT_STEP_S,
                "stop_s": DEFAULT_STOP_S
            },
            "artifacts": {
                "mo_pareto": "pareto.csv",
                "mo_convergence": "convergence.csv",
                "selected_waveforms": "waveforms.csv"
            }
        }),
    )
}

/// Write the cross-simulator candidate grid, `thevenin` metrics, and timestep study.
pub fn write_validation(
    directory: &Path,
    grid: &[ValidationRow],
    timestep: &[TimestepRow],
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut candidates =
        String::from("point_id,u_resistance,u_snubber,resistance_ohm,snubber_resistance_ohm\n");
    let mut thevenin = String::from(
        "point_id,u_resistance,u_snubber,resistance_ohm,snubber_resistance_ohm,rise_time_ns,overshoot_percent,peak_driver_current_a,settling_time_ns,final_gate_voltage_v,timepoints\n",
    );
    for row in grid {
        writeln!(
            candidates,
            "{},{},{},{},{}",
            row.point_id,
            row.controls[0],
            row.controls[1],
            row.design.resistance_ohm,
            row.design.snubber_resistance_ohm,
        )?;
        writeln!(
            thevenin,
            "{},{},{},{},{},{},{},{},{},{},{}",
            row.point_id,
            row.controls[0],
            row.controls[1],
            row.design.resistance_ohm,
            row.design.snubber_resistance_ohm,
            row.metrics.rise_time_ns,
            row.metrics.overshoot_percent,
            row.metrics.peak_driver_current_a,
            row.metrics.settling_time_ns,
            row.metrics.final_gate_voltage_v,
            row.timepoints,
        )?;
    }
    let mut timestep_csv = String::from(
        "design_id,u_resistance,u_snubber,step_s,rise_time_ns,overshoot_percent,peak_driver_current_a,settling_time_ns,final_gate_voltage_v,timepoints\n",
    );
    for row in timestep {
        writeln!(
            timestep_csv,
            "{},{},{},{},{},{},{},{},{},{}",
            row.design_id,
            row.controls[0],
            row.controls[1],
            row.step_s,
            row.metrics.rise_time_ns,
            row.metrics.overshoot_percent,
            row.metrics.peak_driver_current_a,
            row.metrics.settling_time_ns,
            row.metrics.final_gate_voltage_v,
            row.timepoints,
        )?;
    }
    write(&directory.join("candidates.csv"), &candidates)?;
    write(&directory.join("thevenin.csv"), &thevenin)?;
    write(&directory.join("timestep.csv"), &timestep_csv)
}

/// Write repeated worker-scaling observations.
pub fn write_scaling(directory: &Path, rows: &[ScalingRow]) -> Result<(), Box<dyn Error>> {
    let mut csv =
        String::from("workers,repeat,candidates,elapsed_seconds,evaluations_per_second,failures\n");
    for row in rows {
        writeln!(
            csv,
            "{},{},{},{},{},{}",
            row.workers,
            row.repeat,
            row.candidates,
            row.elapsed_seconds,
            row.evaluations_per_second,
            row.failures,
        )?;
    }
    write(&directory.join("scaling.csv"), &csv)
}
