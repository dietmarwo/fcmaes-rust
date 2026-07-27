//! Versioned JSON/CSV artifact writers for the tutorial experiments.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde_json::json;

use crate::mo::MoResult;
use crate::qd::{DESCRIPTOR_LOWER, DESCRIPTOR_UPPER, QdResult, RangeStudyRow};
use crate::so::{SmoothnessRow, SoArmResult};

/// Metadata common to every schema-v1 run manifest.
pub struct RunMetadata<'a> {
    pub directory: &'a Path,
    pub command: &'a str,
    pub seed: u64,
    pub workers: i32,
    pub points: usize,
}

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

/// Write the SO comparison, feature curve, and interpolation study.
pub fn write_so(
    metadata: &RunMetadata<'_>,
    arms: &[SoArmResult],
    feature_curve: &[(f64, f64)],
    smoothness: &[SmoothnessRow],
) -> Result<(), Box<dyn Error>> {
    let directory = metadata.directory;
    fs::create_dir_all(directory)?;
    let mut convergence = String::from("optimizer,evaluations,elapsed_seconds,best_objective\n");
    let mut best = String::from(
        "optimizer,objective,peak_hz,peak_db,q,lower_3db_hz,upper_3db_hz,r1_ohm,r2_ohm,r3_ohm,c1_f,c2_f\n",
    );
    for arm in arms {
        if arm.improvements.is_empty() {
            writeln!(
                convergence,
                "{},{},{},{}",
                arm.optimizer.name(),
                arm.actual_evaluations,
                arm.elapsed.as_secs_f64(),
                arm.best.objective
            )?;
        } else {
            for row in &arm.improvements {
                writeln!(
                    convergence,
                    "{},{},{},{}",
                    arm.optimizer.name(),
                    row.evaluations,
                    row.elapsed_seconds,
                    row.value
                )?;
            }
        }
        let components = arm.best.components;
        let features = arm.best.features;
        writeln!(
            best,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            arm.optimizer.name(),
            arm.best.objective,
            features.peak_hz,
            features.peak_db,
            features.q,
            features.lower_3db_hz,
            features.upper_3db_hz,
            components[0],
            components[1],
            components[2],
            components[3],
            components[4],
        )?;
    }
    let mut curve = String::from("frequency_hz,gain_db\n");
    for (frequency, gain) in feature_curve {
        writeln!(curve, "{frequency},{gain}")?;
    }
    let mut smooth = String::from("sample,r1_ohm,grid_peak_hz,interpolated_peak_hz\n");
    for row in smoothness {
        writeln!(
            smooth,
            "{},{},{},{}",
            row.sample, row.r1_ohm, row.grid_peak_hz, row.interpolated_peak_hz
        )?;
    }
    write(&directory.join("convergence.csv"), &convergence)?;
    write(&directory.join("best.csv"), &best)?;
    write(&directory.join("feature_curve.csv"), &curve)?;
    write(&directory.join("feature_smoothness.csv"), &smooth)?;
    let requested = arms
        .iter()
        .map(|arm| arm.requested_evaluations)
        .sum::<u64>();
    let actual = arms.iter().map(|arm| arm.actual_evaluations).sum::<u64>();
    let elapsed = arms
        .iter()
        .map(|arm| arm.elapsed.as_secs_f64())
        .sum::<f64>();
    let arm_json = arms
        .iter()
        .map(|arm| {
            json!({
                "optimizer": arm.optimizer.name(),
                "requested_evaluations": arm.requested_evaluations,
                "actual_evaluations": arm.actual_evaluations,
                "completed_retries": arm.completed_retries,
                "elapsed_seconds": arm.elapsed.as_secs_f64(),
                "best_objective": arm.best.objective
            })
        })
        .collect::<Vec<_>>();
    write_json(
        &directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "sindr-circuit-design",
            "formulation": "so-comparison",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": requested,
            "actual_evaluations": actual,
            "elapsed_seconds": elapsed,
            "objectives": [
                {"column": "best_objective", "label": "Weighted target error", "unit": "1"}
            ],
            "descriptors": [],
            "ac_points": metadata.points,
            "budget_semantics": "requested and actual totals sum three equal-budget optimizer arms",
            "arms": arm_json,
            "artifacts": {
                "so_convergence": "convergence.csv",
                "so_best": "best.csv",
                "feature_curve": "feature_curve.csv",
                "feature_smoothness": "feature_smoothness.csv"
            }
        }),
    )
}

/// Write the feasible MODE Pareto set and progress.
pub fn write_mo(metadata: &RunMetadata<'_>, result: &MoResult) -> Result<(), Box<dyn Error>> {
    let directory = metadata.directory;
    fs::create_dir_all(directory)?;
    let mut pareto = String::from(
        "point_id,feasible,selected,objective_cutoff_error,objective_passband_ripple_db,objective_total_capacitance_nf,constraint_peak_db,cutoff_hz,peak_above_dc_db,r1_ohm,r2_ohm,r3_ohm,r4_ohm,c1_f,c2_f,c3_f,c4_f\n",
    );
    for (point_id, point) in result.pareto.iter().enumerate() {
        let evaluation = &point.evaluation;
        let components = evaluation.components;
        writeln!(
            pareto,
            "{point_id},1,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            usize::from(point.selected),
            evaluation.objectives[0],
            evaluation.objectives[1],
            evaluation.objectives[2],
            evaluation.constraint,
            evaluation.features.cutoff_hz,
            evaluation.features.peak_above_dc_db,
            components[0],
            components[1],
            components[2],
            components[3],
            components[4],
            components[5],
            components[6],
            components[7],
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
    write(&directory.join("pareto.csv"), &pareto)?;
    write(&directory.join("convergence.csv"), &convergence)?;
    write_json(
        &directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "sindr-circuit-design",
            "formulation": "constrained-mo",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [
                {"column": "objective_cutoff_error", "label": "Cutoff error", "unit": "decades"},
                {"column": "objective_passband_ripple_db", "label": "Pass-band ripple", "unit": "dB"},
                {"column": "objective_total_capacitance_nf", "label": "Total capacitance", "unit": "nF"}
            ],
            "constraints": [
                {"column": "constraint_peak_db", "label": "Peak above 3 dB", "unit": "dB", "feasible": "<= 0"}
            ],
            "descriptors": [],
            "ac_points": metadata.points,
            "pareto_points": result.pareto.len(),
            "artifacts": {
                "mo_pareto": "pareto.csv",
                "mo_convergence": "convergence.csv"
            }
        }),
    )
}

/// Write the descriptor range study and MAP-Elites catalogue.
pub fn write_qd(
    metadata: &RunMetadata<'_>,
    mc_draws: usize,
    range_attempts: usize,
    range_rows: &[RangeStudyRow],
    result: &QdResult,
) -> Result<(), Box<dyn Error>> {
    let directory = metadata.directory;
    fs::create_dir_all(directory)?;
    let mut range = String::from(
        "sample,descriptor_log10_f0,descriptor_peak_gain_db,r1_index,r2_index,r3_index,c1_index,c2_index,r1_ohm,r2_ohm,r3_ohm,c1_f,c2_f\n",
    );
    for row in range_rows {
        writeln!(
            range,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            row.sample,
            row.descriptors[0],
            row.descriptors[1],
            row.indices[0],
            row.indices[1],
            row.indices[2],
            row.indices[3],
            row.indices[4],
            row.components[0],
            row.components[1],
            row.components[2],
            row.components[3],
            row.components[4],
        )?;
    }
    let mut archive = String::from(
        "niche_id,grid_x,grid_y,selected,quality_robustness_db,descriptor_log10_f0,descriptor_peak_gain_db,visit_count,r1_index,r2_index,r3_index,c1_index,c2_index,r1_ohm,r2_ohm,r3_ohm,c1_f,c2_f\n",
    );
    let mut curves =
        String::from("niche_id,frequency_hz,gain_db,descriptor_log10_f0,descriptor_peak_gain_db\n");
    for elite in &result.elites {
        let evaluation = &elite.evaluation;
        writeln!(
            archive,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            elite.niche,
            elite.grid_x,
            elite.grid_y,
            usize::from(elite.selected),
            evaluation.quality,
            evaluation.descriptors[0],
            evaluation.descriptors[1],
            elite.visits,
            evaluation.indices[0],
            evaluation.indices[1],
            evaluation.indices[2],
            evaluation.indices[3],
            evaluation.indices[4],
            evaluation.components[0],
            evaluation.components[1],
            evaluation.components[2],
            evaluation.components[3],
            evaluation.components[4],
        )?;
        if elite.selected {
            for (frequency, gain) in &elite.curve {
                writeln!(
                    curves,
                    "{},{},{},{},{}",
                    elite.niche,
                    frequency,
                    gain,
                    evaluation.descriptors[0],
                    evaluation.descriptors[1]
                )?;
            }
        }
    }
    let mut convergence = String::from(
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction\n",
    );
    for row in &result.progress {
        writeln!(
            convergence,
            "{},{},{},{},{},{}",
            row.evaluations,
            row.elapsed_seconds,
            row.coverage,
            row.qd_score,
            row.best_quality,
            row.invalid_fraction
        )?;
    }
    write(&directory.join("range_study.csv"), &range)?;
    write(&directory.join("archive.csv"), &archive)?;
    write(&directory.join("elites.csv"), &curves)?;
    write(&directory.join("convergence.csv"), &convergence)?;
    let side = (result.capacity as f64).sqrt() as usize;
    write_json(
        &directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "sindr-circuit-design",
            "formulation": "quality-diversity",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [
                {"column": "quality_robustness_db", "label": "Tolerance sensitivity", "unit": "dB"}
            ],
            "descriptors": [
                {"column": "descriptor_log10_f0", "label": "Centre frequency", "unit": "log10(Hz)"},
                {"column": "descriptor_peak_gain_db", "label": "Peak gain", "unit": "dB"}
            ],
            "ac_points": metadata.points,
            "mc_draws": mc_draws,
            "ac_solves": result.ac_solves + range_attempts,
            "optimization_ac_solves": result.ac_solves,
            "range_study_ac_solves": range_attempts,
            "invalid_evaluations": result.invalid_evaluations,
            "out_of_range_descriptors": result.out_of_range_descriptors,
            "distinct_elite_designs": result.distinct_elite_designs,
            "qd": {
                "capacity": result.capacity,
                "occupied": result.elites.len(),
                "grid_shape": [side, side],
                "descriptor_lower": DESCRIPTOR_LOWER,
                "descriptor_upper": DESCRIPTOR_UPPER,
                "range_study_attempts": range_attempts,
                "range_study_valid": range_rows.len()
            },
            "artifacts": {
                "qd_range_study": "range_study.csv",
                "qd_catalogue": "archive.csv",
                "qd_elites": "elites.csv",
                "qd_convergence": "convergence.csv"
            }
        }),
    )
}
