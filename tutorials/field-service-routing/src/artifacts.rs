//! Versioned machine-readable publication artifacts.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde_json::json;

use crate::baseline::BaselineResult;
use crate::decode::Decoded;
use crate::instance::Instance;
use crate::mo::MoResult;
use crate::pilot::PilotSummary;
use crate::qd::QdResult;
use crate::scenarios::RobustEvaluation;
use crate::so::SoResult;

/// Common manifest metadata.
pub struct Metadata<'a> {
    /// Artifact directory.
    pub directory: &'a Path,
    /// Replay command.
    pub command: &'a str,
    /// Preset label.
    pub preset: &'a str,
    /// Root seed.
    pub seed: u64,
    /// Requested workers.
    pub workers: i32,
}

fn prepare(directory: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    Ok(())
}

fn controls(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn route_string(decoded: &Decoded) -> String {
    decoded
        .routes
        .iter()
        .filter(|route| !route.tasks.is_empty())
        .map(|route| {
            format!(
                "{}:{}",
                route.vehicle,
                route
                    .tasks
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("-")
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// Write scalar-arm artifacts.
pub fn write_so(
    metadata: &Metadata<'_>,
    arms: &[SoResult],
    seed: &RobustEvaluation,
) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    let requested = arms
        .iter()
        .map(|arm| arm.requested_evaluations)
        .sum::<u64>();
    let actual = arms.iter().map(|arm| arm.actual_evaluations).sum::<u64>();
    let elapsed = arms
        .iter()
        .map(|arm| arm.elapsed.as_secs_f64())
        .sum::<f64>();
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "tutorial": "field-service-routing",
            "formulation": "robust-hard-window-so",
            "command": metadata.command,
            "preset": metadata.preset,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": requested,
            "actual_evaluations": actual,
            "baseline_evaluations": 1,
            "elapsed_seconds": elapsed,
            "objectives": [{"column":"worst_cost","label":"Worst scenario cost","unit":"currency"}],
            "constraints": [
                {"column":"constraint_capacity","feasible":"<=0"},
                {"column":"constraint_lateness","feasible":"<=0"},
                {"column":"constraint_shift","feasible":"<=0"}
            ],
            "artifacts": {"arms":"arms.csv","convergence":"convergence.csv","routes":"routes.csv"}
        }))?,
    )?;
    let mut arm_csv = String::from(
        "arm,feasible,worst_cost,objective,constraint_capacity,constraint_lateness,constraint_shift,used_vehicles,distance_km,actual_evaluations,elapsed_seconds,delta_vs_seed,search_best_feasible,search_best_cost,search_found_feasible_improvement\n",
    );
    let mut convergence = String::from("arm,evaluations,elapsed_seconds,best_quality\n");
    let mut routes = String::from("arm,scenario,feasible,worst_cost,routes,controls\n");
    let nominal = &seed.nominal().metrics;
    writeln!(
        arm_csv,
        "seed,{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{:.17},1,0.00000000000000000,0.00000000000000000,{},{:.17},false",
        seed.feasible(),
        seed.worst_cost,
        seed.objective,
        seed.constraints[0],
        seed.constraints[1],
        seed.constraints[2],
        nominal.used_vehicles,
        nominal.distance_km,
        seed.feasible(),
        seed.worst_cost
    )?;
    writeln!(
        convergence,
        "seed,0,0.00000000000000000,{:.17}",
        seed.objective
    )?;
    for scenario in &seed.scenarios {
        writeln!(
            routes,
            "seed,{},{},{:.17},\"{}\",\"{}\"",
            scenario.name,
            seed.feasible(),
            seed.worst_cost,
            route_string(&scenario.decoded),
            controls(&seed.controls)
        )?;
    }
    for arm in arms {
        let nominal = &arm.best.nominal().metrics;
        writeln!(
            arm_csv,
            "{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{:.17},{},{:.17},{:.17},{},{:.17},{}",
            arm.optimizer.name(),
            arm.best.feasible(),
            arm.best.worst_cost,
            arm.best.objective,
            arm.best.constraints[0],
            arm.best.constraints[1],
            arm.best.constraints[2],
            nominal.used_vehicles,
            nominal.distance_km,
            arm.actual_evaluations,
            arm.elapsed.as_secs_f64(),
            arm.best.worst_cost - seed.worst_cost,
            arm.search_best.feasible(),
            arm.search_best.worst_cost,
            arm.search_found_feasible_improvement
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
        for scenario in &arm.best.scenarios {
            writeln!(
                routes,
                "{},{},{},{:.17},\"{}\",\"{}\"",
                arm.optimizer.name(),
                scenario.name,
                arm.best.feasible(),
                arm.best.worst_cost,
                route_string(&scenario.decoded),
                controls(&arm.best.controls)
            )?;
        }
    }
    fs::write(metadata.directory.join("arms.csv"), arm_csv)?;
    fs::write(metadata.directory.join("convergence.csv"), convergence)?;
    fs::write(metadata.directory.join("routes.csv"), routes)?;
    Ok(())
}

/// Write descriptor-pilot evidence.
pub fn write_pilot(metadata: &Metadata<'_>, pilot: &PilotSummary) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "tutorial": "field-service-routing",
            "formulation": "descriptor-pilot",
            "command": metadata.command,
            "preset": metadata.preset,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": pilot.attempted,
            "actual_evaluations": pilot.attempted,
            "feasible_candidates": pilot.rows.len(),
            "generator": {
                "local_attempted": pilot.local_attempted,
                "uniform_attempted": pilot.uniform_attempted,
                "local_feasible": pilot.rows.iter().filter(|row| row.source.name() == "local").count(),
                "uniform_feasible": pilot.rows.iter().filter(|row| row.source.name() == "uniform").count()
            },
            "archive": {
                "capacity": pilot.archive_capacity,
                "row_lengths": pilot.archive_row_lengths
            },
            "descriptor_holdout": "geography_uniform",
            "hard_feasibility_holdout": "all four holdout scenarios",
            "qd_decision": pilot.decision.label(),
            "pairs": {
                "D1": pilot.d1,
                "D2": pilot.d2,
                "D3": pilot.d3
            },
            "artifacts": {"samples":"pilot.csv"}
        }))?,
    )?;
    let mut csv = String::from(
        "seed,sample,source,vehicles_train,imbalance_cv_train,mean_waiting_s_train,distance_km_train,vehicles_holdout,imbalance_cv_holdout,mean_waiting_s_holdout,distance_km_holdout,holdout_feasible\n",
    );
    for row in &pilot.rows {
        writeln!(
            csv,
            "{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{}",
            row.seed,
            row.sample,
            row.source.name(),
            row.vehicles,
            row.imbalance,
            row.mean_waiting_s,
            row.distance_km,
            row.holdout_vehicles,
            row.holdout_imbalance,
            row.holdout_mean_waiting_s,
            row.holdout_distance_km,
            row.holdout_feasible
        )?;
    }
    fs::write(metadata.directory.join("pilot.csv"), csv)?;
    Ok(())
}

/// Write QD archive and machine-consumable route catalogue.
pub fn write_qd(metadata: &Metadata<'_>, result: &QdResult) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "tutorial": "field-service-routing",
            "formulation": "robust-hard-window-qd",
            "command": metadata.command,
            "preset": metadata.preset,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "descriptors": [
                {"column":"vehicles_train","label":"Vehicles used","bounds":[3.5,8.5]},
                {"column":"imbalance_train","label":"Route distance CV","bounds":[0.0,1.0]}
            ],
            "qd": {"capacity":result.capacity,"occupied":result.entries.len()},
            "invalid_evaluations":result.invalid_evaluations,
            "clamped_descriptors":result.clamped_descriptors,
            "artifacts": {"archive":"qd_archive.csv","convergence":"qd_convergence.csv","catalogue":"plan_repertoire.csv","holdout":"holdout_migration.csv"}
        }))?,
    )?;
    let mut archive = String::from(
        "niche_id,grid_x,grid_y,quality_train,quality_holdout,vehicles_train,imbalance_train,vehicles_holdout,imbalance_holdout,visit_count,selection_feasible,controls\n",
    );
    let mut catalogue = String::from("niche_id,worst_cost,vehicles,imbalance_cv,routes,controls\n");
    let mut migration = String::from(
        "niche_id,vehicles_train,imbalance_train,vehicles_holdout,imbalance_holdout,holdout_feasible\n",
    );
    for entry in &result.entries {
        let holdout_quality = entry
            .holdout
            .as_ref()
            .map_or(f64::INFINITY, |value| value.worst_cost);
        let holdout_descriptors = entry.holdout.as_ref().map_or([f64::NAN; 2], |value| {
            [
                value.nominal().metrics.used_vehicles as f64,
                value.nominal().metrics.imbalance_cv,
            ]
        });
        let holdout_feasible = entry.holdout.as_ref().is_some_and(|value| value.feasible());
        writeln!(
            archive,
            "{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},{},\"{}\"",
            entry.niche,
            entry.grid_x,
            entry.grid_y,
            entry.quality,
            holdout_quality,
            entry.descriptors[0],
            entry.descriptors[1],
            holdout_descriptors[0],
            holdout_descriptors[1],
            entry.visits,
            holdout_feasible,
            controls(&entry.controls)
        )?;
        writeln!(
            catalogue,
            "{},{:.17},{:.17},{:.17},\"{}\",\"{}\"",
            entry.niche,
            entry.quality,
            entry.descriptors[0],
            entry.descriptors[1],
            route_string(&entry.training.nominal().decoded),
            controls(&entry.controls)
        )?;
        writeln!(
            migration,
            "{},{:.17},{:.17},{:.17},{:.17},{}",
            entry.niche,
            entry.descriptors[0],
            entry.descriptors[1],
            holdout_descriptors[0],
            holdout_descriptors[1],
            holdout_feasible
        )?;
    }
    let mut convergence = String::from(
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction\n",
    );
    for row in &result.progress {
        writeln!(
            convergence,
            "{},{:.17},{:.17},{:.17},{:.17},{:.17}",
            row.evaluations,
            row.elapsed_seconds,
            row.coverage,
            row.qd_score,
            row.best_quality,
            row.invalid_fraction
        )?;
    }
    fs::write(metadata.directory.join("qd_archive.csv"), archive)?;
    fs::write(metadata.directory.join("plan_repertoire.csv"), catalogue)?;
    fs::write(metadata.directory.join("holdout_migration.csv"), migration)?;
    fs::write(metadata.directory.join("qd_convergence.csv"), convergence)?;
    Ok(())
}

/// Write a schema-compliant skipped QD manifest.
pub fn write_qd_skipped(metadata: &Metadata<'_>, reason: &str) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    for stale in [
        "qd_archive.csv",
        "plan_repertoire.csv",
        "holdout_migration.csv",
        "qd_convergence.csv",
    ] {
        let path = metadata.directory.join(stale);
        if path.is_file() {
            fs::remove_file(path)?;
        }
    }
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "tutorial":"field-service-routing",
            "formulation":"robust-hard-window-qd",
            "command":metadata.command,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "status":"skipped",
            "reason":reason,
            "actual_evaluations":serde_json::Value::Null,
            "artifacts":{}
        }))?,
    )?;
    Ok(())
}

/// Write constrained MODE results.
pub fn write_mo(metadata: &Metadata<'_>, result: &MoResult) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"field-service-routing",
            "formulation":"soft-window-mo",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":metadata.workers,
            "requested_evaluations":result.requested_evaluations,
            "actual_evaluations":result.actual_evaluations,
            "elapsed_seconds":result.elapsed.as_secs_f64(),
            "objectives":[
                {"column":"objective_distance_km","unit":"km"},
                {"column":"objective_vehicles","unit":"count"},
                {"column":"objective_makespan_s","unit":"s"},
                {"column":"objective_lateness_s","unit":"s"}
            ],
            "artifacts":{"pareto":"pareto.csv","convergence":"convergence.csv"}
        }))?,
    )?;
    let mut pareto = String::from(
        "point_id,feasible,selected,objective_distance_km,objective_vehicles,objective_makespan_s,objective_lateness_s,constraint_capacity,constraint_shift,controls\n",
    );
    for (index, point) in result.pareto.iter().enumerate() {
        writeln!(
            pareto,
            "{},true,{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},\"{}\"",
            index,
            point.selected,
            point.evaluation.objectives[0],
            point.evaluation.objectives[1],
            point.evaluation.objectives[2],
            point.evaluation.objectives[3],
            point.evaluation.constraints[0],
            point.evaluation.constraints[1],
            controls(&point.evaluation.controls)
        )?;
    }
    let mut convergence =
        String::from("evaluations,elapsed_seconds,feasible_population,pareto_population\n");
    for row in &result.progress {
        writeln!(
            convergence,
            "{},{:.17},{},{}",
            row.evaluations, row.elapsed_seconds, row.feasible, row.pareto
        )?;
    }
    fs::write(metadata.directory.join("pareto.csv"), pareto)?;
    fs::write(metadata.directory.join("convergence.csv"), convergence)?;
    Ok(())
}

/// Write per-instance structural baseline comparison.
pub fn write_baselines(
    metadata: &Metadata<'_>,
    rows: &[(Instance, BaselineResult, f64)],
) -> Result<(), Box<dyn Error>> {
    prepare(metadata.directory)?;
    let mut csv = String::from(
        "instance,seed,baseline_cost,witness_cost,gap_percent,feasible,construction_fallback,operations,elapsed_seconds\n",
    );
    for (instance, result, witness_cost) in rows {
        writeln!(
            csv,
            "{},{},{:.17},{:.17},{:.17},{},{},{},{:.17}",
            instance.name,
            instance.seed,
            result.metrics.cost,
            witness_cost,
            100.0 * (result.metrics.cost - witness_cost) / witness_cost,
            crate::evaluate::feasible(&result.metrics),
            result.construction_fallback,
            result.operations,
            result.elapsed.as_secs_f64()
        )?;
    }
    fs::write(metadata.directory.join("baseline_comparison.csv"), csv)?;
    fs::write(
        metadata.directory.join("run.json"),
        serde_json::to_string_pretty(&json!({
            "schema_version":1,
            "tutorial":"field-service-routing",
            "formulation":"structural-baseline",
            "command":metadata.command,
            "preset":metadata.preset,
            "seed":metadata.seed,
            "workers":1,
            "instances":rows.len(),
            "artifacts":{"comparison":"baseline_comparison.csv"}
        }))?,
    )?;
    Ok(())
}
