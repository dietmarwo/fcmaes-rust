//! Versioned machine-readable tutorial artifacts.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde_json::json;

use crate::annual::AnnualResult;
use crate::archive_grid::ArchiveGrid;
use crate::config::{Preset, Protocol};
use crate::decode::OuterDesign;
use crate::evaluate::{OuterEvaluation, feasible};
use crate::landscape::{LandscapeResult, convexity_violation};
use crate::mo::MoResult;
use crate::pilot::{DescriptorPair, PilotRow, PilotSummary};
use crate::profiles::{Profile, ProfileModifiers, chronological_year};
use crate::qd::QdResult;
use crate::scenarios::all;
use crate::so::SoArmResult;

/// Metadata shared by schema-v1 manifests.
pub struct RunMetadata<'a> {
    /// Artifact directory.
    pub directory: &'a Path,
    /// Exact replay command.
    pub command: &'a str,
    /// Horizon preset.
    pub preset: Preset,
    /// Root seed.
    pub seed: u64,
    /// Candidate workers.
    pub workers: i32,
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

fn limitation(preset: Preset) -> &'static str {
    match preset {
        Preset::Smoke => {
            "four independently cyclic representative days; no seasonal or hydrogen claim"
        }
        Preset::Publication => {
            "twelve independently cyclic representative days; no seasonal or hydrogen claim"
        }
    }
}

fn design_fields(design: &OuterDesign) -> String {
    let cap = design.capacities;
    format!(
        "{},{},{},{},{},{},{},{},{},{},{}",
        cap.pv_kwp,
        cap.wind_kw,
        cap.battery_kwh,
        cap.battery_kw,
        cap.electrolyser_kw,
        cap.hydrogen_kwh,
        cap.grid_kw,
        design.grid_tier,
        usize::from(design.include_wind),
        usize::from(design.include_battery),
        usize::from(design.include_hydrogen)
    )
}

const DESIGN_HEADER: &str = "pv_kwp,wind_kw,battery_kwh,battery_kw,electrolyser_kw,hydrogen_kwh,grid_kw,grid_tier,include_wind,include_battery,include_hydrogen";

fn descriptor_labels(pair: DescriptorPair) -> [&'static str; 2] {
    match pair {
        DescriptorPair::D1 => ["daily_battery_throughput_per_kwh", "peak_import_ratio"],
        DescriptorPair::D2 => ["self_sufficiency", "curtailed_renewable_fraction"],
        DescriptorPair::D3 => ["pv_battery_capacity_ratio", "self_sufficiency"],
    }
}

fn dispatch_csv(evaluation: &OuterEvaluation, profile: &Profile) -> Result<String, Box<dyn Error>> {
    let dispatch = &evaluation.scenarios[0].dispatch;
    let capacity = evaluation.design.capacities;
    let mut csv = String::from(
        "step,hour,load_kw,renewable_kw,import_kw,export_kw,charge_kw,discharge_kw,soc_kwh,curtail_kw,unserved_kw,import_price\n",
    );
    for step in 0..profile.len() {
        let renewable =
            profile.solar_cf[step] * capacity.pv_kwp + profile.wind_cf[step] * capacity.wind_kw;
        writeln!(
            csv,
            "{step},{},{},{},{},{},{},{},{},{},{},{}",
            step as f64 * profile.dt_hours,
            profile.load_kw[step],
            renewable,
            dispatch.trace.import_kw[step],
            dispatch.trace.export_kw[step],
            dispatch.trace.charge_kw[step],
            dispatch.trace.discharge_kw[step],
            dispatch.trace.soc_kwh[step],
            dispatch.trace.curtail_kw[step],
            dispatch.trace.unserved_kw[step],
            profile.import_price[step]
        )?;
    }
    Ok(csv)
}

/// Write the four measured landscape curves.
pub fn write_landscape(
    metadata: &RunMetadata<'_>,
    result: &LandscapeResult,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut csv = String::from(
        "coordinate,convex_total_cost,tiered_total_cost,ratio_lcoe,delivered_lcoe,delivered_grid_tier,delivered_inclusions\n",
    );
    for row in &result.rows {
        writeln!(
            csv,
            "{},{},{},{},{},{},{}",
            row.coordinate,
            row.convex_total_cost,
            row.tiered_total_cost,
            row.ratio_lcoe,
            row.delivered_lcoe,
            row.delivered_grid_tier,
            row.delivered_inclusions
        )?;
    }
    write(&metadata.directory.join("landscape.csv"), &csv)?;
    let candidate_evaluations = result.rows.len() + 4 * result.derivative_probes;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "energy-hub-bilevel",
            "formulation": "landscape",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": candidate_evaluations,
            "actual_evaluations": candidate_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [],
            "descriptors": [],
            "convexity_max_relative_violation": convexity_violation(&result.rows),
            "derivative_probes": result.derivative_probes,
            "derivative_sign_disagreements": result.derivative_disagreements,
            "boundary_probes": result.boundary_probes,
            "boundary_sign_disagreements": result.boundary_disagreements,
            "budget": {
                "candidate_evaluations": candidate_evaluations,
                "lp_solves": result.lp_solves,
                "simplex_iterations": result.simplex_iterations,
                "solver_failures": 0
            },
            "horizon_limitation": limitation(metadata.preset),
            "artifacts": {"landscape": "landscape.csv"}
        }),
    )
}

/// Write the three equal-budget scalar arms and selected dispatch.
pub fn write_so(metadata: &RunMetadata<'_>, arms: &[SoArmResult]) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut best = format!(
        "optimizer,feasible,objective,mean_lcoe,worst_lcoe,min_self_sufficiency,max_unserved_fraction,max_annual_cycles,constraint_self_sufficiency,constraint_unserved,constraint_cycles,constraint_lp_status,{DESIGN_HEADER}\n"
    );
    let mut convergence = String::from("optimizer,evaluations,elapsed_seconds,best_objective\n");
    for arm in arms {
        let evaluation = &arm.best;
        writeln!(
            best,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            arm.optimizer.name(),
            usize::from(feasible(evaluation)),
            evaluation.objective,
            evaluation.mean_lcoe,
            evaluation.worst_lcoe,
            evaluation.min_self_sufficiency,
            evaluation.max_unserved_fraction,
            evaluation.max_annual_cycles,
            evaluation.constraint_self_sufficiency,
            evaluation.constraint_unserved,
            evaluation.constraint_cycles,
            evaluation.constraint_lp_status,
            design_fields(&evaluation.design)
        )?;
        if arm.improvements.is_empty() {
            writeln!(
                convergence,
                "{},{},{},{}",
                arm.optimizer.name(),
                arm.work.candidate_evaluations,
                arm.elapsed.as_secs_f64(),
                evaluation.objective
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
    }
    let selected = arms
        .iter()
        .filter(|arm| feasible(&arm.best))
        .min_by(|left, right| left.best.objective.total_cmp(&right.best.objective))
        .or_else(|| {
            arms.iter()
                .min_by(|left, right| left.best.objective.total_cmp(&right.best.objective))
        })
        .ok_or("SO writer needs at least one arm")?;
    let profile = crate::scenarios::training()[0].profile(metadata.preset);
    write(&metadata.directory.join("best.csv"), &best)?;
    write(&metadata.directory.join("convergence.csv"), &convergence)?;
    write(
        &metadata.directory.join("dispatch.csv"),
        &dispatch_csv(&selected.best, &profile)?,
    )?;
    let arm_json = arms
        .iter()
        .map(|arm| {
            json!({
                "optimizer": arm.optimizer.name(),
                "requested_evaluations": arm.requested_evaluations,
                "actual_evaluations": arm.work.candidate_evaluations,
                "completed_retries": arm.completed_retries,
                "elapsed_seconds": arm.elapsed.as_secs_f64(),
                "best_objective": arm.best.objective,
                "feasible": feasible(&arm.best),
                "budget": {
                    "candidate_evaluations": arm.work.candidate_evaluations,
                    "lp_solves": arm.work.lp_solves,
                    "simplex_iterations": arm.work.simplex_iterations,
                    "solver_failures": arm.work.solver_failures
                }
            })
        })
        .collect::<Vec<_>>();
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "energy-hub-bilevel",
            "formulation": "so-comparison",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": arms.iter().map(|arm| arm.requested_evaluations).sum::<u64>(),
            "actual_evaluations": arms.iter().map(|arm| arm.work.candidate_evaluations).sum::<u64>(),
            "elapsed_seconds": arms.iter().map(|arm| arm.elapsed.as_secs_f64()).sum::<f64>(),
            "objectives": [{"column": "objective", "label": "Robust LCOE plus feasibility penalties", "unit": "currency/kWh"}],
            "constraints": [
                {"column": "constraint_self_sufficiency", "feasible": "<= 0"},
                {"column": "constraint_unserved", "feasible": "<= 0"},
                {"column": "constraint_cycles", "feasible": "<= 0"},
                {"column": "constraint_lp_status", "feasible": "<= 0"}
            ],
            "descriptors": [],
            "horizon_limitation": limitation(metadata.preset),
            "selected_optimizer": selected.optimizer.name(),
            "arms": arm_json,
            "budget": {
                "candidate_evaluations": arms.iter().map(|arm| arm.work.candidate_evaluations).sum::<u64>(),
                "lp_solves": arms.iter().map(|arm| arm.work.lp_solves).sum::<u64>(),
                "simplex_iterations": arms.iter().map(|arm| arm.work.simplex_iterations).sum::<u64>(),
                "solver_failures": arms.iter().map(|arm| arm.work.solver_failures).sum::<u64>()
            },
            "artifacts": {"best": "best.csv", "convergence": "convergence.csv", "dispatch": "dispatch.csv"}
        }),
    )
}

/// Write descriptor-pilot evidence and verdict.
pub fn write_pilot(
    metadata: &RunMetadata<'_>,
    rows: &[PilotRow],
    summary: &PilotSummary,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut csv = String::from(
        "seed,sample,quality,d1_axis1_train,d1_axis2_train,d1_axis1_holdout,d1_axis2_holdout,d1_axis1_quarter_hour,d1_axis2_quarter_hour,d2_axis1_train,d2_axis2_train,d2_axis1_holdout,d2_axis2_holdout,d3_axis1_train,d3_axis2_train\n",
    );
    for row in rows {
        let d1 = DescriptorPair::D1.values(row.training);
        let held_d1 = DescriptorPair::D1.values(row.holdout);
        let fine_d1 = DescriptorPair::D1.values(row.quarter_hour);
        let d2 = DescriptorPair::D2.values(row.training);
        let held_d2 = DescriptorPair::D2.values(row.holdout);
        let d3 = DescriptorPair::D3.values(row.training);
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            row.seed,
            row.sample,
            row.quality,
            d1[0],
            d1[1],
            held_d1[0],
            held_d1[1],
            fine_d1[0],
            fine_d1[1],
            d2[0],
            d2[1],
            held_d2[0],
            held_d2[1],
            d3[0],
            d3[1]
        )?;
    }
    write(&metadata.directory.join("pilot.csv"), &csv)?;
    let layout = ArchiveGrid::new(metadata.preset.protocol().qd_capacity);
    let report = format!(
        "# Descriptor pilot verdict\n\n**{}** — {}\n\n- attempted: {}\n- feasible: {}\n- archive capacity: {}\n- archive row lengths: {:?}\n- D1 Spearman: {:.4}\n- D1 coverage: {:.2}%\n- D1 minimum per-seed coverage: {:.2}%\n- D1 holdout retention: {:.2}%\n- D1 quarter-hour normalized shift: {:.4}\n- D2 Spearman: {:.4}\n- D3 control Spearman: {:.4}\n",
        summary.decision.label(),
        summary.reason,
        summary.attempted_candidates,
        summary.feasible_candidates,
        layout.capacity(),
        layout.row_lengths(),
        summary.d1.rank_correlation,
        100.0 * summary.d1.coverage,
        100.0 * summary.d1.minimum_seed_coverage,
        100.0 * summary.d1.holdout_retention,
        summary.timestep_mean_normalized_shift,
        summary.d2.rank_correlation,
        summary.d3.rank_correlation
    );
    write(&metadata.directory.join("pilot.md"), &report)?;
    let mut qd_metadata = json!({
        "capacity": layout.capacity(),
        "grid_row_lengths": layout.row_lengths()
    });
    if let Some(shape) = layout.rectangular_shape() {
        qd_metadata["grid_shape"] = json!(shape);
    }
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "energy-hub-bilevel",
            "formulation": "descriptor-pilot",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": summary.attempted_candidates,
            "actual_evaluations": summary.attempted_candidates,
            "elapsed_seconds": summary.elapsed_seconds,
            "objectives": [],
            "descriptors": [],
            "qd_decision": summary.decision,
            "selected_pair": summary.selected_pair,
            "reason": summary.reason,
            "qd": qd_metadata,
            "diagnostics": {"d1": summary.d1, "d2": summary.d2, "d3": summary.d3},
            "timestep_mean_normalized_shift": summary.timestep_mean_normalized_shift,
            "horizon_limitation": limitation(metadata.preset),
            "artifacts": {"pilot": "pilot.csv", "report": "pilot.md"}
        }),
    )
}

/// Write MAP-Elites archive, portfolio, progress, and holdout migration.
pub fn write_qd(metadata: &RunMetadata<'_>, result: &QdResult) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let labels = descriptor_labels(result.descriptor_pair);
    let mut archive = format!(
        "niche_id,grid_x,grid_y,quality_train,quality_holdout,descriptor_{}_train,descriptor_{}_train,descriptor_{}_holdout,descriptor_{}_holdout,visit_count,retained_niche,constraint_self_sufficiency,constraint_unserved,constraint_cycles,{DESIGN_HEADER}\n",
        labels[0], labels[1], labels[0], labels[1]
    );
    let mut portfolio = format!(
        "niche_id,quality,descriptor_{},descriptor_{},{DESIGN_HEADER}\n",
        labels[0], labels[1]
    );
    let mut migration = format!(
        "niche_id,train_{},train_{},holdout_{},holdout_{},moved\n",
        labels[0], labels[1], labels[0], labels[1]
    );
    let layout = ArchiveGrid::new(result.capacity);
    for entry in &result.entries {
        let quality_holdout = entry.holdout.iter().map(|row| row.lcoe).sum::<f64>()
            / entry.holdout.len().max(1) as f64;
        let retained = layout.niche(
            entry.holdout_descriptors,
            result.descriptor_pair.lower(),
            result.descriptor_pair.upper(),
        ) == Some(entry.niche);
        writeln!(
            archive,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            entry.niche,
            entry.grid_x,
            entry.grid_y,
            entry.quality,
            quality_holdout,
            entry.descriptors[0],
            entry.descriptors[1],
            entry.holdout_descriptors[0],
            entry.holdout_descriptors[1],
            entry.visits,
            usize::from(retained),
            entry.training.constraint_self_sufficiency,
            entry.training.constraint_unserved,
            entry.training.constraint_cycles,
            design_fields(&entry.training.design)
        )?;
        writeln!(
            portfolio,
            "{},{},{},{},{}",
            entry.niche,
            entry.quality,
            entry.descriptors[0],
            entry.descriptors[1],
            design_fields(&entry.training.design)
        )?;
        writeln!(
            migration,
            "{},{},{},{},{},{}",
            entry.niche,
            entry.descriptors[0],
            entry.descriptors[1],
            entry.holdout_descriptors[0],
            entry.holdout_descriptors[1],
            usize::from(!retained)
        )?;
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
    write(&metadata.directory.join("qd_archive.csv"), &archive)?;
    write(&metadata.directory.join("portfolio.csv"), &portfolio)?;
    write(
        &metadata.directory.join("holdout_migration.csv"),
        &migration,
    )?;
    write(&metadata.directory.join("qd_convergence.csv"), &convergence)?;
    let mut qd_metadata = json!({
        "descriptor_pair": result.descriptor_pair,
        "capacity": result.capacity,
        "occupied": result.entries.len(),
        "clamped_descriptors": result.clamped_descriptors,
        "invalid_evaluations": result.invalid_evaluations
    });
    if let Some(shape) = layout.rectangular_shape() {
        qd_metadata["grid_shape"] = json!(shape);
    } else {
        qd_metadata["grid_row_lengths"] = json!(layout.row_lengths());
    }
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "energy-hub-bilevel",
            "formulation": "quality-diversity",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [{"column": "quality_train", "label": "Robust mean LCOE", "unit": "currency/kWh"}],
            "descriptors": [
                {"column": format!("descriptor_{}_train", labels[0]), "label": labels[0], "bounds": [result.descriptor_pair.lower()[0], result.descriptor_pair.upper()[0]]},
                {"column": format!("descriptor_{}_train", labels[1]), "label": labels[1], "bounds": [result.descriptor_pair.lower()[1], result.descriptor_pair.upper()[1]]}
            ],
            "qd": qd_metadata,
            "budget": {
                "candidate_evaluations": result.actual_evaluations,
                "lp_solves": result.lp_solves,
                "simplex_iterations": result.simplex_iterations,
                "solver_failures": 0
            },
            "horizon_limitation": limitation(metadata.preset),
            "artifacts": {
                "archive": "qd_archive.csv",
                "portfolio": "portfolio.csv",
                "convergence": "qd_convergence.csv",
                "holdout_migration": "holdout_migration.csv"
            }
        }),
    )
}

/// Record a pilot-rejected QD arm without pretending it executed.
pub fn write_qd_skipped(
    metadata: &RunMetadata<'_>,
    summary: &PilotSummary,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "energy-hub-bilevel",
            "formulation": "quality-diversity",
            "status": "skipped",
            "reason": summary.reason,
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": 0,
            "actual_evaluations": null,
            "elapsed_seconds": 0.0,
            "objectives": [],
            "descriptors": [],
            "qd_decision": summary.decision,
            "artifacts": {}
        }),
    )
}

/// Write constrained MODE Pareto evidence.
pub fn write_mo(metadata: &RunMetadata<'_>, result: &MoResult) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut pareto = format!(
        "point_id,feasible,selected,objective_annualized_capex,objective_unserved_kwh,objective_co2_kg,objective_curtailed_kwh,constraint_self_sufficiency,constraint_cycles,constraint_lp_status,mean_lcoe,{DESIGN_HEADER}\n"
    );
    for (point_id, point) in result.pareto.iter().enumerate() {
        let evaluation = &point.evaluation;
        writeln!(
            pareto,
            "{point_id},1,{},{},{},{},{},{},{},{},{},{}",
            usize::from(point.selected),
            evaluation.objectives[0],
            evaluation.objectives[1],
            evaluation.objectives[2],
            evaluation.objectives[3],
            evaluation.constraints[0],
            evaluation.constraints[1],
            evaluation.constraints[2],
            evaluation.outer.mean_lcoe,
            design_fields(&evaluation.outer.design)
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
            row.best_compromise,
            row.feasible_population,
            row.pareto_population
        )?;
    }
    write(&metadata.directory.join("pareto.csv"), &pareto)?;
    write(&metadata.directory.join("convergence.csv"), &convergence)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "energy-hub-bilevel",
            "formulation": "constrained-mo",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [
                {"column": "objective_annualized_capex", "label": "Annualized CAPEX", "unit": "currency/year"},
                {"column": "objective_unserved_kwh", "label": "Worst unserved electricity", "unit": "kWh/year"},
                {"column": "objective_co2_kg", "label": "Mean grid CO2", "unit": "kg/year"},
                {"column": "objective_curtailed_kwh", "label": "Mean curtailed renewable energy", "unit": "kWh/year"}
            ],
            "constraints": [
                {"column": "constraint_self_sufficiency", "feasible": "<= 0"},
                {"column": "constraint_cycles", "feasible": "<= 0"},
                {"column": "constraint_lp_status", "feasible": "<= 0"}
            ],
            "descriptors": [],
            "pareto_points": result.pareto.len(),
            "budget": {
                "candidate_evaluations": result.actual_evaluations,
                "lp_solves": result.lp_solves,
                "simplex_iterations": result.simplex_iterations,
                "solver_failures": 0
            },
            "horizon_limitation": limitation(metadata.preset),
            "artifacts": {"pareto": "pareto.csv", "convergence": "convergence.csv"}
        }),
    )
}

/// Write coarse annual sizing and independent hourly replay.
pub fn write_annual(
    metadata: &RunMetadata<'_>,
    result: &AnnualResult,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut summary = format!(
        "resolution_hours,objective,delivered_energy_cost,electricity_self_sufficiency,unserved_fraction,onsite_hydrogen_fraction,purchased_hydrogen_kwh,hydrogen_amplitude_kwh,simplex_iterations,{DESIGN_HEADER}\n"
    );
    for evaluation in [&result.coarse, &result.hourly] {
        writeln!(
            summary,
            "{},{},{},{},{},{},{},{},{},{}",
            evaluation.dt_hours,
            evaluation.objective,
            evaluation.delivered_energy_cost,
            evaluation.electricity_self_sufficiency,
            evaluation.unserved_fraction,
            evaluation.onsite_hydrogen_fraction,
            evaluation.dispatch.purchased_hydrogen_kwh,
            evaluation.dispatch.hydrogen_amplitude_kwh,
            evaluation.dispatch.simplex_iterations,
            design_fields(&evaluation.design)
        )?;
    }
    let profile = chronological_year(1, ProfileModifiers::default());
    let trace = &result.hourly.dispatch.trace;
    let cap = result.hourly.design.capacities;
    let mut dispatch = String::from(
        "hour,load_kw,renewable_kw,import_kw,export_kw,battery_soc_kwh,electrolyser_kw,hydrogen_store_kwh,hydrogen_buy_kw,hydrogen_demand_kw\n",
    );
    for hour in 0..profile.len() {
        let renewable = profile.solar_cf[hour] * cap.pv_kwp + profile.wind_cf[hour] * cap.wind_kw;
        writeln!(
            dispatch,
            "{hour},{},{},{},{},{},{},{},{},{}",
            profile.load_kw[hour],
            renewable,
            trace.import_kw[hour],
            trace.export_kw[hour],
            trace.soc_kwh[hour],
            trace.electrolyser_kw[hour],
            trace.hydrogen_kwh[hour],
            trace.hydrogen_buy_kw[hour],
            profile.hydrogen_demand_kw[hour]
        )?;
    }
    write(&metadata.directory.join("summary.csv"), &summary)?;
    write(&metadata.directory.join("dispatch.csv"), &dispatch)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "energy-hub-bilevel",
            "formulation": "chronological-annual-extension",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": 1,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [{"column": "delivered_energy_cost", "label": "Combined delivered-energy cost", "unit": "currency/kWh"}],
            "constraints": [
                {"column": "constraint_self_sufficiency", "feasible": "<= 0"},
                {"column": "constraint_unserved", "feasible": "<= 0"}
            ],
            "descriptors": [],
            "sizing_steps": 1460,
            "validation_steps": 8760,
            "hourly_validation": {
                "objective": result.hourly.objective,
                "hydrogen_amplitude_kwh": result.hourly.dispatch.hydrogen_amplitude_kwh,
                "max_balance_residual_kw": result.hourly.dispatch.max_balance_residual_kw,
                "max_storage_residual_kwh": result.hourly.dispatch.max_storage_residual_kwh
            },
            "budget": {
                "candidate_evaluations": result.actual_evaluations,
                "selection_replays": result.selection_replays,
                "validation_replays": result.validation_replays,
                "lp_solves": result.lp_solves,
                "simplex_iterations": result.simplex_iterations,
                "solver_failures": 0
            },
            "artifacts": {"summary": "summary.csv", "dispatch": "dispatch.csv"}
        }),
    )
}

/// Emit deterministic publication scenario profiles as checked-in evidence.
pub fn write_scenario_profiles(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "scenario,set,step,hour,dt_hours,solar_cf,wind_cf,load_kw,import_price,export_price,hydrogen_demand_kw,hydrogen_price\n",
    );
    for scenario in all() {
        let profile = scenario.profile(Preset::Publication);
        for step in 0..profile.len() {
            writeln!(
                csv,
                "{},{:?},{step},{},{},{},{},{},{},{},{},{}",
                scenario.name,
                scenario.set,
                step as f64 * profile.dt_hours,
                profile.dt_hours,
                profile.solar_cf[step],
                profile.wind_cf[step],
                profile.load_kw[step],
                profile.import_price[step],
                profile.export_price[step],
                profile.hydrogen_demand_kw[step],
                profile.hydrogen_price[step]
            )?;
        }
    }
    write(path, &csv)
}

/// Frozen budgets used by a preset.
#[must_use]
pub fn protocol_json(protocol: Protocol) -> serde_json::Value {
    json!({
        "representative_days": protocol.representative_days,
        "so_evaluations_per_arm": protocol.so_evaluations,
        "so_retries": protocol.so_retries,
        "pilot_samples_per_seed": protocol.pilot_samples,
        "qd_evaluations": protocol.qd_evaluations,
        "qd_capacity": protocol.qd_capacity,
        "mo_evaluations": protocol.mo_evaluations,
        "mo_population": protocol.mo_population,
        "annual_evaluations": protocol.annual_evaluations
    })
}

/// Write the frozen protocol beside an aggregate result bundle.
pub fn write_protocol(
    path: &Path,
    preset: Preset,
    protocol: Protocol,
    command: &str,
    seed: u64,
    workers: i32,
) -> Result<(), Box<dyn Error>> {
    write_json(
        path,
        &json!({
            "schema_version": 1,
            "tutorial": "energy-hub-bilevel",
            "formulation": "aggregate-protocol",
            "preset": preset.label(),
            "command": command,
            "seed": seed,
            "workers": workers,
            "horizon_limitation": limitation(preset),
            "protocol": protocol_json(protocol)
        }),
    )
}
