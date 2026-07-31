//! Machine-readable publication artifact writers.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde_json::json;

use crate::catalogue::sections;
use crate::config::Protocol;
use crate::decode::{baseline_controls, dimension};
use crate::evaluate::Evaluation;
use crate::fem::{RCOND_MIN, Scenario, WorkCounter, triangular_oracle};
use crate::ground::GroundStructure;
use crate::mo::MoResult;
use crate::pilot::{
    BROAD_UNIFORM_STRIDE, PILOT_PROTOCOL_REVISION, PUBLICATION_CAPACITY, PilotResult,
};
use crate::qd::QdOutcome;
use crate::so::SoArmResult;

/// Metadata common to schema-v1 manifests.
pub struct RunMetadata<'a> {
    /// Artifact directory.
    pub directory: &'a Path,
    /// Exact replay command.
    pub command: &'a str,
    /// Root optimizer seed.
    pub seed: u64,
    /// Requested candidate worker count.
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

fn controls(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn constraints(evaluation: &Evaluation) -> [f64; 6] {
    evaluation.constraints.optimizer_values()
}

fn evaluation_json(evaluation: &Evaluation) -> serde_json::Value {
    let metrics = evaluation.metrics.as_ref();
    let redundancy = evaluation.redundancy.as_ref();
    json!({
        "feasible": evaluation.feasible(),
        "objective": evaluation.objective,
        "mass_kg": evaluation.mass_kg,
        "carbon_kg_co2e_indicative": evaluation.carbon_kg_co2e,
        "active_count": evaluation.active_count,
        "depth_to_span": evaluation.depth_to_span,
        "rcond": metrics.map(|value| value.rcond),
        "max_stress_ratio": metrics.map(|value| value.max_stress_ratio),
        "max_buckling_ratio": metrics.map(|value| value.max_buckling_ratio),
        "max_displacement_m": metrics.map(|value| value.max_displacement_m),
        "compliance_j": metrics.map(|value| value.compliance_j),
        "redundancy_degradation": redundancy.map(|value| value.degradation),
        "removal_survival": redundancy.map(|value| value.survival),
        "failed_removals": redundancy.map(|value| value.failed_removals),
        "failure": evaluation.failure.as_ref().map(|failure| failure.kind()),
        "constraints": constraints(evaluation)
    })
}

/// Write the computed circular-section catalogue.
pub fn write_catalogue(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut csv = String::from(
        "section_index,name,outer_diameter_m,wall_m,area_m2,inertia_m4,radius_gyration_m,mass_kg_m,carbon_kg_co2e_per_kg_indicative\n",
    );
    for (index, section) in sections().iter().enumerate() {
        writeln!(
            csv,
            "{index},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            section.name,
            section.outer_diameter_m,
            section.wall_m,
            section.area_m2,
            section.inertia_m4,
            section.radius_gyration_m,
            section.mass_kg_m,
            section.carbon_kg_co2e_per_kg
        )?;
    }
    write(path, &csv)
}

/// Write the exact triangular oracle and conditioning-gate sweep.
pub fn write_validation(metadata: &RunMetadata<'_>) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let evidence = triangular_oracle()
        .map_err(|failure| format!("triangular oracle failed: {}", failure.kind()))?;
    let mut oracle = String::from("quantity,analytic,fem,absolute_error,relative_error,unit\n");
    for (index, (analytic, fem)) in evidence
        .analytic_forces_n
        .iter()
        .zip(evidence.fem_forces_n)
        .enumerate()
    {
        writeln!(
            oracle,
            "member_{}_force,{analytic:.17},{fem:.17},{:.17},{:.17},N",
            index + 1,
            (analytic - fem).abs(),
            (analytic - fem).abs() / analytic.abs().max(1.0e-30)
        )?;
    }
    writeln!(
        oracle,
        "apex_vertical_displacement,{:.17},{:.17},{:.17},{:.17},m",
        evidence.analytic_displacement_m,
        evidence.fem_displacement_m,
        (evidence.analytic_displacement_m - evidence.fem_displacement_m).abs(),
        (evidence.analytic_displacement_m - evidence.fem_displacement_m).abs()
            / evidence.analytic_displacement_m.abs().max(1.0e-30)
    )?;

    let ground = GroundStructure::reference();
    let counter = WorkCounter::default();
    let baseline = crate::evaluate::evaluate(
        &baseline_controls(&ground),
        &ground,
        Scenario::TRAINING,
        false,
        &counter,
    )
    .ok_or("baseline failed validation replay")?;
    let measured = baseline
        .metrics
        .as_ref()
        .ok_or("baseline has no physical metrics")?
        .rcond;
    let mut sensitivity = String::from("rcond_threshold,measured_rcond,passes\n");
    for threshold in [1.0e-4, 1.0e-6, 1.0e-8, RCOND_MIN, 1.0e-12, 1.0e-14] {
        writeln!(
            sensitivity,
            "{threshold:.17},{measured:.17},{}",
            usize::from(measured >= threshold)
        )?;
    }
    write(&metadata.directory.join("oracle.csv"), &oracle)?;
    write(
        &metadata.directory.join("condition_sensitivity.csv"),
        &sensitivity,
    )?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "truss-sizing",
            "formulation": "validation",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": 0,
            "actual_evaluations": 0,
            "elapsed_seconds": 0.0,
            "objectives": [],
            "descriptors": [],
            "reference_dimension": dimension(&ground),
            "rcond_min": RCOND_MIN,
            "oracle_work": {
                "candidate_evaluations": evidence.work.candidate_evaluations,
                "fem_solves": evidence.work.fem_solves,
                "factorizations": evidence.work.factorizations
            },
            "baseline": evaluation_json(&baseline),
            "model_limitation": "Educational linear-elastic pin-jointed 2-D truss; not a code-compliant structural design.",
            "artifacts": {
                "oracle": "oracle.csv",
                "condition_sensitivity": "condition_sensitivity.csv"
            }
        }),
    )
}

/// Write the scalar equal-budget comparison and selected member table.
pub fn write_so(
    metadata: &RunMetadata<'_>,
    seed: &Evaluation,
    arms: &[SoArmResult],
) -> Result<(), Box<dyn Error>> {
    if arms.is_empty() {
        return Err("SO artifact writer needs at least one arm".into());
    }
    fs::create_dir_all(metadata.directory)?;
    let mut convergence = String::from("optimizer,evaluations,elapsed_seconds,best_objective\n");
    let mut summary = String::from(
        "optimizer,feasible,objective,mass_kg,active_count,metrics_available,rcond,max_stress_ratio,max_buckling_ratio,max_displacement_m,constraint_disconnected,constraint_mechanism,constraint_conditioning,constraint_stress,constraint_buckling,constraint_displacement,delta_vs_seed,controls\n",
    );
    let seed_metrics = seed.metrics.as_ref();
    let seed_constraints = constraints(seed);
    writeln!(
        summary,
        "seed,{},{:.17},{:.17},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},\"{}\"",
        usize::from(seed.feasible()),
        seed.objective,
        seed.mass_kg,
        seed.active_count,
        usize::from(seed_metrics.is_some()),
        seed_metrics.map_or(f64::NAN, |value| value.rcond),
        seed_metrics.map_or(f64::NAN, |value| value.max_stress_ratio),
        seed_metrics.map_or(f64::NAN, |value| value.max_buckling_ratio),
        seed_metrics.map_or(f64::NAN, |value| value.max_displacement_m),
        seed_constraints[0],
        seed_constraints[1],
        seed_constraints[2],
        seed_constraints[3],
        seed_constraints[4],
        seed_constraints[5],
        0.0,
        controls(&seed.controls)
    )?;
    for arm in arms {
        if arm.improvements.is_empty() {
            writeln!(
                convergence,
                "{},{},{:.17},{:.17}",
                arm.optimizer.name(),
                arm.actual_evaluations,
                arm.elapsed.as_secs_f64(),
                arm.best.objective
            )?;
        } else {
            for row in &arm.improvements {
                writeln!(
                    convergence,
                    "{},{},{:.17},{:.17}",
                    arm.optimizer.name(),
                    row.evaluations,
                    row.elapsed_seconds,
                    row.value
                )?;
            }
        }
        let metrics = arm.best.metrics.as_ref();
        let values = constraints(&arm.best);
        writeln!(
            summary,
            "{},{},{:.17},{:.17},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},\"{}\"",
            arm.optimizer.name(),
            usize::from(arm.best.feasible()),
            arm.best.objective,
            arm.best.mass_kg,
            arm.best.active_count,
            usize::from(metrics.is_some()),
            metrics.map_or(f64::NAN, |value| value.rcond),
            metrics.map_or(f64::NAN, |value| value.max_stress_ratio),
            metrics.map_or(f64::NAN, |value| value.max_buckling_ratio),
            metrics.map_or(f64::NAN, |value| value.max_displacement_m),
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            arm.best.objective - seed.objective,
            controls(&arm.best.controls)
        )?;
    }
    let selected = arms
        .iter()
        .filter(|arm| arm.best.feasible())
        .min_by(|left, right| left.best.mass_kg.total_cmp(&right.best.mass_kg))
        .map(|arm| &arm.best)
        .unwrap_or(seed);
    let ground = GroundStructure::reference();
    let selected_metrics = selected
        .metrics
        .as_ref()
        .ok_or("selected SO design lacks response metrics")?;
    let catalogue = sections();
    let mut members = String::from(
        "active_index,member_index,node_a,node_b,x_a_m,y_a_m,x_b_m,y_b_m,length_m,section_index,section_name,axial_force_n,utilization\n",
    );
    for (active_index, active) in selected.design.active.iter().enumerate() {
        let member = ground.members[active.member_index];
        let a = selected.design.nodes[member.a];
        let b = selected.design.nodes[member.b];
        writeln!(
            members,
            "{active_index},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{},{:.17},{:.17}",
            active.member_index,
            member.a,
            member.b,
            a.x,
            a.y,
            b.x,
            b.y,
            (b.x - a.x).hypot(b.y - a.y),
            active.section_index,
            catalogue[active.section_index].name,
            selected_metrics.member_forces_n[active_index],
            selected_metrics.member_utilizations[active_index]
        )?;
    }
    write(&metadata.directory.join("arms.csv"), &summary)?;
    write(&metadata.directory.join("convergence.csv"), &convergence)?;
    write(&metadata.directory.join("best_members.csv"), &members)?;
    let arm_json = arms
        .iter()
        .map(|arm| {
            json!({
                "optimizer": arm.optimizer.name(),
                "requested_evaluations": arm.requested_evaluations,
                "actual_evaluations": arm.actual_evaluations,
                "completed_retries": arm.completed_retries,
                "elapsed_seconds": arm.elapsed.as_secs_f64(),
                "best": evaluation_json(&arm.best),
                "work": {
                    "candidate_evaluations": arm.work.candidate_evaluations,
                    "fem_solves": arm.work.fem_solves,
                    "factorizations": arm.work.factorizations
                }
            })
        })
        .collect::<Vec<_>>();
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "truss-sizing",
            "formulation": "so-comparison",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": arms.iter().map(|arm| arm.requested_evaluations).sum::<u64>(),
            "actual_evaluations": arms.iter().map(|arm| arm.actual_evaluations).sum::<u64>(),
            "elapsed_seconds": arms.iter().map(|arm| arm.elapsed.as_secs_f64()).sum::<f64>(),
            "objectives": [{"column": "objective", "label": "Penalized structural mass", "unit": "kg"}],
            "constraints": [
                {"column": "constraint_disconnected", "feasible": "<= 0"},
                {"column": "constraint_mechanism", "feasible": "<= 0"},
                {"column": "constraint_conditioning", "feasible": "<= 0"},
                {"column": "constraint_stress", "feasible": "<= 0"},
                {"column": "constraint_buckling", "feasible": "<= 0"},
                {"column": "constraint_displacement", "feasible": "<= 0"}
            ],
            "descriptors": [],
            "seed_baseline": evaluation_json(seed),
            "selected": evaluation_json(selected),
            "arms": arm_json,
            "model_limitation": "Educational linear-elastic pin-jointed 2-D truss; not a code-compliant structural design.",
            "artifacts": {
                "arms": "arms.csv",
                "convergence": "convergence.csv",
                "best_members": "best_members.csv"
            }
        }),
    )
}

/// Write descriptor observations, registered gate metrics, and verdict.
pub fn write_pilot(metadata: &RunMetadata<'_>, result: &PilotResult) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut csv = String::from(
        "arm,observation,generator,mass_kg,active_count,depth_to_span_train,depth_to_span_holdout,utilization_spread_train,utilization_spread_holdout,survival_train,survival_holdout,controls\n",
    );
    for row in &result.rows {
        writeln!(
            csv,
            "{},{},{},{:.17},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},\"{}\"",
            row.arm,
            row.observation,
            row.generator.name(),
            row.mass_kg,
            row.active_count,
            row.depth_to_span_train,
            row.depth_to_span_holdout,
            row.utilization_spread_train,
            row.utilization_spread_holdout,
            row.survival_train,
            row.survival_holdout,
            controls(&row.controls)
        )?;
    }
    let mut markdown = format!(
        "# Descriptor-pilot verdict\n\n- protocol revision: `{}`\n- decision: `{}`\n- feasible observations: {} / {}\n\n## Frozen generator mixture\n",
        PILOT_PROTOCOL_REVISION,
        result.decision.name(),
        result.feasible,
        result.attempted
    );
    for generator in &result.generators {
        writeln!(
            markdown,
            "- `{}`: {} / {} feasible; per-arm feasible {:?} from attempts {:?}",
            generator.name,
            generator.feasible(),
            generator.attempted(),
            generator.feasible_by_arm,
            generator.attempted_by_arm
        )?;
    }
    markdown.push_str("\n## Registered descriptor gates\n");
    for pair in &result.pairs {
        writeln!(
            markdown,
            "- {}: passed={}, bounds={:?} to {:?}, reachable={:?} to {:?}, lower clipping={:?}, upper clipping={:?}, rho={:.6}, minimum arm coverage={:.3}%, holdout retention={:.3}%",
            pair.name,
            pair.passed,
            pair.lower_bound,
            pair.upper_bound,
            pair.reachable_min,
            pair.reachable_max,
            pair.lower_clipping,
            pair.upper_clipping,
            pair.spearman,
            100.0 * pair.minimum_arm_coverage,
            100.0 * pair.holdout_niche_retention
        )?;
    }
    write(&metadata.directory.join("pilot.csv"), &csv)?;
    write(&metadata.directory.join("pilot.md"), &markdown)?;
    let pairs = result
        .pairs
        .iter()
        .map(|pair| {
            json!({
                "name": pair.name,
                "lower_bound": pair.lower_bound,
                "upper_bound": pair.upper_bound,
                "reachable_min": pair.reachable_min,
                "reachable_max": pair.reachable_max,
                "spearman": pair.spearman,
                "lower_clipping": pair.lower_clipping,
                "upper_clipping": pair.upper_clipping,
                "arm_coverage": pair.arm_coverage,
                "minimum_arm_coverage": pair.minimum_arm_coverage,
                "holdout_niche_retention": pair.holdout_niche_retention,
                "coarse_holdout_niche_retention": pair.coarse_holdout_niche_retention,
                "passed": pair.passed
            })
        })
        .collect::<Vec<_>>();
    let generators = result
        .generators
        .iter()
        .map(|generator| {
            json!({
                "name": generator.name,
                "attempted": generator.attempted(),
                "feasible": generator.feasible(),
                "attempted_by_arm": generator.attempted_by_arm,
                "feasible_by_arm": generator.feasible_by_arm
            })
        })
        .collect::<Vec<_>>();
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "truss-sizing",
            "formulation": "descriptor-pilot",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.attempted,
            "actual_evaluations": result.attempted,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [],
            "descriptors": [
                {"pair": "D1", "columns": ["depth_to_span_train", "survival_train"]},
                {"pair": "D2", "columns": ["utilization_spread_train", "survival_train"]},
                {"pair": "D3", "columns": ["active_count", "mass_kg"]}
            ],
            "qd": {
                "capacity": PUBLICATION_CAPACITY,
                "grid_shape": [12, 10],
                "pilot_protocol_revision": PILOT_PROTOCOL_REVISION,
                "generator_mixture": {
                    "broad_uniform_fraction": 1.0 / BROAD_UNIFORM_STRIDE as f64,
                    "policy": "Every fourth attempt uses uniform topology ranks and node offsets at maximum cardinality with conservative sections; all others use structured local perturbations.",
                    "components": generators
                },
                "bound_revision": "Protocol v1 used survival [0,1] and a local generator. Protocol v2 reports a mixed-generator calibration, then freezes round [0,0.30] survival bounds and a [0,0.30] D2 utilization-spread bound. Exact zero-survival removals remain lower-bound clipping.",
                "decision": result.decision.name(),
                "pairs": pairs
            },
            "work": {
                "candidate_evaluations": result.work.candidate_evaluations,
                "fem_solves": result.work.fem_solves,
                "factorizations": result.work.factorizations
            },
            "artifacts": {"pilot": "pilot.csv", "verdict": "pilot.md"}
        }),
    )
}

/// Write the explicit skipped-QD manifest required by the pilot gate.
pub fn write_qd(metadata: &RunMetadata<'_>, outcome: &QdOutcome) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let QdOutcome::Skipped { reason } = outcome;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "truss-sizing",
            "formulation": "quality-diversity",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "status": "skipped",
            "reason": reason,
            "requested_evaluations": 0,
            "actual_evaluations": null,
            "elapsed_seconds": 0.0,
            "objectives": [],
            "descriptors": [],
            "artifacts": {}
        }),
    )
}

/// Write constrained-MODE Pareto points and convergence.
pub fn write_mo(metadata: &RunMetadata<'_>, result: &MoResult) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut pareto = String::from(
        "point_id,feasible,selected,objective_mass_kg,objective_displacement_m,objective_redundancy_degradation,objective_active_count,constraint_disconnected,constraint_mechanism,constraint_conditioning,constraint_stress,constraint_buckling,carbon_kg_co2e_indicative,removal_survival,failed_removals,controls\n",
    );
    for (point_id, point) in result.pareto.iter().enumerate() {
        let values = constraints(&point.evaluation);
        let redundancy = point.evaluation.redundancy.as_ref();
        writeln!(
            pareto,
            "{point_id},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},\"{}\"",
            usize::from(point.evaluation.feasible()),
            usize::from(point.selected),
            point.objectives[0],
            point.objectives[1],
            point.objectives[2],
            point.objectives[3],
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            point.evaluation.carbon_kg_co2e,
            redundancy.map_or(f64::NAN, |value| value.survival),
            redundancy.map_or(0, |value| value.failed_removals),
            controls(&point.evaluation.controls)
        )?;
    }
    let mut convergence = String::from(
        "evaluations,elapsed_seconds,feasible_population,pareto_population,best_quality\n",
    );
    for row in &result.progress {
        writeln!(
            convergence,
            "{},{:.17},{},{},{:.17}",
            row.evaluations,
            row.elapsed_seconds,
            row.feasible_population,
            row.pareto_population,
            row.best_quality
        )?;
    }
    write(&metadata.directory.join("pareto.csv"), &pareto)?;
    write(&metadata.directory.join("convergence.csv"), &convergence)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "truss-sizing",
            "formulation": "mo",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [
                {"column": "objective_mass_kg", "label": "Structural mass", "unit": "kg"},
                {"column": "objective_displacement_m", "label": "Maximum displacement", "unit": "m"},
                {"column": "objective_redundancy_degradation", "label": "Worst removal compliance degradation", "unit": "ratio"},
                {"column": "objective_active_count", "label": "Active members", "unit": "count"}
            ],
            "constraints": [
                {"column": "constraint_disconnected", "feasible": "<= 0"},
                {"column": "constraint_mechanism", "feasible": "<= 0"},
                {"column": "constraint_conditioning", "feasible": "<= 0"},
                {"column": "constraint_stress", "feasible": "<= 0"},
                {"column": "constraint_buckling", "feasible": "<= 0"}
            ],
            "descriptors": [],
            "convergence_metrics": ["feasible_population", "pareto_population", "best_quality"],
            "work": {
                "candidate_evaluations": result.work.candidate_evaluations,
                "fem_solves": result.work.fem_solves,
                "factorizations": result.work.factorizations
            },
            "artifacts": {"pareto": "pareto.csv", "convergence": "convergence.csv"}
        }),
    )
}

/// Write the frozen protocol shared by all modes.
pub fn write_protocol(
    path: &Path,
    protocol: Protocol,
    seed: u64,
    workers: i32,
) -> Result<(), Box<dyn Error>> {
    write_json(
        path,
        &json!({
            "schema_version": 1,
            "tutorial": "truss-sizing",
            "seed": seed,
            "workers": workers,
            "ground_structure": {
                "nodes": 18,
                "candidate_members": 75,
                "movable_nodes": 10,
                "decision_dimension": 171,
                "active_member_bounds": [8, 40]
            },
            "budgets": {
                "so_evaluations_per_arm": protocol.so_evaluations,
                "so_retries_per_arm": protocol.so_retries,
                "pilot_attempts_per_arm": protocol.pilot_per_arm,
                "mo_evaluations": protocol.mo_evaluations,
                "mo_population": protocol.mo_population
            },
            "rcond_min": RCOND_MIN,
            "model_limitation": "Educational linear-elastic pin-jointed 2-D truss; not a code-compliant structural design."
        }),
    )
}
