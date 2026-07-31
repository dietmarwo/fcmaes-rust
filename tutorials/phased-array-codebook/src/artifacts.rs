//! Machine-readable tutorial artifacts.

use std::error::Error;
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use num_complex::Complex64;
use serde_json::json;

use crate::archive_grid::ArchiveGrid;
use crate::decode::decode_beam;
use crate::geometry::GeometryResult;
use crate::kernel::field_direct;
use crate::mo::MoResult;
use crate::pilot::{
    DESCRIPTOR_LOWER, DESCRIPTOR_UPPER, PUBLICATION_CAPACITY, PilotRow, PilotSummary,
};
use crate::qd::QdResult;
use crate::so::{BeamContext, ELEMENTS, SoArmResult, analytic_seed, evaluate_beam};

/// Metadata common to schema-v1 manifests.
pub struct RunMetadata<'a> {
    /// Artifact directory.
    pub directory: &'a Path,
    /// Replay command.
    pub command: &'a str,
    /// Root seed.
    pub seed: u64,
    /// Candidate workers.
    pub workers: i32,
    /// Angular samples.
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

fn relative_db(value: Complex64, peak: f64) -> f64 {
    if value.norm() <= 0.0 {
        -300.0
    } else {
        (20.0 * (value.norm() / peak).log10()).max(-300.0)
    }
}

/// Write scalar comparison, convergence, decoded best controls, and pattern cuts.
pub fn write_so(
    metadata: &RunMetadata<'_>,
    arms: &[SoArmResult],
    requested_deg: f64,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut convergence = String::from("optimizer,evaluations,elapsed_seconds,best_objective\n");
    let mut best = String::from(
        "optimizer,feasible,objective,peak_deg,hpbw_deg,nominal_psll_db,worst_psll_db,taper_efficiency,constraint_pointing,constraint_psll,constraint_kernel,delta_vs_seed,phase_codes,attenuator_codes\n",
    );
    // Every retry starts from `analytic_seed` at a taper drawn from
    // [0.35, 0.95]. The published baseline is the best seed over a
    // deterministic sweep of that same range, so it upper-bounds what seeding
    // alone provides. An arm whose `delta_vs_seed` is zero did not improve on
    // its own starting point.
    let seed_context = BeamContext::stage_a(metadata.points);
    let seed = (0..13)
        .filter_map(|index| {
            let taper = 0.35 + 0.6 * f64::from(index) / 12.0;
            evaluate_beam(
                &analytic_seed(requested_deg, taper),
                &seed_context,
                requested_deg,
            )
        })
        .min_by(|left, right| left.objective.total_cmp(&right.objective))
        .ok_or("no analytic seed could be replayed for the baseline row")?;
    writeln!(
        best,
        "seed,{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},\"{}\",\"{}\"",
        usize::from(
            seed.constraint_pointing <= 0.0
                && seed.constraint_psll <= 0.0
                && seed.constraint_kernel <= 0.0
        ),
        seed.objective,
        seed.robust.nominal.peak_theta_deg,
        seed.robust.nominal.hpbw_deg,
        seed.robust.nominal.psll_db,
        seed.robust.worst_psll_db,
        seed.robust.nominal.taper_efficiency,
        seed.constraint_pointing,
        seed.constraint_psll,
        seed.constraint_kernel,
        0.0,
        seed.excitation
            .phase_codes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(";"),
        seed.excitation
            .attenuator_codes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(";")
    )?;
    let seed_objective = seed.objective;
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
        let feasible = arm.best.constraint_pointing <= 0.0
            && arm.best.constraint_psll <= 0.0
            && arm.best.constraint_kernel <= 0.0;
        writeln!(
            best,
            "{},{},{},{},{},{},{},{},{},{},{},{:.17},\"{}\",\"{}\"",
            arm.optimizer.name(),
            usize::from(feasible),
            arm.best.objective,
            arm.best.robust.nominal.peak_theta_deg,
            arm.best.robust.nominal.hpbw_deg,
            arm.best.robust.nominal.psll_db,
            arm.best.robust.worst_psll_db,
            arm.best.robust.nominal.taper_efficiency,
            arm.best.constraint_pointing,
            arm.best.constraint_psll,
            arm.best.constraint_kernel,
            arm.best.objective - seed_objective,
            join_codes(&arm.best.excitation.phase_codes),
            join_codes(&arm.best.excitation.attenuator_codes)
        )?;
    }
    let selected = arms
        .iter()
        .filter(|arm| {
            arm.best.constraint_pointing <= 0.0
                && arm.best.constraint_psll <= 0.0
                && arm.best.constraint_kernel <= 0.0
        })
        .min_by(|left, right| left.best.objective.total_cmp(&right.best.objective))
        .or_else(|| {
            arms.iter()
                .min_by(|left, right| left.best.objective.total_cmp(&right.best.objective))
        })
        .ok_or("SO artifact writer needs at least one arm")?;
    let context = BeamContext::stage_a(metadata.points);
    let uniform = decode_beam(
        &analytic_seed(requested_deg, 0.0),
        ELEMENTS,
        &context.quantization,
    )?;
    let uniform_field = field_direct(&context.array, &context.grid, &uniform.weights)?;
    let optimized_field = field_direct(
        &context.array,
        &context.grid,
        &selected.best.excitation.weights,
    )?;
    const CHEB_30: [f64; ELEMENTS] = [
        0.290_988_871_257_774_8,
        0.317_296_191_540_353_9,
        0.455_688_938_631_810_3,
        0.601_756_006_455_069_8,
        0.742_386_845_754_509_2,
        0.863_659_696_720_422_3,
        0.952_789_152_816_859_2,
        1.0,
        1.0,
        0.952_789_152_816_859_2,
        0.863_659_696_720_422_3,
        0.742_386_845_754_509_2,
        0.601_756_006_455_069_8,
        0.455_688_938_631_810_3,
        0.317_296_191_540_353_9,
        0.290_988_871_257_774_8,
    ];
    let target_u = requested_deg.to_radians().sin();
    let center = (ELEMENTS - 1) as f64 / 2.0;
    let chebyshev = CHEB_30
        .iter()
        .enumerate()
        .map(|(index, amplitude)| {
            Complex64::from_polar(*amplitude, -PI * (index as f64 - center) * target_u)
        })
        .collect::<Vec<_>>();
    let chebyshev_field = field_direct(&context.array, &context.grid, &chebyshev)?;
    let uniform_peak = uniform_field
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let optimized_peak = optimized_field
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let chebyshev_peak = chebyshev_field
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let mut pattern = String::from("angle_deg,uniform_db,optimized_db,chebyshev_reference_db\n");
    let mut failure_fields = Vec::with_capacity(ELEMENTS);
    for failed in 0..ELEMENTS {
        let mut weights = selected.best.excitation.weights.clone();
        weights[failed] = Complex64::new(0.0, 0.0);
        failure_fields.push(field_direct(&context.array, &context.grid, &weights)?);
    }
    let mut envelope = String::from("angle_deg,nominal_db,envelope_min_db,envelope_max_db\n");
    for index in 0..context.grid.len() {
        let angle = context.grid.directions[index]
            .u
            .clamp(-1.0, 1.0)
            .asin()
            .to_degrees();
        writeln!(
            pattern,
            "{angle},{},{},{}",
            relative_db(uniform_field[index], uniform_peak),
            relative_db(optimized_field[index], optimized_peak),
            relative_db(chebyshev_field[index], chebyshev_peak)
        )?;
        let levels = failure_fields
            .iter()
            .map(|field| relative_db(field[index], optimized_peak))
            .collect::<Vec<_>>();
        writeln!(
            envelope,
            "{angle},{},{},{}",
            relative_db(optimized_field[index], optimized_peak),
            levels.iter().copied().fold(f64::INFINITY, f64::min),
            levels.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        )?;
    }
    write(&metadata.directory.join("convergence.csv"), &convergence)?;
    write(&metadata.directory.join("best.csv"), &best)?;
    write(&metadata.directory.join("pattern.csv"), &pattern)?;
    write(&metadata.directory.join("failure_envelope.csv"), &envelope)?;
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
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "phased-array-codebook",
            "formulation": "so-comparison",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": arms.iter().map(|arm| arm.requested_evaluations).sum::<u64>(),
            "actual_evaluations": arms.iter().map(|arm| arm.actual_evaluations).sum::<u64>(),
            "elapsed_seconds": arms.iter().map(|arm| arm.elapsed.as_secs_f64()).sum::<f64>(),
            "objectives": [{"column": "best_objective", "label": "Worst training PSLL plus feasibility penalties", "unit": "dB"}],
            "constraints": [
                {"column": "constraint_pointing", "feasible": "<= 0", "unit": "deg"},
                {"column": "constraint_psll", "feasible": "<= 0", "unit": "dB"},
                {"column": "constraint_kernel", "feasible": "<= 0", "unit": "scaled"}
            ],
            "descriptors": [],
            "angular_points": metadata.points,
            "requested_steer_deg": requested_deg,
            "arms": arm_json,
            "artifacts": {
                "best": "best.csv",
                "convergence": "convergence.csv",
                "pattern": "pattern.csv",
                "failure_envelope": "failure_envelope.csv"
            }
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
        "seed,sample,peak_deg,hpbw_deg,holdout_peak_deg,holdout_hpbw_deg,taper_efficiency,holdout_taper_efficiency,active_count,worst_psll_db,holdout_worst_psll_db\n",
    );
    for row in rows {
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{},{}",
            row.seed,
            row.sample,
            row.descriptors[0],
            row.descriptors[1],
            row.holdout_descriptors[0],
            row.holdout_descriptors[1],
            row.taper_efficiency,
            row.holdout_taper_efficiency,
            row.active_count,
            row.worst_psll_db,
            row.holdout_worst_psll_db
        )?;
    }
    let markdown = format!(
        "# Descriptor-pilot verdict\n\n- decision: `{}`\n- feasible candidates: {} / {} ({:.3}%)\n- D1 rank correlation: {:.6}\n- D1 coverage: {:.3}%\n- D1 holdout niche retention: {:.3}%\n- D2 coverage: {:.3}%\n- D2 holdout niche retention: {:.3}%\n- reason: {}\n",
        summary.decision.label(),
        summary.feasible_candidates,
        summary.attempted_candidates,
        100.0 * summary.feasible_fraction,
        summary.d1_rank_correlation,
        100.0 * summary.coverage,
        100.0 * summary.holdout_niche_retention,
        100.0 * summary.d2_coverage,
        100.0 * summary.d2_holdout_niche_retention,
        summary.reason
    );
    write(&metadata.directory.join("pilot.csv"), &csv)?;
    write(&metadata.directory.join("pilot.md"), &markdown)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "phased-array-codebook",
            "formulation": "descriptor-pilot",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": summary.attempted_candidates,
            "actual_evaluations": summary.attempted_candidates,
            "elapsed_seconds": summary.elapsed_seconds,
            "objectives": [],
            "descriptors": [
                {"column": "peak_deg", "label": "Measured peak direction", "unit": "deg"},
                {"column": "hpbw_deg", "label": "Measured half-power beamwidth", "unit": "deg"}
            ],
            "qd": {
                "grid_shape": ArchiveGrid::new(PUBLICATION_CAPACITY).rectangular_shape(),
                "capacity": PUBLICATION_CAPACITY,
                "decision": summary.decision,
                "summary": summary
            },
            "artifacts": {"pilot": "pilot.csv", "verdict": "pilot.md"}
        }),
    )
}

fn join_codes(codes: &[u32]) -> String {
    codes
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(";")
}

/// Write MAP-Elites archive, machine codebook, progress, and migration.
pub fn write_qd(metadata: &RunMetadata<'_>, result: &QdResult) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut archive = String::from(
        "niche_id,grid_x,grid_y,quality_train,quality_validation,descriptor_peak_deg_train,descriptor_hpbw_deg_train,descriptor_peak_deg_validation,descriptor_hpbw_deg_validation,visit_count,retained_niche,decision_phase_codes,decision_attenuator_codes,constraint_robust_psll\n",
    );
    let mut codebook = String::from(
        "niche_id,peak_theta_deg,hpbw_deg,psll_db,worst_psll_db,holdout_worst_psll_db,taper_efficiency,active_count,phase_codes,attenuator_codes\n",
    );
    let mut migration = String::from(
        "niche_id,train_peak_deg,train_hpbw_deg,holdout_peak_deg,holdout_hpbw_deg,moved\n",
    );
    let layout = ArchiveGrid::new(result.capacity);
    for entry in &result.entries {
        let retained = layout.niche(
            entry.holdout_descriptors,
            DESCRIPTOR_LOWER,
            DESCRIPTOR_UPPER,
        ) == Some(entry.niche);
        writeln!(
            archive,
            "{},{},{},{},{},{},{},{},{},{},{},\"{}\",\"{}\",{}",
            entry.niche,
            entry.grid_x,
            entry.grid_y,
            entry.quality,
            10_f64.powf(entry.holdout.worst_psll_db / 20.0),
            entry.descriptors[0],
            entry.descriptors[1],
            entry.holdout_descriptors[0],
            entry.holdout_descriptors[1],
            entry.visits,
            usize::from(retained),
            join_codes(&entry.excitation.phase_codes),
            join_codes(&entry.excitation.attenuator_codes),
            entry.robust.worst_psll_db + 10.0
        )?;
        writeln!(
            codebook,
            "{},{},{},{},{},{},{},{},\"{}\",\"{}\"",
            entry.niche,
            entry.descriptors[0],
            entry.descriptors[1],
            entry.robust.nominal.psll_db,
            entry.robust.worst_psll_db,
            entry.holdout.worst_psll_db,
            entry.robust.nominal.taper_efficiency,
            entry
                .excitation
                .active
                .iter()
                .filter(|value| **value)
                .count(),
            join_codes(&entry.excitation.phase_codes),
            join_codes(&entry.excitation.attenuator_codes)
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
        "evaluations,elapsed_seconds,coverage,qd_score,best_quality,invalid_fraction,infeasible_fraction\n",
    );
    for row in &result.progress {
        writeln!(
            convergence,
            "{},{},{},{},{},{},{}",
            row.evaluations,
            row.elapsed_seconds,
            row.coverage,
            row.qd_score,
            row.best_quality,
            row.invalid_fraction,
            row.infeasible_fraction
        )?;
    }
    write(&metadata.directory.join("qd_archive.csv"), &archive)?;
    write(&metadata.directory.join("codebook.csv"), &codebook)?;
    write(
        &metadata.directory.join("holdout_migration.csv"),
        &migration,
    )?;
    write(&metadata.directory.join("qd_convergence.csv"), &convergence)?;
    let mut qd_metadata = json!({
        "capacity": result.capacity,
        "occupied": result.entries.len(),
        "clamped_descriptors": result.clamped_descriptors,
        "invalid_evaluations": result.invalid_evaluations,
        "infeasible_evaluations": result.infeasible_evaluations
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
            "tutorial": "phased-array-codebook",
            "formulation": "quality-diversity",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [{"column": "quality_train", "label": "Worst-case sidelobe ratio", "unit": "linear"}],
            "descriptors": [
                {"column": "descriptor_peak_deg_train", "label": "Measured peak direction", "unit": "deg"},
                {"column": "descriptor_hpbw_deg_train", "label": "Measured HPBW", "unit": "deg"}
            ],
            "qd": qd_metadata,
            "artifacts": {
                "archive": "qd_archive.csv",
                "codebook": "codebook.csv",
                "convergence": "qd_convergence.csv",
                "holdout_migration": "holdout_migration.csv"
            }
        }),
    )
}

/// Write constrained MODE Pareto evidence.
pub fn write_mo(metadata: &RunMetadata<'_>, result: &MoResult) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut pareto = String::from(
        "point_id,feasible,selected,objective_negative_peak_gain_db,objective_psll_db,objective_active_count,objective_robustness_margin_db,constraint_null_db,constraint_kernel,peak_deg,hpbw_deg,phase_codes,attenuator_codes\n",
    );
    for (point_id, point) in result.pareto.iter().enumerate() {
        let evaluation = &point.evaluation;
        writeln!(
            pareto,
            "{point_id},1,{},{},{},{},{},{},{},{},{},\"{}\",\"{}\"",
            usize::from(point.selected),
            evaluation.objectives[0],
            evaluation.objectives[1],
            evaluation.objectives[2],
            evaluation.objectives[3],
            evaluation.constraints[0],
            evaluation.constraints[1],
            evaluation.robust.nominal.peak_theta_deg,
            evaluation.robust.nominal.hpbw_deg,
            join_codes(&evaluation.excitation.phase_codes),
            join_codes(&evaluation.excitation.attenuator_codes)
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
    write(&metadata.directory.join("pareto.csv"), &pareto)?;
    write(&metadata.directory.join("convergence.csv"), &convergence)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "phased-array-codebook",
            "formulation": "constrained-mo",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [
                {"column": "objective_negative_peak_gain_db", "label": "Negative peak gain", "unit": "dB"},
                {"column": "objective_psll_db", "label": "Nominal PSLL", "unit": "dB"},
                {"column": "objective_active_count", "label": "Active elements", "unit": "count"},
                {"column": "objective_robustness_margin_db", "label": "Robustness degradation", "unit": "dB"}
            ],
            "constraints": [
                {"column": "constraint_null_db", "feasible": "<= 0"},
                {"column": "constraint_kernel", "feasible": "<= 0"}
            ],
            "descriptors": [],
            "pareto_points": result.pareto.len(),
            "artifacts": {"pareto": "pareto.csv", "convergence": "convergence.csv"}
        }),
    )
}

/// Write the non-uniform geometry result.
pub fn write_geometry(
    metadata: &RunMetadata<'_>,
    result: &GeometryResult,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(metadata.directory)?;
    let mut best = String::from(
        "objective,psll_db,peak_deg,hpbw_deg,minimum_spacing_lambda,constraint_spacing,positions_lambda,phase_codes,attenuator_codes\n",
    );
    writeln!(
        best,
        "{},{},{},{},{},{},\"{}\",\"{}\",\"{}\"",
        result.best.objective,
        result.best.metrics.psll_db,
        result.best.metrics.peak_theta_deg,
        result.best.metrics.hpbw_deg,
        result.best.array.minimum_spacing_lambda(),
        result.best.constraint_spacing,
        result
            .best
            .array
            .positions
            .iter()
            .map(|position| position[0].to_string())
            .collect::<Vec<_>>()
            .join(";"),
        join_codes(&result.best.excitation.phase_codes),
        join_codes(&result.best.excitation.attenuator_codes)
    )?;
    let grid = BeamContext::stage_a(metadata.points).grid;
    let field = field_direct(&result.best.array, &grid, &result.best.excitation.weights)?;
    let peak = field.iter().map(|value| value.norm()).fold(0.0, f64::max);
    let mut pattern = String::from("angle_deg,level_db\n");
    for (direction, value) in grid.directions.iter().zip(field) {
        writeln!(
            pattern,
            "{},{}",
            direction.u.asin().to_degrees(),
            relative_db(value, peak)
        )?;
    }
    write(&metadata.directory.join("best.csv"), &best)?;
    write(&metadata.directory.join("pattern.csv"), &pattern)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "phased-array-codebook",
            "formulation": "nonuniform-geometry",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [{"column": "objective", "label": "PSLL plus feasibility penalties", "unit": "dB"}],
            "constraints": [{"column": "constraint_spacing", "feasible": "<= 0", "unit": "lambda"}],
            "descriptors": [],
            "fft_supported": false,
            "artifacts": {"best": "best.csv", "pattern": "pattern.csv"}
        }),
    )
}

/// Write quantization-staircase evidence.
pub fn write_staircase(directory: &Path) -> Result<(), Box<dyn Error>> {
    let context = BeamContext::stage_a(721);
    let mut controls = analytic_seed(20.0, 0.8);
    let mut csv = String::from("coordinate,phase_code,objective\n");
    for sample in 0..=640 {
        controls[0] = sample as f64 / 640.0;
        if let Some(evaluation) = crate::so::evaluate_beam(&controls, &context, 20.0) {
            writeln!(
                csv,
                "{},{},{}",
                controls[0], evaluation.excitation.phase_codes[0], evaluation.objective
            )?;
        }
    }
    write(&directory.join("staircase.csv"), &csv)
}

/// Write kernel/directivity validation evidence.
pub struct ValidationEvidence {
    /// Coarse polar-grid directivity.
    pub coarse_directivity_dbi: f64,
    /// Fine polar-grid directivity.
    pub fine_directivity_dbi: f64,
    /// Fine steering-matrix allocation.
    pub steering_memory_bytes: usize,
    /// Direct linear-cut kernel time.
    pub direct_linear_us: f64,
    /// Direct planar kernel time.
    pub direct_planar_ms: f64,
    /// Optional 1-D FFT time.
    pub fft_linear_us: Option<f64>,
    /// Optional 2-D FFT time.
    pub fft_planar_us: Option<f64>,
}

/// Write kernel/directivity validation evidence.
pub fn write_validation(
    metadata: &RunMetadata<'_>,
    evidence: &ValidationEvidence,
) -> Result<(), Box<dyn Error>> {
    let ValidationEvidence {
        coarse_directivity_dbi,
        fine_directivity_dbi,
        steering_memory_bytes,
        direct_linear_us,
        direct_planar_ms,
        fft_linear_us,
        fft_planar_us,
    } = evidence;
    fs::create_dir_all(metadata.directory)?;
    let csv = format!(
        "metric,value,unit\ncoarse_planar_directivity,{coarse_directivity_dbi},dBi\nfine_planar_directivity,{fine_directivity_dbi},dBi\nsteering_matrix_memory,{steering_memory_bytes},bytes\ndirect_linear,{direct_linear_us},us\ndirect_planar,{direct_planar_ms},ms\nfft_linear,{},us\nfft_planar,{},us\n",
        fft_linear_us.map_or_else(String::new, |value| value.to_string()),
        fft_planar_us.map_or_else(String::new, |value| value.to_string())
    );
    write(&metadata.directory.join("validation.csv"), &csv)?;
    write_json(
        &metadata.directory.join("run.json"),
        &json!({
            "schema_version": 1,
            "tutorial": "phased-array-codebook",
            "formulation": "kernel-validation",
            "command": metadata.command,
            "seed": metadata.seed,
            "workers": metadata.workers,
            "requested_evaluations": 0,
            "actual_evaluations": 0,
            "elapsed_seconds": 0.0,
            "objectives": [],
            "descriptors": [],
            "directivity_convention": "one-sided upper-hemisphere aperture; field is zero behind the array",
            "coarse_directivity_dbi": coarse_directivity_dbi,
            "fine_directivity_dbi": fine_directivity_dbi,
            "steering_memory_bytes": steering_memory_bytes,
            "artifacts": {"validation": "validation.csv", "staircase": "staircase.csv"}
        }),
    )
}
