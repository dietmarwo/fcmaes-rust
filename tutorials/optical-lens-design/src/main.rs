use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use optical_lens_design::{
    CONSTRAINTS, DIMENSION, Evaluation, MoConfig, MoResult, PUBLICATION_GRID_RADIUS, SoConfig,
    SoOptimizer, SoResult, WAVELENGTHS_UM, evaluate, optimize_mo, optimize_so, pupil_points,
    trace_ray, validate_reference,
};
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Validate,
    So,
    Mo,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Preset {
    Smoke,
    Publication,
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    preset: Preset,
    workers: i32,
    seed: u64,
    output: PathBuf,
    write_output: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::All,
            preset: Preset::Smoke,
            workers: 0,
            seed: 42,
            output: PathBuf::from("results"),
            write_output: true,
        }
    }
}

fn usage() {
    println!(
        "Cooke-triplet optimization\n\n\
         cargo run --release -- [OPTIONS]\n\n\
         --mode NAME       validate, so, mo, or all (all)\n\
         --preset NAME     smoke or publication (smoke)\n\
         --workers N       Candidate workers; 0 uses available CPUs (0)\n\
         --seed N          Root optimizer seed (42)\n\
         --output DIR      Artifact root (results)\n\
         --no-output       Run without writing artifacts\n\
         -h, --help        Show this help"
    );
}

fn parse_args() -> Result<Option<Args>, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = |name: &str| {
            arguments
                .next()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        match argument.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--mode" => {
                parsed.mode = match value("--mode")?.as_str() {
                    "validate" => Mode::Validate,
                    "so" => Mode::So,
                    "mo" => Mode::Mo,
                    "all" => Mode::All,
                    other => return Err(format!("unknown mode: {other}").into()),
                }
            }
            "--preset" => {
                parsed.preset = match value("--preset")?.as_str() {
                    "smoke" => Preset::Smoke,
                    "publication" => Preset::Publication,
                    other => return Err(format!("unknown preset: {other}").into()),
                }
            }
            "--workers" => parsed.workers = value("--workers")?.parse()?,
            "--seed" => parsed.seed = value("--seed")?.parse()?,
            "--output" => parsed.output = value("--output")?.into(),
            "--no-output" => parsed.write_output = false,
            other => return Err(format!("unknown option: {other}").into()),
        }
    }
    if parsed.workers < 0 {
        return Err("--workers must be non-negative".into());
    }
    Ok(Some(parsed))
}

fn command_line() -> String {
    let tail = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if tail.is_empty() {
        "cargo run --release".to_owned()
    } else {
        format!("cargo run --release -- {tail}")
    }
}

fn write(path: &Path, contents: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    Ok(())
}

fn evaluation_csv(evaluation: &Evaluation) -> String {
    let mut output = String::from(
        "rms_spot_um,efl_mm,bfl_mm,track_length_mm,glass_volume_mm3,minimum_edge_thickness_mm,lost_rays,total_rays",
    );
    for index in 0..DIMENSION {
        let _ = write!(output, ",decision_{index}");
    }
    output.push('\n');
    let _ = write!(
        output,
        "{},{},{},{},{},{},{},{}",
        evaluation.rms_spot_mm * 1_000.0,
        evaluation.efl_mm,
        evaluation.bfl_mm,
        evaluation.track_length_mm,
        evaluation.glass_volume_mm3,
        evaluation.minimum_edge_thickness_mm,
        evaluation.lost_rays,
        evaluation.total_rays,
    );
    for value in evaluation.design.values {
        let _ = write!(output, ",{value}");
    }
    output.push('\n');
    output
}

fn run_validation(root: &Path, write_output: bool) -> Result<(), Box<dyn Error>> {
    let summary = validate_reference()?;
    if write_output {
        let directory = root.join("validation");
        fs::create_dir_all(&directory)?;
        let mut convergence = String::from("grid_radius,pupil_rays,weighted_rms_spot_um\n");
        for (radius, rays, rms) in &summary.convergence {
            writeln!(convergence, "{radius},{rays},{rms}")?;
        }
        write(&directory.join("ray_convergence.csv"), &convergence)?;
        write(
            &directory.join("summary.json"),
            &(serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "tutorial": "optical-lens-design",
            "reference": {
                "name": "Optiland Tutorial 5c final Cooke triplet",
                "url": "https://optiland.readthedocs.io/en/stable/examples/Tutorial_5c_Optimization_Case_Study.html",
                "published_efl_mm": summary.reference_efl_mm,
                "published_on_axis_rms_mm": summary.reference_on_axis_rms_mm
            },
            "rust": {
                "efl_mm": summary.efl_mm,
                "on_axis_rms_mm": summary.on_axis_rms_mm
            },
            "acceptance": {
                "paraxial_focus_relative_residual_limit": 0.001,
                "efl_relative_error_limit": 0.001,
                "maximum_on_axis_spot_relative_error_limit": 0.20,
                "production_and_two_finer_grid_relative_deviation_limit": 0.03
            },
            "measured": {
                "paraxial_focus_relative_residual": summary.paraxial_focus_relative_residual,
                "efl_relative_error": summary.efl_relative_error,
                "maximum_on_axis_spot_relative_error": summary.maximum_spot_relative_error
            },
            "checks": {
                "paraxial_focus": summary.paraxial_pass,
                "efl": summary.efl_pass,
                "on_axis_spot": summary.spot_pass,
                "ray_convergence": summary.convergence_pass
            },
            "passed": summary.passed(),
            "note": "The on-axis comparison can retain pupil-sampling differences; off-axis values additionally use different entrance-pupil/stop ray-aiming conventions and are not an equality gate."
            }))? + "\n"),
        )?;
    }
    if !summary.passed() {
        return Err("optical validation gate failed".into());
    }
    println!(
        "VALIDATION pass efl={:.6} mm max_on_axis_error={:.3}% convergence_rows={}",
        summary.efl_mm,
        100.0 * summary.maximum_spot_relative_error,
        summary.convergence.len()
    );
    Ok(())
}

fn write_so_artifacts(
    directory: &Path,
    command: &str,
    args: &Args,
    results: &[SoResult],
    grid_radius: usize,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut convergence = String::from("optimizer,evaluations,elapsed_seconds,best_objective\n");
    let mut best = String::from("optimizer,");
    best.push_str(
        evaluation_csv(&results[0].best)
            .lines()
            .next()
            .unwrap_or_default(),
    );
    best.push('\n');
    for result in results {
        for row in &result.improvements {
            writeln!(
                convergence,
                "{},{},{},{}",
                result.optimizer.name(),
                row.evaluations,
                row.elapsed_seconds,
                row.value
            )?;
        }
        let serialized = evaluation_csv(&result.best);
        let row = serialized.lines().nth(1).unwrap_or_default();
        writeln!(best, "{},{}", result.optimizer.name(), row)?;
    }
    write(&directory.join("convergence.csv"), &convergence)?;
    write(&directory.join("best.csv"), &best)?;
    let reference = optical_lens_design::Design::reference();
    let optimized = results
        .iter()
        .min_by(|left, right| {
            left.best
                .scalar_score()
                .total_cmp(&right.best.scalar_score())
        })
        .map(|row| &row.best.design)
        .ok_or("SO result list is empty")?;
    let mut spots = String::from("design,field_deg,wavelength_um,ray_id,x_mm,y_mm\n");
    for (label, design) in [("reference", &reference), ("optimized", optimized)] {
        for field in optical_lens_design::FIELDS_DEG {
            for wavelength in WAVELENGTHS_UM {
                for (ray_id, pupil) in pupil_points(grid_radius).iter().enumerate() {
                    if let Some(hit) = trace_ray(design, pupil[0], pupil[1], field, wavelength) {
                        writeln!(
                            spots,
                            "{label},{field},{wavelength},{ray_id},{},{}",
                            hit[0], hit[1]
                        )?;
                    }
                }
            }
        }
    }
    write(&directory.join("spot_diagrams.csv"), &spots)?;
    let requested = results
        .iter()
        .map(|row| row.requested_evaluations)
        .sum::<u64>();
    let actual = results
        .iter()
        .map(|row| row.actual_evaluations)
        .sum::<u64>();
    let elapsed = results
        .iter()
        .map(|row| row.elapsed.as_secs_f64())
        .sum::<f64>();
    write(
        &directory.join("run.json"),
        &(serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "tutorial": "optical-lens-design",
            "formulation": "so-comparison",
            "command": command,
            "seed": args.seed,
            "workers": args.workers,
            "grid_radius": grid_radius,
            "pupil_rays_per_field_wavelength": pupil_points(grid_radius).len(),
            "requested_evaluations": requested,
            "actual_evaluations": actual,
            "elapsed_seconds": elapsed,
            "objectives": [{"column": "best_objective", "label": "Penalized polychromatic RMS spot", "unit": "um"}],
            "descriptors": [],
            "arms": results.iter().map(|row| json!({
                "optimizer": row.optimizer.name(),
                "requested_evaluations": row.requested_evaluations,
                "actual_evaluations": row.actual_evaluations,
                "elapsed_seconds": row.elapsed.as_secs_f64(),
                "best_objective": row.best.scalar_score(),
                "feasible": row.best.feasible()
            })).collect::<Vec<_>>(),
            "artifacts": {"so_convergence": "convergence.csv", "so_best": "best.csv", "spot_diagrams": "spot_diagrams.csv"}
        }))? + "\n"),
    )
}

fn write_mo_artifacts(
    directory: &Path,
    command: &str,
    args: &Args,
    result: &MoResult,
    grid_radius: usize,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let mut pareto = String::from(
        "point_id,feasible,selected,objective_rms_spot_um,objective_track_length_mm,objective_glass_volume_mm3,constraint_edge_thickness_mm,constraint_efl_mm,constraint_lost_rays",
    );
    for index in 0..DIMENSION {
        write!(pareto, ",decision_{index}")?;
    }
    pareto.push('\n');
    for (point_id, point) in result.pareto.iter().enumerate() {
        let evaluation = &point.evaluation;
        let objectives = evaluation.objectives();
        write!(
            pareto,
            "{point_id},1,{},{},{},{},{},{},{}",
            usize::from(point.selected),
            objectives[0],
            objectives[1],
            objectives[2],
            evaluation.constraints[0],
            evaluation.constraints[1],
            evaluation.constraints[2]
        )?;
        for value in evaluation.design.values {
            write!(pareto, ",{value}")?;
        }
        pareto.push('\n');
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
    write(
        &directory.join("run.json"),
        &(serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "tutorial": "optical-lens-design",
            "formulation": "constrained-mo",
            "command": command,
            "seed": args.seed,
            "workers": args.workers,
            "grid_radius": grid_radius,
            "pupil_rays_per_field_wavelength": pupil_points(grid_radius).len(),
            "initialization": {
                "centre": "disclosed reference prescription",
                "relative_bound_half_width": 0.01,
                "includes_exact_reference": true
            },
            "requested_evaluations": result.requested_evaluations,
            "actual_evaluations": result.actual_evaluations,
            "elapsed_seconds": result.elapsed.as_secs_f64(),
            "objectives": [
                {"column": "objective_rms_spot_um", "label": "Polychromatic RMS spot", "unit": "um"},
                {"column": "objective_track_length_mm", "label": "Optical track length", "unit": "mm"},
                {"column": "objective_glass_volume_mm3", "label": "Glass volume", "unit": "mm3"}
            ],
            "constraints": [
                {"column": "constraint_edge_thickness_mm", "label": "Minimum edge thickness deficit", "unit": "mm", "feasible": "<= 0"},
                {"column": "constraint_efl_mm", "label": "EFL tolerance excess", "unit": "mm", "feasible": "<= 0"},
                {"column": "constraint_lost_rays", "label": "Lost rays", "unit": "rays", "feasible": "<= 0"}
            ],
            "descriptors": [],
            "constraints_count": CONSTRAINTS,
            "pareto_points": result.pareto.len(),
            "artifacts": {"mo_pareto": "pareto.csv", "mo_convergence": "convergence.csv"}
        }))? + "\n"),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let name = match args.preset {
        Preset::Smoke => "smoke",
        Preset::Publication => "publication",
    };
    let root = args.output.join(name);
    let command = command_line();
    run_validation(&root, args.write_output)?;
    if args.mode == Mode::Validate {
        return Ok(());
    }
    let (so_evaluations, retries, mo_evaluations, popsize, grid_radius) = match args.preset {
        Preset::Smoke => (2_000, 4, 16_384, 256, 4),
        Preset::Publication => (30_000, 12, 100_000, 256, PUBLICATION_GRID_RADIUS),
    };
    if matches!(args.mode, Mode::So | Mode::All) {
        let mut results = Vec::new();
        for optimizer in SoOptimizer::ALL {
            let result = optimize_so(
                optimizer,
                &SoConfig {
                    evaluations_per_arm: so_evaluations,
                    retries,
                    workers: args.workers as usize,
                    seed: args.seed,
                    grid_radius,
                },
            )?;
            println!(
                "SO {} spot={:.6} um efl={:.6} feasible={} evals={} wall={:.3}s",
                optimizer.name(),
                result.best.rms_spot_mm * 1_000.0,
                result.best.efl_mm,
                result.best.feasible(),
                result.actual_evaluations,
                result.elapsed.as_secs_f64()
            );
            results.push(result);
        }
        if args.write_output {
            write_so_artifacts(&root.join("so"), &command, &args, &results, grid_radius)?;
        }
    }
    if matches!(args.mode, Mode::Mo | Mode::All) {
        let result = optimize_mo(&MoConfig {
            evaluations: mo_evaluations,
            popsize,
            workers: args.workers,
            seed: args.seed ^ 0xA076_1D64_78BD_642F,
            grid_radius,
        })?;
        println!(
            "MO pareto={} evals={} wall={:.3}s",
            result.pareto.len(),
            result.actual_evaluations,
            result.elapsed.as_secs_f64()
        );
        if args.write_output {
            write_mo_artifacts(&root.join("mo"), &command, &args, &result, grid_radius)?;
        }
    }
    // Keep the initial/reference prescription available even for --no-output
    // smoke checks, which also exercises the full metric path.
    let reference = evaluate(&optical_lens_design::REFERENCE_DESIGN, grid_radius)
        .ok_or("reference design evaluation failed")?;
    println!(
        "REFERENCE spot={:.6} um efl={:.6} rays={}/{}",
        reference.rms_spot_mm * 1_000.0,
        reference.efl_mm,
        reference.total_rays - reference.lost_rays,
        reference.total_rays
    );
    Ok(())
}
