use std::error::Error;
use std::path::PathBuf;

use rapier_trebuchet::{
    Design, INITIAL_DESIGN, MoProgress, MultiOptions, ParetoPoint, QdOptions, ScalarOptions,
    SimulationConfig, SimulationResult, optimize_multi, optimize_qd, optimize_scalar, simulate,
    write_artifacts, write_qd_artifacts,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    Single,
    Multi,
    Both,
    Qd,
    All,
    Simulate,
}

#[derive(Debug)]
struct Args {
    mode: RunMode,
    target: f64,
    evaluations: u64,
    retries: usize,
    workers: usize,
    depth: i32,
    mo_evaluations: usize,
    popsize: usize,
    qd_evaluations: usize,
    qd_capacity: usize,
    qd_chunk_size: usize,
    seed: u64,
    output: PathBuf,
    write_output: bool,
    design: Option<Design>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: RunMode::Both,
            target: 35.0,
            evaluations: 5_000,
            retries: 8,
            workers: 0,
            depth: 6,
            mo_evaluations: 20_000,
            popsize: 128,
            qd_evaluations: 20_000,
            qd_capacity: 400,
            qd_chunk_size: 128,
            seed: 42,
            output: PathBuf::from("results"),
            write_output: true,
            design: None,
        }
    }
}

fn usage() {
    println!(
        "Rapier trebuchet optimization\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --mode NAME              single, multi, qd, both, all, or simulate (both)\n\
         --target METRES          Landing target in metres (35)\n\
         --evaluations N          BiteOpt evaluations per retry (5000)\n\
         --retries N              Independent BiteOpt retries (8)\n\
         --workers N              fcmaes workers; 0 uses available CPUs (0)\n\
         --depth N                BiteOpt deep populations, 1..36 (6)\n\
         --mo-evaluations N       Total requested MODE evaluations (20000)\n\
         --popsize N              MODE population size (128)\n\
         --qd-evaluations N       Total requested MAP-Elites evaluations (20000)\n\
         --qd-capacity N          Square MAP-Elites archive capacity (400)\n\
         --qd-chunk-size N        Even QD evaluation batch size (128)\n\
         --seed N                 Optimizer root seed (42)\n\
         --output DIR             CSV/replay output directory (results)\n\
         --no-output              Do not write CSV or replay files\n\
         --x CSV                  Eight physical values for simulate mode\n\
         -h, --help               Show this help"
    );
}

fn take_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_design(value: &str) -> Result<Design, Box<dyn Error>> {
    let values = value
        .split(',')
        .map(|field| field.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Design::from_slice(&values)?)
}

fn parse_args() -> Result<Option<Args>, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--mode" => {
                parsed.mode = match take_value(&mut args, "--mode")?.as_str() {
                    "single" => RunMode::Single,
                    "multi" => RunMode::Multi,
                    "both" => RunMode::Both,
                    "qd" => RunMode::Qd,
                    "all" => RunMode::All,
                    "simulate" => RunMode::Simulate,
                    value => return Err(format!("unknown mode: {value}").into()),
                };
            }
            "--target" => parsed.target = take_value(&mut args, "--target")?.parse()?,
            "--evaluations" => {
                parsed.evaluations = take_value(&mut args, "--evaluations")?.parse()?
            }
            "--retries" => parsed.retries = take_value(&mut args, "--retries")?.parse()?,
            "--workers" => parsed.workers = take_value(&mut args, "--workers")?.parse()?,
            "--depth" => parsed.depth = take_value(&mut args, "--depth")?.parse()?,
            "--mo-evaluations" => {
                parsed.mo_evaluations = take_value(&mut args, "--mo-evaluations")?.parse()?
            }
            "--popsize" => parsed.popsize = take_value(&mut args, "--popsize")?.parse()?,
            "--qd-evaluations" => {
                parsed.qd_evaluations = take_value(&mut args, "--qd-evaluations")?.parse()?
            }
            "--qd-capacity" => {
                parsed.qd_capacity = take_value(&mut args, "--qd-capacity")?.parse()?
            }
            "--qd-chunk-size" => {
                parsed.qd_chunk_size = take_value(&mut args, "--qd-chunk-size")?.parse()?
            }
            "--seed" => parsed.seed = take_value(&mut args, "--seed")?.parse()?,
            "--output" => parsed.output = take_value(&mut args, "--output")?.into(),
            "--no-output" => parsed.write_output = false,
            "--x" => parsed.design = Some(parse_design(&take_value(&mut args, "--x")?)?),
            value => return Err(format!("unknown argument: {value}").into()),
        }
    }
    if !parsed.target.is_finite() || parsed.target <= 0.0 {
        return Err("--target must be finite and positive".into());
    }
    Ok(Some(parsed))
}

fn print_simulation(label: &str, design: &Design, simulation: &SimulationResult) {
    println!(
        "{label} status={} landing={:.6} target_error={:.6} energy_j={:.6} peak_load_n={:.6} apex_m={:.6} score={:.9} release_s={}",
        simulation.status.as_str(),
        simulation.landing_position,
        simulation.target_error,
        simulation.input_energy,
        simulation.peak_joint_force,
        simulation.apex_height,
        simulation.scalar_score,
        simulation
            .release_time
            .map_or_else(|| "none".to_string(), |value| format!("{value:.6}")),
    );
    println!("{label}_X {:?}", design.to_vec());
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let simulation_config = SimulationConfig {
        target_position: args.target,
        ..Default::default()
    };
    let mut replay_config = simulation_config.clone();
    replay_config.record_trajectory = true;
    let initial_design = Design::from_slice(&INITIAL_DESIGN)?;
    let initial = simulate(&initial_design, &replay_config);
    print_simulation("INITIAL", &initial_design, &initial);

    let mut selected_design = args
        .design
        .clone()
        .unwrap_or_else(|| initial_design.clone());
    let mut selected_simulation = initial.clone();
    let mut convergence = Vec::new();
    let mut pareto: Vec<ParetoPoint> = Vec::new();
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let mut actual_evaluations = 0usize;
    let mut elapsed_seconds = 0.0;

    if matches!(args.mode, RunMode::Single | RunMode::Both | RunMode::All) {
        let outcome = optimize_scalar(
            &simulation_config,
            &ScalarOptions {
                evaluations_per_retry: args.evaluations,
                retries: args.retries,
                workers: args.workers,
                depth: args.depth,
                seed: args.seed,
            },
        )?;
        println!(
            "SO evaluations={} retries={} workers={} seconds={:.6} evaluations_per_second={:.0}",
            outcome.evaluations,
            outcome.completed_retries,
            args.workers,
            outcome.elapsed.as_secs_f64(),
            outcome.evaluations as f64 / outcome.elapsed.as_secs_f64().max(1.0e-9),
        );
        print_simulation("SO_BEST", &outcome.design, &outcome.simulation);
        convergence = outcome
            .improvements
            .iter()
            .map(|sample| MoProgress {
                evaluations: sample.evaluations as usize,
                elapsed_seconds: sample.elapsed_seconds,
                best_quality: -sample.value,
            })
            .collect();
        actual_evaluations = outcome.evaluations as usize;
        elapsed_seconds = outcome.elapsed.as_secs_f64();
        selected_design = outcome.design;
        selected_simulation = outcome.simulation;
    }

    if matches!(args.mode, RunMode::Multi | RunMode::Both | RunMode::All) {
        let outcome = optimize_multi(
            &simulation_config,
            &MultiOptions {
                evaluations: args.mo_evaluations,
                popsize: args.popsize,
                workers: args.workers,
                seed: args.seed ^ 0xE703_7ED1_A0B4_28DB,
            },
        )?;
        println!(
            "MO evaluations={} generations={} pareto={} workers={} quality={:.9} seconds={:.6} evaluations_per_second={:.0}",
            outcome.evaluations,
            outcome.generations,
            outcome.pareto.len(),
            args.workers,
            outcome.quality,
            outcome.elapsed.as_secs_f64(),
            outcome.evaluations as f64 / outcome.elapsed.as_secs_f64().max(1.0e-9),
        );
        print_simulation(
            "MO_REPRESENTATIVE",
            &outcome.representative.design,
            &outcome.simulation,
        );
        for (rank, point) in outcome.pareto.iter().take(12).enumerate() {
            println!(
                "MO_POINT rank={} error={:.6} energy_j={:.6} peak_load_n={:.6} x={:?}",
                rank + 1,
                point.objectives[0],
                point.objectives[1],
                point.objectives[2],
                point.design.to_vec(),
            );
        }
        if args.mode == RunMode::Multi {
            selected_design = outcome.representative.design.clone();
            selected_simulation = outcome.simulation.clone();
            convergence = outcome.convergence.clone();
        }
        if matches!(args.mode, RunMode::Multi) {
            actual_evaluations = outcome.evaluations;
            elapsed_seconds = outcome.elapsed.as_secs_f64();
        }
        pareto = outcome.pareto;
    }

    if matches!(args.mode, RunMode::Qd | RunMode::All) {
        let options = QdOptions {
            evaluations: args.qd_evaluations,
            capacity: args.qd_capacity,
            chunk_size: args.qd_chunk_size,
            workers: args.workers,
            seed: args.seed,
        };
        let outcome = optimize_qd(&simulation_config, &options)?;
        println!(
            "QD evaluations={} occupied={} capacity={} coverage={:.6} qd_score={:.9} best_quality={:.9} invalid={} clipped={} workers={} seconds={:.6} evaluations_per_second={:.0}",
            outcome.evaluations,
            outcome.occupied,
            outcome.capacity,
            outcome.occupied as f64 / outcome.capacity as f64,
            outcome.qd_score,
            outcome.representative.quality,
            outcome.invalid_evaluations,
            outcome.clipped_descriptors,
            args.workers,
            outcome.elapsed.as_secs_f64(),
            outcome.evaluations as f64 / outcome.elapsed.as_secs_f64().max(1.0e-9),
        );
        print_simulation(
            "QD_BEST",
            &outcome.representative.design,
            &outcome.simulation,
        );
        let qd_output = if args.mode == RunMode::All {
            args.output.join("qd")
        } else {
            args.output.clone()
        };
        if args.write_output {
            write_qd_artifacts(
                &qd_output,
                args.target,
                &initial,
                &outcome,
                &options,
                &command,
            )?;
            println!(
                "QD_ARTIFACTS directory={} archive={}",
                qd_output.display(),
                qd_output.join("qd_archive.csv").display(),
            );
        }
        if args.mode == RunMode::Qd {
            selected_design = outcome.representative.design;
            selected_simulation = outcome.simulation;
        }
    }

    if args.mode == RunMode::Simulate {
        selected_simulation = simulate(&selected_design, &replay_config);
        print_simulation("SIMULATED", &selected_design, &selected_simulation);
    }

    if args.write_output && args.mode != RunMode::Qd {
        write_artifacts(
            &args.output,
            args.target,
            &initial,
            &selected_simulation,
            &convergence,
            &pareto,
        )?;
        println!(
            "ARTIFACTS directory={} replay={}",
            args.output.display(),
            args.output.join("replay.html").display(),
        );
        if matches!(args.mode, RunMode::Single | RunMode::Multi) {
            write_run_manifest(
                &args.output,
                &args,
                actual_evaluations,
                elapsed_seconds,
                &command,
            )?;
        }
    }
    Ok(())
}

fn effective_workers(requested: usize) -> usize {
    if requested == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        requested
    }
}

fn write_run_manifest(
    directory: &std::path::Path,
    args: &Args,
    actual_evaluations: usize,
    elapsed_seconds: f64,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let formulation = match args.mode {
        RunMode::Single => "scalar",
        RunMode::Multi => "mo",
        _ => return Ok(()),
    };
    let mut artifacts = serde_json::Map::new();
    artifacts.insert("convergence".into(), "convergence.csv".into());
    artifacts.insert("trajectory".into(), "trajectory.csv".into());
    artifacts.insert("replay".into(), "replay.html".into());
    if args.mode == RunMode::Multi {
        artifacts.insert("pareto".into(), "pareto.csv".into());
    }
    let objectives = if args.mode == RunMode::Multi {
        serde_json::json!([
            {"column": "objective_target_error", "label": "Target error", "unit": "m"},
            {"column": "objective_input_energy", "label": "Input energy", "unit": "J"},
            {"column": "objective_peak_joint_force", "label": "Peak pivot load", "unit": "N"}
        ])
    } else {
        serde_json::json!([])
    };
    let requested = if args.mode == RunMode::Multi {
        args.mo_evaluations
    } else {
        args.evaluations.saturating_mul(args.retries as u64) as usize
    };
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "rapier-trebuchet",
        "formulation": formulation,
        "command": command,
        "seed": args.seed,
        "workers": effective_workers(args.workers),
        "requested_evaluations": requested,
        "actual_evaluations": actual_evaluations,
        "elapsed_seconds": elapsed_seconds,
        "objectives": objectives,
        "descriptors": [],
        "convergence_metrics": ["best_quality"],
        "artifacts": artifacts
    });
    std::fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_design() {
        let design = parse_design("3.5,80,4,2.5,-0.75,-0.05,2,2").unwrap();
        assert_eq!(design, Design::default());
    }

    #[test]
    fn rejects_bad_design_text() {
        assert!(parse_design("1,2,3").is_err());
        assert!(parse_design("bad,2,3,4,5,6,7,8").is_err());
    }
}
