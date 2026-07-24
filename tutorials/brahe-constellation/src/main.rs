use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use brahe_constellation::{
    AccessMetrics, BASELINE_DESIGN, ConstellationModel, DIMENSION, MoProgress, ModelConfig,
    MultiOptions, Parallelism, ParetoPoint, QdOptions, ScalarOptions, optimize_multi, optimize_qd,
    optimize_scalar, scalar_objective, write_artifacts, write_qd_artifacts,
};
use fcmaes_core::{Rng, parallel_batch};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    Single,
    Multi,
    Both,
    Qd,
    All,
    Simulate,
    Benchmark,
}

#[derive(Debug)]
struct Args {
    mode: RunMode,
    model: ModelConfig,
    evaluations: u64,
    retries: usize,
    depth: i32,
    mo_evaluations: usize,
    popsize: usize,
    qd_evaluations: usize,
    qd_capacity: usize,
    qd_chunk_size: usize,
    seed: u64,
    numerical_validation: bool,
    output: PathBuf,
    write_output: bool,
    design: Option<Vec<f64>>,
    benchmark_candidates: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: RunMode::Both,
            model: ModelConfig::default(),
            evaluations: 250,
            retries: 8,
            depth: 6,
            mo_evaluations: 4_096,
            popsize: 128,
            qd_evaluations: 4_096,
            qd_capacity: 400,
            qd_chunk_size: 128,
            seed: 42,
            numerical_validation: false,
            output: PathBuf::from("results"),
            write_output: true,
            design: None,
            benchmark_candidates: 128,
        }
    }
}

fn usage() {
    println!(
        "Brahe constellation ground-contact optimization\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --mode NAME              single, multi, qd, both, all, simulate, or benchmark (both)\n\
         --workers N              CPU workers; 0 uses available CPUs (0)\n\
         --parallel NAME          outer or inner (outer)\n\
         --hours N                Access horizon in hours (24)\n\
         --provider NAME          Embedded Brahe station provider (ksat)\n\
         --stations CSV           Station names (six global KSAT stations)\n\
         --min-elevation DEG      Minimum access elevation (10)\n\
         --min-pass SECONDS       Minimum accepted pass duration (180)\n\
         --access-step SECONDS    Access grid step, 1..=300 (60)\n\
         --evaluations N          BiteOpt evaluations per retry (250)\n\
         --retries N              Independent BiteOpt retries (8)\n\
         --depth N                BiteOpt deep populations, 1..=36 (6)\n\
         --mo-evaluations N       Requested MODE evaluations (4096)\n\
         --popsize N              MODE population size (128)\n\
         --qd-evaluations N       Requested MAP-Elites evaluations (4096)\n\
         --qd-capacity N          Square MAP-Elites archive capacity (400)\n\
         --qd-chunk-size N        Even QD evaluation batch size (128)\n\
         --seed N                 Optimizer root seed (42)\n\
         --numerical-validation   Validate final design with 20x20 gravity\n\
         --output DIR             CSV/HTML output directory (results)\n\
         --no-output              Do not write result files\n\
         --x CSV                  Ten design values for simulate mode\n\
         --benchmark-candidates N Fixed candidates for parallel benchmark (128)\n\
         -h, --help               Show this help\n\
         \n\
         outer: fcmaes owns the worker pool; Brahe uses one access thread.\n\
         inner: fcmaes uses one objective worker; Brahe owns the worker pool."
    );
}

fn take_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_csv_f64(value: &str, expected: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let values = value
        .split(',')
        .map(|field| field.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != expected {
        return Err(format!("expected {expected} comma-separated values").into());
    }
    Ok(values)
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
                    "benchmark" => RunMode::Benchmark,
                    value => return Err(format!("unknown mode: {value}").into()),
                };
            }
            "--workers" => parsed.model.workers = take_value(&mut args, "--workers")?.parse()?,
            "--parallel" => {
                parsed.model.parallelism = match take_value(&mut args, "--parallel")?.as_str() {
                    "outer" => Parallelism::Outer,
                    "inner" => Parallelism::Inner,
                    value => return Err(format!("unknown parallel mode: {value}").into()),
                };
            }
            "--hours" => parsed.model.horizon_hours = take_value(&mut args, "--hours")?.parse()?,
            "--provider" => parsed.model.provider = take_value(&mut args, "--provider")?,
            "--stations" => {
                parsed.model.station_names = take_value(&mut args, "--stations")?
                    .split(',')
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(ToString::to_string)
                    .collect()
            }
            "--min-elevation" => {
                parsed.model.minimum_elevation_deg =
                    take_value(&mut args, "--min-elevation")?.parse()?
            }
            "--min-pass" => {
                parsed.model.minimum_pass_seconds = take_value(&mut args, "--min-pass")?.parse()?
            }
            "--access-step" => {
                parsed.model.access_step_seconds =
                    take_value(&mut args, "--access-step")?.parse()?
            }
            "--evaluations" => {
                parsed.evaluations = take_value(&mut args, "--evaluations")?.parse()?
            }
            "--retries" => parsed.retries = take_value(&mut args, "--retries")?.parse()?,
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
            "--numerical-validation" => parsed.numerical_validation = true,
            "--output" => parsed.output = take_value(&mut args, "--output")?.into(),
            "--no-output" => parsed.write_output = false,
            "--x" => {
                parsed.design = Some(parse_csv_f64(&take_value(&mut args, "--x")?, DIMENSION)?)
            }
            "--benchmark-candidates" => {
                parsed.benchmark_candidates =
                    take_value(&mut args, "--benchmark-candidates")?.parse()?
            }
            value => return Err(format!("unknown argument: {value}").into()),
        }
    }
    Ok(Some(parsed))
}

fn print_design(label: &str, design: &[f64]) {
    println!(
        "{label}_X {}",
        design
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    );
}

fn print_metrics(label: &str, metrics: &AccessMetrics) {
    println!(
        "{label} score={:.9} max_gap_hours={:.9} total_contact_hours={:.9} min_passes={} missing_passes={} min_accepted_pass_seconds={:.3} altitude_cost={:.9} plane_spread={:.9} launch_complexity={:.9} quality={:.9}",
        metrics.scalar_score,
        metrics.maximum_gap_hours,
        metrics.total_contact_hours,
        metrics.minimum_passes,
        metrics.missing_passes,
        metrics.minimum_accepted_pass_seconds,
        metrics.altitude_cost,
        metrics.plane_spread,
        metrics.launch_complexity,
        metrics.quality(),
    );
    for station in &metrics.stations {
        println!(
            "{label}_STATION name={} passes={} short={} contact_hours={:.6} max_gap_hours={:.6}",
            station.station,
            station.accepted_passes,
            station.rejected_short_passes,
            station.contact_hours,
            station.maximum_gap_hours,
        );
    }
}

fn run_parallelism_benchmark(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.benchmark_candidates == 0 {
        return Err("benchmark candidate count must be positive".into());
    }
    let mut rng = Rng::new(args.seed);
    let candidates: Vec<Vec<f64>> = (0..args.benchmark_candidates)
        .map(|_| {
            brahe_constellation::LOWER_BOUNDS
                .iter()
                .zip(brahe_constellation::UPPER_BOUNDS)
                .map(|(&lower, upper)| lower + rng.uniform01() * (upper - lower))
                .collect()
        })
        .collect();

    let mut outer_config = args.model.clone();
    outer_config.parallelism = Parallelism::Outer;
    let outer = ConstellationModel::new(outer_config)?;
    let started = Instant::now();
    let outer_values = parallel_batch(&candidates, args.model.workers as i32, |candidate| {
        scalar_objective(candidate, &outer)
    });
    let outer_elapsed = started.elapsed();

    let mut inner_config = args.model.clone();
    inner_config.parallelism = Parallelism::Inner;
    let inner = ConstellationModel::new(inner_config)?;
    let started = Instant::now();
    let inner_values = parallel_batch(&candidates, 1, |candidate| {
        scalar_objective(candidate, &inner)
    });
    let inner_elapsed = started.elapsed();

    let maximum_difference = outer_values
        .iter()
        .zip(&inner_values)
        .map(|(outer, inner)| (outer - inner).abs())
        .fold(0.0_f64, f64::max);
    let checksum = outer_values.iter().sum::<f64>();
    println!(
        "BENCHMARK candidates={} workers={} outer_seconds={:.6} inner_seconds={:.6} outer_eval_per_second={:.3} inner_eval_per_second={:.3} outer_speedup={:.3} checksum={:.12} maximum_result_difference={:.3e}",
        candidates.len(),
        args.model.resolved_workers(),
        outer_elapsed.as_secs_f64(),
        inner_elapsed.as_secs_f64(),
        candidates.len() as f64 / outer_elapsed.as_secs_f64().max(1.0e-9),
        candidates.len() as f64 / inner_elapsed.as_secs_f64().max(1.0e-9),
        inner_elapsed.as_secs_f64() / outer_elapsed.as_secs_f64().max(1.0e-9),
        checksum,
        maximum_difference,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    if args.mode == RunMode::Benchmark {
        return run_parallelism_benchmark(&args);
    }
    let model = ConstellationModel::new(args.model.clone())?;
    let (outer_workers, brahe_threads) = match args.model.parallelism {
        Parallelism::Outer => (args.model.resolved_workers(), 1),
        Parallelism::Inner => (1, args.model.resolved_workers()),
    };
    println!(
        "CONFIG parallel={:?} fcmaes_workers={} brahe_access_threads={} horizon_hours={} provider={} stations={} required_passes_per_station={} min_pass_seconds={} access_step_seconds={}",
        args.model.parallelism,
        outer_workers,
        brahe_threads,
        args.model.horizon_hours,
        args.model.provider,
        args.model.station_names.join("|"),
        args.model.required_passes_per_station(),
        args.model.minimum_pass_seconds,
        args.model.access_step_seconds,
    );

    let initial_design = args
        .design
        .clone()
        .unwrap_or_else(|| BASELINE_DESIGN.to_vec());
    let initial_metrics = model.evaluate(&initial_design)?;
    print_design("INITIAL", &initial_design);
    print_metrics("INITIAL", &initial_metrics);

    let mut selected_design = initial_design;
    let mut selected_metrics = initial_metrics;
    let mut convergence: Vec<MoProgress> = Vec::new();
    let mut pareto: Vec<ParetoPoint> = Vec::new();
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let mut run_manifest: Option<serde_json::Value> = None;

    if matches!(args.mode, RunMode::Single | RunMode::Both | RunMode::All) {
        let outcome = optimize_scalar(
            &model,
            &ScalarOptions {
                evaluations_per_retry: args.evaluations,
                retries: args.retries,
                workers: args.model.outer_workers(),
                depth: args.depth,
                seed: args.seed,
            },
        )?;
        println!(
            "SO evaluations={} retries={} seconds={:.6} evaluations_per_second={:.3}",
            outcome.evaluations,
            outcome.completed_retries,
            outcome.elapsed.as_secs_f64(),
            outcome.evaluations as f64 / outcome.elapsed.as_secs_f64().max(1.0e-9),
        );
        print_design("SO_BEST", &outcome.design);
        print_metrics("SO_BEST", &outcome.metrics);
        convergence = outcome
            .improvements
            .iter()
            .map(|sample| MoProgress {
                evaluations: sample.evaluations as usize,
                elapsed_seconds: sample.elapsed_seconds,
                best_quality: 1.0 / (1.0 + sample.value),
            })
            .collect();
        selected_design = outcome.design;
        selected_metrics = outcome.metrics;
    }

    if matches!(args.mode, RunMode::Multi | RunMode::Both | RunMode::All) {
        let outcome = optimize_multi(
            &model,
            &MultiOptions {
                evaluations: args.mo_evaluations,
                popsize: args.popsize,
                workers: args.model.outer_workers(),
                seed: args.seed ^ 0xA076_1D64_78BD_642F,
            },
        )?;
        println!(
            "MODE evaluations={} generations={} pareto={} seconds={:.6} evaluations_per_second={:.3} quality={:.9}",
            outcome.evaluations,
            outcome.generations,
            outcome.pareto.len(),
            outcome.elapsed.as_secs_f64(),
            outcome.evaluations as f64 / outcome.elapsed.as_secs_f64().max(1.0e-9),
            outcome.quality,
        );
        print_design("MODE_REPRESENTATIVE", &outcome.representative.design);
        print_metrics("MODE_REPRESENTATIVE", &outcome.metrics);
        if args.mode == RunMode::Multi {
            run_manifest = Some(serde_json::json!({
                "schema_version": 1,
                "tutorial": "brahe-constellation",
                "formulation": "mo",
                "strategy": format!("{:?}", args.model.parallelism).to_lowercase(),
                "command": &command,
                "root_seed": args.seed,
                "seed": args.seed ^ 0xA076_1D64_78BD_642F,
                "workers": outer_workers,
                "requested_evaluations": args.mo_evaluations,
                "actual_evaluations": outcome.evaluations,
                "elapsed_seconds": outcome.elapsed.as_secs_f64(),
                "simulation": {
                    "horizon_hours": args.model.horizon_hours,
                    "minimum_pass_seconds": args.model.minimum_pass_seconds,
                    "access_step_seconds": args.model.access_step_seconds,
                    "provider": &args.model.provider,
                    "stations": &args.model.station_names
                },
                "objectives": [
                    {
                        "column": "objective_maximum_gap_hours",
                        "label": "Maximum contact gap",
                        "unit": "hours"
                    },
                    {
                        "column": "objective_negative_contact_hours",
                        "label": "Total contact",
                        "unit": "hours",
                        "display_sign": -1
                    },
                    {
                        "column": "objective_launch_complexity",
                        "label": "Launch complexity"
                    }
                ],
                "descriptors": [],
                "convergence_metrics": ["best_quality"],
                "artifacts": {
                    "pareto": "pareto.csv",
                    "convergence": "convergence.csv",
                    "access_windows": "access_windows.csv",
                    "stations": "stations.csv",
                    "design": "design.csv",
                    "report": "report.html"
                }
            }));
        }
        convergence = outcome.convergence;
        selected_design = outcome.representative.design;
        selected_metrics = outcome.metrics;
        pareto = outcome.pareto;
    }

    if matches!(args.mode, RunMode::Qd | RunMode::All) {
        let options = QdOptions {
            evaluations: args.qd_evaluations,
            capacity: args.qd_capacity,
            chunk_size: args.qd_chunk_size,
            workers: args.model.outer_workers(),
            seed: args.seed,
        };
        let outcome = optimize_qd(&model, &options)?;
        println!(
            "QD evaluations={} occupied={} capacity={} coverage={:.6} qd_score={:.9} best_quality={:.9} invalid={} clipped={} seconds={:.6} evaluations_per_second={:.3}",
            outcome.evaluations,
            outcome.occupied,
            outcome.capacity,
            outcome.occupied as f64 / outcome.capacity as f64,
            outcome.qd_score,
            outcome.representative.quality,
            outcome.invalid_evaluations,
            outcome.clipped_descriptors,
            outcome.elapsed.as_secs_f64(),
            outcome.evaluations as f64 / outcome.elapsed.as_secs_f64().max(1.0e-9),
        );
        print_design("QD_BEST", &outcome.representative.design);
        print_metrics("QD_BEST", &outcome.representative.metrics);
        let directory = if args.mode == RunMode::All {
            args.output.join("qd")
        } else {
            args.output.clone()
        };
        if args.write_output {
            write_qd_artifacts(&directory, &model, &outcome, &options, &command)?;
            println!("QD_OUTPUT {}", directory.display());
        }
        if args.mode == RunMode::Qd {
            selected_design = outcome.representative.design;
            selected_metrics = outcome.representative.metrics;
        }
    }

    if args.numerical_validation {
        println!("NUMERICAL_VALIDATION force_model=EGM2008 degree=20 order=20");
        let numerical = model.evaluate_numerical(&selected_design)?;
        print_metrics("NUMERICAL_FINALIST", &numerical);
        println!(
            "NUMERICAL_DELTA max_gap_hours={:.9} total_contact_hours={:.9} scalar_score={:.9}",
            numerical.maximum_gap_hours - selected_metrics.maximum_gap_hours,
            numerical.total_contact_hours - selected_metrics.total_contact_hours,
            numerical.scalar_score - selected_metrics.scalar_score,
        );
    }

    if args.write_output && args.mode != RunMode::Qd {
        write_artifacts(
            &args.output,
            &model,
            &selected_design,
            &selected_metrics,
            &convergence,
            &pareto,
        )?;
        if let Some(manifest) = run_manifest {
            std::fs::write(
                args.output.join("run.json"),
                serde_json::to_string_pretty(&manifest)? + "\n",
            )?;
        }
        println!("OUTPUT {}", args.output.display());
    }
    Ok(())
}
