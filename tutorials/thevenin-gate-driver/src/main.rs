use std::error::Error;
use std::path::PathBuf;

use thevenin_gate_driver::DEFAULT_STEP_S;
use thevenin_gate_driver::artifacts::{RunMetadata, write_mode, write_scaling, write_validation};
use thevenin_gate_driver::mode::{ModeConfig, optimize_mode};
use thevenin_gate_driver::studies::{scaling_study, timestep_study, validation_grid};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Optimize,
    Validate,
    Benchmark,
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
    evaluations: Option<usize>,
    popsize: Option<usize>,
    seed: u64,
    output: Option<PathBuf>,
    validation_side: Option<usize>,
    benchmark_candidates: Option<usize>,
    benchmark_repeats: Option<usize>,
    write_output: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::All,
            preset: Preset::Smoke,
            workers: 0,
            evaluations: None,
            popsize: None,
            seed: 42,
            output: None,
            validation_side: None,
            benchmark_candidates: None,
            benchmark_repeats: None,
            write_output: true,
        }
    }
}

#[derive(Clone, Copy)]
struct Protocol {
    evaluations: usize,
    popsize: usize,
    validation_side: usize,
    benchmark_candidates: usize,
    benchmark_repeats: usize,
}

fn protocol(preset: Preset) -> Protocol {
    match preset {
        Preset::Smoke => Protocol {
            evaluations: 256,
            popsize: 32,
            validation_side: 3,
            benchmark_candidates: 32,
            benchmark_repeats: 1,
        },
        Preset::Publication => Protocol {
            evaluations: 4_096,
            popsize: 128,
            validation_side: 7,
            benchmark_candidates: 512,
            benchmark_repeats: 5,
        },
    }
}

fn usage() {
    println!(
        "Transient gate-driver optimization with fcmaes-core and thevenin\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --mode NAME              optimize, validate, benchmark, or all (all)\n\
         --preset NAME            smoke or publication (smoke)\n\
         --workers N              MODE candidate threads; 0 uses CPUs (0)\n\
         --evaluations N          Override the MODE budget\n\
         --popsize N              Override the MODE population size\n\
         --seed N                 Optimizer and benchmark seed (42)\n\
         --validation-side N      Inclusive N×N cross-simulator grid\n\
         --benchmark-candidates N Candidates per scaling repeat\n\
         --benchmark-repeats N    Repeats per worker count\n\
         --output DIR             Artifact root (results/<preset>)\n\
         --no-output              Execute without writing artifacts\n\
         -h, --help               Show this help\n\
         \n\
         ngspice is intentionally not called here. The independent reference\n\
         harness consumes validation/candidates.csv after this command."
    );
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_args() -> Result<Option<Args>, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--mode" => {
                parsed.mode = match next_value(&mut arguments, "--mode")?.as_str() {
                    "optimize" | "mo" => Mode::Optimize,
                    "validate" => Mode::Validate,
                    "benchmark" => Mode::Benchmark,
                    "all" => Mode::All,
                    value => return Err(format!("unknown mode: {value}").into()),
                }
            }
            "--preset" => {
                parsed.preset = match next_value(&mut arguments, "--preset")?.as_str() {
                    "smoke" => Preset::Smoke,
                    "publication" => Preset::Publication,
                    value => return Err(format!("unknown preset: {value}").into()),
                }
            }
            "--workers" => parsed.workers = next_value(&mut arguments, "--workers")?.parse()?,
            "--evaluations" => {
                parsed.evaluations = Some(next_value(&mut arguments, "--evaluations")?.parse()?)
            }
            "--popsize" => parsed.popsize = Some(next_value(&mut arguments, "--popsize")?.parse()?),
            "--seed" => parsed.seed = next_value(&mut arguments, "--seed")?.parse()?,
            "--validation-side" => {
                parsed.validation_side =
                    Some(next_value(&mut arguments, "--validation-side")?.parse()?)
            }
            "--benchmark-candidates" => {
                parsed.benchmark_candidates =
                    Some(next_value(&mut arguments, "--benchmark-candidates")?.parse()?)
            }
            "--benchmark-repeats" => {
                parsed.benchmark_repeats =
                    Some(next_value(&mut arguments, "--benchmark-repeats")?.parse()?)
            }
            "--output" => parsed.output = Some(next_value(&mut arguments, "--output")?.into()),
            "--no-output" => parsed.write_output = false,
            value => return Err(format!("unknown option: {value}").into()),
        }
    }
    if parsed.workers < 0 {
        return Err("--workers must be non-negative".into());
    }
    Ok(Some(parsed))
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let defaults = protocol(args.preset);
    let preset_name = match args.preset {
        Preset::Smoke => "smoke",
        Preset::Publication => "publication",
    };
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("results").join(preset_name));
    let forwarded = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let command = if forwarded.is_empty() {
        "cargo run --release".to_owned()
    } else {
        format!("cargo run --release -- {forwarded}")
    };

    if matches!(args.mode, Mode::Optimize | Mode::All) {
        let result = optimize_mode(&ModeConfig {
            evaluations: args.evaluations.unwrap_or(defaults.evaluations),
            popsize: args.popsize.unwrap_or(defaults.popsize),
            workers: args.workers,
            seed: args.seed,
        })?;
        println!(
            "MODE pareto={} evaluations={} wall={:.6}s",
            result.pareto.len(),
            result.actual_evaluations,
            result.elapsed.as_secs_f64()
        );
        if args.write_output {
            write_mode(
                &RunMetadata {
                    directory: &output.join("mo"),
                    command: &command,
                    seed: args.seed,
                    workers: args.workers,
                },
                &result,
            )?;
        }
    }

    if matches!(args.mode, Mode::Validate | Mode::All) {
        let side = args.validation_side.unwrap_or(defaults.validation_side);
        let grid = validation_grid(side, DEFAULT_STEP_S)?;
        let timestep = timestep_study()?;
        println!(
            "VALIDATION grid={}x{} rows={} timestep_rows={}",
            side,
            side,
            grid.len(),
            timestep.len()
        );
        if args.write_output {
            write_validation(&output.join("validation"), &grid, &timestep)?;
        }
    }

    if matches!(args.mode, Mode::Benchmark | Mode::All) {
        let candidates = args
            .benchmark_candidates
            .unwrap_or(defaults.benchmark_candidates);
        let repeats = args.benchmark_repeats.unwrap_or(defaults.benchmark_repeats);
        let available = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let mut worker_counts = vec![1];
        for workers in [4, 16] {
            let effective = workers.min(available) as i32;
            if !worker_counts.contains(&effective) {
                worker_counts.push(effective);
            }
        }
        let scaling = scaling_study(
            candidates,
            repeats,
            &worker_counts,
            args.seed.wrapping_add(9_000),
        )?;
        for row in &scaling {
            println!(
                "SCALING workers={} repeat={} evaluations_per_second={:.1} failures={}",
                row.workers, row.repeat, row.evaluations_per_second, row.failures
            );
        }
        if args.write_output {
            write_scaling(&output.join("validation"), &scaling)?;
        }
    }
    Ok(())
}
