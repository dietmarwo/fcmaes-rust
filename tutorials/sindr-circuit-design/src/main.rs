use std::error::Error;
use std::path::PathBuf;

use sindr_circuit_design::artifacts::{RunMetadata, write_mo, write_qd, write_so};
use sindr_circuit_design::mo::{MoConfig, optimize_mode};
use sindr_circuit_design::qd::{QdConfig, optimize_qd, range_study};
use sindr_circuit_design::so::{SoConfig, SoOptimizer, feature_demo, optimize_arm};
use sindr_circuit_design::{PUBLICATION_MO_POINTS, PUBLICATION_QD_POINTS, PUBLICATION_SO_POINTS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    So,
    Mo,
    Qd,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OptimizerSelection {
    Cma,
    De,
    Bite,
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
    optimizer: OptimizerSelection,
    preset: Preset,
    workers: i32,
    evaluations: Option<usize>,
    seed: u64,
    output: Option<PathBuf>,
    points: Option<usize>,
    mc_draws: Option<usize>,
    popsize: Option<usize>,
    qd_capacity: Option<usize>,
    qd_chunk_size: Option<usize>,
    range_samples: Option<usize>,
    write_output: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::All,
            optimizer: OptimizerSelection::All,
            preset: Preset::Smoke,
            workers: 0,
            evaluations: None,
            seed: 42,
            output: None,
            points: None,
            mc_draws: None,
            popsize: None,
            qd_capacity: None,
            qd_chunk_size: None,
            range_samples: None,
            write_output: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Protocol {
    so_evaluations: usize,
    so_retries: usize,
    mo_evaluations: usize,
    mo_popsize: usize,
    qd_evaluations: usize,
    qd_capacity: usize,
    qd_chunk_size: usize,
    range_samples: usize,
    so_points: usize,
    mo_points: usize,
    qd_points: usize,
    mc_draws: usize,
}

fn protocol(preset: Preset) -> Protocol {
    match preset {
        Preset::Smoke => Protocol {
            so_evaluations: 600,
            so_retries: 3,
            mo_evaluations: 512,
            mo_popsize: 64,
            qd_evaluations: 256,
            qd_capacity: 100,
            qd_chunk_size: 32,
            range_samples: 128,
            so_points: 31,
            mo_points: 31,
            qd_points: 31,
            mc_draws: 4,
        },
        Preset::Publication => Protocol {
            so_evaluations: 6_000,
            so_retries: 6,
            mo_evaluations: 8_192,
            mo_popsize: 128,
            qd_evaluations: 4_096,
            qd_capacity: 400,
            qd_chunk_size: 64,
            range_samples: 1_000,
            so_points: PUBLICATION_SO_POINTS,
            mo_points: PUBLICATION_MO_POINTS,
            qd_points: PUBLICATION_QD_POINTS,
            mc_draws: 16,
        },
    }
}

fn usage() {
    println!(
        "Circuit-design optimization with fcmaes-core and sindr\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --mode NAME              so, mo, qd, or all (all)\n\
         --optimizer NAME         cma, de, bite, or all for SO (all)\n\
         --preset NAME            smoke or publication (smoke)\n\
         --workers N              Candidate threads; 0 uses available CPUs (0)\n\
         --evaluations N          Override the selected module budget\n\
         --seed N                 Root optimizer and tolerance seed (42)\n\
         --output DIR             Artifact root (results/<preset>)\n\
         --points N               Override AC sweep points for every module\n\
         --mc-draws N             QD tolerance draws\n\
         --popsize N              MODE population size\n\
         --qd-capacity N          Perfect-square regular-grid capacity\n\
         --qd-chunk-size N        Even MAP-Elites batch size\n\
         --range-samples N        QD descriptor range-study candidates\n\
         --no-output              Execute without writing artifacts\n\
         -h, --help               Show this help\n\
         \n\
         Budgets count logical candidates. QD additionally records physical\n\
         AC solves = candidates × (1 + tolerance draws)."
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
                    "so" => Mode::So,
                    "mo" => Mode::Mo,
                    "qd" => Mode::Qd,
                    "all" => Mode::All,
                    value => return Err(format!("unknown mode: {value}").into()),
                };
            }
            "--optimizer" => {
                parsed.optimizer = match next_value(&mut arguments, "--optimizer")?.as_str() {
                    "cma" => OptimizerSelection::Cma,
                    "de" => OptimizerSelection::De,
                    "bite" => OptimizerSelection::Bite,
                    "all" => OptimizerSelection::All,
                    value => return Err(format!("unknown optimizer: {value}").into()),
                };
            }
            "--preset" => {
                parsed.preset = match next_value(&mut arguments, "--preset")?.as_str() {
                    "smoke" => Preset::Smoke,
                    "publication" => Preset::Publication,
                    value => return Err(format!("unknown preset: {value}").into()),
                };
            }
            "--workers" => parsed.workers = next_value(&mut arguments, "--workers")?.parse()?,
            "--evaluations" => {
                parsed.evaluations = Some(next_value(&mut arguments, "--evaluations")?.parse()?)
            }
            "--seed" => parsed.seed = next_value(&mut arguments, "--seed")?.parse()?,
            "--output" => parsed.output = Some(next_value(&mut arguments, "--output")?.into()),
            "--points" => parsed.points = Some(next_value(&mut arguments, "--points")?.parse()?),
            "--mc-draws" => {
                parsed.mc_draws = Some(next_value(&mut arguments, "--mc-draws")?.parse()?)
            }
            "--popsize" => parsed.popsize = Some(next_value(&mut arguments, "--popsize")?.parse()?),
            "--qd-capacity" => {
                parsed.qd_capacity = Some(next_value(&mut arguments, "--qd-capacity")?.parse()?)
            }
            "--qd-chunk-size" => {
                parsed.qd_chunk_size = Some(next_value(&mut arguments, "--qd-chunk-size")?.parse()?)
            }
            "--range-samples" => {
                parsed.range_samples = Some(next_value(&mut arguments, "--range-samples")?.parse()?)
            }
            "--no-output" => parsed.write_output = false,
            value => return Err(format!("unknown option: {value}").into()),
        }
    }
    if parsed.workers < 0 {
        return Err("--workers must be non-negative".into());
    }
    Ok(Some(parsed))
}

fn command_line() -> String {
    let forwarded = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if forwarded.is_empty() {
        "cargo run --release".to_owned()
    } else {
        format!("cargo run --release -- {forwarded}")
    }
}

fn run_so(
    args: &Args,
    protocol: Protocol,
    output: &std::path::Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let selected: Vec<SoOptimizer> = match args.optimizer {
        OptimizerSelection::Cma => vec![SoOptimizer::Cma],
        OptimizerSelection::De => vec![SoOptimizer::De],
        OptimizerSelection::Bite => vec![SoOptimizer::Bite],
        OptimizerSelection::All => SoOptimizer::ALL.to_vec(),
    };
    let evaluations = args.evaluations.unwrap_or(protocol.so_evaluations) as u64;
    let retries = protocol.so_retries.min(evaluations as usize).max(1);
    let config = SoConfig {
        evaluations_per_arm: evaluations,
        retries,
        workers: args.workers as usize,
        seed: args.seed,
        points: protocol.so_points,
    };
    let mut results = Vec::new();
    for optimizer in selected {
        let result = optimize_arm(optimizer, &config)?;
        println!(
            "SO {:>4}: objective={:.6}, f0={:.2} Hz, Q={:.3}, evaluations={}, wall={:.3}s",
            optimizer.name(),
            result.best.objective,
            result.best.features.peak_hz,
            result.best.features.q,
            result.actual_evaluations,
            result.elapsed.as_secs_f64(),
        );
        results.push(result);
    }
    let (feature_curve, smoothness) = feature_demo(protocol.so_points)?;
    if args.write_output {
        let directory = output.join("so");
        write_so(
            &RunMetadata {
                directory: &directory,
                command,
                seed: args.seed,
                workers: args.workers,
                points: protocol.so_points,
            },
            &results,
            &feature_curve,
            &smoothness,
        )?;
    }
    Ok(())
}

fn run_mo(
    args: &Args,
    protocol: Protocol,
    output: &std::path::Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let config = MoConfig {
        evaluations: args.evaluations.unwrap_or(protocol.mo_evaluations),
        popsize: protocol.mo_popsize,
        workers: args.workers,
        seed: args.seed,
        points: protocol.mo_points,
    };
    let result = optimize_mode(&config)?;
    println!(
        "MO: {} feasible nondominated designs, evaluations={}, wall={:.3}s",
        result.pareto.len(),
        result.actual_evaluations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        let directory = output.join("mo");
        write_mo(
            &RunMetadata {
                directory: &directory,
                command,
                seed: args.seed,
                workers: args.workers,
                points: protocol.mo_points,
            },
            &result,
        )?;
    }
    Ok(())
}

fn run_qd(
    args: &Args,
    protocol: Protocol,
    output: &std::path::Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let range_rows = range_study(
        protocol.range_samples,
        args.seed.wrapping_add(700),
        protocol.qd_points,
    )?;
    let config = QdConfig {
        evaluations: args.evaluations.unwrap_or(protocol.qd_evaluations),
        capacity: protocol.qd_capacity,
        chunk_size: protocol.qd_chunk_size,
        workers: args.workers,
        seed: args.seed,
        points: protocol.qd_points,
        mc_draws: protocol.mc_draws,
    };
    let result = optimize_qd(&config)?;
    println!(
        "QD: {}/{} niches ({:.1}%), evaluations={}, search AC solves={}, range-study solves={}, invalid={}, outside={}, wall={:.3}s",
        result.elites.len(),
        result.capacity,
        100.0 * result.elites.len() as f64 / result.capacity as f64,
        result.actual_evaluations,
        result.ac_solves,
        protocol.range_samples,
        result.invalid_evaluations,
        result.out_of_range_descriptors,
        result.elapsed.as_secs_f64(),
    );
    if args.write_output {
        let directory = output.join("qd");
        write_qd(
            &RunMetadata {
                directory: &directory,
                command,
                seed: args.seed,
                workers: args.workers,
                points: protocol.qd_points,
            },
            protocol.mc_draws,
            protocol.range_samples,
            &range_rows,
            &result,
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let mut selected_protocol = protocol(args.preset);
    if let Some(points) = args.points {
        selected_protocol.so_points = points;
        selected_protocol.mo_points = points;
        selected_protocol.qd_points = points;
    }
    if let Some(draws) = args.mc_draws {
        selected_protocol.mc_draws = draws;
    }
    if let Some(popsize) = args.popsize {
        selected_protocol.mo_popsize = popsize;
    }
    if let Some(capacity) = args.qd_capacity {
        selected_protocol.qd_capacity = capacity;
    }
    if let Some(chunk) = args.qd_chunk_size {
        selected_protocol.qd_chunk_size = chunk;
    }
    if let Some(samples) = args.range_samples {
        selected_protocol.range_samples = samples;
    }
    let preset_name = match args.preset {
        Preset::Smoke => "smoke",
        Preset::Publication => "publication",
    };
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("results").join(preset_name));
    let command = command_line();
    if matches!(args.mode, Mode::So | Mode::All) {
        run_so(&args, selected_protocol, &output, &command)?;
    }
    if matches!(args.mode, Mode::Mo | Mode::All) {
        run_mo(&args, selected_protocol, &output, &command)?;
    }
    if matches!(args.mode, Mode::Qd | Mode::All) {
        run_qd(&args, selected_protocol, &output, &command)?;
    }
    Ok(())
}
