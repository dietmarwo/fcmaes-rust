use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use nexosim_production_line::{
    OptimizationConfig, OptimizationResult, ParallelStrategy, QdOptions, optimize, optimize_qd,
    write_mode_artifacts, write_qd_artifacts,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunMode {
    Mo,
    Qd,
    All,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StrategySelection {
    Outer,
    Inner,
    Both,
}

#[derive(Clone, Debug)]
struct Args {
    mode: RunMode,
    strategy: StrategySelection,
    config: OptimizationConfig,
    front_limit: usize,
    qd: QdOptions,
    output: PathBuf,
    write_output: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: RunMode::Mo,
            strategy: StrategySelection::Both,
            config: OptimizationConfig::default(),
            front_limit: 12,
            qd: QdOptions::default(),
            output: PathBuf::from("results"),
            write_output: true,
        }
    }
}

impl Args {
    fn parse() -> Result<Option<Self>, String> {
        Self::from_args(env::args().skip(1))
    }

    fn from_args(mut arguments: impl Iterator<Item = String>) -> Result<Option<Self>, String> {
        let mut args = Self::default();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--mode" => {
                    args.mode = match next_value(&mut arguments, "--mode")?.as_str() {
                        "mo" | "multi" => RunMode::Mo,
                        "qd" => RunMode::Qd,
                        "all" => RunMode::All,
                        _ => return Err("--mode must be mo, qd, or all".to_string()),
                    }
                }
                "--strategy" => {
                    args.strategy = match next_value(&mut arguments, "--strategy")?.as_str() {
                        "outer" | "outer-fcmaes" => StrategySelection::Outer,
                        "inner" | "inner-nexosim" => StrategySelection::Inner,
                        "both" => StrategySelection::Both,
                        _ => return Err("--strategy must be outer, inner, or both".to_string()),
                    }
                }
                "--evaluations" => {
                    args.config.evaluations = parse_value(&mut arguments, "--evaluations")?
                }
                "--popsize" => args.config.popsize = parse_value(&mut arguments, "--popsize")?,
                "--replications" => {
                    args.config.replications = parse_value(&mut arguments, "--replications")?
                }
                "--workers" => args.config.workers = parse_value(&mut arguments, "--workers")?,
                "--horizon" => {
                    args.config.horizon_minutes = parse_value(&mut arguments, "--horizon")?
                }
                "--seed" => args.config.seed = parse_value(&mut arguments, "--seed")?,
                "--front-limit" => args.front_limit = parse_value(&mut arguments, "--front-limit")?,
                "--qd-evaluations" => {
                    args.qd.evaluations = parse_value(&mut arguments, "--qd-evaluations")?
                }
                "--qd-capacity" => args.qd.capacity = parse_value(&mut arguments, "--qd-capacity")?,
                "--qd-chunk-size" => {
                    args.qd.chunk_size = parse_value(&mut arguments, "--qd-chunk-size")?
                }
                "--validation-replications" => {
                    args.qd.validation_replications =
                        parse_value(&mut arguments, "--validation-replications")?
                }
                "--output" => args.output = PathBuf::from(next_value(&mut arguments, "--output")?),
                "--no-output" => args.write_output = false,
                "-h" | "--help" => {
                    print_help();
                    return Ok(None);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        args.qd.seed = args.config.seed ^ 0x243F_6A88_85A3_08D3;
        args.config.validate()?;
        Ok(Some(args))
    }
}

fn next_value(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("missing value after {option}"))
}

fn parse_value<T: std::str::FromStr>(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    next_value(arguments, option)?
        .parse()
        .map_err(|_| format!("invalid value for {option}"))
}

fn print_help() {
    println!(
        "NeXosim production-line MODE and MAP-Elites optimization\n\
         \nUsage: cargo run --release -- [OPTIONS]\n\
         \n  --mode NAME           mo, qd, or all (mo)\n\
         \n  --strategy NAME       outer, inner, or both (both)\n\
         \n  --evaluations N       Requested MODE evaluations (512)\n\
         \n  --popsize N           MODE population size, at least 4 (32)\n\
         \n  --replications N      Stochastic replications per candidate (4)\n\
         \n  --workers N           Total worker budget; 0 uses available CPUs (0)\n\
         \n  --horizon MINUTES     Modeled production horizon (240)\n\
         \n  --seed N              Common optimizer/replication root seed (42)\n\
         \n  --front-limit N       Pareto members printed per strategy (12)\n\
         \n  --qd-evaluations N    Requested MAP-Elites evaluations (4096)\n\
         \n  --qd-capacity N       Square QD archive capacity (400)\n\
         \n  --qd-chunk-size N     Even QD evaluation batch size (128)\n\
         \n  --validation-replications N  Disjoint paths per QD elite (8)\n\
         \n  --output DIR          JSON/CSV output directory (results)\n\
         \n  --no-output           Do not write artifacts\n\
         \nOuter: MODE evaluates candidates concurrently; every NeXosim bench is serial.\n\
         \nInner: MODE evaluates candidates serially; every NeXosim bench uses N threads.\n\
         \nQD always uses outer candidate parallelism and serial NeXosim benches."
    );
}

fn print_result(result: &OptimizationResult, workers: usize, front_limit: usize) {
    let (evaluation_workers, nexosim_threads) = match result.strategy {
        ParallelStrategy::Outer => (workers, 1),
        ParallelStrategy::Inner => (1, workers),
    };
    println!(
        "RESULT strategy={} evaluation_workers={} nexosim_threads={} evaluations={} simulation_replications={} pareto={} balanced_score={:.9} wall_seconds={:.6}",
        result.strategy.name(),
        evaluation_workers,
        nexosim_threads,
        result.evaluations,
        result.simulation_replications,
        result.front.len(),
        result.balanced_score,
        result.wall_seconds
    );
    for (rank, member) in result.front.iter().take(front_limit).enumerate() {
        let design = member.design;
        let objective = member.objectives;
        println!(
            "FRONT strategy={} rank={} throughput_per_hour={:.6} mean_lead_time={:.6} mean_wip={:.6} cost_rate={:.6} buffer={} speed_a={:.4} speed_b={:.4} pm_a={:.4} pm_b={:.4} rework_probability={:.4} dispatch_priority={:.4} staff_a={} staff_b={}",
            result.strategy.name(),
            rank + 1,
            -objective[0],
            objective[1],
            objective[2],
            objective[3],
            design.buffer_capacity,
            design.speed_a,
            design.speed_b,
            design.maintenance_threshold_a,
            design.maintenance_threshold_b,
            design.rework_probability,
            design.dispatch_priority,
            design.staff_a,
            design.staff_b,
        );
    }
}

fn run(args: Args) -> Result<(), String> {
    let workers = args.config.resolved_workers();
    eprintln!(
        "configuration strategy={:?} workers={} popsize={} evaluations={} replications={} horizon={} seed={}",
        args.strategy,
        workers,
        args.config.popsize,
        args.config.evaluations,
        args.config.replications,
        args.config.horizon_minutes,
        args.config.seed
    );
    let command = env::args().collect::<Vec<_>>().join(" ");
    let mut results = Vec::new();
    if matches!(args.mode, RunMode::Mo | RunMode::All) {
        let strategies: &[ParallelStrategy] = match args.strategy {
            StrategySelection::Outer => &[ParallelStrategy::Outer],
            StrategySelection::Inner => &[ParallelStrategy::Inner],
            StrategySelection::Both => &[ParallelStrategy::Outer, ParallelStrategy::Inner],
        };
        for &strategy in strategies {
            eprintln!("starting {} without nested parallelism", strategy.name());
            let result = optimize(strategy, &args.config, true)?;
            print_result(&result, workers, args.front_limit);
            if args.write_output {
                let directory = if strategies.len() == 1 && args.mode == RunMode::Mo {
                    args.output.clone()
                } else {
                    args.output.join(strategy.name())
                };
                write_mode_artifacts(&directory, &result, &args.config, &command)
                    .map_err(|error| error.to_string())?;
                println!("ARTIFACTS formulation=mo directory={}", directory.display());
            }
            results.push(result);
        }
    }
    if results.len() == 2 {
        let outer = &results[0];
        let inner = &results[1];
        println!(
            "COMPARISON outer_speedup_vs_inner={:.6} outer_score={:.9} inner_score={:.9}",
            inner.wall_seconds / outer.wall_seconds.max(1.0e-12),
            outer.balanced_score,
            inner.balanced_score
        );
    }
    if matches!(args.mode, RunMode::Qd | RunMode::All) {
        eprintln!("starting MAP-Elites with fcmaes-owned outer parallelism");
        let result = optimize_qd(&args.config, &args.qd)?;
        println!(
            "QD evaluation_workers={} nexosim_threads=1 evaluations={} simulation_replications={} validation_replications={} occupied={} capacity={} coverage={:.6} qd_score={:.9} best_quality={:.9} invalid={} clipped={} same_validation_niche={:.6} wall_seconds={:.6} validation_seconds={:.6}",
            workers,
            result.evaluations,
            result.simulation_replications,
            result.validation_replications,
            result.occupied,
            result.capacity,
            result.occupied as f64 / result.capacity as f64,
            result.qd_score,
            result.representative.quality_train,
            result.invalid_evaluations,
            result.clipped_descriptors,
            result.validation_same_niche_fraction,
            result.elapsed.as_secs_f64(),
            result.validation_elapsed.as_secs_f64(),
        );
        println!(
            "QD_BEST throughput={:.6} wip={:.6} lead_time={:.6} cost_rate={:.6} validation_throughput={:.6} validation_wip={:.6} x={:?}",
            result.representative.training.throughput_per_hour,
            result.representative.training.mean_wip,
            result.representative.training.mean_lead_time,
            result.representative.training.cost_rate,
            result.representative.validation.throughput_per_hour,
            result.representative.validation.mean_wip,
            result.representative.design.as_vector(),
        );
        if args.write_output {
            let directory = if args.mode == RunMode::All {
                args.output.join("qd")
            } else {
                args.output.clone()
            };
            write_qd_artifacts(&directory, &result, &args.config, &args.qd, &command)
                .map_err(|error| error.to_string())?;
            println!("ARTIFACTS formulation=qd directory={}", directory.display());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match Args::parse() {
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!("use --help for usage");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> std::vec::IntoIter<String> {
        values
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn defaults_are_an_explicit_two_strategy_benchmark() {
        let args = Args::from_args(strings(&[])).unwrap().unwrap();
        assert_eq!(args.strategy, StrategySelection::Both);
        assert_eq!(args.mode, RunMode::Mo);
        assert_eq!(args.config.evaluations, 512);
        assert_eq!(args.config.replications, 4);
    }

    #[test]
    fn parses_configuration_and_rejects_bad_values() {
        let args = Args::from_args(strings(&[
            "--strategy",
            "outer",
            "--mode",
            "all",
            "--workers",
            "16",
            "--evaluations",
            "64",
            "--popsize",
            "8",
            "--replications",
            "2",
            "--horizon",
            "60",
            "--seed",
            "7",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(args.strategy, StrategySelection::Outer);
        assert_eq!(args.mode, RunMode::All);
        assert_eq!(args.config.workers, 16);
        assert_eq!(args.config.evaluations, 64);
        assert!(Args::from_args(strings(&["--popsize", "3"])).is_err());
        assert!(Args::from_args(strings(&["--strategy", "nested"])).is_err());
    }
}
