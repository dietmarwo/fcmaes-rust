//! Multi-UAV task assignment with native Rust BiteOpt retry and MODE.

use std::env;
use std::error::Error;
use std::time::Instant;

use fcmaes_core::{
    BiteParams, Fitness, Mode, ModeParams, RetryBounds, RetryConfig, RetryRunResult, Rng,
    optimize_bite, parallel_batch, pareto_indices, retry,
};
use fcmaes_examples::uav::UavProblem;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunMode {
    Single,
    Multi,
    Both,
}

impl RunMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "single" | "so" => Ok(Self::Single),
            "multi" | "mo" => Ok(Self::Multi),
            "both" => Ok(Self::Both),
            _ => Err("--mode must be single, multi, or both".to_string()),
        }
    }

    fn includes_single(self) -> bool {
        matches!(self, Self::Single | Self::Both)
    }

    fn includes_multi(self) -> bool {
        matches!(self, Self::Multi | Self::Both)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Multi => "multi",
            Self::Both => "both",
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    vehicles: usize,
    targets: usize,
    map_size: usize,
    mode: RunMode,
    evaluations: u64,
    mo_evaluations: usize,
    retries: usize,
    workers: usize,
    popsize: usize,
    depth: i32,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            vehicles: 5,
            targets: 30,
            map_size: 5_000,
            mode: RunMode::Both,
            evaluations: 50_000,
            mo_evaluations: 100_000,
            retries: 8,
            workers: 0,
            popsize: 256,
            depth: 6,
            seed: 65,
        }
    }
}

impl Args {
    fn parse() -> Result<Self, String> {
        Self::from_args(env::args().skip(1))
    }

    fn from_args(mut arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut parsed = Self::default();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--size" => {
                    let size = next_value(&mut arguments, "--size")?;
                    (parsed.vehicles, parsed.targets, parsed.map_size) = match size.as_str() {
                        "small" => (5, 30, 5_000),
                        "medium" => (10, 60, 10_000),
                        "large" => (15, 90, 15_000),
                        _ => return Err("--size must be small, medium, or large".to_string()),
                    };
                }
                "--vehicles" => parsed.vehicles = parse_value(&mut arguments, "--vehicles")?,
                "--targets" => parsed.targets = parse_value(&mut arguments, "--targets")?,
                "--map-size" => parsed.map_size = parse_value(&mut arguments, "--map-size")?,
                "--mode" => parsed.mode = RunMode::parse(&next_value(&mut arguments, "--mode")?)?,
                "--evaluations" => {
                    parsed.evaluations = parse_value(&mut arguments, "--evaluations")?
                }
                "--mo-evaluations" => {
                    parsed.mo_evaluations = parse_value(&mut arguments, "--mo-evaluations")?
                }
                "--retries" => parsed.retries = parse_value(&mut arguments, "--retries")?,
                "--workers" => parsed.workers = parse_value(&mut arguments, "--workers")?,
                "--popsize" => parsed.popsize = parse_value(&mut arguments, "--popsize")?,
                "--depth" => parsed.depth = parse_value(&mut arguments, "--depth")?,
                "--seed" => parsed.seed = parse_value(&mut arguments, "--seed")?,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<(), String> {
        if self.vehicles == 0 || self.targets == 0 || self.map_size == 0 {
            return Err("vehicle count, target count, and map size must be positive".to_string());
        }
        if self.evaluations == 0 || self.mo_evaluations == 0 || self.retries == 0 {
            return Err("evaluation budgets and retries must be positive".to_string());
        }
        if self.popsize < 4 {
            return Err("--popsize must be at least four".to_string());
        }
        if !(1..=36).contains(&self.depth) {
            return Err("--depth must be between 1 and 36".to_string());
        }
        if self.workers > i32::MAX as usize {
            return Err("--workers is too large".to_string());
        }
        Ok(())
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
        "Native Multi-UAV Task Assignment benchmark\n\
         \nUsage: cargo run --release -p fcmaes-examples --bin uav-task-assignment -- [OPTIONS]\n\
         \n  --size NAME             small, medium, or large preset (small)\n\
         \n  --vehicles N            Override UAV count (5)\n\
         \n  --targets N             Override target count (30)\n\
         \n  --map-size N            Override square map side length (5000)\n\
         \n  --mode NAME             single, multi, or both (both)\n\
         \n  --evaluations N         BiteOpt evaluations per retry (50000)\n\
         \n  --mo-evaluations N      Total requested MODE evaluations (100000)\n\
         \n  --retries N             Independent BiteOpt retries (8)\n\
         \n  --workers N             Retry/MODE workers; 0 uses available CPUs (0)\n\
         \n  --popsize N              MODE population size (256)\n\
         \n  --depth N               BiteOpt deep populations, 1..36 (6)\n\
         \n  --seed N                Instance and optimizer root seed (65)"
    );
}

fn run_single(problem: &UavProblem, args: &Args) -> Result<(), Box<dyn Error>> {
    let lower = vec![0.0; problem.dimension()];
    let upper = vec![1.0; problem.dimension()];
    let bounds = RetryBounds::new(lower, upper)?;
    let theoretical_best = -problem.total_reward();
    let config = RetryConfig {
        num_retries: args.retries,
        workers: args.workers,
        max_evaluations: args.evaluations,
        seed: args.seed ^ 0xA076_1D64_78BD_642F,
        stop_fitness: theoretical_best,
        capacity: args.retries.min(500),
        ..Default::default()
    };
    let objective = |x: &[f64]| problem.fitness(x);
    let started = Instant::now();
    let result = retry(&objective, &bounds, &config, |objective, context| {
        let mut rng = Rng::new(context.seed);
        let random_guess: Vec<f64> = context
            .bounds
            .lower()
            .iter()
            .zip(context.bounds.upper())
            .map(|(&lower, &upper)| lower + rng.uniform01() * (upper - lower))
            .collect();
        let guess = context.guess.as_deref().unwrap_or(&random_guess);
        let optimized = optimize_bite(
            objective,
            context.bounds.lower(),
            context.bounds.upper(),
            Some(guess),
            &BiteParams {
                max_evaluations: context.max_evaluations,
                stop_fitness: theoretical_best,
                seed: rng.next_u64(),
                runid: context.run_id as i64,
                ..Default::default()
            },
            args.depth,
        );
        RetryRunResult {
            x: optimized.x,
            y: optimized.y,
            evaluations: optimized.evaluations,
        }
    });
    if !result.success {
        return Err("BiteOpt retry returned no finite solution".into());
    }
    let elapsed = started.elapsed().as_secs_f64();
    let solution = problem.solution(&result.x);
    println!(
        "SO reward={:.0} reward_fraction={:.9} max_time={:.6} energy={:.6} evaluations={} retries={} seconds={:.6} evaluations_per_second={:.0}",
        solution.metrics.reward,
        solution.metrics.reward / problem.total_reward(),
        solution.metrics.max_time,
        solution.metrics.energy,
        result.evaluations,
        result.runs,
        elapsed,
        result.evaluations as f64 / elapsed.max(1.0e-9)
    );
    println!("SO_ASSIGNMENTS {:?}", solution.assignments);
    Ok(())
}

fn run_multi(problem: &UavProblem, args: &Args) -> Result<(), Box<dyn Error>> {
    let lower = vec![0.0; problem.dimension()];
    let upper = vec![1.0; problem.dimension()];
    let fitness = Fitness::bounded(problem.dimension(), 3, &lower, &upper);
    let parameters = ModeParams {
        popsize: args.popsize as i32,
        nsga_update: true,
        seed: args.seed ^ 0xE703_7ED1_A0B4_28DB,
        ..Default::default()
    };
    let mut mode = Mode::try_new(fitness, 3, 0, None, &parameters)?;
    let generations = args.mo_evaluations.div_ceil(args.popsize);
    let started = Instant::now();
    for _ in 0..generations {
        let xs = mode.ask();
        let ys = parallel_batch(&xs, args.workers as i32, |x| problem.multi_objective(x));
        mode.tell(&ys);
    }

    let population = mode.population();
    let values: Vec<Vec<f64>> = population
        .iter()
        .map(|x| problem.multi_objective(x))
        .collect();
    let front_indices = pareto_indices(&values, 3)?;
    let elapsed = started.elapsed().as_secs_f64();
    let evaluations = generations * args.popsize;
    let best_reward = front_indices
        .iter()
        .map(|&index| -values[index][0])
        .fold(0.0_f64, f64::max);
    println!(
        "MO pareto={} best_reward={:.0} reward_fraction={:.9} evaluations={} generations={} seconds={:.6} evaluations_per_second={:.0}",
        front_indices.len(),
        best_reward,
        best_reward / problem.total_reward(),
        evaluations,
        generations,
        elapsed,
        evaluations as f64 / elapsed.max(1.0e-9)
    );
    for (rank, &index) in front_indices.iter().take(12).enumerate() {
        let solution = problem.solution(&population[index]);
        println!(
            "MO_POINT rank={} reward={:.0} max_time={:.6} energy={:.6} assignments={:?}",
            rank + 1,
            solution.metrics.reward,
            solution.metrics.max_time,
            solution.metrics.energy,
            solution.assignments
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let problem = UavProblem::generate(args.vehicles, args.targets, args.map_size, args.seed)?;
    println!(
        "CONFIG language=rust problem=multi-uav-task-assignment mode={} vehicles={} targets={} dimension={} map_size={} time_limit={:.6} total_reward={:.0} speeds={:?} workers={} retries={} evaluations_per_retry={} mo_evaluations={} popsize={} depth={} seed={}",
        args.mode.name(),
        problem.vehicle_count(),
        problem.target_count(),
        problem.dimension(),
        args.map_size,
        problem.time_limit(),
        problem.total_reward(),
        problem.speeds(),
        args.workers,
        args.retries,
        args.evaluations,
        args.mo_evaluations,
        args.popsize,
        args.depth,
        args.seed
    );
    if args.mode.includes_single() {
        run_single(&problem, &args)?;
    }
    if args.mode.includes_multi() {
        run_multi(&problem, &args)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> std::vec::IntoIter<String> {
        values
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn defaults_select_small_both_mode() {
        let args = Args::default();
        assert_eq!((args.vehicles, args.targets, args.map_size), (5, 30, 5_000));
        assert_eq!(args.mode, RunMode::Both);
        assert_eq!(args.workers, 0);
    }

    #[test]
    fn parses_presets_and_optimization_controls() {
        let args = Args::from_args(arguments(&[
            "--size",
            "medium",
            "--mode",
            "mo",
            "--evaluations",
            "1000",
            "--mo-evaluations",
            "2000",
            "--workers",
            "16",
            "--retries",
            "32",
            "--popsize",
            "128",
            "--depth",
            "2",
            "--seed",
            "7",
        ]))
        .unwrap();
        assert_eq!(
            (args.vehicles, args.targets, args.map_size),
            (10, 60, 10_000)
        );
        assert_eq!(args.mode, RunMode::Multi);
        assert_eq!(args.workers, 16);
        assert_eq!(args.retries, 32);
        assert_eq!(args.depth, 2);
        assert_eq!(args.seed, 7);
    }

    #[test]
    fn rejects_invalid_arguments() {
        assert!(Args::from_args(arguments(&["--size", "huge"])).is_err());
        assert!(Args::from_args(arguments(&["--mode", "none"])).is_err());
        assert!(Args::from_args(arguments(&["--vehicles", "0"])).is_err());
        assert!(Args::from_args(arguments(&["--popsize", "3"])).is_err());
        assert!(Args::from_args(arguments(&["--depth", "37"])).is_err());
        assert!(Args::from_args(arguments(&["--unknown"])).is_err());
    }

    #[test]
    fn native_single_and_multi_optimizers_smoke() {
        let problem = UavProblem::generate(2, 4, 100, 9).unwrap();
        let args = Args {
            vehicles: 2,
            targets: 4,
            map_size: 100,
            mode: RunMode::Both,
            evaluations: 256,
            mo_evaluations: 256,
            retries: 1,
            workers: 1,
            popsize: 32,
            depth: 1,
            seed: 9,
        };
        run_single(&problem, &args).unwrap();
        run_multi(&problem, &args).unwrap();
    }
}
