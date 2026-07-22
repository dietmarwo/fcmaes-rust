//! Multi-objective Mazda factory-design example using Rust MODE.

use std::env;
use std::error::Error;

use fcmaes_core::{Fitness, Mode, ModeParams, parallel_batch, pareto_indices};
use fcmaes_examples::mazda::{
    MAZDA_CONSTRAINTS, MAZDA_DIM, MAZDA_OBJECTIVES, MAZDA_VALUE_WIDTH, MazdaDecisionSpace,
    MazdaEvaluator, is_feasible,
};

struct Args {
    evaluations: usize,
    popsize: usize,
    workers: i32,
    seed: u64,
    nsga_update: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        Self::from_args(env::args().skip(1))
    }

    fn from_args(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut parsed = Self {
            evaluations: 100_000,
            popsize: 256,
            workers: 1,
            seed: 42,
            nsga_update: true,
        };
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--evaluations" => parsed.evaluations = parse_value(&mut args, "--evaluations")?,
                "--popsize" => parsed.popsize = parse_value(&mut args, "--popsize")?,
                "--workers" => parsed.workers = parse_value(&mut args, "--workers")?,
                "--seed" => parsed.seed = parse_value(&mut args, "--seed")?,
                "--de-update" => parsed.nsga_update = false,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        if parsed.popsize < 4 {
            return Err("--popsize must be at least four".to_string());
        }
        if parsed.workers < 0 {
            return Err("--workers must be non-negative".to_string());
        }
        Ok(parsed)
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {option}"))
}

fn parse_value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    next_value(args, option)?
        .parse()
        .map_err(|_| format!("invalid value for {option}"))
}

fn print_help() {
    println!(
        "Mazda multi-objective MODE example\n\
         \nUsage: cargo run --release -p fcmaes-examples --bin mazda-mo -- [OPTIONS]\n\
         \n  --evaluations N      Evaluation budget (100000)\n\
         \n  --popsize N          MODE population size (256)\n\
         \n  --workers N          Evaluation threads; 0 uses available parallelism (1)\n\
         \n  --seed N             RNG seed (42)\n\
         \n  --de-update           Use MODE's DE update instead of NSGA-II"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let space = MazdaDecisionSpace::new()?;
    let evaluator = MazdaEvaluator::new()?;
    let lower = space.lower();
    let upper = space.upper();
    let fitness = Fitness::bounded(MAZDA_DIM, MAZDA_VALUE_WIDTH, &lower, &upper);
    let params = ModeParams {
        popsize: args.popsize as i32,
        nsga_update: args.nsga_update,
        seed: args.seed,
        ..Default::default()
    };
    let mut mode = Mode::try_new(
        fitness,
        MAZDA_OBJECTIVES,
        MAZDA_CONSTRAINTS,
        Some(vec![true; MAZDA_DIM]),
        &params,
    )?;
    eprintln!(
        "Mazda MODE configuration: workers={} popsize={} budget={}",
        args.workers, args.popsize, args.evaluations
    );

    let generations = args.evaluations.div_ceil(args.popsize);
    let mut evaluations = 0usize;
    for generation in 0..generations {
        let xs = mode.ask();
        let evaluated =
            parallel_batch(&xs, args.workers, |x| evaluator.evaluate_indices(&space, x));
        let ys: Vec<Vec<f64>> = evaluated.into_iter().collect::<Result<_, _>>()?;
        evaluations += ys.len();
        mode.tell(&ys);
        if generation == 0 || (generation + 1) % 25 == 0 || generation + 1 == generations {
            let feasible = ys.iter().filter(|values| is_feasible(values)).count();
            eprintln!(
                "generation={} evaluations={} feasible_offspring={}",
                generation + 1,
                evaluations,
                feasible
            );
        }
    }

    let result = mode.result();
    let feasible: Vec<(usize, &Vec<f64>)> = result
        .y
        .iter()
        .enumerate()
        .filter(|(_, values)| is_feasible(values))
        .collect();
    let feasible_values: Vec<Vec<f64>> = feasible.iter().map(|(_, y)| (*y).clone()).collect();
    let front = pareto_indices(&feasible_values, MAZDA_OBJECTIVES)?;

    println!(
        "Mazda MODE: dim={} objectives={} constraints={} workers={} evaluations={} feasible={} pareto={}",
        space.dim(),
        MAZDA_OBJECTIVES,
        MAZDA_CONSTRAINTS,
        args.workers,
        evaluations,
        feasible.len(),
        front.len()
    );
    for &front_index in front.iter().take(30) {
        let (population_index, values) = feasible[front_index];
        println!(
            "population={} mass={:.8} common_parts={:.0}",
            population_index, values[0], -values[1]
        );
    }
    if feasible.is_empty() {
        let least_violation = result
            .y
            .iter()
            .map(|values| {
                values[MAZDA_OBJECTIVES..]
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max)
            })
            .fold(f64::INFINITY, f64::min);
        println!("no feasible population member; least max violation={least_violation:.6}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> std::vec::IntoIter<String> {
        values
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn parses_worker_count_and_rejects_negative_values() {
        assert_eq!(Args::from_args(args(&[])).unwrap().workers, 1);
        assert_eq!(
            Args::from_args(args(&["--workers", "16"])).unwrap().workers,
            16
        );
        assert!(Args::from_args(args(&["--workers", "-1"])).is_err());
    }
}
