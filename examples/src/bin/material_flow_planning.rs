//! Native material-flow simulation and BiteOpt speed example.

use std::env;
use std::error::Error;
use std::time::Instant;

use fcmaes_core::{BiteParams, DeepBiteOpt};
use fcmaes_examples::material_flow_planning::{Plant, benchmark_candidate};

#[derive(Clone, Debug)]
struct Args {
    benchmark_evaluations: usize,
    optimize_evaluations: u64,
    batch_size: usize,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            benchmark_evaluations: 256,
            optimize_evaluations: 256,
            batch_size: 8,
            seed: 42,
        }
    }
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut parsed = Self::default();
        let mut args = env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--benchmark-evaluations" => {
                    parsed.benchmark_evaluations =
                        parse_value(&mut args, "--benchmark-evaluations")?
                }
                "--optimize-evaluations" => {
                    parsed.optimize_evaluations = parse_value(&mut args, "--optimize-evaluations")?
                }
                "--batch-size" => parsed.batch_size = parse_value(&mut args, "--batch-size")?,
                "--seed" => parsed.seed = parse_value(&mut args, "--seed")?,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        if parsed.benchmark_evaluations == 0
            || parsed.optimize_evaluations == 0
            || parsed.batch_size == 0
        {
            return Err("evaluation counts and batch size must be positive".to_string());
        }
        Ok(parsed)
    }
}

fn parse_value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {option}"))?
        .parse()
        .map_err(|_| format!("invalid value for {option}"))
}

fn print_help() {
    println!(
        "Siemens material-flow objective speed example\n\
         \nUsage: cargo run --release -p fcmaes-examples --bin material-flow-planning -- [OPTIONS]\n\
         \n  --benchmark-evaluations N  Fixed objective calls (256)\n\
         \n  --optimize-evaluations N   BiteOpt objective budget (256)\n\
         \n  --batch-size N             BiteOpt ask/tell batch size (8)\n\
         \n  --seed N                   BiteOpt seed (42)"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let plant = Plant::original();
    println!(
        "CONFIG language=rust workers=1 benchmark_evaluations={} optimize_evaluations={} batch_size={} seed={} day_seconds=86400",
        args.benchmark_evaluations, args.optimize_evaluations, args.batch_size, args.seed
    );

    let benchmark_start = Instant::now();
    let mut checksum = 0i64;
    let mut benchmark_best = f64::INFINITY;
    for index in 0..args.benchmark_evaluations {
        let y = plant.fitness(&benchmark_candidate(index));
        checksum += y as i64;
        benchmark_best = benchmark_best.min(y);
    }
    let benchmark_seconds = benchmark_start.elapsed().as_secs_f64();
    println!(
        "OBJECTIVE evaluations={} best={:.0} checksum={} seconds={:.6} evaluations_per_second={:.3}",
        args.benchmark_evaluations,
        benchmark_best,
        checksum,
        benchmark_seconds,
        args.benchmark_evaluations as f64 / benchmark_seconds
    );

    let optimize_start = Instant::now();
    let params = BiteParams {
        max_evaluations: args.optimize_evaluations,
        seed: args.seed,
        ..Default::default()
    };
    let mut optimizer = DeepBiteOpt::new(&[1.0, 1.0], &[50.0, 50.0], None, &params, 1);
    loop {
        let xs = optimizer.ask(args.batch_size);
        if xs.is_empty() {
            break;
        }
        let ys: Vec<f64> = xs.iter().map(|x| plant.fitness(x)).collect();
        optimizer.tell(&ys);
    }
    let result = optimizer.result_public();
    let optimize_seconds = optimize_start.elapsed().as_secs_f64();
    let batches: Vec<usize> = result
        .x
        .iter()
        .map(|value| value.clamp(1.0, 50.0) as usize)
        .collect();
    println!(
        "OPTIMIZE evaluations={} best={:.0} batches={:?} seconds={:.6} evaluations_per_second={:.3}",
        result.evaluations,
        result.y,
        batches,
        optimize_seconds,
        result.evaluations as f64 / optimize_seconds
    );
    println!(
        "RESULT language=rust objective_best={:.0} checksum={} optimize_best={:.0} objective_seconds={:.6} optimize_seconds={:.6}",
        benchmark_best, checksum, result.y, benchmark_seconds, optimize_seconds
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_define_the_matched_benchmark() {
        let args = Args::default();
        assert_eq!(args.benchmark_evaluations, 256);
        assert_eq!(args.optimize_evaluations, 256);
        assert_eq!(args.batch_size, 8);
        assert_eq!(args.seed, 42);
    }
}
