//! EMA/SMA trading strategy optimization with native Rust MODE and
//! MAP-Elites.

use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use fcmaes_core::{
    Archive, Fitness, MapElitesParams, Mode, ModeParams, Rng, map_elites, pareto_indices,
};
use fcmaes_examples::trading::{
    DIM, LOWER, QD_LOWER, QD_UPPER, TradingProblem, UPPER, archive_quality, load_histories,
    parameters, pareto_quality,
};

const DEFAULT_TICKERS: &str = "NVDA,GOOGL,AAPL,MSFT";

#[derive(Clone, Debug)]
struct Args {
    tickers: Vec<String>,
    start: String,
    end: String,
    cache_dir: PathBuf,
    offline: bool,
    mo_evaluations: usize,
    qd_evaluations: usize,
    popsize: usize,
    max_trades: usize,
    capacity: usize,
    chunk_size: usize,
    samples_per_niche: usize,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            tickers: parse_tickers(DEFAULT_TICKERS).expect("valid default tickers"),
            start: "2020-01-01".to_string(),
            end: "2026-01-01".to_string(),
            cache_dir: "ticker_cache".into(),
            offline: false,
            mo_evaluations: 16_384,
            qd_evaluations: 16_384,
            popsize: 64,
            max_trades: 12,
            capacity: 256,
            chunk_size: 64,
            samples_per_niche: 4,
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
                "--tickers" => {
                    parsed.tickers = parse_tickers(&next_value(&mut args, "--tickers")?)?
                }
                "--start" => parsed.start = next_value(&mut args, "--start")?,
                "--end" => parsed.end = next_value(&mut args, "--end")?,
                "--cache-dir" => parsed.cache_dir = next_value(&mut args, "--cache-dir")?.into(),
                "--offline" => parsed.offline = true,
                "--mo-evaluations" => {
                    parsed.mo_evaluations = parse_value(&mut args, "--mo-evaluations")?
                }
                "--qd-evaluations" => {
                    parsed.qd_evaluations = parse_value(&mut args, "--qd-evaluations")?
                }
                "--popsize" => parsed.popsize = parse_value(&mut args, "--popsize")?,
                "--max-trades" => parsed.max_trades = parse_value(&mut args, "--max-trades")?,
                "--capacity" => parsed.capacity = parse_value(&mut args, "--capacity")?,
                "--chunk-size" => parsed.chunk_size = parse_value(&mut args, "--chunk-size")?,
                "--samples-per-niche" => {
                    parsed.samples_per_niche = parse_value(&mut args, "--samples-per-niche")?
                }
                "--seed" => parsed.seed = parse_value(&mut args, "--seed")?,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        if parsed.tickers.len() != 4 {
            return Err("--tickers must contain exactly four comma-separated symbols".to_string());
        }
        if parsed.popsize < 4 || parsed.popsize % 2 != 0 {
            return Err("--popsize must be even and at least four".to_string());
        }
        if parsed.chunk_size < 2 || parsed.chunk_size % 2 != 0 {
            return Err("--chunk-size must be even and at least two".to_string());
        }
        if parsed.capacity == 0 || parsed.mo_evaluations == 0 || parsed.qd_evaluations == 0 {
            return Err("capacities and evaluation budgets must be positive".to_string());
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

fn parse_tickers(value: &str) -> Result<Vec<String>, String> {
    let tickers: Vec<String> = value
        .split(',')
        .map(str::trim)
        .filter(|ticker| !ticker.is_empty())
        .map(str::to_uppercase)
        .collect();
    if tickers.is_empty() {
        return Err("ticker list must not be empty".to_string());
    }
    Ok(tickers)
}

fn print_help() {
    println!(
        "Trading strategy MODE and MAP-Elites example\n\
         \nUsage: cargo run --release -p fcmaes-examples --bin trading -- [OPTIONS]\n\
         \n  --tickers LIST          Four comma-separated symbols ({DEFAULT_TICKERS})\n\
         \n  --start DATE            Inclusive history start (2020-01-01)\n\
         \n  --end DATE              Exclusive history end (2026-01-01)\n\
         \n  --cache-dir PATH        Shared adjusted-close CSV cache (ticker_cache)\n\
         \n  --offline               Refuse network access when a cache is missing\n\
         \n  --mo-evaluations N      MODE budget (16384)\n\
         \n  --qd-evaluations N      MAP-Elites budget (16384)\n\
         \n  --popsize N             MODE population (64)\n\
         \n  --max-trades N          Per-stock trade constraint (12)\n\
         \n  --capacity N            Archive niches (256)\n\
         \n  --chunk-size N          MAP-Elites offspring per generation (64)\n\
         \n  --samples-per-niche N   CVT samples per niche (4)\n\
         \n  --seed N                RNG seed (42)"
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let total_start = Instant::now();
    let histories = load_histories(
        &args.tickers,
        &args.start,
        &args.end,
        &args.cache_dir,
        args.offline,
    )
    .await?;
    let problem = TradingProblem::new(histories, args.max_trades)?;
    println!(
        "CONFIG language=rust tickers={} start={} end={} seed={} workers=1 popsize={} capacity={} chunk_size={} samples_per_niche={} max_trades={}",
        args.tickers.join(","),
        args.start,
        args.end,
        args.seed,
        args.popsize,
        args.capacity,
        args.chunk_size,
        args.samples_per_niche,
        args.max_trades
    );
    println!(
        "DATA rows={:?} hodl={:?}",
        problem.lengths(),
        problem
            .hodl_factors()
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
    );

    let mo_start = Instant::now();
    let mo_generations = args.mo_evaluations.div_ceil(args.popsize);
    let mo_evaluations = mo_generations * args.popsize;
    let fitness = Fitness::bounded(DIM, 2 * problem.nobj(), &LOWER, &UPPER);
    let mode_parameters = ModeParams {
        popsize: args.popsize as i32,
        seed: args.seed,
        nsga_update: true,
        ..Default::default()
    };
    let mut mode = Mode::try_new(
        fitness,
        problem.nobj(),
        problem.nobj(),
        None,
        &mode_parameters,
    )?;
    for _ in 0..mo_generations {
        let xs = mode.ask();
        let ys: Vec<Vec<f64>> = xs.iter().map(|x| problem.mo_values(x)).collect();
        mode.tell(&ys);
    }
    // Re-evaluate the retained population in both language versions so the
    // quality calculation follows exactly the same path.
    let population = mode.population();
    let population_values: Vec<Vec<f64>> =
        population.iter().map(|x| problem.mo_values(x)).collect();
    let feasible_values: Vec<Vec<f64>> = population_values
        .into_iter()
        .filter(|values| values[problem.nobj()..].iter().all(|value| *value <= 0.0))
        .collect();
    let front_indices = pareto_indices(&feasible_values, problem.nobj())?;
    let front: Vec<Vec<f64>> = front_indices
        .iter()
        .map(|&index| feasible_values[index].clone())
        .collect();
    let mo_quality = pareto_quality(&front, problem.nobj());
    let mo_seconds = mo_start.elapsed().as_secs_f64();
    println!(
        "MO evaluations={} feasible={} pareto={} quality={:.12} seconds={:.6}",
        mo_evaluations,
        feasible_values.len(),
        front.len(),
        mo_quality,
        mo_seconds
    );
    for (index, values) in front.iter().take(8).enumerate() {
        println!(
            "MO_POINT rank={} factors={:?} trades={:?}",
            index + 1,
            values[..problem.nobj()]
                .iter()
                .map(|value| format!("{:.6}", -value))
                .collect::<Vec<_>>(),
            values[problem.nobj()..]
                .iter()
                .map(|value| (value + problem.max_trades() as f64) as usize)
                .collect::<Vec<_>>()
        );
    }

    let qd_start = Instant::now();
    let qd_generations = args.qd_evaluations.div_ceil(args.chunk_size);
    let qd_evaluations = qd_generations * args.chunk_size;
    let mut rng = Rng::new(args.seed);
    let mut archive = Archive::try_new(
        DIM,
        &QD_LOWER,
        &QD_UPPER,
        args.capacity,
        args.samples_per_niche,
        &mut rng,
    )?;
    archive.seed_uniform(&LOWER, &UPPER, &mut rng);
    let mut qd_fitness = |x: &[f64]| problem.qd_value(x);
    map_elites(
        &mut archive,
        &mut qd_fitness,
        &LOWER,
        &UPPER,
        &MapElitesParams {
            generations: qd_generations,
            chunk_size: args.chunk_size,
            ..Default::default()
        },
        &mut rng,
    );
    let qd_quality = archive_quality(archive.ys());
    let qd_seconds = qd_start.elapsed().as_secs_f64();
    let coverage = archive.occupied() as f64 / archive.capacity() as f64;
    println!(
        "QD evaluations={} occupied={} capacity={} coverage={:.12} best={:.12} quality={:.12} seconds={:.6}",
        qd_evaluations,
        archive.occupied(),
        archive.capacity(),
        coverage,
        -archive.best_y(),
        qd_quality,
        qd_seconds
    );
    let mut occupied = archive.occupied_data();
    occupied.sort_by(|left, right| left.1.total_cmp(&right.1));
    for (index, (x, y, descriptor)) in occupied.iter().take(8).enumerate() {
        println!(
            "QD_POINT rank={} quality={:.6} factors={:?} x={:?}",
            index + 1,
            -y,
            descriptor
                .iter()
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>(),
            parameters(x)
        );
    }
    println!(
        "RESULT language=rust mo_quality={:.12} qd_quality={:.12} mo_seconds={:.6} qd_seconds={:.6} total_seconds={:.6}",
        mo_quality,
        qd_quality,
        mo_seconds,
        qd_seconds,
        total_start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_cross_language_benchmark() {
        let args = Args::default();
        assert_eq!(args.tickers, ["NVDA", "GOOGL", "AAPL", "MSFT"]);
        assert_eq!(args.mo_evaluations, args.qd_evaluations);
        assert_eq!(args.seed, 42);
    }

    #[test]
    fn ticker_parser_normalizes_and_rejects_empty_input() {
        assert_eq!(parse_tickers(" nvda, AAPL ").unwrap(), ["NVDA", "AAPL"]);
        assert!(parse_tickers(" , ").is_err());
    }
}
