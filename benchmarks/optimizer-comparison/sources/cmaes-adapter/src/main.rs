use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cmaes::restart::{RestartOptions, RestartStrategy};
use cmaes::{CMAESOptions, DVector};
use gtop_benchmark_common::{Adapter, Case, Config, Outcome, run_benchmark};

#[derive(Clone, Copy)]
enum Mode {
    Population,
    Bipop,
}

fn main() {
    if let Err(message) = try_main() {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

fn try_main() -> Result<(), String> {
    let (mode, args) = parse_mode(std::env::args().skip(1))?;
    let Some(config) = Config::from_args(args)? else {
        println!("--mode population|bipop\n{}", Config::USAGE);
        return Ok(());
    };
    let (parallel_mode, solve) = match mode {
        Mode::Population => (
            "parallel-population",
            solve_population as fn(&Case, u64, &Config) -> Result<Outcome, String>,
        ),
        Mode::Bipop => (
            "parallel-bipop",
            solve_bipop as fn(&Case, u64, &Config) -> Result<Outcome, String>,
        ),
    };
    run_benchmark(
        Adapter {
            library: "cmaes",
            version: "0.2.2",
            algorithm: if matches!(mode, Mode::Bipop) {
                "BIPOP-CMA-ES"
            } else {
                "CMA-ES"
            },
            parallel_mode,
        },
        &config,
        solve,
    )
}

fn parse_mode<I, S>(args: I) -> Result<(Mode, Vec<String>), String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut mode = Mode::Population;
    let mut filtered = Vec::new();
    let mut args = args.into_iter().map(Into::into);
    while let Some(argument) = args.next() {
        if argument == "--mode" {
            mode = match args.next().as_deref() {
                Some("population") => Mode::Population,
                Some("bipop") => Mode::Bipop,
                _ => return Err("--mode requires population or bipop".to_owned()),
            };
        } else {
            filtered.push(argument);
        }
    }
    Ok((mode, filtered))
}

fn solve_population(case: &Case, seed: u64, config: &Config) -> Result<Outcome, String> {
    let evaluations = Arc::new(AtomicU64::new(0));
    let evaluation_counter = Arc::clone(&evaluations);
    let budget = config.evaluations;
    let objective = |point: &DVector<f64>| {
        evaluate_with_budget(case, point.as_slice(), &evaluation_counter, budget)
    };
    // A population of 24 exposes one candidate per configured worker.
    let population = config.workers.max(4);
    let mut optimizer = CMAESOptions::new(vec![0.5; case.dimension()], 0.3)
        .population_size(population)
        .max_function_evals(config.evaluations as usize)
        .fun_target(case.stop_fitness)
        // The benchmark has one common stopping rule: target or evaluation
        // budget. Disable the crate's optional tolerance/stagnation limits so
        // a flat population does not silently receive a smaller budget.
        .tol_fun(0.0)
        .tol_fun_rel(0.0)
        .tol_fun_hist(0.0)
        .tol_x(-1.0)
        .tol_stagnation(usize::MAX)
        .tol_x_up(f64::MAX)
        .tol_condition_cov(f64::MAX)
        .seed(seed)
        .build(objective)
        .map_err(|error| format!("{error:?}"))?;
    optimizer.run_parallel();
    let value = optimizer
        .overall_best_individual()
        .map(|individual| individual.value)
        .unwrap_or(f64::INFINITY);
    Ok(Outcome {
        value,
        evaluations: evaluations.load(Ordering::Relaxed),
        workers_used: config.workers,
        population_or_batch: population,
        optimizer_runs: 1,
    })
}

fn solve_bipop(case: &Case, seed: u64, config: &Config) -> Result<Outcome, String> {
    let evaluations = Arc::new(AtomicU64::new(0));
    let restarter = RestartOptions::new(
        case.dimension(),
        0.0..=1.0,
        RestartStrategy::BIPOP(Default::default()),
    )
    .max_function_evals(config.evaluations as usize)
    .max_function_evals_per_run(config.evaluations as usize)
    .fun_target(case.stop_fitness)
    .seed(seed)
    .build()
    .map_err(|error| format!("{error:?}"))?;
    let results = restarter.run_parallel(|| {
        let evaluation_counter = Arc::clone(&evaluations);
        move |point: &DVector<f64>| {
            evaluate_with_budget(
                case,
                point.as_slice(),
                &evaluation_counter,
                config.evaluations,
            )
        }
    });
    let value = results
        .best
        .as_ref()
        .map(|individual| individual.value)
        .unwrap_or(f64::INFINITY);
    Ok(Outcome {
        value,
        evaluations: evaluations.load(Ordering::Relaxed),
        workers_used: config.workers,
        population_or_batch: 0,
        optimizer_runs: results.runs,
    })
}

#[inline]
fn evaluate_reflected(case: &Case, unit: &[f64]) -> f64 {
    let point: Vec<f64> = unit
        .iter()
        .zip(case.lower.iter().zip(&case.upper))
        .map(|(&value, (&lower, &upper))| {
            let period = value.rem_euclid(2.0);
            let reflected = if period <= 1.0 {
                period
            } else {
                2.0 - period
            };
            lower + reflected * (upper - lower)
        })
        .collect();
    (case.objective)(&point)
}

#[inline]
fn evaluate_with_budget(
    case: &Case,
    unit: &[f64],
    evaluations: &AtomicU64,
    budget: u64,
) -> f64 {
    if evaluations
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
            (count < budget).then_some(count + 1)
        })
        .is_err()
    {
        return f64::INFINITY;
    }
    evaluate_reflected(case, unit)
}
