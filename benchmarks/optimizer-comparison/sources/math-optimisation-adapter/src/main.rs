use gtop_benchmark_common::{Adapter, Case, Config, Outcome, run_benchmark};
use math_audio_optimisation::{
    CallbackAction, DEConfigBuilder, Strategy, differential_evolution,
};

const POPULATION_MULTIPLIER: usize = 15;

fn main() {
    if let Err(message) = try_main() {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

fn try_main() -> Result<(), String> {
    let Some(config) = Config::from_env()? else {
        println!("{}", Config::USAGE);
        return Ok(());
    };
    run_benchmark(
        Adapter {
            library: "math-optimisation",
            version: "0.5.10",
            algorithm: "DE/best/1/bin",
            parallel_mode: "parallel-population",
        },
        &config,
        solve,
    )
}

fn solve(case: &Case, seed: u64, config: &Config) -> Result<Outcome, String> {
    let population = POPULATION_MULTIPLIER * case.dimension();
    let generations = config
        .evaluations
        .checked_div(population as u64)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| "evaluation budget is smaller than the DE population".to_owned())?
        as usize;
    let target = case.stop_fitness;
    let de_config = DEConfigBuilder::new()
        .maxiter(generations)
        .popsize(POPULATION_MULTIPLIER)
        .strategy(Strategy::Best1Bin)
        .seed(seed)
        .tol(0.0)
        // This crate always checks std(population) <= atol + tol*|mean|.
        // A negative absolute threshold is its only way to disable that
        // unrelated convergence exit and retain the shared target/budget
        // stopping rule.
        .atol(-1.0)
        .enable_parallel(true)
        .parallel_threads(config.workers)
        .callback(Box::new(move |state| {
            if state.fun < target {
                CallbackAction::Stop
            } else {
                CallbackAction::Continue
            }
        }))
        .build()
        .map_err(|error| error.to_string())?;
    let bounds: Vec<(f64, f64)> = case
        .lower
        .iter()
        .copied()
        .zip(case.upper.iter().copied())
        .collect();
    let report = differential_evolution(
        &|point| (case.objective)(point.as_slice().expect("contiguous DE point")),
        &bounds,
        de_config,
    )
    .map_err(|error| error.to_string())?;
    Ok(Outcome {
        value: report.fun,
        evaluations: report.nfev as u64,
        workers_used: config.workers,
        population_or_batch: population,
        optimizer_runs: 1,
    })
}
