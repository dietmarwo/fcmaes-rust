use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::particleswarm::ParticleSwarm;
use gtop_benchmark_common::{Adapter, Case, Config, Objective, Outcome, run_benchmark};
use rand::SeedableRng;
use rand::rngs::StdRng;

const PARTICLES: usize = 48;

#[derive(Clone)]
struct GtopCost {
    objective: Objective,
    evaluations: Arc<AtomicU64>,
}

impl CostFunction for GtopCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, point: &Self::Param) -> Result<Self::Output, Error> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        Ok((self.objective)(point))
    }
}

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
            library: "argmin",
            version: "0.11.0",
            algorithm: "PSO",
            parallel_mode: "parallel-population",
        },
        &config,
        solve,
    )
}

fn solve(case: &Case, seed: u64, config: &Config) -> Result<Outcome, String> {
    let evaluations = Arc::new(AtomicU64::new(0));
    let problem = GtopCost {
        objective: case.objective,
        evaluations: Arc::clone(&evaluations),
    };
    let iterations = config
        .evaluations
        .checked_div(PARTICLES as u64)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| "evaluation budget is smaller than the PSO population".to_owned())?;
    let solver: ParticleSwarm<Vec<f64>, f64, _> =
        ParticleSwarm::new((case.lower.clone(), case.upper.clone()), PARTICLES)
            .with_rng_generator(StdRng::seed_from_u64(seed));
    let result = Executor::new(problem, solver)
        .configure(|state| {
            state
                .max_iters(iterations)
                .target_cost(case.stop_fitness)
        })
        .run()
        .map_err(|error| error.to_string())?;
    Ok(Outcome {
        value: result.state().get_best_cost(),
        evaluations: evaluations.load(Ordering::Relaxed),
        workers_used: config.workers,
        population_or_batch: PARTICLES,
        optimizer_runs: 1,
    })
}

