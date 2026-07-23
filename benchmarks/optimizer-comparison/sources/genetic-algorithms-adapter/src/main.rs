use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use genetic_algorithms::chromosomes::Range as RangeChromosome;
use genetic_algorithms::de::{DeAdaptive, DeConfiguration, DeEngine, DeMutationStrategy};
use genetic_algorithms::genotypes::Range as RangeGene;
use genetic_algorithms::rng;
use genetic_algorithms::traits::{LinearChromosome, RealGene};
use gtop_benchmark_common::{Adapter, Case, Config, Outcome, run_benchmark};
use rand::Rng;

const POPULATION: usize = 48;

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
            library: "genetic_algorithms",
            version: "3.0.0",
            algorithm: "L-SHADE",
            // DeEngine 3.0.0 has no parallel DE evaluation API.
            parallel_mode: "native-serial",
        },
        &config,
        solve,
    )
}

fn solve(case: &Case, seed: u64, config: &Config) -> Result<Outcome, String> {
    rng::set_seed(Some(seed));
    let dimensions = case.dimension();
    let lower = case.lower.clone();
    let upper = case.upper.clone();
    let fitness_lower = case.lower.clone();
    let fitness_upper = case.upper.clone();
    let objective = case.objective;
    let evaluations = Arc::new(AtomicU64::new(0));
    let evaluation_counter = Arc::clone(&evaluations);
    let generations = config
        .evaluations
        .checked_div(POPULATION as u64)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| "evaluation budget is smaller than the DE population".to_owned())?
        as usize;

    let initialize = move |size: usize| {
        let mut random = rng::make_rng();
        (0..size)
            .map(|_| {
                let genes: Vec<RangeGene<f64>> = (0..dimensions)
                    .map(|index| {
                        let value = random.random_range(lower[index]..=upper[index]);
                        RangeGene::new(
                            index as i32,
                            vec![(lower[index], upper[index])],
                            value,
                        )
                    })
                    .collect();
                let mut chromosome = RangeChromosome::<f64>::default();
                chromosome.set_dna(Cow::Owned(genes));
                chromosome
            })
            .collect()
    };
    let fitness = move |genes: &[RangeGene<f64>]| {
        evaluation_counter.fetch_add(1, Ordering::Relaxed);
        // DeEngine's mutation arithmetic does not apply RangeGene bounds.
        // Reflect every trial into the declared box before evaluating it so
        // infeasible coordinates cannot receive credit.
        let point: Vec<f64> = genes
            .iter()
            .zip(fitness_lower.iter().zip(&fitness_upper))
            .map(|(gene, (&lower, &upper))| {
                reflect(RealGene::real_value(gene), lower, upper)
            })
            .collect();
        objective(&point)
    };
    let de_config = DeConfiguration::default()
        .with_population_size(POPULATION)
        .with_max_generations(generations)
        .with_mutation_strategy(DeMutationStrategy::Rand1)
        .with_adaptive(DeAdaptive::LShade { history_size: 5 })
        .with_fitness_target(case.stop_fitness);
    let mut engine: DeEngine<RangeChromosome<f64>> =
        DeEngine::new(de_config, initialize, fitness);
    let result = engine.run();

    Ok(Outcome {
        value: result.best_fitness,
        evaluations: evaluations.load(Ordering::Relaxed),
        workers_used: 1,
        population_or_batch: POPULATION,
        optimizer_runs: 1,
    })
}

#[inline]
fn reflect(value: f64, lower: f64, upper: f64) -> f64 {
    let unit = (value - lower) / (upper - lower);
    let period = unit.rem_euclid(2.0);
    let reflected = if period <= 1.0 {
        period
    } else {
        2.0 - period
    };
    lower + reflected * (upper - lower)
}

#[cfg(test)]
mod tests {
    use super::reflect;

    #[test]
    fn reflection_preserves_and_folds_the_box() {
        assert_eq!(reflect(2.0, 0.0, 10.0), 2.0);
        assert!((reflect(12.0, 0.0, 10.0) - 8.0).abs() < 1.0e-12);
        assert!((reflect(-2.0, 0.0, 10.0) - 2.0).abs() < 1.0e-12);
    }
}
