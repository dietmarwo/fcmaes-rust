use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fcmaes_core::{
    AdvancedRetryConfig, BiteParams, Cmaes, CmaesParams, De, DeParams, DeepBiteOpt, Fitness,
    RetryBounds, RetryConfig, RetryContext, RetryRunResult, Rng, advanced_retry, optimize_bite,
    parallel_batch, retry,
};
use gtop_benchmark_common::{Adapter, Case, Config, Outcome, run_benchmark};

#[derive(Clone, Copy)]
enum Mode {
    Retry,
    Advanced,
    Batch,
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
        println!("--mode retry|advanced|batch\n{}", Config::USAGE);
        return Ok(());
    };
    match mode {
        Mode::Retry => run_benchmark(
            Adapter {
                library: "fcmaes",
                version: "0.1.0",
                algorithm: "BiteOpt",
                parallel_mode: "independent-retries",
            },
            &config,
            solve_retry,
        ),
        Mode::Advanced => run_benchmark(
            Adapter {
                library: "fcmaes",
                version: "0.1.0",
                algorithm: "DE→CMA",
                parallel_mode: "coordinated-retries",
            },
            &config,
            solve_advanced,
        ),
        Mode::Batch => {
            // Construct the cached evaluation pool outside measured regions.
            let _: Vec<usize> =
                parallel_batch(&[0_usize, 1], config.workers as i32, |value| *value);
            run_benchmark(
                Adapter {
                    library: "fcmaes",
                    version: "0.1.0",
                    algorithm: "BiteOpt",
                    parallel_mode: "ask-tell-batch",
                },
                &config,
                solve_batch,
            )
        }
    }
}

fn parse_mode<I, S>(args: I) -> Result<(Mode, Vec<String>), String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut mode = None;
    let mut filtered = Vec::new();
    let mut args = args.into_iter().map(Into::into);
    while let Some(argument) = args.next() {
        if argument == "--mode" {
            let value = args
                .next()
                .ok_or_else(|| "--mode requires retry or batch".to_owned())?;
            mode = Some(match value.as_str() {
                "retry" => Mode::Retry,
                "advanced" => Mode::Advanced,
                "batch" => Mode::Batch,
                _ => return Err("--mode requires retry, advanced, or batch".to_owned()),
            });
        } else {
            filtered.push(argument);
        }
    }
    Ok((mode.unwrap_or(Mode::Retry), filtered))
}

fn solve_retry(case: &Case, seed: u64, config: &Config) -> Result<Outcome, String> {
    let bounds = RetryBounds::new(case.lower.clone(), case.upper.clone())
        .map_err(str::to_owned)?;
    let retry_config = RetryConfig {
        num_retries: config.retries,
        workers: config.workers,
        capacity: config.retries,
        value_limit: f64::INFINITY,
        stop_fitness: case.stop_fitness,
        max_evaluations: config.evaluations_per_retry,
        seed,
        statistic_num: 0,
    };
    let result = retry(
        &case.objective,
        &bounds,
        &retry_config,
        |objective, context| bite_retry_run(objective, context, case.stop_fitness),
    );
    Ok(Outcome {
        value: result.y,
        evaluations: result.evaluations,
        workers_used: config.workers,
        population_or_batch: 9 + 3 * case.dimension(),
        optimizer_runs: result.runs,
    })
}

fn solve_advanced(case: &Case, seed: u64, config: &Config) -> Result<Outcome, String> {
    const MAX_EVAL_FAC: f64 = 3.0;
    let num_retries = config
        .retries
        .checked_mul(2)
        .ok_or_else(|| "advanced retry count overflow".to_owned())?;
    // With factor 3 the average retry limit is twice the initial limit.
    let initial_evaluations = config.evaluations / (2 * num_retries as u64);
    if initial_evaluations < 100 {
        return Err("evaluation budget is too small for coordinated DE→CMA retry".to_owned());
    }
    let bounds = RetryBounds::new(case.lower.clone(), case.upper.clone()).map_err(str::to_owned)?;
    let retry_config = RetryConfig {
        num_retries,
        workers: config.workers,
        capacity: num_retries,
        value_limit: f64::INFINITY,
        stop_fitness: case.stop_fitness,
        max_evaluations: initial_evaluations,
        seed,
        statistic_num: 0,
    };
    let advanced_config = AdvancedRetryConfig {
        retry: retry_config,
        check_interval: 4,
        max_eval_fac: MAX_EVAL_FAC,
        crossover_probability: 0.5,
        diversity_threshold: 0.15,
    };
    let evaluations = Arc::new(AtomicU64::new(0));
    let evaluation_counter = Arc::clone(&evaluations);
    let budget = config.evaluations;
    let objective = |point: &[f64]| {
        if evaluation_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < budget).then_some(count + 1)
            })
            .is_err()
        {
            f64::INFINITY
        } else {
            (case.objective)(point)
        }
    };
    let result = advanced_retry(&objective, &bounds, &advanced_config, de_cma_retry_run);
    Ok(Outcome {
        value: result.y,
        evaluations: evaluations.load(Ordering::Relaxed),
        workers_used: config.workers,
        population_or_batch: 0,
        optimizer_runs: result.runs,
    })
}

fn de_cma_retry_run<O>(objective: &O, context: &RetryContext) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let dim = context.bounds.dim();
    let de_budget = (context.max_evaluations * 2 / 5).max(31);
    let cma_budget = context.max_evaluations.saturating_sub(de_budget).max(31);
    let de_fit = Fitness::bounded(dim, 1, context.bounds.lower(), context.bounds.upper());
    let de_sigma: Vec<f64> = context
        .sdev
        .iter()
        .zip(context.bounds.lower().iter().zip(context.bounds.upper()))
        .map(|(&sigma, (&lower, &upper))| sigma * (upper - lower))
        .collect();
    let guess = context.guess.as_deref().unwrap_or(&[]);
    let mut de = De::new(
        de_fit,
        guess,
        if guess.is_empty() { &[] } else { &de_sigma },
        None,
        &DeParams {
            max_evaluations: de_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed,
            runid: context.run_id as i64,
            ..Default::default()
        },
    );
    let de_result = de.optimize(objective);

    let mut cma_fit = Fitness::bounded(dim, 1, context.bounds.lower(), context.bounds.upper());
    cma_fit.set_normalize(true);
    let mut cma = Cmaes::new(
        cma_fit,
        &de_result.x,
        &context.sdev,
        &CmaesParams {
            max_evaluations: cma_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed ^ 0xA076_1D64_78BD_642F,
            runid: context.run_id as i64,
            ..Default::default()
        },
    );
    let cma_result = cma.optimize(objective, 1);
    let (x, y) = if cma_result.y < de_result.y {
        (cma_result.x, cma_result.y)
    } else {
        (de_result.x, de_result.y)
    };
    RetryRunResult {
        x,
        y,
        evaluations: de_result.evaluations + cma_result.evaluations,
    }
}

fn bite_retry_run<O>(
    objective: &O,
    context: &RetryContext,
    stop_fitness: f64,
) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let mut rng = Rng::new(context.seed);
    let sampled_guess: Vec<f64> = context
        .bounds
        .lower()
        .iter()
        .zip(context.bounds.upper())
        .map(|(&lower, &upper)| lower + rng.uniform01() * (upper - lower))
        .collect();
    let guess = context.guess.as_deref().unwrap_or(&sampled_guess);
    let bite_seed = (rng.uniform01() * u32::MAX as f64) as u64;
    let result = optimize_bite(
        objective,
        context.bounds.lower(),
        context.bounds.upper(),
        Some(guess),
        &BiteParams {
            max_evaluations: context.max_evaluations,
            stop_fitness,
            seed: bite_seed,
            runid: context.run_id as i64,
            ..Default::default()
        },
        1,
    );
    RetryRunResult {
        x: result.x,
        y: result.y,
        evaluations: result.evaluations,
    }
}

fn solve_batch(case: &Case, seed: u64, config: &Config) -> Result<Outcome, String> {
    let params = BiteParams {
        max_evaluations: config.evaluations,
        stop_fitness: case.stop_fitness,
        seed,
        runid: 0,
        ..Default::default()
    };
    let mut optimizer = DeepBiteOpt::new(&case.lower, &case.upper, None, &params, 1);
    while optimizer.stop_code() == 0 {
        let candidates = optimizer.ask(config.workers);
        if candidates.is_empty() {
            break;
        }
        let values = parallel_batch(&candidates, config.workers as i32, |x| (case.objective)(x));
        if optimizer.tell(&values) < 0 {
            return Err("BiteOpt rejected its ask/tell batch".to_owned());
        }
    }
    let result = optimizer.result_public();
    Ok(Outcome {
        value: result.y,
        evaluations: result.evaluations,
        workers_used: config.workers,
        population_or_batch: config.workers,
        optimizer_runs: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mode_without_leaking_it_to_common_cli() {
        let (mode, args) = parse_mode(["--runs", "2", "--mode", "batch"]).unwrap();
        assert!(matches!(mode, Mode::Batch));
        assert_eq!(args, ["--runs", "2"]);
        let (mode, _) = parse_mode(["--mode", "advanced"]).unwrap();
        assert!(matches!(mode, Mode::Advanced));
    }
}
