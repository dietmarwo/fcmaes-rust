// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

mod model;

use std::env;
use std::error::Error;
use std::time::Instant;

use fcmaes_core::{
    AdvancedRetryConfig, BiteParams, Cmaes, CmaesParams, De, DeParams, Fitness, RetryConfig,
    RetryContext, RetryResult, RetryRunResult, advanced_retry, optimize_bite, retry,
};
use model::{
    JPL_SCORE, WINNING_DECISION, bounds, evaluate, minimum_solar_distance_au, objective,
    refinement_bounds,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Algorithm {
    Inspect,
    DeCma,
    Cma,
    Bite,
}

impl Algorithm {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "inspect" => Ok(Self::Inspect),
            "de-cma" => Ok(Self::DeCma),
            "cma" => Ok(Self::Cma),
            "bite" => Ok(Self::Bite),
            _ => Err("--algorithm must be inspect, de-cma, cma, or bite".to_owned()),
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    algorithm: Algorithm,
    broad: bool,
    fraction: f64,
    stages: usize,
    workers: usize,
    retries: usize,
    evaluations: u64,
    max_eval_fac: f64,
    seed: u64,
    stop: f64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Inspect,
            broad: false,
            fraction: 0.01,
            stages: 1,
            workers: 0,
            retries: 128,
            evaluations: 500_000,
            max_eval_fac: 20.0,
            seed: 900,
            stop: -1_851_000.0,
        }
    }
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut parsed = Self::default();
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            if argument == "--broad" {
                parsed.broad = true;
                continue;
            }
            if matches!(argument.as_str(), "--help" | "-h") {
                help();
                std::process::exit(0);
            }
            let value = arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))?;
            match argument.as_str() {
                "--algorithm" => parsed.algorithm = Algorithm::parse(&value)?,
                "--fraction" => parsed.fraction = parse(&value, "--fraction")?,
                "--stages" => parsed.stages = parse(&value, "--stages")?,
                "--workers" => parsed.workers = parse(&value, "--workers")?,
                "--retries" => parsed.retries = parse(&value, "--retries")?,
                "--evaluations" => parsed.evaluations = parse(&value, "--evaluations")?,
                "--max-eval-fac" => parsed.max_eval_fac = parse(&value, "--max-eval-fac")?,
                "--seed" => parsed.seed = parse(&value, "--seed")?,
                "--stop" => parsed.stop = parse(&value, "--stop")?,
                _ => return Err(format!("unknown option {argument}")),
            }
        }
        if parsed.stages == 0 || parsed.retries == 0 || parsed.evaluations == 0 {
            return Err("stages, retries, and evaluations must be positive".to_owned());
        }
        if !parsed.fraction.is_finite() || parsed.fraction <= 0.0 {
            return Err("--fraction must be positive".to_owned());
        }
        if !parsed.max_eval_fac.is_finite() || parsed.max_eval_fac < 1.0 {
            return Err("--max-eval-fac must be at least one".to_owned());
        }
        if !parsed.stop.is_finite() {
            return Err("--stop must be finite".to_owned());
        }
        if parsed.broad && parsed.algorithm != Algorithm::DeCma {
            return Err("--broad is supported only by de-cma".to_owned());
        }
        Ok(parsed)
    }
}

fn parse<T: std::str::FromStr>(value: &str, option: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("{option} is invalid"))
}

fn help() {
    println!(
        "Real GTOC1 EVEEEJSJA Rust tutorial\n\
         \nUsage: cargo run --release -- [OPTIONS]\n\
         \n  --algorithm NAME    inspect, de-cma, cma, or bite (inspect)\n\
         \n  --broad             use the complete box with coordinated DE-CMA-ES\n\
         \n  --fraction N        incumbent refinement-box fraction (0.01)\n\
         \n  --stages N          successive strict-improvement stages (1)\n\
         \n  --workers N         worker threads; 0 uses all logical CPUs (0)\n\
         \n  --retries N         parallel retry cap (128)\n\
         \n  --evaluations N     initial evaluations per retry (500000)\n\
         \n  --max-eval-fac N    coordinated retry maximum budget factor (20)\n\
         \n  --seed N            deterministic root seed (900)\n\
         \n  --stop N            early-stop objective; must beat incumbent (-1851000)"
    );
}

fn de_cma_run<O>(
    function: &O,
    context: &RetryContext,
    initial_guess: Option<&[f64]>,
) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let dimension = context.bounds.dim();
    let de_budget = if initial_guess.is_some() {
        (context.max_evaluations / 10).max(31)
    } else {
        (context.max_evaluations * 2 / 5).max(31)
    };
    let cma_budget = context.max_evaluations.saturating_sub(de_budget).max(31);
    let guess = context.guess.as_deref().or(initial_guess).unwrap_or(&[]);
    let sigma = context
        .sdev
        .iter()
        .zip(context.bounds.lower().iter().zip(context.bounds.upper()))
        .map(|(&value, (&lower, &upper))| value * (upper - lower))
        .collect::<Vec<_>>();
    let de_fitness = Fitness::bounded(dimension, 1, context.bounds.lower(), context.bounds.upper());
    let mut de = De::new(
        de_fitness,
        guess,
        if guess.is_empty() { &[] } else { &sigma },
        None,
        &DeParams {
            max_evaluations: de_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed,
            runid: i64::try_from(context.run_id).expect("retry identifier fits i64"),
            ..Default::default()
        },
    );
    let de_result = de.optimize(function);

    let mut cma_fitness =
        Fitness::bounded(dimension, 1, context.bounds.lower(), context.bounds.upper());
    cma_fitness.set_normalize(true);
    let mut cma = Cmaes::new(
        cma_fitness,
        &de_result.x,
        &context.sdev,
        &CmaesParams {
            max_evaluations: cma_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed ^ 0xA076_1D64_78BD_642F,
            runid: i64::try_from(context.run_id).expect("retry identifier fits i64"),
            ..Default::default()
        },
    );
    let cma_result = cma.optimize(function, 1);
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

fn cma_run<O>(function: &O, context: &RetryContext, initial_guess: &[f64]) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let mut fitness = Fitness::bounded(
        context.bounds.dim(),
        1,
        context.bounds.lower(),
        context.bounds.upper(),
    );
    fitness.set_normalize(true);
    let guess = context.guess.as_deref().unwrap_or(initial_guess);
    let mut cma = Cmaes::new(
        fitness,
        guess,
        &context.sdev,
        &CmaesParams {
            max_evaluations: context.max_evaluations,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed,
            runid: i64::try_from(context.run_id).expect("retry identifier fits i64"),
            ..Default::default()
        },
    );
    let result = cma.optimize(function, 1);
    RetryRunResult {
        x: result.x,
        y: result.y,
        evaluations: result.evaluations,
    }
}

fn bite_run<O>(function: &O, context: &RetryContext, initial_guess: &[f64]) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let guess = context.guess.as_deref().unwrap_or(initial_guess);
    let result = optimize_bite(
        function,
        context.bounds.lower(),
        context.bounds.upper(),
        Some(guess),
        &BiteParams {
            max_evaluations: context.max_evaluations,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed,
            runid: i64::try_from(context.run_id).expect("retry identifier fits i64"),
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

fn validate_stop(stop: f64, incumbent: f64) -> Result<(), String> {
    if stop < incumbent {
        Ok(())
    } else {
        Err(format!(
            "--stop {stop} must be lower than the stored incumbent objective {incumbent}"
        ))
    }
}

fn report(kind: &str, x: &[f64], wall_seconds: Option<f64>) -> Result<(), Box<dyn Error>> {
    let result = evaluate(x)?;
    println!(
        "{kind}_RESULT objective={:.12} score={:.12} beats_jpl={} final_mass_kg={:.9}",
        result.objective,
        result.score,
        result.score > JPL_SCORE,
        result.final_mass_kg
    );
    println!(
        "{kind}_FEASIBILITY mismatch_norm={:.12e} powered_delta_v_km_s={:.12e} \
         minimum_periapsis_margin_km={:.6}",
        result.mismatch_norm, result.powered_delta_v_km_s, result.minimum_periapsis_margin_km
    );
    println!(
        "{kind}_SOLAR minimum_distance_au={:.12}",
        minimum_solar_distance_au(x)?
    );
    println!("{kind}_EPOCHS mjd2000={:?}", result.epochs_mjd2000);
    if let Some(seconds) = wall_seconds {
        println!("TIMING wall_seconds={seconds:.6}");
    }
    println!("{kind}_DECISION x={x:?}");
    Ok(())
}

fn report_stage(index: usize, result: &RetryResult) {
    match evaluate(&result.x) {
        Ok(evaluation) => println!(
            "STAGE_RESULT index={index} model_valid=true objective={:.12} score={:.12} \
             constraint_penalty={:.12}",
            result.y,
            evaluation.score,
            (result.y + evaluation.score).max(0.0)
        ),
        Err(error) => println!(
            "STAGE_RESULT index={index} model_valid=false objective={:.12} error={error}",
            result.y
        ),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    if args.algorithm == Algorithm::Inspect {
        return report("STORED", &WINNING_DECISION, None);
    }

    let stored_objective = objective(&WINNING_DECISION);
    validate_stop(args.stop, stored_objective)?;
    println!(
        "CONFIG algorithm={:?} broad={} workers={} seed={} stages={} retries={} \
         evaluations={} fraction={} stop={} stored_incumbent={}",
        args.algorithm,
        args.broad,
        args.workers,
        args.seed,
        args.stages,
        args.retries,
        args.evaluations,
        args.fraction,
        args.stop,
        stored_objective
    );
    let started = Instant::now();
    let mut incumbent = WINNING_DECISION.to_vec();
    let mut best = RetryResult {
        y: objective(&incumbent),
        x: incumbent.clone(),
        evaluations: 0,
        runs: 0,
        success: true,
        entries: Vec::new(),
        improvements: Vec::new(),
    };
    let mut improved_stages = 0;
    for stage in 0..args.stages {
        let search_bounds = if args.broad {
            bounds()
        } else {
            refinement_bounds(&incumbent, args.fraction)
        };
        let guess = (!args.broad).then_some(incumbent.as_slice());
        let retry_config = RetryConfig {
            num_retries: args.retries,
            workers: args.workers,
            max_evaluations: args.evaluations,
            seed: args
                .seed
                .wrapping_add(u64::try_from(stage).expect("stage index fits u64")),
            value_limit: f64::INFINITY,
            stop_fitness: args.stop,
            statistic_num: 100,
            ..Default::default()
        };
        let result = match args.algorithm {
            Algorithm::DeCma => advanced_retry(
                &objective,
                &search_bounds,
                &AdvancedRetryConfig {
                    retry: retry_config,
                    check_interval: 100,
                    max_eval_fac: args.max_eval_fac,
                    ..Default::default()
                },
                |function, context| de_cma_run(function, context, guess),
            ),
            Algorithm::Cma => retry(
                &objective,
                &search_bounds,
                &retry_config,
                |function, context| {
                    cma_run(
                        function,
                        context,
                        guess.expect("local CMA-ES has an incumbent"),
                    )
                },
            ),
            Algorithm::Bite => retry(
                &objective,
                &search_bounds,
                &retry_config,
                |function, context| {
                    bite_run(
                        function,
                        context,
                        guess.expect("local BiteOpt has an incumbent"),
                    )
                },
            ),
            Algorithm::Inspect => unreachable!("inspect returned before optimization"),
        };
        let improved = result.y < best.y;
        println!(
            "STAGE index={} objective={:.12} evaluations={} retries={} improved={improved}",
            stage + 1,
            result.y,
            result.evaluations,
            result.runs
        );
        report_stage(stage + 1, &result);
        if improved {
            incumbent.clone_from(&result.x);
            best = result;
            improved_stages += 1;
        }
        if best.y <= args.stop {
            break;
        }
    }
    let wall_seconds = started.elapsed().as_secs_f64();
    if improved_stages == 0 {
        println!(
            "NO_IMPROVEMENT optimized_result=false stored_incumbent_retained=true \
             stages_completed={}",
            args.stages
        );
        report("INCUMBENT", &best.x, Some(wall_seconds))
    } else {
        println!("OPTIMIZATION_SUMMARY optimized_result=true improved_stages={improved_stages}");
        report("OPTIMIZED", &best.x, Some(wall_seconds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algorithm_names_are_explicit() {
        assert_eq!(Algorithm::parse("inspect").unwrap(), Algorithm::Inspect);
        assert_eq!(Algorithm::parse("de-cma").unwrap(), Algorithm::DeCma);
        assert_eq!(Algorithm::parse("cma").unwrap(), Algorithm::Cma);
        assert_eq!(Algorithm::parse("bite").unwrap(), Algorithm::Bite);
        assert!(Algorithm::parse("unknown").is_err());
    }

    #[test]
    fn stop_target_must_improve_the_stored_incumbent() {
        let incumbent = objective(&WINNING_DECISION);
        assert!(validate_stop(-1_851_000.0, incumbent).is_ok());
        assert!(validate_stop(-1_850_001.0, incumbent).is_err());
        assert!(Args::default().stop < incumbent);
    }
}
