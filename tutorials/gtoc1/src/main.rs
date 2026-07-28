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
use model::tour;
use model::{
    JPL_SCORE, VALIDATED_DECISION, bounds, dop853_validation, evaluate, objective,
    refinement_bounds, repair_zoh,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Algorithm {
    Inspect,
    ZohRepair,
    TourInspect,
    TourDeCma,
    TourBite,
    TourMesh,
    DeCma,
    Cma,
    Bite,
}

impl Algorithm {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "inspect" => Ok(Self::Inspect),
            "zoh-repair" => Ok(Self::ZohRepair),
            "tour-inspect" => Ok(Self::TourInspect),
            "tour-de-cma" => Ok(Self::TourDeCma),
            "tour-bite" => Ok(Self::TourBite),
            "tour-mesh" => Ok(Self::TourMesh),
            "de-cma" => Ok(Self::DeCma),
            "cma" => Ok(Self::Cma),
            "bite" => Ok(Self::Bite),
            _ => Err(
                "--algorithm must be inspect, zoh-repair, tour-inspect, tour-de-cma, \
                 tour-bite, tour-mesh, de-cma, cma, or bite"
                    .to_owned(),
            ),
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    algorithm: Algorithm,
    segments_per_leg: usize,
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
            segments_per_leg: 5,
            broad: false,
            fraction: 0.01,
            stages: 1,
            workers: 0,
            retries: 128,
            evaluations: 500_000,
            max_eval_fac: 20.0,
            seed: 900,
            stop: -1_843_301.0,
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
                "--segments-per-leg" => {
                    parsed.segments_per_leg = parse(&value, "--segments-per-leg")?
                }
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
        if !(5..=8).contains(&parsed.segments_per_leg) {
            return Err("--segments-per-leg must be in 5..=8".to_owned());
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
         \n  --algorithm NAME    inspect, zoh-repair, tour-inspect, tour-de-cma,\n\
         \n                      tour-bite, tour-mesh, de-cma, cma, or bite (inspect)\n\
         \n  --segments-per-leg N  whole-tour ZOH segments on each leg, 5..=8 (5)\n\
         \n  --broad             use the complete box with coordinated DE-CMA-ES\n\
         \n  --fraction N        incumbent refinement-box fraction (0.01)\n\
         \n  --stages N          optimization stages or ZOH repair iterations (1)\n\
         \n  --workers N         worker threads; 0 uses all logical CPUs (0)\n\
         \n  --retries N         parallel retry cap (128)\n\
         \n  --evaluations N     initial evaluations per retry (500000)\n\
         \n  --max-eval-fac N    coordinated retry maximum budget factor (20)\n\
         \n  --seed N            deterministic root seed (900)\n\
         \n  --stop N            early-stop objective; must beat incumbent (-1843301)"
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
    let validation = dop853_validation(x)?;
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
        "{kind}_VALIDATION taylor_mismatch_norm={:.12e} dop853_mismatch_norm={:.12e} \
         maximum_backend_difference={:.12e}",
        validation.taylor_mismatch_norm,
        validation.dop853_mismatch_norm,
        validation.maximum_backend_difference
    );
    println!(
        "{kind}_SOLAR minimum_distance_au={:.12}",
        validation.minimum_solar_distance_au
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

fn report_tour(
    kind: &str,
    x: &[f64],
    segments_per_leg: usize,
    wall_seconds: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let validation = tour::validate_backends(x, segments_per_leg)?;
    println!(
        "{kind}_TOUR_RESULT segments_per_leg={} objective={:.12} score={:.12} \
         final_mass_kg={:.9}",
        segments_per_leg,
        validation.taylor.objective,
        validation.taylor.score,
        validation.taylor.final_mass_kg
    );
    println!(
        "{kind}_TOUR_FEASIBILITY taylor_position_mismatch={:.12e} \
         dop853_position_mismatch={:.12e} maximum_component={:.12e} \
         maximum_backend_difference={:.12e}",
        validation.taylor.position_mismatch_norm,
        validation.dop853.position_mismatch_norm,
        validation.dop853.maximum_position_mismatch,
        validation.maximum_backend_difference
    );
    if let Some(seconds) = wall_seconds {
        println!("TIMING wall_seconds={seconds:.6}");
    }
    println!("{kind}_TOUR_DECISION x={x:?}");
    Ok(())
}

fn optimize_tour(args: &Args, segments: usize, initial: &[f64]) -> RetryResult {
    let limits = tour::bounds(segments);
    let function = |x: &[f64]| tour::objective(x, segments);
    let config = RetryConfig {
        num_retries: args.retries,
        workers: args.workers,
        max_evaluations: args.evaluations,
        seed: args
            .seed
            .wrapping_add(u64::try_from(segments - 5).expect("supported segment count fits u64")),
        value_limit: f64::INFINITY,
        stop_fitness: args.stop,
        statistic_num: 100,
        ..Default::default()
    };
    println!(
        "TOUR_CONFIG algorithm={:?} segments_per_leg={} dimension={} workers={} \
         retries={} evaluations={} seed={} stop={}",
        args.algorithm,
        segments,
        tour::dimension(segments),
        args.workers,
        args.retries,
        args.evaluations,
        config.seed,
        args.stop
    );
    let mut result = match args.algorithm {
        Algorithm::TourDeCma | Algorithm::TourMesh => advanced_retry(
            &function,
            &limits,
            &AdvancedRetryConfig {
                retry: config,
                check_interval: 100,
                max_eval_fac: args.max_eval_fac,
                ..Default::default()
            },
            |objective, context| de_cma_run(objective, context, Some(initial)),
        ),
        Algorithm::TourBite => retry(&function, &limits, &config, |objective, context| {
            bite_run(objective, context, initial)
        }),
        _ => unreachable!("run_tour accepts only whole-tour algorithms"),
    };
    let incumbent = tour::objective(initial, segments);
    let improved = result.y < incumbent;
    if !improved {
        result.y = incumbent;
        result.x = initial.to_vec();
    }
    println!(
        "TOUR_OPTIMIZATION objective={:.12} evaluations={} retries={} improved={improved}",
        result.y, result.evaluations, result.runs,
    );
    result
}

fn run_tour(args: &Args) -> Result<(), Box<dyn Error>> {
    let segments = args.segments_per_leg;
    let initial = tour::seed(segments)?;
    if args.algorithm == Algorithm::TourInspect {
        return report_tour("SEED", &initial, segments, None);
    }
    let started = Instant::now();
    if args.algorithm == Algorithm::TourMesh {
        let mut mesh = 5;
        let mut incumbent = tour::seed(mesh)?;
        loop {
            let stage_started = Instant::now();
            let result = optimize_tour(args, mesh, &incumbent);
            report_tour(
                "MESH_STAGE",
                &result.x,
                mesh,
                Some(stage_started.elapsed().as_secs_f64()),
            )?;
            if mesh == segments {
                return report_tour(
                    "OPTIMIZED",
                    &result.x,
                    mesh,
                    Some(started.elapsed().as_secs_f64()),
                );
            }
            let next_mesh = mesh + 1;
            let resampled = tour::resample(&result.x, mesh, next_mesh)?;
            let fresh = tour::seed(next_mesh)?;
            let resampled_objective = tour::objective(&resampled, next_mesh);
            let fresh_objective = tour::objective(&fresh, next_mesh);
            let use_resampled = resampled_objective <= fresh_objective;
            println!(
                "MESH_TRANSFER from={} to={} resampled_objective={:.12} \
                 fresh_objective={:.12} selected={}",
                mesh,
                next_mesh,
                resampled_objective,
                fresh_objective,
                if use_resampled { "resampled" } else { "fresh" }
            );
            incumbent = if use_resampled { resampled } else { fresh };
            mesh = next_mesh;
        }
    }
    let result = optimize_tour(args, segments, &initial);
    report_tour(
        "OPTIMIZED",
        &result.x,
        segments,
        Some(started.elapsed().as_secs_f64()),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    if matches!(
        args.algorithm,
        Algorithm::TourInspect | Algorithm::TourDeCma | Algorithm::TourBite | Algorithm::TourMesh
    ) {
        return run_tour(&args);
    }
    if args.algorithm == Algorithm::Inspect {
        return report("STORED", &VALIDATED_DECISION, None);
    }
    if args.algorithm == Algorithm::ZohRepair {
        let started = Instant::now();
        let repaired = repair_zoh(&VALIDATED_DECISION, args.stages)?;
        return report("REPAIRED", &repaired, Some(started.elapsed().as_secs_f64()));
    }

    let seed = repair_zoh(&VALIDATED_DECISION, 200)?;
    let stored_objective = objective(&seed);
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
    let mut incumbent = seed;
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
            Algorithm::Inspect
            | Algorithm::ZohRepair
            | Algorithm::TourInspect
            | Algorithm::TourDeCma
            | Algorithm::TourBite
            | Algorithm::TourMesh => {
                unreachable!("non-optimizer algorithms returned before retry")
            }
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
        assert_eq!(
            Algorithm::parse("zoh-repair").unwrap(),
            Algorithm::ZohRepair
        );
        assert_eq!(
            Algorithm::parse("tour-inspect").unwrap(),
            Algorithm::TourInspect
        );
        assert_eq!(
            Algorithm::parse("tour-de-cma").unwrap(),
            Algorithm::TourDeCma
        );
        assert_eq!(Algorithm::parse("tour-bite").unwrap(), Algorithm::TourBite);
        assert_eq!(Algorithm::parse("tour-mesh").unwrap(), Algorithm::TourMesh);
        assert_eq!(Algorithm::parse("de-cma").unwrap(), Algorithm::DeCma);
        assert_eq!(Algorithm::parse("cma").unwrap(), Algorithm::Cma);
        assert_eq!(Algorithm::parse("bite").unwrap(), Algorithm::Bite);
        assert!(Algorithm::parse("unknown").is_err());
    }

    #[test]
    fn stop_target_must_improve_the_stored_incumbent() {
        let incumbent = objective(&VALIDATED_DECISION);
        assert!(validate_stop(-1_843_301.0, incumbent).is_ok());
        assert!(validate_stop(-1_843_300.0, incumbent).is_err());
        assert!(Args::default().stop < incumbent);
    }
}
