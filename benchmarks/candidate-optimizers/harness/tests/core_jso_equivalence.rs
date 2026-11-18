//! Core `fcmaes_core::Jso` is the benchmarked `cand_de::Jso`.
//!
//! Seed mapping: `cand_de::Jso::default().minimize(f, lower, upper, B, s)`
//! corresponds to `Jso::new(Fitness::bounded(D, 1, lower, upper), &p)` with
//! `p = JsoParams { max_evaluations: B, seed: s, runid: 0, ..Default::default() }`
//! (core seeds its stream with `seed + runid`, so any pair summing to `s`
//! works). With that mapping the serial `Jso::optimize` evaluates exactly the
//! same points, bit for bit and in the same order, and reports the same best
//! point, value and evaluation count.
//!
//! The second part checks that the harness `jso_cma` arm, which ran the
//! campaign with `cand_de::Jso`, equals a jSO→CMA sequence assembled from core
//! `Jso` and core `Cmaes` with the configuration documented in the README
//! (section "Sequence arms").

#![cfg(feature = "candidates")]

use std::sync::Mutex;

use candidate_harness::problems;
use candidate_harness::solvers;
use fcmaes_core::{Cmaes, CmaesParams, Fitness, Jso, JsoParams, Rng};
use solver_api::{Guard, Solver};

/// Shifted Rastrigin, optimum off the box center.
fn rastrigin(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let z = v - (1.3 - 0.17 * (i % 11) as f64);
            z * z - 10.0 * (2.0 * std::f64::consts::PI * z).cos() + 10.0
        })
        .sum()
}

/// Asymmetric box, so that the bound repair and the uniform sampling are
/// exercised away from a symmetric `[-a, a]`.
fn rastrigin_box(dim: usize) -> (Vec<f64>, Vec<f64>) {
    let lower = (0..dim).map(|i| -4.0 - 0.1 * i as f64).collect();
    let upper = (0..dim).map(|i| 5.5 + 0.05 * i as f64).collect();
    (lower, upper)
}

type Trace = Vec<Vec<u64>>;

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Runs `minimize` with a recording objective; returns the evaluated points
/// as bit patterns and the reported best point, value and count.
fn record(
    f: &(dyn Fn(&[f64]) -> f64 + Sync),
    run: impl FnOnce(&(dyn Fn(&[f64]) -> f64 + Sync)) -> (Vec<f64>, f64, u64),
) -> (Trace, (Vec<u64>, u64, u64)) {
    let trace = Mutex::new(Vec::new());
    let recording = |x: &[f64]| {
        trace.lock().unwrap().push(bits(x));
        f(x)
    };
    let (x, y, evaluations) = run(&recording);
    (
        trace.into_inner().unwrap(),
        (bits(&x), y.to_bits(), evaluations),
    )
}

fn candidate(
    f: &(dyn Fn(&[f64]) -> f64 + Sync),
    lower: &[f64],
    upper: &[f64],
    budget: u64,
    seed: u64,
) -> (Trace, (Vec<u64>, u64, u64)) {
    record(f, |g| {
        let outcome = cand_de::Jso::default().minimize(g, lower, upper, budget, seed);
        (outcome.x, outcome.y, outcome.evaluations)
    })
}

fn core(
    f: &(dyn Fn(&[f64]) -> f64 + Sync),
    lower: &[f64],
    upper: &[f64],
    budget: u64,
    seed: u64,
) -> (Trace, (Vec<u64>, u64, u64)) {
    record(f, |g| {
        let params = JsoParams {
            max_evaluations: budget,
            seed,
            runid: 0,
            ..Default::default()
        };
        let mut jso = Jso::new(Fitness::bounded(lower.len(), 1, lower, upper), &params);
        let result = jso.optimize(&g);
        (result.x, result.y, result.evaluations)
    })
}

/// Initial population size `round(25 ln(D) sqrt(D))`.
fn n_init(dim: usize) -> u64 {
    let d = dim as f64;
    (25.0 * d.ln() * d.sqrt()).round() as u64
}

fn assert_same(
    label: &str,
    f: &(dyn Fn(&[f64]) -> f64 + Sync),
    lower: &[f64],
    upper: &[f64],
    budget: u64,
    seed: u64,
) {
    let (cand_trace, cand_result) = candidate(f, lower, upper, budget, seed);
    let (core_trace, core_result) = core(f, lower, upper, budget, seed);
    assert_eq!(cand_trace.len() as u64, budget, "{label}: candidate count");
    assert_eq!(
        core_trace.len(),
        cand_trace.len(),
        "{label} budget {budget} seed {seed}: evaluation counts differ"
    );
    if let Some(i) = (0..cand_trace.len()).find(|&i| cand_trace[i] != core_trace[i]) {
        panic!("{label} budget {budget} seed {seed}: evaluation {i} differs");
    }
    assert_eq!(
        core_result, cand_result,
        "{label} budget {budget} seed {seed}: results differ"
    );
}

#[test]
fn core_jso_evaluates_the_benchmarked_sequence_on_rastrigin() {
    for dim in [2, 5, 10, 22, 39] {
        let (lower, upper) = rastrigin_box(dim);
        let n = n_init(dim);
        // Inside, at and just past the initial population, a cut generation,
        // and runs long enough for every stage of the parameter schedule.
        for budget in [1, n - 1, n, n + 3, 2_003, 30_000] {
            for seed in [1, 2, 0x9E37_79B9_7F4A_7C15] {
                assert_same("rastrigin", &rastrigin, &lower, &upper, budget, seed);
            }
        }
    }
}

#[test]
fn core_jso_evaluates_the_benchmarked_sequence_on_campaign_problems() {
    // D = 5 (band-pass filter), 22 (Cassini2, Rosetta), 39 (Lennard-Jones 13).
    for name in ["bandpass", "cassini2", "rosetta", "lj13"] {
        let problem = problems::by_name(name).unwrap();
        let f = |x: &[f64]| (problem.objective)(x);
        let budgets: &[u64] = if name == "bandpass" {
            &[500, 3_000]
        } else {
            &[1_000, 40_000]
        };
        for &budget in budgets {
            for seed in [3, 4] {
                assert_same(name, &f, &problem.lower, &problem.upper, budget, seed);
            }
        }
    }
}

/// jSO→CMA from core parts, as configured in the README ("Sequence arms"):
/// `Rng::new(seed)` draws σ = 0.05 + 0.05 U, then the run seed; jSO runs with
/// budget `min(max(⌊2B/5⌋, 31), B)` and the run seed; core CMA-ES (defaults,
/// normalized box) starts at jSO's best point with σ in every coordinate,
/// seed `run_seed ^ 0xA0761D6478BD642F` and limit `max(B − share, 31)`, and
/// stops before a generation that does not fit into the remaining budget.
/// Every point is projected onto the box before evaluation, as in the
/// harness.
fn core_jso_cma(
    f: &(dyn Fn(&[f64]) -> f64 + Sync),
    lower: &[f64],
    upper: &[f64],
    budget: u64,
    seed: u64,
) -> (Trace, u64) {
    let trace = Mutex::new(Vec::new());
    let best = Mutex::new(f64::INFINITY);
    let eval = |x: &[f64]| {
        let projected: Vec<f64> = x
            .iter()
            .zip(lower.iter().zip(upper))
            .map(|(v, (lo, hi))| v.clamp(*lo, *hi))
            .collect();
        trace.lock().unwrap().push(bits(&projected));
        let y = f(&projected);
        let mut b = best.lock().unwrap();
        if y < *b {
            *b = y;
        }
        y
    };
    let used = || trace.lock().unwrap().len() as u64;

    let mut rng = Rng::new(seed);
    let sdev = 0.05 + 0.05 * rng.uniform01();
    let run_seed = rng.next_u64();
    let share = (budget * 2 / 5).max(31);
    let mut jso = Jso::new(
        Fitness::bounded(lower.len(), 1, lower, upper),
        &JsoParams {
            max_evaluations: share.min(budget),
            seed: run_seed,
            ..Default::default()
        },
    );
    let start = jso.optimize(&eval).x;
    if used() < budget {
        let mut fitness = Fitness::bounded(lower.len(), 1, lower, upper);
        fitness.set_normalize(true);
        let mut cma = Cmaes::new(
            fitness,
            &start,
            &vec![sdev; lower.len()],
            &CmaesParams {
                max_evaluations: budget.saturating_sub(share).max(31),
                seed: run_seed ^ 0xA076_1D64_78BD_642F,
                ..Default::default()
            },
        );
        let popsize = cma.popsize() as u64;
        cma.optimize_until(&eval, 1, &|| budget - used() < popsize);
    }
    let best = best.into_inner().unwrap();
    (trace.into_inner().unwrap(), best.to_bits())
}

#[test]
fn harness_jso_cma_equals_a_core_jso_cma_sequence() {
    let harness = solvers::by_name("jso_cma").unwrap();
    for name in ["cassini1", "messenger", "lj13"] {
        let problem = problems::by_name(name).unwrap();
        let f = |x: &[f64]| (problem.objective)(x);
        let (lower, upper) = (&problem.lower, &problem.upper);
        for budget in [30, 2_000, 25_000] {
            for seed in [5, 6, 7] {
                let trace = Mutex::new(Vec::new());
                let recording = |x: &[f64]| {
                    trace.lock().unwrap().push(bits(x));
                    f(x)
                };
                let guard = Guard::new(&recording, lower, upper, budget);
                let guarded = |x: &[f64]| guard.eval(x);
                harness.minimize(&guarded, lower, upper, budget, seed);
                assert_eq!((guard.over_budget(), guard.out_of_bounds()), (0, 0));
                let harness_best = guard.best_y().to_bits();
                drop(guard);
                let harness_trace = trace.into_inner().unwrap();

                let (core_trace, core_best) = core_jso_cma(&f, lower, upper, budget, seed);
                assert_eq!(
                    core_trace.len(),
                    harness_trace.len(),
                    "{name} budget {budget} seed {seed}: evaluation counts differ"
                );
                assert!(
                    core_trace == harness_trace,
                    "{name} budget {budget} seed {seed}: evaluated points differ"
                );
                assert_eq!(
                    core_best, harness_best,
                    "{name} budget {budget} seed {seed}: best values differ"
                );
            }
        }
    }
}
