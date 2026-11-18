//! Validation gates from PROTOCOL.md section 3.4 plus informational runs
//! (print with `--nocapture`).

use std::time::Instant;

use cand_eda::Amalgam;
use solver_api::{Guard, Solver};

/// Shifted optimum, neither at the box center nor at the origin.
fn shift(i: usize) -> f64 {
    -3.1 + 0.67 * i as f64
}

fn shifted_sphere(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - shift(i)).powi(2))
        .sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn shifted_rastrigin(x: &[f64]) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let z = v - 0.5 * shift(i);
            z * z - 10.0 * (two_pi * z).cos() + 10.0
        })
        .sum()
}

fn fmt(values: &[f64]) -> String {
    let v: Vec<String> = values.iter().map(|y| format!("{y:.3e}")).collect();
    v.join(", ")
}

fn run(f: &(dyn Fn(&[f64]) -> f64 + Sync), d: usize, budget: u64, seed: u64) -> f64 {
    let (lo, hi) = (vec![-5.0; d], vec![5.0; d]);
    let guard = Guard::new(f, &lo, &hi, budget);
    Amalgam::default().minimize(&|x| guard.eval(x), &lo, &hi, budget, seed);
    assert_eq!((guard.over_budget(), guard.out_of_bounds()), (0, 0));
    guard.best_y()
}

#[test]
fn sphere_10d_below_1e_8_within_50k() {
    let y = run(&shifted_sphere, 10, 50_000, 1);
    println!("shifted sphere D=10, 50,000 evaluations, seed 1: {y:.3e}");
    assert!(y < 1e-8, "sphere reached only {y:e}");
}

#[test]
fn rosenbrock_10d_below_1e_6_in_3_of_5() {
    let values: Vec<f64> = (1..=5).map(|s| run(&rosenbrock, 10, 200_000, s)).collect();
    println!(
        "Rosenbrock D=10, 200,000 evaluations, seeds 1-5: {}",
        fmt(&values)
    );
    let hits = values.iter().filter(|&&v| v < 1e-6).count();
    assert!(hits >= 3, "only {hits} of 5 seeds below 1e-6: {values:?}");
}

#[test]
fn rastrigin_10d_informational() {
    let values: Vec<f64> = (1..=5)
        .map(|s| run(&shifted_rastrigin, 10, 200_000, s))
        .collect();
    println!(
        "shifted Rastrigin D=10, 200,000 evaluations, seeds 1-5 (informational): {}",
        fmt(&values)
    );
}

#[test]
fn timing_d114_informational() {
    let d = 114;
    let budget = 200_000;
    let f = |x: &[f64]| x[0];
    let (lo, hi) = (vec![-5.0; d], vec![5.0; d]);
    let start = Instant::now();
    let out = Amalgam::default().minimize(&f, &lo, &hi, budget, 1);
    let secs = start.elapsed().as_secs_f64();
    assert_eq!(out.evaluations, budget);
    println!(
        "timing D=114, population {}, trivial objective: {:.0} evaluations/s ({:.1} us per evaluation, {secs:.2} s total)",
        Amalgam::default().base_population_size(d),
        budget as f64 / secs,
        1e6 * secs / budget as f64
    );
}
