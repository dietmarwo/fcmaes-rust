//! Validation gates of PROTOCOL.md 3.4 (sphere and Rosenbrock) and an
//! informational shifted Rastrigin run. Values are printed; see them with
//! `cargo test -p cand-de --test validation -- --nocapture`.

use cand_de::{Jso, LShade, LSrtde};
use solver_api::{Guard, Solver};

const DIM: usize = 10;
const SEEDS: [u64; 5] = [1, 2, 3, 4, 5];

/// Optimum of the shifted sphere and Rastrigin: off the box center and origin.
fn shift(i: usize) -> f64 {
    1.7 - 0.31 * i as f64
}

fn sphere(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - shift(i)).powi(2))
        .sum()
}

/// Standard Rosenbrock; its optimum (1, ..., 1) is off the box center.
fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn rastrigin(x: &[f64]) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let z = v - shift(i);
            z * z - 10.0 * (two_pi * z).cos() + 10.0
        })
        .sum()
}

/// Best value recorded by the guard, with contract checks.
fn run(solver: &dyn Solver, f: fn(&[f64]) -> f64, budget: u64, seed: u64) -> f64 {
    let (lower, upper) = (vec![-5.0; DIM], vec![5.0; DIM]);
    let guard = Guard::new(&f, &lower, &upper, budget);
    solver.minimize(&|x| guard.eval(x), &lower, &upper, budget, seed);
    assert_eq!(guard.over_budget() + guard.out_of_bounds(), 0);
    guard.best_y()
}

fn sphere_gate(solver: &dyn Solver) {
    let y = run(solver, sphere, 50_000, 1);
    println!(
        "{:>7} sphere-10 (50k evals, seed 1): {y:.3e}",
        solver.name()
    );
    assert!(y < 1e-8, "{}: sphere {y:e} >= 1e-8", solver.name());
}

fn rosenbrock_gate(solver: &dyn Solver) {
    let values: Vec<f64> = SEEDS
        .iter()
        .map(|&s| run(solver, rosenbrock, 200_000, s))
        .collect();
    let passed = values.iter().filter(|&&y| y < 1e-6).count();
    println!(
        "{:>7} rosenbrock-10 (200k evals, seeds 1-5): {} -> {passed}/5 below 1e-6",
        solver.name(),
        fmt(&values)
    );
    assert!(passed >= 3, "{}: only {passed}/5 below 1e-6", solver.name());
}

fn rastrigin_info(solver: &dyn Solver) {
    let values: Vec<f64> = SEEDS
        .iter()
        .map(|&s| run(solver, rastrigin, 200_000, s))
        .collect();
    println!(
        "{:>7} rastrigin-10 (200k evals, seeds 1-5, informational): {}",
        solver.name(),
        fmt(&values)
    );
}

fn fmt(values: &[f64]) -> String {
    values
        .iter()
        .map(|y| format!("{y:.3e}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn lshade_sphere() {
    sphere_gate(&LShade::default());
}

#[test]
fn lshade_rosenbrock() {
    rosenbrock_gate(&LShade::default());
}

#[test]
fn lshade_rastrigin_info() {
    rastrigin_info(&LShade::default());
}

#[test]
fn jso_sphere() {
    sphere_gate(&Jso::default());
}

#[test]
fn jso_rosenbrock() {
    rosenbrock_gate(&Jso::default());
}

#[test]
fn jso_rastrigin_info() {
    rastrigin_info(&Jso::default());
}

#[test]
fn lsrtde_sphere() {
    sphere_gate(&LSrtde::default());
}

#[test]
fn lsrtde_rosenbrock() {
    rosenbrock_gate(&LSrtde::default());
}

#[test]
fn lsrtde_rastrigin_info() {
    rastrigin_info(&LSrtde::default());
}
