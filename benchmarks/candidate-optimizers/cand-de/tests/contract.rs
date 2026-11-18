//! Contract tests (PROTOCOL.md 3.4): budget, bounds, determinism and
//! non-finite objective values, for every solver of the crate.

use std::sync::Mutex;

use cand_de::{Jso, LShade, LSrtde};
use solver_api::{Guard, Solver};

fn solvers() -> Vec<Box<dyn Solver>> {
    vec![
        Box::new(LShade::default()),
        Box::new(Jso::default()),
        Box::new(LSrtde::default()),
    ]
}

/// Shifted sphere whose optimum is neither the box center nor the origin.
fn shifted_sphere(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let d = v - (1.3 - 0.27 * i as f64);
            d * d
        })
        .sum()
}

/// Narrow asymmetric box so that mutation often leaves it.
fn narrow_box(dim: usize) -> (Vec<f64>, Vec<f64>) {
    let lower: Vec<f64> = (0..dim).map(|i| -0.5 + 0.1 * i as f64).collect();
    let upper: Vec<f64> = lower.iter().map(|lo| lo + 0.3).collect();
    (lower, upper)
}

#[test]
fn names_are_stable() {
    let names: Vec<&str> = solvers().iter().map(|s| s.name()).collect();
    assert_eq!(names, ["lshade", "jso", "lsrtde"]);
}

#[test]
fn budget_and_bounds_are_respected() {
    for solver in solvers() {
        for dim in [1usize, 2, 10] {
            let (lower, upper) = narrow_box(dim);
            for budget in [0u64, 1, 7, 1000] {
                let guard = Guard::new(&shifted_sphere, &lower, &upper, budget);
                let outcome = solver.minimize(&|x| guard.eval(x), &lower, &upper, budget, 11);
                let label = format!("{} dim={dim} budget={budget}", solver.name());
                assert_eq!(guard.over_budget(), 0, "{label}: over budget");
                assert_eq!(guard.out_of_bounds(), 0, "{label}: out of bounds");
                assert_eq!(guard.calls(), budget, "{label}: budget not used up");
                assert_eq!(outcome.evaluations, guard.calls(), "{label}");
                assert_eq!(outcome.x.len(), dim, "{label}");
                if budget > 0 {
                    assert_eq!(outcome.y, guard.best_y(), "{label}: reported best");
                    assert_eq!(shifted_sphere(&outcome.x), outcome.y, "{label}");
                }
            }
        }
    }
}

#[test]
fn small_budgets_return_best_initial_point() {
    let dim = 10;
    let (lower, upper) = (vec![-5.0; dim], vec![5.0; dim]);
    for solver in solvers() {
        for budget in [1u64, 7] {
            let seen = Mutex::new(Vec::new());
            let f = |x: &[f64]| {
                let y = shifted_sphere(x);
                seen.lock().unwrap().push(y);
                y
            };
            let outcome = solver.minimize(&f, &lower, &upper, budget, 3);
            let seen = seen.into_inner().unwrap();
            assert_eq!(seen.len() as u64, budget, "{}", solver.name());
            let best = seen.iter().copied().fold(f64::INFINITY, f64::min);
            assert_eq!(outcome.y, best, "{}", solver.name());
        }
    }
}

#[test]
fn same_seed_reproduces_the_run() {
    let dim = 5;
    let (lower, upper) = (vec![-5.0; dim], vec![5.0; dim]);
    for solver in solvers() {
        let run = |seed: u64| {
            let trace = Mutex::new(Vec::new());
            let f = |x: &[f64]| {
                trace.lock().unwrap().extend_from_slice(x);
                shifted_sphere(x)
            };
            let outcome = solver.minimize(&f, &lower, &upper, 3000, seed);
            (outcome.x, outcome.y, trace.into_inner().unwrap())
        };
        let (a, b, c) = (run(42), run(42), run(43));
        assert_eq!(a, b, "{}: same seed differs", solver.name());
        assert_ne!(a.2, c.2, "{}: different seeds agree", solver.name());
    }
}

#[test]
fn non_finite_values_do_not_panic() {
    let dim = 4;
    let (lower, upper) = (vec![-5.0; dim], vec![5.0; dim]);
    let always_nan = |_: &[f64]| f64::NAN;
    // NaN on half the box, +inf and -inf on parts of the rest.
    let mixed = |x: &[f64]| {
        if x[0] > 0.0 {
            f64::NAN
        } else if x[1] > 3.0 {
            f64::INFINITY
        } else if x[1] < -4.0 {
            f64::NEG_INFINITY
        } else {
            shifted_sphere(x)
        }
    };
    for solver in solvers() {
        let guard = Guard::new(&always_nan, &lower, &upper, 2000);
        let outcome = solver.minimize(&|x| guard.eval(x), &lower, &upper, 2000, 5);
        assert_eq!(guard.calls(), 2000, "{}", solver.name());
        assert_eq!(outcome.y, f64::INFINITY, "{}", solver.name());
        assert_eq!(outcome.x.len(), dim);

        let guard = Guard::new(&mixed, &lower, &upper, 5000);
        let outcome = solver.minimize(&|x| guard.eval(x), &lower, &upper, 5000, 5);
        assert_eq!(guard.over_budget() + guard.out_of_bounds(), 0);
        assert!(outcome.y.is_finite(), "{}", solver.name());
        assert!(outcome.x[0] <= 0.0, "{}", solver.name());
    }
}
