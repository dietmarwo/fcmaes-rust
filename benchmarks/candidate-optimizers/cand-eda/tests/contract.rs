//! Contract tests from PROTOCOL.md section 3.4: budget, bounds, determinism
//! and robustness against non-finite objective values.

use std::sync::Mutex;

use cand_eda::Amalgam;
use solver_api::{Guard, Solver};

fn sphere(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - 0.3 * i as f64).powi(2))
        .sum()
}

/// Runs `solver` behind a guard and checks budget and bounds.
fn check_contract(
    solver: &Amalgam,
    f: &(dyn Fn(&[f64]) -> f64 + Sync),
    lo: &[f64],
    hi: &[f64],
    budget: u64,
) {
    let guard = Guard::new(f, lo, hi, budget);
    let out = solver.minimize(&|x| guard.eval(x), lo, hi, budget, 11);
    assert_eq!(guard.over_budget(), 0, "budget {budget} exceeded");
    assert_eq!(guard.out_of_bounds(), 0, "point outside the box");
    assert!(guard.calls() <= budget);
    assert_eq!(out.evaluations, guard.calls());
    assert_eq!(out.x.len(), lo.len());
    assert!(
        out.x
            .iter()
            .zip(lo.iter().zip(hi))
            .all(|(v, (l, h))| v >= l && v <= h)
    );
    if guard.best_y().is_finite() {
        assert_eq!(out.y, guard.best_y());
    }
}

#[test]
fn name_is_amalgam() {
    assert_eq!(Amalgam::default().name(), "amalgam");
}

#[test]
fn budgets_and_bounds_are_respected() {
    let solver = Amalgam::default();
    for d in [5usize, 10] {
        let lo = vec![-5.0; d];
        let hi = vec![5.0; d];
        for budget in [0u64, 1, 7, 1000] {
            check_contract(&solver, &sphere, &lo, &hi, budget);
        }
    }
}

#[test]
fn small_budget_uses_it_fully() {
    let solver = Amalgam::default();
    let (lo, hi) = (vec![-5.0; 10], vec![5.0; 10]);
    for budget in [1u64, 7, 1000] {
        let guard = Guard::new(&sphere, &lo, &hi, budget);
        let out = solver.minimize(&|x| guard.eval(x), &lo, &hi, budget, 3);
        assert_eq!(guard.calls(), budget);
        assert_eq!(out.y, guard.best_y());
    }
}

#[test]
fn optimum_outside_and_on_the_box_edge() {
    // Optimum at 7 lies outside [-5, 5]; the best point is on the upper edge,
    // so many Gaussian samples fall outside and must be redrawn or clamped.
    let f = |x: &[f64]| x.iter().map(|v| (v - 7.0).powi(2)).sum::<f64>();
    let (lo, hi) = (vec![-5.0; 5], vec![5.0; 5]);
    check_contract(&Amalgam::default(), &f, &lo, &hi, 20_000);
    // Asymmetric, narrow box.
    let (lo, hi) = (vec![0.999; 5], vec![1.001; 5]);
    check_contract(&Amalgam::default(), &sphere, &lo, &hi, 5_000);
}

#[test]
fn degenerate_box_does_not_panic() {
    // One fixed variable (lower == upper) makes the covariance singular.
    let lo = vec![-5.0, 2.0, -5.0, -5.0, -5.0];
    let hi = vec![5.0, 2.0, 5.0, 5.0, 5.0];
    check_contract(&Amalgam::default(), &sphere, &lo, &hi, 5_000);
}

#[test]
fn same_seed_reproduces_the_run() {
    let solver = Amalgam::default();
    let (lo, hi) = (vec![-5.0; 6], vec![5.0; 6]);
    let record = |seed: u64| {
        let trace = Mutex::new(Vec::new());
        let f = |x: &[f64]| {
            trace.lock().unwrap().extend_from_slice(x);
            sphere(x)
        };
        let out = solver.minimize(&f, &lo, &hi, 3_000, seed);
        (trace.into_inner().unwrap(), out.y, out.x)
    };
    let a = record(5);
    let b = record(5);
    assert_eq!(a.0.len(), 3_000 * 6);
    assert_eq!(a, b);
    let c = record(6);
    assert_ne!(a.0, c.0);
}

#[test]
fn non_finite_values_do_not_panic() {
    let solver = Amalgam::default();
    let (lo, hi) = (vec![-5.0; 5], vec![5.0; 5]);
    // Always NaN.
    let nan = |_: &[f64]| f64::NAN;
    check_contract(&solver, &nan, &lo, &hi, 3_000);
    let out = solver.minimize(&nan, &lo, &hi, 500, 1);
    assert_eq!(out.y, f64::INFINITY);
    // NaN or infinity on half the box: the finite half must win.
    let half = |x: &[f64]| {
        if x[0] < 0.0 {
            f64::NAN
        } else if x[1] < 0.0 {
            f64::NEG_INFINITY
        } else {
            sphere(x)
        }
    };
    let guard = Guard::new(&half, &lo, &hi, 5_000);
    let out = solver.minimize(&|x| guard.eval(x), &lo, &hi, 5_000, 2);
    assert_eq!((guard.over_budget(), guard.out_of_bounds()), (0, 0));
    assert!(out.y.is_finite());
    assert!(out.x[0] >= 0.0 && out.x[1] >= 0.0);
    // Constant objective: the selected set is flat, runs restart until the
    // budget is used.
    let flat = |_: &[f64]| 1.0;
    check_contract(&solver, &flat, &lo, &hi, 4_000);
}

#[test]
fn incremental_variant_respects_the_contract() {
    let solver = Amalgam {
        incremental: true,
        ..Amalgam::default()
    };
    let (lo, hi) = (vec![-5.0; 10], vec![5.0; 10]);
    for budget in [1u64, 7, 1000, 20_000] {
        check_contract(&solver, &sphere, &lo, &hi, budget);
    }
}

#[test]
fn published_defaults() {
    let a = Amalgam::default();
    assert_eq!(a.tau, 0.35);
    assert_eq!(a.eta_dec, 0.9);
    assert_eq!(a.theta_sdr, 1.0);
    assert_eq!(a.delta_ams, 2.0);
    assert_eq!(a.min_multiplier, 1e-10);
    assert!(!a.incremental);
    assert_eq!(a.base_population_size(10), 111);
    assert_eq!(a.base_population_size(114), 3668);
    assert_eq!(a.learning_rates(39, 10), (1.0, 1.0));
}
