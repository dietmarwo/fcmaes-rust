//! Contract, reproducibility, BIPOP validation and resumability tests.

use std::sync::Mutex;

use candidate_harness::bipop::Bipop;
use candidate_harness::campaign::{
    CampaignConfig, Protocol, RAW_HEADER, run_campaign, run_repetition,
};
use candidate_harness::problems::Problem;
use candidate_harness::solvers;
use solver_api::{Guard, Solver};

const SHIFT: [f64; 10] = [1.2, -0.7, 3.1, -2.4, 0.3, 2.2, -4.1, 0.9, -1.6, 4.4];

fn shifted_sphere(x: &[f64]) -> f64 {
    x.iter()
        .zip(SHIFT.iter().cycle())
        .map(|(v, s)| (v - s) * (v - s))
        .sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

struct Trace {
    calls: u64,
    over: u64,
    out: u64,
    best: f64,
    sequence_hash: u64,
}

/// Run `solver` behind a guard and hash the sequence of evaluated points.
fn traced(solver: &dyn Solver, dim: usize, budget: u64, seed: u64) -> Trace {
    let (lower, upper) = (vec![-5.0; dim], vec![5.0; dim]);
    let hash = Mutex::new(0xCBF2_9CE4_8422_2325_u64);
    let recorded = |x: &[f64]| {
        let mut h = hash.lock().unwrap();
        for v in x {
            *h = (*h ^ v.to_bits()).wrapping_mul(0x0000_0100_0000_01B3);
        }
        shifted_sphere(x)
    };
    let guard = Guard::new(&recorded, &lower, &upper, budget);
    let guarded = |x: &[f64]| guard.eval(x);
    let outcome = solver.minimize(&guarded, &lower, &upper, budget, seed);
    assert!(outcome.evaluations <= budget, "{}", solver.name());
    let sequence_hash = *hash.lock().unwrap();
    Trace {
        calls: guard.calls(),
        over: guard.over_budget(),
        out: guard.out_of_bounds(),
        best: guard.best_y(),
        sequence_hash,
    }
}

#[test]
fn every_solver_respects_budgets_and_bounds_and_reproduces() {
    for solver in solvers::registry() {
        for (dim, budget) in [(5, 1), (5, 7), (5, 1000), (10, 20_000), (3, 3_001)] {
            for seed in [0, 17] {
                let first = traced(solver.as_ref(), dim, budget, seed);
                assert_eq!(
                    (first.over, first.out),
                    (0, 0),
                    "{} budget {budget}: violations",
                    solver.name()
                );
                assert!(first.calls <= budget);
                let second = traced(solver.as_ref(), dim, budget, seed);
                assert_eq!(first.calls, second.calls, "{}", solver.name());
                assert_eq!(
                    first.sequence_hash,
                    second.sequence_hash,
                    "{}",
                    solver.name()
                );
                assert_eq!(first.best.to_bits(), second.best.to_bits());
            }
        }
    }
}

#[test]
fn exact_prefix_solvers_spend_the_whole_budget() {
    // BiteOpt and DE stop before every evaluation, so they use the budget
    // exactly; DE->CMA does so whenever the budget ends in its DE phase.
    for name in ["bite", "de"] {
        let solver = solvers::by_name(name).unwrap();
        for budget in [1, 7, 1000, 4321] {
            assert_eq!(
                traced(solver.as_ref(), 5, budget, 3).calls,
                budget,
                "{name}"
            );
        }
    }
    let de_cma = solvers::by_name("de_cma").unwrap();
    assert_eq!(traced(de_cma.as_ref(), 5, 7, 3).calls, 7);
}

#[test]
fn different_seeds_give_different_runs() {
    for solver in solvers::registry() {
        let a = traced(solver.as_ref(), 5, 1000, 1);
        let b = traced(solver.as_ref(), 5, 1000, 2);
        assert_ne!(a.sequence_hash, b.sequence_hash, "{}", solver.name());
    }
}

fn bipop_best(f: fn(&[f64]) -> f64, budget: u64, seed: u64) -> (f64, u64) {
    let (lower, upper) = (vec![-5.0; 10], vec![5.0; 10]);
    let guard = Guard::new(&f, &lower, &upper, budget);
    let guarded = |x: &[f64]| guard.eval(x);
    Bipop.minimize(&guarded, &lower, &upper, budget, seed);
    assert_eq!((guard.over_budget(), guard.out_of_bounds()), (0, 0));
    (guard.best_y(), guard.calls())
}

#[test]
fn bipop_solves_shifted_sphere() {
    for seed in 0..5 {
        let (best, calls) = bipop_best(shifted_sphere, 50_000, seed);
        println!("bipop sphere-10 seed {seed}: {best:e} after {calls} evaluations");
        assert!(best < 1e-8, "seed {seed}: {best} after {calls}");
    }
}

#[test]
fn bipop_solves_rosenbrock_in_three_of_five_seeds() {
    let results: Vec<f64> = (0..5)
        .map(|seed| bipop_best(rosenbrock, 200_000, seed).0)
        .collect();
    println!("bipop rosenbrock-10 best values: {results:?}");
    let solved = results.iter().filter(|value| **value < 1e-6).count();
    assert!(solved >= 3, "{results:?}");
}

fn toy_problem() -> Problem {
    Problem {
        name: "toy",
        domain: "test",
        lower: vec![-5.0; 4],
        upper: vec![5.0; 4],
        objective: Box::new(shifted_sphere),
    }
}

#[test]
fn campaign_resumes_without_duplicates() {
    let directory = std::env::temp_dir().join(format!("candidate-harness-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let out = directory.join("raw.tsv");
    let _ = std::fs::remove_file(&out);
    let problem = toy_problem();
    let registry = solvers::registry();
    let config = |reps| CampaignConfig {
        problems: vec![(&problem, 800)],
        solvers: registry.iter().take(3).map(|s| s.as_ref()).collect(),
        protocols: vec![Protocol::S, Protocol::R],
        repetitions: reps,
        threads: 2,
        budget_scale: 1.0,
    };
    assert_eq!(run_campaign(&config(2), &out).unwrap(), 12);
    assert_eq!(run_campaign(&config(2), &out).unwrap(), 0);
    // Simulate an interrupted write: a partial last line is dropped and
    // the missing row is recomputed.
    let text = std::fs::read_to_string(&out).unwrap();
    let cut = text.trim_end_matches('\n').rfind('\n').unwrap() + 1;
    std::fs::write(&out, format!("{}bite\ttoy\t", &text[..cut])).unwrap();
    assert_eq!(run_campaign(&config(3), &out).unwrap(), 7);
    let text = std::fs::read_to_string(&out).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(RAW_HEADER));
    let mut keys: Vec<(String, String, String)> = lines
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 14);
            assert_eq!((f[11], f[12]), ("0", "0"));
            (f[0].to_owned(), f[4].to_owned(), f[5].to_owned())
        })
        .collect();
    assert_eq!(keys.len(), 18);
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), 18);
    // A different budget scale is refused for the same file.
    let mut scaled = config(3);
    scaled.budget_scale = 0.5;
    assert!(run_campaign(&scaled, &out).is_err());
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn protocol_r_sums_eight_guarded_runs() {
    let problem = toy_problem();
    let solver = solvers::by_name("bite").unwrap();
    let result = run_repetition(solver.as_ref(), &problem, Protocol::R, 0, 803);
    assert_eq!(result.budget, 800);
    assert_eq!(result.evaluations, 800);
    assert_eq!((result.over_budget, result.out_of_bounds), (0, 0));
    let single = run_repetition(solver.as_ref(), &problem, Protocol::S, 0, 803);
    assert_eq!(single.evaluations, 803);
    assert_ne!(single.seed, result.seed);
}

fn shifted_rastrigin(x: &[f64]) -> f64 {
    x.iter()
        .zip(SHIFT.iter().cycle())
        .map(|(v, s)| {
            let z = v - s * 0.5;
            z * z - 10.0 * (2.0 * std::f64::consts::PI * z).cos() + 10.0
        })
        .sum()
}

#[test]
fn every_solver_respects_budgets_on_a_multimodal_function() {
    let (lower, upper) = (vec![-5.0; 10], vec![5.0; 10]);
    for solver in solvers::registry() {
        for budget in [7, 333, 5_000] {
            for seed in 0..4 {
                let guard = Guard::new(&shifted_rastrigin, &lower, &upper, budget);
                let guarded = |x: &[f64]| guard.eval(x);
                solver.minimize(&guarded, &lower, &upper, budget, seed);
                assert_eq!(
                    (guard.over_budget(), guard.out_of_bounds()),
                    (0, 0),
                    "{} budget {budget} seed {seed}",
                    solver.name()
                );
            }
        }
    }
}

#[test]
fn every_solver_respects_budgets_on_every_problem() {
    for problem in candidate_harness::problems::all() {
        let budget = if problem.name == "phased_array" {
            160
        } else {
            400
        };
        for solver in solvers::registry() {
            for protocol in [Protocol::S, Protocol::R] {
                let result = run_repetition(solver.as_ref(), &problem, protocol, 0, budget);
                assert_eq!(
                    (result.over_budget, result.out_of_bounds),
                    (0, 0),
                    "{} on {} ({protocol:?})",
                    solver.name(),
                    problem.name
                );
                assert!(result.evaluations <= result.budget);
                assert!(result.best_value.is_finite() || result.evaluations == 0);
            }
        }
    }
}
