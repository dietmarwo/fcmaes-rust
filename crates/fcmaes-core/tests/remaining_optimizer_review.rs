//! Properties of the capped and observed entry points checked over ranges
//! of budgets and populations, and against the uncapped loops.

use std::cell::RefCell;

use fcmaes_core::{
    Archive, Crfmnes, CrfmnesParams, DiversifierParams, Fitness, MoRetryConfig, Pgpe, PgpeParams,
    RetryBounds, RetryRunResult, Rng, diversify_batch, diversify_batch_capped, moretry_observed,
};

fn loss(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - 0.3 + 0.1 * i as f64).powi(2))
        .sum()
}

fn holes(x: &[f64]) -> f64 {
    if x[0] > 0.2 { f64::NAN } else { loss(x) }
}

const DIM: usize = 3;

fn nes(budget: u64, popsize: i32, seed: u64) -> Crfmnes {
    let fit = Fitness::bounded(DIM, 1, &[-1.0; DIM], &[1.0; DIM]);
    let p = CrfmnesParams {
        max_evaluations: budget,
        popsize,
        seed,
        ..Default::default()
    };
    Crfmnes::new(fit, &[0.9; DIM], 0.5, &p)
}

fn pgpe(budget: u64, popsize: i32, seed: u64) -> Pgpe {
    let fit = Fitness::bounded(DIM, 1, &[-1.0; DIM], &[1.0; DIM]);
    let p = PgpeParams {
        max_evaluations: budget,
        popsize,
        seed,
        ..Default::default()
    };
    Pgpe::new(fit, &[0.9; DIM], &[0.5], &p)
}

/// Every budget from one to past four populations: the callback sees exactly
/// the budget, every point lies in the box, and the result is the best
/// finite value among them at the point that produced it.
#[test]
fn capped_loops_spend_exactly_the_budget_for_every_budget_and_population() {
    for f in [loss as fn(&[f64]) -> f64, holes] {
        for popsize in [2, 7, 8, 31, 32] {
            for budget in 1..=(4 * popsize as u64 + 3) {
                for which in 0..2 {
                    let seen = RefCell::new(Vec::<(Vec<f64>, f64)>::new());
                    let eval = |xs: &[Vec<f64>]| {
                        xs.iter()
                            .map(|x| {
                                let y = f(x);
                                seen.borrow_mut().push((x.clone(), y));
                                y
                            })
                            .collect::<Vec<_>>()
                    };
                    let (x, y, n) = if which == 0 {
                        let r = nes(budget, popsize, 3)
                            .optimize_batch_capped_until(eval, &|| false)
                            .unwrap();
                        (r.x, r.y, r.evaluations)
                    } else {
                        let r = pgpe(budget, popsize, 3)
                            .optimize_batch_capped_until(eval, &|| false)
                            .unwrap();
                        (r.x, r.y, r.evaluations)
                    };
                    let seen = seen.borrow();
                    let tag = format!("solver {which} popsize {popsize} budget {budget}");
                    assert_eq!(seen.len() as u64, budget, "{tag}");
                    assert_eq!(n, budget, "{tag}");
                    assert!(
                        seen.iter()
                            .all(|(x, _)| x.iter().all(|v| (-1.0..=1.0).contains(v))),
                        "{tag}"
                    );
                    let best = seen
                        .iter()
                        .filter(|(_, y)| y.is_finite())
                        .map(|(_, y)| *y)
                        .fold(f64::INFINITY, f64::min);
                    assert_eq!(y, best, "{tag}");
                    if best.is_finite() {
                        assert!(seen.iter().any(|(p, q)| *q == y && *p == x), "{tag}");
                    }
                }
            }
        }
    }
}

/// On a budget of whole populations the capped loop asks for the points of
/// the ordinary loop, in the same order.
#[test]
fn capped_loops_ask_the_same_points_as_the_uncapped_loops_on_whole_populations() {
    for popsize in [8, 32] {
        for generations in [1u64, 5, 20] {
            let budget = generations * popsize as u64;
            let trace = |capped: bool, which: usize| {
                let seen = RefCell::new(Vec::<Vec<f64>>::new());
                let eval = |xs: &[Vec<f64>]| {
                    seen.borrow_mut().extend_from_slice(xs);
                    xs.iter().map(|x| loss(x)).collect::<Vec<_>>()
                };
                match (which, capped) {
                    (0, true) => {
                        nes(budget, popsize, 5)
                            .optimize_batch_capped_until(eval, &|| false)
                            .unwrap();
                    }
                    (0, false) => {
                        nes(budget, popsize, 5).optimize_batch(eval);
                    }
                    (_, true) => {
                        pgpe(budget, popsize, 5)
                            .optimize_batch_capped_until(eval, &|| false)
                            .unwrap();
                    }
                    (_, false) => {
                        pgpe(budget, popsize, 5).optimize_batch(eval);
                    }
                }
                seen.into_inner()
            };
            for which in 0..2 {
                assert_eq!(
                    trace(true, which),
                    trace(false, which),
                    "{which} {popsize} {budget}"
                );
            }
        }
    }
}

fn qd(x: &[f64]) -> (f64, Vec<f64>) {
    (loss(x), vec![x[0], x[1]])
}

fn archive(seed: u64) -> (Archive, Rng) {
    let mut rng = Rng::new(seed);
    let mut a = Archive::new(DIM, &[-1.0; 2], &[1.0; 2], 64, 0, &mut rng);
    a.seed_uniform(&[-1.0; DIM], &[1.0; DIM], &mut rng);
    (a, rng)
}

/// The capped Diversifier makes exactly the budget of calls for every budget
/// and, on whole populations, the calls of the ordinary one.
#[test]
fn capped_diversifier_spends_the_budget_and_matches_the_uncapped_one_on_whole_populations() {
    for popsize in [8, 32] {
        for budget in 1..=(5 * popsize as u64 + 3) {
            let p = DiversifierParams {
                max_evaluations: budget,
                popsize,
                stall_criterion: 4,
            };
            let run = |capped: bool| {
                let (mut a, mut rng) = archive(11);
                let mut seen = Vec::<Vec<f64>>::new();
                let mut fit = |xs: &[Vec<f64>]| {
                    seen.extend_from_slice(xs);
                    xs.iter().map(|x| qd(x)).collect::<Vec<_>>()
                };
                let call = if capped {
                    diversify_batch_capped
                } else {
                    diversify_batch
                };
                let (x, y) =
                    call(&mut a, &mut fit, &[-1.0; DIM], &[1.0; DIM], &p, &mut rng).unwrap();
                (seen, x, y, a.counts().iter().sum::<u64>())
            };
            let (seen, x, y, counted) = run(true);
            assert_eq!(
                seen.len() as u64,
                budget,
                "popsize {popsize} budget {budget}"
            );
            assert_eq!(counted, budget);
            assert_eq!(
                y,
                seen.iter().map(|x| loss(x)).fold(f64::INFINITY, f64::min)
            );
            assert_eq!(y, loss(&x));
            let (plain, ..) = run(false);
            assert!(plain.len() as u64 >= budget);
            assert_eq!(
                seen[..],
                plain[..seen.len()],
                "popsize {popsize} budget {budget}"
            );
        }
    }
}

/// An observed vector goes through the checks of a reevaluated one.
#[test]
fn observed_vectors_are_validated_like_reevaluated_ones() {
    let bounds = RetryBounds::new(vec![0.0], vec![1.0]).unwrap();
    let mut p = MoRetryConfig::new(vec![0.1; 2], vec![1.0; 2]);
    p.retry.num_retries = 6;
    p.retry.workers = 1;
    let objective = |x: &[f64]| vec![x[0], 1.0 - x[0]];
    let (result, extra) = moretry_observed(&objective, &bounds, &p, |_, c| {
        let x = vec![c.run_id as f64 / 6.0];
        let observed = match c.run_id {
            0 => Some(vec![x[0]]),                  // too short
            1 => Some(vec![x[0], f64::NAN]),        // not finite
            2 => Some(vec![x[0], 1.0 - x[0], 0.0]), // too long
            3 => None,
            _ => Some(vec![x[0], 1.0 - x[0]]),
        };
        (
            RetryRunResult {
                x,
                y: 0.5,
                evaluations: 1,
            },
            observed,
        )
    })
    .unwrap();
    assert_eq!(extra, 1);
    for entry in &result.entries {
        assert_eq!(entry.y, vec![entry.x[0], 1.0 - entry.x[0]]);
    }
    let kept: Vec<usize> = result
        .entries
        .iter()
        .map(|e| (e.x[0] * 6.0).round() as usize)
        .collect();
    assert!(kept.iter().all(|id| *id >= 3), "{kept:?}");
    assert_eq!(kept.len(), 3);
}

/// One retry worker runs on the calling thread, so Rayon work started inside
/// the closure uses the caller's pool. Two workers use a private pool.
#[test]
fn a_single_retry_worker_runs_on_the_calling_thread() {
    use fcmaes_core::{RetryConfig, retry};
    let bounds = RetryBounds::new(vec![0.0], vec![1.0]).unwrap();
    let caller = std::thread::current().id();
    for (workers, inline) in [(1usize, true), (2, false)] {
        let config = RetryConfig {
            num_retries: 4,
            workers,
            ..Default::default()
        };
        let result = retry(&|x: &[f64]| x[0], &bounds, &config, |f, _| {
            assert_eq!(std::thread::current().id() == caller, inline);
            RetryRunResult {
                x: vec![0.5],
                y: f(&[0.5]),
                evaluations: 1,
            }
        });
        assert_eq!((result.runs, result.evaluations), (4, 4));
    }
}
