//! Round-2 validation (`ROUND2.md`, section 6): the sequence arms and the
//! protocol-T driver.

use std::sync::Mutex;

use candidate_harness::campaign::{
    CampaignConfig, Protocol, RAW_HEADER_T, TSettings, run_campaign, run_campaign_t,
    run_repetition_t,
};
use candidate_harness::problems::Problem;
use candidate_harness::seeds::run_seed_from;
use candidate_harness::solvers;
use fcmaes_core::Rng;
use solver_api::{Objective, Outcome, Solver};

const SHIFT: [f64; 10] = [1.2, -0.7, 3.1, -2.4, 0.3, 2.2, -4.1, 0.9, -1.6, 4.4];

fn shifted_sphere(x: &[f64]) -> f64 {
    x.iter()
        .zip(SHIFT.iter().cycle())
        .map(|(v, s)| (v - s) * (v - s))
        .sum()
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

// ---------------------------------------------------------------- sequence arms

#[cfg(feature = "candidates")]
mod sequences {
    use super::*;
    use solver_api::Guard;

    /// Evaluated points of one guarded run, with the guard's counters.
    struct Run {
        points: Vec<Vec<f64>>,
        calls: u64,
        over: u64,
        out: u64,
        best: f64,
    }

    fn guarded_run(solver: &dyn Solver, dim: usize, budget: u64, seed: u64) -> Run {
        let (lower, upper) = (vec![-5.0; dim], vec![5.0; dim]);
        let points = Mutex::new(Vec::new());
        let recorded = |x: &[f64]| {
            points.lock().unwrap().push(x.to_vec());
            shifted_sphere(x)
        };
        let guard = Guard::new(&recorded, &lower, &upper, budget);
        let guarded = |x: &[f64]| guard.eval(x);
        solver.minimize(&guarded, &lower, &upper, budget, seed);
        let (calls, over, out, best) = (
            guard.calls(),
            guard.over_budget(),
            guard.out_of_bounds(),
            guard.best_y(),
        );
        drop(guard);
        Run {
            points: points.into_inner().unwrap(),
            calls,
            over,
            out,
            best,
        }
    }

    const ARMS: [&str; 2] = ["jso_cma", "lsrtde_cma"];

    #[test]
    fn sequence_arms_are_registered_for_round_two() {
        let names = solvers::round2_solver_names();
        assert_eq!(names.len(), 12);
        for arm in ARMS {
            assert!(names.contains(&arm));
            assert!(solvers::by_name(arm).is_some(), "{arm}");
        }
        // Round-1 solvers stay registered so round 1 stays reproducible.
        for name in solvers::INCUMBENTS.iter().chain(&solvers::CANDIDATES) {
            assert!(solvers::by_name(name).is_some(), "{name}");
        }
    }

    #[test]
    fn sequence_arms_satisfy_the_contract() {
        for arm in ARMS {
            let solver = solvers::by_name(arm).unwrap();
            for (dim, budget) in [(5, 1), (5, 7), (5, 1000), (10, 50_000)] {
                for seed in [0, 17, 4242] {
                    let first = guarded_run(solver.as_ref(), dim, budget, seed);
                    assert_eq!((first.over, first.out), (0, 0), "{arm} budget {budget}");
                    assert!(first.calls <= budget);
                    let (lower, upper) = (-5.0, 5.0);
                    assert!(
                        first
                            .points
                            .iter()
                            .flatten()
                            .all(|v| (lower..=upper).contains(v))
                    );
                    let second = guarded_run(solver.as_ref(), dim, budget, seed);
                    assert_eq!(first.points, second.points, "{arm} budget {budget}");
                    assert_eq!(first.best.to_bits(), second.best.to_bits());
                }
            }
            let a = guarded_run(solver.as_ref(), 5, 1000, 1);
            let b = guarded_run(solver.as_ref(), 5, 1000, 2);
            assert_ne!(a.points, b.points, "{arm}");
        }
    }

    /// The first stage of each arm is exactly the candidate run with the
    /// first-stage budget `max(2B/5, 31)` and the run seed drawn after the
    /// step size; CMA-ES then spends the rest.
    #[test]
    fn first_stage_is_the_candidate_with_the_de_share() {
        let jso = cand_de::Jso::default();
        let lsrtde_b = cand_de::LSrtde {
            pbest_difference_from_base: true,
            ..Default::default()
        };
        let pairs: [(&str, &dyn Solver); 2] = [("jso_cma", &jso), ("lsrtde_cma", &lsrtde_b)];
        for (arm, first) in pairs {
            let solver = solvers::by_name(arm).unwrap();
            for (dim, budget, seed) in [(5, 2_000, 3), (10, 50_000, 8), (5, 20, 1)] {
                let sequence = guarded_run(solver.as_ref(), dim, budget, seed);
                let mut rng = Rng::new(seed);
                let _sdev = rng.uniform01();
                let run_seed = rng.next_u64();
                let stop = (budget * 2 / 5).max(31).min(budget);
                let stage = guarded_run(first, dim, stop, run_seed);
                assert_eq!(stage.calls, stop, "{arm}: candidates spend their budget");
                assert_eq!(
                    &sequence.points[..stage.points.len()],
                    &stage.points[..],
                    "{arm} B={budget}"
                );
                if budget > 31 {
                    // CMA-ES continues with the rest of the budget (it may
                    // stop earlier on its own termination criteria, as in
                    // `de_cma`).
                    assert!(sequence.calls > stop, "{arm}");
                    assert!(sequence.calls <= budget, "{arm}");
                } else {
                    assert_eq!(sequence.calls, budget);
                }
            }
        }
    }

    #[test]
    fn sequence_arms_run_on_every_round_two_problem_under_t() {
        let settings = TSettings {
            root_seed: 2,
            total: 1_200,
            run_cap: 500,
            min_run: 40,
        };
        for name in candidate_harness::problems::round2_names() {
            let problem = candidate_harness::problems::by_name(name).unwrap();
            for arm in ARMS {
                let solver = solvers::by_name(arm).unwrap();
                let t = run_repetition_t(solver.as_ref(), &problem, 0, &settings);
                assert_eq!(
                    (t.result.over_budget, t.result.out_of_bounds),
                    (0, 0),
                    "{arm} on {name}"
                );
                assert!(t.result.evaluations <= settings.total);
                assert!(t.result.best_value.is_finite());
            }
        }
    }
}

// ---------------------------------------------------------------- protocol T

/// Toy solver: evaluates `min(per_run, budget)` uniform points (or
/// `budget + overshoot` when `overshoot > 0`), records each seed it is given,
/// and claims a wrong evaluation count.
struct Toy {
    per_run: Option<u64>,
    overshoot: u64,
    empty_after: Option<usize>,
    seeds: Mutex<Vec<(u64, u64)>>,
}

impl Toy {
    fn new(per_run: Option<u64>) -> Self {
        Self {
            per_run,
            overshoot: 0,
            empty_after: None,
            seeds: Mutex::new(Vec::new()),
        }
    }
}

impl Solver for Toy {
    fn name(&self) -> &'static str {
        "toy"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let mut seeds = self.seeds.lock().unwrap();
        let index = seeds.len();
        seeds.push((seed, max_evaluations));
        drop(seeds);
        let count = if self.empty_after.is_some_and(|n| index >= n) {
            0
        } else {
            self.per_run
                .map_or(max_evaluations + self.overshoot, |n| n.min(max_evaluations))
        };
        let mut rng = Rng::new(seed);
        let mut best = (f64::INFINITY, Vec::new());
        for _ in 0..count {
            let x: Vec<f64> = lower
                .iter()
                .zip(upper)
                .map(|(lo, hi)| lo + rng.uniform01() * (hi - lo))
                .collect();
            let y = objective(&x);
            if y < best.0 {
                best = (y, x);
            }
        }
        // A false claim: the driver must count the guard's evaluations.
        Outcome {
            x: best.1,
            y: best.0,
            evaluations: 7,
        }
    }
}

fn defaults() -> TSettings {
    TSettings::with_root_seed(2)
}

#[test]
fn a_solver_that_spends_everything_makes_four_runs() {
    let problem = toy_problem();
    let toy = Toy::new(None);
    let t = run_repetition_t(&toy, &problem, 3, &defaults());
    assert_eq!(t.runs, 4);
    assert!(!t.stopped_empty);
    assert_eq!(t.result.evaluations, 400_000);
    assert_eq!(t.result.budget, 400_000);
    assert_eq!((t.result.over_budget, t.result.out_of_bounds), (0, 0));
    let seeds = toy.seeds.into_inner().unwrap();
    assert_eq!(seeds.len(), 4);
    for (k, (seed, budget)) in seeds.iter().enumerate() {
        assert_eq!(*budget, 100_000);
        assert_eq!(*seed, run_seed_from(2, "toy", 'T', 3, k as u32));
    }
    assert_eq!(t.result.seed, seeds[0].0);
}

#[test]
fn an_early_stopping_solver_gets_more_runs_with_distinct_seeds() {
    let problem = toy_problem();
    // 30,000 evaluations per run: runs get min(100,000, remaining), so the
    // last three get 70,000, 40,000 and 10,000; then 0 < 1,000 remain.
    let toy = Toy::new(Some(30_000));
    let t = run_repetition_t(&toy, &problem, 0, &defaults());
    assert_eq!(t.runs, 14);
    assert_eq!(t.result.evaluations, 400_000);
    let seeds = toy.seeds.into_inner().unwrap();
    let budgets: Vec<u64> = seeds.iter().map(|(_, b)| *b).collect();
    assert_eq!(budgets[..11], [100_000; 11]);
    assert_eq!(budgets[11..], [70_000, 40_000, 10_000]);
    let mut distinct: Vec<u64> = seeds.iter().map(|(s, _)| *s).collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 14);
    // Restarting ends below the minimum run: 30,000 per run with a total of
    // 90,500 leaves 500 < 1,000 after three runs.
    let toy = Toy::new(Some(30_000));
    let settings = TSettings {
        total: 90_500,
        ..defaults()
    };
    let t = run_repetition_t(&toy, &problem, 0, &settings);
    let budgets: Vec<u64> = toy
        .seeds
        .into_inner()
        .unwrap()
        .iter()
        .map(|(_, b)| *b)
        .collect();
    assert_eq!(budgets, vec![90_500, 60_500, 30_500]);
    assert_eq!((t.runs, t.result.evaluations), (3, 90_000));
    assert!(!t.stopped_empty);
}

#[test]
fn the_total_is_never_exceeded_and_overshoot_is_refused() {
    let problem = toy_problem();
    let mut toy = Toy::new(None);
    toy.overshoot = 5;
    let settings = TSettings {
        root_seed: 2,
        total: 10_000,
        run_cap: 3_000,
        min_run: 100,
    };
    let t = run_repetition_t(&toy, &problem, 0, &settings);
    // Runs of 3,000, 3,000, 3,000 and 1,000; each tries five extra calls.
    assert_eq!(t.runs, 4);
    assert_eq!(t.result.evaluations, 10_000);
    assert_eq!(t.result.over_budget, 20);
    for solver in solvers::registry() {
        for (total, run_cap, min_run) in [(5_000, 1_300, 50), (777, 100, 1), (90, 200, 10)] {
            let settings = TSettings {
                root_seed: 2,
                total,
                run_cap,
                min_run,
            };
            let t = run_repetition_t(solver.as_ref(), &problem, 1, &settings);
            assert!(t.result.evaluations <= total, "{}", solver.name());
            assert!(total - t.result.evaluations < min_run || t.stopped_empty);
            assert_eq!(
                (t.result.over_budget, t.result.out_of_bounds),
                (0, 0),
                "{} total {total}",
                solver.name()
            );
        }
    }
}

#[test]
fn a_run_without_evaluations_stops_restarting() {
    let problem = toy_problem();
    let toy = Toy::new(Some(0));
    let t = run_repetition_t(&toy, &problem, 0, &defaults());
    assert!(t.stopped_empty);
    assert_eq!((t.runs, t.result.evaluations), (1, 0));
    assert_eq!(t.result.best_value, f64::INFINITY);
    let mut toy = Toy::new(Some(1_000));
    toy.empty_after = Some(3);
    let t = run_repetition_t(&toy, &problem, 0, &defaults());
    assert!(t.stopped_empty);
    assert_eq!((t.runs, t.result.evaluations), (4, 3_000));
}

fn t_config<'a>(
    problem: &'a Problem,
    registry: &'a [Box<dyn Solver>],
    reps: u32,
) -> CampaignConfig<'a> {
    CampaignConfig {
        problems: vec![(problem, 0)],
        solvers: registry
            .iter()
            .filter(|s| ["bite", "cma", "de_cma"].contains(&s.name()))
            .map(|s| s.as_ref())
            .collect(),
        protocols: vec![Protocol::T],
        repetitions: reps,
        threads: 2,
        budget_scale: 1.0,
    }
}

fn small_settings() -> TSettings {
    TSettings {
        root_seed: 2,
        total: 3_000,
        run_cap: 800,
        min_run: 40,
    }
}

/// Rows without the wall-time column, sorted by key.
fn rows_without_time(path: &std::path::Path) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(RAW_HEADER_T));
    let mut rows: Vec<Vec<String>> = lines
        .map(|line| {
            let mut fields: Vec<String> = line.split('\t').map(str::to_owned).collect();
            assert_eq!(fields.len(), 19);
            fields.remove(13);
            fields
        })
        .collect();
    rows.sort();
    rows
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let directory =
        std::env::temp_dir().join(format!("candidate-harness-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

#[test]
fn protocol_t_rows_are_reproduced_by_a_rerun() {
    let directory = temp_dir("t-rerun");
    let problem = toy_problem();
    let registry = solvers::registry();
    let config = t_config(&problem, &registry, 2);
    let (a, b) = (directory.join("a.tsv"), directory.join("b.tsv"));
    assert_eq!(run_campaign_t(&config, &small_settings(), &a).unwrap(), 6);
    assert_eq!(run_campaign_t(&config, &small_settings(), &b).unwrap(), 6);
    let rows = rows_without_time(&a);
    assert_eq!(rows, rows_without_time(&b));
    for row in &rows {
        // protocol, budget, scale, violations, root seed, total, cap, min run
        assert_eq!(row[4], "T");
        assert_eq!((row[7].as_str(), row[8].as_str()), ("3000", "1e0"));
        assert_eq!((row[11].as_str(), row[12].as_str()), ("0", "0"));
        assert_eq!(row[14..], ["2", "3000", "800", "40"]);
        let runs: u32 = row[13].parse().unwrap();
        assert!(runs >= 4, "{row:?}");
    }
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn protocol_t_resumes_without_duplicates() {
    let directory = temp_dir("t-resume");
    let out = directory.join("raw.tsv");
    let problem = toy_problem();
    let registry = solvers::registry();
    assert_eq!(
        run_campaign_t(&t_config(&problem, &registry, 2), &small_settings(), &out).unwrap(),
        6
    );
    assert_eq!(
        run_campaign_t(&t_config(&problem, &registry, 2), &small_settings(), &out).unwrap(),
        0
    );
    // An interrupted write: the partial last line is dropped and recomputed.
    let text = std::fs::read_to_string(&out).unwrap();
    let cut = text.trim_end_matches('\n').rfind('\n').unwrap() + 1;
    std::fs::write(&out, format!("{}bite\ttoy\t", &text[..cut])).unwrap();
    assert_eq!(
        run_campaign_t(&t_config(&problem, &registry, 3), &small_settings(), &out).unwrap(),
        4
    );
    let rows = rows_without_time(&out);
    assert_eq!(rows.len(), 9);
    let mut keys: Vec<(String, String)> =
        rows.iter().map(|r| (r[0].clone(), r[5].clone())).collect();
    keys.dedup();
    assert_eq!(keys.len(), 9);
    // Other settings, or a round-1 run, are refused for the same file.
    let other = TSettings {
        root_seed: 3,
        ..small_settings()
    };
    assert!(run_campaign_t(&t_config(&problem, &registry, 3), &other, &out).is_err());
    let mut round_one = t_config(&problem, &registry, 3);
    round_one.protocols = vec![Protocol::S];
    assert!(run_campaign(&round_one, &out).is_err());
    // A round-1 runner cannot be asked for protocol T.
    assert!(run_campaign(&t_config(&problem, &registry, 3), &directory.join("x.tsv")).is_err());
    std::fs::remove_dir_all(&directory).unwrap();
}
