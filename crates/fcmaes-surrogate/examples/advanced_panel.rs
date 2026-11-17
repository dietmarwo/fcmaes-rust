//! One bounded benchmark cell; orchestration supplies process watchdogs.
use fcmaes_core::{BiteOpt, BiteParams};
use fcmaes_surrogate::{Config, Observation, RunSpec, Space, Value, advanced};
use serde_json::json;
use std::{
    env,
    time::{Duration, Instant},
};

fn objective(name: &str, x: &[f64]) -> f64 {
    match name {
        "sphere" => x.iter().map(|v| (v - 0.37).powi(2)).sum(),
        "ackley" => {
            let n = x.len() as f64;
            -20.0 * (-0.2 * (x.iter().map(|v| v * v).sum::<f64>() / n).sqrt()).exp()
                - (x.iter()
                    .map(|v| (std::f64::consts::TAU * v).cos())
                    .sum::<f64>()
                    / n)
                    .exp()
                + 20.0
                + std::f64::consts::E
        }
        "styblinski" => x
            .iter()
            .map(|v| (v.powi(4) - 16.0 * v * v + 5.0 * v) * 0.5 + 39.16616570377142)
            .sum(),
        "oscillator" => {
            let mut y = 1.0;
            let mut velocity = 0.0;
            let dt = 0.01;
            for _ in 0..200 {
                let accel = -x[0] * y - x[1] * velocity;
                velocity += dt * accel;
                y += dt * velocity;
            }
            (y - 0.1).powi(2) + 0.01 * velocity.powi(2)
        }
        _ => panic!("unknown problem"),
    }
}
fn main() {
    let a: Vec<_> = env::args().collect();
    let variant = &a[1];
    let problem = &a[2];
    let seed: u64 = a[3].parse().unwrap();
    let budget: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(48);
    let d = if problem == "sphere" {
        3
    } else if problem == "oscillator" {
        2
    } else {
        6
    };
    let (lo, hi) = if problem == "oscillator" {
        (0.1, 4.0)
    } else {
        (-5.0, 5.0)
    };
    let space = Space::new(vec![lo; d], vec![hi; d], vec![false; d]).unwrap();
    let base = Config {
        initial_points: 8,
        candidates: 128,
        fit_restarts: 1,
        fit_iterations: 12,
        ..Config::default()
    };
    let began = Instant::now();
    let evaluate = |x: &[f64]| {
        if variant == "async" || variant == "sync-cost" {
            std::thread::sleep(Duration::from_millis(if x[0] > 0.0 { 30 } else { 5 }));
        }
        Ok(objective(problem, x))
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let result = if variant == "biteopt" {
        // Population-only baseline: BiteOpt alone in unit coordinates, as
        // after a handoff, for exactly `budget` calls.
        let params = BiteParams {
            max_evaluations: (budget as u64).saturating_mul(1000),
            seed,
            ..BiteParams::default()
        };
        let mut bite = BiteOpt::new(&vec![0.0; d], &vec![1.0; d], None, &params);
        let mut history = Vec::new();
        for id in 0..budget as u64 {
            let unit = bite.ask(1).pop().unwrap();
            let x = space.from_unit(&unit).unwrap();
            let y = evaluate(&x).unwrap();
            bite.tell(&[y]);
            history.push(Observation {
                id,
                x,
                value: Value::Finite(y),
            });
        }
        json!({"history":history,"proposal_seconds":0.0,"evaluations":budget})
    } else if variant == "legacy" {
        let r = pool
            .install(|| {
                fcmaes_surrogate::run(
                    RunSpec {
                        space: &space,
                        config: &base,
                        evaluations: budget,
                        concurrent: 1,
                        seed,
                        warm: &[],
                        target: None,
                    },
                    &evaluate,
                    &mut |_| Ok(()),
                    &|| None,
                )
            })
            .unwrap();
        json!({"history":r.history,"proposal_seconds":r.proposal_seconds,"evaluations":r.evaluations})
    } else {
        let options = advanced::Options {
            base,
            training_cap: 64,
            concurrent: if variant == "async" || variant == "sync-cost" {
                3
            } else {
                1
            },
            schedule: if variant == "async" {
                advanced::Schedule::Asynchronous
            } else {
                advanced::Schedule::Synchronous
            },
            acquisition: if variant == "thompson" {
                advanced::Acquisition::Thompson
            } else {
                advanced::Acquisition::LogEi
            },
            handoff_after: if variant == "handoff" {
                Some(budget / 2)
            } else {
                None
            },
            // handoff-25, handoff-50 and handoff-75: a share of the budget,
            // subject to the default minimum of remaining BiteOpt calls.
            handoff_fraction: variant
                .strip_prefix("handoff-")
                .map(|percent| percent.parse::<f64>().unwrap() / 100.0),
            governor: variant == "governor",
            ..advanced::Options::default()
        };
        let mut events = Vec::new();
        let r = advanced::run(
            advanced::RunSpec {
                space: &space,
                options: &options,
                evaluations: budget,
                seed,
                warm: &[],
                target: None,
                inner_threads: 4,
            },
            &evaluate,
            &mut |e| {
                events.push(e.clone());
                Ok(())
            },
            &|| None,
        )
        .unwrap();
        advanced::replay(&events).unwrap();
        json!({"history":r.history.iter().map(|s|&s.observation).collect::<Vec<_>>(),"proposal_seconds":r.proposal_seconds,
            "evaluations":r.evaluations,"peak_pending":r.peak_pending,"trace":events})
    };
    let best = result["history"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|o| {
            let value: Value = serde_json::from_value(o["value"].clone()).unwrap();
            match value {
                Value::Finite(v) => Some(v),
                _ => None,
            }
        })
        .fold(f64::INFINITY, f64::min);
    println!(
        "{}",
        json!({"variant":variant,"problem":problem,"seed":seed,"budget":budget,"best":best,"seconds":began.elapsed().as_secs_f64(),"result":result})
    );
}
