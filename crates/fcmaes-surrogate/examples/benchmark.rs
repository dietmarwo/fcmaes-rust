//! Small equal-budget quality/overhead smoke campaign; not a published comparison.
use fcmaes_core::{BiteOpt, BiteParams, Cmaes, CmaesParams, Fitness, Rng};
use fcmaes_surrogate::{
    Config, Event, Observation, RunSpec, Space, Value as Observed, design, problems, run,
    stream_seed,
};
use rayon::{ThreadPoolBuilder, prelude::*};
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

struct Problem {
    name: &'static str,
    space: Space,
    objective: fn(&[f64]) -> f64,
    optimum: f64,
}
fn score(xs: &[Vec<f64>], objective: &(dyn Fn(&[f64]) -> f64 + Sync)) -> Vec<f64> {
    xs.par_iter().map(|x| objective(x)).collect()
}
fn append(history: &mut Vec<Observation>, xs: Vec<Vec<f64>>, ys: Vec<f64>) {
    for (x, y) in xs.into_iter().zip(ys) {
        history.push(Observation {
            id: history.len() as u64,
            x,
            value: Observed::Finite(y),
        });
    }
}
fn best(history: &[Observation]) -> &Observation {
    history
        .iter()
        .min_by(|a, b| {
            let value = |o: &Observation| {
                if let Observed::Finite(y) = o.value {
                    y
                } else {
                    f64::INFINITY
                }
            };
            value(a).total_cmp(&value(b)).then(a.id.cmp(&b.id))
        })
        .unwrap()
}

fn baseline(
    problem: &Problem,
    config: &Config,
    method: &str,
    seed: u64,
    budget: usize,
    q: usize,
    objective: &(dyn Fn(&[f64]) -> f64 + Sync),
) -> Vec<Observation> {
    let mut history = Vec::new();
    let initial: Vec<Vec<f64>> = design::initial_design(
        config.initial_points.min(budget),
        problem.space.dimension(),
        seed,
    )
    .iter()
    .map(|x| problem.space.from_unit(x).unwrap())
    .collect();
    for xs in initial.chunks(q) {
        append(&mut history, xs.to_vec(), score(xs, objective));
    }
    if history.len() == budget {
        return history;
    }
    let incumbent = best(&history).x.clone();
    if method == "random" {
        let mut rng = Rng::new(stream_seed(seed, 91, 0));
        while history.len() < budget {
            let xs: Vec<Vec<f64>> = (0..q.min(budget - history.len()))
                .map(|_| {
                    problem
                        .space
                        .from_unit(&rng.uniform_vec(problem.space.dimension()))
                        .unwrap()
                })
                .collect();
            let ys = score(&xs, objective);
            append(&mut history, xs, ys);
        }
    } else if method == "cmaes" {
        let d = problem.space.dimension();
        let population = q.max(4);
        let fitness = Fitness::bounded(d, 1, &vec![0.0; d], &vec![1.0; d]);
        let mut optimizer = Cmaes::new(
            fitness,
            &problem.space.to_unit(&incumbent).unwrap(),
            &[0.25],
            &CmaesParams {
                popsize: population as i32,
                max_evaluations: u64::MAX,
                seed,
                ..CmaesParams::default()
            },
        );
        while history.len() < budget {
            let ask = optimizer.ask();
            let take = ask.len().min(budget - history.len());
            let xs: Vec<Vec<f64>> = ask[..take]
                .iter()
                .map(|x| {
                    problem
                        .space
                        .from_unit(&x.iter().map(|v| v.clamp(0.0, 1.0)).collect::<Vec<_>>())
                        .unwrap()
                })
                .collect();
            let mut ys = Vec::new();
            for batch in xs.chunks(q) {
                ys.extend(score(batch, objective));
            }
            append(&mut history, xs, ys.clone());
            if take == ask.len() {
                optimizer.tell(&ys);
            } // Never fabricate values to fill the last generation.
        }
    } else {
        let mut optimizer = BiteOpt::new(
            &problem.space.lower,
            &problem.space.upper,
            Some(&incumbent),
            &BiteParams {
                popsize: (9 + 3 * problem.space.dimension()) as i32,
                max_evaluations: (budget - history.len()) as u64,
                seed,
                ..BiteParams::default()
            },
        );
        while history.len() < budget {
            let xs = optimizer.ask(q.min(budget - history.len()));
            let ys = score(&xs, objective);
            append(&mut history, xs, ys.clone());
            optimizer.tell(&ys);
        }
    }
    history
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = PathBuf::from("surrogate-smoke.json");
    let mut threads = 8;
    let mut q = 4;
    let mut budget = 48;
    let mut sleep_ms = 0;
    let mut seeds = vec![7, 11, 19, 23, 41];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("expected option value")?;
        match arg.as_str() {
            "--output" => output = value.into(),
            "--threads" => threads = value.parse()?,
            "--concurrent" => q = value.parse()?,
            "--evaluations" => budget = value.parse()?,
            "--sleep-ms" => sleep_ms = value.parse()?,
            "--seeds" => seeds = value.split(',').map(str::parse).collect::<Result<_, _>>()?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if threads == 0 || threads > 16 || q == 0 || q > 8 || !(8..=256).contains(&budget) {
        return Err("threads 1..16, batch size 1..8, evaluations 8..256".into());
    }
    let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
    let problems = [
        Problem {
            name: "branin",
            space: Space::new(vec![-5.0, 0.0], vec![10.0, 15.0], vec![false; 2])?,
            objective: problems::branin,
            optimum: 0.39788735772973816,
        },
        Problem {
            name: "hartmann3",
            space: Space::new(vec![0.0; 3], vec![1.0; 3], vec![false; 3])?,
            objective: problems::hartmann3,
            optimum: -3.86278214782076,
        },
    ];
    let config = Config::default();
    let mut runs = Vec::new();
    for problem in &problems {
        for &seed in &seeds {
            for method in ["gp", "random", "cmaes", "biteopt"] {
                let objective = |x: &[f64]| {
                    if sleep_ms > 0 {
                        std::thread::sleep(Duration::from_millis(sleep_ms));
                    }
                    (problem.objective)(x)
                };
                let began = Instant::now();
                let mut events = Vec::<Event>::new();
                let mut proposal_seconds = 0.0;
                let mut fit_evaluations = 0;
                let history = pool.install(|| {
                    if method == "gp" {
                        let result = run(
                            RunSpec {
                                space: &problem.space,
                                config: &config,
                                evaluations: budget,
                                concurrent: q,
                                seed,
                                warm: &[],
                                target: None,
                            },
                            &|x| Ok(objective(x)),
                            &mut |e| {
                                events.push(e.clone());
                                Ok(())
                            },
                            &|| None,
                        )
                        .unwrap();
                        proposal_seconds = result.proposal_seconds;
                        fit_evaluations = result.fit_evaluations;
                        result.history
                    } else {
                        baseline(problem, &config, method, seed, budget, q, &objective)
                    }
                });
                assert_eq!(history.len(), budget);
                let elapsed = began.elapsed().as_secs_f64();
                let Observed::Finite(best_value) = best(&history).value else {
                    unreachable!()
                };
                let first = best(&history[..config.initial_points.min(budget)]);
                let Observed::Finite(initial_best) = first.value else {
                    unreachable!()
                };
                runs.push(json!({"problem":problem.name,"method":method,"seed":seed,"evaluations":history.len(),"best":best_value,
            "regret":(best_value-problem.optimum).max(0.0),"initial_best":initial_best,"wall_seconds":elapsed,
            "proposal_seconds":proposal_seconds,"fit_evaluations":fit_evaluations,"history":history}));
            }
        }
    }
    let result: Value = json!({"schema_version":"fcmaes.surrogate.benchmark/1","kind":"smoke; no statistical superiority claim",
        "threads":threads,"concurrent":q,"effective_evaluation_concurrency":q.min(threads),"sleep_ms":sleep_ms,"evaluations":budget,"seeds":seeds,"parameters":config,
        "fairness":"all methods share centre/LHS evaluations and their incumbent; CMA-ES population=max(q,4), partial last generations count only evaluated points; no fictitious objective calls",
        "timing":"real wall time, installed pool; GP sink in memory, excluding worker fsync; nondedicated host","runs":runs});
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&result)?)?;
    println!(
        "Wrote {} runs to {}",
        result["runs"].as_array().unwrap().len(),
        output.display()
    );
    Ok(())
}
