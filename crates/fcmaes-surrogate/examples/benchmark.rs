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
    evaluations: usize,
}
/// The tutorial's RK4 damped-oscillator calibration (four replicas, 2048
/// steps): fit damping and stiffness, minimum 0 at (0.4, 2.3) up to
/// integration error.
fn oscillator(x: &[f64]) -> f64 {
    let (c, k) = (x[0], x[1]);
    let rhs = |y: [f64; 2]| [y[1], -c * y[1] - k * y[0]];
    let steps = 2048;
    let h = 4.0 / steps as f64;
    let mut total = 0.0;
    for replica in 0..4 {
        let initial = 1.0 + 0.1 * replica as f64;
        let mut y = [initial, 0.0];
        let mut loss = 0.0;
        for i in 0..steps {
            let a = rhs(y);
            let b = rhs([y[0] + h * a[0] / 2.0, y[1] + h * a[1] / 2.0]);
            let d = rhs([y[0] + h * b[0] / 2.0, y[1] + h * b[1] / 2.0]);
            let e = rhs([y[0] + h * d[0], y[1] + h * d[1]]);
            y = [
                y[0] + h * (a[0] + 2.0 * b[0] + 2.0 * d[0] + e[0]) / 6.0,
                y[1] + h * (a[1] + 2.0 * b[1] + 2.0 * d[1] + e[1]) / 6.0,
            ];
            if (i + 1) % (steps / 8) == 0 {
                let t = (i + 1) as f64 * h;
                let omega = (2.3_f64 - 0.04).sqrt();
                let target = initial
                    * (-0.2 * t).exp()
                    * ((omega * t).cos() + 0.2 / omega * (omega * t).sin());
                loss += (y[0] - target).powi(2);
            }
        }
        total += loss / 8.0;
    }
    total / 4.0
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
    let mut budget = None;
    let mut selected = vec!["branin".to_string(), "hartmann3".to_string()];
    let mut sleep_ms = 0;
    let mut seeds = vec![7, 11, 19, 23, 41];
    let mut config = Config::default();
    let mut methods: Vec<String> = ["gp", "random", "cmaes", "biteopt"]
        .map(String::from)
        .to_vec();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("expected option value")?;
        match arg.as_str() {
            "--output" => output = value.into(),
            "--threads" => threads = value.parse()?,
            "--concurrent" => q = value.parse()?,
            "--evaluations" => budget = Some(value.parse()?),
            "--problems" => selected = value.split(',').map(str::to_string).collect(),
            "--sleep-ms" => sleep_ms = value.parse()?,
            "--seeds" => seeds = value.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--parameters" => config = serde_json::from_str(&value)?,
            "--methods" => methods = value.split(',').map(str::to_string).collect(),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if threads == 0
        || threads > 16
        || q == 0
        || q > 8
        || budget.is_some_and(|b| !(8..=256).contains(&b))
    {
        return Err("threads 1..16, batch size 1..8, evaluations 8..256".into());
    }
    let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
    let catalogue = [
        Problem {
            name: "branin",
            space: Space::new(vec![-5.0, 0.0], vec![10.0, 15.0], vec![false; 2])?,
            objective: problems::branin,
            optimum: 0.39788735772973816,
            evaluations: 48,
        },
        Problem {
            name: "hartmann3",
            space: Space::new(vec![0.0; 3], vec![1.0; 3], vec![false; 3])?,
            objective: problems::hartmann3,
            optimum: -3.8627797873326624,
            evaluations: 48,
        },
        Problem {
            name: "hartmann6",
            space: Space::new(vec![0.0; 6], vec![1.0; 6], vec![false; 6])?,
            objective: problems::hartmann6,
            optimum: -3.322368011415515,
            evaluations: 96,
        },
        Problem {
            name: "styblinski10",
            space: Space::new(vec![-5.0; 10], vec![5.0; 10], vec![false; 10])?,
            objective: problems::styblinski_tang,
            optimum: -391.6616570377142,
            evaluations: 128,
        },
        Problem {
            name: "rosenbrock8",
            space: Space::new(vec![-2.048; 8], vec![2.048; 8], vec![false; 8])?,
            objective: problems::rosenbrock,
            optimum: 0.0,
            evaluations: 128,
        },
        Problem {
            name: "oscillator",
            space: Space::new(vec![0.05, 0.5], vec![1.5, 4.0], vec![false; 2])?,
            objective: oscillator,
            optimum: 0.0,
            evaluations: 32,
        },
    ];
    let mut problems = Vec::new();
    for name in &selected {
        let problem = catalogue
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| format!("unknown problem {name}"))?;
        problems.push(problem);
    }
    config.validate()?;
    if let Some(m) = methods
        .iter()
        .find(|m| !["gp", "random", "cmaes", "biteopt"].contains(&m.as_str()))
    {
        return Err(format!("unknown method {m}").into());
    }
    let mut runs = Vec::new();
    for &problem in &problems {
        let budget = budget.unwrap_or(problem.evaluations);
        for &seed in &seeds {
            for method in methods.iter().map(String::as_str) {
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
                runs.push(json!({
                    "problem": problem.name,
                    "dimension": problem.space.dimension(),
                    "method": method,
                    "seed": seed,
                    "evaluations": history.len(),
                    "best": best_value,
                    "regret": (best_value - problem.optimum).max(0.0),
                    "initial_best": initial_best,
                    "wall_seconds": elapsed,
                    "proposal_seconds": proposal_seconds,
                    "fit_evaluations": fit_evaluations,
                    "history": history,
                }));
            }
        }
    }
    let result: Value = json!({
        "schema_version": "fcmaes.surrogate.benchmark/1",
        "kind": "smoke; no statistical superiority claim",
        "threads": threads,
        "concurrent": q,
        "effective_evaluation_concurrency": q.min(threads),
        "sleep_ms": sleep_ms,
        "evaluations": budget,
        "problems": selected,
        "seeds": seeds,
        "parameters": config,
        "fairness": "all methods share centre/LHS evaluations and their incumbent; CMA-ES population=max(q,4), partial last generations count only evaluated points; no fictitious objective calls",
        "timing": "real wall time, installed pool; GP sink in memory, excluding worker fsync; nondedicated host",
        "runs": runs,
    });
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
