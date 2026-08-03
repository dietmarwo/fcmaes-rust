use std::fmt;
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

use cmaes::{CMAESOptions, DVector, Weights};
use cpu_time::ProcessTime;
use fcmaes_core::{Cmaes, CmaesParams, Fitness};
use rayon::ThreadPool;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::objective::SharedObjective;
use crate::problems::Problem;

const HARD_EVALUATION_LIMIT: usize = 1_000_000_000;
const SIGMA0: f64 = 0.3;
const SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Arm {
    A,
    B,
    C,
}

impl Arm {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "a" => Ok(Self::A),
            "b" => Ok(Self::B),
            "c" => Ok(Self::C),
            _ => Err(format!("unknown arm '{value}'; expected a, b, or c")),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::A => "A: serial single run",
            Self::B => "B: population parallel",
            Self::C => "C: independent multistart",
        }
    }
}

impl fmt::Display for Arm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::A => "a",
            Self::B => "b",
            Self::C => "c",
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Library {
    Fcmaes,
    Cmaes,
}

impl fmt::Display for Library {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Fcmaes => "fcmaes-core",
            Self::Cmaes => "cmaes",
        })
    }
}

#[derive(Clone, Debug)]
pub struct RunMetrics {
    pub best: f64,
    pub evaluations: u64,
    pub wall_seconds: f64,
    pub allocated_seconds: f64,
    pub cpu_seconds: f64,
    pub active_cores: f64,
    pub allocated_cores: f64,
    pub workers: usize,
    pub optimizer_runs: usize,
    pub termination: String,
}

pub struct RunRequest<'a> {
    pub library: Library,
    pub arm: Arm,
    pub problem: &'a Problem,
    pub injected_cost_ns: u64,
    pub deadline: Duration,
    pub workers: usize,
    pub population: usize,
    pub seed: u64,
    pub pool: &'a ThreadPool,
}

pub fn run_one(request: RunRequest<'_>) -> RunMetrics {
    let RunRequest {
        library,
        arm,
        problem,
        injected_cost_ns,
        deadline,
        workers,
        population,
        seed,
        pool,
    } = request;
    let objective = SharedObjective::new(problem, injected_cost_ns);
    let cpu_start = ProcessTime::now();
    let (optimizer_runs, termination) = match (library, arm) {
        (Library::Fcmaes, Arm::A) => (
            1,
            run_fcmaes_loop(&objective, problem, deadline, population, seed, None),
        ),
        (Library::Fcmaes, Arm::B) => (
            1,
            run_fcmaes_loop(&objective, problem, deadline, population, seed, Some(pool)),
        ),
        (Library::Fcmaes, Arm::C) => (
            workers,
            run_fcmaes_multistart(
                &objective, problem, deadline, population, seed, workers, pool,
            ),
        ),
        (Library::Cmaes, Arm::A) => (
            1,
            run_cmaes_loop(&objective, problem, deadline, population, seed, None),
        ),
        (Library::Cmaes, Arm::B) => (
            1,
            run_cmaes_loop(&objective, problem, deadline, population, seed, Some(pool)),
        ),
        (Library::Cmaes, Arm::C) => (
            workers,
            run_cmaes_multistart(
                &objective, problem, deadline, population, seed, workers, pool,
            ),
        ),
    };
    let cpu_seconds = cpu_start.elapsed().as_secs_f64();
    let wall_seconds = objective.elapsed().as_secs_f64();
    let allocated_seconds = wall_seconds.max(deadline.as_secs_f64());
    RunMetrics {
        best: objective.best(),
        evaluations: objective.calls(),
        wall_seconds,
        allocated_seconds,
        cpu_seconds,
        active_cores: if wall_seconds > 0.0 {
            cpu_seconds / wall_seconds
        } else {
            0.0
        },
        allocated_cores: if allocated_seconds > 0.0 {
            cpu_seconds / allocated_seconds
        } else {
            0.0
        },
        workers: if arm == Arm::A { 1 } else { workers },
        optimizer_runs,
        termination,
    }
}

fn fcmaes_optimizer(problem: &Problem, population: usize, seed: u64) -> Cmaes {
    let fitness = Fitness::new(problem.dimension, 1, Vec::new(), Vec::new());
    Cmaes::new(
        fitness,
        &problem.initial_mean(),
        &[SIGMA0],
        &CmaesParams {
            popsize: population as i32,
            mu: 0,
            max_evaluations: HARD_EVALUATION_LIMIT as u64,
            accuracy: 0.0,
            stop_fitness: f64::NEG_INFINITY,
            stop_tol_hist_fun: 0.0,
            seed,
            ..Default::default()
        },
    )
}

fn run_fcmaes_loop(
    objective: &SharedObjective,
    problem: &Problem,
    deadline: Duration,
    population: usize,
    seed: u64,
    pool: Option<&ThreadPool>,
) -> String {
    let mut optimizer = fcmaes_optimizer(problem, population, seed);
    while objective.elapsed() < deadline && optimizer.stop() == 0 {
        let candidates = optimizer.ask();
        let values: Vec<f64> = if let Some(pool) = pool {
            pool.install(|| {
                candidates
                    .par_iter()
                    .map(|candidate| objective.evaluate(candidate))
                    .collect()
            })
        } else {
            candidates
                .iter()
                .map(|candidate| objective.evaluate(candidate))
                .collect()
        };
        optimizer.tell(&values);
    }
    if optimizer.stop() == 0 {
        "deadline".to_owned()
    } else {
        format!("internal-stop-{}", optimizer.stop())
    }
}

fn cmaes_options(problem: &Problem, population: usize, seed: u64) -> CMAESOptions {
    CMAESOptions::new(problem.initial_mean(), SIGMA0)
        .population_size(population)
        .weights(Weights::Negative)
        .max_function_evals(HARD_EVALUATION_LIMIT)
        .tol_fun(0.0)
        .tol_fun_rel(0.0)
        .tol_fun_hist(0.0)
        .tol_x(-1.0)
        .tol_stagnation(usize::MAX)
        .tol_x_up(f64::MAX)
        .tol_condition_cov(f64::MAX)
        .seed(seed)
}

fn run_cmaes_loop(
    objective: &SharedObjective,
    problem: &Problem,
    deadline: Duration,
    population: usize,
    seed: u64,
    pool: Option<&ThreadPool>,
) -> String {
    let shared = objective.clone();
    let function = move |point: &DVector<f64>| shared.evaluate(point.as_slice());
    let mut optimizer = match cmaes_options(problem, population, seed).build(function) {
        Ok(optimizer) => optimizer,
        Err(error) => return format!("build-error-{error:?}"),
    };
    while objective.elapsed() < deadline {
        let termination = if let Some(pool) = pool {
            pool.install(|| optimizer.next_parallel())
        } else {
            optimizer.next()
        };
        if let Some(data) = termination {
            return format!("internal-stop-{:?}", data.reasons);
        }
    }
    "deadline".to_owned()
}

fn run_fcmaes_multistart(
    objective: &SharedObjective,
    problem: &Problem,
    deadline: Duration,
    population: usize,
    seed: u64,
    workers: usize,
    pool: &ThreadPool,
) -> String {
    let terminations = Arc::new(Mutex::new(Vec::with_capacity(workers)));
    let barrier = Arc::new(Barrier::new(workers));
    pool.scope(|scope| {
        for worker in 0..workers {
            let objective = objective.clone();
            let problem = problem.clone();
            let terminations = Arc::clone(&terminations);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move |_| {
                barrier.wait();
                let termination = run_fcmaes_loop(
                    &objective,
                    &problem,
                    deadline,
                    population,
                    seed.wrapping_add((worker as u64).wrapping_mul(SEED_STRIDE)),
                    None,
                );
                terminations
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(termination);
            });
        }
    });
    summarize_terminations(&terminations)
}

fn run_cmaes_multistart(
    objective: &SharedObjective,
    problem: &Problem,
    deadline: Duration,
    population: usize,
    seed: u64,
    workers: usize,
    pool: &ThreadPool,
) -> String {
    let terminations = Arc::new(Mutex::new(Vec::with_capacity(workers)));
    let barrier = Arc::new(Barrier::new(workers));
    pool.scope(|scope| {
        for worker in 0..workers {
            let objective = objective.clone();
            let problem = problem.clone();
            let terminations = Arc::clone(&terminations);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move |_| {
                barrier.wait();
                let termination = run_cmaes_loop(
                    &objective,
                    &problem,
                    deadline,
                    population,
                    seed.wrapping_add((worker as u64).wrapping_mul(SEED_STRIDE)),
                    None,
                );
                terminations
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(termination);
            });
        }
    });
    summarize_terminations(&terminations)
}

fn summarize_terminations(terminations: &Mutex<Vec<String>>) -> String {
    let terminations = terminations
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let early = terminations
        .iter()
        .filter(|termination| termination.as_str() != "deadline")
        .count();
    if early == 0 {
        "deadline".to_owned()
    } else {
        format!("{early}-of-{}-internal-stops", terminations.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn cmaes_contract_is_active_and_slice_is_zero_copy() {
        let options = CMAESOptions::new(vec![0.0; 3], SIGMA0);
        assert_eq!(options.weights, Weights::Negative);
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(vector.as_ptr(), vector.as_slice().as_ptr());
    }

    #[test]
    fn declared_initial_distributions_have_matching_moments() {
        let problem = Problem {
            key: "moment",
            label: "moment",
            dimension: 1,
            lower: vec![-10.0],
            upper: vec![10.0],
            initial_normalized: 0.25,
            optimum: 0.0,
            natural_cost_only: false,
            evaluator: |_| 0.0,
        };
        let mut fcmaes_samples = Vec::new();
        let cmaes_samples = Arc::new(Mutex::new(Vec::new()));
        for seed in 0..200 {
            let mut optimizer = fcmaes_optimizer(&problem, 10, seed);
            fcmaes_samples.extend(optimizer.ask().into_iter().map(|point| point[0]));

            let captured = Arc::clone(&cmaes_samples);
            let objective = move |point: &DVector<f64>| {
                captured
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(point[0]);
                0.0
            };
            let mut optimizer = cmaes_options(&problem, 10, seed).build(objective).unwrap();
            let _ = optimizer.next();
        }
        let cmaes_samples = cmaes_samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for samples in [&fcmaes_samples, &*cmaes_samples] {
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let variance = samples
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / samples.len() as f64;
            assert!((mean - 0.25).abs() < 0.03, "mean={mean}");
            assert!(
                (variance.sqrt() - SIGMA0).abs() < 0.03,
                "sd={}",
                variance.sqrt()
            );
        }
    }
}
