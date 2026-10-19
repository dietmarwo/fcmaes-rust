//! `fcmaes.retry/1`: independent BiteOpt, DE or CMA-ES retries on registered
//! scalar objectives. BiteOpt and DE resolve through the legacy
//! `fcmaes.job/0.1` definitions, so both profiles run identical retries.
use fcmaes_core::{Cmaes, CmaesParams, Fitness, RetryBounds, RetryRunResult, Rng, retry_run_seed};
use fcmaes_native_worker::{objective_by_id, objective_for, solve};
use optimization_job_api::{
    JobSpec, Result,
    legacy::{self, JobSpec as LegacySpec},
    result::{Evidence, Outcome},
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub const PROFILE: &str = "fcmaes.retry/1";
/// Default CMA-ES population, the `fcmaes-core` default.
pub const CMAES_POPULATION: usize = 31;
/// Default initial CMA-ES step size in the normalized `[-1, 1]` box.
pub const CMAES_SIGMA: f64 = 0.3;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub problem: legacy::Problem,
    pub optimizer: Optimizer,
    pub seed: u64,
    pub retries: usize,
    pub evaluations_per_retry: u64,
    pub evaluation_limit_mode: String,
}
/// Serializes like `legacy::Optimizer` when `sigma` is absent, so BiteOpt and
/// DE payloads and their job digests are unchanged by the CMA-ES addition.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Optimizer {
    pub kind: String,
    pub parameters: Parameters,
}
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub population: Option<usize>,
    /// CMA-ES only: initial step size in the normalized `[-1, 1]` box.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma: Option<f64>,
}

enum Engine {
    /// BiteOpt or DE through the shared native solver.
    Legacy(LegacySpec),
    Cmaes {
        problem: legacy::Problem,
        population: usize,
        sigma: f64,
    },
}
pub struct Run {
    engine: Engine,
    payload: Payload,
    bounds: RetryBounds,
    /// Largest evaluation count one retry may report.
    per_retry_limit: u64,
}

pub fn prepare(s: &mut JobSpec) -> Result<Box<dyn Prepared>> {
    let p: Payload = serde_json::from_value(s.payload.clone())?;
    let (engine, payload) = if p.optimizer.kind == "cmaes" {
        cmaes(p)?
    } else {
        legacy_retry(s, p)?
    };
    let problem = match &engine {
        Engine::Legacy(spec) => &spec.problem,
        Engine::Cmaes { problem, .. } => problem,
    };
    let bounds = RetryBounds::new(problem.bounds.lower.clone(), problem.bounds.upper.clone())
        .map_err(|e| format!("bounds: {e}"))?;
    let per_retry_limit = match &engine {
        Engine::Legacy(spec) => spec.maximum_evaluations()? / spec.budget.retries as u64,
        Engine::Cmaes { population, .. } => payload
            .evaluations_per_retry
            .checked_add(*population as u64 - 1)
            .ok_or("evaluation budget overflow")?,
    };
    s.payload = serde_json::to_value(&payload)?;
    Ok(Box::new(Run {
        engine,
        payload,
        bounds,
        per_retry_limit,
    }))
}

fn legacy_retry(s: &JobSpec, p: Payload) -> Result<(Engine, Payload)> {
    if p.optimizer.parameters.sigma.is_some() {
        return Err("optimizer.parameters.sigma applies only to cmaes".into());
    }
    let spec = LegacySpec {
        schema_version: "fcmaes.job/0.1".into(),
        request_key: s.request_key.clone(),
        problem: p.problem,
        optimizer: legacy::Optimizer {
            kind: p.optimizer.kind,
            parameters: legacy::Parameters {
                population: p.optimizer.parameters.population,
            },
        },
        execution: legacy::Execution {
            strategy: "independent_retry".into(),
            seed: p.seed,
            workers: s.resources.outer_workers,
            retry_wave_size: s.resources.outer_workers,
        },
        budget: legacy::Budget {
            retries: p.retries,
            evaluations_per_retry: p.evaluations_per_retry,
            evaluation_limit_mode: p.evaluation_limit_mode,
            wall_time_seconds: s.limits.wall_time_seconds,
            grace_seconds: s.limits.grace_seconds,
        },
        outputs: legacy::Outputs {
            progress: "completed_retries".into(),
        },
    }
    .resolve()?;
    let payload = Payload {
        problem: spec.problem.clone(),
        optimizer: Optimizer {
            kind: spec.optimizer.kind.clone(),
            parameters: Parameters {
                population: spec.optimizer.parameters.population,
                sigma: None,
            },
        },
        seed: spec.execution.seed,
        retries: spec.budget.retries,
        evaluations_per_retry: spec.budget.evaluations_per_retry,
        evaluation_limit_mode: spec.budget.evaluation_limit_mode.clone(),
    };
    Ok((Engine::Legacy(spec), payload))
}

fn cmaes(mut p: Payload) -> Result<(Engine, Payload)> {
    let problem = p.problem.clone().resolve()?;
    if p.retries == 0 || p.evaluations_per_retry == 0 {
        return Err("positive retries and evaluations_per_retry required".into());
    }
    if p.evaluation_limit_mode != "soft" {
        return Err("cmaes evaluates whole generations; use evaluation_limit_mode soft".into());
    }
    let population = p
        .optimizer
        .parameters
        .population
        .unwrap_or(CMAES_POPULATION);
    if !(4..=1_000_000).contains(&population) {
        return Err("population must be 4..1000000".into());
    }
    let sigma = p.optimizer.parameters.sigma.unwrap_or(CMAES_SIGMA);
    if !sigma.is_finite() || sigma <= 0.0 || sigma > 2.0 {
        return Err("cmaes sigma must lie in (0, 2] (normalized box units)".into());
    }
    p.evaluations_per_retry
        .checked_add(population as u64 - 1)
        .and_then(|e| e.checked_mul(p.retries as u64))
        .ok_or("evaluation budget overflow")?;
    p.problem = problem.clone();
    p.optimizer.parameters = Parameters {
        population: Some(population),
        sigma: Some(sigma),
    };
    Ok((
        Engine::Cmaes {
            problem,
            population,
            sigma,
        },
        p,
    ))
}

/// One CMA-ES retry: a uniform random start in the box (from a stream
/// derived from the retry seed), the job's `sigma` in the normalized box, and
/// serial evaluation.
pub fn cmaes_retry(
    bounds: &RetryBounds,
    seed: u64,
    population: usize,
    sigma: f64,
    evaluations: u64,
    objective: fn(&[f64]) -> f64,
) -> (RetryRunResult, i32) {
    let mut start = Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let guess: Vec<f64> = bounds
        .lower()
        .iter()
        .zip(bounds.upper())
        .map(|(l, u)| l + (u - l) * start.uniform01())
        .collect();
    let mut fitness = Fitness::bounded(bounds.dim(), 1, bounds.lower(), bounds.upper());
    fitness.set_normalize(true);
    let mut cma = Cmaes::new(
        fitness,
        &guess,
        &[sigma],
        &CmaesParams {
            popsize: population as i32,
            max_evaluations: evaluations,
            seed,
            ..Default::default()
        },
    );
    let r = cma.optimize(&objective, 1);
    // CMA-ES evaluates the clamped decode of its sample but stores the
    // unclamped one, which can lie an ulp outside the box on a bound. Clamping
    // gives exactly the point that was evaluated.
    let x =
        r.x.iter()
            .zip(bounds.lower().iter().zip(bounds.upper()))
            .map(|(v, (l, u))| v.clamp(*l, *u))
            .collect();
    (
        RetryRunResult {
            x,
            y: r.y,
            evaluations: r.evaluations,
        },
        r.stop,
    )
}

/// Judges one retry's answer: the best point it evaluated and its value.
/// The point is re-evaluated with `objective`; a value at or above `1e99`
/// (the fcmaes penalty) or non-finite is no answer. A point outside the box
/// refutes `best_found` (`infeasible_point`); a value that differs from the
/// re-evaluation refutes it too (`objective_mismatch`), leaving a feasible
/// point with the recomputed value. Returns the outcome without termination
/// and details (apart from `reported_objective`), and the number of
/// validation evaluations.
pub fn judge(
    r: RetryRunResult,
    bounds: &RetryBounds,
    objective: fn(&[f64]) -> f64,
) -> Result<(Outcome, u64)> {
    let usable = |v: f64| v.is_finite() && v < 1e99;
    if !usable(r.y) {
        let outcome = Outcome {
            claim: "none".into(),
            feasible: Some(false),
            evidence: Some(Evidence::claimed(&[])),
            ..Outcome::default()
        };
        return Ok((outcome, 0));
    }
    if r.x.len() != bounds.dim() || r.x.iter().any(|v| !v.is_finite()) {
        return Err("native candidate has the wrong dimension or a non-finite entry".into());
    }
    let checks = Evidence::rechecked(&["objective", "bounds"]);
    let outside =
        r.x.iter()
            .zip(bounds.lower().iter().zip(bounds.upper()))
            .position(|(v, (l, u))| v < l || v > u);
    let value = objective(&r.x);
    let outcome = if let Some(i) = outside {
        let message = format!(
            "x[{i}] = {} lies outside [{}, {}]",
            r.x[i],
            bounds.lower()[i],
            bounds.upper()[i]
        );
        Outcome {
            claim: "none".into(),
            feasible: Some(false),
            objective: value.is_finite().then_some(value),
            x: Some(r.x),
            evidence: Some(checks.refuting("best_found", "infeasible_point", message)),
            ..Outcome::default()
        }
    } else if value != r.y {
        let message = format!("reported {} but the point evaluates to {value}", r.y);
        let refuted = checks.refuting("best_found", "objective_mismatch", message);
        let feasible = usable(value);
        Outcome {
            claim: if feasible { "feasible" } else { "none" }.into(),
            feasible: Some(feasible),
            objective: value.is_finite().then_some(value),
            x: Some(r.x),
            details: json!({"reported_objective":r.y}),
            evidence: Some(refuted),
            ..Outcome::default()
        }
    } else {
        Outcome {
            claim: "best_found".into(),
            feasible: Some(true),
            objective: Some(value),
            x: Some(r.x),
            evidence: Some(checks),
            ..Outcome::default()
        }
    };
    Ok((outcome, 1))
}

impl Prepared for Run {
    fn tasks(&self) -> usize {
        self.payload.retries
    }
    fn cancel(&self) {} // fcmaes core has no in-retry stop token.
    fn solve(&self, task: usize, _: f64) -> Result<TaskResult> {
        let seed = retry_run_seed(self.payload.seed, task);
        let (r, objective, stop) = match &self.engine {
            Engine::Legacy(spec) => {
                let objective = objective_for(spec);
                (solve(spec, &self.bounds, seed, &objective), objective, 0)
            }
            Engine::Cmaes {
                problem,
                population,
                sigma,
            } => {
                let objective = objective_by_id(&problem.id);
                let (r, stop) = cmaes_retry(
                    &self.bounds,
                    seed,
                    *population,
                    *sigma,
                    self.payload.evaluations_per_retry,
                    objective,
                );
                (r, objective, stop)
            }
        };
        if r.evaluations == 0 || r.evaluations > self.per_retry_limit {
            return Err("native evaluation budget mismatch".into());
        }
        // BiteOpt and DE always spend their budget; CMA-ES can stop earlier
        // on its own convergence tests.
        let termination = if r.evaluations >= self.payload.evaluations_per_retry {
            "evaluation_limit"
        } else {
            "optimizer_stop"
        };
        let evaluations = r.evaluations;
        let (outcome, validations) = judge(r, &self.bounds, objective)?;
        let mut details = json!({"run_seed":seed,"evaluation_limit_mode":self.payload.evaluation_limit_mode,"requested_evaluations":self.payload.evaluations_per_retry,"global_optimum_certified":false});
        if matches!(self.engine, Engine::Cmaes { .. }) {
            details["stop_code"] = json!(stop);
        }
        if let Some(reported) = outcome.details.get("reported_objective") {
            details["reported_objective"] = reported.clone();
        }
        Ok(TaskResult {
            outcome: Outcome {
                termination: termination.into(),
                details,
                ..outcome
            },
            work: [
                ("evaluations".into(), evaluations),
                ("validation_evaluations".into(), validations),
            ]
            .into(),
        })
    }
}
