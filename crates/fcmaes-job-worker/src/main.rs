//! Common job-protocol adapter. The original fcmaes-native-worker is preserved.
#![forbid(unsafe_code)]
use fcmaes_core::{
    BiteOpt, BiteParams, De, DeParams, Fitness, RetryBounds, RetryRunResult, retry_run_seed,
};
use optimization_job_api::{
    Result,
    legacy::{self, JobSpec},
    result::Outcome,
    worker::{self, Prepared, TaskResult, Worker},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    problem: legacy::Problem,
    optimizer: legacy::Optimizer,
    seed: u64,
    retries: usize,
    evaluations_per_retry: u64,
    evaluation_limit_mode: String,
}
struct Adapter;
struct Run {
    spec: JobSpec,
    bounds: RetryBounds,
}
impl Worker for Adapter {
    fn describe(&self) -> Value {
        let mut d = worker::description(
            "fcmaes",
            "0.1.5",
            json!(["fcmaes.retry/1"]),
            legacy::discovery()["problems"].clone(),
            1,
        );
        d["optimizers"] = legacy::discovery()["optimizers"].clone();
        d["cancellation"] =
            json!("between retries; active retries run until their budget or process deadline");
        d
    }
    fn prepare(
        &self,
        s: &mut optimization_job_api::JobSpec,
        _: &Path,
    ) -> Result<Box<dyn Prepared>> {
        if s.payload_schema != "fcmaes.retry/1"
            || s.resources.inner_threads != 1
            || !s.inputs.is_empty()
        {
            return Err("fcmaes.retry/1 requires inner_threads=1 and no inputs".into());
        }
        let p: Payload = serde_json::from_value(s.payload.clone())?;
        let spec = JobSpec {
            schema_version: "fcmaes.job/0.1".into(),
            request_key: s.request_key.clone(),
            problem: p.problem,
            optimizer: p.optimizer,
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
        s.payload = serde_json::to_value(Payload {
            problem: spec.problem.clone(),
            optimizer: spec.optimizer.clone(),
            seed: spec.execution.seed,
            retries: spec.budget.retries,
            evaluations_per_retry: spec.budget.evaluations_per_retry,
            evaluation_limit_mode: spec.budget.evaluation_limit_mode.clone(),
        })?;
        let bounds = RetryBounds::new(
            spec.problem.bounds.lower.clone(),
            spec.problem.bounds.upper.clone(),
        )
        .map_err(|e| format!("bounds: {e}"))?;
        Ok(Box::new(Run { spec, bounds }))
    }
}
impl Prepared for Run {
    fn tasks(&self) -> usize {
        self.spec.budget.retries
    }
    fn cancel(&self) {} // fcmaes core has no in-retry stop token.
    fn solve(&self, task: usize, _: f64) -> Result<TaskResult> {
        let seed = retry_run_seed(self.spec.execution.seed, task);
        let objective = objective_for(&self.spec);
        let r = solve(&self.spec, &self.bounds, seed, &objective);
        let valid = r.y.is_finite() && r.y < 1e99;
        if r.evaluations == 0
            || r.evaluations > self.spec.maximum_evaluations()? / self.spec.budget.retries as u64
        {
            return Err("native evaluation budget mismatch".into());
        }
        if valid
            && (r.x.len() != self.bounds.dim()
                || r.x
                    .iter()
                    .zip(self.bounds.lower().iter().zip(self.bounds.upper()))
                    .any(|(x, (l, u))| !x.is_finite() || x < l || x > u)
                || objective(&r.x) != r.y)
        {
            return Err("native candidate failed independent objective/bounds check".into());
        }
        Ok(TaskResult {
            outcome: Outcome {
                termination: "evaluation_limit".into(),
                claim: if valid { "best_found" } else { "none" }.into(),
                feasible: Some(valid),
                objective: valid.then_some(r.y),
                x: valid.then_some(r.x),
                details: json!({"run_seed":seed,"evaluation_limit_mode":self.spec.budget.evaluation_limit_mode,"requested_evaluations":self.spec.budget.evaluations_per_retry,"global_optimum_certified":false}),
            },
            work: [
                ("evaluations".into(), r.evaluations),
                ("validation_evaluations".into(), u64::from(valid)),
            ]
            .into(),
        })
    }
}
fn main() {
    worker::run(Adapter);
}
fn solve(
    spec: &JobSpec,
    bounds: &RetryBounds,
    seed: u64,
    obj: &(impl Fn(&[f64]) -> f64 + Sync),
) -> RetryRunResult {
    if spec.optimizer.kind == "biteopt" {
        let mut solver = BiteOpt::new(
            bounds.lower(),
            bounds.upper(),
            None,
            &BiteParams {
                seed,
                runid: 0,
                popsize: spec.optimizer.parameters.population.unwrap() as i32,
                max_evaluations: spec.budget.evaluations_per_retry,
                ..Default::default()
            },
        );
        let r = solver.optimize(obj);
        RetryRunResult {
            x: r.x,
            y: r.y,
            evaluations: r.evaluations,
        }
    } else {
        let fit = Fitness::new(
            bounds.dim(),
            1,
            bounds.lower().to_vec(),
            bounds.upper().to_vec(),
        );
        let mut solver = De::new(
            fit,
            &[],
            &[],
            None,
            &DeParams {
                seed,
                runid: 0,
                popsize: spec.optimizer.parameters.population.unwrap() as i32,
                max_evaluations: spec.budget.evaluations_per_retry,
                ..Default::default()
            },
        );
        let r = solver.optimize(obj);
        RetryRunResult {
            x: r.x,
            y: r.y,
            evaluations: r.evaluations,
        }
    }
}
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}
fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}
/// Resolve the objective once, so the evaluation loop does not dispatch on
/// the problem name for every candidate.
fn objective_for(spec: &JobSpec) -> fn(&[f64]) -> f64 {
    match spec.problem.id.as_str() {
        "benchmark.sphere" => sphere,
        "benchmark.rosenbrock" => rosenbrock,
        "gtop.cassini1" => fcmaes_gtop::cassini1,
        _ => unreachable!("problem id is validated by JobSpec::resolve"),
    }
}
