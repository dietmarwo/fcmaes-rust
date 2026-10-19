//! Common job-protocol adapter. The original fcmaes-native-worker is preserved.
#![forbid(unsafe_code)]
use fcmaes_core::{RetryBounds, retry_run_seed};
use fcmaes_native_worker::{objective_for, solve};
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
            // The worker inherits the workspace version, which fcmaes-core shares.
            env!("CARGO_PKG_VERSION"),
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
