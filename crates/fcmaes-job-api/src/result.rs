use crate::{JobSpec, Result, storage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetryRecord {
    pub retry_id: usize,
    pub run_seed: u64,
    pub spec_digest: String,
    pub initialization: String,
    pub x: Vec<f64>,
    pub y: Option<f64>,
    pub actual_evaluations: u64,
    pub elapsed_seconds: f64,
    pub valid: bool,
    pub feasible: bool,
}
pub fn replay(dir: &Path, spec: &JobSpec) -> Result<Vec<RetryRecord>> {
    let path = dir.join("retries.jsonl");
    if !path.exists() {
        return Ok(vec![]);
    }
    let records: Vec<RetryRecord> = storage::read_checked(&path, 1)?;
    let digest = spec.digest()?;
    let mut ids = BTreeSet::new();
    for r in &records {
        if !ids.insert(r.retry_id)
            || r.retry_id >= spec.budget.retries
            || r.spec_digest != digest
            || r.x.len() != spec.problem.config.dimension
            || r.x.iter().any(|x| !x.is_finite())
            || r.x
                .iter()
                .zip(
                    spec.problem
                        .bounds
                        .lower
                        .iter()
                        .zip(&spec.problem.bounds.upper),
                )
                .any(|(x, (l, u))| x < l || x > u)
            || !r.elapsed_seconds.is_finite()
            || r.elapsed_seconds < 0.0
            || r.y.is_some_and(|y| !y.is_finite())
            || r.valid != r.y.is_some()
            || (r.feasible && !r.valid)
            || r.actual_evaluations == 0
            || r.actual_evaluations > spec.maximum_evaluations()? / (spec.budget.retries as u64)
        {
            return Err("invalid or duplicate retry journal record".into());
        }
    }
    Ok(records)
}
pub fn manifest(
    dir: &Path,
    spec: &JobSpec,
    status: &str,
    complete: bool,
    elapsed: f64,
    worker_digest: &str,
) -> Result<Value> {
    let records = replay(dir, spec)?;
    let known = records.iter().try_fold(0u64, |sum, r| {
        sum.checked_add(r.actual_evaluations)
            .ok_or("evaluation sum overflow")
    })?;
    let best = records
        .iter()
        .filter(|r| r.feasible)
        .min_by(|a, b| a.y.unwrap().total_cmp(&b.y.unwrap()));
    // Link only artifacts that exist: a job cancelled while waiting for CPU
    // slots never created a journal.
    let artifacts = if dir.join("retries.jsonl").exists() {
        json!({"retries":"retries.jsonl"})
    } else {
        json!({})
    };
    Ok(
        json!({"schema_version":1,"tutorial":"fcmaes-native-worker","formulation":"single_objective",
            "command":"fcmaes-native-worker execute JOB_DIRECTORY", "status":status,
            "seed":spec.execution.seed,"workers":spec.execution.workers,
            "requested_evaluations":spec.requested()?,"actual_evaluations":if complete {Some(known)}else{None},
            "elapsed_seconds":elapsed,"objectives":[{"column":"objective","label":spec.problem.id,"unit":if spec.problem.id=="gtop.cassini1" {"km/s"}else{"dimensionless"}}],
            "descriptors":[],"feasible_results":records.iter().filter(|r|r.feasible).count(),
            "best":best.map(|r|json!({"x":r.x,"y":r.y,"retry_id":r.retry_id})),
            "artifacts":artifacts,
            "runner":{"profile":"fcmaes.runner/0.1","spec_digest":spec.digest()?,"worker_digest":worker_digest,
                "problem":spec.problem,"optimizer":spec.optimizer,"budget":spec.budget,"execution":spec.execution,
                "termination_reason":status,"accounting_complete":complete,"known_evaluations_lower_bound":known,
                "completed_retries":records.len(),"constraints":"none; finite objective required","data":"built-in",
                "initialization":"uniform bounds; solver seed=retry_run_seed(root_seed,global_retry_id); runid=0"}
        }),
    )
}
/// Comparison preserves all arms and labels comparability; never invents a winner.
pub fn compare(runs: &[Value]) -> Result<Value> {
    if runs.len() < 2 {
        return Err("comparison needs at least two manifests".into());
    }
    for r in runs {
        if r["schema_version"] != 1
            || (!r["runner"]["profile"].is_null() && !r["objectives"].is_array())
            || !r["tutorial"].is_string()
            || !r["formulation"].is_string()
        {
            return Err("not a run.json v1 manifest".into());
        }
    }
    let first = &runs[0];
    let mut warnings = Vec::<String>::new();
    for r in &runs[1..] {
        for key in ["tutorial", "formulation", "objectives"] {
            if r[key] != first[key] {
                return Err(format!("incompatible {key}").into());
            }
        }
        if r["runner"]["profile"].is_null() || first["runner"]["profile"].is_null() {
            warnings.push("Legacy manifest: compatibility of data, transformations and constraints requires domain review; descriptive comparison only".into());
        } else {
            for key in ["problem", "data", "constraints"] {
                if r["runner"][key] != first["runner"][key] {
                    return Err(format!("incompatible runner.{key}").into());
                }
            }
            if r["runner"]["worker_digest"] != first["runner"]["worker_digest"] {
                warnings.push("Different worker binaries; review code provenance".into());
            }
        }
        if r["seed"] != first["seed"] {
            warnings.push("Seeds differ; arms are unpaired".into());
        }
        if r["workers"] != first["workers"] {
            warnings.push("Worker allocations differ".into());
        }
        let mode = &r["runner"]["budget"]["evaluation_limit_mode"];
        if !mode.is_null() && *mode != first["runner"]["budget"]["evaluation_limit_mode"] {
            warnings.push(
                "Evaluation limit modes differ (hard vs soft); compare actual, not requested, work"
                    .into(),
            );
        }
    }
    if first["formulation"] != "single_objective" {
        warnings.push("No automatic Pareto or hypervolume comparison; common scaling/reference definitions required".into());
    }
    for r in runs {
        if r["actual_evaluations"].is_null() {
            warnings.push(
                "At least one arm has incomplete accounting (forced stop); its actual work is only a lower bound"
                    .into(),
            );
        }
    }
    warnings.sort();
    warnings.dedup();
    let budget = |r: &Value| r["runner"]["budget"]["wall_time_seconds"].clone();
    let summary: Vec<Value> = runs
        .iter()
        .enumerate()
        .map(|(arm, r)| {
            json!({"arm":arm,"optimizer":r["runner"]["optimizer"]["kind"],
                "evaluation_limit_mode":r["runner"]["budget"]["evaluation_limit_mode"],
                "status":r["status"],"seed":r["seed"],"workers":r["workers"],
                "requested_evaluations":r["requested_evaluations"],"actual_evaluations":r["actual_evaluations"],
                "known_evaluations_lower_bound":r["runner"]["known_evaluations_lower_bound"],
                "completed_retries":r["runner"]["completed_retries"],"feasible_results":r["feasible_results"],
                "best_y":r["best"]["y"],"elapsed_seconds":r["elapsed_seconds"]})
        })
        .collect();
    Ok(
        json!({"summary":summary,"arms":runs,"equal_requested_work":runs.iter().all(|r|r["requested_evaluations"]==first["requested_evaluations"]),
        "equal_actual_work":!first["actual_evaluations"].is_null() && runs.iter().all(|r|r["actual_evaluations"]==first["actual_evaluations"]),
        "equal_wall_time_budget":!budget(first).is_null() && runs.iter().all(|r|budget(r)==budget(first)),
        "warnings":warnings,"interpretation":"Descriptive arms; no statistical winner inferred. Incomplete and zero-feasible arms retained."}),
    )
}
