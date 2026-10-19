use crate::{JobSpec, Request, Result, legacy, storage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub termination: String,
    /// none, feasible, best_found, numerical_optimum, first_order,
    /// non_dominated_set or elite_archive. A non_dominated_set describes a
    /// set of points in `details` and has no single `x` or `objective`.
    pub claim: String,
    pub feasible: Option<bool>,
    pub objective: Option<f64>,
    pub x: Option<Vec<f64>>,
    pub details: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub task_id: usize,
    pub spec_digest: String,
    pub elapsed_seconds: f64,
    pub work: BTreeMap<String, u64>,
    pub outcome: Outcome,
}
pub fn replay(dir: &Path, spec: &JobSpec) -> Result<Vec<Record>> {
    let path = dir.join("results.jsonl");
    if !path.exists() {
        return Ok(vec![]);
    }
    let mut records: Vec<Record> = storage::read_checked(&path, 1)?;
    let digest = spec.digest()?;
    let mut ids = BTreeSet::new();
    for r in &records {
        let o = &r.outcome;
        let set = o.claim == "non_dominated_set";
        if r.task_id >= spec.task_count.ok_or("unresolved task_count")?
            || !ids.insert(r.task_id)
            || r.spec_digest != digest
            || !r.elapsed_seconds.is_finite()
            || r.elapsed_seconds < 0.0
            || o.termination.is_empty()
            || !matches!(
                o.claim.as_str(),
                "none"
                    | "feasible"
                    | "best_found"
                    | "numerical_optimum"
                    | "first_order"
                    | "non_dominated_set"
                    | "elite_archive"
            )
            || o.objective.is_some_and(|v| !v.is_finite())
            || o.x
                .as_ref()
                .is_some_and(|v| v.is_empty() || v.iter().any(|x| !x.is_finite()))
            || o.objective.is_some() != o.x.is_some()
            || (o.claim != "none" && o.feasible != Some(true))
            // A set claim lives in details; every other claim names one point.
            || (set && o.x.is_some())
            || (!set && o.feasible == Some(true) && o.x.is_none())
            || r.work.keys().any(|k| !crate::safe_name(k))
        {
            return Err("invalid, mismatched or duplicate result record".into());
        }
    }
    records.sort_by_key(|r| r.task_id);
    Ok(records)
}
pub fn counts(records: &[Record]) -> Result<BTreeMap<String, u64>> {
    let mut counts = BTreeMap::<String, u64>::new();
    for r in records {
        for (k, v) in &r.work {
            let n = counts.entry(k.clone()).or_default();
            *n = n.checked_add(*v).ok_or("work count overflow")?;
        }
    }
    Ok(counts)
}
pub fn manifest(
    dir: &Path,
    spec: &Request,
    status: &str,
    complete: bool,
    elapsed: f64,
    worker_digest: &str,
) -> Result<Value> {
    let s = match spec {
        Request::Legacy(s) => {
            return legacy::result::manifest(dir, s, status, complete, elapsed, worker_digest);
        }
        Request::Job(s) => s,
    };
    let records = replay(dir, s)?;
    let known = counts(&records)?;
    Ok(json!({
        "schema_version":"optimization.result/0.1", "status":status,
        "worker":s.worker,"payload_schema":s.payload_schema,"elapsed_seconds":elapsed,
        "completed_tasks":records.len(),"requested_tasks":s.task_count,
        "work":if complete { Some(&known) } else { None },"known_work_lower_bound":known,
        "results":records,
        "artifacts":if dir.join("results.jsonl").exists(){json!({"results":"results.jsonl"})}else{json!({})},
        "runner":{"profile":"optimization.runner/0.1","spec_digest":s.digest()?,"worker_digest":worker_digest,"accounting_complete":complete,"resources":s.resources,"recovery":"journal_only_no_resume"}
    }))
}
pub fn progress(dir: &Path, s: &Request) -> Result<Value> {
    match s {
        Request::Legacy(s) => {
            let r = legacy::result::replay(dir, s)?;
            Ok(
                json!({"completed_retries":r.len(),"known_evaluations_lower_bound":r.iter().map(|r|r.actual_evaluations).sum::<u64>()}),
            )
        }
        Request::Job(s) => {
            let r = replay(dir, s)?;
            Ok(
                json!({"completed_tasks":r.len(),"requested_tasks":s.task_count,"known_work_lower_bound":counts(&r)?}),
            )
        }
    }
}
pub fn repair(dir: &Path, s: &Request) -> Result<()> {
    match s {
        Request::Legacy(s) => {
            legacy::result::replay(dir, s)?;
            let p = dir.join("retries.jsonl");
            if p.exists() {
                storage::repair_tail::<legacy::result::RetryRecord>(&p, 1)?;
            }
        }
        Request::Job(s) => {
            replay(dir, s)?;
            let p = dir.join("results.jsonl");
            if p.exists() {
                storage::repair_tail::<Record>(&p, 1)?;
            }
        }
    }
    Ok(())
}
pub fn worker_outcome(dir: &Path, s: &Request) -> Result<Option<(String, f64)>> {
    if !dir.join("run.json").exists() {
        return Ok(None);
    }
    let r: Value = serde_json::from_slice(&std::fs::read(dir.join("run.json"))?)?;
    let status = r["status"].as_str().unwrap_or_default();
    if r["runner"]["spec_digest"] != s.digest()?
        || r["runner"]["accounting_complete"] != true
        || !matches!(status, "completed" | "cancelled" | "timed_out")
    {
        return Ok(None);
    }
    let (done, total, valid) = match s {
        Request::Legacy(s) => {
            let rr = legacy::result::replay(dir, s)?;
            (
                rr.len(),
                s.budget.retries,
                r["actual_evaluations"].as_u64()
                    == Some(rr.iter().map(|r| r.actual_evaluations).sum()),
            )
        }
        Request::Job(s) => {
            let rr = replay(dir, s)?;
            (
                rr.len(),
                s.task_count.ok_or("unresolved task count")?,
                r["work"] == serde_json::to_value(counts(&rr)?)?
                    && r["results"] == serde_json::to_value(&rr)?,
            )
        }
    };
    if !valid {
        return Ok(None);
    }
    if status == "completed" && done != total {
        return Err("worker completed without all tasks".into());
    }
    let elapsed = r["elapsed_seconds"]
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .ok_or("invalid worker elapsed time")?;
    Ok(Some((status.into(), elapsed)))
}
pub fn compare(runs: &[Value]) -> Result<Value> {
    if runs
        .iter()
        .all(|r| r["schema_version"] != "optimization.result/0.1")
    {
        return legacy::result::compare(runs);
    }
    if runs.iter().any(|r| {
        r["schema_version"] != "optimization.result/0.1"
            || r["payload_schema"] != runs[0]["payload_schema"]
            || r["runner"]["spec_digest"] != runs[0]["runner"]["spec_digest"]
    }) {
        return Err("comparison requires the same resolved job content and profile; LP and SQP claims are not interchangeable".into());
    }
    Ok(
        json!({"arms":runs,"warnings":["No automatic winner; inspect mathematical outcomes, worker hashes and complete accounting."]}),
    )
}
