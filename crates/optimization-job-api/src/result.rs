use crate::{JobSpec, Request, Result, legacy, safe_name, storage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

/// Result manifests written by this build.
pub const SCHEMA: &str = "optimization.result/0.2";
/// Result manifests `compare` reads. A 0.1 manifest is a valid 0.2 manifest
/// without evidence, objective bounds or the claims added in 0.2.
pub const SCHEMAS: [&str; 2] = ["optimization.result/0.1", SCHEMA];

/// Every claim. A claim states what the worker asserts about the task's
/// answer; `evidence` states how much of that was verified.
pub const CLAIMS: [&str; 12] = [
    "none",
    "feasible",
    "best_found",
    "numerical_optimum",
    "first_order",
    "certified_optimum",
    "non_dominated_set",
    "elite_archive",
    "infeasible",
    "unbounded",
    "infeasible_or_unbounded",
    "locally_infeasible",
];
/// Claims about one returned point, which must be feasible.
const POINT_CLAIMS: [&str; 6] = [
    "feasible",
    "best_found",
    "numerical_optimum",
    "first_order",
    "certified_optimum",
    "elite_archive",
];
/// Point claims that also need the point's (scalar) objective. A feasible
/// point of a multi-objective or constraint-only model has none.
const OBJECTIVE_CLAIMS: [&str; 5] = [
    "best_found",
    "numerical_optimum",
    "first_order",
    "certified_optimum",
    "elite_archive",
];
/// Claims that rule out a feasible returned point: the problem has none
/// (`infeasible`), or the point is a local minimizer of the infeasibility
/// (`locally_infeasible`). `infeasible_or_unbounded` allows one: with a
/// feasible point, the problem is unbounded if the claim holds.
const NO_FEASIBLE_POINT: [&str; 2] = ["infeasible", "locally_infeasible"];
/// Claims about the whole problem that no recomputation of returned points
/// can confirm: their evidence is `claimed` or `certified`, never `rechecked`.
const PROBLEM_CLAIMS: [&str; 3] = ["infeasible", "unbounded", "infeasible_or_unbounded"];
/// Evidence levels, weakest first.
pub const LEVELS: [&str; 3] = ["claimed", "rechecked", "certified"];
/// Properties a worker recomputes independently of the solver.
pub const CHECKS: [&str; 7] = [
    "objective",
    "bounds",
    "constraints",
    "integrality",
    "kkt",
    "non_dominance",
    "niche",
];
/// Why an independent check refuted the solver's claim; the solver
/// competitions' wrong-answer categories.
pub const VERDICTS: [&str; 5] = [
    "infeasible_point",
    "objective_mismatch",
    "false_optimality",
    "false_infeasibility",
    "invalid_certificate",
];

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    /// Why the task stopped, in the solver's words (`evaluation_limit`,
    /// `optimal`, `converged`, `cancelled`, ...). Never a claim.
    pub termination: String,
    /// One of [`CLAIMS`]. A non_dominated_set describes a set of points in
    /// `details` and has no single `x` or `objective`.
    pub claim: String,
    pub feasible: Option<bool>,
    pub objective: Option<f64>,
    pub x: Option<Vec<f64>>,
    pub details: Value,
    /// How the claim is supported. Workers of protocol 0.2 always report it;
    /// a record without it predates 0.2 and is only `claimed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Evidence>,
    /// Best objective of a feasible point (primal) and a proven bound on the
    /// optimum (dual), in the model's own sense.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_bounds: Option<ObjectiveBounds>,
    /// The engine reported success only at relaxed accuracy. Qualifies the
    /// claim; never set together with `none`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub imprecise: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    /// One of [`LEVELS`]: `claimed` (the solver's report alone), `rechecked`
    /// (returned points recomputed from the registered model, as `checks`
    /// list) or `certified` (a proof verified by an independent checker).
    pub level: String,
    /// Properties recomputed independently of the solver, from [`CHECKS`].
    #[serde(default)]
    pub checks: Vec<String>,
    /// The solver's claim that an independent check rejected. `claim` then
    /// states only what survived the check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refuted: Option<Refutation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certificate: Option<Certificate>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Refutation {
    /// The rejected claim.
    pub claim: String,
    /// One of [`VERDICTS`].
    pub verdict: String,
    pub message: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Certificate {
    /// Proof format, for example `vipr`, `lrat`, `veripb` or `farkas`.
    pub format: String,
    /// File name of the proof in the attempt directory, if it was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
    /// SHA-256 of the proof bytes.
    pub sha256: String,
    /// Name and version of the checker that judged it.
    pub checker: String,
    /// `verified` or `rejected`.
    pub verdict: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveBounds {
    /// `minimize` or `maximize`: the dual bound is a lower bound when
    /// minimizing and an upper bound when maximizing.
    pub sense: String,
    /// Objective of the best feasible point, if one was found.
    pub primal: Option<f64>,
    /// Proven bound on the optimal objective, if the solver proved one.
    pub dual: Option<f64>,
}

impl Evidence {
    /// The solver's report alone; `checks` may list recomputations of
    /// returned points that the claim does not depend on.
    pub fn claimed(checks: &[&str]) -> Self {
        Self {
            level: "claimed".into(),
            checks: checks.iter().map(|c| c.to_string()).collect(),
            ..Self::default()
        }
    }
    /// Returned points recomputed independently: `checks` names what.
    pub fn rechecked(checks: &[&str]) -> Self {
        Self {
            level: "rechecked".into(),
            ..Self::claimed(checks)
        }
    }
    /// Records that the checks rejected the solver's `claim`.
    pub fn refuting(mut self, claim: &str, verdict: &str, message: impl Into<String>) -> Self {
        self.refuted = Some(Refutation {
            claim: claim.into(),
            verdict: verdict.into(),
            message: message.into(),
        });
        self
    }
}

fn lowercase_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Outcome {
    /// Checks the rules of the result contract (docs/job-api.md, "Claims and
    /// evidence"). The runner applies them to every journal record.
    pub fn check(&self) -> std::result::Result<(), String> {
        let claim = self.claim.as_str();
        if self.termination.is_empty() {
            return Err("empty termination".into());
        }
        if !CLAIMS.contains(&claim) {
            return Err(format!("unknown claim {claim:?}"));
        }
        if self.objective.is_some_and(|v| !v.is_finite())
            || self
                .x
                .as_ref()
                .is_some_and(|v| v.is_empty() || v.iter().any(|x| !x.is_finite()))
        {
            return Err("objective and x must be finite and x nonempty".into());
        }
        if self.objective.is_some() && self.x.is_none() {
            return Err("an objective needs its point x".into());
        }
        let set = claim == "non_dominated_set";
        if self.feasible == Some(true) && self.x.is_none() && !set {
            return Err("a feasible outcome needs its point x".into());
        }
        if POINT_CLAIMS.contains(&claim) && (self.feasible != Some(true) || self.x.is_none()) {
            return Err(format!("{claim} needs a feasible point x"));
        }
        if OBJECTIVE_CLAIMS.contains(&claim) && self.objective.is_none() {
            return Err(format!("{claim} needs an objective"));
        }
        // A set claim lives in details: no single point or objective.
        if set && (self.feasible != Some(true) || self.x.is_some()) {
            return Err("non_dominated_set needs feasible true and no single point".into());
        }
        if NO_FEASIBLE_POINT.contains(&claim) && self.feasible == Some(true) {
            return Err(format!("{claim} contradicts a feasible point"));
        }
        if self.imprecise && claim == "none" {
            return Err("imprecise qualifies a claim; none has nothing to qualify".into());
        }
        if let Some(b) = &self.objective_bounds
            && (!matches!(b.sense.as_str(), "minimize" | "maximize")
                || b.primal.is_some_and(|v| !v.is_finite())
                || b.dual.is_some_and(|v| !v.is_finite()))
        {
            return Err("objective_bounds need a sense and finite (or null) bounds".into());
        }
        let level = self
            .evidence
            .as_ref()
            .map_or("claimed", |e| e.level.as_str());
        if claim == "certified_optimum" && level != "certified" {
            return Err("certified_optimum needs certified evidence".into());
        }
        if PROBLEM_CLAIMS.contains(&claim) && level == "rechecked" {
            return Err(format!(
                "{claim} concerns the whole problem; it cannot be rechecked"
            ));
        }
        let Some(e) = &self.evidence else {
            return Ok(());
        };
        let mut seen = BTreeSet::new();
        if !LEVELS.contains(&level)
            || e.checks
                .iter()
                .any(|c| !CHECKS.contains(&c.as_str()) || !seen.insert(c))
        {
            return Err("unknown evidence level, or unknown or repeated check".into());
        }
        if level == "rechecked" && e.checks.is_empty() {
            return Err("rechecked evidence must name its checks".into());
        }
        if let Some(c) = &e.certificate
            && (!safe_name(&c.format)
                || !safe_name(&c.checker)
                || !lowercase_sha256(&c.sha256)
                || c.artifact.as_deref().is_some_and(|a| !safe_name(a))
                || !matches!(c.verdict.as_str(), "verified" | "rejected"))
        {
            return Err("certificate needs format, checker, lowercase SHA-256 and verdict verified or rejected".into());
        }
        let verified = e
            .certificate
            .as_ref()
            .is_some_and(|c| c.verdict == "verified");
        if level == "certified" && !verified {
            return Err("certified evidence needs a verified certificate".into());
        }
        if let Some(r) = &e.refuted {
            let rejected = e
                .certificate
                .as_ref()
                .is_some_and(|c| c.verdict == "rejected");
            if !CLAIMS.contains(&r.claim.as_str())
                || r.claim == "none"
                || r.claim == claim
                || !VERDICTS.contains(&r.verdict.as_str())
                || (r.verdict == "invalid_certificate") != rejected
                || (r.verdict != "invalid_certificate" && level != "rechecked")
            {
                return Err(
                    "a refutation names another claim, a known verdict and the check that failed"
                        .into(),
                );
            }
        }
        Ok(())
    }
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
    let mut records: Vec<Record> =
        storage::read_checked(&path, 1).map_err(|e| format!("{}: {e}", path.display()))?;
    let digest = spec.digest()?;
    let mut ids = BTreeSet::new();
    for r in &records {
        let invalid = |why: String| {
            format!(
                "{}: invalid, mismatched or duplicate result record (task {}): {why}",
                path.display(),
                r.task_id
            )
        };
        if r.task_id >= spec.task_count.ok_or("unresolved task_count")?
            || !ids.insert(r.task_id)
            || r.spec_digest != digest
            || !r.elapsed_seconds.is_finite()
            || r.elapsed_seconds < 0.0
            || r.work.keys().any(|k| !safe_name(k))
        {
            return Err(invalid("task, digest, elapsed time or work counters".into()).into());
        }
        r.outcome.check().map_err(invalid)?;
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
        "schema_version":SCHEMA, "status":status,
        "worker":s.worker,"payload_schema":s.payload_schema,"elapsed_seconds":elapsed,
        "completed_tasks":records.len(),"requested_tasks":s.task_count,
        "work":if complete { Some(&known) } else { None },"known_work_lower_bound":known,
        "results":records,
        "artifacts":if dir.join("results.jsonl").exists(){json!({"results":"results.jsonl"})}else{json!({})},
        "runner":{"profile":"optimization.runner/0.1","spec_digest":s.digest()?,"worker_digest":worker_digest,"accounting_complete":complete,"resources":s.resources,"recovery":"journal_only_no_resume"}
    }))
}
/// The manifest of an attempt whose journal cannot be replayed (a corrupt or
/// invalid record before its final line). No record is trusted, so there are
/// no results and no work totals; `runner.journal_error` says why.
pub fn unreadable_manifest(
    spec: &Request,
    elapsed: f64,
    worker_digest: &str,
    error: &str,
) -> Value {
    match spec {
        Request::Legacy(_) => json!({
            "schema_version":1,"status":"failed","actual_evaluations":null,
            "elapsed_seconds":elapsed,
            "runner":{"worker_digest":worker_digest,"accounting_complete":false,"journal_error":error}
        }),
        Request::Job(s) => json!({
            "schema_version":SCHEMA, "status":"failed",
            "worker":s.worker,"payload_schema":s.payload_schema,"elapsed_seconds":elapsed,
            "completed_tasks":null,"requested_tasks":s.task_count,
            "work":null,"known_work_lower_bound":null,"results":[],"artifacts":{},
            "runner":{"profile":"optimization.runner/0.1","spec_digest":s.digest().ok(),"worker_digest":worker_digest,"accounting_complete":false,"resources":s.resources,"recovery":"journal_only_no_resume","journal_error":error}
        }),
    }
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
    let common = |r: &Value| {
        r["schema_version"]
            .as_str()
            .is_some_and(|v| SCHEMAS.contains(&v))
    };
    if !runs.iter().any(common) {
        return legacy::result::compare(runs);
    }
    if runs.iter().any(|r| {
        !common(r)
            || r["payload_schema"] != runs[0]["payload_schema"]
            || r["runner"]["spec_digest"] != runs[0]["runner"]["spec_digest"]
    }) {
        return Err("comparison requires the same resolved job content and profile; LP and SQP claims are not interchangeable".into());
    }
    let mut warnings = vec![json!(
        "No automatic winner; inspect mathematical outcomes, worker hashes and complete accounting."
    )];
    if runs
        .iter()
        .any(|r| r["schema_version"] != runs[0]["schema_version"])
    {
        warnings.push(json!(
            "Arms use different result schemas: 0.1 records carry no evidence and report infeasibility as claim none."
        ));
    }
    Ok(json!({"arms":runs,"warnings":warnings}))
}
