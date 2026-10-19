//! Helpers shared by the integration tests that run the worker binary.
#![allow(dead_code)]
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

pub const WORKER: &str = env!("CARGO_BIN_EXE_fcmaes-job-worker");

pub fn worker(args: &[&str]) -> std::process::Output {
    Command::new(WORKER).args(args).output().unwrap()
}

/// `validate` output for `spec`, or the worker's error message.
pub fn validate(dir: &Path, spec: &Value) -> Result<Value, String> {
    let path = dir.join("spec.json");
    fs::write(&path, spec.to_string()).unwrap();
    let out = worker(&["validate", path.to_str().unwrap()]);
    if out.status.success() {
        Ok(serde_json::from_slice::<Value>(&out.stdout).unwrap()["resolved"].clone())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

/// Validates and executes `spec` in a fresh job directory; returns run.json.
pub fn execute(spec: &Value) -> Value {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), spec).unwrap();
    fs::write(dir.path().join("resolved.json"), resolved.to_string()).unwrap();
    let out = worker(&["execute", dir.path().to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&fs::read(dir.path().join("run.json")).unwrap()).unwrap()
}

/// A job envelope with `outer` workers of `inner` threads each.
pub fn envelope(profile: &str, payload: Value, outer: usize, inner: usize) -> Value {
    json!({
        "schema_version": "optimization.job/0.1",
        "request_key": "worker-test",
        "worker": "fcmaes",
        "payload_schema": profile,
        "payload": payload,
        "resources": {"cpu_slots": outer * inner, "outer_workers": outer, "inner_threads": inner},
        "limits": {"wall_time_seconds": 300.0, "grace_seconds": 1.0},
        "inputs": []
    })
}

/// A problem with broadcast bounds.
pub fn problem(id: &str, dim: usize, lower: f64, upper: f64) -> Value {
    json!({"id": id, "version": "1", "config": {"dimension": dim},
           "bounds": {"lower": [lower], "upper": [upper], "broadcast": true}})
}

/// Outcome and work of every task, ordered by task, without timings.
pub fn results(run: &Value) -> Vec<Value> {
    let mut results: Vec<Value> = run["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| json!({"task_id": r["task_id"], "outcome": r["outcome"], "work": r["work"]}))
        .collect();
    results.sort_by_key(|r| r["task_id"].as_u64());
    results
}
