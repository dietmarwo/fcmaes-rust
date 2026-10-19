//! Black-box checks of the `fcmaes.retry/1` adapter: validation, resolved
//! defaults, budgets, and per-retry agreement with the shared native solver.
use fcmaes_core::{RetryBounds, retry_run_seed};
use fcmaes_native_worker::{objective_for, solve};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

const WORKER: &str = env!("CARGO_BIN_EXE_fcmaes-job-worker");

fn job(optimizer: &str, problem: &str, dim: usize) -> Value {
    let (population, mode) = if optimizer == "de" {
        (json!(12), "soft")
    } else {
        (Value::Null, "hard")
    };
    let bounds = if problem == "gtop.cassini1" {
        json!({"lower": [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
               "upper": [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0], "broadcast": false})
    } else {
        json!({"lower": [-5.0], "upper": [5.0], "broadcast": true})
    };
    json!({
        "schema_version": "optimization.job/0.1",
        "request_key": "worker-test",
        "worker": "fcmaes",
        "payload_schema": "fcmaes.retry/1",
        "payload": {
            "problem": {"id": problem, "version": "1", "config": {"dimension": dim}, "bounds": bounds},
            "optimizer": {"kind": optimizer, "parameters": {"population": population}},
            "seed": 11, "retries": 5, "evaluations_per_retry": 997,
            "evaluation_limit_mode": mode
        },
        "resources": {"cpu_slots": 3, "outer_workers": 3, "inner_threads": 1},
        "limits": {"wall_time_seconds": 120.0, "grace_seconds": 1.0},
        "inputs": []
    })
}

fn worker(args: &[&str]) -> std::process::Output {
    Command::new(WORKER).args(args).output().unwrap()
}

/// `validate` output for `spec`, or the worker's error message.
fn validate(dir: &Path, spec: &Value) -> Result<Value, String> {
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
fn execute(spec: &Value) -> Value {
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

#[test]
fn describe_reports_profile_and_workspace_engine_version() {
    let out = worker(&["describe"]);
    assert!(out.status.success());
    let d: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(d["protocol"], "optimization.worker/0.1");
    assert_eq!(d["profiles"], json!(["fcmaes.retry/1"]));
    assert_eq!(d["resources"]["inner_threads_max"], 1);
    // fcmaes-core and this worker both inherit [workspace.package] version.
    let manifest = include_str!("../../../Cargo.toml");
    let workspace = &manifest[manifest.find("[workspace.package]").unwrap()..];
    let version = workspace
        .lines()
        .find_map(|l| l.strip_prefix("version = "))
        .unwrap()
        .trim_matches('"');
    assert_eq!(d["build"]["engine_version"], version);
}

#[test]
fn validation_resolves_defaults_and_rejects_unsupported_requests() {
    let dir = tempfile::tempdir().unwrap();
    let resolved = validate(dir.path(), &job("biteopt", "benchmark.sphere", 4)).unwrap();
    assert_eq!(resolved["task_count"], 5);
    assert_eq!(
        resolved["payload"]["problem"]["bounds"]["lower"],
        json!(vec![-5.0; 4])
    );
    assert_eq!(
        resolved["payload"]["optimizer"]["parameters"]["population"],
        9 + 3 * 4
    );

    let mut threads = job("biteopt", "benchmark.sphere", 4);
    threads["resources"] = json!({"cpu_slots": 4, "outer_workers": 2, "inner_threads": 2});
    assert!(
        validate(dir.path(), &threads)
            .unwrap_err()
            .contains("inner_threads=1")
    );

    let mut hard_de = job("de", "benchmark.sphere", 4);
    hard_de["payload"]["evaluation_limit_mode"] = json!("hard");
    assert!(validate(dir.path(), &hard_de).is_err());

    let mut unknown = job("biteopt", "benchmark.sphere", 4);
    unknown["payload"]["stop_fitness"] = json!(0.0);
    assert!(validate(dir.path(), &unknown).is_err());

    let mut cassini = job("biteopt", "gtop.cassini1", 6);
    cassini["payload"]["problem"]["bounds"] =
        json!({"lower": [-1e6], "upper": [1e6], "broadcast": true});
    assert!(validate(dir.path(), &cassini).is_err());
}

#[test]
fn retries_match_the_shared_native_solver_and_honour_budgets() {
    for (problem, dim) in [("benchmark.rosenbrock", 6), ("gtop.cassini1", 6)] {
        for optimizer in ["biteopt", "de"] {
            let spec = job(optimizer, problem, dim);
            let run = execute(&spec);
            assert_eq!(run["status"], "completed");
            assert_eq!(run["runner"]["accounting_complete"], true);
            let results = run["results"].as_array().unwrap();
            assert_eq!(results.len(), 5);

            // Re-solve each retry in-process with the legacy definitions the
            // native worker uses, from the same retry seed.
            let dir = tempfile::tempdir().unwrap();
            let resolved = validate(dir.path(), &spec).unwrap();
            let legacy: fcmaes_job_api::JobSpec = serde_json::from_value(json!({
                "schema_version": "fcmaes.job/0.1", "request_key": "worker-test",
                "problem": resolved["payload"]["problem"],
                "optimizer": resolved["payload"]["optimizer"],
                "execution": {"strategy": "independent_retry", "seed": 11, "workers": 1,
                              "retry_wave_size": 1},
                "budget": {"retries": 5, "evaluations_per_retry": 997,
                           "evaluation_limit_mode": resolved["payload"]["evaluation_limit_mode"],
                           "wall_time_seconds": 120.0},
                "outputs": {"progress": "completed_retries"}
            }))
            .unwrap();
            let legacy = legacy.resolve().unwrap();
            let bounds = RetryBounds::new(
                legacy.problem.bounds.lower.clone(),
                legacy.problem.bounds.upper.clone(),
            )
            .unwrap();
            let objective = objective_for(&legacy);
            let mut total = 0;
            for r in results {
                let task = r["task_id"].as_u64().unwrap() as usize;
                let seed = retry_run_seed(11, task);
                let native = solve(&legacy, &bounds, seed, &objective);
                assert_eq!(r["outcome"]["details"]["run_seed"], seed);
                assert_eq!(
                    r["outcome"]["objective"],
                    json!(native.y),
                    "{problem}/{optimizer}"
                );
                assert_eq!(r["outcome"]["x"], json!(native.x));
                assert_eq!(r["work"]["evaluations"], native.evaluations);
                let evaluations = native.evaluations;
                if optimizer == "biteopt" {
                    assert_eq!(evaluations, 997);
                } else {
                    assert!((997..=997 + 2 * 12 - 1).contains(&evaluations));
                }
                total += evaluations;
            }
            assert_eq!(run["work"]["evaluations"], total);
        }
    }
}
