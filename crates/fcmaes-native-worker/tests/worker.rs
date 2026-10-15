//! End-to-end checks of the native worker's budget, seed and journal contract.
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path, process::Command};

const WORKER: &str = env!("CARGO_BIN_EXE_fcmaes-native-worker");

fn spec(problem: &str, dim: usize, optimizer: &str, workers: usize, wave: usize) -> Value {
    let bounds = if problem == "gtop.cassini1" {
        json!({"lower": [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
               "upper": [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0], "broadcast": false})
    } else {
        json!({"lower": [-5.0], "upper": [5.0], "broadcast": true})
    };
    let (population, mode) = if optimizer == "de" {
        (json!(12), "soft")
    } else {
        (Value::Null, "hard")
    };
    json!({
        "schema_version": "fcmaes.job/0.1",
        "request_key": format!("{problem}-{optimizer}-{workers}-{wave}"),
        "problem": {"id": problem, "version": "1", "config": {"dimension": dim}, "bounds": bounds},
        "optimizer": {"kind": optimizer, "parameters": {"population": population}},
        "execution": {"strategy": "independent_retry", "seed": 7, "workers": workers,
                      "retry_wave_size": wave},
        "budget": {"retries": 6, "evaluations_per_retry": 997, "evaluation_limit_mode": mode,
                   "wall_time_seconds": 300.0},
        "outputs": {"progress": "completed_retries"}
    })
}

/// Runs `execute` in a fresh job directory and returns run.json plus the
/// journal records keyed by retry ID.
fn execute(spec: &Value) -> (Value, BTreeMap<u64, Value>) {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("resolved.json"), spec.to_string()).unwrap();
    let out = Command::new(WORKER)
        .arg("execute")
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    (
        read(&dir.path().join("run.json")),
        journal(&dir.path().join("retries.jsonl")),
    )
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn journal(path: &Path) -> BTreeMap<u64, Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| {
            let envelope: Value = serde_json::from_str(line).unwrap();
            let mut record: Value =
                serde_json::from_str(envelope["payload_json"].as_str().unwrap()).unwrap();
            // Timing and the spec digest legitimately differ between configurations.
            let id = record["retry_id"].as_u64().unwrap();
            record.as_object_mut().unwrap().remove("elapsed_seconds");
            record.as_object_mut().unwrap().remove("spec_digest");
            (id, record)
        })
        .collect()
}

#[test]
fn describe_reports_the_capability_profile() {
    let out = Command::new(WORKER).arg("describe").output().unwrap();
    assert!(out.status.success());
    let d: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(d["protocol"], "fcmaes.worker/0.1");
    assert_eq!(d["optimizers"]["biteopt"]["limit"], "hard");
    assert_eq!(d["optimizers"]["de"]["limit"], "soft");
}

#[test]
fn biteopt_meets_the_hard_cap_exactly() {
    let (run, records) = execute(&spec("benchmark.rosenbrock", 8, "biteopt", 2, 2));
    assert_eq!(run["status"], "completed");
    assert_eq!(run["requested_evaluations"], 6 * 997);
    assert_eq!(run["actual_evaluations"], 6 * 997);
    assert_eq!(records.len(), 6);
    assert!(
        records
            .values()
            .all(|r| r["actual_evaluations"].as_u64() == Some(997))
    );
}

#[test]
fn de_stays_within_the_declared_soft_bound() {
    let (run, records) = execute(&spec("benchmark.rosenbrock", 8, "de", 2, 2));
    assert_eq!(run["status"], "completed");
    let maximum_per_retry = 997 + 2 * 12 - 1;
    for r in records.values() {
        let actual = r["actual_evaluations"].as_u64().unwrap();
        assert!(
            (997..=maximum_per_retry).contains(&actual),
            "retry used {actual} evaluations"
        );
    }
    let total: u64 = records
        .values()
        .map(|r| r["actual_evaluations"].as_u64().unwrap())
        .sum();
    assert_eq!(run["actual_evaluations"], total);
}

#[test]
fn retry_results_do_not_depend_on_workers_or_wave_size() {
    for (problem, dim) in [("benchmark.rosenbrock", 8), ("gtop.cassini1", 6)] {
        for optimizer in ["biteopt", "de"] {
            let reference = execute(&spec(problem, dim, optimizer, 1, 1)).1;
            for (workers, wave) in [(2, 2), (3, 4)] {
                let other = execute(&spec(problem, dim, optimizer, workers, wave)).1;
                assert_eq!(
                    reference, other,
                    "{problem}/{optimizer} differs at workers={workers}, wave={wave}"
                );
            }
            let distinct: std::collections::BTreeSet<String> =
                reference.values().map(|r| r["y"].to_string()).collect();
            assert!(
                distinct.len() > 1,
                "{problem}/{optimizer}: degenerate check"
            );
        }
    }
}

#[test]
fn an_attempt_is_never_restarted() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("resolved.json"),
        spec("benchmark.sphere", 4, "biteopt", 1, 1).to_string(),
    )
    .unwrap();
    let run = || {
        Command::new(WORKER)
            .arg("execute")
            .arg(dir.path())
            .output()
            .unwrap()
    };
    assert!(run().status.success());
    let second = run();
    assert!(!second.status.success());
    assert!(String::from_utf8_lossy(&second.stderr).contains("refuses to restart"));
}
