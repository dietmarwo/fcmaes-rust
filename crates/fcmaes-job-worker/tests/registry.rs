//! Application models in a worker of their own: parameters, hashed inputs,
//! model kinds and their validation, run in-process through the worker SDK.
use fcmaes_job_worker::{
    Adapter,
    registry::{MultiObjective, QualityDiversity, Registry, Scalar},
};
use optimization_job_api::{
    JobSpec,
    storage::digest,
    worker::{Prepared, Worker, execute},
};
use serde_json::{Value, json};
use std::{fs, path::Path};

/// `y = 2t + 1` at t = 0..9.
const DATA: &str = "t,y\n0,1\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17\n9,19\n";

fn registry() -> Registry {
    let mut registry = Registry::benchmarks();
    registry
        .scalar(
            "app.line-fit",
            "2",
            json!({"dimensions":2,"inputs":["data"],"parameters":{"power":"default 2"}}),
            |m| {
                m.require_dimension(2..=2)?;
                let power: i32 = m.parameter("power", 2)?;
                let bytes = m.input("data")?;
                let rows: Vec<(f64, f64)> = std::str::from_utf8(&bytes)?
                    .lines()
                    .skip(1)
                    .map(|l| {
                        let (t, y) = l.split_once(',').unwrap();
                        (t.parse().unwrap(), y.parse().unwrap())
                    })
                    .collect();
                Ok(Scalar::new(move |x: &[f64]| {
                    rows.iter()
                        .map(|(t, y)| (x[0] * t + x[1] - y).abs().powi(power))
                        .sum()
                }))
            },
        )
        .scalar("app.needs-scale", "1", json!({}), |m| {
            let scale: f64 = m.required("scale")?;
            Ok(Scalar::new(move |x: &[f64]| scale * x[0] * x[0]).domain(vec![-1.0], vec![1.0]))
        })
        .multi_objective("app.pair", "1", json!({"objectives":2}), |_| {
            Ok(MultiObjective::new(2, 0, |x: &[f64]| {
                vec![x[0] * x[0], (x[0] - 1.0) * (x[0] - 1.0)]
            }))
        })
        .multi_objective("app.miscounted", "1", json!({}), |_| {
            Ok(
                MultiObjective::new(2, 0, |x: &[f64]| vec![x[0]; 3])
                    .reference_point(vec![2.0, 2.0]),
            )
        })
        .quality_diversity("app.qd3", "1", json!({"descriptors":3}), |m| {
            let n = m.dimension();
            Ok(QualityDiversity::new(
                vec![-1.0; 3],
                vec![1.0; 3],
                move |x: &[f64]| {
                    let f = x.iter().map(|v| v * v).sum();
                    (f, (0..3).map(|i| x[i % n].sin()).collect())
                },
            ))
        });
    registry
}

fn spec(profile: &str, payload: Value, inputs: Value) -> JobSpec {
    serde_json::from_value(json!({
        "schema_version": "optimization.job/0.1",
        "request_key": "registry-test",
        "worker": "app",
        "payload_schema": profile,
        "payload": payload,
        "resources": {"cpu_slots": 4, "outer_workers": 2, "inner_threads": 2},
        "limits": {"wall_time_seconds": 120.0, "grace_seconds": 1.0},
        "inputs": inputs
    }))
    .unwrap()
}

fn line_fit(parameters: Option<Value>) -> Value {
    let mut problem = json!({"id": "app.line-fit", "version": "2", "config": {"dimension": 2},
        "bounds": {"lower": [-10.0], "upper": [10.0], "broadcast": true}});
    if let Some(p) = parameters {
        problem["parameters"] = p;
    }
    json!({"problem": problem, "optimizer": {"kind": "cmaes", "parameters": {}},
        "seed": 1, "retries": 2, "evaluations_per_retry": 3000, "evaluation_limit_mode": "soft"})
}

/// A job directory with `DATA` staged as input `data`.
fn job_dir() -> (tempfile::TempDir, Value) {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("inputs")).unwrap();
    fs::write(dir.path().join("inputs/data.csv"), DATA).unwrap();
    let inputs =
        json!([{"name": "data", "file": "inputs/data.csv", "sha256": digest(DATA.as_bytes())}]);
    (dir, inputs)
}

/// What `validate` does: the resolved job with its task count.
fn resolve(adapter: &Adapter, mut s: JobSpec, dir: &Path) -> Result<JobSpec, String> {
    s.validate().map_err(|e| e.to_string())?;
    let prepared: Box<dyn Prepared> = adapter.prepare(&mut s, dir).map_err(|e| e.to_string())?;
    s.task_count = Some(prepared.tasks());
    Ok(s)
}

fn run(adapter: &Adapter, s: JobSpec, dir: &Path) -> Result<Value, String> {
    let resolved = resolve(adapter, s, dir)?;
    fs::write(
        dir.join("resolved.json"),
        serde_json::to_vec(&resolved).unwrap(),
    )
    .unwrap();
    execute(adapter, dir).map_err(|e| e.to_string())?;
    Ok(serde_json::from_slice(&fs::read(dir.join("run.json")).unwrap()).unwrap())
}

#[test]
fn describe_lists_application_models_with_kind_and_profile() {
    let d = Adapter::new(registry()).describe();
    let problems = d["problems"].as_array().unwrap();
    let find = |id: &str| problems.iter().find(|p| p["id"] == id).unwrap().clone();
    let fit = find("app.line-fit");
    assert_eq!(fit["version"], "2");
    assert_eq!(fit["kind"], "scalar");
    assert_eq!(fit["profile"], "fcmaes.retry/1");
    assert_eq!(fit["inputs"], json!(["data"]));
    assert_eq!(find("app.qd3")["profile"], "fcmaes.map-elites/1");
    assert_eq!(find("benchmark.zdt1")["profile"], "fcmaes.mode/1");
    assert_eq!(find("gtop.cassini1")["kind"], "scalar");
}

#[test]
fn a_model_reads_hashed_inputs_and_records_its_parameters() {
    let adapter = Adapter::new(registry());
    let (dir, inputs) = job_dir();
    let resolved = resolve(
        &adapter,
        spec("fcmaes.retry/1", line_fit(None), inputs.clone()),
        dir.path(),
    )
    .unwrap();
    assert_eq!(
        resolved.payload["problem"]["parameters"],
        json!({"power": 2})
    );
    let run = run(
        &adapter,
        spec(
            "fcmaes.retry/1",
            line_fit(Some(json!({"power": 2}))),
            inputs.clone(),
        ),
        dir.path(),
    )
    .unwrap();
    assert_eq!(run["status"], "completed");
    for r in run["results"].as_array().unwrap() {
        let o = &r["outcome"];
        assert_eq!(o["claim"], "best_found");
        let x: Vec<f64> = serde_json::from_value(o["x"].clone()).unwrap();
        assert!(
            (x[0] - 2.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6,
            "{x:?}"
        );
    }

    // Built-in models keep their payload shape: no parameters appear.
    let sphere = json!({"problem": {"id": "benchmark.sphere", "version": "1", "config": {"dimension": 3},
            "bounds": {"lower": [-1.0], "upper": [1.0], "broadcast": true}},
        "optimizer": {"kind": "biteopt", "parameters": {}}, "seed": 1, "retries": 1,
        "evaluations_per_retry": 100, "evaluation_limit_mode": "hard"});
    let mut s = spec("fcmaes.retry/1", sphere, json!([]));
    s.resources.inner_threads = 1;
    let resolved = resolve(&adapter, s, dir.path()).unwrap();
    assert!(resolved.payload["problem"].get("parameters").is_none());
}

#[test]
fn parameters_inputs_versions_and_kinds_are_checked() {
    let adapter = Adapter::new(registry());
    let (dir, inputs) = job_dir();
    let check = |s: JobSpec, message: &str| {
        let err = resolve(&adapter, s, dir.path()).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    };
    check(
        spec(
            "fcmaes.retry/1",
            line_fit(Some(json!({"pow": 2}))),
            inputs.clone(),
        ),
        "app.line-fit has no parameter pow",
    );
    check(
        spec(
            "fcmaes.retry/1",
            line_fit(Some(json!({"power": "two"}))),
            inputs.clone(),
        ),
        "parameter power of app.line-fit",
    );
    check(
        spec("fcmaes.retry/1", line_fit(None), json!([])),
        "requires the job input data",
    );
    let wrong = json!([{"name": "data", "file": "inputs/data.csv", "sha256": digest(b"other")}]);
    check(spec("fcmaes.retry/1", line_fit(None), wrong), "SHA-256");
    let mut extra = inputs.as_array().unwrap().clone();
    extra.push(json!({"name": "notes", "file": "inputs/data.csv",
        "sha256": digest(DATA.as_bytes())}));
    let needs_scale = json!({"problem": {"id": "app.needs-scale", "version": "1",
            "config": {"dimension": 1}, "bounds": {"lower": [-1.0], "upper": [1.0], "broadcast": false}},
        "points": [[0.5]]});
    let mut evaluate = spec("fcmaes.evaluate/1", needs_scale.clone(), json!(extra));
    evaluate.resources.inner_threads = 1;
    evaluate.payload["problem"]["parameters"] = json!({"scale": 2.0});
    check(
        evaluate.clone(),
        "input data is not used by app.needs-scale",
    );
    evaluate.inputs.clear();
    evaluate.payload["problem"]["parameters"] = json!({});
    check(evaluate.clone(), "requires parameter scale");
    evaluate.payload["problem"]["parameters"] = json!({"scale": 2.0});
    evaluate.payload["problem"]["bounds"] =
        json!({"lower": [-2.0], "upper": [1.0], "broadcast": false});
    check(evaluate.clone(), "bounds must lie inside [-1, 1]");
    let mut version = line_fit(None);
    version["problem"]["version"] = json!("1");
    check(
        spec("fcmaes.retry/1", version, inputs.clone()),
        "has version 2",
    );
    let mut mode = spec(
        "fcmaes.mode/1",
        json!({"problem": line_fit(None)["problem"], "optimizer": {"kind": "mode"},
            "seed": 1, "runs": 1, "generations": 5}),
        inputs.clone(),
    );
    check(
        mode.clone(),
        "app.line-fit is a scalar model; use fcmaes.retry/1",
    );
    mode.payload["problem"] = json!({"id": "app.pair", "version": "1", "config": {"dimension": 1},
        "bounds": {"lower": [0.0], "upper": [1.0], "broadcast": false}});
    mode.inputs.clear();
    check(mode.clone(), "defines no reference point");
    mode.payload["reference_point"] = json!([2.0, 2.0]);
    resolve(&adapter, mode.clone(), dir.path()).unwrap();

    // A model that breaks its own declaration fails the task, not silently.
    let run_dir = tempfile::tempdir().unwrap();
    mode.payload["problem"]["id"] = json!("app.miscounted");
    mode.payload["reference_point"] = Value::Null;
    let err = run(&adapter, mode, run_dir.path()).unwrap_err();
    assert!(err.contains("returned 3 values"), "{err}");
}

#[test]
fn multi_dimensional_descriptors_use_a_cvt_archive() {
    let adapter = Adapter::new(registry());
    let dir = tempfile::tempdir().unwrap();
    let payload = json!({"problem": {"id": "app.qd3", "version": "1", "config": {"dimension": 4},
            "bounds": {"lower": [-2.0], "upper": [2.0], "broadcast": true}},
        "archive": {"capacity": 30},
        "optimizer": {"kind": "map_elites", "parameters": {"cma_generations": 5}},
        "seed": 9, "runs": 2, "generations": 10});
    let resolved = resolve(
        &adapter,
        spec("fcmaes.map-elites/1", payload.clone(), json!([])),
        dir.path(),
    )
    .unwrap();
    assert_eq!(
        resolved.payload["archive"],
        json!({"capacity": 30, "kind": "cvt", "samples_per_niche": 20, "seed": 9})
    );
    let run = run(
        &adapter,
        spec("fcmaes.map-elites/1", payload.clone(), json!([])),
        dir.path(),
    )
    .unwrap();
    assert_eq!(run["status"], "completed");
    assert_eq!(run["work"]["evaluations"], 2 * (10 * 20 + 5 * 4 * 16));
    for r in run["results"].as_array().unwrap() {
        assert_eq!(r["outcome"]["claim"], "elite_archive");
        assert_eq!(r["outcome"]["details"]["archive_kind"], "cvt");
        for e in r["outcome"]["details"]["elites"].as_array().unwrap() {
            assert_eq!(e["descriptor"].as_array().unwrap().len(), 3);
        }
    }
    let mut grid = payload;
    grid["archive"]["kind"] = json!("grid");
    let err = resolve(
        &adapter,
        spec("fcmaes.map-elites/1", grid, json!([])),
        dir.path(),
    )
    .unwrap_err();
    assert!(err.contains("two-dimensional descriptors"), "{err}");
}

#[test]
#[should_panic(expected = "registered twice")]
fn a_model_identifier_is_registered_once() {
    let mut registry = Registry::benchmarks();
    registry.scalar("benchmark.sphere", "1", json!({}), |_| {
        Ok(Scalar::new(|x: &[f64]| x[0]))
    });
}
