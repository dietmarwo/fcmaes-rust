//! `describe`'s discovery keys: `payload_schemas`, `input_schemas` and runnable
//! `examples`, generated from the registry for the stock worker, the example
//! application workers and workers with only models of their own. Python
//! validates the payloads against the schemas (`tutorials/job-api/run.py`).
#[path = "../examples/custom_worker.rs"]
#[allow(dead_code)]
mod custom_worker;
#[path = "../examples/simulation_worker.rs"]
#[allow(dead_code)]
mod simulation_worker;

use fcmaes_job_worker::{
    Adapter,
    registry::{Example, MultiObjective, QualityDiversity, Registry, Scalar},
};
use optimization_job_api::{
    JobSpec,
    storage::digest,
    worker::{Worker, execute},
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs};

/// `t,y` rows of `y = 2t + 1`.
const LINE: &str = "t,y\n0,1\n1,3\n2,5\n3,7\n4,9\n";

/// Models of an application only: a scalar fit to a data input, a
/// multi-objective model without a reference point of its own and a
/// quality-diversity model with three descriptors (a CVT archive).
fn own_models() -> Registry {
    let mut registry = Registry::new();
    registry
        .scalar(
            "app.line-fit",
            "2",
            json!({"inputs":{"data":"CSV t,y"}}),
            |m| {
                m.require_dimension(2..=2)?;
                let power: i32 = m.parameter("power", 2)?;
                let rows: Vec<(f64, f64)> = std::str::from_utf8(&m.input("data")?)?
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
        .example(
            "app.line-fit",
            Example::new(2).input_text("data", "line.csv", LINE),
        )
        .multi_objective("app.pair", "1", json!({"objectives":2}), |_| {
            Ok(MultiObjective::new(2, 0, |x: &[f64]| {
                vec![x[0] * x[0], (x[0] - 1.0) * (x[0] - 1.0)]
            })
            .domain(vec![-2.0], vec![2.0]))
        })
        .example(
            "app.pair",
            Example::new(1).reference_point(vec![10.0, 10.0]),
        )
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

/// A scalar model that needs a parameter the probe cannot guess.
fn scalar_only() -> Registry {
    let mut registry = Registry::new();
    registry
        .scalar("app.needs-scale", "1", json!({}), |m| {
            let scale: f64 = m.required("scale")?;
            Ok(Scalar::new(move |x: &[f64]| scale * x[0] * x[0]).domain(vec![-1.0], vec![1.0]))
        })
        .scalar("app.shifted", "1", json!({}), |m| {
            let shift: f64 = m.parameter("shift", 1.0)?;
            Ok(Scalar::new(move |x: &[f64]| {
                x.iter().map(|v| (v - shift) * (v - shift)).sum()
            }))
        });
    registry
}

fn workers() -> Vec<(&'static str, Adapter)> {
    vec![
        ("fcmaes", Adapter::default()),
        ("curve-fit", Adapter::new(custom_worker::registry())),
        ("simulation", Adapter::new(simulation_worker::registry())),
        ("own-models", Adapter::new(own_models())),
        ("scalar-only", Adapter::new(scalar_only())),
    ]
}

/// The job a client builds from `example`: the envelope fields, and each
/// input written to its file in `dir` with its real SHA-256.
fn job(worker: &str, index: usize, example: &Value, dir: &std::path::Path) -> JobSpec {
    let inputs: Vec<Value> = example["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            let text = i["content_text"].as_str().unwrap();
            let file = i["file"].as_str().unwrap();
            fs::write(dir.join(file), text).unwrap();
            json!({"name": i["name"], "file": file, "sha256": digest(text.as_bytes())})
        })
        .collect();
    serde_json::from_value(json!({
        "schema_version": "optimization.job/0.1",
        "request_key": format!("example-{index}"),
        "worker": worker,
        "payload_schema": example["profile"],
        "payload": example["payload"],
        "resources": example["resources"],
        "limits": example["limits"],
        "inputs": inputs
    }))
    .unwrap()
}

#[test]
fn every_example_is_a_job_that_validates_and_runs() {
    for (worker, adapter) in workers() {
        let d = adapter.describe();
        for (index, example) in d["examples"].as_array().unwrap().iter().enumerate() {
            let dir = tempfile::tempdir().unwrap();
            let mut spec = job(worker, index, example, dir.path());
            spec.validate().unwrap();
            let prepared = adapter
                .prepare(&mut spec, dir.path())
                .unwrap_or_else(|e| panic!("{worker} {example}: {e}"));
            assert!(prepared.tasks() >= 1);
            spec.task_count = Some(prepared.tasks());
            drop(prepared);
            spec.validate().unwrap();
            fs::write(
                dir.path().join("resolved.json"),
                serde_json::to_vec(&spec).unwrap(),
            )
            .unwrap();
            execute(&adapter, dir.path()).unwrap();
            let run: Value =
                serde_json::from_slice(&fs::read(dir.path().join("run.json")).unwrap()).unwrap();
            assert_eq!(run["status"], "completed", "{worker} {example}");
            let results = run["results"].as_array().unwrap();
            assert!(!results.is_empty(), "{worker} {example}");
            for r in results {
                assert_ne!(r["outcome"]["claim"], "none", "{worker} {example}: {r}");
            }
        }
    }
}

#[test]
fn every_advertised_profile_has_a_schema_and_an_example() {
    for (worker, adapter) in workers() {
        let d = adapter.describe();
        let profiles: BTreeSet<&str> = d["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap())
            .collect();
        let schemas = d["payload_schemas"].as_object().unwrap();
        assert_eq!(
            schemas.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            profiles,
            "{worker}"
        );
        for (profile, schema) in schemas {
            assert_eq!(
                schema["$schema"],
                "https://json-schema.org/draft/2020-12/schema"
            );
            assert_eq!(
                schema["$id"],
                format!("urn:optimization-jobs:payload:{profile}")
            );
            assert_eq!(schema["additionalProperties"], false);
        }
        let examples = d["examples"].as_array().unwrap();
        let covered: BTreeSet<&str> = examples
            .iter()
            .map(|e| e["profile"].as_str().unwrap())
            .collect();
        assert_eq!(covered, profiles, "{worker}");
        let formats = d.get("input_schemas").and_then(Value::as_object);
        // Only the surrogate profile reads an input of a fixed format.
        assert_eq!(
            formats.is_some(),
            profiles.contains("fcmaes.surrogate/1"),
            "{worker}"
        );
        for example in examples {
            assert_eq!(
                example["resources"],
                json!({"cpu_slots": 1, "outer_workers": 1, "inner_threads": 1})
            );
            for input in example["inputs"].as_array().unwrap() {
                assert!(input["content_text"].is_string() || input["content"].is_object());
                if let Some(format) = input.get("format") {
                    assert!(formats.unwrap().contains_key(format.as_str().unwrap()));
                }
            }
        }
        // The pinned runner reads at most 64 KiB of describe output.
        let size = d.to_string().len();
        assert!(size < 48 * 1024, "{worker}: {size} bytes");
    }
}

#[test]
fn examples_cover_models_with_inputs_and_profiles_follow_the_models() {
    let d = Adapter::new(custom_worker::registry()).describe();
    let fit: Vec<&Value> = d["examples"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["payload"]["problem"]["id"] == "app.exponential-fit")
        .collect();
    assert_eq!(fit.len(), 1);
    assert_eq!(fit[0]["profile"], "fcmaes.retry/1");
    assert_eq!(fit[0]["inputs"][0]["name"], "data");
    assert_eq!(
        fit[0]["inputs"][0]["content_text"],
        include_str!("../../../tutorials/job-api/decay.csv")
    );
    assert_eq!(
        fit[0]["payload"]["problem"]["parameters"],
        json!({"loss": "squared"})
    );
    let entry = d["problems"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == "app.exponential-fit")
        .unwrap();
    assert_eq!(
        entry["parameters_schema"],
        json!({"type": "object", "additionalProperties": false,
            "properties": {"loss": {"type": "string", "default": "squared"}}})
    );

    // A worker with models of its own only: one example per profile they
    // support, the data model's included.
    let d = Adapter::new(own_models()).describe();
    let examples = d["examples"].as_array().unwrap();
    let on = |profile: &str| -> Vec<&str> {
        examples
            .iter()
            .filter(|e| e["profile"] == profile)
            .map(|e| e["payload"]["problem"]["id"].as_str().unwrap())
            .collect()
    };
    assert_eq!(on("fcmaes.retry/1"), ["app.line-fit", "app.line-fit"]);
    assert_eq!(on("fcmaes.surrogate/1"), ["app.line-fit"]);
    assert_eq!(on("fcmaes.mode/1"), ["app.pair"]);
    assert_eq!(on("fcmaes.map-elites/1"), ["app.qd3"]);
    let mode = examples
        .iter()
        .find(|e| e["profile"] == "fcmaes.mode/1")
        .unwrap();
    assert_eq!(mode["payload"]["reference_point"], json!([10.0, 10.0]));

    // Scalar models only: the multi-objective and quality-diversity profiles
    // are not offered, and prepare rejects them. A model that needs a
    // parameter describe cannot guess has no example and no parameters_schema.
    let adapter = Adapter::new(scalar_only());
    let d = adapter.describe();
    assert_eq!(
        d["profiles"],
        json!(["fcmaes.retry/1", "fcmaes.evaluate/1", "fcmaes.surrogate/1"])
    );
    assert!(
        d["examples"]
            .as_array()
            .unwrap()
            .iter()
            .all(|e| e["payload"]["problem"]["id"] == "app.shifted")
    );
    let entries = d["problems"].as_array().unwrap();
    assert!(
        entries[0].get("parameters_schema").is_none(),
        "{}",
        entries[0]
    );
    assert_eq!(
        entries[1]["parameters_schema"]["properties"]["shift"],
        json!({"type": "number", "default": 1.0})
    );
    let mut spec: JobSpec = serde_json::from_value(json!({
        "schema_version": "optimization.job/0.1", "request_key": "k", "worker": "w",
        "payload_schema": "fcmaes.mode/1", "payload": {},
        "resources": {"cpu_slots": 1, "outer_workers": 1, "inner_threads": 1},
        "limits": {"wall_time_seconds": 10.0, "grace_seconds": 1.0}, "inputs": []
    }))
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let err = adapter
        .prepare(&mut spec, dir.path())
        .err()
        .unwrap()
        .to_string();
    assert!(
        err.contains("this worker accepts fcmaes.retry/1, fcmaes.evaluate/1, fcmaes.surrogate/1"),
        "{err}"
    );
}
