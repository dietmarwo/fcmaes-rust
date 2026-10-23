//! Shared test support for the optional application workers.
use fcmaes_job_worker::{
    Adapter,
    registry::{Model, Registry},
};
use optimization_job_api::{JobSpec, worker::Worker};
use serde_json::{Value, json};
use std::path::Path;

pub fn resolve(r: &Registry, id: &str, parameters: Value) -> Result<Model, String> {
    let p = r.probes().remove(id).expect("model must be discoverable");
    let mut problem = json!({"id":id,"version":"1","config":{"dimension":p.dimension},
        "bounds":{"lower":p.lower,"upper":p.upper,"broadcast":false}});
    problem["parameters"] = parameters;
    let spec: JobSpec = serde_json::from_value(json!({
        "schema_version":"optimization.job/0.1","request_key":"model-contract","worker":"application",
        "payload_schema":"fcmaes.evaluate/1","payload":{"problem":problem,"points":[]},
        "resources":{"cpu_slots":1,"outer_workers":1,"inner_threads":1},
        "limits":{"wall_time_seconds":30.0,"grace_seconds":1.0},"inputs":[]
    })).unwrap();
    r.resolve(
        serde_json::from_value(problem).unwrap(),
        &spec,
        Path::new("."),
    )
    .map(|(_, m)| m)
    .map_err(|e| e.to_string())
}

pub fn values(m: Model, x: &[f64]) -> Vec<f64> {
    match m {
        Model::Scalar(s) => vec![(s.evaluate)(x)],
        Model::MultiObjective(m) => m.values(x).unwrap(),
        Model::QualityDiversity(m) => {
            let (q, d) = m.values(x).unwrap();
            std::iter::once(q).chain(d).collect()
        }
    }
}

pub fn check_discovery(r: Registry, expected: usize) {
    let a = Adapter::new(r);
    let d = a.describe();
    assert_eq!(d["problems"].as_array().unwrap().len(), expected);
    assert!(!d["examples"].as_array().unwrap().is_empty());
    for p in d["problems"].as_array().unwrap() {
        assert!(p.get("parameters_schema").is_some(), "{p}");
    }
}
