use fcmaes_job_worker::{
    Adapter,
    surrogate2::{LogRecord, WarmFile},
};
use fcmaes_surrogate::advanced::{self, Event};
use optimization_job_api::{
    JobSpec, storage,
    worker::{self, Worker},
};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn job() -> JobSpec {
    serde_json::from_value(
        json!({"schema_version":"optimization.job/0.1","request_key":"surrogate2-test",
            "worker":"fcmaes","payload_schema":"fcmaes.surrogate/2",
            "resources":{"cpu_slots":4,"outer_workers":1,"inner_threads":4},
            "limits":{"wall_time_seconds":60.0,"grace_seconds":1.0},"inputs":[],
            "payload":{"problem":{"id":"benchmark.sphere","version":"1","config":{"dimension":2},
                "bounds":{"lower":[-2.0],"upper":[2.0],"broadcast":true}},
                "seed":9,"evaluations":12,"concurrent":2,
                "parameters":{"base":{"initial_points":4,"candidates":64,"fit_restarts":1,"fit_iterations":10},
                    "schedule":"asynchronous","training_cap":16,"archive_cap":100}}
        }),
    )
    .unwrap()
}

fn execute(adapter: &Adapter, mut spec: JobSpec, dir: &Path) -> Value {
    let prepared = adapter.prepare(&mut spec, dir).unwrap();
    spec.task_count = Some(prepared.tasks());
    spec.validate().unwrap();
    drop(prepared);
    fs::write(
        dir.join("resolved.json"),
        serde_json::to_vec(&spec).unwrap(),
    )
    .unwrap();
    worker::execute(adapter, dir).unwrap();
    serde_json::from_slice(&fs::read(dir.join("run.json")).unwrap()).unwrap()
}

fn warm_input(spec: &mut JobSpec, dir: &Path, bytes: &[u8]) {
    fs::write(dir.join("previous.json"), bytes).unwrap();
    spec.inputs = serde_json::from_value(
        json!([{"name":"previous","file":"previous.json","sha256":storage::digest(bytes)}]),
    )
    .unwrap();
    spec.payload["warm_start"] = json!({"input":"previous"});
}

fn rejection(adapter: &Adapter, mut spec: JobSpec, dir: &Path) -> String {
    match adapter.prepare(&mut spec, dir) {
        Ok(_) => panic!("accepted {}", spec.payload),
        Err(e) => e.to_string(),
    }
}

#[test]
fn asynchronous_job_is_rechecked_and_its_trace_replays() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::default();
    let manifest = execute(&adapter, job(), dir.path());
    let outcome = &manifest["results"][0]["outcome"];
    assert_eq!(outcome["claim"], "best_found");
    assert_eq!(outcome["evidence"]["level"], "rechecked");
    assert_eq!(outcome["details"]["schedule"], "asynchronous");
    assert_eq!(manifest["work"]["evaluations"], 12);
    assert_eq!(manifest["work"]["validation_evaluations"], 1);
    let records: Vec<LogRecord> =
        storage::read_checked(&dir.path().join("surrogate-0.jsonl"), 1).unwrap();
    let events: Vec<Event> = records
        .into_iter()
        .filter_map(|r| match r {
            LogRecord::Event { data } => Some(data),
            _ => None,
        })
        .collect();
    assert!(
        matches!(&events[0], Event::Start { schema, .. } if schema == "fcmaes.surrogate.events/2")
    );
    advanced::replay(&events).unwrap();
    let warm: WarmFile = serde_json::from_slice(
        &fs::read(dir.path().join("surrogate-observations-0.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(warm.schema_version, "fcmaes.surrogate.observations/2");
    assert_eq!(warm.observations.len(), 12);
}

#[test]
fn warm_starts_bind_the_model_and_the_snapshot_version() {
    let adapter = Adapter::default();
    let first = tempfile::tempdir().unwrap();
    execute(&adapter, job(), first.path());
    let bytes = fs::read(first.path().join("surrogate-observations-0.json")).unwrap();
    let next = tempfile::tempdir().unwrap();
    let mut spec = job();
    spec.payload["evaluations"] = json!(4);
    warm_input(&mut spec, next.path(), &bytes);
    let run = execute(&adapter, spec, next.path());
    assert_eq!(run["work"]["evaluations"], 4);
    assert_eq!(
        run["results"][0]["outcome"]["details"]["warm_start_points"],
        12
    );
    // Another model version, and a /1 snapshot, are refused.
    let mut file: Value = serde_json::from_slice(&bytes).unwrap();
    file["problem"]["version"] = json!("other");
    let other = tempfile::tempdir().unwrap();
    let mut spec = job();
    warm_input(&mut spec, other.path(), &serde_json::to_vec(&file).unwrap());
    assert!(adapter.prepare(&mut spec, other.path()).is_err());
    let mut file: Value = serde_json::from_slice(&bytes).unwrap();
    file["schema_version"] = json!("fcmaes.surrogate.observations/1");
    let v1 = tempfile::tempdir().unwrap();
    let mut spec = job();
    warm_input(&mut spec, v1.path(), &serde_json::to_vec(&file).unwrap());
    assert!(adapter.prepare(&mut spec, v1.path()).is_err());
}

#[test]
fn payload_limits_and_thread_fields_are_checked_with_the_v2_limits() {
    let adapter = Adapter::default();
    let dir = tempfile::tempdir().unwrap();
    let mut s = job();
    s.payload["evaluations"] = json!(10001);
    assert!(rejection(&adapter, s, dir.path()).contains("evaluations=1..10000"));
    // Asynchronous work reserves 2 * 1 + 1 = 3 threads; 2 are too few.
    let mut s = job();
    s.resources.inner_threads = 2;
    s.resources.cpu_slots = 2;
    assert!(rejection(&adapter, s, dir.path()).contains("surrogate_threads"));
    let mut s = job();
    s.payload["evaluations"] = json!(101);
    assert!(rejection(&adapter, s, dir.path()).contains("archive_cap (100)"));
    // A copy in `parameters` that differs from the request is a conflict.
    for (key, value) in [("concurrent", 1), ("evaluation_threads", 2)] {
        let mut s = job();
        s.payload["parameters"][key] = json!(value);
        assert!(rejection(&adapter, s, dir.path()).contains(&format!("parameters.{key}")));
    }
    // An equal copy, as in a resolved payload, is accepted, and a resolved
    // payload prepares again unchanged.
    let mut s = job();
    s.payload["parameters"]["concurrent"] = json!(2);
    adapter.prepare(&mut s, dir.path()).unwrap();
    let resolved = s.payload.clone();
    adapter.prepare(&mut s, dir.path()).unwrap();
    assert_eq!(s.payload, resolved);
}

#[test]
fn a_handoff_without_enough_remaining_calls_is_skipped_and_reported() {
    let adapter = Adapter::default();
    let dir = tempfile::tempdir().unwrap();
    let mut spec = job();
    spec.payload["parameters"]["handoff_fraction"] = json!(0.5);
    let run = execute(&adapter, spec, dir.path());
    let handoff = &run["results"][0]["outcome"]["details"]["handoff"];
    // 6 of 12 calls remain, fewer than 3 populations of 9 + 3 * 2.
    assert_eq!(
        handoff["plan"],
        json!({"after":6,"remaining":6,"required":45})
    );
    assert_eq!(handoff["after"], Value::Null);
    assert_eq!(run["work"]["evaluations"], 12);
    let mut both = job();
    both.payload["parameters"]["handoff_fraction"] = json!(0.5);
    both.payload["parameters"]["handoff_after"] = json!(6);
    assert!(adapter.prepare(&mut both, dir.path()).is_err());
}
