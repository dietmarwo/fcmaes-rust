use fcmaes_job_worker::{
    Adapter,
    registry::{Registry, Scalar},
    surrogate::{LogRecord, WarmFile},
};
use fcmaes_surrogate::Event;
use optimization_job_api::{
    JobSpec, storage,
    worker::{self, Worker},
};
use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

fn job() -> JobSpec {
    serde_json::from_value(
        json!({"schema_version":"optimization.job/0.1","request_key":"surrogate-test",
            "worker":"fcmaes","payload_schema":"fcmaes.surrogate/1",
            "resources":{"cpu_slots":4,"outer_workers":1,"inner_threads":4},
            "limits":{"wall_time_seconds":60.0,"grace_seconds":1.0},"inputs":[],
            "payload":{"problem":{"id":"benchmark.sphere","version":"1","config":{"dimension":2},
                "bounds":{"lower":[-2.0],"upper":[2.0],"broadcast":true}},
                "seed":9,"evaluations":12,"concurrent":2,
                "parameters":{"initial_points":4,"candidates":64,"fit_restarts":1,"fit_iterations":10}}
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

#[test]
fn profile_executes_through_sdk_and_logs_can_be_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::default();
    let manifest = execute(&adapter, job(), dir.path());
    assert_eq!(manifest["completed_tasks"], 1);
    assert_eq!(manifest["results"][0]["outcome"]["claim"], "best_found");
    assert_eq!(
        manifest["results"][0]["outcome"]["evidence"]["level"],
        "rechecked"
    );
    assert_eq!(manifest["work"]["evaluations"], 12);
    assert_eq!(manifest["work"]["validation_evaluations"], 1);
    let records: Vec<LogRecord> =
        storage::read_checked(&dir.path().join("surrogate-0.jsonl"), 1).unwrap();
    let events: Vec<Event> = records
        .into_iter()
        .filter_map(|r| {
            if let LogRecord::Event { data } = r {
                Some(data)
            } else {
                None
            }
        })
        .collect();
    fcmaes_surrogate::driver::replay(&events).unwrap();
    let mut bytes = fs::read(dir.path().join("surrogate-0.jsonl")).unwrap();
    bytes.extend_from_slice(b"{\"torn\"");
    fs::write(dir.path().join("torn.jsonl"), &bytes).unwrap();
    let _: Vec<LogRecord> = storage::read_checked(&dir.path().join("torn.jsonl"), 1).unwrap();
    bytes[20] = if bytes[20] == b'0' { b'1' } else { b'0' };
    fs::write(dir.path().join("corrupt.jsonl"), bytes).unwrap();
    assert!(storage::read_checked::<LogRecord>(&dir.path().join("corrupt.jsonl"), 1).is_err());
    let warm: WarmFile = serde_json::from_slice(
        &fs::read(dir.path().join("surrogate-observations-0.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(warm.observations.len(), 12);
    assert!(warm.usable_for_warm_start);
}

#[test]
fn warm_starts_bind_model_identity_and_count_only_new_work() {
    let adapter = Adapter::default();
    let first = tempfile::tempdir().unwrap();
    execute(&adapter, job(), first.path());
    let bytes = fs::read(first.path().join("surrogate-observations-0.json")).unwrap();
    let next = tempfile::tempdir().unwrap();
    let mut spec = job();
    spec.payload["evaluations"] = json!(4);
    warm_input(&mut spec, next.path(), &bytes);
    let run = execute(&adapter, spec.clone(), next.path());
    assert_eq!(run["work"]["evaluations"], 4);
    assert_eq!(
        run["results"][0]["outcome"]["details"]["warm_start_points"],
        12
    );
    let different = tempfile::tempdir().unwrap();
    let mut bad = spec.clone();
    let mut file: Value = serde_json::from_slice(&bytes).unwrap();
    file["problem"]["version"] = json!("other");
    warm_input(
        &mut bad,
        different.path(),
        &serde_json::to_vec(&file).unwrap(),
    );
    assert!(adapter.prepare(&mut bad, different.path()).is_err());
    fs::write(next.path().join("previous.json"), b"{}").unwrap();
    assert!(adapter.prepare(&mut spec, next.path()).is_err());
    let narrowed = tempfile::tempdir().unwrap();
    let mut spec = job();
    spec.payload["evaluations"] = json!(4);
    spec.payload["problem"]["bounds"] = json!({"lower":[-0.1],"upper":[0.1],"broadcast":true});
    warm_input(&mut spec, narrowed.path(), &bytes);
    let run = execute(&adapter, spec, narrowed.path());
    assert!(
        run["results"][0]["outcome"]["details"]["dropped_warm_points"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[test]
fn strict_payload_resources_profile_inputs_and_model_kind_are_validated() {
    let adapter = Adapter::default();
    let dir = tempfile::tempdir().unwrap();
    for (key, value) in [
        ("asynchronous", json!(true)),
        ("evaluations", json!(257)),
        ("concurrent", json!(8)),
        ("evaluation_threads", json!(3)),
        ("log_scale", json!([true])),
    ] {
        let mut s = job();
        s.payload[key] = value;
        assert!(adapter.prepare(&mut s, dir.path()).is_err(), "{key}");
    }
    let mut s = job();
    s.payload["problem"]["id"] = json!("benchmark.zdt1");
    assert!(adapter.prepare(&mut s, dir.path()).is_err());
    let mut s = job();
    s.payload["parameters"]["noise"] = json!("inferred");
    assert!(adapter.prepare(&mut s, dir.path()).is_err());
    let mut s = job();
    s.inputs = serde_json::from_value(
        json!([{"name":"unused","file":"unused.json","sha256":"a".repeat(64)}]),
    )
    .unwrap();
    assert!(adapter.prepare(&mut s, dir.path()).is_err());
    let mut s = job();
    s.payload["recheck_best"] = json!(false);
    let run = execute(&adapter, s, dir.path());
    assert_eq!(run["work"]["validation_evaluations"], 0);
    assert_eq!(run["results"][0]["outcome"]["evidence"]["level"], "claimed");
    let d = adapter.describe();
    assert!(
        d["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p == "fcmaes.surrogate/1")
    );
    let sphere = d["problems"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == "benchmark.sphere")
        .unwrap();
    assert!(
        sphere["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p == "fcmaes.surrogate/1")
    );
}

#[test]
fn objective_mismatch_is_refuted_and_cannot_be_warm_started() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let mut registry = Registry::default();
    registry.scalar("app.changing", "1", json!({}), move |_| {
        let counter = Arc::clone(&counter);
        Ok(Scalar::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst) as f64
        }))
    });
    let adapter = Adapter::new(registry);
    let dir = tempfile::tempdir().unwrap();
    let mut spec = job();
    spec.payload["problem"]["id"] = json!("app.changing");
    let run = execute(&adapter, spec, dir.path());
    let outcome = &run["results"][0]["outcome"];
    assert_eq!(outcome["claim"], "feasible");
    assert_eq!(
        outcome["evidence"]["refuted"]["verdict"],
        "objective_mismatch"
    );
    let warm: WarmFile = serde_json::from_slice(
        &fs::read(dir.path().join("surrogate-observations-0.json")).unwrap(),
    )
    .unwrap();
    assert!(!warm.usable_for_warm_start);
}

#[test]
fn simulation_context_is_seeded_bounded_and_cooperatively_cancellable() {
    let started = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&started);
    let mut registry = Registry::default();
    registry.scalar("app.sim", "1", json!({"simulation":true}), move |_| {
        let count = Arc::clone(&count);
        Ok(Scalar::simulation(move |_, ctx| {
            assert_eq!(ctx.threads(), 2);
            assert!(rayon::current_num_threads() <= 4);
            assert!(ctx.remaining_seconds() > 0.0);
            count.fetch_add(1, Ordering::SeqCst);
            for _ in 0..1000 {
                ctx.checkpoint()?;
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(ctx.seed() as f64)
        }))
    });
    let adapter = Adapter::new(registry);
    let dir = tempfile::tempdir().unwrap();
    let mut spec = job();
    spec.payload["problem"]["id"] = json!("app.sim");
    spec.payload["evaluation_threads"] = json!(2);
    let prepared = adapter.prepare(&mut spec, dir.path()).unwrap();
    let began = Instant::now();
    let result = std::thread::scope(|scope| {
        let thread = scope.spawn(|| prepared.solve(0, 30.0));
        while started.load(Ordering::SeqCst) == 0 && began.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(started.load(Ordering::SeqCst) > 0);
        prepared.cancel();
        thread.join().unwrap().unwrap()
    });
    assert!(began.elapsed() < Duration::from_secs(2));
    assert_eq!(result.work["failed_evaluations"], 0);
    assert_eq!(result.work["evaluations"], 0);
    assert!(result.work["cancelled_evaluations"] > 0);
    result.outcome.check().unwrap();
    let mut retry = job();
    retry.payload_schema = "fcmaes.retry/1".into();
    retry.payload = json!({"problem":spec.payload["problem"],"optimizer":{"kind":"biteopt","parameters":{}},"seed":1,"retries":1,"evaluations_per_retry":10,"evaluation_limit_mode":"hard"});
    assert!(adapter.prepare(&mut retry, dir.path()).is_err());
}

#[test]
fn seeded_simulation_recheck_uses_the_same_point_seed() {
    let mut registry = Registry::default();
    registry.scalar("app.seeded", "1", json!({"simulation":true}), |_| {
        Ok(Scalar::simulation(|x, ctx| {
            ctx.checkpoint()?;
            let mut rng = fcmaes_core::Rng::new(ctx.seed());
            Ok(x.iter().map(|x| x * x).sum::<f64>() + rng.uniform01() * 0.01)
        }))
    });
    let adapter = Adapter::new(registry);
    let dir = tempfile::tempdir().unwrap();
    let mut spec = job();
    spec.payload["problem"]["id"] = json!("app.seeded");
    let run = execute(&adapter, spec, dir.path());
    assert_eq!(
        run["results"][0]["outcome"]["evidence"]["level"],
        "rechecked"
    );
}

#[test]
fn simulation_snapshots_bind_the_seed_their_task_simulated_with() {
    let mut registry = Registry::default();
    registry.scalar("app.seeded", "1", json!({"simulation":true}), |_| {
        Ok(Scalar::simulation(|x, ctx| {
            let mut rng = fcmaes_core::Rng::new(ctx.seed());
            Ok(x.iter().map(|x| x * x).sum::<f64>() + rng.uniform01() * 0.01)
        }))
    });
    let adapter = Adapter::new(registry);
    let first = tempfile::tempdir().unwrap();
    let mut spec = job();
    spec.payload["problem"]["id"] = json!("app.seeded");
    spec.payload["runs"] = json!(2);
    spec.payload["evaluations"] = json!(6);
    execute(&adapter, spec.clone(), first.path());
    let snapshot = |task: usize| {
        fs::read(
            first
                .path()
                .join(format!("surrogate-observations-{task}.json")),
        )
        .unwrap()
    };
    for task in 0..2 {
        let warm: WarmFile = serde_json::from_slice(&snapshot(task)).unwrap();
        assert_eq!(
            warm.problem.simulation_seed,
            Some(fcmaes_core::retry_run_seed(9, task))
        );
    }
    // A single-run job with the same seed simulates with task 0's run seed, so
    // only task 0's observations are consistent with it.
    spec.payload["runs"] = json!(1);
    spec.payload["evaluations"] = json!(2);
    for (task, accepted) in [(0, true), (1, false)] {
        let dir = tempfile::tempdir().unwrap();
        let mut next = spec.clone();
        warm_input(&mut next, dir.path(), &snapshot(task));
        assert_eq!(
            adapter.prepare(&mut next, dir.path()).is_ok(),
            accepted,
            "task {task}"
        );
    }
}

#[test]
fn cancelling_validation_keeps_the_historical_claim_without_a_false_refutation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&calls);
    let mut registry = Registry::default();
    registry.scalar(
        "app.validation-cancel",
        "1",
        json!({"simulation":true}),
        move |_| {
            let count = Arc::clone(&count);
            Ok(Scalar::simulation(move |x, ctx| {
                if count.fetch_add(1, Ordering::SeqCst) >= 4 {
                    loop {
                        ctx.checkpoint()?;
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
                Ok(x.iter().map(|v| v * v).sum())
            }))
        },
    );
    let adapter = Adapter::new(registry);
    let dir = tempfile::tempdir().unwrap();
    let mut spec = job();
    spec.payload["problem"]["id"] = json!("app.validation-cancel");
    spec.payload["evaluations"] = json!(4);
    let prepared = adapter.prepare(&mut spec, dir.path()).unwrap();
    let began = Instant::now();
    let result = std::thread::scope(|scope| {
        let thread = scope.spawn(|| prepared.solve(0, 30.0));
        while calls.load(Ordering::SeqCst) < 5 && began.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 5);
        prepared.cancel();
        thread.join().unwrap().unwrap()
    });
    assert!(began.elapsed() < Duration::from_secs(2));
    assert_eq!(result.outcome.claim, "best_found");
    let evidence = serde_json::to_value(result.outcome.evidence).unwrap();
    assert_eq!(evidence["level"], "claimed");
    assert!(evidence.get("refuted").is_none());
    assert_eq!(result.work["validation_evaluations"], 1);
    assert_eq!(result.work["cancelled_validation_evaluations"], 1);
    let warm: WarmFile = serde_json::from_slice(
        &fs::read(dir.path().join("surrogate-observations-0.json")).unwrap(),
    )
    .unwrap();
    assert!(warm.usable_for_warm_start);
}
