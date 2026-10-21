//! `Scalar::simulation` models outside the surrogate profile: retries rank a
//! failed evaluation as `+inf` and count it, a cancel stops the simulation at
//! its checkpoints, and `fcmaes.evaluate/1` reports failures per point.
use fcmaes_job_worker::{
    Adapter,
    registry::{Registry, Scalar},
};
use optimization_job_api::{
    JobSpec,
    worker::{Worker, execute},
};
use serde_json::{Value, json};
use std::{
    fs,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

/// A shifted sphere, minimum 0 at (1, ..., 1), whose simulation fails for
/// `x[0] < 0`, on half of the box [-5, 5]^n; and one that waits at its
/// checkpoints until it is cancelled.
fn registry(started: Arc<AtomicUsize>) -> Registry {
    let mut registry = Registry::new();
    registry
        .scalar("app.holey-sphere", "1", json!({"simulation": true}), |_| {
            Ok(Scalar::simulation(|x, ctx| {
                ctx.checkpoint()?;
                assert_eq!(ctx.threads(), 1);
                if x[0] < 0.0 {
                    return Err(format!("diverged at x[0] = {}", x[0]));
                }
                Ok(x.iter().map(|v| (v - 1.0) * (v - 1.0)).sum())
            }))
        })
        .scalar("app.slow", "1", json!({"simulation": true}), move |_| {
            let started = Arc::clone(&started);
            Ok(Scalar::simulation(move |_, ctx| {
                started.fetch_add(1, Ordering::SeqCst);
                loop {
                    ctx.checkpoint()?;
                    std::thread::sleep(Duration::from_millis(1));
                }
            }))
        })
        .scalar("app.plain", "1", json!({}), |_| {
            Ok(Scalar::new(|x: &[f64]| x[0]))
        });
    registry
}

fn problem(id: &str, dim: usize) -> Value {
    json!({"id": id, "version": "1", "config": {"dimension": dim},
        "bounds": {"lower": [-5.0], "upper": [5.0], "broadcast": true}})
}

fn spec(profile: &str, payload: Value, inner: usize) -> JobSpec {
    serde_json::from_value(json!({
        "schema_version": "optimization.job/0.1", "request_key": "simulation-test", "worker": "app",
        "payload_schema": profile, "payload": payload,
        "resources": {"cpu_slots": 2 * inner, "outer_workers": 2, "inner_threads": inner},
        "limits": {"wall_time_seconds": 120.0, "grace_seconds": 1.0}, "inputs": []
    }))
    .unwrap()
}

fn retry(id: &str, kind: &str, parameters: Value, evaluations: u64) -> Value {
    let mode = if kind == "biteopt" { "hard" } else { "soft" };
    json!({"problem": problem(id, 4), "optimizer": {"kind": kind, "parameters": parameters},
        "seed": 7, "retries": 2, "evaluations_per_retry": evaluations, "evaluation_limit_mode": mode})
}

fn run(adapter: &Adapter, mut s: JobSpec) -> Value {
    let dir = tempfile::tempdir().unwrap();
    let prepared = adapter.prepare(&mut s, dir.path()).unwrap();
    s.task_count = Some(prepared.tasks());
    drop(prepared);
    fs::write(
        dir.path().join("resolved.json"),
        serde_json::to_vec(&s).unwrap(),
    )
    .unwrap();
    execute(adapter, dir.path()).unwrap();
    serde_json::from_slice(&fs::read(dir.path().join("run.json")).unwrap()).unwrap()
}

#[test]
fn retries_find_the_optimum_of_a_simulation_that_fails_on_half_the_box() {
    let adapter = Adapter::new(registry(Arc::default()));
    for (kind, parameters, inner) in [
        ("biteopt", json!({}), 1),
        ("biteopt", json!({"batch": 8}), 2),
        ("de", json!({}), 1),
        ("cmaes", json!({}), 2),
        ("cmaes", json!({"restarts": "bipop"}), 1),
        ("crfmnes", json!({}), 1),
        ("pgpe", json!({}), 1),
    ] {
        let mut payload = retry("app.holey-sphere", kind, parameters.clone(), 6000);
        if kind == "pgpe" {
            // A population that fails entirely sees a flat +inf plateau:
            // CMA-ES widens its step there, PGPE's mean only drifts. Start
            // PGPE where evaluations succeed.
            payload["guess"] = json!([2.0, 2.0, 2.0, 2.0]);
        }
        let run = run(&adapter, spec("fcmaes.retry/1", payload, inner));
        assert_eq!(run["status"], "completed");
        let mut failed = 0;
        for r in run["results"].as_array().unwrap() {
            let o = &r["outcome"];
            assert_eq!(o["claim"], "best_found", "{kind} {parameters}: {o}");
            assert_eq!(o["evidence"]["level"], "rechecked", "{o}");
            let best = o["objective"].as_f64().unwrap();
            // PGPE's diagonal distribution converges more slowly.
            let target = if kind == "pgpe" { 1e-2 } else { 1e-6 };
            assert!(best < target, "{kind} {parameters}: {best}");
            let x: Vec<f64> = serde_json::from_value(o["x"].clone()).unwrap();
            assert!(x[0] >= 0.0, "{x:?}");
            let d = &o["details"];
            assert_eq!(d["failed_evaluations"], r["work"]["failed_evaluations"]);
            assert_eq!(d["cancelled_evaluations"], 0);
            let count = d["failed_evaluations"].as_u64().unwrap();
            assert!(count > 0, "{kind}: {d}");
            // Up to three distinct messages with their counts.
            let messages = d["failures"].as_object().unwrap();
            assert!(!messages.is_empty() && messages.len() <= 3, "{messages:?}");
            assert!(
                messages
                    .keys()
                    .all(|m| m.starts_with("diverged at x[0] = -"))
            );
            assert!(messages.values().map(|c| c.as_u64().unwrap()).sum::<u64>() <= count);
            assert_eq!(r["work"]["evaluations"], r["work"]["evaluation_attempts"]);
            failed += count;
        }
        assert_eq!(run["work"]["failed_evaluations"], failed);
    }
}

#[test]
fn simulation_results_do_not_depend_on_inner_threads() {
    let adapter = Adapter::new(registry(Arc::default()));
    let outcomes = |inner: usize| -> Vec<Value> {
        let run = run(
            &adapter,
            spec(
                "fcmaes.retry/1",
                retry("app.holey-sphere", "cmaes", json!({}), 2000),
                inner,
            ),
        );
        let mut results = run["results"].as_array().unwrap().clone();
        results.sort_by_key(|r| r["task_id"].as_u64());
        results
            .into_iter()
            .map(|r| json!([r["outcome"], r["work"]]))
            .collect()
    };
    assert_eq!(outcomes(1), outcomes(3));
}

#[test]
fn a_cancel_stops_a_retry_at_the_simulations_checkpoints() {
    for kind in ["biteopt", "cmaes"] {
        let started = Arc::new(AtomicUsize::new(0));
        let adapter = Adapter::new(registry(Arc::clone(&started)));
        let dir = tempfile::tempdir().unwrap();
        // Uncancelled, these evaluations would never finish.
        let mut s = spec(
            "fcmaes.retry/1",
            retry("app.slow", kind, json!({}), 20_000),
            1,
        );
        let prepared = adapter.prepare(&mut s, dir.path()).unwrap();
        let began = Instant::now();
        let result = std::thread::scope(|scope| {
            let task = scope.spawn(|| prepared.solve(0, 60.0));
            while started.load(Ordering::SeqCst) == 0 && began.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert!(started.load(Ordering::SeqCst) > 0);
            prepared.cancel();
            task.join().unwrap().unwrap()
        });
        assert!(began.elapsed() < Duration::from_secs(20), "{kind}");
        let o = &result.outcome;
        o.check().unwrap();
        assert_eq!(o.termination, "cancelled", "{kind}");
        // Every evaluation was cancelled: no value, so no claim.
        assert_eq!(o.claim, "none");
        assert_eq!(result.work["failed_evaluations"], 0);
        assert!(result.work["cancelled_evaluations"] > 0);
        assert_eq!(result.work["validation_evaluations"], 0);
        assert_eq!(
            result.work["evaluations"] + result.work["cancelled_evaluations"],
            result.work["evaluation_attempts"]
        );
    }
}

#[test]
fn a_retry_cancelled_after_good_points_keeps_an_unchecked_claim() {
    // Evaluations succeed until the tenth, which waits for the cancel.
    let calls = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&calls);
    let mut registry = Registry::new();
    registry.scalar("app.stalls", "1", json!({"simulation": true}), move |_| {
        let count = Arc::clone(&count);
        Ok(Scalar::simulation(move |x, ctx| {
            if count.fetch_add(1, Ordering::SeqCst) >= 9 {
                loop {
                    ctx.checkpoint()?;
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            Ok(x.iter().map(|v| v * v).sum())
        }))
    });
    let adapter = Adapter::new(registry);
    let dir = tempfile::tempdir().unwrap();
    let mut s = spec(
        "fcmaes.retry/1",
        retry("app.stalls", "biteopt", json!({}), 1000),
        1,
    );
    let prepared = adapter.prepare(&mut s, dir.path()).unwrap();
    let result = std::thread::scope(|scope| {
        let task = scope.spawn(|| prepared.solve(0, 60.0));
        while calls.load(Ordering::SeqCst) < 10 {
            std::thread::sleep(Duration::from_millis(1));
        }
        prepared.cancel();
        task.join().unwrap().unwrap()
    });
    let o = &result.outcome;
    o.check().unwrap();
    assert_eq!(o.termination, "cancelled");
    assert_eq!(o.claim, "best_found");
    let evidence = serde_json::to_value(&o.evidence).unwrap();
    assert_eq!(evidence["level"], "claimed");
    assert!(evidence.get("refuted").is_none());
    assert_eq!(o.details["recheck"], "skipped: cancelled");
    assert_eq!(result.work["evaluations"], 9);
    assert_eq!(result.work["evaluation_attempts"], 1000);
}

#[test]
fn evaluate_reports_failed_simulations_per_point() {
    let adapter = Adapter::new(registry(Arc::default()));
    let mut s = spec(
        "fcmaes.evaluate/1",
        json!({"problem": problem("app.holey-sphere", 2), "points": [[1.0, 1.0], [-1.0, 1.0]]}),
        1,
    );
    let run = run(&adapter, s.clone());
    assert_eq!(run["status"], "completed");
    let (good, bad) = (&run["results"][0]["outcome"], &run["results"][1]["outcome"]);
    assert_eq!(good["claim"], "feasible");
    assert_eq!(good["objective"], 0.0);
    assert_eq!(bad["claim"], "none");
    assert_eq!(bad["feasible"], false);
    assert_eq!(bad["termination"], "evaluated");
    assert_eq!(bad["details"]["error"], "diverged at x[0] = -1");
    assert_eq!(bad["evidence"]["level"], "claimed");
    assert_eq!(run["work"]["evaluations"], 2);
    assert_eq!(run["work"]["failed_evaluations"], 1);
    assert_eq!(run["work"]["cancelled_evaluations"], 0);

    // A run seed is accepted for simulation models only.
    s.payload["run_seed"] = json!(12345);
    let dir = tempfile::tempdir().unwrap();
    adapter.prepare(&mut s.clone(), dir.path()).unwrap();
    s.payload["problem"] = problem("app.plain", 2);
    let err = adapter
        .prepare(&mut s, dir.path())
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("run_seed applies only"), "{err}");
}

#[test]
fn evaluate_uses_the_run_seed_of_a_retry() {
    // A seeded simulation: the value depends on SimContext::seed, so only
    // the retry's run seed reproduces the retry's value.
    let mut registry = Registry::new();
    registry.scalar("app.seeded", "1", json!({"simulation": true}), |_| {
        Ok(Scalar::simulation(|x, ctx| {
            let mut rng = fcmaes_core::Rng::new(ctx.seed());
            Ok(x.iter().map(|v| v * v).sum::<f64>() + 0.01 * rng.uniform01())
        }))
    });
    let adapter = Adapter::new(registry);
    let retried = run(
        &adapter,
        spec(
            "fcmaes.retry/1",
            retry("app.seeded", "cmaes", json!({}), 500),
            1,
        ),
    );
    let o = &retried["results"][0]["outcome"];
    assert_eq!(o["evidence"]["level"], "rechecked", "{o}");
    let evaluate = |run_seed: Value| {
        let mut payload = json!({"problem": problem("app.seeded", 4), "points": [o["x"]]});
        if !run_seed.is_null() {
            payload["run_seed"] = run_seed;
        }
        run(&adapter, spec("fcmaes.evaluate/1", payload, 1))["results"][0]["outcome"]["objective"]
            .clone()
    };
    assert_eq!(evaluate(o["details"]["run_seed"].clone()), o["objective"]);
    assert_ne!(evaluate(Value::Null), o["objective"]);
}

#[test]
fn describe_lists_every_scalar_profile_for_simulation_models() {
    let d = Adapter::new(registry(Arc::default())).describe();
    let entry = d["problems"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == "app.holey-sphere")
        .unwrap()
        .clone();
    assert_eq!(entry["profile"], "fcmaes.surrogate/1");
    assert_eq!(
        entry["profiles"],
        json!(["fcmaes.retry/1", "fcmaes.surrogate/1", "fcmaes.evaluate/1"])
    );
    // A worker with scalar models only advertises the scalar profiles.
    assert_eq!(
        d["profiles"],
        json!(["fcmaes.retry/1", "fcmaes.evaluate/1", "fcmaes.surrogate/1"])
    );
}
