//! `fcmaes.retry/1` with a `guess`: every retry starts there, for the
//! distribution optimizers and for BiteOpt and DE, which take an initial
//! point in fcmaes-core too; a guess of the wrong length or outside the
//! bounds is rejected.
use fcmaes_job_worker::Adapter;
use optimization_job_api::{
    JobSpec,
    worker::{Prepared, Worker, execute},
};
use serde_json::{Value, json};
use std::fs;

/// Sphere on a box whose centre (3, ..., 3) is not the optimum (the origin).
fn payload(kind: &str, parameters: Value, guess: Option<Vec<f64>>, evaluations: u64) -> Value {
    let mode = if kind == "biteopt" { "hard" } else { "soft" };
    let mut p = json!({"problem": {"id": "benchmark.sphere", "version": "1", "config": {"dimension": 8},
            "bounds": {"lower": [-2.0], "upper": [8.0], "broadcast": true}},
        "optimizer": {"kind": kind, "parameters": parameters},
        "seed": 3, "retries": 3, "evaluations_per_retry": evaluations, "evaluation_limit_mode": mode});
    if let Some(guess) = guess {
        p["guess"] = json!(guess);
    }
    p
}

fn spec(payload: Value) -> JobSpec {
    serde_json::from_value(json!({
        "schema_version": "optimization.job/0.1", "request_key": "guess-test", "worker": "fcmaes",
        "payload_schema": "fcmaes.retry/1", "payload": payload,
        "resources": {"cpu_slots": 1, "outer_workers": 1, "inner_threads": 1},
        "limits": {"wall_time_seconds": 120.0, "grace_seconds": 1.0}, "inputs": []
    }))
    .unwrap()
}

/// The resolved job, or the worker's validation error.
fn resolve(payload: Value) -> Result<JobSpec, String> {
    let mut s = spec(payload);
    let dir = tempfile::tempdir().unwrap();
    let prepared: Box<dyn Prepared> = Adapter::default()
        .prepare(&mut s, dir.path())
        .map_err(|e| e.to_string())?;
    s.task_count = Some(prepared.tasks());
    Ok(s)
}

/// The best objective of every retry.
fn bests(payload: Value) -> Vec<f64> {
    let dir = tempfile::tempdir().unwrap();
    let resolved = resolve(payload).unwrap();
    fs::write(
        dir.path().join("resolved.json"),
        serde_json::to_vec(&resolved).unwrap(),
    )
    .unwrap();
    execute(&Adapter::default(), dir.path()).unwrap();
    let run: Value =
        serde_json::from_slice(&fs::read(dir.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(run["status"], "completed");
    run["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            assert_eq!(r["outcome"]["claim"], "best_found", "{r}");
            r["outcome"]["objective"].as_f64().unwrap()
        })
        .collect()
}

#[test]
fn a_guess_at_the_optimum_converges_within_a_small_budget() {
    let origin = vec![0.0; 8];
    for (kind, parameters) in [
        ("cmaes", json!({"sigma": 0.02})),
        ("crfmnes", json!({"sigma": 0.02})),
        ("pgpe", json!({"sigma": 0.02})),
        ("de", json!({"sigma": 0.02})),
        ("biteopt", json!({})),
    ] {
        let guided = bests(payload(kind, parameters.clone(), Some(origin.clone()), 300));
        let mut random_parameters = parameters.clone();
        if kind == "de" {
            // DE takes sigma only with a guess.
            random_parameters = json!({});
        }
        let random = bests(payload(kind, random_parameters, None, 300));
        // The sphere value of the box centre, a typical random start, is 72.
        for (g, r) in guided.iter().zip(&random) {
            assert!(*g < 0.05, "{kind}: {g}");
            assert!(10.0 * g < *r, "{kind}: guided {g} random {r}");
        }
    }
}

#[test]
fn restarts_begin_at_random_points_after_a_guessed_first_run() {
    let p = payload(
        "cmaes",
        json!({"population": 8, "sigma": 0.02, "restarts": "ipop"}),
        Some(vec![0.0; 8]),
        4000,
    );
    let resolved = resolve(p.clone()).unwrap();
    assert_eq!(resolved.payload["guess"], json!(vec![0.0; 8]));
    for best in bests(p) {
        assert!(best < 1e-10, "{best}");
    }
}

#[test]
fn guesses_are_checked_and_de_resolves_its_spread() {
    let check = |payload: Value, message: &str| {
        let err = resolve(payload).unwrap_err();
        assert!(err.contains(message), "{message}: {err}");
    };
    for kind in ["cmaes", "crfmnes", "pgpe", "biteopt", "de"] {
        check(
            payload(kind, json!({}), Some(vec![0.0; 7]), 100),
            "guess has 7 entries; the problem has dimension 8",
        );
        let mut outside = vec![0.0; 8];
        outside[5] = 9.0;
        check(
            payload(kind, json!({}), Some(outside), 100),
            "guess[5] = 9 lies outside the bounds [-2, 8]",
        );
    }
    check(
        payload("de", json!({"sigma": 0.1}), None, 100),
        "sigma applies to de only with a guess",
    );
    check(
        payload("biteopt", json!({"sigma": 0.1}), Some(vec![0.0; 8]), 100),
        "sigma does not apply to biteopt",
    );
    check(
        payload("de", json!({"sigma": 2.5}), Some(vec![0.0; 8]), 100),
        "de sigma",
    );
    let resolved = resolve(payload("de", json!({}), Some(vec![0.0; 8]), 100)).unwrap();
    assert_eq!(
        resolved.payload["optimizer"]["parameters"],
        json!({"population": 31, "sigma": 0.3})
    );
    // Without a guess, BiteOpt and DE payloads, and so their digests, are
    // unchanged: no guess and no sigma appear.
    let resolved = resolve(payload("de", json!({}), None, 100)).unwrap();
    assert_eq!(
        resolved.payload["optimizer"]["parameters"],
        json!({"population": 31})
    );
    assert!(resolved.payload.get("guess").is_none());
}
