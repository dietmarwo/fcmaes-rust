use optimization_job_api::{
    result::{self, Outcome, Record},
    storage, *,
};
use serde_json::json;

fn job() -> JobSpec {
    serde_json::from_value(json!({"schema_version":SCHEMA,"request_key":"test","worker":"test","payload_schema":"test/1","payload":{},"resources":{"cpu_slots":8,"outer_workers":2,"inner_threads":4},"limits":{"wall_time_seconds":2.0,"grace_seconds":0.1},"inputs":[],"task_count":2})).unwrap()
}
#[test]
fn resource_and_schema_boundaries() {
    let s = job();
    s.validate().unwrap();
    for r in [
        Resources {
            cpu_slots: 8,
            outer_workers: 8,
            inner_threads: 8,
        },
        Resources {
            cpu_slots: 8,
            outer_workers: 0,
            inner_threads: 1,
        },
        Resources {
            cpu_slots: 8,
            outer_workers: usize::MAX,
            inner_threads: 2,
        },
    ] {
        let mut bad = s.clone();
        bad.resources = r;
        assert!(bad.validate().is_err());
    }
    let mut bad = serde_json::to_value(&s).unwrap();
    bad["unexpected"] = json!(true);
    assert!(Request::parse(&serde_json::to_vec(&bad).unwrap()).is_err());
    bad.as_object_mut().unwrap().remove("unexpected");
    bad["schema_version"] = json!("optimization.job/99");
    assert!(Request::parse(&serde_json::to_vec(&bad).unwrap()).is_err());
}
#[test]
fn identity_ignores_request_key_but_includes_budget_and_profile() {
    let s = job();
    let mut other = s.clone();
    other.request_key = "other".into();
    assert_eq!(s.digest().unwrap(), other.digest().unwrap());
    other.resources.inner_threads = 1;
    assert_ne!(s.digest().unwrap(), other.digest().unwrap());
    other = s.clone();
    other.payload_schema = "other/1".into();
    assert_ne!(s.digest().unwrap(), other.digest().unwrap());
}
#[test]
fn artifact_staging_hashes_and_path_containment() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let mut s = job();
    std::fs::write(a.path().join("model"), b"model data").unwrap();
    s.inputs.push(Input {
        name: "matrix".into(),
        file: "model".into(),
        sha256: storage::digest(b"model data"),
    });
    s.validate().unwrap();
    stage_inputs(&mut s, a.path(), b.path()).unwrap();
    assert_eq!(s.input_bytes(b.path(), "matrix").unwrap(), b"model data");
    std::fs::write(b.path().join("inputs/matrix"), b"changed").unwrap();
    assert!(s.input_bytes(b.path(), "matrix").is_err());
    s.inputs[0].file = "../escape".into();
    assert!(s.validate().is_err());
    #[cfg(unix)]
    {
        s.inputs[0].file = "link".into();
        std::os::unix::fs::symlink(a.path().join("model"), b.path().join("link")).unwrap();
        assert!(s.input_bytes(b.path(), "matrix").is_err());
    }
}
#[test]
fn journal_claims_duplicates_tail_recovery_and_incomplete_accounting() {
    let d = tempfile::tempdir().unwrap();
    let s = job();
    let r = Record {
        task_id: 0,
        spec_digest: s.digest().unwrap(),
        elapsed_seconds: 0.1,
        work: [("iterations".into(), 3)].into(),
        outcome: Outcome {
            termination: "infeasible".into(),
            claim: "none".into(),
            feasible: None,
            objective: None,
            x: None,
            details: json!({}),
        },
    };
    let path = d.path().join("results.jsonl");
    storage::append_checked(&path, 1, &r).unwrap();
    let partial = result::manifest(
        d.path(),
        &Request::Job(s.clone()),
        "interrupted",
        false,
        0.1,
        "worker",
    )
    .unwrap();
    assert!(partial["work"].is_null());
    assert_eq!(partial["known_work_lower_bound"]["iterations"], 3);
    assert_eq!(
        partial["results"][0]["outcome"]["termination"],
        "infeasible"
    );
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"{broken")
        .unwrap();
    assert_eq!(result::replay(d.path(), &s).unwrap().len(), 1);
    result::repair(d.path(), &Request::Job(s.clone())).unwrap();
    storage::append_checked(&path, 1, &r).unwrap();
    assert!(result::replay(d.path(), &s).is_err());
}
#[test]
fn false_success_and_mixed_claim_comparison_are_rejected() {
    let d = tempfile::tempdir().unwrap();
    let s = job();
    let fake = json!({"status":"completed","elapsed_seconds":0.0,"results":[],"work":{},"runner":{"spec_digest":s.digest().unwrap(),"accounting_complete":true}});
    storage::write_atomic_json(&d.path().join("run.json"), &fake).unwrap();
    assert!(result::worker_outcome(d.path(), &Request::Job(s)).is_err());
    assert!(
        result::compare(&[
            json!({"schema_version":"optimization.result/0.1","payload_schema":"lp-ip.solve/1"}),
            json!({"schema_version":"optimization.result/0.1","payload_schema":"sqp.solve/1"})
        ])
        .is_err()
    );
}

#[test]
fn numerical_json_round_trips_preserve_manifest_and_digest() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = job();
    s.task_count = Some(1);
    // These values include a real near-zero fcmaes result and values across
    // many exponents. A lossy parser can otherwise reject its own manifest.
    let values = vec![
        3.1554436208840472e-30,
        1.8517740610680664e-22,
        9.833351633135928e-23,
        6.999999999999999,
        0.00087198,
        1.7976931348623157e308,
        2.2250738585072014e-308,
    ];
    s.payload = json!({"values":values});
    let bytes = serde_json::to_vec(&s).unwrap();
    let copy: JobSpec = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(s.digest().unwrap(), copy.digest().unwrap());
    let record = Record {
        task_id: 0,
        spec_digest: s.digest().unwrap(),
        elapsed_seconds: 0.001060704,
        work: [("evaluations".into(), 1000)].into(),
        outcome: Outcome {
            termination: "evaluation_limit".into(),
            claim: "best_found".into(),
            feasible: Some(true),
            objective: Some(3.1554436208840472e-30),
            x: Some(values),
            details: json!({}),
        },
    };
    storage::append_checked(&dir.path().join("results.jsonl"), 1, &record).unwrap();
    let request = Request::Job(s);
    let manifest =
        result::manifest(dir.path(), &request, "completed", true, 0.025021547, "test").unwrap();
    storage::write_atomic_json(&dir.path().join("run.json"), &manifest).unwrap();
    assert_eq!(
        result::worker_outcome(dir.path(), &request)
            .unwrap()
            .unwrap()
            .0,
        "completed"
    );
}
