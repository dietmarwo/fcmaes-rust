use optimization_job_api::{
    result::{self, Certificate, Evidence, ObjectiveBounds, Outcome, Record},
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
    // "./model" is relative and contained; "." and "./" name no file.
    s.inputs[0].file = "./model".into();
    s.validate().unwrap();
    for bad in [".", "./", "/model", "a/../model"] {
        s.inputs[0].file = bad.into();
        assert!(s.validate().is_err(), "{bad}");
    }
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
            ..Outcome::default()
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
            json!({"schema_version":"optimization.result/0.2","payload_schema":"lp-ip.solve/1"}),
            json!({"schema_version":"optimization.result/0.2","payload_schema":"sqp.solve/1"})
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
            ..Outcome::default()
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

#[test]
fn set_and_archive_claims_follow_their_rules() {
    let s = job();
    let record = |claim: &str, feasible: Option<bool>, x: Option<Vec<f64>>, objective| Record {
        task_id: 0,
        spec_digest: s.digest().unwrap(),
        elapsed_seconds: 0.1,
        work: [("evaluations".into(), 10)].into(),
        outcome: Outcome {
            termination: "evaluation_limit".into(),
            claim: claim.into(),
            feasible,
            objective,
            x,
            details: json!({"front": [{"x": [0.5], "objectives": [1.0, 2.0]}]}),
            ..Outcome::default()
        },
    };
    let replay = |r: Record| {
        let d = tempfile::tempdir().unwrap();
        storage::append_checked(&d.path().join("results.jsonl"), 1, &r).unwrap();
        result::replay(d.path(), &s).map(|records| records.len())
    };
    // A non-dominated set lives in details: no single point or objective.
    assert_eq!(
        replay(record("non_dominated_set", Some(true), None, None)).unwrap(),
        1
    );
    assert!(replay(record("non_dominated_set", Some(false), None, None)).is_err());
    assert!(
        replay(record(
            "non_dominated_set",
            Some(true),
            Some(vec![0.5]),
            Some(1.0)
        ))
        .is_err()
    );
    // An elite archive names its best elite, like every other point claim.
    assert_eq!(
        replay(record(
            "elite_archive",
            Some(true),
            Some(vec![0.5]),
            Some(1.0)
        ))
        .unwrap(),
        1
    );
    assert!(replay(record("elite_archive", Some(true), None, None)).is_err());
    assert!(replay(record("best_found", Some(true), None, None)).is_err());
    assert!(replay(record("pareto_optimal", Some(true), None, None)).is_err());
}

fn point(
    claim: &str,
    feasible: Option<bool>,
    x: Option<Vec<f64>>,
    objective: Option<f64>,
) -> Outcome {
    Outcome {
        termination: "test".into(),
        claim: claim.into(),
        feasible,
        objective,
        x,
        details: json!({}),
        ..Outcome::default()
    }
}

#[test]
fn claims_of_result_schema_0_2_follow_their_rules() {
    let x = || Some(vec![1.0]);
    let ok = |o: Outcome| o.check().map_err(|e| format!("{o:?}: {e}")).unwrap();
    let bad = |o: Outcome| assert!(o.check().is_err(), "{o:?}");
    // No feasible point: never marked feasible; an infeasible point may stay.
    for claim in ["infeasible", "locally_infeasible"] {
        ok(point(claim, None, None, None));
        ok(point(claim, Some(false), x(), Some(3.0)));
        bad(point(claim, Some(true), x(), Some(3.0)));
    }
    // A feasible point narrows infeasible_or_unbounded down to unbounded.
    ok(point("infeasible_or_unbounded", Some(true), x(), Some(3.0)));
    // Unbounded may carry a feasible point, or none.
    ok(point("unbounded", Some(true), x(), Some(-5.0)));
    ok(point("unbounded", None, None, None));
    // A feasible point of a multi-objective or constraint-only model has no
    // single objective; the objective claims need one.
    ok(point("feasible", Some(true), x(), None));
    bad(point("best_found", Some(true), x(), None));
    bad(point("feasible", Some(false), x(), Some(1.0)));
    bad(point("none", None, None, Some(1.0)));
    // certified_optimum needs a verified certificate.
    let mut o = point("certified_optimum", Some(true), x(), Some(1.0));
    bad(o.clone());
    let certificate = Certificate {
        format: "vipr".into(),
        artifact: Some("proof.vipr".into()),
        sha256: "a".repeat(64),
        checker: "viprchk-1.1".into(),
        verdict: "verified".into(),
    };
    o.evidence = Some(Evidence {
        level: "certified".into(),
        certificate: Some(certificate.clone()),
        ..Evidence::default()
    });
    ok(o.clone());
    o.evidence
        .as_mut()
        .unwrap()
        .certificate
        .as_mut()
        .unwrap()
        .verdict = "rejected".into();
    bad(o.clone());
    o.evidence
        .as_mut()
        .unwrap()
        .certificate
        .as_mut()
        .unwrap()
        .sha256 = "A".repeat(64);
    bad(o);
    // Whole-problem claims are never "rechecked".
    let mut o = point("infeasible", None, None, None);
    o.evidence = Some(Evidence::rechecked(&["constraints"]));
    bad(o.clone());
    o.evidence = Some(Evidence::claimed(&[]));
    ok(o);
    // Evidence names known, distinct checks; rechecked names at least one.
    let mut o = point("best_found", Some(true), x(), Some(1.0));
    for evidence in [
        Evidence::rechecked(&[]),
        Evidence::rechecked(&["objective", "objective"]),
        Evidence::rechecked(&["optimality"]),
        Evidence {
            level: "verified".into(),
            ..Evidence::default()
        },
    ] {
        o.evidence = Some(evidence);
        bad(o.clone());
    }
    o.evidence = Some(Evidence::rechecked(&["objective", "bounds"]));
    ok(o.clone());
    // A refutation names another claim and the verdict of a failed check;
    // the surviving claim is what the checks support.
    let mut refuted = point("feasible", Some(true), x(), Some(1.0));
    refuted.evidence = Some(Evidence::rechecked(&["objective", "kkt"]).refuting(
        "first_order",
        "false_optimality",
        "stationarity 1e-2 above 1e-6",
    ));
    ok(refuted.clone());
    for (claim, verdict) in [
        ("feasible", "false_optimality"),
        ("first_order", "wrong"),
        ("none", "false_optimality"),
        ("first_order", "invalid_certificate"),
    ] {
        let mut o = refuted.clone();
        let r = o.evidence.as_mut().unwrap().refuted.as_mut().unwrap();
        r.claim = claim.into();
        r.verdict = verdict.into();
        bad(o);
    }
    let mut o = refuted.clone();
    o.evidence.as_mut().unwrap().level = "claimed".into();
    bad(o);
    // invalid_certificate goes with a rejected certificate, and only then.
    let mut o = point("numerical_optimum", Some(true), x(), Some(1.0));
    o.evidence = Some(
        Evidence {
            certificate: Some(Certificate {
                verdict: "rejected".into(),
                ..certificate
            }),
            ..Evidence::rechecked(&["objective"])
        }
        .refuting(
            "certified_optimum",
            "invalid_certificate",
            "checker rejected step 12",
        ),
    );
    ok(o);
    // imprecise qualifies a claim.
    let mut o = point("numerical_optimum", Some(true), x(), Some(1.0));
    o.imprecise = true;
    ok(o.clone());
    o.claim = "none".into();
    bad(o);
    // Objective bounds have a sense and finite bounds or null.
    let mut o = point("numerical_optimum", Some(true), x(), Some(7.0));
    o.objective_bounds = Some(ObjectiveBounds {
        sense: "minimize".into(),
        primal: Some(7.0),
        dual: None,
    });
    ok(o.clone());
    o.objective_bounds.as_mut().unwrap().sense = "min".into();
    bad(o);
}

#[test]
fn records_without_0_2_fields_replay_and_serialize_unchanged() {
    // A 0.1 record: no evidence, bounds or imprecise flag in the JSON.
    let s = job();
    let text = format!(
        r#"{{"task_id":0,"spec_digest":"{}","elapsed_seconds":0.5,"work":{{"evaluations":3}},"outcome":{{"termination":"evaluation_limit","claim":"best_found","feasible":true,"objective":1.5,"x":[0.5],"details":{{}}}}}}"#,
        s.digest().unwrap()
    );
    let record: Record = serde_json::from_str(&text).unwrap();
    assert_eq!(record.outcome.evidence, None);
    assert_eq!(serde_json::to_string(&record).unwrap(), text);
    let d = tempfile::tempdir().unwrap();
    storage::append_checked(&d.path().join("results.jsonl"), 1, &record).unwrap();
    assert_eq!(result::replay(d.path(), &s).unwrap().len(), 1);
    // An invalid record names its task and the rule it breaks.
    let mut wrong = record;
    wrong.task_id = 1;
    wrong.outcome.claim = "infeasible".into();
    storage::append_checked(&d.path().join("results.jsonl"), 1, &wrong).unwrap();
    let error = result::replay(d.path(), &s).unwrap_err().to_string();
    assert!(
        error.contains("task 1") && error.contains("contradicts"),
        "{error}"
    );
}

#[test]
fn compare_reads_0_1_and_0_2_manifests_of_one_spec() {
    let arm = |version: &str| json!({"schema_version":version,"payload_schema":"lp-ip.solve/1","runner":{"spec_digest":"d"}});
    let same = result::compare(&[
        arm("optimization.result/0.2"),
        arm("optimization.result/0.2"),
    ])
    .unwrap();
    assert_eq!(same["warnings"].as_array().unwrap().len(), 1);
    let mixed = result::compare(&[
        arm("optimization.result/0.1"),
        arm("optimization.result/0.2"),
    ])
    .unwrap();
    assert_eq!(mixed["warnings"].as_array().unwrap().len(), 2);
    assert!(
        result::compare(&[arm("optimization.result/0.2"), json!({"schema_version":1})]).is_err()
    );
}

#[test]
fn published_json_schema_lists_the_same_vocabulary() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/job-api/schemas/optimization.result-0.2.schema.json");
    let schema: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let names = |pointer: &str| -> Vec<String> {
        let mut v: Vec<String> = schema
            .pointer(pointer)
            .unwrap_or_else(|| panic!("{pointer}"))
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        v.sort();
        v
    };
    let sorted = |a: &[&str]| -> Vec<String> {
        let mut v: Vec<String> = a.iter().map(|s| s.to_string()).collect();
        v.sort();
        v
    };
    assert_eq!(names("/$defs/claim/enum"), sorted(&result::CLAIMS));
    assert_eq!(names("/$defs/check/enum"), sorted(&result::CHECKS));
    assert_eq!(names("/$defs/verdict/enum"), sorted(&result::VERDICTS));
    assert_eq!(
        names("/$defs/evidence/properties/level/enum"),
        sorted(&result::LEVELS)
    );
    assert_eq!(
        schema["properties"]["schema_version"]["enum"],
        json!(result::SCHEMAS)
    );
}

#[test]
fn error_codes_and_worker_protocols() {
    let e = coded("already_terminal", "job ended");
    assert_eq!(error_code(&e), "already_terminal");
    assert_eq!(e.to_string(), "job ended");
    assert_eq!(error_code(&"plain".into()), "runner_error");
    assert!(retryable("attempt_active") && !retryable("already_terminal"));
    assert!(protocol_supported(&json!("optimization.worker/0.1")));
    assert!(protocol_supported(&json!(PROTOCOL)));
    assert!(!protocol_supported(&json!("optimization.worker/0.3")));
    let bad = Request::parse(b"{\"schema_version\":\"x\"}").unwrap_err();
    assert_eq!(error_code(&bad), "invalid_job");
}

/// The same cases check the JSON Schema (scripts/job-api/test_review.py).
#[test]
fn shared_outcome_cases_match_the_rules() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/job-api/outcome_cases.json");
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(cases.len() > 30);
    for case in cases {
        let checked = serde_json::from_value::<Outcome>(case["outcome"].clone())
            .map_err(|e| e.to_string())
            .and_then(|o| o.check());
        assert_eq!(
            checked.is_ok(),
            case["valid"].as_bool().unwrap(),
            "{}: {checked:?}",
            case["name"]
        );
    }
}
