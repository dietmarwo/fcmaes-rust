use fcmaes_job_worker::{
    Adapter,
    registry::{Registry, Scalar},
};
use optimization_job_api::{JobSpec, worker::Worker};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[test]
fn plain_models_stop_on_cancel_and_deadline_without_rechecks() {
    for kind in ["biteopt", "de", "cmaes", "crfmnes", "pgpe"] {
        for batch in [None, Some(4)] {
            if batch.is_some() && kind != "biteopt" && kind != "de" {
                continue;
            }
            for cancelled in [true, false] {
                let calls = Arc::new(AtomicUsize::new(0));
                let count = Arc::clone(&calls);
                let mut registry = Registry::new();
                registry.scalar("app.sleep", "1", json!({}), move |_| {
                    let count = Arc::clone(&count);
                    Ok(Scalar::new(move |x: &[f64]| {
                        count.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(5));
                        x.iter().map(|v| v * v).sum()
                    }))
                });
                let adapter = Adapter::new(registry);
                let mut parameters = json!({"population":4});
                if let Some(batch) = batch {
                    parameters["batch"] = json!(batch);
                }
                let mut s:JobSpec=serde_json::from_value(json!({
                    "schema_version":"optimization.job/0.1","request_key":"stop-test","worker":"app",
                    "payload_schema":"fcmaes.retry/1","payload":{
                        "problem":{"id":"app.sleep","version":"1","config":{"dimension":3},
                            "bounds":{"lower":[-5.0],"upper":[5.0],"broadcast":true}},
                        "optimizer":{"kind":kind,"parameters":parameters},"seed":3,"retries":1,
                        "evaluations_per_retry":10000,"evaluation_limit_mode":if kind=="biteopt" {"hard"} else {"soft"}},
                    "resources":{"cpu_slots":1,"outer_workers":1,"inner_threads":1},
                    "limits":{"wall_time_seconds":60.0,"grace_seconds":1.0},"inputs":[]
                })).unwrap();
                let dir = tempfile::tempdir().unwrap();
                let prepared = adapter.prepare(&mut s, dir.path()).unwrap();
                let began = Instant::now();
                let result = std::thread::scope(|scope| {
                    let task =
                        scope.spawn(|| prepared.solve(0, if cancelled { 60.0 } else { 0.05 }));
                    if cancelled {
                        while calls.load(Ordering::SeqCst) < 3
                            && began.elapsed() < Duration::from_secs(5)
                        {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        prepared.cancel();
                    }
                    task.join().unwrap().unwrap()
                });
                assert!(began.elapsed() < Duration::from_secs(3), "{kind} {batch:?}");
                assert_eq!(result.outcome.termination, "cancelled");
                assert_eq!(
                    result.work["evaluations"],
                    calls.load(Ordering::SeqCst) as u64
                );
                assert!(result.work["evaluations"] < 100);
                assert_eq!(result.work["validation_evaluations"], 0);
                assert_eq!(result.work["cancelled_evaluations"], 0);
                let evidence: Value = serde_json::to_value(result.outcome.evidence).unwrap();
                assert_eq!(evidence["level"], "claimed");
                assert_eq!(result.outcome.details["recheck"], "skipped: cancelled");
            }
        }
    }
}
