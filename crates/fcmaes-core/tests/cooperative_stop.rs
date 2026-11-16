use fcmaes_core::{BiteOpt, BiteParams, De, DeParams, DeepBiteOpt, Fitness};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn stop_retains_exactly_the_evaluated_best() {
    for k in [0, 1, 7, 51, 200] {
        for engine in ["bite", "deep", "de"] {
            let calls = AtomicUsize::new(0);
            let best = Mutex::new(f64::INFINITY);
            let objective = |x: &[f64]| {
                let y = x.iter().map(|x| x * x).sum::<f64>();
                let mut saved = best.lock().unwrap();
                *saved = saved.min(y);
                calls.fetch_add(1, Ordering::SeqCst);
                y
            };
            let stop = || calls.load(Ordering::SeqCst) >= k;
            let (x, y, n, stopped) = match engine {
                "bite" => {
                    let r = BiteOpt::new(
                        &[-5.; 3],
                        &[5.; 3],
                        None,
                        &BiteParams {
                            max_evaluations: 1000,
                            ..Default::default()
                        },
                    )
                    .optimize_until(&objective, &stop);
                    (r.x, r.y, r.evaluations, r.stopped)
                }
                "deep" => {
                    let r = DeepBiteOpt::new(
                        &[-5.; 3],
                        &[5.; 3],
                        None,
                        &BiteParams {
                            max_evaluations: 1000,
                            ..Default::default()
                        },
                        3,
                    )
                    .optimize_until(&objective, &stop);
                    (r.x, r.y, r.evaluations, r.stopped)
                }
                _ => {
                    let r = De::new(
                        Fitness::bounded(3, 1, &[-5.; 3], &[5.; 3]),
                        &[],
                        &[],
                        None,
                        &DeParams {
                            max_evaluations: 1000,
                            ..Default::default()
                        },
                    )
                    .optimize_until(&objective, &stop);
                    (r.x, r.y, r.evaluations, r.stopped)
                }
            };
            assert!(stopped, "{engine} k={k}");
            assert_eq!(n, k as u64, "{engine} k={k}");
            assert_eq!(calls.load(Ordering::SeqCst), k);
            if k > 0 {
                assert_eq!(y, *best.lock().unwrap(), "{engine} k={k}");
                assert_eq!(y, x.iter().map(|x| x * x).sum::<f64>());
            }
        }
    }
}

#[test]
fn batches_drain_in_flight_work_and_count_it() {
    for batch in [1, 4, 16] {
        for engine in ["bite", "de"] {
            let calls = AtomicUsize::new(0);
            let objective = |x: &[f64]| {
                calls.fetch_add(1, Ordering::SeqCst);
                x.iter().map(|v| v * v).sum::<f64>()
            };
            let stop = || calls.load(Ordering::SeqCst) >= 37;
            let (n, stopped) = if engine == "bite" {
                let r = BiteOpt::new(
                    &[-5.; 3],
                    &[5.; 3],
                    None,
                    &BiteParams {
                        max_evaluations: 1000,
                        ..Default::default()
                    },
                )
                .optimize_batch_until(
                    batch,
                    |xs| xs.iter().map(|x| objective(x)).collect(),
                    &stop,
                );
                (r.evaluations, r.stopped)
            } else {
                let r = De::new(
                    Fitness::bounded(3, 1, &[-5.; 3], &[5.; 3]),
                    &[],
                    &[],
                    None,
                    &DeParams {
                        max_evaluations: 1000,
                        ..Default::default()
                    },
                )
                .optimize_delayed_until(&objective, batch, &stop);
                (r.evaluations, r.stopped)
            };
            assert!(stopped);
            assert_eq!(n as usize, calls.load(Ordering::SeqCst));
            assert!(
                (37..37 + batch).contains(&(n as usize)),
                "{engine} {batch} {n}"
            );
        }
    }
}

#[test]
fn never_stop_preserves_results_bit_for_bit() {
    let objective = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    for seed in 1..=10 {
        let p = BiteParams {
            seed,
            max_evaluations: 1000,
            ..Default::default()
        };
        let mut a = BiteOpt::new(&[-5.; 3], &[5.; 3], None, &p);
        let mut b = BiteOpt::new(&[-5.; 3], &[5.; 3], None, &p);
        assert_eq!(
            format!("{:?}", a.optimize(&objective)),
            format!("{:?}", b.optimize_until(&objective, &|| false))
        );
        let p = DeParams {
            seed,
            max_evaluations: 1000,
            ..Default::default()
        };
        let make = || {
            De::new(
                Fitness::bounded(3, 1, &[-5.; 3], &[5.; 3]),
                &[],
                &[],
                None,
                &p,
            )
        };
        assert_eq!(
            format!("{:?}", make().optimize(&objective)),
            format!("{:?}", make().optimize_until(&objective, &|| false))
        );
        assert_eq!(
            format!("{:?}", make().optimize_delayed(&objective, 4)),
            format!(
                "{:?}",
                make().optimize_delayed_until(&objective, 4, &|| false)
            )
        );
    }
}

#[test]
fn retry_until_stops_claiming_new_runs() {
    use fcmaes_core::{RetryBounds, RetryConfig, RetryRunResult, retry_until};
    let calls = AtomicUsize::new(0);
    let objective = |x: &[f64]| x[0];
    let bounds = RetryBounds::new(vec![-1.], vec![1.]).unwrap();
    let config = RetryConfig {
        num_retries: 10,
        workers: 1,
        ..Default::default()
    };
    let result = retry_until(
        &objective,
        &bounds,
        &config,
        |_, context| {
            calls.fetch_add(1, Ordering::SeqCst);
            RetryRunResult {
                x: vec![0.0],
                y: context.run_id as f64,
                evaluations: 1,
            }
        },
        &|| calls.load(Ordering::SeqCst) >= 2,
    );
    assert_eq!(result.runs, 2);
    assert_eq!(result.evaluations, 2);
    let result = retry_until(
        &objective,
        &bounds,
        &config,
        |_, _| panic!("stopped retry must not be claimed"),
        &|| true,
    );
    assert_eq!(result.runs, 0);
}
