use fcmaes_surrogate::{
    Observation, Space, Value,
    advanced::{self, Event, Options, Sample, Schedule},
    gp::{ExactGp, Hyper, Kernel},
    sampling,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn retarget_append_matches_reconstruction() {
    for kernel in [Kernel::Rbf, Kernel::Matern52] {
        let xs = vec![vec![0.1], vec![0.4], vec![0.9]];
        let hyper = Hyper {
            lengthscales: vec![0.3],
        };
        let gp = ExactGp::with_hyper(&xs[..2], &[1.0, 2.0], kernel, hyper.clone()).unwrap();
        let gp = gp
            .with_values(&[-2.0, 3.0])
            .unwrap()
            .fantasize(xs[2].clone(), 0.3)
            .unwrap();
        let rebuilt = ExactGp::with_hyper(&xs, &[-2.0, 3.0, 0.3], kernel, hyper).unwrap();
        for x in [0.0, 0.2, 0.7, 1.0] {
            let a = gp.posterior(&[x]).unwrap();
            let b = rebuilt.posterior(&[x]).unwrap();
            assert!((a.0 - b.0).abs() < 1e-10);
            assert!((a.1 - b.1).abs() < 1e-10);
        }
        assert!(gp.with_values(&[f64::NAN; 3]).is_err());
    }
}
#[test]
fn posterior_paths_have_correlations_and_moments() {
    for kernel in [Kernel::Rbf, Kernel::Matern52] {
        let gp = ExactGp::with_hyper(
            &[vec![0.0], vec![1.0]],
            &[0.0, 0.0],
            kernel,
            Hyper {
                lengthscales: vec![0.3],
            },
        )
        .unwrap();
        let points = vec![vec![0.4], vec![0.45]];
        let mut draws = Vec::new();
        let mut paths = Vec::new();
        for seed in 0..1000 {
            draws.push(sampling::exact_sample(&gp, &points, seed).unwrap());
            let p = sampling::Path::new(&gp, 256, seed).unwrap();
            let values = vec![p.value(&points[0]).unwrap(), p.value(&points[1]).unwrap()];
            assert_eq!(values[0], p.value(&points[0]).unwrap());
            paths.push(values);
        }
        let expected = gp.posterior_covariance(&points[0], &points[1]).unwrap();
        for samples in [&draws, &paths] {
            let mean = samples.iter().map(|v| v[0]).sum::<f64>() / 1000.0;
            let cov = samples.iter().map(|v| v[0] * v[1]).sum::<f64>() / 1000.0;
            assert!(mean.abs() < 0.12, "{mean}");
            assert!((cov - expected).abs() < 0.12, "{cov} vs {expected}");
        }
    }
}
#[test]
fn budgets_async_stop_and_failure_are_exact() {
    let space = Space::new(vec![-1.0], vec![1.0], vec![false]).unwrap();
    for schedule in [Schedule::Synchronous, Schedule::Asynchronous] {
        for budget in [1, 3, 13] {
            let options = Options {
                concurrent: 3,
                schedule,
                ..Options::default()
            };
            let calls = AtomicUsize::new(0);
            let mut events = Vec::new();
            let result = advanced::run(
                advanced::RunSpec {
                    space: &space,
                    options: &options,
                    evaluations: budget,
                    seed: 3,
                    warm: &[],
                    target: None,
                    inner_threads: 4,
                },
                &|x| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    Ok(x[0] * x[0])
                },
                &mut |e| {
                    events.push(e.clone());
                    Ok(())
                },
                &|| None,
            )
            .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), budget);
            assert_eq!(result.evaluations, budget);
            advanced::replay(&events).unwrap();
            assert!(result.peak_pending <= 3);
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::Completed { .. }))
                    .count(),
                budget
            );
        }
    }
    let options = Options {
        concurrent: 1,
        ..Options::default()
    };
    let calls = AtomicUsize::new(0);
    let result = advanced::run(
        advanced::RunSpec {
            space: &space,
            options: &options,
            evaluations: 30,
            seed: 3,
            warm: &[],
            target: None,
            inner_threads: 1,
        },
        &|_| {
            calls.fetch_add(1, Ordering::SeqCst);
            Err("model failed".into())
        },
        &mut |_| Ok(()),
        &|| (calls.load(Ordering::SeqCst) >= 7).then_some("cancelled"),
    )
    .unwrap();
    assert_eq!(result.evaluations, 7);
    assert_eq!(result.failed_evaluations, 7);
    assert!(result.best.is_none());
}
#[test]
fn handoff_partial_and_warm_duplicate_validation() {
    let space = Space::new(vec![0.01], vec![10.0], vec![true]).unwrap();
    for n in [1, 2, 3, 9] {
        let options = Options {
            handoff_after: Some(1),
            // Partial budgets: hand off however few calls remain.
            handoff_min_populations: 0,
            ..Options::default()
        };
        let calls = AtomicUsize::new(0);
        let result = advanced::run(
            advanced::RunSpec {
                space: &space,
                options: &options,
                evaluations: n,
                seed: 21,
                warm: &[],
                target: None,
                inner_threads: 1,
            },
            &|x| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(x[0])
            },
            &mut |_| Ok(()),
            &|| None,
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), n);
        assert_eq!(result.evaluations, n);
        assert_eq!(
            result.best.unwrap().value,
            result
                .history
                .iter()
                .map(|s| s.observation.value.clone())
                .min_by(|a, b| {
                    let v = |x: &Value| match x {
                        Value::Finite(v) => *v,
                        _ => f64::INFINITY,
                    };
                    v(a).total_cmp(&v(b))
                })
                .unwrap()
        );
    }
}
#[test]
fn bounded_archive_retains_incumbent_and_restarts() {
    let space = Space::new(vec![0.0], vec![1.0], vec![false]).unwrap();
    let history: Vec<_> = (0..10000)
        .map(|i| Sample {
            observation: Observation {
                id: i,
                x: vec![i as f64 / 10000.0],
                value: Value::Finite((i as f64 - 5000.0).abs()),
            },
            region: 0,
        })
        .collect();
    let refs: Vec<_> = history.iter().collect();
    let chosen = advanced::select_training(&space, &refs, 64).unwrap();
    assert_eq!(chosen.len(), 64);
    assert!(chosen.iter().any(|s| s.observation.id == 5000));
    let options = Options {
        restart_failures: 4,
        ..Options::default()
    };
    let mut engine = advanced::Engine::new(space, options, 5).unwrap();
    let h: Vec<_> = history[..10]
        .iter()
        .map(|s| Sample {
            observation: Observation {
                value: Value::Finite(1.0),
                ..s.observation.clone()
            },
            region: 0,
        })
        .collect();
    let (_, d) = engine.propose(&h, &[], 1, &|| false).unwrap();
    assert_eq!(d.region, 1);
}
#[test]
fn log_failure_prevents_new_work() {
    let space = Space::new(vec![0.0], vec![1.0], vec![false]).unwrap();
    let options = Options::default();
    let calls = AtomicUsize::new(0);
    let result = advanced::run(
        advanced::RunSpec {
            space: &space,
            options: &options,
            evaluations: 8,
            seed: 0,
            warm: &[],
            target: None,
            inner_threads: 1,
        },
        &|_| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(1.0)
        },
        &mut |event| {
            if matches!(event, Event::Completed { .. }) {
                Err(fcmaes_surrogate::Error::Log("disk full".into()))
            } else {
                Ok(())
            }
        },
        &|| None,
    );
    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn sobol_shifted_net_and_bounds_are_verified() {
    let xs = fcmaes_surrogate::design::sobol_shifted(256, 32, 17).unwrap();
    assert_eq!(
        xs,
        fcmaes_surrogate::design::sobol_shifted(256, 32, 17).unwrap()
    );
    for j in 0..32 {
        let mut bins = vec![0; 256];
        for x in &xs {
            assert!(x[j] > 0.0 && x[j] < 1.0);
            bins[(x[j] * 256.0) as usize] += 1;
        }
        assert!(bins.iter().all(|b| *b == 1));
    }
    let mut squares = vec![0; 256];
    for x in &xs {
        squares[(x[0] * 16.0) as usize * 16 + (x[1] * 16.0) as usize] += 1;
    }
    assert!(squares.iter().all(|v| *v == 1));
    assert!(fcmaes_surrogate::design::sobol_shifted(16, 33, 0).is_err());
}

#[test]
fn large_warm_archive_uses_a_bounded_training_set() {
    let space = Space::new(vec![0.0], vec![1.0], vec![false]).unwrap();
    let options = Options {
        training_cap: 16,
        restart_failures: 100000,
        base: fcmaes_surrogate::Config {
            candidates: 32,
            fit_restarts: 1,
            fit_iterations: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let warm: Vec<_> = (0..9999)
        .map(|i| Sample {
            observation: Observation {
                id: i,
                x: vec![i as f64 / 10000.0],
                value: Value::Finite((i as f64 / 10000.0 - 0.23).powi(2)),
            },
            region: 0,
        })
        .collect();
    let calls = AtomicUsize::new(0);
    let mut events = Vec::new();
    let result = advanced::run(
        advanced::RunSpec {
            space: &space,
            options: &options,
            evaluations: 1,
            seed: 7,
            warm: &warm,
            target: None,
            inner_threads: 1,
        },
        &|x| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok((x[0] - 0.23).powi(2))
        },
        &mut |e| {
            events.push(e.clone());
            Ok(())
        },
        &|| None,
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.history.len(), 10000);
    assert!(events.iter().any(
        |e| matches!(e,Event::Proposal{diagnostics,..} if diagnostics.training_ids.len()==16)
    ));
    assert!(result.best.is_some());
}

#[test]
fn synchronous_advanced_decisions_ignore_numerical_pool_size() {
    let space = Space::new(vec![-2.0; 2], vec![2.0; 2], vec![false; 2]).unwrap();
    let mut histories = Vec::new();
    for threads in [1, 2] {
        let options = Options {
            surrogate_threads: threads,
            concurrent: 2,
            base: fcmaes_surrogate::Config {
                initial_points: 4,
                candidates: 32,
                fit_restarts: 1,
                fit_iterations: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = advanced::run(
            advanced::RunSpec {
                space: &space,
                options: &options,
                evaluations: 12,
                seed: 51,
                warm: &[],
                target: None,
                inner_threads: 2,
            },
            &|x| Ok(x.iter().map(|v| (v - 0.3).powi(2)).sum()),
            &mut |_| Ok(()),
            &|| None,
        )
        .unwrap();
        histories.push(
            result
                .history
                .iter()
                .map(|s| s.observation.clone())
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(histories[0], histories[1]);
}

fn handoff_run(
    evaluations: usize,
    options: &Options,
) -> (advanced::RunResult, Vec<advanced::Event>) {
    let space = Space::new(vec![-2.0; 2], vec![2.0; 2], vec![false; 2]).unwrap();
    let mut events = Vec::new();
    let result = advanced::run(
        advanced::RunSpec {
            space: &space,
            options,
            evaluations,
            seed: 5,
            warm: &[],
            target: None,
            inner_threads: 1,
        },
        &|x| Ok(x.iter().map(|v| (v - 0.3).powi(2)).sum()),
        &mut |e| {
            events.push(e.clone());
            Ok(())
        },
        &|| None,
    )
    .unwrap();
    (result, events)
}

fn quick(options: Options) -> Options {
    Options {
        base: fcmaes_surrogate::Config {
            initial_points: 4,
            candidates: 32,
            fit_restarts: 1,
            fit_iterations: 4,
            ..Default::default()
        },
        training_cap: 32,
        ..options
    }
}

#[test]
fn handoff_plan_needs_enough_biteopt_populations() {
    use advanced::{HandoffPlan, biteopt_population, handoff_plan};
    assert_eq!(biteopt_population(2), 15);
    let fraction = Options {
        handoff_fraction: Some(0.25),
        ..Options::default()
    };
    // 0.25 * 30 = 7.5 rounds to 8; 3 populations of 15 calls are required.
    let plan = handoff_plan(&fraction, 30, 2).unwrap();
    assert_eq!(
        plan,
        HandoffPlan {
            after: 8,
            remaining: 22,
            required: 45
        }
    );
    assert!(!plan.performed());
    assert!(handoff_plan(&fraction, 200, 2).unwrap().performed());
    assert!(handoff_plan(&Options::default(), 200, 2).is_none());
    // Count and fraction are alternatives; fractions lie strictly in (0, 1).
    for options in [
        Options {
            handoff_after: Some(4),
            ..fraction.clone()
        },
        Options {
            handoff_fraction: Some(0.0),
            ..Options::default()
        },
        Options {
            handoff_fraction: Some(1.0),
            ..Options::default()
        },
        Options {
            handoff_min_populations: 101,
            ..Options::default()
        },
    ] {
        assert!(options.validate().is_err());
    }
}

#[test]
fn a_handoff_with_too_few_remaining_calls_is_skipped_and_recorded() {
    // 24 of 48 calls remain, fewer than 3 * 15: the GP keeps the budget.
    let options = quick(Options {
        handoff_after: Some(24),
        ..Options::default()
    });
    let (result, events) = handoff_run(48, &options);
    assert_eq!(result.evaluations, 48);
    assert_eq!(result.handoff_after, None);
    assert!(!result.handoff_plan.unwrap().performed());
    assert!(matches!(events[1], advanced::Event::HandoffSkipped { .. }));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, advanced::Event::Handoff { .. }))
    );
    advanced::replay(&events).unwrap();
    // A skip that disagrees with the plan, or a missing one, is refused.
    let mut tampered = events.clone();
    if let advanced::Event::HandoffSkipped { plan } = &mut tampered[1] {
        plan.required -= 1;
    }
    assert!(advanced::replay(&tampered).is_err());
    let mut missing = events.clone();
    missing.remove(1);
    assert!(advanced::replay(&missing).is_err());
}

#[test]
fn a_fractional_handoff_takes_place_at_its_planned_call() {
    // 0.25 * 100 = 25 BO calls; 75 remain, at least 3 * 15.
    let options = quick(Options {
        handoff_fraction: Some(0.25),
        ..Options::default()
    });
    let (result, events) = handoff_run(100, &options);
    assert_eq!(result.evaluations, 100);
    assert_eq!(result.handoff_after, Some(25));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, advanced::Event::Handoff { after: 25, .. }))
    );
    advanced::replay(&events).unwrap();
    // The same trace with the handoff moved is refused.
    let mut moved = events.clone();
    for e in &mut moved {
        if let advanced::Event::Handoff { after, .. } = e {
            *after = 24;
        }
    }
    assert!(advanced::replay(&moved).is_err());
}
