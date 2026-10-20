use fcmaes_surrogate::{
    Config, Error, Event, Observation, Round, RunSpec, Space, StrategyState, Value, driver,
    propose, run,
};
use rayon::ThreadPoolBuilder;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

fn config() -> Config {
    Config {
        candidates: 64,
        initial_points: 4,
        fit_restarts: 2,
        fit_iterations: 20,
        ..Config::default()
    }
}
fn space() -> Space {
    Space::new(vec![-2.0; 2], vec![2.0; 2], vec![false; 2]).unwrap()
}

#[test]
fn decisions_and_values_are_identical_on_one_two_three_and_eight_threads() {
    let mut baseline = None;
    for threads in [1, 2, 3, 8] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let (result, events) = pool.install(|| {
            let mut events = Vec::new();
            let result = run(
                RunSpec {
                    space: &space(),
                    config: &config(),
                    evaluations: 20,
                    concurrent: 4,
                    seed: 41,
                    warm: &[],
                    target: None,
                },
                &|x| Ok((x[0] - 0.3).powi(2) + (x[1] + 0.4).powi(2)),
                &mut |e| {
                    events.push(e.clone());
                    Ok(())
                },
                &|| None,
            )
            .unwrap();
            (result, events)
        });
        assert_eq!(result.evaluations, 20);
        assert_eq!(result.failed_evaluations, 0);
        assert!(result.model_fits > 0);
        driver::replay(&events).unwrap();
        let decisions: Vec<Event> = events
            .iter()
            .filter(|e| matches!(e, Event::Proposal { .. }))
            .cloned()
            .collect();
        let actual = (result.history.clone(), decisions);
        if let Some(baseline) = &baseline {
            assert_eq!(&actual, baseline);
        } else {
            baseline = Some(actual);
        }
        let encoded = serde_json::to_string(&events).unwrap();
        let decoded: Vec<Event> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(events, decoded);
        driver::replay(&decoded).unwrap();
        let mut incomplete = decoded.clone();
        incomplete.pop();
        assert!(driver::replay(&incomplete).is_err());
        let mut repeated = decoded.clone();
        let start = repeated
            .iter()
            .position(|e| matches!(e, Event::Started { .. }))
            .unwrap();
        repeated.insert(start, repeated[start].clone());
        assert!(driver::replay(&repeated).is_err());
        let mut incorrect_counts = decoded.clone();
        if let Some(Event::Finished { evaluations, .. }) = incorrect_counts.last_mut() {
            *evaluations += 1;
        }
        assert!(driver::replay(&incorrect_counts).is_err());
        let mut corrupt = decoded;
        if let Some(Event::Proposal { points, .. }) = corrupt
            .iter_mut()
            .find(|e| matches!(e, Event::Proposal { .. }))
        {
            points[0][0] += 0.01;
        }
        assert!(driver::replay(&corrupt).is_err());
    }
}

#[test]
fn pending_points_are_fantasized_and_never_repeated() {
    let s = space();
    let config = config();
    let initial = fcmaes_surrogate::design::initial_design(4, 2, 41);
    let history: Vec<Observation> = initial
        .iter()
        .enumerate()
        .map(|(i, x)| Observation {
            id: i as u64,
            x: s.from_unit(x).unwrap(),
            value: Value::Finite(x.iter().map(|x| x * x).sum()),
        })
        .collect();
    let state = StrategyState::default();
    let pending = vec![vec![0.0, 0.0]];
    let p = propose(
        &Round {
            space: &s,
            config: &config,
            seed: 41,
            history: &history,
            pending: &pending,
            state: &state,
            count: 4,
        },
        &|| false,
    )
    .unwrap();
    for (i, x) in p.points.iter().enumerate() {
        assert!(!pending.contains(x));
        assert!(!p.points[..i].contains(x));
        assert!(s.contains(x));
    }
    assert_eq!(
        propose(
            &Round {
                space: &s,
                config: &config,
                seed: 41,
                history: &history,
                pending: &pending,
                state: &state,
                count: 4
            },
            &|| true
        )
        .unwrap_err(),
        Error::Stopped
    );
    assert_eq!(state, StrategyState::default());
}

#[test]
fn completed_evaluations_are_logged_before_a_slower_batch_member_finishes() {
    let fast_logged = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&fast_logged);
    let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    pool.install(|| {
        let cfg = Config {
            initial_points: 2,
            ..config()
        };
        run(
            RunSpec {
                space: &space(),
                config: &cfg,
                evaluations: 2,
                concurrent: 2,
                seed: 42,
                warm: &[],
                target: None,
            },
            &|x| {
                if x == [0.0, 0.0] {
                    let start = Instant::now();
                    while !flag.load(Ordering::Acquire) && start.elapsed() < Duration::from_secs(2)
                    {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    assert!(flag.load(Ordering::Acquire));
                }
                Ok(x.iter().map(|x| x * x).sum())
            },
            &mut |event| {
                if let Event::Completed { observation, .. } = event
                    && observation.x != [0.0, 0.0]
                {
                    fast_logged.store(true, Ordering::Release);
                }
                Ok(())
            },
            &|| None,
        )
        .unwrap();
    });
}

#[test]
fn failures_cancelled_callbacks_and_log_errors_are_not_silently_lost() {
    let calls = AtomicUsize::new(0);
    let result = run(
        RunSpec {
            space: &space(),
            config: &config(),
            evaluations: 4,
            concurrent: 1,
            seed: 2,
            warm: &[],
            target: None,
        },
        &|_| match calls.fetch_add(1, Ordering::SeqCst) {
            0 => panic!("test simulator panic"),
            1 => Ok(f64::NAN),
            2 => Err("domain failure".into()),
            _ => Ok(1e99),
        },
        &mut |_| Ok(()),
        &|| None,
    )
    .unwrap();
    assert_eq!(result.failed_evaluations, 4);
    assert!(result.best.is_none());
    let calls = AtomicUsize::new(0);
    let result = run(
        RunSpec {
            space: &space(),
            config: &config(),
            evaluations: 4,
            concurrent: 1,
            seed: 2,
            warm: &[],
            target: None,
        },
        &|_| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(0.0)
        },
        &mut |event| {
            if matches!(event, Event::Completed { .. }) {
                Err(Error::Log("disk full".into()))
            } else {
                Ok(())
            }
        },
        &|| None,
    );
    assert!(matches!(result, Err(Error::Log(_))));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let cancelled = AtomicBool::new(false);
    let mut events = Vec::new();
    let result = run(
        RunSpec {
            space: &space(),
            config: &config(),
            evaluations: 4,
            concurrent: 1,
            seed: 2,
            warm: &[],
            target: None,
        },
        &|_| {
            cancelled.store(true, Ordering::Release);
            Err("cancelled".into())
        },
        &mut |e| {
            events.push(e.clone());
            Ok(())
        },
        &|| cancelled.load(Ordering::Acquire).then_some("cancelled"),
    )
    .unwrap();
    assert_eq!(result.evaluations, 0);
    assert_eq!(result.failed_evaluations, 0);
    assert_eq!(result.cancelled_evaluations, 1);
    assert!(events.iter().any(|e| matches!(e, Event::Cancelled { .. })));
}

#[test]
fn warm_observations_do_not_spend_the_new_budget_and_targets_stop_work() {
    let warm = vec![Observation {
        id: 17,
        x: vec![0.0, 0.0],
        value: Value::Finite(0.0),
    }];
    let calls = AtomicUsize::new(0);
    let result = run(
        RunSpec {
            space: &space(),
            config: &config(),
            evaluations: 4,
            concurrent: 1,
            seed: 2,
            warm: &warm,
            target: Some(0.0),
        },
        &|_| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(1.0)
        },
        &mut |_| Ok(()),
        &|| None,
    )
    .unwrap();
    assert_eq!(result.termination, "objective_limit");
    assert_eq!(result.evaluations, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let result = run(
        RunSpec {
            space: &space(),
            config: &config(),
            evaluations: 4,
            concurrent: 1,
            seed: 2,
            warm: &warm,
            target: None,
        },
        &|x| Ok(x.iter().map(|v| v * v).sum()),
        &mut |_| Ok(()),
        &|| None,
    )
    .unwrap();
    assert_eq!(result.evaluations, 4);
    assert_eq!(result.history.len(), 5);
    assert_eq!(result.history[1].id, 18);
}
