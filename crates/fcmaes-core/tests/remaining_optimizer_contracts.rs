use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use fcmaes_core::{
    AdvancedRetryConfig, Archive, Crfmnes, CrfmnesParams, DiversifierParams, Fitness,
    MoRetryConfig, Objective, Pgpe, PgpeParams, RetryBounds, RetryConfig, RetryRunResult, Rng,
    advanced_retry_until, diversify_batch_capped, moretry_observed, moretry_with_accounting,
    scalarize,
};

fn loss(x: &[f64]) -> f64 {
    x.iter().map(|v| (v - 0.3).powi(2)).sum()
}

#[test]
fn capped_es_consumes_prefix_and_returns_a_real_observation() {
    for budget in [1, 31, 33, 63, 64] {
        let fit = Fitness::bounded(2, 1, &[-1.0; 2], &[1.0; 2]);
        let mut nes = Crfmnes::new(
            fit.clone(),
            &[0.9; 2],
            0.5,
            &CrfmnesParams {
                max_evaluations: budget,
                ..Default::default()
            },
        );
        let mut pgpe = Pgpe::new(
            fit,
            &[0.9; 2],
            &[0.5],
            &PgpeParams {
                max_evaluations: budget,
                ..Default::default()
            },
        );

        let mut points = Vec::<Vec<f64>>::new();
        let mut evaluate = |xs: &[Vec<f64>]| {
            points.extend_from_slice(xs);
            xs.iter().map(|x| loss(x)).collect::<Vec<_>>()
        };
        let result = nes
            .optimize_batch_capped_until(&mut evaluate, &|| false)
            .unwrap();
        assert_eq!(result.evaluations, budget);
        assert_eq!(result.iterations, (budget / 32) as i32);
        assert_eq!(points.len() as u64, budget);
        assert_eq!(
            result.y,
            points.iter().map(|x| loss(x)).fold(f64::INFINITY, f64::min)
        );
        assert!(points.contains(&result.x));
        assert_eq!(result.y, loss(&result.x));

        points.clear();
        let result = pgpe
            .optimize_batch_capped_until(
                |xs| {
                    points.extend_from_slice(xs);
                    xs.iter().map(|x| loss(x)).collect()
                },
                &|| false,
            )
            .unwrap();
        assert_eq!(result.evaluations, budget);
        assert_eq!(result.iterations, (budget / 32) as i32);
        assert_eq!(points.len() as u64, budget);
        assert_eq!(
            result.y,
            points.iter().map(|x| loss(x)).fold(f64::INFINITY, f64::min)
        );
        assert!(points.contains(&result.x));
        assert_eq!(result.y, loss(&result.x));
    }
}

#[test]
fn capped_pgpe_rejects_wrong_score_count_and_honors_initial_stop() {
    let fit = Fitness::bounded(2, 1, &[-1.; 2], &[1.; 2]);
    let mut pgpe = Pgpe::new(
        fit,
        &[0.; 2],
        &[0.1],
        &PgpeParams {
            max_evaluations: 1,
            ..Default::default()
        },
    );
    assert!(
        pgpe.optimize_batch_capped_until(|_| vec![0.; 2], &|| false)
            .is_err()
    );
    let r = pgpe
        .optimize_batch_capped_until(|_| panic!("callback should be stopped"), &|| true)
        .unwrap();
    assert_eq!(r.evaluations, 0);
}

#[test]
fn diversifier_capped_counts_and_commits_the_final_prefix() {
    for budget in [1, 31, 33, 65] {
        let mut rng = Rng::new(13);
        let mut archive = Archive::new(2, &[-1.0; 2], &[1.0; 2], 64, 0, &mut rng);
        archive.seed_uniform(&[-1.; 2], &[1.; 2], &mut rng);
        let mut called = 0;
        let mut observed = f64::INFINITY;
        let (x, y) = diversify_batch_capped(
            &mut archive,
            &mut |xs: &[Vec<f64>]| {
                called += xs.len() as u64;
                xs.iter()
                    .map(|x| {
                        let value = loss(x);
                        observed = observed.min(value);
                        (value, x.clone())
                    })
                    .collect()
            },
            &[-1.; 2],
            &[1.; 2],
            &DiversifierParams {
                max_evaluations: budget,
                popsize: 32,
                stall_criterion: 8,
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(called, budget);
        assert_eq!(archive.counts().iter().sum::<u64>(), budget);
        assert_eq!(y, observed);
        assert_eq!(y, loss(&x));
    }
}

#[test]
fn weighted_retry_distinguishes_measured_and_reevaluated_vectors() {
    let calls = AtomicU64::new(0);
    let objective = |x: &[f64]| {
        calls.fetch_add(1, Relaxed);
        vec![x[0], 1.0 - x[0], 0.0]
    };
    let bounds = RetryBounds::new(vec![0.0], vec![1.0]).unwrap();
    let mut p = MoRetryConfig::new(vec![0.1; 3], vec![1.0; 3]);
    p.ncon = 1;
    p.retry.num_retries = 4;
    p.retry.workers = 2;

    let (legacy, extra) = moretry_with_accounting(&objective, &bounds, &p, |o, c| {
        let x = vec![c.run_id as f64 / 4.0];
        let y = o.eval_scalar(&x);
        RetryRunResult {
            x,
            y,
            evaluations: 1,
        }
    })
    .unwrap();
    assert_eq!(legacy.evaluations, 4);
    assert_eq!(extra, 4);
    assert_eq!(calls.swap(0, Relaxed), 8);

    let (observed, extra) = moretry_observed(&objective, &bounds, &p, |o, c| {
        let x = vec![c.run_id as f64 / 4.0];
        let values = o.eval_multi(&x);
        let y = scalarize(&values, o.weights(), o.ncon(), o.value_exp());
        (
            RetryRunResult {
                x,
                y,
                evaluations: 1,
            },
            Some(values),
        )
    })
    .unwrap();
    assert_eq!(observed.evaluations, 4);
    assert_eq!(extra, 0);
    assert_eq!(calls.load(Relaxed), 4);
    for entry in observed.entries {
        assert_eq!(entry.y.as_slice(), [entry.x[0], 1.0 - entry.x[0], 0.0]);
    }
}

#[test]
fn coordinated_retry_stops_claiming_after_the_first_completed_run() {
    let calls = AtomicU64::new(0);
    let bounds = RetryBounds::new(vec![0.0], vec![1.0]).unwrap();
    let config = AdvancedRetryConfig {
        retry: RetryConfig {
            num_retries: 32,
            workers: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = advanced_retry_until(
        &|x: &[f64]| loss(x),
        &bounds,
        &config,
        |f, _| {
            let x = vec![0.4];
            let y = f(&x);
            calls.fetch_add(1, Relaxed);
            RetryRunResult {
                x,
                y,
                evaluations: 1,
            }
        },
        &|| calls.load(Relaxed) == 1,
    );
    assert_eq!(result.runs, 1);
    assert_eq!(result.evaluations, 1);
    assert_eq!(result.y, loss(&result.x));
}

#[test]
fn crfmnes_rounds_odd_population_before_sampling() {
    let fit = Fitness::bounded(2, 1, &[-1.0; 2], &[1.0; 2]);
    let mut nes = Crfmnes::new(
        fit,
        &[0.0; 2],
        0.1,
        &CrfmnesParams {
            popsize: 3,
            ..Default::default()
        },
    );
    assert_eq!(nes.popsize(), 4);
    let points = nes.ask_pop();
    assert_eq!(points.len(), 4);
    let scores = points.iter().map(|x| loss(x)).collect::<Vec<_>>();
    nes.tell_pop(&scores);
    assert_eq!(nes.result().evaluations, 4);
}
