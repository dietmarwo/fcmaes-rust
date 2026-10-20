use fcmaes_surrogate::{
    Config, Space, design,
    gp::{self, ExactGp, Hyper, Kernel},
    special,
    transform::{OutputTransform, Transform},
};
use serde_json::Value;

#[test]
fn log_ei_matches_independent_high_precision_tail_tables() {
    let table: Value = serde_json::from_str(include_str!("fixtures/log-h.json")).unwrap();
    for row in table["values"].as_array().unwrap() {
        let z = row["z"].as_f64().unwrap();
        let expected = row["log_h"].as_f64().unwrap();
        let actual = special::log_h(z);
        let tolerance = 1e-10_f64.max(3e-15 * expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "z={z}, {actual} vs {expected}"
        );
        assert!(special::log_ei(0.0, -z, 1.0).is_finite());
    }
    assert_eq!(special::log_ei(1.0, 2.0, 0.0), f64::NEG_INFINITY);
    assert_eq!(special::log_ei(2.0, 1.0, 0.0), 0.0);
    let values: Vec<f64> = [-1000.0, -40.0, -10.0, -1.0, 0.0, 10.0]
        .map(special::log_h)
        .to_vec();
    assert!(values.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn exact_gp_matches_independent_lu_reference() {
    let data: Value = serde_json::from_str(include_str!("fixtures/gp.json")).unwrap();
    for model in data["models"].as_array().unwrap() {
        let xs: Vec<Vec<f64>> = serde_json::from_value(model["x"].clone()).unwrap();
        let ys: Vec<f64> = serde_json::from_value(model["y"].clone()).unwrap();
        let hyper = Hyper {
            lengthscales: serde_json::from_value(model["lengthscales"].clone()).unwrap(),
        };
        let kernel: Kernel = serde_json::from_value(model["kernel"].clone()).unwrap();
        let gp = ExactGp::with_hyper(&xs, &ys, kernel, hyper).unwrap();
        assert!(
            (gp.negative_log_posterior - model["negative_log_posterior"].as_f64().unwrap()).abs()
                < 1e-10
        );
        for q in model["predictions"].as_array().unwrap() {
            let x: Vec<f64> = serde_json::from_value(q["x"].clone()).unwrap();
            let (mean, var) = gp.posterior(&x).unwrap();
            assert!((mean - q["mean"].as_f64().unwrap()).abs() < 1e-10);
            assert!((var - q["variance"].as_f64().unwrap()).abs() < 1e-10);
        }
    }
}

#[test]
fn fantasy_append_matches_fresh_factorization_and_duplicates_are_safe() {
    let mut xs = vec![vec![0.1], vec![0.6], vec![0.9]];
    let mut ys = vec![1.0, -1.0, 0.2];
    for kernel in [Kernel::Rbf, Kernel::Matern52] {
        let hyper = Hyper {
            lengthscales: vec![0.4],
        };
        let mut gp = ExactGp::with_hyper(&xs, &ys, kernel, hyper.clone()).unwrap();
        let x = vec![0.35];
        let value = gp.posterior(&x).unwrap().0;
        gp = gp.fantasize(x.clone(), value).unwrap();
        xs.push(x);
        ys.push(value);
        let fresh = ExactGp::with_hyper(&xs, &ys, kernel, hyper.clone()).unwrap();
        assert!((gp.negative_log_posterior - fresh.negative_log_posterior).abs() < 1e-8);
        for x in [0.0, 0.2, 0.35, 0.8, 1.0] {
            let a = gp.posterior(&[x]).unwrap();
            let b = fresh.posterior(&[x]).unwrap();
            assert!((a.0 - b.0).abs() < 1e-8);
            assert!((a.1 - b.1).abs() < 1e-8);
        }
        let duplicate = gp.fantasize(vec![0.1], 1.0).unwrap();
        assert!(duplicate.posterior(&[0.1]).unwrap().1 >= 0.0);
    }
}

#[test]
fn kernel_covariance_and_map_prior_have_the_expected_properties() {
    for d in [1, 2, 5, 32] {
        let mode = libm::exp(std::f64::consts::SQRT_2 + 0.5 * libm::log(d as f64) - 3.0);
        let hyper = Hyper {
            lengthscales: vec![mode; d],
        };
        for factor in [0.8, 1.2] {
            assert!(
                gp::lengthscale_prior(&hyper)
                    < gp::lengthscale_prior(&Hyper {
                        lengthscales: vec![mode * factor; d]
                    })
            );
        }
        let xs = design::latin_hypercube(12, d, 42);
        for kernel in [Kernel::Rbf, Kernel::Matern52] {
            assert_eq!(gp::covariance(kernel, &xs[0], &xs[0], &hyper), 1.0);
            assert_eq!(
                gp::covariance(kernel, &xs[0], &xs[1], &hyper),
                gp::covariance(kernel, &xs[1], &xs[0], &hyper)
            );
            ExactGp::with_hyper(&xs, &[0.0; 12], kernel, hyper.clone()).unwrap();
        }
    }
}

#[test]
fn spaces_transforms_and_lhs_are_checked_at_extreme_scales() {
    let s = Space::new(vec![1e-10, -1000.0], vec![1e10, 10.0], vec![true, false]).unwrap();
    for x in [[0.0, 0.0], [0.25, 0.8], [1.0, 1.0]] {
        let back = s.to_unit(&s.from_unit(&x).unwrap()).unwrap();
        for (a, b) in x.iter().zip(back) {
            assert!((a - b).abs() < 1e-12);
        }
    }
    assert!(Space::new(vec![-1.0], vec![2.0], vec![true]).is_err());
    assert!(s.from_unit(&[2.0, 0.0]).is_err());
    for ys in [
        vec![1e-12, 3e-12, 7e-12],
        vec![-1e200, 0.0, 1e200],
        vec![5.0; 3],
    ] {
        for kind in [
            OutputTransform::Standardize,
            OutputTransform::Bilog,
            OutputTransform::Auto,
        ] {
            let transform = Transform::fit(&ys, kind).unwrap();
            for y in &ys {
                let z = transform.apply(*y);
                assert!(z.is_finite());
                let back = transform.inverse(z).unwrap();
                assert!((back - y).abs() <= 1e-12_f64.max(y.abs() * 1e-12));
            }
        }
    }
    assert!(
        Transform::fit(&[1.0, 1e12], OutputTransform::Auto)
            .unwrap()
            .bilog
    );
    let xs = design::latin_hypercube(16, 4, 51);
    assert_eq!(xs, design::latin_hypercube(16, 4, 51));
    for j in 0..4 {
        let mut bins: Vec<usize> = xs.iter().map(|x| (x[j] * 16.0) as usize).collect();
        bins.sort();
        assert_eq!(bins, (0..16).collect::<Vec<_>>());
    }
    assert!(serde_json::from_str::<Config>(r#"{"asynchronous":true}"#).is_err());
}

#[test]
fn map_fit_improves_on_its_start_is_reproducible_and_stays_in_bounds() {
    for (n, d) in [(10, 1), (40, 4), (120, 12)] {
        let xs = design::latin_hypercube(n, d, 17);
        let ys: Vec<f64> = xs
            .iter()
            .map(|x| {
                x.iter()
                    .enumerate()
                    .map(|(j, v)| (4.0 * v + j as f64).sin())
                    .sum()
            })
            .collect();
        let mode = libm::exp(std::f64::consts::SQRT_2 + 0.5 * libm::log(d as f64) - 3.0);
        for kernel in [Kernel::Matern52, Kernel::Rbf] {
            let start = Hyper {
                lengthscales: vec![mode; d],
            };
            let fit = |restarts| {
                gp::fit(
                    gp::Fit {
                        xs: &xs,
                        ys: &ys,
                        kernel,
                        warm: Some(&start),
                        seed: 5,
                        restarts,
                        iterations: 50,
                    },
                    &|| false,
                )
                .unwrap()
            };
            let (single, evaluations) = fit(1);
            let initial = ExactGp::with_hyper(&xs, &ys, kernel, start.clone()).unwrap();
            assert!(single.negative_log_posterior < initial.negative_log_posterior);
            assert!(evaluations > 1 && evaluations <= 50 * 31 + 1);
            let (restarted, _) = fit(3);
            assert!(restarted.negative_log_posterior <= single.negative_log_posterior);
            assert_eq!(
                fit(3).0.negative_log_posterior,
                restarted.negative_log_posterior
            );
            // The carried hyperparameters rebuild exactly the fitted model.
            let rebuilt = ExactGp::with_hyper(&xs, &ys, kernel, restarted.hyper.clone()).unwrap();
            assert_eq!(
                rebuilt.negative_log_posterior,
                restarted.negative_log_posterior
            );
            let upper = 1000.0 * libm::sqrt(d as f64);
            assert!(
                restarted
                    .hyper
                    .lengthscales
                    .iter()
                    .all(|l| (0.025..=upper).contains(l))
            );
        }
    }
    assert!(matches!(
        gp::fit(
            gp::Fit {
                xs: &[vec![0.5]],
                ys: &[1.0],
                kernel: Kernel::Rbf,
                warm: None,
                seed: 1,
                restarts: 1,
                iterations: 5,
            },
            &|| true,
        ),
        Err(fcmaes_surrogate::Error::Stopped)
    ));
}
