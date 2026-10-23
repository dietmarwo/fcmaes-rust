//! Native-model parity, discovery, and rejection tests.
#![cfg(feature = "job-worker")]
#[path = "../../job-api/application_contract.rs"]
mod contract;
use nexosim_production_line::{job_models::registry, optimization::*};
use serde_json::json;
#[test]
fn common_random_numbers_and_holdout_replay_match_native_model() {
    let x = [8.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0];
    for seed in [42, 314159] {
        let c = OptimizationConfig {
            seed,
            replications: 2,
            horizon_minutes: 60.0,
            workers: 1,
            ..Default::default()
        };
        let r = registry();
        let p = json!({"simulation_seed":seed,"replications":2,"horizon_minutes":60.0});
        let a = contract::values(
            contract::resolve(&r, "app.production.pareto", p.clone()).unwrap(),
            &x,
        );
        assert_eq!(a, evaluate_design(&x, &c, 1).unwrap().to_vec());
        assert_eq!(
            a,
            contract::values(
                contract::resolve(&r, "app.production.pareto", p.clone()).unwrap(),
                &x
            )
        );
        let (q, d) = qd_objective(&x, &c).unwrap();
        assert_eq!(
            contract::values(contract::resolve(&r, "app.production.qd", p).unwrap(), &x),
            vec![q, d[0], d[1]]
        );
    }
}
#[test]
fn invalid_parameters_fail_before_simulation() {
    let r = registry();
    for p in [
        json!({"replications":0}),
        json!({"horizon_minutes":-1.0}),
        json!({"typo":1}),
    ] {
        assert!(contract::resolve(&r, "app.production.pareto", p).is_err());
    }
}
#[test]
fn discoverable() {
    contract::check_discovery(registry(), 2);
}
