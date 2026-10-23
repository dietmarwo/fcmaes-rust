//! Native-model parity, discovery, and rejection tests.
#![cfg(feature = "job-worker")]
#[path = "../../job-api/application_contract.rs"]
mod contract;
use rapier_trebuchet::{job_models::registry, *};
use serde_json::json;

#[test]
fn native_and_job_objectives_agree_at_two_targets() {
    for target in [25.0, 35.0] {
        let c = SimulationConfig {
            target_position: target,
            ..Default::default()
        };
        let r = registry();
        let p = json!({"target_m":target});
        let x = INITIAL_DESIGN;
        assert_eq!(
            contract::values(
                contract::resolve(&r, "app.trebuchet.scalar", p.clone()).unwrap(),
                &x
            ),
            vec![scalar_objective(&x, &c)]
        );
        assert_eq!(
            contract::values(
                contract::resolve(&r, "app.trebuchet.pareto", p.clone()).unwrap(),
                &x
            ),
            multi_objective(&x, &c)
        );
        let (q, d) = qd_objective(&x, &c);
        assert_eq!(
            contract::values(contract::resolve(&r, "app.trebuchet.qd", p).unwrap(), &x),
            vec![q, d[0], d[1]]
        );
    }
}
#[test]
fn rejects_bad_and_unused_parameters() {
    let r = registry();
    assert!(contract::resolve(&r, "app.trebuchet.scalar", json!({"target_m":-1})).is_err());
    assert!(contract::resolve(&r, "app.trebuchet.scalar", json!({"typo":1})).is_err());
}
#[test]
fn checked_model_rejects_invalid_release_and_discovery_is_complete() {
    let r = registry();
    let d = Design::from_slice(&INITIAL_DESIGN).unwrap();
    let s = simulate(&d, &SimulationConfig::default());
    let v = contract::values(
        contract::resolve(&r, "app.trebuchet.checked", json!({})).unwrap(),
        &INITIAL_DESIGN,
    );
    assert_eq!(v[3], if s.status.is_valid() { 0.0 } else { 1.0 });
    let ground_intersection = [6.0, 100.0, 4.0, 6.0, -1.25, 0.0, 1.0, 1.0];
    let invalid = contract::values(
        contract::resolve(&r, "app.trebuchet.checked", json!({})).unwrap(),
        &ground_intersection,
    );
    assert!(
        invalid[3] > 0.0,
        "a physically invalid design must be infeasible"
    );
    contract::check_discovery(r, 4);
}
