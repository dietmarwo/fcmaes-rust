//! Native-model parity, discovery, and rejection tests.
#![cfg(feature = "job-worker")]
#[path = "../../job-api/application_contract.rs"]
mod contract;
use gtoc1_pykep::{
    job_models::{registry, route_and_bounds},
    mga::evaluate_runtime_mga,
};
use serde_json::json;
#[test]
fn native_mga_matches_registered_model() {
    let (route, lo, hi) = route_and_bounds(vec![3, 2, 3, 5, 10]).unwrap();
    let x: Vec<f64> = lo.iter().zip(&hi).map(|(a, b)| a.midpoint(*b)).collect();
    let expected = evaluate_runtime_mga(
        &route.variant().structure.bodies,
        &route.variant().clockwise,
        &route.maximum_revolutions(),
        &x,
    )
    .unwrap()
    .objective;
    assert_eq!(
        contract::values(
            contract::resolve(&registry(), "app.gtoc1.mga", json!({})).unwrap(),
            &x
        ),
        vec![expected]
    );
}
#[test]
fn grammar_and_dimension_errors_are_rejected() {
    assert!(route_and_bounds(vec![3, 999, 10]).is_err());
    assert!(contract::resolve(&registry(), "app.gtoc1.mga", json!({"bodies":[3,2,10]})).is_err());
    assert!(contract::resolve(&registry(), "app.gtoc1.mga", json!({"typo":1})).is_err());
}
#[test]
fn discoverable() {
    contract::check_discovery(registry(), 1);
}
