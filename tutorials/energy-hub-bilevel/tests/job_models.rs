//! Native-model parity, discovery, and rejection tests.
#![cfg(feature = "job-worker")]
#[path = "../../job-api/application_contract.rs"]
mod contract;
use energy_hub_bilevel::{config::Preset, evaluate::*, job_models::registry, mo::evaluate_mo};
use serde_json::json;
#[test]
fn unchanged_dispatch_and_constraints_match_job_models() {
    let r = registry();
    let x = analytic_seed();
    let e = evaluate_training(&x, Preset::Smoke).unwrap();
    assert_eq!(
        contract::values(
            contract::resolve(&r, "app.energy.scalar", json!({})).unwrap(),
            &x
        ),
        vec![e.objective]
    );
    // The scalar analytic baseline is not necessarily engineering-feasible.
    let qd_baseline = contract::values(
        contract::resolve(&r, "app.energy.qd", json!({})).unwrap(),
        &x,
    );
    assert_eq!(qd_baseline[0].is_finite(), feasible(&e));
    let qd_x = [0.7, 0.65, 0.65, 0.4, 0.3, 0.3, 0.63, 0.75, 0.88, 0.28];
    let qd_native = evaluate_training(&qd_x, Preset::Smoke).unwrap();
    assert!(
        feasible(&qd_native),
        "QD parity fixture must satisfy every engineering constraint"
    );
    let pair = energy_hub_bilevel::pilot::DescriptorPair::D1;
    let descriptors = pair.values(behavior(&qd_native));
    assert_eq!(
        contract::values(
            contract::resolve(&r, "app.energy.qd", json!({})).unwrap(),
            &qd_x
        ),
        vec![
            qd_native.mean_lcoe,
            descriptors[0].clamp(pair.lower()[0], pair.upper()[0]),
            descriptors[1].clamp(pair.lower()[1], pair.upper()[1])
        ]
    );
    // Preserve the original search penalty as well as valid objective values.
    let invalid = [f64::NAN; 10];
    assert!(evaluate_training(&invalid, Preset::Smoke).is_none());
    assert_eq!(
        contract::values(
            contract::resolve(&r, "app.energy.scalar", json!({})).unwrap(),
            &invalid
        ),
        vec![INVALID_OBJECTIVE]
    );
    assert_eq!(
        contract::values(
            contract::resolve(&r, "app.energy.pareto", json!({})).unwrap(),
            &invalid
        ),
        vec![INVALID_OBJECTIVE; 7]
    );
    let mo = evaluate_mo(&x, Preset::Smoke).unwrap();
    assert_eq!(
        contract::values(
            contract::resolve(&r, "app.energy.pareto", json!({})).unwrap(),
            &x
        ),
        mo.objectives
            .into_iter()
            .chain(mo.constraints)
            .collect::<Vec<_>>()
    );
    for set in ["training", "holdout"] {
        let e = if set == "holdout" {
            evaluate_holdout_aggregate(&x, Preset::Smoke)
        } else {
            evaluate_training(&x, Preset::Smoke)
        }
        .unwrap();
        assert_eq!(
            contract::values(
                contract::resolve(&r, "app.energy.checked", json!({"scenario_set":set})).unwrap(),
                &x
            ),
            vec![
                e.mean_lcoe,
                e.constraint_self_sufficiency,
                e.constraint_unserved,
                e.constraint_cycles,
                e.constraint_lp_status
            ]
        );
    }
}
#[test]
fn invalid_parameters_rejected() {
    assert!(contract::resolve(&registry(), "app.energy.scalar", json!({"preset":"typo"})).is_err());
    assert!(
        contract::resolve(
            &registry(),
            "app.energy.checked",
            json!({"scenario_set":"test_data_used_for_training"})
        )
        .is_err()
    );
}
#[test]
fn discoverable() {
    contract::check_discovery(registry(), 4);
}
