//! Fixed-route impulsive MGA jobs for agent-controlled route selection.
//!
//! Reuses the history-blind campaign's direct duration bounds and physics.
//! Generic retry is a separate optimizer protocol from coordinated DE-CMA-ES.
use crate::{
    mga::evaluate_runtime_mga,
    route_grammar::canonical_clockwise,
    route_search::{RouteCase, RouteDerivationConfig, RouteVariant},
};
use fcmaes_job_worker::registry::{Example, Instance, Registry, Scalar};
use optimization_job_api::Result;
use serde_json::json;

/// Build the route and its direct MJD2000/day bounds without historical hints.
///
/// # Errors
/// Returns an error if the body order or its derived timing bounds are invalid.
pub fn route_and_bounds(bodies: Vec<usize>) -> Result<(RouteCase, Vec<f64>, Vec<f64>)> {
    let directions = canonical_clockwise(&bodies);
    let route = RouteCase::derive(
        RouteVariant::new(bodies, directions),
        RouteDerivationConfig::default(),
    )?;
    let mut lo = vec![3653.0];
    let mut hi = vec![10958.0];
    lo.extend(route.profiles().iter().map(|p| p.lower_days));
    hi.extend(route.profiles().iter().map(|p| p.upper_days));
    Ok((route, lo, hi))
}

fn model(m: &Instance) -> Result<Scalar> {
    let bodies: Vec<usize> = m.parameter("bodies", vec![3, 2, 3, 5, 10])?;
    m.require_dimension(bodies.len()..=bodies.len())?;
    let (route, lo, hi) = route_and_bounds(bodies)?;
    Ok(Scalar::new(move |x| {
        evaluate_runtime_mga(
            &route.variant().structure.bodies,
            &route.variant().clockwise,
            &route.maximum_revolutions(),
            x,
        )
        .map_or(1.0e99, |e| e.objective)
    })
    .domain(lo, hi))
}

/// One model accepts any grammar-valid route; its dimension follows the route.
///
/// # Panics
/// Panics if a future grammar change invalidates the built-in discovery route.
pub fn registry() -> Registry {
    let mut r = Registry::new();
    r.scalar("app.gtoc1.mga","1",json!({
        "dimensions":"number of bodies; Earth=3, Venus=2, Jupiter=5, Saturn=6, asteroid=10",
        "variables":"launch MJD2000, then one direct leg duration in days",
        "objective":"negative mass-adjusted impact score; impulsive MGA screening only",
        "claim_boundary":"Finite MGA qualification is not low-thrust feasibility or a competition-valid solution",
        "optimizer_protocol":"generic job retry; not the published coordinated DE-CMA-ES campaign"}),model);
    let bodies = vec![3, 2, 3, 5, 10];
    let (_, lo, hi) = route_and_bounds(bodies.clone()).expect("documented example route is valid");
    r.example(
        "app.gtoc1.mga",
        Example::new(bodies.len())
            .bounds(lo, hi)
            .parameter("bodies", bodies),
    );
    r
}
