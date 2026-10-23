//! Optional durable jobs over the unchanged Rapier model.
use crate::{
    DIMENSION, Design, LOWER_BOUNDS, QD_DESCRIPTOR_LOWER, QD_DESCRIPTOR_UPPER, SimulationConfig,
    UPPER_BOUNDS, multi_objective, qd_objective, scalar_objective, simulate,
};
use fcmaes_job_worker::registry::{
    Example, Instance, MultiObjective, QualityDiversity, Registry, Scalar,
};
use optimization_job_api::Result;
use serde_json::json;

fn config(m: &Instance) -> Result<SimulationConfig> {
    m.require_dimension(DIMENSION..=DIMENSION)?;
    let target: f64 = m.parameter("target_m", 35.0)?;
    if !target.is_finite() || target <= 0.0 {
        return Err("target_m must be finite and positive".into());
    }
    Ok(SimulationConfig {
        target_position: target,
        ..Default::default()
    })
}

/// Models share the original objective and fixed physical integration settings.
pub fn registry() -> Registry {
    let mut r = Registry::new();
    let about = json!({"dimensions":8,"variables":["arm_length_m","counterweight_kg","projectile_kg","sling_m","initial_angle_rad","release_angle_rad","damping","friction"],
        "model":"Rapier trebuchet; original integration and decoder",
        "validation":"Scalar penalties are not a feasibility certificate; use app.trebuchet.checked and replay."});
    r.scalar("app.trebuchet.scalar", "1", about.clone(), |m| {
        let c = config(m)?;
        Ok(Scalar::new(move |x| scalar_objective(x, &c))
            .domain(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec()))
    });
    r.multi_objective("app.trebuchet.pareto", "1", about.clone(), |m| {
        let c = config(m)?;
        Ok(MultiObjective::new(3, 0, move |x| multi_objective(x, &c))
            .reference_point(vec![1000.0, 100000.0, 1.0e8])
            .domain(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec()))
    });
    r.multi_objective("app.trebuchet.checked", "1", about.clone(), |m| {
        let c = config(m)?;
        Ok(
            MultiObjective::new(3, 1, move |x| match Design::from_slice(x) {
                Ok(d) => {
                    let s = simulate(&d, &c);
                    let mut v = s.objectives().to_vec();
                    v.push(if s.status.is_valid() { 0.0 } else { 1.0 });
                    v
                }
                Err(_) => vec![1.0e99; 4],
            })
            .reference_point(vec![1000.0, 100000.0, 1.0e8])
            .domain(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec()),
        )
    });
    r.quality_diversity("app.trebuchet.qd", "1", about, |m| {
        let c = config(m)?;
        Ok(QualityDiversity::new(
            QD_DESCRIPTOR_LOWER.to_vec(),
            QD_DESCRIPTOR_UPPER.to_vec(),
            move |x| {
                let (q, d) = qd_objective(x, &c);
                (q, d.to_vec())
            },
        )
        .domain(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec()))
    });
    for id in [
        "app.trebuchet.scalar",
        "app.trebuchet.pareto",
        "app.trebuchet.checked",
        "app.trebuchet.qd",
    ] {
        r.example(
            id,
            Example::new(8)
                .bounds(LOWER_BOUNDS.to_vec(), UPPER_BOUNDS.to_vec())
                .parameter("target_m", 35.0),
        );
    }
    r
}
