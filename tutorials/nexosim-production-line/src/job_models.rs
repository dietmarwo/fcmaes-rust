//! Durable jobs with explicit fixed replication sets and one owner of parallelism.
use crate::model::{DIM, LOWER, UPPER};
use crate::optimization::{
    OptimizationConfig, QD_DESCRIPTOR_LOWER, QD_DESCRIPTOR_UPPER, evaluate_design, qd_objective,
};
use fcmaes_job_worker::registry::{Example, Instance, MultiObjective, QualityDiversity, Registry};
use optimization_job_api::Result;
use serde_json::json;

fn config(m: &Instance) -> Result<OptimizationConfig> {
    m.require_dimension(DIM..=DIM)?;
    let c = OptimizationConfig {
        replications: m.parameter("replications", 4_usize)?,
        horizon_minutes: m.parameter("horizon_minutes", 240.0)?,
        seed: m.parameter("simulation_seed", 42_u64)?,
        workers: 1,
        ..Default::default()
    };
    c.validate()?;
    if c.replications > 1024 || c.horizon_minutes > 10080.0 {
        return Err("replications <=1024 and horizon_minutes <=10080 required".into());
    }
    Ok(c)
}

/// The optimizer seed and the simulation seed are independent and recorded.
pub fn registry() -> Registry {
    let mut r = Registry::new();
    let about = json!({"dimensions":9,"variables":["buffer_capacity","speed_a","speed_b","maintenance_a","maintenance_b","rework","dispatch","staff_a","staff_b"],
        "integer_indices":[0,7,8],"model":"Original NeXosim decoder and fixed common-random-number aggregation",
        "validation":"Repeat selected designs using disjoint simulation_seed values; deterministic rechecks alone are not a holdout."});
    r.multi_objective("app.production.pareto", "1", about.clone(), |m| {
        let c = config(m)?;
        Ok(MultiObjective::new(4, 0, move |x| {
            evaluate_design(x, &c, 1).map_or_else(|_| vec![1.0e99; 4], |v| v.to_vec())
        })
        .reference_point(vec![0.0, 1.0e6, 1.0e6, 1.0e9])
        .domain(LOWER.to_vec(), UPPER.to_vec()))
    });
    r.quality_diversity("app.production.qd", "1", about, |m| {
        let c = config(m)?;
        Ok(QualityDiversity::new(
            QD_DESCRIPTOR_LOWER.to_vec(),
            QD_DESCRIPTOR_UPPER.to_vec(),
            move |x| {
                qd_objective(x, &c)
                    .map_or((f64::INFINITY, vec![0.0, 0.0]), |(q, d)| (q, d.to_vec()))
            },
        )
        .domain(LOWER.to_vec(), UPPER.to_vec()))
    });
    for id in ["app.production.pareto", "app.production.qd"] {
        r.example(
            id,
            Example::new(DIM)
                .bounds(LOWER.to_vec(), UPPER.to_vec())
                .parameter("replications", 1)
                .parameter("horizon_minutes", 60.0)
                .parameter("simulation_seed", 42),
        );
    }
    r
}
