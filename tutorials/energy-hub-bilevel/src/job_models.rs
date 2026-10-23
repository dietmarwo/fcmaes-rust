//! Durable sizing jobs. Dispatch remains an in-process microlp solve.
use crate::{
    config::Preset,
    decode::DIMENSION,
    evaluate::{
        INVALID_OBJECTIVE, behavior, evaluate_holdout_aggregate, evaluate_training, feasible,
    },
    mo::evaluate_mo,
    pilot::DescriptorPair,
};
use fcmaes_job_worker::registry::{
    Example, Instance, MultiObjective, QualityDiversity, Registry, Scalar,
};
use optimization_job_api::Result;
use serde_json::json;

fn preset(m: &Instance) -> Result<Preset> {
    m.require_dimension(DIMENSION..=DIMENSION)?;
    match m.parameter("preset", "smoke".to_string())?.as_str() {
        "smoke" => Ok(Preset::Smoke),
        "publication" => Ok(Preset::Publication),
        _ => Err("preset must be smoke or publication".into()),
    }
}

/// Scalar, original MODE, strict engineering recheck, and D1 repertoire models.
pub fn registry() -> Registry {
    let mut r = Registry::new();
    let about = json!({"dimensions":10,"model":"Original normalized decoder, training scenarios and microlp dispatch",
        "coordinates":"See energy-hub-bilevel/src/decode.rs; all in [0,1]",
        "validation":"Use app.energy.checked for all four engineering constraints. MODE treats unserved energy as an objective. Synthetic teaching costs, not project economics."});
    r.scalar("app.energy.scalar", "1", about.clone(), |m| {
        let p = preset(m)?;
        Ok(
            Scalar::new(move |x| {
                evaluate_training(x, p).map_or(INVALID_OBJECTIVE, |e| e.objective)
            })
            .domain(vec![0.0; DIMENSION], vec![1.0; DIMENSION]),
        )
    });
    r.multi_objective("app.energy.pareto", "1", about.clone(), |m| {
        let p = preset(m)?;
        Ok(MultiObjective::new(4, 3, move |x| {
            evaluate_mo(x, p).map_or_else(
                || vec![INVALID_OBJECTIVE; 7],
                |e| e.objectives.into_iter().chain(e.constraints).collect(),
            )
        })
        .reference_point(vec![1.0e8, 1.0e9, 1.0e9, 1.0e9])
        .domain(vec![0.0; DIMENSION], vec![1.0; DIMENSION]))
    });
    r.multi_objective("app.energy.checked", "1", about.clone(), |m| {
        let p = preset(m)?;
        let set: String = m.parameter("scenario_set", "training".to_string())?;
        if set != "training" && set != "holdout" {
            return Err("scenario_set must be training or holdout".into());
        }
        Ok(MultiObjective::new(1, 4, move |x| {
            let e = if set == "holdout" {
                evaluate_holdout_aggregate(x, p)
            } else {
                evaluate_training(x, p)
            };
            e.map_or_else(
                || vec![1.0e99; 5],
                |e| {
                    vec![
                        e.mean_lcoe,
                        e.constraint_self_sufficiency,
                        e.constraint_unserved,
                        e.constraint_cycles,
                        e.constraint_lp_status,
                    ]
                },
            )
        })
        .reference_point(vec![100.0])
        .domain(vec![0.0; DIMENSION], vec![1.0; DIMENSION]))
    });
    r.quality_diversity("app.energy.qd", "1", about, |m| {
        let p = preset(m)?;
        let pair = DescriptorPair::D1;
        Ok(
            QualityDiversity::new(pair.lower().to_vec(), pair.upper().to_vec(), move |x| {
                match evaluate_training(x, p).filter(feasible) {
                    Some(e) => {
                        let d = pair.values(behavior(&e));
                        (
                            e.mean_lcoe,
                            (0..2)
                                .map(|i| d[i].clamp(pair.lower()[i], pair.upper()[i]))
                                .collect(),
                        )
                    }
                    None => (f64::INFINITY, vec![0.0, 0.0]),
                }
            })
            .domain(vec![0.0; DIMENSION], vec![1.0; DIMENSION]),
        )
    });
    for id in [
        "app.energy.scalar",
        "app.energy.pareto",
        "app.energy.checked",
        "app.energy.qd",
    ] {
        r.example(
            id,
            Example::new(DIMENSION)
                .bounds(vec![0.0; DIMENSION], vec![1.0; DIMENSION])
                .parameter("preset", "smoke"),
        );
    }
    r
}
