//! `fcmaes.evaluate/1`: evaluates given points on a registered model without
//! optimizing, so a caller (an agent, a person or another solver) can have
//! its own candidates checked. One task per point.
use crate::problems;
use fcmaes_native_worker::objective_by_id;
use optimization_job_api::{
    JobSpec, Result, legacy,
    result::{Evidence, Outcome},
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const PROFILE: &str = "fcmaes.evaluate/1";
/// Most points one job may evaluate (one task each).
const MAX_POINTS: usize = 1_000_000;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub problem: legacy::Problem,
    pub points: Vec<Vec<f64>>,
}

enum Model {
    Scalar(fn(&[f64]) -> f64),
    MultiObjective(problems::MultiObjective),
    QualityDiversity(problems::QualityDiversity),
}

pub struct Run {
    payload: Payload,
    model: Model,
}

pub fn describe() -> Value {
    json!({"profile":PROFILE,"problems":"every registered model of fcmaes.retry/1, fcmaes.mode/1 and fcmaes.map-elites/1",
        "points":"1 or more points (inline: the resolved job, points included, must stay below the runner's 4 MiB validation output), each of the problem's dimension and inside its bounds; one task per point",
        "result":"claim feasible for a usable value (scalar, fitness) or satisfied constraints (multi-objective); objectives, constraints or descriptor in details"})
}

pub fn prepare(s: &mut JobSpec) -> Result<Box<dyn Prepared>> {
    let mut p: Payload = serde_json::from_value(s.payload.clone())?;
    let id = p.problem.id.clone();
    let (problem, model) = match id.as_str() {
        "benchmark.zdt1" | "benchmark.dtlz2" | "benchmark.binh-korn" => {
            let (problem, model) = problems::multi_objective(p.problem.clone())?;
            (problem, Model::MultiObjective(model))
        }
        "benchmark.sphere-projection" | "benchmark.rastrigin-projection" => {
            let (problem, model) = problems::quality_diversity(p.problem.clone())?;
            (problem, Model::QualityDiversity(model))
        }
        _ => {
            let problem = p.problem.clone().resolve()?;
            (problem, Model::Scalar(objective_by_id(&id)))
        }
    };
    if p.points.is_empty() || p.points.len() > MAX_POINTS {
        return Err("points must hold 1..1000000 points".into());
    }
    let (lower, upper) = (&problem.bounds.lower, &problem.bounds.upper);
    for (i, x) in p.points.iter().enumerate() {
        if x.len() != problem.config.dimension {
            return Err(format!(
                "point {i} has {} entries; the problem has dimension {}",
                x.len(),
                problem.config.dimension
            )
            .into());
        }
        // Registered models are defined on their box only (GTOP models
        // are not defined outside it), so a point outside is refused.
        if let Some(j) = x
            .iter()
            .zip(lower.iter().zip(upper))
            .position(|(v, (l, u))| !v.is_finite() || v < l || v > u)
        {
            return Err(format!(
                "point {i}: entry {j} = {} is not finite or lies outside [{}, {}]",
                x[j], lower[j], upper[j]
            )
            .into());
        }
    }
    p.problem = problem;
    s.payload = serde_json::to_value(&p)?;
    Ok(Box::new(Run { payload: p, model }))
}

impl Prepared for Run {
    fn tasks(&self) -> usize {
        self.payload.points.len()
    }
    fn cancel(&self) {} // One evaluation per task; tasks stop between points.
    fn solve(&self, task: usize, _: f64) -> Result<TaskResult> {
        let x = self.payload.points[task].clone();
        // A value at or above 1e99 is the fcmaes penalty: no usable value.
        let usable = |v: f64| v.is_finite() && v < 1e99;
        let outcome = match &self.model {
            Model::Scalar(objective) => {
                let value = objective(&x);
                Outcome {
                    claim: if usable(value) { "feasible" } else { "none" }.into(),
                    feasible: Some(usable(value)),
                    objective: value.is_finite().then_some(value),
                    details: json!({"problem":self.payload.problem.id}),
                    evidence: Some(Evidence::rechecked(&["objective", "bounds"])),
                    x: Some(x),
                    ..Outcome::default()
                }
            }
            Model::MultiObjective(model) => {
                let values = (model.evaluate)(&x);
                let (objectives, constraints) = values.split_at(model.nobj);
                let feasible =
                    values.iter().all(|v| v.is_finite()) && constraints.iter().all(|c| *c <= 0.0);
                let finite = |v: &[f64]| -> Value {
                    v.iter()
                        .map(|v| if v.is_finite() { json!(v) } else { Value::Null })
                        .collect()
                };
                let checks: &[&str] = if model.ncon > 0 {
                    &["objective", "bounds", "constraints"]
                } else {
                    &["objective", "bounds"]
                };
                Outcome {
                    claim: if feasible { "feasible" } else { "none" }.into(),
                    feasible: Some(feasible),
                    // Several objectives: no single objective value.
                    objective: None,
                    details: json!({"problem":model.id,"objectives":finite(objectives),
                        "constraints":finite(constraints),"constraint_rule":"feasible when every constraint <= 0"}),
                    evidence: Some(Evidence::rechecked(checks)),
                    x: Some(x),
                    ..Outcome::default()
                }
            }
            Model::QualityDiversity(model) => {
                let (fitness, descriptor) = (model.evaluate)(&x);
                Outcome {
                    claim: if usable(fitness) { "feasible" } else { "none" }.into(),
                    feasible: Some(usable(fitness)),
                    objective: fitness.is_finite().then_some(fitness),
                    details: json!({"problem":model.id,"descriptor":descriptor,
                        "descriptor_lower":model.descriptor_lower,"descriptor_upper":model.descriptor_upper}),
                    evidence: Some(Evidence::rechecked(&["objective", "bounds"])),
                    x: Some(x),
                    ..Outcome::default()
                }
            }
        };
        Ok(TaskResult {
            outcome: Outcome {
                termination: "evaluated".into(),
                ..outcome
            },
            work: [("evaluations".into(), 1)].into(),
        })
    }
}
