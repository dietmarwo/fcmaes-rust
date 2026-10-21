//! `fcmaes.evaluate/1`: evaluates given points on a registered model without
//! optimizing, so a caller (an agent, a person or another solver) can have
//! its own candidates checked. One task per point.
use crate::{
    Control,
    registry::{Model, Problem, Registry},
    simulation::{Evaluator, Failure},
    threads::Threads,
};
use optimization_job_api::{
    JobSpec, Result,
    result::{Evidence, Outcome},
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

pub const PROFILE: &str = "fcmaes.evaluate/1";
/// Most points one job may evaluate (one task each).
const MAX_POINTS: usize = 1_000_000;
/// What a scalar or quality-diversity outcome's `details.constraints` says:
/// such a model declares no constraints, and a penalty it adds to its
/// objective is part of the value, not a checked constraint.
pub const NO_CONSTRAINTS: &str = "none declared; penalties inside the objective are not checked";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub problem: Problem,
    pub points: Vec<Vec<f64>>,
    /// `Scalar::simulation` models: the run seed their `SimContext` point
    /// seeds derive from (default 0). A retry's or surrogate run's
    /// `details.run_seed` re-evaluates its points with the same seeds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_seed: Option<u64>,
}

pub struct Run {
    payload: Payload,
    model: Model,
    control: Control,
}

pub fn describe() -> Value {
    json!({"profile":PROFILE,"problems":"every registered model, Scalar::simulation models included","inner_threads":"1; one point per task",
        "points":"1 or more points (inline: the resolved job, points included, must stay below the runner's 4 MiB validation output), each of the problem's dimension and inside its bounds; one task per point",
        "result":"claim feasible: a scalar or quality-diversity model gave a usable value at a point inside the bounds; such models declare no constraints, so a penalty inside the objective is not checked (details.constraints says so); a multi-objective model: every declared constraint <= 0; objectives, constraints or descriptor in details",
        "simulation":"Scalar::simulation models get a SimContext with threads() 1, the seed derived from run_seed (default 0) and the point, the remaining time and the job's cancellation; a failed evaluation gives claim none with details.error; a cancelled one termination cancelled",
        "run_seed":"optional, simulation models only: a retry's or surrogate run's details.run_seed re-evaluates its points with the same point seeds"})
}

pub fn prepare(registry: &Registry, s: &mut JobSpec, base: &Path) -> Result<Box<dyn Prepared>> {
    let mut p: Payload = serde_json::from_value(s.payload.clone())?;
    if s.resources.inner_threads != 1 {
        return Err(
            "fcmaes.evaluate/1 evaluates one point per task and needs inner_threads=1; outer_workers evaluates points in parallel"
                .into(),
        );
    }
    let (problem, model) = registry.resolve(p.problem.clone(), s, base)?;
    if p.run_seed.is_some() && !matches!(&model, Model::Scalar(m) if m.is_simulation()) {
        return Err(format!(
            "run_seed applies only to Scalar::simulation models; {} is not one",
            problem.id
        )
        .into());
    }
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
    Ok(Box::new(Run {
        payload: p,
        model,
        control: Control::default(),
    }))
}

impl Prepared for Run {
    fn tasks(&self) -> usize {
        self.payload.points.len()
    }
    /// Tasks stop between points; a simulation also at its checkpoints.
    fn cancel(&self) {
        self.control.cancel();
    }
    fn solve(&self, task: usize, remaining: f64) -> Result<TaskResult> {
        // The whole task runs on its own threads, so no Rayon work of the
        // task (or of the model) reaches the global pool.
        let threads = Threads::new(1)?;
        threads.install(|| self.task(task, remaining, &threads))
    }
}

impl Run {
    fn task(&self, task: usize, remaining: f64, _: &Threads) -> Result<TaskResult> {
        let x = self.payload.points[task].clone();
        let id = &self.payload.problem.id;
        // A value at or above 1e99 is the fcmaes penalty: no usable value.
        let usable = |v: f64| v.is_finite() && v < 1e99;
        let mut work = [("evaluations".to_string(), 1)].into();
        let outcome = match &self.model {
            Model::Scalar(model) => {
                let evaluator = Evaluator::new(
                    model,
                    &self.control,
                    remaining,
                    self.payload.run_seed.unwrap_or(0),
                );
                let value = evaluator.try_value(&x);
                if evaluator.is_simulation() {
                    let (failed, cancelled) = match &value {
                        Err(Failure::Failed(_)) => (1, 0),
                        Err(Failure::Cancelled) => (0, 1),
                        Ok(_) => (0, 0),
                    };
                    work = [
                        ("evaluations".to_string(), 1 - cancelled),
                        ("failed_evaluations".to_string(), failed),
                        ("cancelled_evaluations".to_string(), cancelled),
                    ]
                    .into();
                }
                match value {
                    Ok(value) => Outcome {
                        claim: if usable(value) { "feasible" } else { "none" }.into(),
                        feasible: Some(usable(value)),
                        objective: value.is_finite().then_some(value),
                        details: json!({"problem":id,"constraints":NO_CONSTRAINTS}),
                        evidence: Some(Evidence::rechecked(&["objective", "bounds"])),
                        x: Some(x),
                        ..Outcome::default()
                    },
                    Err(Failure::Failed(error)) => Outcome {
                        claim: "none".into(),
                        feasible: Some(false),
                        details: json!({"problem":id,"error":error,"constraints":NO_CONSTRAINTS}),
                        evidence: Some(Evidence::claimed(&[])),
                        x: Some(x),
                        ..Outcome::default()
                    },
                    Err(Failure::Cancelled) => {
                        return Ok(TaskResult {
                            outcome: Outcome {
                                termination: "cancelled".into(),
                                claim: "none".into(),
                                details: json!({"problem":id,"error":"cancelled"}),
                                evidence: Some(Evidence::claimed(&[])),
                                x: Some(x),
                                ..Outcome::default()
                            },
                            work,
                        });
                    }
                }
            }
            Model::MultiObjective(model) => {
                let values = model.values(&x)?;
                let (objectives, constraints) = values.split_at(model.objectives);
                let feasible =
                    values.iter().all(|v| v.is_finite()) && constraints.iter().all(|c| *c <= 0.0);
                let finite = |v: &[f64]| -> Value {
                    v.iter()
                        .map(|v| if v.is_finite() { json!(v) } else { Value::Null })
                        .collect()
                };
                let checks: &[&str] = if model.constraints > 0 {
                    &["objective", "bounds", "constraints"]
                } else {
                    &["objective", "bounds"]
                };
                Outcome {
                    claim: if feasible { "feasible" } else { "none" }.into(),
                    feasible: Some(feasible),
                    // Several objectives: no single objective value.
                    objective: None,
                    details: json!({"problem":id,"objectives":finite(objectives),
                        "constraints":finite(constraints),"constraint_rule":"feasible when every constraint <= 0"}),
                    evidence: Some(Evidence::rechecked(checks)),
                    x: Some(x),
                    ..Outcome::default()
                }
            }
            Model::QualityDiversity(model) => {
                let (fitness, descriptor) = model.values(&x)?;
                Outcome {
                    claim: if usable(fitness) { "feasible" } else { "none" }.into(),
                    feasible: Some(usable(fitness)),
                    objective: fitness.is_finite().then_some(fitness),
                    details: json!({"problem":id,"descriptor":descriptor,
                        "descriptor_lower":model.descriptor_lower,"descriptor_upper":model.descriptor_upper,
                        "constraints":NO_CONSTRAINTS}),
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
            work,
        })
    }
}
