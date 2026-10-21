//! First synchronous surrogate profile on any registered scalar model.
use crate::{
    Control,
    registry::{Model, Probe, Problem, Registry, Scalar},
    schema,
    simulation::SimContext,
    threads::Threads,
};
use fcmaes_core::retry_run_seed;
use fcmaes_surrogate::{self as bo, Config, Event, Observation, RunSpec, Space, Value as Observed};
use optimization_job_api::{
    JobSpec, Result,
    artifact::{self, Declaration},
    checked_input,
    result::{Evidence, Outcome},
    storage,
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub const PROFILE: &str = "fcmaes.surrogate/1";
/// Format of the observations snapshot a warm start reads (`input_schemas`).
pub const WARM_SCHEMA: &str = "fcmaes.surrogate.observations/1";
fn yes() -> bool {
    true
}
fn one() -> usize {
    1
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub problem: Problem,
    pub seed: u64,
    pub evaluations: usize,
    #[serde(default = "one")]
    pub runs: usize,
    #[serde(default = "one")]
    pub concurrent: usize,
    #[serde(default = "one")]
    pub evaluation_threads: usize,
    #[serde(default)]
    pub parameters: Config,
    #[serde(default)]
    pub log_scale: Vec<bool>,
    #[serde(default)]
    pub target: Option<f64>,
    #[serde(default = "yes")]
    pub recheck_best: bool,
    #[serde(default)]
    pub warm_start: Option<WarmStart>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmStart {
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub id: String,
    pub version: String,
    pub dimension: usize,
    pub parameters: serde_json::Map<String, Value>,
    pub model_inputs: BTreeMap<String, String>,
    /// The run seed passed to `SimContext` (`retry_run_seed(seed, task)`),
    /// so a snapshot is reusable only by a run with the same simulation seed.
    pub simulation_seed: Option<u64>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmFile {
    pub schema_version: String,
    pub problem: Identity,
    pub usable_for_warm_start: bool,
    pub observations: Vec<Observation>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case", deny_unknown_fields)]
pub enum LogRecord {
    Header {
        profile: String,
        problem: Identity,
        run_seed: u64,
    },
    Event {
        data: Event,
    },
    RecheckCancelled {
        id: u64,
    },
    Recheck {
        id: u64,
        value: Observed,
        agreement: bool,
    },
}

struct Run {
    payload: Payload,
    space: Space,
    objective: Scalar,
    control: Control,
    base: PathBuf,
    threads: usize,
    identity: Identity,
    warm: Vec<Observation>,
    dropped_warm: usize,
}

pub fn describe() -> Value {
    json!({"profile":PROFILE,"release":"0.1.0",
        "method":"exact GP: matern52 or rbf, ARD MAP by bounded L-BFGS with analytic gradients; analytic LogEI or minimization LCB with coordinate refinement; TuRBO-style trust region for two of three proposals (trust_region, default true); synchronous Kriging believer",
        "scope":"deterministic scalar box models, 1..32 variables, new plus reused observations <=256, concurrent 1..8",
        "design":"centre plus seeded maximin Latin hypercube; one trust region without Thompson sampling or local-data restarts; no Sobol, asynchronous or noisy support in this release",
        "resources":"concurrent * evaluation_threads <= inner_threads <=8; outer_workers * inner_threads <=cpu_slots",
        "simulation":"Scalar::simulation receives SimContext: checkpoint, remaining_seconds, threads and stable point seed; currently supported by this profile only",
        "artifacts":{"surrogate-TASK":"records (file surrogate-TASK.jsonl): the evaluation trace, a header, then every start, proposal, completion, cancellation and recheck; readable in pages while the job runs",
            "surrogate-observations-TASK":"json (file surrogate-observations-TASK.json): the compact observations snapshot, replaced after every completion; the warm_start input of a later job"},
        "warm_start":"a named hashed compact observations JSON file; matching model id/version/dimension/parameters and model-input hashes; new job, no solver resume",
        "budget":"evaluations counts new optimization callbacks; optional final independent callback is validation_evaluations",
        "default_parameters":Config::default(),"recheck_best":true})
}

pub fn prepare(registry: &Registry, s: &mut JobSpec, base: &Path) -> Result<Box<dyn Prepared>> {
    let explicit_threshold = s
        .payload
        .get("parameters")
        .and_then(|p| p.get("failure_threshold"))
        .is_some();
    let mut p: Payload = serde_json::from_value(s.payload.clone())?;
    if !explicit_threshold && p.problem.id.starts_with("gtop.") {
        p.parameters.failure_threshold = 1e10;
    }
    p.parameters.validate()?;
    if p.runs == 0
        || p.runs > 1024
        || p.concurrent == 0
        || p.concurrent > bo::MAX_BATCH
        || p.evaluation_threads == 0
        || p.concurrent
            .checked_mul(p.evaluation_threads)
            .is_none_or(|n| n > s.resources.inner_threads)
        || s.resources.inner_threads > 8
        || p.evaluations == 0
        || p.evaluations > bo::MAX_OBSERVATIONS
        || p.target.is_some_and(|t| !t.is_finite())
        || (p.warm_start.is_some() && p.runs != 1)
    {
        return Err("surrogate needs runs=1..1024, concurrent=1..8, evaluation_threads>=1, concurrent * evaluation_threads <= inner_threads <=8, evaluations=1..256; warm starts need runs=1".into());
    }
    let profile_inputs: Vec<&str> = p
        .warm_start
        .as_ref()
        .map(|w| vec![w.input.as_str()])
        .unwrap_or_default();
    let (problem, model) =
        registry.resolve_with_profile_inputs(p.problem.clone(), s, base, &profile_inputs)?;
    let Model::Scalar(model) = model else {
        return Err("fcmaes.surrogate/1 requires a registered scalar model".into());
    };
    p.problem = problem;
    let dimension = p.problem.config.dimension;
    if p.log_scale.is_empty() {
        p.log_scale = vec![false; dimension];
    }
    let space = Space::new(
        p.problem.bounds.lower.clone(),
        p.problem.bounds.upper.clone(),
        p.log_scale.clone(),
    )?;
    let identity = Identity {
        id: p.problem.id.clone(),
        version: p.problem.version.clone(),
        dimension,
        parameters: p.problem.parameters.clone().unwrap_or_default(),
        model_inputs: s
            .inputs
            .iter()
            .filter(|i| !profile_inputs.contains(&i.name.as_str()))
            .map(|i| (i.name.clone(), i.sha256.clone()))
            .collect(),
        // Warm starts need runs=1, so the new run is task 0.
        simulation_seed: model.is_simulation().then(|| retry_run_seed(p.seed, 0)),
    };
    let mut warm = Vec::new();
    let mut dropped_warm = 0;
    if let Some(w) = &p.warm_start {
        let input = s
            .inputs
            .iter()
            .find(|i| i.name == w.input)
            .ok_or("missing warm-start input")?;
        let bytes = checked_input(base, input)?;
        let file: WarmFile = serde_json::from_slice(&bytes)?;
        if file.schema_version != WARM_SCHEMA
            || !file.usable_for_warm_start
            || file.problem != identity
            || file.observations.len() > bo::MAX_OBSERVATIONS
        {
            return Err(
                "warm-start schema, model identity, input hashes or trust flag mismatch".into(),
            );
        }
        let mut ids = std::collections::BTreeSet::new();
        for mut o in file.observations {
            if o.x.len() != dimension
                || o.x.iter().any(|v| !v.is_finite())
                || !ids.insert(o.id)
                || matches!(o.value, Observed::Finite(y) if !y.is_finite())
            {
                return Err("malformed warm-start observation".into());
            }
            if !space.contains(&o.x) {
                dropped_warm += 1;
                continue;
            }
            if matches!(o.value, Observed::Finite(y) if y >= p.parameters.failure_threshold) {
                o.value = Observed::Failed("warm_failure_threshold".into());
            }
            if let Observed::Failed(message) = &mut o.value {
                *message = message.chars().take(256).collect();
            }
            warm.push(o);
        }
        warm.sort_by_key(|o| o.id);
        if warm
            .last()
            .is_some_and(|o| o.id > u64::MAX - bo::MAX_OBSERVATIONS as u64 - 1)
        {
            return Err("warm-start observation id overflow".into());
        }
    }
    if warm.len() + p.evaluations > bo::MAX_OBSERVATIONS {
        return Err("new evaluations plus warm observations must be <=256".into());
    }
    s.payload = serde_json::to_value(&p)?;
    Ok(Box::new(Run {
        payload: p,
        space,
        objective: model,
        control: Control::default(),
        base: base.to_owned(),
        threads: s.resources.inner_threads,
        identity,
        warm,
        dropped_warm,
    }))
}

impl Prepared for Run {
    fn tasks(&self) -> usize {
        self.payload.runs
    }
    fn cancel(&self) {
        self.control.cancel();
    }
    fn solve(&self, task: usize, remaining: f64) -> Result<TaskResult> {
        let threads = Threads::new(self.threads)?;
        threads.install(|| self.task(task, remaining))
    }
}

impl Run {
    fn task(&self, task: usize, remaining: f64) -> Result<TaskResult> {
        let began = Instant::now();
        let stop = || {
            self.control
                .stop()
                .or_else(|| (began.elapsed().as_secs_f64() >= remaining).then_some("time_limit"))
        };
        let log_name = format!("surrogate-{task}.jsonl");
        let warm_name = format!("surrogate-observations-{task}.json");
        let log = self.base.join(&log_name);
        let snapshot_path = self.base.join(&warm_name);
        // Declared before their first write, so agents can read them while
        // the job runs (`jobs artifact`, MCP `get_artifact`).
        artifact::declare(
            &self.base,
            &Declaration::records(format!("surrogate-{task}"), &log_name)
                .task(task)
                .describe("Evaluation trace: header, starts, proposals, completions, cancellations, recheck"),
        )?;
        artifact::declare(
            &self.base,
            &Declaration::json(format!("surrogate-observations-{task}"), &warm_name)
                .task(task)
                .describe("Observations snapshot: the warm_start input of a later job"),
        )?;
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&log)?
            .sync_all()?;
        fs::File::open(&self.base)?.sync_all()?;
        let run_seed = retry_run_seed(self.payload.seed, task);
        let context = SimContext::new(
            self.control.token(),
            began + Duration::from_secs_f64(remaining.max(0.0)),
            self.payload.evaluation_threads,
            run_seed,
        );
        let evaluate = |x: &[f64]| self.objective.value(x, &context.at_point(x));
        let identity = Identity {
            simulation_seed: self.identity.simulation_seed.map(|_| run_seed),
            ..self.identity.clone()
        };
        storage::append_checked(
            &log,
            1,
            &LogRecord::Header {
                profile: PROFILE.into(),
                problem: identity.clone(),
                run_seed,
            },
        )?;
        let mut snapshot = WarmFile {
            schema_version: WARM_SCHEMA.into(),
            problem: identity,
            usable_for_warm_start: true,
            observations: self.warm.clone(),
        };
        storage::write_atomic_json(&snapshot_path, &snapshot)?;
        let mut largest_duration = 0.0_f64;
        let result = {
            let mut sink = |event: &Event| -> bo::Result<()> {
                storage::append_checked(
                    &log,
                    1,
                    &LogRecord::Event {
                        data: event.clone(),
                    },
                )
                .map_err(|e| bo::Error::Log(e.to_string()))?;
                if let Event::Completed {
                    observation,
                    seconds,
                } = event
                {
                    snapshot.observations.push(observation.clone());
                    snapshot.observations.sort_by_key(|o| o.id);
                    largest_duration = largest_duration.max(*seconds);
                    storage::write_atomic_json(&snapshot_path, &snapshot)
                        .map_err(|e| bo::Error::Log(e.to_string()))?;
                }
                Ok(())
            };
            bo::run(
                RunSpec {
                    space: &self.space,
                    config: &self.payload.parameters,
                    evaluations: self.payload.evaluations,
                    concurrent: self.payload.concurrent,
                    seed: run_seed,
                    warm: &self.warm,
                    target: self.payload.target,
                },
                &evaluate,
                &mut sink,
                &stop,
            )?
        };
        let mut outcome = Outcome {
            termination: result.termination.clone(),
            claim: "none".into(),
            evidence: Some(Evidence::claimed(&[])),
            ..Outcome::default()
        };
        let mut validation_evaluations = 0;
        let mut recheck = "no usable point";
        let mut validation_cancelled = false;
        let mut reported_objective = None;
        if let Some(best) = &result.best {
            let Observed::Finite(y) = best.value else {
                unreachable!("best is usable")
            };
            outcome.x = Some(best.x.clone());
            outcome.objective = Some(y);
            outcome.feasible = Some(true);
            outcome.claim = "best_found".into();
            if !self.payload.recheck_best {
                recheck = "disabled";
            } else if stop().is_some()
                || remaining - began.elapsed().as_secs_f64() <= largest_duration.max(0.01)
            {
                recheck = "skipped: cancelled or insufficient remaining time estimate";
            } else {
                validation_evaluations = 1;
                let raw = catch_unwind(AssertUnwindSafe(|| evaluate(&best.x)));
                validation_cancelled =
                    matches!(&raw, Ok(Err(s)) if s == "cancelled") && stop().is_some();
                let value = match raw {
                    Ok(Ok(v)) if v.is_finite() && v < self.payload.parameters.failure_threshold => {
                        Observed::Finite(v)
                    }
                    Ok(_) => Observed::Failed("recheck unusable value".into()),
                    Err(_) => Observed::Failed("recheck panic".into()),
                };
                let agreement = matches!(value, Observed::Finite(v) if v == y);
                if validation_cancelled {
                    recheck = "cancelled during validation; historical claim retained";
                    storage::append_checked(&log, 1, &LogRecord::RecheckCancelled { id: best.id })?;
                } else {
                    storage::append_checked(
                        &log,
                        1,
                        &LogRecord::Recheck {
                            id: best.id,
                            value: value.clone(),
                            agreement,
                        },
                    )?;
                    if agreement {
                        recheck = "exact agreement";
                        outcome.evidence = Some(Evidence::rechecked(&["objective", "bounds"]));
                    } else {
                        recheck = "objective mismatch";
                        reported_objective = Some(y);
                        outcome.claim = if matches!(value, Observed::Finite(_)) {
                            "feasible"
                        } else {
                            "none"
                        }
                        .into();
                        outcome.objective = if let Observed::Finite(v) = value {
                            Some(v)
                        } else {
                            None
                        };
                        outcome.feasible = outcome.objective.map(|_| true);
                        outcome.evidence =
                            Some(Evidence::rechecked(&["objective", "bounds"]).refuting(
                                "best_found",
                                "objective_mismatch",
                                "deterministic best-point re-evaluation disagreed or failed",
                            ));
                        snapshot.usable_for_warm_start = false;
                        storage::write_atomic_json(&snapshot_path, &snapshot)?;
                    }
                }
            }
        }
        outcome.details = json!({"profile_version":"0.1.0","method":"gp","schedule":"synchronous","batch_strategy":"kriging_believer",
            "run_seed":run_seed,"evaluations":result.evaluations,"failed_evaluations":result.failed_evaluations,
            "cancelled_evaluations":result.cancelled_evaluations,
            "evaluation_attempts":result.evaluations+result.cancelled_evaluations,
            "validation_cancelled":validation_cancelled,"evaluation_threads":self.payload.evaluation_threads,
            "warm_start_points":self.warm.len(),"dropped_warm_points":self.dropped_warm,
            "model_fits":result.model_fits,"fit_evaluations":result.fit_evaluations,"model_fallbacks":result.model_fallbacks,
            "last_hyperparameters":result.state.hyper,"proposal_seconds":result.proposal_seconds,
            "evaluation_seconds_sum":result.evaluation_seconds,"recheck":recheck,"reported_objective":reported_objective,
            "log_file":log_name,"warm_start_file":warm_name,
            "artifacts":{"log":format!("surrogate-{task}"),"warm_start":format!("surrogate-observations-{task}")},
            "global_optimum_certified":false});
        outcome
            .check()
            .map_err(|e| format!("surrogate outcome: {e}"))?;
        Ok(TaskResult {
            outcome,
            work: [
                ("evaluations".into(), result.evaluations as u64),
                (
                    "failed_evaluations".into(),
                    result.failed_evaluations as u64,
                ),
                (
                    "cancelled_evaluations".into(),
                    result.cancelled_evaluations as u64,
                ),
                (
                    "evaluation_attempts".into(),
                    (result.evaluations + result.cancelled_evaluations) as u64,
                ),
                ("validation_evaluations".into(), validation_evaluations),
                (
                    "cancelled_validation_evaluations".into(),
                    u64::from(validation_cancelled),
                ),
                ("model_fits".into(), result.model_fits as u64),
            ]
            .into(),
        })
    }
}

/// JSON Schema of the payload, with `problem` the schema of its problems.
pub fn payload_schema(problem: Value) -> Value {
    use crate::schema::{integer, number, or_null, payload, u64};
    let range = |r: &std::ops::RangeInclusive<f64>| number(Some(*r.start()), Some(*r.end()), false);
    let parameters = json!({"type":"object","additionalProperties":false,"properties":{
        "kernel":{"enum":["matern52","rbf"]},"acquisition":{"enum":["log_ei","lcb"]},
        "output_transform":{"enum":["standardize","bilog","auto"]},
        "initial_points":integer(1, 128),"candidates":integer(32, 4096),"fit_restarts":integer(1, 4),
        "fit_iterations":integer(1, bo::gp::MAX_FIT_ITERATIONS as u64),"refit_interval":integer(1, 64),
        "lcb_beta":range(&(0.0..=20.0)),"minimum_distance":range(&(1e-12..=0.01)),
        "failure_threshold":number(Some(0.0), None, true),"nugget":range(&bo::gp::NUGGET_RANGE),
        "trust_region":{"type":"boolean"}}});
    let batch = bo::MAX_BATCH as u64;
    payload(
        PROFILE,
        json!({"type":"object","additionalProperties":false,"required":["problem","seed","evaluations"],
        "properties":{"problem":problem,"seed":u64(),"evaluations":integer(1, bo::MAX_OBSERVATIONS as u64),
            "runs":integer(1, 1024),"concurrent":integer(1, batch),"evaluation_threads":integer(1, batch),
            "parameters":parameters,"log_scale":{"type":"array","items":{"type":"boolean"}},
            "target":or_null(json!({"type":"number"})),"recheck_best":{"type":"boolean"},
            "warm_start":or_null(json!({"type":"object","additionalProperties":false,"required":["input"],
                "properties":{"input":{"type":"string"}}}))},
        "if":{"required":["warm_start"],"properties":{"warm_start":{"type":"object"}}},
        "then":{"properties":{"runs":{"const":1}}},
        "$comment":"concurrent times evaluation_threads <= inner_threads <= 8; log_scale is empty or has one entry per variable, and a log-scaled variable needs a positive lower bound; evaluations plus warm-start observations <= 256; warm_start.input names an envelope input in the format fcmaes.surrogate.observations/1 (input_schemas) for the same model, parameters, model inputs and, for simulation models, run seed; gtop models default to failure_threshold 1e10."}),
    )
}

/// JSON Schema of the warm-start observations snapshot.
pub fn warm_start_schema() -> Value {
    use crate::schema::{DRAFT, integer, or_null, u64};
    let value = |kind: &str, value: Value| {
        json!({"type":"object","additionalProperties":false,"required":["kind","value"],
            "properties":{"kind":{"const":kind},"value":value}})
    };
    json!({"$schema":DRAFT,"$id":format!("urn:optimization-jobs:input:{WARM_SCHEMA}"),
        "type":"object","additionalProperties":false,
        "required":["schema_version","problem","usable_for_warm_start","observations"],
        "properties":{"schema_version":{"const":WARM_SCHEMA},
            "problem":{"type":"object","additionalProperties":false,
                "required":["id","version","dimension","parameters","model_inputs"],
                "properties":{"id":{"type":"string"},"version":{"type":"string"},
                    "dimension":integer(1, bo::MAX_DIMENSION as u64),"parameters":{"type":"object"},
                    "model_inputs":{"type":"object","additionalProperties":{"type":"string","pattern":"^[0-9a-f]{64}$"}},
                    "simulation_seed":or_null(u64())}},
            "usable_for_warm_start":{"const":true},
            "observations":{"type":"array","maxItems":bo::MAX_OBSERVATIONS,
                "items":{"type":"object","additionalProperties":false,"required":["id","x","value"],
                    "properties":{"id":u64(),"x":{"type":"array","items":{"type":"number"}},
                        "value":{"oneOf":[value("finite", json!({"type":"number"})),value("failed", json!({"type":"string"}))]}}}}},
        "$comment":"the surrogate-observations-TASK artifact of an earlier job; its problem must equal the new job's resolved model identity (parameters with defaults, SHA-256 of the model inputs, simulation run seed); observation ids are unique and each x has dimension values"})
}

/// The `examples` entry of this profile on `probe`.
pub fn example(probe: &Probe) -> Value {
    schema::example(
        PROFILE,
        format!(
            "exact-GP surrogate on {}: 8 evaluations, 4 of them initial",
            probe.id
        ),
        probe,
        json!({"problem":probe.problem(),"seed":1,"evaluations":8,
            "parameters":{"initial_points":4,"candidates":64,"fit_restarts":1,"fit_iterations":10}}),
    )
}
