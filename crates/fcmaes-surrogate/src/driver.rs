use crate::{
    Config, Error, MAX_BATCH, MAX_OBSERVATIONS, Result, Round, Space, StrategyState, propose,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Value {
    Finite(f64),
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub id: u64,
    pub x: Vec<f64>,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Start {
        schema: String,
        space: Space,
        config: Config,
        seed: u64,
        concurrent: usize,
        evaluations: usize,
        target: Option<f64>,
        warm: Vec<Observation>,
    },
    Proposal {
        round: u64,
        history_count: usize,
        ids: Vec<u64>,
        points: Vec<Vec<f64>>,
        state: StrategyState,
        fallback: Option<String>,
    },
    Started {
        id: u64,
        x: Vec<f64>,
    },
    Completed {
        observation: Observation,
        seconds: f64,
    },
    Cancelled {
        id: u64,
        x: Vec<f64>,
        seconds: f64,
    },
    BatchCommitted {
        ids: Vec<u64>,
    },
    Finished {
        termination: String,
        evaluations: usize,
        failures: usize,
        cancelled: usize,
    },
}

pub struct RunSpec<'a> {
    pub space: &'a Space,
    pub config: &'a Config,
    pub evaluations: usize,
    pub concurrent: usize,
    pub seed: u64,
    pub warm: &'a [Observation],
    pub target: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResult {
    pub history: Vec<Observation>,
    pub best: Option<Observation>,
    pub termination: String,
    pub evaluations: usize,
    pub failed_evaluations: usize,
    pub model_fits: usize,
    pub cancelled_evaluations: usize,
    pub fit_evaluations: usize,
    pub model_fallbacks: usize,
    pub proposal_seconds: f64,
    pub evaluation_seconds: f64,
    pub state: StrategyState,
}

enum EvaluationCompletion {
    Completed(Observation, f64),
    Cancelled(f64),
    NotStarted,
}

fn best(history: &[Observation]) -> Option<Observation> {
    let mut best: Option<Observation> = None;
    for observation in history {
        if let Value::Finite(value) = observation.value
            && best.as_ref().is_none_or(|b| match b.value {
                Value::Finite(y) => value < y,
                _ => true,
            })
        {
            best = Some(observation.clone());
        }
    }
    best
}
fn classify(
    result: std::thread::Result<std::result::Result<f64, String>>,
    threshold: f64,
) -> Value {
    match result {
        Ok(Ok(y)) if y.is_finite() && y < threshold => Value::Finite(y),
        Ok(Ok(y)) => Value::Failed(
            if y.is_finite() {
                "failure_threshold"
            } else {
                "nonfinite"
            }
            .into(),
        ),
        Ok(Err(message)) => Value::Failed(format!(
            "evaluator: {}",
            message.chars().take(256).collect::<String>()
        )),
        Err(_) => Value::Failed("panic".into()),
    }
}

/// Synchronous evaluation batches with immediate, fallible completion logging.
/// The sink is serialized under a mutex; completions may be logged out of order,
/// but GP history is committed in proposal order. No new evaluation starts
/// after its stop check fails; already executing callbacks must finish.
pub fn run(
    spec: RunSpec<'_>,
    evaluator: &(dyn Fn(&[f64]) -> std::result::Result<f64, String> + Sync),
    sink: &mut (dyn FnMut(&Event) -> Result<()> + Send),
    stop: &(dyn Fn() -> Option<&'static str> + Sync),
) -> Result<RunResult> {
    spec.space.validate()?;
    spec.config.validate()?;
    if spec.evaluations == 0
        || spec.concurrent == 0
        || spec.concurrent > MAX_BATCH
        || spec.evaluations.saturating_add(spec.warm.len()) > MAX_OBSERVATIONS
        || spec.target.is_some_and(|v| !v.is_finite())
    {
        return Err(Error::Invalid(
            "positive evaluation/batch budget and total observations <=256 required".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut history = spec.warm.to_vec();
    for o in &history {
        if !spec.space.contains(&o.x)
            || !ids.insert(o.id)
            || matches!(o.value, Value::Finite(y) if !y.is_finite() || y >= spec.config.failure_threshold)
        {
            return Err(Error::Invalid("invalid warm-start observations".into()));
        }
    }
    let mut next_id = history
        .iter()
        .map(|o| o.id)
        .max()
        .map_or(Some(0), |i| i.checked_add(1))
        .ok_or_else(|| Error::Invalid("observation id overflow".into()))?;
    let mut state = StrategyState::default();
    let mut evaluations = 0;
    let mut failures = 0;
    let mut cancelled_evaluations = 0;
    let mut model_fits = 0;
    let mut fit_evaluations = 0;
    let mut model_fallbacks = 0;
    let mut proposal_seconds = 0.0;
    let mut evaluation_seconds = 0.0;
    sink(&Event::Start {
        schema: "fcmaes.surrogate.events/1".into(),
        space: spec.space.clone(),
        config: spec.config.clone(),
        seed: spec.seed,
        concurrent: spec.concurrent,
        evaluations: spec.evaluations,
        target: spec.target,
        warm: history.clone(),
    })?;
    let sink = Mutex::new(sink);
    let log_failed = AtomicBool::new(false);
    let emit = |event: &Event| -> Result<()> {
        let result =
            (sink
                .lock()
                .map_err(|_| Error::Log("event sink mutex poisoned".into()))?)(event);
        if result.is_err() {
            log_failed.store(true, Ordering::Release);
        }
        result
    };
    let mut termination = "evaluation_limit";
    while evaluations < spec.evaluations {
        if let Some(reason) = stop() {
            termination = reason;
            break;
        }
        if let Some(Value::Finite(y)) = best(&history).map(|o| o.value)
            && spec.target.is_some_and(|target| y <= target)
        {
            termination = "objective_limit";
            break;
        }
        let mut count = spec.concurrent.min(spec.evaluations - evaluations);
        if history.len() < spec.config.initial_points {
            count = count.min(spec.config.initial_points - history.len());
        }
        let began = Instant::now();
        let proposal = match propose(
            &Round {
                space: spec.space,
                history: &history,
                pending: &[],
                state: &state,
                config: spec.config,
                seed: spec.seed,
                count,
            },
            &|| stop().is_some(),
        ) {
            Ok(p) => p,
            Err(Error::Stopped) => {
                termination = stop().unwrap_or("cancelled");
                break;
            }
            Err(e) => return Err(e),
        };
        proposal_seconds += began.elapsed().as_secs_f64();
        if let Some(reason) = stop() {
            termination = reason;
            break;
        }
        let task_ids: Vec<u64> = (0..count)
            .map(|i| {
                next_id
                    .checked_add(i as u64)
                    .ok_or_else(|| Error::Invalid("observation id overflow".into()))
            })
            .collect::<Result<_>>()?;
        next_id = next_id
            .checked_add(count as u64)
            .ok_or_else(|| Error::Invalid("observation id overflow".into()))?;
        emit(&Event::Proposal {
            round: state.round,
            history_count: history.len(),
            ids: task_ids.clone(),
            points: proposal.points.clone(),
            state: proposal.state.clone(),
            fallback: proposal.fallback.clone(),
        })?;
        model_fits += proposal.model_fits;
        fit_evaluations += proposal.fit_evaluations;
        model_fallbacks += usize::from(proposal.fallback.is_some());
        state = proposal.state;
        let completed: Vec<Result<EvaluationCompletion>> = proposal
            .points
            .par_iter()
            .zip(&task_ids)
            .map(|(x, &id)| {
                if stop().is_some() || log_failed.load(Ordering::Acquire) {
                    return Ok(EvaluationCompletion::NotStarted);
                }
                emit(&Event::Started { id, x: x.clone() })?;
                let start = Instant::now();
                let raw = catch_unwind(AssertUnwindSafe(|| evaluator(x)));
                let seconds = start.elapsed().as_secs_f64();
                if matches!(&raw, Ok(Err(s)) if s == "cancelled") && stop().is_some() {
                    emit(&Event::Cancelled {
                        id,
                        x: x.clone(),
                        seconds,
                    })?;
                    return Ok(EvaluationCompletion::Cancelled(seconds));
                }
                let value = classify(raw, spec.config.failure_threshold);
                let observation = Observation {
                    id,
                    x: x.clone(),
                    value,
                };
                // Must be durable before waiting for slower evaluations in this batch.
                emit(&Event::Completed {
                    observation: observation.clone(),
                    seconds,
                })?;
                Ok(EvaluationCompletion::Completed(observation, seconds))
            })
            .collect();
        let mut committed = Vec::new();
        for result in completed {
            match result? {
                EvaluationCompletion::Cancelled(seconds) => {
                    cancelled_evaluations += 1;
                    evaluation_seconds += seconds;
                }
                EvaluationCompletion::NotStarted => (),
                EvaluationCompletion::Completed(o, seconds) => {
                    failures += usize::from(matches!(o.value, Value::Failed(_)));
                    evaluations += 1;
                    evaluation_seconds += seconds;
                    committed.push(o.id);
                    history.push(o);
                }
            }
        }
        emit(&Event::BatchCommitted { ids: committed })?;
        if let Some(reason) = stop() {
            termination = reason;
            break;
        }
    }
    emit(&Event::Finished {
        termination: termination.into(),
        evaluations,
        failures,
        cancelled: cancelled_evaluations,
    })?;
    Ok(RunResult {
        best: best(&history),
        history,
        termination: termination.into(),
        evaluations,
        failed_evaluations: failures,
        cancelled_evaluations,
        model_fits,
        fit_evaluations,
        model_fallbacks,
        proposal_seconds,
        evaluation_seconds,
        state,
    })
}

/// Reconstructs synchronous decision history from a complete event stream and
/// checks every proposal and its state. Timing and completion arrival order
/// are intentionally not used for proposal decisions.
pub fn replay(events: &[Event]) -> Result<()> {
    let Some(Event::Start {
        schema,
        space,
        config,
        seed,
        warm,
        concurrent,
        evaluations,
        target,
    }) = events.first()
    else {
        return Err(Error::Invalid("replay needs a start event".into()));
    };
    space.validate()?;
    config.validate()?;
    if schema != "fcmaes.surrogate.events/1"
        || *concurrent == 0
        || *concurrent > MAX_BATCH
        || *evaluations == 0
        || evaluations.saturating_add(warm.len()) > MAX_OBSERVATIONS
        || target.is_some_and(|t| !t.is_finite())
    {
        return Err(Error::Invalid("invalid replay header".into()));
    }
    let mut history = warm.clone();
    let mut seen = BTreeSet::new();
    for o in warm {
        if !space.contains(&o.x)
            || !seen.insert(o.id)
            || matches!(o.value, Value::Finite(y) if !y.is_finite() || y >= config.failure_threshold)
        {
            return Err(Error::Invalid("invalid replay warm observations".into()));
        }
    }
    let mut next_id = seen
        .last()
        .map_or(Some(0), |id| id.checked_add(1))
        .ok_or_else(|| Error::Invalid("replay id overflow".into()))?;
    let mut state = StrategyState::default();
    let mut pending = std::collections::BTreeMap::new();
    let mut completed = std::collections::BTreeMap::new();
    let mut started = BTreeSet::new();
    let mut cancelled = BTreeSet::new();
    let mut total_failures = 0;
    let mut total_cancelled = 0;
    let mut finished = false;
    for (index, event) in events[1..].iter().enumerate() {
        match event {
            Event::Proposal {
                round,
                history_count,
                ids,
                points,
                state: expected,
                fallback,
            } => {
                let mut count = concurrent
                    .min(&(evaluations - (history.len() - warm.len())))
                    .to_owned();
                if history.len() < config.initial_points {
                    count = count.min(config.initial_points - history.len());
                }
                if !pending.is_empty()
                    || *round != state.round
                    || *history_count != history.len()
                    || ids.len() != points.len()
                    || count != points.len()
                {
                    return Err(Error::Invalid("replay proposal ordering mismatch".into()));
                }
                let p = propose(
                    &Round {
                        space,
                        config,
                        seed: *seed,
                        history: &history,
                        pending: &[],
                        state: &state,
                        count: points.len(),
                    },
                    &|| false,
                )?;
                if p.points != *points || p.state != *expected || p.fallback != *fallback {
                    return Err(Error::Invalid("replay proposal/state mismatch".into()));
                }
                for (&id, x) in ids.iter().zip(points) {
                    if id != next_id || !seen.insert(id) || pending.insert(id, x.clone()).is_some()
                    {
                        return Err(Error::Invalid("duplicate proposed id".into()));
                    }
                    next_id = next_id
                        .checked_add(1)
                        .ok_or_else(|| Error::Invalid("replay id overflow".into()))?;
                }
                state = p.state;
            }
            Event::Started { id, x } => {
                if pending.get(id) != Some(x) || !started.insert(*id) {
                    return Err(Error::Invalid(
                        "start not matched to a unique proposal".into(),
                    ));
                }
            }
            Event::Completed {
                observation,
                seconds,
            } => {
                if pending.get(&observation.id) != Some(&observation.x)
                    || !started.contains(&observation.id)
                    || cancelled.contains(&observation.id)
                    || !seconds.is_finite()
                    || *seconds < 0.0
                    || matches!(observation.value, Value::Finite(y) if !y.is_finite() || y >= config.failure_threshold)
                    || completed
                        .insert(observation.id, observation.clone())
                        .is_some()
                {
                    return Err(Error::Invalid("completion not matched to proposal".into()));
                }
                total_failures += usize::from(matches!(observation.value, Value::Failed(_)));
            }
            Event::Cancelled { id, x, seconds } => {
                if pending.get(id) != Some(x)
                    || !started.contains(id)
                    || completed.contains_key(id)
                    || !cancelled.insert(*id)
                    || !seconds.is_finite()
                    || *seconds < 0.0
                {
                    return Err(Error::Invalid(
                        "cancellation not matched to a running evaluation".into(),
                    ));
                }
                total_cancelled += 1;
            }
            Event::BatchCommitted { ids } => {
                if pending.is_empty()
                    || *ids != completed.keys().copied().collect::<Vec<_>>()
                    || started.len() != completed.len() + cancelled.len()
                {
                    return Err(Error::Invalid(
                        "batch commit is incomplete or out of proposal order".into(),
                    ));
                }
                for id in ids {
                    history.push(
                        completed
                            .remove(id)
                            .ok_or_else(|| Error::Invalid("commit without completion".into()))?,
                    );
                }
                if !completed.is_empty() {
                    return Err(Error::Invalid("uncommitted completion".into()));
                }
                pending.clear();
                started.clear();
                cancelled.clear();
            }
            Event::Start { .. } => return Err(Error::Invalid("duplicate start event".into())),
            Event::Finished {
                termination,
                evaluations: actual,
                failures,
                cancelled,
            } => {
                if index + 2 != events.len()
                    || !pending.is_empty()
                    || *actual != history.len() - warm.len()
                    || *failures != total_failures
                    || *cancelled != total_cancelled
                    || termination.is_empty()
                    || (termination == "evaluation_limit" && actual != evaluations)
                {
                    return Err(Error::Invalid(
                        "invalid replay finish or work counts".into(),
                    ));
                }
                finished = true;
            }
        }
    }
    if finished {
        Ok(())
    } else {
        Err(Error::Invalid(
            "incomplete event stream: missing finish".into(),
        ))
    }
}
