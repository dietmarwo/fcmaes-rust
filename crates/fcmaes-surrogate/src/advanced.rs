//! Opt-in deterministic-box optimizer with bounded local fitting and async work.
//! The original `run`/`propose` API and serialized v1 events are unchanged.
use crate::{
    Config, Error, Observation, Result, Space, Value, design,
    gp::{self, ExactGp},
    sampling,
    strategy::{GLOBAL_EVERY, TR_INITIAL, TR_MAX, TR_MIN, TR_SUCCESSES},
    stream_seed,
    transform::Transform,
};
use fcmaes_core::{BiteOpt, BiteParams, Rng};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Acquisition {
    #[default]
    LogEi,
    Thompson,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Schedule {
    #[default]
    Synchronous,
    Asynchronous,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Design {
    #[default]
    LatinHypercube,
    SobolDigitalShift,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub design: Design,
    pub base: Config,
    pub training_cap: usize,
    pub archive_cap: usize,
    pub schedule: Schedule,
    pub acquisition: Acquisition,
    pub features: usize,
    pub restart_failures: usize,
    pub surrogate_threads: usize,
    pub evaluation_threads: usize,
    pub concurrent: usize,
    pub governor: bool,
    pub handoff_after: Option<usize>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            design: Design::LatinHypercube,
            base: Config::default(),
            training_cap: 128,
            archive_cap: 10000,
            schedule: Schedule::Synchronous,
            acquisition: Acquisition::LogEi,
            features: 512,
            restart_failures: 32,
            surrogate_threads: 1,
            evaluation_threads: 1,
            concurrent: 1,
            governor: false,
            handoff_after: None,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        self.base.validate()?;
        if !(8..=240).contains(&self.training_cap)
            || !(self.training_cap..=10000).contains(&self.archive_cap)
            || !(1..=8).contains(&self.concurrent)
            || !(1..=8).contains(&self.surrogate_threads)
            || !(1..=8).contains(&self.evaluation_threads)
            || !(32..=4096).contains(&self.features)
            || self.base.initial_points > self.training_cap
            || self.base.acquisition != crate::strategy::Acquisition::LogEi
            || self.restart_failures < 4
            || self.handoff_after == Some(0)
        {
            return Err(Error::Invalid(
                "advanced resource/configuration limits".into(),
            ));
        }
        Ok(())
    }
    pub fn required_threads(&self) -> usize {
        self.concurrent * self.evaluation_threads
            + if self.schedule == Schedule::Asynchronous {
                self.surrogate_threads
            } else {
                0
            }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub observation: Observation,
    pub region: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Diagnostics {
    pub model_version: u64,
    pub region: u64,
    pub training_ids: Vec<u64>,
    pub jitter: Option<f64>,
    pub pivot_ratio: Option<f64>,
    pub fitted: bool,
    pub reused: bool,
    pub fallback: Option<String>,
    pub candidates: usize,
    pub refit_interval: usize,
    /// Side of the trust region in unit coordinates before lengthscale
    /// shaping, as used by this proposal.
    pub region_length: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Start {
        schema: String,
        space: Space,
        options: Options,
        seed: u64,
        budget: usize,
        warm: Vec<Sample>,
    },
    Proposal {
        request: u64,
        seconds: f64,
        history_ids: Vec<u64>,
        pending: Vec<Vec<f64>>,
        points: Vec<Vec<f64>>,
        diagnostics: Diagnostics,
    },
    Started {
        id: u64,
        x: Vec<f64>,
        region: u64,
    },
    Completed {
        sample: Sample,
        seconds: f64,
    },
    Cancelled {
        id: u64,
        x: Vec<f64>,
        seconds: f64,
    },
    Abandoned {
        request: u64,
        reason: String,
    },
    Handoff {
        after: usize,
        method: String,
    },
    Cached {
        source_id: u64,
        x: Vec<f64>,
    },
    Finished {
        termination: String,
        completed: usize,
        failed: usize,
        cancelled: usize,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResult {
    pub history: Vec<Sample>,
    pub best: Option<Observation>,
    pub termination: String,
    pub evaluations: usize,
    pub failed_evaluations: usize,
    pub cancelled_evaluations: usize,
    pub proposal_seconds: f64,
    pub evaluation_seconds: f64,
    pub proposals: usize,
    pub peak_pending: usize,
}
pub struct RunSpec<'a> {
    pub space: &'a Space,
    pub options: &'a Options,
    pub evaluations: usize,
    pub seed: u64,
    pub warm: &'a [Sample],
    pub target: Option<f64>,
    pub inner_threads: usize,
}

#[derive(Clone)]
struct Cached {
    gp: ExactGp,
    xs: Vec<Vec<f64>>,
    ids: Vec<u64>,
}
pub struct Engine {
    space: Space,
    options: Options,
    seed: u64,
    round: u64,
    region: u64,
    region_start: u64,
    processed: BTreeSet<u64>,
    stalls: usize,
    best: Option<f64>,
    length: f64,
    successes: usize,
    failures: usize,
    cache: Option<Cached>,
    last_cost: f64,
    eval_cost: f64,
}
impl Engine {
    pub fn new(space: Space, options: Options, seed: u64) -> Result<Self> {
        space.validate()?;
        options.validate()?;
        Ok(Self {
            space,
            options,
            seed,
            round: 0,
            region: 0,
            region_start: 0,
            processed: BTreeSet::new(),
            stalls: 0,
            best: None,
            length: TR_INITIAL,
            successes: 0,
            failures: 0,
            cache: None,
            last_cost: 0.0,
            eval_cost: 0.0,
        })
    }
    pub fn observe_cost(&mut self, evaluation_seconds: f64) {
        self.eval_cost = evaluation_seconds;
    }
    /// Pure numerical proposal within one engine. Full archive suppresses
    /// duplicates even when its observations are outside the active GP.
    pub fn propose(
        &mut self,
        history: &[Sample],
        pending: &[Vec<f64>],
        count: usize,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<(Vec<Vec<f64>>, Diagnostics)> {
        if count == 0
            || count > 8
            || history.len() + pending.len() + count > self.options.archive_cap
        {
            return Err(Error::Invalid("advanced archive/proposal cap".into()));
        }
        if stop() {
            return Err(Error::Stopped);
        }
        let began = Instant::now();
        let mut ids = BTreeSet::new();
        let mut occupied = Vec::new();
        for s in history {
            if !ids.insert(s.observation.id) {
                return Err(Error::Invalid("duplicate archive ID".into()));
            }
            occupied.push(self.space.to_unit(&s.observation.x)?);
            if let Value::Finite(y) = s.observation.value
                && (!y.is_finite() || y >= self.options.base.failure_threshold)
            {
                return Err(Error::Invalid("unclassified archive".into()));
            }
        }
        for p in pending {
            occupied.push(self.space.to_unit(p)?);
        }
        // Process observations in ID order, not completion order, so that the
        // region counters and the GP's training order do not depend on which
        // callback of a synchronous batch finished first.
        let mut usable: Vec<_> = history
            .iter()
            .filter(|s| matches!(s.observation.value, Value::Finite(_)))
            .collect();
        usable.sort_by_key(|s| s.observation.id);
        let ys: Vec<_> = usable
            .iter()
            .map(|s| match s.observation.value {
                Value::Finite(y) => y,
                _ => unreachable!(),
            })
            .collect();
        let spread = if ys.is_empty() {
            1.0
        } else {
            let lo = ys.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (hi - lo).abs().max(f64::MIN_POSITIVE)
        };
        for s in &usable {
            if s.region == self.region && self.processed.insert(s.observation.id) {
                let Value::Finite(y) = s.observation.value else {
                    unreachable!()
                };
                if self.best.is_none_or(|b| y < b - 1e-3 * spread) {
                    self.best = Some(y);
                    self.stalls = 0;
                    self.successes += 1;
                    self.failures = 0;
                } else {
                    self.stalls += 1;
                    self.successes = 0;
                    self.failures += 1;
                }
                // The TuRBO-1 side-length rule of the original method, applied
                // per completed evaluation (batch size one).
                if self.successes >= TR_SUCCESSES as usize {
                    self.length = (self.length * 2.0).min(TR_MAX);
                    self.successes = 0;
                } else if self.failures >= 4.max(self.space.dimension()) {
                    self.length *= 0.5;
                    self.failures = 0;
                }
                if self.length < TR_MIN {
                    self.length = TR_INITIAL;
                }
            }
        }
        if self.stalls >= self.options.restart_failures {
            self.region += 1;
            self.region_start = history
                .iter()
                .map(|s| s.observation.id)
                .max()
                .map_or(0, |v| v + 1);
            self.stalls = 0;
            self.successes = 0;
            self.failures = 0;
            self.best = None;
            self.length = TR_INITIAL;
            self.cache = None;
        }
        let local: Vec<_> = usable
            .iter()
            .copied()
            .filter(|s| s.observation.id >= self.region_start)
            .collect();
        let initial = local.len() < self.options.base.initial_points;
        let chosen = select_training(&self.space, &local, self.options.training_cap)?;
        let train_ids: Vec<_> = chosen.iter().map(|s| s.observation.id).collect();
        let xs: Vec<_> = chosen
            .iter()
            .map(|s| self.space.to_unit(&s.observation.x))
            .collect::<Result<_>>()?;
        let raw: Vec<_> = chosen
            .iter()
            .map(|s| match s.observation.value {
                Value::Finite(y) => y,
                _ => unreachable!(),
            })
            .collect();
        let mut model = None;
        let mut fitted = false;
        let mut reused = false;
        let mut fallback = None;
        let interval = if self.options.governor && self.last_cost > self.eval_cost.max(1e-6) * 0.2 {
            self.options.base.refit_interval.max(8)
        } else {
            self.options.base.refit_interval
        };
        let candidates = if self.options.governor && self.last_cost > self.eval_cost.max(1e-6) * 0.2
        {
            self.options.base.candidates.min(128)
        } else {
            self.options.base.candidates
        };
        let mut best = 0.0;
        if !initial && raw.len() >= 2 {
            let transform = Transform::fit(&raw, self.options.base.output_transform)?;
            let target: Vec<_> = raw.iter().map(|y| transform.apply(*y)).collect();
            best = target.iter().copied().fold(f64::INFINITY, f64::min);
            let refit = self.cache.is_none() || self.round.is_multiple_of(interval as u64);
            let result = if refit {
                fitted = true;
                gp::fit(
                    gp::Fit {
                        xs: &xs,
                        ys: &target,
                        kernel: self.options.base.kernel,
                        warm: self.cache.as_ref().map(|c| &c.gp.hyper),
                        seed: stream_seed(self.seed, 301, self.round),
                        restarts: self.options.base.fit_restarts,
                        iterations: self.options.base.fit_iterations,
                        nugget: self.options.base.nugget,
                    },
                    stop,
                )
                .map(|x| x.0)
            } else {
                let c = self.cache.as_ref().expect("checked cache");
                if c.ids.len() <= train_ids.len()
                    && train_ids.starts_with(&c.ids)
                    && xs.starts_with(&c.xs)
                {
                    reused = true;
                    let mut gp = c.gp.with_values(&target[..c.ids.len()])?;
                    for i in c.ids.len()..xs.len() {
                        gp = gp.fantasize(xs[i].clone(), target[i])?;
                    }
                    Ok(gp)
                } else {
                    ExactGp::with_nugget(
                        &xs,
                        &target,
                        self.options.base.kernel,
                        c.gp.hyper.clone(),
                        self.options.base.nugget,
                    )
                }
            };
            match result {
                Ok(gp) => {
                    self.cache = Some(Cached {
                        gp: gp.clone(),
                        xs: xs.clone(),
                        ids: train_ids.clone(),
                    });
                    model = Some(gp);
                }
                Err(Error::Numerical(s)) => {
                    fallback = Some(s);
                    self.cache = None;
                }
                Err(e) => return Err(e),
            }
        }
        let mut rng = Rng::new(stream_seed(self.seed, 302, self.round));
        let d = self.space.dimension();
        let mut pool = match self.options.design {
            Design::LatinHypercube => {
                design::latin_hypercube(candidates, self.space.dimension(), rng.next_u64())
            }
            Design::SobolDigitalShift => {
                design::sobol_shifted(candidates, self.space.dimension(), rng.next_u64())?
            }
        };
        // Two of every three LogEI proposals search a box of side `length`
        // around the incumbent, shaped by the lengthscales as in the original
        // method; the third searches the whole box.
        let full = (vec![0.0; d], vec![1.0; d]);
        let mut region = None;
        if self.options.base.trust_region
            && let Some(gp) = &model
            && let Some((index, _)) = raw.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))
        {
            let mean_log = gp.hyper.lengthscales.iter().map(|l| l.ln()).sum::<f64>() / d as f64;
            let (mut lower, mut upper) = full.clone();
            for (j, l) in gp.hyper.lengthscales.iter().enumerate() {
                let half = 0.5 * self.length * (l.ln() - mean_log).exp();
                lower[j] = (xs[index][j] - half).max(0.0);
                upper[j] = (xs[index][j] + half).min(1.0);
            }
            let boxed: Vec<Vec<f64>> = pool
                .iter()
                .map(|x| {
                    x.iter()
                        .enumerate()
                        .map(|(j, v)| lower[j] + v * (upper[j] - lower[j]))
                        .collect()
                })
                .collect();
            region = Some((boxed, lower, upper));
        }
        // Half sparse local candidates; a global half is always retained.
        if self.options.base.trust_region
            && let Some((index, _)) = raw.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1))
        {
            for _ in 0..candidates / 2 {
                let mut x = xs[index].clone();
                let changed = rng.int_below(x.len() as i64) as usize;
                let probability = (20.0 / x.len() as f64).min(1.0);
                for (j, v) in x.iter_mut().enumerate() {
                    if j == changed || rng.uniform01() < probability {
                        *v = (*v + self.length * 0.5 * rng.gaussian()).clamp(0.0, 1.0);
                    }
                }
                pool.push(x);
            }
        }
        let init_seed = if self.region == 0 {
            self.seed
        } else {
            stream_seed(self.seed, 305, self.region)
        };
        let design = design::initial_design(
            self.options.base.initial_points,
            self.space.dimension(),
            init_seed,
        );
        let mut points = Vec::new();
        for i in 0..count {
            if stop() {
                return Err(Error::Stopped);
            }
            let path = if self.options.acquisition == Acquisition::Thompson {
                model
                    .as_ref()
                    .map(|gp| {
                        sampling::Path::new(
                            gp,
                            self.options.features,
                            stream_seed(self.seed, 303, self.round * 8 + i as u64),
                        )
                    })
                    .transpose()?
            } else {
                None
            };
            let mut scored = Vec::new();
            for x in &pool {
                let physical = self.space.from_unit(x)?;
                let x = self.space.to_unit(&physical)?;
                if occupied.iter().any(|p| {
                    design::squared_distance(p, &x) < self.options.base.minimum_distance.powi(2)
                }) {
                    continue;
                }
                let score = if let Some(p) = &path {
                    -p.value(&x)?
                } else if let Some(gp) = &model {
                    let (m, v) = gp.posterior(&x)?;
                    crate::special::log_ei(best, m, v)
                } else {
                    occupied
                        .iter()
                        .map(|p| design::squared_distance(p, &x))
                        .fold(1.0, f64::min)
                };
                scored.push((score, x, physical));
            }
            let init_index = local.len() + pending.len() + i;
            let first = design
                .get(init_index)
                .filter(|x| {
                    occupied.iter().all(|p| {
                        design::squared_distance(p, x) >= self.options.base.minimum_distance.powi(2)
                    })
                })
                .cloned();
            let unit = if let Some(x) = first {
                x
            } else if self.options.acquisition == Acquisition::LogEi
                && let Some(gp) = model.as_ref()
            {
                let slot = self.round * count as u64 + i as u64;
                let (cands, lower, upper) = match &region {
                    Some((boxed, lower, upper)) if slot % GLOBAL_EVERY != GLOBAL_EVERY - 1 => {
                        (boxed, lower, upper)
                    }
                    _ => (&pool, &full.0, &full.1),
                };
                crate::strategy::optimize(
                    gp,
                    cands,
                    &occupied,
                    best,
                    &self.options.base,
                    lower,
                    upper,
                    stop,
                )?
            } else {
                scored
                    .into_iter()
                    .max_by(|a, b| a.0.total_cmp(&b.0))
                    .ok_or_else(|| Error::Numerical("advanced candidate exhaustion".into()))?
                    .1
            };
            let physical = self.space.from_unit(&unit)?;
            let unit = self.space.to_unit(&physical)?;
            occupied.push(unit.clone());
            points.push(physical);
            if let Some(gp) = &model {
                let mean = gp.posterior(&unit)?.0;
                model = Some(gp.fantasize(unit, mean)?);
            }
        }
        let diagnostics = Diagnostics {
            model_version: self.round,
            region: self.region,
            training_ids: train_ids,
            jitter: self.cache.as_ref().map(|c| c.gp.jitter),
            pivot_ratio: self.cache.as_ref().map(|c| c.gp.pivot_ratio()),
            fitted,
            reused,
            fallback,
            candidates,
            refit_interval: interval,
            region_length: self.length,
        };
        self.round += 1;
        self.last_cost = began.elapsed().as_secs_f64();
        Ok((points, diagnostics))
    }
}

/// Incumbent + nearest local points and deterministic maximin global coverage.
pub fn select_training<'a>(
    space: &Space,
    data: &[&'a Sample],
    cap: usize,
) -> Result<Vec<&'a Sample>> {
    if cap == 0 || cap > 240 {
        return Err(Error::Invalid("training cap 1..240".into()));
    }
    if data.len() <= cap {
        return Ok(data.to_vec());
    }
    let units: Vec<_> = data
        .iter()
        .map(|s| space.to_unit(&s.observation.x))
        .collect::<Result<_>>()?;
    let value = |i: usize| match data[i].observation.value {
        Value::Finite(v) => v,
        _ => f64::INFINITY,
    };
    let best = (0..data.len())
        .min_by(|a, b| value(*a).total_cmp(&value(*b)))
        .unwrap();
    let mut order: Vec<_> = (0..data.len()).collect();
    order.sort_by(|a, b| {
        design::squared_distance(&units[*a], &units[best])
            .total_cmp(&design::squared_distance(&units[*b], &units[best]))
            .then(a.cmp(b))
    });
    let mut chosen: Vec<_> = order.into_iter().take((cap * 3 / 4).max(1)).collect();
    let mut distances: Vec<_> = (0..data.len())
        .map(|i| {
            chosen
                .iter()
                .map(|j| design::squared_distance(&units[i], &units[*j]))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    while chosen.len() < cap {
        let next = (0..data.len())
            .filter(|i| !chosen.contains(i))
            .max_by(|a, b| distances[*a].total_cmp(&distances[*b]).then(b.cmp(a)))
            .unwrap();
        chosen.push(next);
        for i in 0..data.len() {
            distances[i] = distances[i].min(design::squared_distance(&units[i], &units[next]));
        }
    }
    chosen.sort_by_key(|i| data[*i].observation.id);
    Ok(chosen.into_iter().map(|i| data[i]).collect())
}

enum Message {
    Fit(
        Box<Engine>,
        Result<(Vec<Vec<f64>>, Diagnostics)>,
        f64,
        u64,
        Vec<u64>,
        Vec<Vec<f64>>,
    ),
    Evaluation(
        u64,
        Vec<f64>,
        u64,
        std::thread::Result<std::result::Result<f64, String>>,
        f64,
    ),
}

/// Async coordinator; only the bounded numerical pool and at most `concurrent`
/// callbacks can consume work simultaneously. Plain callbacks drain on stop.
pub fn run(
    spec: RunSpec<'_>,
    evaluate: &(dyn Fn(&[f64]) -> std::result::Result<f64, String> + Sync),
    sink: &mut dyn FnMut(&Event) -> Result<()>,
    stop: &(dyn Fn() -> Option<&'static str> + Sync),
) -> Result<RunResult> {
    spec.space.validate()?;
    spec.options.validate()?;
    if spec.evaluations == 0
        || spec.evaluations + spec.warm.len() > spec.options.archive_cap
        || spec.inner_threads
            < spec
                .options
                .required_threads()
                .max(spec.options.surrogate_threads)
        || spec.target.is_some_and(|v| !v.is_finite())
    {
        return Err(Error::Invalid("advanced run budget/resources".into()));
    }
    let mut warm_ids = BTreeSet::new();
    for s in spec.warm {
        if !warm_ids.insert(s.observation.id)
            || !spec.space.contains(&s.observation.x)
            || matches!(s.observation.value,Value::Finite(v) if !v.is_finite()||v>=spec.options.base.failure_threshold)
        {
            return Err(Error::Invalid("advanced warm data".into()));
        }
    }
    let mut next_id = spec
        .warm
        .iter()
        .map(|s| s.observation.id)
        .max()
        .map_or(Some(0), |i| i.checked_add(1))
        .ok_or_else(|| Error::Invalid("ID overflow".into()))?;
    next_id
        .checked_add(spec.evaluations as u64)
        .ok_or_else(|| Error::Invalid("ID overflow".into()))?;
    let model_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(spec.options.surrogate_threads)
            .build()
            .map_err(|e| Error::Invalid(e.to_string()))?,
    );
    let evaluation_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(spec.options.concurrent * spec.options.evaluation_threads)
            .build()
            .map_err(|e| Error::Invalid(e.to_string()))?,
    );
    let halted = AtomicBool::new(false);
    let mut history = spec.warm.to_vec();
    let mut engine = Some(Engine::new(
        spec.space.clone(),
        spec.options.clone(),
        spec.seed,
    )?);
    let mut ready = VecDeque::<(Vec<f64>, u64)>::new();
    let mut pending = BTreeMap::<u64, Vec<f64>>::new();
    let mut dispatched = 0;
    let mut completed = 0;
    let mut cancelled = 0;
    let mut failed = 0;
    let mut fitting = false;
    let mut request = 0;
    let mut proposal_seconds = 0.0;
    let mut evaluation_seconds = 0.0;
    let mut peak_pending = 0;
    let mut termination = "evaluation_limit".to_string();
    let mut log_error = None;
    sink(&Event::Start {
        schema: "fcmaes.surrogate.events/2".into(),
        space: spec.space.clone(),
        options: spec.options.clone(),
        seed: spec.seed,
        budget: spec.evaluations,
        warm: history.clone(),
    })?;
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::sync_channel(spec.options.concurrent + 2);
        loop {
            let reached = spec.target.is_some_and(|t| {
                history
                    .iter()
                    .any(|s| matches!(s.observation.value,Value::Finite(y) if y<=t))
            });
            if let Some(reason) = stop() {
                termination = reason.into();
                halted.store(true, Ordering::Release);
            }
            if reached {
                termination = "objective_limit".into();
                halted.store(true, Ordering::Release);
            }
            if halted.load(Ordering::Acquire) {
                ready.clear();
            }
            // Handoff is serial and starts only after all BO evaluations drain.
            let handoff = spec.options.handoff_after.is_some_and(|n| dispatched >= n);
            if !halted.load(Ordering::Acquire) && !handoff {
                while pending.len() < spec.options.concurrent && dispatched < spec.evaluations {
                    let Some((x, region)) = ready.pop_front() else {
                        break;
                    };
                    let id = next_id;
                    next_id += 1;
                    dispatched += 1;
                    pending.insert(id, x.clone());
                    peak_pending = peak_pending.max(pending.len());
                    if let Err(e) = sink(&Event::Started {
                        id,
                        x: x.clone(),
                        region,
                    }) {
                        log_error = Some(e);
                        halted.store(true, Ordering::Release);
                        pending.remove(&id);
                        dispatched -= 1;
                        break;
                    }
                    let sender = tx.clone();
                    let eval_pool = evaluation_pool.clone();
                    let halted_ref = &halted;
                    scope.spawn(move || {
                        let began = Instant::now();
                        let raw = if halted_ref.load(Ordering::Acquire) || stop().is_some() {
                            Ok(Err("cancelled".into()))
                        } else {
                            catch_unwind(AssertUnwindSafe(|| eval_pool.install(|| evaluate(&x))))
                        };
                        let _ = sender.send(Message::Evaluation(
                            id,
                            x,
                            region,
                            raw,
                            began.elapsed().as_secs_f64(),
                        ));
                    });
                    if spec.options.handoff_after.is_some_and(|n| dispatched >= n) {
                        break;
                    }
                }
                let sync_ready =
                    spec.options.schedule == Schedule::Asynchronous || pending.is_empty();
                if !fitting && ready.is_empty() && sync_ready && dispatched < spec.evaluations {
                    let mut count = (spec.evaluations - dispatched).min(spec.options.concurrent);
                    if let Some(n) = spec.options.handoff_after {
                        count = count.min(n.saturating_sub(dispatched));
                    }
                    if count > 0 {
                        fitting = true;
                        request += 1;
                        let req = request;
                        let mut eng = engine.take().expect("one model task");
                        eng.observe_cost(evaluation_seconds / completed.max(1) as f64);
                        let snapshot = history.clone();
                        let pend: Vec<_> = pending.values().cloned().collect();
                        let sender = tx.clone();
                        let pool = model_pool.clone();
                        let halted_ref = &halted;
                        scope.spawn(move || {
                            let ids = snapshot.iter().map(|s| s.observation.id).collect();
                            let began = Instant::now();
                            let result = catch_unwind(AssertUnwindSafe(|| {
                                pool.install(|| {
                                    eng.propose(&snapshot, &pend, count, &|| {
                                        halted_ref.load(Ordering::Acquire) || stop().is_some()
                                    })
                                })
                            }))
                            .unwrap_or_else(|_| {
                                Err(Error::Numerical("proposal computation panicked".into()))
                            });
                            let _ = sender.send(Message::Fit(
                                Box::new(eng),
                                result,
                                began.elapsed().as_secs_f64(),
                                req,
                                ids,
                                pend,
                            ));
                        });
                    }
                }
            }
            if pending.is_empty()
                && !fitting
                && (halted.load(Ordering::Acquire) || dispatched >= spec.evaluations || handoff)
            {
                break;
            }
            let message = match rx.recv_timeout(Duration::from_millis(5)) {
                Ok(m) => m,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => {
                    log_error = Some(Error::Numerical("coordinator channel".into()));
                    break;
                }
            };
            let event = match message {
                Message::Fit(eng, result, seconds, req, ids, pend) => {
                    engine = Some(*eng);
                    fitting = false;
                    proposal_seconds += seconds;
                    match result {
                        Ok((points, diagnostics)) => {
                            if !halted.load(Ordering::Acquire) {
                                for x in &points {
                                    if !history.iter().any(|s| s.observation.x == *x)
                                        && !pending.values().any(|p| p == x)
                                    {
                                        ready.push_back((x.clone(), diagnostics.region));
                                    }
                                }
                            }
                            Event::Proposal {
                                request: req,
                                seconds,
                                history_ids: ids,
                                pending: pend,
                                points,
                                diagnostics,
                            }
                        }
                        Err(Error::Stopped) => {
                            halted.store(true, Ordering::Release);
                            Event::Abandoned {
                                request: req,
                                reason: "stopped".into(),
                            }
                        }
                        Err(e) => {
                            log_error = Some(e);
                            halted.store(true, Ordering::Release);
                            Event::Abandoned {
                                request: req,
                                reason: "proposal failed".into(),
                            }
                        }
                    }
                }
                Message::Evaluation(id, x, region, raw, seconds) => {
                    pending.remove(&id);
                    evaluation_seconds += seconds;
                    if matches!(&raw,Ok(Err(s)) if s=="cancelled")
                        && (stop().is_some() || halted.load(Ordering::Acquire))
                    {
                        cancelled += 1;
                        Event::Cancelled { id, x, seconds }
                    } else {
                        let value = match raw {
                            Ok(Ok(y))
                                if y.is_finite() && y < spec.options.base.failure_threshold =>
                            {
                                Value::Finite(y)
                            }
                            Ok(Ok(_)) => Value::Failed("nonfinite or failure threshold".into()),
                            Ok(Err(e)) => Value::Failed(e.chars().take(256).collect()),
                            Err(_) => Value::Failed("panic".into()),
                        };
                        if matches!(value, Value::Failed(_)) {
                            failed += 1;
                        }
                        completed += 1;
                        let sample = Sample {
                            observation: Observation { id, x, value },
                            region,
                        };
                        history.push(sample.clone());
                        Event::Completed { sample, seconds }
                    }
                }
            };
            if log_error.is_none()
                && let Err(e) = sink(&event)
            {
                log_error = Some(e);
                halted.store(true, Ordering::Release);
            }
        }
    });
    if let Some(e) = log_error {
        return Err(e);
    }
    if !halted.load(Ordering::Acquire)
        && spec.options.handoff_after.is_some()
        && dispatched < spec.evaluations
    {
        sink(&Event::Handoff {
            after: dispatched,
            method: "biteopt".into(),
        })?;
        // Optimizer uses unit coordinates, preserving log-bound geometry.
        let guess = history
            .iter()
            .filter_map(|s| match s.observation.value {
                Value::Finite(y) => Some((y, &s.observation.x)),
                _ => None,
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, x)| spec.space.to_unit(x))
            .transpose()?;
        let params = BiteParams {
            max_evaluations: spec.evaluations.saturating_mul(1000).saturating_add(1000) as u64,
            seed: stream_seed(spec.seed, 304, 0),
            ..BiteParams::default()
        };
        let mut bite = BiteOpt::new(
            &vec![0.0; spec.space.dimension()],
            &vec![1.0; spec.space.dimension()],
            guess.as_deref(),
            &params,
        );
        let mut attempts = 0;
        while completed + cancelled < spec.evaluations && stop().is_none() && bite.stop_code() == 0
        {
            attempts += 1;
            if attempts > spec.evaluations.saturating_mul(100).max(1000) {
                termination = "handoff_stalled".into();
                break;
            }
            let unit = bite
                .ask(1)
                .pop()
                .ok_or_else(|| Error::Numerical("empty handoff ask".into()))?;
            let x = spec.space.from_unit(&unit)?;
            if let Some(cached) = history.iter().find(|s| s.observation.x == x) {
                let cost = match cached.observation.value {
                    Value::Finite(v) => v,
                    _ => f64::INFINITY,
                };
                sink(&Event::Cached {
                    source_id: cached.observation.id,
                    x,
                })?;
                bite.tell(&[cost]);
                continue;
            }
            let id = next_id;
            next_id += 1;
            sink(&Event::Started {
                id,
                x: x.clone(),
                region: u64::MAX,
            })?;
            let began = Instant::now();
            let raw = catch_unwind(AssertUnwindSafe(|| evaluate(&x)));
            let seconds = began.elapsed().as_secs_f64();
            evaluation_seconds += seconds;
            if matches!(&raw,Ok(Err(s)) if s=="cancelled") && stop().is_some() {
                cancelled += 1;
                sink(&Event::Cancelled { id, x, seconds })?;
                break;
            }
            let value = match raw {
                Ok(Ok(y)) if y.is_finite() && y < spec.options.base.failure_threshold => {
                    Value::Finite(y)
                }
                _ => {
                    failed += 1;
                    Value::Failed("handoff evaluation failed".into())
                }
            };
            let cost = match value {
                Value::Finite(y) => y,
                _ => f64::INFINITY,
            };
            completed += 1;
            let sample = Sample {
                observation: Observation { id, x, value },
                region: u64::MAX,
            };
            history.push(sample.clone());
            sink(&Event::Completed { sample, seconds })?;
            bite.tell(&[cost]);
            if spec.target.is_some_and(|t| cost <= t) {
                termination = "objective_limit".into();
                break;
            }
        }
        if let Some(reason) = stop() {
            termination = reason.into();
        }
    }
    history.sort_by_key(|s| s.observation.id);
    let best = history
        .iter()
        .filter(|s| matches!(s.observation.value, Value::Finite(_)))
        .min_by(|a, b| {
            let value = |s: &Sample| match s.observation.value {
                Value::Finite(v) => v,
                _ => f64::INFINITY,
            };
            value(a).total_cmp(&value(b))
        })
        .map(|s| s.observation.clone());
    sink(&Event::Finished {
        termination: termination.clone(),
        completed,
        failed,
        cancelled,
    })?;
    Ok(RunResult {
        history,
        best,
        termination,
        evaluations: completed,
        failed_evaluations: failed,
        cancelled_evaluations: cancelled,
        proposal_seconds,
        evaluation_seconds,
        proposals: request as usize,
        peak_pending,
    })
}

/// Recompute BO proposals from the recorded decision order and verify callback
/// accounting. Handoff evaluations are checked structurally, not re-executed.
/// Timing-based governor actions are replay inputs rather than fresh timings.
pub fn replay(events: &[Event]) -> Result<()> {
    let Some(Event::Start {
        space,
        options,
        seed,
        budget,
        warm,
        ..
    }) = events.first()
    else {
        return Err(Error::Invalid("missing advanced Start".into()));
    };
    let mut engine = Engine::new(space.clone(), options.clone(), *seed)?;
    let mut observations: BTreeMap<u64, Sample> =
        warm.iter().map(|s| (s.observation.id, s.clone())).collect();
    if observations.len() != warm.len() {
        return Err(Error::Invalid("duplicate warm IDs".into()));
    }
    let mut active = BTreeMap::<u64, (Vec<f64>, u64)>::new();
    let mut allowed = Vec::<(Vec<f64>, u64)>::new();
    let mut completed = 0;
    let mut failures = 0;
    let mut cancelled = 0;
    let mut finished = false;
    let mut handoff = false;
    for event in &events[1..] {
        if finished {
            return Err(Error::Invalid("records after Finished".into()));
        }
        match event {
            Event::Proposal {
                history_ids,
                pending,
                points,
                diagnostics,
                ..
            } => {
                let history: Vec<_> = history_ids
                    .iter()
                    .map(|id| {
                        observations
                            .get(id)
                            .cloned()
                            .ok_or_else(|| Error::Invalid("proposal history ID".into()))
                    })
                    .collect::<Result<_>>()?;
                engine.options.governor = false;
                engine.options.base.candidates = diagnostics.candidates;
                engine.options.base.refit_interval = diagnostics.refit_interval;
                let (generated, diag) =
                    engine.propose(&history, pending, points.len(), &|| false)?;
                if generated != *points
                    || diag.region != diagnostics.region
                    || diag.training_ids != diagnostics.training_ids
                {
                    return Err(Error::Invalid("advanced proposal replay mismatch".into()));
                }
                allowed.extend(points.iter().cloned().map(|x| (x, diagnostics.region)));
            }
            Event::Started { id, x, region } => {
                if !space.contains(x)
                    || observations.contains_key(id)
                    || active.contains_key(id)
                    || (!handoff && !allowed.iter().any(|(p, r)| p == x && r == region))
                {
                    return Err(Error::Invalid("invalid advanced Started".into()));
                }
                if let Some(i) = allowed.iter().position(|(p, r)| p == x && r == region) {
                    allowed.remove(i);
                }
                active.insert(*id, (x.clone(), *region));
            }
            Event::Completed { sample, .. } => {
                let expected = active.remove(&sample.observation.id);
                if expected != Some((sample.observation.x.clone(), sample.region))
                    || matches!(sample.observation.value,Value::Finite(v) if !v.is_finite())
                {
                    return Err(Error::Invalid("unmatched advanced completion".into()));
                }
                completed += 1;
                if matches!(sample.observation.value, Value::Failed(_)) {
                    failures += 1;
                }
                observations.insert(sample.observation.id, sample.clone());
            }
            Event::Cancelled { id, x, .. } => {
                if active.remove(id).is_none_or(|v| v.0 != *x) {
                    return Err(Error::Invalid("unmatched cancellation".into()));
                }
                cancelled += 1;
            }
            Event::Cached { source_id, x } => {
                if !handoff
                    || observations
                        .get(source_id)
                        .is_none_or(|s| s.observation.x != *x)
                {
                    return Err(Error::Invalid("invalid cached handoff".into()));
                }
            }
            Event::Handoff { .. } => {
                handoff = true;
            }
            Event::Finished {
                completed: c,
                failed: f,
                cancelled: k,
                ..
            } => {
                if !active.is_empty()
                    || *c != completed
                    || *f != failures
                    || *k != cancelled
                    || completed + cancelled > *budget
                {
                    return Err(Error::Invalid("advanced finish accounting".into()));
                }
                finished = true;
            }
            Event::Abandoned { .. } => {}
            Event::Start { .. } => return Err(Error::Invalid("duplicate Start".into())),
        }
    }
    if !finished {
        return Err(Error::Invalid("incomplete advanced trace".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: u64, x: [f64; 2], y: f64) -> Sample {
        Sample {
            observation: Observation {
                id,
                x: x.to_vec(),
                value: Value::Finite(y),
            },
            region: 0,
        }
    }

    fn new_engine() -> (Space, Engine) {
        let space = Space::new(vec![0.0; 2], vec![1.0; 2], vec![false; 2]).unwrap();
        let options = Options {
            base: Config {
                initial_points: 4,
                candidates: 64,
                fit_restarts: 1,
                fit_iterations: 8,
                ..Config::default()
            },
            training_cap: 64,
            archive_cap: 200,
            ..Options::default()
        };
        (space.clone(), Engine::new(space, options, 7).unwrap())
    }

    /// The incumbent at (0.3, 0.3), then `failures` worse points.
    fn incumbent_then_failures(failures: u64) -> Vec<Sample> {
        let mut history = vec![sample(0, [0.3, 0.3], 0.0)];
        for i in 1..=failures {
            let (row, column) = ((i - 1) / 6, (i - 1) % 6);
            let x = [0.05 + 0.18 * column as f64, 0.55 + 0.09 * row as f64];
            history.push(sample(i, x, 1.0 + 0.01 * i as f64));
        }
        history
    }

    #[test]
    fn region_halves_after_failures_and_local_proposals_stay_inside_it() {
        let (space, mut engine) = new_engine();
        // One success, then seven failures: one halving after max(4, d) = 4.
        let history = incumbent_then_failures(7);
        let (points, diagnostics) = engine.propose(&history, &[], 3, &|| false).unwrap();
        assert_eq!(diagnostics.region_length, TR_INITIAL / 2.0);
        let hyper = &engine.cache.as_ref().expect("fitted model").gp.hyper;
        let mean_log = hyper.lengthscales.iter().map(|l| l.ln()).sum::<f64>() / 2.0;
        // Slots 0 and 1 are local, slot 2 searches the whole box.
        for x in &points[..2] {
            let unit = space.to_unit(x).unwrap();
            for (j, l) in hyper.lengthscales.iter().enumerate() {
                let half = 0.5 * diagnostics.region_length * (l.ln() - mean_log).exp();
                assert!(
                    (unit[j] - 0.3).abs() <= half + 1e-12,
                    "{unit:?} outside the region"
                );
            }
        }
    }

    #[test]
    fn region_doubles_after_successes_and_resets_below_its_minimum() {
        let (_, mut engine) = new_engine();
        let improving: Vec<_> = (0..4)
            .map(|i| sample(i, [0.1 + 0.2 * i as f64, 0.5], 10.0 - i as f64))
            .collect();
        let (_, diagnostics) = engine.propose(&improving, &[], 1, &|| false).unwrap();
        assert_eq!(diagnostics.region_length, TR_MAX);

        let (_, mut engine) = new_engine();
        // 24 failures halve six times: 0.8 / 64 = 0.0125.
        let history = incumbent_then_failures(28);
        let (_, diagnostics) = engine.propose(&history[..25], &[], 1, &|| false).unwrap();
        assert_eq!(diagnostics.region_length, TR_INITIAL / 64.0);
        // The seventh halving falls below the minimum and resets the side.
        let (_, diagnostics) = engine.propose(&history, &[], 1, &|| false).unwrap();
        assert_eq!(diagnostics.region_length, TR_INITIAL);
    }

    #[test]
    fn proposals_do_not_depend_on_the_completion_order_of_the_history() {
        let history = incumbent_then_failures(9);
        let mut reversed = history.clone();
        reversed.reverse();
        let (_, mut forward) = new_engine();
        let (_, mut backward) = new_engine();
        let (a, da) = forward.propose(&history, &[], 3, &|| false).unwrap();
        let (b, db) = backward.propose(&reversed, &[], 3, &|| false).unwrap();
        assert_eq!(a, b);
        assert_eq!(da.region_length, db.region_length);
        assert_eq!(da.training_ids, db.training_ids);
    }
}
