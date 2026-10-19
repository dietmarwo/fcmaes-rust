//! `fcmaes.retry/1`: independent retries of BiteOpt, DE, CMA-ES, CR-FM-NES or
//! PGPE on registered scalar models. BiteOpt and DE run the shared native
//! solver of the legacy `fcmaes.job/0.1` profile, so both profiles run
//! identical retries on the built-in models. CMA-ES, CR-FM-NES and PGPE
//! evaluate whole populations, in parallel with `inner_threads > 1`; CMA-ES
//! can restart with growing populations (IPOP) or alternate large and small
//! ones (BIPOP) inside the retry's budget.
use crate::{
    registry::{Kind, Model, Problem, Registry, ScalarFn},
    threads::Threads,
};
use fcmaes_core::{
    Cmaes, CmaesParams, Crfmnes, CrfmnesParams, Fitness, Pgpe, PgpeParams, RetryBounds,
    RetryRunResult, Rng, retry_run_seed,
};
use fcmaes_native_worker::solve;
use optimization_job_api::{
    JobSpec, Result,
    legacy::{self, JobSpec as LegacySpec},
    result::{Evidence, Outcome},
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

pub const PROFILE: &str = "fcmaes.retry/1";
/// Default CMA-ES population, the `fcmaes-core` default.
pub const CMAES_POPULATION: usize = 31;
/// Default initial CMA-ES step size in the normalized `[-1, 1]` box.
pub const CMAES_SIGMA: f64 = 0.3;
/// Default CR-FM-NES population, the `fcmaes-core` default.
pub const CRFMNES_POPULATION: usize = 32;
/// Default initial CR-FM-NES step size in the normalized box.
pub const CRFMNES_SIGMA: f64 = 0.3;
/// Default PGPE population, the `fcmaes-core` default.
pub const PGPE_POPULATION: usize = 32;
/// Default initial PGPE standard deviation in the normalized box.
pub const PGPE_SIGMA: f64 = 0.1;
/// Default largest restart population as a multiple of the first one:
/// nine doublings, as in BIPOP-CMA-ES (Hansen 2009).
pub const MAX_POPULATION_FACTOR: usize = 512;
/// Largest population of any run.
const POPULATION_LIMIT: usize = 1_000_000;
/// Largest population (`max_population` with restarts) times dimension of a
/// run, and largest dimension squared of a CMA-ES run, which holds
/// dimension-by-dimension matrices.
const MAX_CELLS: usize = 10_000_000;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub problem: Problem,
    pub optimizer: Optimizer,
    pub seed: u64,
    pub retries: usize,
    pub evaluations_per_retry: u64,
    pub evaluation_limit_mode: String,
}
/// Serializes like `legacy::Optimizer` when only `population` is set, so
/// BiteOpt and DE payloads and their job digests are unchanged.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Optimizer {
    pub kind: String,
    pub parameters: Parameters,
}
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub population: Option<usize>,
    /// CMA-ES, CR-FM-NES, PGPE: initial step size in the normalized
    /// `[-1, 1]` box.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma: Option<f64>,
    /// CMA-ES: `none` (absent when resolved), `ipop` or `bipop`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restarts: Option<String>,
    /// CMA-ES with restarts: largest population of a restart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_population: Option<usize>,
    /// PGPE: Adam learning rate of the distribution center.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center_learning_rate: Option<f64>,
    /// PGPE: learning rate of the standard deviations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdev_learning_rate: Option<f64>,
}

/// The population-based optimizers of this profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EsKind {
    Cmaes,
    Crfmnes,
    Pgpe,
}
/// CMA-ES restart strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Restarts {
    None,
    /// Restart with a doubled population after every stop (Auger and
    /// Hansen 2005).
    Ipop,
    /// Alternate doubled populations with small populations and step sizes,
    /// choosing the regime that has used fewer evaluations (Hansen 2009).
    Bipop,
}
/// Resolved settings of a CMA-ES, CR-FM-NES or PGPE retry.
#[derive(Clone, Debug)]
pub struct EsSettings {
    pub kind: EsKind,
    pub population: usize,
    pub sigma: f64,
    pub restarts: Restarts,
    /// Largest population any run of the retry uses.
    pub max_population: usize,
    pub center_learning_rate: f64,
    pub stdev_learning_rate: f64,
}
impl EsSettings {
    /// CMA-ES without restarts.
    pub fn cmaes(population: usize, sigma: f64) -> Self {
        let d = PgpeParams::default();
        Self {
            kind: EsKind::Cmaes,
            population,
            sigma,
            restarts: Restarts::None,
            max_population: population,
            center_learning_rate: d.center_learning_rate,
            stdev_learning_rate: d.stdev_learning_rate,
        }
    }
}

enum Engine {
    /// BiteOpt or DE through the shared native solver.
    Legacy(Box<LegacySpec>),
    Es(EsSettings),
}
pub struct Run {
    engine: Engine,
    payload: Payload,
    bounds: RetryBounds,
    objective: ScalarFn,
    threads: usize,
    /// Largest evaluation count one retry may report.
    per_retry_limit: u64,
}

pub fn describe() -> Value {
    let pgpe = PgpeParams::default();
    json!({
        "biteopt":{"profile":PROFILE,"limit":"hard","defaults":{"population":"9 + 3 * dimension"},
            "inner_threads":"1; evaluates one point at a time"},
        "de":{"profile":PROFILE,"limit":"soft","maximum_overshoot_per_retry":"2*population-1",
            "defaults":{"population":31},"inner_threads":"1; evaluates one point at a time"},
        "cmaes":{"profile":PROFILE,"limit":"soft",
            "maximum_overshoot_per_retry":"population-1; max_population-1 with restarts",
            "defaults":{"population":CMAES_POPULATION,"sigma":CMAES_SIGMA,"restarts":"none",
                "max_population":format!("with restarts: {MAX_POPULATION_FACTOR} * population, at most 1e6 and 1e7 / dimension")},
            "sigma":"initial step size in the normalized [-1,1] box, (0,2]",
            "limits":"dimension <= 3162 (dimension^2 <= 1e7); population * dimension <= 1e7 (max_population with restarts)",
            "restarts":"none | ipop (double the population after every stop) | bipop (alternate doubled populations with small populations and step sizes); runs restart until the retry's budget is spent, each from a new uniform random point",
            "start":"uniform random point in the bounds from the retry seed",
            "inner_threads":"evaluates each population on inner_threads threads"},
        "crfmnes":{"profile":PROFILE,"limit":"soft","maximum_overshoot_per_retry":"population-1",
            "defaults":{"population":CRFMNES_POPULATION,"sigma":CRFMNES_SIGMA},
            "ranges":{"population":"even, 4..1000000; population * dimension <= 1e7","sigma":"(0,2]"},
            "start":"uniform random point in the bounds from the retry seed",
            "inner_threads":"evaluates each population on inner_threads threads"},
        "pgpe":{"profile":PROFILE,"limit":"soft","maximum_overshoot_per_retry":"population-1",
            "defaults":{"population":PGPE_POPULATION,"sigma":PGPE_SIGMA,
                "center_learning_rate":pgpe.center_learning_rate,"stdev_learning_rate":pgpe.stdev_learning_rate},
            "ranges":{"population":"even, 4..1000000; population * dimension <= 1e7","sigma":"(0,2]","center_learning_rate":"(0,1]","stdev_learning_rate":"(0,1]"},
            "start":"uniform random point in the bounds from the retry seed",
            "inner_threads":"evaluates each population on inner_threads threads"},
        "best":"every optimizer reports the best point it evaluated; for CR-FM-NES and PGPE it is tracked from the evaluated populations"
    })
}

pub fn prepare(registry: &Registry, s: &mut JobSpec, base: &Path) -> Result<Box<dyn Prepared>> {
    let mut p: Payload = serde_json::from_value(s.payload.clone())?;
    let (problem, model) = registry.resolve_as(p.problem.clone(), s, base, Kind::Scalar)?;
    let Model::Scalar(model) = model else {
        unreachable!("resolve_as returns the requested kind")
    };
    p.problem = problem;
    if p.retries == 0 || p.evaluations_per_retry == 0 {
        return Err("positive retries and evaluations_per_retry required".into());
    }
    let threads = s.resources.inner_threads;
    let (engine, per_retry_limit) = match p.optimizer.kind.as_str() {
        "biteopt" | "de" => {
            if threads != 1 {
                return Err(format!(
                    "{} evaluates one point at a time and needs inner_threads=1; outer_workers runs retries in parallel",
                    p.optimizer.kind
                )
                .into());
            }
            let spec = legacy_spec(s, &mut p)?;
            let limit = spec.maximum_evaluations()? / p.retries as u64;
            (Engine::Legacy(Box::new(spec)), limit)
        }
        "cmaes" | "crfmnes" | "pgpe" => {
            let settings = es_settings(&mut p)?;
            let limit = p
                .evaluations_per_retry
                .checked_add(settings.max_population as u64 - 1)
                .ok_or("evaluation budget overflow")?;
            limit
                .checked_mul(p.retries as u64)
                .ok_or("evaluation budget overflow")?;
            (Engine::Es(settings), limit)
        }
        _ => return Err("optimizer.kind must be biteopt, de, cmaes, crfmnes or pgpe".into()),
    };
    let bounds = RetryBounds::new(
        p.problem.bounds.lower.clone(),
        p.problem.bounds.upper.clone(),
    )
    .map_err(|e| format!("bounds: {e}"))?;
    s.payload = serde_json::to_value(&p)?;
    Ok(Box::new(Run {
        engine,
        payload: p,
        bounds,
        objective: model.evaluate,
        threads,
        per_retry_limit,
    }))
}

/// Validates a BiteOpt or DE request like `fcmaes.job/0.1` and returns the
/// legacy specification the shared native solver runs.
fn legacy_spec(s: &JobSpec, p: &mut Payload) -> Result<LegacySpec> {
    let q = &p.optimizer.parameters;
    for (name, given) in [
        ("sigma", q.sigma.is_some()),
        ("restarts", q.restarts.is_some()),
        ("max_population", q.max_population.is_some()),
        ("center_learning_rate", q.center_learning_rate.is_some()),
        ("stdev_learning_rate", q.stdev_learning_rate.is_some()),
    ] {
        if given {
            return Err(format!(
                "optimizer.parameters.{name} does not apply to {}",
                p.optimizer.kind
            )
            .into());
        }
    }
    let kind = p.optimizer.kind.as_str();
    match (kind, p.evaluation_limit_mode.as_str()) {
        ("biteopt", "hard") | ("de", "soft") => (),
        _ => {
            return Err(
                "supported paths: biteopt/hard, de/soft; DE hard cap is unavailable".into(),
            );
        }
    }
    let n = p.problem.config.dimension;
    let population = q
        .population
        .unwrap_or(if kind == "de" { 31 } else { 9 + 3 * n });
    if !(4..=POPULATION_LIMIT).contains(&population) {
        return Err("population must be 4..1000000".into());
    }
    let spec = LegacySpec {
        schema_version: "fcmaes.job/0.1".into(),
        request_key: s.request_key.clone(),
        problem: p.problem.legacy(),
        optimizer: legacy::Optimizer {
            kind: kind.into(),
            parameters: legacy::Parameters {
                population: Some(population),
            },
        },
        execution: legacy::Execution {
            strategy: "independent_retry".into(),
            seed: p.seed,
            workers: s.resources.outer_workers,
            retry_wave_size: s.resources.outer_workers,
        },
        budget: legacy::Budget {
            retries: p.retries,
            evaluations_per_retry: p.evaluations_per_retry,
            evaluation_limit_mode: p.evaluation_limit_mode.clone(),
            wall_time_seconds: s.limits.wall_time_seconds,
            grace_seconds: s.limits.grace_seconds,
        },
        outputs: legacy::Outputs {
            progress: "completed_retries".into(),
        },
    };
    spec.requested()?;
    spec.maximum_evaluations()?;
    p.optimizer.parameters.population = Some(population);
    Ok(spec)
}

fn in_range(name: &str, value: f64, low: f64, high: f64) -> Result<f64> {
    if value.is_finite() && value > low && value <= high {
        Ok(value)
    } else {
        Err(format!("{name} must lie in ({low}, {high}]").into())
    }
}

/// Validates a CMA-ES, CR-FM-NES or PGPE request and resolves its defaults.
fn es_settings(p: &mut Payload) -> Result<EsSettings> {
    if p.evaluation_limit_mode != "soft" {
        return Err(format!(
            "{} evaluates whole generations; use evaluation_limit_mode soft",
            p.optimizer.kind
        )
        .into());
    }
    let q = p.optimizer.parameters.clone();
    let (kind, population, sigma) = match p.optimizer.kind.as_str() {
        "cmaes" => (EsKind::Cmaes, CMAES_POPULATION, CMAES_SIGMA),
        "crfmnes" => (EsKind::Crfmnes, CRFMNES_POPULATION, CRFMNES_SIGMA),
        _ => (EsKind::Pgpe, PGPE_POPULATION, PGPE_SIGMA),
    };
    let only = |name: &str, given: bool, allowed: EsKind, label: &str| -> Result<()> {
        if given && kind != allowed {
            return Err(format!("optimizer.parameters.{name} applies only to {label}").into());
        }
        Ok(())
    };
    only("restarts", q.restarts.is_some(), EsKind::Cmaes, "cmaes")?;
    only(
        "max_population",
        q.max_population.is_some(),
        EsKind::Cmaes,
        "cmaes with restarts",
    )?;
    only(
        "center_learning_rate",
        q.center_learning_rate.is_some(),
        EsKind::Pgpe,
        "pgpe",
    )?;
    only(
        "stdev_learning_rate",
        q.stdev_learning_rate.is_some(),
        EsKind::Pgpe,
        "pgpe",
    )?;
    let population = q.population.unwrap_or(population);
    if !(4..=POPULATION_LIMIT).contains(&population) {
        return Err("population must be 4..1000000".into());
    }
    if kind != EsKind::Cmaes && !population.is_multiple_of(2) {
        return Err(format!(
            "{} samples mirrored pairs; population must be even",
            p.optimizer.kind
        )
        .into());
    }
    let sigma = in_range(
        &format!("{} sigma (normalized box units)", p.optimizer.kind),
        q.sigma.unwrap_or(sigma),
        0.0,
        2.0,
    )?;
    let restarts = match q.restarts.as_deref() {
        None | Some("none") => Restarts::None,
        Some("ipop") => Restarts::Ipop,
        Some("bipop") => Restarts::Bipop,
        Some(_) => return Err("restarts must be none, ipop or bipop".into()),
    };
    let max_population = if restarts == Restarts::None {
        if q.max_population.is_some() {
            return Err("max_population needs restarts ipop or bipop".into());
        }
        population
    } else {
        let max = q.max_population.unwrap_or(
            (population * MAX_POPULATION_FACTOR)
                .min(POPULATION_LIMIT)
                .min((MAX_CELLS / p.problem.config.dimension).max(population)),
        );
        if !(population..=POPULATION_LIMIT).contains(&max) {
            return Err("max_population must lie in population..1000000".into());
        }
        max
    };
    let n = p.problem.config.dimension;
    if max_population.saturating_mul(n) > MAX_CELLS {
        return Err(format!(
            "{} needs population * dimension <= 1e7 (max_population with restarts)",
            p.optimizer.kind
        )
        .into());
    }
    if kind == EsKind::Cmaes && n.saturating_mul(n) > MAX_CELLS {
        return Err(
            "cmaes holds dimension x dimension matrices and needs dimension <= 3162; crfmnes and pgpe scale to higher dimensions"
                .into(),
        );
    }
    let d = PgpeParams::default();
    let settings = EsSettings {
        kind,
        population,
        sigma,
        restarts,
        max_population,
        center_learning_rate: in_range(
            "center_learning_rate",
            q.center_learning_rate.unwrap_or(d.center_learning_rate),
            0.0,
            1.0,
        )?,
        stdev_learning_rate: in_range(
            "stdev_learning_rate",
            q.stdev_learning_rate.unwrap_or(d.stdev_learning_rate),
            0.0,
            1.0,
        )?,
    };
    let pgpe = kind == EsKind::Pgpe;
    let restarting = restarts != Restarts::None;
    p.optimizer.parameters = Parameters {
        population: Some(population),
        sigma: Some(sigma),
        restarts: restarting.then(|| q.restarts.clone().unwrap_or_default()),
        max_population: restarting.then_some(max_population),
        center_learning_rate: pgpe.then_some(settings.center_learning_rate),
        stdev_learning_rate: pgpe.then_some(settings.stdev_learning_rate),
    };
    Ok(settings)
}

/// One retry of a population-based optimizer and what it did.
pub struct EsRun {
    /// The best point evaluated over all runs, its value and the evaluations
    /// of all runs.
    pub result: RetryRunResult,
    /// Stop code of the last run (0: budget spent).
    pub stop: i32,
    /// Number of restarts after the first run.
    pub restarts: usize,
    /// Largest population of any run.
    pub largest_population: usize,
}

fn clamp(x: &[f64], bounds: &RetryBounds) -> Vec<f64> {
    x.iter()
        .zip(bounds.lower().iter().zip(bounds.upper()))
        .map(|(v, (l, u))| v.clamp(*l, *u))
        .collect()
}

fn uniform(bounds: &RetryBounds, rng: &mut Rng) -> Vec<f64> {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper())
        .map(|(l, u)| l + (u - l) * rng.uniform01())
        .collect()
}

fn normalized(bounds: &RetryBounds) -> Fitness {
    let mut fitness = Fitness::bounded(bounds.dim(), 1, bounds.lower(), bounds.upper());
    fitness.set_normalize(true);
    fitness
}

/// One CMA-ES, CR-FM-NES or PGPE retry of `evaluations` evaluations (a soft
/// limit) from a uniform random start drawn from a stream derived from
/// `seed`. Populations are evaluated on `threads` and the result does not
/// depend on their number. With restarts, every restart begins at a new
/// random start and runs until the retry's budget is spent.
pub fn es_retry(
    settings: &EsSettings,
    bounds: &RetryBounds,
    seed: u64,
    evaluations: u64,
    objective: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    threads: &Threads,
) -> EsRun {
    let s = settings;
    let mut start = Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut best: Option<RetryRunResult> = None;
    let mut used = 0;
    let (mut large_used, mut small_used, mut large_runs) = (0u64, 0u64, 0u32);
    let (mut run, mut largest) = (0usize, s.population);
    let mut stop;
    let doubled = |k: u32| {
        s.population
            .saturating_mul(1usize.checked_shl(k).unwrap_or(usize::MAX))
            .min(s.max_population)
    };
    loop {
        let remaining = evaluations - used;
        let (population, sigma, budget, large) = if run == 0 {
            (s.population, s.sigma, remaining, true)
        } else if s.restarts == Restarts::Ipop {
            (doubled(run as u32), s.sigma, remaining, true)
        } else if small_used < large_used {
            // BIPOP small regime: population between the default and the
            // last large one, a smaller step size, and a budget of half the
            // evaluations the large regime used so far, but one generation
            // at least.
            let u = start.uniform01();
            let factor = 2f64.powf(f64::from(large_runs) * u * u);
            let population = ((s.population as f64 * factor) as usize).min(s.max_population);
            let sigma = s.sigma * 10f64.powf(-2.0 * start.uniform01());
            let budget = remaining.min((large_used / 2).max(population as u64));
            (population, sigma, budget, false)
        } else {
            large_runs += 1;
            (doubled(large_runs), s.sigma, remaining, true)
        };
        let guess = uniform(bounds, &mut start);
        let run_seed = if run == 0 {
            seed
        } else {
            retry_run_seed(seed, run)
        };
        let (r, run_stop) = es_run(
            s, bounds, &guess, population, sigma, run_seed, budget, objective, threads,
        );
        used += r.evaluations;
        if large {
            large_used += r.evaluations;
        } else {
            small_used += r.evaluations;
        }
        largest = largest.max(population);
        stop = run_stop;
        let spent = r.evaluations;
        if best.as_ref().is_none_or(|b| r.y < b.y) {
            best = Some(r);
        }
        run += 1;
        if s.restarts == Restarts::None || used >= evaluations || spent == 0 {
            break;
        }
    }
    let mut result = best.expect("one run at least");
    result.evaluations = used;
    EsRun {
        result,
        stop,
        restarts: run - 1,
        largest_population: largest,
    }
}

/// One run without restarts.
#[allow(clippy::too_many_arguments)]
fn es_run(
    s: &EsSettings,
    bounds: &RetryBounds,
    guess: &[f64],
    population: usize,
    sigma: f64,
    seed: u64,
    evaluations: u64,
    objective: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    threads: &Threads,
) -> (RetryRunResult, i32) {
    if s.kind == EsKind::Cmaes {
        let mut cma = Cmaes::new(
            normalized(bounds),
            guess,
            &[sigma],
            &CmaesParams {
                popsize: population as i32,
                max_evaluations: evaluations,
                seed,
                ..Default::default()
            },
        );
        let workers = threads.core_workers();
        let r = threads.install(|| cma.optimize(&|x: &[f64]| objective(x), workers));
        // CMA-ES evaluates the clamped decode of its sample but stores the
        // unclamped one, which can lie an ulp outside the box on a bound.
        // Clamping gives exactly the point that was evaluated.
        let x = clamp(&r.x, bounds);
        return (
            RetryRunResult {
                x,
                y: r.y,
                evaluations: r.evaluations,
            },
            r.stop,
        );
    }
    // CR-FM-NES adds a bound penalty to the values it ranks and keeps
    // unclamped points; PGPE's points can lie an ulp outside the box. The
    // best point is therefore taken from the evaluated (clamped) populations.
    let mut best_x = guess.to_vec();
    let mut best_y = f64::INFINITY;
    let mut evaluate = |rows: &[Vec<f64>]| -> Vec<f64> {
        let xs: Vec<Vec<f64>> = rows.iter().map(|x| clamp(x, bounds)).collect();
        let ys = threads.map(&xs, |x| objective(x));
        for (x, y) in xs.iter().zip(&ys) {
            if *y < best_y {
                best_y = *y;
                best_x.clone_from(x);
            }
        }
        ys
    };
    let (evaluations, stop) = if s.kind == EsKind::Crfmnes {
        let mut nes = Crfmnes::new(
            normalized(bounds),
            guess,
            sigma,
            &CrfmnesParams {
                popsize: population as i32,
                max_evaluations: evaluations,
                seed,
                ..Default::default()
            },
        );
        let r = nes.optimize_batch(&mut evaluate);
        (r.evaluations, r.stop)
    } else {
        let mut pgpe = Pgpe::new(
            normalized(bounds),
            guess,
            &[sigma],
            &PgpeParams {
                popsize: population as i32,
                max_evaluations: evaluations,
                seed,
                center_learning_rate: s.center_learning_rate,
                stdev_learning_rate: s.stdev_learning_rate,
                ..Default::default()
            },
        );
        let r = pgpe.optimize_batch(&mut evaluate);
        (r.evaluations, r.stop)
    };
    (
        RetryRunResult {
            x: best_x,
            y: best_y,
            evaluations,
        },
        stop,
    )
}

/// Judges one retry's answer: the best point it evaluated and its value. A
/// value at or above `1e99` (the fcmaes penalty) or non-finite is no answer.
/// A point outside the box refutes `best_found` (`infeasible_point`) and is
/// not evaluated. Otherwise the point is re-evaluated with `objective`; a
/// value that differs refutes `best_found` too (`objective_mismatch`),
/// leaving a feasible point with the recomputed value if that value is
/// usable. Returns the outcome without termination and details (apart from
/// `reported_objective`), and the number of validation evaluations.
pub fn judge(
    r: RetryRunResult,
    bounds: &RetryBounds,
    objective: &dyn Fn(&[f64]) -> f64,
) -> Result<(Outcome, u64)> {
    let usable = |v: f64| v.is_finite() && v < 1e99;
    if !usable(r.y) {
        let outcome = Outcome {
            claim: "none".into(),
            feasible: Some(false),
            evidence: Some(Evidence::claimed(&[])),
            ..Outcome::default()
        };
        return Ok((outcome, 0));
    }
    if r.x.len() != bounds.dim() || r.x.iter().any(|v| !v.is_finite()) {
        return Err("native candidate has the wrong dimension or a non-finite entry".into());
    }
    let checks = Evidence::rechecked(&["objective", "bounds"]);
    let outside =
        r.x.iter()
            .zip(bounds.lower().iter().zip(bounds.upper()))
            .position(|(v, (l, u))| v < l || v > u);
    if let Some(i) = outside {
        // Registered models are defined on their box only (GTOP models are
        // not defined outside it), so the point is not evaluated.
        let message = format!(
            "x[{i}] = {} lies outside [{}, {}]",
            r.x[i],
            bounds.lower()[i],
            bounds.upper()[i]
        );
        let outcome = Outcome {
            claim: "none".into(),
            feasible: Some(false),
            x: Some(r.x),
            details: json!({"reported_objective":r.y}),
            evidence: Some(Evidence::rechecked(&["bounds"]).refuting(
                "best_found",
                "infeasible_point",
                message,
            )),
            ..Outcome::default()
        };
        return Ok((outcome, 0));
    }
    let value = objective(&r.x);
    let outcome = if value != r.y {
        let message = format!("reported {} but the point evaluates to {value}", r.y);
        let refuted = checks.refuting("best_found", "objective_mismatch", message);
        let feasible = usable(value);
        Outcome {
            claim: if feasible { "feasible" } else { "none" }.into(),
            feasible: Some(feasible),
            objective: value.is_finite().then_some(value),
            x: Some(r.x),
            details: json!({"reported_objective":r.y}),
            evidence: Some(refuted),
            ..Outcome::default()
        }
    } else {
        Outcome {
            claim: "best_found".into(),
            feasible: Some(true),
            objective: Some(value),
            x: Some(r.x),
            evidence: Some(checks),
            ..Outcome::default()
        }
    };
    Ok((outcome, 1))
}

impl Prepared for Run {
    fn tasks(&self) -> usize {
        self.payload.retries
    }
    fn cancel(&self) {} // fcmaes core has no in-retry stop token.
    fn solve(&self, task: usize, _: f64) -> Result<TaskResult> {
        // The whole task runs on its own threads, so no Rayon work of the
        // task (or of the model) reaches the global pool.
        let threads = Threads::new(self.threads)?;
        threads.install(|| self.task(task, &threads))
    }
}

impl Run {
    fn task(&self, task: usize, threads: &Threads) -> Result<TaskResult> {
        let seed = retry_run_seed(self.payload.seed, task);
        let objective = &*self.objective;
        let mut details = json!({"run_seed":seed,"evaluation_limit_mode":self.payload.evaluation_limit_mode,
            "requested_evaluations":self.payload.evaluations_per_retry,"global_optimum_certified":false});
        let r = match &self.engine {
            Engine::Legacy(spec) => solve(spec, &self.bounds, seed, &|x: &[f64]| objective(x)),
            Engine::Es(settings) => {
                let run = es_retry(
                    settings,
                    &self.bounds,
                    seed,
                    self.payload.evaluations_per_retry,
                    objective,
                    threads,
                );
                details["stop_code"] = json!(run.stop);
                if settings.restarts != Restarts::None {
                    details["restarts"] = json!(run.restarts);
                    details["largest_population"] = json!(run.largest_population);
                }
                run.result
            }
        };
        if r.evaluations == 0 || r.evaluations > self.per_retry_limit {
            return Err("native evaluation budget mismatch".into());
        }
        // BiteOpt and DE always spend their budget; CMA-ES, CR-FM-NES and
        // PGPE without restarts can stop earlier on their own tests.
        let termination = if r.evaluations >= self.payload.evaluations_per_retry {
            "evaluation_limit"
        } else {
            "optimizer_stop"
        };
        let evaluations = r.evaluations;
        let (outcome, validations) = judge(r, &self.bounds, objective)?;
        if let Some(reported) = outcome.details.get("reported_objective") {
            details["reported_objective"] = reported.clone();
        }
        Ok(TaskResult {
            outcome: Outcome {
                termination: termination.into(),
                details,
                ..outcome
            },
            work: [
                ("evaluations".into(), evaluations),
                ("validation_evaluations".into(), validations),
            ]
            .into(),
        })
    }
}
