//! `fcmaes.map-elites/1`: independent MAP-Elites runs on registered
//! quality-diversity models. Each task is one seeded run on a regular
//! two-dimensional grid or a CVT archive; its outcome is the re-evaluated
//! archive.
//!
//! A run has two phases with exact budgets: `generations` of the SBX or
//! Iso+LineDD emitter, then optionally `cma_generations` in which
//! `emitters` CMA-ES instances each evaluate `population` candidates ranked
//! by archive improvement: CMA-ME (Fontaine et al. 2020) or CMA-MAE with
//! soft archive thresholds (Fontaine and Nikolaidis 2023). An emitter
//! restarts from a random elite when CMA-ES stops or a generation improves
//! nothing. Every generation is evaluated on `inner_threads` threads; the
//! archive is updated serially in candidate order, so the run does not
//! depend on the number of threads.
use crate::{
    Control,
    registry::{Kind, Model, Problem, QualityDiversity, Registry},
    threads::Threads,
};
use fcmaes_core::{
    Archive, Cmaes, CmaesParams, Fitness, MapElitesParams, Rng,
    mapelites::{iso_dd, variation},
    retry_run_seed,
};
use optimization_job_api::{
    JobSpec, Result,
    result::{Evidence, Outcome},
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{path::Path, sync::OnceLock};

pub const PROFILE: &str = "fcmaes.map-elites/1";
/// Default candidates per generation.
pub const CHUNK_SIZE: usize = 20;
/// Default uniform samples per CVT niche.
pub const SAMPLES_PER_NICHE: usize = 20;
/// Default CMA-ES emitters.
pub const EMITTERS: usize = 4;
/// Default population of each CMA-ES emitter.
pub const EMITTER_POPULATION: usize = 16;
/// Default initial step size of a CMA-ES emitter in the normalized box.
pub const EMITTER_SIGMA: f64 = 0.1;
/// Default CMA-MAE archive learning rate (Fontaine and Nikolaidis 2023).
pub const ARCHIVE_LEARNING_RATE: f64 = 0.01;
/// Largest archive capacity, candidate batch or emitter batch times
/// dimension one run may hold.
const MAX_CELLS: usize = 10_000_000;
/// Largest `capacity^2 * samples_per_niche`, the cost of one k-means step.
const MAX_CVT_WORK: usize = 1_000_000_000;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub problem: Problem,
    pub archive: ArchiveConfig,
    pub optimizer: Optimizer,
    pub seed: u64,
    pub runs: usize,
    pub generations: u64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveConfig {
    /// Number of niches.
    pub capacity: usize,
    /// `grid` (two-dimensional descriptors; absent in the resolved job) or
    /// `cvt`. Absent: `grid` for two-dimensional descriptors, else `cvt`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// CVT: uniform samples per niche of the k-means tessellation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples_per_niche: Option<usize>,
    /// CVT: seed of the tessellation, shared by all runs; the job's seed
    /// when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Optimizer {
    pub kind: String,
    #[serde(default)]
    pub parameters: Parameters,
}
/// Emitter settings. `sbx` uses `dis_c`/`dis_m`; `iso_dd` uses
/// `iso_sigma`/`line_sigma` in raw variable units. Settings of the other
/// variation are rejected. The CMA-ES settings apply only with
/// `cma_generations > 0` and are absent from the resolved job otherwise.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub chunk_size: Option<usize>,
    pub variation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dis_c: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dis_m: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iso_sigma: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_sigma: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cma_generations: Option<u64>,
    /// `cma_me` or `cma_mae`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emitter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emitters: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma: Option<f64>,
    /// `cma_mae`: the threshold learning rate alpha.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive_learning_rate: Option<f64>,
    /// Threshold of an empty niche: required for `cma_mae`; `cma_me` ranks
    /// a candidate for an empty niche by `fitness - threshold_init`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_init: Option<f64>,
}

/// How a CMA-ES emitter ranks candidates; lower is better.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ranking {
    /// CMA-ME improvement ranking: `fitness - elite` for an occupied niche,
    /// `fitness - threshold_init` for an empty one; a candidate improves
    /// the archive if its niche was empty or it beats the elite.
    Elite { threshold_init: f64 },
    /// CMA-MAE: `fitness - threshold` of the niche; a candidate improves
    /// the archive if it lies below the threshold, which then moves
    /// towards its fitness by `learning_rate`.
    Threshold {
        threshold_init: f64,
        learning_rate: f64,
    },
}
/// The CMA-ES emitter phase of a run.
#[derive(Clone, Debug)]
pub struct CmaPhase {
    pub generations: u64,
    pub emitters: usize,
    pub population: usize,
    pub sigma: f64,
    pub ranking: Ranking,
}

pub fn describe() -> Value {
    let d = MapElitesParams::default();
    json!({"profile":PROFILE,
        "limit":"exact: generations * chunk_size + cma_generations * emitters * population per run",
        "archive":{"capacity":"1..10000 niches","kind":"grid (two-dimensional descriptors, the default for them) or cvt (any descriptor dimension, the default otherwise)",
            "samples_per_niche":format!("cvt: uniform samples per niche of the k-means tessellation, default {SAMPLES_PER_NICHE}; capacity^2 * samples_per_niche <= 1e9"),
            "seed":"cvt: seed of the tessellation, default the job seed; every run of a job shares its niches"},
        "defaults":{"chunk_size":CHUNK_SIZE,"variation":"sbx","dis_c":d.dis_c,"dis_m":d.dis_m,
            "iso_sigma":d.iso_sigma,"line_sigma":d.line_sigma,"cma_generations":0,
            "emitter":"cma_me","emitters":EMITTERS,"population":EMITTER_POPULATION,"sigma":EMITTER_SIGMA,
            "archive_learning_rate":ARCHIVE_LEARNING_RATE,"threshold_init":"cma_me: 0; cma_mae: required"},
        "ranges":{"chunk_size":"even, 2..10000","dis_c":"> 0","dis_m":"> 0",
            "iso_sigma":">= 0, raw units","line_sigma":">= 0","emitters":"1..1000","population":"4..10000",
            "sigma":"(0,2], normalized box units","archive_learning_rate":"[0,1]","threshold_init":"finite"},
        "emitter":{"cma_me":"CMA-ME improvement ranking (Fontaine et al. 2020): fitness - elite, or fitness - threshold_init for an empty niche",
            "cma_mae":"CMA-MAE (Fontaine and Nikolaidis 2023): fitness - niche threshold; a lower fitness moves the threshold by archive_learning_rate",
            "restart":"from a random elite when CMA-ES stops or a generation improves nothing"},
        "generations":"SBX or Iso+LineDD generations first, then cma_generations; one of them positive",
        "inner_threads":"evaluates each generation on inner_threads threads",
        "result":"re-evaluated elites with niche, fitness, descriptor and visits; coverage and qd_score"})
}

pub struct Run {
    payload: Payload,
    model: QualityDiversity,
    params: MapElitesParams,
    cma: Option<CmaPhase>,
    /// Empty archive with the niches of every run, built on first use.
    template: OnceLock<std::result::Result<Archive, String>>,
    control: Control,
    threads: usize,
}

pub fn prepare(registry: &Registry, s: &mut JobSpec, base: &Path) -> Result<Box<dyn Prepared>> {
    let mut p: Payload = serde_json::from_value(s.payload.clone())?;
    let (problem, model) =
        registry.resolve_as(p.problem.clone(), s, base, Kind::QualityDiversity)?;
    let Model::QualityDiversity(model) = model else {
        unreachable!("resolve_as returns the requested kind")
    };
    if p.optimizer.kind != "map_elites" {
        return Err("fcmaes.map-elites/1 requires optimizer.kind map_elites".into());
    }
    let n = problem.config.dimension;
    let capacity = p.archive.capacity;
    if !(1..=10_000).contains(&capacity) || capacity.saturating_mul(n) > MAX_CELLS {
        return Err("archive capacity must be 1..10000 with capacity * dimension <= 1e7".into());
    }
    p.archive = archive_config(&p.archive, model.descriptor_lower.len(), p.seed)?;
    let d = MapElitesParams::default();
    let q = p.optimizer.parameters.clone();
    let chunk_size = q.chunk_size.unwrap_or(CHUNK_SIZE);
    if !(2..=10_000).contains(&chunk_size)
        || !chunk_size.is_multiple_of(2)
        || chunk_size.saturating_mul(n) > MAX_CELLS
    {
        return Err(
            "chunk_size must be even and 2..10000 with chunk_size * dimension <= 1e7".into(),
        );
    }
    let variation = q.variation.clone().unwrap_or_else(|| "sbx".into());
    let params = match variation.as_str() {
        "sbx" => {
            if q.iso_sigma.is_some() || q.line_sigma.is_some() {
                return Err("iso_sigma and line_sigma apply only to variation iso_dd".into());
            }
            let dis_c = q.dis_c.unwrap_or(d.dis_c);
            let dis_m = q.dis_m.unwrap_or(d.dis_m);
            if !(dis_c.is_finite() && dis_c > 0.0 && dis_m.is_finite() && dis_m > 0.0) {
                return Err("dis_c and dis_m must be positive and finite".into());
            }
            MapElitesParams {
                use_sbx: true,
                dis_c,
                dis_m,
                ..d
            }
        }
        "iso_dd" => {
            if q.dis_c.is_some() || q.dis_m.is_some() {
                return Err("dis_c and dis_m apply only to variation sbx".into());
            }
            let iso_sigma = q.iso_sigma.unwrap_or(d.iso_sigma);
            let line_sigma = q.line_sigma.unwrap_or(d.line_sigma);
            if !(iso_sigma.is_finite()
                && iso_sigma >= 0.0
                && line_sigma.is_finite()
                && line_sigma >= 0.0)
            {
                return Err("iso_sigma and line_sigma must be finite and nonnegative".into());
            }
            MapElitesParams {
                use_sbx: false,
                iso_sigma,
                line_sigma,
                ..d
            }
        }
        _ => return Err("variation must be sbx or iso_dd".into()),
    };
    let params = MapElitesParams {
        generations: usize::try_from(p.generations).map_err(|_| "generations too large")?,
        chunk_size,
        cma_generations: 0,
        ..params
    };
    let cma = cma_phase(&q, n)?;
    let cma_generations = cma.as_ref().map_or(0, |c| c.generations);
    if p.runs == 0 || (p.generations == 0 && cma_generations == 0) {
        return Err("positive runs and generations (or cma_generations) required".into());
    }
    let per_run = p
        .generations
        .checked_mul(chunk_size as u64)
        .and_then(|e| {
            let c = cma.as_ref().map_or(Some(0), |c| {
                c.generations
                    .checked_mul((c.emitters * c.population) as u64)
            })?;
            e.checked_add(c)
        })
        .ok_or("evaluation budget overflow")?;
    per_run
        .checked_mul(p.runs as u64)
        .ok_or("evaluation budget overflow")?;
    p.problem = problem;
    let mut resolved = Parameters {
        chunk_size: Some(chunk_size),
        variation: Some(variation),
        dis_c: params.use_sbx.then_some(params.dis_c),
        dis_m: params.use_sbx.then_some(params.dis_m),
        iso_sigma: (!params.use_sbx).then_some(params.iso_sigma),
        line_sigma: (!params.use_sbx).then_some(params.line_sigma),
        ..Parameters::default()
    };
    if let Some(c) = &cma {
        resolved.cma_generations = Some(c.generations);
        resolved.emitters = Some(c.emitters);
        resolved.population = Some(c.population);
        resolved.sigma = Some(c.sigma);
        match c.ranking {
            Ranking::Elite { threshold_init } => {
                resolved.emitter = Some("cma_me".into());
                resolved.threshold_init = Some(threshold_init);
            }
            Ranking::Threshold {
                threshold_init,
                learning_rate,
            } => {
                resolved.emitter = Some("cma_mae".into());
                resolved.threshold_init = Some(threshold_init);
                resolved.archive_learning_rate = Some(learning_rate);
            }
        }
    }
    p.optimizer.parameters = resolved;
    s.payload = serde_json::to_value(&p)?;
    Ok(Box::new(Run {
        payload: p,
        model,
        params,
        cma,
        template: OnceLock::new(),
        control: Control::default(),
        threads: s.resources.inner_threads,
    }))
}

/// Resolves the archive kind: grid for two-dimensional descriptors unless
/// `cvt` is asked for, CVT otherwise.
fn archive_config(a: &ArchiveConfig, descriptors: usize, seed: u64) -> Result<ArchiveConfig> {
    let cvt = match a.kind.as_deref() {
        None => descriptors != 2,
        Some("grid") if descriptors == 2 => false,
        Some("grid") => {
            return Err(format!(
                "grid archives need two-dimensional descriptors; this model has {descriptors}; use kind cvt"
            )
            .into());
        }
        Some("cvt") => true,
        Some(_) => return Err("archive kind must be grid or cvt".into()),
    };
    if !cvt {
        if a.samples_per_niche.is_some() || a.seed.is_some() {
            return Err("samples_per_niche and seed apply only to cvt archives".into());
        }
        return Ok(ArchiveConfig {
            capacity: a.capacity,
            kind: None,
            samples_per_niche: None,
            seed: None,
        });
    }
    let samples = a.samples_per_niche.unwrap_or(SAMPLES_PER_NICHE);
    if samples == 0
        || a.capacity
            .checked_mul(a.capacity)
            .and_then(|c| c.checked_mul(samples))
            .is_none_or(|w| w > MAX_CVT_WORK)
    {
        return Err(
            "cvt archives need samples_per_niche >= 1 and capacity^2 * samples_per_niche <= 1e9"
                .into(),
        );
    }
    Ok(ArchiveConfig {
        capacity: a.capacity,
        kind: Some("cvt".into()),
        samples_per_niche: Some(samples),
        seed: Some(a.seed.unwrap_or(seed)),
    })
}

/// Validates the CMA-ES emitter settings; `None` without CMA generations.
fn cma_phase(q: &Parameters, n: usize) -> Result<Option<CmaPhase>> {
    let generations = q.cma_generations.unwrap_or(0);
    if generations == 0 {
        if q.emitter.is_some()
            || q.emitters.is_some()
            || q.population.is_some()
            || q.sigma.is_some()
            || q.archive_learning_rate.is_some()
            || q.threshold_init.is_some()
        {
            return Err(
                "emitter, emitters, population, sigma, archive_learning_rate and threshold_init need cma_generations > 0"
                    .into(),
            );
        }
        return Ok(None);
    }
    let emitters = q.emitters.unwrap_or(EMITTERS);
    let population = q.population.unwrap_or(EMITTER_POPULATION);
    if !(1..=1000).contains(&emitters)
        || !(4..=10_000).contains(&population)
        || (emitters * population).saturating_mul(n) > MAX_CELLS
    {
        return Err(
            "emitters must be 1..1000 and population 4..10000 with emitters * population * dimension <= 1e7"
                .into(),
        );
    }
    let sigma = q.sigma.unwrap_or(EMITTER_SIGMA);
    if !sigma.is_finite() || sigma <= 0.0 || sigma > 2.0 {
        return Err("sigma must lie in (0, 2] (normalized box units)".into());
    }
    let finite = |v: f64| -> Result<f64> {
        if v.is_finite() {
            Ok(v)
        } else {
            Err("threshold_init must be finite".into())
        }
    };
    let ranking = match q.emitter.as_deref().unwrap_or("cma_me") {
        "cma_me" => {
            if q.archive_learning_rate.is_some() {
                return Err("archive_learning_rate applies only to emitter cma_mae".into());
            }
            Ranking::Elite {
                threshold_init: finite(q.threshold_init.unwrap_or(0.0))?,
            }
        }
        "cma_mae" => {
            let threshold_init = q.threshold_init.ok_or(
                "cma_mae requires threshold_init, the threshold of an empty niche (a fitness worth keeping)",
            )?;
            let learning_rate = q.archive_learning_rate.unwrap_or(ARCHIVE_LEARNING_RATE);
            if !(0.0..=1.0).contains(&learning_rate) {
                return Err("archive_learning_rate must lie in [0, 1]".into());
            }
            Ranking::Threshold {
                threshold_init: finite(threshold_init)?,
                learning_rate,
            }
        }
        _ => return Err("emitter must be cma_me or cma_mae".into()),
    };
    Ok(Some(CmaPhase {
        generations,
        emitters,
        population,
        sigma,
        ranking,
    }))
}

/// An empty archive with the job's niches: a grid, or a CVT tessellation
/// from the job's archive seed.
pub fn empty_archive(
    problem: &Problem,
    model: &QualityDiversity,
    config: &ArchiveConfig,
) -> Result<Archive> {
    let mut rng = Rng::new(config.seed.unwrap_or(0));
    Ok(Archive::try_new(
        problem.config.dimension,
        &model.descriptor_lower,
        &model.descriptor_upper,
        config.capacity,
        config.samples_per_niche.unwrap_or(0),
        &mut rng,
    )?)
}

/// The archive of one run, with the number of completed generations of
/// each phase and the reason it ended.
pub struct MapElitesRun {
    pub archive: Archive,
    pub generations: u64,
    pub cma_generations: u64,
    pub emitter_restarts: u64,
    pub termination: &'static str,
}

/// Evaluates `xs` on `threads`, in order.
fn evaluate(
    model: &QualityDiversity,
    xs: &[Vec<f64>],
    threads: &Threads,
) -> Result<Vec<(f64, Vec<f64>)>> {
    threads.map(xs, |x| model.values(x)).into_iter().collect()
}

/// Seeds a copy of `template` from `seed` and runs the SBX or Iso+LineDD
/// emitter, then the CMA-ES emitters. The first phase is
/// `fcmaes_core::map_elites_batch` without its CMA-ES generations, with
/// `stop` checked before every generation; an uninterrupted run without
/// `cma` gives the same archive as that function.
#[allow(clippy::too_many_arguments)]
pub fn run_map_elites(
    problem: &Problem,
    model: &QualityDiversity,
    template: &Archive,
    params: &MapElitesParams,
    cma: Option<&CmaPhase>,
    seed: u64,
    stop: impl Fn() -> Option<&'static str>,
    threads: &Threads,
) -> Result<MapElitesRun> {
    let (lower, upper) = (&problem.bounds.lower, &problem.bounds.upper);
    let mut rng = Rng::new(seed);
    let mut archive = template.clone();
    archive.seed_uniform(lower, upper, &mut rng);
    let mut select_n = archive.capacity();
    let mut done = 0;
    let mut termination = "evaluation_limit";
    while done < params.generations as u64 {
        if let Some(reason) = stop() {
            termination = reason;
            break;
        }
        let xs = if params.use_sbx {
            let pop = archive.random_xs(select_n, params.chunk_size, &mut rng);
            variation(&pop, lower, upper, &mut rng, params.dis_c, params.dis_m)
        } else {
            let x1 = archive.random_xs(select_n, params.chunk_size, &mut rng);
            let x2 = archive.random_xs(select_n, params.chunk_size, &mut rng);
            iso_dd(
                &x1,
                &x2,
                lower,
                upper,
                &mut rng,
                params.iso_sigma,
                params.line_sigma,
            )
        };
        let values = evaluate(model, &xs, threads)?;
        archive.update_evaluated(&xs, &values)?;
        archive.argsort();
        select_n = archive.occupied().max(1);
        done += 1;
    }
    let mut run = MapElitesRun {
        archive,
        generations: done,
        cma_generations: 0,
        emitter_restarts: 0,
        termination,
    };
    if let Some(c) = cma
        && run.termination == "evaluation_limit"
    {
        cma_emitters(problem, model, c, &mut run, &mut rng, &stop, threads)?;
    }
    Ok(run)
}

fn emitter(archive: &Archive, problem: &Problem, c: &CmaPhase, sigma: f64, rng: &mut Rng) -> Cmaes {
    let (lower, upper) = (&problem.bounds.lower, &problem.bounds.upper);
    let (x0, _) = archive.random_x_one(archive.occupied().max(1), rng);
    let mut fitness = Fitness::bounded(problem.config.dimension, 1, lower, upper);
    fitness.set_normalize(true);
    Cmaes::new(
        fitness,
        &x0,
        &[sigma],
        &CmaesParams {
            popsize: c.population as i32,
            max_evaluations: u64::MAX,
            seed: rng.int_below(i64::MAX) as u64,
            ..Default::default()
        },
    )
}

/// The CMA-ES emitter phase: every generation asks each emitter for a
/// population, evaluates all of them together, then updates the archive
/// emitter by emitter in candidate order and tells each emitter its
/// candidates' ranking values.
fn cma_emitters(
    problem: &Problem,
    model: &QualityDiversity,
    c: &CmaPhase,
    run: &mut MapElitesRun,
    rng: &mut Rng,
    stop: &impl Fn() -> Option<&'static str>,
    threads: &Threads,
) -> Result<()> {
    let (lower, upper) = (&problem.bounds.lower, &problem.bounds.upper);
    let archive = &mut run.archive;
    let mut thresholds = match c.ranking {
        Ranking::Threshold { threshold_init, .. } => vec![threshold_init; archive.capacity()],
        Ranking::Elite { .. } => Vec::new(),
    };
    let mut emitters: Vec<Cmaes> = (0..c.emitters)
        .map(|_| emitter(archive, problem, c, c.sigma, rng))
        .collect();
    while run.cma_generations < c.generations {
        if let Some(reason) = stop() {
            run.termination = reason;
            break;
        }
        let mut xs = Vec::with_capacity(c.emitters * c.population);
        for e in emitters.iter_mut() {
            for x in e.ask() {
                // Decoding can leave a coordinate an ulp outside the box.
                let x: Vec<f64> = x
                    .iter()
                    .zip(lower.iter().zip(upper))
                    .map(|(v, (l, u))| v.clamp(*l, *u))
                    .collect();
                xs.push(x);
            }
        }
        let values = evaluate(model, &xs, threads)?;
        let mut restart = vec![false; c.emitters];
        for (k, e) in emitters.iter_mut().enumerate() {
            let range = k * c.population..(k + 1) * c.population;
            let mut ranking = Vec::with_capacity(c.population);
            let mut improved = false;
            for (x, (fitness, descriptor)) in xs[range.clone()].iter().zip(&values[range]) {
                if !fitness.is_finite() || descriptor.iter().any(|v| !v.is_finite()) {
                    ranking.push(f64::MAX);
                    continue;
                }
                let niche = archive.index_of_niche(descriptor);
                let elite = archive.ys()[niche];
                let value = match c.ranking {
                    Ranking::Elite { threshold_init } => {
                        improved |= elite.is_infinite() || *fitness < elite;
                        if elite.is_finite() {
                            fitness - elite
                        } else {
                            fitness - threshold_init
                        }
                    }
                    Ranking::Threshold { learning_rate, .. } => {
                        let threshold = thresholds[niche];
                        if *fitness < threshold {
                            improved = true;
                            thresholds[niche] =
                                (1.0 - learning_rate) * threshold + learning_rate * fitness;
                        }
                        fitness - threshold
                    }
                };
                archive.set(niche, *fitness, descriptor, x);
                ranking.push(value);
            }
            restart[k] = e.tell(&ranking) != 0 || !improved;
        }
        archive.argsort();
        for (k, again) in restart.into_iter().enumerate() {
            if again {
                emitters[k] = emitter(archive, problem, c, c.sigma, rng);
                run.emitter_restarts += 1;
            }
        }
        run.cma_generations += 1;
    }
    Ok(())
}

impl Run {
    fn template(&self, threads: &Threads) -> Result<&Archive> {
        let template = self.template.get_or_init(|| {
            threads.install(|| {
                empty_archive(&self.payload.problem, &self.model, &self.payload.archive)
                    .map_err(|e| e.to_string())
            })
        });
        Ok(template.as_ref().map_err(|e| e.clone())?)
    }
}

impl Prepared for Run {
    fn tasks(&self) -> usize {
        self.payload.runs
    }
    fn cancel(&self) {
        self.control.cancel();
    }
    fn solve(&self, task: usize, _: f64) -> Result<TaskResult> {
        let seed = retry_run_seed(self.payload.seed, task);
        let problem = &self.payload.problem;
        let threads = Threads::new(self.threads)?;
        let run = run_map_elites(
            problem,
            &self.model,
            self.template(&threads)?,
            &self.params,
            self.cma.as_ref(),
            seed,
            || self.control.stop(),
            &threads,
        )?;
        let a = &run.archive;
        let occupied_niches: Vec<usize> = (0..a.capacity())
            .filter(|&i| a.ys()[i].is_finite())
            .collect();
        let values = threads.map(&occupied_niches, |&i| self.model.values(&a.xs()[i]));
        let mut elites = Vec::new();
        let mut best: Option<usize> = None;
        let mut total = 0.0;
        for (&niche, values) in occupied_niches.iter().zip(values) {
            let x = &a.xs()[niche];
            let (fitness, descriptor) = values?;
            if !problem.contains(x)
                || fitness != a.ys()[niche]
                || descriptor != a.descriptors()[niche]
                || a.index_of_niche(&descriptor) != niche
            {
                return Err("MAP-Elites elite failed independent re-evaluation".into());
            }
            if best.is_none_or(|b| fitness < a.ys()[b]) {
                best = Some(niche);
            }
            total += fitness;
            elites.push(json!({"niche":niche,"x":x,"fitness":fitness,
                "descriptor":descriptor,"visits":a.counts()[niche]}));
        }
        let occupied = elites.len();
        let grid = a
            .grid_layout()
            .map(|g| json!({"rows":g.rows,"base_columns":g.base_columns,"extra_columns":g.extra_columns}));
        let mut details = json!({"run_seed":seed,"generations":run.generations,
            "requested_generations":self.payload.generations,
            "chunk_size":self.params.chunk_size,"capacity":a.capacity(),"occupied":occupied,
            "coverage":occupied as f64 / a.capacity() as f64,"qd_score":a.qd_score(),
            "mean_fitness":if occupied > 0 { json!(total / occupied as f64) } else { Value::Null },
            "archive_kind":if grid.is_some() { "grid" } else { "cvt" },
            "grid":grid,"descriptor_lower":self.model.descriptor_lower,
            "descriptor_upper":self.model.descriptor_upper,"elites":elites,
            "global_optimum_certified":false});
        let mut evaluations = run.generations * self.params.chunk_size as u64;
        if let Some(c) = &self.cma {
            details["cma_generations"] = json!(run.cma_generations);
            details["requested_cma_generations"] = json!(c.generations);
            details["emitter_restarts"] = json!(run.emitter_restarts);
            evaluations += run.cma_generations * (c.emitters * c.population) as u64;
        }
        Ok(TaskResult {
            outcome: Outcome {
                termination: run.termination.into(),
                claim: if best.is_some() {
                    "elite_archive"
                } else {
                    "none"
                }
                .into(),
                feasible: best.map(|_| true),
                objective: best.map(|b| a.ys()[b]),
                x: best.map(|b| a.xs()[b].clone()),
                details,
                evidence: Some(if best.is_some() {
                    Evidence::rechecked(&["objective", "bounds", "niche"])
                } else {
                    Evidence::claimed(&[])
                }),
                ..Outcome::default()
            },
            work: [
                ("evaluations".into(), evaluations),
                ("validation_evaluations".into(), occupied as u64),
            ]
            .into(),
        })
    }
}
