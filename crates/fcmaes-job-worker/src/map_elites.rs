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
    registry::{Kind, Model, Probe, Problem, QualityDiversity, Registry},
    schema,
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
/// Largest `capacity^2 * samples_per_niche * descriptor dimension`, the
/// cost of one k-means step.
const MAX_CVT_WORK: usize = 1_000_000_000;
/// Most uniform samples of a CVT tessellation.
const MAX_CVT_SAMPLES: usize = 1_000_000;
/// Largest descriptor dimension.
const MAX_DESCRIPTORS: usize = 100;
/// Largest `emitters * dimension^2`: each CMA-ES emitter holds
/// dimension-by-dimension matrices.
const MAX_EMITTER_MATRIX: usize = 10_000_000;

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
    /// `cma_mae`: the threshold of an empty niche, required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_init: Option<f64>,
}

/// How a CMA-ES emitter ranks candidates; lower is better.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ranking {
    /// CMA-ME's improvement ranking (Fontaine et al. 2020): candidates that
    /// fill an empty niche first, by fitness; then candidates that beat their
    /// niche's elite, by `fitness - elite`; then the rest, by `fitness -
    /// elite`. The first two improve the archive.
    Improvement,
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
            "samples_per_niche":format!("cvt: uniform samples per niche of the k-means tessellation, default {SAMPLES_PER_NICHE}; capacity * samples_per_niche <= 1e6 and capacity^2 * samples_per_niche * descriptor dimension <= 1e9"),
            "descriptors":"1..100 dimensions",
            "seed":"cvt: seed of the tessellation, default the job seed; every run of a job shares its niches"},
        "defaults":{"chunk_size":CHUNK_SIZE,"variation":"sbx","dis_c":d.dis_c,"dis_m":d.dis_m,
            "iso_sigma":d.iso_sigma,"line_sigma":d.line_sigma,"cma_generations":0,
            "emitter":"cma_me","emitters":EMITTERS,"population":EMITTER_POPULATION,"sigma":EMITTER_SIGMA,
            "archive_learning_rate":ARCHIVE_LEARNING_RATE,"threshold_init":"cma_mae: required"},
        "ranges":{"chunk_size":"even, 2..10000","dis_c":"> 0","dis_m":"> 0",
            "iso_sigma":">= 0, raw units","line_sigma":">= 0","emitters":"1..1000; emitters * dimension^2 <= 1e7","population":"4..10000",
            "sigma":"(0,2], normalized box units","archive_learning_rate":"[0,1]","threshold_init":"finite"},
        "emitter":{"cma_me":"CMA-ME improvement ranking (Fontaine et al. 2020): candidates filling an empty niche first (by fitness), then those beating their elite (by fitness - elite), then the rest",
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
    let descriptors = model.descriptor_lower.len();
    if descriptors > MAX_DESCRIPTORS {
        return Err(format!(
            "{} has {descriptors} descriptor dimensions; at most 100",
            problem.id
        )
        .into());
    }
    p.archive = archive_config(&p.archive, descriptors, p.seed)?;
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
            Ranking::Improvement => resolved.emitter = Some("cma_me".into()),
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
            .checked_mul(samples)
            .is_none_or(|n| n > MAX_CVT_SAMPLES)
        || a.capacity
            .checked_mul(a.capacity)
            .and_then(|c| c.checked_mul(samples))
            .and_then(|c| c.checked_mul(descriptors))
            .is_none_or(|w| w > MAX_CVT_WORK)
    {
        return Err(
            "cvt archives need samples_per_niche >= 1, capacity * samples_per_niche <= 1e6 and capacity^2 * samples_per_niche * descriptor dimension <= 1e9"
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
        || emitters.saturating_mul(n).saturating_mul(n) > MAX_EMITTER_MATRIX
    {
        return Err(
            "emitters must be 1..1000 and population 4..10000 with emitters * population * dimension <= 1e7 and emitters * dimension^2 <= 1e7"
                .into(),
        );
    }
    let sigma = q.sigma.unwrap_or(EMITTER_SIGMA);
    if !sigma.is_finite() || sigma <= 0.0 || sigma > 2.0 {
        return Err("sigma must lie in (0, 2] (normalized box units)".into());
    }
    let ranking = match q.emitter.as_deref().unwrap_or("cma_me") {
        "cma_me" => {
            if q.archive_learning_rate.is_some() || q.threshold_init.is_some() {
                return Err(
                    "archive_learning_rate and threshold_init apply only to emitter cma_mae".into(),
                );
            }
            Ranking::Improvement
        }
        "cma_mae" => {
            let threshold_init = q.threshold_init.ok_or(
                "cma_mae requires threshold_init, the threshold of an empty niche (a fitness worth keeping)",
            )?;
            let learning_rate = q.archive_learning_rate.unwrap_or(ARCHIVE_LEARNING_RATE);
            if !(0.0..=1.0).contains(&learning_rate) {
                return Err("archive_learning_rate must lie in [0, 1]".into());
            }
            if !threshold_init.is_finite() {
                return Err("threshold_init must be finite".into());
            }
            Ranking::Threshold {
                threshold_init,
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

/// Ranking groups of an emitter's candidates, best first.
const NEW_NICHE: u8 = 0;
const IMPROVED: u8 = 1;
const UNCHANGED: u8 = 2;
/// A non-finite fitness or descriptor: ranked last, never stored.
const INVALID: u8 = 3;

/// The values CMA-ES ranks an emitter's candidates by. CMA-MAE ranks by
/// `fitness - threshold` alone. CMA-ME ranks by group, then by value within
/// the group: the groups are shifted apart by a multiple of the largest
/// magnitude in the batch, so the order is exact while CMA-ES's step-size
/// and stop rules still see values of the problem's scale.
fn ranking_values(ranked: &[(u8, f64)], ranking: Ranking) -> Vec<f64> {
    if let Ranking::Threshold { .. } = ranking {
        return ranked
            .iter()
            .map(|&(g, v)| if g == INVALID { f64::MAX } else { v })
            .collect();
    }
    let largest = ranked
        .iter()
        .filter(|(g, _)| *g != INVALID)
        .fold(0.0_f64, |m, (_, v)| m.max(v.abs()));
    // With 4L + 1, every group lies at least 2L + 1 below the next one.
    let shift = 4.0 * largest + 1.0;
    if !(2.0 * shift).is_finite() {
        // Values near f64::MAX: rank positions keep the order.
        let mut order: Vec<usize> = (0..ranked.len()).collect();
        order.sort_by(|&a, &b| {
            let (ga, va) = ranked[a];
            let (gb, vb) = ranked[b];
            ga.cmp(&gb).then(va.total_cmp(&vb))
        });
        let mut values = vec![0.0; ranked.len()];
        for (rank, i) in order.into_iter().enumerate() {
            values[i] = rank as f64;
        }
        return values;
    }
    ranked
        .iter()
        .map(|&(g, v)| match g {
            NEW_NICHE => v - 2.0 * shift,
            IMPROVED => v - shift,
            UNCHANGED => v,
            _ => f64::MAX,
        })
        .collect()
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
        Ranking::Improvement => Vec::new(),
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
            // (group, value): ranked by group, then by value.
            let mut ranked = Vec::with_capacity(c.population);
            for (x, (fitness, descriptor)) in xs[range.clone()].iter().zip(&values[range]) {
                if !fitness.is_finite() || descriptor.iter().any(|v| !v.is_finite()) {
                    ranked.push((INVALID, 0.0));
                    continue;
                }
                let niche = archive.index_of_niche(descriptor);
                let elite = archive.ys()[niche];
                ranked.push(match c.ranking {
                    Ranking::Improvement if elite.is_infinite() => (NEW_NICHE, *fitness),
                    Ranking::Improvement if *fitness < elite => (IMPROVED, fitness - elite),
                    Ranking::Improvement => (UNCHANGED, fitness - elite),
                    Ranking::Threshold { learning_rate, .. } => {
                        let threshold = thresholds[niche];
                        let group = if *fitness < threshold {
                            thresholds[niche] =
                                (1.0 - learning_rate) * threshold + learning_rate * fitness;
                            IMPROVED
                        } else {
                            UNCHANGED
                        };
                        (group, fitness - threshold)
                    }
                });
                archive.set(niche, *fitness, descriptor, x);
            }
            let improved = ranked.iter().any(|(g, _)| *g < UNCHANGED);
            restart[k] = e.tell(&ranking_values(&ranked, c.ranking)) != 0 || !improved;
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
        // The whole task runs on its own threads, so no Rayon work of the
        // task (or of the model) reaches the global pool.
        let threads = Threads::new(self.threads)?;
        threads.install(|| self.task(task, &threads))
    }
}

impl Run {
    fn task(&self, task: usize, threads: &Threads) -> Result<TaskResult> {
        let seed = retry_run_seed(self.payload.seed, task);
        let problem = &self.payload.problem;
        let run = run_map_elites(
            problem,
            &self.model,
            self.template(threads)?,
            &self.params,
            self.cma.as_ref(),
            seed,
            || self.control.stop(),
            threads,
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

/// JSON Schema of the payload, with `problem` the schema of its problems.
pub fn payload_schema(problem: Value) -> Value {
    use crate::schema::{absent, integer, number, or_null, payload, u64};
    let nonnegative = || or_null(number(Some(0.0), None, false));
    let positive = || or_null(number(Some(0.0), None, true));
    let cma = [
        "emitter",
        "emitters",
        "population",
        "sigma",
        "archive_learning_rate",
        "threshold_init",
    ];
    let mut chunk = or_null(integer(2, 10_000));
    chunk["multipleOf"] = json!(2);
    let parameters = json!({"type":"object","additionalProperties":false,"properties":{
            "chunk_size":chunk,"variation":{"enum":["sbx","iso_dd",null]},"dis_c":positive(),"dis_m":positive(),
            "iso_sigma":nonnegative(),"line_sigma":nonnegative(),"cma_generations":or_null(integer(0, u64::MAX)),
            "emitter":{"enum":["cma_me","cma_mae",null]},"emitters":or_null(integer(1, 1000)),
            "population":or_null(integer(4, 10_000)),"sigma":or_null(number(Some(0.0), Some(2.0), true)),
            "archive_learning_rate":or_null(number(Some(0.0), Some(1.0), false)),
            "threshold_init":or_null(json!({"type":"number"}))},
        "allOf":[
            {"if":{"required":["variation"],"properties":{"variation":{"const":"iso_dd"}}},
                "then":absent(&["dis_c","dis_m"]),"else":absent(&["iso_sigma","line_sigma"])},
            {"if":{"required":["cma_generations"],"properties":{"cma_generations":{"type":"integer","minimum":1}}},
                "else":absent(&cma)},
            {"if":{"required":["emitter"],"properties":{"emitter":{"const":"cma_mae"}}},
                "then":{"required":["threshold_init"],"properties":{"threshold_init":{"type":"number"}}},
                "else":absent(&["archive_learning_rate","threshold_init"])}]});
    payload(
        PROFILE,
        json!({"type":"object","additionalProperties":false,
        "required":["problem","archive","optimizer","seed","runs","generations"],
        "properties":{"problem":problem,
            "archive":{"type":"object","additionalProperties":false,"required":["capacity"],
                "properties":{"capacity":integer(1, 10_000),"kind":{"enum":["grid","cvt",null]},
                    "samples_per_niche":or_null(integer(1, MAX_CVT_SAMPLES as u64)),"seed":or_null(u64())},
                "if":{"required":["kind"],"properties":{"kind":{"const":"grid"}}},
                "then":absent(&["samples_per_niche","seed"])},
            "optimizer":{"type":"object","additionalProperties":false,"required":["kind"],
                "properties":{"kind":{"const":"map_elites"},"parameters":parameters}},
            "seed":u64(),"runs":integer(1, 1_000_000),"generations":integer(0, u64::MAX)},
        "if":{"properties":{"generations":{"const":0}}},
        "then":{"required":["optimizer"],"properties":{"optimizer":{"required":["parameters"],
            "properties":{"parameters":{"required":["cma_generations"],
                "properties":{"cma_generations":{"type":"integer","minimum":1}}}}}}},
        "$comment":"archive.kind defaults to grid for two-dimensional descriptors (problems: descriptor_bounds) and to cvt otherwise; a grid needs two-dimensional descriptors and takes no samples_per_niche or seed. capacity, chunk_size and emitters times population, each times dimension, <= 1e7; emitters times dimension^2 <= 1e7; cvt: capacity times samples_per_niche <= 1e6 and capacity^2 times samples_per_niche times descriptor dimension <= 1e9; 1..100 descriptor dimensions. inner_threads: the threads evaluating a generation (results do not depend on them)."}),
    )
}

/// The `examples` entry of this profile on `probe`.
pub fn example(probe: &Probe) -> Value {
    schema::example(
        PROFILE,
        format!(
            "MAP-Elites on {}: 16 niches, 1 run of 20 generations of 10",
            probe.id
        ),
        probe,
        json!({"problem":probe.problem(),"archive":{"capacity":16},
            "optimizer":{"kind":"map_elites","parameters":{"chunk_size":10}},"seed":1,"runs":1,"generations":20}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(values: &[f64]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        order
    }

    #[test]
    fn cma_me_ranks_new_niches_first_then_improvements_then_the_rest() {
        let ranked = [
            (UNCHANGED, 1.0),  // fitness 6 against elite 5
            (NEW_NICHE, 10.0), // fills an empty niche
            (IMPROVED, -0.5),
            (INVALID, 0.0),
            (NEW_NICHE, 3.0),
            (IMPROVED, -2.0),
            (UNCHANGED, 0.0),
        ];
        let values = ranking_values(&ranked, Ranking::Improvement);
        assert_eq!(order(&values), [4, 1, 5, 2, 6, 0, 3]);
        // Extreme magnitudes keep the order through rank positions.
        let huge = [(UNCHANGED, -1e308), (NEW_NICHE, 1e308), (IMPROVED, -1e308)];
        assert_eq!(
            order(&ranking_values(&huge, Ranking::Improvement)),
            [1, 2, 0]
        );
        // CMA-MAE ranks by fitness - threshold alone.
        let mae = Ranking::Threshold {
            threshold_init: 0.0,
            learning_rate: 0.01,
        };
        assert_eq!(
            ranking_values(&[(UNCHANGED, 1.0), (IMPROVED, -1.0), (INVALID, 0.0)], mae),
            [1.0, -1.0, f64::MAX]
        );
    }
}
