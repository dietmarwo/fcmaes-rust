//! `fcmaes.map-elites/1`: independent MAP-Elites runs on registered
//! quality-diversity models. Each task is one seeded run on a regular
//! two-dimensional grid archive; its outcome is the re-evaluated archive.
use crate::{Control, problems};
use fcmaes_core::{
    Archive, MapElitesParams, Rng,
    mapelites::{iso_dd, variation},
    retry_run_seed,
};
use optimization_job_api::{
    JobSpec, Result, legacy,
    result::Outcome,
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const PROFILE: &str = "fcmaes.map-elites/1";
/// Default candidates per generation.
pub const CHUNK_SIZE: usize = 20;
/// Largest archive capacity times dimension one run may hold.
const MAX_CELLS: usize = 10_000_000;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub problem: legacy::Problem,
    pub archive: ArchiveConfig,
    pub optimizer: Optimizer,
    pub seed: u64,
    pub runs: usize,
    pub generations: u64,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveConfig {
    /// Number of niches of the regular two-dimensional grid.
    pub capacity: usize,
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
/// variation are rejected.
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
}

pub fn describe() -> Value {
    let d = MapElitesParams::default();
    json!({"profile":PROFILE,"limit":"exact: generations * chunk_size per run",
        "archive":"regular two-dimensional grid with `capacity` niches (1..10000)",
        "defaults":{"chunk_size":CHUNK_SIZE,"variation":"sbx","dis_c":d.dis_c,"dis_m":d.dis_m,
            "iso_sigma":d.iso_sigma,"line_sigma":d.line_sigma},
        "ranges":{"chunk_size":"even, 2..10000","dis_c":"> 0","dis_m":"> 0",
            "iso_sigma":">= 0, raw units","line_sigma":">= 0"},
        "excluded":"CMA-ES emitter generations and CVT archives (unbounded or parallel work)",
        "result":"re-evaluated elites with niche, fitness, descriptor and visits; coverage and qd_score"})
}

pub struct Run {
    payload: Payload,
    model: problems::QualityDiversity,
    params: MapElitesParams,
    control: Control,
}

pub fn prepare(s: &mut JobSpec) -> Result<Box<dyn Prepared>> {
    let mut p: Payload = serde_json::from_value(s.payload.clone())?;
    let (problem, model) = problems::quality_diversity(p.problem.clone())?;
    if p.optimizer.kind != "map_elites" {
        return Err("fcmaes.map-elites/1 requires optimizer.kind map_elites".into());
    }
    if p.runs == 0 || p.generations == 0 {
        return Err("positive runs and generations required".into());
    }
    let n = problem.config.dimension;
    let capacity = p.archive.capacity;
    if !(1..=10_000).contains(&capacity) || capacity.saturating_mul(n) > MAX_CELLS {
        return Err("archive capacity must be 1..10000 with capacity * dimension <= 1e7".into());
    }
    let d = MapElitesParams::default();
    let q = p.optimizer.parameters.clone();
    let chunk_size = q.chunk_size.unwrap_or(CHUNK_SIZE);
    if !(2..=10_000).contains(&chunk_size) || !chunk_size.is_multiple_of(2) {
        return Err("chunk_size must be even and 2..10000".into());
    }
    let variation = q.variation.unwrap_or_else(|| "sbx".into());
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
    p.generations
        .checked_mul(chunk_size as u64)
        .and_then(|e| e.checked_mul(p.runs as u64))
        .ok_or("evaluation budget overflow")?;
    p.problem = problem;
    p.optimizer.parameters = Parameters {
        chunk_size: Some(chunk_size),
        variation: Some(variation),
        dis_c: params.use_sbx.then_some(params.dis_c),
        dis_m: params.use_sbx.then_some(params.dis_m),
        iso_sigma: (!params.use_sbx).then_some(params.iso_sigma),
        line_sigma: (!params.use_sbx).then_some(params.line_sigma),
    };
    s.payload = serde_json::to_value(&p)?;
    Ok(Box::new(Run {
        payload: p,
        model,
        params,
        control: Control::default(),
    }))
}

/// The archive of one run, with the number of completed generations and the
/// reason it ended.
pub struct MapElitesRun {
    pub archive: Archive,
    pub generations: u64,
    pub termination: &'static str,
}

/// Seeds a grid archive from `seed` and runs the SBX or Iso+LineDD emitter.
/// The loop is `fcmaes_core::map_elites_batch` without CMA-ES generations,
/// with `stop` checked before every generation; an uninterrupted run gives
/// the same archive as that function.
pub fn run_map_elites(
    problem: &legacy::Problem,
    model: &problems::QualityDiversity,
    capacity: usize,
    params: &MapElitesParams,
    seed: u64,
    stop: impl Fn() -> Option<&'static str>,
) -> Result<MapElitesRun> {
    let (lower, upper) = (&problem.bounds.lower, &problem.bounds.upper);
    let mut rng = Rng::new(seed);
    let mut archive = Archive::try_new(
        problem.config.dimension,
        &model.descriptor_lower,
        &model.descriptor_upper,
        capacity,
        0,
        &mut rng,
    )?;
    archive.seed_uniform(lower, upper, &mut rng);
    let mut evaluate = |xs: &[Vec<f64>]| xs.iter().map(|x| (model.evaluate)(x)).collect();
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
        archive.update_batch(&xs, &mut evaluate)?;
        archive.argsort();
        select_n = archive.occupied().max(1);
        done += 1;
    }
    Ok(MapElitesRun {
        archive,
        generations: done,
        termination,
    })
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
        let run = run_map_elites(
            problem,
            &self.model,
            self.payload.archive.capacity,
            &self.params,
            seed,
            || self.control.stop(),
        )?;
        let a = &run.archive;
        let mut elites = Vec::new();
        let mut best: Option<usize> = None;
        let mut total = 0.0;
        for niche in (0..a.capacity()).filter(|&i| a.ys()[i].is_finite()) {
            let x = &a.xs()[niche];
            let (fitness, descriptor) = (self.model.evaluate)(x);
            if x.iter()
                .zip(problem.bounds.lower.iter().zip(&problem.bounds.upper))
                .any(|(v, (l, u))| !v.is_finite() || v < l || v > u)
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
                details: json!({"run_seed":seed,"generations":run.generations,
                    "requested_generations":self.payload.generations,
                    "chunk_size":self.params.chunk_size,"capacity":a.capacity(),"occupied":occupied,
                    "coverage":occupied as f64 / a.capacity() as f64,"qd_score":a.qd_score(),
                    "mean_fitness":if occupied > 0 { json!(total / occupied as f64) } else { Value::Null },
                    "grid":grid,"descriptor_lower":self.model.descriptor_lower,
                    "descriptor_upper":self.model.descriptor_upper,"elites":elites,
                    "global_optimum_certified":false}),
            },
            work: [
                (
                    "evaluations".into(),
                    run.generations * self.params.chunk_size as u64,
                ),
                ("validation_evaluations".into(), occupied as u64),
            ]
            .into(),
        })
    }
}
