//! `fcmaes.mode/1`: independent MODE runs on registered multi-objective
//! models. Each task is one seeded run of `generations` asked populations;
//! its outcome is the feasible non-dominated part of the final population,
//! re-evaluated outside the optimizer, with its hypervolume (exact through
//! four objectives, sampled above). Populations are
//! evaluated on `inner_threads` threads; the run does not depend on their
//! number.
use crate::{
    Control,
    registry::{Kind, Model, MultiObjective, Probe, Problem, Registry},
    schema,
    threads::Threads,
};
use fcmaes_core::{
    Fitness, Mode, ModeParams, OutsidePolicy, ReferencePoint, hypervolume_with,
    indicators::HypervolumeEstimate, pareto_indices, retry_run_seed,
};
use optimization_job_api::{
    JobSpec, Result,
    result::{Evidence, Outcome},
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

pub const PROFILE: &str = "fcmaes.mode/1";
/// MODE replaces non-finite values by this sentinel before ranking.
const BIG: f64 = f64::MAX;
/// Largest population times dimension one run may hold.
const MAX_CELLS: usize = 10_000_000;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub problem: Problem,
    pub optimizer: Optimizer,
    pub seed: u64,
    pub runs: usize,
    pub generations: u64,
    #[serde(default)]
    pub reference_point: Option<Vec<f64>>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Optimizer {
    pub kind: String,
    #[serde(default)]
    pub parameters: Parameters,
}
/// `ModeParams` without seeds and integer settings; validation fills in the
/// `fcmaes-core` defaults.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub population: Option<usize>,
    pub nsga_update: Option<bool>,
    pub pareto_update: Option<f64>,
    pub f: Option<f64>,
    pub cr: Option<f64>,
    pub pro_c: Option<f64>,
    pub dis_c: Option<f64>,
    pub pro_m: Option<f64>,
    pub dis_m: Option<f64>,
}

pub fn describe() -> Value {
    let d = ModeParams::default();
    json!({"profile":PROFILE,"limit":"exact: generations * population per run",
        "defaults":{"population":d.popsize,"nsga_update":d.nsga_update,"pareto_update":d.pareto_update,
            "f":d.f,"cr":d.cr,"pro_c":d.pro_c,"dis_c":d.dis_c,"pro_m":d.pro_m,"dis_m":d.dis_m},
        "ranges":{"population":"4..10000; 4..1000 with three objectives","f":"(0,2]","cr":"(0,1]","pro_c":"[0,1]","pro_m":"[0,1]",
            "dis_c":"> 0","dis_m":"> 0","pareto_update":"[0,1]"},
        "reference_point":"required unless the model defines one",
        "inner_threads":"evaluates each population on inner_threads threads",
        "result":"feasible non-dominated final population members, re-evaluated, with their hypervolume",
        "hypervolume":"exact through four objectives (method exact); above four a deterministic Monte Carlo estimate (method monte_carlo, with samples, seed and standard_error)"})
}

pub struct Run {
    payload: Payload,
    model: MultiObjective,
    params: ModeParams,
    reference: ReferencePoint,
    control: Control,
    threads: usize,
}

fn finite(name: &str, value: f64, valid: bool) -> Result<f64> {
    if value.is_finite() && valid {
        Ok(value)
    } else {
        Err(format!("mode parameter {name} = {value} is out of range").into())
    }
}

pub fn prepare(registry: &Registry, s: &mut JobSpec, base: &Path) -> Result<Box<dyn Prepared>> {
    let mut p: Payload = serde_json::from_value(s.payload.clone())?;
    let (problem, model) = registry.resolve_as(p.problem.clone(), s, base, Kind::MultiObjective)?;
    let Model::MultiObjective(model) = model else {
        unreachable!("resolve_as returns the requested kind")
    };
    if p.optimizer.kind != "mode" {
        return Err("fcmaes.mode/1 requires optimizer.kind mode".into());
    }
    if p.runs == 0 || p.generations == 0 {
        return Err("positive runs and generations required".into());
    }
    let d = ModeParams::default();
    let q = &p.optimizer.parameters;
    let population = q.population.unwrap_or(d.popsize as usize);
    let n = problem.config.dimension;
    // The exact hypervolume of a front of F points costs O(F^2) for two
    // objectives and O(F^3) for three; it runs after the last cancel check.
    let max_population = if model.objectives >= 3 { 1_000 } else { 10_000 };
    if !(4..=max_population).contains(&population) || population.saturating_mul(n) > MAX_CELLS {
        return Err(format!(
            "mode population must be 4..{max_population} for {} with population * dimension <= 1e7",
            problem.id
        )
        .into());
    }
    let f = q.f.unwrap_or(d.f);
    let cr = q.cr.unwrap_or(d.cr);
    let pro_c = q.pro_c.unwrap_or(d.pro_c);
    let pro_m = q.pro_m.unwrap_or(d.pro_m);
    let dis_c = q.dis_c.unwrap_or(d.dis_c);
    let dis_m = q.dis_m.unwrap_or(d.dis_m);
    let pareto_update = q.pareto_update.unwrap_or(d.pareto_update);
    let params = ModeParams {
        popsize: population as i32,
        f: finite("f", f, f > 0.0 && f <= 2.0)?,
        cr: finite("cr", cr, cr > 0.0 && cr <= 1.0)?,
        pro_c: finite("pro_c", pro_c, (0.0..=1.0).contains(&pro_c))?,
        dis_c: finite("dis_c", dis_c, dis_c > 0.0)?,
        pro_m: finite("pro_m", pro_m, (0.0..=1.0).contains(&pro_m))?,
        dis_m: finite("dis_m", dis_m, dis_m > 0.0)?,
        nsga_update: q.nsga_update.unwrap_or(d.nsga_update),
        pareto_update: finite(
            "pareto_update",
            pareto_update,
            (0.0..=1.0).contains(&pareto_update),
        )?,
        ..d
    };
    p.generations
        .checked_mul(population as u64)
        .and_then(|e| e.checked_mul(p.runs as u64))
        .ok_or("evaluation budget overflow")?;
    let reference = p
        .reference_point
        .clone()
        .or_else(|| model.reference_point.clone())
        .ok_or_else(|| {
            format!(
                "{} defines no reference point; give reference_point",
                problem.id
            )
        })?;
    if reference.len() != model.objectives {
        return Err(format!("reference_point needs {} coordinates", model.objectives).into());
    }
    let reference =
        ReferencePoint::new(reference).map_err(|e| format!("reference_point: {e:?}"))?;
    // MODE validates its own configuration; construct one to check it now
    // rather than in every task.
    Mode::try_new(
        fitness(&problem, &model),
        model.objectives,
        model.constraints,
        None,
        &params,
    )?;
    p.problem = problem;
    p.reference_point = Some(reference.as_slice().to_vec());
    p.optimizer.parameters = Parameters {
        population: Some(population),
        nsga_update: Some(params.nsga_update),
        pareto_update: Some(params.pareto_update),
        f: Some(params.f),
        cr: Some(params.cr),
        pro_c: Some(params.pro_c),
        dis_c: Some(params.dis_c),
        pro_m: Some(params.pro_m),
        dis_m: Some(params.dis_m),
    };
    s.payload = serde_json::to_value(&p)?;
    Ok(Box::new(Run {
        payload: p,
        model,
        params,
        reference,
        control: Control::default(),
        threads: s.resources.inner_threads,
    }))
}

fn fitness(problem: &Problem, model: &MultiObjective) -> Fitness {
    Fitness::bounded(
        problem.config.dimension,
        model.objectives + model.constraints,
        &problem.bounds.lower,
        &problem.bounds.upper,
    )
}

/// The values MODE stores for an evaluation: non-finite entries become `BIG`.
pub fn sanitized(values: Vec<f64>) -> Vec<f64> {
    values
        .into_iter()
        .map(|v| if v.is_finite() { v } else { BIG })
        .collect()
}

/// Final population of one MODE run, with the number of completed
/// generations and the reason it ended.
pub struct ModeRun {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<Vec<f64>>,
    pub generations: u64,
    pub termination: &'static str,
}

/// Runs MODE from `seed` for at most `generations`, evaluating each asked
/// population on `threads`, in order. `stop` is checked before every
/// generation.
pub fn run_mode(
    problem: &Problem,
    model: &MultiObjective,
    params: &ModeParams,
    seed: u64,
    generations: u64,
    stop: impl Fn() -> Option<&'static str>,
    threads: &Threads,
) -> Result<ModeRun> {
    let mut mode = Mode::try_new(
        fitness(problem, model),
        model.objectives,
        model.constraints,
        None,
        &ModeParams {
            seed,
            runid: 0,
            ..params.clone()
        },
    )?;
    let mut done = 0;
    let mut termination = "evaluation_limit";
    while done < generations {
        if let Some(reason) = stop() {
            termination = reason;
            break;
        }
        let xs = mode.try_ask()?;
        let ys = threads
            .map(&xs, |x| model.values(x))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        mode.try_tell(&ys)?;
        done += 1;
    }
    let r = mode.result();
    Ok(ModeRun {
        x: r.x,
        y: r.y,
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
        let run = run_mode(
            problem,
            &self.model,
            &self.params,
            seed,
            self.payload.generations,
            || self.control.stop(),
            threads,
        )?;
        let population = self.params.popsize as u64;
        let (nobj, ncon) = (self.model.objectives, self.model.constraints);
        let mut validation = 0;
        let mut unevaluated = 0;
        let mut feasible = Vec::new();
        // Survivors that were never evaluated still carry the sentinel.
        let evaluated: Vec<usize> = (0..run.x.len())
            .filter(|&i| run.generations > 0 && !run.y[i].iter().all(|v| *v == BIG))
            .collect();
        let values = threads.map(&evaluated, |&i| self.model.values(&run.x[i]));
        if run.generations > 0 {
            unevaluated = run.x.len() - evaluated.len();
            for (&i, values) in evaluated.iter().zip(values) {
                let (x, stored) = (&run.x[i], &run.y[i]);
                let values = sanitized(values?);
                validation += 1;
                if !problem.contains(x) || &values != stored {
                    return Err("MODE member failed independent re-evaluation".into());
                }
                if values[..nobj].iter().all(|v| *v < BIG)
                    && values[nobj..].iter().all(|c| *c <= 0.0)
                    && !feasible
                        .iter()
                        .any(|(other, _): &(&Vec<f64>, Vec<f64>)| *other == x)
                {
                    feasible.push((x, values));
                }
            }
        }
        let objectives: Vec<Vec<f64>> = feasible.iter().map(|(_, v)| v[..nobj].to_vec()).collect();
        let front = pareto_indices(&objectives, nobj).unwrap_or_default();
        let points: Vec<Value> = front
            .iter()
            .map(|&i| {
                let (x, v) = &feasible[i];
                json!({"x":x,"objectives":&v[..nobj],"constraints":&v[nobj..nobj + ncon]})
            })
            .collect();
        let front_objectives: Vec<Vec<f64>> =
            front.iter().map(|&i| objectives[i].clone()).collect();
        let hypervolume = if front_objectives.is_empty() {
            Value::Null
        } else {
            let h = hypervolume_with(&front_objectives, &self.reference, OutsidePolicy::Exclude)
                .map_err(|e| format!("hypervolume: {e:?}"))?;
            let mut v = json!({"value":h.estimate.value(),
                "exact":matches!(h.estimate, HypervolumeEstimate::Exact(_)),"method":"exact",
                "reference_point":self.reference.as_slice(),"normalization":"none",
                "retained_points":h.retained_points,"outside_reference":h.outside_reference});
            // Above four objectives the volume is sampled: keep what makes
            // the estimate reproducible and comparable.
            if let HypervolumeEstimate::MonteCarlo {
                standard_error,
                samples,
                seed,
                ..
            } = h.estimate
            {
                v["method"] = json!("monte_carlo");
                v["standard_error"] = json!(standard_error);
                v["samples"] = json!(samples);
                v["seed"] = json!(seed);
            }
            v
        };
        let claimed = !points.is_empty();
        let checks: &[&str] = if ncon > 0 {
            &["objective", "bounds", "constraints", "non_dominance"]
        } else {
            &["objective", "bounds", "non_dominance"]
        };
        Ok(TaskResult {
            outcome: Outcome {
                termination: run.termination.into(),
                claim: if claimed { "non_dominated_set" } else { "none" }.into(),
                feasible: (run.generations > 0).then_some(claimed),
                objective: None,
                x: None,
                // The front is the feasible non-dominated part of the final
                // population, not an archive of every point evaluated (as in
                // COCO's bbob-biobj); the hypervolume uses raw objectives.
                details: json!({"run_seed":seed,"generations":run.generations,
                    "requested_generations":self.payload.generations,"population":population,
                    "objectives":nobj,"constraints":ncon,"front_source":"final_population",
                    "front_size":points.len(),"feasible_members":feasible.len(),
                    "unevaluated_members":unevaluated,"front":points,"hypervolume":hypervolume,
                    "pareto_optimality_certified":false}),
                evidence: Some(if claimed {
                    Evidence::rechecked(checks)
                } else {
                    Evidence::claimed(&[])
                }),
                ..Outcome::default()
            },
            work: [
                ("evaluations".into(), run.generations * population),
                ("validation_evaluations".into(), validation),
            ]
            .into(),
        })
    }
}

/// JSON Schema of the payload, with `problem` the schema of its problems.
pub fn payload_schema(problem: Value) -> Value {
    use crate::schema::{integer, number, numbers, or_null, payload, u64};
    let unit = || or_null(number(Some(0.0), Some(1.0), false));
    let positive = || or_null(number(Some(0.0), None, true));
    payload(
        PROFILE,
        json!({"type":"object","additionalProperties":false,
        "required":["problem","optimizer","seed","runs","generations"],
        "properties":{"problem":problem,
            "optimizer":{"type":"object","additionalProperties":false,"required":["kind"],
                "properties":{"kind":{"const":"mode"},
                    "parameters":{"type":"object","additionalProperties":false,"properties":{
                        "population":or_null(integer(4, 10_000)),"nsga_update":or_null(json!({"type":"boolean"})),
                        "pareto_update":unit(),"f":or_null(number(Some(0.0), Some(2.0), true)),
                        "cr":or_null(number(Some(0.0), Some(1.0), true)),"pro_c":unit(),"dis_c":positive(),
                        "pro_m":unit(),"dis_m":positive()}}}},
            "seed":u64(),"runs":integer(1, 1_000_000),"generations":integer(1, u64::MAX),
            "reference_point":or_null(numbers())},
        "$comment":"population <= 1000 for models with three or more objectives (problems: objectives) and population times dimension <= 1e7; reference_point: one value per objective, required unless the model's problems entry has one; inner_threads: the threads evaluating a population (results do not depend on them); generations times population times runs must not overflow."}),
    )
}

/// The `examples` entry of this profile on `probe`.
pub fn example(probe: &Probe) -> Value {
    let mut payload = json!({"problem":probe.problem(),"optimizer":{"kind":"mode","parameters":{"population":16}},
        "seed":1,"runs":1,"generations":20});
    if let Some(point) = &probe.reference_point {
        payload["reference_point"] = json!(point);
    }
    schema::example(
        PROFILE,
        format!("MODE on {}: 1 run of 20 generations of 16", probe.id),
        probe,
        payload,
    )
}
