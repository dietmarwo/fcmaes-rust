use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{self, Write as FmtWrite};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cmaes::{CMAESOptions, DVector, Weights};
use cpu_time::ProcessTime;
use fcmaes_core::{
    Cmaes, CmaesParams, Fitness, RetryConfig, RetryContext, RetryRunResult, Rng, retry,
};
use fcmaes_examples::benchmark_gtop::{BenchmarkCase, selected_bite_cases};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    ExternalSingle,
    ExternalSequential,
    ExternalRetry,
    FcmaesRetry,
}

impl Arm {
    pub const ALL: [Self; 4] = [
        Self::ExternalSingle,
        Self::ExternalSequential,
        Self::ExternalRetry,
        Self::FcmaesRetry,
    ];

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "external-single" => Ok(Self::ExternalSingle),
            "external-sequential" => Ok(Self::ExternalSequential),
            "external-retry" => Ok(Self::ExternalRetry),
            "fcmaes-retry" => Ok(Self::FcmaesRetry),
            _ => Err(format!(
                "unknown arm '{value}'; expected external-single, external-sequential, external-retry, or fcmaes-retry"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ExternalSingle => "external cmaes, one serial lane",
            Self::ExternalSequential => "external cmaes, sequential retries",
            Self::ExternalRetry => "external cmaes + fcmaes retry",
            Self::FcmaesRetry => "fcmaes-core CMA-ES + retry",
        }
    }
}

impl fmt::Display for Arm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ExternalSingle => "external-single",
            Self::ExternalSequential => "external-sequential",
            Self::ExternalRetry => "external-retry",
            Self::FcmaesRetry => "fcmaes-retry",
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Phase {
    EqualWall,
    FixedWork,
    Target,
}

impl Phase {
    pub const ALL: [Self; 3] = [Self::EqualWall, Self::FixedWork, Self::Target];

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "equal-wall" => Ok(Self::EqualWall),
            "fixed-work" => Ok(Self::FixedWork),
            "target" => Ok(Self::Target),
            _ => Err(format!(
                "unknown phase '{value}'; expected equal-wall, fixed-work, or target"
            )),
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EqualWall => "equal-wall",
            Self::FixedWork => "fixed-work",
            Self::Target => "target",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Campaign,
    Report,
    Verify,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Preset {
    Smoke,
    Pilot,
    Publication,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub mode: Mode,
    pub preset: Preset,
    pub phases: Vec<Phase>,
    pub arms: Vec<Arm>,
    pub problem_keys: Vec<String>,
    pub runs: usize,
    pub retries: usize,
    pub evaluations: u64,
    pub wall_time_ms: u64,
    pub workers: Vec<usize>,
    pub population: Option<usize>,
    pub sigma: f64,
    pub seed: u64,
    pub output: PathBuf,
    pub resume: bool,
}

impl Config {
    pub const USAGE: &str = "options:\n\
      --mode MODE              campaign, report, or verify (default campaign)\n\
      --preset NAME            smoke, pilot, or publication (default smoke)\n\
      --phases CSV             equal-wall,fixed-work,target\n\
      --arms CSV               external-single,external-sequential,external-retry,fcmaes-retry\n\
      --problems CSV           GTOP problem keys\n\
      --runs N                 paired top-level experiments\n\
      --retries N              starts in fixed-work and target multistart arms\n\
      --evaluations N          safety ceiling per CMA-ES start\n\
      --wall-time-ms N         deadline per CMA-ES start in equal-wall phase\n\
      --workers CSV            retry lane/scaling points; defaults end at physical cores\n\
      --population N           common CMA-ES population; 0 uses 4+floor(3*ln(dim))\n\
      --sigma X                initial sigma in normalized [-1,1] coordinates\n\
      --seed N                 deterministic campaign root seed\n\
      --output PATH            result directory\n\
      --resume                 append only missing protocol rows\n\
      --help                   show this help\n\
\n\
Publication defaults: seven non-Messenger-Full GTOP cases; 100 paired runs;\n\
4,000 ms per arm; one serial restart lane versus one lane per physical core.\n\
Every individual CMA-ES instance is single-threaded.";

    pub fn from_env() -> Result<Option<Self>, String> {
        Self::from_args(std::env::args().skip(1))
    }

    pub fn from_args<I, S>(args: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut raw: Vec<String> = args
            .into_iter()
            .map(|argument| argument.as_ref().to_owned())
            .collect();
        if raw
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            return Ok(None);
        }
        let preset = value_after(&raw, "--preset")
            .map(parse_preset)
            .transpose()?
            .unwrap_or(Preset::Smoke);
        let physical = num_cpus::get_physical().max(1);
        let mut config = match preset {
            Preset::Smoke => Self {
                mode: Mode::Campaign,
                preset,
                phases: Phase::ALL.to_vec(),
                arms: Arm::ALL.to_vec(),
                problem_keys: vec!["cassini1".to_owned()],
                runs: 1,
                retries: 4,
                evaluations: 200,
                wall_time_ms: 10,
                workers: unique_sorted(vec![1, physical.min(2)]),
                population: None,
                sigma: 0.3,
                seed: 42,
                output: PathBuf::from("results/harness-smoke"),
                resume: false,
            },
            Preset::Pilot => Self {
                mode: Mode::Campaign,
                preset,
                phases: vec![Phase::FixedWork],
                arms: Arm::ALL.to_vec(),
                problem_keys: vec![
                    "cassini1".to_owned(),
                    "rosetta".to_owned(),
                    "tandem".to_owned(),
                ],
                runs: 5,
                retries: 40,
                evaluations: 2_000,
                wall_time_ms: 100,
                workers: scaling_workers(physical),
                population: None,
                sigma: 0.3,
                seed: 42,
                output: PathBuf::from("results/scaling-pilot-v1"),
                resume: false,
            },
            Preset::Publication => Self {
                mode: Mode::Campaign,
                preset,
                phases: vec![Phase::EqualWall],
                arms: vec![Arm::ExternalSingle, Arm::ExternalRetry],
                problem_keys: vec![
                    "cassini1".to_owned(),
                    "cassini2".to_owned(),
                    "gtoc1".to_owned(),
                    "messenger".to_owned(),
                    "rosetta".to_owned(),
                    "sagas".to_owned(),
                    "tandem".to_owned(),
                ],
                runs: 100,
                retries: physical,
                evaluations: 1_000_000_000,
                wall_time_ms: 4_000,
                workers: unique_sorted(vec![1, physical]),
                population: None,
                sigma: 0.3,
                seed: 42,
                output: PathBuf::from("results/equal-wall-100-v2"),
                resume: false,
            },
        };

        let mut index = 0;
        while index < raw.len() {
            let argument = raw[index].as_str();
            match argument {
                "--resume" => {
                    config.resume = true;
                    index += 1;
                }
                "--preset" => index += 2,
                _ => {
                    let value = raw
                        .get(index + 1)
                        .ok_or_else(|| format!("{argument} requires a value"))?;
                    match argument {
                        "--mode" => config.mode = parse_mode(value)?,
                        "--phases" => config.phases = parse_csv(value, Phase::parse)?,
                        "--arms" => config.arms = parse_csv(value, Arm::parse)?,
                        "--problems" => config.problem_keys = parse_strings(value)?,
                        "--runs" => config.runs = parse_positive(value, "--runs")?,
                        "--retries" => config.retries = parse_positive(value, "--retries")?,
                        "--evaluations" => {
                            config.evaluations = parse_positive(value, "--evaluations")?
                        }
                        "--wall-time-ms" => {
                            config.wall_time_ms = parse_positive(value, "--wall-time-ms")?
                        }
                        "--workers" => {
                            config.workers = unique_sorted(parse_csv(value, |item| {
                                parse_positive(item, "--workers")
                            })?)
                        }
                        "--population" => {
                            let parsed: usize = value
                                .parse()
                                .map_err(|_| "--population requires an integer".to_owned())?;
                            config.population = (parsed > 0).then_some(parsed);
                        }
                        "--sigma" => {
                            config.sigma = value
                                .parse()
                                .map_err(|_| "--sigma requires a number".to_owned())?
                        }
                        "--seed" => {
                            config.seed = value
                                .parse()
                                .map_err(|_| "--seed requires an integer".to_owned())?
                        }
                        "--output" => config.output = value.into(),
                        _ => return Err(format!("unknown option '{argument}'")),
                    }
                    index += 2;
                }
            }
        }
        raw.clear();
        if config.phases.is_empty()
            || config.arms.is_empty()
            || config.problem_keys.is_empty()
            || config.workers.is_empty()
        {
            return Err("phase, arm, problem, and worker lists must not be empty".to_owned());
        }
        if !config.sigma.is_finite() || config.sigma <= 0.0 {
            return Err("--sigma must be finite and positive".to_owned());
        }
        if config.population == Some(1) {
            return Err("--population must be zero or at least 2".to_owned());
        }
        Ok(Some(config))
    }

    pub fn max_workers(&self) -> usize {
        self.workers.iter().copied().max().unwrap_or(1)
    }
}

fn value_after<'a>(arguments: &'a [String], option: &str) -> Option<&'a str> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == option)
        .map(|pair| pair[1].as_str())
}

fn parse_mode(value: &str) -> Result<Mode, String> {
    match value {
        "campaign" => Ok(Mode::Campaign),
        "report" => Ok(Mode::Report),
        "verify" => Ok(Mode::Verify),
        _ => Err(format!(
            "unknown mode '{value}'; expected campaign, report, or verify"
        )),
    }
}

fn parse_preset(value: &str) -> Result<Preset, String> {
    match value {
        "smoke" => Ok(Preset::Smoke),
        "pilot" => Ok(Preset::Pilot),
        "publication" => Ok(Preset::Publication),
        _ => Err(format!(
            "unknown preset '{value}'; expected smoke, pilot, or publication"
        )),
    }
}

fn parse_positive<T>(value: &str, option: &str) -> Result<T, String>
where
    T: std::str::FromStr + Default + PartialOrd,
{
    let parsed = value
        .parse::<T>()
        .map_err(|_| format!("{option} requires a positive integer"))?;
    if parsed <= T::default() {
        return Err(format!("{option} requires a positive integer"));
    }
    Ok(parsed)
}

fn parse_csv<T>(value: &str, parse: impl Fn(&str) -> Result<T, String>) -> Result<Vec<T>, String> {
    value.split(',').map(parse).collect()
}

fn parse_strings(value: &str) -> Result<Vec<String>, String> {
    let values: Vec<_> = value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect();
    if values.is_empty() {
        Err("--problems requires at least one key".to_owned())
    } else {
        Ok(values)
    }
}

fn unique_sorted(mut values: Vec<usize>) -> Vec<usize> {
    values.sort_unstable();
    values.dedup();
    values
}

fn scaling_workers(physical: usize) -> Vec<usize> {
    let mut workers = vec![1];
    let mut value = 2;
    while value < physical {
        workers.push(value);
        value = value.saturating_mul(2);
    }
    workers.push(physical);
    unique_sorted(workers)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResultRow {
    pub preset: String,
    pub phase: Phase,
    pub arm: Arm,
    pub problem: String,
    pub run: usize,
    pub seed: u64,
    pub workers: usize,
    pub retries_requested: usize,
    pub retries_completed: usize,
    #[serde(default)]
    pub optimizer_starts: u64,
    pub evaluations_per_retry: u64,
    #[serde(default)]
    pub wall_time_ms: u64,
    pub evaluations_actual: u64,
    pub population: usize,
    pub sigma: f64,
    pub stop_value: f64,
    pub success: bool,
    pub best: f64,
    pub wall_seconds: f64,
    pub cpu_seconds: f64,
    pub average_cores: f64,
}

impl ResultRow {
    pub fn key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}",
            self.phase, self.arm, self.problem, self.run, self.seed, self.workers
        )
    }
}

pub fn resolve_cases(keys: &[String]) -> Result<Vec<BenchmarkCase>, String> {
    let all = selected_bite_cases(None)?;
    let by_key: HashMap<_, _> = all.into_iter().map(|case| (case.key, case)).collect();
    keys.iter()
        .map(|key| {
            by_key
                .get(key.as_str())
                .map(clone_case)
                .ok_or_else(|| format!("unknown GTOP benchmark problem '{key}'"))
        })
        .collect()
}

fn clone_case(case: &BenchmarkCase) -> BenchmarkCase {
    BenchmarkCase {
        key: case.key,
        display_name: case.display_name,
        problem: case.problem.clone(),
        max_retries: case.max_retries,
        value_limit: case.value_limit,
        absolute_best: case.absolute_best,
        absolute_best_label: case.absolute_best_label,
        stop_value: case.stop_value,
        stop_value_label: case.stop_value_label,
        slow: case.slow,
    }
}

pub fn campaign_rows(config: &Config) -> Vec<(Phase, Arm, usize)> {
    let max_workers = config.max_workers();
    let mut rows = Vec::new();
    for &phase in &config.phases {
        for &arm in &config.arms {
            match (phase, arm) {
                (Phase::EqualWall, Arm::ExternalSingle) => {
                    rows.push((phase, arm, 1));
                }
                (Phase::EqualWall, Arm::ExternalRetry) => rows.extend(
                    config
                        .workers
                        .iter()
                        .copied()
                        .filter(|workers| *workers > 1)
                        .map(|workers| (phase, arm, workers)),
                ),
                (Phase::EqualWall, _) => {}
                (_, Arm::ExternalSingle | Arm::ExternalSequential) => {
                    rows.push((phase, arm, 1));
                }
                (_, Arm::ExternalRetry) => rows.extend(
                    config
                        .workers
                        .iter()
                        .copied()
                        .filter(|workers| *workers > 1)
                        .map(|workers| (phase, arm, workers)),
                ),
                (_, Arm::FcmaesRetry) => rows.push((phase, arm, max_workers)),
            }
        }
    }
    rows
}

pub fn run_case(
    config: &Config,
    case: &BenchmarkCase,
    phase: Phase,
    arm: Arm,
    workers: usize,
    run: usize,
    seed: u64,
) -> ResultRow {
    let population = config
        .population
        .unwrap_or_else(|| 4 + (3.0 * (case.problem.bounds.dim() as f64).ln()).floor() as usize);
    let retries = match (phase, arm) {
        (Phase::EqualWall, Arm::ExternalSingle) => 1,
        (Phase::EqualWall, Arm::ExternalRetry) => workers,
        (_, Arm::ExternalSingle) => 1,
        _ => config.retries,
    };
    let retry_workers = match arm {
        Arm::ExternalSingle | Arm::ExternalSequential => 1,
        Arm::ExternalRetry | Arm::FcmaesRetry => workers,
    };
    let target = (phase == Phase::Target).then_some(case.stop_value);
    let wall_time = (phase == Phase::EqualWall).then(|| Duration::from_millis(config.wall_time_ms));
    let retry_config = RetryConfig {
        num_retries: retries,
        workers: retry_workers,
        capacity: retries.max(1),
        value_limit: f64::INFINITY,
        stop_fitness: target.unwrap_or(f64::NEG_INFINITY),
        max_evaluations: config.evaluations,
        seed,
        statistic_num: 0,
    };
    let cpu_started = ProcessTime::now();
    let wall_started = Instant::now();
    let start_barrier = Arc::new(Barrier::new(retry_workers));
    let optimizer_starts = Arc::new(AtomicU64::new(0));
    let result = match arm {
        Arm::ExternalSingle | Arm::ExternalSequential | Arm::ExternalRetry => retry(
            &case.problem.objective,
            &case.problem.bounds,
            &retry_config,
            |objective, context| {
                if let Some(duration) = wall_time {
                    start_barrier.wait();
                    external_cmaes_timed_lane(
                        objective,
                        context,
                        population,
                        config.sigma,
                        duration,
                        &optimizer_starts,
                    )
                } else {
                    optimizer_starts.fetch_add(1, Ordering::Relaxed);
                    external_cmaes_run(
                        objective,
                        context,
                        population,
                        config.sigma,
                        target,
                        context.run_seed,
                        None,
                    )
                }
            },
        ),
        Arm::FcmaesRetry => retry(
            &case.problem.objective,
            &case.problem.bounds,
            &retry_config,
            |objective, context| {
                optimizer_starts.fetch_add(1, Ordering::Relaxed);
                fcmaes_cmaes_run(objective, context, population, config.sigma, target)
            },
        ),
    };
    let wall_seconds = wall_started.elapsed().as_secs_f64();
    let cpu_seconds = cpu_started.elapsed().as_secs_f64();
    ResultRow {
        preset: format!("{:?}", config.preset).to_ascii_lowercase(),
        phase,
        arm,
        problem: case.key.to_owned(),
        run,
        seed,
        workers: retry_workers,
        retries_requested: retries,
        retries_completed: result.runs,
        optimizer_starts: optimizer_starts.load(Ordering::Relaxed),
        evaluations_per_retry: config.evaluations,
        wall_time_ms: if phase == Phase::EqualWall {
            config.wall_time_ms
        } else {
            0
        },
        evaluations_actual: result.evaluations,
        population,
        sigma: config.sigma,
        stop_value: case.stop_value,
        success: result.success && result.y <= case.stop_value,
        best: result.y,
        wall_seconds,
        cpu_seconds,
        average_cores: cpu_seconds / wall_seconds.max(1.0e-12),
    }
}

fn normalized_mean(context: &RetryContext, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..context.bounds.dim())
        .map(|_| -1.0 + 2.0 * rng.uniform01())
        .collect()
}

fn decode(normalized: &[f64], context: &RetryContext) -> Vec<f64> {
    normalized
        .iter()
        .zip(context.bounds.lower().iter().zip(context.bounds.upper()))
        .map(|(&value, (&lower, &upper))| {
            let feasible = value.clamp(-1.0, 1.0);
            0.5 * feasible * (upper - lower) + 0.5 * (upper + lower)
        })
        .collect()
}

fn external_cmaes_run<O>(
    objective: &O,
    context: &RetryContext,
    population: usize,
    sigma: f64,
    target: Option<f64>,
    seed: u64,
    max_time: Option<Duration>,
) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let mean = normalized_mean(context, seed);
    let calls = AtomicU64::new(0);
    let function = |point: &DVector<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        let physical = decode(point.as_slice(), context);
        let value = objective(&physical);
        if value.is_finite() { value } else { 1.0e99 }
    };
    let max_evaluations = usize::try_from(context.max_evaluations).unwrap_or(usize::MAX);
    let mut options = CMAESOptions::new(mean.clone(), sigma)
        .population_size(population)
        .weights(Weights::Negative)
        .max_function_evals(max_evaluations)
        .tol_fun(0.0)
        .tol_fun_rel(0.0)
        .tol_fun_hist(0.0)
        .tol_x(-1.0)
        .tol_stagnation(usize::MAX)
        .tol_x_up(f64::MAX)
        .tol_condition_cov(f64::MAX)
        .seed(seed);
    if let Some(stop_value) = target {
        options = options.fun_target(stop_value);
    }
    if let Some(duration) = max_time {
        options = options.max_time(duration);
    }
    let Ok(mut optimizer) = options.build(function) else {
        return RetryRunResult {
            x: decode(&mean, context),
            y: f64::INFINITY,
            evaluations: 0,
        };
    };
    let termination = optimizer.run();
    let evaluations = calls.load(Ordering::Relaxed);
    if let Some(best) = termination.overall_best {
        RetryRunResult {
            x: decode(best.point.as_slice(), context),
            y: best.value,
            evaluations,
        }
    } else {
        RetryRunResult {
            x: decode(&mean, context),
            y: f64::INFINITY,
            evaluations,
        }
    }
}

fn external_cmaes_timed_lane<O>(
    objective: &O,
    context: &RetryContext,
    population: usize,
    sigma: f64,
    max_time: Duration,
    optimizer_starts: &AtomicU64,
) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let started = Instant::now();
    let mut restart = 0_u64;
    let mut total_evaluations = 0_u64;
    let mut best = RetryRunResult {
        x: decode(&normalized_mean(context, context.run_seed), context),
        y: f64::INFINITY,
        evaluations: 0,
    };
    loop {
        let remaining = max_time.saturating_sub(started.elapsed());
        if remaining.is_zero() || (restart > 0 && remaining < Duration::from_millis(1)) {
            break;
        }
        let seed = if restart == 0 {
            context.run_seed
        } else {
            context
                .run_seed
                .wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(restart))
        };
        optimizer_starts.fetch_add(1, Ordering::Relaxed);
        let result = external_cmaes_run(
            objective,
            context,
            population,
            sigma,
            None,
            seed,
            Some(remaining),
        );
        total_evaluations = total_evaluations.saturating_add(result.evaluations);
        if result.y < best.y {
            best = result;
        }
        restart = restart.saturating_add(1);
    }
    best.evaluations = total_evaluations;
    best
}

fn fcmaes_cmaes_run<O>(
    objective: &O,
    context: &RetryContext,
    population: usize,
    sigma: f64,
    target: Option<f64>,
) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let normalized = normalized_mean(context, context.run_seed);
    let physical = decode(&normalized, context);
    let mut fitness = Fitness::bounded(
        context.bounds.dim(),
        1,
        context.bounds.lower(),
        context.bounds.upper(),
    );
    fitness.set_normalize(true);
    let mut optimizer = Cmaes::new(
        fitness,
        &physical,
        &[sigma],
        &CmaesParams {
            popsize: population as i32,
            max_evaluations: context.max_evaluations,
            accuracy: 0.0,
            stop_fitness: target.unwrap_or(f64::NEG_INFINITY),
            stop_tol_hist_fun: 0.0,
            seed: context.run_seed,
            ..Default::default()
        },
    );
    let result = optimizer.optimize(objective, 1);
    RetryRunResult {
        x: result.x,
        y: result.y,
        evaluations: result.evaluations,
    }
}

pub fn load_rows(path: &Path) -> Result<Vec<ResultRow>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut reader = csv::Reader::from_path(path).map_err(|error| error.to_string())?;
    reader
        .deserialize()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

pub fn append_row(path: &Path, row: &ResultRow) -> Result<(), String> {
    let exists = path.exists();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(!exists)
        .from_writer(file);
    writer.serialize(row).map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

pub fn write_manifest(config: &Config, rows: &[ResultRow], command: &str) -> Result<(), String> {
    let manifest = serde_json::json!({
        "schema_version": 2,
        "preset": format!("{:?}", config.preset).to_ascii_lowercase(),
        "phases": config.phases,
        "arms": config.arms,
        "problems": config.problem_keys,
        "runs": config.runs,
        "retries": config.retries,
        "evaluations_per_retry": config.evaluations,
        "equal_wall_time_ms": config.wall_time_ms,
        "workers": config.workers,
        "physical_cpus": num_cpus::get_physical(),
        "logical_cpus": num_cpus::get(),
        "cpu_model": cpu_model(),
        "os": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "rustc": command_version("rustc", &["--version"]),
        "created_unix_seconds": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|duration| duration.as_secs()),
        "population": config.population,
        "sigma": config.sigma,
        "seed": config.seed,
        "rows": rows.len(),
        "command": command,
    });
    let path = config.output.join("run.json");
    let mut file = File::create(&path).map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(&mut file, &manifest).map_err(|error| error.to_string())?;
    writeln!(file).map_err(|error| error.to_string())
}

fn cpu_model() -> Option<String> {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|value| value.split_once(':'))
                    .map(|(_, value)| value.trim().to_owned())
            })
        })
}

fn command_version(command: &str, arguments: &[&str]) -> Option<String> {
    std::process::Command::new(command)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
}

pub fn render_report(output: &Path) -> Result<(), String> {
    let rows = load_rows(&output.join("results.csv"))?;
    if rows.is_empty() {
        return Err(format!("{} contains no result rows", output.display()));
    }
    let report = build_report(&rows);
    fs::write(output.join("comparison.md"), report).map_err(|error| error.to_string())
}

fn build_report(rows: &[ResultRow]) -> String {
    let equal_wall_rows: Vec<_> = rows
        .iter()
        .filter(|row| row.phase == Phase::EqualWall)
        .collect();
    let mut output = if equal_wall_rows.is_empty() {
        String::from(
            "# GTOP: single-threaded CMA-ES through parallel retry\n\n\
This bundle contains secondary scheduling or target-stopping diagnostics. Every\n\
external `cmaes` optimizer instance is serial; `fcmaes_core::retry` supplies only\n\
outer multistart scheduling. See the experiment README and the separate\n\
equal-wall bundle for the user-facing solution-quality comparison.\n\n",
        )
    } else {
        String::from(
            "# GTOP: single-threaded CMA-ES through parallel retry\n\n\
The primary experiment asks a user-facing question: how much better is the\n\
solution distribution when the same wall-time allowance can use all physical\n\
cores? Every external `cmaes` optimizer instance is serial. Because CMA-ES can\n\
terminate protectively before the deadline, each lane immediately starts a new\n\
CMA-ES run and retains its best result. The serial arm uses one restart lane;\n\
the retry arm coordinates one lane per worker through `fcmaes_core::retry`. Both\n\
arms use the same lane deadline, objective, bounds, population, sigma, and\n\
deterministic root-seed scheme.\n\n",
        )
    };

    if !equal_wall_rows.is_empty() {
        output.push_str("## Equal-wall-time solution quality\n\n");
        output.push_str(
            "Each retry pair includes the serial seed stream as lane zero plus additional\n\
independent streams in the parallel arm. Mean and population standard deviation (`Sdev`) of\n\
the best objective are the primary outcomes; smaller is better. `Retry W/T/L`\n\
is the paired win/tie/loss count for parallel retry. Measured wall time audits\n\
deadline comparability in the separate work table.\n\n\
| Problem | Workers | Pairs | Serial success | Retry success | Serial best mean | Serial best sdev | Retry best mean | Retry best sdev | Retry W/T/L |\n\
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        let serial: HashMap<_, _> = equal_wall_rows
            .iter()
            .filter(|row| row.arm == Arm::ExternalSingle)
            .map(|row| ((row.problem.as_str(), row.seed), *row))
            .collect();
        let mut pairs_by_group: BTreeMap<(&str, usize), Vec<(&ResultRow, &ResultRow)>> =
            BTreeMap::new();
        for parallel in equal_wall_rows
            .iter()
            .filter(|row| row.arm == Arm::ExternalRetry)
        {
            if let Some(serial) = serial.get(&(parallel.problem.as_str(), parallel.seed)) {
                pairs_by_group
                    .entry((parallel.problem.as_str(), parallel.workers))
                    .or_default()
                    .push((*serial, *parallel));
            }
        }
        let mut start_ratios = Vec::new();
        for ((problem, workers), pairs) in pairs_by_group {
            let (serial_best, serial_best_sdev) =
                mean_sdev(pairs.iter().map(|(serial, _)| serial.best));
            let (retry_best, retry_best_sdev) =
                mean_sdev(pairs.iter().map(|(_, retry)| retry.best));
            let wins = pairs
                .iter()
                .filter(|(serial, retry)| retry.best < serial.best)
                .count();
            let ties = pairs
                .iter()
                .filter(|(serial, retry)| retry.best.to_bits() == serial.best.to_bits())
                .count();
            let losses = pairs.len() - wins - ties;
            let serial_successes = pairs.iter().filter(|(serial, _)| serial.success).count();
            let retry_successes = pairs.iter().filter(|(_, retry)| retry.success).count();
            let (serial_starts, _) = mean_sdev(
                pairs
                    .iter()
                    .map(|(serial, _)| serial.optimizer_starts as f64),
            );
            let (retry_starts, _) =
                mean_sdev(pairs.iter().map(|(_, retry)| retry.optimizer_starts as f64));
            if serial_starts > 0.0 {
                start_ratios.push(retry_starts / serial_starts);
            }
            writeln!(
                output,
                "| {problem} | {workers} | {} | {serial_successes}/{} | {retry_successes}/{} | {serial_best:.6} | {serial_best_sdev:.6} | {retry_best:.6} | {retry_best_sdev:.6} | {wins}/{ties}/{losses} |",
                pairs.len(),
                pairs.len(),
                pairs.len(),
            )
            .expect("writing to String cannot fail");
        }
        if !start_ratios.is_empty() {
            let minimum_ratio = start_ratios.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum_ratio = start_ratios
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let minimum_expected = 100.0 * minimum_ratio / (minimum_ratio + 1.0);
            let maximum_expected = 100.0 * maximum_ratio / (maximum_ratio + 1.0);
            writeln!(
                output,
                "\nThe paired win count is primarily a scheduler check. Mean-start ratios range from {minimum_ratio:.2}× to {maximum_ratio:.2}×. Under iid independent restarts with no information sharing, a `k`-to-one ratio predicts a retry win probability of `k/(k+1)`, or {minimum_expected:.1}%–{maximum_expected:.1}% here. The observed W/T/L values are consistent with that baseline; the mean and sdev columns quantify the returned solution distribution."
            )
            .expect("writing to String cannot fail");
        }

        output.push_str("\n### Equal-wall work audit\n\n");
        output.push_str(
            "The arms intentionally do not use equal CPU work. These counts document how\n\
parallel retry converts otherwise idle cores into more independent search within\n\
the same elapsed allowance.\n\n\
| Problem | Workers | Deadline | Serial wall mean | Serial wall sdev | Retry wall mean | Retry wall sdev | Serial starts mean | Retry starts mean | Serial evaluations mean | Retry evaluations mean | Serial active cores | Retry active cores |\n\
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        let serial: HashMap<_, _> = equal_wall_rows
            .iter()
            .filter(|row| row.arm == Arm::ExternalSingle)
            .map(|row| ((row.problem.as_str(), row.seed), *row))
            .collect();
        let mut pairs_by_group: BTreeMap<(&str, usize), Vec<(&ResultRow, &ResultRow)>> =
            BTreeMap::new();
        for parallel in equal_wall_rows
            .iter()
            .filter(|row| row.arm == Arm::ExternalRetry)
        {
            if let Some(serial) = serial.get(&(parallel.problem.as_str(), parallel.seed)) {
                pairs_by_group
                    .entry((parallel.problem.as_str(), parallel.workers))
                    .or_default()
                    .push((*serial, *parallel));
            }
        }
        for ((problem, workers), pairs) in pairs_by_group {
            let (serial_starts, _) = mean_sdev(
                pairs
                    .iter()
                    .map(|(serial, _)| serial.optimizer_starts as f64),
            );
            let (serial_evaluations, _) = mean_sdev(
                pairs
                    .iter()
                    .map(|(serial, _)| serial.evaluations_actual as f64),
            );
            let (retry_starts, _) =
                mean_sdev(pairs.iter().map(|(_, retry)| retry.optimizer_starts as f64));
            let (retry_evaluations, _) = mean_sdev(
                pairs
                    .iter()
                    .map(|(_, retry)| retry.evaluations_actual as f64),
            );
            let (serial_active, _) =
                mean_sdev(pairs.iter().map(|(serial, _)| serial.average_cores));
            let (retry_active, _) = mean_sdev(pairs.iter().map(|(_, retry)| retry.average_cores));
            let (serial_wall, serial_wall_sdev) =
                mean_sdev(pairs.iter().map(|(serial, _)| serial.wall_seconds));
            let (retry_wall, retry_wall_sdev) =
                mean_sdev(pairs.iter().map(|(_, retry)| retry.wall_seconds));
            let deadline_ms = pairs[0].1.wall_time_ms;
            writeln!(
                output,
                "| {problem} | {workers} | {deadline_ms} ms | {serial_wall:.6}s | {serial_wall_sdev:.6}s | {retry_wall:.6}s | {retry_wall_sdev:.6}s | {serial_starts:.1} | {retry_starts:.1} | {serial_evaluations:.0} | {retry_evaluations:.0} | {serial_active:.2} | {retry_active:.2} |",
            )
            .expect("writing to String cannot fail");
        }
    }

    let fixed_rows: Vec<_> = rows
        .iter()
        .filter(|row| row.phase == Phase::FixedWork)
        .collect();
    if !fixed_rows.is_empty() {
        output.push_str("\n## Fixed-work scheduling diagnostic\n\n");
        output.push_str(
        "The fixed-work phase disables target stopping. `Same work` counts paired\n\
runs with equal completed retries, evaluations, and final best value. Speedup is\n\
paired sequential wall time divided by parallel wall time. This is a scheduler\n\
diagnostic, not the solution-quality comparison.\n\n\
| Problem | Workers | Pairs | Same work | Mean speedup | Sdev speedup | Efficiency | Mean active cores | Sdev active cores |\n\
|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        let sequential: HashMap<_, _> = rows
            .iter()
            .filter(|row| row.phase == Phase::FixedWork && row.arm == Arm::ExternalSequential)
            .map(|row| ((row.problem.as_str(), row.seed), row))
            .collect();
        let mut scaling: BTreeMap<(&str, usize), Vec<(&ResultRow, &ResultRow)>> = BTreeMap::new();
        for parallel in rows
            .iter()
            .filter(|row| row.phase == Phase::FixedWork && row.arm == Arm::ExternalRetry)
        {
            if let Some(serial) = sequential.get(&(parallel.problem.as_str(), parallel.seed)) {
                scaling
                    .entry((parallel.problem.as_str(), parallel.workers))
                    .or_default()
                    .push((serial, parallel));
            }
        }
        for ((problem, workers), pairs) in scaling {
            let same = pairs
                .iter()
                .filter(|(serial, parallel)| {
                    serial.retries_completed == parallel.retries_completed
                        && serial.evaluations_actual == parallel.evaluations_actual
                        && serial.best.to_bits() == parallel.best.to_bits()
                })
                .count();
            let (speedup, speedup_sdev) = mean_sdev(pairs.iter().map(|(serial, parallel)| {
                serial.wall_seconds / parallel.wall_seconds.max(1.0e-12)
            }));
            let (active, active_sdev) =
                mean_sdev(pairs.iter().map(|(_, parallel)| parallel.average_cores));
            writeln!(
            output,
            "| {problem} | {workers} | {} | {same} | {speedup:.2}× | {speedup_sdev:.2}× | {:.1}% | {active:.2} | {active_sdev:.2} |",
            pairs.len(),
            100.0 * speedup / workers as f64
        )
        .expect("writing to String cannot fail");
        }

        output.push_str("\n### Fixed-work arm summary\n\n");
        output.push_str(
        "Only the paired external sequential/retry rows above are an exact\n\
scheduling comparison. This table exposes the one-start and native-CMA rows\n\
without treating their different searches as core-count speedups.\n\n\
| Problem | Arm | Workers | Runs | Success | Mean best | Sdev best | Mean wall | Sdev wall | Mean evaluations | Sdev evaluations | Mean active cores | Sdev active cores |\n\
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        let mut fixed_groups: BTreeMap<(&str, Arm, usize), Vec<&ResultRow>> = BTreeMap::new();
        for row in rows.iter().filter(|row| row.phase == Phase::FixedWork) {
            fixed_groups
                .entry((row.problem.as_str(), row.arm, row.workers))
                .or_default()
                .push(row);
        }
        for ((problem, arm, workers), group) in fixed_groups {
            let successes = group.iter().filter(|row| row.success).count();
            let (best, best_sdev) = mean_sdev(group.iter().map(|row| row.best));
            let (wall, wall_sdev) = mean_sdev(group.iter().map(|row| row.wall_seconds));
            let (evaluations, evaluations_sdev) =
                mean_sdev(group.iter().map(|row| row.evaluations_actual as f64));
            let (active, active_sdev) = mean_sdev(group.iter().map(|row| row.average_cores));
            writeln!(
            output,
            "| {problem} | {} | {workers} | {} | {:.0}% | {best:.6} | {best_sdev:.6} | {wall:.4}s | {wall_sdev:.4}s | {evaluations:.0} | {evaluations_sdev:.0} | {active:.2} | {active_sdev:.2} |",
            arm.label(),
            group.len(),
            100.0 * successes as f64 / group.len() as f64,
        )
        .expect("writing to String cannot fail");
        }
    }

    let target_rows: Vec<_> = rows
        .iter()
        .filter(|row| row.phase == Phase::Target)
        .collect();
    if !target_rows.is_empty() {
        output.push_str("\n## Target-oriented results\n\n");
        output.push_str(
            "This phase stops scheduling new retries after reaching the published GTOP\n\
target. Already-running starts are allowed to finish, so wall time is the\n\
user-visible call duration through worker drain for successes and time to\n\
exhaustion for failures. Evaluation counts remain resource\n\
accounting, not the primary outcome.\n\n\
| Problem | Arm | Workers | Runs | Success | Mean best | Sdev best | Mean wall | Sdev wall | Mean evaluations | Sdev evaluations | Mean active cores | Sdev active cores |\n\
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        let mut groups: BTreeMap<(&str, Arm, usize), Vec<&ResultRow>> = BTreeMap::new();
        for row in target_rows {
            groups
                .entry((row.problem.as_str(), row.arm, row.workers))
                .or_default()
                .push(row);
        }
        for ((problem, arm, workers), group) in groups {
            let successes = group.iter().filter(|row| row.success).count();
            let (best, best_sdev) = mean_sdev(group.iter().map(|row| row.best));
            let (wall, wall_sdev) = mean_sdev(group.iter().map(|row| row.wall_seconds));
            let (evaluations, evaluations_sdev) =
                mean_sdev(group.iter().map(|row| row.evaluations_actual as f64));
            let (active, active_sdev) = mean_sdev(group.iter().map(|row| row.average_cores));
            writeln!(
                output,
                "| {problem} | {} | {workers} | {} | {:.0}% | {best:.6} | {best_sdev:.6} | {wall:.4}s | {wall_sdev:.4}s | {evaluations:.0} | {evaluations_sdev:.0} | {active:.2} | {active_sdev:.2} |",
                arm.label(),
                group.len(),
                100.0 * successes as f64 / group.len() as f64,
            )
            .expect("writing to String cannot fail");
        }
    }

    output.push_str("\n## Interpretation boundary\n\n");
    if !equal_wall_rows.is_empty() {
        output.push_str(
            "The equal-wall experiment intentionally spends more aggregate CPU in order to\n\
reduce user waiting time and improve the returned solution distribution. It is\n\
not an equal-CPU or algorithm-efficiency claim. ",
        );
    }
    output.push_str(
        "The fixed-work comparison only isolates `fcmaes_core::retry` scheduling.\n\
The `fcmaes-core` CMA-ES row additionally\n\
changes the optimizer implementation and is not a pure core-count comparison.\n\
The coordinated DE→CMA\n\
results in the parent GTOP report use adaptive budgets and crossover, so they are\n\
a system-level reference rather than another equal-work arm.\n",
    );
    output
}

fn mean_sdev(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let values: Vec<_> = values.collect();
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance.sqrt())
}

pub fn validate_resume(config: &Config, rows: &[ResultRow]) -> Result<(), String> {
    let preset = format!("{:?}", config.preset).to_ascii_lowercase();
    let cases = resolve_cases(&config.problem_keys)?;
    let case_indices: HashMap<_, _> = cases
        .iter()
        .enumerate()
        .map(|(index, case)| (case.key, (index, case)))
        .collect();
    let matrix: HashSet<_> = campaign_rows(config).into_iter().collect();
    let mut keys = HashSet::new();
    for row in rows {
        let Some(&(case_index, case)) = case_indices.get(row.problem.as_str()) else {
            return Err(format!(
                "existing row '{}' uses a problem outside the requested protocol",
                row.key()
            ));
        };
        let expected_population = config.population.unwrap_or_else(|| {
            4 + (3.0 * (case.problem.bounds.dim() as f64).ln()).floor() as usize
        });
        let expected_retries = match (row.phase, row.arm) {
            (Phase::EqualWall, Arm::ExternalSingle) => 1,
            (Phase::EqualWall, Arm::ExternalRetry) => row.workers,
            (_, Arm::ExternalSingle) => 1,
            _ => config.retries,
        };
        let expected_seed = config
            .seed
            .wrapping_add((case_index as u64).wrapping_mul(1_000_003))
            .wrapping_add(row.run.saturating_sub(1) as u64);
        if row.preset != preset
            || !config.phases.contains(&row.phase)
            || !config.arms.contains(&row.arm)
            || !matrix.contains(&(row.phase, row.arm, row.workers))
            || row.run == 0
            || row.run > config.runs
            || row.seed != expected_seed
            || row.retries_requested != expected_retries
            || row.evaluations_per_retry != config.evaluations
            || row.wall_time_ms
                != if row.phase == Phase::EqualWall {
                    config.wall_time_ms
                } else {
                    0
                }
            || row.population != expected_population
            || row.sigma.to_bits() != config.sigma.to_bits()
            || row.stop_value.to_bits() != case.stop_value.to_bits()
        {
            return Err(format!(
                "existing row '{}' conflicts with the requested resume protocol",
                row.key()
            ));
        }
        if !keys.insert(row.key()) {
            return Err(format!("duplicate result row '{}'", row.key()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_defaults_end_at_physical_cores() {
        let config = Config::from_args(["--preset", "publication"])
            .unwrap()
            .unwrap();
        assert_eq!(config.max_workers(), num_cpus::get_physical().max(1));
        assert_eq!(config.runs, 100);
        assert_eq!(config.wall_time_ms, 4_000);
        assert_eq!(config.phases, vec![Phase::EqualWall]);
    }

    #[test]
    fn parser_rejects_invalid_protocol_values() {
        assert!(Config::from_args(["--workers", "0"]).is_err());
        assert!(Config::from_args(["--sigma", "0"]).is_err());
        assert!(Config::from_args(["--population", "1"]).is_err());
        assert!(Config::from_args(["--arms", "unknown"]).is_err());
        assert!(Config::from_args(["--help"]).unwrap().is_none());
    }

    #[test]
    fn external_sequential_and_parallel_do_identical_fixed_work() {
        let config = Config::from_args([
            "--preset",
            "smoke",
            "--phases",
            "fixed-work",
            "--runs",
            "1",
            "--retries",
            "2",
            "--evaluations",
            "40",
            "--workers",
            "1,2",
        ])
        .unwrap()
        .unwrap();
        let case = resolve_cases(&config.problem_keys).unwrap().remove(0);
        let serial = run_case(
            &config,
            &case,
            Phase::FixedWork,
            Arm::ExternalSequential,
            1,
            1,
            7,
        );
        let parallel = run_case(
            &config,
            &case,
            Phase::FixedWork,
            Arm::ExternalRetry,
            2,
            1,
            7,
        );
        assert_eq!(serial.retries_completed, parallel.retries_completed);
        assert_eq!(serial.evaluations_actual, parallel.evaluations_actual);
        assert_eq!(serial.best.to_bits(), parallel.best.to_bits());
    }

    #[test]
    fn campaign_matrix_does_not_duplicate_one_worker_retry() {
        let config = Config::from_args(["--workers", "1,2,4"]).unwrap().unwrap();
        let matrix = campaign_rows(&config);
        assert!(matrix.contains(&(Phase::FixedWork, Arm::ExternalSequential, 1)));
        assert!(!matrix.contains(&(Phase::FixedWork, Arm::ExternalRetry, 1)));
        assert!(matrix.contains(&(Phase::FixedWork, Arm::ExternalRetry, 2)));
        assert!(matrix.contains(&(Phase::FixedWork, Arm::ExternalRetry, 4)));
    }

    #[test]
    fn equal_wall_matrix_is_one_start_against_parallel_retry() {
        let config = Config::from_args(["--preset", "publication", "--workers", "1,4"])
            .unwrap()
            .unwrap();
        assert_eq!(
            campaign_rows(&config),
            vec![
                (Phase::EqualWall, Arm::ExternalSingle, 1),
                (Phase::EqualWall, Arm::ExternalRetry, 4),
            ]
        );
    }

    #[test]
    fn report_statistics_use_population_standard_deviation() {
        let (mean, sdev) = mean_sdev([1.0, 3.0].into_iter());
        assert_eq!((mean, sdev), (2.0, 1.0));
        let (mean, sdev) = mean_sdev([7.0].into_iter());
        assert_eq!((mean, sdev), (7.0, 0.0));
    }
}
