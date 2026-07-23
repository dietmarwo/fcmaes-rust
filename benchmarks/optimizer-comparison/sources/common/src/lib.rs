//! Dependency-free benchmark contract shared by every optimizer adapter.
//!
//! `gtop.rs` is copied verbatim from `examples/src/gtop.rs` by the external
//! workspace generator. Keeping the objective implementation outside the
//! adapter crates prevents optimizer-specific problem variants.

#[allow(dead_code)]
mod gtop;

use std::collections::HashSet;
use std::f64::consts::PI;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub type Objective = fn(&[f64]) -> f64;

#[derive(Clone)]
pub struct Case {
    pub key: &'static str,
    pub name: &'static str,
    pub objective: Objective,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub absolute_best: f64,
    pub stop_fitness: f64,
}

impl Case {
    pub fn dimension(&self) -> usize {
        self.lower.len()
    }
}

fn case(
    key: &'static str,
    name: &'static str,
    objective: Objective,
    lower: &[f64],
    upper: &[f64],
    absolute_best: f64,
    stop_fitness: f64,
) -> Case {
    assert_eq!(lower.len(), upper.len());
    Case {
        key,
        name,
        objective,
        lower: lower.to_vec(),
        upper: upper.to_vec(),
        absolute_best,
        stop_fitness,
    }
}

fn shifted_gtoc1(x: &[f64]) -> f64 {
    gtop::gtoc1(x) - 2_000_000.0
}

fn tandem_5(x: &[f64]) -> f64 {
    gtop::tandem(x, &[3, 2, 3, 3, 6])
}

/// The seven cases and targets from `benchmarks/benchmark_gtop.md`.
pub fn cases() -> Vec<Case> {
    vec![
        case(
            "cassini1",
            "Cassini1",
            gtop::cassini1,
            &[-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
            &[0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
            4.9307,
            4.95535,
        ),
        case(
            "cassini2",
            "Cassini2",
            gtop::cassini2,
            &[
                -1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01,
                0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.15, 1.7, -PI, -PI, -PI,
                -PI,
            ],
            &[
                0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9,
                0.9, 0.9, 0.9, 0.9, 6.0, 6.0, 6.5, 291.0, PI, PI, PI, PI,
            ],
            8.383,
            8.42491,
        ),
        case(
            "gtoc1",
            "Gtoc1",
            shifted_gtoc1,
            &[3000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0],
            &[
                10000.0, 2000.0, 2000.0, 2000.0, 2000.0, 9000.0, 9000.0,
                9000.0,
            ],
            -1_581_950.0,
            -1_574_080.0,
        ),
        case(
            "messenger",
            "Messenger",
            gtop::messenger,
            &[
                1000.0, 1.0, 0.0, 0.0, 200.0, 30.0, 30.0, 30.0, 0.01, 0.01,
                0.01, 0.01, 1.1, 1.1, 1.1, -PI, -PI, -PI,
            ],
            &[
                4000.0, 5.0, 1.0, 1.0, 400.0, 400.0, 400.0, 400.0, 0.99, 0.99,
                0.99, 0.99, 6.0, 6.0, 6.0, PI, PI, PI,
            ],
            8.6299,
            8.673,
        ),
        case(
            "rosetta",
            "Rosetta",
            gtop::rosetta,
            &[
                1460.0, 3.0, 0.0, 0.0, 300.0, 150.0, 150.0, 300.0, 700.0, 0.01,
                0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.05, 1.05, -PI, -PI, -PI,
                -PI,
            ],
            &[
                1825.0, 5.0, 1.0, 1.0, 500.0, 800.0, 800.0, 800.0, 1850.0, 0.9,
                0.9, 0.9, 0.9, 0.9, 9.0, 9.0, 9.0, 9.0, PI, PI, PI, PI,
            ],
            1.3433,
            1.35,
        ),
        case(
            "tandem",
            "Tandem",
            tandem_5,
            &[
                5475.0, 2.5, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 0.01, 0.01,
                0.01, 0.01, 1.05, 1.05, 1.05, -PI, -PI, -PI,
            ],
            &[
                9132.0, 4.9, 1.0, 1.0, 2500.0, 2500.0, 2500.0, 2500.0, 0.99,
                0.99, 0.99, 0.99, 10.0, 10.0, 10.0, PI, PI, PI,
            ],
            -1500.46,
            -1493.0,
        ),
        case(
            "sagas",
            "Sagas",
            gtop::sagas,
            &[
                7000.0, 0.0, 0.0, 0.0, 50.0, 300.0, 0.01, 0.01, 1.05, 8.0,
                -PI, -PI,
            ],
            &[
                9100.0, 7.0, 1.0, 1.0, 2000.0, 2000.0, 0.9, 0.9, 7.0, 500.0,
                PI, PI,
            ],
            18.188,
            18.279,
        ),
    ]
}

/// Stable seed assignment matching the existing Rust GTOP benchmark.
pub fn run_seed(base_seed: u64, case_index: usize, run_index: usize) -> u64 {
    base_seed
        .wrapping_add((case_index as u64).wrapping_mul(1_000_003))
        .wrapping_add(run_index as u64)
}

#[derive(Clone, Debug)]
pub struct Config {
    pub runs: usize,
    pub workers: usize,
    pub evaluations: u64,
    pub retries: usize,
    pub evaluations_per_retry: u64,
    pub seed: u64,
    pub problem: Option<String>,
    pub output: PathBuf,
    pub resume: bool,
}

impl Config {
    pub const USAGE: &str = "options:\n\
      --runs N                    experiments per problem (default 100)\n\
      --workers N                 maximum worker threads (default 24)\n\
      --evaluations N             total evaluations per experiment (default 240000)\n\
      --retries N                 retry topology count (default 24)\n\
      --evaluations-per-retry N   evaluations per retry (default 10000)\n\
      --seed N                    root seed (default 1)\n\
      --problem NAME              run one problem\n\
      --output PATH               raw TSV destination (required)\n\
      --resume                    retain completed rows and continue\n\
      --help                      show this help";

    pub fn from_env() -> Result<Option<Self>, String> {
        Self::from_args(std::env::args().skip(1))
    }

    pub fn from_args<I, S>(args: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut config = Self {
            runs: 100,
            workers: 24,
            evaluations: 240_000,
            retries: 24,
            evaluations_per_retry: 10_000,
            seed: 1,
            problem: None,
            output: PathBuf::new(),
            resume: false,
        };
        let mut args = args.into_iter();
        while let Some(argument) = args.next() {
            let argument = argument.as_ref();
            if argument == "--help" || argument == "-h" {
                return Ok(None);
            }
            if argument == "--resume" {
                config.resume = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))?;
            let value = value.as_ref();
            match argument {
                "--runs" => config.runs = parse_positive(value, "--runs")?,
                "--workers" => config.workers = parse_positive(value, "--workers")?,
                "--evaluations" => {
                    config.evaluations = parse_positive(value, "--evaluations")?
                }
                "--retries" => config.retries = parse_positive(value, "--retries")?,
                "--evaluations-per-retry" => {
                    config.evaluations_per_retry =
                        parse_positive(value, "--evaluations-per-retry")?
                }
                "--seed" => {
                    config.seed = value
                        .parse()
                        .map_err(|_| "--seed requires an integer".to_owned())?
                }
                "--problem" => config.problem = Some(normalize_name(value)),
                "--output" => config.output = value.into(),
                _ => return Err(format!("unknown option '{argument}'")),
            }
        }
        if config.output.as_os_str().is_empty() {
            return Err("--output is required".to_owned());
        }
        let retry_budget = (config.retries as u64)
            .checked_mul(config.evaluations_per_retry)
            .ok_or_else(|| "retry evaluation budget overflow".to_owned())?;
        if retry_budget != config.evaluations {
            return Err(format!(
                "--evaluations ({}) must equal --retries * --evaluations-per-retry ({retry_budget})",
                config.evaluations
            ));
        }
        if let Some(problem) = &config.problem
            && !cases().iter().any(|case| case.key == problem)
        {
            return Err(format!("unknown problem '{problem}'"));
        }
        Ok(Some(config))
    }
}

fn parse_positive<T>(value: &str, option: &str) -> Result<T, String>
where
    T: std::str::FromStr + PartialEq + Default,
{
    let parsed = value
        .parse()
        .map_err(|_| format!("{option} requires a positive integer"))?;
    if parsed == T::default() {
        return Err(format!("{option} must be positive"));
    }
    Ok(parsed)
}

fn normalize_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct Adapter {
    pub library: &'static str,
    pub version: &'static str,
    pub algorithm: &'static str,
    pub parallel_mode: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub value: f64,
    pub evaluations: u64,
    pub workers_used: usize,
    pub population_or_batch: usize,
    pub optimizer_runs: usize,
}

const HEADER: &str = "library\tversion\talgorithm\tparallel_mode\tproblem\trun\tseed\tworkers\tpopulation_or_batch\toptimizer_runs\tconfigured_evaluations\tactual_evaluations\tabsolute_best\tstop_fitness\tsuccess\tvalue\twall_seconds";

pub fn run_benchmark(
    adapter: Adapter,
    config: &Config,
    mut solve: impl FnMut(&Case, u64, &Config) -> Result<Outcome, String>,
) -> Result<(), String> {
    let completed = if config.resume {
        completed_rows(&config.output)?
    } else {
        HashSet::new()
    };
    let exists = config.output.exists() && config.resume;
    let file = if exists {
        OpenOptions::new()
            .append(true)
            .open(&config.output)
            .map_err(|error| format!("opening {}: {error}", config.output.display()))?
    } else {
        File::create(&config.output)
            .map_err(|error| format!("creating {}: {error}", config.output.display()))?
    };
    let mut output = BufWriter::new(file);
    if !exists {
        writeln!(output, "{HEADER}").map_err(|error| error.to_string())?;
    }

    for (case_index, case) in cases().iter().enumerate() {
        if config.problem.as_deref().is_some_and(|key| key != case.key) {
            continue;
        }
        eprintln!(
            "start library={} algorithm={} mode={} problem={} runs={} workers={} budget={}",
            adapter.library,
            adapter.algorithm,
            adapter.parallel_mode,
            case.name,
            config.runs,
            config.workers,
            config.evaluations
        );
        for run_index in 0..config.runs {
            if completed.contains(&(case.key.to_owned(), run_index)) {
                continue;
            }
            let seed = run_seed(config.seed, case_index, run_index);
            let started = Instant::now();
            let result = solve(case, seed, config)?;
            let seconds = started.elapsed().as_secs_f64();
            if result.evaluations > config.evaluations {
                return Err(format!(
                    "{} run {} exceeded its evaluation budget: {} > {}",
                    case.name, run_index, result.evaluations, config.evaluations
                ));
            }
            if !result.value.is_finite() {
                return Err(format!(
                    "{} run {} returned a non-finite value",
                    case.name, run_index
                ));
            }
            writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.12}\t{:.12}\t{}\t{:.15}\t{:.9}",
                adapter.library,
                adapter.version,
                adapter.algorithm,
                adapter.parallel_mode,
                case.name,
                run_index,
                seed,
                result.workers_used,
                result.population_or_batch,
                result.optimizer_runs,
                config.evaluations,
                result.evaluations,
                case.absolute_best,
                case.stop_fitness,
                result.value < case.stop_fitness,
                result.value,
                seconds,
            )
            .map_err(|error| error.to_string())?;
            output.flush().map_err(|error| error.to_string())?;
            if (run_index + 1) % 10 == 0 || run_index + 1 == config.runs {
                eprintln!(
                    "progress library={} problem={} completed={}/{} best={:.9} wall_ms={:.3}",
                    adapter.library,
                    case.name,
                    run_index + 1,
                    config.runs,
                    result.value,
                    seconds * 1000.0
                );
            }
        }
    }
    Ok(())
}

fn completed_rows(path: &Path) -> Result<HashSet<(String, usize)>, String> {
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let input = BufReader::new(
        File::open(path).map_err(|error| format!("opening {}: {error}", path.display()))?,
    );
    let mut completed = HashSet::new();
    for (line_index, line) in input.lines().enumerate() {
        let line = line.map_err(|error| error.to_string())?;
        if line_index == 0 {
            if line != HEADER {
                return Err(format!("{} has an incompatible header", path.display()));
            }
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 17 {
            return Err(format!(
                "{} line {} has {} columns, expected 17",
                path.display(),
                line_index + 1,
                fields.len()
            ));
        }
        let run = fields[5]
            .parse()
            .map_err(|_| format!("invalid run at {}:{}", path.display(), line_index + 1))?;
        completed.insert((normalize_name(fields[4]), run));
    }
    Ok(completed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_matches_published_table() {
        let cases = cases();
        assert_eq!(
            cases.iter().map(|case| case.name).collect::<Vec<_>>(),
            [
                "Cassini1",
                "Cassini2",
                "Gtoc1",
                "Messenger",
                "Rosetta",
                "Tandem",
                "Sagas"
            ]
        );
        for case in cases {
            assert_eq!(case.lower.len(), case.upper.len());
            let midpoint: Vec<f64> = case
                .lower
                .iter()
                .zip(&case.upper)
                .map(|(&lower, &upper)| (lower + upper) * 0.5)
                .collect();
            assert!((case.objective)(&midpoint).is_finite(), "{}", case.name);
        }
    }

    #[test]
    fn seed_schedule_is_stable() {
        assert_eq!(run_seed(1, 0, 0), 1);
        assert_eq!(run_seed(1, 2, 3), 2_000_010);
    }

    #[test]
    fn validates_common_budget() {
        assert!(
            Config::from_args([
                "--output",
                "x.tsv",
                "--evaluations",
                "10",
                "--retries",
                "2",
                "--evaluations-per-retry",
                "4"
            ])
            .is_err()
        );
    }
}
