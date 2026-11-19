//! Budget measurement and the resumable campaign runner.
//!
//! # Files
//!
//! `budgets.tsv` (written by `measure`), one line per problem after a header:
//! `problem, domain, dim, mean_eval_us, budget`.
//!
//! `raw.tsv` (written by `run`), one line per repetition after a header:
//! `solver, problem, domain, dim, protocol, rep, seed, budget, budget_scale,
//! evaluations, best_value, over_budget, out_of_bounds, wall_ms`.
//! `seed` is the seed of restart 0 (protocol R uses restart seeds
//! `run_seed(problem, 'R', rep, k)` for `k = 0..8`); `budget` is the total
//! number of evaluations allowed for the row (protocol R: `8 ⌊B'/8⌋`, with
//! `B' = ⌊B · budget_scale⌋`); `evaluations`, the violation counters and
//! `wall_ms` are summed over the restarts; `best_value` is the lowest value
//! any guard recorded (`inf` if no evaluation was made). Floats are written
//! in Rust's shortest round-trip exponent form.
//!
//! Round 2 (`ROUND2.md`) adds protocol T, written to its own file
//! (`results/round2/raw.tsv`) with the round-1 columns followed by `runs,
//! root_seed, total, run_cap, min_run` ([`RAW_HEADER_T`]). A T row has
//! `budget = total`, `budget_scale = 1`, `seed` = seed of run 0, and
//! `evaluations`, violations and `wall_ms` summed over its `runs` runs. A file
//! holds either S/R rows (round-1 format, unchanged) or T rows, never both.
//!
//! Round 3 (`ROUND3.md`) runs protocol T with two totals (2,000 and 20,000,
//! run cap equal to the total) into one file ([`run_campaign_t_totals`]). The
//! two totals are separate cells: the resume key is (solver, problem,
//! protocol, repetition, total) ([`RowKey`]). The candidate driver
//! `run_catcmawm.py` appends to the same file in the same format; the runner
//! and the driver never write concurrently, which [`RunLock`] enforces with a
//! lock file next to the output.

use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use fcmaes_core::Rng;
use solver_api::{Guard, Solver};

use crate::problems::Problem;
use crate::seeds::{measure_seed, run_seed, run_seed_from};
use crate::solvers::uniform_point;

/// Uniform random points timed per problem.
pub const MEASURE_POINTS: usize = 2_000;

/// Independent runs of protocol R.
pub const RESTARTS_R: u32 = 8;

/// Header of `budgets.tsv`.
pub const BUDGET_HEADER: &str = "problem\tdomain\tdim\tmean_eval_us\tbudget";

/// Header of `raw.tsv`.
pub const RAW_HEADER: &str = "solver\tproblem\tdomain\tdim\tprotocol\trep\tseed\tbudget\tbudget_scale\tevaluations\tbest_value\tover_budget\tout_of_bounds\twall_ms";

/// Header of a protocol-T raw file: [`RAW_HEADER`] plus the protocol
/// settings and the number of runs.
pub const RAW_HEADER_T: &str = "solver\tproblem\tdomain\tdim\tprotocol\trep\tseed\tbudget\tbudget_scale\tevaluations\tbest_value\tover_budget\tout_of_bounds\twall_ms\truns\troot_seed\ttotal\trun_cap\tmin_run";

/// Total evaluations of one protocol-T repetition (`ROUND2.md`, section 4).
pub const T_TOTAL: u64 = 400_000;

/// Largest budget of one protocol-T run.
pub const T_RUN_CAP: u64 = 100_000;

/// Protocol T starts no further run when fewer evaluations remain.
pub const T_MIN_RUN: u64 = 1_000;

/// Budget class table of protocol section 3.3.
pub fn budget_for(mean_eval_us: f64) -> u64 {
    if mean_eval_us <= 10.0 {
        200_000
    } else if mean_eval_us <= 50.0 {
        100_000
    } else if mean_eval_us <= 250.0 {
        40_000
    } else {
        16_000
    }
}

/// Mean single-thread evaluation time in microseconds over
/// [`MEASURE_POINTS`] seeded uniform random points.
pub fn measure_problem(problem: &Problem) -> f64 {
    let mut rng = Rng::new(measure_seed(problem.name));
    let points: Vec<Vec<f64>> = (0..MEASURE_POINTS)
        .map(|_| uniform_point(&problem.lower, &problem.upper, &mut rng))
        .collect();
    let mut sink = 0.0;
    let started = Instant::now();
    for x in &points {
        sink += std::hint::black_box((problem.objective)(std::hint::black_box(x)));
    }
    let elapsed = started.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    elapsed * 1e6 / MEASURE_POINTS as f64
}

/// Points re-evaluated by [`is_deterministic`].
pub const DETERMINISM_POINTS: usize = 200;

/// Whether `problem` returns bit-identical values when the first
/// [`DETERMINISM_POINTS`] points of its measurement sample are evaluated
/// twice (round-2 entry rule).
pub fn is_deterministic(problem: &Problem) -> bool {
    let mut rng = Rng::new(measure_seed(problem.name));
    (0..DETERMINISM_POINTS).all(|_| {
        let x = uniform_point(&problem.lower, &problem.upper, &mut rng);
        (problem.objective)(&x).to_bits() == (problem.objective)(&x).to_bits()
    })
}

/// One line of `budgets.tsv`.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetRow {
    /// Problem name.
    pub problem: String,
    /// Application domain.
    pub domain: String,
    /// Dimension.
    pub dim: usize,
    /// Mean evaluation time in microseconds.
    pub mean_eval_us: f64,
    /// Budget B.
    pub budget: u64,
}

/// Write `budgets.tsv`.
///
/// # Errors
/// Returns an I/O error message.
pub fn write_budgets(path: &Path, rows: &[BudgetRow]) -> Result<(), String> {
    let mut text = format!("{BUDGET_HEADER}\n");
    for row in rows {
        text.push_str(&format!(
            "{}\t{}\t{}\t{:.3}\t{}\n",
            row.problem, row.domain, row.dim, row.mean_eval_us, row.budget
        ));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Read `budgets.tsv`.
///
/// # Errors
/// Returns a message for I/O or format errors.
pub fn read_budgets(path: &Path) -> Result<Vec<BudgetRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = text.lines();
    if lines.next() != Some(BUDGET_HEADER) {
        return Err(format!("{}: unexpected header", path.display()));
    }
    lines
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != 5 {
                return Err(format!("{}: malformed line '{line}'", path.display()));
            }
            let parse_err = |name: &str| format!("{}: bad {name} in '{line}'", path.display());
            Ok(BudgetRow {
                problem: fields[0].to_owned(),
                domain: fields[1].to_owned(),
                dim: fields[2].parse().map_err(|_| parse_err("dim"))?,
                mean_eval_us: fields[3].parse().map_err(|_| parse_err("mean_eval_us"))?,
                budget: fields[4].parse().map_err(|_| parse_err("budget"))?,
            })
        })
        .collect()
}

/// Benchmark protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Protocol {
    /// One run with budget B.
    S,
    /// Eight independent runs with budget ⌊B/8⌋ each.
    R,
    /// Round 2: restarts with fresh seeds until the total is spent
    /// ([`TSettings`]).
    T,
}

impl Protocol {
    /// Protocol letter.
    pub fn letter(self) -> char {
        match self {
            Self::S => 'S',
            Self::R => 'R',
            Self::T => 'T',
        }
    }

    /// Parse a protocol letter.
    ///
    /// # Errors
    /// Returns a message for an unknown letter.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "S" => Ok(Self::S),
            "R" => Ok(Self::R),
            "T" => Ok(Self::T),
            _ => Err(format!("unknown protocol '{text}' (expected S, R or T)")),
        }
    }
}

/// Result of one repetition (one raw row).
#[derive(Clone, Debug, PartialEq)]
pub struct RunResult {
    /// Seed of restart 0.
    pub seed: u64,
    /// Total evaluations allowed.
    pub budget: u64,
    /// Evaluations made (guard count).
    pub evaluations: u64,
    /// Best value recorded by the guards.
    pub best_value: f64,
    /// Calls refused for the budget.
    pub over_budget: u64,
    /// Calls refused for the box.
    pub out_of_bounds: u64,
    /// Wall time in milliseconds.
    pub wall_ms: f64,
}

/// Run budget of one restart for scaled budget `scaled` under `protocol`.
/// (Protocol T has no fixed restart budget; this returns its total, and
/// [`run_repetition_t`] assigns the run budgets.)
pub fn restart_budget(protocol: Protocol, scaled: u64) -> u64 {
    match protocol {
        Protocol::S | Protocol::T => scaled,
        Protocol::R => scaled / u64::from(RESTARTS_R),
    }
}

/// Settings of protocol T, recorded in every T row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TSettings {
    /// Root seed of the seed derivation (round 2: 2).
    pub root_seed: u64,
    /// Total evaluations of one repetition ([`T_TOTAL`]).
    pub total: u64,
    /// Largest budget of one run ([`T_RUN_CAP`]).
    pub run_cap: u64,
    /// No run starts when fewer evaluations remain ([`T_MIN_RUN`]).
    pub min_run: u64,
}

impl TSettings {
    /// The declared defaults with root seed `root_seed`.
    pub fn with_root_seed(root_seed: u64) -> Self {
        Self {
            root_seed,
            total: T_TOTAL,
            run_cap: T_RUN_CAP,
            min_run: T_MIN_RUN,
        }
    }

    /// Whether these are the declared round-2 campaign settings.
    pub fn is_declared(&self) -> bool {
        self.total == T_TOTAL && self.run_cap == T_RUN_CAP && self.min_run == T_MIN_RUN
    }

    fn validate(&self) -> Result<(), String> {
        if self.total == 0 || self.run_cap == 0 || self.min_run == 0 {
            return Err("protocol T needs a positive total, run cap and minimum run".into());
        }
        Ok(())
    }
}

/// Seed of run `restart` of repetition `repetition` of a protocol-T row:
/// `run_seed_from(root_seed, problem, 'T', repetition, restart)`. The total,
/// run cap and minimum run do not enter the derivation, so in round 3 the
/// cells at 2,000 and at 20,000 evaluations use the same run seeds. The
/// candidate driver (`run_catcmawm.py`) replicates this function; the
/// `seeds` subcommand prints it for a grid.
pub fn t_run_seed(settings: &TSettings, problem: &str, repetition: u32, restart: u32) -> u64 {
    run_seed_from(
        settings.root_seed,
        problem,
        Protocol::T.letter(),
        repetition,
        restart,
    )
}

/// Result of one protocol-T repetition.
#[derive(Clone, Debug, PartialEq)]
pub struct TResult {
    /// Summed row fields; `budget` is the total.
    pub result: RunResult,
    /// Runs made (including a final run without evaluations, if any).
    pub runs: u32,
    /// Restarting stopped because a run made no evaluation.
    pub stopped_empty: bool,
}

/// Execute one protocol-T repetition (`ROUND2.md`, section 4).
///
/// Run `k = 0, 1, 2, ...` uses seed `run_seed_from(root, problem, 'T', rep,
/// k)` and a fresh [`Guard`] with budget `min(run_cap, remaining)`, where
/// `remaining` is the total minus the evaluations the guards counted in the
/// earlier runs (not the solvers' claims). Restarting ends when fewer than
/// `min_run` evaluations remain, or when a run made no evaluation at all
/// (a solver that returns at once would otherwise restart forever). The best
/// value over all runs counts; evaluations, violations and wall time are
/// summed.
pub fn run_repetition_t(
    solver: &dyn Solver,
    problem: &Problem,
    repetition: u32,
    settings: &TSettings,
) -> TResult {
    let objective = &problem.objective;
    let seed_of = |restart: u32| t_run_seed(settings, problem.name, repetition, restart);
    let mut result = RunResult {
        seed: seed_of(0),
        budget: settings.total,
        evaluations: 0,
        best_value: f64::INFINITY,
        over_budget: 0,
        out_of_bounds: 0,
        wall_ms: 0.0,
    };
    let mut remaining = settings.total;
    let mut runs = 0u32;
    let mut stopped_empty = false;
    while remaining >= settings.min_run.max(1) {
        let allowance = settings.run_cap.min(remaining);
        let seed = seed_of(runs);
        let guard = Guard::new(
            objective.as_ref(),
            &problem.lower,
            &problem.upper,
            allowance,
        );
        let started = Instant::now();
        let guarded = |x: &[f64]| guard.eval(x);
        let _claimed = solver.minimize(&guarded, &problem.lower, &problem.upper, allowance, seed);
        result.wall_ms += started.elapsed().as_secs_f64() * 1e3;
        let calls = guard.calls();
        result.evaluations += calls;
        result.over_budget += guard.over_budget();
        result.out_of_bounds += guard.out_of_bounds();
        result.best_value = result.best_value.min(guard.best_y());
        runs += 1;
        // The guard never counts more than its allowance.
        remaining -= calls;
        if calls == 0 {
            stopped_empty = true;
            break;
        }
    }
    TResult {
        result,
        runs,
        stopped_empty,
    }
}

/// Scaled budget `⌊B · scale⌋`, at least one evaluation.
pub fn scaled_budget(budget: u64, scale: f64) -> u64 {
    ((budget as f64 * scale).floor() as u64).max(1)
}

/// Execute one repetition of protocol S or R: every run goes through a fresh
/// [`Guard`].
///
/// # Panics
/// Panics for protocol T, which runs through [`run_repetition_t`].
pub fn run_repetition(
    solver: &dyn Solver,
    problem: &Problem,
    protocol: Protocol,
    repetition: u32,
    scaled: u64,
) -> RunResult {
    let restarts = match protocol {
        Protocol::S => 1,
        Protocol::R => RESTARTS_R,
        Protocol::T => panic!("protocol T runs through run_repetition_t"),
    };
    let per_run = restart_budget(protocol, scaled);
    let objective = &problem.objective;
    let mut result = RunResult {
        seed: run_seed(problem.name, protocol.letter(), repetition, 0),
        budget: per_run * u64::from(restarts),
        evaluations: 0,
        best_value: f64::INFINITY,
        over_budget: 0,
        out_of_bounds: 0,
        wall_ms: 0.0,
    };
    for restart in 0..restarts {
        let seed = run_seed(problem.name, protocol.letter(), repetition, restart);
        let guard = Guard::new(objective.as_ref(), &problem.lower, &problem.upper, per_run);
        let started = Instant::now();
        let guarded = |x: &[f64]| guard.eval(x);
        let _claimed = solver.minimize(&guarded, &problem.lower, &problem.upper, per_run, seed);
        result.wall_ms += started.elapsed().as_secs_f64() * 1e3;
        result.evaluations += guard.calls();
        result.over_budget += guard.over_budget();
        result.out_of_bounds += guard.out_of_bounds();
        result.best_value = result.best_value.min(guard.best_y());
    }
    result
}

/// Key of a raw row: solver, problem, protocol, repetition and, for a
/// protocol-T row, its total (0 for S and R rows). Within a round-1 or
/// round-2 file the last component is constant, so the key is the
/// (solver, problem, protocol, repetition) key of those rounds; in a round-3
/// file the two totals are separate cells.
pub type RowKey = (String, String, char, u32, u64);

/// One parsed raw row (the fields the runner needs on resumption).
#[derive(Clone, Debug)]
pub struct RawRow {
    /// Row key.
    pub key: RowKey,
    /// Budget scale of the row.
    pub budget_scale: f64,
    /// Protocol-T settings of the row (T files only).
    pub t_settings: Option<TSettings>,
}

/// Format one raw row (with trailing newline).
pub fn format_row(
    solver: &str,
    problem: &Problem,
    protocol: Protocol,
    repetition: u32,
    budget_scale: f64,
    result: &RunResult,
) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:e}\t{}\t{:e}\t{}\t{}\t{:.3}\n",
        solver,
        problem.name,
        problem.domain,
        problem.dim(),
        protocol.letter(),
        repetition,
        result.seed,
        result.budget,
        budget_scale,
        result.evaluations,
        result.best_value,
        result.over_budget,
        result.out_of_bounds,
        result.wall_ms
    )
}

/// Format one protocol-T raw row (with trailing newline): the round-1
/// columns (budget scale 1) followed by `runs, root_seed, total, run_cap,
/// min_run`.
pub fn format_row_t(
    solver: &str,
    problem: &Problem,
    repetition: u32,
    settings: &TSettings,
    t: &TResult,
) -> String {
    let mut line = format_row(solver, problem, Protocol::T, repetition, 1.0, &t.result);
    line.pop();
    line.push_str(&format!(
        "\t{}\t{}\t{}\t{}\t{}\n",
        t.runs, settings.root_seed, settings.total, settings.run_cap, settings.min_run
    ));
    line
}

/// Load the rows of an existing round-1 (S/R) raw file. A trailing line
/// without newline (an interrupted write) is cut from the file. Creates the
/// file with its header if it does not exist.
///
/// # Errors
/// Returns a message for I/O or format errors.
pub fn load_or_create_raw(path: &Path) -> Result<Vec<RawRow>, String> {
    load_or_create(path, RAW_HEADER)
}

/// Like [`load_or_create_raw`] for a protocol-T file ([`RAW_HEADER_T`]).
///
/// # Errors
/// Returns a message for I/O or format errors.
pub fn load_or_create_raw_t(path: &Path) -> Result<Vec<RawRow>, String> {
    load_or_create(path, RAW_HEADER_T)
}

fn load_or_create(path: &Path, expected_header: &str) -> Result<Vec<RawRow>, String> {
    let field_count = expected_header.split('\t').count();
    let describe = |e: std::io::Error| format!("{}: {e}", path.display());
    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(describe)?;
        }
        std::fs::write(path, format!("{expected_header}\n")).map_err(describe)?;
        return Ok(Vec::new());
    }
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(describe)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(describe)?;
    if bytes.is_empty() {
        file.write_all(format!("{expected_header}\n").as_bytes())
            .map_err(describe)?;
        return Ok(Vec::new());
    }
    if bytes.last() != Some(&b'\n') {
        let keep = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        eprintln!(
            "warning: {}: dropping an incomplete last line ({} bytes)",
            path.display(),
            bytes.len() - keep
        );
        file.set_len(keep as u64).map_err(describe)?;
        file.seek(SeekFrom::End(0)).map_err(describe)?;
        bytes.truncate(keep);
        if bytes.is_empty() {
            file.write_all(format!("{expected_header}\n").as_bytes())
                .map_err(describe)?;
            return Ok(Vec::new());
        }
    }
    let mut rows = Vec::new();
    let mut lines = BufReader::new(bytes.as_slice()).lines();
    let header = lines.next().transpose().map_err(describe)?;
    if header.as_deref() != Some(expected_header) {
        return Err(format!("{}: unexpected header", path.display()));
    }
    for line in lines {
        let line = line.map_err(describe)?;
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != field_count {
            return Err(format!("{}: malformed row '{line}'", path.display()));
        }
        let bad = |name: &str| format!("{}: bad {name} in '{line}'", path.display());
        let protocol = fields[4].chars().next().ok_or_else(|| bad("protocol"))?;
        let t_settings = if field_count > 14 {
            Some(TSettings {
                root_seed: fields[15].parse().map_err(|_| bad("root_seed"))?,
                total: fields[16].parse().map_err(|_| bad("total"))?,
                run_cap: fields[17].parse().map_err(|_| bad("run_cap"))?,
                min_run: fields[18].parse().map_err(|_| bad("min_run"))?,
            })
        } else {
            None
        };
        rows.push(RawRow {
            key: (
                fields[0].to_owned(),
                fields[1].to_owned(),
                protocol,
                fields[5].parse().map_err(|_| bad("rep"))?,
                t_settings.map_or(0, |t| t.total),
            ),
            budget_scale: fields[8].parse().map_err(|_| bad("budget_scale"))?,
            t_settings,
        });
    }
    Ok(rows)
}

/// Campaign settings of the `run` command.
pub struct CampaignConfig<'a> {
    /// Problems to run, each with its budget B.
    pub problems: Vec<(&'a Problem, u64)>,
    /// Solvers to run.
    pub solvers: Vec<&'a dyn Solver>,
    /// Protocols to run.
    pub protocols: Vec<Protocol>,
    /// Repetitions per cell.
    pub repetitions: u32,
    /// Worker threads.
    pub threads: usize,
    /// Budget scale (1.0 for the campaign).
    pub budget_scale: f64,
}

struct Task {
    solver: usize,
    problem: usize,
    protocol: Protocol,
    /// Index into the protocol-T settings (0 for S and R).
    settings: usize,
    repetition: u32,
    cell: usize,
}

/// Exclusive writer lock of a raw file: the file `<out>.lock`, created with
/// `create_new` and holding the process id. The Rust runner and the
/// candidate driver (`run_catcmawm.py`, same lock rule) take it before they
/// read the file for resumption and release it when they finish, so the two
/// never append to one file at the same time. A lock whose process no longer
/// exists (`/proc/<pid>` missing, after a crash) is stale and is replaced.
pub struct RunLock {
    path: std::path::PathBuf,
}

impl RunLock {
    /// Path of the lock file of `out`.
    pub fn path_for(out: &Path) -> std::path::PathBuf {
        let mut name = out.as_os_str().to_owned();
        name.push(".lock");
        std::path::PathBuf::from(name)
    }

    /// Take the lock of `out`.
    ///
    /// # Errors
    /// Returns a message if another live process holds it or on I/O errors.
    pub fn acquire(out: &Path) -> Result<Self, String> {
        let path = Self::path_for(out);
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        for _ in 0..2 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    file.write_all(format!("{}\n", std::process::id()).as_bytes())
                        .map_err(|e| format!("{}: {e}", path.display()))?;
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let holder = std::fs::read_to_string(&path).unwrap_or_default();
                    let pid: Option<u32> = holder.trim().parse().ok();
                    let alive = pid.is_some_and(|pid| Path::new(&format!("/proc/{pid}")).exists());
                    if alive || pid.is_none() {
                        return Err(format!(
                            "{} is locked by process '{}' (the Rust runner or run_catcmawm.py); \
                             they must not write one file concurrently. Remove the lock file \
                             only if no such process runs.",
                            out.display(),
                            holder.trim()
                        ));
                    }
                    eprintln!(
                        "warning: removing stale lock {} of process {}",
                        path.display(),
                        holder.trim()
                    );
                    std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                }
                Err(error) => return Err(format!("{}: {error}", path.display())),
            }
        }
        Err(format!("{}: could not take the lock", path.display()))
    }
}

impl Drop for RunLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Run the campaign (protocols S and R), appending rows to `out`. Returns
/// the number of rows written by this call.
///
/// # Errors
/// Returns a message for I/O errors, inconsistent budget scales, a budget
/// too small for protocol R or a request for protocol T.
pub fn run_campaign(config: &CampaignConfig<'_>, out: &Path) -> Result<usize, String> {
    run_campaign_with(config, None, out)
}

/// Run a protocol-T campaign (round 2), appending rows to `out` in the
/// [`RAW_HEADER_T`] format. The budgets of `config.problems` are ignored;
/// every repetition gets `settings.total`. `config.protocols` must be `[T]`
/// and `config.budget_scale` 1. Resumable like [`run_campaign`]; a file with
/// rows of other settings is refused.
///
/// # Errors
/// Returns a message for I/O errors, invalid settings or a file holding rows
/// with other settings.
pub fn run_campaign_t(
    config: &CampaignConfig<'_>,
    settings: &TSettings,
    out: &Path,
) -> Result<usize, String> {
    run_campaign_with(config, Some(std::slice::from_ref(settings)), out)
}

/// Run a protocol-T campaign with several settings into one file (round 3:
/// totals 2,000 and 20,000, each with run cap equal to the total). Each
/// settings entry is its own set of cells; the resume key includes the total
/// ([`RowKey`]). The totals must be distinct, and every row already in the
/// file must carry one of the given settings, so a file never mixes root
/// seeds or minimum runs. Tasks run problem by problem, then total by total.
///
/// # Errors
/// As [`run_campaign_t`], and for an empty list or repeated totals.
pub fn run_campaign_t_totals(
    config: &CampaignConfig<'_>,
    settings: &[TSettings],
    out: &Path,
) -> Result<usize, String> {
    if settings.is_empty() {
        return Err("no protocol-T settings given".into());
    }
    let mut totals: Vec<u64> = settings.iter().map(|s| s.total).collect();
    totals.sort_unstable();
    totals.dedup();
    if totals.len() != settings.len() {
        return Err("protocol-T totals must be distinct".into());
    }
    run_campaign_with(config, Some(settings), out)
}

fn run_campaign_with(
    config: &CampaignConfig<'_>,
    t_settings: Option<&[TSettings]>,
    out: &Path,
) -> Result<usize, String> {
    let _lock = RunLock::acquire(out)?;
    let existing = if let Some(all_settings) = t_settings {
        for settings in all_settings {
            settings.validate()?;
        }
        if config.protocols != [Protocol::T] {
            return Err("a protocol-T campaign runs protocol T only".into());
        }
        if config.budget_scale != 1.0 {
            return Err("protocol T has no budget scale; use --total for smoke runs".into());
        }
        let existing = load_or_create_raw_t(out)?;
        if let Some(row) = existing.iter().find(|row| {
            !row.t_settings
                .as_ref()
                .is_some_and(|s| all_settings.contains(s))
        }) {
            return Err(format!(
                "{} holds rows with protocol-T settings {:?} (current {all_settings:?}); \
                 use another output file",
                out.display(),
                row.t_settings
            ));
        }
        existing
    } else {
        if config.protocols.contains(&Protocol::T) {
            return Err("protocol T runs through run_campaign_t (round-2 file format)".into());
        }
        load_or_create_raw(out)?
    };
    if let Some(row) = existing
        .iter()
        .find(|row| row.budget_scale != config.budget_scale)
    {
        return Err(format!(
            "{} holds rows with budget_scale {} (current {}); use another output file",
            out.display(),
            row.budget_scale,
            config.budget_scale
        ));
    }
    let done: HashSet<RowKey> = existing.into_iter().map(|row| row.key).collect();

    for &(problem, budget) in &config.problems {
        if t_settings.is_some() {
            break;
        }
        for &protocol in &config.protocols {
            let scaled = scaled_budget(budget, config.budget_scale);
            if restart_budget(protocol, scaled) == 0 {
                return Err(format!(
                    "{}: scaled budget {scaled} is too small for protocol {}",
                    problem.name,
                    protocol.letter()
                ));
            }
        }
    }

    // Tasks ordered cell by cell so cells complete progressively.
    let mut tasks = Vec::new();
    let mut cell_names = Vec::new();
    let mut remaining_per_cell = Vec::new();
    // S/R: one pass with key total 0; T: one pass per settings entry.
    let passes: Vec<(usize, u64)> = match t_settings {
        Some(all) => all.iter().enumerate().map(|(i, s)| (i, s.total)).collect(),
        None => vec![(0, 0)],
    };
    let several_totals = passes.len() > 1;
    for (problem_index, &(problem, _)) in config.problems.iter().enumerate() {
        for &protocol in &config.protocols {
            for &(settings_index, key_total) in &passes {
                for (solver_index, solver) in config.solvers.iter().enumerate() {
                    let cell = cell_names.len();
                    cell_names.push(if several_totals {
                        format!(
                            "{} {} {} {key_total}",
                            solver.name(),
                            problem.name,
                            protocol.letter()
                        )
                    } else {
                        format!("{} {} {}", solver.name(), problem.name, protocol.letter())
                    });
                    let mut pending = 0;
                    for repetition in 0..config.repetitions {
                        let key = (
                            solver.name().to_owned(),
                            problem.name.to_owned(),
                            protocol.letter(),
                            repetition,
                            key_total,
                        );
                        if !done.contains(&key) {
                            tasks.push(Task {
                                solver: solver_index,
                                problem: problem_index,
                                protocol,
                                settings: settings_index,
                                repetition,
                                cell,
                            });
                            pending += 1;
                        }
                    }
                    remaining_per_cell.push(AtomicUsize::new(pending));
                }
            }
        }
    }
    let total = tasks.len();
    println!(
        "{total} runs to do ({} already present), {} threads",
        done.len(),
        config.threads
    );

    let file = OpenOptions::new()
        .append(true)
        .open(out)
        .map_err(|e| format!("{}: {e}", out.display()))?;
    let writer: Mutex<File> = Mutex::new(file);
    let next = AtomicUsize::new(0);
    let written = AtomicUsize::new(0);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let started = Instant::now();
    let cell_started: Vec<Mutex<Option<Instant>>> =
        cell_names.iter().map(|_| Mutex::new(None)).collect();

    std::thread::scope(|scope| {
        for _ in 0..config.threads.max(1) {
            scope.spawn(|| {
                loop {
                    if failure
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .is_some()
                    {
                        return;
                    }
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(task) = tasks.get(index) else {
                        return;
                    };
                    cell_started[task.cell]
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get_or_insert_with(Instant::now);
                    let (problem, budget) = config.problems[task.problem];
                    let solver = config.solvers[task.solver];
                    let line = if let Some(all_settings) = t_settings {
                        let settings = &all_settings[task.settings];
                        let t = run_repetition_t(solver, problem, task.repetition, settings);
                        if t.stopped_empty {
                            eprintln!(
                                "warning: {} on {} rep {}: run {} made no evaluation; \
                                 restarting stopped with {} evaluations left",
                                solver.name(),
                                problem.name,
                                task.repetition,
                                t.runs - 1,
                                settings.total - t.result.evaluations
                            );
                        }
                        format_row_t(solver.name(), problem, task.repetition, settings, &t)
                    } else {
                        let scaled = scaled_budget(budget, config.budget_scale);
                        let result =
                            run_repetition(solver, problem, task.protocol, task.repetition, scaled);
                        format_row(
                            solver.name(),
                            problem,
                            task.protocol,
                            task.repetition,
                            config.budget_scale,
                            &result,
                        )
                    };
                    {
                        let mut file = writer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if let Err(error) =
                            file.write_all(line.as_bytes()).and_then(|()| file.flush())
                        {
                            *failure
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                Some(format!("{}: {error}", out.display()));
                            return;
                        }
                    }
                    let finished = written.fetch_add(1, Ordering::Relaxed) + 1;
                    if remaining_per_cell[task.cell].fetch_sub(1, Ordering::AcqRel) == 1 {
                        let cell_elapsed = cell_started[task.cell]
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .map_or(0.0, |t| t.elapsed().as_secs_f64());
                        println!(
                            "done {} ({finished}/{total} runs, cell {:.1}s, elapsed {:.1}s)",
                            cell_names[task.cell],
                            cell_elapsed,
                            started.elapsed().as_secs_f64()
                        );
                    }
                }
            });
        }
    });
    if let Some(message) = failure
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        return Err(message);
    }
    Ok(written.into_inner())
}

/// Count rows per (solver, problem, protocol) in a round-1 raw file, for
/// summaries.
///
/// # Errors
/// Returns a message for I/O or format errors.
pub fn count_cells(path: &Path) -> Result<BTreeMap<(String, String, char), usize>, String> {
    let mut counts = BTreeMap::new();
    for row in load_or_create_raw(path)? {
        let (solver, problem, protocol, _, _) = row.key;
        *counts.entry((solver, problem, protocol)).or_insert(0) += 1;
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_classes_follow_the_protocol_table() {
        assert_eq!(budget_for(3.0), 200_000);
        assert_eq!(budget_for(10.0), 200_000);
        assert_eq!(budget_for(10.01), 100_000);
        assert_eq!(budget_for(50.0), 100_000);
        assert_eq!(budget_for(250.0), 40_000);
        assert_eq!(budget_for(250.1), 16_000);
    }

    #[test]
    fn restart_budgets() {
        assert_eq!(restart_budget(Protocol::S, 40_000), 40_000);
        assert_eq!(restart_budget(Protocol::R, 40_000), 5_000);
        assert_eq!(restart_budget(Protocol::R, 16_007), 2_000);
        assert_eq!(scaled_budget(200_000, 0.02), 4_000);
    }

    #[test]
    fn t_header_extends_the_round_one_header() {
        assert!(RAW_HEADER_T.starts_with(&format!("{RAW_HEADER}\t")));
        assert_eq!(RAW_HEADER_T.split('\t').count(), 19);
        let defaults = TSettings::with_root_seed(2);
        assert!(defaults.is_declared());
        assert_eq!(
            (defaults.total, defaults.run_cap, defaults.min_run),
            (400_000, 100_000, 1_000)
        );
        assert_eq!(Protocol::parse("T"), Ok(Protocol::T));
    }
}
