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
use crate::seeds::{measure_seed, run_seed};
use crate::solvers::uniform_point;

/// Uniform random points timed per problem.
pub const MEASURE_POINTS: usize = 2_000;

/// Independent runs of protocol R.
pub const RESTARTS_R: u32 = 8;

/// Header of `budgets.tsv`.
pub const BUDGET_HEADER: &str = "problem\tdomain\tdim\tmean_eval_us\tbudget";

/// Header of `raw.tsv`.
pub const RAW_HEADER: &str = "solver\tproblem\tdomain\tdim\tprotocol\trep\tseed\tbudget\tbudget_scale\tevaluations\tbest_value\tover_budget\tout_of_bounds\twall_ms";

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
}

impl Protocol {
    /// Protocol letter.
    pub fn letter(self) -> char {
        match self {
            Self::S => 'S',
            Self::R => 'R',
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
            _ => Err(format!("unknown protocol '{text}' (expected S or R)")),
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
pub fn restart_budget(protocol: Protocol, scaled: u64) -> u64 {
    match protocol {
        Protocol::S => scaled,
        Protocol::R => scaled / u64::from(RESTARTS_R),
    }
}

/// Scaled budget `⌊B · scale⌋`, at least one evaluation.
pub fn scaled_budget(budget: u64, scale: f64) -> u64 {
    ((budget as f64 * scale).floor() as u64).max(1)
}

/// Execute one repetition: every run goes through a fresh [`Guard`].
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

/// Key of a raw row: solver, problem, protocol, repetition.
pub type RowKey = (String, String, char, u32);

/// One parsed raw row (the fields the runner needs on resumption).
#[derive(Clone, Debug)]
pub struct RawRow {
    /// Row key.
    pub key: RowKey,
    /// Budget scale of the row.
    pub budget_scale: f64,
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

/// Load the rows of an existing raw file. A trailing line without newline
/// (an interrupted write) is cut from the file. Creates the file with its
/// header if it does not exist.
///
/// # Errors
/// Returns a message for I/O or format errors.
pub fn load_or_create_raw(path: &Path) -> Result<Vec<RawRow>, String> {
    let describe = |e: std::io::Error| format!("{}: {e}", path.display());
    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(describe)?;
        }
        std::fs::write(path, format!("{RAW_HEADER}\n")).map_err(describe)?;
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
        file.write_all(format!("{RAW_HEADER}\n").as_bytes())
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
            file.write_all(format!("{RAW_HEADER}\n").as_bytes())
                .map_err(describe)?;
            return Ok(Vec::new());
        }
    }
    let mut rows = Vec::new();
    let mut lines = BufReader::new(bytes.as_slice()).lines();
    let header = lines.next().transpose().map_err(describe)?;
    if header.as_deref() != Some(RAW_HEADER) {
        return Err(format!("{}: unexpected header", path.display()));
    }
    for line in lines {
        let line = line.map_err(describe)?;
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 14 {
            return Err(format!("{}: malformed row '{line}'", path.display()));
        }
        let bad = |name: &str| format!("{}: bad {name} in '{line}'", path.display());
        let protocol = fields[4].chars().next().ok_or_else(|| bad("protocol"))?;
        rows.push(RawRow {
            key: (
                fields[0].to_owned(),
                fields[1].to_owned(),
                protocol,
                fields[5].parse().map_err(|_| bad("rep"))?,
            ),
            budget_scale: fields[8].parse().map_err(|_| bad("budget_scale"))?,
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
    repetition: u32,
    cell: usize,
}

/// Run the campaign, appending rows to `out`. Returns the number of rows
/// written by this call.
///
/// # Errors
/// Returns a message for I/O errors, inconsistent budget scales or a
/// budget too small for protocol R.
pub fn run_campaign(config: &CampaignConfig<'_>, out: &Path) -> Result<usize, String> {
    let existing = load_or_create_raw(out)?;
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
    for (problem_index, &(problem, _)) in config.problems.iter().enumerate() {
        for &protocol in &config.protocols {
            for (solver_index, solver) in config.solvers.iter().enumerate() {
                let cell = cell_names.len();
                cell_names.push(format!(
                    "{} {} {}",
                    solver.name(),
                    problem.name,
                    protocol.letter()
                ));
                let mut pending = 0;
                for repetition in 0..config.repetitions {
                    let key = (
                        solver.name().to_owned(),
                        problem.name.to_owned(),
                        protocol.letter(),
                        repetition,
                    );
                    if !done.contains(&key) {
                        tasks.push(Task {
                            solver: solver_index,
                            problem: problem_index,
                            protocol,
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
                    let scaled = scaled_budget(budget, config.budget_scale);
                    let result =
                        run_repetition(solver, problem, task.protocol, task.repetition, scaled);
                    let line = format_row(
                        solver.name(),
                        problem,
                        task.protocol,
                        task.repetition,
                        config.budget_scale,
                        &result,
                    );
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

/// Count rows per (solver, problem, protocol) in a raw file, for summaries.
///
/// # Errors
/// Returns a message for I/O or format errors.
pub fn count_cells(path: &Path) -> Result<BTreeMap<(String, String, char), usize>, String> {
    let mut counts = BTreeMap::new();
    for row in load_or_create_raw(path)? {
        let (solver, problem, protocol, _) = row.key;
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
}
