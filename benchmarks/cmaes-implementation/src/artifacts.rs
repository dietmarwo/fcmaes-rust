use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::adapters::{Arm, Library, RunMetrics};
use crate::config::Config;
use crate::problems::Problem;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResultRow {
    pub preset: String,
    pub problem: String,
    pub dimension: usize,
    pub optimum: f64,
    pub injected_cost_ns: u64,
    pub calibration_ns_per_eval: f64,
    pub arm: Arm,
    pub library: Library,
    pub seed: u64,
    pub workers: usize,
    pub population: usize,
    pub deadline_ms: u64,
    pub wall_seconds: f64,
    pub allocated_seconds: f64,
    pub cpu_seconds: f64,
    pub active_cores: f64,
    pub allocated_cores: f64,
    pub evaluations: u64,
    pub best: f64,
    pub optimizer_runs: usize,
    pub termination: String,
}

impl ResultRow {
    pub fn new(context: ResultContext<'_>, metrics: RunMetrics) -> Self {
        let ResultContext {
            preset,
            problem,
            injected_cost_ns,
            calibration_ns_per_eval,
            arm,
            library,
            seed,
            population,
            deadline_ms,
        } = context;
        Self {
            preset: preset.to_owned(),
            problem: problem.key.to_owned(),
            dimension: problem.dimension,
            optimum: problem.optimum,
            injected_cost_ns,
            calibration_ns_per_eval,
            arm,
            library,
            seed,
            workers: metrics.workers,
            population,
            deadline_ms,
            wall_seconds: metrics.wall_seconds,
            allocated_seconds: metrics.allocated_seconds,
            cpu_seconds: metrics.cpu_seconds,
            active_cores: metrics.active_cores,
            allocated_cores: metrics.allocated_cores,
            evaluations: metrics.evaluations,
            best: metrics.best,
            optimizer_runs: metrics.optimizer_runs,
            termination: metrics.termination,
        }
    }

    pub fn key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}",
            self.problem,
            self.injected_cost_ns,
            self.arm,
            self.library,
            self.seed,
            self.deadline_ms
        )
    }
}

pub struct ResultContext<'a> {
    pub preset: &'a str,
    pub problem: &'a Problem,
    pub injected_cost_ns: u64,
    pub calibration_ns_per_eval: f64,
    pub arm: Arm,
    pub library: Library,
    pub seed: u64,
    pub population: usize,
    pub deadline_ms: u64,
}

type PairKey = (String, u64, Arm, u64, u64);
type LibraryPair<'a> = [Option<&'a ResultRow>; 2];

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
    fs::create_dir_all(&config.output).map_err(|error| error.to_string())?;
    let path = config.output.join("run.json");
    let initial_command = File::open(&path)
        .ok()
        .and_then(|file| serde_json::from_reader::<_, serde_json::Value>(file).ok())
        .and_then(|manifest| manifest["command"].as_str().map(str::to_owned))
        .unwrap_or_else(|| command.to_owned());
    let manifest = json!({
        "schema_version": 1,
        "tutorial": "cmaes-implementation-comparison",
        "formulation": "paired-wall-deadline",
        "status": "completed",
        "command": initial_command,
        "last_command": command,
        "seed": config.root_seed,
        "workers": config.workers,
        "requested_evaluations": null,
        "actual_evaluations": rows.iter().map(|row| row.evaluations).sum::<u64>(),
        "elapsed_seconds": rows.iter().map(|row| row.wall_seconds).sum::<f64>(),
        "allocated_seconds": rows.iter().map(|row| row.allocated_seconds).sum::<f64>(),
        "process_cpu_seconds": rows.iter().map(|row| row.cpu_seconds).sum::<f64>(),
        "allocated_worker_seconds": rows.iter().map(|row| row.allocated_seconds * row.workers as f64).sum::<f64>(),
        "objectives": [{"column": "best", "label": "best objective", "unit": null}],
        "descriptors": [],
        "protocol": {
            "preset": format!("{:?}", config.preset).to_ascii_lowercase(),
            "problems": config.problem_keys,
            "arms": config.arms,
            "injected_cost_ns": config.costs_ns,
            "deadlines_ms": config.deadlines_ms,
            "paired_seeds": config.seeds,
            "population_override": config.population,
            "sigma0_normalized": 0.3,
            "bound_transform": "shared reflection from unbounded normalized coordinates",
            "termination": "external generation-boundary wall deadline; internal stops recorded",
            "cmaes_version": "0.2.2",
            "cmaes_default_features": false
        },
        "machine": {
            "logical_cpus": num_cpus::get(),
            "physical_cpus": num_cpus::get_physical(),
            "cpu_model": cpu_model(),
            "rustc": command_output("rustc", &["--version"]),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH
        },
        "artifacts": {
            "paired_rows": "paired.csv",
            "comparison": "comparison.md"
        }
    });
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(&mut file, &manifest).map_err(|error| error.to_string())?;
    writeln!(file).map_err(|error| error.to_string())
}

pub fn render_report(output: &Path) -> Result<(), String> {
    let rows = load_rows(&output.join("paired.csv"))?;
    if rows.is_empty() {
        return Err("paired.csv contains no rows".to_owned());
    }
    let mut groups: BTreeMap<(String, u64, Arm, u64), Vec<&ResultRow>> = BTreeMap::new();
    for row in &rows {
        groups
            .entry((
                row.problem.clone(),
                row.injected_cost_ns,
                row.arm,
                row.deadline_ms,
            ))
            .or_default()
            .push(row);
    }

    let mut all_pairs: BTreeMap<PairKey, LibraryPair<'_>> = BTreeMap::new();
    for row in &rows {
        let pair = all_pairs
            .entry((
                row.problem.clone(),
                row.injected_cost_ns,
                row.arm,
                row.seed,
                row.deadline_ms,
            ))
            .or_insert([None, None]);
        pair[match row.library {
            Library::Fcmaes => 0,
            Library::Cmaes => 1,
        }] = Some(row);
    }
    let complete_pairs: Vec<_> = all_pairs
        .values()
        .filter_map(|pair| Some((pair[0]?, pair[1]?)))
        .collect();
    let max_deadline = rows.iter().map(|row| row.deadline_ms).max().unwrap_or(0);
    let mut endpoint_outcomes: BTreeMap<Arm, [usize; 3]> = BTreeMap::new();
    for (fcmaes, cmaes) in complete_pairs
        .iter()
        .filter(|pair| pair.0.deadline_ms == max_deadline)
    {
        endpoint_outcomes.entry(fcmaes.arm).or_default()[outcome_index(fcmaes, cmaes)] += 1;
    }

    let mut report = String::new();
    report.push_str("# Controlled active CMA-ES implementation diagnostic\n\n");
    report.push_str(
        "These are paired wall-deadline measurements of `fcmaes-core` and `cmaes` 0.2.2. \
Both use active (negative-weight) CMA-ES, the same normalized reflected objective, explicit \
population size, initial mean, and sigma. This is an implementation diagnostic, not a general \
CMA-ES performance benchmark: its easy analytic functions isolate overhead, scaling, and \
stopping behavior rather than recommending CMA-ES for those functions. Matching numeric seeds \
label pairs but do not create matching random streams. A run may exceed its deadline by one \
generation.\n\n",
    );
    let internal_stops = |library| {
        rows.iter()
            .filter(|row| row.library == library && row.termination != "deadline")
            .count()
    };
    let library_rows = |library| rows.iter().filter(|row| row.library == library).count();
    report.push_str("## Bundle summary\n\n");
    report
        .push_str("| Raw rows | Complete pairs | Objective calls | Active wall | Process CPU |\n");
    report.push_str("|---:|---:|---:|---:|---:|\n");
    report.push_str(&format!(
        "| {} | {} | {} | {:.3} h | {:.3} h |\n\n",
        rows.len(),
        complete_pairs.len(),
        rows.iter().map(|row| row.evaluations).sum::<u64>(),
        rows.iter().map(|row| row.wall_seconds).sum::<f64>() / 3600.0,
        rows.iter().map(|row| row.cpu_seconds).sum::<f64>() / 3600.0,
    ));
    report.push_str("| Library | Internal-stop rows | All rows |\n");
    report.push_str("|---|---:|---:|\n");
    for (library, label) in [(Library::Fcmaes, "fcmaes-core"), (Library::Cmaes, "cmaes")] {
        report.push_str(&format!(
            "| {label} | {} | {} |\n",
            internal_stops(library),
            library_rows(library),
        ));
    }
    report.push_str(&format!(
        "\nPaired final-objective outcomes at the longest endpoint ({max_deadline} ms):\n\n"
    ));
    report.push_str("| Arm | fcmaes-core wins | cmaes wins | Ties | Pairs |\n");
    report.push_str("|---|---:|---:|---:|---:|\n");
    for (arm, outcomes) in endpoint_outcomes {
        report.push_str(&format!(
            "| {arm} | {} | {} | {} | {} |\n",
            outcomes[0],
            outcomes[1],
            outcomes[2],
            outcomes.iter().sum::<usize>(),
        ));
    }
    report.push_str("\n## Detailed results\n\n");
    report.push_str(
        "The table reports medians over available seeds. Wins compare final objective values \
within a relative `1e-10` tie band. `Residual ns/eval` is shown only for serial Arm A and \
subtracts the separately measured shared-objective calibration; it is diagnostic, not a \
library-internal profiler.\n\n",
    );
    report.push_str("| Problem | Cost | Arm | Deadline | Pairs | Wins fc/cma/tie | Internal stops fc/cma | Median best fc | Median best cma | Median eval/s fc | Median eval/s cma | Allocated cores fc | Allocated cores cma | Residual ns/eval fc | Residual ns/eval cma |\n");
    report.push_str("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");

    for ((problem, cost, arm, deadline), group) in groups {
        let mut by_seed: HashMap<u64, [Option<&ResultRow>; 2]> = HashMap::new();
        for row in group {
            let pair = by_seed.entry(row.seed).or_insert([None, None]);
            pair[match row.library {
                Library::Fcmaes => 0,
                Library::Cmaes => 1,
            }] = Some(row);
        }
        let pairs: Vec<_> = by_seed
            .values()
            .filter_map(|pair| Some((pair[0]?, pair[1]?)))
            .collect();
        if pairs.is_empty() {
            continue;
        }
        let mut wins = [0usize; 3];
        for (fcmaes, cmaes) in &pairs {
            wins[outcome_index(fcmaes, cmaes)] += 1;
        }
        let fc: Vec<_> = pairs.iter().map(|pair| pair.0).collect();
        let cm: Vec<_> = pairs.iter().map(|pair| pair.1).collect();
        let internal_stops = [
            fc.iter()
                .filter(|row| row.termination != "deadline")
                .count(),
            cm.iter()
                .filter(|row| row.termination != "deadline")
                .count(),
        ];
        let residual = |rows: &[&ResultRow]| {
            if arm != Arm::A {
                return f64::NAN;
            }
            median(
                rows.iter()
                    .filter(|row| row.evaluations > 0)
                    .map(|row| {
                        (row.wall_seconds * 1e9 / row.evaluations as f64
                            - row.calibration_ns_per_eval)
                            .max(0.0)
                    })
                    .collect(),
            )
        };
        report.push_str(&format!(
            "| {problem} | {cost} ns | {arm} | {deadline} ms | {} | {}/{}/{} | {}/{} | {:.6e} | {:.6e} | {:.0} | {:.0} | {:.2} | {:.2} | {} | {} |\n",
            pairs.len(),
            wins[0],
            wins[1],
            wins[2],
            internal_stops[0],
            internal_stops[1],
            median(fc.iter().map(|row| row.best).collect()),
            median(cm.iter().map(|row| row.best).collect()),
            median(fc.iter().map(|row| row.evaluations as f64 / row.wall_seconds).collect()),
            median(cm.iter().map(|row| row.evaluations as f64 / row.wall_seconds).collect()),
            median(fc.iter().map(|row| row.allocated_cores).collect()),
            median(cm.iter().map(|row| row.allocated_cores).collect()),
            format_optional(residual(&fc)),
            format_optional(residual(&cm)),
        ));
    }
    report.push_str("\n## Interpretation boundary\n\n");
    report.push_str(
        "Arm A is the closest same-family implementation arm. Arm B additionally compares each \
library's population-evaluation path. Arm C gives both implementations the same independent-\
multistart architecture. None of these rows compares fcmaes coordinated DE→CMA retry with \
`cmaes` BIPOP; that is a different-algorithm system comparison already covered by the broader \
optimizer benchmark. The analytic controls do not establish that CMA-ES is the right solver for \
them, and sufficiently costly application objectives make the measured implementation overhead \
irrelevant. Smoke and pilot presets validate the harness but are not publication evidence.\n",
    );
    fs::write(output.join("comparison.md"), report).map_err(|error| error.to_string())
}

fn outcome_index(fcmaes: &ResultRow, cmaes: &ResultRow) -> usize {
    let scale = 1.0_f64.max(fcmaes.best.abs()).max(cmaes.best.abs());
    if (fcmaes.best - cmaes.best).abs() <= 1e-10 * scale {
        2
    } else if fcmaes.best < cmaes.best {
        0
    } else {
        1
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

fn format_optional(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.0}")
    } else {
        "—".to_owned()
    }
}

fn command_output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn cpu_model() -> String {
    let Ok(text) = fs::read_to_string("/proc/cpuinfo") else {
        return "unavailable".to_owned();
    };
    text.lines()
        .find_map(|line| line.strip_prefix("model name\t: "))
        .unwrap_or("unavailable")
        .to_owned()
}
