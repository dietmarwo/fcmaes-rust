use neural_controller_policy_search::{
    Algorithm, PARAMS, RunRecord, ScenarioMode, SearchConfig, baselines, evaluate_frozen_test,
    run_search, summarize, write_baselines, write_convergence, write_frozen_test, write_policy,
    write_records, write_trajectory,
};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;

#[derive(Clone, Debug)]
struct Cli {
    experiment: String,
    algorithms: Vec<Algorithm>,
    config: SearchConfig,
    seed: u64,
    seeds: usize,
    scaling_workers: Vec<usize>,
    output: PathBuf,
    policy: Option<PathBuf>,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            experiment: "single".into(),
            algorithms: Algorithm::ALL.to_vec(),
            config: SearchConfig::default(),
            seed: 42,
            seeds: 1,
            scaling_workers: vec![1, 16, 24],
            output: PathBuf::from("results/latest"),
            policy: None,
        }
    }
}

fn usage() -> &'static str {
    "neural-controller-policy-search

USAGE:
  cargo run --release -- [OPTIONS]

OPTIONS:
  --experiment single|quality|noise|scaling|suite|baselines|final-test
  --algo pgpe|crfmnes|cmaes|biteopt|all
  --evaluations N
  --popsize N                    even, default 64
  --workers N                    evaluation workers, default 16
  --scaling-workers CSV          default 1,16,24
  --train-scenarios N            rollouts per candidate, default 4
  --validation-scenarios N       disjoint final rollouts, default 128
  --horizon N                    simulation steps, default 300
  --scenario-mode fixed|rotating
  --monitor-interval N           candidate evaluations between monitor points
  --seed N                       first independent optimizer seed
  --seeds N                      number of independent seeds
  --output DIR
  --policy FILE                   selected policy CSV for final-test
  --help

EXPERIMENTS:
  single      selected algorithms and scenario mode
  quality     selected algorithms with fixed training scenarios
  noise       selected algorithms with rotating common scenarios
  scaling     selected algorithms at every --scaling-workers value
  suite       baselines, quality, noise, and scaling
  baselines   zero-action, initial-neural, and hand-energy-heuristic only
  final-test  evaluate --policy on one frozen seed stream
"
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_csv_usize(value: &str) -> Result<Vec<usize>, String> {
    let values: Result<Vec<_>, _> = value
        .split(',')
        .map(|item| {
            item.parse::<usize>()
                .map_err(|_| format!("invalid integer '{item}'"))
        })
        .collect();
    let values = values?;
    if values.is_empty() || values.contains(&0) {
        return Err("worker list must contain positive integers".into());
    }
    Ok(values)
}

fn parse_cli() -> Result<Cli, String> {
    let args: Vec<String> = env::args().collect();
    let mut cli = Cli::default();
    let mut index = 1;
    while index < args.len() {
        let flag = &args[index];
        match flag.as_str() {
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            "--experiment" => cli.experiment = take_value(&args, &mut index, flag)?,
            "--algo" => {
                let value = take_value(&args, &mut index, flag)?;
                cli.algorithms = if value == "all" {
                    Algorithm::ALL.to_vec()
                } else {
                    vec![Algorithm::from_str(&value)?]
                };
            }
            "--evaluations" => {
                cli.config.evaluations = take_value(&args, &mut index, flag)?
                    .parse()
                    .map_err(|_| "invalid --evaluations".to_string())?
            }
            "--popsize" => {
                cli.config.popsize = take_value(&args, &mut index, flag)?
                    .parse()
                    .map_err(|_| "invalid --popsize".to_string())?
            }
            "--workers" => {
                cli.config.workers = take_value(&args, &mut index, flag)?
                    .parse()
                    .map_err(|_| "invalid --workers".to_string())?
            }
            "--scaling-workers" => {
                cli.scaling_workers = parse_csv_usize(&take_value(&args, &mut index, flag)?)?
            }
            "--train-scenarios" => {
                cli.config.train_scenarios = take_value(&args, &mut index, flag)?
                    .parse()
                    .map_err(|_| "invalid --train-scenarios".to_string())?
            }
            "--validation-scenarios" => {
                cli.config.validation_scenarios = take_value(&args, &mut index, flag)?
                    .parse()
                    .map_err(|_| "invalid --validation-scenarios".to_string())?
            }
            "--horizon" => {
                cli.config.horizon = take_value(&args, &mut index, flag)?
                    .parse()
                    .map_err(|_| "invalid --horizon".to_string())?
            }
            "--scenario-mode" => {
                cli.config.scenario_mode =
                    ScenarioMode::from_str(&take_value(&args, &mut index, flag)?)?
            }
            "--monitor-interval" => {
                cli.config.monitor_interval = take_value(&args, &mut index, flag)?
                    .parse()
                    .map_err(|_| "invalid --monitor-interval".to_string())?
            }
            "--seed" => {
                cli.seed = take_value(&args, &mut index, flag)?
                    .parse()
                    .map_err(|_| "invalid --seed".to_string())?
            }
            "--seeds" => {
                cli.seeds = take_value(&args, &mut index, flag)?
                    .parse()
                    .map_err(|_| "invalid --seeds".to_string())?
            }
            "--output" => cli.output = PathBuf::from(take_value(&args, &mut index, flag)?),
            "--policy" => cli.policy = Some(PathBuf::from(take_value(&args, &mut index, flag)?)),
            _ => return Err(format!("unknown option '{flag}'\n\n{}", usage())),
        }
        index += 1;
    }
    match cli.experiment.as_str() {
        "single" | "quality" | "noise" | "scaling" | "suite" | "baselines" | "final-test" => {}
        other => return Err(format!("unknown experiment '{other}'")),
    }
    if cli.seeds == 0 {
        return Err("--seeds must be positive".into());
    }
    cli.config.validate()?;
    Ok(cli)
}

fn print_record(record: &RunRecord) {
    println!(
        "{:<8} {:<8} seed={:<3} workers={:<2} evals={:<7} train={:.5} validation={:.5} success={:>6.1}% time={:.3}s",
        record.algorithm,
        record.scenario_mode,
        record.seed,
        record.workers,
        record.evaluations,
        record.train_best,
        record.validation.score,
        100.0 * record.validation.success_rate,
        record.wall_seconds,
    );
}

fn run_group(
    label: &str,
    cli: &Cli,
    config: &SearchConfig,
    records: &mut Vec<RunRecord>,
) -> Result<(), String> {
    for seed_offset in 0..cli.seeds {
        let seed = cli.seed + seed_offset as u64;
        for &algorithm in &cli.algorithms {
            let record = run_search(label, algorithm, seed, config)?;
            print_record(&record);
            records.push(record);
        }
    }
    Ok(())
}

fn print_summary(records: &[RunRecord]) {
    println!("\nSummary (sample standard deviations):");
    println!(
        "{:<10} {:<8} {:>7} {:>22} {:>22} {:>22}",
        "experiment",
        "algo",
        "workers",
        "validation mean ± sd",
        "success mean ± sd",
        "seconds mean ± sd"
    );
    let mut keys: Vec<_> = records
        .iter()
        .map(|r| {
            (
                r.experiment.clone(),
                r.scenario_mode,
                r.algorithm,
                r.workers,
            )
        })
        .collect();
    keys.sort_by_key(|key| (key.0.clone(), key.1.as_str(), key.2.as_str(), key.3));
    keys.dedup();
    for (experiment, mode, algorithm, workers) in keys {
        let rows: Vec<_> = records
            .iter()
            .filter(|r| {
                r.experiment == experiment
                    && r.scenario_mode == mode
                    && r.algorithm == algorithm
                    && r.workers == workers
            })
            .collect();
        let validation = summarize(
            &rows
                .iter()
                .map(|row| row.validation.score)
                .collect::<Vec<_>>(),
        );
        let success = summarize(
            &rows
                .iter()
                .map(|row| row.validation.success_rate)
                .collect::<Vec<_>>(),
        );
        let seconds = summarize(&rows.iter().map(|row| row.wall_seconds).collect::<Vec<_>>());
        println!(
            "{:<10} {:<8} {:>7} {:>9.5} ± {:<9.5} {:>8.3} ± {:<8.3} {:>8.3} ± {:<8.3}",
            format!("{experiment}/{mode}"),
            algorithm,
            workers,
            validation.mean,
            validation.sdev,
            success.mean,
            success.sdev,
            seconds.mean,
            seconds.sdev,
        );
    }
}

fn read_policy(path: &Path) -> Result<Vec<f64>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut policy = Vec::new();
    for (line_number, line) in text.lines().enumerate().skip(1) {
        let (_, value) = line
            .split_once(',')
            .ok_or_else(|| format!("invalid policy row {}", line_number + 1))?;
        policy.push(
            value
                .parse::<f64>()
                .map_err(|_| format!("invalid weight on row {}", line_number + 1))?,
        );
    }
    if policy.len() != PARAMS {
        return Err(format!(
            "policy has {} weights, expected {PARAMS}",
            policy.len()
        ));
    }
    Ok(policy)
}

fn persist(
    output: &Path,
    baselines_only: bool,
    records: &[RunRecord],
    cli: &Cli,
) -> Result<(), String> {
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    let baseline_rows = baselines(cli.seed, &cli.config);
    write_baselines(&output.join("baselines.csv"), &baseline_rows)?;
    println!(
        "\nBaselines on {} disjoint scenarios:",
        cli.config.validation_scenarios
    );
    for row in &baseline_rows {
        println!(
            "{:<18} validation={:.5} success={:>6.1}% mean_steps={:.1}",
            row.name,
            row.validation.score,
            100.0 * row.validation.success_rate,
            row.validation.mean_steps,
        );
    }
    if baselines_only || records.is_empty() {
        return Ok(());
    }
    write_records(&output.join("runs.csv"), records)?;
    write_convergence(&output.join("convergence.csv"), records)?;
    let best = records
        .iter()
        .min_by(|a, b| {
            a.validation
                .score
                .partial_cmp(&b.validation.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("records are non-empty");
    write_policy(&output.join("best_policy.csv"), best)?;
    write_trajectory(
        &output.join("best_trajectory.csv"),
        best,
        cli.seed ^ 0xe703_7ed1_a0b4_28db,
    )?;
    println!(
        "\nBest validated policy: {} seed={} mode={} score={:.6}",
        best.algorithm, best.seed, best.scenario_mode, best.validation.score
    );
    Ok(())
}

fn run(cli: Cli) -> Result<(), String> {
    let mut records = Vec::new();
    match cli.experiment.as_str() {
        "baselines" => persist(&cli.output, true, &records, &cli),
        "final-test" => {
            let path = cli
                .policy
                .as_deref()
                .ok_or_else(|| "--experiment final-test requires --policy".to_string())?;
            let policy = read_policy(path)?;
            let metrics =
                evaluate_frozen_test(&policy, cli.config.validation_scenarios, cli.config.horizon);
            write_frozen_test(
                &cli.output.join("frozen_final_test.csv"),
                cli.config.validation_scenarios,
                metrics,
            )?;
            println!(
                "frozen final test: scenarios={} score={:.6} cvar={:.6} success={:.1}% mean_steps={:.1} rms_force={:.3}",
                cli.config.validation_scenarios,
                metrics.score,
                metrics.cvar_loss,
                100.0 * metrics.success_rate,
                metrics.mean_steps,
                metrics.rms_force,
            );
            Ok(())
        }
        "single" => {
            run_group("single", &cli, &cli.config, &mut records)?;
            print_summary(&records);
            persist(&cli.output, false, &records, &cli)
        }
        "quality" => {
            let mut config = cli.config.clone();
            config.scenario_mode = ScenarioMode::Fixed;
            run_group("quality", &cli, &config, &mut records)?;
            print_summary(&records);
            persist(&cli.output, false, &records, &cli)
        }
        "noise" => {
            let mut config = cli.config.clone();
            config.scenario_mode = ScenarioMode::Rotating;
            run_group("noise", &cli, &config, &mut records)?;
            print_summary(&records);
            persist(&cli.output, false, &records, &cli)
        }
        "scaling" => {
            for &workers in &cli.scaling_workers {
                let mut config = cli.config.clone();
                config.workers = workers;
                run_group("scaling", &cli, &config, &mut records)?;
            }
            print_summary(&records);
            persist(&cli.output, false, &records, &cli)
        }
        "suite" => {
            let mut quality = cli.config.clone();
            quality.scenario_mode = ScenarioMode::Fixed;
            run_group("quality", &cli, &quality, &mut records)?;

            let mut noise = cli.config.clone();
            noise.scenario_mode = ScenarioMode::Rotating;
            run_group("noise", &cli, &noise, &mut records)?;

            for &workers in &cli.scaling_workers {
                let mut scaling = cli.config.clone();
                scaling.scenario_mode = ScenarioMode::Fixed;
                scaling.workers = workers;
                run_group("scaling", &cli, &scaling, &mut records)?;
            }
            print_summary(&records);
            persist(&cli.output, false, &records, &cli)
        }
        _ => unreachable!(),
    }
}

fn main() -> ExitCode {
    match parse_cli().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
