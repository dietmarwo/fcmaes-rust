use std::collections::{HashMap, HashSet};
use std::fs;
use std::time::Duration;

use cmaes_implementation_comparison::{
    Arm, Config, Library, Mode, ResultContext, ResultRow, RunRequest, calibrate, problems,
    render_report, run_one, write_manifest,
};

fn main() {
    if let Err(message) = try_main() {
        eprintln!("Error: {message}");
        std::process::exit(2);
    }
}

fn try_main() -> Result<(), String> {
    let Some(config) = Config::from_env()? else {
        println!("{}", Config::USAGE);
        return Ok(());
    };
    match config.mode {
        Mode::Verify => verify_contract(),
        Mode::Report => render_report(&config.output),
        Mode::Campaign => run_campaign(&config),
    }
}

fn verify_contract() -> Result<(), String> {
    let options = cmaes::CMAESOptions::new(vec![0.0; 3], 0.3);
    if options.weights != cmaes::Weights::Negative {
        return Err("cmaes default is not active negative-weight CMA-ES".to_owned());
    }
    let vector = cmaes::DVector::from_vec(vec![1.0, 2.0, 3.0]);
    if vector.as_ptr() != vector.as_slice().as_ptr() {
        return Err("DVector::as_slice() is not zero-copy".to_owned());
    }
    println!(
        "contract ok: active weights, zero-copy DVector slice, {} physical / {} logical CPUs",
        num_cpus::get_physical(),
        num_cpus::get()
    );
    Ok(())
}

fn run_campaign(config: &Config) -> Result<(), String> {
    fs::create_dir_all(&config.output).map_err(|error| error.to_string())?;
    let csv_path = config.output.join("paired.csv");
    if csv_path.exists() && !config.resume {
        return Err(format!(
            "{} already exists; use --resume or choose another output directory",
            csv_path.display()
        ));
    }
    let registry: HashMap<_, _> = problems()
        .into_iter()
        .map(|problem| (problem.key.to_owned(), problem))
        .collect();
    let selected: Vec<_> = config
        .problem_keys
        .iter()
        .map(|key| {
            registry
                .get(key)
                .cloned()
                .ok_or_else(|| format!("unknown problem '{key}'"))
        })
        .collect::<Result<_, _>>()?;
    let existing = cmaes_implementation_comparison::artifacts::load_rows(&csv_path)?;
    validate_existing(config, &selected, &existing)?;
    let mut completed: HashSet<String> = existing.iter().map(ResultRow::key).collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .thread_name(|index| format!("cmaes-comparison-{index}"))
        .build()
        .map_err(|error| error.to_string())?;
    pool.install(|| assert_eq!(rayon::current_num_threads(), config.workers));

    let preset = format!("{:?}", config.preset).to_ascii_lowercase();
    let mut calibrations = HashMap::new();
    for (problem_index, problem) in selected.iter().enumerate() {
        for (cost_index, &cost_ns) in config.costs_ns.iter().enumerate() {
            if problem.natural_cost_only && cost_ns != 0 {
                continue;
            }
            let calibration = *calibrations
                .entry((problem.key, cost_ns))
                .or_insert_with(|| calibrate(problem, cost_ns));
            for (deadline_index, &deadline_ms) in config.deadlines_ms.iter().enumerate() {
                for seed_index in 0..config.seeds {
                    let seed = config
                        .root_seed
                        .wrapping_add((problem_index as u64).wrapping_mul(1_000_003))
                        .wrapping_add((cost_index as u64).wrapping_mul(100_003))
                        .wrapping_add((deadline_index as u64).wrapping_mul(10_007))
                        .wrapping_add(seed_index as u64);
                    for &arm in &config.arms {
                        let libraries =
                            if (seed_index + deadline_index + arm_index(arm)).is_multiple_of(2) {
                                [Library::Fcmaes, Library::Cmaes]
                            } else {
                                [Library::Cmaes, Library::Fcmaes]
                            };
                        for library in libraries {
                            let population = config
                                .population
                                .unwrap_or_else(|| problem.default_population());
                            let key = format!(
                                "{}|{}|{}|{}|{}|{}",
                                problem.key, cost_ns, arm, library, seed, deadline_ms
                            );
                            if completed.contains(&key) {
                                continue;
                            }
                            let metrics = run_one(RunRequest {
                                library,
                                arm,
                                problem,
                                injected_cost_ns: cost_ns,
                                deadline: Duration::from_millis(deadline_ms),
                                workers: config.workers,
                                population,
                                seed,
                                pool: &pool,
                            });
                            println!(
                                "{} cost={} arm={} deadline={}ms seed={} {} evals={} best={:.6e} wall={:.4}s cores={:.2}",
                                problem.key,
                                cost_ns,
                                arm,
                                deadline_ms,
                                seed,
                                library,
                                metrics.evaluations,
                                metrics.best,
                                metrics.wall_seconds,
                                metrics.allocated_cores,
                            );
                            let row = ResultRow::new(
                                ResultContext {
                                    preset: &preset,
                                    problem,
                                    injected_cost_ns: cost_ns,
                                    calibration_ns_per_eval: calibration,
                                    arm,
                                    library,
                                    seed,
                                    population,
                                    deadline_ms,
                                },
                                metrics,
                            );
                            cmaes_implementation_comparison::artifacts::append_row(
                                &csv_path, &row,
                            )?;
                            completed.insert(row.key());
                        }
                    }
                }
            }
        }
    }
    let rows = cmaes_implementation_comparison::artifacts::load_rows(&csv_path)?;
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    write_manifest(config, &rows, &command)?;
    render_report(&config.output)
}

fn arm_index(arm: Arm) -> usize {
    match arm {
        Arm::A => 0,
        Arm::B => 1,
        Arm::C => 2,
    }
}

fn validate_existing(
    config: &Config,
    problems: &[cmaes_implementation_comparison::Problem],
    rows: &[ResultRow],
) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }
    let preset = format!("{:?}", config.preset).to_ascii_lowercase();
    let problem_keys: HashSet<_> = problems.iter().map(|problem| problem.key).collect();
    for row in rows {
        let expected_population = config.population.unwrap_or_else(|| {
            problems
                .iter()
                .find(|problem| problem.key == row.problem)
                .map_or(row.population, |problem| problem.default_population())
        });
        let expected_workers = if row.arm == Arm::A { 1 } else { config.workers };
        if row.preset != preset
            || !problem_keys.contains(row.problem.as_str())
            || !config.costs_ns.contains(&row.injected_cost_ns)
            || !config.arms.contains(&row.arm)
            || !config.deadlines_ms.contains(&row.deadline_ms)
            || row.population != expected_population
            || row.workers != expected_workers
        {
            return Err(format!(
                "existing row '{}' is incompatible with the requested resume protocol",
                row.key()
            ));
        }
    }
    let manifest_path = config.output.join("run.json");
    if manifest_path.exists() {
        let manifest: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&manifest_path).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if manifest["seed"].as_u64() != Some(config.root_seed)
            || manifest["workers"].as_u64() != Some(config.workers as u64)
        {
            return Err("run.json seed or worker count conflicts with --resume".to_owned());
        }
    }
    Ok(())
}
