use std::collections::HashSet;
use std::fs;
use std::io::Write;

use gtop_cmaes_retry::{
    Config, Mode, campaign_rows, load_rows, render_report, resolve_cases, run_case,
    validate_resume, write_manifest,
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
        Mode::Verify => verify(),
        Mode::Report => render_report(&config.output),
        Mode::Campaign => campaign(&config),
    }
}

fn verify() -> Result<(), String> {
    let options = cmaes::CMAESOptions::new(vec![0.0; 3], 0.3);
    if options.weights != cmaes::Weights::Negative {
        return Err("external cmaes default is not active CMA-ES".to_owned());
    }
    println!(
        "adapter contract ok: active external CMA-ES; every optimizer instance is serial; {} physical / {} logical CPUs",
        num_cpus::get_physical(),
        num_cpus::get()
    );
    Ok(())
}

fn campaign(config: &Config) -> Result<(), String> {
    fs::create_dir_all(&config.output).map_err(|error| error.to_string())?;
    let csv_path = config.output.join("results.csv");
    if csv_path.exists() && !config.resume {
        return Err(format!(
            "{} exists; pass --resume or choose another output directory",
            csv_path.display()
        ));
    }
    let existing = load_rows(&csv_path)?;
    validate_resume(config, &existing)?;
    let mut completed: HashSet<_> = existing.iter().map(|row| row.key()).collect();
    let cases = resolve_cases(&config.problem_keys)?;
    let matrix = campaign_rows(config);
    for (case_index, case) in cases.iter().enumerate() {
        for run_index in 0..config.runs {
            let seed = config
                .seed
                .wrapping_add((case_index as u64).wrapping_mul(1_000_003))
                .wrapping_add(run_index as u64);
            let mut ordered = matrix.clone();
            if run_index.is_multiple_of(2) {
                ordered.reverse();
            }
            for &(phase, arm, workers) in &ordered {
                let provisional = format!(
                    "{}|{}|{}|{}|{}|{}",
                    phase,
                    arm,
                    case.key,
                    run_index + 1,
                    seed,
                    match arm {
                        gtop_cmaes_retry::Arm::ExternalSingle
                        | gtop_cmaes_retry::Arm::ExternalSequential => 1,
                        _ => workers,
                    }
                );
                if completed.contains(&provisional) {
                    continue;
                }
                let row = run_case(config, case, phase, arm, workers, run_index + 1, seed);
                println!(
                    "phase={} problem={} run={}/{} arm={} workers={} success={} best={:.12} lanes={}/{} starts={} evaluations={} wall={:.4}s cpu={:.4}s cores={:.2}",
                    row.phase,
                    row.problem,
                    row.run,
                    config.runs,
                    row.arm,
                    row.workers,
                    row.success,
                    row.best,
                    row.retries_completed,
                    row.retries_requested,
                    row.optimizer_starts,
                    row.evaluations_actual,
                    row.wall_seconds,
                    row.cpu_seconds,
                    row.average_cores,
                );
                std::io::stdout()
                    .flush()
                    .map_err(|error| error.to_string())?;
                gtop_cmaes_retry::append_row(&csv_path, &row)?;
                completed.insert(row.key());
            }
        }
    }
    let rows = load_rows(&csv_path)?;
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    write_manifest(config, &rows, &command)?;
    render_report(&config.output)
}
