use std::error::Error;
use std::path::PathBuf;

use rebop_oscillator::{
    EvaluationConfig, LogRates, MoProgress, MultiOptions, ParetoPoint, QdOptions, RobustMetrics,
    ScalarOptions, evaluate_training, evaluate_validation, optimize_multi, optimize_qd,
    optimize_scalar, write_artifacts, write_qd_artifacts,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    Single,
    Multi,
    Both,
    Qd,
    All,
    Simulate,
}

#[derive(Debug)]
struct Args {
    mode: RunMode,
    target_period: f64,
    replications: usize,
    validation_replications: usize,
    evaluations: u64,
    retries: usize,
    workers: usize,
    depth: i32,
    mo_evaluations: usize,
    popsize: usize,
    qd_evaluations: usize,
    qd_capacity: usize,
    qd_chunk_size: usize,
    seed: u64,
    output: PathBuf,
    write_output: bool,
    design: Option<LogRates>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: RunMode::Both,
            target_period: 20.0,
            replications: 4,
            validation_replications: 8,
            evaluations: 2_000,
            retries: 8,
            workers: 0,
            depth: 6,
            mo_evaluations: 20_000,
            popsize: 128,
            qd_evaluations: 20_000,
            qd_capacity: 400,
            qd_chunk_size: 128,
            seed: 42,
            output: PathBuf::from("results"),
            write_output: true,
            design: None,
        }
    }
}

fn usage() {
    println!(
        "ReBop robust stochastic oscillator optimization\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --mode NAME                 single, multi, qd, both, all, or simulate (both)\n\
         --target-period T           Desired period in model time units (20)\n\
         --replications N            Fixed training seeds per candidate (4)\n\
         --validation-replications N Disjoint holdout seeds for final designs (8)\n\
         --evaluations N             BiteOpt evaluations per retry (2000)\n\
         --retries N                 Independent BiteOpt retries (8)\n\
         --workers N                 fcmaes workers; 0 uses available CPUs (0)\n\
         --depth N                   BiteOpt deep populations, 1..36 (6)\n\
         --mo-evaluations N          Total requested MODE evaluations (20000)\n\
         --popsize N                 MODE population size (128)\n\
         --qd-evaluations N          Total requested MAP-Elites evaluations (20000)\n\
         --qd-capacity N             Square MAP-Elites archive capacity (400)\n\
         --qd-chunk-size N           Even QD evaluation batch size (128)\n\
         --seed N                    Optimizer root seed (42)\n\
         --output DIR                CSV/report output directory (results)\n\
         --no-output                 Do not write CSV or HTML files\n\
         --x CSV                     Fifteen log10 rates for simulate mode\n\
         -h, --help                  Show this help"
    );
}

fn take_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_design(value: &str) -> Result<LogRates, Box<dyn Error>> {
    let values = value
        .split(',')
        .map(|field| field.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LogRates::from_slice(&values)?)
}

fn parse_args() -> Result<Option<Args>, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--mode" => {
                parsed.mode = match take_value(&mut args, "--mode")?.as_str() {
                    "single" => RunMode::Single,
                    "multi" => RunMode::Multi,
                    "both" => RunMode::Both,
                    "qd" => RunMode::Qd,
                    "all" => RunMode::All,
                    "simulate" => RunMode::Simulate,
                    value => return Err(format!("unknown mode: {value}").into()),
                };
            }
            "--target-period" => {
                parsed.target_period = take_value(&mut args, "--target-period")?.parse()?
            }
            "--replications" => {
                parsed.replications = take_value(&mut args, "--replications")?.parse()?
            }
            "--validation-replications" => {
                parsed.validation_replications =
                    take_value(&mut args, "--validation-replications")?.parse()?
            }
            "--evaluations" => {
                parsed.evaluations = take_value(&mut args, "--evaluations")?.parse()?
            }
            "--retries" => parsed.retries = take_value(&mut args, "--retries")?.parse()?,
            "--workers" => parsed.workers = take_value(&mut args, "--workers")?.parse()?,
            "--depth" => parsed.depth = take_value(&mut args, "--depth")?.parse()?,
            "--mo-evaluations" => {
                parsed.mo_evaluations = take_value(&mut args, "--mo-evaluations")?.parse()?
            }
            "--popsize" => parsed.popsize = take_value(&mut args, "--popsize")?.parse()?,
            "--qd-evaluations" => {
                parsed.qd_evaluations = take_value(&mut args, "--qd-evaluations")?.parse()?
            }
            "--qd-capacity" => {
                parsed.qd_capacity = take_value(&mut args, "--qd-capacity")?.parse()?
            }
            "--qd-chunk-size" => {
                parsed.qd_chunk_size = take_value(&mut args, "--qd-chunk-size")?.parse()?
            }
            "--seed" => parsed.seed = take_value(&mut args, "--seed")?.parse()?,
            "--output" => parsed.output = take_value(&mut args, "--output")?.into(),
            "--no-output" => parsed.write_output = false,
            "--x" => parsed.design = Some(parse_design(&take_value(&mut args, "--x")?)?),
            value => return Err(format!("unknown argument: {value}").into()),
        }
    }
    Ok(Some(parsed))
}

fn print_metrics(label: &str, seed_set: &str, metrics: &RobustMetrics) {
    println!(
        "{label} seeds={} replications={} period={:.6} period_error={:.6} amplitude={:.6} spectral_concentration={:.6} autocorrelation_decay={:.6} molecules={:.6} failure_fraction={:.6} period_cv={:.6} amplitude_cv={:.6} oscillation_error={:.9} fragility={:.9} score={:.9}",
        seed_set,
        metrics.replicates.len(),
        metrics.period,
        metrics.period_error,
        metrics.amplitude,
        metrics.spectral_concentration,
        metrics.autocorrelation_decay,
        metrics.mean_molecules,
        metrics.failure_fraction,
        metrics.period_cv,
        metrics.amplitude_cv,
        metrics.oscillation_error,
        metrics.fragility,
        metrics.scalar_score,
    );
}

fn print_design(label: &str, design: &LogRates) {
    println!("{label}_LOG10 {:?}", design.values());
    println!("{label}_RATES {:?}", design.rates());
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let training_config = EvaluationConfig {
        target_period: args.target_period,
        replications: args.replications,
    };
    let validation_config = EvaluationConfig {
        target_period: args.target_period,
        replications: args.validation_replications,
    };
    let initial_design = LogRates::default();
    let initial_training = evaluate_training(initial_design.values(), &training_config, false)?;
    let initial_validation =
        evaluate_validation(initial_design.values(), &validation_config, true)?;
    print_metrics("INITIAL", "training", &initial_training);
    print_metrics("INITIAL", "validation", &initial_validation);
    print_design("INITIAL", &initial_design);

    let mut selected_design = args
        .design
        .clone()
        .unwrap_or_else(|| initial_design.clone());
    let mut selected_training = initial_training.clone();
    let mut selected_validation = initial_validation.clone();
    let mut convergence = Vec::new();
    let mut pareto: Vec<ParetoPoint> = Vec::new();
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let mut actual_evaluations = 0usize;
    let mut elapsed_seconds = 0.0;

    if matches!(args.mode, RunMode::Single | RunMode::Both | RunMode::All) {
        let outcome = optimize_scalar(
            &training_config,
            &validation_config,
            &ScalarOptions {
                evaluations_per_retry: args.evaluations,
                retries: args.retries,
                workers: args.workers,
                depth: args.depth,
                seed: args.seed,
            },
        )?;
        println!(
            "SO evaluations={} stochastic_runs={} retries={} workers={} seconds={:.6} evaluations_per_second={:.0}",
            outcome.evaluations,
            outcome.evaluations as usize * args.replications,
            outcome.completed_retries,
            args.workers,
            outcome.elapsed.as_secs_f64(),
            outcome.evaluations as f64 / outcome.elapsed.as_secs_f64().max(1.0e-9),
        );
        print_metrics("SO_BEST", "training", &outcome.training);
        print_metrics("SO_BEST", "validation", &outcome.validation);
        print_design("SO_BEST", &outcome.design);
        convergence = outcome
            .improvements
            .iter()
            .map(|sample| MoProgress {
                evaluations: sample.evaluations as usize,
                elapsed_seconds: sample.elapsed_seconds,
                best_quality: -sample.value,
            })
            .collect();
        actual_evaluations = outcome.evaluations as usize;
        elapsed_seconds = outcome.elapsed.as_secs_f64();
        selected_design = outcome.design;
        selected_training = outcome.training;
        selected_validation = outcome.validation;
    }

    if matches!(args.mode, RunMode::Multi | RunMode::Both | RunMode::All) {
        let outcome = optimize_multi(
            &training_config,
            &validation_config,
            &MultiOptions {
                evaluations: args.mo_evaluations,
                popsize: args.popsize,
                workers: args.workers,
                seed: args.seed ^ 0xE703_7ED1_A0B4_28DB,
            },
        )?;
        println!(
            "MO evaluations={} stochastic_runs={} generations={} pareto={} workers={} quality={:.9} seconds={:.6} evaluations_per_second={:.0}",
            outcome.evaluations,
            outcome.evaluations * args.replications,
            outcome.generations,
            outcome.pareto.len(),
            args.workers,
            outcome.quality,
            outcome.elapsed.as_secs_f64(),
            outcome.evaluations as f64 / outcome.elapsed.as_secs_f64().max(1.0e-9),
        );
        print_metrics("MO_REPRESENTATIVE", "training", &outcome.training);
        print_metrics("MO_REPRESENTATIVE", "validation", &outcome.validation);
        print_design("MO_REPRESENTATIVE", &outcome.representative.design);
        for (rank, point) in outcome.pareto.iter().take(12).enumerate() {
            println!(
                "MO_POINT rank={} oscillation_error={:.6} molecule_cost={:.6} fragility={:.6} log10_rates={:?}",
                rank + 1,
                point.objectives[0],
                point.objectives[1],
                point.objectives[2],
                point.design.values(),
            );
        }
        if args.mode == RunMode::Multi {
            selected_design = outcome.representative.design.clone();
            selected_training = outcome.training.clone();
            selected_validation = outcome.validation.clone();
            convergence = outcome.convergence.clone();
            actual_evaluations = outcome.evaluations;
            elapsed_seconds = outcome.elapsed.as_secs_f64();
        }
        pareto = outcome.pareto;
    }

    if matches!(args.mode, RunMode::Qd | RunMode::All) {
        let options = QdOptions {
            evaluations: args.qd_evaluations,
            capacity: args.qd_capacity,
            chunk_size: args.qd_chunk_size,
            workers: args.workers,
            seed: args.seed,
        };
        let outcome = optimize_qd(&training_config, &validation_config, &options)?;
        println!(
            "QD evaluations={} stochastic_runs={} validation_runs={} occupied={} capacity={} coverage={:.6} qd_score={:.9} best_quality={:.9} invalid={} clipped={} same_validation_niche={:.6} workers={} seconds={:.6} validation_seconds={:.6}",
            outcome.evaluations,
            outcome.evaluations * args.replications,
            outcome.validation_evaluations * args.validation_replications,
            outcome.occupied,
            outcome.capacity,
            outcome.occupied as f64 / outcome.capacity as f64,
            outcome.qd_score,
            outcome.representative.quality_train,
            outcome.invalid_evaluations,
            outcome.clipped_descriptors,
            outcome.validation_same_niche_fraction,
            args.workers,
            outcome.elapsed.as_secs_f64(),
            outcome.validation_elapsed.as_secs_f64(),
        );
        print_metrics("QD_BEST", "training", &outcome.training);
        print_metrics("QD_BEST", "validation", &outcome.validation);
        print_design("QD_BEST", &outcome.representative.design);
        let qd_output = if args.mode == RunMode::All {
            args.output.join("qd")
        } else {
            args.output.clone()
        };
        if args.write_output {
            write_qd_artifacts(
                &qd_output,
                &initial_training,
                &initial_validation,
                &outcome,
                &training_config,
                &validation_config,
                &options,
                &command,
            )?;
            println!(
                "QD_ARTIFACTS directory={} archive={}",
                qd_output.display(),
                qd_output.join("qd_archive.csv").display(),
            );
        }
        if args.mode == RunMode::Qd {
            selected_design = outcome.representative.design;
            selected_training = outcome.training;
            selected_validation = outcome.validation;
        }
    }

    if args.mode == RunMode::Simulate {
        selected_training = evaluate_training(selected_design.values(), &training_config, false)?;
        selected_validation =
            evaluate_validation(selected_design.values(), &validation_config, true)?;
        print_metrics("SIMULATED", "training", &selected_training);
        print_metrics("SIMULATED", "validation", &selected_validation);
        print_design("SIMULATED", &selected_design);
    }

    if args.write_output && args.mode != RunMode::Qd {
        write_artifacts(
            &args.output,
            &initial_training,
            &initial_validation,
            &selected_training,
            &selected_validation,
            &convergence,
            &pareto,
        )?;
        println!(
            "ARTIFACTS directory={} report={}",
            args.output.display(),
            args.output.join("report.html").display(),
        );
        if matches!(args.mode, RunMode::Single | RunMode::Multi) {
            write_run_manifest(
                &args.output,
                &args,
                actual_evaluations,
                elapsed_seconds,
                &command,
            )?;
        }
    }
    Ok(())
}

fn effective_workers(requested: usize) -> usize {
    if requested == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        requested
    }
}

fn write_run_manifest(
    directory: &std::path::Path,
    args: &Args,
    actual_evaluations: usize,
    elapsed_seconds: f64,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let formulation = match args.mode {
        RunMode::Single => "scalar",
        RunMode::Multi => "mo",
        _ => return Ok(()),
    };
    let mut artifacts = serde_json::Map::new();
    artifacts.insert("convergence".into(), "convergence.csv".into());
    artifacts.insert("replications".into(), "replications.csv".into());
    artifacts.insert("traces".into(), "traces.csv".into());
    artifacts.insert("report".into(), "report.html".into());
    if args.mode == RunMode::Multi {
        artifacts.insert("pareto".into(), "pareto.csv".into());
    }
    let objectives = if args.mode == RunMode::Multi {
        serde_json::json!([
            {
                "column": "objective_oscillation_error",
                "label": "Oscillation error"
            },
            {
                "column": "objective_molecule_cost",
                "label": "Molecule cost",
                "unit": "molecules"
            },
            {
                "column": "objective_fragility",
                "label": "Stochastic fragility"
            }
        ])
    } else {
        serde_json::json!([])
    };
    let requested = if args.mode == RunMode::Multi {
        args.mo_evaluations
    } else {
        args.evaluations.saturating_mul(args.retries as u64) as usize
    };
    let optimizer_seed = if args.mode == RunMode::Multi {
        args.seed ^ 0xE703_7ED1_A0B4_28DB
    } else {
        args.seed
    };
    let manifest = serde_json::json!({
        "schema_version": 1,
        "tutorial": "rebop-oscillator",
        "formulation": formulation,
        "command": command,
        "root_seed": args.seed,
        "seed": optimizer_seed,
        "workers": effective_workers(args.workers),
        "requested_evaluations": requested,
        "actual_evaluations": actual_evaluations,
        "elapsed_seconds": elapsed_seconds,
        "simulation": {
            "target_period": args.target_period,
            "replications": args.replications,
            "validation_replications": args.validation_replications
        },
        "objectives": objectives,
        "descriptors": [],
        "convergence_metrics": ["best_quality"],
        "artifacts": artifacts
    });
    std::fs::write(
        directory.join("run.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_baseline_design() {
        let text = LogRates::default()
            .values()
            .iter()
            .map(f64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(parse_design(&text).unwrap(), LogRates::default());
    }

    #[test]
    fn rejects_bad_design_text() {
        assert!(parse_design("1,2,3").is_err());
        assert!(parse_design("bad,2,3,4,5,6,7,8,9,10,11,12,13,14,15").is_err());
    }
}
