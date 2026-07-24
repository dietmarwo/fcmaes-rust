use std::error::Error;
use std::path::{Path, PathBuf};

use dispersion_source_localization::{
    Dataset, Design, Metrics, MultiOptions, QdOptions, ScalarOptions, evaluate_training,
    evaluate_validation, optimize_multi, optimize_qd, optimize_scalar, write_multi_artifacts,
    write_qd_artifacts, write_scalar_artifacts,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    Scalar,
    Multi,
    Qd,
    All,
    Simulate,
}

#[derive(Debug)]
struct Args {
    mode: RunMode,
    evaluations: u64,
    retries: usize,
    workers: usize,
    depth: i32,
    max_eval_fac: f64,
    mo_evaluations: usize,
    popsize: usize,
    qd_evaluations: usize,
    qd_capacity: usize,
    qd_chunk_size: usize,
    seed: u64,
    output: PathBuf,
    write_output: bool,
    design: Option<Design>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: RunMode::All,
            evaluations: 750,
            retries: 16,
            workers: 0,
            depth: 6,
            max_eval_fac: 4.0,
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
        "Native atmospheric source-localization optimization\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --mode NAME             scalar, multi, qd, all, or simulate (all)\n\
         --evaluations N         Initial BiteOpt evaluations per retry (750)\n\
         --retries N             Coordinated advanced retries (16)\n\
         --workers N             fcmaes workers; 0 uses available CPUs (0)\n\
         --depth N               BiteOpt deep populations, 1..36 (6)\n\
         --max-eval-fac X        Final/initial advanced-retry budget factor (4)\n\
         --mo-evaluations N      Total requested MODE evaluations (20000)\n\
         --popsize N             MODE population size (128)\n\
         --qd-evaluations N      Total requested MAP-Elites evaluations (20000)\n\
         --qd-capacity N         Square MAP-Elites archive capacity (400)\n\
         --qd-chunk-size N       Even QD evaluation batch size (128)\n\
         --seed N                Optimizer root seed (42)\n\
         --output DIR            Artifact directory (results)\n\
         --no-output             Do not write result artifacts\n\
         --x CSV                 Twelve parameters for simulate mode\n\
         -h, --help              Show this help\n\
         \n\
         Educational ISC-3-derived model: not for regulatory analysis."
    );
}

fn take_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_design(value: &str) -> Result<Design, Box<dyn Error>> {
    let values = value
        .split(',')
        .map(|field| field.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Design::from_slice(&values)?)
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
                    "scalar" => RunMode::Scalar,
                    "multi" => RunMode::Multi,
                    "qd" => RunMode::Qd,
                    "all" => RunMode::All,
                    "simulate" => RunMode::Simulate,
                    value => return Err(format!("unknown mode: {value}").into()),
                }
            }
            "--evaluations" => {
                parsed.evaluations = take_value(&mut args, "--evaluations")?.parse()?
            }
            "--retries" => parsed.retries = take_value(&mut args, "--retries")?.parse()?,
            "--workers" => parsed.workers = take_value(&mut args, "--workers")?.parse()?,
            "--depth" => parsed.depth = take_value(&mut args, "--depth")?.parse()?,
            "--max-eval-fac" => {
                parsed.max_eval_fac = take_value(&mut args, "--max-eval-fac")?.parse()?
            }
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

fn print_design(label: &str, design: &Design) {
    println!("{label}_X {:?}", design.values());
    for (index, source) in design.sources().iter().enumerate() {
        println!(
            "{label}_SOURCE_{} x_m={:.6} y_m={:.6} emission_g_s={:.9} height_m={:.6}",
            index + 1,
            source.x_m,
            source.y_m,
            source.emission_g_s,
            source.height_m,
        );
    }
    println!(
        "{label}_BIAS wind_direction_deg={:.6} wind_speed_scale={:.6} lateral_scale={:.6} vertical_scale={:.6}",
        design.wind_direction_bias_deg(),
        design.wind_speed_scale(),
        design.lateral_dispersion_scale(),
        design.vertical_dispersion_scale(),
    );
}

fn print_metrics(label: &str, split: &str, metrics: &Metrics) {
    println!(
        "{label} split={split} observations={} mean_huber={:.9} p95_log={:.9} detection_mismatch={:.6} emission_g_s={:.9} source_error_m={:.6} score={:.9}",
        metrics.observations,
        metrics.mean_huber_error,
        metrics.p95_log_error,
        metrics.detection_mismatch_fraction,
        metrics.total_emission_g_s,
        metrics.source_position_error_m,
        metrics.scalar_score,
    );
}

fn output_directory(base: &Path, child: &str, all: bool) -> PathBuf {
    if all {
        base.join(child)
    } else {
        base.to_path_buf()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let dataset = Dataset::synthetic();
    println!(
        "DATA training_observations={} validation_observations={} model=educational_non_regulatory",
        dataset.training().len(),
        dataset.validation().len(),
    );
    print_design("TRUTH", dataset.truth());
    let baseline = Design::baseline();
    print_design("BASELINE", &baseline);
    print_metrics(
        "BASELINE",
        "training",
        &evaluate_training(baseline.values(), &dataset)?,
    );
    print_metrics(
        "BASELINE",
        "validation",
        &evaluate_validation(baseline.values(), &dataset)?,
    );
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let all = args.mode == RunMode::All;

    if matches!(args.mode, RunMode::Scalar | RunMode::All) {
        let options = ScalarOptions {
            evaluations_per_retry: args.evaluations,
            retries: args.retries,
            workers: args.workers,
            depth: args.depth,
            max_eval_fac: args.max_eval_fac,
            seed: args.seed,
        };
        let outcome = optimize_scalar(&dataset, &options)?;
        println!(
            "SCALAR evaluations={} retries={} workers={} seconds={:.6} evaluations_per_second={:.0}",
            outcome.evaluations,
            outcome.completed_retries,
            args.workers,
            outcome.elapsed.as_secs_f64(),
            outcome.evaluations as f64 / outcome.elapsed.as_secs_f64().max(1.0e-9),
        );
        print_metrics("SCALAR_BEST", "training", &outcome.training);
        print_metrics("SCALAR_BEST", "validation", &outcome.validation);
        print_design("SCALAR_BEST", &outcome.design);
        if args.write_output {
            let output = output_directory(&args.output, "scalar", all);
            write_scalar_artifacts(&output, &dataset, &outcome, &options, &command)?;
            println!("SCALAR_ARTIFACTS directory={}", output.display());
        }
    }

    if matches!(args.mode, RunMode::Multi | RunMode::All) {
        let options = MultiOptions {
            evaluations: args.mo_evaluations,
            popsize: args.popsize,
            workers: args.workers,
            seed: args.seed ^ 0xE703_7ED1_A0B4_28DB,
        };
        let outcome = optimize_multi(&dataset, &options)?;
        println!(
            "MO evaluations={} generations={} pareto={} workers={} quality={:.9} seconds={:.6} evaluations_per_second={:.0}",
            outcome.evaluations,
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
        for (rank, point) in outcome.pareto.iter().take(10).enumerate() {
            println!(
                "MO_POINT rank={} mean_huber={:.9} p95_detection={:.9} emission_g_s={:.9}",
                rank + 1,
                point.objectives[0],
                point.objectives[1],
                point.objectives[2],
            );
        }
        if args.write_output {
            let output = output_directory(&args.output, "mo", all);
            write_multi_artifacts(&output, &dataset, &outcome, &options, &command)?;
            println!("MO_ARTIFACTS directory={}", output.display());
        }
    }

    if matches!(args.mode, RunMode::Qd | RunMode::All) {
        let options = QdOptions {
            evaluations: args.qd_evaluations,
            capacity: args.qd_capacity,
            chunk_size: args.qd_chunk_size,
            workers: args.workers,
            seed: args.seed ^ 0xA076_1D64_78BD_642F,
        };
        let outcome = optimize_qd(&dataset, &options)?;
        println!(
            "QD evaluations={} validation_evaluations={} occupied={} capacity={} coverage={:.6} qd_score={:.9} best_quality={:.9} invalid={} clipped={} workers={} seconds={:.6} validation_seconds={:.6}",
            outcome.evaluations,
            outcome.validation_evaluations,
            outcome.occupied,
            outcome.capacity,
            outcome.occupied as f64 / outcome.capacity as f64,
            outcome.qd_score,
            outcome.representative.quality_train,
            outcome.invalid_evaluations,
            outcome.clipped_descriptors,
            args.workers,
            outcome.elapsed.as_secs_f64(),
            outcome.validation_elapsed.as_secs_f64(),
        );
        print_metrics("QD_BEST", "training", &outcome.training);
        print_metrics("QD_BEST", "validation", &outcome.validation);
        print_design("QD_BEST", &outcome.representative.design);
        if args.write_output {
            let output = output_directory(&args.output, "qd", all);
            write_qd_artifacts(&output, &dataset, &outcome, &options, &command)?;
            println!("QD_ARTIFACTS directory={}", output.display());
        }
    }

    if args.mode == RunMode::Simulate {
        let design = args.design.unwrap_or(baseline);
        print_metrics(
            "SIMULATED",
            "training",
            &evaluate_training(design.values(), &dataset)?,
        );
        print_metrics(
            "SIMULATED",
            "validation",
            &evaluate_validation(design.values(), &dataset)?,
        );
        print_design("SIMULATED", &design);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_twelve_value_design() {
        let text = Design::truth()
            .values()
            .iter()
            .map(f64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let parsed = parse_design(&text).unwrap();
        assert_eq!(parsed.values(), Design::truth().values());
    }

    #[test]
    fn rejects_bad_design_text() {
        assert!(parse_design("1,2,3").is_err());
        assert!(parse_design("bad,2,3,4,5,6,7,8,9,10,11,12").is_err());
    }
}
