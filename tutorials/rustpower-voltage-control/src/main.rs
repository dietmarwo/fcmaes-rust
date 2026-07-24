use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use fcmaes_core::{Rng, parallel_batch};
use rustpower_voltage_control::{
    BASELINE_DESIGN, DIMENSION, LOWER_BOUNDS, OptimizationResult, ParetoPoint, QdOptions,
    QdOutcome, UPPER_BOUNDS, VoltageControlModel, optimize_mode, optimize_qd, write_artifacts,
    write_qd_artifacts,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    Optimize,
    Qd,
    All,
    Simulate,
    Benchmark,
}

#[derive(Debug)]
struct Args {
    mode: RunMode,
    evaluations: usize,
    popsize: usize,
    workers: i32,
    seed: u64,
    qd_evaluations: usize,
    qd_capacity: usize,
    qd_chunk_size: usize,
    output: PathBuf,
    write_output: bool,
    design: Option<Vec<f64>>,
    benchmark_candidates: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: RunMode::Optimize,
            evaluations: 4_096,
            popsize: 128,
            workers: 0,
            seed: 42,
            qd_evaluations: 8_192,
            qd_capacity: 400,
            qd_chunk_size: 128,
            output: PathBuf::from("results"),
            write_output: true,
            design: None,
            benchmark_candidates: 256,
        }
    }
}

fn usage() {
    println!(
        "RustPower robust voltage-control optimization\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --mode NAME              optimize/mo, qd, all, simulate, or benchmark (optimize)\n\
         --evaluations N          Requested MODE evaluations (4096)\n\
         --popsize N              MODE population size (128)\n\
         --qd-evaluations N       Requested MAP-Elites evaluations (8192)\n\
         --qd-capacity N          Square MAP-Elites archive capacity (400)\n\
         --qd-chunk-size N        Even QD evaluation batch size (128)\n\
         --workers N              Candidate-evaluation threads; 0 uses CPUs (0)\n\
         --seed N                 Optimizer/benchmark seed (42)\n\
         --x CSV                  20 design values for simulate mode\n\
         --benchmark-candidates N Fixed candidate count (256)\n\
         --output DIR             CSV/HTML output directory (results)\n\
         --no-output              Do not write output files\n\
         -h, --help               Show this help\n\
         \n\
         RustPower uses the serial pure-Rust RSparse solver. Parallelism is\n\
         exclusively across independent candidates, controlled by fcmaes."
    );
}

fn take_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_design(value: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let values = value
        .split(',')
        .map(|field| field.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != DIMENSION {
        return Err(format!("--x requires {DIMENSION} comma-separated values").into());
    }
    Ok(values)
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
                    "optimize" | "mo" => RunMode::Optimize,
                    "qd" => RunMode::Qd,
                    "all" => RunMode::All,
                    "simulate" => RunMode::Simulate,
                    "benchmark" => RunMode::Benchmark,
                    value => return Err(format!("unknown mode: {value}").into()),
                }
            }
            "--evaluations" => {
                parsed.evaluations = take_value(&mut args, "--evaluations")?.parse()?
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
            "--workers" => parsed.workers = take_value(&mut args, "--workers")?.parse()?,
            "--seed" => parsed.seed = take_value(&mut args, "--seed")?.parse()?,
            "--x" => parsed.design = Some(parse_design(&take_value(&mut args, "--x")?)?),
            "--benchmark-candidates" => {
                parsed.benchmark_candidates =
                    take_value(&mut args, "--benchmark-candidates")?.parse()?
            }
            "--output" => parsed.output = take_value(&mut args, "--output")?.into(),
            "--no-output" => parsed.write_output = false,
            value => return Err(format!("unknown argument: {value}").into()),
        }
    }
    if parsed.workers < 0 {
        return Err("--workers must be non-negative".into());
    }
    if parsed.popsize < 4 {
        return Err("--popsize must be at least four".into());
    }
    if parsed.evaluations == 0 {
        return Err("--evaluations must be positive".into());
    }
    if parsed.qd_evaluations == 0 {
        return Err("--qd-evaluations must be positive".into());
    }
    if parsed.qd_chunk_size < 2 || !parsed.qd_chunk_size.is_multiple_of(2) {
        return Err("--qd-chunk-size must be even and at least two".into());
    }
    let qd_side = (parsed.qd_capacity as f64).sqrt() as usize;
    if qd_side < 2 || qd_side * qd_side != parsed.qd_capacity {
        return Err("--qd-capacity must be a perfect square of at least four".into());
    }
    if parsed.benchmark_candidates == 0 {
        return Err("--benchmark-candidates must be positive".into());
    }
    Ok(Some(parsed))
}

fn print_evaluation(label: &str, evaluation: &rustpower_voltage_control::Evaluation) {
    println!(
        "{label} feasible={} loss_mw={:.9} voltage_deviation_mpu={:.9} lifecycle_cost_musd={:.9} security_index={:.9} quality={:.9} constraints={:?}",
        evaluation.is_feasible(),
        evaluation.objectives[0],
        evaluation.objectives[1],
        evaluation.objectives[2],
        evaluation.objectives[3],
        evaluation.quality(),
        evaluation.constraints,
    );
    let design = &evaluation.design;
    println!(
        "{label}_CONTROLS generator_vm_pu={:?} tap_offsets={:?} capacitor_buses={:?} capacitor_steps={:?} battery_bus={} battery_capacity_mw={:.6} curtailment={:?}",
        design.generator_vm_pu,
        design.tap_offsets,
        design.capacitor_buses(),
        design.capacitor_steps,
        design.battery_bus(),
        design.battery_capacity_mw,
        design.curtailment,
    );
    println!(
        "{label}_X {}",
        design
            .as_vector()
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    for scenario in &evaluation.scenarios {
        println!(
            "{label}_SCENARIO name={} converged={} iterations={} loss_mw={:.6} voltage_rms_mpu={:.6} min_vm={:.6} max_vm={:.6} max_line_percent={:.3} security={:.6} mismatch_pu={:.3e}",
            scenario.name,
            scenario.converged,
            scenario.iterations,
            scenario.line_loss_mw,
            1_000.0 * scenario.rms_voltage_deviation_pu,
            scenario.minimum_voltage_pu,
            scenario.maximum_voltage_pu,
            scenario.maximum_line_loading_percent,
            scenario.security_index,
            scenario.mismatch_pu,
        );
    }
}

fn print_optimization(result: &OptimizationResult, workers: i32) {
    println!(
        "MODE evaluations={} generations={} workers={} pareto={} seconds={:.6} evaluations_per_second={:.1}",
        result.evaluations,
        result.generations,
        workers,
        result.pareto.len(),
        result.elapsed.as_secs_f64(),
        result.evaluations as f64 / result.elapsed.as_secs_f64().max(1.0e-9),
    );
    print_evaluation("MO_REPRESENTATIVE", &result.representative.evaluation);
    for (rank, ParetoPoint { evaluation, .. }) in result.pareto.iter().take(20).enumerate() {
        println!(
            "MO_POINT rank={} loss_mw={:.6} voltage_deviation_mpu={:.6} lifecycle_cost_musd={:.6} security_index={:.6} quality={:.9}",
            rank + 1,
            evaluation.objectives[0],
            evaluation.objectives[1],
            evaluation.objectives[2],
            evaluation.objectives[3],
            evaluation.quality(),
        );
    }
}

fn print_qd(result: &QdOutcome, workers: i32) {
    println!(
        "QD evaluations={} workers={} occupied={} capacity={} coverage={:.6} qd_score={:.9} best_quality={:.9} invalid={} clipped={} seconds={:.6} evaluations_per_second={:.1}",
        result.evaluations,
        workers,
        result.occupied,
        result.capacity,
        result.occupied as f64 / result.capacity as f64,
        result.qd_score,
        result.representative.quality,
        result.invalid_evaluations,
        result.clipped_descriptors,
        result.elapsed.as_secs_f64(),
        result.evaluations as f64 / result.elapsed.as_secs_f64().max(1.0e-9),
    );
    print_evaluation("QD_REPRESENTATIVE", &result.representative.evaluation);
}

fn effective_workers(workers: i32) -> usize {
    if workers == 0 {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    } else {
        workers as usize
    }
}

fn run_benchmark(model: &VoltageControlModel, count: usize, workers: i32, seed: u64) {
    let mut rng = Rng::new(seed);
    let candidates: Vec<Vec<f64>> = (0..count)
        .map(|_| {
            LOWER_BOUNDS
                .iter()
                .zip(UPPER_BOUNDS)
                .map(|(&lower, upper)| lower + rng.uniform01() * (upper - lower))
                .collect()
        })
        .collect();
    let started = Instant::now();
    let evaluations = parallel_batch(&candidates, workers, |x| model.evaluate(x));
    let elapsed = started.elapsed();
    let checksum = evaluations
        .iter()
        .flat_map(|evaluation| evaluation.objectives)
        .sum::<f64>();
    let converged_scenarios = evaluations
        .iter()
        .flat_map(|evaluation| &evaluation.scenarios)
        .filter(|scenario| scenario.converged)
        .count();
    println!(
        "BENCHMARK candidates={} power_flows={} workers={} converged_power_flows={} seconds={:.6} candidates_per_second={:.1} power_flows_per_second={:.1} checksum={:.9}",
        count,
        count * rustpower_voltage_control::SCENARIOS.len(),
        workers,
        converged_scenarios,
        elapsed.as_secs_f64(),
        count as f64 / elapsed.as_secs_f64().max(1.0e-9),
        (count * rustpower_voltage_control::SCENARIOS.len()) as f64
            / elapsed.as_secs_f64().max(1.0e-9),
        checksum,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let model = VoltageControlModel::new()?;
    let baseline = model.evaluate(&BASELINE_DESIGN);
    print_evaluation("REFERENCE", &baseline);

    match args.mode {
        RunMode::Simulate => {
            if let Some(design) = args.design.as_deref() {
                let evaluated = model.evaluate(design);
                print_evaluation("SIMULATED", &evaluated);
            }
        }
        RunMode::Benchmark => {
            run_benchmark(&model, args.benchmark_candidates, args.workers, args.seed);
        }
        RunMode::Optimize | RunMode::All => {
            let result = optimize_mode(
                &model,
                args.evaluations,
                args.popsize,
                args.workers,
                args.seed,
            )?;
            print_optimization(&result, args.workers);
            if args.write_output {
                let directory = if args.mode == RunMode::All {
                    args.output.join("mo")
                } else {
                    args.output.clone()
                };
                write_artifacts(&directory, &baseline, &result)?;
                let manifest = serde_json::json!({
                    "schema_version": 1,
                    "tutorial": "rustpower-voltage-control",
                    "formulation": "mo",
                    "command": &command,
                    "seed": args.seed,
                    "workers": effective_workers(args.workers),
                    "requested_evaluations": args.evaluations,
                    "actual_evaluations": result.evaluations,
                    "elapsed_seconds": result.elapsed.as_secs_f64(),
                    "simulation": {
                        "network": "IEEE-39",
                        "scenarios": rustpower_voltage_control::SCENARIOS
                            .iter()
                            .map(|scenario| scenario.name)
                            .collect::<Vec<_>>(),
                        "solver": "RustPower RSparse Newton-Raphson"
                    },
                    "objectives": [
                        {"column": "objective_loss_mw", "label": "Mean line loss", "unit": "MW"},
                        {"column": "objective_voltage_deviation_mpu", "label": "Voltage RMS deviation", "unit": "mpu"},
                        {"column": "objective_lifecycle_cost_musd", "label": "Lifecycle cost", "unit": "M USD"},
                        {"column": "objective_security_index", "label": "Worst security index"}
                    ],
                    "descriptors": [],
                    "convergence_metrics": ["best_quality", "feasible_population", "pareto_population"],
                    "artifacts": {
                        "pareto": "pareto.csv",
                        "convergence": "convergence.csv",
                        "scenarios": "scenarios.csv",
                        "report": "report.html"
                    }
                });
                std::fs::write(
                    directory.join("run.json"),
                    serde_json::to_string_pretty(&manifest)? + "\n",
                )?;
                println!("MO_OUTPUT {}", directory.display());
            }
            if args.mode == RunMode::All {
                let options = QdOptions {
                    evaluations: args.qd_evaluations,
                    capacity: args.qd_capacity,
                    chunk_size: args.qd_chunk_size,
                    workers: args.workers,
                    seed: args.seed,
                };
                let result = optimize_qd(&model, &options)?;
                print_qd(&result, args.workers);
                if args.write_output {
                    let directory = args.output.join("qd");
                    write_qd_artifacts(&directory, &baseline, &result, &options, &command)?;
                    println!("QD_OUTPUT {}", directory.display());
                }
            }
        }
        RunMode::Qd => {
            let options = QdOptions {
                evaluations: args.qd_evaluations,
                capacity: args.qd_capacity,
                chunk_size: args.qd_chunk_size,
                workers: args.workers,
                seed: args.seed,
            };
            let result = optimize_qd(&model, &options)?;
            print_qd(&result, args.workers);
            if args.write_output {
                write_qd_artifacts(&args.output, &baseline, &result, &options, &command)?;
                println!("QD_OUTPUT {}", args.output.display());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_defaults_and_custom_values() {
        let defaults = Args::default();
        assert_eq!(defaults.mode, RunMode::Optimize);
        assert_eq!(defaults.evaluations, 4_096);
        assert_eq!(defaults.popsize, 128);
        assert_eq!(defaults.workers, 0);
        assert_eq!(defaults.qd_capacity, 400);
        assert_eq!(
            parse_design(&vec!["1"; DIMENSION].join(",")).unwrap().len(),
            DIMENSION
        );
        assert!(parse_design("1,2").is_err());
    }
}
