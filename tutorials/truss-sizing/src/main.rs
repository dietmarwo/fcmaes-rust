use std::error::Error;
use std::path::{Path, PathBuf};

use truss_sizing::artifacts::{
    RunMetadata, write_catalogue, write_mo, write_pilot, write_protocol, write_qd, write_so,
    write_validation,
};
use truss_sizing::config::{Preset, Protocol};
use truss_sizing::decode::{baseline_controls, dimension};
use truss_sizing::evaluate::{Evaluation, evaluate};
use truss_sizing::fem::{Scenario, WorkCounter};
use truss_sizing::ground::GroundStructure;
use truss_sizing::mo::{MoConfig, optimize_mode};
use truss_sizing::pilot::{PilotResult, run_pilot};
use truss_sizing::qd::gate;
use truss_sizing::so::{SoConfig, SoOptimizer, optimize_arm};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Inspect,
    Validation,
    So,
    Pilot,
    Qd,
    Mo,
    All,
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    preset: Preset,
    workers: i32,
    seed: u64,
    output: Option<PathBuf>,
    evaluations: Option<usize>,
    write_output: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::All,
            preset: Preset::Smoke,
            workers: 0,
            seed: 42,
            output: None,
            evaluations: None,
            write_output: true,
        }
    }
}

fn usage() {
    println!(
        "Mixed-variable 2-D truss topology and section sizing\n\
         \n\
         Usage: cargo run --release --locked -- [OPTIONS]\n\
         \n\
         --mode NAME          inspect, validation, so, pilot, qd, mo, or all\n\
         --preset NAME        smoke or publication (smoke)\n\
         --workers N          Candidate threads; 0 uses available CPUs (0)\n\
         --seed N             Root optimizer seed (42)\n\
         --evaluations N      Override the selected optimizer/sample budget\n\
         --output DIR         Artifact root (results/<preset>)\n\
         --no-output          Execute without writing artifacts\n\
         -h, --help           Show this help\n\
         \n\
         QD is conditionally skipped unless the pre-registered descriptor\n\
         pilot passes. Constraints are feasible at values <= 0."
    );
}

fn next(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    arguments
        .next()
        .ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_args() -> Result<Option<Args>, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--mode" => {
                parsed.mode = match next(&mut arguments, "--mode")?.as_str() {
                    "inspect" => Mode::Inspect,
                    "validation" => Mode::Validation,
                    "so" => Mode::So,
                    "pilot" => Mode::Pilot,
                    "qd" => Mode::Qd,
                    "mo" => Mode::Mo,
                    "all" => Mode::All,
                    value => return Err(format!("unknown mode {value}").into()),
                };
            }
            "--preset" => {
                parsed.preset = match next(&mut arguments, "--preset")?.as_str() {
                    "smoke" => Preset::Smoke,
                    "publication" => Preset::Publication,
                    value => return Err(format!("unknown preset {value}").into()),
                };
            }
            "--workers" => parsed.workers = next(&mut arguments, "--workers")?.parse()?,
            "--seed" => parsed.seed = next(&mut arguments, "--seed")?.parse()?,
            "--evaluations" => {
                parsed.evaluations = Some(next(&mut arguments, "--evaluations")?.parse()?)
            }
            "--output" => parsed.output = Some(next(&mut arguments, "--output")?.into()),
            "--no-output" => parsed.write_output = false,
            value => return Err(format!("unknown option {value}").into()),
        }
    }
    if parsed.workers < 0 {
        return Err("--workers must be non-negative".into());
    }
    if parsed.evaluations == Some(0) {
        return Err("--evaluations must be positive".into());
    }
    Ok(Some(parsed))
}

fn command_line() -> String {
    let forwarded = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if forwarded.is_empty() {
        "cargo run --release --locked".to_owned()
    } else {
        format!("cargo run --release --locked -- {forwarded}")
    }
}

fn metadata<'a>(directory: &'a Path, command: &'a str, args: &Args) -> RunMetadata<'a> {
    RunMetadata {
        directory,
        command,
        seed: args.seed,
        workers: args.workers,
    }
}

fn baseline(with_redundancy: bool) -> Result<Evaluation, Box<dyn Error>> {
    let ground = GroundStructure::reference();
    evaluate(
        &baseline_controls(&ground),
        &ground,
        Scenario::TRAINING,
        with_redundancy,
        &WorkCounter::default(),
    )
    .ok_or_else(|| "baseline decoding failed".into())
}

fn inspect() -> Result<(), Box<dyn Error>> {
    let ground = GroundStructure::reference();
    let evaluation = baseline(true)?;
    println!(
        "nodes={} candidates={} dimension={} active={} mass_kg={:.6} feasible={} objective={:.6}",
        ground.nodes.len(),
        ground.members.len(),
        dimension(&ground),
        evaluation.active_count,
        evaluation.mass_kg,
        evaluation.feasible(),
        evaluation.objective
    );
    println!(
        "constraints={:?} failure={:?} redundancy={:?}",
        evaluation.constraints.optimizer_values(),
        evaluation.failure,
        evaluation.redundancy
    );
    Ok(())
}

fn run_validation(args: &Args, output: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let evidence = truss_sizing::fem::triangular_oracle()
        .map_err(|failure| format!("triangular oracle failed: {}", failure.kind()))?;
    let force_error = evidence
        .analytic_forces_n
        .iter()
        .zip(evidence.fem_forces_n)
        .map(|(analytic, fem)| (analytic - fem).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "VALIDATION: max force error={force_error:.3e} N displacement error={:.3e} m",
        (evidence.analytic_displacement_m - evidence.fem_displacement_m).abs()
    );
    if args.write_output {
        write_validation(&metadata(&output.join("validation"), command, args))?;
    }
    Ok(())
}

fn run_so(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let evaluations = args
        .evaluations
        .map_or(protocol.so_evaluations, |value| value as u64);
    let config = SoConfig {
        evaluations_per_arm: evaluations,
        retries: protocol.so_retries.min(evaluations as usize).max(1),
        workers: args.workers as usize,
        seed: args.seed,
    };
    let seed = baseline(false)?;
    let mut results = Vec::new();
    for optimizer in SoOptimizer::ALL {
        let result = optimize_arm(optimizer, &config)?;
        println!(
            "SO {:>13}: mass={:.3} kg feasible={} active={} eval={} wall={:.3}s",
            optimizer.name(),
            result.best.mass_kg,
            result.best.feasible(),
            result.best.active_count,
            result.actual_evaluations,
            result.elapsed.as_secs_f64()
        );
        results.push(result);
    }
    if args.write_output {
        write_so(
            &metadata(&output.join("so"), command, args),
            &seed,
            &results,
        )?;
    }
    Ok(())
}

fn pilot(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<PilotResult, Box<dyn Error>> {
    let per_arm = args.evaluations.unwrap_or(protocol.pilot_per_arm);
    let result = run_pilot(per_arm, args.seed);
    println!(
        "PILOT {}: feasible={}/{} wall={:.3}s",
        result.decision.name(),
        result.feasible,
        result.attempted,
        result.elapsed.as_secs_f64()
    );
    for generator in &result.generators {
        println!(
            "  generator {}: feasible={}/{} per-arm={:?}",
            generator.name,
            generator.feasible(),
            generator.attempted(),
            generator.feasible_by_arm
        );
    }
    for pair in &result.pairs {
        println!(
            "  {} passed={} range={:?}..{:?} clipping={:?}/{:?} rho={:.3} min-coverage={:.1}% retention={:.1}%",
            pair.name,
            pair.passed,
            pair.reachable_min,
            pair.reachable_max,
            pair.lower_clipping,
            pair.upper_clipping,
            pair.spearman,
            100.0 * pair.minimum_arm_coverage,
            100.0 * pair.holdout_niche_retention
        );
    }
    if args.write_output {
        write_pilot(&metadata(&output.join("pilot"), command, args), &result)?;
    }
    Ok(result)
}

fn run_qd(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let pilot_result = pilot(args, protocol, output, command)?;
    let outcome = gate(pilot_result.decision);
    let truss_sizing::qd::QdOutcome::Skipped { reason } = &outcome;
    println!("QD: skipped ({reason})");
    if args.write_output {
        write_qd(&metadata(&output.join("qd"), command, args), &outcome)?;
    }
    Ok(())
}

fn run_mo(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let evaluations = args.evaluations.unwrap_or(protocol.mo_evaluations);
    let population = protocol.mo_population.min(evaluations);
    let population = if population.is_multiple_of(2) {
        population
    } else {
        population - 1
    };
    if population < 4 {
        return Err("MO override needs at least four evaluations".into());
    }
    let result = optimize_mode(&MoConfig {
        evaluations,
        population,
        workers: args.workers,
        seed: args.seed,
    })?;
    println!(
        "MODE: pareto={} eval={} wall={:.3}s",
        result.pareto.len(),
        result.actual_evaluations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_mo(&metadata(&output.join("mo"), command, args), &result)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    if args.mode == Mode::Inspect {
        return inspect();
    }
    let protocol = args.preset.protocol();
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("results").join(args.preset.name()));
    let command = command_line();
    if args.write_output {
        write_protocol(
            &output.join("protocol.json"),
            protocol,
            args.seed,
            args.workers,
        )?;
        write_catalogue(&output.join("sections.csv"))?;
    }
    match args.mode {
        Mode::Inspect => unreachable!(),
        Mode::Validation => run_validation(&args, &output, &command),
        Mode::So => run_so(&args, protocol, &output, &command),
        Mode::Pilot => pilot(&args, protocol, &output, &command).map(drop),
        Mode::Qd => run_qd(&args, protocol, &output, &command),
        Mode::Mo => run_mo(&args, protocol, &output, &command),
        Mode::All => {
            run_validation(&args, &output, &command)?;
            run_so(&args, protocol, &output, &command)?;
            let pilot_result = pilot(&args, protocol, &output, &command)?;
            let outcome = gate(pilot_result.decision);
            if args.write_output {
                write_qd(&metadata(&output.join("qd"), &command, &args), &outcome)?;
            }
            let truss_sizing::qd::QdOutcome::Skipped { reason } = outcome;
            println!("QD: skipped ({reason})");
            run_mo(&args, protocol, &output, &command)
        }
    }
}
