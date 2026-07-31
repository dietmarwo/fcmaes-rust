use std::error::Error;
use std::path::{Path, PathBuf};

use energy_hub_bilevel::annual::{AnnualConfig, optimize_annual};
use energy_hub_bilevel::artifacts::{
    RunMetadata, write_annual, write_landscape, write_mo, write_pilot, write_protocol, write_qd,
    write_qd_skipped, write_scenario_profiles, write_so,
};
use energy_hub_bilevel::config::{Preset, Protocol};
use energy_hub_bilevel::landscape::{convexity_violation, measure_landscape};
use energy_hub_bilevel::mo::{MoConfig, optimize_mode};
use energy_hub_bilevel::pilot::{PilotSummary, run_pilot};
use energy_hub_bilevel::qd::{QdConfig, optimize_qd};
use energy_hub_bilevel::so::{SoConfig, SoOptimizer, optimize_arm};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Landscape,
    So,
    Pilot,
    Qd,
    Mo,
    Annual,
    Scenarios,
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
        "Bilevel energy-hub sizing with fcmaes-core and microlp\n\
         \n\
         Usage: cargo run --release -- [OPTIONS]\n\
         \n\
         --mode NAME          landscape, so, pilot, qd, mo, annual, scenarios, or all\n\
         --preset NAME        smoke or publication (smoke)\n\
         --workers N          Candidate threads; 0 uses available CPUs (0)\n\
         --seed N             Root optimizer seed (42)\n\
         --evaluations N      Override each selected optimizer/pilot budget\n\
         --output DIR         Artifact root (results/<preset>)\n\
         --no-output          Execute without writing artifacts\n\
         -h, --help           Show this help\n\
         \n\
         Representative-day presets exclude hydrogen and seasonal claims.\n\
         The annual arm sizes at six-hour resolution and validates once hourly."
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
                    "landscape" => Mode::Landscape,
                    "so" => Mode::So,
                    "pilot" => Mode::Pilot,
                    "qd" => Mode::Qd,
                    "mo" => Mode::Mo,
                    "annual" => Mode::Annual,
                    "scenarios" => Mode::Scenarios,
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
        preset: args.preset,
        seed: args.seed,
        workers: args.workers,
    }
}

fn run_landscape(args: &Args, output: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let result = measure_landscape(args.preset, args.seed)?;
    println!(
        "LANDSCAPE: convex_violation={:.2e} derivative_disagreement={}/{} LP={} pivots={} wall={:.3}s",
        convexity_violation(&result.rows),
        result.derivative_disagreements,
        result.derivative_probes,
        result.lp_solves,
        result.simplex_iterations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output && args.mode == Mode::All {
        write_landscape(&metadata(&output.join("landscape"), command, args), &result)?;
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
    let mut arms = Vec::new();
    for optimizer in SoOptimizer::ALL {
        let result = optimize_arm(
            optimizer,
            &SoConfig {
                preset: args.preset,
                evaluations_per_arm: evaluations,
                retries: protocol.so_retries.min(evaluations as usize).max(1),
                workers: args.workers as usize,
                seed: args.seed,
            },
        )?;
        println!(
            "SO {:>4}: LCOE={:.5} objective={:.5} feasible={} eval={} LP={} pivots={} wall={:.3}s",
            optimizer.name(),
            result.best.mean_lcoe,
            result.best.objective,
            energy_hub_bilevel::evaluate::feasible(&result.best),
            result.work.candidate_evaluations,
            result.work.lp_solves,
            result.work.simplex_iterations,
            result.elapsed.as_secs_f64()
        );
        arms.push(result);
    }
    if args.write_output {
        write_so(&metadata(&output.join("so"), command, args), &arms)?;
    }
    Ok(())
}

fn run_descriptor_pilot(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<PilotSummary, Box<dyn Error>> {
    let samples = args.evaluations.unwrap_or(protocol.pilot_samples);
    let (rows, summary) = run_pilot(samples, args.preset, protocol.qd_capacity);
    println!(
        "PILOT {}: feasible={}/{} D1 rho={:.3} coverage={:.1}% retention={:.1}%",
        summary.decision.label(),
        summary.feasible_candidates,
        summary.attempted_candidates,
        summary.d1.rank_correlation,
        100.0 * summary.d1.coverage,
        100.0 * summary.d1.holdout_niche_retention
    );
    if args.write_output {
        write_pilot(
            &metadata(&output.join("pilot"), command, args),
            &rows,
            &summary,
        )?;
    }
    Ok(summary)
}

fn run_qd(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
    summary: &PilotSummary,
) -> Result<(), Box<dyn Error>> {
    let qd_directory = output.join("qd");
    let Some(pair) = summary.selected_pair else {
        println!("QD skipped: {}", summary.reason);
        if args.write_output {
            write_qd_skipped(&metadata(&qd_directory, command, args), summary)?;
        }
        return Ok(());
    };
    let result = optimize_qd(&QdConfig {
        preset: args.preset,
        descriptor_pair: pair,
        evaluations: args.evaluations.unwrap_or(protocol.qd_evaluations),
        capacity: protocol.qd_capacity,
        chunk_size: protocol.qd_chunk_size,
        workers: args.workers,
        seed: args.seed,
    })?;
    println!(
        "QD {}: occupied={}/{} coverage={:.1}% eval={} LP={} pivots={} wall={:.3}s",
        pair.label(),
        result.entries.len(),
        result.capacity,
        100.0 * result.entries.len() as f64 / result.capacity as f64,
        result.actual_evaluations,
        result.lp_solves,
        result.simplex_iterations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_qd(&metadata(&qd_directory, command, args), &result)?;
    }
    Ok(())
}

fn run_mo(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let result = optimize_mode(&MoConfig {
        preset: args.preset,
        evaluations: args.evaluations.unwrap_or(protocol.mo_evaluations),
        population: protocol.mo_population,
        workers: args.workers,
        seed: args.seed,
    })?;
    println!(
        "MODE: pareto={} eval={} LP={} pivots={} wall={:.3}s",
        result.pareto.len(),
        result.actual_evaluations,
        result.lp_solves,
        result.simplex_iterations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_mo(&metadata(&output.join("mo"), command, args), &result)?;
    }
    Ok(())
}

fn run_annual(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let result = optimize_annual(&AnnualConfig {
        evaluations: args
            .evaluations
            .map_or(protocol.annual_evaluations, |value| value as u64),
        seed: args.seed,
    })?;
    println!(
        "ANNUAL: hourly cost={:.5} onsite_H2={:.1}% H2_amplitude={:.1} kWh residual={:.2e} eval={} wall={:.3}s",
        result.hourly.delivered_energy_cost,
        100.0 * result.hourly.onsite_hydrogen_fraction,
        result.hourly.dispatch.hydrogen_amplitude_kwh,
        result.hourly.dispatch.max_storage_residual_kwh,
        result.actual_evaluations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_annual(&metadata(&output.join("annual"), command, args), &result)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let protocol = args.preset.protocol();
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("results").join(args.preset.label()));
    let command = command_line();
    if args.write_output {
        write_protocol(
            &output.join("protocol.json"),
            args.preset,
            protocol,
            &command,
            args.seed,
            args.workers,
        )?;
    }
    match args.mode {
        Mode::Landscape => run_landscape(&args, &output, &command)?,
        Mode::So => run_so(&args, protocol, &output, &command)?,
        Mode::Pilot => {
            run_descriptor_pilot(&args, protocol, &output, &command)?;
        }
        Mode::Qd => {
            let summary = run_descriptor_pilot(&args, protocol, &output, &command)?;
            run_qd(&args, protocol, &output, &command, &summary)?;
        }
        Mode::Mo => run_mo(&args, protocol, &output, &command)?,
        Mode::Annual => run_annual(&args, protocol, &output, &command)?,
        Mode::Scenarios => {
            write_scenario_profiles(Path::new("scenarios/generated-publication.csv"))?;
            println!("SCENARIOS: wrote scenarios/generated-publication.csv");
        }
        Mode::All => {
            run_landscape(&args, &output, &command)?;
            run_so(&args, protocol, &output, &command)?;
            let summary = run_descriptor_pilot(&args, protocol, &output, &command)?;
            run_qd(&args, protocol, &output, &command, &summary)?;
            run_mo(&args, protocol, &output, &command)?;
            run_annual(&args, protocol, &output, &command)?;
        }
    }
    Ok(())
}
