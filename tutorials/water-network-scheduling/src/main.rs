//! Reproducible water-network scheduling campaigns.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use water_network_scheduling::artifacts::{
    Metadata, ResolutionRow, write_benchmark, write_mo, write_pilot, write_qd, write_qd_skipped,
    write_resolution, write_scenarios, write_so, write_validation,
};
use water_network_scheduling::bench;
use water_network_scheduling::config::Preset;
use water_network_scheduling::decode::{decode, override_witness_plan, seed_controls};
use water_network_scheduling::driver::simulate;
use water_network_scheduling::energy::validate_energy_oracle;
use water_network_scheduling::evaluate::{ScenarioEvaluation, evaluate_scenarios};
use water_network_scheduling::mo::{MoConfig, optimize as optimize_mo};
use water_network_scheduling::network;
use water_network_scheduling::pilot::{QdDecision, run as run_pilot};
use water_network_scheduling::qd::{QdConfig, optimize as optimize_qd};
use water_network_scheduling::scenarios::{holdout, training};
use water_network_scheduling::so::{SoConfig, SoOptimizer, optimize as optimize_so};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Generate,
    Validate,
    Scenarios,
    Resolution,
    So,
    Pilot,
    Qd,
    Mo,
    Bench,
    All,
}

struct Args {
    mode: Mode,
    preset: Preset,
    workers: i32,
    seed: u64,
    evaluations: Option<usize>,
    output: Option<PathBuf>,
    write_output: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::All,
            preset: Preset::Smoke,
            workers: 0,
            seed: 42,
            evaluations: None,
            output: None,
            write_output: true,
        }
    }
}

fn usage() {
    println!(
        "Water-network pump scheduling with epanet-rs and fcmaes-core\n\
         \n\
         cargo run --release --locked -- [OPTIONS]\n\
         \n\
         --mode NAME       generate, validate, scenarios, resolution, so, pilot,\n\
                           qd, mo, bench, or all\n\
         --preset NAME     smoke or publication (smoke)\n\
         --workers N       candidate threads; 0 uses available CPUs (0)\n\
         --seed N          root seed (42)\n\
         --evaluations N   override the selected campaign budget\n\
         --output DIR      artifact root (results/<preset>)\n\
         --no-output       run without writing artifacts\n\
         -h, --help        show this help"
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
                    "generate" => Mode::Generate,
                    "validate" => Mode::Validate,
                    "scenarios" => Mode::Scenarios,
                    "resolution" => Mode::Resolution,
                    "so" => Mode::So,
                    "pilot" => Mode::Pilot,
                    "qd" => Mode::Qd,
                    "mo" => Mode::Mo,
                    "bench" => Mode::Bench,
                    "all" => Mode::All,
                    value => return Err(format!("unknown mode {value}").into()),
                };
            }
            "--preset" => {
                parsed.preset = match next(&mut arguments, "--preset")?.as_str() {
                    "smoke" => Preset::Smoke,
                    "publication" => Preset::Publication,
                    value => return Err(format!("unknown preset {value}").into()),
                }
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

fn command() -> String {
    let tail = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if tail.is_empty() {
        "cargo run --release --locked".to_owned()
    } else {
        format!("cargo run --release --locked -- {tail}")
    }
}

fn metadata<'a>(directory: &'a Path, command: &'a str, args: &Args) -> Metadata<'a> {
    Metadata {
        directory,
        command,
        preset: match args.preset {
            Preset::Smoke => "smoke",
            Preset::Publication => "publication",
        },
        seed: args.seed,
        workers: args.workers,
    }
}

fn validation(args: &Args, root: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let base = network::load()?;
    let plan = decode(&seed_controls())?;
    let trace = simulate(&base, &plan, &training()[0], 3_600)?;
    let override_trace = simulate(&base, &override_witness_plan(), &training()[0], 1_800)?;
    let energy_replay = trace
        .steps
        .iter()
        .map(|step| step.pump_power_kw.iter().sum::<f64>() * step.interval_s as f64 / 3_600.0)
        .sum::<f64>();
    let energy_oracle = validate_energy_oracle()?;
    let oracle_error = energy_oracle
        .iter()
        .map(|check| check.relative_error)
        .fold(0.0_f64, f64::max);
    let pipe_error = network::analytic_pipe_relative_error()?;
    let continuity = trace
        .steps
        .iter()
        .map(|step| step.continuity_residual_m3_s)
        .fold(0.0, f64::max);
    let override_steps = override_trace
        .steps
        .iter()
        .filter(|step| step.safety_override)
        .count();
    if trace.failed_at_step.is_some()
        || override_trace.failed_at_step.is_some()
        || continuity > 1e-6
        || pipe_error > 1e-5
        || oracle_error > 1e-6
        || override_steps == 0
    {
        return Err(format!(
            "validation failed: failed_at={:?} continuity={continuity:.3e} pipe={pipe_error:.3e} energy={oracle_error:.3e} overrides={override_steps}",
            trace.failed_at_step
        )
        .into());
    }
    println!(
        "VALIDATE steps={} energy={:.3} kWh continuity={continuity:.3e} pipe_rel={pipe_error:.3e} oracle_rel={oracle_error:.3e} overrides={override_steps}",
        trace.steps.len(),
        trace.energy_kwh
    );
    if args.write_output {
        write_validation(
            &metadata(&root.join("validation"), command, args),
            &trace,
            &override_trace,
            energy_replay,
            &energy_oracle,
            pipe_error,
        )?;
    }
    Ok(())
}

fn scenario_campaign(args: &Args, root: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let base = network::load()?;
    let controls = seed_controls();
    let mut rows: Vec<(&str, ScenarioEvaluation)> = Vec::new();
    for scenario in training() {
        let evaluation =
            evaluate_scenarios(&controls, &base, std::slice::from_ref(&scenario), 3_600)?;
        rows.push(("training", evaluation.scenarios[0].clone()));
    }
    for scenario in holdout() {
        let timestep = if scenario.name == "hydraulic_timestep_halved" {
            1_800
        } else {
            3_600
        };
        let evaluation =
            evaluate_scenarios(&controls, &base, std::slice::from_ref(&scenario), timestep)?;
        rows.push(("holdout", evaluation.scenarios[0].clone()));
    }
    println!(
        "SCENARIOS count={} failed={} pda={}",
        rows.len(),
        rows.iter().filter(|(_, row)| row.failed).count(),
        rows.iter()
            .filter(|(_, row)| row.analysis.name() == "PDA")
            .count()
    );
    if args.write_output {
        write_scenarios(&metadata(&root.join("scenarios"), command, args), &rows)?;
    }
    Ok(())
}

fn resolution(args: &Args, root: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let base = network::load()?;
    let scenario = training()[0].clone();
    let mut rows = Vec::new();
    let plans = [
        ("baseline", decode(&seed_controls())?),
        ("override-witness", override_witness_plan()),
    ];
    for (case, plan) in &plans {
        for timestep in [3_600, 1_800, 900, 300] {
            let started = Instant::now();
            let trace = simulate(&base, plan, &scenario, timestep)?;
            rows.push(ResolutionRow {
                case,
                timestep_s: timestep,
                energy_kwh: trace.energy_kwh,
                energy_cost: trace.energy_cost,
                peak_kw_hourly: trace.peak_kw_hourly,
                peak_kw_native: trace.peak_kw_native,
                starts: trace.starts.iter().sum(),
                min_pressure_m: trace
                    .steps
                    .iter()
                    .map(|step| step.min_pressure_m)
                    .fold(f64::INFINITY, f64::min),
                max_velocity_m_s: trace
                    .steps
                    .iter()
                    .map(|step| step.max_velocity_m_s)
                    .fold(0.0, f64::max),
                failed_at_step: trace.failed_at_step,
                override_steps: trace
                    .steps
                    .iter()
                    .filter(|step| step.safety_override)
                    .count(),
                wall_seconds: started.elapsed().as_secs_f64(),
            });
        }
    }
    let baseline = rows.iter().filter(|row| row.case == "baseline");
    let min_energy = baseline
        .clone()
        .map(|row| row.energy_kwh)
        .fold(f64::INFINITY, f64::min);
    let max_energy = baseline.map(|row| row.energy_kwh).fold(0.0, f64::max);
    let stress_overrides = rows
        .iter()
        .filter(|row| row.case == "override-witness")
        .map(|row| row.override_steps)
        .max()
        .unwrap_or(0);
    println!(
        "RESOLUTION energy_spread={:.2}% stress_overrides={} finest_wall={:.3}s",
        100.0 * (max_energy - min_energy) / min_energy,
        stress_overrides,
        rows.last().map_or(0.0, |row| row.wall_seconds)
    );
    if args.write_output {
        write_resolution(&metadata(&root.join("resolution"), command, args), &rows)?;
    }
    Ok(())
}

fn scalar(
    args: &Args,
    root: &Path,
    command: &str,
) -> Result<Vec<water_network_scheduling::so::SoResult>, Box<dyn Error>> {
    let protocol = args.preset.protocol();
    let evaluations = args
        .evaluations
        .map_or(protocol.so_evaluations, |value| value as u64);
    let network = network::load()?;
    let mut arms = Vec::new();
    for optimizer in SoOptimizer::ALL {
        let result = optimize_so(
            optimizer,
            &network,
            &SoConfig {
                evaluations,
                retries: protocol.so_retries,
                workers: args.workers as usize,
                seed: args.seed,
            },
        )?;
        println!(
            "SO arm={} objective={:.3} cost={:.3} violation={:.3e} evals={} wall={:.2}s",
            optimizer.name(),
            result.best.objective,
            result.best.operating_cost,
            result.best.violation,
            result.actual_evaluations,
            result.elapsed.as_secs_f64()
        );
        arms.push(result);
    }
    if args.write_output {
        write_so(&metadata(&root.join("so"), command, args), &network, &arms)?;
    }
    Ok(arms)
}

fn pilot(
    args: &Args,
    root: &Path,
    command: &str,
) -> Result<water_network_scheduling::pilot::PilotSummary, Box<dyn Error>> {
    let samples = args
        .evaluations
        .unwrap_or(args.preset.protocol().pilot_samples);
    let network = network::load()?;
    let result = run_pilot(
        &network,
        samples,
        args.seed,
        args.preset.protocol().qd_capacity,
    );
    println!(
        "PILOT decision={} feasible={}/{} D1_coverage={:.3} D1_rho={:.3}",
        result.decision.label(),
        result.rows.len(),
        result.attempted,
        result.d1.coverage,
        result.d1.rank_correlation
    );
    if args.write_output {
        write_pilot(&metadata(&root.join("pilot"), command, args), &result)?;
    }
    Ok(result)
}

fn qd(args: &Args, root: &Path, command: &str, decision: QdDecision) -> Result<(), Box<dyn Error>> {
    let directory = root.join("qd");
    if decision == QdDecision::Rejected {
        println!("QD skipped: descriptor pilot rejected both emergent pairs");
        if args.write_output {
            write_qd_skipped(
                &metadata(&directory, command, args),
                "descriptor pilot rejected D1 and D2",
            )?;
        }
        return Ok(());
    }
    let protocol = args.preset.protocol();
    let evaluations = args.evaluations.unwrap_or(protocol.qd_evaluations);
    let result = optimize_qd(
        &network::load()?,
        decision,
        &QdConfig {
            evaluations,
            capacity: protocol.qd_capacity,
            chunk_size: if args.preset == Preset::Smoke { 20 } else { 50 },
            workers: args.workers,
            seed: args.seed,
        },
    )?;
    println!(
        "QD occupied={}/{} invalid={} clamped={} wall={:.2}s",
        result.entries.len(),
        result.capacity,
        result.invalid_evaluations,
        result.clamped_evaluations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_qd(&metadata(&directory, command, args), &result)?;
    }
    Ok(())
}

fn multiobjective(args: &Args, root: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let protocol = args.preset.protocol();
    let result = optimize_mo(
        &network::load()?,
        &MoConfig {
            evaluations: args.evaluations.unwrap_or(protocol.mo_evaluations),
            population: protocol.mo_population,
            workers: args.workers,
            seed: args.seed,
        },
    )?;
    println!(
        "MO pareto={} evals={} wall={:.2}s",
        result.pareto.len(),
        result.actual_evaluations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_mo(&metadata(&root.join("mo"), command, args), &result)?;
    }
    Ok(())
}

fn benchmark(args: &Args, root: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let candidates = args
        .evaluations
        .unwrap_or(args.preset.protocol().parallel_benchmark_candidates);
    let rows = bench::run(candidates, args.workers)?;
    for row in &rows {
        println!(
            "BENCH arrangement={} throughput={:.1}/s wall={:.3}s",
            row.arrangement, row.candidates_per_second, row.wall_seconds
        );
    }
    if args.write_output {
        write_benchmark(&metadata(&root.join("benchmark"), command, args), &rows)?;
    }
    Ok(())
}

fn generate(args: &Args, root: &Path) -> Result<(), Box<dyn Error>> {
    if !args.write_output {
        let parsed = network::load()?;
        println!(
            "GENERATE checked deterministic template nodes={} links={} (no output)",
            parsed.nodes.len(),
            parsed.links.len()
        );
        return Ok(());
    }
    let destination = root.join("generated").join("synthetic-zone.inp");
    network::write(&destination)?;
    fs::write(
        root.join("generated").join("generator.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "seed":20260730_u64,
            "builder":"deterministic checked-in INP template",
            "nodes":25,
            "links":39
        }))?,
    )?;
    println!("GENERATE {}", destination.display());
    Ok(())
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    network::assert_thread_safety();
    let command = command();
    let default = match args.preset {
        Preset::Smoke => "smoke",
        Preset::Publication => "publication",
    };
    let root = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("results").join(default));
    match args.mode {
        Mode::Generate => generate(args, &root),
        Mode::Validate => validation(args, &root, &command),
        Mode::Scenarios => scenario_campaign(args, &root, &command),
        Mode::Resolution => resolution(args, &root, &command),
        Mode::So => scalar(args, &root, &command).map(|_| ()),
        Mode::Pilot => pilot(args, &root, &command).map(|_| ()),
        Mode::Qd => {
            let result = pilot(args, &root, &command)?;
            qd(args, &root, &command, result.decision)
        }
        Mode::Mo => multiobjective(args, &root, &command),
        Mode::Bench => benchmark(args, &root, &command),
        Mode::All => {
            generate(args, &root)?;
            validation(args, &root, &command)?;
            scenario_campaign(args, &root, &command)?;
            resolution(args, &root, &command)?;
            scalar(args, &root, &command)?;
            let evidence = pilot(args, &root, &command)?;
            qd(args, &root, &command, evidence.decision)?;
            multiobjective(args, &root, &command)?;
            benchmark(args, &root, &command)
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    run(&args)
}
