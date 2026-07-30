//! Reproducible field-service routing campaigns.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use fcmaes_core::Rng;
use field_service_routing::artifacts::{
    Metadata, write_baselines, write_mo, write_pilot, write_qd, write_qd_skipped, write_so,
};
use field_service_routing::baseline;
use field_service_routing::config::Preset;
use field_service_routing::decode::{decode, witness_controls};
use field_service_routing::evaluate::{EvalConfig, evaluate};
use field_service_routing::instance::{DIMENSION, SEEDS, generate, load_primary, write_instances};
use field_service_routing::mo::{MoConfig, optimize as optimize_mo};
use field_service_routing::pilot::{QdDecision, run as run_pilot};
use field_service_routing::qd::{QdConfig, optimize as optimize_qd};
use field_service_routing::scenarios::{
    evaluate_holdout, evaluate_training, holdout, robust_seed_controls, training,
};
use field_service_routing::scorer2::{max_discrepancy, score};
use field_service_routing::so::{SoConfig, SoOptimizer, optimize as optimize_so};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Generate,
    Validate,
    Staircase,
    Baseline,
    Scenarios,
    So,
    Pilot,
    Qd,
    Mo,
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
        "Random-key field-service routing with fcmaes-core\n\
         \n\
         cargo run --release --locked -- [OPTIONS]\n\
         \n\
         --mode NAME       generate, validate, staircase, baseline, scenarios,\n\
                           so, pilot, qd, mo, or all\n\
         --preset NAME     smoke or publication (smoke)\n\
         --workers N       candidate threads; 0 uses available CPUs (0)\n\
         --seed N          root seed (42)\n\
         --evaluations N   override selected optimization/sample budget\n\
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
                    "staircase" => Mode::Staircase,
                    "baseline" => Mode::Baseline,
                    "scenarios" => Mode::Scenarios,
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
    let options = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if options.is_empty() {
        "cargo run --release --locked".to_owned()
    } else {
        format!("cargo run --release --locked -- {options}")
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

fn run_validation(args: &Args, root: &Path) -> Result<(), Box<dyn Error>> {
    let mut rng = Rng::new(args.seed);
    let mut csv = String::from("instance,sample,max_absolute_discrepancy\n");
    let mut maximum: f64 = 0.0;
    let mut mean = 0.0;
    let mut count = 0;
    for (index, seed) in SEEDS.iter().copied().enumerate() {
        let instance = generate(seed, index);
        for sample in 0..100 {
            let controls = (0..DIMENSION).map(|_| rng.uniform01()).collect::<Vec<_>>();
            let decoded = decode(&controls, &instance)?;
            let primary = evaluate(&decoded, &instance, EvalConfig::default());
            let independent = score(&decoded, &instance, EvalConfig::default());
            let discrepancy = max_discrepancy(&primary, independent);
            maximum = maximum.max(discrepancy);
            mean += discrepancy;
            count += 1;
            writeln!(csv, "{},{},{:.17}", instance.name, sample, discrepancy)?;
        }
    }
    mean /= count as f64;
    println!(
        "VALIDATE: supplied_routes={} max_abs={maximum:.3e} mean_abs={mean:.3e}",
        count
    );
    if args.write_output {
        let directory = root.join("validation");
        fs::create_dir_all(&directory)?;
        fs::write(directory.join("validator_discrepancy.csv"), csv)?;
        fs::write(
            directory.join("summary.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "validator":"independent in-repository scorer2",
                "external_validation":false,
                "expected_bit_exact":true,
                "scope":"cost and constraint arithmetic for supplied decoded routes",
                "decoder_covered_separately_by_tests":true,
                "limitation":"shared interpretation errors remain possible",
                "supplied_routes":count,
                "max_absolute_discrepancy":maximum,
                "mean_absolute_discrepancy":mean
            }))?,
        )?;
    }
    Ok(())
}

fn run_staircase(args: &Args, root: &Path) -> Result<(), Box<dyn Error>> {
    let instance = load_primary()?;
    let mut controls = robust_seed_controls(&instance);
    let task = instance.witness_routes[0][2];
    let mut csv = String::from("key,objective,cost,vehicle,position\n");
    let mut states = std::collections::HashSet::new();
    for step in 0..=250 {
        let key = step as f64 / 250.0;
        controls[instance.tasks.len() + task] = key;
        let decoded = decode(&controls, &instance)?;
        let evaluation =
            evaluate_training(&controls, &instance).ok_or("staircase evaluation failed")?;
        let (vehicle, position) = decoded
            .routes
            .iter()
            .find_map(|route| {
                route
                    .tasks
                    .iter()
                    .position(|candidate| *candidate == task)
                    .map(|position| (route.vehicle, position))
            })
            .unwrap();
        states.insert((vehicle, position));
        writeln!(
            csv,
            "{key:.17},{:.17},{:.17},{vehicle},{position}",
            evaluation.objective, evaluation.worst_cost
        )?;
    }
    println!(
        "STAIRCASE: task={task} states={} exact_bound={}",
        states.len(),
        instance.witness_routes[0].len()
    );
    if args.write_output {
        let directory = root.join("staircase");
        fs::create_dir_all(&directory)?;
        fs::write(directory.join("staircase.csv"), csv)?;
    }
    Ok(())
}

fn run_baseline(args: &Args, root: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let moves = args
        .evaluations
        .unwrap_or(args.preset.protocol().baseline_moves);
    let limit = if args.preset == Preset::Smoke {
        3
    } else {
        SEEDS.len()
    };
    let rows = SEEDS[..limit]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, seed)| {
            let instance = generate(seed, index);
            let witness = evaluate(
                &decode(&witness_controls(&instance), &instance).unwrap(),
                &instance,
                EvalConfig::default(),
            );
            let result = baseline::optimize(&instance, moves);
            println!(
                "BASELINE {}: cost={:.2} witness={:.2} fallback={} ops={} wall={:.3}s",
                instance.name,
                result.metrics.cost,
                witness.cost,
                result.construction_fallback,
                result.operations,
                result.elapsed.as_secs_f64()
            );
            (instance, result, witness.cost)
        })
        .collect::<Vec<_>>();
    if args.write_output {
        write_baselines(&metadata(&root.join("baseline"), command, args), &rows)?;
    }
    Ok(())
}

fn run_scenarios(args: &Args, root: &Path) -> Result<(), Box<dyn Error>> {
    let instance = load_primary()?;
    let controls = robust_seed_controls(&instance);
    let mut csv = String::from(
        "set,scenario,cost,distance_km,lateness_s,capacity_excess_kg,shift_excess_s\n",
    );
    for (set, cases) in [
        ("training", training(&instance)),
        ("holdout", holdout(&instance)),
    ] {
        for case in cases {
            let evaluated =
                field_service_routing::scenarios::evaluate_cases(&controls, &[case], true)
                    .ok_or("scenario replay failed")?;
            let metrics = &evaluated.nominal().metrics;
            writeln!(
                csv,
                "{set},{},{:.17},{:.17},{:.17},{:.17},{:.17}",
                evaluated.nominal().name,
                metrics.cost,
                metrics.distance_km,
                metrics.total_lateness_s,
                metrics.capacity_excess_kg,
                metrics.shift_excess_s
            )?;
        }
    }
    let training_eval = evaluate_training(&controls, &instance).unwrap();
    let holdout_eval = evaluate_holdout(&controls, &instance).unwrap();
    println!(
        "SCENARIOS: training_feasible={} holdout_feasible={} train_worst={:.2}",
        training_eval.feasible(),
        holdout_eval.feasible(),
        training_eval.worst_cost
    );
    if args.write_output {
        let directory = root.join("scenarios");
        fs::create_dir_all(&directory)?;
        fs::write(directory.join("robustness.csv"), csv)?;
    }
    Ok(())
}

fn run_scalar(args: &Args, root: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let instance = load_primary()?;
    let seed = evaluate_training(&robust_seed_controls(&instance), &instance)
        .ok_or("structured scalar seed cannot replay")?;
    let protocol = args.preset.protocol();
    let evaluations = args
        .evaluations
        .map_or(protocol.so_evaluations, |value| value as u64);
    let config = SoConfig {
        evaluations,
        retries: protocol.so_retries.min(evaluations as usize).max(1),
        workers: args.workers as usize,
        seed: args.seed,
    };
    let mut arms = Vec::new();
    for optimizer in SoOptimizer::ALL {
        let result = optimize_so(optimizer, &instance, &config)?;
        println!(
            "SO {:>4}: cost={:.2} delta={:+.2} improved={} feasible={} vehicles={} eval={} wall={:.3}s",
            optimizer.name(),
            result.best.worst_cost,
            result.best.worst_cost - seed.worst_cost,
            result.search_found_feasible_improvement,
            result.best.feasible(),
            result.best.nominal().metrics.used_vehicles,
            result.actual_evaluations,
            result.elapsed.as_secs_f64()
        );
        arms.push(result);
    }
    if args.write_output {
        write_so(&metadata(&root.join("so"), command, args), &arms, &seed)?;
    }
    Ok(())
}

fn run_descriptor_pilot(
    args: &Args,
    root: &Path,
    command: &str,
) -> Result<field_service_routing::pilot::PilotSummary, Box<dyn Error>> {
    let samples = args
        .evaluations
        .unwrap_or(args.preset.protocol().pilot_samples);
    let protocol = args.preset.protocol();
    let pilot = run_pilot(&load_primary()?, samples, args.seed, protocol.qd_capacity);
    println!(
        "PILOT {}: feasible={}/{} D1 rho={:.3} coverage={:.1}% feasible_holdout={:.1}% niche_retention={:.1}%",
        pilot.decision.label(),
        pilot.rows.len(),
        pilot.attempted,
        pilot.d1.rank_correlation,
        100.0 * pilot.d1.coverage,
        100.0 * pilot.d1.holdout_feasible_fraction,
        100.0 * pilot.d1.holdout_niche_retention
    );
    if args.write_output {
        write_pilot(&metadata(&root.join("pilot"), command, args), &pilot)?;
    }
    Ok(pilot)
}

fn run_repertoire(args: &Args, root: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let protocol = args.preset.protocol();
    let result = optimize_qd(
        &load_primary()?,
        &QdConfig {
            evaluations: args.evaluations.unwrap_or(protocol.qd_evaluations),
            capacity: protocol.qd_capacity,
            chunk_size: protocol.qd_chunk_size,
            workers: args.workers,
            seed: args.seed,
        },
    )?;
    println!(
        "QD: occupied={}/{} best={:.2} invalid={:.1}% wall={:.3}s",
        result.entries.len(),
        result.capacity,
        result
            .entries
            .iter()
            .map(|entry| entry.quality)
            .fold(f64::INFINITY, f64::min),
        100.0 * result.invalid_evaluations as f64 / result.actual_evaluations.max(1) as f64,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_qd(&metadata(&root.join("qd"), command, args), &result)?;
    }
    Ok(())
}

fn run_multi(args: &Args, root: &Path, command: &str) -> Result<(), Box<dyn Error>> {
    let protocol = args.preset.protocol();
    let result = optimize_mo(
        &load_primary()?,
        &MoConfig {
            evaluations: args.evaluations.unwrap_or(protocol.mo_evaluations),
            population: protocol.mo_population,
            workers: args.workers,
            seed: args.seed,
        },
    )?;
    println!(
        "MO: pareto={} eval={} wall={:.3}s",
        result.pareto.len(),
        result.actual_evaluations,
        result.elapsed.as_secs_f64()
    );
    if args.write_output {
        write_mo(&metadata(&root.join("mo"), command, args), &result)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let preset = match args.preset {
        Preset::Smoke => "smoke",
        Preset::Publication => "publication",
    };
    let root = args.output.clone().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("results")
            .join(preset)
    });
    let command = command();
    match args.mode {
        Mode::Generate => {
            write_instances(&Path::new(env!("CARGO_MANIFEST_DIR")).join("instances"))?
        }
        Mode::Validate => run_validation(&args, &root)?,
        Mode::Staircase => run_staircase(&args, &root)?,
        Mode::Baseline => run_baseline(&args, &root, &command)?,
        Mode::Scenarios => run_scenarios(&args, &root)?,
        Mode::So => run_scalar(&args, &root, &command)?,
        Mode::Pilot => {
            run_descriptor_pilot(&args, &root, &command)?;
        }
        Mode::Qd => run_repertoire(&args, &root, &command)?,
        Mode::Mo => run_multi(&args, &root, &command)?,
        Mode::All => {
            run_validation(&args, &root)?;
            run_staircase(&args, &root)?;
            run_baseline(&args, &root, &command)?;
            run_scenarios(&args, &root)?;
            run_scalar(&args, &root, &command)?;
            let pilot = run_descriptor_pilot(&args, &root, &command)?;
            if pilot.decision == QdDecision::Rejected {
                if args.write_output {
                    write_qd_skipped(
                        &metadata(&root.join("qd"), &command, &args),
                        "pre-registered descriptor pilot rejected D1 and D2",
                    )?;
                }
                println!("QD: skipped because the pre-registered pilot rejected both pairs");
            } else {
                run_repertoire(&args, &root, &command)?;
            }
            run_multi(&args, &root, &command)?;
        }
    }
    Ok(())
}
