use std::error::Error;
use std::path::{Path, PathBuf};

use network_coverage::artifacts::{
    RunMetadata, write_mo, write_protocol, write_so, write_throughput, write_validation,
};
use network_coverage::config::{Preset, Protocol};
use network_coverage::coverage::{GROUP_WEIGHT_EXPONENT, evaluate};
use network_coverage::instance::{FIXTURES, Instance, generate, read_edge_list};
use network_coverage::mo::{MoConfig, optimize as optimize_mo};
use network_coverage::oracle::{matching_cover, weighted_primal_dual};
use network_coverage::so::{SoConfig, SoObjective, optimize as optimize_so};
use network_coverage::throughput::{measure, select_publication_scale};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Generate,
    Inspect,
    Validation,
    Throughput,
    So,
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
    graph: Option<PathBuf>,
    instance: Option<String>,
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
            graph: None,
            instance: None,
            write_output: true,
        }
    }
}

fn usage() {
    println!(
        "Weighted ordinary-edge and native-group network coverage\n\
         \n\
         Usage: cargo run --release --locked -- [OPTIONS]\n\
         \n\
         --mode NAME          generate, inspect, validation, throughput, so, mo, all\n\
         --preset NAME        smoke or publication (smoke)\n\
         --workers N          Candidate threads; 0 uses available CPUs (0)\n\
         --seed N             Root optimizer seed (42)\n\
         --evaluations N      Override optimizer/sample budgets\n\
         --instance NAME      tiny, small, reference-1k, or reference-4k\n\
         --graph PATH         Read a local whitespace edge list (never vendored)\n\
         --output DIR         Artifact root (results/<preset>)\n\
         --no-output          Execute without writing artifacts\n\
         -h, --help           Show this help"
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
                    "inspect" => Mode::Inspect,
                    "validation" => Mode::Validation,
                    "throughput" => Mode::Throughput,
                    "so" => Mode::So,
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
            "--instance" => parsed.instance = Some(next(&mut arguments, "--instance")?),
            "--graph" => parsed.graph = Some(next(&mut arguments, "--graph")?.into()),
            "--output" => parsed.output = Some(next(&mut arguments, "--output")?.into()),
            "--no-output" => parsed.write_output = false,
            value => return Err(format!("unknown option {value}").into()),
        }
    }
    if parsed.workers < 0 || parsed.evaluations == Some(0) {
        return Err("workers must be non-negative and evaluations positive".into());
    }
    if parsed.graph.is_some() && parsed.instance.is_some() {
        return Err("--graph and --instance are mutually exclusive".into());
    }
    Ok(Some(parsed))
}

fn fixture(name: &str) -> Result<Instance, Box<dyn Error>> {
    let config = FIXTURES
        .iter()
        .find(|config| config.name == name)
        .ok_or_else(|| format!("unknown fixture {name}"))?;
    generate(config)
}

fn generate_fixtures() -> Result<(), Box<dyn Error>> {
    for config in &FIXTURES {
        let instance = generate(config)?;
        let path = Path::new("instances").join(config.name);
        instance.write_csv(&path)?;
        let replay = Instance::read_csv(&path)?;
        if replay != instance {
            return Err(format!("fixture {} failed CSV round trip", config.name).into());
        }
        println!(
            "GENERATED {} nodes={} edges={} groups={}",
            config.name,
            instance.nodes(),
            instance.edges.len(),
            instance.groups.len()
        );
    }
    Ok(())
}

fn command_line() -> String {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if arguments.is_empty() {
        "cargo run --release --locked".to_owned()
    } else {
        format!("cargo run --release --locked -- {arguments}")
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

fn chosen_instance(args: &Args, gated: Option<&str>) -> Result<Instance, Box<dyn Error>> {
    if let Some(path) = &args.graph {
        return read_edge_list(path, args.seed);
    }
    if let Some(name) = &args.instance {
        return fixture(name);
    }
    fixture(gated.unwrap_or(match args.preset {
        Preset::Smoke => "small",
        Preset::Publication => "reference-1k",
    }))
}

fn inspect(instance: &Instance) {
    let empty = evaluate(
        instance,
        &vec![false; instance.nodes()],
        GROUP_WEIGHT_EXPONENT,
    )
    .unwrap();
    let full = evaluate(
        instance,
        &vec![true; instance.nodes()],
        GROUP_WEIGHT_EXPONENT,
    )
    .unwrap();
    println!(
        "instance={} nodes={} edges={} groups={} connected={} cost={:.6} coverage={:.6}",
        instance.metadata.name,
        instance.nodes(),
        instance.edges.len(),
        instance.groups.len(),
        instance.connected(),
        full.cost,
        full.coverage
    );
    if instance.metadata.dropped_self_loops > 0 {
        println!(
            "input_cleaning dropped_self_loops={}",
            instance.metadata.dropped_self_loops
        );
    }
    println!("empty_roi={:.6} full_roi={:.6}", empty.roi, full.roi);
}

fn run_throughput(
    args: &Args,
    protocol: Protocol,
    output: &Path,
    command: &str,
) -> Result<
    (
        Vec<network_coverage::throughput::ThroughputResult>,
        &'static str,
    ),
    Box<dyn Error>,
> {
    let samples = args.evaluations.unwrap_or(protocol.throughput_samples);
    let one = fixture("reference-1k")?;
    let four = fixture("reference-4k")?;
    let rows = vec![
        measure(&one, samples, 1, args.seed),
        measure(&one, samples, args.workers, args.seed),
        measure(&four, samples, 1, args.seed),
        measure(&four, samples, args.workers, args.seed),
    ];
    let selected = if matches!(args.preset, Preset::Smoke) {
        "small"
    } else {
        select_publication_scale(&rows[2])
    };
    for row in &rows {
        println!(
            "THROUGHPUT {:>12} workers={:>2} {:.0} candidates/s",
            row.instance, row.workers, row.candidates_per_second
        );
    }
    println!("SCALE gate selected {selected}");
    if args.write_output {
        write_throughput(
            &metadata(&output.join("throughput"), command, args),
            &rows,
            selected,
        )?;
    }
    Ok((rows, selected))
}

fn run_so(
    args: &Args,
    protocol: Protocol,
    instance: &Instance,
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
    let mut results = Vec::new();
    for objective in [SoObjective::Cardinality, SoObjective::Weighted] {
        let result = optimize_so(instance, objective, &config)?;
        println!(
            "SO {:>14}: selected={} cost={:.6} retained={} raw_objective={:.6} eval={} wall={:.3}s",
            objective.name(),
            result.best.selected_count,
            result.best.cost,
            result.retained_source,
            result.optimizer_objective,
            result.actual_evaluations,
            result.elapsed.as_secs_f64()
        );
        results.push(result);
    }
    if args.write_output {
        write_so(
            &metadata(&output.join("so"), command, args),
            instance,
            &results,
        )?;
    }
    Ok(())
}

fn run_mo(
    args: &Args,
    protocol: Protocol,
    instance: &Instance,
    output: &Path,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    let evaluations = args.evaluations.unwrap_or(protocol.mo_evaluations);
    let population = protocol.mo_population.min(evaluations).max(4);
    let population = if population.is_multiple_of(2) {
        population
    } else {
        population - 1
    };
    let result = optimize_mo(
        instance,
        &MoConfig {
            evaluations,
            population,
            workers: args.workers,
            seed: args.seed,
        },
    )?;
    println!(
        "MO front={} greedy_front={} greedy_prefixes={} eval={} wall={:.3}s",
        result.pareto.len(),
        result.greedy.len(),
        result.greedy_prefixes.len(),
        result.actual_evaluations,
        result.elapsed.as_secs_f64()
    );
    let sensitivity = if args.evaluations.is_none()
        && protocol.mo_sensitivity_evaluations > evaluations
        && instance.metadata.name == "reference-4k"
        && !instance.metadata.external_edges
    {
        let campaign = optimize_mo(
            instance,
            &MoConfig {
                evaluations: protocol.mo_sensitivity_evaluations,
                population,
                workers: args.workers,
                seed: args.seed,
            },
        )?;
        println!(
            "MO high-budget front={} greedy_front={} eval={} wall={:.3}s",
            campaign.pareto.len(),
            campaign.greedy.len(),
            campaign.actual_evaluations,
            campaign.elapsed.as_secs_f64()
        );
        Some(campaign)
    } else {
        None
    };
    if args.write_output {
        write_mo(
            &metadata(&output.join("mo"), command, args),
            instance,
            &result,
            sensitivity.as_ref(),
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    if matches!(args.mode, Mode::Generate) {
        return generate_fixtures();
    }
    let protocol = args.preset.protocol();
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("results").join(args.preset.name()));
    let command = command_line();
    if matches!(args.mode, Mode::Validation | Mode::All) && args.write_output {
        write_validation(&metadata(&output.join("validation"), &command, &args))?;
    }
    let gated = if matches!(args.mode, Mode::Throughput | Mode::All) {
        Some(run_throughput(&args, protocol, &output, &command)?.1)
    } else {
        None
    };
    let instance = chosen_instance(&args, gated)?;
    if matches!(args.mode, Mode::Inspect) {
        inspect(&instance);
    }
    if matches!(args.mode, Mode::So | Mode::All) {
        run_so(&args, protocol, &instance, &output, &command)?;
    }
    if matches!(args.mode, Mode::Mo | Mode::All) {
        run_mo(&args, protocol, &instance, &output, &command)?;
    }
    if args.write_output && matches!(args.mode, Mode::All) {
        write_protocol(
            &output.join("protocol.json"),
            args.preset.name(),
            &instance.metadata.name,
            protocol,
            args.evaluations,
            args.evaluations.is_none()
                && instance.metadata.name == "reference-4k"
                && !instance.metadata.external_edges,
        )?;
    }
    if matches!(args.mode, Mode::Validation) {
        let tiny = fixture("tiny")?;
        let matching = matching_cover(&tiny);
        let weighted = weighted_primal_dual(&tiny);
        println!(
            "VALIDATION tiny matching={}/{} weighted={:.6}/{:.6}",
            matching.selected.iter().filter(|value| **value).count(),
            matching.lower_bound,
            weighted.cost,
            weighted.lower_bound
        );
    }
    Ok(())
}
