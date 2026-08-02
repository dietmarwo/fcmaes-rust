// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! CLI for equal-budget GTOC1 route-search campaigns.

use std::env;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

use gtoc1_pykep::mga::{evaluate_mga, optimize_mga};
use gtoc1_pykep::route_agent::AgentTransport;
use gtoc1_pykep::route_archive::Strategy;
use gtoc1_pykep::route_archive::load_archive;
use gtoc1_pykep::route_campaign::optimize_route;
use gtoc1_pykep::route_campaign::{CampaignConfig, MaximumLevel, run_campaign};
use gtoc1_pykep::route_refine::RefinementConfig;
use gtoc1_pykep::route_refine::refine_route;
use gtoc1_pykep::route_search::{PhysicalDecision, RouteCase, RouteDerivationConfig, RouteVariant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Campaign,
    Inspect,
    MgaInspect,
    MgaScout,
    Scout,
    Refine,
}

fn parse_mode(value: &str) -> Result<Mode, String> {
    match value.to_ascii_lowercase().as_str() {
        "campaign" => Ok(Mode::Campaign),
        "inspect" => Ok(Mode::Inspect),
        "mga-inspect" => Ok(Mode::MgaInspect),
        "mga-scout" => Ok(Mode::MgaScout),
        "scout" => Ok(Mode::Scout),
        "refine" => Ok(Mode::Refine),
        _ => Err(
            "--mode must be campaign, inspect, mga-inspect, mga-scout, scout, or refine".to_owned(),
        ),
    }
}

fn parse_strategy(value: &str) -> Result<Strategy, String> {
    match value.to_ascii_lowercase().as_str() {
        "agent" => Ok(Strategy::Agent),
        "random" => Ok(Strategy::Random),
        "evolutionary" => Ok(Strategy::Evolutionary),
        _ => Err("--strategy must be agent, random, or evolutionary".to_owned()),
    }
}

fn parse_level(value: &str) -> Result<MaximumLevel, String> {
    match value.to_ascii_lowercase().as_str() {
        "l0" => Ok(MaximumLevel::L0),
        "l1" => Ok(MaximumLevel::L1),
        "l2" => Ok(MaximumLevel::L2),
        _ => Err("--max-level must be l0, l1, or l2".to_owned()),
    }
}

fn parse<T: std::str::FromStr>(value: &str, name: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("{name} is invalid"))
}

fn load_base(arguments: &[String]) -> Result<CampaignConfig, Box<dyn Error>> {
    let mut base = CampaignConfig::default();
    if let Some(index) = arguments.iter().position(|argument| argument == "--config") {
        let path = arguments.get(index + 1).ok_or("--config requires a path")?;
        base = serde_json::from_reader(File::open(path)?)?;
    }
    if arguments.iter().any(|argument| argument == "--smoke") {
        base = CampaignConfig::smoke(base.strategy, base.results.clone());
    }
    Ok(base)
}

fn parse_args(arguments: &[String]) -> Result<CampaignConfig, Box<dyn Error>> {
    if arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        print_help();
        std::process::exit(0);
    }
    let mut config = load_base(arguments)?;
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if argument == "--smoke" {
            index += 1;
            continue;
        }
        if argument == "--l1-smoke" {
            config.refinement = RefinementConfig::smoke();
            index += 1;
            continue;
        }
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("{argument} requires a value"))?;
        match argument.as_str() {
            "--config" | "--mode" => {}
            "--strategy" => config.strategy = parse_strategy(value)?,
            "--accepted-candidates" => {
                config.accepted_candidates = parse(value, "--accepted-candidates")?;
            }
            "--max-proposal-attempts" => {
                config.maximum_proposal_attempts = parse(value, "--max-proposal-attempts")?;
            }
            "--bootstrap-candidates" => {
                config.bootstrap_candidates = parse(value, "--bootstrap-candidates")?;
            }
            "--portfolio-size" => {
                config.portfolio_size = parse(value, "--portfolio-size")?;
            }
            "--retries" => config.inner_budget.retries = parse(value, "--retries")?,
            "--evaluations" => {
                config.inner_budget.initial_evaluations = parse(value, "--evaluations")?;
            }
            "--max-eval-fac" => {
                config.inner_budget.maximum_evaluation_factor = parse(value, "--max-eval-fac")?;
            }
            "--workers" => config.inner_budget.workers = parse(value, "--workers")?,
            "--seed" => config.root_seed = parse(value, "--seed")?,
            "--max-level" => config.maximum_level = parse_level(value)?,
            "--promote-every" => config.promotion.every = parse(value, "--promote-every")?,
            "--promote-batch" => config.promotion.batch = parse(value, "--promote-batch")?,
            "--control-promotion-rate" => {
                config.promotion.control_rate = parse(value, "--control-promotion-rate")?;
            }
            "--promote-variants" => {
                config.promotion.variants =
                    value.split(',').map(str::trim).map(str::to_owned).collect();
            }
            "--results" => {
                config.results = PathBuf::from(value);
                config.agent.log_path = config.results.join("agent_log.jsonl");
            }
            "--agent-command-json" => {
                config.agent.command = serde_json::from_str(value)?;
                config.agent.transport = AgentTransport::Command;
            }
            "--agent-replay" => {
                config.agent.transport = AgentTransport::Replay;
                config.agent.replay_path = Some(PathBuf::from(value));
            }
            "--agent-timeout-seconds" => {
                config.agent.timeout_seconds = parse(value, "--agent-timeout-seconds")?;
            }
            "--agent-max-tokens" => {
                config.agent.maximum_tokens = parse(value, "--agent-max-tokens")?;
            }
            "--agent-max-retries" => {
                config.agent.maximum_retries = parse(value, "--agent-max-retries")?;
            }
            _ => return Err(format!("unknown option {argument}").into()),
        }
        index += 2;
    }
    Ok(config)
}

fn print_help() {
    println!(
        "GTOC1 split-brain route search\n\
         \nUsage: cargo run --release -- [OPTIONS]\n\
         \n  --mode MODE                    campaign, inspect, mga-inspect, mga-scout, scout, or refine\n\
         \n  --route ROUTE                  compact EV...A or numeric comma-separated bodies\n\
         \n  --clockwise BITS               one 0/1 Lambert direction bit per leg\n\
         \n  --schedule CSV                 launch followed by one duration per leg\n\
         \n  --from-result PATH#VARIANT     archived L0 record for refine mode\n\
         \n  --config PATH                  load complete JSON configuration\n\
         \n  --smoke                        use the tiny offline CI budget\n\
         \n  --strategy NAME                agent, random, evolutionary\n\
         \n  --accepted-candidates N        unique MGA body-order target\n\
         \n  --max-proposal-attempts N      hard proposal-attempt cap\n\
         \n  --bootstrap-candidates N       score-withholding prefix\n\
         \n  --retries N                    coordinated MGA retries\n\
         \n  --evaluations N                first-retry MGA evaluation cap\n\
         \n  --max-eval-fac N               last-retry cap multiplier\n\
         \n  --workers N                    zero means physical CPU cores\n\
         \n  --seed N                       root seed\n\
         \n  --portfolio-size N             sum the best N MGA-qualified scores\n\
         \n  --max-level LEVEL              campaign requires l0 (MGA-only)\n\
         \n  --l1-smoke                     tiny L1 continuation for protocol tests\n\
         \n  --promote-every N              L0 acceptance cadence for L1\n\
         \n  --promote-batch N              maximum L1 promotions per cadence\n\
         \n  --control-promotion-rate N     lower-ranked promotion probability\n\
         \n  --promote-variants CSV         exact predeclared L1 variant order\n\
         \n  --results PATH                 arm-specific artifact directory\n\
         \n  --agent-command-json JSON      exact command argv array\n\
         \n  --agent-replay PATH            replay a prior agent log\n\
         \n  --agent-timeout-seconds N      hard subprocess deadline\n\
         \n  --agent-max-tokens N           provider output-token cap per call\n\
         \n  --agent-max-retries N          transport retries per provider call"
    );
}

fn option_value<'a>(arguments: &'a [String], name: &str) -> Option<&'a str> {
    arguments
        .iter()
        .position(|argument| argument == name)
        .and_then(|index| arguments.get(index + 1))
        .map(String::as_str)
}

fn parse_route(value: &str) -> Result<Vec<usize>, String> {
    if value.contains(',') || value.contains('-') {
        return value
            .split([',', '-'])
            .map(|field| {
                field
                    .trim()
                    .parse()
                    .map_err(|_| "--route contains an invalid numeric body".to_owned())
            })
            .collect();
    }
    let mut bodies = Vec::new();
    let mut remaining = value;
    while !remaining.is_empty() {
        let (body, consumed) = if remaining.starts_with("Me") {
            (1, 2)
        } else if remaining.starts_with("Ma") {
            (4, 2)
        } else {
            match remaining.as_bytes()[0] {
                b'V' => (2, 1),
                b'E' => (3, 1),
                b'J' => (5, 1),
                b'S' => (6, 1),
                b'A' => (10, 1),
                _ => return Err("--route uses an unknown compact body token".to_owned()),
            }
        };
        bodies.push(body);
        remaining = &remaining[consumed..];
    }
    Ok(bodies)
}

fn parse_variant(arguments: &[String]) -> Result<RouteVariant, Box<dyn Error>> {
    let route = option_value(arguments, "--route").ok_or("--route is required")?;
    let bodies = parse_route(route)?;
    let legs = bodies
        .len()
        .checked_sub(1)
        .ok_or("--route must contain at least two bodies")?;
    let clockwise = option_value(arguments, "--clockwise").map_or_else(
        || Ok(vec![false; legs]),
        |bits| {
            if bits.len() != legs || bits.bytes().any(|bit| !matches!(bit, b'0' | b'1')) {
                return Err("--clockwise must contain one 0/1 bit per leg");
            }
            Ok(bits.bytes().map(|bit| bit == b'1').collect())
        },
    )?;
    Ok(RouteVariant::new(bodies, clockwise))
}

fn physical_decision(
    arguments: &[String],
    route: &RouteCase,
) -> Result<PhysicalDecision, Box<dyn Error>> {
    if let Some(value) = option_value(arguments, "--schedule") {
        let values = value
            .split(',')
            .map(|field| field.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != route.variant().structure.bodies.len() {
            return Err("--schedule requires launch followed by one duration per leg".into());
        }
        return Ok(PhysicalDecision {
            launch_mjd2000: values[0],
            leg_days: values[1..].to_vec(),
        });
    }
    let bounds = route.codec().optimizer_bounds();
    let midpoint = bounds
        .lower()
        .iter()
        .zip(bounds.upper())
        .map(|(&lower, &upper)| 0.5 * (lower + upper))
        .collect::<Vec<_>>();
    Ok(route.codec().decode(&midpoint)?)
}

fn run_inspect(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let route = RouteCase::derive(parse_variant(arguments)?, RouteDerivationConfig::default())?;
    let physical = physical_decision(arguments, &route)?;
    let coordinates = route.codec().encode(&physical)?;
    let evaluation = route.evaluate(&coordinates)?;
    println!(
        "INSPECT variant={} schedule={:?} branches={:?} constraint={:.12e} \
         estimated_score={:.9} fixed_mass_score={:.9}",
        route.variant().variant_key(),
        physical.as_sequence_decision(),
        evaluation.sequence.branches,
        evaluation.sequence.constraint,
        evaluation.sequence.estimated_score,
        evaluation.sequence.score
    );
    Ok(())
}

fn run_scout(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let route = RouteCase::derive(parse_variant(arguments)?, RouteDerivationConfig::default())?;
    let mut config = CampaignConfig::default();
    if let Some(value) = option_value(arguments, "--retries") {
        config.inner_budget.retries = parse(value, "--retries")?;
    }
    if let Some(value) = option_value(arguments, "--evaluations") {
        config.inner_budget.initial_evaluations = parse(value, "--evaluations")?;
    }
    if let Some(value) = option_value(arguments, "--max-eval-fac") {
        config.inner_budget.maximum_evaluation_factor = parse(value, "--max-eval-fac")?;
    }
    if let Some(value) = option_value(arguments, "--workers") {
        config.inner_budget.workers = parse(value, "--workers")?;
    }
    if let Some(value) = option_value(arguments, "--seed") {
        config.root_seed = parse(value, "--seed")?;
    }
    let result = optimize_route(&route, &config.inner_budget, config.root_seed)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn run_mga_inspect(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let route = RouteCase::derive(parse_variant(arguments)?, RouteDerivationConfig::default())?;
    let physical = physical_decision(arguments, &route)?;
    let evaluation = evaluate_mga(&route, &physical)?;
    println!("{}", serde_json::to_string_pretty(&evaluation)?);
    Ok(())
}

fn run_mga_scout(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let route = RouteCase::derive(parse_variant(arguments)?, RouteDerivationConfig::default())?;
    let mut config = CampaignConfig::default();
    if let Some(value) = option_value(arguments, "--retries") {
        config.inner_budget.retries = parse(value, "--retries")?;
    }
    if let Some(value) = option_value(arguments, "--evaluations") {
        config.inner_budget.initial_evaluations = parse(value, "--evaluations")?;
    }
    if let Some(value) = option_value(arguments, "--max-eval-fac") {
        config.inner_budget.maximum_evaluation_factor = parse(value, "--max-eval-fac")?;
    }
    if let Some(value) = option_value(arguments, "--workers") {
        config.inner_budget.workers = parse(value, "--workers")?;
    }
    if let Some(value) = option_value(arguments, "--seed") {
        config.root_seed = parse(value, "--seed")?;
    }
    let initial = option_value(arguments, "--schedule")
        .map(|_| physical_decision(arguments, &route))
        .transpose()?;
    let result = optimize_mga(
        &route,
        &config.inner_budget,
        config.root_seed,
        initial.as_ref(),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn run_refine(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let reference =
        option_value(arguments, "--from-result").ok_or("--from-result PATH#VARIANT is required")?;
    let (path, variant_key) = reference
        .split_once('#')
        .ok_or("--from-result must contain #VARIANT")?;
    let archive = load_archive(PathBuf::from(path).as_path())?;
    let result = archive
        .results
        .iter()
        .find(|result| result.variant_key == variant_key)
        .ok_or("variant key is not present in the archive")?;
    let refinement = if arguments.iter().any(|argument| argument == "--l1-smoke") {
        RefinementConfig::smoke()
    } else {
        RefinementConfig::default()
    };
    let seed = option_value(arguments, "--seed").map_or(Ok(42), |value| parse(value, "--seed"))?;
    let promoted = refine_route(result, &RouteDerivationConfig::default(), &refinement, seed)?;
    println!("{}", serde_json::to_string_pretty(&promoted)?);
    Ok(())
}

fn run_campaign_mode(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let config = parse_args(arguments)?;
    let outcome = run_campaign(&config)?;
    println!(
        "CAMPAIGN strategy={:?} accepted={} attempts={} niches={} \
         requested_evaluations={} actual_evaluations={} worker_seconds={:.6} \
         qualified={} top_n={} portfolio_score_sum={:.9} \
         elapsed_seconds={:.6} results={}",
        config.strategy,
        outcome.archive.len(),
        outcome.manifest.budget.proposal_attempts,
        outcome.manifest.budget.niches,
        outcome.manifest.requested_evaluations,
        outcome.manifest.actual_evaluations,
        outcome.manifest.budget.l0_worker_seconds + outcome.manifest.budget.l1_worker_seconds,
        outcome.manifest.qualified_candidates,
        outcome.manifest.portfolio_size,
        outcome.manifest.portfolio_score_sum,
        outcome.manifest.elapsed_seconds,
        config.results.display()
    );
    if let Some(best) = outcome.archive.best() {
        println!(
            "BEST variant={} mga_score={:.9} \
             fixed_mass_score={:.9} evaluation_found={}",
            best.variant_key,
            best.l0.estimated_score,
            best.l0.fixed_mass_score,
            best.l0.evaluation_found
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
    {
        print_help();
        return Ok(());
    }
    let mode = option_value(&arguments, "--mode").map_or(Ok(Mode::Campaign), parse_mode)?;
    match mode {
        Mode::Campaign => run_campaign_mode(&arguments),
        Mode::Inspect => run_inspect(&arguments),
        Mode::MgaInspect => run_mga_inspect(&arguments),
        Mode::MgaScout => run_mga_scout(&arguments),
        Mode::Scout => run_scout(&arguments),
        Mode::Refine => run_refine(&arguments),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_and_numeric_routes_have_identical_semantics() {
        assert_eq!(
            parse_route("EVEEEJSJA").unwrap(),
            parse_route("3,2,3,3,3,5,6,5,10").unwrap()
        );
        assert_eq!(parse_route("EMeMaA").unwrap(), [3, 1, 4, 10]);
    }

    #[test]
    fn mode_and_direction_errors_are_explicit() {
        assert_eq!(parse_mode("refine").unwrap(), Mode::Refine);
        assert_eq!(parse_mode("mga-scout").unwrap(), Mode::MgaScout);
        assert!(parse_mode("unknown").is_err());
        let arguments = vec![
            "--route".to_owned(),
            "EVA".to_owned(),
            "--clockwise".to_owned(),
            "0".to_owned(),
        ];
        assert!(parse_variant(&arguments).is_err());
    }

    #[test]
    fn provider_budget_overrides_are_explicit() {
        let arguments = vec![
            "--agent-max-tokens".to_owned(),
            "4096".to_owned(),
            "--agent-max-retries".to_owned(),
            "0".to_owned(),
        ];
        let config = parse_args(&arguments).unwrap();
        assert_eq!(config.agent.maximum_tokens, 4096);
        assert_eq!(config.agent.maximum_retries, 0);
    }
}
