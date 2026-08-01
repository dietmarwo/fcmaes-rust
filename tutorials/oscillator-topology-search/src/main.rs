use std::error::Error;
use std::path::{Path, PathBuf};

use oscillator_topology_search::archive::Archive;
use oscillator_topology_search::artifacts::{
    restore_campaign_snapshot, write_campaign, write_comparison, write_pilot, write_qd_skipped,
};
use oscillator_topology_search::config::{Preset, Protocol};
use oscillator_topology_search::grammar::Topology;
use oscillator_topology_search::outer::{self, Campaign, Strategy};
use oscillator_topology_search::{pilot, qd};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    All,
    Reference,
    Campaign,
    Pilot,
    Report,
    Inspect,
}

struct Args {
    mode: Mode,
    preset: Preset,
    strategy: Strategy,
    seed: u64,
    target: Option<usize>,
    inner_retries: Option<usize>,
    workers: Option<i32>,
    evaluations: Option<u64>,
    output: Option<PathBuf>,
    agent_command: Option<String>,
    resume: bool,
    topology: Option<Topology>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::All,
            preset: Preset::Smoke,
            strategy: Strategy::Random,
            seed: 42,
            target: None,
            inner_retries: None,
            workers: None,
            evaluations: None,
            output: None,
            agent_command: None,
            resume: false,
            topology: None,
        }
    }
}

fn usage() {
    println!(
        "Split-brain stochastic oscillator topology search\n\
         \n\
         Usage: cargo run --release --locked -- [OPTIONS]\n\
         \n\
         --mode NAME             all, reference, campaign, pilot, report, inspect (all)\n\
         --preset NAME           smoke or publication (smoke)\n\
         --strategy NAME         random, evolutionary, or agent (random)\n\
         --accepted-candidates N Override equal outer-arm target\n\
         --inner-retries N       Restarts per topology (physical cores)\n\
         --workers N             Concurrent retries; 0 uses logical CPUs (physical cores)\n\
         --evaluations N         Evaluations per inner retry\n\
         --seed N                Root seed (42)\n\
         --agent-command CMD     External JSON proposer; required for agent arm\n\
         --topology VECTOR       Nine digits or comma-separated values for inspect\n\
         --output DIR            Artifact root (results/<preset>)\n\
         --resume                Restore the selected arm's candidates.jsonl\n\
         -h, --help              Show this help"
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
    let mut args = Args::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--mode" => {
                args.mode = match next(&mut arguments, "--mode")?.as_str() {
                    "all" => Mode::All,
                    "reference" => Mode::Reference,
                    "campaign" => Mode::Campaign,
                    "pilot" => Mode::Pilot,
                    "report" => Mode::Report,
                    "inspect" => Mode::Inspect,
                    value => return Err(format!("unknown mode {value}").into()),
                }
            }
            "--preset" => {
                let value = next(&mut arguments, "--preset")?;
                args.preset =
                    Preset::parse(&value).ok_or_else(|| format!("unknown preset {value}"))?;
            }
            "--strategy" => {
                args.strategy = match next(&mut arguments, "--strategy")?.as_str() {
                    "random" => Strategy::Random,
                    "evolutionary" => Strategy::Evolutionary,
                    "agent" => Strategy::Agent,
                    value => return Err(format!("unknown strategy {value}").into()),
                }
            }
            "--accepted-candidates" => {
                args.target = Some(next(&mut arguments, "--accepted-candidates")?.parse()?)
            }
            "--inner-retries" => {
                args.inner_retries = Some(next(&mut arguments, "--inner-retries")?.parse()?)
            }
            "--workers" => args.workers = Some(next(&mut arguments, "--workers")?.parse()?),
            "--evaluations" => {
                args.evaluations = Some(next(&mut arguments, "--evaluations")?.parse()?)
            }
            "--seed" => args.seed = next(&mut arguments, "--seed")?.parse()?,
            "--agent-command" => {
                args.agent_command = Some(next(&mut arguments, "--agent-command")?)
            }
            "--topology" => {
                args.topology = Some(Topology::parse(&next(&mut arguments, "--topology")?)?)
            }
            "--output" => args.output = Some(next(&mut arguments, "--output")?.into()),
            "--resume" => args.resume = true,
            value => return Err(format!("unknown option {value}").into()),
        }
    }
    if args.target == Some(0)
        || args.inner_retries == Some(0)
        || args.evaluations == Some(0)
        || args.workers.is_some_and(|workers| workers < 0)
    {
        return Err(
            "candidate, retry and evaluation budgets must be positive; workers must be non-negative"
                .into(),
        );
    }
    Ok(Some(args))
}

fn command_line() -> String {
    let rest = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if rest.is_empty() {
        "cargo run --release --locked".to_owned()
    } else {
        format!("cargo run --release --locked -- {rest}")
    }
}

fn restored(
    root: &Path,
    strategy: Strategy,
    resume: bool,
    preset: Preset,
    protocol: Protocol,
    seed: u64,
) -> Result<Option<Campaign>, Box<dyn Error>> {
    if resume {
        let campaign = restore_campaign_snapshot(root, strategy, preset, protocol, seed)?;
        if strategy == Strategy::Agent
            && campaign.as_ref().is_some_and(|campaign| {
                campaign
                    .message
                    .as_deref()
                    .is_some_and(|message| message.starts_with("agent circuit breaker opened"))
            })
        {
            return Err(
                "agent resume rejected because its circuit breaker is open; preserve or rename the failed agent directory, fix and preflight the adapter, then start a fresh agent arm"
                    .into(),
            );
        }
        Ok(campaign)
    } else {
        Ok(None)
    }
}

fn write(
    root: &Path,
    campaign: &Campaign,
    args: &Args,
    protocol: Protocol,
    command: &str,
) -> Result<(), Box<dyn Error>> {
    write_campaign(root, campaign, args.preset, protocol, command, args.seed)?;
    let best = campaign
        .archive
        .best()
        .map(|row| {
            format!(
                "{} score={:.6}",
                row.topology_key, row.validation.scalar_score
            )
        })
        .unwrap_or_else(|| "none".to_owned());
    println!(
        "ARM {} status={} accepted={} attempts={} best={best}",
        campaign.strategy.label(),
        campaign.status,
        campaign.accepted_candidates,
        campaign.proposal_attempts,
    );
    Ok(())
}

fn archived_campaign(root: &Path, strategy: Strategy) -> Result<Campaign, Box<dyn Error>> {
    let archive = Archive::read_jsonl(&root.join(strategy.label()).join("candidates.jsonl"))?;
    let accepted_candidates = archive.candidates.len();
    Ok(Campaign {
        strategy,
        status: if accepted_candidates > 0 {
            "complete"
        } else {
            "not-run"
        }
        .to_owned(),
        proposal_attempts: archive
            .candidates
            .iter()
            .map(|candidate| candidate.proposal)
            .max()
            .unwrap_or(0),
        accepted_candidates,
        duplicate_or_invalid_proposals: 0,
        transport_failures: 0,
        input_tokens: 0,
        output_tokens: 0,
        message: None,
        archive,
    })
}

fn report_campaign(
    root: &Path,
    strategy: Strategy,
    preset: Preset,
    protocol: Protocol,
    seed: u64,
    expected_candidates: usize,
) -> Result<Campaign, Box<dyn Error>> {
    let campaign = restore_campaign_snapshot(root, strategy, preset, protocol, seed)?
        .ok_or_else(|| format!("report requires a completed {} arm", strategy.label()))?;
    if campaign.status != "complete" || campaign.accepted_candidates != expected_candidates {
        return Err(format!(
            "report requires {} complete {} candidates for {}, found status={} accepted={}",
            expected_candidates,
            if strategy == Strategy::Reference {
                "reference"
            } else {
                "matched"
            },
            strategy.label(),
            campaign.status,
            campaign.accepted_candidates
        )
        .into());
    }
    Ok(campaign)
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    if args.mode == Mode::Inspect {
        let topology = args
            .topology
            .ok_or("--topology is required with --mode inspect")?;
        println!(
            "topology={} active={} dimension={} motifs={} niche={}",
            topology.key(),
            topology.active_edges(),
            topology.parameter_dimension(),
            topology.motif_flags().join("+"),
            topology.niche_key()
        );
        return Ok(());
    }

    let mut protocol = Protocol::for_preset(args.preset);
    if let Some(evaluations) = args.evaluations {
        protocol.inner_evaluations = evaluations;
    }
    if let Some(inner_retries) = args.inner_retries {
        protocol.inner_retries = inner_retries;
    }
    if let Some(workers) = args.workers {
        protocol.workers = workers;
    }
    let target = args.target.unwrap_or(protocol.candidates_per_arm);
    let root = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("results").join(args.preset.label()));
    std::fs::create_dir_all(&root)?;
    let command = command_line();

    match args.mode {
        Mode::Reference => {
            let campaign = outer::references(&protocol, args.seed);
            write(&root, &campaign, &args, protocol, &command)?;
        }
        Mode::Campaign => {
            let archive = restored(
                &root,
                args.strategy,
                args.resume,
                args.preset,
                protocol,
                args.seed,
            )?;
            let campaign = outer::run(
                args.strategy,
                &protocol,
                args.seed,
                target,
                args.agent_command.as_deref(),
                archive,
            );
            write(&root, &campaign, &args, protocol, &command)?;
        }
        Mode::Pilot => {
            let random = archived_campaign(&root, Strategy::Random)?;
            let evolutionary = archived_campaign(&root, Strategy::Evolutionary)?;
            let agent = archived_campaign(&root, Strategy::Agent)?;
            let mut campaigns = vec![&random, &evolutionary];
            if agent.status == "complete" {
                campaigns.push(&agent);
            }
            let descriptor_pilot = pilot::evaluate(&campaigns);
            write_pilot(&root, &descriptor_pilot)?;
            if descriptor_pilot.status == "accepted" {
                return Err("descriptor pilot passed; run --mode all to execute the QD arm".into());
            }
            write_qd_skipped(&root, &descriptor_pilot, args.preset, &command, args.seed)?;
            println!(
                "PILOT status={} arms={}/{} coverage={:.3}% retention={:.3}% coarse={:.3}% high-rep={:.3}%",
                descriptor_pilot.status,
                descriptor_pilot.observed_arm_count,
                descriptor_pilot.required_arm_count,
                100.0 * descriptor_pilot.minimum_arm_coverage,
                100.0 * descriptor_pilot.holdout_niche_retention,
                100.0 * descriptor_pilot.coarse_holdout_niche_retention,
                100.0 * descriptor_pilot.high_replication_holdout_niche_retention,
            );
        }
        Mode::Report => {
            let random = report_campaign(
                &root,
                Strategy::Random,
                args.preset,
                protocol,
                args.seed,
                target,
            )?;
            let evolutionary = report_campaign(
                &root,
                Strategy::Evolutionary,
                args.preset,
                protocol,
                args.seed,
                target,
            )?;
            let agent = report_campaign(
                &root,
                Strategy::Agent,
                args.preset,
                protocol,
                args.seed,
                target,
            )?;
            let campaigns = [&random, &evolutionary, &agent];
            write_comparison(&root, &campaigns)?;
            println!(
                "REPORT arms=3 accepted_per_proposal_arm={} comparison={}",
                target,
                root.join("comparison.md").display(),
            );
        }
        Mode::All => {
            let reference = outer::references(&protocol, args.seed);
            let random = outer::run(
                Strategy::Random,
                &protocol,
                args.seed,
                target,
                None,
                restored(
                    &root,
                    Strategy::Random,
                    args.resume,
                    args.preset,
                    protocol,
                    args.seed,
                )?,
            );
            let evolutionary = outer::run(
                Strategy::Evolutionary,
                &protocol,
                args.seed,
                target,
                None,
                restored(
                    &root,
                    Strategy::Evolutionary,
                    args.resume,
                    args.preset,
                    protocol,
                    args.seed,
                )?,
            );
            let agent = outer::run(
                Strategy::Agent,
                &protocol,
                args.seed,
                target,
                args.agent_command.as_deref(),
                restored(
                    &root,
                    Strategy::Agent,
                    args.resume,
                    args.preset,
                    protocol,
                    args.seed,
                )?,
            );
            for campaign in [&reference, &random, &evolutionary, &agent] {
                write(&root, campaign, &args, protocol, &command)?;
            }
            let mut pilot_campaigns = vec![&random, &evolutionary];
            if agent.status == "complete" {
                pilot_campaigns.push(&agent);
            }
            let descriptor_pilot = pilot::evaluate(&pilot_campaigns);
            write_pilot(&root, &descriptor_pilot)?;
            let qd_campaign = if descriptor_pilot.status == "accepted" {
                let campaign = qd::run(&protocol, args.seed, target);
                write(&root, &campaign, &args, protocol, &command)?;
                Some(campaign)
            } else {
                write_qd_skipped(&root, &descriptor_pilot, args.preset, &command, args.seed)?;
                None
            };
            let mut comparisons = vec![&reference, &random, &evolutionary, &agent];
            if let Some(qd) = &qd_campaign {
                comparisons.push(qd);
            }
            write_comparison(&root, &comparisons)?;
            println!(
                "PILOT status={} arms={}/{} coverage={:.3}% retention={:.3}% coarse={:.3}% high-rep={:.3}%",
                descriptor_pilot.status,
                descriptor_pilot.observed_arm_count,
                descriptor_pilot.required_arm_count,
                100.0 * descriptor_pilot.minimum_arm_coverage,
                100.0 * descriptor_pilot.holdout_niche_retention,
                100.0 * descriptor_pilot.coarse_holdout_niche_retention,
                100.0 * descriptor_pilot.high_replication_holdout_niche_retention,
            );
        }
        Mode::Inspect => unreachable!(),
    }
    Ok(())
}
