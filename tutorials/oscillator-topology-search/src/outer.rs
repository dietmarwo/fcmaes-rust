//! Equal-budget outer proposal strategies.

use serde::{Deserialize, Serialize};

use crate::agent;
use crate::archive::Archive;
use crate::config::Protocol;
use crate::grammar::{REFERENCES, Topology, mutate, random_valid};
use crate::inner;
use fcmaes_core::Rng;

/// Number of independently ranked parents retained by the evolutionary arm.
pub const EVOLUTIONARY_ELITES: usize = 8;
/// Every fifth evolutionary proposal is an independent grammar sample.
pub const EVOLUTIONARY_IMMIGRANT_INTERVAL: usize = 5;
/// Consecutive failed adapter calls allowed before the agent arm stops.
pub const MAX_CONSECUTIVE_AGENT_FAILURES: usize = 3;

/// Outer arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Strategy {
    Reference,
    Random,
    Evolutionary,
    Agent,
    Qd,
}

impl Strategy {
    pub fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Random => "random",
            Self::Evolutionary => "evolutionary",
            Self::Agent => "agent",
            Self::Qd => "qd",
        }
    }
}

/// Versioned proposal contract stored in campaign manifests.
pub fn proposal_policy(strategy: Strategy) -> &'static str {
    match strategy {
        Strategy::Reference => "held-out-references-v1",
        Strategy::Random => "independent-grammar-sampling-v1",
        Strategy::Evolutionary => "elite8-mutation+20pct-random-immigrant-v1",
        Strategy::Agent => "external-feedback-candidate-menu-agent-circuit3-v4",
        Strategy::Qd => "descriptor-archive-v1",
    }
}

/// Budget and transport accounting for one arm.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Campaign {
    pub strategy: Strategy,
    pub status: String,
    pub proposal_attempts: usize,
    pub accepted_candidates: usize,
    pub duplicate_or_invalid_proposals: usize,
    pub transport_failures: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub message: Option<String>,
    pub archive: Archive,
}

/// Optimize held-out references separately. These rows cannot count as an
/// outer-arm proposal.
pub fn references(protocol: &Protocol, seed: u64) -> Campaign {
    let mut archive = Archive::default();
    for (index, (_, topology)) in REFERENCES.iter().enumerate() {
        archive.insert(inner::optimize(
            *topology,
            Strategy::Reference.label(),
            index + 1,
            protocol,
            seed.wrapping_add(10_000 + index as u64),
        ));
    }
    Campaign {
        strategy: Strategy::Reference,
        status: "complete".to_owned(),
        proposal_attempts: REFERENCES.len(),
        accepted_candidates: REFERENCES.len(),
        duplicate_or_invalid_proposals: 0,
        transport_failures: 0,
        input_tokens: 0,
        output_tokens: 0,
        message: Some("reference rows are excluded from rediscovery counts".to_owned()),
        archive,
    }
}

fn attempt_rng(seed: u64, strategy: Strategy, attempt: usize) -> Rng {
    let arm = match strategy {
        Strategy::Random => 0x9e37_79b9_7f4a_7c15,
        Strategy::Evolutionary => 0xd1b5_4a32_d192_ed03,
        Strategy::Agent => 0x94d0_49bb_1331_11eb,
        Strategy::Qd => 0xda94_2042_e4dd_58b5,
        Strategy::Reference => 0,
    };
    Rng::new(seed ^ arm ^ (attempt as u64).wrapping_mul(0x5851_f42d_4c95_7f2d))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvolutionaryProposalKind {
    Bootstrap,
    RandomImmigrant,
    EliteMutation(usize),
}

fn evolutionary_proposal(
    archive: &Archive,
    rng: &mut Rng,
    attempt: usize,
) -> (Topology, EvolutionaryProposalKind) {
    if archive.candidates.len() < EVOLUTIONARY_ELITES {
        return (random_valid(rng), EvolutionaryProposalKind::Bootstrap);
    }
    if attempt.is_multiple_of(EVOLUTIONARY_IMMIGRANT_INTERVAL) {
        return (random_valid(rng), EvolutionaryProposalKind::RandomImmigrant);
    }
    let mut elites = archive.candidates.iter().collect::<Vec<_>>();
    elites.sort_by(|left, right| {
        left.validation
            .scalar_score
            .total_cmp(&right.validation.scalar_score)
            .then_with(|| left.topology_key.cmp(&right.topology_key))
    });
    elites.truncate(EVOLUTIONARY_ELITES);
    let rank = rng.int_below(elites.len() as i64) as usize;
    (
        mutate(&elites[rank].topology, rng),
        EvolutionaryProposalKind::EliteMutation(rank),
    )
}

fn offline_proposal(strategy: Strategy, archive: &Archive, seed: u64, attempt: usize) -> Topology {
    let mut rng = attempt_rng(seed, strategy, attempt);
    match strategy {
        Strategy::Random => random_valid(&mut rng),
        Strategy::Evolutionary => evolutionary_proposal(archive, &mut rng, attempt).0,
        _ => unreachable!("offline proposal only supports control arms"),
    }
}

fn push_unique_rejection(rejected: &mut Vec<String>, detail: String) {
    if !rejected.contains(&detail) {
        rejected.push(detail);
    }
}

/// Run a random, evolutionary or external-agent campaign. Resumed archives
/// retain evaluated candidates; duplicates are rejected before the optimizer.
pub fn run(
    strategy: Strategy,
    protocol: &Protocol,
    seed: u64,
    target: usize,
    agent_command: Option<&str>,
    previous: Option<Campaign>,
) -> Campaign {
    assert!(matches!(
        strategy,
        Strategy::Random | Strategy::Evolutionary | Strategy::Agent
    ));
    let previous = previous.unwrap_or_else(|| Campaign {
        strategy,
        status: "not-run".to_owned(),
        proposal_attempts: 0,
        accepted_candidates: 0,
        duplicate_or_invalid_proposals: 0,
        transport_failures: 0,
        input_tokens: 0,
        output_tokens: 0,
        message: None,
        archive: Archive::default(),
    });
    assert_eq!(previous.strategy, strategy);
    if strategy == Strategy::Agent
        && previous
            .message
            .as_deref()
            .is_some_and(|message| message.starts_with("agent circuit breaker opened"))
    {
        return previous;
    }
    let Campaign {
        proposal_attempts: mut attempts,
        duplicate_or_invalid_proposals: mut duplicates,
        transport_failures: mut failures,
        mut input_tokens,
        mut output_tokens,
        mut archive,
        ..
    } = previous;
    if strategy == Strategy::Agent && agent_command.is_none() {
        return Campaign {
            strategy,
            status: "not-run".to_owned(),
            proposal_attempts: attempts,
            accepted_candidates: archive.candidates.len(),
            duplicate_or_invalid_proposals: duplicates,
            transport_failures: failures,
            input_tokens,
            output_tokens,
            message: Some(
                "configure --agent-command, a provider/model and a deliberate token budget"
                    .to_owned(),
            ),
            archive,
        };
    }

    let mut rejected = Vec::new();
    let mut consecutive_agent_failures = 0;
    let mut last_agent_error = None;
    let maximum_attempts = target.saturating_mul(25).max(25);
    while archive.candidates.len() < target && attempts < maximum_attempts {
        attempts += 1;
        let topology = if strategy == Strategy::Agent {
            match agent::propose(
                agent_command.expect("checked above"),
                attempts,
                &archive,
                &rejected,
            ) {
                Ok(outcome) => {
                    consecutive_agent_failures = 0;
                    input_tokens += outcome.input_tokens;
                    output_tokens += outcome.output_tokens;
                    outcome.topology
                }
                Err(error) => {
                    match error.kind() {
                        agent::AgentErrorKind::Transport => failures += 1,
                        agent::AgentErrorKind::InvalidResponse => duplicates += 1,
                    }
                    consecutive_agent_failures += 1;
                    let detail = error.to_string();
                    push_unique_rejection(&mut rejected, format!("agent:{detail}"));
                    last_agent_error = Some(detail);
                    if consecutive_agent_failures >= MAX_CONSECUTIVE_AGENT_FAILURES {
                        break;
                    }
                    continue;
                }
            }
        } else {
            offline_proposal(strategy, &archive, seed, attempts)
        };
        let key = topology.key();
        if archive.contains_key(&key) {
            duplicates += 1;
            push_unique_rejection(&mut rejected, key);
            continue;
        }
        let candidate = inner::optimize(
            topology,
            strategy.label(),
            attempts,
            protocol,
            seed.wrapping_add(attempts as u64 * 1_000_003),
        );
        archive.insert(candidate);
    }
    let complete = archive.candidates.len() >= target;
    Campaign {
        strategy,
        status: if complete { "complete" } else { "incomplete" }.to_owned(),
        proposal_attempts: attempts,
        accepted_candidates: archive.candidates.len(),
        duplicate_or_invalid_proposals: duplicates,
        transport_failures: failures,
        input_tokens,
        output_tokens,
        message: (!complete).then(|| {
            if consecutive_agent_failures >= MAX_CONSECUTIVE_AGENT_FAILURES {
                format!(
                    "agent circuit breaker opened after {consecutive_agent_failures} consecutive failures at attempt {attempts}: {}",
                    last_agent_error.as_deref().unwrap_or("unknown adapter failure")
                )
            } else {
                format!(
                    "stopped after {attempts} attempts with {} accepted",
                    archive.candidates.len()
                )
            }
        }),
        archive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Preset, Protocol};
    use std::collections::BTreeSet;

    #[test]
    fn controls_receive_equal_candidate_and_inner_budgets() {
        let protocol = Protocol {
            inner_retries: 1,
            workers: 1,
            inner_evaluations: 8,
            ..Protocol::for_preset(Preset::Smoke)
        };
        let random = run(Strategy::Random, &protocol, 42, 2, None, None);
        let evolutionary = run(Strategy::Evolutionary, &protocol, 42, 2, None, None);
        assert_eq!(random.accepted_candidates, evolutionary.accepted_candidates);
        assert!(
            random
                .archive
                .candidates
                .iter()
                .chain(&evolutionary.archive.candidates)
                .all(|candidate| candidate.requested_evaluations == 8)
        );
    }

    #[test]
    fn absent_agent_is_an_explicit_status() {
        let campaign = run(
            Strategy::Agent,
            &Protocol::for_preset(Preset::Smoke),
            42,
            3,
            None,
            None,
        );
        assert_eq!(campaign.status, "not-run");
        assert!(campaign.archive.candidates.is_empty());
    }

    #[test]
    fn agent_circuit_breaker_stops_and_cannot_be_resumed_silently() {
        let protocol = Protocol {
            inner_retries: 1,
            workers: 1,
            inner_evaluations: 1,
            ..Protocol::for_preset(Preset::Smoke)
        };
        let failed = run(Strategy::Agent, &protocol, 42, 200, Some("false"), None);
        assert_eq!(failed.status, "incomplete");
        assert_eq!(failed.proposal_attempts, MAX_CONSECUTIVE_AGENT_FAILURES);
        assert_eq!(failed.transport_failures, MAX_CONSECUTIVE_AGENT_FAILURES);
        assert!(
            failed
                .message
                .as_deref()
                .unwrap()
                .contains("agent circuit breaker opened")
        );
        let replay = run(
            Strategy::Agent,
            &protocol,
            42,
            200,
            Some("false"),
            Some(failed.clone()),
        );
        assert_eq!(replay.proposal_attempts, failed.proposal_attempts);
        assert_eq!(replay.transport_failures, failed.transport_failures);
    }

    #[test]
    fn resumed_campaign_preserves_attempt_and_failure_accounting() {
        let protocol = Protocol {
            inner_retries: 1,
            workers: 1,
            inner_evaluations: 1,
            ..Protocol::for_preset(Preset::Smoke)
        };
        let first = run(Strategy::Random, &protocol, 42, 3, None, None);
        let first_attempts = first.proposal_attempts;
        let first_duplicates = first.duplicate_or_invalid_proposals;
        let resumed = run(Strategy::Random, &protocol, 42, 6, None, Some(first));
        assert_eq!(resumed.accepted_candidates, 6);
        assert!(resumed.proposal_attempts >= first_attempts + 3);
        assert!(resumed.duplicate_or_invalid_proposals >= first_duplicates);
    }

    #[test]
    fn evolutionary_search_uses_multiple_elites_and_random_immigrants() {
        let protocol = Protocol {
            inner_retries: 1,
            workers: 1,
            inner_evaluations: 1,
            training_replications: 1,
            validation_replications: 1,
            bite_depth: 1,
            ..Protocol::for_preset(Preset::Smoke)
        };
        let bootstrap = run(
            Strategy::Random,
            &protocol,
            42,
            EVOLUTIONARY_ELITES,
            None,
            None,
        );
        assert_eq!(bootstrap.accepted_candidates, EVOLUTIONARY_ELITES);

        let mut parent_ranks = BTreeSet::new();
        let mut immigrants = 0;
        for attempt in EVOLUTIONARY_ELITES + 1..=100 {
            let mut rng = attempt_rng(42, Strategy::Evolutionary, attempt);
            let (_, kind) = evolutionary_proposal(&bootstrap.archive, &mut rng, attempt);
            match kind {
                EvolutionaryProposalKind::RandomImmigrant => immigrants += 1,
                EvolutionaryProposalKind::EliteMutation(rank) => {
                    parent_ranks.insert(rank);
                }
                EvolutionaryProposalKind::Bootstrap => {
                    panic!("a full elite pool must not return to bootstrap")
                }
            }
        }
        assert!(immigrants > 0);
        assert!(parent_ranks.len() > 1);
    }

    #[test]
    fn evolutionary_search_grows_beyond_one_parent_neighborhood() {
        let protocol = Protocol {
            inner_retries: 1,
            workers: 1,
            inner_evaluations: 1,
            training_replications: 1,
            validation_replications: 1,
            bite_depth: 1,
            ..Protocol::for_preset(Preset::Smoke)
        };
        let campaign = run(Strategy::Evolutionary, &protocol, 42, 30, None, None);
        assert_eq!(campaign.status, "complete");
        assert_eq!(campaign.accepted_candidates, 30);
    }
}
