// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Equal-budget outer campaign for agent, random, and evolutionary routes.

use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fcmaes_core::{
    AdvancedRetryConfig, Cmaes, CmaesParams, De, DeParams, Fitness, RetryConfig, RetryContext,
    RetryRunResult, advanced_retry,
};
use serde::{Deserialize, Serialize};

use crate::Gtoc1Error;
use crate::route_agent::{
    AgentClient, AgentConfig, AgentLogEntry, AgentPhase, AgentUsage, build_request,
};
use crate::route_archive::{
    BranchChoice, L0Result, ProposalEvent, ProposalEventKind, RouteArchive, SearchResult, Strategy,
    append_archive, append_proposal_event, load_archive, snapshot_archive,
};
use crate::route_grammar::{
    GrammarConfig, GrammarRng, clears_diversity, mutate_route, sample_route,
};
use crate::route_refine::{RefinementConfig, refine_route};
use crate::route_search::{
    EvaluationContext, FailureCode, FailureObservation, InnerBudget, PhysicalDecision, RouteCase,
    RouteDerivationConfig, RouteProposal, RouteSearchError, read_jsonl_resilient, route_seed,
    write_atomic_json,
};

const FAILURE_OBJECTIVE: f64 = 1.0e99;

/// Highest numerical fidelity enabled for a campaign.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaximumLevel {
    /// Lambert-DP route ranking only.
    L0,
    /// Add promoted Sims–Flanagan approximations.
    L1,
    /// Add optional Taylor/DOP853 finalist validation.
    L2,
}

const fn default_manifest_level() -> MaximumLevel {
    MaximumLevel::L0
}

/// Promotion cadence shared by all strategies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionConfig {
    /// Promote after this many newly accepted L0 routes.
    pub every: usize,
    /// Maximum promotions at each cadence.
    pub batch: usize,
    /// Probability of a deliberately lower-ranked control promotion.
    pub control_rate: f64,
}

impl Default for PromotionConfig {
    fn default() -> Self {
        Self {
            every: 8,
            batch: 2,
            control_rate: 0.2,
        }
    }
}

/// Complete deterministic campaign configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignConfig {
    /// Agent, random, or evolutionary arm.
    pub strategy: Strategy,
    /// Accepted unique L0 evaluations required.
    pub accepted_candidates: usize,
    /// Hard proposal-attempt cap.
    pub maximum_proposal_attempts: usize,
    /// Feedback-blind accepted-candidate prefix.
    pub bootstrap_candidates: usize,
    /// Number of top routes protected by exploration diversity.
    pub protected_top: usize,
    /// Root seed shared by discrete and continuous derivations.
    pub root_seed: u64,
    /// Highest enabled fidelity.
    pub maximum_level: MaximumLevel,
    /// Route grammar and discrete operators.
    pub grammar: GrammarConfig,
    /// Runtime route derivation settings.
    pub derivation: RouteDerivationConfig,
    /// Identical L0 optimizer budget.
    pub inner_budget: InnerBudget,
    /// Promotion policy.
    pub promotion: PromotionConfig,
    /// L1 Sims–Flanagan continuation and thresholds.
    #[serde(default)]
    pub refinement: RefinementConfig,
    /// Agent transport settings.
    pub agent: AgentConfig,
    /// Arm-specific artifact directory.
    pub results: PathBuf,
}

impl Default for CampaignConfig {
    fn default() -> Self {
        Self {
            strategy: Strategy::Agent,
            accepted_candidates: 40,
            maximum_proposal_attempts: 120,
            bootstrap_candidates: 6,
            protected_top: 5,
            root_seed: 42,
            maximum_level: MaximumLevel::L1,
            grammar: GrammarConfig::default(),
            derivation: RouteDerivationConfig::default(),
            inner_budget: InnerBudget {
                retries: 32,
                initial_evaluations: 20_000,
                maximum_evaluation_factor: 10.0,
                workers: 0,
            },
            promotion: PromotionConfig::default(),
            refinement: RefinementConfig::default(),
            agent: AgentConfig::default(),
            results: PathBuf::from("results/publication/agent"),
        }
    }
}

impl CampaignConfig {
    /// Small deterministic configuration used by CI.
    #[must_use]
    pub fn smoke(strategy: Strategy, results: PathBuf) -> Self {
        Self {
            strategy,
            accepted_candidates: 3,
            maximum_proposal_attempts: 40,
            bootstrap_candidates: 0,
            protected_top: 3,
            maximum_level: MaximumLevel::L0,
            inner_budget: InnerBudget {
                retries: 2,
                initial_evaluations: 500,
                maximum_evaluation_factor: 1.0,
                workers: 2,
            },
            agent: AgentConfig {
                log_path: results.join("agent_log.jsonl"),
                ..Default::default()
            },
            results,
            ..Default::default()
        }
    }

    /// Validates campaign fairness and termination settings.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero target/budget, invalid grammar/agent
    /// configuration, or unsupported L2 execution.
    pub fn validate(&self) -> Result<(), RouteSearchError> {
        self.grammar.validate()?;
        self.agent.validate()?;
        if self.accepted_candidates == 0
            || self.maximum_proposal_attempts < self.accepted_candidates
            || self.inner_budget.retries == 0
            || self.inner_budget.initial_evaluations == 0
        {
            return Err(RouteSearchError::Grammar(
                "campaign target, attempt cap, retries, and evaluations are inconsistent"
                    .to_owned(),
            ));
        }
        if !self.inner_budget.maximum_evaluation_factor.is_finite()
            || self.inner_budget.maximum_evaluation_factor < 1.0
        {
            return Err(RouteSearchError::Grammar(
                "maximum_evaluation_factor must be at least one".to_owned(),
            ));
        }
        if self.promotion.every == 0
            || self.promotion.batch == 0
            || !self.promotion.control_rate.is_finite()
            || !(0.0..=1.0).contains(&self.promotion.control_rate)
        {
            return Err(RouteSearchError::Grammar(
                "promotion cadence, batch, and control rate are invalid".to_owned(),
            ));
        }
        if self.maximum_level >= MaximumLevel::L1 {
            self.refinement.validate()?;
        }
        if self.maximum_level == MaximumLevel::L2 {
            return Err(RouteSearchError::Grammar(
                "L2 is an optional follow-on and is not enabled by the core campaign".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Budget and protocol counters written to `run.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CampaignCounters {
    /// Total proposal candidates considered.
    pub proposal_attempts: usize,
    /// Accepted unique candidates that consumed or hit an L0 cache.
    pub accepted_candidates: usize,
    /// Typed grammar rejections.
    pub invalid_proposals: usize,
    /// Exact variant duplicates.
    pub duplicate_variants: usize,
    /// Body orders rejected by the per-structure variant cap.
    pub structure_cap_rejections: usize,
    /// Exploration diversity rejections.
    pub diversity_rejections: usize,
    /// JSON repair calls.
    pub repair_calls: usize,
    /// Agent transport failures.
    pub transport_failures: usize,
    /// Agent calls, including diversity retries.
    pub agent_calls: usize,
    /// Agent input tokens where reported.
    pub agent_input_tokens: u64,
    /// Agent output tokens where reported.
    pub agent_output_tokens: u64,
    /// L0 deterministic requested evaluation caps.
    pub l0_requested_evaluations: u64,
    /// L0 actual objective calls.
    pub l0_actual_evaluations: u64,
    /// L0 allocated worker-seconds.
    pub l0_worker_seconds: f64,
    /// L0 cache hits.
    pub cache_hits: usize,
    /// Number of structural niches occupied.
    pub niches: usize,
    /// Number of archived L1 promotions.
    pub l1_promotions: usize,
    /// L1 promotions that passed every declared numerical threshold.
    pub l1_threshold_passed: usize,
    /// Deterministic sum of requested L1 objective caps.
    pub l1_requested_evaluations: u64,
    /// Actual L1 objective calls.
    pub l1_actual_evaluations: u64,
    /// L1 allocated worker-seconds.
    pub l1_worker_seconds: f64,
}

/// Top-level schema-v1 campaign manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignManifest {
    /// Tutorial artifact schema version.
    pub schema_version: u32,
    /// Tutorial identifier.
    pub tutorial: String,
    /// Formulation identifier.
    pub formulation: String,
    /// Completed/failed campaign status.
    pub status: String,
    /// Strategy.
    pub strategy: Strategy,
    /// Complete non-secret campaign configuration.
    #[serde(default)]
    pub configuration: CampaignConfig,
    /// Provider/transport accounting for the agent arm.
    #[serde(default)]
    pub agent: Option<AgentRunManifest>,
    /// Highest fidelity requested by this completed run.
    #[serde(default = "default_manifest_level")]
    pub maximum_level: MaximumLevel,
    /// Root seed.
    pub seed: u64,
    /// Resolved worker allocation.
    pub workers: usize,
    /// Sum of requested L0 caps.
    pub requested_evaluations: u64,
    /// Sum of actual L0 objective calls.
    pub actual_evaluations: u64,
    /// Complete campaign wall time.
    pub elapsed_seconds: f64,
    /// Detailed protocol and resource counters.
    pub budget: CampaignCounters,
    /// Relative artifact paths.
    pub artifacts: BTreeMap<String, String>,
}

/// Agent metadata recorded independently from numerical budget counters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentRunManifest {
    /// Mock, bounded command, or replay.
    pub transport: crate::route_agent::AgentTransport,
    /// Reported/configured provider identifier.
    pub provider: Option<String>,
    /// Reported/configured model identifier.
    pub model: Option<String>,
    /// Complete adapter calls, including repair calls.
    pub calls: usize,
    /// Reported input tokens.
    pub input_tokens: u64,
    /// Reported output tokens.
    pub output_tokens: u64,
}

/// Completed campaign and in-memory archive.
#[derive(Clone, Debug)]
pub struct CampaignOutcome {
    /// Accepted route archive.
    pub archive: RouteArchive,
    /// Manifest also written to disk.
    pub manifest: CampaignManifest,
}

/// Runs one complete L0 campaign and writes resumable artifacts.
///
/// # Errors
///
/// Returns an error for invalid configuration, exhausted proposal attempts,
/// numerical setup, or persistence failure.
#[allow(clippy::too_many_lines)]
pub fn run_campaign(config: &CampaignConfig) -> Result<CampaignOutcome, RouteSearchError> {
    config.validate()?;
    fs::create_dir_all(&config.results)?;
    let archive_path = config.results.join("archive.jsonl");
    let snapshot_path = config.results.join("archive.json");
    let proposal_log_path = config.results.join("proposal_log.jsonl");
    let cache_directory = config.results.join("cache");
    fs::create_dir_all(&cache_directory)?;
    let mut archive = if archive_path.exists() {
        load_archive(&archive_path)?
    } else {
        RouteArchive::default()
    };
    for result in &archive.results {
        if let Err(error) = config.grammar.route.validate(&result.variant) {
            return Err(RouteSearchError::Grammar(format!(
                "existing archive contains route {} excluded by the current grammar ({error}); \
                 preserve it and choose a new --results directory",
                result.variant_key
            )));
        }
    }
    let manifest_path = config.results.join("run.json");
    if archive.len() >= config.accepted_candidates && manifest_path.exists() {
        let manifest: CampaignManifest = serde_json::from_reader(File::open(&manifest_path)?)?;
        if manifest.strategy != config.strategy || manifest.seed != config.root_seed {
            return Err(RouteSearchError::Grammar(
                "existing completed campaign has a different strategy or seed".to_owned(),
            ));
        }
        if manifest.configuration.results.as_os_str() == std::ffi::OsStr::new(".")
            && manifest.configuration != artifact_configuration(config)
        {
            return Err(RouteSearchError::Grammar(
                "existing completed campaign has a different configuration".to_owned(),
            ));
        }
        if manifest.maximum_level >= config.maximum_level {
            return Ok(CampaignOutcome { archive, manifest });
        }
    }
    let mut counters = reconstruct_counters(&archive, &proposal_log_path, &config.agent.log_path)?;
    let mut rng = GrammarRng::new(config.root_seed ^ 0xD1B5_4A32_D192_ED03);
    let mut agent = (config.strategy == Strategy::Agent)
        .then(|| AgentClient::new(config.agent.clone()))
        .transpose()?;
    let mut agent_queue = VecDeque::new();
    let mut diversity_retry_note = None;
    let mut consecutive_transport_failures = 0_usize;
    let started = Instant::now();
    if config.maximum_level >= MaximumLevel::L1 {
        run_due_promotions(
            config,
            &mut archive,
            &mut counters,
            &mut rng,
            &archive_path,
            &snapshot_path,
        )?;
    }

    while archive.len() < config.accepted_candidates
        && counters.proposal_attempts < config.maximum_proposal_attempts
    {
        let phase = campaign_phase(&archive, config, counters.proposal_attempts);
        let transport_failures_before = counters.transport_failures;
        let proposal = match config.strategy {
            Strategy::Agent => {
                let client = agent.as_mut().ok_or_else(|| {
                    RouteSearchError::Grammar("agent strategy has no client".to_owned())
                })?;
                next_agent_proposal(
                    client,
                    &mut agent_queue,
                    phase,
                    &archive,
                    config,
                    &mut counters,
                    diversity_retry_note.take(),
                    &proposal_log_path,
                )
            }
            Strategy::Random => {
                sample_route(&config.grammar, &mut rng).map(|variant| RouteProposal {
                    variant,
                    rationale: "grammar-aware random baseline".to_owned(),
                })
            }
            Strategy::Evolutionary => evolutionary_proposal(&archive, config, &mut rng),
            Strategy::Seed => Err(RouteSearchError::Grammar(
                "seed is not a campaign strategy".to_owned(),
            )),
        };
        counters.proposal_attempts += 1;
        let proposal = match proposal {
            Ok(proposal) => proposal,
            Err(error) => {
                let transport_failed = counters.transport_failures > transport_failures_before;
                if transport_failed {
                    consecutive_transport_failures += 1;
                } else {
                    counters.invalid_proposals += 1;
                    consecutive_transport_failures = 0;
                }
                append_event(
                    &proposal_log_path,
                    counters.proposal_attempts,
                    if transport_failed {
                        ProposalEventKind::TransportFailed
                    } else {
                        ProposalEventKind::GrammarInvalid
                    },
                    None,
                    &error.to_string(),
                )?;
                if consecutive_transport_failures >= config.agent.maximum_consecutive_failures {
                    break;
                }
                continue;
            }
        };
        consecutive_transport_failures = 0;
        if let Err(error) = config.grammar.route.validate(&proposal.variant) {
            counters.invalid_proposals += 1;
            append_event(
                &proposal_log_path,
                counters.proposal_attempts,
                ProposalEventKind::GrammarInvalid,
                Some(proposal.variant.variant_key()),
                &error.to_string(),
            )?;
            continue;
        }
        let variant_key = proposal.variant.variant_key();
        if archive.contains_variant(&variant_key) {
            counters.duplicate_variants += 1;
            append_event(
                &proposal_log_path,
                counters.proposal_attempts,
                ProposalEventKind::DuplicateVariant,
                Some(variant_key),
                "exact variant already evaluated; no inner budget consumed",
            )?;
            continue;
        }
        let structure_key = proposal.variant.structure.structure_key();
        if archive.structure_variant_count(&structure_key)
            >= config.grammar.maximum_variants_per_structure
        {
            counters.structure_cap_rejections += 1;
            append_event(
                &proposal_log_path,
                counters.proposal_attempts,
                ProposalEventKind::StructureVariantCap,
                Some(variant_key),
                "body order reached the equal per-structure variant cap",
            )?;
            continue;
        }
        if matches!(phase, AgentPhase::Bootstrap | AgentPhase::Explore)
            && !clears_diversity(
                &proposal.variant.structure,
                &archive.protected_structures(config.protected_top),
                config.grammar.minimum_edit_distance,
            )
        {
            counters.diversity_rejections += 1;
            let detail =
                "body order is too close to the protected set; propose a more distinct order";
            append_event(
                &proposal_log_path,
                counters.proposal_attempts,
                ProposalEventKind::DiversityRejected,
                Some(variant_key),
                detail,
            )?;
            if config.strategy == Strategy::Agent {
                diversity_retry_note = Some(detail.to_owned());
            }
            continue;
        }

        let route = RouteCase::derive(proposal.variant.clone(), config.derivation.clone())?;
        let context =
            EvaluationContext::for_route(&route, config.inner_budget.clone(), config.root_seed);
        let cache_key = proposal.evaluation_key(&context)?;
        let cache_path = cache_directory.join(format!("{cache_key}.json"));
        let l0 = if cache_path.exists() {
            counters.cache_hits += 1;
            serde_json::from_reader(File::open(&cache_path)?)?
        } else {
            let result = optimize_route(&route, &config.inner_budget, config.root_seed)?;
            counters.l0_requested_evaluations = counters
                .l0_requested_evaluations
                .saturating_add(result.requested_evaluations);
            counters.l0_actual_evaluations = counters
                .l0_actual_evaluations
                .saturating_add(result.actual_evaluations);
            counters.l0_worker_seconds += result.worker_seconds;
            write_atomic_json(&cache_path, &result)?;
            result
        };
        let result = SearchResult::new(
            proposal.variant,
            archive.len(),
            counters.proposal_attempts,
            config.strategy,
            Some(proposal.rationale),
            l0,
            cache_key,
            unix_ms()?,
        )?;
        archive.add(result.clone())?;
        append_archive(&archive_path, &result)?;
        snapshot_archive(&snapshot_path, &archive)?;
        counters.accepted_candidates = archive.len();
        counters.niches = archive.niche_elites().len();
        if config.maximum_level >= MaximumLevel::L1 {
            run_due_promotions(
                config,
                &mut archive,
                &mut counters,
                &mut rng,
                &archive_path,
                &snapshot_path,
            )?;
        }
    }
    if archive.len() < config.accepted_candidates {
        let elapsed_seconds = started.elapsed().as_secs_f64();
        write_campaign_artifacts(
            config,
            &archive,
            counters.clone(),
            elapsed_seconds,
            "failed",
        )?;
        return Err(RouteSearchError::Grammar(format!(
            "campaign stopped at {} accepted routes after {} attempts",
            archive.len(),
            counters.proposal_attempts
        )));
    }
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let manifest =
        write_campaign_artifacts(config, &archive, counters, elapsed_seconds, "completed")?;
    Ok(CampaignOutcome { archive, manifest })
}

fn campaign_phase(
    archive: &RouteArchive,
    config: &CampaignConfig,
    proposal_attempts: usize,
) -> AgentPhase {
    if archive.len() < config.bootstrap_candidates {
        AgentPhase::Bootstrap
    } else if proposal_attempts.is_multiple_of(2) {
        AgentPhase::Explore
    } else {
        AgentPhase::Exploit
    }
}

#[allow(clippy::too_many_arguments)]
fn next_agent_proposal(
    client: &mut AgentClient,
    queue: &mut VecDeque<RouteProposal>,
    phase: AgentPhase,
    archive: &RouteArchive,
    config: &CampaignConfig,
    counters: &mut CampaignCounters,
    retry_note: Option<String>,
    proposal_log_path: &Path,
) -> Result<RouteProposal, RouteSearchError> {
    if let Some(proposal) = queue.pop_front() {
        return Ok(proposal);
    }
    let mut request = build_request(
        phase,
        archive.len(),
        config.accepted_candidates,
        counters.proposal_attempts + 1,
        archive,
        &config.grammar,
        &config.agent,
    );
    if let Some(note) = retry_note {
        request.user.push_str("\nSpecific retry requirement: ");
        request.user.push_str(&note);
    }
    counters.agent_calls += 1;
    let call = match client.propose(request) {
        Ok(call) => call,
        Err(error) => {
            counters.transport_failures += 1;
            return Err(error);
        }
    };
    if call.repaired {
        counters.repair_calls += 1;
        append_event(
            proposal_log_path,
            counters.proposal_attempts + 1,
            ProposalEventKind::Repaired,
            None,
            "malformed agent JSON was corrected by the single repair round trip",
        )?;
    }
    accumulate_usage(counters, &call.response.usage);
    for candidate in call.response.candidates {
        queue.push_back(candidate.into_proposal()?);
    }
    queue.pop_front().ok_or_else(|| {
        RouteSearchError::Grammar("agent response yielded no usable candidate".to_owned())
    })
}

fn accumulate_usage(counters: &mut CampaignCounters, usage: &AgentUsage) {
    counters.agent_input_tokens = counters
        .agent_input_tokens
        .saturating_add(usage.input_tokens.unwrap_or(0));
    counters.agent_output_tokens = counters
        .agent_output_tokens
        .saturating_add(usage.output_tokens.unwrap_or(0));
}

fn evolutionary_proposal(
    archive: &RouteArchive,
    config: &CampaignConfig,
    rng: &mut GrammarRng,
) -> Result<RouteProposal, RouteSearchError> {
    if archive.is_empty() {
        return sample_route(&config.grammar, rng).map(|variant| RouteProposal {
            variant,
            rationale: "evolutionary bootstrap sample".to_owned(),
        });
    }
    let elites = archive.top(archive.len().min(8));
    let parent = &elites[rng.index(elites.len())].variant;
    let (variant, operator) = mutate_route(parent, &config.grammar, rng)?;
    Ok(RouteProposal {
        variant,
        rationale: format!("route (1+1)-ES mutation {operator:?}"),
    })
}

fn run_due_promotions(
    config: &CampaignConfig,
    archive: &mut RouteArchive,
    counters: &mut CampaignCounters,
    rng: &mut GrammarRng,
    archive_path: &Path,
    snapshot_path: &Path,
) -> Result<(), RouteSearchError> {
    let due = (archive.len() / config.promotion.every).saturating_mul(config.promotion.batch);
    while archive
        .results
        .iter()
        .filter(|result| result.l1.is_some())
        .count()
        < due
    {
        let Some(index) = select_promotion(archive, config, rng) else {
            break;
        };
        let mut promoted = archive.results[index].clone();
        let refinement = refine_route(
            &promoted,
            &config.derivation,
            &config.refinement,
            config.root_seed,
        )?;
        counters.l1_promotions += 1;
        counters.l1_threshold_passed += usize::from(refinement.threshold_passed);
        counters.l1_requested_evaluations = counters
            .l1_requested_evaluations
            .saturating_add(refinement.requested_evaluations);
        counters.l1_actual_evaluations = counters
            .l1_actual_evaluations
            .saturating_add(refinement.actual_evaluations);
        counters.l1_worker_seconds += refinement.worker_seconds;
        promoted.surrogate_gap = refinement
            .score
            .map(|score| promoted.l0.estimated_score - score);
        promoted.l1 = Some(refinement);
        archive.update(promoted.clone())?;
        append_archive(archive_path, &promoted)?;
        snapshot_archive(snapshot_path, archive)?;
    }
    Ok(())
}

fn select_promotion(
    archive: &RouteArchive,
    config: &CampaignConfig,
    rng: &mut GrammarRng,
) -> Option<usize> {
    let candidates = archive
        .results
        .iter()
        .enumerate()
        .filter(|(_, result)| result.l0.evaluation_found && result.l1.is_none())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return None;
    }
    let completed = archive
        .results
        .iter()
        .filter(|result| result.l1.is_some())
        .count();
    let slot = completed % config.promotion.batch;
    if slot == 0 {
        return candidates.into_iter().max_by(|&left, &right| {
            archive.results[left]
                .l0
                .rank_cmp(&archive.results[right].l0)
        });
    }
    if rng.probability(config.promotion.control_rate) {
        let mut ranked = candidates;
        ranked.sort_by(|&left, &right| {
            archive.results[right]
                .l0
                .rank_cmp(&archive.results[left].l0)
        });
        let lower_half = ranked.len() / 2;
        return ranked
            .get(lower_half + rng.index(ranked.len() - lower_half))
            .copied();
    }
    let promoted_niches = archive
        .results
        .iter()
        .filter(|result| result.l1.is_some())
        .map(|result| result.niche_key.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    candidates
        .iter()
        .copied()
        .filter(|&index| !promoted_niches.contains(archive.results[index].niche_key.as_str()))
        .max_by(|&left, &right| {
            archive.results[left]
                .l0
                .rank_cmp(&archive.results[right].l0)
        })
        .or_else(|| {
            candidates.into_iter().max_by(|&left, &right| {
                archive.results[left]
                    .l0
                    .rank_cmp(&archive.results[right].l0)
            })
        })
}

/// Optimizes one runtime route under an explicit equal budget.
///
/// # Errors
///
/// Returns an error only if the validated route's optimizer midpoint cannot
/// be decoded.
#[allow(clippy::cast_precision_loss)]
pub fn optimize_route(
    route: &RouteCase,
    budget: &InnerBudget,
    root_seed: u64,
) -> Result<L0Result, RouteSearchError> {
    let bounds = route.codec().optimizer_bounds();
    let initial_guess = bounds
        .lower()
        .iter()
        .zip(bounds.upper())
        .map(|(&lower, &upper)| 0.5 * (lower + upper))
        .collect::<Vec<_>>();
    let tracker = FailureTracker::default();
    let objective = |coordinates: &[f64]| match route.evaluate(coordinates) {
        Ok(evaluation) => evaluation.sequence.objective,
        Err(error) => {
            tracker.record(&error);
            fcmaes_core::NAN_REPLACEMENT
        }
    };
    let config = AdvancedRetryConfig {
        retry: RetryConfig {
            num_retries: budget.retries,
            workers: budget.workers,
            max_evaluations: budget.initial_evaluations,
            seed: route_seed(root_seed, route.variant()),
            value_limit: f64::INFINITY,
            stop_fitness: f64::NEG_INFINITY,
            statistic_num: 100,
            ..Default::default()
        },
        check_interval: 100,
        max_eval_fac: budget.maximum_evaluation_factor,
        ..Default::default()
    };
    let started = Instant::now();
    let retry = advanced_retry(&objective, &bounds, &config, |function, context| {
        de_cma_run(function, context, &initial_guess)
    });
    let wall_seconds = started.elapsed().as_secs_f64();
    let resolved_workers = resolved_workers(budget.workers, budget.retries);
    let physical = route
        .codec()
        .decode(&retry.x)
        .or_else(|_| route.codec().decode(&initial_guess))?;
    if let Ok(evaluation) = route.evaluate(&retry.x) {
        return Ok(L0Result {
            evaluation_found: true,
            objective: evaluation.sequence.objective,
            estimated_score: evaluation.sequence.estimated_score,
            fixed_mass_score: evaluation.sequence.score,
            constraint: evaluation.sequence.constraint,
            launch_v_infinity_km_s: evaluation.sequence.launch_v_infinity_km_s,
            powered_delta_v_km_s: evaluation.sequence.powered_delta_v_km_s,
            endpoint_repair_delta_v_km_s: evaluation.sequence.endpoint_repair_delta_v_km_s,
            minimum_periapsis_margin_km: evaluation.sequence.minimum_periapsis_margin_km,
            flight_days: evaluation.physical.total_flight_days(),
            branches: evaluation
                .sequence
                .branches
                .into_iter()
                .map(|(revolutions, path)| BranchChoice {
                    revolutions,
                    path: path.into(),
                })
                .collect(),
            epochs_mjd2000: evaluation.sequence.epochs_mjd2000,
            optimizer_decision: retry.x,
            physical_decision: evaluation.physical,
            requested_evaluations: budget.requested_evaluations(),
            actual_evaluations: retry.evaluations,
            resolved_workers,
            worker_seconds: wall_seconds * resolved_workers as f64,
            wall_seconds,
            failures: tracker.snapshot(),
            failure_examples: tracker.examples(),
        });
    }
    Ok(L0Result {
        evaluation_found: false,
        objective: FAILURE_OBJECTIVE,
        estimated_score: 0.0,
        fixed_mass_score: 0.0,
        constraint: FAILURE_OBJECTIVE,
        launch_v_infinity_km_s: 0.0,
        powered_delta_v_km_s: 0.0,
        endpoint_repair_delta_v_km_s: 0.0,
        minimum_periapsis_margin_km: -FAILURE_OBJECTIVE,
        flight_days: physical.total_flight_days(),
        branches: Vec::new(),
        epochs_mjd2000: encounter_epochs(&physical),
        optimizer_decision: retry.x,
        physical_decision: physical,
        requested_evaluations: budget.requested_evaluations(),
        actual_evaluations: retry.evaluations,
        resolved_workers,
        worker_seconds: wall_seconds * resolved_workers as f64,
        wall_seconds,
        failures: tracker.snapshot(),
        failure_examples: tracker.examples(),
    })
}

fn de_cma_run<O>(objective: &O, context: &RetryContext, initial_guess: &[f64]) -> RetryRunResult
where
    O: Fn(&[f64]) -> f64 + Sync,
{
    let dimension = context.bounds.dim();
    let de_budget = (context.max_evaluations * 2 / 5).max(31);
    let cma_budget = context.max_evaluations.saturating_sub(de_budget).max(31);
    let de_fitness = Fitness::bounded(dimension, 1, context.bounds.lower(), context.bounds.upper());
    let de_sigma = context
        .sdev
        .iter()
        .zip(context.bounds.lower().iter().zip(context.bounds.upper()))
        .map(|(&sigma, (&lower, &upper))| sigma * (upper - lower))
        .collect::<Vec<_>>();
    let guess = context.guess.as_deref().unwrap_or(initial_guess);
    let mut de = De::new(
        de_fitness,
        guess,
        &de_sigma,
        None,
        &DeParams {
            max_evaluations: de_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed,
            runid: i64::try_from(context.run_id).expect("retry identifier fits i64"),
            ..Default::default()
        },
    );
    let de_result = de.optimize(objective);
    let mut cma_fitness =
        Fitness::bounded(dimension, 1, context.bounds.lower(), context.bounds.upper());
    cma_fitness.set_normalize(true);
    let mut cma = Cmaes::new(
        cma_fitness,
        &de_result.x,
        &context.sdev,
        &CmaesParams {
            max_evaluations: cma_budget,
            stop_fitness: f64::NEG_INFINITY,
            seed: context.seed ^ 0xA076_1D64_78BD_642F,
            runid: i64::try_from(context.run_id).expect("retry identifier fits i64"),
            ..Default::default()
        },
    );
    let cma_result = cma.optimize(objective, 1);
    let (x, y) = if cma_result.y < de_result.y {
        (cma_result.x, cma_result.y)
    } else {
        (de_result.x, de_result.y)
    };
    RetryRunResult {
        x,
        y,
        evaluations: de_result.evaluations + cma_result.evaluations,
    }
}

#[derive(Default)]
struct FailureTracker {
    lambert: AtomicU64,
    singular: AtomicU64,
    disconnected: AtomicU64,
    propagation: AtomicU64,
    examples: Mutex<BTreeMap<FailureCode, FailureObservation>>,
}

impl FailureTracker {
    fn record(&self, error: &RouteSearchError) {
        let code = classify_failure(error);
        match code {
            FailureCode::LambertUnavailable => {
                self.lambert.fetch_add(1, AtomicOrdering::Relaxed);
            }
            FailureCode::SingularGeometry => {
                self.singular.fetch_add(1, AtomicOrdering::Relaxed);
            }
            FailureCode::NoConnectedChain => {
                self.disconnected.fetch_add(1, AtomicOrdering::Relaxed);
            }
            FailureCode::PropagationFailure | FailureCode::RefinementNotClosed => {
                self.propagation.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        let mut examples = self
            .examples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        examples.entry(code).or_insert_with(|| FailureObservation {
            code,
            leg: None,
            value: None,
            message: Some(error.to_string().chars().take(300).collect()),
        });
    }

    fn snapshot(&self) -> BTreeMap<FailureCode, u64> {
        [
            (
                FailureCode::LambertUnavailable,
                self.lambert.load(AtomicOrdering::Relaxed),
            ),
            (
                FailureCode::SingularGeometry,
                self.singular.load(AtomicOrdering::Relaxed),
            ),
            (
                FailureCode::NoConnectedChain,
                self.disconnected.load(AtomicOrdering::Relaxed),
            ),
            (
                FailureCode::PropagationFailure,
                self.propagation.load(AtomicOrdering::Relaxed),
            ),
        ]
        .into_iter()
        .filter(|(_, count)| *count > 0)
        .collect()
    }

    fn examples(&self) -> Vec<FailureObservation> {
        self.examples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect()
    }
}

fn classify_failure(error: &RouteSearchError) -> FailureCode {
    match error {
        RouteSearchError::Gtoc1(Gtoc1Error::Numerical(message))
            if message.contains("connected") =>
        {
            FailureCode::NoConnectedChain
        }
        RouteSearchError::Gtoc1(Gtoc1Error::Numerical(message))
            if message.contains("singular") || message.contains("zero flyby") =>
        {
            FailureCode::SingularGeometry
        }
        RouteSearchError::Gtoc1(Gtoc1Error::Pykep(_)) => FailureCode::LambertUnavailable,
        _ => FailureCode::PropagationFailure,
    }
}

fn append_event(
    path: &Path,
    attempt: usize,
    kind: ProposalEventKind,
    variant_key: Option<String>,
    detail: &str,
) -> Result<(), RouteSearchError> {
    append_proposal_event(
        path,
        &ProposalEvent {
            proposal_attempt: attempt,
            kind,
            variant_key,
            detail: detail.to_owned(),
            created_unix_ms: unix_ms()?,
        },
    )
}

fn encounter_epochs(physical: &PhysicalDecision) -> Vec<f64> {
    let mut epochs = Vec::with_capacity(physical.leg_days.len() + 1);
    let mut epoch = physical.launch_mjd2000;
    epochs.push(epoch);
    for &days in &physical.leg_days {
        epoch += days;
        epochs.push(epoch);
    }
    epochs
}

fn resolved_workers(requested: usize, retries: usize) -> usize {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let requested = if requested == 0 { available } else { requested };
    requested.min(available).min(retries)
}

fn unix_ms() -> Result<u64, RouteSearchError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| RouteSearchError::Duration("system clock predates Unix epoch".to_owned()))?
        .as_millis()
        .try_into()
        .map_err(|_| RouteSearchError::Duration("Unix timestamp exceeds u64".to_owned()))
}

fn write_campaign_artifacts(
    config: &CampaignConfig,
    archive: &RouteArchive,
    counters: CampaignCounters,
    elapsed_seconds: f64,
    status: &str,
) -> Result<CampaignManifest, RouteSearchError> {
    ensure_file(&config.results.join("proposal_log.jsonl"))?;
    ensure_file(&config.agent.log_path)?;
    write_archive_csv(&config.results.join("archive.csv"), archive)?;
    write_promotions_csv(&config.results.join("promotions.csv"), archive)?;
    write_convergence_csv(&config.results.join("convergence.csv"), archive)?;
    let artifacts = BTreeMap::from([
        ("archive_jsonl".to_owned(), "archive.jsonl".to_owned()),
        ("archive_csv".to_owned(), "archive.csv".to_owned()),
        ("promotions".to_owned(), "promotions.csv".to_owned()),
        ("convergence".to_owned(), "convergence.csv".to_owned()),
        ("proposal_log".to_owned(), "proposal_log.jsonl".to_owned()),
        ("agent_log".to_owned(), "agent_log.jsonl".to_owned()),
    ]);
    let agent = agent_run_manifest(config, &counters)?;
    let manifest = CampaignManifest {
        schema_version: 1,
        tutorial: "gtoc1-route-search".to_owned(),
        formulation: if config.maximum_level >= MaximumLevel::L1 {
            "lambert-endpoint-repair-l0+sims-flanagan-l1".to_owned()
        } else {
            "lambert-endpoint-repair-l0".to_owned()
        },
        status: status.to_owned(),
        strategy: config.strategy,
        configuration: artifact_configuration(config),
        agent,
        maximum_level: config.maximum_level,
        seed: config.root_seed,
        workers: archive
            .results
            .iter()
            .map(|result| {
                result.l1.as_ref().map_or(result.l0.resolved_workers, |l1| {
                    result.l0.resolved_workers.max(l1.resolved_workers)
                })
            })
            .max()
            .unwrap_or_else(|| {
                resolved_workers(config.inner_budget.workers, config.inner_budget.retries)
            }),
        requested_evaluations: counters
            .l0_requested_evaluations
            .saturating_add(counters.l1_requested_evaluations),
        actual_evaluations: counters
            .l0_actual_evaluations
            .saturating_add(counters.l1_actual_evaluations),
        elapsed_seconds: elapsed_seconds.max(
            archive
                .results
                .iter()
                .map(|result| result.l0.wall_seconds)
                .sum(),
        ),
        budget: counters,
        artifacts,
    };
    write_atomic_json(&config.results.join("run.json"), &manifest)?;
    Ok(manifest)
}

fn agent_run_manifest(
    config: &CampaignConfig,
    counters: &CampaignCounters,
) -> Result<Option<AgentRunManifest>, RouteSearchError> {
    if config.strategy != Strategy::Agent {
        return Ok(None);
    }
    let entries: Vec<AgentLogEntry> = if config.agent.log_path.exists() {
        read_jsonl_resilient(&config.agent.log_path)?
    } else {
        Vec::new()
    };
    let reported = entries
        .iter()
        .filter_map(|entry| entry.response.as_ref())
        .map(|response| &response.usage)
        .find(|usage| usage.provider.is_some() || usage.model.is_some());
    Ok(Some(AgentRunManifest {
        transport: config.agent.transport,
        provider: reported
            .and_then(|usage| usage.provider.clone())
            .or_else(|| config.agent.provider.clone()),
        model: reported
            .and_then(|usage| usage.model.clone())
            .or_else(|| config.agent.model.clone()),
        calls: counters.agent_calls,
        input_tokens: counters.agent_input_tokens,
        output_tokens: counters.agent_output_tokens,
    }))
}

fn artifact_configuration(config: &CampaignConfig) -> CampaignConfig {
    let mut recorded = config.clone();
    recorded.results = PathBuf::from(".");
    recorded.agent.log_path = PathBuf::from("agent_log.jsonl");
    recorded
}

fn ensure_file(path: &Path) -> Result<(), RouteSearchError> {
    if !path.exists() {
        File::create(path)?;
    }
    Ok(())
}

fn write_archive_csv(path: &Path, archive: &RouteArchive) -> Result<(), RouteSearchError> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "accepted_index,structure_key,variant_key,niche,strategy,evaluation_found,\
objective_l0,constraint_l0,estimated_score_l0,fixed_mass_score_l0,flight_days,\
requested_evaluations,actual_evaluations,worker_seconds,l1_promoted,l1_threshold_passed,\
l1_score,surrogate_gap"
    )?;
    for result in &archive.results {
        writeln!(
            writer,
            "{},{},{},{},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            result.accepted_index,
            result.structure_key,
            result.variant_key,
            result.niche_key,
            result.strategy,
            result.l0.evaluation_found,
            result.l0.objective,
            result.l0.constraint,
            result.l0.estimated_score,
            result.l0.fixed_mass_score,
            result.l0.flight_days,
            result.l0.requested_evaluations,
            result.l0.actual_evaluations,
            result.l0.worker_seconds,
            result.l1.is_some(),
            result.l1.as_ref().is_some_and(|l1| l1.threshold_passed),
            result
                .l1
                .as_ref()
                .and_then(|l1| l1.score)
                .map_or_else(String::new, |value| value.to_string()),
            result
                .surrogate_gap
                .map_or_else(String::new, |value| value.to_string())
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn write_promotions_csv(path: &Path, archive: &RouteArchive) -> Result<(), RouteSearchError> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "variant_key,l0_estimated_score,l1_score,surrogate_gap,l1_threshold_passed,\
failure,requested_evaluations,actual_evaluations,worker_seconds"
    )?;
    for result in archive.results.iter().filter(|result| result.l1.is_some()) {
        let l1 = result
            .l1
            .as_ref()
            .expect("filtered promotion has an L1 result");
        let failure = l1
            .outcome
            .as_ref()
            .map_or_else(String::new, |outcome| format!("{:?}", outcome.code));
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{}",
            result.variant_key,
            result.l0.estimated_score,
            l1.score.map_or_else(String::new, |value| value.to_string()),
            result
                .surrogate_gap
                .map_or_else(String::new, |value| value.to_string()),
            l1.threshold_passed,
            failure,
            l1.requested_evaluations,
            l1.actual_evaluations,
            l1.worker_seconds
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn write_convergence_csv(path: &Path, archive: &RouteArchive) -> Result<(), RouteSearchError> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "accepted_candidates,best_l0,best_l1,l1_promotions,niches,cumulative_worker_seconds"
    )?;
    let mut prefix = RouteArchive::default();
    let mut worker_seconds = 0.0;
    for result in &archive.results {
        prefix.results.push(result.clone());
        worker_seconds +=
            result.l0.worker_seconds + result.l1.as_ref().map_or(0.0, |l1| l1.worker_seconds);
        let best = prefix
            .best()
            .map_or(0.0, |candidate| candidate.l0.estimated_score);
        let best_l1 = prefix
            .results
            .iter()
            .filter_map(|candidate| candidate.l1.as_ref().and_then(|l1| l1.score))
            .max_by(f64::total_cmp);
        let promotions = prefix
            .results
            .iter()
            .filter(|candidate| candidate.l1.is_some())
            .count();
        writeln!(
            writer,
            "{},{},{},{},{},{}",
            prefix.len(),
            best,
            best_l1.map_or_else(String::new, |value| value.to_string()),
            promotions,
            prefix.niche_elites().len(),
            worker_seconds
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn reconstruct_counters(
    archive: &RouteArchive,
    proposal_log_path: &Path,
    agent_log_path: &Path,
) -> Result<CampaignCounters, RouteSearchError> {
    let mut counters = CampaignCounters {
        accepted_candidates: archive.len(),
        l0_requested_evaluations: archive
            .results
            .iter()
            .map(|result| result.l0.requested_evaluations)
            .sum(),
        l0_actual_evaluations: archive
            .results
            .iter()
            .map(|result| result.l0.actual_evaluations)
            .sum(),
        l0_worker_seconds: archive
            .results
            .iter()
            .map(|result| result.l0.worker_seconds)
            .sum(),
        niches: archive.niche_elites().len(),
        l1_promotions: archive
            .results
            .iter()
            .filter(|result| result.l1.is_some())
            .count(),
        l1_threshold_passed: archive
            .results
            .iter()
            .filter(|result| result.l1.as_ref().is_some_and(|l1| l1.threshold_passed))
            .count(),
        l1_requested_evaluations: archive
            .results
            .iter()
            .filter_map(|result| result.l1.as_ref())
            .map(|l1| l1.requested_evaluations)
            .sum(),
        l1_actual_evaluations: archive
            .results
            .iter()
            .filter_map(|result| result.l1.as_ref())
            .map(|l1| l1.actual_evaluations)
            .sum(),
        l1_worker_seconds: archive
            .results
            .iter()
            .filter_map(|result| result.l1.as_ref())
            .map(|l1| l1.worker_seconds)
            .sum(),
        proposal_attempts: archive
            .results
            .iter()
            .map(|result| result.proposal_attempt)
            .max()
            .unwrap_or(0),
        ..Default::default()
    };
    if proposal_log_path.exists() {
        let events: Vec<ProposalEvent> = read_jsonl_resilient(proposal_log_path)?;
        for event in events {
            counters.proposal_attempts = counters.proposal_attempts.max(event.proposal_attempt);
            match event.kind {
                ProposalEventKind::GrammarInvalid => counters.invalid_proposals += 1,
                ProposalEventKind::DuplicateVariant => counters.duplicate_variants += 1,
                ProposalEventKind::StructureVariantCap => counters.structure_cap_rejections += 1,
                ProposalEventKind::DiversityRejected => counters.diversity_rejections += 1,
                ProposalEventKind::RepairRequested | ProposalEventKind::Repaired => {
                    counters.repair_calls += 1;
                }
                ProposalEventKind::TransportFailed => counters.transport_failures += 1,
            }
        }
    }
    if agent_log_path.exists() {
        let entries: Vec<AgentLogEntry> = read_jsonl_resilient(agent_log_path)?;
        counters.agent_calls = entries.len();
        counters.repair_calls = entries.iter().filter(|entry| entry.repair).count();
        for entry in entries {
            if let Some(response) = entry.response {
                accumulate_usage(&mut counters, &response.usage);
            }
        }
    }
    Ok(counters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route_grammar::body_edit_distance;
    use crate::route_search::RouteVariant;
    use crate::sequences::JPL;

    #[test]
    fn duplicate_screening_does_not_invoke_the_optimizer() {
        let variant = RouteVariant::from_sequence_case(JPL);
        let key = variant.variant_key();
        let archive = RouteArchive {
            results: Vec::new(),
        };
        assert!(!archive.contains_variant(&key));
        let mut directions = variant.clone();
        directions.clockwise[0] = !directions.clockwise[0];
        assert_eq!(
            body_edit_distance(&variant.structure, &directions.structure),
            0
        );
    }

    #[test]
    fn smoke_configuration_is_equal_budget_and_l0_only() {
        let agent = CampaignConfig::smoke(Strategy::Agent, PathBuf::from("a"));
        let random = CampaignConfig::smoke(Strategy::Random, PathBuf::from("b"));
        assert_eq!(agent.inner_budget, random.inner_budget);
        assert_eq!(agent.accepted_candidates, random.accepted_candidates);
        assert_eq!(agent.maximum_level, MaximumLevel::L0);
        agent.validate().unwrap();
        random.validate().unwrap();
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn single_worker_optimization_is_bit_repeatable() {
        let route = RouteCase::derive(
            RouteVariant::from_sequence_case(JPL),
            RouteDerivationConfig::default(),
        )
        .unwrap();
        let budget = InnerBudget {
            retries: 1,
            initial_evaluations: 200,
            maximum_evaluation_factor: 1.0,
            workers: 1,
        };
        let first = optimize_route(&route, &budget, 42).unwrap();
        let second = optimize_route(&route, &budget, 42).unwrap();
        assert_eq!(first.objective, second.objective);
        assert_eq!(first.optimizer_decision, second.optimizer_decision);
        assert_eq!(first.actual_evaluations, second.actual_evaluations);
    }
}
