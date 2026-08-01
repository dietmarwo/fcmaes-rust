//! Versioned campaign, comparison and descriptor-gate artifacts.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::archive::Archive;
use crate::config::{Preset, Protocol, VALIDATION_SEEDS};
use crate::network::simulate;
use crate::outer::{Campaign, Strategy, proposal_policy};
use crate::pilot::DescriptorPilot;

/// Current resumable campaign schema.
pub const CAMPAIGN_SCHEMA_VERSION: u32 = 2;

#[derive(Serialize)]
struct RunManifest<'a> {
    schema_version: u32,
    tutorial: &'static str,
    formulation: &'static str,
    status: &'a str,
    strategy: Strategy,
    preset: Preset,
    command: &'a str,
    seed: u64,
    protocol: Protocol,
    resolved_workers: usize,
    proposal_policy: &'static str,
    proposal_attempts: usize,
    accepted_candidates: usize,
    duplicates_or_invalid: usize,
    transport_failures: usize,
    input_tokens: u64,
    output_tokens: u64,
    best_validation_score: Option<f64>,
    best_topology: Option<&'a str>,
    exact_rediscoveries: BTreeMap<String, usize>,
    motif_classes: Vec<String>,
    message: &'a Option<String>,
    artifacts: BTreeMap<&'static str, serde_json::Value>,
}

#[derive(Deserialize)]
struct ResumeManifest {
    schema_version: u32,
    tutorial: String,
    status: String,
    strategy: Strategy,
    preset: Preset,
    seed: u64,
    protocol: Protocol,
    resolved_workers: usize,
    proposal_policy: String,
    proposal_attempts: usize,
    accepted_candidates: usize,
    duplicates_or_invalid: usize,
    transport_failures: usize,
    input_tokens: u64,
    output_tokens: u64,
    message: Option<String>,
}

fn resume_error(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

/// Restore an arm only when its manifest matches the complete run protocol.
///
/// Legacy schema-v1 artifacts remain valid evidence, but cannot be extended
/// because they do not identify the retry or proposal policy that created
/// every archived candidate.
pub fn restore_campaign(
    root: &Path,
    strategy: Strategy,
    preset: Preset,
    protocol: Protocol,
    seed: u64,
) -> Result<Archive, Box<dyn Error>> {
    Ok(
        restore_campaign_snapshot(root, strategy, preset, protocol, seed)?
            .map_or_else(Archive::default, |campaign| campaign.archive),
    )
}

/// Load a complete stored campaign without rerunning or rewriting it.
///
/// `None` means the arm has no manifest or archive yet. A partially present,
/// legacy, mismatched, or internally inconsistent arm is an error.
pub fn restore_campaign_snapshot(
    root: &Path,
    strategy: Strategy,
    preset: Preset,
    protocol: Protocol,
    seed: u64,
) -> Result<Option<Campaign>, Box<dyn Error>> {
    let directory = root.join(strategy.label());
    let archive_path = directory.join("candidates.jsonl");
    let manifest_path = directory.join("run.json");
    match (archive_path.is_file(), manifest_path.is_file()) {
        (false, false) => return Ok(None),
        (true, false) => {
            return Err(resume_error(format!(
                "resume rejected for {}: candidates.jsonl exists without run.json; \
                 preserve it and choose a new --output directory",
                strategy.label()
            )));
        }
        (false, true) => {
            return Err(resume_error(format!(
                "resume rejected for {}: run.json exists without candidates.jsonl; \
                 preserve it and choose a new --output directory",
                strategy.label()
            )));
        }
        (true, true) => {}
    }

    let value: serde_json::Value = serde_json::from_reader(fs::File::open(&manifest_path)?)?;
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version != u64::from(CAMPAIGN_SCHEMA_VERSION) {
        return Err(resume_error(format!(
            "resume rejected for {}: artifact schema {schema_version} cannot be extended by \
             campaign schema {CAMPAIGN_SCHEMA_VERSION}; preserve it and choose a new --output \
             directory",
            strategy.label()
        )));
    }
    let manifest: ResumeManifest = serde_json::from_value(value)?;
    let expected_policy = proposal_policy(strategy);
    if manifest.schema_version != CAMPAIGN_SCHEMA_VERSION
        || manifest.tutorial != "oscillator-topology-search"
        || manifest.strategy != strategy
        || manifest.preset != preset
        || manifest.seed != seed
        || manifest.protocol != protocol
        || manifest.resolved_workers != protocol.resolved_workers()
        || manifest.proposal_policy != expected_policy
    {
        return Err(resume_error(format!(
            "resume rejected for {}: stored seed, preset, optimizer protocol, worker count, or \
             proposal policy differs from this invocation; preserve it and choose a new --output \
             directory",
            strategy.label()
        )));
    }

    let archive = Archive::read_jsonl(&archive_path)?;
    let requested = protocol.requested_evaluations_per_topology();
    if archive.candidates.iter().any(|candidate| {
        candidate.strategy != strategy.label()
            || candidate.requested_evaluations != requested
            || candidate.training.replicates.len() != protocol.training_replications
            || candidate.validation.replicates.len() != protocol.validation_replications
    }) {
        return Err(resume_error(format!(
            "resume rejected for {}: candidates.jsonl contains rows from a different strategy or \
             numerical budget",
            strategy.label()
        )));
    }
    if manifest.accepted_candidates != archive.candidates.len() {
        return Err(resume_error(format!(
            "resume rejected for {}: run.json records {} accepted candidates but the archive has {}",
            strategy.label(),
            manifest.accepted_candidates,
            archive.candidates.len()
        )));
    }
    Ok(Some(Campaign {
        strategy,
        status: manifest.status,
        proposal_attempts: manifest.proposal_attempts,
        accepted_candidates: manifest.accepted_candidates,
        duplicate_or_invalid_proposals: manifest.duplicates_or_invalid,
        transport_failures: manifest.transport_failures,
        input_tokens: manifest.input_tokens,
        output_tokens: manifest.output_tokens,
        message: manifest.message,
        archive,
    }))
}

/// Write one complete arm.
pub fn write_campaign(
    root: &Path,
    campaign: &Campaign,
    preset: Preset,
    protocol: Protocol,
    command: &str,
    seed: u64,
) -> Result<(), Box<dyn Error>> {
    let directory = root.join(campaign.strategy.label());
    fs::create_dir_all(&directory)?;
    campaign
        .archive
        .write_jsonl(&directory.join("candidates.jsonl"))?;
    let best = campaign.archive.best();
    let mut artifacts = BTreeMap::from([
        (
            "candidates",
            json!({"path": "candidates.csv", "kind": "table"}),
        ),
        (
            "archive",
            json!({"path": "candidates.jsonl", "kind": "jsonl"}),
        ),
        (
            "convergence",
            json!({"path": "convergence.csv", "kind": "table"}),
        ),
    ]);
    if best.is_some() {
        artifacts.insert(
            "best_trace",
            json!({"path": "best_trace.csv", "kind": "table"}),
        );
    }
    let manifest = RunManifest {
        schema_version: CAMPAIGN_SCHEMA_VERSION,
        tutorial: "oscillator-topology-search",
        formulation: campaign.strategy.label(),
        status: &campaign.status,
        strategy: campaign.strategy,
        preset,
        command,
        seed,
        protocol,
        resolved_workers: protocol.resolved_workers(),
        proposal_policy: proposal_policy(campaign.strategy),
        proposal_attempts: campaign.proposal_attempts,
        accepted_candidates: campaign.accepted_candidates,
        duplicates_or_invalid: campaign.duplicate_or_invalid_proposals,
        transport_failures: campaign.transport_failures,
        input_tokens: campaign.input_tokens,
        output_tokens: campaign.output_tokens,
        best_validation_score: best.map(|candidate| candidate.validation.scalar_score),
        best_topology: best.map(|candidate| candidate.topology_key.as_str()),
        exact_rediscoveries: campaign.archive.exact_rediscoveries(),
        motif_classes: campaign.archive.motif_classes().into_iter().collect(),
        message: &campaign.message,
        artifacts,
    };
    fs::write(
        directory.join("run.json"),
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    )?;

    let mut csv = "proposal,topology,active_edges,dimension,motifs,exact_reference,training_score,validation_score,generalization_gap,period_train,period_validation,amplitude_train,amplitude_validation,participation_validation,failure_fraction_validation,requested_evaluations,actual_evaluations,wall_seconds\n".to_owned();
    let mut convergence =
        "accepted,proposal,best_validation_score,motif_classes_found,exact_references_found\n"
            .to_owned();
    let mut best_score = f64::INFINITY;
    let mut classes = std::collections::BTreeSet::new();
    let mut references = std::collections::BTreeSet::new();
    for (accepted, candidate) in campaign.archive.candidates.iter().enumerate() {
        writeln!(
            csv,
            "{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{:.9}",
            candidate.proposal,
            candidate.topology_key,
            candidate.topology.active_edges(),
            candidate.parameter_dimension,
            candidate.motif_flags.join("+"),
            candidate.exact_reference.as_deref().unwrap_or(""),
            candidate.training.scalar_score,
            candidate.validation.scalar_score,
            candidate.generalization_gap,
            candidate.training.period,
            candidate.validation.period,
            candidate.training.amplitude,
            candidate.validation.amplitude,
            candidate.validation.participation,
            candidate.validation.failure_fraction,
            candidate.requested_evaluations,
            candidate.actual_evaluations,
            candidate.wall_seconds,
        )?;
        best_score = best_score.min(candidate.validation.scalar_score);
        classes.extend(
            candidate
                .motif_flags
                .iter()
                .filter(|name| name.as_str() != "other")
                .cloned(),
        );
        if let Some(reference) = &candidate.exact_reference {
            references.insert(reference.clone());
        }
        writeln!(
            convergence,
            "{},{},{:.12},{},{}",
            accepted + 1,
            candidate.proposal,
            best_score,
            classes.len(),
            references.len()
        )?;
    }
    fs::write(directory.join("candidates.csv"), csv)?;
    fs::write(directory.join("convergence.csv"), convergence)?;

    if let Some(best) = best {
        let trace = simulate(&best.topology, &best.parameters, VALIDATION_SEEDS[0])?;
        let mut output = "time,A,B,C\n".to_owned();
        for (time, values) in trace.time.iter().zip(trace.values) {
            writeln!(
                output,
                "{time:.9},{:.9},{:.9},{:.9}",
                values[0], values[1], values[2]
            )?;
        }
        fs::write(directory.join("best_trace.csv"), output)?;
    }
    Ok(())
}

/// Lead with rediscovery and only then show score.
pub fn write_comparison(root: &Path, campaigns: &[&Campaign]) -> Result<(), Box<dyn Error>> {
    let mut markdown = "# Oscillator topology-search comparison\n\n".to_owned();
    markdown.push_str(
        "All scores are minimized (**lower is better**). Held-out reference encodings are excluded from proposal histories. A dash means a proposal arm did not exactly rediscover that topology.\n\n",
    );
    markdown.push_str("| Arm | Repressilator | Goodwin-like | Positive cycle | Toggle control | Classes | Accepted | Best | Median | Score < 1 | Agent tokens |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    let display = |value: Option<usize>| {
        value
            .filter(|value| *value > 0)
            .map_or_else(|| "—".to_owned(), |value| value.to_string())
    };
    for campaign in campaigns {
        let exact = campaign.archive.exact_rediscoveries();
        let rediscovery = |name: &str| {
            if campaign.strategy == Strategy::Reference {
                "n/a".to_owned()
            } else {
                display(exact.get(name).copied())
            }
        };
        let best = campaign.archive.best().map_or_else(
            || "—".to_owned(),
            |row| format!("{:.6}", row.validation.scalar_score),
        );
        let mut scores = campaign
            .archive
            .candidates
            .iter()
            .map(|row| row.validation.scalar_score)
            .collect::<Vec<_>>();
        scores.sort_by(f64::total_cmp);
        let median = match scores.len() {
            0 => "—".to_owned(),
            length if length % 2 == 0 => {
                format!("{:.6}", (scores[length / 2 - 1] + scores[length / 2]) / 2.0)
            }
            length => format!("{:.6}", scores[length / 2]),
        };
        let below_one = scores.iter().filter(|score| **score < 1.0).count();
        writeln!(
            markdown,
            "| {} ({}) | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            campaign.strategy.label(),
            campaign.status,
            rediscovery("repressilator"),
            rediscovery("goodwin-like"),
            rediscovery("positive-cycle"),
            rediscovery("toggle-control"),
            campaign.archive.motif_classes().len(),
            campaign.accepted_candidates,
            best,
            median,
            below_one,
            campaign.input_tokens + campaign.output_tokens,
        )?;
    }
    fs::write(root.join("comparison.md"), markdown)?;
    Ok(())
}

/// Write descriptor-pilot evidence and the QD gate manifest.
pub fn write_pilot(root: &Path, pilot: &DescriptorPilot) -> Result<(), Box<dyn Error>> {
    let directory = root.join("pilot");
    fs::create_dir_all(&directory)?;
    fs::write(
        directory.join("pilot.json"),
        format!("{}\n", serde_json::to_string_pretty(pilot)?),
    )?;
    let reasons = if pilot.rejection_reasons.is_empty() {
        "none".to_owned()
    } else {
        pilot.rejection_reasons.join("; ")
    };
    let arm_limit = pilot.arm_limit.as_deref().unwrap_or("none");
    let markdown = format!(
        "# Descriptor pilot\n\nStatus: **{}**.\n\n- pair: measured period × measured amplitude\n- candidates: {}\n- arms: {} observed / {} required ({})\n- replications: {} training / {} validation; sensitivity row uses {} training\n- native grid: {}×{}; coarse grid: {}×{}\n- observed period range: {:.6}–{:.6}\n- observed amplitude range: {:.6}–{:.6}\n- minimum per-arm coverage: {:.3}%\n- period below / above bounds: {:.3}% / {:.3}%\n- amplitude below / above bounds: {:.3}% / {:.3}%\n- any descriptor out of range: {:.3}%\n- absolute correlation: {:.4}\n- native-grid holdout retention: {:.3}%\n- coarse-grid holdout retention: {:.3}%\n- {}-replication training retention on the native grid: {:.3}%\n- reasons: {}\n\nThe structural E–A–I–S–motif key is decision-derived and remains a control. The high-replication row remeasures the frozen kinetic vectors; it does not rerun or retune the optimizer.\n",
        pilot.status,
        pilot.candidate_count,
        pilot.observed_arm_count,
        pilot.required_arm_count,
        arm_limit,
        pilot.training_replications,
        pilot.validation_replications,
        pilot.high_replication_training_replications,
        pilot.grid_side,
        pilot.grid_side,
        pilot.coarse_grid_side,
        pilot.coarse_grid_side,
        pilot.observed_period_range[0],
        pilot.observed_period_range[1],
        pilot.observed_amplitude_range[0],
        pilot.observed_amplitude_range[1],
        100.0 * pilot.minimum_arm_coverage,
        100.0 * pilot.period_below_fraction,
        100.0 * pilot.period_above_fraction,
        100.0 * pilot.amplitude_below_fraction,
        100.0 * pilot.amplitude_above_fraction,
        100.0 * pilot.out_of_range_fraction,
        pilot.descriptor_correlation.abs(),
        100.0 * pilot.holdout_niche_retention,
        100.0 * pilot.coarse_holdout_niche_retention,
        pilot.high_replication_training_replications,
        100.0 * pilot.high_replication_holdout_niche_retention,
        reasons,
    );
    fs::write(directory.join("pilot.md"), markdown)?;
    fs::write(
        directory.join("run.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "schema_version": 1,
                "tutorial": "oscillator-topology-search",
                "formulation": "descriptor-pilot",
                "strategy": "descriptor-pilot",
                "status": pilot.status,
                "candidate_count": pilot.candidate_count,
                "observed_arm_count": pilot.observed_arm_count,
                "required_arm_count": pilot.required_arm_count,
                "descriptor_pair": pilot.descriptor_pair,
                "holdout_niche_retention": pilot.holdout_niche_retention,
                "coarse_holdout_niche_retention": pilot.coarse_holdout_niche_retention,
                "high_replication_holdout_niche_retention":
                    pilot.high_replication_holdout_niche_retention,
                "artifacts": {
                    "pilot": {"path": "pilot.json", "kind": "json"},
                    "report": {"path": "pilot.md", "kind": "markdown"}
                }
            }))?
        ),
    )?;
    Ok(())
}

/// Write an explicit QD skip instead of silently omitting the arm.
pub fn write_qd_skipped(
    root: &Path,
    pilot: &DescriptorPilot,
    preset: Preset,
    command: &str,
    seed: u64,
) -> Result<(), Box<dyn Error>> {
    let directory = root.join("qd");
    fs::create_dir_all(&directory)?;
    let placeholder = directory.join("candidates.jsonl");
    if placeholder.is_file() {
        fs::remove_file(placeholder)?;
    }
    fs::write(
        directory.join("run.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "schema_version": 1,
                "tutorial": "oscillator-topology-search",
                "formulation": "qd",
                "strategy": "qd",
                "status": "skipped",
                "preset": preset,
                "command": command,
                "seed": seed,
                "reason": "descriptor pilot rejected",
                "pilot_rejection_reasons": pilot.rejection_reasons,
                "actual_evaluations": null,
                "artifacts": {}
            }))?
        ),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(label: &str) -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("oscillator-{label}-{suffix}"))
    }

    fn empty_campaign(strategy: Strategy) -> Campaign {
        Campaign {
            strategy,
            status: "complete".to_owned(),
            proposal_attempts: 0,
            accepted_candidates: 0,
            duplicate_or_invalid_proposals: 0,
            transport_failures: 0,
            input_tokens: 0,
            output_tokens: 0,
            message: None,
            archive: Archive::default(),
        }
    }

    #[test]
    fn resume_requires_an_exact_schema_protocol_and_policy_match() {
        let root = temporary_root("resume-contract");
        let preset = Preset::Smoke;
        let protocol = Protocol {
            inner_retries: 2,
            workers: 2,
            inner_evaluations: 17,
            ..Protocol::for_preset(preset)
        };
        let mut stored = empty_campaign(Strategy::Random);
        stored.proposal_attempts = 7;
        stored.duplicate_or_invalid_proposals = 2;
        stored.input_tokens = 123;
        stored.output_tokens = 17;
        write_campaign(&root, &stored, preset, protocol, "test", 42).unwrap();
        assert!(
            restore_campaign(&root, Strategy::Random, preset, protocol, 42)
                .unwrap()
                .candidates
                .is_empty()
        );
        let snapshot = restore_campaign_snapshot(&root, Strategy::Random, preset, protocol, 42)
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.proposal_attempts, 7);
        assert_eq!(snapshot.duplicate_or_invalid_proposals, 2);
        assert_eq!(snapshot.input_tokens, 123);
        assert_eq!(snapshot.output_tokens, 17);

        let different_budget = Protocol {
            inner_evaluations: 18,
            ..protocol
        };
        let error = restore_campaign(&root, Strategy::Random, preset, different_budget, 42)
            .unwrap_err()
            .to_string();
        assert!(error.contains("stored seed, preset, optimizer protocol"));

        let manifest_path = root.join("random/run.json");
        let mut value: serde_json::Value =
            serde_json::from_reader(fs::File::open(&manifest_path).unwrap()).unwrap();
        value["schema_version"] = serde_json::Value::from(1);
        fs::write(
            &manifest_path,
            format!("{}\n", serde_json::to_string_pretty(&value).unwrap()),
        )
        .unwrap();
        let error = restore_campaign(&root, Strategy::Random, preset, protocol, 42)
            .unwrap_err()
            .to_string();
        assert!(error.contains("artifact schema 1"));

        fs::remove_dir_all(root).unwrap();
    }
}
