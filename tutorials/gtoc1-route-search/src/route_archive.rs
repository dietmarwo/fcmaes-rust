// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Persistent archive and niche logic for GTOC1 route search.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use pykep_core::astro::lambert::LambertPath;
use serde::{Deserialize, Serialize};

use crate::route_grammar::compact_route;
use crate::route_search::{
    FailureCode, FailureObservation, PhysicalDecision, RouteSearchError, RouteStructure,
    RouteVariant, append_jsonl, read_jsonl_resilient, write_atomic_json,
};

/// Constraint-passing threshold used by archive and promotion ordering.
pub const L0_CONSTRAINT_THRESHOLD: f64 = 1.0e-8;

/// Campaign arm that generated an evaluated route.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// AI-agent proposal.
    Agent,
    /// Grammar-aware random proposal.
    Random,
    /// Grammar-aware (1+1)-ES proposal.
    Evolutionary,
    /// Historical regression seed.
    Seed,
}

/// Serializable multi-revolution Lambert branch side.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchPath {
    /// Unique zero-revolution family.
    Zero,
    /// Left multi-revolution family.
    Left,
    /// Right multi-revolution family.
    Right,
}

impl From<LambertPath> for BranchPath {
    fn from(path: LambertPath) -> Self {
        match path {
            LambertPath::ZeroRevolution => Self::Zero,
            LambertPath::Left => Self::Left,
            LambertPath::Right => Self::Right,
        }
    }
}

impl From<BranchPath> for LambertPath {
    fn from(path: BranchPath) -> Self {
        match path {
            BranchPath::Zero => Self::ZeroRevolution,
            BranchPath::Left => Self::Left,
            BranchPath::Right => Self::Right,
        }
    }
}

/// One selected Lambert family on a route leg.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BranchChoice {
    /// Revolution count.
    pub revolutions: usize,
    /// Zero/left/right family.
    pub path: BranchPath,
}

/// Complete cheap-model result and budget accounting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L0Result {
    /// Whether at least one complete finite Lambert chain was found.
    pub evaluation_found: bool,
    /// Penalized minimization objective.
    pub objective: f64,
    /// Rocket-equation endpoint-repair score used for ranking.
    pub estimated_score: f64,
    /// Fixed-1442.9-kg impact score.
    pub fixed_mass_score: f64,
    /// Dimensionless hard-constraint violation.
    pub constraint: f64,
    /// Earth departure excess in kilometres per second.
    pub launch_v_infinity_km_s: f64,
    /// Sum of powered flyby impulses in kilometres per second.
    pub powered_delta_v_km_s: f64,
    /// Endpoint repair sum in kilometres per second.
    pub endpoint_repair_delta_v_km_s: f64,
    /// Smallest flyby periapsis margin in kilometres.
    pub minimum_periapsis_margin_km: f64,
    /// Total flight duration in days.
    pub flight_days: f64,
    /// Selected Lambert family on each leg.
    pub branches: Vec<BranchChoice>,
    /// Encounter epochs in MJD2000 days.
    pub epochs_mjd2000: Vec<f64>,
    /// Raw launch/total/logit optimizer coordinates.
    pub optimizer_decision: Vec<f64>,
    /// Decoded launch and physical durations.
    pub physical_decision: PhysicalDecision,
    /// Deterministic sum of requested retry caps.
    pub requested_evaluations: u64,
    /// Actual optimizer objective calls.
    pub actual_evaluations: u64,
    /// Resolved worker count.
    pub resolved_workers: usize,
    /// Allocated worker-seconds (`workers × wall`).
    pub worker_seconds: f64,
    /// Measured route wall time.
    pub wall_seconds: f64,
    /// Per-code evaluation failure counts.
    pub failures: BTreeMap<FailureCode, u64>,
    /// At most one bounded representative per failure code.
    pub failure_examples: Vec<FailureObservation>,
}

impl L0Result {
    /// Feasibility-first comparison used by the archive.
    #[must_use]
    pub fn rank_cmp(&self, other: &Self) -> Ordering {
        let self_passes = self.constraint <= L0_CONSTRAINT_THRESHOLD;
        let other_passes = other.constraint <= L0_CONSTRAINT_THRESHOLD;
        match (self_passes, other_passes) {
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (true, true) => self.estimated_score.total_cmp(&other.estimated_score),
            (false, false) => other
                .constraint
                .total_cmp(&self.constraint)
                .then_with(|| self.estimated_score.total_cmp(&other.estimated_score)),
        }
    }

    fn validate_finite(&self) -> Result<(), RouteSearchError> {
        finite_slice(
            &[
                self.objective,
                self.estimated_score,
                self.fixed_mass_score,
                self.constraint,
                self.launch_v_infinity_km_s,
                self.powered_delta_v_km_s,
                self.endpoint_repair_delta_v_km_s,
                self.minimum_periapsis_margin_km,
                self.flight_days,
                self.worker_seconds,
                self.wall_seconds,
                self.physical_decision.launch_mjd2000,
            ],
            "L0 scalar",
        )?;
        finite_slice(&self.epochs_mjd2000, "L0 epoch")?;
        finite_slice(&self.optimizer_decision, "L0 optimizer decision")?;
        finite_slice(&self.physical_decision.leg_days, "L0 physical duration")?;
        for observation in &self.failure_examples {
            if observation.value.is_some_and(|value| !value.is_finite()) {
                return Err(RouteSearchError::Duration(
                    "failure observations must contain finite values or null".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// Result of an impulsive Sims–Flanagan promotion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefinementResult {
    /// Whether all declared numerical thresholds pass.
    pub threshold_passed: bool,
    /// Retained final mass, when the solve closed.
    pub final_mass_kg: Option<f64>,
    /// Model impact score, when the solve closed.
    pub score: Option<f64>,
    /// Largest normalized endpoint mismatch.
    pub maximum_normalized_mismatch: Option<f64>,
    /// Largest throttle-vector norm.
    pub maximum_throttle_norm: Option<f64>,
    /// Powered flyby impulse in kilometres per second.
    pub powered_delta_v_km_s: Option<f64>,
    /// Smallest flyby margin in kilometres.
    pub minimum_periapsis_margin_km: Option<f64>,
    /// Sampled heliocentric distance in AU.
    pub minimum_solar_distance_au: Option<f64>,
    /// Per-leg propellant use in kilograms.
    pub leg_fuel_kg: Vec<f64>,
    /// Final-stage per-leg decisions, retained as an L2 warm start.
    pub controls: Vec<Vec<f64>>,
    /// Sims–Flanagan impulses per leg.
    pub segments: usize,
    /// Deterministic sum of requested optimizer caps.
    pub requested_evaluations: u64,
    /// Actual objective calls.
    pub actual_evaluations: u64,
    /// Resolved retry worker allocation.
    pub resolved_workers: usize,
    /// Allocated worker-seconds.
    pub worker_seconds: f64,
    /// Wall time.
    pub wall_seconds: f64,
    /// Failure when no threshold-passing result was found.
    pub outcome: Option<FailureObservation>,
}

/// Optional continuous-thrust finalist validation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationResult {
    /// Whether the declared model-qualified validation gate passed.
    pub threshold_passed: bool,
    /// Taylor transcription residual norm.
    pub taylor_residual_norm: Option<f64>,
    /// Independently repropagated DOP853 residual norm.
    pub dop853_residual_norm: Option<f64>,
    /// Maximum Taylor/DOP853 component difference.
    pub maximum_backend_difference: Option<f64>,
    /// Daily-sampled minimum solar distance in AU.
    pub minimum_solar_distance_au: Option<f64>,
    /// Failure when no validated result was found.
    pub outcome: Option<FailureObservation>,
}

/// One evaluated route and all fidelity levels reached so far.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchResult {
    /// Body-order identity.
    pub structure: RouteStructure,
    /// Exact body-plus-direction variant.
    pub variant: RouteVariant,
    /// Stable body-only key.
    pub structure_key: String,
    /// Stable evaluated variant key.
    pub variant_key: String,
    /// Route-shape niche descriptor.
    pub niche_key: String,
    /// Zero-based accepted-candidate index.
    pub accepted_index: usize,
    /// One-based proposal attempt.
    pub proposal_attempt: usize,
    /// Campaign strategy.
    pub strategy: Strategy,
    /// Untrusted human/model explanation, never treated as evidence.
    pub rationale: Option<String>,
    /// Cheap Lambert-DP result.
    pub l0: L0Result,
    /// Optional Sims–Flanagan promotion.
    pub l1: Option<RefinementResult>,
    /// Optional Taylor/DOP853 finalist validation.
    pub l2: Option<ValidationResult>,
    /// L0 estimated score minus L1 score.
    pub surrogate_gap: Option<f64>,
    /// Complete numerical cache key.
    pub cache_key: String,
    /// Observation timestamp; excluded from semantic replay.
    pub created_unix_ms: u64,
}

impl SearchResult {
    /// Constructs and validates an archive result.
    ///
    /// # Errors
    ///
    /// Returns an error when identity fields disagree or any serialized
    /// floating-point value is non-finite.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        variant: RouteVariant,
        accepted_index: usize,
        proposal_attempt: usize,
        strategy: Strategy,
        rationale: Option<String>,
        l0: L0Result,
        cache_key: String,
        created_unix_ms: u64,
    ) -> Result<Self, RouteSearchError> {
        l0.validate_finite()?;
        let structure = variant.structure.clone();
        let structure_key = structure.structure_key();
        let variant_key = variant.variant_key();
        let niche_key = niche_key(&variant, l0.flight_days);
        Ok(Self {
            structure,
            variant,
            structure_key,
            variant_key,
            niche_key,
            accepted_index,
            proposal_attempt,
            strategy,
            rationale: rationale.map(|value| bounded_text(&value, 2_000)),
            l0,
            l1: None,
            l2: None,
            surrogate_gap: None,
            cache_key,
            created_unix_ms,
        })
    }

    /// Validates all optional fidelity fields before persistence.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent identity or non-finite numbers.
    pub fn validate(&self) -> Result<(), RouteSearchError> {
        if self.structure != self.variant.structure
            || self.structure_key != self.structure.structure_key()
            || self.variant_key != self.variant.variant_key()
            || self.niche_key != niche_key(&self.variant, self.l0.flight_days)
        {
            return Err(RouteSearchError::Duration(
                "archive identity fields disagree".to_owned(),
            ));
        }
        self.l0.validate_finite()?;
        if self.surrogate_gap.is_some_and(|value| !value.is_finite()) {
            return Err(RouteSearchError::Duration(
                "surrogate gap must be finite or null".to_owned(),
            ));
        }
        if let Some(refinement) = &self.l1 {
            validate_optional_finite(
                &[
                    refinement.final_mass_kg,
                    refinement.score,
                    refinement.maximum_normalized_mismatch,
                    refinement.maximum_throttle_norm,
                    refinement.powered_delta_v_km_s,
                    refinement.minimum_periapsis_margin_km,
                    refinement.minimum_solar_distance_au,
                    Some(refinement.worker_seconds),
                    Some(refinement.wall_seconds),
                ],
                "L1 scalar",
            )?;
            finite_slice(&refinement.leg_fuel_kg, "L1 leg fuel")?;
            for controls in &refinement.controls {
                finite_slice(controls, "L1 controls")?;
            }
        }
        if let Some(validation) = &self.l2 {
            validate_optional_finite(
                &[
                    validation.taylor_residual_norm,
                    validation.dop853_residual_norm,
                    validation.maximum_backend_difference,
                    validation.minimum_solar_distance_au,
                ],
                "L2 scalar",
            )?;
        }
        Ok(())
    }

    /// Stable semantic digest that excludes timestamps and resource timing.
    ///
    /// # Errors
    ///
    /// Returns an error only if serialization fails.
    pub fn semantic_digest(&self) -> Result<String, RouteSearchError> {
        let mut semantic = self.clone();
        semantic.created_unix_ms = 0;
        semantic.l0.wall_seconds = 0.0;
        semantic.l0.worker_seconds = 0.0;
        if let Some(refinement) = &mut semantic.l1 {
            refinement.wall_seconds = 0.0;
            refinement.worker_seconds = 0.0;
        }
        checksum(&semantic)
    }
}

/// Proposal outcome that did not consume an L0 candidate budget.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalEventKind {
    /// Typed grammar rejection.
    GrammarInvalid,
    /// Exact evaluated variant duplicate.
    DuplicateVariant,
    /// Body order already reached its variant cap.
    StructureVariantCap,
    /// Exploration edit-distance rejection.
    DiversityRejected,
    /// First response was malformed and a repair was requested.
    RepairRequested,
    /// Agent transport failed or timed out.
    TransportFailed,
    /// Repaired response was accepted.
    Repaired,
}

/// Append-only audit event for non-evaluated proposals.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalEvent {
    /// One-based proposal attempt.
    pub proposal_attempt: usize,
    /// Stable event category.
    pub kind: ProposalEventKind,
    /// Optional route variant key.
    pub variant_key: Option<String>,
    /// Bounded detail.
    pub detail: String,
    /// Observation timestamp.
    pub created_unix_ms: u64,
}

/// In-memory archive with feasibility-first elites.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteArchive {
    /// All accepted route evaluations in accepted-index order.
    pub results: Vec<SearchResult>,
}

impl RouteArchive {
    /// Adds one unique, validated route result.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite data or duplicate variant identity.
    pub fn add(&mut self, result: SearchResult) -> Result<(), RouteSearchError> {
        result.validate()?;
        if self.contains_variant(&result.variant_key) {
            return Err(RouteSearchError::Grammar(format!(
                "duplicate evaluated variant {}",
                result.variant_key
            )));
        }
        self.results.push(result);
        Ok(())
    }

    /// Replaces an existing route with a promotion/validation revision.
    ///
    /// Only L1, L2, surrogate-gap, and volatile timing fields may differ.
    /// This keeps append-only archive revisions from silently changing L0.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid data, a missing variant, or changed
    /// immutable/L0 fields.
    pub fn update(&mut self, result: SearchResult) -> Result<(), RouteSearchError> {
        result.validate()?;
        let current = self
            .results
            .iter_mut()
            .find(|current| current.variant_key == result.variant_key)
            .ok_or_else(|| {
                RouteSearchError::Grammar(format!(
                    "cannot update missing archive variant {}",
                    result.variant_key
                ))
            })?;
        if current.structure != result.structure
            || current.variant != result.variant
            || current.structure_key != result.structure_key
            || current.niche_key != result.niche_key
            || current.accepted_index != result.accepted_index
            || current.proposal_attempt != result.proposal_attempt
            || current.strategy != result.strategy
            || current.rationale != result.rationale
            || current.l0 != result.l0
            || current.cache_key != result.cache_key
        {
            return Err(RouteSearchError::Grammar(
                "archive revision changed immutable or L0 fields".to_owned(),
            ));
        }
        *current = result;
        Ok(())
    }

    /// Number of accepted evaluations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Whether the archive is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Whether an exact variant is already evaluated.
    #[must_use]
    pub fn contains_variant(&self, variant_key: &str) -> bool {
        self.results
            .iter()
            .any(|result| result.variant_key == variant_key)
    }

    /// Number of evaluated direction variants for a body order.
    #[must_use]
    pub fn structure_variant_count(&self, structure_key: &str) -> usize {
        self.results
            .iter()
            .filter(|result| result.structure_key == structure_key)
            .count()
    }

    /// Feasibility-first best result.
    #[must_use]
    pub fn best(&self) -> Option<&SearchResult> {
        self.results
            .iter()
            .max_by(|left, right| left.l0.rank_cmp(&right.l0))
    }

    /// Best `count` results under the archive ordering.
    #[must_use]
    pub fn top(&self, count: usize) -> Vec<&SearchResult> {
        let mut ranked = self.results.iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.l0.rank_cmp(&left.l0));
        ranked.truncate(count);
        ranked
    }

    /// Best result in every structural niche.
    #[must_use]
    pub fn niche_elites(&self) -> BTreeMap<&str, &SearchResult> {
        let mut elites = BTreeMap::new();
        for result in &self.results {
            let replace = elites
                .get(result.niche_key.as_str())
                .is_none_or(|current: &&SearchResult| result.l0.rank_cmp(&current.l0).is_gt());
            if replace {
                elites.insert(result.niche_key.as_str(), result);
            }
        }
        elites
    }

    /// Protected body orders used by the exploration diversity gate.
    #[must_use]
    pub fn protected_structures(&self, top: usize) -> Vec<RouteStructure> {
        let mut seen = BTreeSet::new();
        self.top(top)
            .into_iter()
            .chain(self.niche_elites().into_values())
            .filter_map(|result| {
                seen.insert(result.structure_key.clone())
                    .then_some(result.structure.clone())
            })
            .collect()
    }

    /// Compact, prompt-safe archive summary.
    #[must_use]
    pub fn summary(&self, top: usize, niche_top: usize) -> String {
        if self.results.is_empty() {
            return "Archive empty.".to_owned();
        }
        let mut lines = vec![format!(
            "Evaluated {} routes; {} structures; {} niches.",
            self.results.len(),
            self.results
                .iter()
                .map(|result| &result.structure_key)
                .collect::<BTreeSet<_>>()
                .len(),
            self.niche_elites().len()
        )];
        lines.push("Top feasibility-first L0 routes:".to_owned());
        for result in self.top(top) {
            lines.push(format!(
                "- {} [{}] violation={:.3e} estimated_score={:.3}",
                compact_route(&result.structure),
                result.variant_key,
                result.l0.constraint,
                result.l0.estimated_score
            ));
        }
        lines.push("Niche elites:".to_owned());
        for result in self.niche_elites().into_values().take(niche_top) {
            lines.push(format!(
                "- {} {} estimated_score={:.3}",
                result.niche_key, result.variant_key, result.l0.estimated_score
            ));
        }
        lines.join("\n")
    }
}

/// Checksummed archive line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchiveLine {
    schema_version: u32,
    checksum: String,
    payload_json: String,
}

/// Appends one checksummed archive result.
///
/// # Errors
///
/// Returns an error for invalid data, serialization, or I/O.
pub fn append_archive(path: &Path, result: &SearchResult) -> Result<(), RouteSearchError> {
    result.validate()?;
    let payload_json = serde_json::to_string(result)?;
    append_jsonl(
        path,
        &ArchiveLine {
            schema_version: 2,
            checksum: checksum_bytes(payload_json.as_bytes()),
            payload_json,
        },
    )
}

/// Loads and verifies a checksummed archive, tolerating only a truncated final
/// JSONL record.
///
/// # Errors
///
/// Returns an error for I/O, mid-file corruption, checksum mismatch,
/// duplicates, or invalid values.
pub fn load_archive(path: &Path) -> Result<RouteArchive, RouteSearchError> {
    let lines: Vec<ArchiveLine> = read_jsonl_resilient(path)?;
    let mut archive = RouteArchive::default();
    for line in lines {
        let actual_checksum = checksum_bytes(line.payload_json.as_bytes());
        if line.schema_version != 2 || line.checksum != actual_checksum {
            return Err(RouteSearchError::Duration(format!(
                "archive checksum or schema mismatch: stored={}, actual={actual_checksum}",
                line.checksum
            )));
        }
        let result: SearchResult = serde_json::from_str(&line.payload_json)?;
        if archive.contains_variant(&result.variant_key) {
            archive.update(result)?;
        } else {
            archive.add(result)?;
        }
    }
    Ok(archive)
}

/// Writes the complete archive snapshot atomically.
///
/// # Errors
///
/// Returns an error for invalid data, serialization, or I/O.
pub fn snapshot_archive(path: &Path, archive: &RouteArchive) -> Result<(), RouteSearchError> {
    for result in &archive.results {
        result.validate()?;
    }
    write_atomic_json(path, archive)
}

/// Appends one proposal event.
///
/// # Errors
///
/// Returns an error for serialization or I/O.
pub fn append_proposal_event(path: &Path, event: &ProposalEvent) -> Result<(), RouteSearchError> {
    let mut bounded = event.clone();
    bounded.detail = bounded_text(&bounded.detail, 2_000);
    append_jsonl(path, &bounded)
}

/// Human-readable route-shape descriptor.
#[must_use]
pub fn niche_key(variant: &RouteVariant, flight_days: f64) -> String {
    let bodies = &variant.structure.bodies;
    let inner = bodies.iter().filter(|&&body| matches!(body, 1..=4)).count();
    let outer_tail = bodies
        .iter()
        .filter_map(|&body| match body {
            5 => Some('J'),
            6 => Some('S'),
            _ => None,
        })
        .collect::<String>();
    let changes = variant
        .clockwise
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let years = (flight_days / 365.25).floor().max(0.0) as usize;
    format!(
        "L{}-I{}-T{}-R{}-F{}y",
        bodies.len(),
        inner,
        if outer_tail.is_empty() {
            "none"
        } else {
            &outer_tail
        },
        changes,
        years
    )
}

fn finite_slice(values: &[f64], name: &str) -> Result<(), RouteSearchError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(RouteSearchError::Duration(format!(
            "{name} contains a non-finite value"
        )))
    }
}

fn validate_optional_finite(values: &[Option<f64>], name: &str) -> Result<(), RouteSearchError> {
    if values.iter().all(|value| value.is_none_or(f64::is_finite)) {
        Ok(())
    } else {
        Err(RouteSearchError::Duration(format!(
            "{name} contains a non-finite value"
        )))
    }
}

fn bounded_text(value: &str, maximum_chars: usize) -> String {
    value.chars().take(maximum_chars).collect()
}

fn checksum<T: Serialize>(value: &T) -> Result<String, RouteSearchError> {
    Ok(checksum_bytes(&serde_json::to_vec(value)?))
}

fn checksum_bytes(value: &[u8]) -> String {
    use std::fmt::Write as _;

    use sha2::{Digest, Sha256};

    let digest = Sha256::digest(value);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::io::Write;

    use super::*;
    use crate::real::JPL_DECISION;
    use crate::route_search::RouteVariant;
    use crate::sequences::{JPL, JPL2, JPL2_HISTORICAL_DECISION};

    fn result(
        case: crate::sequences::SequenceCase,
        decision: &[f64],
        index: usize,
    ) -> SearchResult {
        let variant = RouteVariant::from_sequence_case(case);
        let evaluation = case.evaluate_endpoint_repair_scout(decision).unwrap();
        SearchResult::new(
            variant,
            index,
            index + 1,
            Strategy::Seed,
            None,
            L0Result {
                evaluation_found: true,
                objective: evaluation.objective,
                estimated_score: evaluation.estimated_score,
                fixed_mass_score: evaluation.score,
                constraint: evaluation.constraint,
                launch_v_infinity_km_s: evaluation.launch_v_infinity_km_s,
                powered_delta_v_km_s: evaluation.powered_delta_v_km_s,
                endpoint_repair_delta_v_km_s: evaluation.endpoint_repair_delta_v_km_s,
                minimum_periapsis_margin_km: evaluation.minimum_periapsis_margin_km,
                flight_days: decision[1..].iter().sum(),
                branches: evaluation
                    .branches
                    .into_iter()
                    .map(|(revolutions, path)| BranchChoice {
                        revolutions,
                        path: path.into(),
                    })
                    .collect(),
                epochs_mjd2000: evaluation.epochs_mjd2000,
                optimizer_decision: decision.to_vec(),
                physical_decision: PhysicalDecision {
                    launch_mjd2000: decision[0],
                    leg_days: decision[1..].to_vec(),
                },
                requested_evaluations: 10,
                actual_evaluations: 10,
                resolved_workers: 1,
                worker_seconds: 1.0,
                wall_seconds: 1.0,
                failures: BTreeMap::new(),
                failure_examples: Vec::new(),
            },
            format!("key-{index}"),
            100,
        )
        .unwrap()
    }

    #[test]
    fn known_routes_have_stable_niches_and_constraint_ordering() {
        let jpl = result(JPL, &JPL_DECISION, 0);
        let jpl2 = result(JPL2, &JPL2_HISTORICAL_DECISION, 1);
        assert!(jpl.niche_key.starts_with("L9-I5-TJSJ-R1-F"));
        assert!(jpl2.niche_key.starts_with("L11-I7-TJSJ-R1-F"));
        let mut archive = RouteArchive::default();
        archive.add(jpl).unwrap();
        archive.add(jpl2).unwrap();
        assert_eq!(archive.len(), 2);
        assert_eq!(archive.niche_elites().len(), 2);
        assert!(archive.summary(2, 2).contains("feasibility-first"));
    }

    #[test]
    fn archive_rejects_non_finite_values_and_duplicate_variants() {
        let mut invalid = result(JPL, &JPL_DECISION, 0);
        invalid.l0.estimated_score = f64::NAN;
        assert!(invalid.validate().is_err());

        let valid = result(JPL, &JPL_DECISION, 0);
        let mut archive = RouteArchive::default();
        archive.add(valid.clone()).unwrap();
        assert!(archive.add(valid).is_err());
    }

    #[test]
    fn checksummed_jsonl_recovers_final_truncation_and_rejects_corruption() {
        let directory = std::env::temp_dir().join(format!("gtoc1-archive-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("archive.jsonl");
        let first = result(JPL, &JPL_DECISION, 0);
        append_archive(&path, &first).unwrap();
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(br#"{"schema_version":1"#).unwrap();
        }
        let loaded = load_archive(&path).unwrap();
        assert_eq!(loaded.results, vec![first.clone()]);

        let snapshot = directory.join("archive.json");
        snapshot_archive(&snapshot, &loaded).unwrap();
        let decoded: RouteArchive =
            serde_json::from_reader(std::fs::File::open(&snapshot).unwrap()).unwrap();
        assert_eq!(decoded, loaded);

        let corrupted = directory.join("corrupt.jsonl");
        fs::write(&corrupted, b"{bad}\n").unwrap();
        assert!(load_archive(&corrupted).is_err());

        let mut replay = first.clone();
        replay.created_unix_ms += 1;
        replay.l0.wall_seconds += 2.0;
        replay.l0.worker_seconds += 2.0;
        assert_eq!(
            first.semantic_digest().unwrap(),
            replay.semantic_digest().unwrap()
        );

        fs::remove_file(path).unwrap();
        fs::remove_file(snapshot).unwrap();
        fs::remove_file(corrupted).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn append_only_promotion_revision_preserves_one_logical_candidate() {
        let directory =
            std::env::temp_dir().join(format!("gtoc1-promotion-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("archive.jsonl");
        let first = result(JPL, &JPL_DECISION, 0);
        append_archive(&path, &first).unwrap();
        let mut promoted = first.clone();
        promoted.l1 = Some(RefinementResult {
            threshold_passed: false,
            final_mass_kg: Some(1_400.0),
            score: Some(1_700_000.0),
            maximum_normalized_mismatch: Some(1.0e-4),
            maximum_throttle_norm: Some(1.0),
            powered_delta_v_km_s: Some(0.0),
            minimum_periapsis_margin_km: Some(0.0),
            minimum_solar_distance_au: Some(0.5),
            leg_fuel_kg: vec![100.0],
            controls: vec![vec![0.0]],
            segments: 1,
            requested_evaluations: 100,
            actual_evaluations: 90,
            resolved_workers: 1,
            worker_seconds: 1.0,
            wall_seconds: 1.0,
            outcome: Some(FailureObservation {
                code: FailureCode::RefinementNotClosed,
                leg: Some(0),
                value: Some(1.0e-4),
                message: Some("not found within budget".to_owned()),
            }),
        });
        promoted.surrogate_gap = Some(
            promoted.l0.estimated_score - promoted.l1.as_ref().and_then(|l1| l1.score).unwrap(),
        );
        append_archive(&path, &promoted).unwrap();
        let loaded = load_archive(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.results[0], promoted);

        let mut changed_l0 = promoted.clone();
        changed_l0.l0.constraint += 1.0;
        assert!(loaded.clone().update(changed_l0).is_err());
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
