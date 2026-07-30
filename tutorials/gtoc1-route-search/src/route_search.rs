// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Phase-0 infrastructure for runtime GTOC1 route evaluation.
//!
//! This module deliberately stops at the deterministic, non-agent part of the
//! split-brain experiment: route semantics, route-derived bounds, a
//! total-flight-safe duration decoder, cache identity, conservative
//! diagnostics, and crash-safe persistence.

use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use fcmaes_core::RetryBounds;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Gtoc1Error;
use crate::real::MAXIMUM_FLIGHT_DAYS;
use crate::sequences::{SequenceCase, SequenceEvaluation, evaluate_runtime_endpoint_repair};

#[cfg(test)]
use crate::real::JPL_DECISION;
#[cfg(test)]
use crate::sequences::{
    DEIMOS, DEIMOS_HISTORICAL_DECISIONS, JENA, JENA_HISTORICAL_DECISIONS, JPL, JPL2,
    JPL2_HISTORICAL_DECISION,
};

const EARTH: usize = 3;
const ASTEROID: usize = 10;
const LAUNCH_LOWER_MJD2000: f64 = 3_653.0;
const LAUNCH_UPPER_MJD2000: f64 = 10_958.0;
const LOGIT_LIMIT: f64 = 30.0;
const THRUST_NEWTONS: f64 = 0.04;
const EXHAUST_VELOCITY_M_S: f64 = 24_516.625;
const INITIAL_MASS_KG: f64 = 1_500.0;
const SOLAR_EXCLUSION_AU: f64 = 0.2;

/// Stable Phase-0 route-search error.
#[derive(Debug)]
pub enum RouteSearchError {
    /// A route violates the deterministic grammar.
    Grammar(String),
    /// A vector has an unexpected length.
    Dimension {
        /// Human-readable vector name.
        name: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// A non-finite or out-of-range coordinate was supplied.
    Coordinate {
        /// Coordinate index.
        index: usize,
        /// Rejected value.
        value: f64,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// A physical duration cannot be represented by the decoder.
    Duration(String),
    /// The underlying GTOC1 evaluator failed.
    Gtoc1(Gtoc1Error),
    /// Persistence failed.
    Io(io::Error),
    /// Serialized data was malformed.
    Json(serde_json::Error),
}

impl fmt::Display for RouteSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grammar(reason) => write!(formatter, "invalid route: {reason}"),
            Self::Dimension {
                name,
                expected,
                actual,
            } => write!(formatter, "{name} has length {actual}, expected {expected}"),
            Self::Coordinate {
                index,
                value,
                reason,
            } => write!(formatter, "coordinate {index}={value} is invalid: {reason}"),
            Self::Duration(reason) => write!(formatter, "invalid duration allocation: {reason}"),
            Self::Gtoc1(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "persistence I/O failed: {error}"),
            Self::Json(error) => write!(formatter, "invalid persisted JSON: {error}"),
        }
    }
}

impl std::error::Error for RouteSearchError {}

impl From<Gtoc1Error> for RouteSearchError {
    fn from(error: Gtoc1Error) -> Self {
        Self::Gtoc1(error)
    }
}

impl From<io::Error> for RouteSearchError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for RouteSearchError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// Body-order identity, independent of Lambert direction.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteStructure {
    /// GTOP body identifiers from Earth through asteroid 2001 TW229.
    pub bodies: Vec<usize>,
}

impl RouteStructure {
    /// Constructs a body-order identity.
    #[must_use]
    pub fn new(bodies: Vec<usize>) -> Self {
        Self { bodies }
    }

    /// Returns the stable body-only archive key.
    #[must_use]
    pub fn structure_key(&self) -> String {
        self.bodies
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-")
    }
}

/// Fully evaluated route variant with one exact direction per leg.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteVariant {
    /// Body-order identity.
    pub structure: RouteStructure,
    /// Exact `LambertProblem::new` clockwise argument for every leg.
    pub clockwise: Vec<bool>,
}

impl RouteVariant {
    /// Constructs a route variant without changing its direction flags.
    #[must_use]
    pub fn new(bodies: Vec<usize>, clockwise: Vec<bool>) -> Self {
        Self {
            structure: RouteStructure::new(bodies),
            clockwise,
        }
    }

    /// Converts one historical compile-time case to runtime semantics.
    #[must_use]
    pub fn from_sequence_case(case: SequenceCase) -> Self {
        let legs = case.bodies.len() - 1;
        Self::new(case.bodies.to_vec(), case.rev_flags[..legs].to_vec())
    }

    /// Returns the stable body-plus-direction evaluation key.
    #[must_use]
    pub fn variant_key(&self) -> String {
        let bits = self
            .clockwise
            .iter()
            .map(|&clockwise| if clockwise { '1' } else { '0' })
            .collect::<String>();
        format!("{}|{bits}", self.structure.structure_key())
    }
}

/// Deterministic grammar limits shared by every campaign arm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteGrammar {
    /// Largest allowed number of encounters, including endpoints.
    pub maximum_encounters: usize,
    /// Largest run of identical bodies.
    pub maximum_same_body_run: usize,
    /// Largest number of Jupiter/Saturn encounters.
    pub maximum_outer_encounters: usize,
    /// Maximum competition flight time in days.
    pub maximum_flight_days: f64,
}

impl Default for RouteGrammar {
    fn default() -> Self {
        Self {
            maximum_encounters: 14,
            maximum_same_body_run: 4,
            maximum_outer_encounters: 4,
            maximum_flight_days: MAXIMUM_FLIGHT_DAYS,
        }
    }
}

impl RouteGrammar {
    /// Validates the route structure and its exact direction vector.
    ///
    /// # Errors
    ///
    /// Returns the first deterministic grammar violation.
    pub fn validate(&self, variant: &RouteVariant) -> Result<(), RouteSearchError> {
        let bodies = &variant.structure.bodies;
        if !(3..=self.maximum_encounters).contains(&bodies.len()) {
            return Err(RouteSearchError::Grammar(format!(
                "encounter count {} is outside 3..={}",
                bodies.len(),
                self.maximum_encounters
            )));
        }
        if bodies.first() != Some(&EARTH) {
            return Err(RouteSearchError::Grammar(
                "the first encounter must be Earth".to_owned(),
            ));
        }
        if bodies.last() != Some(&ASTEROID)
            || bodies.iter().filter(|&&body| body == ASTEROID).count() != 1
        {
            return Err(RouteSearchError::Grammar(
                "TW229 must appear exactly once, as the final body".to_owned(),
            ));
        }
        if variant.clockwise.len() + 1 != bodies.len() {
            return Err(RouteSearchError::Dimension {
                name: "clockwise",
                expected: bodies.len() - 1,
                actual: variant.clockwise.len(),
            });
        }
        if let Some(&body) = bodies[1..bodies.len() - 1]
            .iter()
            .find(|&&body| !matches!(body, 2 | 3 | 5 | 6))
        {
            return Err(RouteSearchError::Grammar(format!(
                "unsupported interior body {body}"
            )));
        }
        let mut longest_run = 1;
        let mut current_run = 1;
        for pair in bodies.windows(2) {
            if pair[0] == pair[1] {
                current_run += 1;
                longest_run = longest_run.max(current_run);
            } else {
                current_run = 1;
            }
        }
        if longest_run > self.maximum_same_body_run {
            return Err(RouteSearchError::Grammar(format!(
                "same-body run {longest_run} exceeds {}",
                self.maximum_same_body_run
            )));
        }
        let outer = bodies.iter().filter(|&&body| matches!(body, 5 | 6)).count();
        if outer > self.maximum_outer_encounters {
            return Err(RouteSearchError::Grammar(format!(
                "{outer} outer encounters exceed {}",
                self.maximum_outer_encounters
            )));
        }
        let minimum_days = bodies
            .windows(2)
            .map(|pair| leg_profile(pair[0], pair[1]).map(|profile| profile.lower_days))
            .sum::<Result<f64, _>>()?;
        if minimum_days > self.maximum_flight_days {
            return Err(RouteSearchError::Grammar(format!(
                "minimum flight {minimum_days} days exceeds {}",
                self.maximum_flight_days
            )));
        }
        Ok(())
    }
}

/// Route-derived duration bounds and Lambert family width for one leg.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegProfile {
    /// Inclusive lower duration in days.
    pub lower_days: f64,
    /// Inclusive upper duration in days.
    pub upper_days: f64,
    /// Largest Lambert revolution count evaluated.
    pub maximum_revolutions: usize,
}

fn leg_profile(from: usize, to: usize) -> Result<LegProfile, RouteSearchError> {
    let profile = if to == ASTEROID {
        LegProfile {
            lower_days: 300.0,
            upper_days: 1_500.0,
            maximum_revolutions: 1,
        }
    } else if is_inner(from) && is_inner(to) && from == to {
        // A 3.4-period ceiling excluded supplied JPL2/Jena schedules. Keep the
        // historical 14-day floor and a wider resonance-search interval.
        LegProfile {
            lower_days: 14.0,
            upper_days: 3_000.0,
            maximum_revolutions: 5,
        }
    } else if is_inner(from) && is_inner(to) {
        LegProfile {
            lower_days: 14.0,
            upper_days: 2_000.0,
            maximum_revolutions: 5,
        }
    } else if (is_inner(from) && to == 5) || (from == 5 && is_inner(to)) {
        LegProfile {
            lower_days: 200.0,
            upper_days: 2_500.0,
            maximum_revolutions: 2,
        }
    } else if (is_inner(from) && to == 6) || (from == 6 && is_inner(to)) {
        LegProfile {
            lower_days: 400.0,
            upper_days: 4_200.0,
            maximum_revolutions: 2,
        }
    } else if is_outer(from) && is_outer(to) {
        LegProfile {
            lower_days: 300.0,
            upper_days: 4_200.0,
            maximum_revolutions: 2,
        }
    } else {
        return Err(RouteSearchError::Grammar(format!(
            "unsupported leg {from}->{to}"
        )));
    };
    Ok(profile)
}

const fn is_inner(body: usize) -> bool {
    matches!(body, 1..=4)
}

const fn is_outer(body: usize) -> bool {
    matches!(body, 5 | 6)
}

/// Versioned settings used to derive a runtime route problem.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteDerivationConfig {
    /// Shared route grammar.
    pub grammar: RouteGrammar,
    /// Duration decoder identity included in cache keys.
    pub duration_decoder_version: String,
    /// L0 ephemeris identifier.
    pub ephemeris_id: String,
    /// VSOP coefficient threshold.
    pub vsop_threshold: f64,
    /// Version of the L0 objective and DP formulation.
    pub scout_formulation_version: String,
}

impl Default for RouteDerivationConfig {
    fn default() -> Self {
        Self {
            grammar: RouteGrammar::default(),
            duration_decoder_version: "capped-softmax-v1".to_owned(),
            ephemeris_id: "vsop2013".to_owned(),
            vsop_threshold: 1.0e-9,
            scout_formulation_version: "endpoint-repair-lexicographic-v1".to_owned(),
        }
    }
}

/// A physical launch and decoded leg-duration vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalDecision {
    /// Launch epoch in MJD2000 days.
    pub launch_mjd2000: f64,
    /// One duration in days per leg.
    pub leg_days: Vec<f64>,
}

impl PhysicalDecision {
    /// Returns launch followed by leg durations for the numerical scout.
    #[must_use]
    pub fn as_sequence_decision(&self) -> Vec<f64> {
        let mut values = Vec::with_capacity(self.leg_days.len() + 1);
        values.push(self.launch_mjd2000);
        values.extend_from_slice(&self.leg_days);
        values
    }

    /// Returns total time of flight in days.
    #[must_use]
    pub fn total_flight_days(&self) -> f64 {
        self.leg_days.iter().sum()
    }
}

/// Total-flight-safe capped-softmax duration codec.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_field_names)]
pub struct DurationCodec {
    lower_days: Vec<f64>,
    upper_days: Vec<f64>,
    maximum_flight_days: f64,
}

impl DurationCodec {
    /// Constructs and validates one route-specific decoder.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, non-finite, inconsistent, or globally
    /// impossible bounds.
    pub fn new(
        lower_days: Vec<f64>,
        upper_days: Vec<f64>,
        maximum_flight_days: f64,
    ) -> Result<Self, RouteSearchError> {
        if lower_days.is_empty() || lower_days.len() != upper_days.len() {
            return Err(RouteSearchError::Duration(
                "lower/upper vectors must have one entry per leg".to_owned(),
            ));
        }
        for (index, (&lower, &upper)) in lower_days.iter().zip(&upper_days).enumerate() {
            if !lower.is_finite() || !upper.is_finite() || lower <= 0.0 || lower >= upper {
                return Err(RouteSearchError::Coordinate {
                    index,
                    value: lower,
                    reason: "duration bounds must be finite and strictly increasing",
                });
            }
        }
        if !maximum_flight_days.is_finite() || maximum_flight_days < lower_days.iter().sum::<f64>()
        {
            return Err(RouteSearchError::Duration(
                "sum of lower bounds exceeds maximum flight time".to_owned(),
            ));
        }
        Ok(Self {
            lower_days,
            upper_days,
            maximum_flight_days,
        })
    }

    /// Number of route legs.
    #[must_use]
    pub fn legs(&self) -> usize {
        self.lower_days.len()
    }

    /// Per-leg lower bounds.
    #[must_use]
    pub fn lower_days(&self) -> &[f64] {
        &self.lower_days
    }

    /// Per-leg upper bounds.
    #[must_use]
    pub fn upper_days(&self) -> &[f64] {
        &self.upper_days
    }

    /// Largest representable total flight time.
    #[must_use]
    pub fn maximum_total_days(&self) -> f64 {
        self.maximum_flight_days.min(self.upper_days.iter().sum())
    }

    /// Returns bounds for launch, total flight, and `L-1` logits.
    ///
    /// # Panics
    ///
    /// Panics only if the already-validated codec is internally inconsistent.
    #[must_use]
    pub fn optimizer_bounds(&self) -> RetryBounds {
        let mut lower = Vec::with_capacity(self.legs() + 1);
        let mut upper = Vec::with_capacity(self.legs() + 1);
        lower.push(LAUNCH_LOWER_MJD2000);
        upper.push(LAUNCH_UPPER_MJD2000);
        lower.push(self.lower_days.iter().sum());
        upper.push(self.maximum_total_days());
        for _ in 1..self.legs() {
            lower.push(-LOGIT_LIMIT);
            upper.push(LOGIT_LIMIT);
        }
        RetryBounds::new(lower, upper).expect("validated duration codec defines valid bounds")
    }

    /// Decodes launch, total flight, and bounded logits to physical durations.
    ///
    /// # Errors
    ///
    /// Returns an error for wrong dimension or out-of-range coordinates.
    pub fn decode(&self, coordinates: &[f64]) -> Result<PhysicalDecision, RouteSearchError> {
        let expected = self.legs() + 1;
        if coordinates.len() != expected {
            return Err(RouteSearchError::Dimension {
                name: "optimizer coordinates",
                expected,
                actual: coordinates.len(),
            });
        }
        let bounds = self.optimizer_bounds();
        for (index, (&value, (&lower, &upper))) in coordinates
            .iter()
            .zip(bounds.lower().iter().zip(bounds.upper()))
            .enumerate()
        {
            if !value.is_finite() || value < lower || value > upper {
                return Err(RouteSearchError::Coordinate {
                    index,
                    value,
                    reason: "outside route-specific optimizer bounds",
                });
            }
        }
        let total = coordinates[1];
        let minimum = self.lower_days.iter().sum::<f64>();
        let extra = total - minimum;
        let capacities = self
            .upper_days
            .iter()
            .zip(&self.lower_days)
            .map(|(&upper, &lower)| upper - lower)
            .collect::<Vec<_>>();
        let mut logits = coordinates[2..].to_vec();
        logits.push(0.0);
        let maximum_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut weights = logits
            .iter()
            .map(|&logit| (logit - maximum_logit).exp())
            .collect::<Vec<_>>();
        let weight_sum = weights.iter().sum::<f64>();
        for weight in &mut weights {
            *weight /= weight_sum;
        }
        let allocation = capped_allocation(extra, &capacities, &weights);
        let leg_days = self
            .lower_days
            .iter()
            .zip(allocation)
            .map(|(&lower, allocated)| lower + allocated)
            .collect();
        Ok(PhysicalDecision {
            launch_mjd2000: coordinates[0],
            leg_days,
        })
    }

    /// Inverse-encodes a physical schedule.
    ///
    /// Schedules whose omitted final leg is exactly at its lower bound while
    /// another leg has positive excess do not have a finite softmax inverse.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bounds, total flight, or a non-finite
    /// inverse.
    pub fn encode(
        &self,
        physical_decision: &PhysicalDecision,
    ) -> Result<Vec<f64>, RouteSearchError> {
        if physical_decision.leg_days.len() != self.legs() {
            return Err(RouteSearchError::Dimension {
                name: "physical leg durations",
                expected: self.legs(),
                actual: physical_decision.leg_days.len(),
            });
        }
        if !physical_decision.launch_mjd2000.is_finite()
            || !(LAUNCH_LOWER_MJD2000..=LAUNCH_UPPER_MJD2000)
                .contains(&physical_decision.launch_mjd2000)
        {
            return Err(RouteSearchError::Coordinate {
                index: 0,
                value: physical_decision.launch_mjd2000,
                reason: "launch is outside the competition window",
            });
        }
        let mut excess = Vec::with_capacity(self.legs());
        for (index, ((&duration, &lower), &upper)) in physical_decision
            .leg_days
            .iter()
            .zip(&self.lower_days)
            .zip(&self.upper_days)
            .enumerate()
        {
            if !duration.is_finite() || duration < lower || duration > upper {
                return Err(RouteSearchError::Coordinate {
                    index: index + 1,
                    value: duration,
                    reason: "physical duration is outside its pair-class bounds",
                });
            }
            excess.push(duration - lower);
        }
        let total = physical_decision.total_flight_days();
        if total > self.maximum_total_days() {
            return Err(RouteSearchError::Coordinate {
                index: 1,
                value: total,
                reason: "total flight exceeds the competition cap",
            });
        }
        let mut coordinates = Vec::with_capacity(self.legs() + 1);
        coordinates.push(physical_decision.launch_mjd2000);
        coordinates.push(total);
        let extra = excess.iter().sum::<f64>();
        if extra == 0.0 {
            coordinates.resize(self.legs() + 1, 0.0);
            return Ok(coordinates);
        }
        let reference = excess[self.legs() - 1];
        if reference <= 0.0 {
            return Err(RouteSearchError::Duration(
                "the omitted final softmax weight is zero".to_owned(),
            ));
        }
        for &value in &excess[..self.legs() - 1] {
            if value <= 0.0 {
                return Err(RouteSearchError::Duration(
                    "a finite softmax inverse requires positive excess on every leg".to_owned(),
                ));
            }
            let logit = (value / reference).ln();
            if !(-LOGIT_LIMIT..=LOGIT_LIMIT).contains(&logit) {
                return Err(RouteSearchError::Duration(format!(
                    "inverse logit {logit} exceeds ±{LOGIT_LIMIT}"
                )));
            }
            coordinates.push(logit);
        }
        Ok(coordinates)
    }
}

fn capped_allocation(extra: f64, capacities: &[f64], weights: &[f64]) -> Vec<f64> {
    if extra <= 0.0 {
        return vec![0.0; capacities.len()];
    }
    let capacity_sum = capacities.iter().sum::<f64>();
    if extra >= capacity_sum {
        return capacities.to_vec();
    }
    let allocated_at = |scale: f64| {
        capacities
            .iter()
            .zip(weights)
            .map(|(&capacity, &weight)| capacity.min(scale * weight))
            .sum::<f64>()
    };
    let mut high = extra;
    while allocated_at(high) < extra {
        high *= 2.0;
    }
    let mut low = 0.0;
    for _ in 0..100 {
        let middle = 0.5 * (low + high);
        if allocated_at(middle) < extra {
            low = middle;
        } else {
            high = middle;
        }
    }
    let mut allocation = capacities
        .iter()
        .zip(weights)
        .map(|(&capacity, &weight)| capacity.min(high * weight))
        .collect::<Vec<_>>();
    let mut residual = extra - allocation.iter().sum::<f64>();
    if residual > 0.0 {
        for (value, &capacity) in allocation.iter_mut().zip(capacities) {
            let increment = residual.min(capacity - *value);
            *value += increment;
            residual -= increment;
        }
    } else if residual < 0.0 {
        for value in &mut allocation {
            let decrement = (-residual).min(*value);
            *value -= decrement;
            residual += decrement;
        }
    }
    allocation
}

/// Runtime route problem with route-derived bounds and revolution caps.
#[derive(Clone, Debug)]
pub struct RouteCase {
    variant: RouteVariant,
    profiles: Vec<LegProfile>,
    codec: DurationCodec,
    config: RouteDerivationConfig,
}

impl RouteCase {
    /// Derives a runtime route problem from deterministic configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when route grammar or derived bounds are invalid.
    pub fn derive(
        variant: RouteVariant,
        config: RouteDerivationConfig,
    ) -> Result<Self, RouteSearchError> {
        config.grammar.validate(&variant)?;
        let profiles = variant
            .structure
            .bodies
            .windows(2)
            .map(|pair| leg_profile(pair[0], pair[1]))
            .collect::<Result<Vec<_>, _>>()?;
        let codec = DurationCodec::new(
            profiles.iter().map(|profile| profile.lower_days).collect(),
            profiles.iter().map(|profile| profile.upper_days).collect(),
            config.grammar.maximum_flight_days,
        )?;
        Ok(Self {
            variant,
            profiles,
            codec,
            config,
        })
    }

    /// Exact runtime route variant.
    #[must_use]
    pub fn variant(&self) -> &RouteVariant {
        &self.variant
    }

    /// Effective per-leg profiles.
    #[must_use]
    pub fn profiles(&self) -> &[LegProfile] {
        &self.profiles
    }

    /// Route-specific duration codec.
    #[must_use]
    pub fn codec(&self) -> &DurationCodec {
        &self.codec
    }

    /// Derivation settings that define the numerical problem.
    #[must_use]
    pub fn config(&self) -> &RouteDerivationConfig {
        &self.config
    }

    /// Effective per-leg revolution caps.
    #[must_use]
    pub fn maximum_revolutions(&self) -> Vec<usize> {
        self.profiles
            .iter()
            .map(|profile| profile.maximum_revolutions)
            .collect()
    }

    /// Evaluates an optimizer-space decision through the generalized L0 scout.
    ///
    /// # Errors
    ///
    /// Returns an error from decoding or the numerical route evaluator.
    pub fn evaluate(&self, coordinates: &[f64]) -> Result<RouteEvaluation, RouteSearchError> {
        let physical = self.codec.decode(coordinates)?;
        let sequence_decision = physical.as_sequence_decision();
        let sequence = evaluate_runtime_endpoint_repair(
            &self.variant.structure.bodies,
            &self.variant.clockwise,
            &self.maximum_revolutions(),
            &sequence_decision,
        )?;
        let diagnostics = L0Diagnostics::from_evaluation(&sequence, &physical);
        Ok(RouteEvaluation {
            physical,
            sequence,
            diagnostics,
        })
    }

    /// Scalar callback for `fcmaes-core`.
    #[must_use]
    pub fn objective(&self, coordinates: &[f64]) -> f64 {
        self.evaluate(coordinates)
            .map_or(fcmaes_core::NAN_REPLACEMENT, |evaluation| {
                evaluation.sequence.objective
            })
    }
}

/// Generalized L0 result with physical and diagnostic data.
#[derive(Clone, Debug)]
pub struct RouteEvaluation {
    /// Decoded schedule.
    pub physical: PhysicalDecision,
    /// Existing Lambert-DP sequence evaluation.
    pub sequence: SequenceEvaluation,
    /// Conservative Phase-0 diagnostics.
    pub diagnostics: L0Diagnostics,
}

/// Whether a diagnostic can reject a route.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DiagnosticAuthority {
    /// Independently established necessary condition.
    Necessary,
    /// Heuristic or incomplete check; may inform but never reject.
    WarningOnly,
}

/// Outcome of one named diagnostic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticOutcome {
    /// Stable diagnostic name.
    pub name: String,
    /// Whether this check is allowed to prune.
    pub authority: DiagnosticAuthority,
    /// Whether the observed value passes the check.
    pub passes: bool,
    /// Optional finite observed value.
    pub value: Option<f64>,
    /// Optional finite threshold.
    pub threshold: Option<f64>,
    /// Short explanation suitable for artifacts.
    pub message: String,
}

impl DiagnosticOutcome {
    /// Returns `true` only for a failed, proven necessary condition.
    #[must_use]
    pub fn prunes(&self) -> bool {
        self.authority == DiagnosticAuthority::Necessary && !self.passes
    }
}

/// Diagnostics available from a cheap L0 route evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L0Diagnostics {
    /// Full-thrust rocket-equation capacity compared with total endpoint
    /// repair. This comparison is warning-only because repair is not a proved
    /// lower bound on continuous-thrust effort.
    pub thrust_budget: DiagnosticOutcome,
    /// L0 has no validated direction-aware perihelion-span test yet, so solar
    /// exclusion remains explicitly unevaluated and warning-only.
    pub solar_distance: DiagnosticOutcome,
}

impl L0Diagnostics {
    fn from_evaluation(evaluation: &SequenceEvaluation, physical: &PhysicalDecision) -> Self {
        let mass_flow_kg_s = THRUST_NEWTONS / EXHAUST_VELOCITY_M_S;
        let duration_seconds = physical.total_flight_days() * 86_400.0;
        let consumed = mass_flow_kg_s * duration_seconds;
        let available_delta_v_km_s = if consumed < INITIAL_MASS_KG {
            Some(
                EXHAUST_VELOCITY_M_S * (INITIAL_MASS_KG / (INITIAL_MASS_KG - consumed)).ln()
                    / 1_000.0,
            )
        } else {
            None
        };
        let required = evaluation.endpoint_repair_delta_v_km_s
            + (evaluation.launch_v_infinity_km_s - 2.5).max(0.0);
        Self {
            thrust_budget: DiagnosticOutcome {
                name: "global_full_thrust_capacity".to_owned(),
                authority: DiagnosticAuthority::WarningOnly,
                passes: available_delta_v_km_s.is_none_or(|available| available >= required),
                value: available_delta_v_km_s.map(|available| available - required),
                threshold: Some(0.0),
                message:
                    "warning only: endpoint repair is not a proved continuous-thrust lower bound"
                        .to_owned(),
            },
            solar_distance: DiagnosticOutcome {
                name: "lambert_arc_solar_exclusion".to_owned(),
                authority: DiagnosticAuthority::WarningOnly,
                passes: true,
                value: None,
                threshold: Some(SOLAR_EXCLUSION_AU),
                message:
                    "not pruned in Phase 0: direction-aware perihelion-span validation is pending"
                        .to_owned(),
            },
        }
    }

    /// Returns `true` when any diagnostic is entitled to prune.
    #[must_use]
    pub fn prunes(&self) -> bool {
        self.thrust_budget.prunes() || self.solar_distance.prunes()
    }
}

/// Validates diagnostics measured by a stored L1 Sims–Flanagan solution.
///
/// Solar distance and throttle norm are direct necessary-condition
/// observations at L1, unlike the warning-only L0 thrust surrogate.
#[must_use]
pub fn stored_l1_diagnostics(
    minimum_solar_distance_au: f64,
    maximum_throttle_norm: f64,
) -> [DiagnosticOutcome; 2] {
    [
        DiagnosticOutcome {
            name: "sampled_solar_distance".to_owned(),
            authority: DiagnosticAuthority::Necessary,
            passes: minimum_solar_distance_au >= SOLAR_EXCLUSION_AU,
            value: minimum_solar_distance_au
                .is_finite()
                .then_some(minimum_solar_distance_au),
            threshold: Some(SOLAR_EXCLUSION_AU),
            message: "sampled L1 coast points must remain outside 0.2 AU".to_owned(),
        },
        DiagnosticOutcome {
            name: "throttle_norm".to_owned(),
            authority: DiagnosticAuthority::Necessary,
            passes: maximum_throttle_norm <= 1.0 + 1.0e-8,
            value: maximum_throttle_norm
                .is_finite()
                .then_some(maximum_throttle_norm),
            threshold: Some(1.0 + 1.0e-8),
            message: "Sims-Flanagan Cartesian throttles must remain inside the unit ball"
                .to_owned(),
        },
    ]
}

/// Identical inner-optimizer budget assigned to every accepted route.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InnerBudget {
    /// Coordinated retry count.
    pub retries: usize,
    /// First-retry objective-evaluation cap.
    pub initial_evaluations: u64,
    /// Last-retry budget multiplier.
    pub maximum_evaluation_factor: f64,
    /// Requested workers; zero means all logical CPUs.
    pub workers: usize,
}

impl InnerBudget {
    /// Deterministic sum of all per-retry evaluation caps.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub fn requested_evaluations(&self) -> u64 {
        (0..self.retries)
            .map(|run| {
                let progress = if self.retries <= 1 {
                    1.0
                } else {
                    run as f64 / (self.retries - 1) as f64
                };
                let factor = 1.0 + (self.maximum_evaluation_factor.max(1.0) - 1.0) * progress;
                (self.initial_evaluations as f64 * factor)
                    .round()
                    .clamp(1.0, u64::MAX as f64) as u64
            })
            .sum()
    }
}

/// Numerical context that defines an L0 cache entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationContext {
    /// Effective per-leg Lambert revolution caps.
    pub effective_revolution_caps: Vec<usize>,
    /// Duration decoder version.
    pub duration_decoder_version: String,
    /// Ephemeris family identifier.
    pub ephemeris_id: String,
    /// VSOP threshold.
    pub vsop_threshold: f64,
    /// Scout formulation version.
    pub scout_formulation_version: String,
    /// Published pykep-core version.
    pub pykep_core_version: String,
    /// Published fcmaes-core version.
    pub fcmaes_core_version: String,
    /// Source revision or clean implementation identifier.
    pub implementation_revision: String,
    /// Equal inner-optimizer budget.
    pub budget: InnerBudget,
    /// Campaign root seed.
    pub root_seed: u64,
}

impl EvaluationContext {
    /// Builds the default Phase-0 identity context for one route.
    #[must_use]
    pub fn for_route(route: &RouteCase, budget: InnerBudget, root_seed: u64) -> Self {
        Self {
            effective_revolution_caps: route.maximum_revolutions(),
            duration_decoder_version: route.config.duration_decoder_version.clone(),
            ephemeris_id: route.config.ephemeris_id.clone(),
            vsop_threshold: route.config.vsop_threshold,
            scout_formulation_version: route.config.scout_formulation_version.clone(),
            pykep_core_version: "0.1.4".to_owned(),
            fcmaes_core_version: "0.1.3".to_owned(),
            implementation_revision: env!("CARGO_PKG_VERSION").to_owned(),
            budget,
            root_seed,
        }
    }
}

/// Agent proposal metadata. Only the route affects numerical identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteProposal {
    /// Exact evaluated variant.
    pub variant: RouteVariant,
    /// Human explanation, archived but excluded from the evaluation key.
    pub rationale: String,
}

impl RouteProposal {
    /// Computes a stable SHA-256 L0 key, excluding rationale text.
    ///
    /// # Errors
    ///
    /// Returns an error only if the finite identity cannot be serialized.
    pub fn evaluation_key(&self, context: &EvaluationContext) -> Result<String, RouteSearchError> {
        #[derive(Serialize)]
        struct Identity<'a> {
            variant: &'a RouteVariant,
            context: &'a EvaluationContext,
        }
        sha256_json(&Identity {
            variant: &self.variant,
            context,
        })
    }
}

/// Derives a process-independent optimizer seed from the root seed and exact
/// route variant.
#[must_use]
pub fn route_seed(root_seed: u64, variant: &RouteVariant) -> u64 {
    let digest = Sha256::digest(variant.variant_key().as_bytes());
    let mut first = [0_u8; 8];
    first.copy_from_slice(&digest[..8]);
    root_seed ^ splitmix64(u64::from_be_bytes(first))
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, RouteSearchError> {
    let bytes = serde_json::to_vec(value)?;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(encoded)
}

/// Stable evaluated-failure category.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum FailureCode {
    /// No Lambert family was available.
    LambertUnavailable,
    /// Endpoint geometry was singular or near-collinear.
    SingularGeometry,
    /// No branch chain connected every leg.
    NoConnectedChain,
    /// Ephemeris or propagation failed.
    PropagationFailure,
    /// A promoted refinement did not close within budget.
    RefinementNotClosed,
}

/// Bounded representative observation for one failure code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureObservation {
    /// Stable category.
    pub code: FailureCode,
    /// Optional zero-based route leg.
    pub leg: Option<usize>,
    /// Optional finite observed value.
    pub value: Option<f64>,
    /// Optional bounded human-readable detail.
    pub message: Option<String>,
}

/// One persisted Phase-0 evaluation summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phase0Record {
    /// Stable evaluation key.
    pub evaluation_key: String,
    /// Exact evaluated route.
    pub variant: RouteVariant,
    /// Raw launch/total/logit optimizer result.
    pub optimizer_coordinates: Vec<f64>,
    /// Decoded physical decision.
    pub physical_decision: PhysicalDecision,
    /// Deterministic requested evaluation cap.
    pub requested_evaluations: u64,
    /// Actual optimizer evaluations completed.
    pub actual_evaluations: u64,
    /// Whether a finite route result was accepted.
    pub accepted: bool,
    /// Penalized L0 minimization objective, when accepted.
    pub objective: Option<f64>,
    /// Fixed-mass impact score, when accepted.
    pub score: Option<f64>,
    /// Rocket-equation endpoint-repair score, when accepted.
    pub estimated_score: Option<f64>,
    /// Dimensionless hard-constraint violation, when accepted.
    pub hard_violation: Option<f64>,
    /// Evaluation failure counts.
    pub failures: BTreeMap<FailureCode, u64>,
    /// Worker count after resolving zero=all.
    pub resolved_workers: usize,
    /// Resolved workers multiplied by wall time.
    pub worker_seconds: f64,
    /// Measured wall time, excluded from semantic replay.
    pub wall_seconds: f64,
    /// Observation timestamp, excluded from semantic replay.
    pub timestamp_unix_ms: u64,
}

impl Phase0Record {
    /// Returns a digest of semantic content, excluding time and wall-clock
    /// observations.
    ///
    /// # Errors
    ///
    /// Returns an error only if serialization fails.
    pub fn semantic_digest(&self) -> Result<String, RouteSearchError> {
        #[derive(Serialize)]
        struct Semantic<'a> {
            evaluation_key: &'a str,
            variant: &'a RouteVariant,
            optimizer_coordinates: &'a [f64],
            physical_decision: &'a PhysicalDecision,
            requested_evaluations: u64,
            actual_evaluations: u64,
            accepted: bool,
            objective: Option<f64>,
            score: Option<f64>,
            estimated_score: Option<f64>,
            hard_violation: Option<f64>,
            failures: &'a BTreeMap<FailureCode, u64>,
            resolved_workers: usize,
        }
        sha256_json(&Semantic {
            evaluation_key: &self.evaluation_key,
            variant: &self.variant,
            optimizer_coordinates: &self.optimizer_coordinates,
            physical_decision: &self.physical_decision,
            requested_evaluations: self.requested_evaluations,
            actual_evaluations: self.actual_evaluations,
            accepted: self.accepted,
            objective: self.objective,
            score: self.score,
            estimated_score: self.estimated_score,
            hard_violation: self.hard_violation,
            failures: &self.failures,
            resolved_workers: self.resolved_workers,
        })
    }
}

/// Appends one complete, newline-terminated JSONL record and syncs it.
///
/// # Errors
///
/// Returns an I/O or serialization error.
pub fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<(), RouteSearchError> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_data()?;
    Ok(())
}

/// Reads JSONL while ignoring only an incomplete final record.
///
/// A malformed newline-terminated record or corruption before the final line
/// remains fatal.
///
/// # Errors
///
/// Returns an error for I/O or non-final JSON corruption.
pub fn read_jsonl_resilient<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, RouteSearchError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        let bytes = reader.read_until(b'\n', &mut line)?;
        if bytes == 0 {
            break;
        }
        let terminated = line.last() == Some(&b'\n');
        if terminated {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        match serde_json::from_slice(&line) {
            Ok(record) => records.push(record),
            Err(_) if !terminated => break,
            Err(error) => return Err(RouteSearchError::Json(error)),
        }
    }
    Ok(records)
}

/// Writes a JSON snapshot to a sibling temporary file and atomically renames
/// it over the target.
///
/// # Errors
///
/// Returns an I/O or serialization error.
pub fn write_atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<(), RouteSearchError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| RouteSearchError::Duration("snapshot path has no file name".to_owned()))?;
    let temporary = temporary_path(parent, filename);
    let result = (|| {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok::<(), RouteSearchError>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn temporary_path(parent: &Path, filename: &str) -> PathBuf {
    parent.join(format!(".{filename}.{}.tmp", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_case(case: SequenceCase) -> RouteCase {
        RouteCase::derive(
            RouteVariant::from_sequence_case(case),
            RouteDerivationConfig::default(),
        )
        .unwrap()
    }

    fn historical_schedules() -> Vec<(RouteCase, Vec<f64>)> {
        let mut schedules = vec![
            (runtime_case(JPL), JPL_DECISION.to_vec()),
            (runtime_case(JPL2), JPL2_HISTORICAL_DECISION.to_vec()),
        ];
        schedules.extend(
            JENA_HISTORICAL_DECISIONS
                .iter()
                .map(|schedule| (runtime_case(JENA), schedule.to_vec())),
        );
        schedules.extend(
            DEIMOS_HISTORICAL_DECISIONS
                .iter()
                .map(|schedule| (runtime_case(DEIMOS), schedule.to_vec())),
        );
        schedules
    }

    #[test]
    fn g0_historical_routes_round_trip_exact_direction_semantics() {
        for case in [JPL, JPL2, JENA, DEIMOS] {
            let variant = RouteVariant::from_sequence_case(case);
            let encoded = serde_json::to_string(&variant).unwrap();
            let decoded: RouteVariant = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded, variant);
            assert_eq!(decoded.structure.bodies, case.bodies);
            assert_eq!(decoded.clockwise, case.rev_flags[..case.bodies.len() - 1]);
        }
        assert_ne!(
            pykep_core::astro::lambert::LambertPath::Left,
            pykep_core::astro::lambert::LambertPath::Right
        );
    }

    #[test]
    fn g1_decoded_points_always_respect_leg_and_total_bounds() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        for route in [
            runtime_case(JPL),
            runtime_case(JPL2),
            runtime_case(JENA),
            runtime_case(DEIMOS),
        ] {
            let bounds = route.codec.optimizer_bounds();
            for _ in 0..256 {
                let coordinates = bounds
                    .lower()
                    .iter()
                    .zip(bounds.upper())
                    .map(|(&lower, &upper)| {
                        state = test_splitmix64(state);
                        #[allow(clippy::cast_precision_loss)]
                        let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
                        lower + unit * (upper - lower)
                    })
                    .collect::<Vec<_>>();
                let physical = route.codec.decode(&coordinates).unwrap();
                assert!(physical.total_flight_days() <= MAXIMUM_FLIGHT_DAYS + 1.0e-9);
                for ((&duration, &lower), &upper) in physical
                    .leg_days
                    .iter()
                    .zip(route.codec.lower_days())
                    .zip(route.codec.upper_days())
                {
                    assert!(duration >= lower - 1.0e-9);
                    assert!(duration <= upper + 1.0e-9);
                }
            }
        }
    }

    #[test]
    fn g1_all_historical_schedules_round_trip() {
        for (route, schedule) in historical_schedules() {
            let physical = PhysicalDecision {
                launch_mjd2000: schedule[0],
                leg_days: schedule[1..].to_vec(),
            };
            let encoded = route.codec.encode(&physical).unwrap();
            let decoded = route.codec.decode(&encoded).unwrap();
            for (&expected, actual) in physical.leg_days.iter().zip(&decoded.leg_days) {
                let relative = (actual - expected).abs() / expected.abs().max(1.0);
                assert!(
                    relative <= 1.0e-10,
                    "{}: expected {expected}, decoded {actual}, relative {relative}",
                    route.variant.variant_key()
                );
            }
        }
    }

    #[test]
    fn g2_every_numerical_factor_changes_the_key_but_rationale_does_not() {
        let route = runtime_case(JPL);
        let budget = InnerBudget {
            retries: 8,
            initial_evaluations: 20_000,
            maximum_evaluation_factor: 1.0,
            workers: 8,
        };
        let context = EvaluationContext::for_route(&route, budget, 42);
        let proposal = RouteProposal {
            variant: route.variant.clone(),
            rationale: "first explanation".to_owned(),
        };
        let baseline = proposal.evaluation_key(&context).unwrap();
        let mut rationale = proposal.clone();
        rationale.rationale = "unrelated prose".to_owned();
        assert_eq!(baseline, rationale.evaluation_key(&context).unwrap());

        let mut changed = proposal.clone();
        changed.variant.structure.bodies[1] = 1;
        assert_ne!(baseline, changed.evaluation_key(&context).unwrap());
        for index in 0..proposal.variant.clockwise.len() {
            let mut changed = proposal.clone();
            changed.variant.clockwise[index] = !changed.variant.clockwise[index];
            assert_ne!(baseline, changed.evaluation_key(&context).unwrap());
        }

        for mutate in [
            |value: &mut EvaluationContext| value.effective_revolution_caps[0] += 1,
            |value: &mut EvaluationContext| value.duration_decoder_version.push_str("-other"),
            |value: &mut EvaluationContext| value.ephemeris_id.push_str("-other"),
            |value: &mut EvaluationContext| value.vsop_threshold *= 10.0,
            |value: &mut EvaluationContext| value.scout_formulation_version.push_str("-other"),
            |value: &mut EvaluationContext| value.pykep_core_version.push_str("-other"),
            |value: &mut EvaluationContext| value.fcmaes_core_version.push_str("-other"),
            |value: &mut EvaluationContext| value.implementation_revision.push_str("-other"),
            |value: &mut EvaluationContext| value.budget.retries += 1,
            |value: &mut EvaluationContext| value.budget.initial_evaluations += 1,
            |value: &mut EvaluationContext| value.budget.maximum_evaluation_factor += 1.0,
            |value: &mut EvaluationContext| value.budget.workers += 1,
            |value: &mut EvaluationContext| value.root_seed += 1,
        ] {
            let mut changed_context = context.clone();
            mutate(&mut changed_context);
            assert_ne!(baseline, proposal.evaluation_key(&changed_context).unwrap());
        }
    }

    #[test]
    fn g3_stored_l1_solutions_pass_and_unproved_checks_cannot_prune() {
        for (solar, throttle) in [(0.654_921, 0.999_976), (0.578_735, 0.999_999_998_9)] {
            let checks = stored_l1_diagnostics(solar, throttle);
            assert!(checks.iter().all(|check| check.passes));
            assert!(checks.iter().all(|check| !check.prunes()));
        }
        let warning = DiagnosticOutcome {
            name: "unproved".to_owned(),
            authority: DiagnosticAuthority::WarningOnly,
            passes: false,
            value: Some(-1.0),
            threshold: Some(0.0),
            message: "heuristic".to_owned(),
        };
        assert!(!warning.prunes());

        for (route, schedule) in [
            (runtime_case(JPL2), JPL2_HISTORICAL_DECISION.as_slice()),
            (runtime_case(JENA), JENA_HISTORICAL_DECISIONS[0].as_slice()),
        ] {
            let physical = PhysicalDecision {
                launch_mjd2000: schedule[0],
                leg_days: schedule[1..].to_vec(),
            };
            let coordinates = route.codec.encode(&physical).unwrap();
            let evaluation = route.evaluate(&coordinates).unwrap();
            assert!(!evaluation.diagnostics.prunes());
            assert_eq!(
                evaluation.diagnostics.thrust_budget.authority,
                DiagnosticAuthority::WarningOnly
            );
            assert_eq!(
                evaluation.diagnostics.solar_distance.authority,
                DiagnosticAuthority::WarningOnly
            );
        }
    }

    #[test]
    fn g5_truncated_jsonl_atomic_snapshot_and_semantic_replay() {
        let directory = std::env::temp_dir().join(format!(
            "gtoc1-phase0-{}-{}",
            std::process::id(),
            test_splitmix64(17)
        ));
        fs::create_dir_all(&directory).unwrap();
        let log = directory.join("events.jsonl");
        append_jsonl(&log, &vec![1_u64, 2]).unwrap();
        {
            let mut file = OpenOptions::new().append(true).open(&log).unwrap();
            file.write_all(br"[3,4").unwrap();
        }
        let records: Vec<Vec<u64>> = read_jsonl_resilient(&log).unwrap();
        assert_eq!(records, vec![vec![1, 2]]);

        let snapshot = directory.join("snapshot.json");
        write_atomic_json(&snapshot, &records).unwrap();
        assert_eq!(
            serde_json::from_reader::<_, Vec<Vec<u64>>>(File::open(&snapshot).unwrap()).unwrap(),
            records
        );
        assert!(!temporary_path(&directory, "snapshot.json").exists());

        let route = runtime_case(JPL);
        let record = Phase0Record {
            evaluation_key: "key".to_owned(),
            variant: route.variant.clone(),
            optimizer_coordinates: vec![1.0, 2.0],
            physical_decision: PhysicalDecision {
                launch_mjd2000: 8_000.0,
                leg_days: vec![500.0; 8],
            },
            requested_evaluations: 10,
            actual_evaluations: 9,
            accepted: true,
            objective: Some(-1.0),
            score: Some(2.0),
            estimated_score: Some(1.0),
            hard_violation: Some(0.0),
            failures: BTreeMap::new(),
            resolved_workers: 2,
            worker_seconds: 20.0,
            wall_seconds: 10.0,
            timestamp_unix_ms: 100,
        };
        let mut replay = record.clone();
        replay.worker_seconds = 200.0;
        replay.wall_seconds = 100.0;
        replay.timestamp_unix_ms = 200;
        assert_eq!(
            record.semantic_digest().unwrap(),
            replay.semantic_digest().unwrap()
        );

        fs::remove_file(log).unwrap();
        fs::remove_file(snapshot).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    const fn test_splitmix64(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}
