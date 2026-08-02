// Copyright (c) 2026 Dietmar Wolz
// SPDX-License-Identifier: MIT

//! Deterministic variable-length grammar for GTOC1 route proposals.

use serde::{Deserialize, Serialize};

use crate::route_search::{RouteGrammar, RouteSearchError, RouteStructure, RouteVariant};

const EARTH: usize = 3;
const ASTEROID: usize = 10;
const INTERIOR_BODIES: [usize; 4] = [2, 3, 5, 6];

/// Versioned sampling and diversity settings shared by all campaign arms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrammarConfig {
    /// Deterministic route validity rules.
    pub route: RouteGrammar,
    /// Maximum evaluated direction variants per body order.
    pub maximum_variants_per_structure: usize,
    /// Minimum body-order edit distance during bootstrap/exploration.
    pub minimum_edit_distance: usize,
}

impl Default for GrammarConfig {
    fn default() -> Self {
        Self {
            route: RouteGrammar::default(),
            maximum_variants_per_structure: 1,
            minimum_edit_distance: 3,
        }
    }
}

impl GrammarConfig {
    /// Validates campaign-level grammar configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero variant cap.
    pub fn validate(&self) -> Result<(), RouteSearchError> {
        if self.maximum_variants_per_structure == 0 {
            return Err(RouteSearchError::Grammar(
                "maximum_variants_per_structure must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Derives the direction pattern shared by the historical JPL, JPL2, Jena,
/// and Deimos trajectories.
///
/// All inner and outward legs use the default direction. Only the final
/// Saturn-to-Jupiter and Jupiter-to-asteroid legs use the reverse direction.
#[must_use]
pub fn canonical_clockwise(bodies: &[usize]) -> Vec<bool> {
    bodies
        .windows(2)
        .map(|pair| matches!(pair, [6, 5] | [5, 10]))
        .collect()
}

/// Small process-independent PRNG used only by the discrete route grammar.
#[derive(Clone, Copy, Debug)]
pub struct GrammarRng {
    state: u64,
}

impl GrammarRng {
    /// Constructs a deterministic grammar RNG.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next uniformly distributed 64-bit word.
    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    /// Returns a uniform index in `0..length`.
    ///
    /// # Panics
    ///
    /// Panics when `length` is zero.
    #[must_use]
    pub fn index(&mut self, length: usize) -> usize {
        assert!(length > 0, "random index requires a non-empty range");
        debug_assert!(length > 0);
        #[allow(clippy::cast_possible_truncation)]
        let value = (u128::from(self.next_u64()) * length as u128) >> 64;
        usize::try_from(value).expect("scaled random index fits usize")
    }

    /// Bernoulli draw with a caller-validated probability.
    #[must_use]
    pub fn probability(&mut self, probability: f64) -> bool {
        #[allow(clippy::cast_precision_loss)]
        let unit = (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        unit < probability
    }
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Samples one grammar-valid route using a uniform encounter count.
///
/// # Errors
///
/// Returns an error for invalid configuration or if rejection sampling cannot
/// find a valid route in 1,000 attempts.
pub fn sample_route(
    config: &GrammarConfig,
    rng: &mut GrammarRng,
) -> Result<RouteVariant, RouteSearchError> {
    config.validate()?;
    for _ in 0..1_000 {
        let length = 3 + rng.index(config.route.maximum_encounters - 2);
        let mut bodies = Vec::with_capacity(length);
        bodies.push(EARTH);
        for _ in 1..length - 1 {
            bodies.push(INTERIOR_BODIES[rng.index(INTERIOR_BODIES.len())]);
        }
        bodies.push(ASTEROID);
        let clockwise = canonical_clockwise(&bodies);
        let variant = RouteVariant::new(bodies, clockwise);
        if config.route.validate(&variant).is_ok() {
            return Ok(variant);
        }
    }
    Err(RouteSearchError::Grammar(
        "route sampler exhausted 1,000 attempts".to_owned(),
    ))
}

/// Mutation operator used by the route (1+1)-ES baseline.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RouteMutation {
    /// Replace one interior encounter.
    SubstituteBody,
    /// Insert one interior encounter and corresponding direction bit.
    InsertBody,
    /// Delete one interior encounter and corresponding direction bit.
    DeleteBody,
    /// Exchange two adjacent interior encounters.
    SwapAdjacent,
    /// Replace the Jupiter/Saturn subsequence while preserving its length.
    ResampleOuterTail,
}

/// Mutates a route, retrying deterministic operators until grammar-valid.
///
/// # Errors
///
/// Returns an error for invalid input/configuration or if 256 attempts fail.
pub fn mutate_route(
    parent: &RouteVariant,
    config: &GrammarConfig,
    rng: &mut GrammarRng,
) -> Result<(RouteVariant, RouteMutation), RouteSearchError> {
    config.validate()?;
    config.route.validate(parent)?;
    let operators = [
        RouteMutation::SubstituteBody,
        RouteMutation::InsertBody,
        RouteMutation::DeleteBody,
        RouteMutation::SwapAdjacent,
        RouteMutation::ResampleOuterTail,
    ];
    for _ in 0..256 {
        let operator = operators[rng.index(operators.len())];
        let mut candidate = parent.clone();
        if apply_mutation(&mut candidate, operator, config, rng)
            && candidate != *parent
            && config.route.validate(&candidate).is_ok()
        {
            return Ok((candidate, operator));
        }
    }
    Err(RouteSearchError::Grammar(
        "route mutation exhausted 256 attempts".to_owned(),
    ))
}

fn apply_mutation(
    candidate: &mut RouteVariant,
    operator: RouteMutation,
    config: &GrammarConfig,
    rng: &mut GrammarRng,
) -> bool {
    let bodies = &mut candidate.structure.bodies;
    match operator {
        RouteMutation::SubstituteBody if bodies.len() > 2 => {
            let index = 1 + rng.index(bodies.len() - 2);
            bodies[index] = INTERIOR_BODIES[rng.index(INTERIOR_BODIES.len())];
        }
        RouteMutation::InsertBody if bodies.len() < config.route.maximum_encounters => {
            let index = 1 + rng.index(bodies.len() - 1);
            bodies.insert(index, INTERIOR_BODIES[rng.index(INTERIOR_BODIES.len())]);
        }
        RouteMutation::DeleteBody if bodies.len() > 3 => {
            let index = 1 + rng.index(bodies.len() - 2);
            bodies.remove(index);
        }
        RouteMutation::SwapAdjacent if bodies.len() > 3 => {
            let index = 1 + rng.index(bodies.len() - 3);
            bodies.swap(index, index + 1);
        }
        RouteMutation::ResampleOuterTail => {
            let indices = bodies
                .iter()
                .enumerate()
                .filter_map(|(index, &body)| matches!(body, 5 | 6).then_some(index))
                .collect::<Vec<_>>();
            if indices.is_empty() {
                return false;
            }
            for index in indices {
                bodies[index] = if rng.probability(0.5) { 5 } else { 6 };
            }
        }
        _ => return false,
    }
    candidate.clockwise = canonical_clockwise(bodies);
    true
}

/// Levenshtein edit distance between body orders, ignoring direction bits.
#[must_use]
pub fn body_edit_distance(left: &RouteStructure, right: &RouteStructure) -> usize {
    let mut previous = (0..=right.bodies.len()).collect::<Vec<_>>();
    let mut current = vec![0; right.bodies.len() + 1];
    for (left_index, &left_body) in left.bodies.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, &right_body) in right.bodies.iter().enumerate() {
            let substitution = usize::from(left_body != right_body);
            current[right_index + 1] = (previous[right_index + 1] + 1)
                .min(current[right_index] + 1)
                .min(previous[right_index] + substitution);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.bodies.len()]
}

/// Returns whether a proposal clears the body-only diversity threshold.
#[must_use]
pub fn clears_diversity(
    candidate: &RouteStructure,
    protected: &[RouteStructure],
    minimum_distance: usize,
) -> bool {
    protected
        .iter()
        .all(|route| body_edit_distance(candidate, route) >= minimum_distance)
}

/// Returns the canonical compact body-order label used in reports.
#[must_use]
pub fn compact_route(structure: &RouteStructure) -> String {
    structure
        .bodies
        .iter()
        .map(|&body| match body {
            1 => "Me",
            2 => "V",
            3 => "E",
            4 => "Ma",
            5 => "J",
            6 => "S",
            10 => "A",
            _ => "?",
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequences::{DEIMOS, JENA, JPL, JPL2};

    fn valid() -> RouteVariant {
        RouteVariant::from_sequence_case(JPL2)
    }

    #[test]
    fn every_route_rule_has_acceptance_and_rejection_coverage() {
        let grammar = RouteGrammar::default();
        grammar.validate(&valid()).unwrap();

        let invalid = [
            RouteVariant::new(vec![2, 3, 10], vec![false, false]),
            RouteVariant::new(vec![3, 10, 2], vec![false, false]),
            RouteVariant::new(vec![3, 10], vec![false]),
            RouteVariant::new(vec![3, 1, 10], vec![false, false]),
            RouteVariant::new(vec![3, 4, 10], vec![false, false]),
            RouteVariant::new(vec![3, 7, 10], vec![false, false]),
            RouteVariant::new(
                vec![3, 3, 3, 3, 3, 10],
                vec![false, false, false, false, false],
            ),
            RouteVariant::new(
                vec![3, 5, 6, 5, 6, 5, 10],
                vec![false, false, false, false, false, false],
            ),
            RouteVariant::new(vec![3, 2, 10], vec![false]),
        ];
        for route in invalid {
            assert!(grammar.validate(&route).is_err());
        }
        assert_eq!(valid().structure.bodies[3..7], [EARTH, EARTH, EARTH, EARTH]);

        let short_flight_grammar = RouteGrammar {
            maximum_flight_days: 100.0,
            ..RouteGrammar::default()
        };
        let duration_impossible = RouteVariant::new(vec![EARTH, 6, ASTEROID], vec![false, false]);
        assert!(short_flight_grammar.validate(&duration_impossible).is_err());
    }

    #[test]
    fn sampling_and_all_mutation_families_preserve_the_grammar() {
        let config = GrammarConfig::default();
        let mut rng = GrammarRng::new(42);
        for _ in 0..100 {
            let route = sample_route(&config, &mut rng).unwrap();
            config.route.validate(&route).unwrap();
            assert!(
                route.structure.bodies[1..route.structure.bodies.len() - 1]
                    .iter()
                    .all(|body| INTERIOR_BODIES.contains(body))
            );
        }
        let mut parent = RouteVariant::from_sequence_case(DEIMOS);
        let mut observed = [false; 5];
        for _ in 0..2_000 {
            let (route, operator) = mutate_route(&parent, &config, &mut rng).unwrap();
            config.route.validate(&route).unwrap();
            observed[operator as usize] = true;
            parent = route;
        }
        assert!(observed.into_iter().all(|seen| seen));
    }

    #[test]
    fn canonical_directions_match_historical_route_fixtures() {
        for case in [JPL, JPL2, JENA, DEIMOS] {
            assert_eq!(
                canonical_clockwise(case.bodies),
                case.rev_flags[..case.bodies.len() - 1]
            );
        }
    }

    #[test]
    fn edit_distance_uses_bodies_but_not_direction() {
        let left = valid();
        let mut direction = left.clone();
        direction.clockwise[0] = !direction.clockwise[0];
        assert_eq!(body_edit_distance(&left.structure, &direction.structure), 0);
        let inserted = RouteStructure::new(vec![3, 2, 2, 3, 3, 3, 3, 4, 5, 6, 5, 10]);
        assert_eq!(body_edit_distance(&left.structure, &inserted), 1);
        assert!(!clears_diversity(
            &inserted,
            std::slice::from_ref(&left.structure),
            3
        ));
    }
}
