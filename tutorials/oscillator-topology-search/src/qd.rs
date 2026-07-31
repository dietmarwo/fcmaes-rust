//! Grammar-native QD arm, executed only after the descriptor pilot passes.

use std::collections::BTreeMap;

use fcmaes_core::Rng;

use crate::archive::Archive;
use crate::config::Protocol;
use crate::grammar::{mutate, random_valid};
use crate::inner;
use crate::outer::{Campaign, Strategy};
use crate::pilot::cell;

/// Mutate elites selected round-robin from occupied period/amplitude cells.
pub fn run(protocol: &Protocol, seed: u64, target: usize) -> Campaign {
    let mut archive = Archive::default();
    let mut elites: BTreeMap<usize, usize> = BTreeMap::new();
    let mut attempts = 0;
    let mut duplicates = 0;
    while archive.candidates.len() < target && attempts < target * 25 {
        attempts += 1;
        let mut rng = Rng::new(
            seed ^ 0xda94_2042_e4dd_58b5 ^ (attempts as u64).wrapping_mul(0x5851_f42d_4c95_7f2d),
        );
        let topology = if elites.is_empty() || archive.candidates.len() < 4 {
            random_valid(&mut rng)
        } else {
            let indices: Vec<_> = elites.values().copied().collect();
            let parent = archive.candidates[indices[attempts % indices.len()]].topology;
            mutate(&parent, &mut rng)
        };
        if archive.contains_key(&topology.key()) {
            duplicates += 1;
            continue;
        }
        let candidate = inner::optimize(
            topology,
            Strategy::Qd.label(),
            attempts,
            protocol,
            seed.wrapping_add(attempts as u64 * 1_000_003),
        );
        let candidate_index = archive.candidates.len();
        if let Some(niche) = cell(candidate.training.period, candidate.training.amplitude) {
            let replace = elites.get(&niche).is_none_or(|&old| {
                candidate.training.scalar_score < archive.candidates[old].training.scalar_score
            });
            if replace {
                elites.insert(niche, candidate_index);
            }
        }
        archive.insert(candidate);
    }
    Campaign {
        strategy: Strategy::Qd,
        status: if archive.candidates.len() >= target {
            "complete"
        } else {
            "incomplete"
        }
        .to_owned(),
        proposal_attempts: attempts,
        accepted_candidates: archive.candidates.len(),
        duplicate_or_invalid_proposals: duplicates,
        transport_failures: 0,
        input_tokens: 0,
        output_tokens: 0,
        message: Some(format!(
            "{} occupied training niches after {} accepted topologies",
            elites.len(),
            archive.candidates.len()
        )),
        archive,
    }
}
