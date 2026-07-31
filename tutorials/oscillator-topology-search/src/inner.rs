//! Fixed-evaluation inner optimization for one topology.

use std::time::Instant;

use fcmaes_core::{BiteParams, optimize_bite, parallel_batch};
use serde::{Deserialize, Serialize};

use crate::config::Protocol;
use crate::grammar::{Topology, exact_reference};
use crate::network::bounds;
use crate::score::{Metrics, training, validation};

/// Fully replayable topology evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateResult {
    pub strategy: String,
    pub proposal: usize,
    pub topology: Topology,
    pub topology_key: String,
    pub exact_reference: Option<String>,
    pub motif_flags: Vec<String>,
    pub structural_niche: String,
    pub parameter_dimension: usize,
    pub requested_evaluations: u64,
    pub actual_evaluations: u64,
    pub optimizer_seed: u64,
    pub parameters: Vec<f64>,
    pub training: Metrics,
    pub validation: Metrics,
    pub generalization_gap: f64,
    pub wall_seconds: f64,
}

/// Stable, human-readable cache key fields.
pub fn cache_key(topology: &Topology, protocol: &Protocol, seed: u64) -> String {
    format!(
        "topology={}|retries={}|workers={}|evals_per_retry={}|train={}|valid={}|seed={seed}|fcmaes=0.1.3|rebop=0.9.7-expr-public-v1",
        topology.key(),
        protocol.inner_retries,
        protocol.resolved_workers(),
        protocol.inner_evaluations,
        protocol.training_replications,
        protocol.validation_replications,
    )
}

fn retry_seed(seed: u64, retry: usize) -> u64 {
    if retry == 0 {
        return seed;
    }
    let mut value = seed ^ (retry as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Tune kinetics and validate on disjoint seeds.
pub fn optimize(
    topology: Topology,
    strategy: &str,
    proposal: usize,
    protocol: &Protocol,
    seed: u64,
) -> CandidateResult {
    let (lower, upper) = bounds(&topology);
    let objective = |decision: &[f64]| {
        training(&topology, decision, protocol.training_replications).scalar_score
    };
    let started = Instant::now();
    let retries: Vec<_> = (0..protocol.inner_retries).collect();
    let results = parallel_batch(&retries, protocol.resolved_workers() as i32, |&retry| {
        optimize_bite(
            &objective,
            &lower,
            &upper,
            None,
            &BiteParams {
                max_evaluations: protocol.inner_evaluations,
                seed: retry_seed(seed, retry),
                runid: proposal as i64 + retry as i64 * 1_000_000,
                ..Default::default()
            },
            protocol.bite_depth,
        )
    });
    let actual_evaluations = results.iter().map(|result| result.evaluations).sum();
    let result = results
        .into_iter()
        .min_by(|left, right| left.y.total_cmp(&right.y))
        .expect("validated inner retry count is positive");
    let training = training(&topology, &result.x, protocol.training_replications);
    let validation = validation(&topology, &result.x, protocol.validation_replications);
    CandidateResult {
        strategy: strategy.to_owned(),
        proposal,
        topology,
        topology_key: topology.key(),
        exact_reference: exact_reference(&topology).map(str::to_owned),
        motif_flags: topology.motif_flags(),
        structural_niche: topology.niche_key(),
        parameter_dimension: topology.parameter_dimension(),
        requested_evaluations: protocol.requested_evaluations_per_topology(),
        actual_evaluations,
        optimizer_seed: seed,
        parameters: result.x,
        generalization_gap: validation.scalar_score - training.scalar_score,
        training,
        validation,
        wall_seconds: started.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Preset, Protocol};
    use crate::grammar::REFERENCES;

    #[test]
    fn cache_key_includes_all_replay_boundaries() {
        let protocol = Protocol::for_preset(Preset::Smoke);
        let key = cache_key(&REFERENCES[0].1, &protocol, 42);
        for field in [
            "topology=",
            "retries=",
            "workers=",
            "evals_per_retry=",
            "train=",
            "valid=",
            "seed=",
            "fcmaes=",
            "rebop=",
        ] {
            assert!(key.contains(field));
        }
    }

    #[test]
    fn equal_budget_is_explicit_despite_variable_dimension() {
        let protocol = Protocol::for_preset(Preset::Smoke);
        let three_edges = REFERENCES[0].1;
        let six_edges = Topology::try_new([1, 1, 1, 2, 2, 2, 0, 0, 0]).unwrap();
        assert_ne!(
            three_edges.parameter_dimension(),
            six_edges.parameter_dimension()
        );
        assert_eq!(protocol.inner_evaluations, protocol.inner_evaluations);
    }

    #[test]
    fn parallel_inner_retries_are_worker_count_invariant() {
        let serial = Protocol {
            inner_retries: 2,
            workers: 1,
            inner_evaluations: 16,
            ..Protocol::for_preset(Preset::Smoke)
        };
        let parallel = Protocol {
            workers: 2,
            ..serial
        };
        let left = optimize(REFERENCES[0].1, "test", 1, &serial, 91);
        let right = optimize(REFERENCES[0].1, "test", 1, &parallel, 91);
        assert_eq!(left.parameters, right.parameters);
        assert_eq!(left.actual_evaluations, right.actual_evaluations);
        assert_eq!(left.requested_evaluations, 32);
        assert_eq!(left.training.scalar_score, right.training.scalar_score);
    }
}
