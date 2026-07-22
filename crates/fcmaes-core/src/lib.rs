//! Pure-Rust core of the fcmaes optimization library.
//!
//! It contains the shared RNG and fitness layer, native optimizers, and retry
//! coordinators. GTOP benchmarks and standalone drivers live in the sibling
//! `fcmaes-examples` crate.

pub mod biteopt;
pub mod cmaes;
pub mod crfmnes;
pub mod da;
pub mod de;
pub mod fitness;
pub mod mapelites;
pub mod mode;
pub mod moretry;
pub mod pgpe;
pub mod retry;
pub mod rng;

pub use biteopt::{
    BiteOpt, BiteParams, BiteResult, DeepBiteOpt, optimize_bite, validate_bite_inputs,
};
pub use cmaes::{AcmaResult, Cmaes, CmaesParams};
pub use crfmnes::{Crfmnes, CrfmnesParams, CrfmnesResult};
pub use da::{DaParams, DaResult, optimize_da};
pub use de::{De, DeParams, DeResult};
pub use fitness::{Fitness, NAN_REPLACEMENT, Objective};
pub use mapelites::{
    Archive, DiversifierParams, MapElitesParams, QdFitness, diversify, map_elites,
};
pub use mode::{Mode, ModeParams, ModeResult};
pub use moretry::{
    MoRetryConfig, MoRetryEntry, MoRetryResult, MultiObjective, WeightedObjective, moretry,
    pareto_indices, scalarize,
};
pub use pgpe::{Pgpe, PgpeParams, PgpeResult};
pub use retry::{
    AdvancedRetryConfig, RetryBounds, RetryConfig, RetryContext, RetryEntry, RetryImprovement,
    RetryResult, RetryRunResult, advanced_retry, retry,
};
pub use rng::Rng;

/// Version string of the core crate, surfaced through the Python build-info.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Sum of a slice of f64 — the Phase 0 probe used to prove the
/// Python → PyO3 → core call path is wired end to end.
pub fn probe_sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_sum_adds_values() {
        assert_eq!(probe_sum(&[1.0, 2.0, 3.5]), 6.5);
    }

    #[test]
    fn probe_sum_empty_is_zero() {
        assert_eq!(probe_sum(&[]), 0.0);
    }
}
