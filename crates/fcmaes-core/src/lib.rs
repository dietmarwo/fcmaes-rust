#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
// Public entry points must state how they fail. Optimizer misuse (wrong batch
// length, ask/tell out of order, mismatched bounds) is the most common
// integration error, so the panicking and fallible variants both document
// their contract.
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]

pub mod biteopt;
pub mod cmaes;
pub mod crfmnes;
pub mod da;
pub mod de;
pub mod fitness;
pub mod indicators;
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
pub use fitness::{Fitness, NAN_REPLACEMENT, Objective, parallel_batch};
pub use indicators::{
    HypervolumeEstimate, HypervolumeReport, IndicatorError, OutsidePolicy, ReferencePoint,
    additive_epsilon, crowding_distance, gd, gd_plus, hypervolume, hypervolume_monte_carlo,
    hypervolume_with, igd, igd_plus, nondominated_sort, spacing, spread,
};
pub use mapelites::{
    Archive, DiversifierParams, GridLayout, MapElitesParams, QdBatchFitness, QdFitness, diversify,
    diversify_batch, map_elites, map_elites_batch, map_elites_batch_with_progress,
};
pub use mode::{Mode, ModeParams, ModeResult};
pub use moretry::{
    MoRetryConfig, MoRetryEntry, MoRetryResult, MultiObjective, WeightedObjective, moretry,
    pareto_indices, scalarize,
};
pub use pgpe::{Pgpe, PgpeParams, PgpeResult};
pub use retry::{
    AdvancedRetryConfig, RetryBounds, RetryConfig, RetryContext, RetryEntry, RetryImprovement,
    RetryResult, RetryRunResult, advanced_retry, retry, retry_run_seed,
};
pub use rng::Rng;

/// Version string of the core crate, surfaced through the Python build-info.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Sum a slice of `f64`.
///
/// This exists as an installation probe: it is the smallest call that proves
/// the Python → PyO3 → Rust path is wired end to end, and it backs
/// `fcmaes_rust._fcmaes_ext._phase1_probe_sum`. It is not a numerical
/// reduction API — it performs no compensated summation and applications
/// should use [`Iterator::sum`] directly.
///
/// # Examples
///
/// ```
/// assert_eq!(fcmaes_core::probe_sum(&[1.0, 2.0, 3.5]), 6.5);
/// assert_eq!(fcmaes_core::probe_sum(&[]), 0.0);
/// ```
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
