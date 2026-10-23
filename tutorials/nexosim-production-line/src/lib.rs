//! NeXosim production-line digital twin optimized with MODE and MAP-Elites.

/// Optional models for durable optimization jobs.
#[cfg(feature = "job-worker")]
pub mod job_models;

pub mod model;
pub mod optimization;

pub use model::{Design, Metrics, OBJECTIVES, simulate};
pub use optimization::{
    OptimizationConfig, OptimizationResult, ParallelStrategy, QdOptions, QdResult, optimize,
    optimize_qd, write_mode_artifacts, write_qd_artifacts,
};
