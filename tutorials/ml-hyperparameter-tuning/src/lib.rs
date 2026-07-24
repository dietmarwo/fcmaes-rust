//! Validation-aware ML hyperparameter optimization with fcmaes-rust.
//!
//! The tutorial keeps model fitting in Rust, uses fixed cross-validation folds
//! during optimization, selects shortlisted configurations on disjoint data,
//! and reserves a final test set for the separately invoked finalization stage.

pub mod benchmark;
pub mod data;
pub mod metrics;
pub mod model;
pub mod objective;
pub mod optimize;
pub mod protocol;
pub mod report;
pub mod space;

pub use benchmark::{BenchmarkOptions, BenchmarkOutcome, run_benchmark};
pub use data::{DataConfig, Dataset, Partition, Preset};
pub use metrics::Metrics;
pub use model::{FitOutcome, ProbabilityForest, TrainFailure};
pub use objective::{CandidateEvaluation, Evaluator, ValidationEvaluation};
pub use optimize::{
    BaselineMethod, BaselineOptions, BaselineOutcome, MultiOptions, MultiOutcome, QdOptions,
    QdOutcome, ScalarOptions, ScalarOutcome, optimize_baseline, optimize_multi, optimize_qd,
    optimize_scalar,
};
pub use protocol::{FinalStudyPlan, FinalStudyResult, finalize_study};
pub use report::{
    write_baseline_artifacts, write_benchmark_artifacts, write_final_artifacts,
    write_multi_artifacts, write_qd_artifacts, write_scalar_artifacts,
};
pub use space::{Criterion, DIMENSION, ForestConfig, decode};
