//! Native inverse atmospheric-dispersion tutorial for fcmaes-rust.
//!
//! The Gaussian-plume and Briggs plume-rise equations are adapted from
//! `joshuanunn/really-simple-dispersion-wasm`. Browser bindings, random weather
//! generation, full-grid rendering and PNG encoding are deliberately excluded
//! from objective evaluation.

mod model;
mod optimize;
mod output;

pub use model::{
    DESIGN_NAMES, DIMENSION, Dataset, Design, Metrics, OBJECTIVES, Observation, Sensor, Source,
    Split, Weather, concentration_ug_m3, evaluate_training, evaluate_validation, lower_bounds,
    multi_objective, qd_objective, scalar_objective, upper_bounds,
};
pub use optimize::{
    MoProgress, MultiOptions, MultiOutcome, ParetoPoint, QdOptions, QdOutcome, QdPoint, QdProgress,
    ScalarOptions, ScalarOutcome, optimize_multi, optimize_qd, optimize_scalar,
};
pub use output::{write_multi_artifacts, write_qd_artifacts, write_scalar_artifacts};
