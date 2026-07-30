//! Quantized phased-array beam synthesis with native Rust evaluation.

pub mod archive_grid;
pub mod array;
pub mod artifacts;
pub mod config;
pub mod decode;
pub mod geometry;
pub mod kernel;
pub mod metrics;
pub mod mo;
pub mod pilot;
pub mod qd;
pub mod scenarios;
pub mod so;

/// Finite replacement used only when an optimizer API cannot accept infinity.
pub const INVALID_COST: f64 = 1.0e12;

/// Calibrated failure constraint, comparable to the dB constraints.
pub const KERNEL_FAILURE_SCALE: f64 = 100.0;
