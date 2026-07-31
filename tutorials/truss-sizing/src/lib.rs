//! Native 2-D truss topology and catalogue-section sizing.

pub mod archive_grid;
pub mod artifacts;
pub mod catalogue;
pub mod config;
pub mod decode;
pub mod evaluate;
pub mod fem;
pub mod ground;
pub mod mo;
pub mod pilot;
pub mod qd;
pub mod so;

/// Finite penalty transported through optimizers when physics is unavailable.
pub const INVALID_COST: f64 = 1.0e12;
