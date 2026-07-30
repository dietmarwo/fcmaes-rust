//! Field-service routing with continuous random keys.

pub mod archive_grid;
pub mod artifacts;
pub mod baseline;
pub mod config;
pub mod decode;
pub mod evaluate;
pub mod instance;
pub mod mo;
pub mod pilot;
pub mod qd;
pub mod scenarios;
pub mod scorer2;
pub mod so;

/// Objective returned for malformed candidates.
pub const INVALID_OBJECTIVE: f64 = 1.0e99;
