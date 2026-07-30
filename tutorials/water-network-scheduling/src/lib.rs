//! Water-network pump scheduling with explicit hydraulic, control and artifact contracts.

pub mod archive_grid;
pub mod artifacts;
pub mod bench;
pub mod config;
pub mod decode;
pub mod driver;
pub mod energy;
pub mod evaluate;
pub mod mo;
pub mod network;
pub mod pilot;
pub mod qd;
pub mod scenarios;
pub mod so;

/// Decision-vector dimension.
pub const DIMENSION: usize = 28;
/// Large but finite score for invalid candidates.
pub const INVALID_OBJECTIVE: f64 = 1.0e12;
