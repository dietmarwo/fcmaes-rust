//! Circuit-design optimization models used by the tutorial binary.

pub mod artifacts;
pub mod decode;
pub mod features;
pub mod mo;
pub mod netlist;
pub mod qd;
pub mod so;

/// Decision-space dimension shared by the band-pass SO and QD formulations.
pub const BANDPASS_DIMENSION: usize = 5;

/// Publication AC-grid size for Q-sensitive scalar tuning.
pub const PUBLICATION_SO_POINTS: usize = 201;

/// Publication AC-grid size for pass-band-ripple optimization.
pub const PUBLICATION_MO_POINTS: usize = 201;

/// Publication AC-grid size for the already-converged QD descriptors.
pub const PUBLICATION_QD_POINTS: usize = 41;

/// Convert a simulator or feature-extraction failure into a finite optimizer cost.
pub const INVALID_COST: f64 = 1.0e12;
