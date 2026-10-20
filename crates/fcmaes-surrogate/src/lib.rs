//! Small-budget, deterministic Bayesian optimization of continuous boxes.
//!
//! Version 0.1 implements exact RBF/Matérn-5/2 GPs with an analytic-gradient
//! bounded L-BFGS MAP fit, LogEI or LCB maximized by coordinate refinement of
//! the best candidates, a TuRBO-style trust region around the incumbent for
//! two of every three proposals, and synchronous Kriging-believer batches. It uses
//! the current Rayon pool and constructs no per-run thread pools. Callers
//! must install a bounded pool; otherwise Rayon's default global pool applies.
//! The numerical library has no dependency on the optimization job API.
#![forbid(unsafe_code)]

pub mod design;
pub mod driver;
pub mod gp;
mod linalg;
mod optim;
pub mod problems;
pub mod space;
pub mod special;
pub mod strategy;
pub mod transform;

pub use driver::{Event, Observation, RunResult, RunSpec, Value, run};
pub use space::Space;
pub use strategy::{Acquisition, Config, Proposal, Round, StrategyState, propose};

use std::fmt;

pub const MAX_DIMENSION: usize = 32;
pub const MAX_OBSERVATIONS: usize = 256;
pub const MAX_BATCH: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Invalid(String),
    Numerical(String),
    Log(String),
    Stopped,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid surrogate input: {s}"),
            Self::Numerical(s) => write!(f, "surrogate numerical failure: {s}"),
            Self::Log(s) => write!(f, "surrogate log failure: {s}"),
            Self::Stopped => f.write_str("surrogate stopped"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

/// Stable independent streams, derived without scheduling-dependent draws.
pub fn stream_seed(seed: u64, purpose: u64, index: u64) -> u64 {
    let mut z =
        seed ^ purpose.wrapping_mul(0x9e3779b97f4a7c15) ^ index.wrapping_mul(0xd1b54a32d192ed03);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
