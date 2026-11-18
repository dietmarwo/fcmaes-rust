//! Adaptive differential evolution candidates: L-SHADE, jSO and L-SRTDE.
//!
//! Each solver implements [`solver_api::Solver`], evaluates serially, never
//! exceeds the evaluation budget, keeps every evaluated point inside the box,
//! is deterministic for a given seed, and treats NaN or infinite objective
//! values as worse than every finite value. All randomness comes from
//! [`fcmaes_core::Rng`]. Parameters are public fields whose defaults are the
//! published values; each module documents its deviations and the details
//! its sources leave open.

#![forbid(unsafe_code)]

mod common;
mod jso;
mod lshade;
mod lsrtde;

pub use jso::Jso;
pub use lshade::LShade;
pub use lsrtde::LSrtde;
