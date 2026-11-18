//! Benchmark harness for the candidate-optimizer inclusion experiment.
//!
//! The binding rules live in `PROTOCOL.md` of this workspace. This crate
//! provides the pieces the campaign needs:
//!
//! - [`seeds`]: the SplitMix64-style seed derivation shared by all solvers;
//! - [`solvers`]: adapters for the six `fcmaes-core` incumbents, the
//!   BIPOP-aCMA-ES restart schedule ([`bipop`]) and, with the `candidates`
//!   feature, the four DE/EDA candidates and the round-2 sequences
//!   jSO→CMA and L-SRTDE→CMA;
//! - [`problems`]: the thirteen real-world problems of protocol section 3.2
//!   and the two problems examined for round 2 (`ROUND2.md`);
//! - [`campaign`]: budget measurement, the resumable campaign runner
//!   (protocols S and R, and protocol T of round 2) and the raw TSV formats.

#![forbid(unsafe_code)]

pub mod bipop;
pub mod campaign;
pub mod problems;
pub mod seeds;
pub mod solvers;
