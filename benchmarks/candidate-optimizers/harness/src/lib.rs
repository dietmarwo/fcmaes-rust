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
//!   (protocols S and R, and protocol T of rounds 2 and 3) and the raw TSV
//!   formats;
//! - [`round3`]: the round-3 problems, type tables, decoder probes and
//!   campaign sets;
//! - [`serve`]: the line protocol through which the round-3 candidate driver
//!   evaluates the harness objectives.

#![forbid(unsafe_code)]

pub mod bipop;
pub mod campaign;
pub mod examples;
pub mod problems;
pub mod round3;
pub mod seeds;
pub mod serve;
pub mod solvers;
