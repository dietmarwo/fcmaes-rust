//! Bilevel energy-hub sizing tutorial.

#![deny(missing_docs)]

/// Optional models for durable optimization jobs.
#[cfg(feature = "job-worker")]
pub mod job_models;

pub mod annual;
pub mod archive_grid;
pub mod artifacts;
pub mod capex;
pub mod config;
pub mod decode;
pub mod dispatch;
pub mod evaluate;
pub mod landscape;
pub mod mo;
pub mod pilot;
pub mod profiles;
pub mod qd;
pub mod scenarios;
pub mod so;
