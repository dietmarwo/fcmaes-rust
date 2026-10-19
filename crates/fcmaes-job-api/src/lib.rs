//! Backwards-compatible facade; common code lives in optimization-job-api.
pub use optimization_job_api::{Error, Result, legacy::*, process, storage};
