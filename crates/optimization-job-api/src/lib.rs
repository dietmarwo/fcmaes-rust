//! Versioned optimization jobs, durable journals, worker SDK and legacy compatibility.
pub mod legacy;
pub mod process;
pub mod result;
pub mod spec;
pub mod storage;
pub mod worker;
pub use spec::*;
pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
