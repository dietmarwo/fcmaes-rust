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

/// An error with a stable, machine-readable code. The runner prints the code
/// next to the message; docs/job-api.md lists every code.
#[derive(Debug)]
pub struct Coded {
    pub code: &'static str,
    pub message: String,
}
impl std::fmt::Display for Coded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Coded {}
/// An [`Error`] carrying `code`.
pub fn coded(code: &'static str, message: impl Into<String>) -> Error {
    Box::new(Coded {
        code,
        message: message.into(),
    })
}
/// The code of `error`, `runner_error` if it has none.
pub fn error_code(error: &Error) -> &'static str {
    error
        .downcast_ref::<Coded>()
        .map_or("runner_error", |c| c.code)
}
/// True for error codes after which the identical command can succeed later
/// without any change: only `attempt_active` (wait for the processes of the
/// attempt to exit). Every other error needs a different request, a
/// re-registered worker or operator action.
pub fn retryable(code: &str) -> bool {
    code == "attempt_active"
}
