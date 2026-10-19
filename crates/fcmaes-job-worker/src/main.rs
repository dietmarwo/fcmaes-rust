//! Common job-protocol adapter. The original fcmaes-native-worker is preserved.
#![forbid(unsafe_code)]
fn main() {
    optimization_job_api::worker::run(fcmaes_job_worker::Adapter);
}
