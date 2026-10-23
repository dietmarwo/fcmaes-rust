//! Durable application worker; the numerical model stays in the tutorial library.
fn main() {
    optimization_job_api::worker::run(fcmaes_job_worker::Adapter::new(
        rapier_trebuchet::job_models::registry(),
    ));
}
