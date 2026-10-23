//! Durable application worker; the numerical model stays in the tutorial library.
fn main() {
    optimization_job_api::worker::run(fcmaes_job_worker::Adapter::new(
        nexosim_production_line::job_models::registry(),
    ));
}
