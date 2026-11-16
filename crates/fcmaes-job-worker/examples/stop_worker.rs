//! A plain sleeping objective for the cooperative cancellation CLI regression.
use fcmaes_job_worker::{
    Adapter,
    registry::{Registry, Scalar},
};
use serde_json::json;
use std::time::Duration;
fn main() {
    let mut registry = Registry::new();
    registry.scalar(
        "app.sleep",
        "1",
        json!({"purpose":"plain-model cancellation test"}),
        |model| {
            model.require_dimension(1..=64)?;
            let milliseconds: u64 = model.parameter("milliseconds", 20)?;
            if !(1..=100).contains(&milliseconds) {
                return Err("milliseconds must be 1..=100".into());
            }
            Ok(Scalar::new(move |x: &[f64]| {
                std::thread::sleep(Duration::from_millis(milliseconds));
                x.iter().map(|v| v * v).sum()
            }))
        },
    );
    optimization_job_api::worker::run(Adapter::new(registry));
}
