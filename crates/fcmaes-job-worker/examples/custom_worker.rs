//! An application worker: the built-in benchmarks plus one model of its own,
//! an exponential decay `y = a * exp(-b * t) + c` fitted to a data file.
//!
//! The job names the model, its parameters and its data input; the data
//! reaches the model as SHA-256-checked bytes, and the worker records the
//! parameters it used (defaults included) in the resolved job:
//!
//! ```json
//! "payload": {"problem": {"id": "app.exponential-fit", "version": "1",
//!     "config": {"dimension": 3},
//!     "bounds": {"lower": [0, 0, -10], "upper": [10, 5, 10], "broadcast": false},
//!     "parameters": {"loss": "absolute"}}, ...},
//! "inputs": [{"name": "data", "file": "decay.csv", "sha256": "..."}]
//! ```
//!
//! Build it and register it with the runner like the stock worker:
//!
//! ```text
//! cargo build --release -p fcmaes-job-worker --example custom_worker
//! optimization-run workers add --id curve-fit --executable target/release/examples/custom_worker
//! ```
//!
//! `tutorials/job-api` runs it on `decay.csv` (`exponential-fit.json`). The
//! registered [`Example`] makes `describe` advertise a runnable job with that
//! data inline, so a client without this repository can build one.
use fcmaes_job_worker::{
    Adapter,
    registry::{Example, Instance, Registry, Scalar},
};
use optimization_job_api::Result;
use serde_json::json;

/// `(t, y)` rows of a CSV file; a first line that is not numeric is a header.
fn samples(bytes: &[u8]) -> Result<Vec<(f64, f64)>> {
    let text = std::str::from_utf8(bytes)?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let parsed = match fields.as_slice() {
            [t, y] => t.parse::<f64>().ok().zip(y.parse::<f64>().ok()),
            _ => None,
        };
        match parsed {
            Some((t, y)) if t.is_finite() && y.is_finite() => rows.push((t, y)),
            None if i == 0 => (), // header
            _ => {
                return Err(format!("data line {}: expected two finite numbers t,y", i + 1).into());
            }
        }
    }
    if rows.len() < 3 {
        return Err("data needs at least three rows".into());
    }
    Ok(rows)
}

fn exponential_fit(model: &Instance) -> Result<Scalar> {
    model.require_dimension(3..=3)?;
    let loss: String = model.parameter("loss", "squared".to_string())?;
    let absolute = match loss.as_str() {
        "squared" => false,
        "absolute" => true,
        _ => return Err("loss must be squared or absolute".into()),
    };
    let data = samples(&model.input("data")?)?;
    Ok(Scalar::new(move |x: &[f64]| {
        let (a, b, c) = (x[0], x[1], x[2]);
        data.iter()
            .map(|(t, y)| {
                let r = a * (-b * t).exp() + c - y;
                if absolute { r.abs() } else { r * r }
            })
            .sum()
    }))
}

/// The data of the tutorial's job: `a = 3, b = 0.7, c = 0.5` plus noise.
const DECAY_CSV: &str = include_str!("../../../tutorials/job-api/decay.csv");

/// The benchmarks plus the application's model.
pub fn registry() -> Registry {
    let mut registry = Registry::benchmarks();
    registry
        .scalar(
            "app.exponential-fit",
            "1",
            json!({"dimensions":3,"variables":["a","b","c"],
                "objective":"sum over the data rows of the loss of a*exp(-b*t) + c - y",
                "parameters":{"loss":"squared (default) or absolute"},
                "inputs":{"data":"CSV rows t,y; an optional header line"}}),
            exponential_fit,
        )
        // The model reads a data file: describe can only build it, and
        // advertise a job on it, with an example of that file.
        .example(
            "app.exponential-fit",
            Example::new(3)
                .bounds(vec![0.0, 0.0, -10.0], vec![10.0, 5.0, 10.0])
                .parameter("loss", "squared")
                .input_text("data", "decay.csv", DECAY_CSV),
        );
    registry
}

fn main() {
    optimization_job_api::worker::run(Adapter::new(registry()));
}
