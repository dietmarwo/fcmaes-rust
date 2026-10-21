//! Damped-oscillator calibration: a fallible, checkpointed Rust simulation.
//! Every scalar profile accepts it: `fcmaes.surrogate/1` (its default),
//! `fcmaes.retry/1` and `fcmaes.evaluate/1`.
use fcmaes_job_worker::{
    Adapter,
    registry::{Registry, Scalar},
    simulation::SimContext,
};
use rayon::prelude::*;
use serde_json::json;

fn rhs(y: [f64; 2], damping: f64, stiffness: f64) -> [f64; 2] {
    [y[1], -damping * y[1] - stiffness * y[0]]
}
fn advance(y: [f64; 2], h: f64, c: f64, k: f64) -> [f64; 2] {
    let a = rhs(y, c, k);
    let b = rhs([y[0] + h * a[0] / 2.0, y[1] + h * a[1] / 2.0], c, k);
    let d = rhs([y[0] + h * b[0] / 2.0, y[1] + h * b[1] / 2.0], c, k);
    let e = rhs([y[0] + h * d[0], y[1] + h * d[1]], c, k);
    [
        y[0] + h * (a[0] + 2.0 * b[0] + 2.0 * d[0] + e[0]) / 6.0,
        y[1] + h * (a[1] + 2.0 * b[1] + 2.0 * d[1] + e[1]) / 6.0,
    ]
}
fn replica(
    x: &[f64],
    ctx: &SimContext,
    steps: usize,
    index: usize,
    pause_ms: u64,
) -> Result<f64, String> {
    let initial = 1.0 + 0.1 * index as f64;
    let mut state = [initial, 0.0];
    let h = 4.0 / steps as f64;
    let mut loss = 0.0;
    for i in 0..steps {
        if i.is_multiple_of(64) {
            ctx.checkpoint()?;
            if pause_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(pause_ms));
            }
        }
        state = advance(state, h, x[0], x[1]);
        if (i + 1).is_multiple_of((steps / 8).max(1)) {
            let t = (i + 1) as f64 * h;
            let a = 0.2_f64;
            let omega = (2.3 - a * a).sqrt();
            let target =
                initial * (-a * t).exp() * ((omega * t).cos() + a / omega * (omega * t).sin());
            loss += (state[0] - target).powi(2);
        }
    }
    Ok(loss / 8.0)
}
/// The benchmarks plus the oscillator simulation.
pub fn registry() -> Registry {
    let mut registry = Registry::benchmarks();
    let about = json!({
        "simulation": true,
        "dimensions": 2,
        "parameters": {
            "steps": "default 2048, multiple of 8",
            "replicas": "default 4",
            "pause_ms": "default 0; optional synthetic latency",
        },
        "objective": "RK4 trajectory squared error; target damping 0.4, stiffness 2.3",
    });
    registry.scalar("app.damped-oscillator", "1", about, |model| {
        model.require_dimension(2..=2)?;
        let steps: usize = model.parameter("steps", 2048)?;
        let replicas: usize = model.parameter("replicas", 4)?;
        let pause_ms: u64 = model.parameter("pause_ms", 0)?;
        if !(64..=1_000_000).contains(&steps)
            || !steps.is_multiple_of(8)
            || !(1..=64).contains(&replicas)
            || pause_ms > 10
        {
            return Err("steps 64..1e6 multiple of 8, replicas 1..64, pause_ms 0..10".into());
        }
        let simulate = move |x: &[f64], ctx: &SimContext| -> Result<f64, String> {
            ctx.checkpoint()?;
            let run = |i| replica(x, ctx, steps, i, pause_ms);
            let values: Vec<Result<f64, String>> = if ctx.threads() > 1 {
                (0..replicas).into_par_iter().map(run).collect()
            } else {
                (0..replicas).map(run).collect()
            };
            // Ordered collection and serial reduction, so any pool size gives
            // the same value.
            let mut total = 0.0;
            for value in values {
                total += value?;
            }
            Ok(total / replicas as f64)
        };
        Ok(Scalar::simulation(simulate).domain(vec![0.05, 0.5], vec![1.5, 4.0]))
    });
    registry
}

fn main() {
    optimization_job_api::worker::run(Adapter::new(registry()));
}
