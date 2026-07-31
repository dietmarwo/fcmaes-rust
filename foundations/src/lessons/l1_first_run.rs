//! L1: one bounded objective and deterministic retry.

use fcmaes_core::{De, DeParams, Fitness, RetryBounds, RetryConfig, RetryRunResult, retry};

pub fn run(_workers: i32) -> Result<String, String> {
    let objective = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();
    let bounds = RetryBounds::new(vec![-5.0; 3], vec![5.0; 3]).map_err(str::to_owned)?;
    let config = RetryConfig {
        num_retries: 3,
        workers: 1,
        max_evaluations: 400,
        seed: 42,
        ..Default::default()
    };
    let result = retry(&objective, &bounds, &config, |function, context| {
        let fitness = Fitness::bounded(3, 1, context.bounds.lower(), context.bounds.upper());
        let parameters = DeParams {
            max_evaluations: context.max_evaluations,
            seed: context.seed,
            ..Default::default()
        };
        let outcome = De::new(fitness, &[], &[], None, &parameters).optimize(function);
        RetryRunResult {
            x: outcome.x,
            y: outcome.y,
            evaluations: outcome.evaluations,
        }
    });
    Ok(format!(
        "L1 first run | retries={} evaluations={} best={:.6e}\n",
        result.runs, result.evaluations, result.y
    ))
}
