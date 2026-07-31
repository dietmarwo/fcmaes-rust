//! L3: run-id-derived seeds make parallel scheduling irrelevant.

use fcmaes_core::{De, DeParams, Fitness, parallel_batch};

fn runs(workers: i32) -> Vec<f64> {
    let ids: Vec<u64> = (0..8).collect();
    parallel_batch(&ids, workers, |run_id| {
        let objective = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();
        let fitness = Fitness::bounded(4, 1, &[-5.0; 4], &[5.0; 4]);
        De::new(
            fitness,
            &[],
            &[],
            None,
            &DeParams {
                max_evaluations: 300,
                seed: 42_u64.wrapping_add(*run_id * 1_000_003),
                ..Default::default()
            },
        )
        .optimize(&objective)
        .y
    })
}

pub fn run(workers: i32) -> Result<String, String> {
    let serial = runs(1);
    let parallel = runs(workers.max(2));
    if serial != parallel {
        return Err("run-id seeds changed with worker count".to_owned());
    }
    Ok(format!(
        "L3 deterministic workers | runs={} identical=true best={:.6e}\n",
        serial.len(),
        serial.iter().copied().fold(f64::INFINITY, f64::min)
    ))
}
