//! L4: MODE receives finite objective and explicit constraint columns.

use fcmaes_core::{Fitness, Mode, ModeParams, NAN_REPLACEMENT};

pub fn run(_workers: i32) -> Result<String, String> {
    let fitness = Fitness::bounded(1, 2, &[-2.0], &[3.0]);
    let mut mode = Mode::try_new(
        fitness,
        1,
        1,
        None,
        &ModeParams {
            popsize: 32,
            seed: 42,
            ..Default::default()
        },
    )
    .map_err(str::to_owned)?;
    for _ in 0..20 {
        let decisions = mode.ask();
        let values: Vec<Vec<f64>> = decisions
            .iter()
            .map(|x| {
                let objective = if x[0].is_finite() {
                    x[0] * x[0]
                } else {
                    NAN_REPLACEMENT
                };
                vec![objective, 1.0 - x[0]]
            })
            .collect();
        mode.try_tell(&values).map_err(str::to_owned)?;
    }
    let result = mode.result();
    let best = result
        .y
        .iter()
        .filter(|row| row[1] <= 0.0)
        .min_by(|a, b| a[0].total_cmp(&b[0]));
    let best = best.ok_or_else(|| "MODE found no feasible point".to_owned())?;
    Ok(format!(
        "L4 explicit constraint | feasible=true objective={:.6e} violation={:.6e}\n",
        best[0], best[1]
    ))
}
