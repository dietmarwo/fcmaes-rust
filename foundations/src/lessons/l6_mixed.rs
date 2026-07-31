//! L6: integer coordinates use physical `[0, k)`-style bounds.

use fcmaes_core::{Fitness, Mode, ModeParams};

pub fn run(_workers: i32) -> Result<String, String> {
    let fitness = Fitness::bounded(2, 1, &[0.0, 0.0], &[7.999_999, 1.0]);
    let mut mode = Mode::try_new(
        fitness,
        1,
        0,
        Some(vec![true, false]),
        &ModeParams {
            popsize: 32,
            seed: 42,
            ..Default::default()
        },
    )
    .map_err(str::to_owned)?;
    for _ in 0..15 {
        let decisions = mode.ask();
        let values = decisions
            .iter()
            .map(|x| {
                let category = x[0].round().clamp(0.0, 7.0);
                vec![(category - 3.0).powi(2) + (x[1] - 0.25).powi(2)]
            })
            .collect::<Vec<_>>();
        mode.try_tell(&values).map_err(str::to_owned)?;
    }
    let result = mode.result();
    let (index, value) = result
        .y
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a[0].total_cmp(&b[0]))
        .ok_or_else(|| "empty MODE population".to_owned())?;
    Ok(format!(
        "L6 mixed variables | integer_mask=true decoded_category={:.0} raw={:.6} continuous={:.6} objective={:.6e}\n",
        result.x[index][0].round().clamp(0.0, 7.0),
        result.x[index][0],
        result.x[index][1],
        value[0]
    ))
}
