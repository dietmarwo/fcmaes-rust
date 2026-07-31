//! L2: equal-budget optimizer comparison on Rastrigin-10.

use fcmaes_core::{
    BiteOpt, BiteParams, Cmaes, CmaesParams, Crfmnes, CrfmnesParams, De, DeParams, Fitness,
};

pub fn run(_workers: i32) -> Result<String, String> {
    let objective = |x: &[f64]| {
        10.0 * x.len() as f64
            + x.iter()
                .map(|value| value * value - 10.0 * (2.0 * std::f64::consts::PI * value).cos())
                .sum::<f64>()
    };
    let lower = vec![-5.12; 10];
    let upper = vec![5.12; 10];
    let fitness = || Fitness::bounded(10, 1, &lower, &upper);
    let budget = 1_500;
    let cma = Cmaes::new(
        fitness(),
        &[2.5; 10],
        &[1.5],
        &CmaesParams {
            max_evaluations: budget,
            seed: 42,
            ..Default::default()
        },
    )
    .optimize(&objective, 1)
    .y;
    let de = De::new(
        fitness(),
        &[],
        &[],
        None,
        &DeParams {
            max_evaluations: budget,
            seed: 42,
            ..Default::default()
        },
    )
    .optimize(&objective)
    .y;
    let bite = BiteOpt::new(
        &lower,
        &upper,
        None,
        &BiteParams {
            max_evaluations: budget,
            seed: 42,
            ..Default::default()
        },
    )
    .optimize(&objective)
    .y;
    let mut cr = Crfmnes::new(
        fitness(),
        &[2.5; 10],
        1.5,
        &CrfmnesParams {
            max_evaluations: budget,
            seed: 42,
            ..Default::default()
        },
    );
    let cr = cr
        .optimize_batch(|xs| xs.iter().map(|x| objective(x)).collect())
        .y;
    Ok(format!(
        "L2 equal budget | cma={cma:.6e} de={de:.6e} bite={bite:.6e} crfmnes={cr:.6e}\n"
    ))
}
