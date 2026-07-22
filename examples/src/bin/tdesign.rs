mod common;

use std::error::Error;
use std::f64::consts::PI;
use std::time::Instant;

use common::{Options, optimize};
use fcmaes_examples::tdesign;

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let points: usize = options
        .positional
        .first()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(10);
    let degree: usize = options
        .positional
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(4);
    let mut lower = vec![0.0; 3 * points];
    let mut upper = vec![PI; 3 * points];
    upper[points..2 * points].fill(2.0 * PI);
    upper[2 * points..].fill(2.0);
    for i in 0..points {
        lower[i] = 0.0;
        lower[points + i] = 0.0;
        lower[2 * points + i] = 0.0;
    }
    let started = Instant::now();
    let (x, value, evaluations) = optimize(&lower, &upper, &options, |x| {
        tdesign::weighted_objective(x, degree)
    });
    println!(
        "t-design points={points} degree={degree} evaluations={evaluations} error={value:.12e} seconds={:.6}",
        started.elapsed().as_secs_f64()
    );
    println!("x={x:?}");
    Ok(())
}
