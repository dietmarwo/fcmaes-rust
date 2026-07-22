mod common;

use std::error::Error;
use std::time::Instant;

use common::{Options, optimize};
use fcmaes_examples::f8;

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let dimension: usize = options
        .positional
        .first()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(6);
    let lower = vec![0.0; dimension];
    let upper = vec![2.0; dimension];
    let started = Instant::now();
    let (x, value, evaluations) = optimize(&lower, &upper, &options, f8::fitness);
    println!(
        "f8 dimension={dimension} evaluations={evaluations} fitness={value:.12e} final_state={:?} seconds={:.6}",
        f8::final_state(&x)?,
        started.elapsed().as_secs_f64()
    );
    println!("x={x:?}");
    Ok(())
}
