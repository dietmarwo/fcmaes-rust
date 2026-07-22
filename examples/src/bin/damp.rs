mod common;

use std::error::Error;
use std::time::Instant;

use common::{Options, optimize};
use fcmaes_examples::damp;

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let dimension: usize = options
        .positional
        .first()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(12);
    if !dimension.is_multiple_of(2) {
        return Err("damp dimension must be positive and even".into());
    }
    let lower = vec![0.0; dimension];
    let upper = vec![1.0; dimension];
    let started = Instant::now();
    let (x, value, evaluations) = optimize(&lower, &upper, &options, damp::fitness);
    println!(
        "damp dimension={dimension} evaluations={evaluations} amplitude={value:.12e} descriptor={:?} seconds={:.6}",
        damp::descriptor(&x),
        started.elapsed().as_secs_f64()
    );
    println!("x={x:?}");
    Ok(())
}
