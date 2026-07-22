mod common;

use std::error::Error;
use std::time::Instant;

use common::{Options, optimize};
use fcmaes_examples::lotka;

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let lower = vec![-1.0; lotka::DIMENSION];
    let upper = vec![1.0; lotka::DIMENSION];
    let started = Instant::now();
    let (x, value, evaluations) = optimize(&lower, &upper, &options, lotka::fitness);
    println!(
        "lotka dimension={} evaluations={evaluations} fitness={value:.12e} kill_times={:?} seconds={:.6}",
        lotka::DIMENSION,
        lotka::kill_times(&x),
        started.elapsed().as_secs_f64()
    );
    println!("x={x:?}");
    Ok(())
}
