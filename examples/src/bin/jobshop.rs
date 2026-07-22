mod common;

use std::error::Error;
use std::time::Instant;

use common::{Options, optimize};
use fcmaes_examples::jobshop::JobShop;

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let problem = options
        .data
        .as_ref()
        .map(JobShop::from_path)
        .transpose()?
        .unwrap_or_else(JobShop::mini);
    let lower = vec![1e-7; problem.dimension()];
    let upper = vec![0.999_999_9; problem.dimension()];
    let started = Instant::now();
    let (x, value, evaluations) = optimize(&lower, &upper, &options, |x| problem.fitness(x));
    println!(
        "jobshop jobs={} machines={} operations={} evaluations={} fitness={value:.9} objectives={:?} seconds={:.6}",
        problem.jobs,
        problem.machines,
        problem.operations.len(),
        evaluations,
        problem.evaluate(&x),
        started.elapsed().as_secs_f64()
    );
    println!("x={x:?}");
    Ok(())
}
