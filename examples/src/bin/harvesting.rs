mod common;

use std::error::Error;
use std::time::Instant;

use common::{Options, optimize};
use fcmaes_examples::{harvesting, jobshop::JobShop};

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let problem = options
        .data
        .as_ref()
        .map(JobShop::from_path)
        .transpose()?
        .unwrap_or_else(JobShop::mini);
    let max_active = options
        .positional
        .first()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(4usize)
        .min(problem.machines);
    let dimension = problem.dimension() + 2 * problem.machines + 1;
    let lower = vec![1e-7; dimension];
    let mut upper = vec![0.999_999_9; dimension];
    upper[dimension - 1] = problem.sum_time;
    let started = Instant::now();
    let (x, value, evaluations) = optimize(&lower, &upper, &options, |x| {
        harvesting::fitness(&problem, x, max_active)
    });
    println!(
        "harvesting jobs={} machines={} max_active={} evaluations={} fitness={value:.9} objectives={:?} seconds={:.6}",
        problem.jobs,
        problem.machines,
        max_active,
        evaluations,
        harvesting::evaluate(&problem, &x, max_active),
        started.elapsed().as_secs_f64()
    );
    println!("x={x:?}");
    Ok(())
}
