mod common;

use std::error::Error;
use std::time::Instant;

use common::{Options, optimize};
use fcmaes_examples::scheduling::{STATION_COUNT, Scheduler};

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let scheduler = options
        .data
        .as_ref()
        .map(Scheduler::from_path)
        .transpose()?
        .unwrap_or_else(Scheduler::sample);
    let dimension = scheduler.dimension();
    let lower = vec![1e-7; dimension];
    let mut upper = vec![0.999_999_9; dimension];
    upper[..10].fill(scheduler.trajectory_count as f64 - 0.000_01);
    let started = Instant::now();
    let (x, value, evaluations) = optimize(&lower, &upper, &options, |x| scheduler.fitness(x));
    let result = scheduler.evaluate(&x);
    println!(
        "scheduling transfers={} trajectories={} stations={} evaluations={} fitness={value:.9} score={:.9} shaped_mass={:.6e} objectives={:?} seconds={:.6}",
        scheduler.transfer_count(),
        scheduler.trajectory_count,
        STATION_COUNT,
        evaluations,
        scheduler.score(&x),
        result.shaped_mass,
        scheduler.objectives(&x),
        started.elapsed().as_secs_f64()
    );
    println!("x={x:?}");
    Ok(())
}
