//! Measured parallelism-ownership benchmark.

use std::error::Error;
use std::path::Path;
use std::time::Instant;

use epanet_rs::model::network::Network;
use epanet_rs::simulation::Simulation;
use fcmaes_core::parallel_batch;

/// One equal-work arrangement.
#[derive(Clone, Debug)]
pub struct BenchmarkRow {
    pub arrangement: &'static str,
    pub candidates: usize,
    pub workers: i32,
    pub internal_parallel: bool,
    pub wall_seconds: f64,
    pub candidates_per_second: f64,
    pub checksum: f64,
}

fn load_variant() -> Result<Network, Box<dyn Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("network")
        .join("benchmark-zone.inp");
    Ok(Network::from_file(
        path.to_str().ok_or("invalid network path")?,
    )?)
}

fn solve(mut network: Network, multiplier: f64, parallel: bool) -> Result<f64, String> {
    network.options.demand_multiplier = multiplier;
    let mut simulation = Simulation::new(network);
    let result = simulation
        .solve_hydraulics(parallel)
        .map_err(|error| error.to_string())?;
    Ok(result
        .heads
        .iter()
        .flat_map(|step| step.iter())
        .copied()
        .sum())
}

/// Compare candidate-owned and EPS-owned parallelism at equal total work.
pub fn run(candidates: usize, workers: i32) -> Result<Vec<BenchmarkRow>, Box<dyn Error>> {
    let base = load_variant()?;
    let multipliers = (0..candidates)
        .map(|index| 0.8 + 0.4 * index as f64 / candidates.max(1) as f64)
        .collect::<Vec<_>>();
    let started = Instant::now();
    let candidate_results = parallel_batch(&multipliers, workers, {
        let base = base.clone();
        move |multiplier| solve(base.clone(), *multiplier, false)
    });
    let candidate_seconds = started.elapsed().as_secs_f64();
    let candidate_checksum = candidate_results
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<f64>();
    let started = Instant::now();
    let mut internal_checksum = 0.0;
    for multiplier in &multipliers {
        internal_checksum += solve(base.clone(), *multiplier, true)?;
    }
    if (candidate_checksum - internal_checksum).abs() > 1e-8 * candidate_checksum.abs().max(1.0) {
        return Err("parallel arrangements produced different hydraulic checksums".into());
    }
    let internal_seconds = started.elapsed().as_secs_f64();
    Ok(vec![
        BenchmarkRow {
            arrangement: "candidate_parallel",
            candidates,
            workers,
            internal_parallel: false,
            wall_seconds: candidate_seconds,
            candidates_per_second: candidates as f64 / candidate_seconds,
            checksum: candidate_checksum,
        },
        BenchmarkRow {
            arrangement: "internal_eps_parallel",
            candidates,
            workers,
            internal_parallel: true,
            wall_seconds: internal_seconds,
            candidates_per_second: candidates as f64 / internal_seconds,
            checksum: internal_checksum,
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrangements_are_numerically_equivalent() {
        let rows = run(4, 2).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(
            (rows[0].checksum - rows[1].checksum).abs() < 1e-8 * rows[0].checksum.abs().max(1.0)
        );
    }
}
