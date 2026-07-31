//! Candidate-throughput measurements for the native coverage kernel.

use std::hint::black_box;
use std::time::{Duration, Instant};

use fcmaes_core::{Rng, parallel_batch};

use crate::coverage::{GROUP_WEIGHT_EXPONENT, evaluate};
use crate::instance::Instance;

/// One serial or parallel throughput measurement.
#[derive(Clone, Debug)]
pub struct ThroughputResult {
    /// Instance label.
    pub instance: String,
    /// Candidate count.
    pub samples: usize,
    /// Resolved worker request.
    pub workers: i32,
    /// Elapsed wall time.
    pub elapsed: Duration,
    /// Candidate evaluations per second.
    pub candidates_per_second: f64,
    /// Ordinary-edge visits per second.
    pub edge_visits_per_second: f64,
    /// Native group-membership visits per second.
    pub group_memberships_per_second: f64,
    /// Value preventing dead-code elimination.
    pub checksum: f64,
}

fn candidates(nodes: usize, samples: usize, seed: u64) -> Vec<Vec<bool>> {
    let mut rng = Rng::new(seed);
    (0..samples)
        .map(|sample| {
            let probability = 0.05 + 0.90 * (sample as f64 + 0.5) / samples.max(1) as f64;
            (0..nodes)
                .map(|_| rng.uniform01() < probability)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Measure the physical kernel with a fixed candidate stream.
#[must_use]
pub fn measure(instance: &Instance, samples: usize, workers: i32, seed: u64) -> ThroughputResult {
    let selections = candidates(instance.nodes(), samples, seed);
    let started = Instant::now();
    let values = if workers == 1 {
        selections
            .iter()
            .map(|selected| {
                evaluate(instance, selected, GROUP_WEIGHT_EXPONENT)
                    .expect("candidate dimension is fixed")
                    .coverage
            })
            .collect::<Vec<_>>()
    } else {
        parallel_batch(&selections, workers, |selected| {
            evaluate(instance, selected, GROUP_WEIGHT_EXPONENT)
                .expect("candidate dimension is fixed")
                .coverage
        })
    };
    let elapsed = started.elapsed();
    let seconds = elapsed.as_secs_f64().max(1.0e-12);
    let rate = samples as f64 / seconds;
    let memberships = instance.groups.iter().map(Vec::len).sum::<usize>();
    ThroughputResult {
        instance: instance.metadata.name.clone(),
        samples,
        workers,
        elapsed,
        candidates_per_second: rate,
        edge_visits_per_second: rate * instance.edges.len() as f64,
        group_memberships_per_second: rate * memberships as f64,
        checksum: black_box(values.into_iter().sum()),
    }
}

/// Frozen, conservative gate for selecting the 4k publication fixture.
#[must_use]
pub fn select_publication_scale(reference_4k_serial: &ThroughputResult) -> &'static str {
    if reference_4k_serial.candidates_per_second >= 20_000.0 {
        "reference-4k"
    } else {
        "reference-1k"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{FIXTURES, generate};

    #[test]
    fn serial_and_parallel_checksums_match() {
        let instance = generate(&FIXTURES[1]).unwrap();
        let serial = measure(&instance, 64, 1, 42);
        let parallel = measure(&instance, 64, 2, 42);
        assert!((serial.checksum - parallel.checksum).abs() < 1.0e-9);
    }
}
