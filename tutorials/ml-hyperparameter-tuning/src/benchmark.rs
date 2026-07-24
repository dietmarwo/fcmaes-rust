use std::collections::BTreeSet;
use std::error::Error;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fcmaes_core::{Rng, parallel_batch};
use serde::{Deserialize, Serialize};

use crate::objective::Evaluator;
use crate::report::peak_rss_kib;
use crate::space::{DIMENSION, ForestConfig, decode, default_coordinates};

#[derive(Clone, Debug)]
pub struct BenchmarkOptions {
    pub candidates: usize,
    pub maximum_workers: usize,
    pub prediction_repetitions: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencySample {
    pub candidate_id: usize,
    pub config: ForestConfig,
    pub model_bytes: usize,
    pub structural_cost: u64,
    pub fit_seconds: f64,
    pub microseconds_per_row: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalingSample {
    pub workers: usize,
    pub candidates: usize,
    pub wall_seconds: f64,
    pub candidates_per_second: f64,
    pub peak_rss_kib: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct BenchmarkOutcome {
    pub latency: Vec<LatencySample>,
    pub scaling: Vec<ScalingSample>,
    pub elapsed: Duration,
}

pub fn run_benchmark(
    evaluator: Arc<Evaluator>,
    options: &BenchmarkOptions,
) -> Result<BenchmarkOutcome, Box<dyn Error>> {
    if options.candidates == 0
        || options.maximum_workers == 0
        || options.prediction_repetitions == 0
    {
        return Err("benchmark candidates, workers, and repetitions must be positive".into());
    }
    let started = Instant::now();
    let candidates = benchmark_candidates(options.candidates, options.seed);
    let mut latency = Vec::with_capacity(candidates.len());
    for (candidate_id, values) in candidates.iter().enumerate() {
        let config = decode(values)?;
        let fit_started = Instant::now();
        let model = match evaluator.forest.fit(
            &config,
            &evaluator.dataset.tuning.features,
            &evaluator.dataset.tuning.labels,
            crate::data::stream_seed(options.seed, 500 + candidate_id as u64),
        ) {
            Ok(model) => model,
            Err(_) => continue,
        };
        let fit_seconds = fit_started.elapsed().as_secs_f64();
        black_box(model.predict(&evaluator.dataset.selection.features)?);
        let prediction_started = Instant::now();
        for _ in 0..options.prediction_repetitions {
            black_box(model.predict(&evaluator.dataset.selection.features)?);
        }
        let rows = evaluator.dataset.selection.len() * options.prediction_repetitions;
        latency.push(LatencySample {
            candidate_id,
            config,
            model_bytes: model.serialized_bytes,
            structural_cost: model.structural_cost,
            fit_seconds,
            microseconds_per_row: 1.0e6 * prediction_started.elapsed().as_secs_f64() / rows as f64,
        });
    }
    if latency.len() < 2 {
        return Err("benchmark produced fewer than two valid latency samples".into());
    }

    let scaling_candidates: Vec<Vec<f64>> = (0..options.candidates.max(16))
        .map(|index| candidates[index % candidates.len()].clone())
        .collect();
    let mut worker_counts = BTreeSet::from([1, options.maximum_workers]);
    for workers in [2, 4, 8, 16] {
        if workers <= options.maximum_workers {
            worker_counts.insert(workers);
        }
    }
    let mut scaling = Vec::with_capacity(worker_counts.len());
    for workers in worker_counts {
        black_box(parallel_batch(
            &scaling_candidates[..1],
            workers as i32,
            |values| evaluator.evaluate(values).scalar_fitness,
        ));
        let wall_started = Instant::now();
        let values = parallel_batch(&scaling_candidates, workers as i32, |candidate| {
            evaluator.evaluate(candidate).scalar_fitness
        });
        black_box(values);
        let wall_seconds = wall_started.elapsed().as_secs_f64();
        scaling.push(ScalingSample {
            workers,
            candidates: scaling_candidates.len(),
            wall_seconds,
            candidates_per_second: scaling_candidates.len() as f64 / wall_seconds.max(1.0e-12),
            peak_rss_kib: peak_rss_kib(),
        });
    }
    Ok(BenchmarkOutcome {
        latency,
        scaling,
        elapsed: started.elapsed(),
    })
}

fn benchmark_candidates(count: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(count);
    result.push(default_coordinates().to_vec());
    let mut rng = Rng::new(seed);
    while result.len() < count {
        result.push((0..DIMENSION).map(|_| rng.uniform01()).collect());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataConfig, Dataset, Preset};

    #[test]
    fn tiny_benchmark_measures_latency_and_scaling() {
        let dataset = Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap());
        let evaluator = Arc::new(Evaluator::new(dataset, 0.1, 42));
        let outcome = run_benchmark(
            evaluator,
            &BenchmarkOptions {
                candidates: 2,
                maximum_workers: 2,
                prediction_repetitions: 1,
                seed: 42,
            },
        )
        .unwrap();
        assert!(outcome.latency.iter().all(|sample| {
            sample.microseconds_per_row > 0.0
                && sample.model_bytes > 0
                && sample.structural_cost > 0
        }));
        assert_eq!(outcome.scaling.len(), 2);
    }
}
