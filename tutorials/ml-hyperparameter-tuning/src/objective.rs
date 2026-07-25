use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::data::{Dataset, Partition, stream_seed};
use crate::metrics::{Metrics, mean_and_sdev};
use crate::model::{ProbabilityForest, TrainFailure};
use crate::space::{ForestConfig, decode};

pub const LOG_LOSS_CEILING: f64 = 13.815_510_557_964_274;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub x: Vec<f64>,
    pub config: Option<ForestConfig>,
    pub metrics: Option<Metrics>,
    pub mean_model_bytes: f64,
    pub mean_structural_cost: f64,
    pub estimated_structural_cost: u64,
    pub model_fits: usize,
    pub trees_fitted: usize,
    pub elapsed_seconds: f64,
    pub recall_violation: f64,
    pub structural_violation: f64,
    pub scalar_fitness: f64,
    pub failure: Option<TrainFailure>,
}

impl CandidateEvaluation {
    pub fn feasible(&self) -> bool {
        self.failure.is_none() && self.recall_violation <= 0.0 && self.structural_violation <= 0.0
    }

    pub fn mode_values(&self) -> Vec<f64> {
        if let Some(metrics) = self.metrics {
            vec![
                -metrics.pr_auc,
                metrics.brier,
                self.mean_model_bytes,
                self.mean_structural_cost,
                self.recall_violation,
                self.structural_violation,
            ]
        } else {
            vec![
                1.0,
                1.0,
                1.0e12,
                1.0e12,
                self.recall_violation.max(1.0),
                self.structural_violation.max(1.0),
            ]
        }
    }

    pub fn qd_values(&self) -> (f64, [f64; 2]) {
        if !self.feasible() {
            return (f64::INFINITY, [f64::NAN, f64::NAN]);
        }
        let metrics = self.metrics.expect("feasible evaluation has metrics");
        (metrics.log_loss, metrics.qd_descriptors())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationEvaluation {
    pub metrics: Option<Metrics>,
    /// Mean QD descriptors across independently fitted single forests.
    ///
    /// `metrics` describes the seed-averaged probability ensemble used for
    /// robust model selection. QD retention instead compares like with like:
    /// tuning descriptors come from single-forest out-of-fold predictions, so
    /// validation descriptors must not be computed from a multi-seed ensemble.
    pub mean_single_forest_qd_descriptors: Option<[f64; 2]>,
    pub log_loss_sdev: f64,
    pub mean_model_bytes: f64,
    pub mean_structural_cost: f64,
    pub model_fits: usize,
    pub trees_fitted: usize,
    pub elapsed_seconds: f64,
    pub failure: Option<TrainFailure>,
}

impl ValidationEvaluation {
    pub fn score(&self) -> f64 {
        self.metrics
            .map_or(f64::INFINITY, |metrics| metrics.log_loss)
    }
}

#[derive(Clone, Debug)]
pub struct Evaluator {
    pub dataset: Arc<Dataset>,
    pub forest: ProbabilityForest,
    pub min_recall: f64,
    pub tuning_model_seed: u64,
}

impl Evaluator {
    pub fn new(dataset: Arc<Dataset>, min_recall: f64, tuning_model_seed: u64) -> Self {
        Self {
            dataset,
            forest: ProbabilityForest::default(),
            min_recall,
            tuning_model_seed,
        }
    }

    pub fn evaluate(&self, values: &[f64]) -> CandidateEvaluation {
        let started = Instant::now();
        let config = match decode(values) {
            Ok(config) => config,
            Err(error) => {
                return failure_evaluation(
                    values,
                    None,
                    TrainFailure::InvalidConfig(error.to_string()),
                    started.elapsed().as_secs_f64(),
                );
            }
        };
        let fold_train_rows =
            self.dataset.tuning.len() - self.dataset.folds.iter().map(Vec::len).max().unwrap_or(0);
        let estimated_structural_cost = config.structural_upper_bound(fold_train_rows);
        if estimated_structural_cost > self.forest.structural_cost_limit {
            let limit = self.forest.structural_cost_limit;
            return failure_evaluation(
                values,
                Some(config),
                TrainFailure::CostLimitExceeded {
                    estimated: estimated_structural_cost,
                    limit,
                },
                started.elapsed().as_secs_f64(),
            );
        }

        let mut out_of_fold = vec![f64::NAN; self.dataset.tuning.len()];
        let mut total_bytes = 0usize;
        let mut total_structural = 0u64;
        let mut trees_fitted = 0usize;
        for (fold_index, validation_indices) in self.dataset.folds.iter().enumerate() {
            let mut is_validation = vec![false; self.dataset.tuning.len()];
            for &index in validation_indices {
                is_validation[index] = true;
            }
            let training_indices: Vec<usize> = (0..self.dataset.tuning.len())
                .filter(|&index| !is_validation[index])
                .collect();
            let training_features = select_rows(&self.dataset.tuning.features, &training_indices);
            let training_labels: Vec<u8> = training_indices
                .iter()
                .map(|&index| self.dataset.tuning.labels[index])
                .collect();
            let validation_features =
                select_rows(&self.dataset.tuning.features, validation_indices);
            let outcome = match self.forest.fit_predict(
                &config,
                &training_features,
                &training_labels,
                &validation_features,
                stream_seed(self.tuning_model_seed, fold_index as u64),
            ) {
                Ok(outcome) => outcome,
                Err(failure) => {
                    return failure_evaluation(
                        values,
                        Some(config),
                        failure,
                        started.elapsed().as_secs_f64(),
                    );
                }
            };
            for (&row, probability) in validation_indices.iter().zip(outcome.probabilities) {
                out_of_fold[row] = probability;
            }
            total_bytes += outcome.serialized_bytes;
            total_structural += outcome.structural_cost;
            trees_fitted += outcome.trees;
        }
        let metrics = match Metrics::calculate(&self.dataset.tuning.labels, &out_of_fold) {
            Ok(metrics) => metrics,
            Err(error) => {
                let _ = error;
                return failure_evaluation(
                    values,
                    Some(config),
                    TrainFailure::NonFinitePrediction,
                    started.elapsed().as_secs_f64(),
                );
            }
        };
        let recall_violation = self.min_recall - metrics.recall;
        let structural_violation =
            estimated_structural_cost as f64 / self.forest.structural_cost_limit as f64 - 1.0;
        let normalized_violation =
            recall_violation.max(0.0) / self.min_recall.max(1.0e-9) + structural_violation.max(0.0);
        let scalar_fitness = if normalized_violation <= 0.0 {
            metrics.log_loss
        } else {
            LOG_LOSS_CEILING + 1.0 + normalized_violation
        };
        CandidateEvaluation {
            x: values.to_vec(),
            config: Some(config),
            metrics: Some(metrics),
            mean_model_bytes: total_bytes as f64 / self.dataset.folds.len() as f64,
            mean_structural_cost: total_structural as f64 / self.dataset.folds.len() as f64,
            estimated_structural_cost,
            model_fits: self.dataset.folds.len(),
            trees_fitted,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            recall_violation,
            structural_violation,
            scalar_fitness,
            failure: None,
        }
    }

    pub fn evaluate_selection(&self, config: &ForestConfig, seeds: &[u64]) -> ValidationEvaluation {
        evaluate_partition(
            &self.forest,
            config,
            &self.dataset.tuning,
            &self.dataset.selection,
            seeds,
        )
    }

    pub fn evaluate_final(&self, config: &ForestConfig, seeds: &[u64]) -> ValidationEvaluation {
        let training = self.dataset.training_and_selection();
        evaluate_partition(&self.forest, config, &training, &self.dataset.test, seeds)
    }
}

fn failure_evaluation(
    values: &[f64],
    config: Option<ForestConfig>,
    failure: TrainFailure,
    elapsed_seconds: f64,
) -> CandidateEvaluation {
    let estimated_structural_cost = config
        .as_ref()
        .map_or(0, |value| value.structural_upper_bound(1));
    CandidateEvaluation {
        x: values.to_vec(),
        config,
        metrics: None,
        mean_model_bytes: 1.0e12,
        mean_structural_cost: 1.0e12,
        estimated_structural_cost,
        model_fits: 0,
        trees_fitted: 0,
        elapsed_seconds,
        recall_violation: 1.0,
        structural_violation: 1.0,
        scalar_fitness: LOG_LOSS_CEILING + 3.0,
        failure: Some(failure),
    }
}

fn evaluate_partition(
    forest: &ProbabilityForest,
    config: &ForestConfig,
    training: &Partition,
    validation: &Partition,
    seeds: &[u64],
) -> ValidationEvaluation {
    let started = Instant::now();
    if seeds.is_empty() {
        return ValidationEvaluation {
            metrics: None,
            mean_single_forest_qd_descriptors: None,
            log_loss_sdev: f64::NAN,
            mean_model_bytes: f64::NAN,
            mean_structural_cost: f64::NAN,
            model_fits: 0,
            trees_fitted: 0,
            elapsed_seconds: 0.0,
            failure: Some(TrainFailure::InvalidConfig(
                "validation requires at least one model seed".to_string(),
            )),
        };
    }
    let mut probability_sum = vec![0.0; validation.len()];
    let mut log_losses = Vec::with_capacity(seeds.len());
    let mut qd_descriptor_sum = [0.0; 2];
    let mut qd_descriptor_count = 0usize;
    let mut total_bytes = 0usize;
    let mut total_structural = 0u64;
    let mut trees_fitted = 0usize;
    for &seed in seeds {
        let outcome = match forest.fit_predict(
            config,
            &training.features,
            &training.labels,
            &validation.features,
            seed,
        ) {
            Ok(outcome) => outcome,
            Err(failure) => {
                return ValidationEvaluation {
                    metrics: None,
                    mean_single_forest_qd_descriptors: None,
                    log_loss_sdev: f64::NAN,
                    mean_model_bytes: f64::NAN,
                    mean_structural_cost: f64::NAN,
                    model_fits: log_losses.len(),
                    trees_fitted,
                    elapsed_seconds: started.elapsed().as_secs_f64(),
                    failure: Some(failure),
                };
            }
        };
        if let Ok(metrics) = Metrics::calculate(&validation.labels, &outcome.probabilities) {
            log_losses.push(metrics.log_loss);
            let descriptors = metrics.qd_descriptors();
            qd_descriptor_sum[0] += descriptors[0];
            qd_descriptor_sum[1] += descriptors[1];
            qd_descriptor_count += 1;
        }
        for (sum, probability) in probability_sum.iter_mut().zip(outcome.probabilities) {
            *sum += probability;
        }
        total_bytes += outcome.serialized_bytes;
        total_structural += outcome.structural_cost;
        trees_fitted += outcome.trees;
    }
    for probability in &mut probability_sum {
        *probability /= seeds.len() as f64;
    }
    let metrics = Metrics::calculate(&validation.labels, &probability_sum).ok();
    let mean_single_forest_qd_descriptors = (qd_descriptor_count == seeds.len()).then(|| {
        [
            qd_descriptor_sum[0] / qd_descriptor_count as f64,
            qd_descriptor_sum[1] / qd_descriptor_count as f64,
        ]
    });
    let (_, log_loss_sdev) = mean_and_sdev(&log_losses);
    ValidationEvaluation {
        metrics,
        mean_single_forest_qd_descriptors,
        log_loss_sdev,
        mean_model_bytes: total_bytes as f64 / seeds.len() as f64,
        mean_structural_cost: total_structural as f64 / seeds.len() as f64,
        model_fits: seeds.len(),
        trees_fitted,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        failure: metrics
            .is_none()
            .then_some(TrainFailure::NonFinitePrediction),
    }
}

fn select_rows(source: &[Vec<f64>], indices: &[usize]) -> Vec<Vec<f64>> {
    indices.iter().map(|&index| source[index].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataConfig, Preset};
    use crate::space::DIMENSION;

    #[test]
    fn candidate_evaluation_is_repeatable() {
        let dataset = Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap());
        let evaluator = Evaluator::new(dataset, 0.2, 42);
        let first = evaluator.evaluate(&[0.25; DIMENSION]);
        let second = evaluator.evaluate(&[0.25; DIMENSION]);
        assert_eq!(
            first.metrics.unwrap().log_loss,
            second.metrics.unwrap().log_loss
        );
        assert_eq!(first.mean_model_bytes, second.mean_model_bytes);
        assert!(first.scalar_fitness.is_finite());
    }

    #[test]
    fn selection_uses_disjoint_partition() {
        let dataset = Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap());
        let evaluator = Evaluator::new(dataset, 0.2, 42);
        let config = decode(&[0.2; DIMENSION]).unwrap();
        let seeds = [101, 102];
        let selection = evaluator.evaluate_selection(&config, &seeds);
        assert!(selection.metrics.is_some());
        assert_eq!(selection.model_fits, 2);
        assert_eq!(selection.trees_fitted, 2 * config.n_trees);
        let mut expected = [0.0; 2];
        for seed in seeds {
            let outcome = evaluator
                .forest
                .fit_predict(
                    &config,
                    &evaluator.dataset.tuning.features,
                    &evaluator.dataset.tuning.labels,
                    &evaluator.dataset.selection.features,
                    seed,
                )
                .unwrap();
            let descriptors =
                Metrics::calculate(&evaluator.dataset.selection.labels, &outcome.probabilities)
                    .unwrap()
                    .qd_descriptors();
            expected[0] += descriptors[0] / seeds.len() as f64;
            expected[1] += descriptors[1] / seeds.len() as f64;
        }
        let actual = selection
            .mean_single_forest_qd_descriptors
            .expect("single-forest descriptors");
        assert!((actual[0] - expected[0]).abs() < 1.0e-12);
        assert!((actual[1] - expected[1]).abs() < 1.0e-12);
    }

    #[test]
    fn ordered_parallel_evaluation_matches_serial_values() {
        let dataset = Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap());
        let evaluator = Arc::new(Evaluator::new(dataset, 0.1, 42));
        let candidates = vec![
            vec![0.05; DIMENSION],
            vec![0.10; DIMENSION],
            vec![0.15; DIMENSION],
            vec![0.20; DIMENSION],
        ];
        let serial = fcmaes_core::parallel_batch(&candidates, 1, {
            let evaluator = Arc::clone(&evaluator);
            move |candidate| evaluator.evaluate(candidate)
        });
        let parallel = fcmaes_core::parallel_batch(&candidates, 4, {
            let evaluator = Arc::clone(&evaluator);
            move |candidate| evaluator.evaluate(candidate)
        });
        assert_eq!(serial.len(), parallel.len());
        for (serial_value, parallel_value) in serial.iter().zip(&parallel) {
            assert_eq!(
                serial_value
                    .config
                    .as_ref()
                    .expect("serial configuration")
                    .canonical_key(),
                parallel_value
                    .config
                    .as_ref()
                    .expect("parallel configuration")
                    .canonical_key()
            );
            assert_eq!(serial_value.scalar_fitness, parallel_value.scalar_fitness);
            assert_eq!(
                serial_value.metrics.expect("serial metrics"),
                parallel_value.metrics.expect("parallel metrics")
            );
        }
    }
}
