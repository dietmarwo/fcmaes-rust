use std::fmt;
use std::time::{Duration, Instant};

use fcmaes_core::Rng;
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters,
};

use crate::data::stream_seed;
use crate::space::ForestConfig;

const PROBABILITY_EPSILON: f64 = 1.0e-6;
type Tree = DecisionTreeClassifier<f64, u8, DenseMatrix<f64>, Vec<u8>>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainFailure {
    InvalidConfig(String),
    Backend(String),
    NonFinitePrediction,
    CostLimitExceeded { estimated: u64, limit: u64 },
}

impl fmt::Display for TrainFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid configuration: {message}"),
            Self::Backend(message) => write!(formatter, "SmartCore backend failure: {message}"),
            Self::NonFinitePrediction => formatter.write_str("non-finite model prediction"),
            Self::CostLimitExceeded { estimated, limit } => {
                write!(
                    formatter,
                    "estimated structural cost {estimated} exceeds limit {limit}"
                )
            }
        }
    }
}

impl std::error::Error for TrainFailure {}

#[derive(Clone, Debug)]
pub struct FitOutcome {
    pub probabilities: Vec<f64>,
    pub serialized_bytes: usize,
    pub structural_cost: u64,
    pub trees: usize,
    pub elapsed: Duration,
}

#[derive(Clone, Debug)]
pub struct ProbabilityForest {
    pub structural_cost_limit: u64,
}

#[derive(Debug)]
struct FittedTree {
    features: Vec<usize>,
    tree: Tree,
}

#[derive(Debug)]
pub struct FittedForest {
    trees: Vec<FittedTree>,
    pub serialized_bytes: usize,
    pub structural_cost: u64,
}

impl FittedForest {
    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>, TrainFailure> {
        validate_prediction_data(features)?;
        let mut probabilities = vec![0.0; features.len()];
        for fitted in &self.trees {
            let predict_matrix = matrix_for_all(features, &fitted.features)?;
            let predicted = fitted
                .tree
                .predict_proba(&predict_matrix)
                .map_err(|error| TrainFailure::Backend(error.to_string()))?;
            if predicted.shape().1 != 2 {
                return Err(TrainFailure::Backend(
                    "tree probability output did not contain two classes".to_string(),
                ));
            }
            for (row, total) in probabilities.iter_mut().enumerate() {
                *total += *predicted.get((row, 1));
            }
        }
        for probability in &mut probabilities {
            *probability = (*probability / self.trees.len() as f64)
                .clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON);
            if !probability.is_finite() {
                return Err(TrainFailure::NonFinitePrediction);
            }
        }
        Ok(probabilities)
    }
}

impl Default for ProbabilityForest {
    fn default() -> Self {
        Self {
            structural_cost_limit: 2_000_000,
        }
    }
}

impl ProbabilityForest {
    pub fn fit(
        &self,
        config: &ForestConfig,
        train_features: &[Vec<f64>],
        train_labels: &[u8],
        seed: u64,
    ) -> Result<FittedForest, TrainFailure> {
        validate_config(config)?;
        validate_training_data(train_features, train_labels)?;
        let estimated = config.structural_upper_bound(train_labels.len());
        if estimated > self.structural_cost_limit {
            return Err(TrainFailure::CostLimitExceeded {
                estimated,
                limit: self.structural_cost_limit,
            });
        }
        let negative: Vec<usize> = train_labels
            .iter()
            .enumerate()
            .filter_map(|(index, &label)| (label == 0).then_some(index))
            .collect();
        let positive: Vec<usize> = train_labels
            .iter()
            .enumerate()
            .filter_map(|(index, &label)| (label == 1).then_some(index))
            .collect();
        if negative.is_empty() || positive.is_empty() {
            return Err(TrainFailure::InvalidConfig(
                "training data must contain both classes".to_string(),
            ));
        }

        let mut trees = Vec::with_capacity(config.n_trees);
        let mut serialized_bytes = 0usize;
        let mut structural_cost = 0u64;
        let feature_count = train_features[0].len();
        let selected_features = ((feature_count as f64 * config.feature_fraction).round() as usize)
            .clamp(1, feature_count);
        let sample_rows = ((train_labels.len() as f64 * config.row_sample_fraction).round()
            as usize)
            .clamp(2, train_labels.len());

        for tree_index in 0..config.n_trees {
            let tree_seed = stream_seed(seed, tree_index as u64);
            let mut rng = Rng::new(tree_seed);
            let features = feature_sample(feature_count, selected_features, &mut rng);
            let rows = weighted_bootstrap(
                sample_rows,
                &negative,
                &positive,
                config.positive_sampling_weight,
                &mut rng,
            );
            let train_matrix = matrix_for_rows(train_features, &rows, &features)?;
            let labels: Vec<u8> = rows.iter().map(|&index| train_labels[index]).collect();
            let parameters = DecisionTreeClassifierParameters {
                criterion: config.criterion.smartcore(),
                max_depth: Some(config.max_depth),
                min_samples_leaf: config.min_samples_leaf,
                min_samples_split: config.min_samples_split,
                seed: Some(stream_seed(tree_seed, 1)),
            };
            let tree = DecisionTreeClassifier::fit(&train_matrix, &labels, parameters)
                .map_err(|error| TrainFailure::Backend(error.to_string()))?;
            structural_cost += u64::from(tree.depth()) + 1;
            serialized_bytes += serde_json::to_vec(&tree)
                .map_err(|error| TrainFailure::Backend(error.to_string()))?
                .len()
                + features.len() * std::mem::size_of::<usize>();
            trees.push(FittedTree { features, tree });
        }
        Ok(FittedForest {
            trees,
            serialized_bytes,
            structural_cost,
        })
    }

    pub fn fit_predict(
        &self,
        config: &ForestConfig,
        train_features: &[Vec<f64>],
        train_labels: &[u8],
        predict_features: &[Vec<f64>],
        seed: u64,
    ) -> Result<FitOutcome, TrainFailure> {
        let started = Instant::now();
        let fitted = self.fit(config, train_features, train_labels, seed)?;
        let probabilities = fitted.predict(predict_features)?;
        Ok(FitOutcome {
            probabilities,
            serialized_bytes: fitted.serialized_bytes,
            structural_cost: fitted.structural_cost,
            trees: fitted.tree_count(),
            elapsed: started.elapsed(),
        })
    }
}

fn validate_config(config: &ForestConfig) -> Result<(), TrainFailure> {
    if config.n_trees == 0 {
        return Err(TrainFailure::InvalidConfig(
            "n_trees must be positive".to_string(),
        ));
    }
    if config.max_depth == 0 {
        return Err(TrainFailure::InvalidConfig(
            "max_depth must be positive".to_string(),
        ));
    }
    if config.min_samples_leaf == 0 || config.min_samples_split < 2 {
        return Err(TrainFailure::InvalidConfig(
            "minimum leaf/split sizes must be positive and split at least two".to_string(),
        ));
    }
    if !config.row_sample_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.row_sample_fraction)
        || config.row_sample_fraction == 0.0
    {
        return Err(TrainFailure::InvalidConfig(
            "row sample fraction must lie in (0, 1]".to_string(),
        ));
    }
    if !config.feature_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.feature_fraction)
        || config.feature_fraction == 0.0
    {
        return Err(TrainFailure::InvalidConfig(
            "feature fraction must lie in (0, 1]".to_string(),
        ));
    }
    if !config.positive_sampling_weight.is_finite() || config.positive_sampling_weight <= 0.0 {
        return Err(TrainFailure::InvalidConfig(
            "positive sampling weight must be finite and positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_training_data(
    train_features: &[Vec<f64>],
    train_labels: &[u8],
) -> Result<(), TrainFailure> {
    if train_features.len() != train_labels.len() || train_features.len() < 2 {
        return Err(TrainFailure::InvalidConfig(
            "training features and labels must be row aligned".to_string(),
        ));
    }
    let feature_count = train_features[0].len();
    if feature_count == 0
        || train_features
            .iter()
            .any(|row| row.len() != feature_count || row.iter().any(|value| !value.is_finite()))
    {
        return Err(TrainFailure::InvalidConfig(
            "feature rows must be finite and rectangular".to_string(),
        ));
    }
    if train_labels.iter().any(|&label| label > 1) {
        return Err(TrainFailure::InvalidConfig(
            "labels must be binary zero/one values".to_string(),
        ));
    }
    Ok(())
}

fn validate_prediction_data(features: &[Vec<f64>]) -> Result<(), TrainFailure> {
    if features.is_empty() {
        return Err(TrainFailure::InvalidConfig(
            "prediction features must be non-empty".to_string(),
        ));
    }
    let feature_count = features[0].len();
    if feature_count == 0
        || features
            .iter()
            .any(|row| row.len() != feature_count || row.iter().any(|value| !value.is_finite()))
    {
        return Err(TrainFailure::InvalidConfig(
            "prediction features must be finite and rectangular".to_string(),
        ));
    }
    Ok(())
}

fn feature_sample(feature_count: usize, selected: usize, rng: &mut Rng) -> Vec<usize> {
    let mut features: Vec<usize> = (0..feature_count).collect();
    shuffle(&mut features, rng);
    features.truncate(selected);
    features.sort_unstable();
    features
}

fn weighted_bootstrap(
    sample_rows: usize,
    negative: &[usize],
    positive: &[usize],
    positive_weight: f64,
    rng: &mut Rng,
) -> Vec<usize> {
    let positive_mass = positive.len() as f64 * positive_weight;
    let positive_probability = positive_mass / (positive_mass + negative.len() as f64);
    let mut rows = Vec::with_capacity(sample_rows);
    rows.push(negative[rng.int_below(negative.len() as i64) as usize]);
    rows.push(positive[rng.int_below(positive.len() as i64) as usize]);
    for _ in 2..sample_rows {
        let class = if rng.uniform01() < positive_probability {
            positive
        } else {
            negative
        };
        rows.push(class[rng.int_below(class.len() as i64) as usize]);
    }
    shuffle(&mut rows, rng);
    rows
}

fn shuffle<T>(values: &mut [T], rng: &mut Rng) {
    for index in (1..values.len()).rev() {
        let swap = rng.int_below((index + 1) as i64) as usize;
        values.swap(index, swap);
    }
}

fn matrix_for_rows(
    source: &[Vec<f64>],
    rows: &[usize],
    features: &[usize],
) -> Result<DenseMatrix<f64>, TrainFailure> {
    if rows.iter().any(|&row| row >= source.len())
        || features
            .iter()
            .any(|&column| source.iter().any(|row| column >= row.len()))
    {
        return Err(TrainFailure::InvalidConfig(
            "training row or feature index is out of bounds".to_string(),
        ));
    }
    let values: Vec<Vec<f64>> = rows
        .iter()
        .map(|&row| features.iter().map(|&column| source[row][column]).collect())
        .collect();
    DenseMatrix::from_2d_vec(&values).map_err(|error| TrainFailure::Backend(error.to_string()))
}

fn matrix_for_all(
    source: &[Vec<f64>],
    features: &[usize],
) -> Result<DenseMatrix<f64>, TrainFailure> {
    if features
        .iter()
        .any(|&column| source.iter().any(|row| column >= row.len()))
    {
        return Err(TrainFailure::InvalidConfig(
            "prediction data have fewer columns than the fitted model".to_string(),
        ));
    }
    let values: Vec<Vec<f64>> = source
        .iter()
        .map(|row| features.iter().map(|&column| row[column]).collect())
        .collect();
    DenseMatrix::from_2d_vec(&values).map_err(|error| TrainFailure::Backend(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataConfig, Dataset, Preset};

    #[test]
    fn probability_forest_is_seeded_and_finite() {
        let dataset = Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap();
        let config = ForestConfig {
            n_trees: 8,
            max_depth: 5,
            ..ForestConfig::default_config()
        };
        let forest = ProbabilityForest::default();
        let first = forest
            .fit_predict(
                &config,
                &dataset.tuning.features,
                &dataset.tuning.labels,
                &dataset.selection.features,
                42,
            )
            .unwrap();
        let second = forest
            .fit_predict(
                &config,
                &dataset.tuning.features,
                &dataset.tuning.labels,
                &dataset.selection.features,
                42,
            )
            .unwrap();
        assert_eq!(first.probabilities, second.probabilities);
        assert!(first.probabilities.iter().all(|p| (0.0..1.0).contains(p)));
        assert!(first.serialized_bytes > 0);
        assert!(first.structural_cost >= config.n_trees as u64);
    }

    #[test]
    fn prefit_cost_limit_is_enforced() {
        let dataset = Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap();
        let forest = ProbabilityForest {
            structural_cost_limit: 1,
        };
        let result = forest.fit_predict(
            &ForestConfig::default_config(),
            &dataset.tuning.features,
            &dataset.tuning.labels,
            &dataset.selection.features,
            42,
        );
        assert!(matches!(
            result,
            Err(TrainFailure::CostLimitExceeded { .. })
        ));
    }

    #[test]
    fn malformed_configuration_returns_a_typed_error() {
        let dataset = Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap();
        let config = ForestConfig {
            n_trees: 0,
            ..ForestConfig::default_config()
        };
        let result = ProbabilityForest::default().fit(
            &config,
            &dataset.tuning.features,
            &dataset.tuning.labels,
            42,
        );
        assert!(matches!(result, Err(TrainFailure::InvalidConfig(_))));
    }

    #[test]
    fn prediction_dimension_mismatch_returns_a_typed_error() {
        let dataset = Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap();
        let config = ForestConfig {
            n_trees: 2,
            feature_fraction: 1.0,
            ..ForestConfig::default_config()
        };
        let forest = ProbabilityForest::default()
            .fit(
                &config,
                &dataset.tuning.features,
                &dataset.tuning.labels,
                42,
            )
            .unwrap();
        let result = forest.predict(&vec![vec![0.0; 3]; 2]);
        assert!(matches!(result, Err(TrainFailure::InvalidConfig(_))));
    }
}
