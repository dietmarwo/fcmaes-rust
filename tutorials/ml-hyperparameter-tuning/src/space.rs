use serde::{Deserialize, Serialize};
use smartcore::tree::decision_tree_classifier::SplitCriterion;

pub const DIMENSION: usize = 8;
pub const LOWER_BOUNDS: [f64; DIMENSION] = [0.0; DIMENSION];
pub const UPPER_BOUNDS: [f64; DIMENSION] = [1.0; DIMENSION];

pub const DECISION_NAMES: [&str; DIMENSION] = [
    "n_trees",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "row_sample_fraction",
    "feature_fraction",
    "positive_sampling_weight",
    "criterion_index",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Criterion {
    Gini,
    Entropy,
}

impl Criterion {
    pub fn smartcore(self) -> SplitCriterion {
        match self {
            Self::Gini => SplitCriterion::Gini,
            Self::Entropy => SplitCriterion::Entropy,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Gini => 0,
            Self::Entropy => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForestConfig {
    pub n_trees: usize,
    pub max_depth: u16,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub row_sample_fraction: f64,
    pub feature_fraction: f64,
    pub positive_sampling_weight: f64,
    pub criterion: Criterion,
}

impl ForestConfig {
    pub fn default_config() -> Self {
        Self {
            n_trees: 64,
            max_depth: 12,
            min_samples_leaf: 2,
            min_samples_split: 4,
            row_sample_fraction: 0.8,
            feature_fraction: 0.7,
            positive_sampling_weight: 2.0,
            criterion: Criterion::Gini,
        }
    }

    pub fn canonical_key(&self) -> String {
        format!(
            "{}:{}:{}:{}:{:016x}:{:016x}:{:016x}:{}",
            self.n_trees,
            self.max_depth,
            self.min_samples_leaf,
            self.min_samples_split,
            self.row_sample_fraction.to_bits(),
            self.feature_fraction.to_bits(),
            self.positive_sampling_weight.to_bits(),
            self.criterion.index(),
        )
    }

    pub fn as_decisions(&self) -> [f64; DIMENSION] {
        [
            self.n_trees as f64,
            f64::from(self.max_depth),
            self.min_samples_leaf as f64,
            self.min_samples_split as f64,
            self.row_sample_fraction,
            self.feature_fraction,
            self.positive_sampling_weight,
            self.criterion.index() as f64,
        ]
    }

    pub fn structural_upper_bound(&self, training_rows: usize) -> u64 {
        let sample_rows = ((training_rows as f64 * self.row_sample_fraction).round() as u64)
            .clamp(2, training_rows.max(2) as u64);
        let depth_nodes = (1_u64 << (u32::from(self.max_depth) + 1)).saturating_sub(1);
        let sample_nodes = (2 * sample_rows / self.min_samples_leaf as u64).saturating_sub(1);
        self.n_trees as u64 * depth_nodes.min(sample_nodes.max(1))
    }
}

pub fn decode(values: &[f64]) -> Result<ForestConfig, &'static str> {
    if values.len() != DIMENSION || values.iter().any(|value| !value.is_finite()) {
        return Err("decision vector must contain eight finite values");
    }
    let x: Vec<f64> = values.iter().map(|value| value.clamp(0.0, 1.0)).collect();
    Ok(ForestConfig {
        n_trees: log_integer(x[0], 8, 256),
        max_depth: linear_integer(x[1], 2, 24) as u16,
        min_samples_leaf: log_integer(x[2], 1, 64),
        min_samples_split: log_integer(x[3], 2, 64),
        row_sample_fraction: 0.4 + 0.6 * x[4],
        feature_fraction: 0.25 + 0.75 * x[5],
        positive_sampling_weight: 0.5 * 8.0_f64.powf(x[6]),
        criterion: if categorical_index(x[7], 2) == 0 {
            Criterion::Gini
        } else {
            Criterion::Entropy
        },
    })
}

pub fn default_coordinates() -> [f64; DIMENSION] {
    [
        (64.0_f64 / 8.0).ln() / (256.0_f64 / 8.0).ln(),
        10.0 / 22.0,
        2.0_f64.ln() / 64.0_f64.ln(),
        (4.0_f64 / 2.0).ln() / (64.0_f64 / 2.0).ln(),
        (0.8 - 0.4) / 0.6,
        (0.7 - 0.25) / 0.75,
        (2.0_f64 / 0.5).ln() / 8.0_f64.ln(),
        0.0,
    ]
}

fn linear_integer(value: f64, lower: usize, upper: usize) -> usize {
    (lower as f64 + value * (upper - lower) as f64)
        .round()
        .clamp(lower as f64, upper as f64) as usize
}

fn log_integer(value: f64, lower: usize, upper: usize) -> usize {
    let log_lower = (lower as f64).ln();
    let log_upper = (upper as f64).ln();
    (log_lower + value * (log_upper - log_lower))
        .exp()
        .round()
        .clamp(lower as f64, upper as f64) as usize
}

fn categorical_index(value: f64, categories: usize) -> usize {
    ((value.clamp(0.0, 1.0) * categories as f64).floor() as usize).min(categories - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_includes_both_endpoints() {
        let lower = decode(&[0.0; DIMENSION]).unwrap();
        assert_eq!(lower.n_trees, 8);
        assert_eq!(lower.max_depth, 2);
        assert_eq!(lower.min_samples_leaf, 1);
        assert_eq!(lower.min_samples_split, 2);
        assert_eq!(lower.criterion, Criterion::Gini);
        let upper = decode(&[1.0; DIMENSION]).unwrap();
        assert_eq!(upper.n_trees, 256);
        assert_eq!(upper.max_depth, 24);
        assert_eq!(upper.min_samples_leaf, 64);
        assert_eq!(upper.min_samples_split, 64);
        assert_eq!(upper.criterion, Criterion::Entropy);
    }

    #[test]
    fn logarithmic_dimensions_are_monotone() {
        let mut previous = decode(&[0.0; DIMENSION]).unwrap();
        for step in 1..=100 {
            let mut values = [0.0; DIMENSION];
            values[0] = step as f64 / 100.0;
            values[2] = values[0];
            values[3] = values[0];
            let current = decode(&values).unwrap();
            assert!(current.n_trees >= previous.n_trees);
            assert!(current.min_samples_leaf >= previous.min_samples_leaf);
            assert!(current.min_samples_split >= previous.min_samples_split);
            previous = current;
        }
    }

    #[test]
    fn canonical_key_tracks_decoded_configuration() {
        let first = decode(&[0.5; DIMENSION]).unwrap();
        let second = decode(&[0.5; DIMENSION]).unwrap();
        assert_eq!(first.canonical_key(), second.canonical_key());
        assert_eq!(first.as_decisions().len(), DIMENSION);
    }
}
