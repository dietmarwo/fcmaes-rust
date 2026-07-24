use std::f64::consts::TAU;

use fcmaes_core::Rng;
use serde::{Deserialize, Serialize};

pub const FEATURE_COUNT: usize = 24;
const INFORMATIVE_COUNT: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Preset {
    Smoke,
    Publication,
}

impl Preset {
    pub fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Publication => "publication",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataConfig {
    pub tuning_rows: usize,
    pub selection_rows: usize,
    pub test_rows: usize,
    pub reference_rows: usize,
    pub data_seed: u64,
    pub tuning_folds: usize,
}

impl DataConfig {
    pub fn for_preset(preset: Preset) -> Self {
        match preset {
            Preset::Smoke => Self {
                tuning_rows: 240,
                selection_rows: 120,
                test_rows: 400,
                reference_rows: 10_000,
                data_seed: 20_260_724,
                tuning_folds: 5,
            },
            Preset::Publication => Self {
                tuning_rows: 6_000,
                selection_rows: 4_000,
                test_rows: 20_000,
                reference_rows: 1_000_000,
                data_seed: 20_260_724,
                tuning_folds: 5,
            },
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.tuning_folds < 2 {
            return Err("at least two tuning folds are required");
        }
        if self.tuning_rows < self.tuning_folds * 4 {
            return Err("tuning data are too small for the requested folds");
        }
        if self.selection_rows < 2 || self.test_rows < 2 || self.reference_rows < 100 {
            return Err("selection, test, and reference sets must be non-empty");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Partition {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<u8>,
    pub eta: Vec<f64>,
}

impl Partition {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn prevalence(&self) -> f64 {
        self.labels
            .iter()
            .map(|&label| f64::from(label))
            .sum::<f64>()
            / self.len() as f64
    }

    pub fn content_hash(&self) -> String {
        // Stable FNV-1a over an explicitly framed byte stream. Rust's
        // DefaultHasher is not a cross-version persistence format.
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        hash_u64(&mut hash, self.features.len() as u64);
        hash_u64(&mut hash, self.features.first().map_or(0, Vec::len) as u64);
        hash_u64(&mut hash, self.labels.len() as u64);
        for &label in &self.labels {
            hash_byte(&mut hash, label);
        }
        for row in &self.features {
            for value in row {
                hash_u64(&mut hash, value.to_bits());
            }
        }
        for value in &self.eta {
            hash_u64(&mut hash, value.to_bits());
        }
        format!("{hash:016x}")
    }
}

fn hash_byte(hash: &mut u64, value: u8) {
    *hash ^= u64::from(value);
    *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
}

fn hash_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        hash_byte(hash, byte);
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BayesReference {
    pub rows: usize,
    pub error: f64,
    pub error_standard_error: f64,
    pub log_loss: f64,
    pub log_loss_standard_error: f64,
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub config: DataConfig,
    pub tuning: Partition,
    pub selection: Partition,
    pub test: Partition,
    pub folds: Vec<Vec<usize>>,
    pub bayes: BayesReference,
}

impl Dataset {
    pub fn generate(config: DataConfig) -> Result<Self, &'static str> {
        config.validate()?;
        let tuning = generate_partition(config.tuning_rows, stream_seed(config.data_seed, 1), true);
        let selection = generate_partition(
            config.selection_rows,
            stream_seed(config.data_seed, 2),
            true,
        );
        let test = generate_partition(config.test_rows, stream_seed(config.data_seed, 3), true);
        let folds = stratified_folds(
            &tuning.labels,
            config.tuning_folds,
            stream_seed(config.data_seed, 4),
        );
        let bayes = bayes_reference(config.reference_rows, stream_seed(config.data_seed, 5));
        Ok(Self {
            config,
            tuning,
            selection,
            test,
            folds,
            bayes,
        })
    }

    pub fn training_and_selection(&self) -> Partition {
        let mut features =
            Vec::with_capacity(self.tuning.features.len() + self.selection.features.len());
        features.extend(self.tuning.features.iter().cloned());
        features.extend(self.selection.features.iter().cloned());
        let mut labels = Vec::with_capacity(self.tuning.labels.len() + self.selection.labels.len());
        labels.extend_from_slice(&self.tuning.labels);
        labels.extend_from_slice(&self.selection.labels);
        let mut eta = Vec::with_capacity(self.tuning.eta.len() + self.selection.eta.len());
        eta.extend_from_slice(&self.tuning.eta);
        eta.extend_from_slice(&self.selection.eta);
        Partition {
            features,
            labels,
            eta,
        }
    }

    pub fn hashes(&self) -> DatasetHashes {
        DatasetHashes {
            tuning: self.tuning.content_hash(),
            selection: self.selection.content_hash(),
            test: self.test.content_hash(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetHashes {
    pub tuning: String,
    pub selection: String,
    pub test: String,
}

pub fn stream_seed(master: u64, stream: u64) -> u64 {
    let mut z = master.wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(stream + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn generate_partition(rows: usize, seed: u64, draw_labels: bool) -> Partition {
    let mut rng = Rng::new(seed);
    let mut features = Vec::with_capacity(rows);
    let mut labels = Vec::with_capacity(rows);
    let mut etas = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut informative = [0.0; INFORMATIVE_COUNT];
        for value in &mut informative {
            *value = 2.0 * rng.uniform01() - 1.0;
        }
        let score = latent_score(&informative);
        let eta = sigmoid(score);
        let label = u8::from(draw_labels && rng.uniform01() < eta);
        let mut row = Vec::with_capacity(FEATURE_COUNT);
        row.extend_from_slice(&informative);
        for _ in INFORMATIVE_COUNT..20 {
            row.push(2.0 * rng.uniform01() - 1.0);
        }
        row.push(informative[0] + 0.03 * standard_normal(&mut rng));
        row.push(informative[2] + 0.03 * standard_normal(&mut rng));
        row.push(0.7 * informative[5] + 0.3 * informative[6] + 0.03 * standard_normal(&mut rng));
        row.push(informative[7] + 0.03 * standard_normal(&mut rng));
        features.push(row);
        labels.push(label);
        etas.push(eta);
    }
    Partition {
        features,
        labels,
        eta: etas,
    }
}

fn latent_score(x: &[f64; INFORMATIVE_COUNT]) -> f64 {
    -2.15
        + 1.25 * (std::f64::consts::PI * x[0] * x[1]).sin()
        + 0.95 * x[2]
        + 0.85 * x[2] * x[2]
        + if x[3] > 0.25 { 1.0 } else { -0.15 }
        - 0.75 * x[4].abs()
        + 1.05 * x[5] * x[6]
        + 0.65 * x[7]
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn standard_normal(rng: &mut Rng) -> f64 {
    let u1 = rng.uniform01().max(f64::MIN_POSITIVE);
    let u2 = rng.uniform01();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

fn shuffle(values: &mut [usize], rng: &mut Rng) {
    for index in (1..values.len()).rev() {
        let swap = rng.int_below((index + 1) as i64) as usize;
        values.swap(index, swap);
    }
}

fn stratified_folds(labels: &[u8], folds: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut negative = Vec::new();
    let mut positive = Vec::new();
    for (index, &label) in labels.iter().enumerate() {
        if label == 0 {
            negative.push(index);
        } else {
            positive.push(index);
        }
    }
    let mut rng = Rng::new(seed);
    shuffle(&mut negative, &mut rng);
    shuffle(&mut positive, &mut rng);
    let mut result = vec![Vec::new(); folds];
    for (offset, index) in negative.into_iter().enumerate() {
        result[offset % folds].push(index);
    }
    for (offset, index) in positive.into_iter().enumerate() {
        result[offset % folds].push(index);
    }
    for fold in &mut result {
        fold.sort_unstable();
    }
    result
}

fn bayes_reference(rows: usize, seed: u64) -> BayesReference {
    let partition = generate_partition(rows, seed, false);
    let errors: Vec<f64> = partition
        .eta
        .iter()
        .map(|&eta| eta.min(1.0 - eta))
        .collect();
    let losses: Vec<f64> = partition
        .eta
        .iter()
        .map(|&eta| -eta * eta.ln() - (1.0 - eta) * (1.0 - eta).ln())
        .collect();
    let (error, error_standard_error) = mean_and_standard_error(&errors);
    let (log_loss, log_loss_standard_error) = mean_and_standard_error(&losses);
    BayesReference {
        rows,
        error,
        error_standard_error,
        log_loss,
        log_loss_standard_error,
    }
}

fn mean_and_standard_error(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, (variance / values.len() as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_data_are_repeatable_and_stratified() {
        let config = DataConfig::for_preset(Preset::Smoke);
        let first = Dataset::generate(config.clone()).unwrap();
        let second = Dataset::generate(config).unwrap();
        assert_eq!(first.hashes(), second.hashes());
        assert_eq!(first.hashes().tuning, "a0cc2b6131802e81");
        assert_eq!(first.hashes().selection, "d171c2051913607e");
        assert_eq!(first.hashes().test, "9c29e9a4564bde89");
        assert!((0.08..0.25).contains(&first.tuning.prevalence()));
        assert_eq!(
            first.folds.iter().map(Vec::len).sum::<usize>(),
            first.tuning.len()
        );
        for fold in &first.folds {
            assert!(fold.iter().any(|&index| first.tuning.labels[index] == 1));
            assert!(fold.iter().any(|&index| first.tuning.labels[index] == 0));
        }
    }

    #[test]
    fn partitions_and_streams_are_disjoint() {
        let dataset = Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap();
        let hashes = dataset.hashes();
        assert_ne!(hashes.tuning, hashes.selection);
        assert_ne!(hashes.tuning, hashes.test);
        assert_ne!(hashes.selection, hashes.test);
        assert_ne!(stream_seed(1, 1), stream_seed(1, 2));
    }

    #[test]
    fn bayes_reference_is_finite_and_bounded() {
        let dataset = Dataset::generate(DataConfig::for_preset(Preset::Smoke)).unwrap();
        assert!((0.0..0.5).contains(&dataset.bayes.error));
        assert!((0.0..std::f64::consts::LN_2).contains(&dataset.bayes.log_loss));
        assert!(dataset.bayes.error_standard_error > 0.0);
        assert!(dataset.bayes.log_loss_standard_error > 0.0);
    }
}
