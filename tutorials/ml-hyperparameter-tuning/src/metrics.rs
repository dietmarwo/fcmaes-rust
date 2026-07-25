use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    pub log_loss: f64,
    pub brier: f64,
    pub pr_auc: f64,
    pub roc_auc: f64,
    pub ece: f64,
    pub recall: f64,
    pub precision: f64,
    pub predicted_positive_rate: f64,
    /// Standard deviation of the predicted probabilities. High values are
    /// decisive forests that push probabilities toward 0 and 1; low values are
    /// hedging forests that keep every prediction near the base rate. It is an
    /// aggregate over every row, so unlike a confusion count it does not hinge
    /// on the handful of cases that straddle the 0.5 threshold.
    pub sharpness: f64,
    pub false_positives: usize,
    pub false_negatives: usize,
}

impl Metrics {
    pub fn calculate(labels: &[u8], probabilities: &[f64]) -> Result<Self, &'static str> {
        if labels.len() != probabilities.len() || labels.is_empty() {
            return Err("labels and probabilities must be non-empty and row aligned");
        }
        if labels.iter().any(|&label| label > 1)
            || probabilities
                .iter()
                .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
        {
            return Err("labels must be binary and probabilities finite in [0, 1]");
        }
        let n = labels.len() as f64;
        let mut log_loss = 0.0;
        let mut brier = 0.0;
        let mut true_positives = 0usize;
        let mut false_positives = 0usize;
        let mut false_negatives = 0usize;
        let mut predicted_positives = 0usize;
        for (&label, &probability) in labels.iter().zip(probabilities) {
            let p = probability.clamp(1.0e-6, 1.0 - 1.0e-6);
            let y = f64::from(label);
            log_loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            brier += (p - y).powi(2);
            if probability >= 0.5 {
                predicted_positives += 1;
                if label == 1 {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            } else if label == 1 {
                false_negatives += 1;
            }
        }
        let mean_probability = probabilities.iter().sum::<f64>() / n;
        let sharpness = (probabilities
            .iter()
            .map(|probability| (probability - mean_probability).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let positives = labels.iter().filter(|&&label| label == 1).count();
        let recall = true_positives as f64 / positives.max(1) as f64;
        let precision = true_positives as f64 / predicted_positives.max(1) as f64;
        Ok(Self {
            log_loss: log_loss / n,
            brier: brier / n,
            pr_auc: average_precision(labels, probabilities),
            roc_auc: roc_auc(labels, probabilities),
            ece: expected_calibration_error(labels, probabilities, 10),
            recall,
            precision,
            predicted_positive_rate: predicted_positives as f64 / n,
            sharpness,
            false_positives,
            false_negatives,
        })
    }

    /// log10 false-positive / false-negative ratio.
    ///
    /// Retained as a reported diagnostic. It is deliberately **not** a QD
    /// descriptor: with the operating threshold fixed at 0.5 it is a monotone
    /// function of [`Self::predicted_positive_rate`] (measured rank
    /// correlation +0.999715 over 271 feasible publication candidates), so
    /// pairing the two collapses the archive onto a one-dimensional ribbon.
    pub fn error_ratio(self) -> f64 {
        ((self.false_positives as f64 + 1.0) / (self.false_negatives as f64 + 1.0)).log10()
    }

    /// The two QD behavior descriptors: where the forest sits on the
    /// precision/recall trade-off, and how decisive its probabilities are.
    ///
    /// The axes are chosen to be jointly reachable rather than redundant.
    /// Precision is driven mainly by the positive-class sampling weight, while
    /// sharpness is driven by depth, leaf size and ensemble size, so feasible
    /// designs spread over a genuinely two-dimensional region.
    pub fn qd_descriptors(self) -> [f64; 2] {
        [self.precision, self.sharpness]
    }
}

pub fn mean_and_sdev(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() == 1 {
        return (mean, 0.0);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, variance.sqrt())
}

fn average_precision(labels: &[u8], probabilities: &[f64]) -> f64 {
    let positives = labels.iter().filter(|&&label| label == 1).count();
    if positives == 0 {
        return 0.0;
    }
    let mut order: Vec<usize> = (0..labels.len()).collect();
    order.sort_by(|&left, &right| {
        probabilities[right]
            .total_cmp(&probabilities[left])
            .then_with(|| left.cmp(&right))
    });
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut area = 0.0;
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len()
            && probabilities[order[end]].to_bits() == probabilities[order[start]].to_bits()
        {
            end += 1;
        }
        let group_positives = order[start..end]
            .iter()
            .filter(|&&index| labels[index] == 1)
            .count();
        true_positives += group_positives;
        false_positives += end - start - group_positives;
        let recall_increment = group_positives as f64 / positives as f64;
        let precision = true_positives as f64 / (true_positives + false_positives) as f64;
        area += recall_increment * precision;
        start = end;
    }
    area
}

fn roc_auc(labels: &[u8], probabilities: &[f64]) -> f64 {
    let positives = labels.iter().filter(|&&label| label == 1).count();
    let negatives = labels.len() - positives;
    if positives == 0 || negatives == 0 {
        return 0.5;
    }
    let mut order: Vec<usize> = (0..labels.len()).collect();
    order.sort_by(|&left, &right| {
        probabilities[left]
            .total_cmp(&probabilities[right])
            .then_with(|| left.cmp(&right))
    });
    let mut positive_rank_sum = 0.0;
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len()
            && probabilities[order[end]].to_bits() == probabilities[order[start]].to_bits()
        {
            end += 1;
        }
        let average_rank = ((start + 1 + end) as f64) / 2.0;
        for &index in &order[start..end] {
            if labels[index] == 1 {
                positive_rank_sum += average_rank;
            }
        }
        start = end;
    }
    (positive_rank_sum - positives as f64 * (positives as f64 + 1.0) / 2.0)
        / (positives * negatives) as f64
}

fn expected_calibration_error(labels: &[u8], probabilities: &[f64], bins: usize) -> f64 {
    let mut counts = vec![0usize; bins];
    let mut probability_sum = vec![0.0; bins];
    let mut label_sum = vec![0.0; bins];
    for (&label, &probability) in labels.iter().zip(probabilities) {
        let bin = ((probability * bins as f64).floor() as usize).min(bins - 1);
        counts[bin] += 1;
        probability_sum[bin] += probability;
        label_sum[bin] += f64::from(label);
    }
    let mut ece = 0.0;
    for bin in 0..bins {
        if counts[bin] > 0 {
            let count = counts[bin] as f64;
            ece += count / labels.len() as f64
                * (probability_sum[bin] / count - label_sum[bin] / count).abs();
        }
    }
    ece
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_predictions_have_expected_metrics() {
        let labels = [0, 0, 1, 1];
        let probabilities = [0.01, 0.1, 0.9, 0.99];
        let metrics = Metrics::calculate(&labels, &probabilities).unwrap();
        assert_eq!(metrics.pr_auc, 1.0);
        assert_eq!(metrics.roc_auc, 1.0);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.precision, 1.0);
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.false_negatives, 0);
        assert!(metrics.log_loss < 0.06);
    }

    #[test]
    fn ties_receive_half_auc() {
        let metrics = Metrics::calculate(&[0, 1, 0, 1], &[0.5; 4]).unwrap();
        assert!((metrics.roc_auc - 0.5).abs() < 1.0e-12);
        assert!((metrics.pr_auc - 0.5).abs() < 1.0e-12);
        assert_eq!(metrics.predicted_positive_rate, 1.0);
    }

    #[test]
    fn average_precision_is_invariant_to_order_within_ties() {
        let first = Metrics::calculate(&[1, 0, 0, 1], &[0.5; 4]).unwrap();
        let second = Metrics::calculate(&[0, 1, 1, 0], &[0.5; 4]).unwrap();
        assert_eq!(first.pr_auc, 0.5);
        assert_eq!(first.pr_auc, second.pr_auc);
    }

    #[test]
    fn mean_and_sample_deviation_are_correct() {
        let (mean, sdev) = mean_and_sdev(&[1.0, 2.0, 3.0]);
        assert_eq!(mean, 2.0);
        assert!((sdev - 1.0).abs() < 1.0e-12);
    }
}
