//! Pieces shared by the three adaptive differential-evolution variants:
//! budgeted evaluation, Cauchy sampling, the weighted Lehmer mean,
//! midpoint-target bound repair, linear population size reduction, rank-based
//! selection and the external archive.
//!
//! All randomness comes from [`fcmaes_core::Rng`].

use fcmaes_core::Rng;
use solver_api::{Objective, Outcome};

/// Budgeted, serial evaluator that maps non-finite values to `+inf` and
/// tracks the best point seen.
pub(crate) struct Evaluator<'a> {
    objective: Objective<'a>,
    max_evaluations: u64,
    used: u64,
    best_x: Vec<f64>,
    best_y: f64,
}

impl<'a> Evaluator<'a> {
    /// Wrap `objective` with a hard budget of `max_evaluations` calls.
    pub(crate) fn new(objective: Objective<'a>, max_evaluations: u64) -> Self {
        Self {
            objective,
            max_evaluations,
            used: 0,
            best_x: Vec::new(),
            best_y: f64::INFINITY,
        }
    }

    /// True when no further call is allowed.
    pub(crate) fn exhausted(&self) -> bool {
        self.used >= self.max_evaluations
    }

    /// Objective calls made so far.
    pub(crate) fn used(&self) -> u64 {
        self.used
    }

    /// Evaluate `x`; NaN and infinite values become `+inf` (worse than every
    /// finite value). Callers must check [`exhausted`](Self::exhausted) first.
    pub(crate) fn eval(&mut self, x: &[f64]) -> f64 {
        debug_assert!(!self.exhausted());
        self.used += 1;
        let raw = (self.objective)(x);
        let y = if raw.is_finite() { raw } else { f64::INFINITY };
        if self.best_x.is_empty() || y < self.best_y {
            self.best_y = y;
            self.best_x = x.to_vec();
        }
        y
    }

    /// Final result; for a zero budget the box center is reported with `+inf`.
    pub(crate) fn outcome(self, lower: &[f64], upper: &[f64]) -> Outcome {
        let x = if self.best_x.is_empty() {
            lower
                .iter()
                .zip(upper)
                .map(|(lo, hi)| 0.5 * (lo + hi))
                .collect()
        } else {
            self.best_x
        };
        Outcome {
            x,
            y: self.best_y,
            evaluations: self.used,
        }
    }
}

/// Check the box and return its dimension.
pub(crate) fn check_box(lower: &[f64], upper: &[f64]) -> usize {
    assert_eq!(
        lower.len(),
        upper.len(),
        "lower and upper bounds differ in length"
    );
    assert!(
        lower.iter().zip(upper).all(|(lo, hi)| lo <= hi),
        "every lower bound must not exceed its upper bound"
    );
    lower.len()
}

/// Uniform random point in `[lower, upper]`.
pub(crate) fn uniform_point(rng: &mut Rng, lower: &[f64], upper: &[f64]) -> Vec<f64> {
    lower
        .iter()
        .zip(upper)
        .map(|(lo, hi)| (lo + rng.uniform01() * (hi - lo)).clamp(*lo, *hi))
        .collect()
}

/// Uniform index in `0..n` (`n > 0`).
pub(crate) fn rand_index(rng: &mut Rng, n: usize) -> usize {
    debug_assert!(n > 0);
    (rng.int_below(n as i64) as usize).min(n - 1)
}

/// Cauchy sample with location `loc` and scale `scale`.
pub(crate) fn cauchy(rng: &mut Rng, loc: f64, scale: f64) -> f64 {
    loc + scale * (std::f64::consts::PI * (rng.uniform01() - 0.5)).tan()
}

/// SHADE scaling-factor sample: Cauchy(`loc`, `scale`), regenerated while it
/// is not positive, truncated to 1.
pub(crate) fn sample_f_cauchy(rng: &mut Rng, loc: f64, scale: f64) -> f64 {
    loop {
        let f = cauchy(rng, loc, scale);
        if f > 0.0 {
            return f.min(1.0);
        }
    }
}

/// Normal sample clipped to `[0, 1]`.
pub(crate) fn sample_clipped_normal(rng: &mut Rng, mean: f64, sigma: f64) -> f64 {
    rng.normreal(mean, sigma).clamp(0.0, 1.0)
}

/// Weighted Lehmer mean `sum(w v^2) / sum(w v)` with weights proportional to
/// `weights` (fitness improvements).
///
/// Infinite improvements (a finite trial replacing a non-finite parent)
/// dominate: in the limit the weight is shared equally among them. If all
/// weights are zero, equal weights are used. Returns `None` for an empty set
/// or a zero denominator.
pub(crate) fn weighted_lehmer_mean(values: &[f64], weights: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let any_infinite = weights.iter().any(|w| w.is_infinite());
    let total: f64 = weights.iter().filter(|w| w.is_finite()).sum();
    let weight = |w: f64| -> f64 {
        if any_infinite {
            if w.is_infinite() { 1.0 } else { 0.0 }
        } else if total > 0.0 {
            w.max(0.0)
        } else {
            1.0
        }
    };
    let (mut num, mut den) = (0.0, 0.0);
    for (&v, &w) in values.iter().zip(weights) {
        let w = weight(w);
        num += w * v * v;
        den += w * v;
    }
    (den > 0.0).then(|| num / den)
}

/// Midpoint-target repair: a coordinate below (above) its bound is moved
/// halfway between the bound and the reference (parent or target) coordinate.
pub(crate) fn repair_midpoint(v: &mut [f64], reference: &[f64], lower: &[f64], upper: &[f64]) {
    for j in 0..v.len() {
        if v[j] < lower[j] {
            v[j] = 0.5 * (lower[j] + reference[j]);
        } else if v[j] > upper[j] {
            v[j] = 0.5 * (upper[j] + reference[j]);
        }
        // Guards against rounding only; the midpoint of two in-box values is
        // already inside the box.
        v[j] = v[j].clamp(lower[j], upper[j]);
    }
}

/// Binomial crossover with a forced coordinate `j_rand`: `trial[j] = mutant[j]`
/// if `rand <= cr` or `j == j_rand`, otherwise `target[j]`. Returns the number
/// of coordinates taken from the mutant.
pub(crate) fn binomial_crossover(
    rng: &mut Rng,
    mutant: &[f64],
    target: &[f64],
    cr: f64,
    trial: &mut [f64],
) -> usize {
    let dim = mutant.len();
    let j_rand = rand_index(rng, dim);
    let mut taken = 0;
    for j in 0..dim {
        if j == j_rand || rng.uniform01() <= cr {
            trial[j] = mutant[j];
            taken += 1;
        } else {
            trial[j] = target[j];
        }
    }
    taken
}

/// Linear population size reduction:
/// `round(n_init + (n_min - n_init) * fes / max_fes)`, never below `n_min`.
pub(crate) fn lpsr_size(n_init: usize, n_min: usize, fes: u64, max_fes: u64) -> usize {
    if max_fes == 0 {
        return n_min;
    }
    let ratio = (fes as f64 / max_fes as f64).min(1.0);
    let n = (n_init as f64 + (n_min as f64 - n_init as f64) * ratio).round();
    (n.max(n_min as f64) as usize).min(n_init.max(n_min))
}

/// Indices `0..fitness.len()` sorted by ascending fitness (best first).
pub(crate) fn sorted_indices(fitness: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..fitness.len()).collect();
    order.sort_by(|&a, &b| fitness[a].total_cmp(&fitness[b]).then(a.cmp(&b)));
    order
}

/// Size of the p-best set: `round(p * n)`, at least `min_count`, at most `n`.
pub(crate) fn pbest_count(p: f64, n: usize, min_count: usize) -> usize {
    ((p * n as f64).round() as usize)
        .max(min_count)
        .min(n)
        .max(1)
}

/// Exponential rank-based selection: rank `k` (0 = best) of `n` is chosen with
/// probability proportional to `exp(-kp * k / n)`.
pub(crate) struct RankSelector {
    cumulative: Vec<f64>,
}

impl RankSelector {
    /// Selector over `n` ranks with selective pressure `kp`.
    pub(crate) fn new(n: usize, kp: f64) -> Self {
        let mut cumulative = Vec::with_capacity(n);
        let mut acc = 0.0;
        for k in 0..n {
            acc += (-kp * k as f64 / n as f64).exp();
            cumulative.push(acc);
        }
        Self { cumulative }
    }

    /// Draw a rank (0 = best).
    pub(crate) fn sample(&self, rng: &mut Rng) -> usize {
        let total = *self.cumulative.last().unwrap_or(&0.0);
        let u = rng.uniform01() * total;
        self.cumulative
            .partition_point(|&c| c <= u)
            .min(self.cumulative.len() - 1)
    }
}

/// External archive of replaced parents with random removal.
#[derive(Default)]
pub(crate) struct Archive {
    members: Vec<Vec<f64>>,
}

impl Archive {
    /// Number of stored vectors.
    pub(crate) fn len(&self) -> usize {
        self.members.len()
    }

    /// Stored vector `i`.
    pub(crate) fn get(&self, i: usize) -> &[f64] {
        &self.members[i]
    }

    /// Add `x`. If the archive already holds `capacity` members, `x`
    /// overwrites a random member, so a new parent always enters. With
    /// `capacity == 0` nothing is stored.
    pub(crate) fn push(&mut self, rng: &mut Rng, x: Vec<f64>, capacity: usize) {
        self.shrink(rng, capacity);
        if capacity == 0 {
            return;
        }
        if self.members.len() < capacity {
            self.members.push(x);
        } else {
            let r = rand_index(rng, self.members.len());
            self.members[r] = x;
        }
    }

    /// Remove random members until at most `capacity` remain.
    pub(crate) fn shrink(&mut self, rng: &mut Rng, capacity: usize) {
        while self.members.len() > capacity {
            let r = rand_index(rng, self.members.len());
            self.members.swap_remove(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lehmer_mean_matches_definition() {
        let m = weighted_lehmer_mean(&[0.2, 0.6], &[1.0, 3.0]).unwrap();
        let expected = (0.25 * 0.04 + 0.75 * 0.36) / (0.25 * 0.2 + 0.75 * 0.6);
        assert!((m - expected).abs() < 1e-12);
        assert_eq!(weighted_lehmer_mean(&[], &[]), None);
        assert_eq!(weighted_lehmer_mean(&[0.0, 0.0], &[1.0, 1.0]), None);
        let inf = weighted_lehmer_mean(&[0.3, 0.9], &[f64::INFINITY, 2.0]).unwrap();
        assert!((inf - 0.3).abs() < 1e-12);
        let zero = weighted_lehmer_mean(&[0.5, 0.5], &[0.0, 0.0]).unwrap();
        assert!((zero - 0.5).abs() < 1e-12);
    }

    #[test]
    fn lpsr_endpoints() {
        assert_eq!(lpsr_size(180, 4, 0, 1000), 180);
        assert_eq!(lpsr_size(180, 4, 1000, 1000), 4);
        assert_eq!(lpsr_size(180, 4, 500, 1000), 92);
        assert_eq!(lpsr_size(180, 4, 5000, 1000), 4);
    }

    #[test]
    fn midpoint_repair_stays_inside() {
        let (lo, hi) = ([-1.0, -1.0], [1.0, 1.0]);
        let mut v = [-3.0, 5.0];
        repair_midpoint(&mut v, &[0.5, 0.5], &lo, &hi);
        assert_eq!(v, [-0.25, 0.75]);
    }

    #[test]
    fn rank_selector_prefers_good_ranks() {
        let sel = RankSelector::new(10, 3.0);
        let mut rng = Rng::new(3);
        let mut counts = [0usize; 10];
        for _ in 0..20_000 {
            counts[sel.sample(&mut rng)] += 1;
        }
        assert!(counts[0] > counts[9] * 10);
    }

    #[test]
    fn archive_insert_always_enters_when_full() {
        let mut rng = Rng::new(9);
        let mut archive = Archive::default();
        for v in 0..3 {
            archive.push(&mut rng, vec![v as f64], 3);
        }
        for v in 3..60 {
            archive.push(&mut rng, vec![v as f64], 3);
            assert_eq!(archive.len(), 3);
            assert!(
                (0..3).any(|i| archive.get(i)[0] == v as f64),
                "member {v} missing"
            );
        }
        archive.push(&mut rng, vec![-1.0], 0);
        assert_eq!(archive.len(), 0);
    }

    #[test]
    fn cauchy_f_is_in_unit_interval() {
        let mut rng = Rng::new(5);
        for _ in 0..10_000 {
            let f = sample_f_cauchy(&mut rng, 0.5, 0.1);
            assert!(f > 0.0 && f <= 1.0);
        }
    }
}
