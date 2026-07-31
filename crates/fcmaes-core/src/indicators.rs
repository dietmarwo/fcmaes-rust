//! Auditable quality indicators for minimized multi-objective fronts.
//!
//! Every function rejects empty, non-finite, or dimensionally inconsistent
//! input. Hypervolume additionally requires an explicit reference point that
//! is weakly worse than every approximation point. Exact and sampled results
//! are different enum variants so an approximation cannot be reported as an
//! exact value accidentally.

use std::error::Error;
use std::fmt;

use crate::Rng;

const DEFAULT_MONTE_CARLO_SAMPLES: usize = 100_000;
const DEFAULT_MONTE_CARLO_SEED: u64 = 0x4856_2d4d_4f4e_5445;

/// Validation failures returned by quality indicators.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndicatorError {
    /// A front or reference set was empty.
    EmptySet,
    /// A point had no objective coordinates.
    EmptyPoint,
    /// Point and reference dimensions differed.
    DimensionMismatch,
    /// An objective coordinate was NaN or infinite.
    NonFiniteValue,
    /// A hypervolume point lay outside the reference box.
    ReferencePointViolation,
    /// A sampled estimate requested fewer than two samples.
    InsufficientSamples,
}

impl fmt::Display for IndicatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptySet => "indicator sets must not be empty",
            Self::EmptyPoint => "indicator points must have at least one objective",
            Self::DimensionMismatch => "indicator point dimensions must match",
            Self::NonFiniteValue => "indicator coordinates must be finite",
            Self::ReferencePointViolation => {
                "hypervolume points must be weakly better than the reference point"
            }
            Self::InsufficientSamples => "Monte Carlo hypervolume requires at least two samples",
        };
        formatter.write_str(message)
    }
}

impl Error for IndicatorError {}

/// Explicit, validated hypervolume reference point for minimized objectives.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferencePoint(Vec<f64>);

impl ReferencePoint {
    /// Validate and construct a reference point.
    ///
    /// # Errors
    ///
    /// Returns [`IndicatorError::EmptyPoint`] for zero objectives and
    /// [`IndicatorError::NonFiniteValue`] for NaN or infinite coordinates.
    pub fn new(coordinates: Vec<f64>) -> Result<Self, IndicatorError> {
        if coordinates.is_empty() {
            return Err(IndicatorError::EmptyPoint);
        }
        if coordinates.iter().any(|value| !value.is_finite()) {
            return Err(IndicatorError::NonFiniteValue);
        }
        Ok(Self(coordinates))
    }

    /// Borrow the reference coordinates.
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// Number of minimized objectives.
    pub fn dimension(&self) -> usize {
        self.0.len()
    }
}

/// Exact or explicitly sampled hypervolume estimate.
#[derive(Clone, Debug, PartialEq)]
pub enum HypervolumeEstimate {
    /// Exact union volume of the dominated axis-aligned boxes.
    Exact(f64),
    /// Uniform Monte Carlo estimate and its Bernoulli standard error.
    MonteCarlo {
        /// Estimated dominated volume.
        value: f64,
        /// One-standard-deviation sampling uncertainty.
        standard_error: f64,
        /// Number of uniform samples used.
        samples: usize,
        /// Seed used by the deterministic sampler.
        seed: u64,
    },
}

impl HypervolumeEstimate {
    /// Numeric volume regardless of exact or sampled provenance.
    pub fn value(&self) -> f64 {
        match self {
            Self::Exact(value) | Self::MonteCarlo { value, .. } => *value,
        }
    }
}

/// Hypervolume value plus deterministic front-cleanup accounting.
#[derive(Clone, Debug, PartialEq)]
pub struct HypervolumeReport {
    /// Exact or Monte Carlo volume.
    pub estimate: HypervolumeEstimate,
    /// Number of points supplied by the caller.
    pub input_points: usize,
    /// Number of unique nondominated points used in the calculation.
    pub retained_points: usize,
    /// Exact duplicate points removed before dominance filtering.
    pub duplicates_collapsed: usize,
    /// Unique dominated points removed before integration.
    pub dominated_removed: usize,
    /// Points excluded because at least one coordinate was worse than the
    /// reference point. Always zero under [`OutsidePolicy::Strict`].
    pub outside_reference: usize,
}

/// Policy for hypervolume points outside the minimized reference box.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutsidePolicy {
    /// Reject the complete front with [`IndicatorError::ReferencePointViolation`].
    Strict,
    /// Exclude outside points, report their count, and never clip coordinates.
    Exclude,
}

fn same_point(left: &[f64], right: &[f64]) -> bool {
    left.iter().zip(right).all(|(a, b)| a == b)
}

fn dominates(left: &[f64], right: &[f64]) -> bool {
    let mut strict = false;
    for (&a, &b) in left.iter().zip(right) {
        if a > b {
            return false;
        }
        strict |= a < b;
    }
    strict
}

struct CleanedHypervolumeFront {
    points: Vec<Vec<f64>>,
    duplicates: usize,
    dominated: usize,
    outside: usize,
}

fn clean_hypervolume_front(
    front: &[Vec<f64>],
    reference: &ReferencePoint,
    policy: OutsidePolicy,
) -> Result<CleanedHypervolumeFront, IndicatorError> {
    validate_set(front, Some(reference.dimension()))?;
    let outside = |point: &Vec<f64>| {
        point
            .iter()
            .zip(reference.as_slice())
            .any(|(&value, &limit)| value > limit)
    };
    let outside_reference = front.iter().filter(|point| outside(point)).count();
    if policy == OutsidePolicy::Strict && outside_reference > 0 {
        return Err(IndicatorError::ReferencePointViolation);
    }

    let mut unique: Vec<Vec<f64>> = front
        .iter()
        .filter(|point| !outside(point))
        .cloned()
        .collect();
    unique.sort_by(|left, right| {
        left.iter()
            .zip(right)
            .find_map(|(a, b)| (a != b).then(|| a.total_cmp(b)))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    unique.dedup_by(|left, right| same_point(left, right));
    let duplicates = front.len() - outside_reference - unique.len();
    let retained: Vec<Vec<f64>> = unique
        .iter()
        .enumerate()
        .filter(|(index, point)| {
            !unique
                .iter()
                .enumerate()
                .any(|(other, candidate)| other != *index && dominates(candidate, point))
        })
        .map(|(_, point)| point.clone())
        .collect();
    let dominated = unique.len() - retained.len();
    Ok(CleanedHypervolumeFront {
        points: retained,
        duplicates,
        dominated,
        outside: outside_reference,
    })
}

fn exact_union(points: &[&[f64]], reference: &[f64], dimensions: usize) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    if dimensions == 1 {
        let minimum = points
            .iter()
            .map(|point| point[0])
            .fold(reference[0], f64::min);
        return (reference[0] - minimum).max(0.0);
    }
    let axis = dimensions - 1;
    let mut cuts: Vec<f64> = points
        .iter()
        .map(|point| point[axis])
        .filter(|value| *value < reference[axis])
        .collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|left, right| left == right);
    let mut volume = 0.0;
    for (index, &lower) in cuts.iter().enumerate() {
        let upper = cuts.get(index + 1).copied().unwrap_or(reference[axis]);
        if upper <= lower {
            continue;
        }
        let active: Vec<&[f64]> = points
            .iter()
            .copied()
            .filter(|point| point[axis] <= lower)
            .collect();
        volume += (upper - lower) * exact_union(&active, reference, axis);
    }
    volume
}

fn report(
    input_points: usize,
    retained: &[Vec<f64>],
    duplicates_collapsed: usize,
    dominated_removed: usize,
    outside_reference: usize,
    estimate: HypervolumeEstimate,
) -> HypervolumeReport {
    HypervolumeReport {
        estimate,
        input_points,
        retained_points: retained.len(),
        duplicates_collapsed,
        dominated_removed,
        outside_reference,
    }
}

/// Compute exact hypervolume through four objectives and deterministic Monte
/// Carlo hypervolume above four objectives.
///
/// The default approximation uses 100,000 samples and a fixed seed. Published
/// experiments should call [`hypervolume_monte_carlo`] with an explicit seed.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for empty, non-finite, dimensionally
/// inconsistent input or for a point outside the reference box.
pub fn hypervolume(
    front: &[Vec<f64>],
    reference: &ReferencePoint,
) -> Result<HypervolumeReport, IndicatorError> {
    hypervolume_with(front, reference, OutsidePolicy::Strict)
}

/// Compute hypervolume with an explicit outside-reference policy.
///
/// Exact integration is used through four objectives and deterministic Monte
/// Carlo integration above four. [`OutsidePolicy::Exclude`] removes points
/// with an empty dominated box and records the count in
/// [`HypervolumeReport::outside_reference`]; it never clips coordinates.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for empty, non-finite, or dimensionally
/// inconsistent input. Strict policy also rejects any point outside the
/// reference box.
pub fn hypervolume_with(
    front: &[Vec<f64>],
    reference: &ReferencePoint,
    policy: OutsidePolicy,
) -> Result<HypervolumeReport, IndicatorError> {
    if reference.dimension() > 4 {
        return hypervolume_monte_carlo_impl(
            front,
            reference,
            DEFAULT_MONTE_CARLO_SAMPLES,
            DEFAULT_MONTE_CARLO_SEED,
            policy,
        );
    }
    let cleaned = clean_hypervolume_front(front, reference, policy)?;
    let borrowed: Vec<&[f64]> = cleaned.points.iter().map(Vec::as_slice).collect();
    let value = exact_union(&borrowed, reference.as_slice(), reference.dimension());
    Ok(report(
        front.len(),
        &cleaned.points,
        cleaned.duplicates,
        cleaned.dominated,
        cleaned.outside,
        HypervolumeEstimate::Exact(value),
    ))
}

/// Estimate hypervolume uniformly inside the front/reference bounding box.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for invalid front/reference data or when
/// `samples < 2`.
pub fn hypervolume_monte_carlo(
    front: &[Vec<f64>],
    reference: &ReferencePoint,
    samples: usize,
    seed: u64,
) -> Result<HypervolumeReport, IndicatorError> {
    hypervolume_monte_carlo_impl(front, reference, samples, seed, OutsidePolicy::Strict)
}

fn hypervolume_monte_carlo_impl(
    front: &[Vec<f64>],
    reference: &ReferencePoint,
    samples: usize,
    seed: u64,
    policy: OutsidePolicy,
) -> Result<HypervolumeReport, IndicatorError> {
    if samples < 2 {
        return Err(IndicatorError::InsufficientSamples);
    }
    let cleaned = clean_hypervolume_front(front, reference, policy)?;
    let dimensions = reference.dimension();
    let lower: Vec<f64> = (0..dimensions)
        .map(|axis| {
            cleaned
                .points
                .iter()
                .map(|point| point[axis])
                .fold(reference.as_slice()[axis], f64::min)
        })
        .collect();
    let bounding_volume: f64 = lower
        .iter()
        .zip(reference.as_slice())
        .map(|(&lo, &hi)| hi - lo)
        .product();
    let mut rng = Rng::new(seed);
    let mut dominated_samples = 0usize;
    for _ in 0..samples {
        let sample: Vec<f64> = lower
            .iter()
            .zip(reference.as_slice())
            .map(|(&lo, &hi)| lo + (hi - lo) * rng.uniform01())
            .collect();
        if cleaned.points.iter().any(|point| {
            point
                .iter()
                .zip(&sample)
                .all(|(&value, &draw)| value <= draw)
        }) {
            dominated_samples += 1;
        }
    }
    let probability = dominated_samples as f64 / samples as f64;
    let value = bounding_volume * probability;
    let standard_error =
        bounding_volume * (probability * (1.0 - probability) / samples as f64).sqrt();
    Ok(report(
        front.len(),
        &cleaned.points,
        cleaned.duplicates,
        cleaned.dominated,
        cleaned.outside,
        HypervolumeEstimate::MonteCarlo {
            value,
            standard_error,
            samples,
            seed,
        },
    ))
}

fn validate_set(points: &[Vec<f64>], dimension: Option<usize>) -> Result<usize, IndicatorError> {
    let Some(first) = points.first() else {
        return Err(IndicatorError::EmptySet);
    };
    if first.is_empty() {
        return Err(IndicatorError::EmptyPoint);
    }
    let dimension = dimension.unwrap_or(first.len());
    if points.iter().any(|point| point.len() != dimension) {
        return Err(IndicatorError::DimensionMismatch);
    }
    if points.iter().flatten().any(|value| !value.is_finite()) {
        return Err(IndicatorError::NonFiniteValue);
    }
    Ok(dimension)
}

/// Partition minimized objective vectors into successive non-dominated fronts.
///
/// Returned values are original input indices. Exact duplicate points do not
/// dominate one another and therefore remain together in the same front.
/// Front zero is the Pareto set; removing it and repeating produces each later
/// front.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for empty, non-finite, empty-point, or
/// dimensionally inconsistent input.
pub fn nondominated_sort(points: &[Vec<f64>]) -> Result<Vec<Vec<usize>>, IndicatorError> {
    validate_set(points, None)?;
    let count = points.len();
    let mut dominates_indices = vec![Vec::new(); count];
    let mut domination_count = vec![0usize; count];
    for left in 0..count {
        for right in left + 1..count {
            if dominates(&points[left], &points[right]) {
                dominates_indices[left].push(right);
                domination_count[right] += 1;
            } else if dominates(&points[right], &points[left]) {
                dominates_indices[right].push(left);
                domination_count[left] += 1;
            }
        }
    }
    let mut current: Vec<usize> = domination_count
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value == 0).then_some(index))
        .collect();
    let mut fronts = Vec::new();
    while !current.is_empty() {
        let mut next = Vec::new();
        for &index in &current {
            for &dominated in &dominates_indices[index] {
                domination_count[dominated] -= 1;
                if domination_count[dominated] == 0 {
                    next.push(dominated);
                }
            }
        }
        fronts.push(current);
        current = next;
    }
    Ok(fronts)
}

/// Compute normalized NSGA-II crowding distances for one supplied front.
///
/// Every point attaining a finite objective's minimum or maximum receives
/// infinity. Exact duplicates are retained; duplicate interior points can
/// consequently receive zero distance. An objective with zero range adds no
/// distance. For one or two points, every distance is infinite.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for empty, non-finite, empty-point, or
/// dimensionally inconsistent input.
pub fn crowding_distance(points: &[Vec<f64>]) -> Result<Vec<f64>, IndicatorError> {
    validate_set(points, None)?;
    let count = points.len();
    if count <= 2 {
        return Ok(vec![f64::INFINITY; count]);
    }
    let mut distance = vec![0.0; count];
    for (objective, _) in points[0].iter().enumerate() {
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&left, &right| points[left][objective].total_cmp(&points[right][objective]));
        let minimum = points[order[0]][objective];
        let maximum = points[order[count - 1]][objective];
        let span = maximum - minimum;
        if span <= 0.0 {
            continue;
        }
        for &index in &order {
            if points[index][objective] == minimum || points[index][objective] == maximum {
                distance[index] = f64::INFINITY;
            }
        }
        for position in 1..count - 1 {
            let index = order[position];
            if distance[index].is_finite() {
                distance[index] += (points[order[position + 1]][objective]
                    - points[order[position - 1]][objective])
                    / span;
            }
        }
    }
    Ok(distance)
}

fn validate_pair(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<usize, IndicatorError> {
    let dimension = validate_set(left, None)?;
    validate_set(right, Some(dimension))?;
    Ok(dimension)
}

fn euclidean(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn plus_distance(approximation: &[f64], reference: &[f64]) -> f64 {
    approximation
        .iter()
        .zip(reference)
        .map(|(&a, &r)| (a - r).max(0.0).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn mean_min_distance(
    sources: &[Vec<f64>],
    targets: &[Vec<f64>],
    distance: impl Fn(&[f64], &[f64]) -> f64,
) -> f64 {
    sources
        .iter()
        .map(|source| {
            targets
                .iter()
                .map(|target| distance(target, source))
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>()
        / sources.len() as f64
}

/// Inverted generational distance: mean nearest Euclidean distance from each
/// reference point to the approximation front.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for invalid or inconsistent sets.
pub fn igd(front: &[Vec<f64>], reference_set: &[Vec<f64>]) -> Result<f64, IndicatorError> {
    validate_pair(front, reference_set)?;
    Ok(mean_min_distance(reference_set, front, euclidean))
}

/// IGD+ using only approximation coordinates that are worse than a reference
/// point under minimization.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for invalid or inconsistent sets.
pub fn igd_plus(front: &[Vec<f64>], reference_set: &[Vec<f64>]) -> Result<f64, IndicatorError> {
    validate_pair(front, reference_set)?;
    Ok(mean_min_distance(reference_set, front, plus_distance))
}

/// Generational distance: mean nearest Euclidean distance from each
/// approximation point to the reference set.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for invalid or inconsistent sets.
pub fn gd(front: &[Vec<f64>], reference_set: &[Vec<f64>]) -> Result<f64, IndicatorError> {
    validate_pair(front, reference_set)?;
    Ok(mean_min_distance(front, reference_set, euclidean))
}

/// GD+ using only approximation coordinates that are worse than a reference
/// point under minimization.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for invalid or inconsistent sets.
pub fn gd_plus(front: &[Vec<f64>], reference_set: &[Vec<f64>]) -> Result<f64, IndicatorError> {
    validate_pair(front, reference_set)?;
    Ok(front
        .iter()
        .map(|approximation| {
            reference_set
                .iter()
                .map(|reference| plus_distance(approximation, reference))
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>()
        / front.len() as f64)
}

/// Unary additive epsilon indicator `Iε+(front, reference_set)` for minimized
/// objectives. Smaller values are better and negative values indicate strict
/// improvement over the complete reference set.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for invalid or inconsistent sets.
pub fn additive_epsilon(
    front: &[Vec<f64>],
    reference_set: &[Vec<f64>],
) -> Result<f64, IndicatorError> {
    validate_pair(front, reference_set)?;
    Ok(reference_set
        .iter()
        .map(|reference| {
            front
                .iter()
                .map(|approximation| {
                    approximation
                        .iter()
                        .zip(reference)
                        .map(|(&a, &r)| a - r)
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max))
}

/// Standard spacing of nearest-neighbor Manhattan distances within a front.
/// A single-point front has zero spacing.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for invalid points.
pub fn spacing(front: &[Vec<f64>]) -> Result<f64, IndicatorError> {
    validate_set(front, None)?;
    if front.len() == 1 {
        return Ok(0.0);
    }
    let nearest: Vec<f64> = front
        .iter()
        .enumerate()
        .map(|(index, point)| {
            front
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, candidate)| {
                    point
                        .iter()
                        .zip(candidate)
                        .map(|(&a, &b)| (a - b).abs())
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let mean = nearest.iter().sum::<f64>() / nearest.len() as f64;
    Ok((nearest
        .iter()
        .map(|distance| (distance - mean).powi(2))
        .sum::<f64>()
        / (nearest.len() - 1) as f64)
        .sqrt())
}

/// Generalized Deb spread using nearest-neighbor Euclidean distances and
/// explicit extreme points. Zero is perfectly even; larger is less uniform.
///
/// # Errors
///
/// Returns an [`IndicatorError`] for invalid/inconsistent sets or fewer than
/// two front points.
pub fn spread(front: &[Vec<f64>], extremes: &[Vec<f64>]) -> Result<f64, IndicatorError> {
    validate_pair(front, extremes)?;
    if front.len() < 2 {
        return Err(IndicatorError::EmptySet);
    }
    let nearest: Vec<f64> = front
        .iter()
        .enumerate()
        .map(|(index, point)| {
            front
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, candidate)| euclidean(point, candidate))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let mean = nearest.iter().sum::<f64>() / nearest.len() as f64;
    let edge = extremes
        .iter()
        .map(|extreme| {
            front
                .iter()
                .map(|point| euclidean(extreme, point))
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>();
    let deviation = nearest
        .iter()
        .map(|distance| (distance - mean).abs())
        .sum::<f64>();
    let denominator = edge + nearest.len() as f64 * mean;
    Ok(if denominator == 0.0 {
        0.0
    } else {
        (edge + deviation) / denominator
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(values: &[f64]) -> ReferencePoint {
        ReferencePoint::new(values.to_vec()).unwrap()
    }

    #[test]
    fn exact_hypervolume_has_auditable_cleanup() {
        let front = vec![
            vec![1.0, 4.0],
            vec![2.0, 2.0],
            vec![4.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let result = hypervolume(&front, &reference(&[5.0, 5.0])).unwrap();
        assert_eq!(result.estimate, HypervolumeEstimate::Exact(11.0));
        assert_eq!(result.input_points, 5);
        assert_eq!(result.retained_points, 3);
        assert_eq!(result.duplicates_collapsed, 1);
        assert_eq!(result.dominated_removed, 1);
        assert_eq!(result.outside_reference, 0);
    }

    #[test]
    fn outside_reference_policy_is_explicit_and_audited() {
        let front = vec![
            vec![1.0, 4.0],
            vec![2.0, 2.0],
            vec![6.0, 1.0],
            vec![7.0, 0.0],
        ];
        let reference = reference(&[5.0, 5.0]);
        assert_eq!(
            hypervolume(&front, &reference),
            Err(IndicatorError::ReferencePointViolation)
        );
        let report = hypervolume_with(&front, &reference, OutsidePolicy::Exclude).unwrap();
        assert_eq!(report.estimate, HypervolumeEstimate::Exact(10.0));
        assert_eq!(report.input_points, 4);
        assert_eq!(report.retained_points, 2);
        assert_eq!(report.outside_reference, 2);
        assert_eq!(report.duplicates_collapsed, 0);
        assert_eq!(report.dominated_removed, 0);

        let outside_only =
            hypervolume_with(&[vec![6.0, 1.0]], &reference, OutsidePolicy::Exclude).unwrap();
        assert_eq!(outside_only.estimate, HypervolumeEstimate::Exact(0.0));
        assert_eq!(outside_only.retained_points, 0);
        assert_eq!(outside_only.outside_reference, 1);
    }

    #[test]
    fn exact_recursive_volume_handles_four_dimensions() {
        let front = vec![vec![0.0; 4], vec![0.5; 4]];
        let result = hypervolume(&front, &reference(&[1.0; 4])).unwrap();
        assert_eq!(result.estimate, HypervolumeEstimate::Exact(1.0));
        assert_eq!(result.dominated_removed, 1);
    }

    #[test]
    fn exact_two_dimensional_volume_matches_an_independent_cell_union() {
        let hv_reference = reference(&[10.0, 10.0]);
        let mut rng = Rng::new(19);
        for case in 0..1_000 {
            let point_count = 1 + case % 7;
            let front: Vec<Vec<f64>> = (0..point_count)
                .map(|_| {
                    vec![
                        (9.0 * rng.uniform01()).floor(),
                        (9.0 * rng.uniform01()).floor(),
                    ]
                })
                .collect();
            let brute_force_cells = (0..10)
                .flat_map(|x| (0..10).map(move |y| (x, y)))
                .filter(|&(x, y)| {
                    front
                        .iter()
                        .any(|point| point[0] <= x as f64 && point[1] <= y as f64)
                })
                .count() as f64;
            assert_eq!(
                hypervolume(&front, &hv_reference).unwrap().estimate.value(),
                brute_force_cells
            );
        }
    }

    #[test]
    fn sampled_volume_agrees_with_exact_within_reported_uncertainty() {
        let front = vec![
            vec![0.1, 0.8, 0.8, 0.8],
            vec![0.8, 0.1, 0.8, 0.8],
            vec![0.8, 0.8, 0.1, 0.8],
            vec![0.8, 0.8, 0.8, 0.1],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        let exact = hypervolume(&front, &reference(&[1.0; 4]))
            .unwrap()
            .estimate
            .value();
        let sampled = hypervolume_monte_carlo(&front, &reference(&[1.0; 4]), 500_000, 7).unwrap();
        let HypervolumeEstimate::MonteCarlo {
            value,
            standard_error,
            samples,
            seed,
        } = sampled.estimate
        else {
            panic!("sampled API returned exact hypervolume")
        };
        assert_eq!(samples, 500_000);
        assert_eq!(seed, 7);
        assert!((value - exact).abs() <= 3.0 * standard_error);
    }

    #[test]
    fn high_dimensional_default_is_typed_as_monte_carlo() {
        let result = hypervolume(&[vec![0.0; 5]], &reference(&[1.0; 5])).unwrap();
        assert!(matches!(
            result.estimate,
            HypervolumeEstimate::MonteCarlo {
                samples: 100_000,
                ..
            }
        ));
    }

    #[test]
    fn distances_and_epsilon_match_hand_computed_fixture() {
        let front = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let ideal = vec![vec![1.0, 1.0]];
        assert_eq!(igd(&front, &ideal).unwrap(), 1.0);
        assert_eq!(igd_plus(&front, &ideal).unwrap(), 1.0);
        assert_eq!(gd(&front, &ideal).unwrap(), 1.0);
        assert_eq!(gd_plus(&front, &ideal).unwrap(), 1.0);
        assert_eq!(additive_epsilon(&front, &ideal).unwrap(), 1.0);
        assert_eq!(spacing(&front).unwrap(), 0.0);
    }

    #[test]
    fn nondominated_sort_preserves_indices_and_duplicates() {
        let points = vec![
            vec![0.0, 2.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 2.0],
            vec![1.0, 1.0],
            vec![3.0, 3.0],
        ];
        assert_eq!(
            nondominated_sort(&points).unwrap(),
            vec![vec![0, 1, 2, 4], vec![3], vec![5]]
        );
    }

    #[test]
    fn crowding_distance_matches_nsga_fixture() {
        let distance =
            crowding_distance(&[vec![0.0, 2.0], vec![1.0, 1.0], vec![2.0, 0.0]]).unwrap();
        assert!(distance[0].is_infinite());
        assert_eq!(distance[1], 2.0);
        assert!(distance[2].is_infinite());
        assert_eq!(
            crowding_distance(&[vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]]).unwrap(),
            vec![0.0; 3]
        );
    }

    #[test]
    fn translation_and_positive_scaling_have_correct_factors() {
        fn transform(points: &[Vec<f64>], scale: f64, offset: f64) -> Vec<Vec<f64>> {
            points
                .iter()
                .map(|point| point.iter().map(|value| scale * value + offset).collect())
                .collect()
        }

        fn close(actual: f64, expected: f64) {
            assert!((actual - expected).abs() <= 1.0e-12 * expected.abs().max(1.0));
        }

        let front = vec![vec![1.0, 4.0], vec![2.0, 2.0], vec![4.5, 1.0]];
        let reference_set = vec![vec![0.5, 4.5], vec![2.5, 2.5], vec![4.5, 0.5]];
        let extremes = vec![reference_set[0].clone(), reference_set[2].clone()];
        let translated = transform(&front, 1.0, 10.0);
        let translated_reference = transform(&reference_set, 1.0, 10.0);
        let translated_extremes = transform(&extremes, 1.0, 10.0);
        let scaled = transform(&front, 3.0, 0.0);
        let scaled_reference = transform(&reference_set, 3.0, 0.0);
        let scaled_extremes = transform(&extremes, 3.0, 0.0);
        let base = hypervolume(&front, &reference(&[5.0, 5.0]))
            .unwrap()
            .estimate
            .value();
        let shifted = hypervolume(&translated, &reference(&[15.0, 15.0]))
            .unwrap()
            .estimate
            .value();
        let expanded = hypervolume(&scaled, &reference(&[15.0, 15.0]))
            .unwrap()
            .estimate
            .value();
        close(shifted, base);
        close(expanded, 9.0 * base);

        let distances = [
            igd(&front, &reference_set).unwrap(),
            igd_plus(&front, &reference_set).unwrap(),
            gd(&front, &reference_set).unwrap(),
            gd_plus(&front, &reference_set).unwrap(),
            additive_epsilon(&front, &reference_set).unwrap(),
            spacing(&front).unwrap(),
        ];
        let translated_distances = [
            igd(&translated, &translated_reference).unwrap(),
            igd_plus(&translated, &translated_reference).unwrap(),
            gd(&translated, &translated_reference).unwrap(),
            gd_plus(&translated, &translated_reference).unwrap(),
            additive_epsilon(&translated, &translated_reference).unwrap(),
            spacing(&translated).unwrap(),
        ];
        let scaled_distances = [
            igd(&scaled, &scaled_reference).unwrap(),
            igd_plus(&scaled, &scaled_reference).unwrap(),
            gd(&scaled, &scaled_reference).unwrap(),
            gd_plus(&scaled, &scaled_reference).unwrap(),
            additive_epsilon(&scaled, &scaled_reference).unwrap(),
            spacing(&scaled).unwrap(),
        ];
        for ((original, shifted), expanded) in distances
            .into_iter()
            .zip(translated_distances)
            .zip(scaled_distances)
        {
            close(shifted, original);
            close(expanded, 3.0 * original);
        }
        let base_spread = spread(&front, &extremes).unwrap();
        close(
            spread(&translated, &translated_extremes).unwrap(),
            base_spread,
        );
        close(spread(&scaled, &scaled_extremes).unwrap(), base_spread);
    }

    #[test]
    fn strict_dominance_improves_compliant_indicators_for_ten_thousand_pairs() {
        let ideal = vec![vec![0.0, 0.0]];
        let hv_reference = reference(&[2.0, 2.0]);
        let mut rng = Rng::new(42);
        for _ in 0..10_000 {
            let worse = vec![vec![0.2 + rng.uniform01(), 0.2 + rng.uniform01()]];
            let better = vec![vec![worse[0][0] - 0.1, worse[0][1] - 0.1]];
            assert!(
                hypervolume(&better, &hv_reference)
                    .unwrap()
                    .estimate
                    .value()
                    > hypervolume(&worse, &hv_reference).unwrap().estimate.value()
            );
            assert!(igd_plus(&better, &ideal).unwrap() < igd_plus(&worse, &ideal).unwrap());
            assert!(
                additive_epsilon(&better, &ideal).unwrap()
                    < additive_epsilon(&worse, &ideal).unwrap()
            );
        }
    }

    #[test]
    fn invalid_input_fails_closed() {
        assert_eq!(ReferencePoint::new(vec![]), Err(IndicatorError::EmptyPoint));
        assert_eq!(
            ReferencePoint::new(vec![f64::NAN]),
            Err(IndicatorError::NonFiniteValue)
        );
        assert_eq!(
            hypervolume(&[], &reference(&[1.0])),
            Err(IndicatorError::EmptySet)
        );
        assert_eq!(
            hypervolume(&[vec![2.0]], &reference(&[1.0])),
            Err(IndicatorError::ReferencePointViolation)
        );
        assert_eq!(
            igd(&[vec![0.0]], &[vec![0.0, 1.0]]),
            Err(IndicatorError::DimensionMismatch)
        );
        assert_eq!(nondominated_sort(&[]), Err(IndicatorError::EmptySet));
        assert_eq!(
            crowding_distance(&[vec![0.0], vec![f64::INFINITY]]),
            Err(IndicatorError::NonFiniteValue)
        );
    }
}
