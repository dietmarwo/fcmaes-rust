// Copyright (c) Dietmar Wolz (fcmaes adaptation).
// Copyright (c) 2026 Aleksey Vaneev (original archive idea and C++ source).
// Copyright (c) 2026 bitehv2-rust contributors (Rust implementation).
// MIT license; retain NOTICE.pareto with copies of this module.
// Simplified to the cached-map design; see PARETO_ARCHIVE.md for provenance.
//! Insertion-only, unique-vector 2D Pareto archives with a fixed reference.
//!
//! All objectives are minimized. Bounds and scores are immutable; original
//! coordinates decide dominance. Hypervolume uses finite f64 arithmetic and
//! compensated sums, not certified intervals. A single-point numeric error
//! leaves membership and HV unchanged. Batch errors retain successful earlier
//! rows. Duplicate objectives retain the first payload.
//!
//! ```
//! use fcmaes_core::{ParetoArchive2D, ParetoBounds2D, ParetoScore};
//! let mut archive = ParetoArchive2D::new(
//!     ParetoBounds2D::nonnegative([4.0, 4.0])?, ParetoScore::Linear,
//! );
//! archive.add_batch(&[[1.0, 3.0], [3.0, 1.0]], &["left", "right"])?;
//! assert_eq!(archive.hypervolume(), 5.0);
//! assert_eq!(archive.front()[0], ([1.0, 3.0], "left"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::{cmp::Ordering, collections::BTreeMap, fmt};

/// Preference in fixed, bounded nonnegative coordinates, not Pareto layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Preference score used only for processing order and cached rejection choice.
pub enum Score {
    /// Mean of the two normalized objective coordinates.
    Linear,
    /// Mean of the squared normalized objective coordinates.
    SquaredEuclidean,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Invalid configuration or arithmetic; numeric errors leave membership and HV unchanged.
pub enum Error {
    /// Nonfinite or reversed objective bounds.
    InvalidBounds,
    /// Nonfinite objective or a point outside the fixed objective bounds.
    InvalidPoint,
    /// Preference normalization could not produce a finite score.
    NonFiniteScore,
    /// The new rectangle or accumulated hypervolume is not representable.
    NonFiniteArea,
    /// Objective and payload batch lengths differ.
    PayloadLength,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PayloadLength => "objective and payload batch lengths differ",
            Self::InvalidBounds => "bounds must be finite and lower <= reference",
            Self::InvalidPoint => "point must be finite and inside the fixed bounds",
            Self::NonFiniteScore => "score must be finite",
            Self::NonFiniteArea => "hypervolume arithmetic exceeds finite f64 range",
        })
    }
}
impl std::error::Error for Error {}

/// Batch insertion failed after the entries in `accepted` had been processed.
/// Shape errors have no row and leave the archive unchanged. Numeric errors
/// leave the failing row unapplied; earlier successful rows remain.
#[derive(Debug, PartialEq)]
pub struct BatchUpdateError {
    /// Failing row, or `None` for a mismatched payload batch length.
    pub row: Option<usize>,
    /// Acceptance decisions for the successfully processed prefix.
    pub accepted: Vec<bool>,
    /// Underlying numerical or shape error.
    pub source: Error,
}

impl fmt::Display for BatchUpdateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.row {
            Some(row) => write!(f, "row {row}: {}; earlier rows retained", self.source),
            None => self.source.fmt(f),
        }
    }
}

impl std::error::Error for BatchUpdateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Fixed origin and upper-right hypervolume reference. Signed inputs are allowed.
#[derive(Clone, Copy, Debug, PartialEq)]
/// Immutable finite objective lower bounds and hypervolume reference.
pub struct Bounds {
    lower: [f64; 2],
    reference: [f64; 2],
    lower_half: [f64; 2],
    width_half: [f64; 2],
}
impl Bounds {
    /// Construct objective bounds.
    ///
    /// # Errors
    /// Returns [`Error::InvalidBounds`] for nonfinite or reversed bounds.
    pub fn new(lower: [f64; 2], reference: [f64; 2]) -> Result<Self, Error> {
        if lower.iter().chain(&reference).any(|x| !x.is_finite())
            || (0..2).any(|j| lower[j] > reference[j])
        {
            return Err(Error::InvalidBounds);
        }
        let lower = lower.map(canonical);
        let reference = reference.map(canonical);
        let lower_half = lower.map(|x| x * 0.5);
        let width_half = std::array::from_fn(|j| reference[j] * 0.5 - lower_half[j]);
        Ok(Self {
            lower,
            reference,
            lower_half,
            width_half,
        })
    }
    /// Construct zero lower bounds with the supplied finite reference.
    ///
    /// # Errors
    /// Returns [`Error::InvalidBounds`] for a negative or nonfinite reference.
    pub fn nonnegative(reference: [f64; 2]) -> Result<Self, Error> {
        Self::new([0.0; 2], reference)
    }
    /// Return the fixed objective lower bounds.
    pub fn lower(&self) -> [f64; 2] {
        self.lower
    }
    /// Return the fixed hypervolume reference.
    pub fn reference(&self) -> [f64; 2] {
        self.reference
    }
    /// Compute a preference score without changing geometric dominance.
    ///
    /// # Errors
    /// Returns an error for invalid objective coordinates or nonfinite normalization.
    pub fn score(&self, point: [f64; 2], score: Score) -> Result<f64, Error> {
        self.prepare(point, score).map(|(_, s)| s)
    }
    fn prepare(&self, point: [f64; 2], score: Score) -> Result<([f64; 2], f64), Error> {
        let point = self.validate(point)?;
        self.prepare_validated(point, score).map(|s| (point, s))
    }
    #[inline]
    fn validate(&self, point: [f64; 2]) -> Result<[f64; 2], Error> {
        if (0..2).any(|j| {
            !point[j].is_finite() || point[j] < self.lower[j] || point[j] > self.reference[j]
        }) {
            return Err(Error::InvalidPoint);
        }
        Ok(point.map(canonical))
    }
    #[inline]
    fn prepare_validated(&self, point: [f64; 2], score: Score) -> Result<f64, Error> {
        // The common map is monotone. Subnormal collapse only creates ties;
        // ties are settled on the original coordinates, never on these keys.
        let u: [f64; 2] = std::array::from_fn(|j| {
            if self.width_half[j] == 0.0 {
                0.0
            } else {
                (point[j] * 0.5 - self.lower_half[j]) / self.width_half[j]
            }
        });
        let s = match score {
            Score::Linear => u[0] * 0.5 + u[1] * 0.5,
            Score::SquaredEuclidean => u[0] * u[0] * 0.5 + u[1] * u[1] * 0.5,
        };
        if !s.is_finite() || !(0.0..=1.0).contains(&s) {
            return Err(Error::NonFiniteScore);
        }
        Ok(canonical(s))
    }
}
fn canonical(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}
fn weak(p: [f64; 2], q: [f64; 2]) -> bool {
    p[0] <= q[0] && p[1] <= q[1]
}
fn coordinate_key(x: f64) -> u64 {
    let bits = canonical(x).to_bits();
    if bits >> 63 == 0 {
        bits ^ (1 << 63)
    } else {
        !bits
    }
}
fn compare(score: f64, point: [f64; 2], e: &Entry<impl Sized>) -> Ordering {
    score
        .total_cmp(&e.score)
        .then(point[0].total_cmp(&e.point[0]))
        .then(point[1].total_cmp(&e.point[1]))
}

#[derive(Clone, Debug, PartialEq)]
/// One unique nondominated objective pair and its owned payload.
pub struct Entry<P> {
    point: [f64; 2],
    score: f64,
    payload: P,
}
impl<P> Entry<P> {
    /// Return the original objective coordinates with canonical zero.
    pub fn point(&self) -> [f64; 2] {
        self.point
    }
    /// Return the fixed normalized preference score.
    pub fn score(&self) -> f64 {
        self.score
    }
    /// Borrow the payload retained with this objective pair.
    pub fn payload(&self) -> &P {
        &self.payload
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Result of inserting a point into an exact archive.
pub enum Update {
    /// A duplicate or weakly dominated point; the payload factory was not called.
    Rejected,
    /// The point was inserted; `removed` counts erased dominated entries.
    Accepted {
        /// Number of dominated entries removed by this insertion.
        removed: usize,
    },
}
impl Update {
    /// Whether this insertion added a new point.
    pub fn accepted(self) -> bool {
        matches!(self, Self::Accepted { .. })
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, x: f64) -> Result<(), Error> {
        if !x.is_finite() || x < 0.0 {
            return Err(Error::NonFiniteArea);
        }
        let y = x - self.correction;
        let next = self.value + y;
        if !next.is_finite() || next < 0.0 {
            return Err(Error::NonFiniteArea);
        }
        self.correction = (next - self.value) - y;
        self.value = next;
        Ok(())
    }
}
/// Positive rectangle area. Empty rectangles are handled before subtraction,
/// avoiding 0 * infinity for huge signed intervals.
fn rectangle(hi_x: f64, lo_x: f64, hi_y: f64, lo_y: f64) -> Result<f64, Error> {
    debug_assert!(hi_x >= lo_x && hi_y >= lo_y);
    if hi_x == lo_x || hi_y == lo_y {
        return Ok(0.0);
    }
    let width = hi_x - lo_x;
    let height = hi_y - lo_y;
    let area = width * height;
    if !width.is_finite() || !height.is_finite() || !area.is_finite() {
        return Err(Error::NonFiniteArea);
    }
    Ok(area)
}
#[cfg(test)]
fn recompute(mut points: Vec<[f64; 2]>, reference: [f64; 2]) -> Result<Sum, Error> {
    points.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let mut sum = Sum::default();
    let mut prev_y = reference[1];
    for p in points {
        debug_assert!(p[1] <= prev_y);
        sum.add(rectangle(reference[0], p[0], prev_y, p[1])?)?;
        prev_y = p[1];
    }
    Ok(sum)
}

/// Scalar-order port of the reviewed algorithm. Accepted inserts recompute HV;
/// deletions use one retain pass rather than repeated vector::erase.
#[cfg(test)]
struct ScoredArchive2d<P> {
    bounds: Bounds,
    score: Score,
    entries: Vec<Entry<P>>,
    area: Sum,
}
#[cfg(test)]
impl<P> ScoredArchive2d<P> {
    /// Construct an empty archive with immutable bounds and preference scoring.
    pub fn new(bounds: Bounds, score: Score) -> Self {
        Self {
            bounds,
            score,
            entries: Vec::new(),
            area: Sum::default(),
        }
    }
    /// Current dominated union area, maintained with compensated accumulation.
    pub fn hypervolume(&self) -> f64 {
        self.area.value
    }
    /// Borrow entries in increasing first-objective order without materialization.
    pub fn entries(&self) -> &[Entry<P>] {
        &self.entries
    }
    /// Borrow the lowest preference-score entry; does not compute Pareto layers.
    pub fn preferred(&self) -> Option<&Entry<P>> {
        self.entries.first()
    }
    /// Insert with an already constructed payload.
    ///
    /// # Errors
    /// Invalid coordinates or area arithmetic leave this archive unchanged.
    pub fn insert(&mut self, point: [f64; 2], payload: P) -> Result<Update, Error> {
        self.insert_with(point, || payload)
    }
    /// Construct/copy decision metadata only after the point is accepted.
    /// Insert with a payload factory called only after geometric acceptance and numeric checks.
    ///
    /// # Errors
    /// Invalid coordinates or area arithmetic leave membership and hypervolume unchanged.
    ///
    /// # Panics
    /// Propagates a panic from the payload factory before membership mutation.
    pub fn insert_with(
        &mut self,
        point: [f64; 2],
        make_payload: impl FnOnce() -> P,
    ) -> Result<Update, Error> {
        let (point, score) = self.bounds.prepare(point, self.score)?;
        let hi = self.entries.partition_point(|e| e.score <= score);
        if self.entries[..hi].iter().any(|e| weak(e.point, point)) {
            return Ok(Update::Rejected);
        }
        // Validate the entire proposed HV before modifying the archive.
        let mut remaining: Vec<_> = self
            .entries
            .iter()
            .filter(|e| !weak(point, e.point))
            .map(|e| e.point)
            .collect();
        remaining.push(point);
        let area = recompute(remaining, self.bounds.reference)?;
        let entry = Entry {
            point,
            score,
            payload: make_payload(),
        };
        let before = self.entries.len();
        self.entries.retain(|e| !weak(point, e.point));
        let removed = before - self.entries.len();
        let slot = self
            .entries
            .partition_point(|e| compare(score, point, e) == Ordering::Greater);
        self.entries.insert(slot, entry);
        self.area = area;
        Ok(Update::Accepted { removed })
    }
}

/// Exact insertion-only 2D minimization archive with incremental hypervolume.
/// One coordinate B-tree and a cached dominance witness. Predecessor rejection
/// and contiguous successor deletion decide membership on original coordinates.
/// Positive compensated HV increments are validated before any mutation.
/// Duplicate objective pairs keep the first payload. No capacity truncation occurs.
/// Acceptance shortcuts cache coordinates, never references into the B-tree.
/// Cloning is available for cloneable payloads and creates independent storage.
#[derive(Clone)]
pub struct ParetoArchive2D<P = ()> {
    bounds: Bounds,
    score: Score,
    entries: BTreeMap<u64, Entry<P>>,
    // Scratch keys separate numeric preflight from mutation on stable Rust.
    victims: Vec<u64>,
    area: Sum,
    best: Option<u64>,
    best_score: f64,
    best_point: [f64; 2],
    last: Option<[f64; 2]>,
    gap: Option<([f64; 2], [f64; 2])>,
}
impl<P> ParetoArchive2D<P> {
    /// Remove all entries, retaining the objective bounds and preference policy.
    pub fn clear(&mut self) {
        *self = Self::new(self.bounds, self.score);
    }

    /// Number of unique nondominated objective pairs.
    pub fn size(&self) -> usize {
        self.len()
    }

    /// Insert a point and an owned payload. Equivalent to [`Self::insert`].
    /// # Errors
    /// Invalid objectives or nonfinite area arithmetic leave the row unapplied.
    pub fn add(&mut self, point: [f64; 2], payload: P) -> Result<Update, Error> {
        self.insert(point, payload)
    }

    /// Insert a batch in supplied order, cloning payloads only for accepted rows.
    /// # Errors
    /// A length mismatch leaves the archive unchanged. Numeric failure returns
    /// the failing row and processed acceptance prefix; earlier updates remain.
    /// # Panics
    /// Propagates a payload clone panic before that row mutates membership.
    pub fn add_batch(
        &mut self,
        points: &[[f64; 2]],
        payloads: &[P],
    ) -> Result<Vec<bool>, BatchUpdateError>
    where
        P: Clone,
    {
        if points.len() != payloads.len() {
            return Err(BatchUpdateError {
                row: None,
                accepted: Vec::new(),
                source: Error::PayloadLength,
            });
        }
        let mut accepted = Vec::with_capacity(points.len());
        for (row, (&point, payload)) in points.iter().zip(payloads).enumerate() {
            match self.insert_with(point, || payload.clone()) {
                Ok(update) => accepted.push(update.accepted()),
                Err(source) => {
                    return Err(BatchUpdateError {
                        row: Some(row),
                        accepted,
                        source,
                    });
                }
            }
        }
        Ok(accepted)
    }

    /// Owned objective/payload snapshot in increasing first-objective order.
    /// # Panics
    /// Propagates a panic from a payload's clone implementation.
    pub fn front(&self) -> Vec<([f64; 2], P)>
    where
        P: Clone,
    {
        self.entries()
            .map(|entry| (entry.point, entry.payload.clone()))
            .collect()
    }
    /// Construct an empty archive with immutable bounds and preference scoring.
    pub fn new(bounds: Bounds, score: Score) -> Self {
        Self {
            bounds,
            score,
            entries: BTreeMap::new(),
            victims: Vec::new(),
            area: Sum::default(),
            best: None,
            best_score: 0.0,
            best_point: [0.0; 2],
            last: None,
            gap: None,
        }
    }
    /// Return the fixed objective bounds.
    pub fn bounds(&self) -> Bounds {
        self.bounds
    }
    /// Number of unique nondominated objective pairs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether the archive contains no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Current dominated union area, maintained with compensated accumulation.
    pub fn hypervolume(&self) -> f64 {
        self.area.value
    }
    /// Borrow entries in increasing first-objective order without materialization.
    pub fn entries(&self) -> impl DoubleEndedIterator<Item = &Entry<P>> {
        self.entries.values()
    }
    /// Borrow the lowest preference-score entry; does not compute Pareto layers.
    pub fn preferred(&self) -> Option<&Entry<P>> {
        self.best.and_then(|key| self.entries.get(&key))
    }
    /// Build the complete score order only when requested.
    pub fn ranked(&self) -> Vec<&Entry<P>> {
        let mut result: Vec<_> = self.entries().collect();
        result.sort_unstable_by(|a, b| compare(a.score, a.point, b));
        result
    }
    /// Insert with an already constructed payload.
    ///
    /// # Errors
    /// Invalid coordinates or area arithmetic leave this archive unchanged.
    pub fn insert(&mut self, point: [f64; 2], payload: P) -> Result<Update, Error> {
        self.insert_with(point, || payload)
    }
    /// Insert with a payload factory called only after geometric acceptance and numeric checks.
    ///
    /// # Errors
    /// Invalid coordinates or area arithmetic leave membership and hypervolume unchanged.
    ///
    /// # Panics
    /// Propagates a panic from the payload factory before membership mutation.
    pub fn insert_with(
        &mut self,
        point: [f64; 2],
        make_payload: impl FnOnce() -> P,
    ) -> Result<Update, Error> {
        // Validate even rejected input; score arithmetic is needed only after
        // geometry accepts. Within finite bounds its monotone normalization
        // cannot produce a nonfinite score for a rejected valid point.
        let point = self.bounds.validate(point)?;
        if self.best.is_some() && weak(self.best_point, point) {
            return Ok(Update::Rejected);
        }
        self.insert_validated(point, make_payload)
    }

    // Keep validation and common cached rejection small enough to inline.
    fn insert_validated(
        &mut self,
        point: [f64; 2],
        make_payload: impl FnOnce() -> P,
    ) -> Result<Update, Error> {
        if let Some(last) = self.last {
            if last[0] < point[0] && point[1] < last[1] {
                return self.insert_gap(point, last[1], None, make_payload);
            }
        }
        if let Some((previous, successor)) = self.gap {
            if previous[0] < point[0]
                && point[0] < successor[0]
                && successor[1] < point[1]
                && point[1] < previous[1]
            {
                return self.insert_gap(point, previous[1], Some(successor), make_payload);
            }
        }
        let key = coordinate_key(point[0]);
        let mut predecessors = self.entries.range(..=key).rev();
        let pred = predecessors.next();
        if pred.is_some_and(|(_, q)| q.point[1] <= point[1]) {
            return Ok(Update::Rejected);
        }
        let mut old_y = match pred {
            Some((&k, q)) if k < key => q.point[1],
            Some(_) => predecessors
                .next()
                .map_or(self.bounds.reference[1], |(_, q)| q.point[1]),
            None => self.bounds.reference[1],
        };
        let score = self.bounds.prepare_validated(point, self.score)?;
        let replace_best = self.best.is_none()
            || score
                .total_cmp(&self.best_score)
                .then(point[0].total_cmp(&self.best_point[0]))
                .then(point[1].total_cmp(&self.best_point[1]))
                == Ordering::Less;
        let mut old_x = point[0];
        let mut next_x = self.bounds.reference[0];
        let mut successor = None;
        let mut delta = Sum::default();
        self.victims.clear();
        for (&k, q) in self.entries.range(key..) {
            if q.point[1] < point[1] {
                next_x = q.point[0];
                successor = Some(q.point);
                break;
            }
            delta.add(rectangle(q.point[0], old_x, old_y, point[1])?)?;
            old_x = q.point[0];
            old_y = q.point[1];
            self.victims.push(k);
        }
        let removed = self.victims.len();
        delta.add(rectangle(next_x, old_x, old_y, point[1])?)?;
        let mut next_area = self.area;
        next_area.add(delta.value)?;
        let entry = Entry {
            point,
            score,
            payload: make_payload(),
        };
        // Payload construction and all numeric checks precede front mutation.
        debug_assert!(replace_best || !self.victims.iter().any(|&k| Some(k) == self.best));
        if removed == self.entries.len() {
            self.entries.clear();
        } else {
            for k in &self.victims {
                self.entries.remove(k);
            }
        }
        self.entries.insert(key, entry);
        if replace_best {
            self.best = Some(key);
            self.best_score = score;
            self.best_point = point;
        }
        self.area = next_area;
        self.gap = successor.map(|next| (point, next));
        if successor.is_none() {
            self.last = Some(point);
        }
        Ok(Update::Accepted { removed })
    }

    // Geometry establishes incomparability without range walks. Stable
    // BTreeMap insertion still costs O(log n); it has no stable hinted insert.
    fn insert_gap(
        &mut self,
        point: [f64; 2],
        previous_y: f64,
        successor: Option<[f64; 2]>,
        make_payload: impl FnOnce() -> P,
    ) -> Result<Update, Error> {
        let score = self.bounds.prepare_validated(point, self.score)?;
        let key = coordinate_key(point[0]);
        let next_x = successor.map_or(self.bounds.reference[0], |p| p[0]);
        let mut next_area = self.area;
        next_area.add(rectangle(next_x, point[0], previous_y, point[1])?)?;
        let entry = Entry {
            point,
            score,
            payload: make_payload(),
        };
        // Factory panics occur before mutation, including all cached state.
        self.entries.insert(key, entry);
        if self.best.is_none()
            || score
                .total_cmp(&self.best_score)
                .then(point[0].total_cmp(&self.best_point[0]))
                .then(point[1].total_cmp(&self.best_point[1]))
                == Ordering::Less
        {
            self.best = Some(key);
            self.best_score = score;
            self.best_point = point;
        }
        self.area = next_area;
        self.gap = successor.map(|next| (point, next));
        if successor.is_none() {
            self.last = Some(point);
        }
        Ok(Update::Accepted { removed: 0 })
    }
}

#[cfg(test)]
#[path = "pareto_tests.rs"]
mod tests;
