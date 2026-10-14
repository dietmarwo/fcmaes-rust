// Copyright (c) 2026 Dietmar Wolz (fcmaes adaptation and Rust implementation).
// Copyright (c) 2026 Aleksey Vaneev (original archive, BiteOpt bitehv2.h).
// MIT license; see NOTICE.pareto.
#![doc = include_str!("../PARETO_ARCHIVE.md")]

use std::{cmp::Ordering, collections::BTreeMap, fmt};

/// Preference score used by [`ParetoArchive2D::preferred`] and
/// [`ParetoArchive2D::ranked`]. It never changes archive membership.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Score {
    /// Mean of the two normalized objective coordinates.
    Linear,
    /// Mean of the squared normalized objective coordinates.
    SquaredEuclidean,
}

/// Errors reported by the archive. A failed insertion leaves the archive unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Non-finite reference or normalization bounds, or `lower > reference`.
    InvalidBounds,
    /// A non-finite objective value.
    InvalidPoint,
    /// The new rectangle or the accumulated hypervolume is not representable.
    NonFiniteArea,
    /// Objective and payload batch lengths differ.
    PayloadLength,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidBounds => "bounds must be finite and lower <= reference",
            Self::InvalidPoint => "objectives must be finite",
            Self::NonFiniteArea => "hypervolume arithmetic exceeds the finite f64 range",
            Self::PayloadLength => "objective and payload batch lengths differ",
        })
    }
}
impl std::error::Error for Error {}

/// Batch insertion failed. A length mismatch (`row == None`) leaves the archive
/// unchanged. Otherwise the failing row is not applied, the rows before it are,
/// and `accepted` holds their acceptance flags.
#[derive(Debug, PartialEq)]
pub struct BatchUpdateError {
    /// Failing row, or `None` for a mismatched payload batch length.
    pub row: Option<usize>,
    /// Acceptance flags for the rows processed before the failure.
    pub accepted: Vec<bool>,
    /// Underlying error.
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

/// Objective range used to normalize preference scores. The upper corner is the
/// hypervolume reference. `lower` is only the normalization origin, for example
/// an estimate of the ideal point: points below it are archived normally.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    lower: [f64; 2],
    reference: [f64; 2],
    lower_half: [f64; 2],
    width_half: [f64; 2],
}
impl Bounds {
    /// Construct normalization bounds.
    ///
    /// # Errors
    /// Returns [`Error::InvalidBounds`] for non-finite values or `lower > reference`.
    pub fn new(lower: [f64; 2], reference: [f64; 2]) -> Result<Self, Error> {
        if lower.iter().chain(&reference).any(|x| !x.is_finite())
            || (0..2).any(|j| lower[j] > reference[j])
        {
            return Err(Error::InvalidBounds);
        }
        let lower = lower.map(canonical);
        let reference = reference.map(canonical);
        // Halves keep the width finite even for [-f64::MAX, f64::MAX].
        let lower_half = lower.map(|x| x * 0.5);
        let width_half = std::array::from_fn(|j| reference[j] * 0.5 - lower_half[j]);
        Ok(Self {
            lower,
            reference,
            lower_half,
            width_half,
        })
    }
    /// Construct bounds with the origin as normalization corner.
    ///
    /// # Errors
    /// Returns [`Error::InvalidBounds`] for a negative or non-finite reference.
    pub fn nonnegative(reference: [f64; 2]) -> Result<Self, Error> {
        Self::new([0.0; 2], reference)
    }
    /// Normalization origin.
    pub fn lower(&self) -> [f64; 2] {
        self.lower
    }
    /// Hypervolume reference.
    pub fn reference(&self) -> [f64; 2] {
        self.reference
    }
    /// Preference score in `[0, 1]`. Each coordinate is normalized to the bounds
    /// and clamped, so the score never decreases when an objective grows.
    ///
    /// # Errors
    /// Returns [`Error::InvalidPoint`] for non-finite objectives.
    pub fn score(&self, point: [f64; 2], score: Score) -> Result<f64, Error> {
        if !is_finite(point) {
            return Err(Error::InvalidPoint);
        }
        Ok(self.normalized(point.map(canonical), score))
    }
    fn normalized(&self, point: [f64; 2], score: Score) -> f64 {
        // Clamping keeps the map monotone (and finite) outside the bounds. Ties
        // are settled on the original coordinates, never on these values.
        let u: [f64; 2] = std::array::from_fn(|j| {
            if self.width_half[j] == 0.0 {
                0.0
            } else {
                ((point[j] * 0.5 - self.lower_half[j]) / self.width_half[j]).clamp(0.0, 1.0)
            }
        });
        canonical(match score {
            Score::Linear => u[0] * 0.5 + u[1] * 0.5,
            Score::SquaredEuclidean => u[0] * u[0] * 0.5 + u[1] * u[1] * 0.5,
        })
    }
}

/// Preference score of a finite point: normalized if bounds are given, else
/// the plain mean of the objectives. Both are monotone in each objective.
fn preference_score(preference: Option<(Bounds, Score)>, point: [f64; 2]) -> f64 {
    match preference {
        Some((bounds, score)) => bounds.normalized(point, score),
        None => canonical(point[0] * 0.5 + point[1] * 0.5),
    }
}
fn canonical(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}
fn is_finite(point: [f64; 2]) -> bool {
    point[0].is_finite() && point[1].is_finite()
}
fn outside(point: [f64; 2], reference: [f64; 2]) -> bool {
    point[0] > reference[0] || point[1] > reference[1]
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

/// One unique nondominated objective pair and its payload.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry<P> {
    point: [f64; 2],
    score: f64,
    payload: P,
}
impl<P> Entry<P> {
    /// Objective coordinates, with `-0.0` stored as `0.0`.
    pub fn point(&self) -> [f64; 2] {
        self.point
    }
    /// Preference score, see [`ParetoArchive2D::with_preference`].
    pub fn score(&self) -> f64 {
        self.score
    }
    /// Payload stored with this objective pair.
    pub fn payload(&self) -> &P {
        &self.payload
    }
}

/// Result of an insertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Update {
    /// A duplicate or weakly dominated point; the payload was not built.
    Rejected,
    /// The point lies outside the reference box and was not archived.
    Outside,
    /// The point was archived.
    Accepted {
        /// Number of archived entries it dominated and removed.
        removed: usize,
    },
}
impl Update {
    /// Whether the point was archived.
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
    /// Kahan-compensated addition of a nonnegative increment.
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
/// Positive rectangle area. Empty rectangles return before the subtraction,
/// which avoids `0 * inf` for huge signed intervals.
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

/// Exact, unbounded 2D minimization archive with incremental hypervolume.
///
/// Entries live in one B-tree ordered by the first objective, so the second
/// objective strictly decreases along the front. An insertion needs one
/// predecessor lookup to reject a dominated point and removes the contiguous
/// run of successors it dominates; appends and inserts into the gap after the
/// previous insertion skip the lookup. The entry with the lowest preference
/// score doubles as a cached dominator that rejects most points in O(1).
/// Hypervolume increments are Kahan-summed and validated before any change,
/// so a failed insertion leaves the archive untouched. Duplicate objective
/// pairs keep the first payload. There is no capacity limit: an antichain
/// stream keeps every point.
#[derive(Clone)]
pub struct ParetoArchive2D<P = ()> {
    reference: [f64; 2],
    preference: Option<(Bounds, Score)>,
    entries: BTreeMap<u64, Entry<P>>,
    // Scratch keys separate the numeric preflight from the mutation.
    victims: Vec<u64>,
    area: Sum,
    best: Option<u64>,
    best_score: f64,
    best_point: [f64; 2],
    last: Option<[f64; 2]>,
    gap: Option<([f64; 2], [f64; 2])>,
}
impl<P> ParetoArchive2D<P> {
    /// Empty archive with the given hypervolume reference. The preference
    /// score is the plain mean of both objectives.
    ///
    /// # Errors
    /// Returns [`Error::InvalidBounds`] for a non-finite reference.
    pub fn new(reference: [f64; 2]) -> Result<Self, Error> {
        if !is_finite(reference) {
            return Err(Error::InvalidBounds);
        }
        Ok(Self::build(reference.map(canonical), None))
    }
    /// Empty archive whose preference score is normalized to `bounds`. The
    /// reference is `bounds.reference()`; points below `bounds.lower()` are
    /// archived normally.
    pub fn with_preference(bounds: Bounds, score: Score) -> Self {
        Self::build(bounds.reference, Some((bounds, score)))
    }
    fn build(reference: [f64; 2], preference: Option<(Bounds, Score)>) -> Self {
        Self {
            reference,
            preference,
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
    /// Remove all entries, keeping the reference and preference.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.victims.clear();
        self.area = Sum::default();
        self.best = None;
        self.last = None;
        self.gap = None;
    }
    /// Hypervolume reference.
    pub fn reference(&self) -> [f64; 2] {
        self.reference
    }
    /// Number of archived objective pairs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether the archive is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Area dominated by the archive within the reference box.
    pub fn hypervolume(&self) -> f64 {
        self.area.value
    }
    /// Entries in increasing first-objective order.
    pub fn entries(&self) -> impl DoubleEndedIterator<Item = &Entry<P>> {
        self.entries.values()
    }
    /// Copy of all objective pairs and payloads in increasing first-objective order.
    pub fn front(&self) -> Vec<([f64; 2], P)>
    where
        P: Clone,
    {
        self.entries()
            .map(|entry| (entry.point, entry.payload.clone()))
            .collect()
    }
    /// Entry with the lowest preference score, ties broken by coordinates.
    pub fn preferred(&self) -> Option<&Entry<P>> {
        self.best.and_then(|key| self.entries.get(&key))
    }
    /// Entries sorted by preference score, ties broken by coordinates.
    pub fn ranked(&self) -> Vec<&Entry<P>> {
        let mut result: Vec<_> = self.entries().collect();
        result.sort_unstable_by(|a, b| compare(a.score, a.point, b));
        result
    }

    /// Insert a point with its payload.
    ///
    /// # Errors
    /// Non-finite objectives and non-representable areas leave the archive unchanged.
    pub fn insert(&mut self, point: [f64; 2], payload: P) -> Result<Update, Error> {
        self.insert_with(point, || payload)
    }
    /// Insert a point; `make_payload` runs only if the point is archived.
    ///
    /// # Errors
    /// Non-finite objectives and non-representable areas leave the archive unchanged.
    ///
    /// # Panics
    /// Propagates a panic from `make_payload`; the archive stays unchanged.
    pub fn insert_with(
        &mut self,
        point: [f64; 2],
        make_payload: impl FnOnce() -> P,
    ) -> Result<Update, Error> {
        if !is_finite(point) {
            return Err(Error::InvalidPoint);
        }
        if outside(point, self.reference) {
            return Ok(Update::Outside);
        }
        let point = point.map(canonical);
        if self.best.is_some() && weak(self.best_point, point) {
            return Ok(Update::Rejected);
        }
        self.insert_validated(point, make_payload)
    }
    /// Insert rows in order, cloning payloads only for archived rows. Returns
    /// one flag per row; outside and dominated rows are `false`.
    ///
    /// # Errors
    /// A length mismatch leaves the archive unchanged. Otherwise the failing
    /// row and the flags of the rows before it are returned; those rows stay applied.
    ///
    /// # Panics
    /// Propagates a panic from a payload clone before that row changes the archive.
    pub fn insert_batch(
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

    fn replaces_best(&self, score: f64, point: [f64; 2]) -> bool {
        self.best.is_none()
            || score
                .total_cmp(&self.best_score)
                .then(point[0].total_cmp(&self.best_point[0]))
                .then(point[1].total_cmp(&self.best_point[1]))
                == Ordering::Less
    }

    fn insert_validated(
        &mut self,
        point: [f64; 2],
        make_payload: impl FnOnce() -> P,
    ) -> Result<Update, Error> {
        // Append after the last entry, or fill the gap after the previous
        // insertion: geometry alone proves that nothing is dominated.
        if let Some(last) = self.last
            && last[0] < point[0]
            && point[1] < last[1]
        {
            return self.insert_gap(point, last[1], None, make_payload);
        }
        if let Some((previous, successor)) = self.gap
            && previous[0] < point[0]
            && point[0] < successor[0]
            && successor[1] < point[1]
            && point[1] < previous[1]
        {
            return self.insert_gap(point, previous[1], Some(successor), make_payload);
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
                .map_or(self.reference[1], |(_, q)| q.point[1]),
            None => self.reference[1],
        };
        let score = preference_score(self.preference, point);
        let replace_best = self.replaces_best(score, point);
        let mut old_x = point[0];
        let mut next_x = self.reference[0];
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
        // All numeric checks and the payload factory ran before this point.
        // The score is monotone, so a removed best entry is always replaced.
        debug_assert!(replace_best || !self.victims.iter().any(|&k| Some(k) == self.best));
        if removed == self.entries.len() {
            self.entries.clear();
        } else {
            for k in &self.victims {
                self.entries.remove(k);
            }
        }
        self.entries.insert(key, entry);
        self.commit(key, point, score, replace_best, next_area, successor);
        Ok(Update::Accepted { removed })
    }

    fn insert_gap(
        &mut self,
        point: [f64; 2],
        previous_y: f64,
        successor: Option<[f64; 2]>,
        make_payload: impl FnOnce() -> P,
    ) -> Result<Update, Error> {
        let score = preference_score(self.preference, point);
        let key = coordinate_key(point[0]);
        let next_x = successor.map_or(self.reference[0], |p| p[0]);
        let mut next_area = self.area;
        next_area.add(rectangle(next_x, point[0], previous_y, point[1])?)?;
        let entry = Entry {
            point,
            score,
            payload: make_payload(),
        };
        self.entries.insert(key, entry);
        let replace_best = self.replaces_best(score, point);
        self.commit(key, point, score, replace_best, next_area, successor);
        Ok(Update::Accepted { removed: 0 })
    }

    fn commit(
        &mut self,
        key: u64,
        point: [f64; 2],
        score: f64,
        replace_best: bool,
        area: Sum,
        successor: Option<[f64; 2]>,
    ) {
        if replace_best {
            self.best = Some(key);
            self.best_score = score;
            self.best_point = point;
        }
        self.area = area;
        self.gap = successor.map(|next| (point, next));
        if successor.is_none() {
            self.last = Some(point);
        }
    }
}

#[cfg(test)]
#[path = "pareto_tests.rs"]
mod tests;
