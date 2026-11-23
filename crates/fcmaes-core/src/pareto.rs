// Exact 2D Pareto archive with incremental hypervolume.
//
// This is a Rust port derived from the C++ code of bitehv2.h in Aleksey
// Vaneev's BiteOpt repository, https://github.com/avaneev/biteopt (bitehv2.h
// as of commit 5f86f50, 2026-10-13). From there come the front ordered by the
// first objective, the delta hypervolume updates, the cached rejection point
// and the append/gap fast paths. The Rust port and its adaptations
// (compensated area sums, error handling, outside points, payloads,
// preference scores, paged storage) are by Dietmar Wolz; Aleksey Vaneev's copyright covers
// the original C++ code only.
//
// Original C++ code (bitehv2.h): Copyright (c) 2026 Aleksey Vaneev.
// Rust port: Copyright (c) 2026 Dietmar Wolz.
// MIT license; keep NOTICE.pareto with copies of this file.
#![doc = include_str!("../PARETO_ARCHIVE.md")]

use std::{cmp::Ordering, fmt};
mod pages;
use pages::Pages;

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
pub(crate) fn coordinate_key(x: f64) -> u64 {
    let bits = canonical(x).to_bits();
    if bits >> 63 == 0 {
        bits ^ (1 << 63)
    } else {
        !bits
    }
}
/// Membership on a lexicographically sorted 2D stream, preserving equal
/// nondominated rows. The caller chooses whether to collapse duplicates.
pub(crate) fn sorted_skyline_member(last: &mut Option<[f64; 2]>, p: [f64; 2]) -> bool {
    if last.is_none_or(|q| p == q || p[1] < q[1]) {
        *last = Some(p);
        true
    } else {
        false
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
/// Entries live in sorted pages ordered by the first objective, so the second
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
    entries: Pages<P>,
    area: Sum,
    best_score: f64,
    best_point: [f64; 2],
    witness: [f64; 2],
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
            entries: Pages::new(),
            area: Sum::default(),
            best_score: 0.0,
            best_point: [0.0; 2],
            witness: [f64::INFINITY; 2],
        }
    }
    /// Remove all entries, keeping the reference and preference.
    pub fn clear(&mut self) {
        *self = Self::build(self.reference, self.preference);
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
        self.entries.len() == 0
    }
    /// Area dominated by the archive within the reference box.
    pub fn hypervolume(&self) -> f64 {
        self.area.value
    }
    /// Entries in increasing first-objective order.
    pub fn entries(&self) -> impl DoubleEndedIterator<Item = &Entry<P>> {
        self.entries.iter()
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
        self.entries.get(self.best_point[0])
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
        if (!self.is_empty() && weak(self.best_point, point)) || weak(self.witness, point) {
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
        self.is_empty()
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
        let [x, y] = point;
        // Extremes and improvement at the last x each add exactly one slab.
        let left = self
            .entries
            .first()
            .filter(|e| x < e.point[0] && y > e.point[1])
            .map(|e| e.point[0]);
        let last = self.entries.last().map(|e| e.point);
        if let Some(p) = last.filter(|&p| weak(p, point)) {
            self.witness = p;
            return Ok(Update::Rejected);
        }
        if left.is_some() || last.is_none_or(|p| p[0] <= x) {
            let replaces_last = last.is_some_and(|p| p[0] == x);
            let delta = if let Some(next) = left {
                rectangle(next, x, self.reference[1], y)?
            } else {
                rectangle(
                    self.reference[0],
                    x,
                    last.map_or(self.reference[1], |p| p[1]),
                    y,
                )?
            };
            let mut area = self.area;
            area.add(delta)?;
            let score = preference_score(self.preference, point);
            let best = self.replaces_best(score, point);
            let entry = Entry {
                point,
                score,
                payload: make_payload(),
            };
            if left.is_some() {
                self.entries.prepend(entry)
            } else if replaces_last {
                self.entries.replace_last(entry)
            } else {
                self.entries.append(entry)
            }
            self.commit(point, score, best, area);
            return Ok(Update::Accepted {
                removed: usize::from(replaces_last),
            });
        }
        self.entries.flush_left();
        let start = self.entries.locate(x);
        let previous = self.entries.before(start).map(|p| p.point);
        if let Some(p) = previous.filter(|p| p[1] <= y).or_else(|| {
            self.entries
                .at(start)
                .map(|e| e.point)
                .filter(|p| p[0] == x && p[1] <= y)
        }) {
            self.witness = p;
            return Ok(Update::Rejected);
        }
        let mut old_y = previous.map_or(self.reference[1], |p| p[1]);
        let mut old_x = x;
        let mut delta = Sum::default();
        let mut end = start;
        let mut removed = 0;
        while let Some(entry) = self.entries.at(end) {
            if entry.point[1] < y {
                break;
            }
            delta.add(rectangle(entry.point[0], old_x, old_y, y)?)?;
            [old_x, old_y] = entry.point;
            removed += 1;
            end = self.entries.next(end);
        }
        let next_x = self
            .entries
            .at(end)
            .map_or(self.reference[0], |e| e.point[0]);
        delta.add(rectangle(next_x, old_x, old_y, y)?)?;
        let mut area = self.area;
        area.add(delta.value)?;
        let score = preference_score(self.preference, point);
        let best = self.replaces_best(score, point);
        let entry = Entry {
            point,
            score,
            payload: make_payload(),
        };
        self.entries.replace(start, end, removed, entry);
        self.commit(point, score, best, area);
        Ok(Update::Accepted { removed })
    }
    fn commit(&mut self, point: [f64; 2], score: f64, best: bool, area: Sum) {
        if best {
            self.best_score = score;
            self.best_point = point;
        }
        self.area = area;
    }
}

#[cfg(test)]
#[path = "pareto_tests.rs"]
mod tests;
