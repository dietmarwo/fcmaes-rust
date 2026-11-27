// Numeric kernels index parallel arrays by a shared counter.
#![allow(clippy::needless_range_loop)]

//! Quality-Diversity search with CVT-MAP-Elites and the Diversifier.
//!
//! A CVT (centroidal Voronoi tessellation) archive partitions the behavior
//! space into `capacity` niches (via k-means over uniform samples); each niche
//! keeps the best solution found for it. MAP-Elites fills the archive with an
//! SBX / Iso+LineDD emitter driven by a quality-diversity fitness
//! `x -> (fitness, behavior descriptor)`. The Diversifier generalizes CMA-ME:
//! it drives an ask/tell optimizer (here Rust CMA-ES) whose objective is the
//! per-niche *improvement*, so the search fills and improves niches.
//!
//! Archive mutation remains serial and deterministic. Objective evaluation can
//! be supplied either point-by-point through [`QdFitness`] or in parallel
//! batches through [`QdBatchFitness`].
//!
//! # References
//!
//! - J.-B. Mouret and J. Clune, [“Illuminating Search Spaces by Mapping
//!   Elites”](https://arxiv.org/abs/1504.04909) (2015).
//! - V. Vassiliades, K. Chatzilygeroudis, and J.-B. Mouret, [“Using
//!   Centroidal Voronoi Tessellations to Scale Up the Multi-dimensional
//!   Archive of Phenotypic Elites
//!   Algorithm”](https://arxiv.org/abs/1610.05729) (2016).
//!
//! # Example
//!
//! ```
//! use fcmaes_core::{map_elites, Archive, MapElitesParams, Rng};
//!
//! let mut rng = Rng::new(42);
//! let mut archive = Archive::new(2, &[0.0, 0.0], &[1.0, 1.0], 16, 0, &mut rng);
//! archive.seed_uniform(&[0.0; 2], &[1.0; 2], &mut rng);
//! let mut fitness = |x: &[f64]| {
//!     let quality = x.iter().map(|v| (v - 0.5).powi(2)).sum();
//!     (quality, x.to_vec())
//! };
//! let params = MapElitesParams {
//!     generations: 4,
//!     chunk_size: 8,
//!     ..Default::default()
//! };
//! map_elites(
//!     &mut archive,
//!     &mut fitness,
//!     &[0.0; 2],
//!     &[1.0; 2],
//!     &params,
//!     &mut rng,
//! );
//! assert!(archive.occupied() > 0);
//! ```

use crate::cmaes::{Cmaes, CmaesParams};
use crate::fitness::Fitness;
use crate::rng::Rng;
use rayon::prelude::*;

/// Quality-diversity fitness: maps a solution to `(fitness, behavior)`.
pub trait QdFitness {
    /// Evaluate one decoded solution, returning minimized quality and behavior descriptors.
    fn eval(&mut self, x: &[f64]) -> (f64, Vec<f64>);
}

impl<F> QdFitness for F
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    fn eval(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        self(x)
    }
}

/// Batch quality-diversity fitness. Implementations may evaluate `xs` in
/// parallel, but must return one result per input in the same order.
pub trait QdBatchFitness {
    /// Evaluate decoded solutions in order.
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)>;
}

impl<F> QdBatchFitness for F
where
    F: FnMut(&[Vec<f64>]) -> Vec<(f64, Vec<f64>)>,
{
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        self(xs)
    }
}

struct SerialQdBatchFitness<'a> {
    fitness: &'a mut dyn QdFitness,
}

impl QdBatchFitness for SerialQdBatchFitness<'_> {
    fn eval_batch(&mut self, xs: &[Vec<f64>]) -> Vec<(f64, Vec<f64>)> {
        xs.iter().map(|x| self.fitness.eval(x)).collect()
    }
}

// ---------------------------------------------------------------------------
// k-means++ (CVT niche centers)
// ---------------------------------------------------------------------------

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[derive(Clone, Debug)]
struct CenterTree {
    order: Vec<usize>,
}
impl CenterTree {
    fn new(rows: &[f64], dim: usize) -> Self {
        let mut tree = Self {
            order: (0..rows.len() / dim).collect(),
        };
        Self::split(&mut tree.order, rows, dim, 0);
        tree
    }

    fn split(indices: &mut [usize], rows: &[f64], dim: usize, axis: usize) {
        if indices.len() < 2 {
            return;
        }
        let middle = indices.len() / 2;
        let (left, _, right) = indices.select_nth_unstable_by(middle, |&a, &b| {
            rows[a * dim + axis]
                .total_cmp(&rows[b * dim + axis])
                .then(a.cmp(&b))
        });
        let next_axis = (axis + 1) % dim;
        Self::split(left, rows, dim, next_axis);
        Self::split(right, rows, dim, next_axis);
    }

    fn nearest(&self, rows: &[f64], point: &[f64]) -> usize {
        let mut best = (f64::MAX, 0);
        Self::visit(&self.order, rows, point, 0, &mut best);
        best.1
    }

    fn visit(indices: &[usize], rows: &[f64], point: &[f64], axis: usize, best: &mut (f64, usize)) {
        if indices.is_empty() {
            return;
        }
        let dim = point.len();
        let middle = indices.len() / 2;
        let i = indices[middle];
        let center = &rows[i * dim..(i + 1) * dim];
        let d = dist2(point, center);
        if d < best.0 || (d == best.0 && i < best.1) {
            *best = (d, i);
        }
        let delta = point[axis] - center[axis];
        let left = &indices[..middle];
        let right = &indices[middle + 1..];
        let (near, far) = if delta < 0.0 {
            (left, right)
        } else {
            (right, left)
        };
        let next_axis = if axis + 1 == dim { 0 } else { axis + 1 };
        Self::visit(near, rows, point, next_axis, best);
        if delta * delta <= best.0 {
            Self::visit(far, rows, point, next_axis, best);
        }
    }
}

fn nearest_center(point: &[f64], centers: &[Vec<f64>]) -> usize {
    let mut best = 0;
    let mut best_distance = f64::MAX;
    for (index, center) in centers.iter().enumerate() {
        let distance = dist2(point, center);
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    best
}

fn grid_index(layout: GridLayout, x: f64, y: f64) -> usize {
    let GridLayout {
        rows,
        base_columns,
        extra_columns,
    } = layout;
    let y = y.clamp(0.0, 1.0 - f64::EPSILON);
    let row = (y * rows as f64) as usize;
    let columns = base_columns + usize::from(row < extra_columns);
    let x = x.clamp(0.0, 1.0 - f64::EPSILON);
    let column = (x * columns as f64) as usize;
    row * base_columns + row.min(extra_columns) + column
}

/// Fewest samples per partial sum of a Lloyd step.
const LLOYD_CHUNK: usize = 1024;
/// Most partial sums of a Lloyd step; bounds their memory to this many
/// copies of the centers.
const LLOYD_PARTIALS: usize = 64;

/// Compute `k` niche centers in `[0,1]^dim` via k-means++ over
/// `k * samples_per_niche` uniform samples (Lloyd iterations). The centers
/// depend only on the arguments and the random stream, not on the Rayon
/// pool the function runs in.
fn cvt_centers(k: usize, dim: usize, samples_per_niche: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let n = (k * samples_per_niche).max(k);
    let samples: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.uniform01()).collect())
        .collect();

    // k-means++ init.
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    centers.push(samples[rng.int_below(n as i64) as usize].clone());
    let mut d2: Vec<f64> = samples.iter().map(|s| dist2(s, &centers[0])).collect();
    while centers.len() < k {
        let sum: f64 = d2.iter().sum();
        let mut target = rng.uniform01() * sum;
        let mut idx = 0;
        for (i, &d) in d2.iter().enumerate() {
            target -= d;
            if target <= 0.0 {
                idx = i;
                break;
            }
        }
        centers.push(samples[idx].clone());
        let c = centers.last().unwrap();
        let update = |(distance, s): (&mut f64, &Vec<f64>)| {
            let candidate = dist2(s, c);
            if candidate < *distance {
                *distance = candidate;
            }
        };
        if samples.len() < 4096 || rayon::current_num_threads() == 1 {
            d2.iter_mut().zip(&samples).for_each(update);
        } else {
            d2.par_iter_mut().zip(&samples).for_each(update);
        }
    }

    // A few Lloyd iterations. Assignment is the dominant O(samples * k)
    // kernel. Fixed-size chunks are summed in parallel and their partial
    // sums are added in chunk order, so the centers do not depend on the
    // number of threads or on work stealing.
    let chunk_size = LLOYD_CHUNK.max(n.div_ceil(LLOYD_PARTIALS));
    for _ in 0..10 {
        // Rebuild for the old centers. Assignment is independent of the
        // fixed-order reduction of the new centers.
        let index = (dim == 2 && k >= 256).then(|| {
            let rows = centers.iter().flatten().copied().collect::<Vec<_>>();
            let tree = CenterTree::new(&rows, dim);
            (rows, tree)
        });
        let partials: Vec<(Vec<f64>, Vec<usize>)> = samples
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut sums = vec![0.0; k * dim];
                let mut counts = vec![0usize; k];
                for s in chunk {
                    let best = match &index {
                        Some((rows, tree)) => tree.nearest(rows, s),
                        None => nearest_center(s, &centers),
                    };
                    counts[best] += 1;
                    for j in 0..dim {
                        sums[best * dim + j] += s[j];
                    }
                }
                (sums, counts)
            })
            .collect();
        let mut sums = vec![0.0; k * dim];
        let mut counts = vec![0usize; k];
        for (partial_sums, partial_counts) in &partials {
            for index in 0..k {
                counts[index] += partial_counts[index];
            }
            for index in 0..k * dim {
                sums[index] += partial_sums[index];
            }
        }
        let mut max_shift = 0.0_f64;
        for ci in 0..k {
            if counts[ci] > 0 {
                let old = centers[ci].clone();
                for j in 0..dim {
                    centers[ci][j] = sums[ci * dim + j] / counts[ci] as f64;
                }
                max_shift = max_shift.max(dist2(&old, &centers[ci]));
            }
        }
        if max_shift <= 1e-12 {
            break;
        }
    }
    centers
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

/// Exact row layout of a regular two-dimensional MAP-Elites archive.
///
/// The first `extra_columns` rows contain `base_columns + 1` cells; remaining
/// rows contain `base_columns`. This represents every capacity exactly,
/// including ragged grids such as 60 cells arranged over seven rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GridLayout {
    /// Number of descriptor rows.
    pub rows: usize,
    /// Columns present in every row.
    pub base_columns: usize,
    /// Number of leading rows containing one additional column.
    pub extra_columns: usize,
}

impl GridLayout {
    /// Exact number of cells described by this layout.
    pub fn cells(&self) -> usize {
        self.rows * self.base_columns + self.extra_columns
    }

    /// Number of columns in `row`, or `None` when the row is out of range.
    pub fn columns_in_row(&self, row: usize) -> Option<usize> {
        (row < self.rows).then(|| self.base_columns + usize::from(row < self.extra_columns))
    }

    /// Maximum number of columns in any row.
    pub fn max_columns(&self) -> usize {
        self.base_columns + usize::from(self.extra_columns > 0)
    }
}

/// CVT quality-diversity archive: `capacity` niches, each holding the best
/// solution found for it. Cloning copies the niches with their elites, so an
/// expensive CVT tessellation can be computed once and reused by several runs.
#[derive(Clone, Debug)]
pub struct Archive {
    dim: usize,
    qd_dim: usize,
    capacity: usize,
    desc_lb: Vec<f64>,
    desc_scale: Vec<f64>,
    geometry: Geometry,
    xs: Vec<Vec<f64>>,
    ds: Vec<Vec<f64>>,
    ys: Vec<f64>,
    counts: Vec<u64>,
    occupied: usize,
    si: Vec<usize>, // niche indices sorted ascending by fitness
    rank_changes: Option<RankChanges>,
}

#[derive(Clone, Debug)]
enum Geometry {
    Grid(GridLayout),
    Cvt(std::sync::Arc<CvtGeometry>),
}

#[derive(Debug)]
struct CvtGeometry {
    centers: Vec<f64>,
    tree: Option<CenterTree>,
}

#[derive(Clone, Debug)]
struct RankChanges {
    indices: Vec<usize>,
    marked: Vec<bool>,
    merged: Vec<usize>,
}

impl RankChanges {
    fn new(capacity: usize) -> Self {
        Self {
            indices: Vec::new(),
            marked: vec![false; capacity],
            merged: Vec::new(),
        }
    }

    fn record(&mut self, index: usize) {
        if !self.marked[index] {
            self.marked[index] = true;
            self.indices.push(index);
        }
    }
}

impl Archive {
    /// Construct a validated CVT archive.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` or `capacity` is zero, if the descriptor
    /// bound slices are empty or of unequal length, or if any bound pair is
    /// non-finite or does not satisfy `lower < upper`.
    pub fn try_new(
        dim: usize,
        qd_lb: &[f64],
        qd_ub: &[f64],
        capacity: usize,
        samples_per_niche: usize,
        rng: &mut Rng,
    ) -> Result<Self, &'static str> {
        if dim == 0 {
            return Err("archive decision dimension must be positive");
        }
        if qd_lb.is_empty() || qd_lb.len() != qd_ub.len() {
            return Err("descriptor bounds must be non-empty and have equal lengths");
        }
        if qd_lb
            .iter()
            .zip(qd_ub)
            .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo >= hi)
        {
            return Err("descriptor bounds must be finite and satisfy lower < upper");
        }
        if capacity == 0 {
            return Err("archive capacity must be positive");
        }
        Ok(Self::new_unchecked(
            dim,
            qd_lb,
            qd_ub,
            capacity,
            samples_per_niche,
            rng,
        ))
    }

    /// Construct a CVT archive, panicking on invalid configuration. Prefer
    /// [`Archive::try_new`] for user-supplied inputs.
    ///
    /// # Panics
    ///
    /// Panics on any configuration [`Archive::try_new`] rejects.
    pub fn new(
        dim: usize,
        qd_lb: &[f64],
        qd_ub: &[f64],
        capacity: usize,
        samples_per_niche: usize,
        rng: &mut Rng,
    ) -> Self {
        Self::try_new(dim, qd_lb, qd_ub, capacity, samples_per_niche, rng)
            .expect("invalid archive configuration")
    }

    fn new_unchecked(
        dim: usize,
        qd_lb: &[f64],
        qd_ub: &[f64],
        capacity: usize,
        samples_per_niche: usize,
        rng: &mut Rng,
    ) -> Self {
        let qd_dim = qd_lb.len();
        let desc_lb = qd_lb.to_vec();
        let desc_scale: Vec<f64> = qd_ub.iter().zip(qd_lb).map(|(u, l)| u - l).collect();
        let geometry = if samples_per_niche == 0 && qd_dim == 2 {
            let rows = (capacity as f64).sqrt().floor().max(1.0) as usize;
            Geometry::Grid(GridLayout {
                rows,
                base_columns: capacity / rows,
                extra_columns: capacity % rows,
            })
        } else {
            let centers = cvt_centers(capacity, qd_dim, samples_per_niche.max(1), rng)
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let tree = (qd_dim == 2 && capacity >= 256).then(|| CenterTree::new(&centers, qd_dim));
            Geometry::Cvt(std::sync::Arc::new(CvtGeometry { centers, tree }))
        };
        Archive {
            dim,
            qd_dim,
            capacity,
            desc_lb,
            desc_scale,
            geometry,
            xs: vec![vec![0.0; dim]; capacity],
            ds: vec![vec![0.0; qd_dim]; capacity],
            ys: vec![f64::INFINITY; capacity],
            counts: vec![0; capacity],
            occupied: 0,
            si: (0..capacity).collect(),
            rank_changes: (capacity >= 512).then(|| RankChanges::new(capacity)),
        }
    }

    /// Number of decision variables stored for each elite.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Number of behavior-descriptor dimensions.
    pub fn qd_dim(&self) -> usize {
        self.qd_dim
    }
    /// Total number of niches.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Number of niches containing an evaluated elite.
    pub fn occupied(&self) -> usize {
        self.occupied
    }

    /// Exact layout of a regular two-dimensional archive.
    ///
    /// Returns `None` for CVT archives and descriptor dimensions other than
    /// two. Use [`capacity`](Self::capacity) as the coverage denominator and
    /// this layout when mapping a niche index to a rendered row.
    pub fn grid_layout(&self) -> Option<GridLayout> {
        match self.geometry {
            Geometry::Grid(layout) => Some(layout),
            Geometry::Cvt(_) => None,
        }
    }

    /// Shape of a regular two-dimensional archive.
    ///
    /// The first component is the number of columns and the second the number
    /// of rows. Returns `None` for CVT archives and for descriptor dimensions
    /// other than two. Non-rectangular regular grids report the maximum column
    /// count; early rows may contain one additional cell as documented by the
    /// archive's exact-capacity construction.
    pub fn grid_shape(&self) -> Option<(usize, usize)> {
        self.grid_layout()
            .map(|layout| (layout.max_columns(), layout.rows))
    }

    /// Seed all niche solutions with uniform random samples in `[lower, upper]`
    /// (never evaluated — they serve as the initial SBX/Iso parent pool, as the
    /// Python original documents).
    ///
    /// # Panics
    ///
    /// Panics if `lower.len()` or `upper.len()` differs from the archive's
    /// decision dimension.
    pub fn seed_uniform(&mut self, lower: &[f64], upper: &[f64], rng: &mut Rng) {
        assert_eq!(lower.len(), self.dim, "lower bounds length must equal dim");
        assert_eq!(upper.len(), self.dim, "upper bounds length must equal dim");
        assert!(
            lower
                .iter()
                .zip(upper)
                .all(|(&lo, &hi)| lo.is_finite() && hi.is_finite() && lo < hi),
            "decision bounds must be finite and satisfy lower < upper"
        );
        for x in self.xs.iter_mut() {
            for i in 0..self.dim {
                x[i] = lower[i] + (upper[i] - lower[i]) * rng.uniform01();
            }
        }
    }

    fn encode_d(&self, d: &[f64]) -> Vec<f64> {
        (0..self.qd_dim)
            .map(|i| (d[i] - self.desc_lb[i]) / self.desc_scale[i])
            .collect()
    }

    /// Index of the niche whose center is nearest the (encoded) descriptor.
    ///
    /// # Panics
    ///
    /// Panics if `d.len()` differs from the archive's descriptor dimension.
    pub fn index_of_niche(&self, d: &[f64]) -> usize {
        assert_eq!(d.len(), self.qd_dim, "descriptor length must equal qd_dim");
        let cvt = match &self.geometry {
            Geometry::Grid(layout) => {
                let x = (d[0] - self.desc_lb[0]) / self.desc_scale[0];
                let y = (d[1] - self.desc_lb[1]) / self.desc_scale[1];
                return grid_index(*layout, x, y);
            }
            Geometry::Cvt(geometry) => geometry,
        };
        let encoded = self.encode_d(d);
        if let Some(tree) = &cvt.tree
            && encoded.iter().all(|value| value.is_finite())
        {
            return tree.nearest(&cvt.centers, &encoded);
        }
        let mut best = 0;
        let mut bd = f64::MAX;
        for (i, c) in cvt.centers.chunks_exact(self.qd_dim).enumerate() {
            let dist = match *encoded.as_slice() {
                [a, b] => (a - c[0]) * (a - c[0]) + (b - c[1]) * (b - c[1]),
                [a, b, d, f] => {
                    let s0 = a - c[0];
                    let s1 = b - c[1];
                    let s2 = d - c[2];
                    let s3 = f - c[3];
                    ((s0 * s0 + s1 * s1) + s2 * s2) + s3 * s3
                }
                _ => dist2(&encoded, c),
            };
            if dist < bd {
                bd = dist;
                best = i;
            }
        }
        best
    }

    /// Add a solution to niche `i` if it improves it.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not below the archive capacity, or if `d` or `x` do
    /// not match the descriptor and decision dimensions.
    pub fn set(&mut self, i: usize, y: f64, d: &[f64], x: &[f64]) {
        assert!(i < self.capacity, "niche index out of bounds");
        assert_eq!(d.len(), self.qd_dim, "descriptor length mismatch");
        assert_eq!(x.len(), self.dim, "solution length mismatch");
        self.counts[i] += 1;
        if y.is_finite() && d.iter().all(|v| v.is_finite()) && y < self.ys[i] {
            if self.ys[i].is_infinite() {
                self.occupied += 1;
            }
            if let Some(changes) = &mut self.rank_changes {
                changes.record(i);
            }
            self.ys[i] = y;
            self.xs[i].copy_from_slice(x);
            self.ds[i].copy_from_slice(d);
        }
    }

    /// Evaluate `xs`, add to the archive, and return `(improvements, real_ys)`
    /// where `improvement = fitness - niche's previous fitness` (negative is an
    /// improvement — the objective the Diversifier's optimizer minimizes).
    ///
    /// # Panics
    ///
    /// Panics if `fitness` returns a descriptor whose length differs from the
    /// archive's descriptor dimension.
    pub fn update(&mut self, xs: &[Vec<f64>], fitness: &mut dyn QdFitness) -> (Vec<f64>, Vec<f64>) {
        let evaluations: Vec<(f64, Vec<f64>)> = xs.iter().map(|x| fitness.eval(x)).collect();
        self.update_evaluated(xs, &evaluations)
            .expect("serial QD evaluation preserves batch length")
    }

    /// Apply already evaluated `(fitness, descriptor)` values in input order.
    /// Keeping this step separate lets callers parallelize expensive objective
    /// functions without concurrently mutating the archive.
    ///
    /// # Errors
    ///
    /// Returns an error if `evaluated` does not have the same length as `xs`.
    pub fn update_evaluated(
        &mut self,
        xs: &[Vec<f64>],
        evaluations: &[(f64, Vec<f64>)],
    ) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
        if xs.len() != evaluations.len() {
            return Err("QD evaluation batch length must match candidate batch length");
        }
        let mut improvements = Vec::with_capacity(xs.len());
        let mut real_ys = Vec::with_capacity(xs.len());
        for (x, (y, desc)) in xs.iter().zip(evaluations) {
            if x.len() != self.dim
                || desc.len() != self.qd_dim
                || !y.is_finite()
                || desc.iter().any(|value| !value.is_finite())
            {
                improvements.push(f64::INFINITY);
                real_ys.push(f64::INFINITY);
                continue;
            }
            let niche = self.index_of_niche(desc);
            let oldy = self.ys[niche];
            let improvement = if oldy.is_infinite() { *y } else { *y - oldy };
            self.set(niche, *y, desc, x);
            improvements.push(improvement);
            real_ys.push(*y);
        }
        Ok((improvements, real_ys))
    }

    /// Evaluate and apply a complete batch. Evaluation may be parallel inside
    /// `fitness`; archive updates are deterministic and retain input order.
    ///
    /// # Errors
    ///
    /// Returns an error if `fitness` does not return exactly one
    /// `(fitness, descriptor)` pair per candidate.
    pub fn update_batch(
        &mut self,
        xs: &[Vec<f64>],
        fitness: &mut dyn QdBatchFitness,
    ) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
        let evaluations = fitness.eval_batch(xs);
        self.update_evaluated(xs, &evaluations)
    }

    /// Re-sort niche indices ascending by fitness.
    pub fn argsort(&mut self) {
        let Some(changes) = &mut self.rank_changes else {
            self.si.clear();
            self.si.extend(0..self.capacity);
            self.si.sort_by(|&a, &b| self.ys[a].total_cmp(&self.ys[b]));
            return;
        };
        if changes.indices.is_empty() {
            return;
        }
        let compare = |&a: &usize, &b: &usize| self.ys[a].total_cmp(&self.ys[b]).then(a.cmp(&b));
        if changes.indices.len() <= 64 {
            // The relative order of niches whose qualities did not change
            // is still valid. Merge those niches with the changed subset.
            changes.indices.sort_unstable_by(compare);
            changes.merged.clear();
            let mut changed = changes.indices.iter().peekable();
            for &index in &self.si {
                if changes.marked[index] {
                    continue;
                }
                while let Some(&&next) = changed.peek() {
                    if !compare(&next, &index).is_lt() {
                        break;
                    }
                    changes.merged.push(next);
                    changed.next();
                }
                changes.merged.push(index);
            }
            changes.merged.extend(changed.copied());
            std::mem::swap(&mut self.si, &mut changes.merged);
        } else {
            self.si.clear();
            if self.occupied < self.capacity - self.capacity / 4 {
                self.si
                    .extend((0..self.capacity).filter(|&i| self.ys[i].is_finite()));
                self.si.sort_unstable_by(compare);
                self.si
                    .extend((0..self.capacity).filter(|&i| !self.ys[i].is_finite()));
            } else {
                self.si.extend(0..self.capacity);
                self.si.sort_unstable_by(compare);
            }
        }
        for index in changes.indices.drain(..) {
            changes.marked[index] = false;
        }
    }

    /// Sample `chunk` solutions from the best `best_n` niches (by fitness).
    pub fn random_xs(&self, best_n: usize, chunk: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        let bn = best_n.max(1).min(self.capacity);
        (0..chunk)
            .map(|_| {
                let sel = rng.int_below(bn as i64) as usize;
                let niche = if bn < self.capacity {
                    self.si[sel]
                } else {
                    sel
                };
                self.xs[niche].clone()
            })
            .collect()
    }

    /// A random solution from the best `best_n` niches (with fitness).
    pub fn random_x_one(&self, best_n: usize, rng: &mut Rng) -> (Vec<f64>, f64) {
        let bn = best_n.max(1).min(self.capacity);
        let selected = rng.int_below(bn as i64) as usize;
        let niche = if bn < self.capacity {
            self.si[selected]
        } else {
            selected
        };
        (self.xs[niche].clone(), self.ys[niche])
    }

    /// Lowest finite quality currently stored, or infinity for an empty archive.
    pub fn best_y(&self) -> f64 {
        self.ys.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// QD-score matching Python `Archive.get_qd_score`: for an all-positive
    /// archive, sum reciprocal fitness; otherwise sum the negated negative
    /// fitness values. Higher is better in both cases.
    pub fn qd_score(&self) -> f64 {
        let score: f64 = if self.best_y() > 0.0 {
            self.ys
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| value.recip())
                .sum()
        } else {
            self.ys
                .iter()
                .filter(|&&value| value < 0.0 && value.is_finite())
                .map(|&value| -value)
                .sum()
        };
        // A sum without terms is -0.0; report 0.0 for an archive without
        // elites, as before, and for one whose rule has no term.
        score + 0.0
    }

    /// Per-niche quality values; empty niches contain infinity.
    pub fn ys(&self) -> &[f64] {
        &self.ys
    }
    /// Per-niche decision vectors.
    pub fn xs(&self) -> &[Vec<f64>] {
        &self.xs
    }
    /// Per-niche behavior descriptors.
    pub fn descriptors(&self) -> &[Vec<f64>] {
        &self.ds
    }
    /// Number of evaluated candidates mapped to each niche.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Occupied `(x, y, descriptor)` triples.
    pub fn occupied_data(&self) -> Vec<(Vec<f64>, f64, Vec<f64>)> {
        (0..self.capacity)
            .filter(|&i| self.ys[i].is_finite())
            .map(|i| (self.xs[i].clone(), self.ys[i], self.ds[i].clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Emitters
// ---------------------------------------------------------------------------

/// SBX (simulated binary crossover) + polynomial mutation (the Python
/// `variation_`), clamped to `[lower, upper]`.
pub fn variation(
    pop: &[Vec<f64>],
    lower: &[f64],
    upper: &[f64],
    rng: &mut Rng,
    dis_c: f64,
    dis_m: f64,
) -> Vec<Vec<f64>> {
    let dis_c = dis_c * (0.5 + 0.5 * rng.uniform01());
    let dis_m = dis_m * (0.5 + 0.5 * rng.uniform01());
    let n = (pop.len() / 2) * 2;
    let d = lower.len();
    let half = n / 2;
    let mut offspring = vec![vec![0.0; d]; n];
    for p in 0..half {
        let p1 = &pop[p];
        let p2 = &pop[half + p];
        for i in 0..d {
            let mu = rng.uniform01();
            let negative = rng.int_below(2) == 1;
            let override_beta = rng.uniform01() < 0.5;
            let beta = if override_beta {
                1.0
            } else {
                let spread = if mu <= 0.5 {
                    (2.0 * mu).powf(1.0 / (dis_c + 1.0))
                } else {
                    (2.0 * mu).powf(-1.0 / (dis_c + 1.0))
                };
                if negative { -spread } else { spread }
            };
            let mean = (p1[i] + p2[i]) * 0.5;
            let diff = (p1[i] - p2[i]) * 0.5;
            offspring[p][i] = mean + beta * diff;
            offspring[half + p][i] = mean - beta * diff;
        }
    }
    // polynomial mutation
    let site_p = 1.0 / d as f64;
    for op in offspring.iter_mut() {
        for i in 0..d {
            if rng.uniform01() < site_p {
                let mu = rng.uniform01();
                let span = upper[i] - lower[i];
                if mu <= 0.5 {
                    let norm = (op[i] - lower[i]) / span;
                    op[i] += span
                        * ((2.0 * mu + (1.0 - 2.0 * mu) * (1.0 - norm).abs().powf(dis_m + 1.0))
                            .powf(1.0 / (dis_m + 1.0))
                            - 1.0);
                } else {
                    let norm = (upper[i] - op[i]) / span;
                    op[i] += span
                        * (1.0
                            - (2.0 * (1.0 - mu)
                                + 2.0 * (mu - 0.5) * (1.0 - norm).abs().powf(dis_m + 1.0))
                            .powf(1.0 / (dis_m + 1.0)));
                }
            }
            op[i] = op[i].clamp(lower[i], upper[i]);
        }
    }
    offspring
}

/// Iso+LineDD emitter (the Python `iso_dd_`): `x1 + N(0,iso) + N(0,line)*(x1-x2)`.
pub fn iso_dd(
    x1: &[Vec<f64>],
    x2: &[Vec<f64>],
    lower: &[f64],
    upper: &[f64],
    rng: &mut Rng,
    iso_sigma: f64,
    line_sigma: f64,
) -> Vec<Vec<f64>> {
    let d = lower.len();
    x1.iter()
        .zip(x2)
        .map(|(a, b)| {
            (0..d)
                .map(|i| {
                    let z = a[i]
                        + rng.normreal(0.0, iso_sigma)
                        + rng.normreal(0.0, line_sigma) * (a[i] - b[i]);
                    z.clamp(lower[i], upper[i])
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// MAP-Elites parameters.
#[derive(Clone, Debug)]
pub struct MapElitesParams {
    /// Number of ordinary emitter generations.
    pub generations: usize,
    /// Candidates requested per ordinary generation.
    pub chunk_size: usize,
    /// Use simulated binary crossover; false selects Iso+LineDD.
    pub use_sbx: bool,
    /// Simulated-binary-crossover distribution index.
    pub dis_c: f64,
    /// Polynomial-mutation distribution index.
    pub dis_m: f64,
    /// Isotropic noise standard deviation for Iso+LineDD.
    pub iso_sigma: f64,
    /// Directional noise standard deviation for Iso+LineDD.
    pub line_sigma: f64,
    /// Additional CMA-ES emitter generations after ordinary generations.
    pub cma_generations: usize,
}

impl Default for MapElitesParams {
    fn default() -> Self {
        Self {
            generations: 100,
            chunk_size: 20,
            use_sbx: true,
            dis_c: 20.0,
            dis_m: 20.0,
            iso_sigma: 0.02,
            line_sigma: 0.2,
            cma_generations: 0,
        }
    }
}

/// Run CVT-MAP-Elites into `archive` using the SBX / Iso+LineDD emitter, with
/// optional CMA-ES emitter generations.
///
/// # Panics
///
/// Panics if the serial adapter around `fitness` returns a batch of the wrong
/// length, which indicates a bug in this crate rather than in caller code.
/// Use [`map_elites_batch`] to handle batch-length mismatches as an error.
pub fn map_elites(
    archive: &mut Archive,
    fitness: &mut dyn QdFitness,
    lower: &[f64],
    upper: &[f64],
    p: &MapElitesParams,
    rng: &mut Rng,
) {
    let mut batch_fitness = SerialQdBatchFitness { fitness };
    map_elites_batch(archive, &mut batch_fitness, lower, upper, p, rng)
        .expect("serial QD evaluation preserves batch length");
}

/// Batch-evaluation variant of [`map_elites`]. Candidate generation and
/// archive updates remain deterministic; `fitness` controls evaluation
/// parallelism.
///
/// # Errors
///
/// Returns an error if `fitness` does not return one result per requested
/// candidate.
pub fn map_elites_batch(
    archive: &mut Archive,
    fitness: &mut dyn QdBatchFitness,
    lower: &[f64],
    upper: &[f64],
    p: &MapElitesParams,
    rng: &mut Rng,
) -> Result<(), &'static str> {
    map_elites_batch_with_progress(archive, fitness, lower, upper, p, rng, &mut |_, _| {})
}

/// Batch MAP-Elites with an ordered callback after every archive update.
///
/// The callback runs after the evaluated generation has been committed and
/// sorted. It is intended for convergence logging and must not mutate the
/// archive. The generation index is one-based and continues through optional
/// CMA-emitter generations.
///
/// # Errors
///
/// Returns an error if `fitness` does not return one result per requested
/// candidate. The archive keeps every generation committed before the
/// failure.
pub fn map_elites_batch_with_progress(
    archive: &mut Archive,
    fitness: &mut dyn QdBatchFitness,
    lower: &[f64],
    upper: &[f64],
    p: &MapElitesParams,
    rng: &mut Rng,
    progress: &mut dyn FnMut(usize, &Archive),
) -> Result<(), &'static str> {
    let mut select_n = archive.capacity();
    for generation in 0..p.generations {
        let xs = if p.use_sbx {
            let pop = archive.random_xs(select_n, p.chunk_size, rng);
            variation(&pop, lower, upper, rng, p.dis_c, p.dis_m)
        } else {
            let x1 = archive.random_xs(select_n, p.chunk_size, rng);
            let x2 = archive.random_xs(select_n, p.chunk_size, rng);
            iso_dd(&x1, &x2, lower, upper, rng, p.iso_sigma, p.line_sigma)
        };
        archive.update_batch(&xs, fitness)?;
        archive.argsort();
        select_n = archive.occupied().max(1);
        progress(generation + 1, archive);
    }
    for generation in 0..p.cma_generations {
        cma_emitter_batch(archive, fitness, lower, upper, rng)?;
        archive.argsort();
        progress(p.generations + generation + 1, archive);
    }
    Ok(())
}

/// One CMA-ES emitter run: seed CMA-ES at a random good niche and drive it by
/// per-niche improvement (the Python `optimize_cma_`).
fn cma_emitter_batch(
    archive: &mut Archive,
    fitness: &mut dyn QdBatchFitness,
    lower: &[f64],
    upper: &[f64],
    rng: &mut Rng,
) -> Result<(), &'static str> {
    let best_n = 100.min(archive.capacity());
    let (x0, _) = archive.random_x_one(best_n, rng);
    let sigma = {
        let u = 0.03 + rng.uniform01() * 0.27;
        u * u
    };
    let mut fit = Fitness::bounded(archive.dim(), 1, lower, upper);
    fit.set_normalize(true);
    let params = CmaesParams {
        popsize: 31,
        max_evaluations: 100_000,
        seed: rng.int_below(i64::MAX) as u64,
        ..Default::default()
    };
    let mut es = Cmaes::new(fit, &x0, &[sigma], &params);
    let stall = 5;
    let mut last_improve = 0i32;
    let mut old_ys: Option<Vec<f64>> = None;
    for iter in 0..100 {
        let xs = es.ask();
        let (improvement, _real) = archive.update_batch(&xs, fitness)?;
        let mut sorted = improvement.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(oy) = &old_ys
            && sorted.iter().zip(oy).any(|(&a, &b)| a < b)
        {
            last_improve = iter;
        }
        if last_improve + stall < iter {
            break;
        }
        if es.tell(&improvement) != 0 {
            break;
        }
        old_ys = Some(sorted);
    }
    Ok(())
}

/// Diversifier parameters.
#[derive(Clone, Debug)]
pub struct DiversifierParams {
    /// Maximum number of candidate evaluations.
    pub max_evaluations: u64,
    /// CMA-ES population size used by each emitter.
    pub popsize: i32,
    /// Stop an emitter after this many generations without improvement.
    pub stall_criterion: i32,
}

impl Default for DiversifierParams {
    fn default() -> Self {
        Self {
            max_evaluations: 100_000,
            popsize: 31,
            stall_criterion: 20,
        }
    }
}

/// The Diversifier meta-algorithm (CMA-ME-style): drive a CMA-ES ask/tell loop
/// whose objective is per-niche improvement, filling the archive. Returns the
/// best real solution found.
///
/// # Panics
///
/// Panics if the serial adapter around `fitness` returns a batch of the wrong
/// length, which indicates a bug in this crate rather than in caller code.
/// Use [`diversify_batch`] to handle batch-length mismatches as an error.
pub fn diversify(
    archive: &mut Archive,
    fitness: &mut dyn QdFitness,
    lower: &[f64],
    upper: &[f64],
    p: &DiversifierParams,
    rng: &mut Rng,
) -> (Vec<f64>, f64) {
    let mut batch_fitness = SerialQdBatchFitness { fitness };
    diversify_batch(archive, &mut batch_fitness, lower, upper, p, rng)
        .expect("serial QD evaluation preserves batch length")
}

/// Batch-evaluation variant of [`diversify`]. CMA-ES asks and tells remain
/// serial while each requested population can be evaluated concurrently.
///
/// # Errors
///
/// Returns an error if `fitness` does not return one result per requested
/// candidate.
pub fn diversify_batch(
    archive: &mut Archive,
    fitness: &mut dyn QdBatchFitness,
    lower: &[f64],
    upper: &[f64],
    p: &DiversifierParams,
    rng: &mut Rng,
) -> Result<(Vec<f64>, f64), &'static str> {
    diversify_batch_impl(archive, fitness, lower, upper, p, rng, false)
}

/// Run the Diversifier with a hard cap on physical evaluations.
/// The final prefix can update the archive and best solution. Only complete
/// populations are told to their emitter.
///
/// # Errors
/// Returns an error if the callback gives a result count different from the request.
pub fn diversify_batch_capped(
    archive: &mut Archive,
    fitness: &mut dyn QdBatchFitness,
    lower: &[f64],
    upper: &[f64],
    p: &DiversifierParams,
    rng: &mut Rng,
) -> Result<(Vec<f64>, f64), &'static str> {
    diversify_batch_impl(archive, fitness, lower, upper, p, rng, true)
}

fn diversify_batch_impl(
    archive: &mut Archive,
    fitness: &mut dyn QdBatchFitness,
    lower: &[f64],
    upper: &[f64],
    p: &DiversifierParams,
    rng: &mut Rng,
    capped: bool,
) -> Result<(Vec<f64>, f64), &'static str> {
    let mut best_x = vec![0.0; archive.dim()];
    let mut best_y = f64::INFINITY;
    let mut evals: u64 = 0;
    let dim = archive.dim();
    let mut xs: Vec<Vec<f64>> = Vec::new();
    while evals < p.max_evaluations {
        let (x0, _) = archive.random_x_one(archive.occupied().max(1), rng);
        let sigma = {
            let u = 0.03 + rng.uniform01() * 0.27;
            u * u
        };
        let mut fit = Fitness::bounded(archive.dim(), 1, lower, upper);
        fit.set_normalize(true);
        let params = CmaesParams {
            popsize: p.popsize,
            max_evaluations: 100_000,
            seed: rng.int_below(i64::MAX) as u64,
            ..Default::default()
        };
        let mut es = Cmaes::new(fit, &x0, &[sigma], &params);
        let max_iters = 50_000 / p.popsize.max(1) as usize;
        let stall = p.stall_criterion;
        let mut last_improve = 0i32;
        let mut old_ys: Option<Vec<f64>> = None;
        for iter in 0..max_iters as i32 {
            let flat = es.ask_flat();
            let whole_population = flat.len() / dim;
            let count = if capped {
                whole_population
                    .min(usize::try_from(p.max_evaluations - evals).unwrap_or(usize::MAX))
            } else {
                whole_population
            };
            xs.resize_with(count, || vec![0.0; dim]);
            for (row, slice) in xs.iter_mut().zip(flat.chunks_exact(dim)) {
                row.copy_from_slice(slice);
            }
            let (improvement, real_ys) = archive.update_batch(&xs, fitness)?;
            evals += xs.len() as u64;
            // track best real solution
            for (x, &ry) in xs.iter().zip(&real_ys) {
                if ry < best_y {
                    best_y = ry;
                    best_x.copy_from_slice(x);
                }
            }
            if xs.len() < whole_population {
                break;
            }
            let mut sorted = improvement.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(oy) = &old_ys
                && sorted.iter().zip(oy).any(|(&a, &b)| a < b)
            {
                last_improve = iter;
            }
            if last_improve + stall < iter {
                break;
            }
            if es.tell(&improvement) != 0 || evals >= p.max_evaluations {
                break;
            }
            old_ys = Some(sorted);
        }
        archive.argsort();
    }
    Ok((best_x, best_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qd_score_without_terms_is_positive_zero() {
        let mut rng = Rng::new(3);
        let mut archive = Archive::new(2, &[0.0, 0.0], &[1.0, 1.0], 9, 0, &mut rng);
        assert_eq!(archive.qd_score().to_bits(), 0.0f64.to_bits());
        // A zero quality selects the negative-only rule, which then has no term.
        archive.set(4, 0.0, &[0.5, 0.5], &[0.0, 0.0]);
        assert_eq!(archive.qd_score().to_bits(), 0.0f64.to_bits());
        archive.set(5, -2.5, &[0.5, 0.9], &[0.0, 0.0]);
        assert_eq!(archive.qd_score(), 2.5);
    }

    #[test]
    fn nearest_center_keeps_original_id_on_ties_and_matches_scan() {
        let mut centers = vec![vec![0.25, 0.5], vec![0.75, 0.5], vec![0.25, 0.5]];
        let mut rng = Rng::new(7);
        centers.extend((0..509).map(|_| vec![rng.uniform01(), -rng.uniform01()]));
        let rows: Vec<_> = centers.iter().flatten().copied().collect();
        let tree = CenterTree::new(&rows, 2);
        assert_eq!(tree.nearest(&rows, &[0.5, 0.5]), 0);
        assert_eq!(tree.nearest(&rows, &[0.25, 0.5]), 0);
        for _ in 0..2000 {
            let x = [2.0 * rng.uniform01() - 0.5, 2.0 * rng.uniform01() - 0.5];
            assert_eq!(tree.nearest(&rows, &x), nearest_center(&x, &centers));
        }
    }

    #[test]
    fn ranked_niches_match_a_fresh_sort_after_sparse_and_dense_updates() {
        let mut rng = Rng::new(7);
        let mut archive = Archive::new(2, &[0.0, 0.0], &[1.0, 1.0], 521, 0, &mut rng);
        for wave in 0..12 {
            let count = if wave % 3 == 0 { 175 } else { 31 };
            for _ in 0..count {
                let index = rng.int_below(521) as usize;
                let score = if index.is_multiple_of(7) {
                    -0.0
                } else {
                    rng.int_below(11) as f64
                };
                archive.set(index, score, &[0.5, 0.5], &[0.0, 1.0]);
                archive.set(index, score, &[0.5, 0.5], &[0.0, 0.0]);
            }
            let mut expected: Vec<_> = (0..521).collect();
            expected.sort_by(|&a, &b| archive.ys()[a].total_cmp(&archive.ys()[b]));
            archive.argsort();
            assert_eq!(archive.si, expected);
            archive.argsort();
            assert_eq!(archive.si, expected);
        }
    }

    // A QD problem: minimize sphere; behavior = first two coordinates.
    fn qd(x: &[f64]) -> (f64, Vec<f64>) {
        let f = x.iter().map(|v| v * v).sum();
        (f, vec![x[0], x[1]])
    }

    #[test]
    fn cvt_centers_are_in_unit_box() {
        let mut rng = Rng::new(1);
        let c = cvt_centers(20, 2, 10, &mut rng);
        assert_eq!(c.len(), 20);
        for center in &c {
            for &v in center {
                assert!((0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn cvt_centers_do_not_depend_on_the_thread_count() {
        // 300 * 40 samples give twelve partial sums of 1024 samples.
        let centers = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| cvt_centers(300, 3, 40, &mut Rng::new(7)))
        };
        let serial = centers(1);
        for threads in [2, 3, 8] {
            assert_eq!(centers(threads), serial, "{threads} threads");
        }
        let mut rng = Rng::new(7);
        let archive = Archive::new(2, &[0.0; 3], &[1.0; 3], 300, 40, &mut rng);
        let copy = archive.clone();
        let Geometry::Cvt(ref geometry) = copy.geometry else {
            panic!("expected a CVT archive");
        };
        assert_eq!(
            geometry.centers,
            serial.into_iter().flatten().collect::<Vec<_>>()
        );
        assert_eq!(copy.capacity(), 300);
    }

    #[test]
    fn arithmetic_grid_covers_every_niche() {
        let mut rng = Rng::new(1);
        for capacity in [1, 10, 64, 100, 1_000, 1_025] {
            let archive = Archive::new(2, &[0.0, 0.0], &[1.0, 1.0], capacity, 0, &mut rng);
            let layout = archive.grid_layout().unwrap();
            let mut expected = 0;
            for row in 0..layout.rows {
                let columns = layout.columns_in_row(row).unwrap();
                for column in 0..columns {
                    let descriptor = [
                        (column as f64 + 0.5) / columns as f64,
                        (row as f64 + 0.5) / layout.rows as f64,
                    ];
                    assert_eq!(archive.index_of_niche(&descriptor), expected);
                    expected += 1;
                }
            }
            assert_eq!(expected, layout.cells());
            assert_eq!(archive.index_of_niche(&[0.0, 0.0]), 0);
            assert_eq!(archive.index_of_niche(&[1.0, 1.0]), capacity - 1);
        }
    }

    #[test]
    fn grid_layout_represents_rectangular_and_ragged_capacities_exactly() {
        let mut rng = Rng::new(11);
        let rectangular = Archive::new(2, &[0.0, 0.0], &[1.0, 1.0], 120, 0, &mut rng);
        let layout = rectangular.grid_layout().unwrap();
        assert_eq!(
            layout,
            GridLayout {
                rows: 10,
                base_columns: 12,
                extra_columns: 0,
            }
        );
        assert_eq!(layout.cells(), rectangular.capacity());
        assert_eq!(layout.columns_in_row(9), Some(12));
        assert_eq!(layout.columns_in_row(10), None);
        assert_eq!(rectangular.grid_shape(), Some((12, 10)));

        let ragged = Archive::new(2, &[0.0, 0.0], &[1.0, 1.0], 60, 0, &mut rng);
        let layout = ragged.grid_layout().unwrap();
        assert_eq!(
            layout,
            GridLayout {
                rows: 7,
                base_columns: 8,
                extra_columns: 4,
            }
        );
        assert_eq!(layout.cells(), ragged.capacity());
        assert_eq!(layout.columns_in_row(0), Some(9));
        assert_eq!(layout.columns_in_row(3), Some(9));
        assert_eq!(layout.columns_in_row(4), Some(8));
        assert_eq!(ragged.grid_shape(), Some((9, 7)));
    }

    #[test]
    fn map_elites_fills_niches() {
        let mut rng = Rng::new(2);
        let lower = vec![-2.0; 4];
        let upper = vec![2.0; 4];
        let mut archive = Archive::new(4, &[-2.0, -2.0], &[2.0, 2.0], 64, 10, &mut rng);
        // seed the archive with random samples so SBX has parents
        let seed_pop: Vec<Vec<f64>> = (0..64)
            .map(|_| (0..4).map(|_| -2.0 + 4.0 * rng.uniform01()).collect())
            .collect();
        archive.update(&seed_pop, &mut qd);
        archive.argsort();
        let params = MapElitesParams {
            generations: 200,
            chunk_size: 16,
            ..Default::default()
        };
        map_elites(&mut archive, &mut qd, &lower, &upper, &params, &mut rng);
        assert!(archive.occupied() > 20, "occupied={}", archive.occupied());
        assert!(archive.best_y() < 0.5, "best_y={}", archive.best_y());
    }

    #[test]
    fn evaluated_batches_match_serial_updates_and_validate_lengths() {
        let mut rng_a = Rng::new(7);
        let mut rng_b = Rng::new(7);
        let mut serial = Archive::new(2, &[-2.0, -2.0], &[2.0, 2.0], 16, 0, &mut rng_a);
        let mut batch = Archive::new(2, &[-2.0, -2.0], &[2.0, 2.0], 16, 0, &mut rng_b);
        let xs = vec![vec![-1.0, 0.5], vec![0.25, -0.75], vec![1.0, 1.0]];
        let expected = serial.update(&xs, &mut qd);
        let evaluations: Vec<_> = xs.iter().map(|x| qd(x)).collect();
        let actual = batch.update_evaluated(&xs, &evaluations).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(batch.ys(), serial.ys());
        assert_eq!(batch.descriptors(), serial.descriptors());
        assert!(batch.update_evaluated(&xs, &evaluations[..2]).is_err());
    }

    #[test]
    fn batch_map_elites_preserves_serial_results() {
        let lower = vec![-2.0; 4];
        let upper = vec![2.0; 4];
        let mut rng_a = Rng::new(19);
        let mut rng_b = Rng::new(19);
        let mut serial = Archive::new(4, &[-2.0, -2.0], &[2.0, 2.0], 32, 0, &mut rng_a);
        let mut batch = Archive::new(4, &[-2.0, -2.0], &[2.0, 2.0], 32, 0, &mut rng_b);
        serial.seed_uniform(&lower, &upper, &mut rng_a);
        batch.seed_uniform(&lower, &upper, &mut rng_b);
        let initial = serial.xs().to_vec();
        let initial_evaluations: Vec<_> = initial.iter().map(|x| qd(x)).collect();
        serial.update(&initial, &mut qd);
        batch
            .update_evaluated(&initial, &initial_evaluations)
            .unwrap();
        serial.argsort();
        batch.argsort();
        let params = MapElitesParams {
            generations: 10,
            chunk_size: 8,
            ..Default::default()
        };
        map_elites(&mut serial, &mut qd, &lower, &upper, &params, &mut rng_a);
        let mut batch_qd = |xs: &[Vec<f64>]| xs.iter().map(|x| qd(x)).collect();
        map_elites_batch(
            &mut batch,
            &mut batch_qd,
            &lower,
            &upper,
            &params,
            &mut rng_b,
        )
        .unwrap();
        assert_eq!(batch.ys(), serial.ys());
        assert_eq!(batch.xs(), serial.xs());
        assert_eq!(batch.descriptors(), serial.descriptors());
    }

    #[test]
    fn batch_progress_callback_is_ordered_and_observes_committed_updates() {
        let lower = vec![-2.0; 4];
        let upper = vec![2.0; 4];
        let mut rng = Rng::new(23);
        let mut archive = Archive::new(4, &[-2.0, -2.0], &[2.0, 2.0], 16, 0, &mut rng);
        archive.seed_uniform(&lower, &upper, &mut rng);
        let params = MapElitesParams {
            generations: 5,
            chunk_size: 8,
            ..Default::default()
        };
        let mut observed = Vec::new();
        let mut batch_qd = |xs: &[Vec<f64>]| xs.iter().map(|x| qd(x)).collect();
        map_elites_batch_with_progress(
            &mut archive,
            &mut batch_qd,
            &lower,
            &upper,
            &params,
            &mut rng,
            &mut |generation, committed| {
                observed.push((generation, committed.occupied(), committed.best_y()));
            },
        )
        .unwrap();
        assert_eq!(
            observed.iter().map(|sample| sample.0).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );
        assert!(
            observed
                .iter()
                .all(|(_, occupied, best)| *occupied > 0 && best.is_finite())
        );
        assert!(observed.windows(2).all(|pair| pair[1].2 <= pair[0].2));
    }

    #[test]
    fn diversify_improves_and_fills() {
        let mut rng = Rng::new(3);
        let lower = vec![-2.0; 4];
        let upper = vec![2.0; 4];
        let mut archive = Archive::new(4, &[-2.0, -2.0], &[2.0, 2.0], 64, 10, &mut rng);
        let seed_pop: Vec<Vec<f64>> = (0..64)
            .map(|_| (0..4).map(|_| -2.0 + 4.0 * rng.uniform01()).collect())
            .collect();
        archive.update(&seed_pop, &mut qd);
        let params = DiversifierParams {
            max_evaluations: 20_000,
            ..Default::default()
        };
        let (bx, by) = diversify(&mut archive, &mut qd, &lower, &upper, &params, &mut rng);
        assert_eq!(bx.len(), 4);
        assert!(by < 1e-3, "diversifier best_y={by}");
        assert!(archive.occupied() > 20, "occupied={}", archive.occupied());
    }

    #[test]
    fn archive_validation_and_bad_evaluations() {
        let mut rng = Rng::new(4);
        assert!(Archive::try_new(0, &[0.0], &[1.0], 4, 2, &mut rng).is_err());
        assert!(Archive::try_new(2, &[0.0], &[0.0], 4, 2, &mut rng).is_err());
        assert!(Archive::try_new(2, &[0.0], &[1.0], 0, 2, &mut rng).is_err());

        let mut archive = Archive::try_new(2, &[0.0], &[1.0], 4, 2, &mut rng).unwrap();
        let (improvements, values) =
            archive.update(&[vec![0.0, 0.0]], &mut |_: &[f64]| (f64::NAN, vec![0.5]));
        assert!(improvements[0].is_infinite());
        assert!(values[0].is_infinite());
        assert_eq!(archive.occupied(), 0);
    }

    #[test]
    fn qd_score_matches_positive_and_negative_python_rules() {
        let mut rng = Rng::new(5);
        let mut positive = Archive::new(1, &[0.0], &[1.0], 3, 2, &mut rng);
        positive.set(0, 2.0, &[0.1], &[0.0]);
        positive.set(1, 4.0, &[0.5], &[0.0]);
        assert_eq!(positive.qd_score(), 0.75);

        let mut mixed = Archive::new(1, &[0.0], &[1.0], 3, 2, &mut rng);
        mixed.set(0, -2.0, &[0.1], &[0.0]);
        mixed.set(1, 4.0, &[0.5], &[0.0]);
        assert_eq!(mixed.qd_score(), 2.0);
    }

    #[test]
    fn random_x_one_uses_sorted_best_niches() {
        let mut rng = Rng::new(6);
        let mut archive = Archive::new(1, &[0.0], &[1.0], 4, 2, &mut rng);
        for (index, value) in [4.0, 1.0, 3.0, 2.0].into_iter().enumerate() {
            archive.set(index, value, &[index as f64 / 4.0], &[index as f64]);
        }
        archive.argsort();
        for _ in 0..10 {
            let (x, y) = archive.random_x_one(1, &mut rng);
            assert_eq!(x, vec![1.0]);
            assert_eq!(y, 1.0);
        }
    }
}
