// Numeric kernels index parallel arrays by a shared counter.
#![allow(clippy::needless_range_loop)]

//! Quality-Diversity: CVT-MAP-Elites and the Diversifier meta-algorithm —
//! Rust equivalents of the pure-Python `fcmaes/mapelites.py` and
//! `fcmaes/diversifier.py`.
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

use crate::cmaes::{Cmaes, CmaesParams};
use crate::fitness::Fitness;
use crate::rng::Rng;
use rayon::prelude::*;

/// Quality-diversity fitness: maps a solution to `(fitness, behavior)`.
pub trait QdFitness {
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

/// Uniform two-dimensional grid with exactly `k` centers. Rows differ by at
/// most one column when `k` is not a square.
fn grid_centers_2d(k: usize) -> Vec<Vec<f64>> {
    let rows = (k as f64).sqrt().floor().max(1.0) as usize;
    let base_columns = k / rows;
    let extra_columns = k % rows;
    let mut centers = Vec::with_capacity(k);
    for row in 0..rows {
        let columns = base_columns + usize::from(row < extra_columns);
        for column in 0..columns {
            centers.push(vec![
                (column as f64 + 0.5) / columns as f64,
                (row as f64 + 0.5) / rows as f64,
            ]);
        }
    }
    centers
}

fn grid_index_2d(k: usize, descriptor: &[f64]) -> usize {
    let rows = (k as f64).sqrt().floor().max(1.0) as usize;
    let base_columns = k / rows;
    let extra_columns = k % rows;
    let y = descriptor[1].clamp(0.0, 1.0 - f64::EPSILON);
    let row = (y * rows as f64) as usize;
    let columns = base_columns + usize::from(row < extra_columns);
    let x = descriptor[0].clamp(0.0, 1.0 - f64::EPSILON);
    let column = (x * columns as f64) as usize;
    row * base_columns + row.min(extra_columns) + column
}

/// Compute `k` niche centers in `[0,1]^dim` via k-means++ over
/// `k * samples_per_niche` uniform samples (Lloyd iterations).
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
        d2.par_iter_mut().zip(&samples).for_each(|(distance, s)| {
            let candidate = dist2(s, c);
            if candidate < *distance {
                *distance = candidate;
            }
        });
    }

    // A few Lloyd iterations. Assignment is the dominant O(samples * k)
    // kernel, so use thread-local accumulators and merge them in parallel.
    for _ in 0..10 {
        let (sums, counts) = samples
            .par_iter()
            .fold(
                || (vec![0.0; k * dim], vec![0usize; k]),
                |(mut sums, mut counts), s| {
                    let mut best = 0;
                    let mut best_distance = f64::MAX;
                    for (index, center) in centers.iter().enumerate() {
                        let distance = dist2(s, center);
                        if distance < best_distance {
                            best_distance = distance;
                            best = index;
                        }
                    }
                    counts[best] += 1;
                    for j in 0..dim {
                        sums[best * dim + j] += s[j];
                    }
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![0.0; k * dim], vec![0usize; k]),
                |(mut left_sums, mut left_counts), (right_sums, right_counts)| {
                    for index in 0..k {
                        left_counts[index] += right_counts[index];
                    }
                    for index in 0..k * dim {
                        left_sums[index] += right_sums[index];
                    }
                    (left_sums, left_counts)
                },
            );
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

/// CVT quality-diversity archive: `capacity` niches, each holding the best
/// solution found for it.
pub struct Archive {
    dim: usize,
    qd_dim: usize,
    capacity: usize,
    desc_lb: Vec<f64>,
    desc_scale: Vec<f64>,
    centers: Vec<Vec<f64>>, // normalized [0,1]^qd_dim
    grid_2d: bool,
    xs: Vec<Vec<f64>>,
    ds: Vec<Vec<f64>>,
    ys: Vec<f64>,
    counts: Vec<u64>,
    occupied: usize,
    si: Vec<usize>, // niche indices sorted ascending by fitness
}

impl Archive {
    /// Construct a validated CVT archive.
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
        let grid_2d = samples_per_niche == 0 && qd_dim == 2;
        let centers = if grid_2d {
            grid_centers_2d(capacity)
        } else {
            cvt_centers(capacity, qd_dim, samples_per_niche.max(1), rng)
        };
        Archive {
            dim,
            qd_dim,
            capacity,
            desc_lb,
            desc_scale,
            centers,
            grid_2d,
            xs: vec![vec![0.0; dim]; capacity],
            ds: vec![vec![0.0; qd_dim]; capacity],
            ys: vec![f64::INFINITY; capacity],
            counts: vec![0; capacity],
            occupied: 0,
            si: (0..capacity).collect(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn qd_dim(&self) -> usize {
        self.qd_dim
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn occupied(&self) -> usize {
        self.occupied
    }

    /// Seed all niche solutions with uniform random samples in `[lower, upper]`
    /// (never evaluated — they serve as the initial SBX/Iso parent pool, as the
    /// Python original documents).
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
    pub fn index_of_niche(&self, d: &[f64]) -> usize {
        assert_eq!(d.len(), self.qd_dim, "descriptor length must equal qd_dim");
        let e = self.encode_d(d);
        if self.grid_2d {
            return grid_index_2d(self.capacity, &e);
        }
        let mut best = 0;
        let mut bd = f64::MAX;
        for (i, c) in self.centers.iter().enumerate() {
            let dist = dist2(&e, c);
            if dist < bd {
                bd = dist;
                best = i;
            }
        }
        best
    }

    /// Add a solution to niche `i` if it improves it.
    pub fn set(&mut self, i: usize, y: f64, d: &[f64], x: &[f64]) {
        assert!(i < self.capacity, "niche index out of bounds");
        assert_eq!(d.len(), self.qd_dim, "descriptor length mismatch");
        assert_eq!(x.len(), self.dim, "solution length mismatch");
        self.counts[i] += 1;
        if y.is_finite() && d.iter().all(|v| v.is_finite()) && y < self.ys[i] {
            if self.ys[i].is_infinite() {
                self.occupied += 1;
            }
            self.ys[i] = y;
            self.xs[i].copy_from_slice(x);
            self.ds[i].copy_from_slice(d);
        }
    }

    /// Evaluate `xs`, add to the archive, and return `(improvements, real_ys)`
    /// where `improvement = fitness - niche's previous fitness` (negative is an
    /// improvement — the objective the Diversifier's optimizer minimizes).
    pub fn update(&mut self, xs: &[Vec<f64>], fitness: &mut dyn QdFitness) -> (Vec<f64>, Vec<f64>) {
        let evaluations: Vec<(f64, Vec<f64>)> = xs.iter().map(|x| fitness.eval(x)).collect();
        self.update_evaluated(xs, &evaluations)
            .expect("serial QD evaluation preserves batch length")
    }

    /// Apply already evaluated `(fitness, descriptor)` values in input order.
    /// Keeping this step separate lets callers parallelize expensive objective
    /// functions without concurrently mutating the archive.
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
        let mut si: Vec<usize> = (0..self.capacity).collect();
        si.sort_by(|&a, &b| self.ys[a].total_cmp(&self.ys[b]));
        self.si = si;
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

    pub fn best_y(&self) -> f64 {
        self.ys.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// QD-score matching Python `Archive.get_qd_score`: for an all-positive
    /// archive, sum reciprocal fitness; otherwise sum the negated negative
    /// fitness values. Higher is better in both cases.
    pub fn qd_score(&self) -> f64 {
        let finite: Vec<f64> = self
            .ys
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if finite.is_empty() {
            return 0.0;
        }
        if finite.iter().copied().fold(f64::INFINITY, f64::min) > 0.0 {
            finite
                .iter()
                .filter(|&&value| value != 0.0)
                .map(|&value| value.recip())
                .sum()
        } else {
            finite
                .iter()
                .filter(|&&value| value < 0.0)
                .map(|&value| -value)
                .sum()
        }
    }

    pub fn ys(&self) -> &[f64] {
        &self.ys
    }
    pub fn xs(&self) -> &[Vec<f64>] {
        &self.xs
    }
    pub fn descriptors(&self) -> &[Vec<f64>] {
        &self.ds
    }
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
            let mut beta = if mu <= 0.5 {
                (2.0 * mu).powf(1.0 / (dis_c + 1.0))
            } else {
                (2.0 * mu).powf(-1.0 / (dis_c + 1.0))
            };
            if rng.int_below(2) == 1 {
                beta = -beta;
            }
            if rng.uniform01() < 0.5 {
                beta = 1.0;
            }
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
    pub generations: usize,
    pub chunk_size: usize,
    pub use_sbx: bool,
    pub dis_c: f64,
    pub dis_m: f64,
    pub iso_sigma: f64,
    pub line_sigma: f64,
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
pub fn map_elites_batch(
    archive: &mut Archive,
    fitness: &mut dyn QdBatchFitness,
    lower: &[f64],
    upper: &[f64],
    p: &MapElitesParams,
    rng: &mut Rng,
) -> Result<(), &'static str> {
    let mut select_n = archive.capacity();
    for _ in 0..p.generations {
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
    }
    for _ in 0..p.cma_generations {
        cma_emitter_batch(archive, fitness, lower, upper, rng)?;
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
    pub max_evaluations: u64,
    pub popsize: i32,
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
pub fn diversify_batch(
    archive: &mut Archive,
    fitness: &mut dyn QdBatchFitness,
    lower: &[f64],
    upper: &[f64],
    p: &DiversifierParams,
    rng: &mut Rng,
) -> Result<(Vec<f64>, f64), &'static str> {
    let mut best_x = vec![0.0; archive.dim()];
    let mut best_y = f64::INFINITY;
    let mut evals: u64 = 0;
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
            let xs = es.ask();
            let (improvement, real_ys) = archive.update_batch(&xs, fitness)?;
            evals += xs.len() as u64;
            // track best real solution
            for (x, &ry) in xs.iter().zip(&real_ys) {
                if ry < best_y {
                    best_y = ry;
                    best_x.copy_from_slice(x);
                }
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
    fn grid_centers_are_fast_exact_and_cover_the_box() {
        for capacity in [1, 10, 64, 1_000] {
            let centers = grid_centers_2d(capacity);
            assert_eq!(centers.len(), capacity);
            assert!(
                centers
                    .iter()
                    .flatten()
                    .all(|value| (0.0..=1.0).contains(value))
            );
        }
        let mut rng = Rng::new(1);
        let archive = Archive::new(2, &[0.0, 0.0], &[1.0, 1.0], 100, 0, &mut rng);
        assert_eq!(archive.centers.len(), 100);
        assert_eq!(archive.index_of_niche(&[0.0, 0.0]), 0);
        assert_eq!(archive.index_of_niche(&[1.0, 1.0]), 99);
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
