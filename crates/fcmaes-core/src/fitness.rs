//! Bounds handling, coordinate normalization, and (parallel) fitness
//! evaluation — the Rust port of `Fitness` / `evaluator` from the C++
//! `evaluator.h`.
//!
//! The C++ `Fitness` owned the objective function pointer directly. Here the
//! objective is decoupled behind the [`Objective`] trait so the same bounds /
//! encode / decode logic serves pure-Rust closures (tests, `fcmaes-cli`) and
//! the Python-callback bridge in `fcmaes-py`. The parallel worker-thread pool
//! (`blocking_queue` + `std::thread`) is replaced by `rayon`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::rng::Rng;

/// Value substituted for a non-finite objective result, matching the C++
/// `1E99` sentinel.
pub const NAN_REPLACEMENT: f64 = 1e99;

/// An objective function over `dim` decision variables returning `nobj`
/// objective values. Implementations must be `Sync` so populations can be
/// evaluated in parallel.
pub trait Objective: Sync {
    /// Number of objective values returned by [`eval`](Objective::eval).
    fn nobj(&self) -> usize;
    /// Evaluate the (already decoded, in-bounds) point `x`. May return
    /// non-finite values; the caller sanitizes them.
    fn eval(&self, x: &[f64]) -> Vec<f64>;

    /// Evaluate a scalar objective without forcing callers to allocate a
    /// one-element vector. Multi-objective implementations may rely on this
    /// default; scalar implementations should override it.
    fn eval_scalar(&self, x: &[f64]) -> f64 {
        self.eval(x).first().copied().unwrap_or(NAN_REPLACEMENT)
    }
}

/// Blanket impl so a plain `Fn(&[f64]) -> f64` is a single-objective
/// [`Objective`].
impl<F> Objective for F
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    fn nobj(&self) -> usize {
        1
    }
    fn eval(&self, x: &[f64]) -> Vec<f64> {
        vec![self(x)]
    }

    #[inline]
    fn eval_scalar(&self, x: &[f64]) -> f64 {
        self(x)
    }
}

#[inline]
fn sanitize(v: &mut [f64]) {
    for r in v.iter_mut() {
        if !r.is_finite() {
            *r = NAN_REPLACEMENT;
        }
    }
}

/// Rayon pools keyed by an explicitly requested worker count. Constructing a
/// pool starts threads and used to happen once per optimizer generation. The
/// cache makes that setup cost a once-per-process operation instead.
fn worker_pool(workers: usize) -> Arc<ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut pools = pools
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Arc::clone(pools.entry(workers).or_insert_with(|| {
        Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(move |index| format!("fcmaes-eval-{workers}-{index}"))
                .build()
                .expect("failed to build rayon evaluation pool"),
        )
    }))
}

/// Evaluate an ordered batch, optionally in parallel.
///
/// `workers`: `1` runs serially, values above one use exactly that many cached
/// Rayon worker threads, and values at or below zero use Rayon's global pool.
/// Results retain input order.
pub fn parallel_batch<T, R>(
    items: &[T],
    workers: i32,
    evaluate: impl Fn(&T) -> R + Sync + Send,
) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    if workers == 1 || items.len() <= 1 {
        items.iter().map(evaluate).collect()
    } else if workers <= 0 {
        items.par_iter().map(evaluate).collect()
    } else {
        worker_pool(workers as usize).install(|| items.par_iter().map(evaluate).collect())
    }
}

/// Bounds- and normalization-aware wrapper around the decision space.
///
/// When `normalize` is enabled, optimizers work in normalized coordinates
/// where the box maps to `[-1, 1]` per dimension; [`encode`](Fitness::encode)
/// and [`decode`](Fitness::decode) convert between real and normalized space.
#[derive(Clone, Debug)]
pub struct Fitness {
    dim: usize,
    nobj: usize,
    /// Empty when the problem is unbounded.
    lower: Vec<f64>,
    upper: Vec<f64>,
    scale: Vec<f64>,
    typx: Vec<f64>,
    normalize: bool,
    eval_counter: u64,
    terminate: bool,
}

impl Fitness {
    /// Create a bounded (or, with empty `lower`/`upper`, unbounded) fitness
    /// wrapper. `scale = upper - lower`, `typx = 0.5 * (upper + lower)`.
    pub fn new(dim: usize, nobj: usize, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        let bounded = !lower.is_empty();
        assert!(
            !bounded || (lower.len() == dim && upper.len() == dim),
            "bounds length must equal dim"
        );
        let (scale, typx) = if bounded {
            let scale = upper.iter().zip(&lower).map(|(u, l)| u - l).collect();
            let typx = upper
                .iter()
                .zip(&lower)
                .map(|(u, l)| 0.5 * (u + l))
                .collect();
            (scale, typx)
        } else {
            (vec![1.0; dim], vec![0.0; dim])
        };
        Self {
            dim,
            nobj,
            lower,
            upper,
            scale,
            typx,
            normalize: false,
            eval_counter: 0,
            terminate: false,
        }
    }

    /// Convenience constructor from bound slices.
    pub fn bounded(dim: usize, nobj: usize, lower: &[f64], upper: &[f64]) -> Self {
        Self::new(dim, nobj, lower.to_vec(), upper.to_vec())
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn nobj(&self) -> usize {
        self.nobj
    }
    pub fn has_bounds(&self) -> bool {
        !self.lower.is_empty()
    }
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }
    pub fn typx(&self) -> &[f64] {
        &self.typx
    }
    pub fn normalize(&self) -> bool {
        self.normalize
    }
    pub fn set_normalize(&mut self, normalize: bool) {
        self.normalize = normalize;
    }

    pub fn evaluations(&self) -> u64 {
        self.eval_counter
    }
    pub fn reset_evaluations(&mut self) {
        self.eval_counter = 0;
    }
    pub fn incr_evaluations(&mut self, by: u64) {
        self.eval_counter += by;
    }

    pub fn terminate(&self) -> bool {
        self.terminate
    }
    pub fn set_terminate(&mut self) {
        self.terminate = true;
    }

    /// Clamp `x` into the box (identity when unbounded).
    pub fn closest_feasible(&self, x: &[f64]) -> Vec<f64> {
        if !self.has_bounds() {
            return x.to_vec();
        }
        x.iter()
            .enumerate()
            .map(|(i, &v)| v.min(self.upper[i]).max(self.lower[i]))
            .collect()
    }

    /// Clamp respecting the active coordinate system: to `[-1, 1]` when
    /// normalized, else to the real box.
    pub fn closest_feasible_normed(&self, x: &[f64]) -> Vec<f64> {
        if !self.has_bounds() {
            return x.to_vec();
        }
        if self.normalize {
            x.iter().map(|&v| v.clamp(-1.0, 1.0)).collect()
        } else {
            self.closest_feasible(x)
        }
    }

    /// Map a real point to `[0, 1]^dim`: `(x - lower) / scale`.
    pub fn norm(&self, x: &[f64]) -> Vec<f64> {
        debug_assert!(self.has_bounds(), "norm requires bounds");
        x.iter()
            .enumerate()
            .map(|(i, &v)| ((v - self.lower[i]) / self.scale[i]).clamp(0.0, 1.0))
            .collect()
    }

    /// Normalize coordinate `i` into `[0, 1]` (the C++ `norm_i`).
    pub fn norm_i(&self, i: usize, x: f64) -> f64 {
        debug_assert!(self.has_bounds(), "norm_i requires bounds");
        ((x - self.lower[i]) / self.scale[i]).clamp(0.0, 1.0)
    }

    /// Lower bound of coordinate `i`.
    pub fn lower_i(&self, i: usize) -> f64 {
        self.lower[i]
    }

    /// Encode a real point into the optimizer's working coordinates: when
    /// normalized, `2 * (x - typx) / scale` (the box → `[-1, 1]`); else `x`.
    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        if !self.normalize {
            return x.to_vec();
        }
        x.iter()
            .enumerate()
            .map(|(i, &v)| 2.0 * (v - self.typx[i]) / self.scale[i])
            .collect()
    }

    /// Inverse of [`encode`](Fitness::encode): when normalized,
    /// `0.5 * x * scale + typx`; else `x`.
    pub fn decode(&self, x: &[f64]) -> Vec<f64> {
        if !self.normalize {
            return x.to_vec();
        }
        x.iter()
            .enumerate()
            .map(|(i, &v)| 0.5 * v * self.scale[i] + self.typx[i])
            .collect()
    }

    /// Decode and clamp for evaluation. Already feasible real-space points
    /// are borrowed, so the common non-normalized path does not allocate.
    fn decode_clamped<'a>(&self, x: &'a [f64]) -> Cow<'a, [f64]> {
        if self.normalize {
            if self.has_bounds() {
                Cow::Owned(
                    x.iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            (0.5 * v * self.scale[i] + self.typx[i])
                                .clamp(self.lower[i], self.upper[i])
                        })
                        .collect(),
                )
            } else {
                Cow::Owned(
                    x.iter()
                        .enumerate()
                        .map(|(i, &v)| 0.5 * v * self.scale[i] + self.typx[i])
                        .collect(),
                )
            }
        } else if self.has_bounds() {
            if x.iter()
                .enumerate()
                .all(|(i, &value)| value >= self.lower[i] && value <= self.upper[i])
            {
                Cow::Borrowed(x)
            } else {
                Cow::Owned(
                    x.iter()
                        .enumerate()
                        .map(|(i, &v)| v.clamp(self.lower[i], self.upper[i]))
                        .collect(),
                )
            }
        } else {
            Cow::Borrowed(x)
        }
    }

    /// Uniform random point inside the box.
    pub fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        debug_assert!(self.has_bounds(), "sample requires bounds");
        (0..self.dim)
            .map(|i| self.lower[i] + self.scale[i] * rng.uniform01())
            .collect()
    }

    /// Uniform random value for coordinate `i` (the C++ `sample_i`).
    pub fn sample_i(&self, i: usize, rng: &mut Rng) -> f64 {
        debug_assert!(self.has_bounds(), "sample_i requires bounds");
        self.lower[i] + self.scale[i] * rng.uniform01()
    }

    /// Clamp coordinate `i` into its bound (the C++ `getClosestFeasible_i`).
    pub fn closest_feasible_i(&self, i: usize, x: f64) -> f64 {
        if !self.has_bounds() {
            return x;
        }
        x.min(self.upper[i]).max(self.lower[i])
    }

    /// Whether coordinate `i` of value `x` lies within its bound.
    pub fn feasible_i(&self, i: usize, x: f64) -> bool {
        !self.has_bounds() || (x >= self.lower[i] && x <= self.upper[i])
    }

    /// Penalized bound-violation magnitude for a (possibly normalized) point.
    pub fn violation(&self, x: &[f64], penalty_coef: f64) -> f64 {
        if !self.has_bounds() {
            return 0.0;
        }
        let mut sum = 0.0;
        for (i, &value) in x.iter().enumerate() {
            let decoded = if self.normalize {
                0.5 * value * self.scale[i] + self.typx[i]
            } else {
                value
            };
            sum += (self.lower[i] - decoded).max(0.0);
            sum += (decoded - self.upper[i]).max(0.0);
        }
        penalty_coef * sum
    }

    /// Evaluate one *encoded* candidate: decode, clamp into the box, call the
    /// objective, sanitize non-finite results, and bump the eval counter.
    pub fn eval_encoded(&mut self, encoded: &[f64], obj: &impl Objective) -> Vec<f64> {
        let x = self.decode_clamped(encoded);
        let mut res = obj.eval(&x);
        sanitize(&mut res);
        self.eval_counter += 1;
        res
    }

    /// Allocation-light scalar counterpart of [`eval_encoded`](Self::eval_encoded).
    pub fn eval_encoded_scalar(&mut self, encoded: &[f64], obj: &impl Objective) -> f64 {
        let x = self.decode_clamped(encoded);
        let value = obj.eval_scalar(&x);
        self.eval_counter += 1;
        if value.is_finite() {
            value
        } else {
            NAN_REPLACEMENT
        }
    }

    /// Evaluate a whole population of *encoded* candidates, optionally in
    /// parallel (the port of `Fitness::values` + the worker-thread pool).
    ///
    /// `workers`: `1` = serial; `>1` = that many rayon threads; `<=0` = all
    /// available cores. Each row is decoded, clamped, evaluated, and
    /// sanitized. The eval counter advances by the population size.
    pub fn eval_population(
        &mut self,
        pop: &[Vec<f64>],
        obj: &impl Objective,
        workers: i32,
    ) -> Vec<Vec<f64>> {
        let eval_one = |enc: &Vec<f64>| -> Vec<f64> {
            let x = self.decode_clamped(enc);
            let mut res = obj.eval(&x);
            sanitize(&mut res);
            res
        };

        let ys = parallel_batch(pop, workers, eval_one);

        self.eval_counter += pop.len() as u64;
        ys
    }

    /// Single-objective convenience over [`eval_population`](Fitness::eval_population).
    pub fn eval_population_scalar(
        &mut self,
        pop: &[Vec<f64>],
        obj: &impl Objective,
        workers: i32,
    ) -> Vec<f64> {
        let eval_one = |enc: &Vec<f64>| -> f64 {
            let x = self.decode_clamped(enc);
            let value = obj.eval_scalar(&x);
            if value.is_finite() {
                value
            } else {
                NAN_REPLACEMENT
            }
        };

        let ys = parallel_batch(pop, workers, eval_one);
        self.eval_counter += pop.len() as u64;
        ys
    }

    /// Evaluate a scalar population stored as contiguous fixed-width chunks.
    /// This avoids building `Vec<Vec<f64>>` adapters for column-major matrix
    /// populations such as CMA-ES (`DMatrix` columns are contiguous).
    pub fn eval_population_scalar_flat(
        &mut self,
        population: &[f64],
        dimensions: usize,
        obj: &impl Objective,
        workers: i32,
    ) -> Vec<f64> {
        assert!(dimensions > 0, "population dimensions must be positive");
        assert_eq!(
            population.len() % dimensions,
            0,
            "flat population length must be divisible by dimensions"
        );
        let candidates = population.len() / dimensions;
        let eval_one = |encoded: &[f64]| {
            let decoded = self.decode_clamped(encoded);
            let value = obj.eval_scalar(&decoded);
            if value.is_finite() {
                value
            } else {
                NAN_REPLACEMENT
            }
        };
        let values = if workers == 1 || candidates <= 1 {
            population.chunks_exact(dimensions).map(eval_one).collect()
        } else if workers <= 0 {
            population
                .par_chunks_exact(dimensions)
                .map(eval_one)
                .collect()
        } else {
            worker_pool(workers as usize).install(|| {
                population
                    .par_chunks_exact(dimensions)
                    .map(eval_one)
                    .collect()
            })
        };
        self.eval_counter += candidates as u64;
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-12, "{x} != {y}");
        }
    }

    #[test]
    fn scale_and_typx() {
        let f = Fitness::bounded(3, 1, &[-1.0, 0.0, 2.0], &[1.0, 10.0, 4.0]);
        approx(f.scale(), &[2.0, 10.0, 2.0]);
        approx(f.typx(), &[0.0, 5.0, 3.0]);
    }

    #[test]
    fn encode_decode_roundtrip_normalized() {
        let mut f = Fitness::bounded(2, 1, &[-2.0, 10.0], &[2.0, 20.0]);
        f.set_normalize(true);
        // Box corners map to +/-1.
        approx(&f.encode(&[-2.0, 10.0]), &[-1.0, -1.0]);
        approx(&f.encode(&[2.0, 20.0]), &[1.0, 1.0]);
        approx(&f.encode(&[0.0, 15.0]), &[0.0, 0.0]);
        // decode is the exact inverse.
        let x = [0.3, 17.5];
        approx(&f.decode(&f.encode(&x)), &x);
    }

    #[test]
    fn encode_decode_identity_when_not_normalized() {
        let f = Fitness::bounded(2, 1, &[-2.0, 10.0], &[2.0, 20.0]);
        let x = [0.3, 17.5];
        approx(&f.encode(&x), &x);
        approx(&f.decode(&x), &x);
    }

    #[test]
    fn norm_maps_to_unit_and_clamps() {
        let f = Fitness::bounded(2, 1, &[0.0, -5.0], &[10.0, 5.0]);
        approx(&f.norm(&[5.0, 0.0]), &[0.5, 0.5]);
        approx(&f.norm(&[-100.0, 100.0]), &[0.0, 1.0]); // clamped
    }

    #[test]
    fn closest_feasible_clamps_to_box() {
        let f = Fitness::bounded(2, 1, &[0.0, 0.0], &[1.0, 1.0]);
        approx(&f.closest_feasible(&[-1.0, 2.0]), &[0.0, 1.0]);
    }

    #[test]
    fn closest_feasible_normed_uses_unit_box_when_normalized() {
        let mut f = Fitness::bounded(1, 1, &[0.0], &[10.0]);
        f.set_normalize(true);
        approx(&f.closest_feasible_normed(&[5.0]), &[1.0]); // clamped to [-1,1]
        f.set_normalize(false);
        approx(&f.closest_feasible_normed(&[5.0]), &[5.0]); // within real box
    }

    #[test]
    fn violation_penalizes_out_of_box() {
        let f = Fitness::bounded(2, 1, &[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(f.violation(&[0.5, 0.5], 100.0), 0.0);
        assert_eq!(f.violation(&[-1.0, 2.0], 10.0), 10.0 * (1.0 + 1.0));
    }

    #[test]
    fn sample_within_bounds() {
        let f = Fitness::bounded(3, 1, &[-1.0, 5.0, 0.0], &[1.0, 6.0, 100.0]);
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let x = f.sample(&mut rng);
            for ((&xi, &lo), &up) in x.iter().zip(f.lower()).zip(f.upper()) {
                assert!(xi >= lo && xi < up);
            }
        }
    }

    #[test]
    fn eval_sanitizes_and_counts() {
        let mut f = Fitness::bounded(1, 1, &[0.0], &[1.0]);
        let obj = |x: &[f64]| if x[0] > 0.5 { f64::NAN } else { x[0] };
        let a = f.eval_encoded(&[0.25], &obj);
        approx(&a, &[0.25]);
        let b = f.eval_encoded(&[0.9], &obj);
        approx(&b, &[NAN_REPLACEMENT]);
        assert_eq!(f.eval_encoded_scalar(&[0.9], &obj), NAN_REPLACEMENT);
        assert_eq!(f.evaluations(), 3);
    }

    #[test]
    fn eval_clamps_before_calling_objective() {
        // Objective records the x it sees; out-of-box input must be clamped.
        let mut f = Fitness::bounded(1, 1, &[0.0], &[1.0]);
        let obj = |x: &[f64]| x[0];
        let y = f.eval_encoded(&[5.0], &obj);
        approx(&y, &[1.0]);
    }

    #[test]
    fn parallel_matches_serial() {
        let mut f = Fitness::bounded(2, 1, &[-5.0, -5.0], &[5.0, 5.0]);
        let obj = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
        let pop: Vec<Vec<f64>> = (0..64)
            .map(|i| vec![(i as f64) * 0.1 - 3.0, (i as f64) * -0.05 + 1.0])
            .collect();
        let serial = f.eval_population_scalar(&pop, &obj, 1);
        let parallel = f.eval_population_scalar(&pop, &obj, 4);
        approx(&serial, &parallel);
        assert_eq!(f.evaluations(), 2 * pop.len() as u64);
        let inputs: Vec<i32> = (0..64).collect();
        let expected: Vec<i32> = inputs.iter().map(|value| value * value).collect();
        for workers in [1, 0, 4] {
            assert_eq!(
                parallel_batch(&inputs, workers, |value| value * value),
                expected
            );
        }
    }

    struct MultiObjective;

    impl Objective for MultiObjective {
        fn nobj(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> Vec<f64> {
            vec![x.iter().sum(), f64::INFINITY]
        }
    }

    struct EmptyObjective;

    impl Objective for EmptyObjective {
        fn nobj(&self) -> usize {
            0
        }

        fn eval(&self, _: &[f64]) -> Vec<f64> {
            Vec::new()
        }
    }

    #[test]
    fn multiobjective_population_and_default_scalar_paths() {
        let pop = vec![vec![0.25, 0.5], vec![0.75, 1.5]];
        for workers in [1, 0, 2, 2] {
            let mut fitness = Fitness::bounded(2, 2, &[0.0, 0.0], &[1.0, 1.0]);
            let values = fitness.eval_population(&pop, &MultiObjective, workers);
            assert_eq!(values[0], vec![0.75, NAN_REPLACEMENT]);
            assert_eq!(values[1], vec![1.75, NAN_REPLACEMENT]);
            assert_eq!(fitness.evaluations(), 2);
            assert_eq!(fitness.nobj(), 2);
            assert_eq!(MultiObjective.nobj(), 2);
        }
        assert_eq!(EmptyObjective.eval_scalar(&[]), NAN_REPLACEMENT);
    }

    #[test]
    fn unbounded_and_lifecycle_paths() {
        let mut fitness = Fitness::new(2, 1, Vec::new(), Vec::new());
        assert!(!fitness.has_bounds());
        assert_eq!(fitness.scale(), &[1.0, 1.0]);
        assert_eq!(fitness.typx(), &[0.0, 0.0]);
        assert_eq!(fitness.closest_feasible(&[-2.0, 3.0]), vec![-2.0, 3.0]);
        assert_eq!(
            fitness.closest_feasible_normed(&[-2.0, 3.0]),
            vec![-2.0, 3.0]
        );
        assert_eq!(fitness.closest_feasible_i(0, -2.0), -2.0);
        assert_eq!(fitness.violation(&[-2.0, 3.0], 10.0), 0.0);
        fitness.set_normalize(true);
        assert!(fitness.normalize());
        let objective = |x: &[f64]| x.iter().sum();
        assert_eq!(fitness.eval_encoded(&[2.0, 4.0], &objective), vec![3.0]);
        fitness.incr_evaluations(4);
        assert_eq!(fitness.evaluations(), 5);
        fitness.reset_evaluations();
        assert_eq!(fitness.evaluations(), 0);
        assert!(!fitness.terminate());
        fitness.set_terminate();
        assert!(fitness.terminate());
    }

    #[test]
    fn scalar_population_sanitizes_on_global_pool_and_singleton() {
        let objective = |_: &[f64]| f64::NEG_INFINITY;
        let pop = vec![vec![0.5], vec![0.2]];
        let mut fitness = Fitness::bounded(1, 1, &[0.0], &[1.0]);
        assert_eq!(
            fitness.eval_population_scalar(&pop, &objective, 0),
            vec![NAN_REPLACEMENT; 2]
        );
        assert_eq!(
            fitness.eval_population_scalar(&pop[..1], &objective, 8),
            vec![NAN_REPLACEMENT]
        );
    }

    #[test]
    fn flat_scalar_population_matches_nested_for_all_worker_paths() {
        let objective = |x: &[f64]| x.iter().sum();
        let flat = [0.0, 0.5, 0.25, 0.75, 1.0, 1.0];
        for workers in [1, 0, 3] {
            let mut fitness = Fitness::bounded(2, 1, &[0.0; 2], &[1.0; 2]);
            assert_eq!(
                fitness.eval_population_scalar_flat(&flat, 2, &objective, workers),
                vec![0.5, 1.0, 2.0]
            );
            assert_eq!(fitness.evaluations(), 3);
        }
    }

    #[test]
    #[should_panic(expected = "population dimensions must be positive")]
    fn flat_population_rejects_zero_dimensions() {
        let mut fitness = Fitness::new(0, 1, Vec::new(), Vec::new());
        let _ = fitness.eval_population_scalar_flat(&[], 0, &|_: &[f64]| 0.0, 1);
    }
}
