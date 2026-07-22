//! Random number generation for the optimizers.
//!
//! Wraps `rand_pcg::Pcg64` (the same PCG family the C++ backend used) behind a
//! small API mirroring the free helpers in the old `evaluator.h`
//! (`rand01`, `randInt`, `normreal`, `uniformVec`, `normalVec`).
//!
//! Cross-implementation parity is *statistical*: we make no attempt to
//! reproduce an historical backend's exact PCG stream, only to provide a
//! well-distributed, deterministically seedable generator.

use rand::{Rng as _, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;

/// Deterministic, seedable RNG shared by all fcmaes optimizers.
#[derive(Clone, Debug)]
pub struct Rng {
    inner: Pcg64,
}

impl Rng {
    /// Seed from a single 64-bit value (convenience for `seed`/`runid` pairs).
    pub fn new(seed: u64) -> Self {
        Self {
            inner: Pcg64::seed_from_u64(seed),
        }
    }

    /// Seed from a full 128-bit state + stream selector.
    pub fn from_state_stream(state: u128, stream: u128) -> Self {
        Self {
            inner: Pcg64::new(state, stream),
        }
    }

    /// Uniform double in `[0, 1)` — the C++ `rand01`.
    #[inline]
    pub fn uniform01(&mut self) -> f64 {
        self.inner.r#gen::<f64>()
    }

    /// Draw a full-width seed for an independent child operation.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.inner.r#gen::<u64>()
    }

    /// Standard normal sample `N(0, 1)`.
    #[inline]
    pub fn gaussian(&mut self) -> f64 {
        self.inner.sample(StandardNormal)
    }

    /// Normal sample with mean `mu` and stdev `sdev` — the C++ `normreal`.
    #[inline]
    pub fn normreal(&mut self, mu: f64, sdev: f64) -> f64 {
        self.gaussian() * sdev + mu
    }

    /// Integer in `[0, max)` matching the C++ `randInt`: `(int)(max * rand01())`.
    /// Returns 0 for `max <= 0`.
    #[inline]
    pub fn int_below(&mut self, max: i64) -> i64 {
        if max <= 0 {
            return 0;
        }
        (max as f64 * self.uniform01()) as i64
    }

    /// Vector of `dim` uniform `[0, 1)` samples.
    pub fn uniform_vec(&mut self, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| self.uniform01()).collect()
    }

    /// Vector of `dim` standard-normal samples.
    pub fn normal_vec(&mut self, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| self.gaussian()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_stream() {
        let mut a = Rng::new(12345);
        let mut b = Rng::new(12345);
        for _ in 0..100 {
            assert_eq!(a.uniform01(), b.uniform01());
        }
    }

    #[test]
    fn different_seed_diverges() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        // Extremely unlikely the first draws coincide.
        assert_ne!(a.uniform01(), b.uniform01());
    }

    #[test]
    fn full_width_seeds_reproduce_and_advance() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let first = a.next_u64();
        assert_eq!(first, b.next_u64());
        assert_ne!(first, a.next_u64());
    }

    #[test]
    fn uniform01_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let u = rng.uniform01();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn int_below_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let v = rng.int_below(5);
            assert!((0..5).contains(&v));
        }
        assert_eq!(rng.int_below(0), 0);
        assert_eq!(rng.int_below(-3), 0);
    }

    #[test]
    fn gaussian_statistics_are_sane() {
        let mut rng = Rng::new(2024);
        let n = 200_000;
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for _ in 0..n {
            let g = rng.gaussian();
            sum += g;
            sumsq += g * g;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean={mean}");
        assert!((var - 1.0).abs() < 0.03, "var={var}");
    }

    #[test]
    fn state_stream_normal_and_vector_helpers() {
        let mut first = Rng::from_state_stream(123, 7);
        let mut second = Rng::from_state_stream(123, 7);
        assert_eq!(first.uniform_vec(8), second.uniform_vec(8));
        let normal = first.normal_vec(16);
        assert_eq!(normal.len(), 16);
        assert!(normal.iter().all(|value| value.is_finite()));
        let sample = first.normreal(10.0, 0.0);
        assert_eq!(sample, 10.0);
        assert!(first.uniform_vec(0).is_empty());
        assert!(first.normal_vec(0).is_empty());
    }
}
