//! L-SRTDE: success-rate based adaptive differential evolution with a
//! population of newest and a population of top individuals.
//!
//! This is an independent implementation written from the published
//! description only. The authors' reference code was not consulted.
//!
//! Primary source: V. Stanovov, E. Semenkin, "Success rate-based adaptive
//! differential evolution L-SRTDE for CEC 2024 competition", Proc. IEEE
//! Congress on Evolutionary Computation (CEC 2024), Section III and
//! Algorithm 1. The default configuration follows Algorithm 1.
//!
//! Background: V. Stanovov, S. Akhmedova, E. Semenkin, "Dual-population
//! adaptive differential evolution algorithm L-NTADE", Mathematics
//! 10(24):4666, 2022 (the base algorithm); V. Stanovov, E. Semenkin,
//! "Adaptation of the scaling factor based on the success rate in
//! differential evolution", Mathematics 12(4):516, 2024.
//!
//! # Algorithm (Algorithm 1 of the paper)
//!
//! - Two populations of size `N`: newest `x^new` and top `x^top` (the best
//!   points found so far). `N_init = 20·D`, linear population size reduction
//!   to `N_min = 4`, applied to both. Initially `x^top = x^new`, `SR = 0.5`.
//! - At the start of every generation `x^new` is sorted by fitness (best
//!   first) and ranked; the loop index `i`, `r1`, `r2` and the cyclic
//!   replacement index `nc` (kept across generations) refer to positions in
//!   this sorted array.
//! - Per individual: `F ~ N(mF, 0.02)` redrawn until `F ∈ (0, 1)`, with
//!   `mF = 0.4 + 0.25·tanh(5·SR)`; memory cell `r` uniform among `H = 5`,
//!   `CR = clip(N(M_CR[r], 0.05), 0, 1)`; `r1` uniform; then `pbest` uniform
//!   among the `max(2, pb·N)` best of `x^top` with `pb = 0.7·exp(-7·SR)`,
//!   `r2` by exponential rank selection on `x^new` (probability
//!   ∝ `exp(-kp·rank/N)`, `kp = 3`) and `r3` uniform, redrawn until `r1`,
//!   `r2`, `r3` and `pbest` are pairwise different.
//! - Mutation r-new-to-ptop/n/t (Eq. 7):
//!   `v = x^new_r1 + F (x^top_pbest - x^new_i) + F (x^new_r2 - x^top_r3)`;
//!   binomial crossover with `x^new_i` (Eq. 2) and a forced coordinate; the
//!   actual rate `CR_a` = (coordinates taken from `v`)/D is recorded.
//! - Selection: if `f(u) < f(x^new_r1)` (strict, Algorithm 1 line 32), record
//!   `CR_a` and `f(x^new_r1) - f(u)`, put `u` in the temporary pool, write
//!   `u` to `x^new_nc` and advance `nc` cyclically.
//! - End of generation: `SR = NS/N` (`NS` successes); new size from LPSR;
//!   `x^top` = the best of `x^top` joined with the pool of successful trials;
//!   the worst members of `x^new` are removed when the size shrinks;
//!   `M_CR[k] = (M_CR[k] + mean_WL(S_CR)) / 2` (Eq. 5, weighted Lehmer mean
//!   with weights ∝ improvement); `k` advances every generation.
//!
//! # Deviations and unspecified details
//!
//! - **Selection operator.** Eq. (8), which describes L-NTADE, prints `<=`;
//!   Algorithm 1 (L-SRTDE) uses `<`. Algorithm 1 is followed.
//! - **Size of the p-best set.** Algorithm 1 draws `randi(1, N·pb)`; the text
//!   uses `max(2, pb·N)`. Read literally as a random integer from 1 to that
//!   real bound, i.e. `max(2, floor(pb·N))` (capped at `N`).
//! - **Bound handling.** The paper only calls it classical as in L-NTADE.
//!   Midpoint-target repair towards the crossover target `x^new_i` is used.
//! - **Memory update without successes.** Eq. (5) is undefined for an empty
//!   `S_CR`; the cell is left unchanged, and `k` still advances.
//! - **`nc` after a reduction.** If `nc` points beyond the reduced newest
//!   population it wraps (`nc mod N`).
//! - **Order after removal.** Removing the worst newest members keeps the
//!   remaining ones in place; the next generation sorts them anyway.
//! - **Rank weights.** Eq. (6) gives `exp(-kp·i/N)` for `i = 1..N`; ranks
//!   `0..N-1` are used, which differs only by a constant factor.
//! - **Within-generation updates.** The newest population is overwritten
//!   immediately, so later trials of the same generation may use the new
//!   members; ranks are not recomputed until the next generation.
//! - **Alternative difference vector.** `pbest_difference_from_base = true`
//!   (not the default) uses `x^new_r1` instead of `x^new_i` in the p-best
//!   difference, as printed for the predecessor LL-NADE (HMMOCS 2022).
//! - Budget handling and non-finite values as in [`LShade`](crate::LShade):
//!   if the budget ends inside a generation the run stops; NaN and infinite
//!   values count as `+inf`.

use fcmaes_core::Rng;
use solver_api::{Objective, Outcome, Solver};

use crate::common::{
    Evaluator, RankSelector, binomial_crossover, check_box, lpsr_size, rand_index, repair_midpoint,
    sample_clipped_normal, sorted_indices, uniform_point, weighted_lehmer_mean,
};

/// L-SRTDE with the parameter values of the CEC 2024 paper as defaults.
#[derive(Clone, Debug)]
pub struct LSrtde {
    /// Initial population size factor: `N_init = round(n_init_factor · D)` (20).
    pub n_init_factor: f64,
    /// Final population size `N_min` (4).
    pub n_min: usize,
    /// Crossover-rate memory size `H` (5).
    pub memory_size: usize,
    /// Initial value of every `M_CR` cell (1.0).
    pub memory_cr_init: f64,
    /// Standard deviation of the normal distribution for `CR` (0.05).
    pub cr_sigma: f64,
    /// Base of the mean scaling factor: `mF = f_mean_base + ...` (0.4).
    pub f_mean_base: f64,
    /// Amplitude of the `tanh` term of `mF` (0.25).
    pub f_mean_amplitude: f64,
    /// Slope of the `tanh` term of `mF`: `tanh(f_mean_slope · SR)` (5).
    pub f_mean_slope: f64,
    /// Standard deviation of the normal distribution for `F` (0.02).
    pub f_sigma: f64,
    /// Scale of the greediness: `pb = pb_scale · exp(-pb_decay · SR)` (0.7).
    pub pb_scale: f64,
    /// Decay of the greediness with the success rate (7).
    pub pb_decay: f64,
    /// Minimum size of the p-best set (2).
    pub pbest_min: usize,
    /// Selective pressure `kp` of the exponential rank selection of `r2` (3).
    pub rank_pressure: f64,
    /// Success rate before the first generation (0.5).
    pub initial_success_rate: f64,
    /// Subtract the base `x^new_r1` instead of `x^new_i` in the p-best
    /// difference, as printed for the predecessor LL-NADE (false: Eq. 7 of
    /// the L-SRTDE paper).
    pub pbest_difference_from_base: bool,
}

impl Default for LSrtde {
    fn default() -> Self {
        Self {
            n_init_factor: 20.0,
            n_min: 4,
            memory_size: 5,
            memory_cr_init: 1.0,
            cr_sigma: 0.05,
            f_mean_base: 0.4,
            f_mean_amplitude: 0.25,
            f_mean_slope: 5.0,
            f_sigma: 0.02,
            pb_scale: 0.7,
            pb_decay: 7.0,
            pbest_min: 2,
            rank_pressure: 3.0,
            initial_success_rate: 0.5,
            pbest_difference_from_base: false,
        }
    }
}

/// `F ~ N(mean, sigma)`, redrawn until it lies in the open interval `(0, 1)`.
fn sample_f(rng: &mut Rng, mean: f64, sigma: f64) -> f64 {
    loop {
        let f = rng.normreal(mean, sigma);
        if f > 0.0 && f < 1.0 {
            return f;
        }
    }
}

/// Number of p-best candidates: a random integer from 1 to `max(min, pb·n)`
/// has `max(min, floor(pb·n))` possible values, capped at `n`.
fn pbest_range(pb: f64, n: usize, min: usize) -> usize {
    ((pb * n as f64).floor() as usize).max(min).min(n).max(1)
}

/// Reorder `pop` and `fit` by ascending fitness (best first).
fn sort_population(pop: &mut Vec<Vec<f64>>, fit: &mut Vec<f64>) {
    let order = sorted_indices(fit);
    *pop = order.iter().map(|&j| std::mem::take(&mut pop[j])).collect();
    *fit = order.iter().map(|&j| fit[j]).collect();
}

impl Solver for LSrtde {
    fn name(&self) -> &'static str {
        "lsrtde"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let dim = check_box(lower, upper);
        let mut rng = Rng::new(seed);
        let mut ev = Evaluator::new(objective, max_evaluations);
        let n_min = self.n_min.max(4);
        let n_init = ((self.n_init_factor * dim as f64).round() as usize).max(n_min);
        let h = self.memory_size.max(1);

        let mut new_pop: Vec<Vec<f64>> = Vec::with_capacity(n_init);
        let mut new_fit: Vec<f64> = Vec::with_capacity(n_init);
        for _ in 0..n_init {
            if ev.exhausted() {
                return ev.outcome(lower, upper);
            }
            let x = uniform_point(&mut rng, lower, upper);
            new_fit.push(ev.eval(&x));
            new_pop.push(x);
        }
        if dim == 0 {
            return ev.outcome(lower, upper);
        }

        // Top population, kept sorted best first.
        let mut top_pop = new_pop.clone();
        let mut top_fit = new_fit.clone();
        sort_population(&mut top_pop, &mut top_fit);

        let mut m_cr = vec![self.memory_cr_init; h];
        let mut k = 0usize;
        let mut nc = 0usize;
        let mut sr = self.initial_success_rate;

        while !ev.exhausted() {
            // Lines 10-11: sort the newest population; rank = sorted position.
            sort_population(&mut new_pop, &mut new_fit);
            let n = new_pop.len();
            let selector = RankSelector::new(n, self.rank_pressure);
            let m_f = self.f_mean_base + self.f_mean_amplitude * (self.f_mean_slope * sr).tanh();
            let pb = self.pb_scale * (-self.pb_decay * sr).exp();
            let n_pbest = pbest_range(pb, n, self.pbest_min);

            let mut successes = 0usize;
            let (mut s_cr, mut s_df) = (Vec::new(), Vec::new());
            let mut pool: Vec<(Vec<f64>, f64)> = Vec::new();

            for i in 0..n {
                if ev.exhausted() {
                    break;
                }
                let f = sample_f(&mut rng, m_f, self.f_sigma);
                let r = rand_index(&mut rng, h);
                let cr = sample_clipped_normal(&mut rng, m_cr[r], self.cr_sigma);
                let r1 = rand_index(&mut rng, n);
                let (pbest, r2, r3) = loop {
                    let pbest = rand_index(&mut rng, n_pbest);
                    let r2 = selector.sample(&mut rng);
                    let r3 = rand_index(&mut rng, n);
                    let distinct = r1 != r2
                        && r1 != r3
                        && r1 != pbest
                        && r2 != r3
                        && r2 != pbest
                        && r3 != pbest;
                    if distinct {
                        break (pbest, r2, r3);
                    }
                };

                let base = &new_pop[r1];
                let current = &new_pop[i];
                let subtracted = if self.pbest_difference_from_base {
                    base
                } else {
                    current
                };
                let v: Vec<f64> = (0..dim)
                    .map(|j| {
                        base[j]
                            + f * (top_pop[pbest][j] - subtracted[j])
                            + f * (new_pop[r2][j] - top_pop[r3][j])
                    })
                    .collect();
                let mut u = vec![0.0; dim];
                let taken = binomial_crossover(&mut rng, &v, current, cr, &mut u);
                let cr_actual = taken as f64 / dim as f64;
                repair_midpoint(&mut u, current, lower, upper);

                let fu = ev.eval(&u);
                let f_base = new_fit[r1];
                if fu < f_base {
                    successes += 1;
                    s_cr.push(cr_actual);
                    s_df.push(f_base - fu);
                    pool.push((u.clone(), fu));
                    new_pop[nc] = u;
                    new_fit[nc] = fu;
                    nc = (nc + 1) % n;
                }
            }

            sr = successes as f64 / n as f64;
            let n_next = lpsr_size(n_init, n_min, ev.used(), max_evaluations).min(n);

            // Lines 43-44: the best of the top population and the successful trials.
            for (x, y) in pool {
                top_pop.push(x);
                top_fit.push(y);
            }
            sort_population(&mut top_pop, &mut top_fit);
            top_pop.truncate(n_next);
            top_fit.truncate(n_next);

            // Lines 45-47: remove the worst newest members when shrinking.
            if n_next < n {
                let order = sorted_indices(&new_fit);
                let mut keep: Vec<usize> = order[..n_next].to_vec();
                keep.sort_unstable();
                new_pop = keep.iter().map(|&j| new_pop[j].clone()).collect();
                new_fit = keep.iter().map(|&j| new_fit[j]).collect();
                nc %= n_next;
            }

            // Lines 48-49: memory update (Eq. 5); k advances every generation.
            if let Some(mean) = weighted_lehmer_mean(&s_cr, &s_df) {
                m_cr[k] = 0.5 * (m_cr[k] + mean);
            }
            k = (k + 1) % h;
        }
        ev.outcome(lower, upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_follow_the_paper() {
        let d = LSrtde::default();
        assert_eq!(d.initial_success_rate, 0.5);
        assert_eq!(d.cr_sigma, 0.05);
        assert_eq!(d.f_sigma, 0.02);
        assert_eq!((d.n_init_factor, d.n_min, d.memory_size), (20.0, 4, 5));
        assert_eq!((d.memory_cr_init, d.rank_pressure), (1.0, 3.0));
        assert!(!d.pbest_difference_from_base);
    }

    #[test]
    fn f_is_inside_the_open_unit_interval() {
        let mut rng = Rng::new(8);
        for _ in 0..10_000 {
            let f = sample_f(&mut rng, 1.0, 0.02);
            assert!(f > 0.0 && f < 1.0, "{f}");
        }
    }

    #[test]
    fn pbest_range_truncates_and_has_minimum_two() {
        assert_eq!(pbest_range(0.25, 10, 2), 2);
        assert_eq!(pbest_range(0.7, 600, 2), 420);
        assert_eq!(pbest_range(0.39, 10, 2), 3);
        assert_eq!(pbest_range(0.01, 10, 2), 2);
        assert_eq!(pbest_range(0.7, 4, 2), 2);
    }

    #[test]
    fn sorting_puts_the_best_first() {
        let mut pop = vec![vec![3.0], vec![1.0], vec![2.0]];
        let mut fit = vec![3.0, 1.0, 2.0];
        sort_population(&mut pop, &mut fit);
        assert_eq!(fit, [1.0, 2.0, 3.0]);
        assert_eq!(pop, [vec![1.0], vec![2.0], vec![3.0]]);
    }
}
