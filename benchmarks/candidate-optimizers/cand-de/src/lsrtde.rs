//! L-SRTDE: success-rate based adaptive differential evolution with a
//! population of newest and a population of top individuals.
//!
//! This is an independent implementation written from the published
//! descriptions only. The authors' reference code was not consulted.
//!
//! References:
//!
//! - V. Stanovov, E. Semenkin, "Success rate-based adaptive differential
//!   evolution L-SRTDE for CEC 2024 competition", Proc. IEEE Congress on
//!   Evolutionary Computation (CEC 2024). (Not openly accessible; not read.)
//! - V. Stanovov, S. Khodenkov, I. Rozhnov, L. Kazakovtsev, "Searching for
//!   multimode resonator topologies with adaptive differential evolution",
//!   Sensors 25(20):6447, 2025: Eqs. (2)–(6) describe L-SRTDE's mutation,
//!   selection, `F` and `pb` adaptation and population size reduction.
//! - J. Brest, M. S. Maučec, "Comparative study of modern differential
//!   evolution algorithms: perspectives on mechanisms and performance",
//!   Mathematics 13(10):1556, 2025, Section 4.7: success rate, repaired
//!   crossover rate, and "all other characteristics taken from L-NTADE".
//! - V. Stanovov, L. Kazakovtsev, E. Semenkin, "Crossover rate sorting in
//!   adaptive differential evolution", Algorithms 16(3):133, 2023: the L-NTADE
//!   description and pseudocode (newest/top populations, index sampling,
//!   cyclic replacement index, top update, `CR` memory).
//! - V. Stanovov, E. Semenkin, "Adaptation of the scaling factor based on the
//!   success rate in differential evolution", Mathematics 12(4):516, 2024:
//!   success rate, exponential rank selection, midpoint-target repair, LPSR.
//!
//! # Algorithm
//!
//! - Two populations of size `N`: newest `x^new` and top `x^top` (the best
//!   points found so far, sorted). `N_init = 20·D`, linear reduction to 4
//!   applied to both.
//! - Success rate `SR = NS / N` of the previous generation. The scaling factor
//!   is drawn around `mF = 0.4 + 0.25·tanh(5·SR)`; the greediness is
//!   `pb = 0.7·exp(-7·SR)`.
//! - Mutation r-new-to-ptop/n/t:
//!   `v_i = x^new_r1 + F (x^top_pbest - x^new_i) + F (x^new_r2 - x^top_r3)`,
//!   with `r1`, `r3` uniform, `r2` by exponential rank selection on the
//!   newest population (probability ∝ `exp(-kp·rank/N)`, `kp = 3`), `pbest`
//!   uniform among the `round(pb·N)` best of the top population.
//! - `CR` from a success-history memory (`H = 5`, initial 1.0),
//!   `CR ~ N(M_CR[r], σ)` clipped to `[0, 1]`; binomial crossover of `v_i`
//!   with `x^new_i`; the recorded value is the repaired rate
//!   `CR_a` = (coordinates taken from the mutant)/D.
//! - Selection against the base `x^new_r1`: if `f(u) <= f(x^new_r1)` the
//!   trial overwrites `x^new_nc`, `nc` advances cyclically, and the success
//!   is counted and recorded.
//! - After each generation: memory update with the weighted Lehmer mean
//!   averaged with the old cell value, then `x^top` = the `N` best of
//!   `x^top` and the trials of the generation, and the worst members of
//!   `x^new` are removed when `N` shrinks.
//!
//! # Deviations and unspecified details
//!
//! Where a choice is a parameter, it is a named field of [`LSrtde`].
//!
//! - **Distribution of `F`.** The open sources give only the mean `mF`. A
//!   normal distribution with standard deviation `f_sigma = 0.02` is used,
//!   regenerated while `F <= 0` and truncated to 1. Not verified.
//! - **`CR` standard deviation.** Not given for L-SRTDE. Brest and Maučec state
//!   that all characteristics other than the success-rate `F`, `pb` and the
//!   repaired `CR` come from L-NTADE, whose description uses
//!   `CR ~ N(M_CR, 0.1)`; `cr_sigma = 0.1` is used.
//! - **Memory cells.** The L-NTADE pseudocode samples the memory index from
//!   `1..H+1` without explaining the extra index; `H = 5` cells are used.
//! - **Memory update.** The L-NTADE description averages the old cell with
//!   the weighted Lehmer mean (`memory_averaging = true`), Lehmer power 2.
//!   Weights are the improvements `f(x^new_r1) - f(u)`; if all are zero
//!   (equal-value acceptances), equal weights are used.
//! - **Which index is rank-selected.** Both the Sensors 2025 text and the
//!   L-NTADE pseudocode apply rank-based selection to `r2` (newest
//!   population), not to `r3`. Ranks are computed once per generation.
//! - **Distinct indices.** As in the L-NTADE pseudocode, `r1`, `r2`, `r3` and
//!   `pbest` are resampled until pairwise different (`i` is not excluded).
//! - **Acceptance.** Sensors 2025, Eq. (3), uses `<=`; the L-NTADE pseudocode
//!   uses `<`. `accept_equal = true` follows the L-SRTDE source. An accepted
//!   trial counts as a success for `SR` and the memory.
//! - **Top-population pool.** Sensors 2025 says the top population is updated
//!   from all individuals created in the generation; the L-NTADE pseudocode
//!   adds only successful trials. `top_from_all_trials = true` follows the
//!   L-SRTDE source.
//! - **Minimum p-best set.** `round(pb·N)` can drop to 0 or 1; at least
//!   `pbest_min = 2` is used, as in the predecessor LL-NADE.
//! - **Crossover partner and bound repair.** Binomial crossover takes the
//!   non-mutant coordinates from `x^new_i`, and midpoint-target repair moves
//!   towards `x^new_i`. This follows the printed crossover equation; not
//!   otherwise confirmed. The randomly chosen `x^new_r1` is the mutation base
//!   and the vector the trial is compared with.
//! - **Vector in the p-best difference.** The default follows the printed
//!   L-SRTDE equation (Sensors 2025, Eq. (2), checked in the raw page text)
//!   and the L-NTADE equation (Algorithms 2023): `F (x^top_pbest - x^new_i)`.
//!   The predecessor LL-NADE (HMMOCS 2022) prints `F (x_pbest - x_r1)`
//!   instead; `pbest_difference_from_base = true` selects that form. With the
//!   default, a step relative to the compared vector `x^new_r1` is not
//!   directed towards `x^top_pbest`, and this implementation fails the
//!   30-variable rotated-ellipsoid regression test; with the LL-NADE form it
//!   passes. Which form the authors' L-SRTDE uses is not confirmed by an open
//!   source.
//! - **Initial success rate.** Not specified; `initial_success_rate = 0`.
//! - **Within-generation updates.** Trials are evaluated one by one and the
//!   newest population is overwritten immediately (as in the L-NTADE
//!   pseudocode), so later trials of the same generation may use them.
//! - Budget handling and non-finite values as in [`LShade`](crate::LShade).

use fcmaes_core::Rng;
use solver_api::{Objective, Outcome, Solver};

use crate::common::{
    Evaluator, RankSelector, binomial_crossover, check_box, lpsr_size, pbest_count, rand_index,
    repair_midpoint, sample_clipped_normal, sorted_indices, uniform_point, weighted_lehmer_mean,
};

/// L-SRTDE with the published parameter values as defaults (see the module
/// documentation for the values that are not published).
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
    /// Standard deviation of the normal distribution for `CR` (0.1, unverified).
    pub cr_sigma: f64,
    /// Average the old memory cell with the new Lehmer mean (true).
    pub memory_averaging: bool,
    /// Base of the mean scaling factor: `mF = f_mean_base + ...` (0.4).
    pub f_mean_base: f64,
    /// Amplitude of the `tanh` term of `mF` (0.25).
    pub f_mean_amplitude: f64,
    /// Slope of the `tanh` term of `mF`: `tanh(f_mean_slope · SR)` (5).
    pub f_mean_slope: f64,
    /// Standard deviation of the normal distribution for `F` (0.02, unverified).
    pub f_sigma: f64,
    /// Scale of the greediness: `pb = pb_scale · exp(-pb_decay · SR)` (0.7).
    pub pb_scale: f64,
    /// Decay of the greediness with the success rate (7).
    pub pb_decay: f64,
    /// Minimum size of the p-best set (2).
    pub pbest_min: usize,
    /// Selective pressure `kp` of the exponential rank selection of `r2` (3).
    pub rank_pressure: f64,
    /// Success rate assumed before the first generation (0).
    pub initial_success_rate: f64,
    /// Accept a trial whose value equals the target's (true).
    pub accept_equal: bool,
    /// Offer every trial of a generation to the top population, not only the
    /// successful ones (true).
    pub top_from_all_trials: bool,
    /// Subtract the base `x^new_r1` instead of `x^new_i` in the p-best
    /// difference, as printed for the predecessor LL-NADE (false: the
    /// L-SRTDE/L-NTADE equation with `x^new_i`).
    pub pbest_difference_from_base: bool,
}

impl Default for LSrtde {
    fn default() -> Self {
        Self {
            n_init_factor: 20.0,
            n_min: 4,
            memory_size: 5,
            memory_cr_init: 1.0,
            cr_sigma: 0.1,
            memory_averaging: true,
            f_mean_base: 0.4,
            f_mean_amplitude: 0.25,
            f_mean_slope: 5.0,
            f_sigma: 0.02,
            pb_scale: 0.7,
            pb_decay: 7.0,
            pbest_min: 2,
            rank_pressure: 3.0,
            initial_success_rate: 0.0,
            accept_equal: true,
            top_from_all_trials: true,
            pbest_difference_from_base: false,
        }
    }
}

impl LSrtde {
    fn sample_f(&self, rng: &mut Rng, mean: f64) -> f64 {
        loop {
            let f = rng.normreal(mean, self.f_sigma);
            if f > 0.0 {
                return f.min(1.0);
            }
        }
    }
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
        let order = sorted_indices(&new_fit);
        let mut top_pop: Vec<Vec<f64>> = order.iter().map(|&j| new_pop[j].clone()).collect();
        let mut top_fit: Vec<f64> = order.iter().map(|&j| new_fit[j]).collect();

        let mut m_cr = vec![self.memory_cr_init; h];
        let mut k = 0usize;
        let mut nc = 0usize;
        let mut sr = self.initial_success_rate;

        while !ev.exhausted() {
            let n = new_pop.len();
            let m_f = self.f_mean_base + self.f_mean_amplitude * (self.f_mean_slope * sr).tanh();
            let pb = self.pb_scale * (-self.pb_decay * sr).exp();
            let n_pbest = pbest_count(pb, n, self.pbest_min);
            let new_rank = sorted_indices(&new_fit);
            let selector = RankSelector::new(n, self.rank_pressure);

            let mut successes = 0usize;
            let (mut s_cr, mut s_df) = (Vec::new(), Vec::new());
            let mut pool: Vec<(Vec<f64>, f64)> = Vec::new();

            for i in 0..n {
                if ev.exhausted() {
                    break;
                }
                let r = rand_index(&mut rng, h);
                let cr = sample_clipped_normal(&mut rng, m_cr[r], self.cr_sigma);
                let f = self.sample_f(&mut rng, m_f);

                let (r1, r2, r3, pbest) = loop {
                    let r1 = rand_index(&mut rng, n);
                    let r2 = new_rank[selector.sample(&mut rng)];
                    let r3 = rand_index(&mut rng, n);
                    let pbest = rand_index(&mut rng, n_pbest);
                    let distinct = r1 != r2
                        && r1 != r3
                        && r1 != pbest
                        && r2 != r3
                        && r2 != pbest
                        && r3 != pbest;
                    if distinct {
                        break (r1, r2, r3, pbest);
                    }
                };

                let base = &new_pop[r1];
                let current = &new_pop[i];
                let subtracted = if self.pbest_difference_from_base {
                    base
                } else {
                    current
                };
                let mut v: Vec<f64> = (0..dim)
                    .map(|j| {
                        base[j]
                            + f * (top_pop[pbest][j] - subtracted[j])
                            + f * (new_pop[r2][j] - top_pop[r3][j])
                    })
                    .collect();
                repair_midpoint(&mut v, current, lower, upper);
                let mut u = vec![0.0; dim];
                let taken = binomial_crossover(&mut rng, &v, current, cr, &mut u);
                let cr_repaired = taken as f64 / dim as f64;

                let fu = ev.eval(&u);
                let f_target = new_fit[r1];
                let accepted = if self.accept_equal {
                    fu <= f_target
                } else {
                    fu < f_target
                };
                if accepted {
                    successes += 1;
                    s_cr.push(cr_repaired);
                    s_df.push(f_target - fu);
                }
                if accepted || self.top_from_all_trials {
                    pool.push((u.clone(), fu));
                }
                if accepted {
                    new_pop[nc] = u;
                    new_fit[nc] = fu;
                    nc = (nc + 1) % n;
                }
            }

            sr = successes as f64 / n as f64;

            if let Some(mean) = weighted_lehmer_mean(&s_cr, &s_df) {
                m_cr[k] = if self.memory_averaging {
                    0.5 * (m_cr[k] + mean)
                } else {
                    mean
                };
                k = (k + 1) % h;
            }

            let n_next = lpsr_size(n_init, n_min, ev.used(), max_evaluations).min(n);

            // Top population: the n_next best of the top population and the pool.
            let mut merged_x = std::mem::take(&mut top_pop);
            let mut merged_f = std::mem::take(&mut top_fit);
            for (x, y) in pool {
                merged_x.push(x);
                merged_f.push(y);
            }
            let order = sorted_indices(&merged_f);
            top_pop = order[..n_next]
                .iter()
                .map(|&j| merged_x[j].clone())
                .collect();
            top_fit = order[..n_next].iter().map(|&j| merged_f[j]).collect();

            // Newest population: remove the worst when the size shrinks.
            if n_next < n {
                let order = sorted_indices(&new_fit);
                let mut keep: Vec<usize> = order[..n_next].to_vec();
                keep.sort_unstable();
                new_pop = keep.iter().map(|&j| new_pop[j].clone()).collect();
                new_fit = keep.iter().map(|&j| new_fit[j]).collect();
                nc %= n_next;
            }
        }
        ev.outcome(lower, upper)
    }
}
