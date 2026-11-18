//! L-SHADE: success-history based adaptive differential evolution with linear
//! population size reduction.
//!
//! Reference: R. Tanabe, A. Fukunaga, "Improving the search performance of
//! SHADE using linear population size reduction", Proc. IEEE Congress on
//! Evolutionary Computation (CEC 2014), pp. 1658–1665.
//!
//! # Algorithm
//!
//! - Population of `round(18·D)` uniform random points, reduced linearly in the
//!   evaluation count to 4 after every generation (worst individuals removed).
//! - Historical memory of `H = 6` cells for `F` and `CR`, initialized to 0.5.
//!   Each individual picks a cell `r` uniformly; `CR ~ N(M_CR[r], 0.1)`
//!   clipped to `[0, 1]` (0 if the cell holds the terminal value);
//!   `F ~ Cauchy(M_F[r], 0.1)`, regenerated while `F <= 0`, truncated to 1.
//! - Mutation current-to-pbest/1 with archive:
//!   `v = x_i + F (x_pbest - x_i) + F (x_r1 - x_r2)`; `pbest` is uniform among
//!   the `max(2, round(p·N))` best (`p = 0.11`), `r1 != i` from the population,
//!   `r2 != i, r1` from population ∪ archive.
//! - Midpoint-target bound repair towards the parent, binomial crossover with
//!   a forced coordinate.
//! - Generational selection: the trial replaces its parent if `f(u) <= f(x)`;
//!   on strict improvement the parent enters the archive (capacity
//!   `round(2.6·N)`; when full, the parent overwrites a random member; after
//!   a population reduction, random members are removed) and
//!   `(F, CR, f(x) - f(u))` is recorded.
//! - Memory cell `k` (cyclic) becomes the weighted Lehmer mean of the recorded
//!   values (weights proportional to the improvement); `M_CR[k]` becomes
//!   terminal (and stays terminal) when `max(S_CR) = 0`. `k` advances only
//!   when something was recorded.
//!
//! # Deviations and unspecified details
//!
//! - The trials of a generation are generated from the population at its
//!   start and evaluated in order; if the budget ends inside a generation, the
//!   evaluated trials are still selected and the run ends.
//! - Non-finite objective values count as `+inf`. An improvement from `+inf`
//!   to a finite value has infinite weight; the Lehmer mean then shares the
//!   weight equally among such records.
//! - If the budget is smaller than the initial population, only that many
//!   initial points are evaluated and the best is returned.

use fcmaes_core::Rng;
use solver_api::{Objective, Outcome, Solver};

use crate::common::{
    Archive, Evaluator, binomial_crossover, check_box, lpsr_size, pbest_count, rand_index,
    repair_midpoint, sample_clipped_normal, sample_f_cauchy, sorted_indices, uniform_point,
    weighted_lehmer_mean,
};

/// L-SHADE with the published tuned parameter values as defaults.
#[derive(Clone, Debug)]
pub struct LShade {
    /// Initial population size factor: `N_init = round(n_init_factor · D)` (18).
    pub n_init_factor: f64,
    /// Final population size `N_min` (4).
    pub n_min: usize,
    /// Historical memory size `H` (6).
    pub memory_size: usize,
    /// Initial value of every `M_F` cell (0.5).
    pub memory_f_init: f64,
    /// Initial value of every `M_CR` cell (0.5).
    pub memory_cr_init: f64,
    /// Fraction `p` of the population forming the p-best set (0.11).
    pub p_best_rate: f64,
    /// Archive capacity relative to the current population size (2.6).
    pub archive_rate: f64,
    /// Scale of the Cauchy distribution for `F` (0.1).
    pub f_scale: f64,
    /// Standard deviation of the normal distribution for `CR` (0.1).
    pub cr_sigma: f64,
}

impl Default for LShade {
    fn default() -> Self {
        Self {
            n_init_factor: 18.0,
            n_min: 4,
            memory_size: 6,
            memory_f_init: 0.5,
            memory_cr_init: 0.5,
            p_best_rate: 0.11,
            archive_rate: 2.6,
            f_scale: 0.1,
            cr_sigma: 0.1,
        }
    }
}

/// Marker for the terminal crossover-rate memory value.
const TERMINAL: f64 = -1.0;

impl Solver for LShade {
    fn name(&self) -> &'static str {
        "lshade"
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

        // Initial population; stop early if the budget is smaller.
        let mut pop: Vec<Vec<f64>> = Vec::with_capacity(n_init);
        let mut fit: Vec<f64> = Vec::with_capacity(n_init);
        for _ in 0..n_init {
            if ev.exhausted() {
                return ev.outcome(lower, upper);
            }
            let x = uniform_point(&mut rng, lower, upper);
            fit.push(ev.eval(&x));
            pop.push(x);
        }
        if dim == 0 {
            return ev.outcome(lower, upper);
        }

        let mut m_f = vec![self.memory_f_init; h];
        let mut m_cr = vec![self.memory_cr_init; h];
        let mut k = 0usize;
        let mut archive = Archive::default();

        while !ev.exhausted() {
            let n = pop.len();
            let order = sorted_indices(&fit);
            let n_pbest = pbest_count(self.p_best_rate, n, 2);

            // Generate all trials from the current population.
            let mut trials: Vec<(Vec<f64>, f64, f64)> = Vec::with_capacity(n);
            for i in 0..n {
                let r = rand_index(&mut rng, h);
                let cr = if m_cr[r] == TERMINAL {
                    0.0
                } else {
                    sample_clipped_normal(&mut rng, m_cr[r], self.cr_sigma)
                };
                let f = sample_f_cauchy(&mut rng, m_f[r], self.f_scale);

                let pbest = order[rand_index(&mut rng, n_pbest)];
                let r1 = loop {
                    let c = rand_index(&mut rng, n);
                    if c != i {
                        break c;
                    }
                };
                let r2 = loop {
                    let c = rand_index(&mut rng, n + archive.len());
                    if c != i && c != r1 {
                        break c;
                    }
                };
                let x_r2: &[f64] = if r2 < n {
                    &pop[r2]
                } else {
                    archive.get(r2 - n)
                };
                let xi = &pop[i];
                let mut v: Vec<f64> = (0..dim)
                    .map(|j| xi[j] + f * (pop[pbest][j] - xi[j]) + f * (pop[r1][j] - x_r2[j]))
                    .collect();
                repair_midpoint(&mut v, xi, lower, upper);
                let mut u = vec![0.0; dim];
                binomial_crossover(&mut rng, &v, xi, cr, &mut u);
                trials.push((u, f, cr));
            }

            // Evaluate and select.
            let (mut s_f, mut s_cr, mut s_df) = (Vec::new(), Vec::new(), Vec::new());
            for (i, (u, f, cr)) in trials.into_iter().enumerate() {
                if ev.exhausted() {
                    break;
                }
                let fu = ev.eval(&u);
                if fu <= fit[i] {
                    if fu < fit[i] {
                        s_f.push(f);
                        s_cr.push(cr);
                        s_df.push(fit[i] - fu);
                        let capacity = (self.archive_rate * n as f64).round() as usize;
                        let parent = std::mem::replace(&mut pop[i], u);
                        archive.push(&mut rng, parent, capacity);
                    } else {
                        pop[i] = u;
                    }
                    fit[i] = fu;
                }
            }

            // Memory update.
            if !s_f.is_empty() {
                if let Some(mean) = weighted_lehmer_mean(&s_f, &s_df) {
                    m_f[k] = mean;
                }
                let max_cr = s_cr.iter().copied().fold(0.0_f64, f64::max);
                m_cr[k] = if m_cr[k] == TERMINAL || max_cr == 0.0 {
                    TERMINAL
                } else {
                    weighted_lehmer_mean(&s_cr, &s_df).unwrap_or(m_cr[k])
                };
                k = (k + 1) % h;
            }

            // Linear population size reduction.
            let n_next = lpsr_size(n_init, n_min, ev.used(), max_evaluations);
            if n_next < pop.len() {
                let order = sorted_indices(&fit);
                let keep = &order[..n_next];
                pop = keep.iter().map(|&j| pop[j].clone()).collect();
                fit = keep.iter().map(|&j| fit[j]).collect();
            }
            let capacity = (self.archive_rate * pop.len() as f64).round() as usize;
            archive.shrink(&mut rng, capacity);
        }
        ev.outcome(lower, upper)
    }
}
