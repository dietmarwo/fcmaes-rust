//! jSO: L-SHADE/iL-SHADE descendant with weighted current-to-pbest mutation
//! and staged parameter limits.
//!
//! Reference: J. Brest, M. S. Maučec, B. Bošković, "Single objective
//! real-parameter optimization: algorithm jSO", Proc. IEEE Congress on
//! Evolutionary Computation (CEC 2017), pp. 1311–1318. Details the paper
//! leaves open follow the authors' posted CEC 2017 code as reported by an
//! independent review (marked below).
//!
//! # Algorithm (differences from L-SHADE)
//!
//! - `N_init = round(25 · ln(D) · √D)`, `N_min = 4`, linear reduction.
//! - Memory of `H = 5` cells, initialized to `M_F = 0.3`, `M_CR = 0.8`.
//!   Sampling the last cell always yields `M_F = M_CR = 0.9`.
//! - `p = 0.25 · (1 - 0.5 · nfes / max_nfes)`, decreasing linearly from
//!   `p_max = 0.25` to `p_min = 0.125`; archive capacity `round(1.0 · N)`.
//! - Mutation current-to-pbest-w/1 with archive:
//!   `v = x_i + Fw (x_pbest - x_i) + F (x_r1 - x_r2)`, with `Fw = 0.7 F` while
//!   `nfes < 0.2 max_nfes`, `0.8 F` while `nfes < 0.4 max_nfes`, else `1.2 F`.
//!   While `nfes < 0.5 max_nfes`, `pbest` is redrawn until it differs from `i`.
//! - `F` is capped at 0.7 while `nfes < 0.6 max_nfes`; `CR = max(CR, 0.7)`
//!   while `nfes < 0.25 max_nfes`, `CR = max(CR, 0.6)` while
//!   `nfes < 0.5 max_nfes`.
//! - Memory update of cell `k` (cyclic over all `H` cells):
//!   `M_F[k] = (M_F[k] + mean_WL(S_F)) / 2` and
//!   `M_CR[k] = (M_CR[k] + c) / 2` with `c = -1` if `max(S_CR) = 0`, otherwise
//!   `c = mean_WL(S_CR)`. A negative `M_CR` cell samples `CR = 0`.
//!
//! # Deviations and unspecified details
//!
//! - **Fixed cell inside `H`.** The paper sets `M_F,H = M_CR,H = 0.9` for the
//!   last of the `H = 5` cells; it is not an additional sixth cell. As in the
//!   authors' code, the cyclic update index runs over all `H` cells, so an
//!   update that lands on the last cell has no effect.
//! - **Initial `M_F`.** Section IV-B of the paper gives 0.3, Algorithm 1
//!   line 3 gives 0.5; 0.3 (the stated parameter setting) is used.
//! - **Direction of `p`.** The prose and the authors' code decrease `p` from
//!   `p_max` to `p_min`; only the printed Eq. (1) increases. The decreasing
//!   schedule is used.
//! - **`pbest` different from `i`.** Not in the paper; from the authors' code.
//! - **Terminal `CR`.** Algorithm 1 sets `CR = 0` for a negative memory value.
//!   The terminal value `-1` is averaged into the cell like any other mean
//!   (authors' code), so it is not sticky, unlike L-SHADE.
//! - **Stage clock.** Algorithm 1 phrases the `F` and `CR` stages with the
//!   generation ratio `g / G_max`, which is undefined under population size
//!   reduction; the evaluation ratio `nfes / max_nfes` is used, as for `Fw`.
//!   All stage decisions use the evaluation count at the start of the
//!   generation.
//! - **Bound handling.** The paper only says a repair is applied; the
//!   midpoint-target repair of L-SHADE (towards the parent) is used.
//! - **Archive rate and `N_min`.** Not stated in the paper ("kept unchanged"
//!   from L-SHADE/iL-SHADE); 1.0 and 4 as fixed in the benchmark protocol.
//! - **Recorded values.** `S_F` stores the capped `F` (not `Fw`), `S_CR` the
//!   floored `CR`.
//! - Budget, non-finite values and partial generations are handled as in
//!   [`LShade`](crate::LShade).

use fcmaes_core::Rng;
use solver_api::{Objective, Outcome, Solver};

use crate::common::{
    Archive, Evaluator, binomial_crossover, check_box, lpsr_size, pbest_count, rand_index,
    repair_midpoint, sample_clipped_normal, sample_f_cauchy, sorted_indices, uniform_point,
    weighted_lehmer_mean,
};

/// jSO with the published parameter values as defaults.
#[derive(Clone, Debug)]
pub struct Jso {
    /// Initial population factor: `N_init = round(n_init_factor · ln(D) · √D)` (25).
    pub n_init_factor: f64,
    /// Final population size `N_min` (4).
    pub n_min: usize,
    /// Memory size `H` including the fixed last cell (5).
    pub memory_size: usize,
    /// Initial value of the `M_F` cells (0.3).
    pub memory_f_init: f64,
    /// Initial value of the `M_CR` cells (0.8).
    pub memory_cr_init: f64,
    /// Value the last memory cell yields for both `F` and `CR` (0.9).
    pub memory_fixed: f64,
    /// Initial p-best fraction `p_max` (0.25).
    pub p_max: f64,
    /// Final p-best fraction `p_min` (0.125).
    pub p_min: f64,
    /// Archive capacity relative to the current population size (1.0).
    pub archive_rate: f64,
    /// Scale of the Cauchy distribution for `F` (0.1).
    pub f_scale: f64,
    /// Standard deviation of the normal distribution for `CR` (0.1).
    pub cr_sigma: f64,
}

impl Default for Jso {
    fn default() -> Self {
        Self {
            n_init_factor: 25.0,
            n_min: 4,
            memory_size: 5,
            memory_f_init: 0.3,
            memory_cr_init: 0.8,
            memory_fixed: 0.9,
            p_max: 0.25,
            p_min: 0.125,
            archive_rate: 1.0,
            f_scale: 0.1,
            cr_sigma: 0.1,
        }
    }
}

/// Value averaged into `M_CR` when every recorded `CR` is zero.
const TERMINAL: f64 = -1.0;

/// Memory values `(M_F, M_CR)` for sampled cell `r`; the last cell is fixed.
fn memory_cell(m_f: &[f64], m_cr: &[f64], r: usize, fixed: f64) -> (f64, f64) {
    if r + 1 == m_f.len() {
        (fixed, fixed)
    } else {
        (m_f[r], m_cr[r])
    }
}

/// New `M_CR` cell value: the old value averaged with the weighted Lehmer
/// mean of `s_cr`, or with the terminal value `-1` if `max(S_CR) = 0`.
fn updated_cr_cell(old: f64, s_cr: &[f64], s_df: &[f64]) -> f64 {
    let max_cr = s_cr.iter().copied().fold(0.0_f64, f64::max);
    let target = if max_cr == 0.0 {
        TERMINAL
    } else {
        match weighted_lehmer_mean(s_cr, s_df) {
            Some(mean) => mean,
            None => return old,
        }
    };
    0.5 * (old + target)
}

/// Uniform index among the `n_pbest` best (`order` is best first); while
/// `redraw_self` holds it is redrawn until it differs from `i`.
fn draw_pbest(
    rng: &mut Rng,
    order: &[usize],
    n_pbest: usize,
    i: usize,
    redraw_self: bool,
) -> usize {
    loop {
        let c = order[rand_index(rng, n_pbest)];
        if !(redraw_self && c == i && n_pbest > 1) {
            return c;
        }
    }
}

impl Solver for Jso {
    fn name(&self) -> &'static str {
        "jso"
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
        let d = dim.max(1) as f64;
        let n_init = ((self.n_init_factor * d.ln() * d.sqrt()).round() as usize).max(n_min);
        let h = self.memory_size.max(2);

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
        let max_fes = max_evaluations as f64;

        while !ev.exhausted() {
            let n = pop.len();
            let progress = (ev.used() as f64 / max_fes).min(1.0);
            let p = self.p_max - (self.p_max - self.p_min) * progress;
            let order = sorted_indices(&fit);
            let n_pbest = pbest_count(p, n, 2);
            let fw_factor = if progress < 0.2 {
                0.7
            } else if progress < 0.4 {
                0.8
            } else {
                1.2
            };

            let mut trials: Vec<(Vec<f64>, f64, f64)> = Vec::with_capacity(n);
            for i in 0..n {
                let r = rand_index(&mut rng, h);
                let (mean_f, mean_cr) = memory_cell(&m_f, &m_cr, r, self.memory_fixed);
                let mut cr = if mean_cr < 0.0 {
                    0.0
                } else {
                    sample_clipped_normal(&mut rng, mean_cr, self.cr_sigma)
                };
                if progress < 0.25 {
                    cr = cr.max(0.7);
                } else if progress < 0.5 {
                    cr = cr.max(0.6);
                }
                let mut f = sample_f_cauchy(&mut rng, mean_f, self.f_scale);
                if progress < 0.6 && f > 0.7 {
                    f = 0.7;
                }
                let fw = fw_factor * f;

                let pbest = draw_pbest(&mut rng, &order, n_pbest, i, progress < 0.5);
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
                    .map(|j| xi[j] + fw * (pop[pbest][j] - xi[j]) + f * (pop[r1][j] - x_r2[j]))
                    .collect();
                repair_midpoint(&mut v, xi, lower, upper);
                let mut u = vec![0.0; dim];
                binomial_crossover(&mut rng, &v, xi, cr, &mut u);
                trials.push((u, f, cr));
            }

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

            if !s_f.is_empty() {
                if let Some(mean) = weighted_lehmer_mean(&s_f, &s_df) {
                    m_f[k] = 0.5 * (m_f[k] + mean);
                }
                m_cr[k] = updated_cr_cell(m_cr[k], &s_cr, &s_df);
                k = (k + 1) % h;
            }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_cell_is_fixed_and_others_are_read() {
        let m_f = [0.1, 0.2, 0.3, 0.4, 0.5];
        let m_cr = [0.6, 0.7, 0.8, 0.9, -1.0];
        assert_eq!(memory_cell(&m_f, &m_cr, 4, 0.9), (0.9, 0.9));
        assert_eq!(memory_cell(&m_f, &m_cr, 1, 0.9), (0.2, 0.7));
    }

    #[test]
    fn terminal_cr_is_averaged_not_sticky() {
        // All recorded CR zero: average with -1.
        assert!((updated_cr_cell(0.6, &[0.0, 0.0], &[1.0, 2.0]) - (-0.2)).abs() < 1e-12);
        // A negative cell recovers through averaging with positive means.
        let once = updated_cr_cell(-1.0, &[0.8], &[1.0]);
        assert!((once - (-0.1)).abs() < 1e-12);
        let twice = updated_cr_cell(once, &[0.8], &[1.0]);
        assert!((twice - 0.35).abs() < 1e-12);
    }

    #[test]
    fn pbest_avoids_target_only_when_asked() {
        let mut rng = Rng::new(4);
        let order = [7, 3, 0, 1];
        for _ in 0..1000 {
            assert_ne!(draw_pbest(&mut rng, &order, 2, 7, true), 7);
        }
        let hits = (0..1000)
            .filter(|_| draw_pbest(&mut rng, &order, 2, 7, false) == 7)
            .count();
        assert!(hits > 300);
    }
}
