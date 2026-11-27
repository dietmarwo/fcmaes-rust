//! Estimation-of-distribution candidate: AMaLGaM-IDEA with a full covariance
//! matrix.
//!
//! AMaLGaM (Adapted Maximum-Likelihood Gaussian Model) is a Gaussian
//! estimation-of-distribution algorithm. Every generation it selects the best
//! fraction τ of the population, estimates the maximum-likelihood mean and
//! covariance of the selected solutions, and samples new solutions from that
//! Gaussian. Two mechanisms keep the maximum-likelihood model from converging
//! prematurely:
//!
//! - **Adaptive variance scaling (AVS)** with the standard-deviation-ratio
//!   (SDR) trigger: the covariance is multiplied by a distribution multiplier
//!   `c` that grows while improvements are found far from the mean (measured
//!   in standard deviations along the principal axes of the Cholesky factor)
//!   and shrinks when no improvement is found.
//! - **Anticipated mean shift (AMS):** a fraction of the new solutions is
//!   moved by `c · δ_AMS · μ_Shift`, where `μ_Shift` is the change of the
//!   mean between subsequent generations.
//!
//! One elitist solution (the best selected solution) is carried over
//! unchanged into the next population.
//!
//! # Algorithm as implemented
//!
//! One generation, with population size `n`, `|S| = ⌊τ n⌋` and dimension `D`:
//!
//! 1. Select the `|S|` best solutions `S` (truncation selection).
//! 2. Estimate `μ(t) = mean(S)` and
//!    `Σ_ML = (1/|S|) Σ_i (S_i − μ(t))(S_i − μ(t))ᵀ`; update
//!    `Σ ← (1 − η_Σ) Σ + η_Σ Σ_ML` and
//!    `μ_Shift ← (1 − η_Shift) μ_Shift + η_Shift (μ(t) − μ(t − 1))`.
//!    For AMaLGaM (the default) `η_Σ = η_Shift = 1`.
//! 3. Factorize `Σ = L Lᵀ` (Cholesky) once per generation.
//! 4. `P_0` ← best solution of `S` (kept with its value, not re-evaluated).
//! 5. Sample `P_1 … P_{n−1}` from `N(μ, c Σ)`; move `n_AMS = ⌊α_AMS (n − 1)⌋`
//!    randomly chosen ones by `c · δ_AMS · μ_Shift`.
//! 6. Evaluate. If any `P_i` is better than `P_0`: `NIS ← 0`,
//!    `c ← max(c, 1)`, compute the average `x_avg-imp` of the improving
//!    samples and `SDR = max_i |(L⁻¹ (x_avg-imp − μ))_i|`; if
//!    `SDR > θ_SDR` then `c ← η_INC c`. Otherwise: if `c ≤ 1` then
//!    `NIS ← NIS + 1`; if `c > 1` or `NIS ≥ NIS_MAX` then `c ← η_DEC c`; if
//!    `c < 1` and `NIS < NIS_MAX` then `c ← 1`.
//! 7. A run terminates when `c < 10⁻¹⁰`.
//!
//! While `c < 1` (which is only possible after a no-improvement stretch of
//! `NIS_MAX` generations) the mean used for sampling is set to the best
//! solution, as described in Section 2.5 of the 2013 paper.
//!
//! # Published constants (defaults)
//!
//! | Field | Default | Source |
//! |---|---|---|
//! | [`tau`](Amalgam::tau) | 0.35 | Bosman et al. 2013, Sec. 4 and Alg. 2 |
//! | base population | `⌊17 + 3 D^1.5⌋` (AMaLGaM, full covariance) | 2013, Sec. 4; GECCO 2009 |
//! | [`eta_dec`](Amalgam::eta_dec) | 0.9, `η_INC = 1/η_DEC` | 2013, Alg. 2 |
//! | [`theta_sdr`](Amalgam::theta_sdr) | 1.0 | 2013, Alg. 2 |
//! | `NIS_MAX` | `25 + D` | 2013, Alg. 2 |
//! | [`delta_ams`](Amalgam::delta_ams) | 2.0 | 2013, Sec. 2.7 |
//! | `α_AMS` | `½ τ n / (n − 1)` | 2013, Sec. 2.7 and Alg. 2 |
//! | elitist solutions | 1 | 2013, Sec. 2.7 |
//! | [`min_multiplier`](Amalgam::min_multiplier) | 1e-10 | 2013, Alg. 1 |
//! | `η_Σ`, `η_Shift` | 1 (AMaLGaM); Table 1 "full" row for iAMaLGaM | 2013, Sec. 2.2 and Table 1 |
//!
//! With [`incremental`](Amalgam::incremental) set, the solver becomes
//! iAMaLGaM: `η_Σ = 1 − exp(−1.1 |S|^1.2 / D^1.6)`,
//! `η_Shift = 1 − exp(−1.2 |S|^0.31 / D^0.5)` and base population
//! `⌊10 D^0.5⌋` (2013, Table 1 and Sec. 4). The default is the
//! non-incremental AMaLGaM, because the base population `17 + 3 D^1.5` is
//! published for that variant.
//!
//! # Deviations and unspecified details
//!
//! - **Restarts.** The paper's parameter-free scheme (Algorithm 2) alternates
//!   between one run with a larger population and several interleaved
//!   parallel runs started from a clustering of the initial population. This
//!   implementation uses the simpler scheme of the benchmark protocol
//!   instead: each restart doubles the population size (IPOP style),
//!   re-initializes uniformly in the box, and the overall best solution is
//!   kept outside the runs.
//! - **Termination of a run.** Besides the published `c < 10⁻¹⁰`, a run
//!   stops when the variance of the selected objective values is at most
//!   [`fitness_variance_tolerance`](Amalgam::fitness_variance_tolerance)
//!   (default 0, i.e. only when all selected values are equal), and when the
//!   covariance matrix cannot be factorized even after regularization (all
//!   selected solutions identical, or non-finite entries). The paper names
//!   no such tolerance.
//! - **Box constraints.** The paper does not specify boundary handling.
//!   Initialization is uniform in the box. A Gaussian sample outside the box
//!   is redrawn up to
//!   [`max_resample_attempts`](Amalgam::max_resample_attempts) times, then
//!   clamped to the box. An anticipated mean shift that would leave the box
//!   is not applied to that sample (the in-box unshifted sample is used).
//! - **Mean reset timing.** The paper states that the mean is set to the
//!   best solution after a no-improvement stretch of more than `NIS_MAX`
//!   generations, without saying whether once or every generation. Here the
//!   sampling mean equals the best selected solution in every generation
//!   with `c < 1`. The covariance is still estimated around the
//!   maximum-likelihood mean, and `μ_Shift` uses maximum-likelihood means.
//! - **SDR scale.** The SDR uses the Cholesky factor of the estimated
//!   covariance without the multiplier `c`, matching the printed formula
//!   `max_i |(L⁻¹(x_avg-imp − μ))_i|`.
//! - **Rounding.** `n`, `|S|` and `n_AMS` are truncated to integers
//!   (`|S| ≥ 1`, `n ≥ 2`).
//! - **Numerical safeguard.** If the Cholesky factorization fails, a ridge of
//!   `10⁻¹²`, `10⁻¹⁰`, `10⁻⁸` and then `10⁻⁶` times the mean diagonal entry is
//!   added before giving up on the run; the paper does not discuss singular
//!   covariance matrices.
//! - **Small budgets.** If the budget is smaller than the first population,
//!   as many uniform initial points as the budget allows are evaluated and
//!   the best is returned.
//!
//! Bosman's reference C implementation was not consulted; this code is
//! written from the published descriptions only.
//!
//! # References
//!
//! - P. A. N. Bosman, J. Grahl, D. Thierens, "Benchmarking Parameter-Free
//!   AMaLGaM on Functions With and Without Noise", Evolutionary Computation
//!   21(3):445–469, 2013.
//! - P. A. N. Bosman, J. Grahl, D. Thierens, "AMaLGaM IDEAs in Noiseless
//!   Black-Box Optimization Benchmarking", GECCO 2009 companion (BBOB-2009
//!   workshop).
//! - P. A. N. Bosman, J. Grahl, D. Thierens, "Enhancing the Performance of
//!   Maximum-Likelihood Gaussian EDAs Using Anticipated Mean Shift", PPSN X,
//!   2008.

#![forbid(unsafe_code)]

use fcmaes_core::Rng;
use nalgebra::{Cholesky, DMatrix};
use solver_api::{Objective, Outcome, Solver};

/// AMaLGaM-IDEA with a full covariance matrix, adaptive variance scaling,
/// anticipated mean shift, one elitist solution and population-doubling
/// restarts. See the [module documentation](crate) for the algorithm, the
/// sources of every constant and the documented deviations.
#[derive(Clone, Debug)]
pub struct Amalgam {
    /// Truncation-selection fraction τ (published: 0.35).
    pub tau: f64,
    /// Population size of the first run; `None` uses the published base
    /// size (`⌊17 + 3 D^1.5⌋`, or `⌊10 D^0.5⌋` when
    /// [`incremental`](Self::incremental) is set).
    pub population_size: Option<usize>,
    /// Use incremental (memory) learning of covariance and mean shift with
    /// the published iAMaLGaM learning rates instead of the plain
    /// maximum-likelihood estimates (published AMaLGaM: `false`).
    pub incremental: bool,
    /// Variance-multiplier decrease factor η_DEC (published: 0.9);
    /// η_INC = 1/η_DEC.
    pub eta_dec: f64,
    /// Standard-deviation-ratio threshold θ_SDR (published: 1.0).
    pub theta_sdr: f64,
    /// Maximum no-improvement stretch; `None` uses the published `25 + D`.
    pub nis_max: Option<usize>,
    /// Anticipated-mean-shift step factor δ_AMS (published: 2.0).
    pub delta_ams: f64,
    /// Fraction of new solutions that are shifted; `None` uses the published
    /// `½ τ n / (n − 1)`.
    pub alpha_ams: Option<f64>,
    /// A run terminates when the distribution multiplier drops below this
    /// value (published: 1e-10).
    pub min_multiplier: f64,
    /// A run terminates when the variance of the selected objective values
    /// is at most this value (not in the paper; default 0).
    pub fitness_variance_tolerance: f64,
    /// Redraws of an out-of-box Gaussian sample before it is clamped (not in
    /// the paper; default 100).
    pub max_resample_attempts: usize,
    /// Population growth factor per restart (protocol: 2.0).
    pub restart_population_factor: f64,
}

impl Default for Amalgam {
    fn default() -> Self {
        Self {
            tau: 0.35,
            population_size: None,
            incremental: false,
            eta_dec: 0.9,
            theta_sdr: 1.0,
            nis_max: None,
            delta_ams: 2.0,
            alpha_ams: None,
            min_multiplier: 1e-10,
            fitness_variance_tolerance: 0.0,
            max_resample_attempts: 100,
            restart_population_factor: 2.0,
        }
    }
}

impl Amalgam {
    /// Published base population size for dimension `dim`.
    pub fn base_population_size(&self, dim: usize) -> usize {
        let d = dim as f64;
        let n = if self.incremental {
            10.0 * d.sqrt()
        } else {
            17.0 + 3.0 * d.powf(1.5)
        };
        self.population_size.unwrap_or(n as usize).max(2)
    }

    /// Learning rates `(η_Σ, η_Shift)` for `selected` solutions in dimension
    /// `dim` (1 and 1 unless [`incremental`](Self::incremental) is set).
    pub fn learning_rates(&self, selected: usize, dim: usize) -> (f64, f64) {
        if !self.incremental {
            return (1.0, 1.0);
        }
        let s = selected as f64;
        let d = dim as f64;
        let eta_sigma = 1.0 - (-1.1 * s.powf(1.2) / d.powf(1.6)).exp();
        let eta_shift = 1.0 - (-1.2 * s.powf(0.31) / d.powf(0.5)).exp();
        (eta_sigma, eta_shift)
    }
}

/// Non-finite objective values rank behind every finite value.
#[inline]
fn sanitize(y: f64) -> f64 {
    if y.is_finite() { y } else { f64::INFINITY }
}

#[inline]
fn inside(x: &[f64], lower: &[f64], upper: &[f64]) -> bool {
    x.iter()
        .zip(lower.iter().zip(upper.iter()))
        .all(|(v, (lo, hi))| *v >= *lo && *v <= *hi)
}

#[inline]
fn clamp_into(x: &mut [f64], lower: &[f64], upper: &[f64]) {
    for ((v, lo), hi) in x.iter_mut().zip(lower).zip(upper) {
        // `max`/`min` (unlike `clamp`) never panic and map NaN to a bound.
        *v = v.max(*lo).min(*hi);
    }
}

/// Evaluation bookkeeping shared by all runs of one `minimize` call.
struct Evaluator<'a> {
    objective: Objective<'a>,
    budget: u64,
    used: u64,
    best_x: Vec<f64>,
    best_y: f64,
}

impl Evaluator<'_> {
    fn exhausted(&self) -> bool {
        self.used >= self.budget
    }

    /// Evaluate `x`; returns the sanitized value. Must not be called when
    /// [`exhausted`](Self::exhausted).
    fn eval(&mut self, x: &[f64]) -> f64 {
        debug_assert!(!self.exhausted());
        self.used += 1;
        let y = sanitize((self.objective)(x));
        if y < self.best_y || self.best_x.is_empty() {
            self.best_y = y;
            self.best_x.clear();
            self.best_x.extend_from_slice(x);
        }
        y
    }
}

/// Cholesky factor of `sigma`, with a growing ridge if the plain
/// factorization fails. `None` if the matrix is degenerate or non-finite.
fn robust_cholesky(sigma: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if sigma.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if let Some(ch) = Cholesky::new(sigma.clone()) {
        return Some(ch.l());
    }
    let d = sigma.nrows();
    let mean_diag = sigma.diagonal().sum() / d as f64;
    if mean_diag.is_nan() || mean_diag <= 0.0 {
        return None;
    }
    for ridge in [1e-12, 1e-10, 1e-8, 1e-6] {
        let mut m = sigma.clone();
        for i in 0..d {
            m[(i, i)] += ridge * mean_diag;
        }
        if let Some(ch) = Cholesky::new(m) {
            return Some(ch.l());
        }
    }
    None
}

/// `out = L z` for lower-triangular `L` (column-major access).
#[inline]
fn lower_mul(l: &DMatrix<f64>, z: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for (j, zj) in z.iter().enumerate() {
        let col = l.column(j);
        for i in j..out.len() {
            out[i] += col[i] * zj;
        }
    }
}

/// Solve `L y = b` in place by forward substitution.
#[inline]
fn lower_solve(l: &DMatrix<f64>, b: &mut [f64]) {
    let d = b.len();
    for i in 0..d {
        let mut s = b[i];
        for j in 0..i {
            s -= l[(i, j)] * b[j];
        }
        b[i] = s / l[(i, i)];
    }
}

/// Settings fixed for one run.
struct RunSettings<'a> {
    lower: &'a [f64],
    upper: &'a [f64],
    n: usize,
}

impl Amalgam {
    /// One AMaLGaM run with population size `n` until its own termination
    /// criterion or until the budget is exhausted.
    fn run(&self, ev: &mut Evaluator<'_>, rng: &mut Rng, rs: &RunSettings<'_>) {
        let (lower, upper, n) = (rs.lower, rs.upper, rs.n);
        let d = lower.len();
        let sel = ((self.tau * n as f64) as usize).clamp(1, n);
        let (eta_sigma, eta_shift) = self.learning_rates(sel, d);
        let nis_max = self.nis_max.unwrap_or(25 + d);
        let alpha_ams = self
            .alpha_ams
            .unwrap_or(0.5 * self.tau * n as f64 / (n as f64 - 1.0));
        let n_ams = ((alpha_ams * (n - 1) as f64).max(0.0) as usize).min(n - 1);
        let eta_inc = 1.0 / self.eta_dec;

        // A run that cannot complete its initial population only evaluates
        // uniform points; skip allocating the full population for it.
        if n as u64 > ev.budget - ev.used {
            let mut x = vec![0.0; d];
            while !ev.exhausted() {
                for k in 0..d {
                    x[k] = lower[k] + rng.uniform01() * (upper[k] - lower[k]);
                }
                clamp_into(&mut x, lower, upper);
                ev.eval(&x);
            }
            return;
        }

        // Population, row per individual, and sanitized objective values.
        let mut pop = vec![0.0; n * d];
        let mut fit = vec![f64::INFINITY; n];
        for i in 0..n {
            if ev.exhausted() {
                return;
            }
            let x = &mut pop[i * d..(i + 1) * d];
            for k in 0..d {
                x[k] = lower[k] + rng.uniform01() * (upper[k] - lower[k]);
            }
            clamp_into(x, lower, upper);
            fit[i] = ev.eval(x);
        }

        let mut c = 1.0_f64;
        let mut nis = 0usize;
        let mut prev_mean: Option<Vec<f64>> = None;
        let mut shift = vec![0.0; d];
        let mut sigma = DMatrix::<f64>::zeros(d, d);
        let mut first = true;

        // Buffers reused across generations.
        let mut order: Vec<usize> = (0..n).collect();
        let mut centered = DMatrix::<f64>::zeros(sel, d);
        let mut sigma_ml = DMatrix::<f64>::zeros(d, d);
        let mut z = DMatrix::<f64>::zeros(d, n - 1);
        let mut y = DMatrix::<f64>::zeros(d, n - 1);
        let mut new_pop = vec![0.0; n * d];
        let mut mean = vec![0.0; d];
        let mut sample_mean = vec![0.0; d];
        let mut zbuf = vec![0.0; d];
        let mut ybuf = vec![0.0; d];
        let mut xbuf = vec![0.0; d];
        let mut avg_imp = vec![0.0; d];
        let mut idx: Vec<usize> = (1..n).collect();

        loop {
            // 1. Truncation selection (stable order for determinism).
            let compare = |&a: &usize, &b: &usize| fit[a].total_cmp(&fit[b]).then(a.cmp(&b));
            if !self.incremental && sel < order.len() {
                order.select_nth_unstable_by(sel, compare);
                order[..sel].sort_unstable_by(compare);
            } else {
                order.sort_by(compare);
            }
            let selected = &order[..sel];

            // Termination on a flat selected set.
            let fmean = selected.iter().map(|&i| fit[i]).sum::<f64>() / sel as f64;
            let fvar = if fmean.is_finite() {
                selected
                    .iter()
                    .map(|&i| (fit[i] - fmean).powi(2))
                    .sum::<f64>()
                    / sel as f64
            } else {
                f64::INFINITY
            };
            if fvar <= self.fitness_variance_tolerance {
                return;
            }

            // 2. Maximum-likelihood estimates and memory updates.
            mean.iter_mut().for_each(|v| *v = 0.0);
            for &i in selected {
                for k in 0..d {
                    mean[k] += pop[i * d + k];
                }
            }
            mean.iter_mut().for_each(|v| *v /= sel as f64);
            for (r, &i) in selected.iter().enumerate() {
                for k in 0..d {
                    centered[(r, k)] = pop[i * d + k] - mean[k];
                }
            }
            sigma_ml.gemm_tr(1.0 / sel as f64, &centered, &centered, 0.0);
            if first {
                sigma.copy_from(&sigma_ml);
            } else {
                sigma *= 1.0 - eta_sigma;
                sigma += &sigma_ml * eta_sigma;
            }
            if let Some(pm) = &prev_mean {
                for k in 0..d {
                    shift[k] = (1.0 - eta_shift) * shift[k] + eta_shift * (mean[k] - pm[k]);
                }
            }
            prev_mean = Some(mean.clone());
            first = false;

            // 3. One factorization per generation.
            let Some(l) = robust_cholesky(&sigma) else {
                return;
            };

            // 4. Elitist P_0 and the sampling mean (best solution while c < 1).
            let best = order[0];
            let best_f = fit[best];
            new_pop[..d].copy_from_slice(&pop[best * d..(best + 1) * d]);
            if c < 1.0 {
                sample_mean.copy_from_slice(&pop[best * d..(best + 1) * d]);
            } else {
                sample_mean.copy_from_slice(&mean);
            }

            // 5. Sample n - 1 solutions from N(mean, c Σ).
            let sc = c.sqrt();
            for v in z.iter_mut() {
                *v = rng.gaussian();
            }
            l.mul_to(&z, &mut y);
            for j in 0..n - 1 {
                let x = &mut new_pop[(j + 1) * d..(j + 2) * d];
                let col = y.column(j);
                for k in 0..d {
                    x[k] = sample_mean[k] + sc * col[k];
                }
                if !inside(x, lower, upper) {
                    let mut ok = false;
                    for _ in 0..self.max_resample_attempts {
                        for v in zbuf.iter_mut() {
                            *v = rng.gaussian();
                        }
                        lower_mul(&l, &zbuf, &mut ybuf);
                        for k in 0..d {
                            x[k] = sample_mean[k] + sc * ybuf[k];
                        }
                        if inside(x, lower, upper) {
                            ok = true;
                            break;
                        }
                    }
                    if !ok {
                        clamp_into(x, lower, upper);
                    }
                }
            }

            // Anticipated mean shift on n_AMS randomly chosen new solutions.
            if n_ams > 0 {
                for (pos, v) in idx.iter_mut().enumerate() {
                    *v = pos + 1;
                }
                for a in 0..n_ams {
                    let r = a + rng.int_below((n - 1 - a) as i64) as usize;
                    idx.swap(a, r.min(n - 2));
                    let i = idx[a];
                    let x = &new_pop[i * d..(i + 1) * d];
                    for k in 0..d {
                        xbuf[k] = x[k] + c * self.delta_ams * shift[k];
                    }
                    if inside(&xbuf, lower, upper) {
                        new_pop[i * d..(i + 1) * d].copy_from_slice(&xbuf);
                    }
                }
            }

            // 6. Evaluate and adapt the variance multiplier.
            std::mem::swap(&mut pop, &mut new_pop);
            fit[0] = best_f;
            let mut n_imp = 0usize;
            avg_imp.iter_mut().for_each(|v| *v = 0.0);
            for i in 1..n {
                if ev.exhausted() {
                    return;
                }
                let x = &pop[i * d..(i + 1) * d];
                fit[i] = ev.eval(x);
                if fit[i] < best_f {
                    n_imp += 1;
                    for k in 0..d {
                        avg_imp[k] += x[k];
                    }
                }
            }
            if n_imp > 0 {
                nis = 0;
                c = c.max(1.0);
                for k in 0..d {
                    avg_imp[k] = avg_imp[k] / n_imp as f64 - sample_mean[k];
                }
                lower_solve(&l, &mut avg_imp);
                let sdr = avg_imp.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                if sdr > self.theta_sdr {
                    c *= eta_inc;
                }
            } else {
                if c <= 1.0 {
                    nis += 1;
                }
                if c > 1.0 || nis >= nis_max {
                    c *= self.eta_dec;
                }
                if c < 1.0 && nis < nis_max {
                    c = 1.0;
                }
            }

            // 7. Published termination of a run.
            if c < self.min_multiplier {
                return;
            }
        }
    }
}

impl Solver for Amalgam {
    fn name(&self) -> &'static str {
        "amalgam"
    }

    fn minimize(
        &self,
        objective: Objective<'_>,
        lower: &[f64],
        upper: &[f64],
        max_evaluations: u64,
        seed: u64,
    ) -> Outcome {
        let d = lower.len().min(upper.len());
        let (lower, upper) = (&lower[..d], &upper[..d]);
        let mut rng = Rng::new(seed);
        let mut ev = Evaluator {
            objective,
            budget: max_evaluations,
            used: 0,
            best_x: Vec::new(),
            best_y: f64::INFINITY,
        };
        if d > 0 {
            let mut n = self.base_population_size(d) as f64;
            while !ev.exhausted() {
                let rs = RunSettings {
                    lower,
                    upper,
                    n: (n as usize).max(2),
                };
                self.run(&mut ev, &mut rng, &rs);
                n = (n * self.restart_population_factor).min(1e9);
            }
        }
        let x = if ev.best_x.is_empty() {
            lower
                .iter()
                .zip(upper)
                .map(|(lo, hi)| 0.5 * (lo + hi))
                .collect()
        } else {
            ev.best_x
        };
        Outcome {
            x,
            y: ev.best_y,
            evaluations: ev.used,
        }
    }
}
