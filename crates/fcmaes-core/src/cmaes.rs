//! Active CMA-ES — Rust port of the C++ `acmaesoptimizer.cpp`.
//!
//! Faithful translation of the Eigen-based active CMA-ES (covariance matrix
//! adaptation with the negative/active rank-mu update). Matrix algebra uses
//! `nalgebra`; the self-adjoint eigensolver replaces Eigen's
//! `SelfAdjointEigenSolver`. Parity with the reference is statistical, so the
//! RNG stream is not matched bit-for-bit.
//!
//! The single collapsed implementation replaces both the C++ optimizer and the
//! pure-Python `fcmaes/cmaes.py`.

use nalgebra::{DMatrix, DVector};

use crate::fitness::{Fitness, Objective};
use crate::rng::Rng;

/// Outcome of a CMA-ES run (mirrors the C++ `AcmaResult`).
#[derive(Clone, Debug)]
pub struct AcmaResult {
    pub x: Vec<f64>,
    pub y: f64,
    pub evaluations: u64,
    pub iterations: i32,
    pub stop: i32,
}

/// Tunable inputs for [`Cmaes::new`]. `popsize`/`mu` <= 0 select CMA defaults.
#[derive(Clone, Debug)]
pub struct CmaesParams {
    pub popsize: i32,
    pub mu: i32,
    pub max_evaluations: u64,
    pub accuracy: f64,
    pub stop_fitness: f64,
    pub stop_tol_hist_fun: f64,
    pub update_gap: i32,
    pub seed: u64,
    pub runid: i64,
}

impl Default for CmaesParams {
    fn default() -> Self {
        Self {
            popsize: 31,
            mu: 0,
            max_evaluations: 100_000,
            accuracy: 1.0,
            stop_fitness: f64::NEG_INFINITY,
            stop_tol_hist_fun: -1.0,
            update_gap: -1,
            seed: 0,
            runid: 0,
        }
    }
}

/// Ascending order of indices by value (the C++ `sort_index`).
fn sort_index(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// Inverse permutation: `inv[perm[i]] = i` (the C++ `inverse`).
fn inverse_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Column of `m` as an owned `DVector`.
fn col(m: &DMatrix<f64>, j: usize) -> DVector<f64> {
    DVector::from_iterator(m.nrows(), m.column(j).iter().copied())
}

pub struct Cmaes {
    fitfun: Fitness,
    rng: Rng,
    dim: usize,
    popsize: usize,
    mu: usize,

    // termination
    max_evaluations: u64,
    accuracy: f64,
    stopfitness: f64,
    stop_tol_upx: f64,
    stop_tol_x: f64,
    stop_tol_fun: f64,
    stop_tol_hist_fun: f64,

    // strategy parameters / constants
    weights: DVector<f64>,
    mueff: f64,
    sigma: f64,
    cc: f64,
    cs: f64,
    damps: f64,
    ccov1: f64,
    ccovmu: f64,
    chi_n: f64,
    lazy_update_gap: f64,

    // dynamic state
    xmean: DVector<f64>,
    pc: DVector<f64>,
    ps: DVector<f64>,
    normps: f64,
    b: DMatrix<f64>,
    bd: DMatrix<f64>,
    diag_d: DVector<f64>,
    diag_c: DVector<f64>,
    c: DMatrix<f64>,
    arz: DMatrix<f64>,
    arx: DMatrix<f64>,
    fitness: DVector<f64>,

    iterations: i32,
    last_update: i32,
    fitness_history: Vec<f64>,
    best_value: f64,
    best_x: Vec<f64>,
    stop: i32,
    told: usize,
    compute_arz: bool,

    // ask/tell bookkeeping (mirrors AcmaState::Impl)
    asked_x: DMatrix<f64>,
    population_decoded: DMatrix<f64>,
    external_evaluations: u64,
}

impl Cmaes {
    /// Build a CMA-ES optimizer. `guess` is the initial mean, `input_sigma`
    /// the per-coordinate step sizes (length 1 broadcasts to `dim`).
    pub fn new(mut fitfun: Fitness, guess: &[f64], input_sigma: &[f64], p: &CmaesParams) -> Self {
        let dim = guess.len();
        assert_eq!(fitfun.dim(), dim, "fitness dim must match guess");
        fitfun.reset_evaluations();

        let popsize = if p.popsize > 0 {
            p.popsize as usize
        } else {
            (4.0 + 3.0 * (dim as f64).ln()).floor() as usize
        };
        let input_sigma: DVector<f64> = if input_sigma.len() == 1 {
            DVector::from_element(dim, input_sigma[0])
        } else {
            DVector::from_row_slice(input_sigma)
        };
        let sigma = input_sigma.max();

        let mu = if p.mu > 0 { p.mu as usize } else { popsize / 2 };

        // weights for weighted recombination
        let mut weights = DVector::from_iterator(
            mu,
            (1..=mu).map(|i| -((i as f64).ln()) + (mu as f64 + 0.5).ln()),
        );
        let sumw = weights.sum();
        let sumwq = weights.dot(&weights);
        weights /= sumw;
        let mueff = sumw * sumw / sumwq;

        let dimf = dim as f64;
        let cc = (4.0 + mueff / dimf) / (dimf + 4.0 + 2.0 * mueff / dimf);
        let cs = (mueff + 2.0) / (dimf + mueff + 3.0);
        let damps = (1.0 + 2.0 * f64::max(0.0, ((mueff - 1.0) / (dimf + 1.0)).sqrt() - 1.0))
            * f64::max(
                0.3,
                1.0 - dimf / (1e-6 + (p.max_evaluations / popsize as u64) as f64),
            )
            + cs;
        let ccov1 = 2.0 / ((dimf + 1.3) * (dimf + 1.3) + mueff);
        let ccovmu = f64::min(
            1.0 - ccov1,
            2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dimf + 2.0) * (dimf + 2.0) + mueff),
        );
        let chi_n = dimf.sqrt() * (1.0 - 1.0 / (4.0 * dimf) + 1.0 / (21.0 * dimf * dimf));
        let lazy_update_gap = if p.update_gap >= 0 {
            p.update_gap as f64
        } else {
            1.0 / (ccov1 + ccovmu + 1e-23) / dimf / 10.0
        };

        let xmean = DVector::from_vec(fitfun.encode(guess));
        let diag_d = &input_sigma / sigma;
        let diag_c = diag_d.component_mul(&diag_d);
        let b = DMatrix::identity(dim, dim);
        // BD = B * diag(diag_d): column j of B scaled by diag_d[j].
        let mut bd = b.clone();
        for j in 0..dim {
            for i in 0..dim {
                bd[(i, j)] *= diag_d[j];
            }
        }
        let c = &b * b.transpose();

        let history_size = 10 + (30.0 * dimf / popsize as f64) as usize;
        let best_value = f64::MAX;
        let mut fitness_history = vec![f64::MAX; history_size];
        fitness_history[0] = best_value;

        let stop_tol_hist_fun = if p.stop_tol_hist_fun < 0.0 {
            1e-13 * p.accuracy
        } else {
            p.stop_tol_hist_fun
        };

        Cmaes {
            dim,
            popsize,
            mu,
            max_evaluations: p.max_evaluations,
            accuracy: p.accuracy,
            stopfitness: p.stop_fitness,
            stop_tol_upx: 1e3 * sigma,
            stop_tol_x: 1e-11 * sigma * p.accuracy,
            stop_tol_fun: 1e-12 * p.accuracy,
            stop_tol_hist_fun,
            weights,
            mueff,
            sigma,
            cc,
            cs,
            damps,
            ccov1,
            ccovmu,
            chi_n,
            lazy_update_gap,
            xmean,
            pc: DVector::zeros(dim),
            ps: DVector::zeros(dim),
            normps: 0.0,
            b,
            bd,
            diag_d,
            diag_c,
            c,
            arz: DMatrix::zeros(dim, popsize),
            arx: DMatrix::zeros(dim, popsize),
            fitness: DVector::from_element(popsize, 0.0),
            iterations: 1,
            last_update: 0,
            fitness_history,
            best_value,
            best_x: guess.to_vec(),
            stop: 0,
            told: 0,
            compute_arz: true,
            asked_x: DMatrix::zeros(dim, popsize),
            population_decoded: DMatrix::zeros(dim, popsize),
            external_evaluations: 0,
            rng: Rng::new(p.seed.wrapping_add(p.runid as u64)),
            fitfun,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn popsize(&self) -> usize {
        self.popsize
    }
    pub fn stop(&self) -> i32 {
        self.stop
    }

    fn normal_matrix(&mut self, rows: usize, cols: usize) -> DMatrix<f64> {
        // Column-major fill so the ask ordering is deterministic per seed.
        let mut m = DMatrix::zeros(rows, cols);
        for j in 0..cols {
            for i in 0..rows {
                m[(i, j)] = self.rng.gaussian();
            }
        }
        m
    }

    /// Sample `popsize` offspring; returns the encoded (normed) columns.
    fn ask_all(&mut self) -> DMatrix<f64> {
        self.arz = self.normal_matrix(self.dim, self.popsize);
        let mut xs = DMatrix::zeros(self.dim, self.popsize);
        for k in 0..self.popsize {
            let delta = (&self.bd * col(&self.arz, k)) * self.sigma;
            let cand = &self.xmean + delta;
            let feasible = self.fitfun.closest_feasible_normed(cand.as_slice());
            for i in 0..self.dim {
                xs[(i, k)] = feasible[i];
            }
        }
        self.compute_arz = false;
        xs
    }

    fn tell_one(&mut self, y: f64, x: &DVector<f64>) -> i32 {
        self.fitness[self.told] = if y.is_finite() { y } else { f64::MAX };
        self.arx.set_column(self.told, x);
        self.told += 1;
        if self.told >= self.popsize {
            let feasible = self.fitfun.closest_feasible_normed(self.xmean.as_slice());
            self.xmean = DVector::from_vec(feasible);
            if self.compute_arz {
                // arz = BD^{-1} * ((arx - xmean) / sigma)
                let mut diff = self.arx.clone();
                for k in 0..self.popsize {
                    for i in 0..self.dim {
                        diff[(i, k)] = (diff[(i, k)] - self.xmean[i]) / self.sigma;
                    }
                }
                self.arz = match self.bd.clone().try_inverse() {
                    Some(inv) => inv * diff,
                    None => self.normal_matrix(self.dim, self.popsize),
                };
            }
            self.update_cma();
            self.told = 0;
            self.iterations += 1;
        }
        self.stop
    }

    fn update_evolution_paths(&mut self, zmean: &DVector<f64>, xold: &DVector<f64>) -> bool {
        self.ps = &self.ps * (1.0 - self.cs)
            + (&self.b * zmean) * (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        self.normps = self.ps.norm();
        let hsig = self.normps
            / (1.0 - (1.0 - self.cs).powf(2.0 * self.iterations as f64)).sqrt()
            / self.chi_n
            < 1.4 + 2.0 / (self.dim as f64 + 1.0);
        self.pc *= 1.0 - self.cc;
        if hsig {
            self.pc += (&self.xmean - xold)
                * ((self.cc * (2.0 - self.cc) * self.mueff).sqrt() / self.sigma);
        }
        hsig
    }

    fn update_covariance(
        &mut self,
        hsig: bool,
        best_arx: &DMatrix<f64>,
        arindex: &[usize],
        xold: &DVector<f64>,
    ) -> f64 {
        let mut negccov = 0.0;
        if self.ccov1 + self.ccovmu > 0.0 {
            let dimf = self.dim as f64;
            // mu difference vectors: arpos = (best_arx - xold) / sigma
            let mut arpos = best_arx.clone();
            for k in 0..self.mu {
                for i in 0..self.dim {
                    arpos[(i, k)] = (arpos[(i, k)] - xold[i]) / self.sigma;
                }
            }
            let roneu = (&self.pc * self.pc.transpose()) * self.ccov1;
            let mut old_fac = if hsig {
                0.0
            } else {
                self.ccov1 * self.cc * (2.0 - self.cc)
            };
            old_fac += 1.0 - self.ccov1 - self.ccovmu;

            negccov = (1.0 - self.ccovmu) * 0.25 * self.mueff
                / ((dimf + 2.0).powf(1.5) + 2.0 * self.mueff);
            let negminresidualvariance = 0.66;
            let negalphaold = 0.5;

            // worst-mu columns of arz (arindex reversed, first mu)
            let rev: Vec<usize> = arindex.iter().rev().copied().collect();
            let worst_mu = &rev[0..self.mu];
            let mut arzneg = DMatrix::zeros(self.dim, self.mu);
            for (k, &src) in worst_mu.iter().enumerate() {
                arzneg.set_column(k, &col(&self.arz, src));
            }
            let arnorms: Vec<f64> = (0..self.mu).map(|k| arzneg.column(k).norm()).collect();
            let idxnorms = sort_index(&arnorms);
            let idx_reverse: Vec<usize> = idxnorms.iter().rev().copied().collect();
            // arnorms = arnorms[reverse] / arnorms[sorted]
            let ratio: Vec<f64> = (0..self.mu)
                .map(|k| arnorms[idx_reverse[k]] / arnorms[idxnorms[k]])
                .collect();
            let inv_idx = inverse_perm(&idxnorms);
            let arnorms_inv: Vec<f64> = (0..self.mu).map(|k| ratio[inv_idx[k]]).collect();

            // sqarnw = (arnorms_inv^2) . weights
            let sqarnw: f64 = (0..self.mu)
                .map(|k| arnorms_inv[k] * arnorms_inv[k] * self.weights[k])
                .sum();
            let negcov_max = (1.0 - negminresidualvariance) / sqarnw;
            if negccov > negcov_max {
                negccov = negcov_max;
            }
            // scale each column k of arzneg by arnorms_inv[k]
            for k in 0..self.mu {
                let s = arnorms_inv[k];
                for i in 0..self.dim {
                    arzneg[(i, k)] *= s;
                }
            }
            let artmp = &self.bd * &arzneg;
            let w = DMatrix::from_diagonal(&self.weights);
            let cneg = &artmp * &w * artmp.transpose();

            old_fac += negalphaold * negccov;
            let rankmu = &arpos * &w * arpos.transpose();
            let s = self.ccovmu + (1.0 - negalphaold) * negccov;
            self.c = &self.c * old_fac + roneu + rankmu * s - cneg * negccov;
        }
        negccov
    }

    fn update_bd(&mut self, negccov: f64) {
        let denom = self.ccov1 + self.ccovmu + negccov;
        if denom > 0.0 && (self.iterations as f64) % (1.0 / denom / self.dim as f64 / 10.0) < 1.0 {
            // enforce symmetry
            let mut sym = self.c.clone();
            for i in 0..self.dim {
                for j in (i + 1)..self.dim {
                    let avg = 0.5 * (sym[(i, j)] + sym[(j, i)]);
                    sym[(i, j)] = avg;
                    sym[(j, i)] = avg;
                }
            }
            self.c = sym.clone();
            let eig = sym.symmetric_eigen();
            let mut diag_d = eig.eigenvalues.clone();
            self.b = eig.eigenvectors;
            if diag_d.min() <= 0.0 {
                for i in 0..self.dim {
                    if diag_d[i] < 0.0 {
                        diag_d[i] = 0.0;
                    }
                }
                let tfac = diag_d.max() / 1e14;
                for i in 0..self.dim {
                    self.c[(i, i)] += tfac;
                    diag_d[i] += tfac;
                }
            }
            if diag_d.max() > 1e14 * diag_d.min() {
                let tfac = diag_d.max() / 1e14 - diag_d.min();
                for i in 0..self.dim {
                    self.c[(i, i)] += tfac;
                    diag_d[i] += tfac;
                }
            }
            self.diag_c = self.c.diagonal();
            diag_d = diag_d.map(|v| v.sqrt());
            self.diag_d = diag_d;
            // BD = B * diag(diag_d)
            let mut bd = self.b.clone();
            for j in 0..self.dim {
                for i in 0..self.dim {
                    bd[(i, j)] *= self.diag_d[j];
                }
            }
            self.bd = bd;
        }
    }

    fn update_cma(&mut self) {
        let arindex = sort_index(self.fitness.as_slice());
        let xold = self.xmean.clone();
        let best_index = &arindex[0..self.mu];

        let mut best_arx = DMatrix::zeros(self.dim, self.mu);
        let mut best_arz = DMatrix::zeros(self.dim, self.mu);
        for (k, &idx) in best_index.iter().enumerate() {
            best_arx.set_column(k, &col(&self.arx, idx));
            best_arz.set_column(k, &col(&self.arz, idx));
        }
        self.xmean = &best_arx * &self.weights;
        let zmean = &best_arz * &self.weights;
        let hsig = self.update_evolution_paths(&zmean, &xold);

        self.sigma *= f64::exp(f64::min(
            1.0,
            (self.normps / self.chi_n - 1.0) * self.cs / self.damps,
        ));
        let best_fitness = self.fitness[arindex[0]];
        let worst_fitness = self.fitness[arindex[self.popsize - 1]];
        if self.best_value > best_fitness {
            self.best_value = best_fitness;
            self.best_x = self.fitfun.decode(col(&best_arx, 0).as_slice());
            if self.stopfitness.is_finite() && best_fitness < self.stopfitness {
                self.stop = 1;
                return;
            }
        }
        if self.iterations >= self.last_update + self.lazy_update_gap as i32 {
            self.last_update = self.iterations;
            let negccov = self.update_covariance(hsig, &best_arx, &arindex, &xold);
            self.update_bd(negccov);
            let sqrt_diag_c = self.diag_c.map(|v| v.sqrt());
            let mut all_below = true;
            for i in 0..self.dim {
                if self.sigma * f64::max(self.pc[i].abs(), sqrt_diag_c[i]) > self.stop_tol_x {
                    all_below = false;
                    break;
                }
            }
            if all_below {
                self.stop = 2;
                return;
            }
            for i in 0..self.dim {
                if self.sigma * sqrt_diag_c[i] > self.stop_tol_upx {
                    self.stop = 3;
                }
            }
            if self.stop > 0 {
                return;
            }
        }
        let history_best = self
            .fitness_history
            .iter()
            .cloned()
            .fold(f64::MAX, f64::min);
        let history_worst = self
            .fitness_history
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        if self.iterations > 2
            && f64::max(history_worst, worst_fitness) - f64::min(history_best, best_fitness)
                < self.stop_tol_fun
        {
            self.stop = 4;
            return;
        }
        if self.iterations as usize > self.fitness_history.len()
            && history_worst - history_best < self.stop_tol_hist_fun
        {
            self.stop = 5;
            return;
        }
        if self.diag_d.max() / self.diag_d.min() > 1e7 / self.accuracy.sqrt() {
            self.stop = 6;
            return;
        }
        // flat-fitness step-size adjustments
        let flat_idx = arindex[self.popsize / 4];
        if self.best_value == self.fitness[flat_idx] {
            self.sigma *= f64::exp(0.2 + self.cs / self.damps);
        }
        if self.iterations > 2
            && f64::max(history_worst, best_fitness) - f64::min(history_best, best_fitness) == 0.0
        {
            self.sigma *= f64::exp(0.2 + self.cs / self.damps);
        }
        // shift history
        for i in (1..self.fitness_history.len()).rev() {
            self.fitness_history[i] = self.fitness_history[i - 1];
        }
        self.fitness_history[0] = best_fitness;
    }

    /// Run the full generational loop (the C++ `doOptimize`), evaluating each
    /// generation through `obj` (parallelized per `workers`).
    pub fn optimize(&mut self, obj: &impl Objective, workers: i32) -> AcmaResult {
        self.iterations = 0;
        self.fitfun.reset_evaluations();
        while self.fitfun.evaluations() < self.max_evaluations && !self.fitfun.terminate() {
            let xs = self.ask_all();
            let ys = self
                .fitfun
                .eval_population_scalar_flat(xs.as_slice(), self.dim, obj, workers);
            self.told = 0;
            for (k, &y) in ys.iter().enumerate() {
                if self.stop != 0 {
                    break;
                }
                self.tell_one(y, &col(&xs, k));
            }
            if self.stop != 0 {
                break;
            }
        }
        self.make_result(self.fitfun.evaluations())
    }

    fn make_result(&self, evaluations: u64) -> AcmaResult {
        AcmaResult {
            x: self.best_x.clone(),
            y: self.best_value,
            evaluations,
            iterations: self.iterations,
            stop: self.stop,
        }
    }

    // ---- ask/tell interface (mirrors AcmaState::Impl) ----

    fn decode_population(&self, encoded: &DMatrix<f64>) -> DMatrix<f64> {
        let mut decoded = DMatrix::zeros(self.dim, self.popsize);
        for p in 0..self.popsize {
            let feasible = self
                .fitfun
                .closest_feasible_normed(col(encoded, p).as_slice());
            let dec = self.fitfun.decode(&feasible);
            for i in 0..self.dim {
                decoded[(i, p)] = dec[i];
            }
        }
        decoded
    }

    fn encode_population(&self, decoded: &DMatrix<f64>) -> DMatrix<f64> {
        let mut encoded = DMatrix::zeros(self.dim, self.popsize);
        for p in 0..self.popsize {
            let enc = self.fitfun.encode(col(decoded, p).as_slice());
            for i in 0..self.dim {
                encoded[(i, p)] = enc[i];
            }
        }
        encoded
    }

    /// Ask for a decoded population (rows = individuals).
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        self.asked_x = self.ask_all();
        self.population_decoded = self.decode_population(&self.asked_x);
        (0..self.popsize)
            .map(|p| col(&self.population_decoded, p).as_slice().to_vec())
            .collect()
    }

    /// Tell fitness values for the population returned by [`ask`](Cmaes::ask).
    pub fn tell(&mut self, ys: &[f64]) -> i32 {
        self.told = 0;
        let asked = self.asked_x.clone();
        for (p, &y) in ys.iter().enumerate() {
            self.tell_one(y, &col(&asked, p));
        }
        self.compute_arz = false;
        self.external_evaluations += ys.len() as u64;
        self.stop
    }

    /// Tell fitness values for an externally supplied decoded population.
    pub fn tell_x(&mut self, ys: &[f64], xs_decoded: &[Vec<f64>]) -> i32 {
        let mut decoded = DMatrix::zeros(self.dim, self.popsize);
        for (p, row) in xs_decoded.iter().enumerate() {
            for i in 0..self.dim {
                decoded[(i, p)] = row[i];
            }
        }
        self.asked_x = self.encode_population(&decoded);
        self.population_decoded = decoded;
        self.told = 0;
        let asked = self.asked_x.clone();
        for (p, &y) in ys.iter().enumerate() {
            self.tell_one(y, &col(&asked, p));
        }
        self.compute_arz = true;
        self.external_evaluations += ys.len() as u64;
        self.stop
    }

    /// Current decoded population (rows = individuals).
    pub fn population(&self) -> Vec<Vec<f64>> {
        (0..self.popsize)
            .map(|p| col(&self.population_decoded, p).as_slice().to_vec())
            .collect()
    }

    /// Result snapshot for the ask/tell interface.
    pub fn result(&self) -> AcmaResult {
        self.make_result(self.external_evaluations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(name: &str, obj: impl Objective, dim: usize, lo: f64, hi: f64, seed: u64) -> f64 {
        let fit = Fitness::bounded(dim, 1, &vec![lo; dim], &vec![hi; dim]);
        let mut fit = fit;
        fit.set_normalize(true);
        let guess = vec![0.5 * (lo + hi); dim];
        let params = CmaesParams {
            popsize: 31,
            max_evaluations: 5000,
            seed,
            ..Default::default()
        };
        let mut opt = Cmaes::new(fit, &guess, &[0.3], &params);
        let r = opt.optimize(&obj, 1);
        assert!(r.evaluations > 0, "{name} made no evaluations");
        r.y
    }

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn rosen(x: &[f64]) -> f64 {
        (0..x.len() - 1)
            .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
            .sum()
    }

    #[test]
    fn minimizes_sphere() {
        let y = run("sphere", sphere as fn(&[f64]) -> f64, 5, -5.0, 5.0, 1);
        assert!(y < 1e-6, "sphere not solved: {y}");
    }

    #[test]
    fn minimizes_rosenbrock() {
        // Median over a few seeds should be small; Rosenbrock is harder.
        let mut vals: Vec<f64> = (0..5)
            .map(|s| run("rosen", rosen as fn(&[f64]) -> f64, 5, -5.0, 5.0, s))
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(vals[2] < 1.0, "rosenbrock median too large: {:?}", vals);
    }

    #[test]
    fn ask_tell_matches_optimize_shape() {
        let fit = {
            let mut f = Fitness::bounded(3, 1, &[-5.0, -5.0, -5.0], &[5.0, 5.0, 5.0]);
            f.set_normalize(true);
            f
        };
        let params = CmaesParams {
            popsize: 12,
            max_evaluations: 2000,
            seed: 42,
            ..Default::default()
        };
        let mut opt = Cmaes::new(fit, &[0.0, 0.0, 0.0], &[0.3], &params);
        for _ in 0..50 {
            let pop = opt.ask();
            assert_eq!(pop.len(), 12);
            assert_eq!(pop[0].len(), 3);
            let ys: Vec<f64> = pop.iter().map(|x| sphere(x)).collect();
            if opt.tell(&ys) != 0 {
                break;
            }
        }
        assert!(opt.result().y < 1e-3, "ask/tell did not converge");
    }

    #[test]
    fn external_population_defaults_getters_and_stop_fitness() {
        let mut fit = Fitness::bounded(3, 1, &[-2.0; 3], &[2.0; 3]);
        fit.set_normalize(true);
        let params = CmaesParams {
            popsize: 0,
            stop_fitness: 0.0,
            stop_tol_hist_fun: 0.0,
            update_gap: 0,
            seed: 19,
            ..Default::default()
        };
        let mut optimizer = Cmaes::new(fit, &[0.5; 3], &[0.2, 0.3, 0.4], &params);
        assert_eq!(optimizer.dim(), 3);
        assert!(optimizer.popsize() >= 4);
        assert_eq!(optimizer.stop(), 0);
        let population = vec![vec![0.25; 3]; optimizer.popsize()];
        let values = vec![-1.0; optimizer.popsize()];
        assert_eq!(optimizer.tell_x(&values, &population), 1);
        assert_eq!(optimizer.stop(), 1);
        assert_eq!(optimizer.population(), population);
        assert_eq!(optimizer.result().evaluations, values.len() as u64);
        assert_eq!(optimizer.result().y, -1.0);
    }
}
