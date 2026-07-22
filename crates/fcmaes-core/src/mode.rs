// Numeric kernels index several parallel arrays by a shared loop counter, where
// range loops read more clearly than zipped iterators.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

//! MODE — Rust port of the C++ `modeoptimizer.cpp`.
//!
//! Multi-objective / constrained Differential Evolution (DE/all/1) with an
//! optional NSGA-II-style population update. Features enhanced multiple
//! constraint ranking, oscillating CR/F, SBX + polynomial variation, mixed
//! integer handling, and normalized all-objective crowding distance.
//!
//! Ask/tell only (the caller evaluates objectives+constraints and feeds them
//! back), so the core needs no objective callback. Row-per-individual layout
//! (the C++ used column-per-individual Eigen matrices). Replaces both the C++
//! optimizer and the pure-Python `fcmaes/mode.py`; parity is statistical.

use crate::fitness::Fitness;
use crate::rng::Rng;

const BIG: f64 = f64::MAX;

/// Outcome/result snapshot of a MODE run.
#[derive(Clone, Debug)]
pub struct ModeResult {
    /// Current population (rows = individuals, `dim` columns).
    pub x: Vec<Vec<f64>>,
    /// Objective+constraint values of the population.
    pub y: Vec<Vec<f64>>,
    pub iterations: i32,
    pub stop: i32,
}

/// Tunable inputs for [`Mode::new`].
#[derive(Clone, Debug)]
pub struct ModeParams {
    pub popsize: i32,
    pub f: f64,
    pub cr: f64,
    pub pro_c: f64,
    pub dis_c: f64,
    pub pro_m: f64,
    pub dis_m: f64,
    pub nsga_update: bool,
    pub pareto_update: f64,
    pub min_mutate: f64,
    pub max_mutate: f64,
    pub seed: u64,
    pub runid: i64,
}

impl Default for ModeParams {
    fn default() -> Self {
        Self {
            popsize: 64,
            f: 0.5,
            cr: 0.9,
            pro_c: 0.5,
            dis_c: 15.0,
            pro_m: 0.9,
            dis_m: 20.0,
            nsga_update: true,
            pareto_update: 0.0,
            min_mutate: 0.1,
            max_mutate: 0.5,
            seed: 0,
            runid: 0,
        }
    }
}

fn sort_index(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    idx
}

fn validate_mode_inputs(
    fitfun: &Fitness,
    nobj: usize,
    ncon: usize,
    ints: Option<&[bool]>,
    p: &ModeParams,
) -> Result<(), &'static str> {
    if fitfun.dim() == 0 || !fitfun.has_bounds() {
        return Err("MODE requires a non-empty bounded decision space");
    }
    if fitfun
        .lower()
        .iter()
        .zip(fitfun.upper())
        .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo >= hi)
    {
        return Err("MODE bounds must be finite and satisfy lower < upper");
    }
    if nobj == 0 || fitfun.nobj() != nobj + ncon {
        return Err("MODE requires nobj > 0 and Fitness::nobj == nobj + ncon");
    }
    let popsize = if p.popsize > 0 { p.popsize } else { 128 };
    if popsize < 4 {
        return Err("MODE population size must be at least four");
    }
    if ints.is_some_and(|values| values.len() != fitfun.dim()) {
        return Err("MODE integer mask length must equal the decision dimension");
    }
    if !p.f.is_finite()
        || !p.cr.is_finite()
        || !p.pro_c.is_finite()
        || !p.dis_c.is_finite()
        || !p.pro_m.is_finite()
        || !p.dis_m.is_finite()
        || !p.pareto_update.is_finite()
        || !p.min_mutate.is_finite()
        || !p.max_mutate.is_finite()
    {
        return Err("MODE parameters must be finite");
    }
    if !(0.0..=1.0).contains(&p.pro_c) || !(0.0..=1.0).contains(&p.pro_m) {
        return Err("MODE crossover and mutation probabilities must be in [0, 1]");
    }
    if p.dis_c <= 0.0 || p.dis_m <= 0.0 {
        return Err("MODE distribution indices must be positive");
    }
    if p.min_mutate > 0.0 && p.max_mutate > 0.0 && p.min_mutate > p.max_mutate {
        return Err("MODE min_mutate must not exceed max_mutate");
    }
    Ok(())
}

pub struct Mode {
    fitfun: Fitness,
    rng: Rng,
    dim: usize,
    nobj: usize,
    ncon: usize,
    nobj_ncon: usize,
    popsize: usize,

    f0: f64,
    cr0: f64,
    f: f64,
    cr: f64,
    pro_c: f64,
    dis_c: f64,
    pro_m: f64,
    dis_m: f64,
    nsga_update: bool,
    pareto_update: f64,
    min_mutate: f64,
    max_mutate: f64,
    is_int: Option<Vec<bool>>,

    // population: [0..popsize] current, [popsize..2*popsize] offspring
    pop_x: Vec<Vec<f64>>,
    pop_y: Vec<Vec<f64>>,
    v_x: Vec<Vec<f64>>, // NSGA variation buffer
    vp: usize,

    last_con: Option<Vec<Vec<f64>>>,
    last_eps: Vec<f64>,

    iterations: i32,
    stop: i32,
    pending: bool,
}

impl Mode {
    /// Construct MODE after validating dimensions, bounds, population size,
    /// probabilities, and the optional integer mask.
    pub fn try_new(
        fitfun: Fitness,
        nobj: usize,
        ncon: usize,
        ints: Option<Vec<bool>>,
        p: &ModeParams,
    ) -> Result<Self, &'static str> {
        validate_mode_inputs(&fitfun, nobj, ncon, ints.as_deref(), p)?;
        Ok(Self::new_unchecked(fitfun, nobj, ncon, ints, p))
    }

    /// Construct MODE, panicking on invalid configuration. Applications that
    /// accept user input should prefer [`Mode::try_new`].
    pub fn new(
        fitfun: Fitness,
        nobj: usize,
        ncon: usize,
        ints: Option<Vec<bool>>,
        p: &ModeParams,
    ) -> Self {
        Self::try_new(fitfun, nobj, ncon, ints, p).expect("invalid MODE configuration")
    }

    fn new_unchecked(
        fitfun: Fitness,
        nobj: usize,
        ncon: usize,
        ints: Option<Vec<bool>>,
        p: &ModeParams,
    ) -> Self {
        let dim = fitfun.dim();
        let popsize = if p.popsize > 0 {
            p.popsize as usize
        } else {
            128
        };
        let f0 = if p.f > 0.0 { p.f } else { 0.5 };
        let cr0 = if p.cr > 0.0 { p.cr } else { 0.9 };
        let mut m = Mode {
            dim,
            nobj,
            ncon,
            nobj_ncon: nobj + ncon,
            popsize,
            f0,
            cr0,
            f: f0,
            cr: cr0,
            pro_c: p.pro_c,
            dis_c: p.dis_c,
            pro_m: p.pro_m,
            dis_m: p.dis_m,
            nsga_update: p.nsga_update,
            pareto_update: p.pareto_update,
            min_mutate: if p.min_mutate > 0.0 {
                p.min_mutate
            } else {
                0.1
            },
            max_mutate: if p.max_mutate > 0.0 {
                p.max_mutate
            } else {
                0.5
            },
            is_int: ints,
            pop_x: vec![],
            pop_y: vec![],
            v_x: vec![],
            vp: 0,
            last_con: None,
            last_eps: vec![0.0; ncon],
            iterations: 0,
            stop: 0,
            pending: false,
            rng: Rng::new(p.seed.wrapping_add(p.runid as u64)),
            fitfun,
        };
        m.init();
        m
    }

    fn init(&mut self) {
        let n = 2 * self.popsize;
        self.pop_x = (0..n)
            .map(|i| {
                if i < self.popsize {
                    self.fitfun.sample(&mut self.rng)
                } else {
                    vec![0.0; self.dim]
                }
            })
            .collect();
        self.pop_y = vec![vec![BIG; self.nobj_ncon]; n];
        self.v_x = self.pop_x[0..self.popsize].to_vec();
        self.vp = 0;
        self.pending = false;
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn nobj(&self) -> usize {
        self.nobj
    }
    pub fn ncon(&self) -> usize {
        self.ncon
    }
    pub fn popsize(&self) -> usize {
        self.popsize
    }
    pub fn stop(&self) -> i32 {
        self.stop
    }

    // ---- variation (SBX crossover + polynomial mutation) ----

    fn variation(&mut self, pop: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let dim = self.dim;
        let dis_c = (0.5 * self.rng.uniform01() + 0.5) * self.dis_c;
        let dis_m = (0.5 * self.rng.uniform01() + 0.5) * self.dis_m;
        let n2 = pop.len() / 2;
        let n = 2 * n2;
        // beta[p][i]
        let mut beta = vec![vec![0.0; dim]; n2];
        for pb in beta.iter_mut() {
            let cross_pair = self.rng.uniform01() < self.pro_c;
            for i in 0..dim {
                if !cross_pair || self.rng.uniform01() < 0.5 {
                    pb[i] = 1.0;
                } else {
                    let r = self.rng.uniform01();
                    let mut b = if r <= 0.5 {
                        (2.0 * r).powf(1.0 / (dis_c + 1.0))
                    } else {
                        (2.0 * r).powf(-1.0 / (dis_c + 1.0))
                    };
                    if self.rng.uniform01() > 0.5 {
                        b = -b;
                    }
                    pb[i] = b;
                }
            }
        }
        let mut offspring: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut off2: Vec<Vec<f64>> = Vec::with_capacity(n2);
        for p in 0..n2 {
            let p1 = &pop[p];
            let p2 = &pop[n2 + p];
            let mut o1 = vec![0.0; dim];
            let mut o2 = vec![0.0; dim];
            for i in 0..dim {
                let base = (p1[i] + p2[i]) * 0.5;
                let delta = beta[p][i] * (p1[i] - p2[i]) * 0.5;
                o1[i] = base + delta;
                o2[i] = base - delta;
            }
            offspring.push(o1);
            off2.push(o2);
        }
        offspring.extend(off2);

        // The Python implementation truncates odd populations, which leaves
        // the ask/tell batch one candidate short. Preserve the final parent so
        // polynomial mutation can still produce exactly `pop.len()` children.
        if offspring.len() < pop.len() {
            offspring.push(pop[pop.len() - 1].clone());
        }

        let limit = self.pro_m / dim as f64;
        for op in offspring.iter_mut() {
            for i in 0..dim {
                if self.rng.uniform01() < limit {
                    let mu = self.rng.uniform01();
                    let norm = self.fitfun.norm_i(i, op[i]);
                    let scale = self.fitfun.scale()[i];
                    if mu <= 0.5 {
                        op[i] += scale
                            * ((2.0 * mu + (1.0 - 2.0 * mu) * (1.0 - norm).powf(dis_m + 1.0))
                                .powf(1.0 / (dis_m + 1.0))
                                - 1.0);
                    } else {
                        op[i] += scale
                            * (1.0
                                - (2.0 * (1.0 - mu)
                                    + 2.0 * (mu - 0.5) * (1.0 - norm).powf(dis_m + 1.0))
                                .powf(1.0 / (dis_m + 1.0)));
                    }
                }
            }
        }
        for op in offspring.iter_mut() {
            *op = self.fitfun.closest_feasible(op);
        }
        offspring
    }

    fn modify(&mut self, x: &mut [f64]) {
        let Some(is_int) = self.is_int.clone() else {
            return;
        };
        let n_ints = is_int.iter().filter(|&&b| b).count() as f64;
        if n_ints == 0.0 {
            return;
        }
        let to_mutate =
            self.min_mutate + self.rng.uniform01() * (self.max_mutate - self.min_mutate);
        for i in 0..self.dim {
            if is_int[i] && self.rng.uniform01() < to_mutate / n_ints {
                x[i] = self.fitfun.sample_i(i, &mut self.rng).trunc();
            }
        }
    }

    fn next_x(&mut self, p: usize) -> Vec<f64> {
        if p == 0 {
            self.iterations += 1;
        }
        if self.nsga_update {
            let x = self.v_x[self.vp].clone();
            self.vp = (self.vp + 1) % self.v_x.len();
            return x;
        }
        if p == 0 {
            self.cr = if self.iterations % 2 == 0 {
                0.5 * self.cr0
            } else {
                self.cr0
            };
            self.f = if self.iterations % 2 == 0 {
                0.5 * self.f0
            } else {
                self.f0
            };
        }
        let ps = self.popsize;
        let (mut r1, mut r2, mut r3);
        loop {
            r1 = self.rng.int_below(ps as i64) as usize;
            r2 = self.rng.int_below(ps as i64) as usize;
            r3 = if self.pareto_update > 0.0 {
                (self.rng.uniform01().powf(1.0 + self.pareto_update) * ps as f64) as usize
            } else {
                self.rng.int_below(ps as i64) as usize
            };
            if r3 != p && r3 != r1 && r3 != r2 && r2 != p && r2 != r1 && r1 != p {
                break;
            }
        }
        let xp = self.pop_x[p].clone();
        let x1 = &self.pop_x[r1];
        let x2 = &self.pop_x[r2];
        let x3 = &self.pop_x[r3];
        let mut x: Vec<f64> = (0..self.dim)
            .map(|j| x3[j] + (x1[j] - x2[j]) * self.f)
            .collect();
        let r = self.rng.int_below(self.dim as i64) as usize;
        for j in 0..self.dim {
            if j != r && self.rng.uniform01() > self.cr {
                x[j] = xp[j];
            }
        }
        self.modify(&mut x);
        self.fitfun.closest_feasible(&x)
    }

    // ---- pareto ranking ----

    /// `true` if individual `i` is dominated by `index` (index is <= i in all
    /// objectives). `objs[k]` is the length-`nobj` objective vector of k.
    fn is_dominated(objs: &[Vec<f64>], i: usize, index: usize) -> bool {
        for j in 0..objs[i].len() {
            if objs[i][j] < objs[index][j] {
                return false;
            }
        }
        true
    }

    fn pareto_levels(objs: &[Vec<f64>]) -> Vec<f64> {
        let n = objs.len();
        let mut domination = vec![0.0; n];
        let mut mask = vec![true; n];
        let mut index = 0;
        while index < n {
            for i in 0..n {
                if i != index && mask[i] && Self::is_dominated(objs, i, index) {
                    mask[i] = false;
                }
            }
            for i in 0..n {
                if mask[i] {
                    domination[i] += 1.0;
                }
            }
            index += 1;
            while index < n && !mask[index] {
                index += 1;
            }
        }
        domination
    }

    fn objranks(objs: &[Vec<f64>]) -> Vec<f64> {
        let n = objs.len();
        let nobj = objs[0].len();
        let mut rank_sum = vec![0.0; n];
        for j in 0..nobj {
            let col: Vec<f64> = objs.iter().map(|o| o[j]).collect();
            let order = sort_index(&col);
            for (pos, &idx) in order.iter().enumerate() {
                rank_sum[idx] += pos as f64;
            }
        }
        rank_sum
    }

    fn ranks(cons: &[Vec<f64>], eps: &[f64]) -> Vec<f64> {
        let n = cons.len();
        let ncon = eps.len();
        let mut rank = vec![vec![0.0; ncon]; n];
        let mut alpha = vec![0.0; n];
        for j in 0..ncon {
            let col: Vec<f64> = cons.iter().map(|c| c[j]).collect();
            let order = sort_index(&col);
            for (pos, &idx) in order.iter().enumerate() {
                if cons[idx][j] <= eps[j] {
                    rank[idx][j] = 0.0;
                } else {
                    rank[idx][j] = pos as f64;
                    alpha[idx] += 1.0;
                }
            }
        }
        let mut csum = vec![0.0; n];
        for i in 0..n {
            for j in 0..ncon {
                csum[i] += rank[i][j] * alpha[i] / ncon as f64;
            }
        }
        csum
    }

    fn pareto(&mut self, ys: &[Vec<f64>]) -> Vec<f64> {
        if self.ncon == 0 {
            return Self::pareto_levels(ys);
        }
        let popn = ys.len();
        let objs: Vec<Vec<f64>> = ys.iter().map(|y| y[0..self.nobj].to_vec()).collect();
        let cons: Vec<Vec<f64>> = ys
            .iter()
            .map(|y| {
                y[self.nobj..self.nobj_ncon]
                    .iter()
                    .map(|&c| c.max(0.0))
                    .collect()
            })
            .collect();

        let mut eps = vec![0.0; self.ncon];
        if self.iterations > 1
            && let Some(last) = &self.last_con
        {
            let last_max = last
                .iter()
                .flat_map(|c| c.iter().cloned())
                .fold(f64::MIN, f64::max);
            if last_max < 1e90 {
                let mut eps_mean = vec![0.0; self.ncon];
                for j in 0..self.ncon {
                    let mean_j = last.iter().map(|c| c[j]).sum::<f64>() / last.len() as f64;
                    eps_mean[j] = 0.5 * (self.last_eps[j] + 0.5 * mean_j);
                }
                if eps_mean.iter().cloned().fold(f64::MIN, f64::max) > 1e-8 {
                    eps = eps_mean;
                }
            }
        }
        self.last_con = Some(cons.clone());
        self.last_eps = eps.clone();

        let feasible: Vec<bool> = cons
            .iter()
            .map(|c| c.iter().zip(&eps).all(|(&cv, &ev)| cv <= ev))
            .collect();
        let has_feasible = feasible.iter().any(|&f| f);
        let has_infeasible = feasible.iter().any(|&f| !f);

        let mut csum = Self::ranks(&cons, &eps);
        if has_feasible {
            let orank = Self::objranks(&objs);
            for i in 0..popn {
                csum[i] += orank[i];
            }
        }
        let ci = sort_index(&csum);
        let mut fiv = vec![];
        let mut viv = vec![];
        for &i in &ci {
            if feasible[i] {
                fiv.push(i);
            } else {
                viv.push(i);
            }
        }
        let mut domination = vec![0.0; popn];
        if has_feasible {
            let feas_objs: Vec<Vec<f64>> = fiv.iter().map(|&i| objs[i].clone()).collect();
            let ypar = Self::pareto_levels(&feas_objs);
            for (k, &i) in fiv.iter().enumerate() {
                domination[i] += ypar[k];
            }
        }
        if has_infeasible {
            for (i, &vi) in viv.iter().enumerate() {
                domination[vi] += (viv.len() - i) as f64;
            }
            for &fi in &fiv {
                domination[fi] += (viv.len() + 1) as f64;
            }
        }
        domination
    }

    fn crowd_dist(sub: &[Vec<f64>], nobj: usize) -> Vec<f64> {
        let n = sub.len();
        if n == 0 {
            return Vec::new();
        }
        if n <= 2 {
            return vec![BIG; n];
        }
        let mut distance = vec![0.0; n];
        for objective in 0..nobj {
            let values: Vec<f64> = sub.iter().map(|y| y[objective]).collect();
            let order = sort_index(&values);
            let lo = values[order[0]];
            let hi = values[order[n - 1]];
            let span = hi - lo;
            if !span.is_finite() || span <= 0.0 {
                continue;
            }
            distance[order[0]] = BIG;
            distance[order[n - 1]] = BIG;
            for position in 1..n - 1 {
                let index = order[position];
                if distance[index] != BIG {
                    distance[index] +=
                        (values[order[position + 1]] - values[order[position - 1]]) / span;
                }
            }
        }
        if distance.iter().all(|&value| value == 0.0) {
            return vec![0.0; n];
        }
        distance
    }

    fn pop_update(&mut self) {
        let n = 2 * self.popsize;
        let mut x0 = self.pop_x[0..n].to_vec();
        let mut y0 = self.pop_y[0..n].to_vec();
        if self.nobj == 1 {
            let col: Vec<f64> = y0.iter().map(|y| y[0]).collect();
            let mut yi = sort_index(&col);
            yi.reverse();
            x0 = yi.iter().map(|&i| x0[i].clone()).collect();
            y0 = yi.iter().map(|&i| y0[i].clone()).collect();
        }
        let domination = self.pareto(&y0);
        let maxdom = domination.iter().cloned().fold(f64::MIN, f64::max) as i32;
        let mut newx: Vec<Vec<f64>> = Vec::with_capacity(self.popsize);
        let mut newy: Vec<Vec<f64>> = Vec::with_capacity(self.popsize);
        for dom in (0..=maxdom).rev() {
            let level: Vec<usize> = (0..n).filter(|&i| domination[i] as i32 == dom).collect();
            if level.is_empty() {
                continue;
            }
            if newx.len() + level.len() <= self.popsize {
                for &i in &level {
                    newx.push(x0[i].clone());
                    newy.push(y0[i].clone());
                }
            } else {
                if level.len() > 1 {
                    let domy: Vec<Vec<f64>> = level.iter().map(|&i| y0[i].clone()).collect();
                    let cd = Self::crowd_dist(&domy, self.nobj);
                    let mut si = sort_index(&cd);
                    si.reverse();
                    for &k in &si {
                        if newx.len() >= self.popsize {
                            break;
                        }
                        let i = level[k];
                        newx.push(x0[i].clone());
                        newy.push(y0[i].clone());
                    }
                } else {
                    newx.push(x0[level[0]].clone());
                    newy.push(y0[level[0]].clone());
                }
                break;
            }
        }
        for i in 0..self.popsize {
            self.pop_x[i] = newx[i].clone();
            self.pop_y[i] = newy[i].clone();
        }
        if self.nsga_update {
            let cur = self.pop_x[0..self.popsize].to_vec();
            self.v_x = self.variation(&cur);
        }
    }

    // ---- ask/tell interface ----

    /// Ask for `popsize` offspring rows.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        self.try_ask().expect("invalid MODE ask call")
    }

    /// Fallible ask variant for interfaces that need to report call-order
    /// errors rather than panic.
    pub fn try_ask(&mut self) -> Result<Vec<Vec<f64>>, &'static str> {
        if self.pending {
            return Err("MODE ask called before telling the pending batch");
        }
        for p in 0..self.popsize {
            let x = self.next_x(p);
            self.pop_x[self.popsize + p] = x;
        }
        self.pending = true;
        Ok(self.pop_x[self.popsize..2 * self.popsize].to_vec())
    }

    fn set_x(&mut self, xs: &[Vec<f64>]) {
        for (p, row) in xs.iter().enumerate().take(self.popsize) {
            self.pop_x[self.popsize + p] = row.clone();
        }
    }

    /// Tell objective+constraint values for the offspring from [`ask`](Mode::ask).
    pub fn tell(&mut self, ys: &[Vec<f64>]) -> i32 {
        self.try_tell(ys).expect("invalid MODE tell call")
    }

    /// Fallible tell variant validating call order and matrix shape.
    pub fn try_tell(&mut self, ys: &[Vec<f64>]) -> Result<i32, &'static str> {
        if !self.pending {
            return Err("MODE tell called without a pending ask batch");
        }
        if ys.len() != self.popsize {
            return Err("MODE tell batch length must equal popsize");
        }
        for (p, row) in ys.iter().enumerate() {
            if row.len() != self.nobj_ncon {
                return Err("MODE tell row width must equal nobj + ncon");
            }
            self.pop_y[self.popsize + p] = row
                .iter()
                .map(|&value| if value.is_finite() { value } else { BIG })
                .collect();
        }
        self.pop_update();
        self.pending = false;
        Ok(self.stop)
    }

    /// Tell with a switched update mode (the C++ `tell_switch`).
    pub fn tell_switch(&mut self, ys: &[Vec<f64>], nsga_update: bool, pareto_update: f64) -> i32 {
        self.try_tell_switch(ys, nsga_update, pareto_update)
            .expect("invalid MODE tell_switch call")
    }

    pub fn try_tell_switch(
        &mut self,
        ys: &[Vec<f64>],
        nsga_update: bool,
        pareto_update: f64,
    ) -> Result<i32, &'static str> {
        if !pareto_update.is_finite() {
            return Err("MODE pareto_update must be finite");
        }
        self.nsga_update = nsga_update;
        self.pareto_update = pareto_update;
        self.try_tell(ys)
    }

    /// Replace the population/offspring and tell (the C++ `set_population`).
    pub fn set_population(&mut self, xs: &[Vec<f64>], ys: &[Vec<f64>]) -> i32 {
        self.try_set_population(xs, ys)
            .expect("invalid MODE set_population call")
    }

    pub fn try_set_population(
        &mut self,
        xs: &[Vec<f64>],
        ys: &[Vec<f64>],
    ) -> Result<i32, &'static str> {
        if xs.len() < 4 {
            return Err("MODE population size must be at least four");
        }
        if xs.len() != ys.len() {
            return Err("MODE population x/y length mismatch");
        }
        if xs.iter().any(|row| row.len() != self.dim) {
            return Err("MODE population row width must equal dim");
        }
        if xs.len() != self.popsize {
            self.popsize = xs.len();
            self.init();
        }
        self.set_x(xs);
        self.pending = true;
        self.try_tell(ys)
    }

    /// Current population (rows = individuals).
    pub fn population(&self) -> Vec<Vec<f64>> {
        self.pop_x[0..self.popsize].to_vec()
    }

    pub fn result(&self) -> ModeResult {
        ModeResult {
            x: self.population(),
            y: self.pop_y[0..self.popsize].to_vec(),
            iterations: self.iterations,
            stop: self.stop,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two-objective test: minimize (sum x^2, sum (x-2)^2) — a convex Pareto
    // front between 0 and 2 in each coordinate.
    fn eval(x: &[f64]) -> Vec<f64> {
        let o1: f64 = x.iter().map(|v| v * v).sum();
        let o2: f64 = x.iter().map(|v| (v - 2.0) * (v - 2.0)).sum();
        vec![o1, o2]
    }

    fn run(nsga: bool) -> Mode {
        let fit = Fitness::bounded(3, 2, &[-5.0; 3], &[5.0; 3]);
        let params = ModeParams {
            popsize: 32,
            nsga_update: nsga,
            seed: 1,
            ..Default::default()
        };
        let mut opt = Mode::new(fit, 2, 0, None, &params);
        for _ in 0..80 {
            let xs = opt.ask();
            let ys: Vec<Vec<f64>> = xs.iter().map(|x| eval(x)).collect();
            opt.tell(&ys);
        }
        opt
    }

    #[test]
    fn nsga_finds_pareto_front() {
        let opt = run(true);
        let r = opt.result();
        // Front should contain points near both extremes (o1~0 and o2~0).
        let min_o1 = r.y.iter().map(|y| y[0]).fold(f64::MAX, f64::min);
        let min_o2 = r.y.iter().map(|y| y[1]).fold(f64::MAX, f64::min);
        assert!(min_o1 < 0.1, "no low-o1 solution: {min_o1}");
        assert!(min_o2 < 0.1, "no low-o2 solution: {min_o2}");
    }

    #[test]
    fn de_update_finds_pareto_front() {
        let opt = run(false);
        let r = opt.result();
        let min_o1 = r.y.iter().map(|y| y[0]).fold(f64::MAX, f64::min);
        let min_o2 = r.y.iter().map(|y| y[1]).fold(f64::MAX, f64::min);
        assert!(min_o1 < 0.2, "no low-o1 solution: {min_o1}");
        assert!(min_o2 < 0.2, "no low-o2 solution: {min_o2}");
    }

    #[test]
    fn constrained_run_progresses() {
        // 1 objective, 1 constraint: minimize sum x^2 s.t. sum x >= 1
        // (constraint value = 1 - sum x, feasible when <= 0).
        let fit = Fitness::bounded(3, 2, &[-5.0; 3], &[5.0; 3]);
        let params = ModeParams {
            popsize: 24,
            nsga_update: false,
            seed: 2,
            ..Default::default()
        };
        let mut opt = Mode::new(fit, 1, 1, None, &params);
        for _ in 0..100 {
            let xs = opt.ask();
            let ys: Vec<Vec<f64>> = xs
                .iter()
                .map(|x| {
                    let o: f64 = x.iter().map(|v| v * v).sum();
                    let c: f64 = 1.0 - x.iter().sum::<f64>();
                    vec![o, c]
                })
                .collect();
            opt.tell(&ys);
        }
        let r = opt.result();
        // Some feasible solution (sum x >= 1) should exist with small objective.
        let best =
            r.y.iter()
                .filter(|y| y[1] <= 0.0)
                .map(|y| y[0])
                .fold(f64::MAX, f64::min);
        assert!(best < 2.0, "constrained best too large: {best}");
    }

    #[test]
    fn rejects_invalid_configuration() {
        let fit = Fitness::bounded(2, 2, &[-1.0; 2], &[1.0; 2]);
        let mut params = ModeParams {
            popsize: 3,
            ..Default::default()
        };
        assert!(Mode::try_new(fit.clone(), 2, 0, None, &params).is_err());
        params.popsize = 5;
        assert!(Mode::try_new(fit.clone(), 0, 2, None, &params).is_err());
        assert!(Mode::try_new(fit.clone(), 2, 0, Some(vec![true]), &params).is_err());
        params.pro_m = 1.5;
        assert!(Mode::try_new(fit, 2, 0, None, &params).is_err());
    }

    #[test]
    fn odd_population_preserves_batch_size() {
        let fit = Fitness::bounded(2, 2, &[-1.0; 2], &[1.0; 2]);
        let params = ModeParams {
            popsize: 5,
            nsga_update: true,
            seed: 7,
            ..Default::default()
        };
        let mut mode = Mode::try_new(fit, 2, 0, None, &params).unwrap();
        for _ in 0..3 {
            let xs = mode.ask();
            assert_eq!(xs.len(), 5);
            let ys: Vec<Vec<f64>> = xs.iter().map(|x| vec![x[0], x[1]]).collect();
            mode.tell(&ys);
        }
    }

    #[test]
    fn crowding_uses_every_objective() {
        let values = vec![
            vec![0.0, 0.5],
            vec![0.25, 0.0],
            vec![0.5, 0.5],
            vec![0.75, 1.0],
            vec![1.0, 0.5],
        ];
        let distance = Mode::crowd_dist(&values, 2);
        assert_eq!(distance.iter().filter(|&&d| d == BIG).count(), 4);
        assert!(distance[2].is_finite() && distance[2] > 0.0);
        assert_eq!(Mode::crowd_dist(&vec![vec![1.0, 1.0]; 4], 2), vec![0.0; 4]);
    }

    #[test]
    fn zero_crossover_and_mutation_preserve_parents() {
        let fit = Fitness::bounded(2, 2, &[-1.0; 2], &[1.0; 2]);
        let params = ModeParams {
            popsize: 5,
            pro_c: 0.0,
            pro_m: 0.0,
            seed: 9,
            ..Default::default()
        };
        let mut mode = Mode::try_new(fit, 2, 0, None, &params).unwrap();
        let parents = vec![
            vec![-0.8, -0.7],
            vec![-0.4, -0.3],
            vec![0.1, 0.2],
            vec![0.5, 0.6],
            vec![0.8, 0.9],
        ];
        let offspring = mode.variation(&parents);
        for (child, parent) in offspring.iter().zip(&parents) {
            for (&actual, &expected) in child.iter().zip(parent) {
                assert!((actual - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn tell_sanitizes_non_finite_values() {
        let fit = Fitness::bounded(2, 1, &[-1.0; 2], &[1.0; 2]);
        let params = ModeParams {
            popsize: 4,
            nsga_update: false,
            ..Default::default()
        };
        let mut mode = Mode::try_new(fit, 1, 0, None, &params).unwrap();
        mode.ask();
        mode.tell(&[vec![f64::NAN], vec![1.0], vec![2.0], vec![3.0]]);
        assert!(
            mode.result()
                .y
                .iter()
                .flatten()
                .all(|value| !value.is_nan())
        );
    }

    #[test]
    fn ask_tell_enforces_call_order_and_shapes() {
        let fit = Fitness::bounded(2, 1, &[-1.0; 2], &[1.0; 2]);
        let params = ModeParams {
            popsize: 4,
            ..Default::default()
        };
        let mut mode = Mode::try_new(fit, 1, 0, None, &params).unwrap();
        assert!(mode.try_tell(&vec![vec![0.0]; 4]).is_err());
        mode.try_ask().unwrap();
        assert!(mode.try_ask().is_err());
        assert!(mode.try_tell(&vec![vec![0.0]; 3]).is_err());
        assert!(mode.try_tell(&vec![vec![0.0, 1.0]; 4]).is_err());
        assert_eq!(mode.try_tell(&vec![vec![0.0]; 4]).unwrap(), 0);
    }
}
