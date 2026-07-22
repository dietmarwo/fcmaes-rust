// Numeric kernels index several parallel arrays by a shared loop counter, where
// range loops read more clearly than zipped iterators.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

//! BiteOpt — Rust port of Aleksey Vaneev's BiteOpt
//! (<https://github.com/avaneev/biteopt>), from the C++ `biteopt.h`/`biteaux.h`.
//!
//! The implementation includes the LCG-hash `BiteRnd`, 58-bit integer-mantissa
//! parameters, adaptive sparse selectors, dynamic and diverging populations,
//! all primary generators, CSpherOpt, sequential Nelder-Mead, deep
//! multi-population solution exchange, and delayed-feedback batch ask/tell.
//! Parity is validated by convergence because no independent Python twin
//! exists.

use std::collections::VecDeque;

use crate::fitness::Objective;

const INT_MANT_BITS: u32 = 58;
const INT_MANT_MULT: i64 = 1i64 << INT_MANT_BITS;
const INT_MANT_MASK: i64 = INT_MANT_MULT - 1;
const BAD_COST: f64 = 1e300;
const MAX_DEEP_OPTIMIZERS: i32 = 36;

#[inline]
fn sanitize_cost(cost: f64) -> f64 {
    if cost.is_finite() { cost } else { BAD_COST }
}

#[inline]
fn objective_cost(objective: &dyn Objective, values: &[f64]) -> f64 {
    sanitize_cost(objective.eval_scalar(values))
}

/// Validate the dimensions and numeric domain required by BiteOpt.
pub fn validate_bite_inputs(
    lower: &[f64],
    upper: &[f64],
    init: Option<&[f64]>,
    params: &BiteParams,
    depth: i32,
) -> Result<(), String> {
    if lower.is_empty() || lower.len() != upper.len() {
        return Err("bounds must be non-empty and have equal lengths".into());
    }
    for (&lo, &hi) in lower.iter().zip(upper) {
        if !lo.is_finite() || !hi.is_finite() || lo >= hi || !(hi - lo).is_finite() {
            return Err("bounds must contain finite intervals with lower < upper".into());
        }
    }
    if let Some(values) = init {
        if values.len() != lower.len() {
            return Err("initial guess must match the bounds dimension".into());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("initial guess must contain only finite values".into());
        }
    }
    if params.stop_fitness.is_nan() {
        return Err("stop_fitness must not be NaN".into());
    }
    if (1..4).contains(&params.popsize) {
        return Err("popsize must be non-positive (automatic) or at least 4".into());
    }
    if depth > MAX_DEEP_OPTIMIZERS {
        return Err(format!(
            "M must not exceed {MAX_DEEP_OPTIMIZERS} deep optimizers"
        ));
    }
    Ok(())
}

/// A generated-but-not-yet-applied candidate (frozen population state).
struct Candidate {
    enc: Vec<i64>,
    real: Vec<f64>,
    sels: Vec<SelUse>,
    is_init: bool,
    /// Set by `generateSolPar`: the parallel optimizer already evaluated it.
    precomputed_cost: Option<f64>,
}

/// Which population a generator draws from.
#[derive(Clone, Copy)]
enum PopSel {
    Main,
    Par(usize),
    ParOpt,
    ParOpt2,
}

/// Outcome of a BiteOpt run.
#[derive(Clone, Debug)]
pub struct BiteResult {
    pub x: Vec<f64>,
    pub y: f64,
    pub evaluations: u64,
    pub iterations: i32,
    pub stop: i32,
}

/// Tunable inputs for [`optimize_bite`].
#[derive(Clone, Debug)]
pub struct BiteParams {
    pub popsize: i32,
    pub max_evaluations: u64,
    pub stop_fitness: f64,
    pub stall_criterion: i32,
    pub seed: u64,
    pub runid: i64,
}

impl Default for BiteParams {
    fn default() -> Self {
        Self {
            popsize: 0,
            max_evaluations: 100_000,
            stop_fitness: f64::NEG_INFINITY,
            stall_criterion: 0,
            seed: 0,
            runid: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// PRNG (faithful port of CBiteRnd)
// ---------------------------------------------------------------------------

pub struct BiteRnd {
    seed: u64,
    lcg: u64,
    hash: u64,
    bit_pool: u64,
    bits_left: i32,
}

impl BiteRnd {
    pub fn new(seed: u64) -> Self {
        let mut r = BiteRnd {
            seed,
            lcg: 0,
            hash: 0,
            bit_pool: 0,
            bits_left: 0,
        };
        for _ in 0..5 {
            r.advance();
        }
        r
    }

    #[inline]
    fn advance(&mut self) -> u64 {
        self.seed = self
            .seed
            .wrapping_mul(self.lcg.wrapping_mul(2).wrapping_add(1));
        let rs = self.seed.rotate_left(32);
        self.hash = self.hash.wrapping_add(rs).wrapping_add(0xAAAAAAAAAAAAAAAA);
        self.lcg = self
            .lcg
            .wrapping_add(self.seed)
            .wrapping_add(0x5555555555555555);
        self.seed ^= self.hash;
        self.lcg ^ rs
    }

    #[inline]
    pub fn get(&mut self) -> f64 {
        (self.advance() >> (64 - 53)) as f64 * (-53f64).exp2()
    }
    #[inline]
    pub fn get_int(&mut self, n: i32) -> i32 {
        (self.get() * n as f64) as i32
    }
    #[inline]
    fn get_sqr(&mut self) -> f64 {
        let v = self.get();
        v * v
    }
    #[inline]
    pub fn get_sqr_int(&mut self, n: i32) -> i32 {
        (self.get_sqr() * n as f64) as i32
    }
    fn get_pow(&mut self, p: f64) -> f64 {
        let v = self.get();
        // These are the powers used by BiteOpt's hot paths. Matching the
        // upstream algebra avoids a comparatively expensive generic pow().
        match p {
            0.25 => v.sqrt().sqrt(),
            0.5 => v.sqrt(),
            1.0 => v,
            1.5 => v * v.sqrt(),
            1.75 => {
                let sv = v.sqrt();
                v * sv * sv.sqrt()
            }
            2.0 => v * v,
            3.0 => v * v * v,
            4.0 => {
                let v2 = v * v;
                v2 * v2
            }
            _ => v.powf(p),
        }
    }
    #[inline]
    pub fn get_pow_int(&mut self, p: f64, n: i32) -> i32 {
        (self.get_pow(p) * n as f64) as i32
    }
    #[inline]
    pub fn get_raw(&mut self) -> u64 {
        self.advance()
    }
    #[inline]
    pub fn get_tpdf(&mut self) -> f64 {
        let v1 = (self.advance() >> (64 - 53)) as i64;
        let v2 = (self.advance() >> (64 - 53)) as i64;
        (v1 - v2) as f64 * (-53f64).exp2()
    }
    #[inline]
    pub fn get_bit(&mut self) -> i32 {
        if self.bits_left == 0 {
            self.bit_pool = self.advance();
            let b = (self.bit_pool & 1) as i32;
            self.bits_left = 63;
            self.bit_pool >>= 1;
            return b;
        }
        let b = (self.bit_pool & 1) as i32;
        self.bits_left -= 1;
        self.bit_pool >>= 1;
        b
    }
    /// Leva's fast normal generator (as in CBiteRnd::getGaussian).
    fn get_gaussian(&mut self) -> f64 {
        loop {
            let mut u = self.get();
            let mut v = self.get();
            if u == 0.0 || v == 0.0 {
                u = 1.0;
                v = 1.0;
            }
            v = 1.7156 * (v - 0.5);
            let x = u - 0.449871;
            let y = v.abs() + 0.386595;
            let q = x * x + y * (0.19600 * y - 0.25472 * x);
            if q < 0.27597 {
                return v / u;
            }
            if q <= 0.27846 && v * v <= -4.0 * u.ln() * u * u {
                return v / u;
            }
        }
    }
}

fn wrap_param(rnd: &mut BiteRnd, v: i64) -> i64 {
    if v < 0 {
        if v > -INT_MANT_MULT {
            (rnd.get() * (-v) as f64) as i64
        } else {
            (rnd.get_raw() as i64) & INT_MANT_MASK
        }
    } else if v > INT_MANT_MULT {
        if v < INT_MANT_MULT * 2 {
            (INT_MANT_MULT as f64 - rnd.get() * (v - INT_MANT_MULT) as f64) as i64
        } else {
            (rnd.get_raw() as i64) & INT_MANT_MASK
        }
    } else {
        v
    }
}

fn gaussian_int(rnd: &mut BiteRnd, sd: f64, mean: i64) -> i64 {
    loop {
        let r = rnd.get_gaussian() * sd;
        if r > -8.0 && r < 8.0 {
            return (r * INT_MANT_MULT as f64) as i64 + mean;
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive selector (faithful port of CBiteSelBase)
// ---------------------------------------------------------------------------

const SLOT_COUNT: usize = 5;

/// Selector state captured when a candidate is generated. Delayed batch
/// feedback must restore this state instead of updating the last selection
/// subsequently made for another candidate.
#[derive(Clone, Copy)]
struct SelUse {
    index: usize,
    value: i32,
    position: usize,
    slot_id: u8,
    entry_id: u8,
}

struct BiteSel {
    count: usize,
    count_sp: usize,
    count_sp1: usize,
    accum_coeff: f64,
    slot_accums: [f64; SLOT_COUNT],
    slot_ids: [u8; SLOT_COUNT],
    sels: [Vec<i32>; SLOT_COUNT],
    entry_ids: [Vec<u8>; SLOT_COUNT],
    sel: i32,
    sel_id: u8,
    selp: usize,
    slot: usize,
}

impl BiteSel {
    fn new(count: usize) -> Self {
        BiteSel {
            count,
            count_sp: 0,
            count_sp1: 0,
            accum_coeff: 0.0,
            slot_accums: [0.0; SLOT_COUNT],
            slot_ids: [0, 1, 2, 3, 4],
            sels: Default::default(),
            entry_ids: Default::default(),
            sel: 0,
            sel_id: 0,
            selp: 0,
            slot: 0,
        }
    }

    fn reset(&mut self, rnd: &mut BiteRnd, param_count: usize) {
        let sparse_mul = 5usize;
        self.count_sp = self.count * sparse_mul;
        self.count_sp1 = self.count_sp - 1;
        self.accum_coeff = 1.0 / (param_count as f64).sqrt();
        for j in 0..SLOT_COUNT {
            let mut sp = vec![0i32; self.count_sp];
            let mut ids: Vec<u8> = (0..self.count_sp as u8).collect();
            for i in 0..self.count {
                for k in 0..sparse_mul {
                    sp[i * sparse_mul + k] = i as i32;
                }
            }
            for _ in 0..self.count_sp * 5 {
                let i1 = rnd.get_int(self.count_sp as i32) as usize;
                let i2 = rnd.get_int(self.count_sp as i32) as usize;
                sp.swap(i1, i2);
                ids.swap(i1, i2);
            }
            self.sels[j] = sp;
            self.entry_ids[j] = ids;
            self.slot_accums[j] = 0.0;
            self.slot_ids[j] = j as u8;
        }
        self.slot = 0;
        self.select(rnd);
    }

    fn select(&mut self, rnd: &mut BiteRnd) -> i32 {
        self.slot = rnd.get_pow_int(1.5, SLOT_COUNT as i32) as usize;
        self.selp = rnd.get_pow_int(1.5, self.count_sp as i32) as usize;
        self.sel = self.sels[self.slot][self.selp];
        self.sel_id = self.entry_ids[self.slot][self.selp];
        self.sel
    }

    fn incr(&mut self, v: f64) {
        let dp = (-(self.selp as f64) * v * v) as i64;
        if dp < 0 {
            if dp == -1 {
                self.sels[self.slot].swap(self.selp, self.selp - 1);
                self.entry_ids[self.slot].swap(self.selp, self.selp - 1);
            } else {
                let np = (self.selp as i64 + dp) as usize;
                self.sels[self.slot].copy_within(np..self.selp, np + 1);
                self.entry_ids[self.slot].copy_within(np..self.selp, np + 1);
                self.sels[self.slot][np] = self.sel;
                self.entry_ids[self.slot][np] = self.sel_id;
            }
        }
        self.slot_accums[self.slot] += self.accum_coeff;
        if self.slot_accums[self.slot] >= 1.0 {
            let a = self.slot_accums[self.slot] - 1.0;
            if self.slot > 0 {
                self.sels.swap(self.slot, self.slot - 1);
                self.entry_ids.swap(self.slot, self.slot - 1);
                self.slot_ids.swap(self.slot, self.slot - 1);
                self.slot_accums[self.slot] = self.slot_accums[self.slot - 1];
                self.slot_accums[self.slot - 1] = a;
            } else {
                self.slot_accums[self.slot] = a;
            }
        }
    }

    fn decr(&mut self) {
        if self.selp < self.count_sp1 {
            self.sels[self.slot].swap(self.selp, self.selp + 1);
            self.entry_ids[self.slot].swap(self.selp, self.selp + 1);
        }
        self.slot_accums[self.slot] -= self.accum_coeff;
        if self.slot_accums[self.slot] <= -1.0 {
            let a = self.slot_accums[self.slot] + 1.0;
            if self.slot < SLOT_COUNT - 1 {
                self.sels.swap(self.slot, self.slot + 1);
                self.entry_ids.swap(self.slot, self.slot + 1);
                self.slot_ids.swap(self.slot, self.slot + 1);
                self.slot_accums[self.slot] = self.slot_accums[self.slot + 1];
                self.slot_accums[self.slot + 1] = a;
            } else {
                self.slot_accums[self.slot] = a;
            }
        }
    }

    fn captured(&self, index: usize) -> SelUse {
        SelUse {
            index,
            value: self.sel,
            position: self.selp,
            slot_id: self.slot_ids[self.slot],
            entry_id: self.sel_id,
        }
    }

    fn restore(&mut self, selection: SelUse) {
        self.slot = self
            .slot_ids
            .iter()
            .position(|&id| id == selection.slot_id)
            .unwrap_or(0);
        self.sel = selection.value;
        self.sel_id = selection.entry_id;
        self.selp = self.entry_ids[self.slot]
            .iter()
            .position(|&id| id == selection.entry_id)
            .or_else(|| {
                self.sels[self.slot]
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value == selection.value)
                    .min_by_key(|(position, _)| position.abs_diff(selection.position))
                    .map(|(position, _)| position)
            })
            .unwrap_or(selection.position.min(self.count_sp1));
    }

    fn incr_captured(&mut self, selection: SelUse, value: f64) {
        self.restore(selection);
        self.incr(value);
    }

    fn decr_captured(&mut self, selection: SelUse) {
        self.restore(selection);
        self.decr();
    }
}

// Selector indices (full CBiteOpt roster).
mod sel {
    pub const METHOD: usize = 0;
    pub const M1: usize = 1;
    pub const M1A: usize = 2;
    pub const M1B: usize = 3;
    pub const M1C: usize = 4;
    pub const M2: usize = 5;
    pub const M2B: usize = 6;
    pub const POP_CHANGE_INCR: usize = 7;
    pub const POP_CHANGE_DECR: usize = 8;
    pub const PAR_OPT2: usize = 9;
    pub const PAR_POP_P: usize = 10; // [0..4] = 10..14
    pub const ALT_POP_P: usize = 14;
    pub const ALT_POP: usize = 15; // [0..4] = 15..19
    pub const MIN_SOL_PWR: usize = 19; // [0..4] = 19..23
    pub const MIN_SOL_MUL: usize = 23; // [0..4] = 23..27
    pub const GEN1_ALLP: usize = 27;
    pub const GEN1_MOVE_ASYNC: usize = 28;
    pub const GEN1_MOVE_SPAN: usize = 29;
    pub const GEN2_MODE: usize = 30;
    pub const GEN2B_MODE: usize = 31;
    pub const GEN2C_MODE: usize = 32;
    pub const GEN2D_MODE: usize = 33;
    pub const GEN3_MODE: usize = 34;
    pub const GEN4_MIX_FAC: usize = 35;
    pub const GEN5B_MODE: usize = 36;
    pub const GEN7_POW_FAC: usize = 37;
    pub const GEN8_MODE: usize = 38;
    pub const GEN8_NUM: usize = 39;
    pub const GEN8_SPAN: usize = 40; // [0..2] = 40..42
    pub const COUNT: usize = 42;
}

fn build_selectors() -> Vec<BiteSel> {
    let mut s = Vec::with_capacity(sel::COUNT);
    s.push(BiteSel::new(4)); // METHOD
    s.push(BiteSel::new(4)); // M1
    s.push(BiteSel::new(3)); // M1A
    s.push(BiteSel::new(2)); // M1B
    s.push(BiteSel::new(2)); // M1C
    s.push(BiteSel::new(2)); // M2
    s.push(BiteSel::new(4)); // M2B
    s.push(BiteSel::new(2)); // POP_CHANGE_INCR
    s.push(BiteSel::new(2)); // POP_CHANGE_DECR
    s.push(BiteSel::new(2)); // PAR_OPT2
    for _ in 0..4 {
        s.push(BiteSel::new(2)); // PAR_POP_P[gi]
    }
    s.push(BiteSel::new(2)); // ALT_POP_P
    for _ in 0..4 {
        s.push(BiteSel::new(2)); // ALT_POP[gi]
    }
    for _ in 0..4 {
        s.push(BiteSel::new(4)); // MIN_SOL_PWR[gi]
    }
    for _ in 0..4 {
        s.push(BiteSel::new(4)); // MIN_SOL_MUL[gi]
    }
    s.push(BiteSel::new(2)); // GEN1_ALLP
    s.push(BiteSel::new(2)); // GEN1_MOVE_ASYNC
    s.push(BiteSel::new(4)); // GEN1_MOVE_SPAN
    s.push(BiteSel::new(2)); // GEN2_MODE
    s.push(BiteSel::new(2)); // GEN2B_MODE
    s.push(BiteSel::new(2)); // GEN2C_MODE
    s.push(BiteSel::new(2)); // GEN2D_MODE
    s.push(BiteSel::new(4)); // GEN3_MODE
    s.push(BiteSel::new(4)); // GEN4_MIX_FAC
    s.push(BiteSel::new(2)); // GEN5B_MODE
    s.push(BiteSel::new(4)); // GEN7_POW_FAC
    s.push(BiteSel::new(2)); // GEN8_MODE
    s.push(BiteSel::new(4)); // GEN8_NUM
    for _ in 0..2 {
        s.push(BiteSel::new(4)); // GEN8_SPAN[i]
    }
    s
}

// ---------------------------------------------------------------------------
// Population (faithful port of CBitePop, fixed capacity and dynamic active size)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BitePop {
    param_count: usize,
    pop_size: usize,
    params: Vec<Vec<i64>>, // ordered by cost ascending
    costs: Vec<f64>,
    cent: Vec<i64>,
    cur_pop_pos: usize,
    // Dynamic "active window" size (the C++ CurPopSize); <= pop_size.
    cur_pop_size: usize,
    cur_pop_size1: usize,
    cur_pop_size_i: f64,
    need_cent: bool,
    cent_lpc: f64,
}

fn calc_lp1_coeff(count: f64) -> f64 {
    let theta = 2.8 / count;
    let costheta2 = 2.0 - theta.cos();
    1.0 - (costheta2 - (costheta2 * costheta2 - 1.0).sqrt())
}

impl BitePop {
    fn new(param_count: usize, pop_size: usize) -> Self {
        BitePop {
            param_count,
            pop_size,
            params: vec![vec![0i64; param_count]; pop_size],
            costs: vec![1e300; pop_size],
            cent: vec![0i64; param_count],
            cur_pop_pos: 0,
            cur_pop_size: pop_size,
            cur_pop_size1: pop_size - 1,
            cur_pop_size_i: 1.0 / pop_size as f64,
            need_cent: false,
            cent_lpc: calc_lp1_coeff(pop_size as f64),
        }
    }

    fn reset_cur_pop_pos(&mut self) {
        self.cur_pop_pos = 0;
        self.cur_pop_size = self.pop_size;
        self.cur_pop_size1 = self.pop_size - 1;
        self.cur_pop_size_i = 1.0 / self.pop_size as f64;
        self.need_cent = false;
        self.cent_lpc = calc_lp1_coeff(self.pop_size as f64);
    }

    fn incr_cur_pop_size(&mut self) {
        self.cur_pop_size += 1;
        self.cur_pop_size1 += 1;
        self.cur_pop_size_i = 1.0 / self.cur_pop_size as f64;
        self.need_cent = true;
        self.cent_lpc = calc_lp1_coeff(self.cur_pop_size as f64);
    }

    fn decr_cur_pop_size(&mut self) {
        self.cur_pop_size -= 1;
        self.cur_pop_size1 -= 1;
        self.cur_pop_size_i = 1.0 / self.cur_pop_size as f64;
        self.need_cent = true;
        self.cent_lpc = calc_lp1_coeff(self.cur_pop_size as f64);
    }

    /// Insert a solution keeping the population cost-sorted; returns the
    /// insertion index, or `pop_size` if the solution was rejected.
    fn update_pop(&mut self, mut cost: f64, up: &[i64], do_update_centroid: bool) -> usize {
        let ri;
        if self.cur_pop_pos < self.pop_size {
            ri = self.cur_pop_pos;
            if cost.is_nan() {
                cost = 1e300;
            }
        } else {
            ri = self.pop_size - 1;
            if cost.is_nan() || cost >= self.costs[ri] {
                return self.pop_size;
            }
        }
        // binary search for insertion position
        let mut p = 0usize;
        let mut i = ri;
        while p < i {
            let mid = (p + i) >> 1;
            if self.costs[mid] >= cost {
                i = mid;
            } else {
                p = mid + 1;
            }
        }
        if self.cur_pop_pos < self.pop_size {
            self.cur_pop_pos += 1;
        }
        // shift [p..ri) right by one, insert at p
        for k in (p + 1..=ri).rev() {
            self.params.swap(k, k - 1);
            self.costs[k] = self.costs[k - 1];
        }
        self.costs[p] = cost;
        if self.params[p] != up {
            if do_update_centroid {
                for (c, &u) in self.cent.iter_mut().zip(up) {
                    *c += ((u - *c) as f64 * self.cent_lpc) as i64;
                }
                self.params[p].copy_from_slice(up);
            } else {
                self.params[p].copy_from_slice(up);
                self.need_cent = true;
            }
        } else {
            self.need_cent = true;
        }
        p
    }

    fn update_centroid(&mut self) {
        self.need_cent = false;
        let cm = 1.0 / self.pop_size as f64;
        for j in 0..self.param_count {
            let mut sum = 0i128;
            for row in &self.params {
                sum += row[j] as i128;
            }
            self.cent[j] = (sum as f64 * cm) as i64;
        }
    }

    #[inline]
    fn ordered(&self, i: usize) -> &[i64] {
        &self.params[i]
    }

    #[inline]
    fn cur_pop_size(&self) -> usize {
        self.cur_pop_size
    }

    fn get_centroid(&mut self) -> &[i64] {
        if self.need_cent {
            self.update_centroid();
        }
        &self.cent
    }

    /// Copy another (same-size) population wholesale (the C++ `CBitePop::copy`).
    fn copy_from(&mut self, src: &BitePop) {
        for (d, s) in self.params.iter_mut().zip(&src.params) {
            d.copy_from_slice(s);
        }
        self.costs.copy_from_slice(&src.costs);
        self.cent.copy_from_slice(&src.cent);
        self.cur_pop_pos = src.cur_pop_pos;
        self.cur_pop_size = src.cur_pop_size;
        self.cur_pop_size1 = src.cur_pop_size1;
        self.cur_pop_size_i = src.cur_pop_size_i;
        self.need_cent = src.need_cent;
        self.cent_lpc = src.cent_lpc;
    }
}

// ---------------------------------------------------------------------------
// Secondary ("parallel") optimizers (ports of CSpherOpt and CNMSeqOpt)
// ---------------------------------------------------------------------------

/// Wrap a normalized value into `[0, 1]` (the C++ double-branch `wrapParam`).
fn wrap01(rnd: &mut BiteRnd, v: f64) -> f64 {
    if v < 0.0 {
        if v > -1.0 { rnd.get() * -v } else { rnd.get() }
    } else if v > 1.0 {
        if v < 2.0 {
            1.0 - rnd.get() * (v - 1.0)
        } else {
            rnd.get()
        }
    } else {
        v
    }
}

/// Reflect a real value back into `[minv, minv+diffv]` (the C++ `wrapParamReal`).
fn wrap_param_real(rnd: &mut BiteRnd, v: f64, minv: f64, diffv: f64) -> f64 {
    if v < minv {
        if v > minv - diffv {
            minv + rnd.get() * (minv - v)
        } else {
            minv + rnd.get() * diffv
        }
    } else {
        let maxv = minv + diffv;
        if v > maxv {
            if v < maxv + diffv {
                maxv - rnd.get() * (v - maxv)
            } else {
                maxv - rnd.get() * diffv
            }
        } else {
            v
        }
    }
}

/// One evaluated step from a secondary optimizer.
struct ParStep {
    stall: i64,
    cost: f64,
    values: Vec<f64>,
}

/// CSpherOpt — converging hyper-spheroid optimizer (normalized `[0,1]` space).
struct SpherOpt {
    dim: usize,
    pop_size: usize,
    params: Vec<Vec<f64>>, // sorted ascending by cost
    costs: Vec<f64>,
    cur_pop_pos: usize,
    cent: Vec<f64>,
    min_values: Vec<f64>,
    diff_values: Vec<f64>,
    sels: [BiteSel; 3], // CentPow(4), RadPow(4), EvalFac(3)
    apply: Vec<usize>,
    radius: f64,
    eval_fac: f64,
    cure: i32,
    curem: i32,
    do_cent_eval: bool,
    jit_mult: f64,
    jit_offs: f64,
    avg_cost: f64,
    hi_bound: f64,
    stall_count: i64,
    best_cost: f64,
    best_values: Vec<f64>,
}

impl SpherOpt {
    fn new(dim: usize, min_values: Vec<f64>, diff_values: Vec<f64>, pop_size: usize) -> Self {
        let dim_i = 1.0 / dim as f64;
        SpherOpt {
            dim,
            pop_size,
            params: vec![vec![0.0; dim]; pop_size],
            costs: vec![1e300; pop_size],
            cur_pop_pos: 0,
            cent: vec![0.5; dim],
            min_values,
            diff_values,
            sels: [BiteSel::new(4), BiteSel::new(4), BiteSel::new(3)],
            apply: Vec::new(),
            radius: 0.5,
            eval_fac: 2.0,
            cure: 0,
            curem: 0,
            do_cent_eval: false,
            jit_mult: 5.0 * dim_i,
            jit_offs: 1.0 - 5.0 * dim_i * 0.5,
            avg_cost: 0.0,
            hi_bound: 1e300,
            stall_count: 0,
            best_cost: 1e300,
            best_values: vec![0.0; dim],
        }
    }

    fn real_value(&self, norm: &[f64], i: usize) -> f64 {
        self.min_values[i] + self.diff_values[i] * norm[i]
    }

    fn init(&mut self, rnd: &mut BiteRnd, init_params: Option<&[f64]>, radius: f64) {
        self.best_cost = 1e300;
        self.stall_count = 0;
        self.hi_bound = 1e300;
        self.avg_cost = 0.0;
        for sel in self.sels.iter_mut() {
            sel.reset(rnd, self.dim);
        }
        self.cur_pop_pos = 0;
        self.radius = 0.5 * radius;
        self.eval_fac = 2.0;
        self.cure = 0;
        self.curem = (self.pop_size as f64 * self.eval_fac).ceil() as i32;
        match init_params {
            None => {
                self.cent = vec![0.5; self.dim];
                self.do_cent_eval = false;
            }
            Some(ip) => {
                for i in 0..self.dim {
                    self.cent[i] = wrap01(rnd, (ip[i] - self.min_values[i]) / self.diff_values[i]);
                }
                self.do_cent_eval = true;
            }
        }
    }

    fn update_pop(&mut self, cost: f64, params: &[f64]) {
        let ri;
        if self.cur_pop_pos < self.pop_size {
            ri = self.cur_pop_pos;
        } else {
            ri = self.pop_size - 1;
            if cost >= self.costs[ri] {
                return;
            }
        }
        let mut p = 0usize;
        let mut i = ri;
        while p < i {
            let mid = (p + i) >> 1;
            if self.costs[mid] >= cost {
                i = mid;
            } else {
                p = mid + 1;
            }
        }
        if self.cur_pop_pos < self.pop_size {
            self.cur_pop_pos += 1;
        }
        for k in (p + 1..=ri).rev() {
            self.params.swap(k, k - 1);
            self.costs[k] = self.costs[k - 1];
        }
        self.params[p].copy_from_slice(params);
        self.costs[p] = cost;
    }

    fn optimize(&mut self, rnd: &mut BiteRnd, obj: &dyn Objective) -> ParStep {
        let mut params = vec![0.0; self.dim];
        let mut new_values = vec![0.0; self.dim];
        if self.do_cent_eval {
            self.do_cent_eval = false;
            for i in 0..self.dim {
                params[i] = self.cent[i];
                new_values[i] = self.real_value(&self.cent, i);
            }
        } else {
            let mut s2 = 1e-300;
            for pi in params.iter_mut() {
                *pi = rnd.get() - 0.5;
                s2 += *pi * *pi;
            }
            let d = self.radius / s2.sqrt();
            if self.dim > 4 {
                for i in 0..self.dim {
                    params[i] = wrap01(rnd, self.cent[i] + params[i] * d);
                    new_values[i] = self.real_value(&params, i);
                }
            } else {
                for i in 0..self.dim {
                    let m = self.jit_offs + rnd.get() * self.jit_mult;
                    params[i] = wrap01(rnd, self.cent[i] + params[i] * d * m);
                    new_values[i] = self.real_value(&params, i);
                }
            }
        }
        let cost = objective_cost(obj, &new_values);
        self.update_pop(cost, &params);
        if cost <= self.best_cost {
            self.best_cost = cost;
            self.best_values.copy_from_slice(&new_values);
        }
        self.avg_cost += cost;
        self.cure += 1;
        if self.cure >= self.curem {
            self.avg_cost /= self.cure as f64;
            if self.avg_cost < self.hi_bound {
                self.hi_bound = self.avg_cost;
                self.stall_count = 0;
                for &s in &self.apply {
                    self.sels[s].incr(1.0);
                }
            } else {
                self.stall_count += self.cure as i64;
                for &s in &self.apply {
                    self.sels[s].decr();
                }
            }
            self.apply.clear();
            self.cur_pop_pos = 0;
            self.avg_cost = 0.0;
            self.cure = 0;
            self.update(rnd);
            self.curem = (self.pop_size as f64 * self.eval_fac).ceil() as i32;
        }
        ParStep {
            stall: self.stall_count,
            cost,
            values: new_values,
        }
    }

    fn sel(&mut self, idx: usize, rnd: &mut BiteRnd) -> i32 {
        self.apply.push(idx);
        self.sels[idx].select(rnd)
    }

    fn update(&mut self, rnd: &mut BiteRnd) {
        const WCENT: [f64; 4] = [4.5, 6.0, 7.5, 10.0];
        const WRAD: [f64; 4] = [14.0, 16.0, 18.0, 20.0];
        const EVAL_FACS: [f64; 3] = [2.1, 2.0, 1.9];
        let cent_fac = WCENT[self.sel(0, rnd) as usize];
        let rad_fac = WRAD[self.sel(1, rnd) as usize];
        self.eval_fac = EVAL_FACS[self.sel(2, rnd) as usize];

        let lm = 1.0 / self.curem as f64;
        let mut wc = vec![0.0; self.pop_size];
        let mut wr = vec![0.0; self.pop_size];
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for i in 0..self.pop_size {
            let l = 1.0 - i as f64 * lm;
            let v1 = l.powf(cent_fac);
            wc[i] = v1;
            s1 += v1;
            let v2 = l.powf(rad_fac);
            wr[i] = v2;
            s2 += v2;
        }
        s1 = 1.0 / s1;
        s2 = 1.0 / s2;
        for j in 0..self.dim {
            let mut acc = 0.0;
            for i in 0..self.pop_size {
                acc += self.params[i][j] * wc[i] * s1;
            }
            self.cent[j] = acc;
        }
        let mut radius = 0.0;
        for i in 0..self.pop_size {
            let mut s = 0.0;
            for j in 0..self.dim {
                let d = self.params[i][j] - self.cent[j];
                s += d * d;
            }
            radius += s * wr[i];
        }
        self.radius = (radius * s2).sqrt();
    }
}

/// CNMSeqOpt — sequential Nelder-Mead simplex (real-value space).
struct NMSeqOpt {
    n: usize,
    m: usize,
    m1: usize,
    m1i: f64,
    param_count_i: f64,
    x: Vec<Vec<f64>>, // simplex points (real)
    y: Vec<f64>,      // costs
    x0: Vec<f64>,     // centroid
    x1: Vec<f64>,
    x2: Vec<f64>,
    y1: f64,
    xlo: usize,
    xhi: usize,
    xhi2: usize,
    rx: usize, // index of lowest-cost vector during reduction
    rj: usize,
    do_init_evals: bool,
    cur_pop_pos: usize,
    state: NmState,
    stall_count: i64,
    min_values: Vec<f64>,
    diff_values: Vec<f64>,
    best_cost: f64,
    best_values: Vec<f64>,
}

#[derive(Clone, Copy, PartialEq)]
enum NmState {
    Reflection,
    Expansion,
    Contraction,
    Reduction,
}

impl NMSeqOpt {
    fn new(dim: usize, min_values: Vec<f64>, diff_values: Vec<f64>) -> Self {
        let m = (dim + 1) * 4;
        NMSeqOpt {
            n: dim,
            m,
            m1: m - 1,
            m1i: 1.0 / (m - 1) as f64,
            param_count_i: 1.0 / dim as f64,
            x: vec![vec![0.0; dim]; m],
            y: vec![1e300; m],
            x0: vec![0.0; dim],
            x1: vec![0.0; dim],
            x2: vec![0.0; dim],
            y1: 0.0,
            xlo: 0,
            xhi: 0,
            xhi2: 0,
            rx: 0,
            rj: 0,
            do_init_evals: true,
            cur_pop_pos: 0,
            state: NmState::Reflection,
            stall_count: 0,
            min_values,
            diff_values,
            best_cost: 1e300,
            best_values: vec![0.0; dim],
        }
    }

    fn init(&mut self, rnd: &mut BiteRnd, init_params: Option<&[f64]>, radius: f64) {
        self.best_cost = 1e300;
        self.stall_count = 0;
        match init_params {
            Some(ip) => self.x[0].copy_from_slice(ip),
            None => {
                for i in 0..self.n {
                    self.x[0][i] = self.min_values[i] + self.diff_values[i] * 0.5;
                }
            }
        }
        self.xlo = 0;
        let base = self.x[0].clone();
        if radius <= 0.0 {
            for j in 1..self.m {
                for i in 0..self.n {
                    self.x[j][i] = self.min_values[i] + self.diff_values[i] * rnd.get();
                }
            }
        } else {
            let sd = 0.25 * radius;
            for j in 1..self.m {
                for i in 0..self.n {
                    self.x[j][i] = base[i] + self.diff_values[i] * rnd.get_gaussian() * sd;
                }
            }
        }
        self.state = NmState::Reflection;
        self.do_init_evals = true;
        self.cur_pop_pos = 0;
    }

    fn eval(&mut self, rnd: &mut BiteRnd, params: &[f64], obj: &dyn Objective) -> (f64, Vec<f64>) {
        let mut nv = vec![0.0; self.n];
        for i in 0..self.n {
            nv[i] = wrap_param_real(rnd, params[i], self.min_values[i], self.diff_values[i]);
        }
        let cost = objective_cost(obj, &nv);
        if cost <= self.best_cost {
            self.best_cost = cost;
            self.best_values.copy_from_slice(&nv);
        }
        (cost, nv)
    }

    fn find_hi(&mut self) {
        self.xhi2 = if self.y[0] > self.y[1] { 0 } else { 1 };
        self.xhi = 1 - self.xhi2;
        for j in 2..self.m {
            if self.y[j] > self.y[self.xhi] {
                self.xhi2 = self.xhi;
                self.xhi = j;
            } else if self.y[j] > self.y[self.xhi2] {
                self.xhi2 = j;
            }
        }
    }

    fn calc_cent(&mut self) {
        self.find_hi();
        let mut xc = vec![0.0; self.n];
        for (j, xj) in self.x.iter().enumerate() {
            if j == self.xhi {
                continue;
            }
            for i in 0..self.n {
                xc[i] += xj[i];
            }
        }
        for c in xc.iter_mut() {
            *c *= self.m1i;
        }
        self.x0 = xc;
    }

    fn copy(&mut self, ip: &[f64], cost: f64) {
        let replaced_index = self.xhi;
        self.y[replaced_index] = cost;
        self.x[replaced_index].copy_from_slice(ip);
        let replacement = self.x[replaced_index].clone();
        self.find_hi();
        if replaced_index != self.xhi {
            for i in 0..self.n {
                self.x0[i] += (replacement[i] - self.x[self.xhi][i]) * self.m1i;
            }
        }
        self.stall_count = 0;
    }

    fn optimize(&mut self, rnd: &mut BiteRnd, obj: &dyn Objective) -> ParStep {
        #[allow(unused_assignments)]
        let mut out_cost = 0.0;
        let mut out_values = vec![0.0; self.n];

        if self.do_init_evals {
            let xp = self.x[self.cur_pop_pos].clone();
            let (c, v) = self.eval(rnd, &xp, obj);
            self.y[self.cur_pop_pos] = c;
            out_cost = c;
            out_values = v;
            if self.y[self.cur_pop_pos] < self.y[self.xlo] {
                self.xlo = self.cur_pop_pos;
            }
            self.cur_pop_pos += 1;
            if self.cur_pop_pos == self.m {
                self.do_init_evals = false;
                self.calc_cent();
            }
            return ParStep {
                stall: 0,
                cost: out_cost,
                values: out_values,
            };
        }

        self.stall_count += 1;
        let sn = 0.5 * self.param_count_i.sqrt();
        let alpha = 1.0;
        let gamma = 1.5 + sn;
        let rho = -0.75 + sn;
        let sigma = 1.0 - sn;
        let xh = self.x[self.xhi].clone();

        match self.state {
            NmState::Reflection => {
                for i in 0..self.n {
                    self.x1[i] = self.x0[i] + alpha * (self.x0[i] - xh[i]);
                }
                let x1 = self.x1.clone();
                let (c, v) = self.eval(rnd, &x1, obj);
                self.y1 = c;
                out_cost = c;
                out_values = v;
                if self.y1 > self.y[self.xlo] && self.y1 < self.y[self.xhi2] {
                    let x1c = self.x1.clone();
                    self.copy(&x1c, self.y1);
                } else if self.y1 < self.y[self.xlo] {
                    self.state = NmState::Expansion;
                    self.stall_count -= 1;
                } else {
                    self.state = NmState::Contraction;
                }
            }
            NmState::Expansion => {
                for i in 0..self.n {
                    self.x2[i] = self.x0[i] + gamma * (self.x0[i] - xh[i]);
                }
                let x2 = self.x2.clone();
                let (y2, v) = self.eval(rnd, &x2, obj);
                out_cost = y2;
                out_values = v;
                self.xlo = self.xhi;
                if y2 < self.y1 {
                    let x2c = self.x2.clone();
                    self.copy(&x2c, y2);
                } else {
                    let x1c = self.x1.clone();
                    self.copy(&x1c, self.y1);
                }
                self.state = NmState::Reflection;
            }
            NmState::Contraction => {
                for i in 0..self.n {
                    self.x2[i] = self.x0[i] + rho * (self.x0[i] - xh[i]);
                }
                let x2 = self.x2.clone();
                let (y2, v) = self.eval(rnd, &x2, obj);
                out_cost = y2;
                out_values = v;
                if y2 < self.y[self.xhi] {
                    if y2 < self.y[self.xlo] {
                        self.xlo = self.xhi;
                    }
                    let x2c = self.x2.clone();
                    self.copy(&x2c, y2);
                    self.state = NmState::Reflection;
                } else {
                    self.rx = self.xlo;
                    self.rj = 0;
                    self.state = NmState::Reduction;
                }
            }
            NmState::Reduction => {
                if self.rj == self.rx {
                    self.rj += 1;
                }
                let rxv = self.x[self.rx].clone();
                for i in 0..self.n {
                    self.x[self.rj][i] = rxv[i] + sigma * (self.x[self.rj][i] - rxv[i]);
                }
                let xx = self.x[self.rj].clone();
                let (c, v) = self.eval(rnd, &xx, obj);
                self.y[self.rj] = c;
                out_cost = c;
                out_values = v;
                if self.y[self.rj] < self.y[self.xlo] {
                    self.xlo = self.rj;
                    self.stall_count = 0;
                }
                self.rj += 1;
                if self.rj == self.m || (self.rj == self.m1 && self.rj == self.rx) {
                    self.calc_cent();
                    self.state = NmState::Reflection;
                }
            }
        }

        ParStep {
            stall: self.stall_count,
            cost: out_cost,
            values: out_values,
        }
    }
}

// ---------------------------------------------------------------------------
// BiteOpt optimizer core (port of CBiteOpt)
// ---------------------------------------------------------------------------

pub struct BiteOpt {
    param_count: usize,
    param_count_i: f64,
    pop_size: usize,
    min_values: Vec<f64>,
    diff_values: Vec<f64>,
    diff_values_i: Vec<f64>,

    pop: BitePop,
    old_pop: BitePop,
    par_pop_count: usize,
    par_pops: Vec<BitePop>,
    par_opt_pop: BitePop,
    par_opt2_pop: BitePop,
    spher: SpherOpt,
    nmseq: NMSeqOpt,
    use_par_opt: i32,

    sels: Vec<BiteSel>,
    apply_sels: Vec<SelUse>,
    deferred_sels: VecDeque<SelUse>,
    rnd: BiteRnd,

    tmp: Vec<i64>,
    real_tmp: Vec<f64>,
    best_cost: f64,
    best_values: Vec<f64>,
    stall_count: i64,

    // ask/tell (delayed-feedback batching)
    init_queue: VecDeque<Vec<i64>>,
    asked: Vec<Candidate>,

    max_evaluations: u64,
    stopfitness: f64,
    stall_criterion: i32,
    evaluations: u64,
    iterations: i32,
    stop: i32,
}

impl BiteOpt {
    pub fn new(lower: &[f64], upper: &[f64], init: Option<&[f64]>, p: &BiteParams) -> Self {
        validate_bite_inputs(lower, upper, init, p, 1).expect("invalid BiteOpt configuration");
        let param_count = lower.len();
        let pop_size = if p.popsize > 0 {
            p.popsize as usize
        } else {
            9 + param_count * 3
        };
        let min_values = lower.to_vec();
        let diff_values: Vec<f64> = upper
            .iter()
            .zip(lower)
            .map(|(u, l)| (u - l) / INT_MANT_MULT as f64)
            .collect();
        let diff_values_i: Vec<f64> = diff_values.iter().map(|d| 1.0 / d).collect();
        let real_diff: Vec<f64> = upper.iter().zip(lower).map(|(u, l)| u - l).collect();
        let par_pop_count = 4;
        let mut b = BiteOpt {
            param_count,
            param_count_i: 1.0 / param_count as f64,
            pop_size,
            min_values: min_values.clone(),
            diff_values,
            diff_values_i,
            pop: BitePop::new(param_count, pop_size),
            old_pop: BitePop::new(param_count, pop_size),
            par_pop_count,
            par_pops: (0..par_pop_count)
                .map(|_| BitePop::new(param_count, pop_size))
                .collect(),
            par_opt_pop: BitePop::new(param_count, pop_size),
            par_opt2_pop: BitePop::new(param_count, pop_size),
            spher: SpherOpt::new(
                param_count,
                min_values.clone(),
                real_diff.clone(),
                14 + param_count,
            ),
            nmseq: NMSeqOpt::new(param_count, min_values, real_diff),
            use_par_opt: 0,
            sels: build_selectors(),
            apply_sels: Vec::with_capacity(32),
            deferred_sels: VecDeque::with_capacity(2),
            rnd: BiteRnd::new(p.seed.wrapping_add(p.runid as u64)),
            tmp: vec![0; param_count],
            real_tmp: vec![0.0; param_count],
            best_cost: 1e300,
            best_values: vec![0.0; param_count],
            stall_count: 0,
            init_queue: VecDeque::new(),
            asked: Vec::new(),
            max_evaluations: if p.max_evaluations > 0 {
                p.max_evaluations
            } else {
                50_000
            },
            stopfitness: p.stop_fitness,
            stall_criterion: p.stall_criterion.max(0),
            evaluations: 0,
            iterations: 0,
            stop: 0,
        };
        b.init(init);
        b
    }

    fn init(&mut self, init: Option<&[f64]>) {
        let seed_reset: Vec<usize> = (0..self.sels.len()).collect();
        for i in seed_reset {
            self.sels[i].reset(&mut self.rnd, self.param_count);
        }
        self.pop.reset_cur_pop_pos();
        self.old_pop.reset_cur_pop_pos();
        self.par_opt_pop.reset_cur_pop_pos();
        self.par_opt2_pop.reset_cur_pop_pos();
        let init_slice = init.map(|x| x.to_vec());
        self.spher.init(&mut self.rnd, init_slice.as_deref(), 1.0);
        self.nmseq.init(&mut self.rnd, init_slice.as_deref(), 1.0);
        self.use_par_opt = 0;
        self.init_queue.clear();
        self.asked.clear();
        self.deferred_sels.clear();
        let sd = 0.25;
        let mut members: Vec<Vec<i64>> = vec![vec![0i64; self.param_count]; self.pop_size];
        match init {
            None => {
                for member in members.iter_mut() {
                    for slot in member.iter_mut() {
                        let g = gaussian_int(&mut self.rnd, sd, INT_MANT_MULT >> 1);
                        *slot = wrap_param(&mut self.rnd, g);
                    }
                }
            }
            Some(x0) => {
                for i in 0..self.param_count {
                    let v = ((x0[i] - self.min_values[i]) / self.diff_values[i]) as i64;
                    members[0][i] = wrap_param(&mut self.rnd, v);
                }
                for j in 1..self.pop_size {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..self.param_count {
                        let mean = members[0][i];
                        let g = gaussian_int(&mut self.rnd, sd, mean);
                        members[j][i] = wrap_param(&mut self.rnd, g);
                    }
                }
            }
        }
        self.init_queue = members.into_iter().collect();
        self.best_cost = 1e300;
        self.stall_count = 0;
    }

    #[inline]
    fn real_value(&self, params: &[i64], i: usize) -> f64 {
        self.min_values[i] + self.diff_values[i] * params[i] as f64
    }

    #[inline]
    fn take_tmp(&mut self) -> Vec<i64> {
        let mut params = std::mem::take(&mut self.tmp);
        params.resize(self.param_count, 0);
        params.fill(0);
        params
    }

    fn recycle_candidate(&mut self, mut candidate: Candidate) {
        if candidate.enc.capacity() >= self.tmp.capacity() {
            candidate.enc.resize(self.param_count, 0);
            self.tmp = candidate.enc;
        }
        if candidate.real.capacity() >= self.real_tmp.capacity() {
            candidate.real.resize(self.param_count, 0.0);
            self.real_tmp = candidate.real;
        }
        if candidate.sels.capacity() >= self.apply_sels.capacity() {
            candidate.sels.clear();
            self.apply_sels = candidate.sels;
        }
    }

    fn select(&mut self, sel_idx: usize) -> i32 {
        let value = self.sels[sel_idx].select(&mut self.rnd);
        self.apply_sels.push(self.sels[sel_idx].captured(sel_idx));
        value
    }

    fn get_min_sol_index(&mut self, gi: usize, ps: usize) -> usize {
        const PP: [f64; 4] = [0.05, 0.125, 0.25, 0.5];
        const RM: [f64; 4] = [0.0, 0.125, 0.25, 0.5];
        let pwr = self.select(sel::MIN_SOL_PWR + gi) as usize;
        let r = ps as f64 * self.rnd.get_pow(ps as f64 * PP[pwr]);
        let mul = self.select(sel::MIN_SOL_MUL + gi) as usize;
        (r * RM[mul]) as usize
    }

    fn update_best_cost(&mut self, cost: f64, values: &[f64], p: i64) {
        if cost.is_nan() {
            return;
        }
        if p == 0 || (p < 0 && cost <= self.best_cost) {
            self.best_cost = cost;
            self.best_values.copy_from_slice(values);
        }
    }

    // ---- population selection ----

    fn pop_ref(&self, s: PopSel) -> &BitePop {
        match s {
            PopSel::Main => &self.pop,
            PopSel::Par(i) => &self.par_pops[i],
            PopSel::ParOpt => &self.par_opt_pop,
            PopSel::ParOpt2 => &self.par_opt2_pop,
        }
    }

    fn ordered_of(&self, s: PopSel, i: usize) -> Vec<i64> {
        self.pop_ref(s).ordered(i).to_vec()
    }

    fn cur_pop_size_of(&self, s: PopSel) -> usize {
        self.pop_ref(s).cur_pop_size()
    }

    fn select_par_pop(&mut self, gi: usize) -> PopSel {
        if self.select(sel::PAR_POP_P + gi) != 0 {
            PopSel::Par(self.rnd.get_int(self.par_pop_count as i32) as usize)
        } else {
            PopSel::Main
        }
    }

    fn select_alt_pop(&mut self, gi: usize) -> PopSel {
        if self.select(sel::ALT_POP_P) != 0 {
            if self.select(sel::ALT_POP + gi) != 0 {
                if self.par_opt_pop.cur_pop_pos >= self.pop.cur_pop_size() {
                    return PopSel::ParOpt;
                }
            } else if self.par_opt2_pop.cur_pop_pos >= self.pop.cur_pop_size() {
                return PopSel::ParOpt2;
            }
        }
        PopSel::Main
    }

    fn update_par_pop(&mut self, cost: f64, params: &[i64]) {
        let p = self.get_min_dist_par_pop(params);
        self.par_pops[p].update_pop(cost, params, true);
    }

    fn get_min_dist_par_pop(&mut self, params: &[i64]) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::MAX;
        for pi in 0..self.par_pop_count {
            let c = self.par_pops[pi].get_centroid();
            let mut s = 0.0;
            for i in 0..self.param_count {
                let d = (c[i] - params[i]) as f64;
                s += d * d;
            }
            if s <= best_d {
                best_d = s;
                best = pi;
            }
        }
        best
    }

    // ---- solution generators ----

    fn generate_sol1(&mut self) {
        let par = self.select_par_pop(0);
        let par_ps = self.cur_pop_size_of(par);
        let si = self.get_min_sol_index(0, par_ps);
        let mut params = self.take_tmp();
        params.copy_from_slice(self.pop_ref(par).ordered(si));

        let mut a;
        let mut b;
        let mut do_allp = false;
        if self.rnd.get() < 1.8 * self.param_count_i && self.select(sel::GEN1_ALLP) != 0 {
            do_allp = true;
        }
        if do_allp {
            a = 0;
            b = self.param_count;
        } else {
            a = self.rnd.get_int(self.param_count as i32) as usize;
            b = a + 1;
        }

        let r1 = self.rnd.get();
        let r12 = r1 * r1;
        let ims = (r12 * r12 * 48.0) as u32;
        let imask = INT_MANT_MASK >> ims;
        let im2s = self.rnd.get_sqr_int(96);
        let imask2 = if im2s > 63 { 0 } else { INT_MANT_MASK >> im2s };
        let si1 = (r1 * r12 * par_ps as f64) as usize;
        {
            let rp1 = self.pop_ref(par).ordered(si1);
            for i in a..b {
                params[i] = ((params[i] ^ imask) + (rp1[i] ^ imask2)) >> 1;
            }
        }
        if self.rnd.get() < 1.0 - self.param_count_i {
            let ri = self.rnd.get_sqr_int(self.pop.cur_pop_size as i32) as usize;
            if self.rnd.get() < self.param_count_i.sqrt() && self.select(sel::GEN1_MOVE_ASYNC) != 0
            {
                a = 0;
                b = self.param_count;
            }
            const SPAN_MULTS: [f64; 4] = [0.5, 1.5, 2.0, 2.5];
            let m = SPAN_MULTS[self.select(sel::GEN1_MOVE_SPAN) as usize];
            let m1 = self.rnd.get_tpdf() * m;
            let m2 = self.rnd.get_tpdf() * m;
            let rp2 = self.pop.ordered(ri);
            for i in a..b {
                params[i] += ((rp2[i] - params[i]) as f64 * m1) as i64;
                params[i] += ((rp2[i] - params[i]) as f64 * m2) as i64;
            }
        }
        self.tmp = params;
    }

    fn generate_sol2(&mut self) {
        let ps = self.pop.cur_pop_size;
        let ps1 = self.pop.cur_pop_size1;
        let si1 = self.get_min_sol_index(1, ps);
        let si2 = 1 + self.rnd.get_int(ps1 as i32) as usize;
        let si4 = self.rnd.get_sqr_int(ps as i32) as usize;
        let mode = self.select(sel::GEN2_MODE);
        let si1b = (mode != 0).then(|| self.rnd.get_sqr_int(ps as i32) as usize);
        let mut params = self.take_tmp();
        let rp1 = self.pop.ordered(si1);
        let rp2 = self.pop.ordered(si2);
        let rp3 = self.pop.ordered(ps1 - si1);
        let rp4 = self.pop.ordered(si4);
        let rp5 = self.pop.ordered(ps1 - si4);
        if mode == 0 {
            for i in 0..self.param_count {
                params[i] = rp1[i] + (((rp2[i] - rp3[i]) + (rp4[i] - rp5[i])) >> 1);
            }
        } else {
            let rp1b = self.pop.ordered(si1b.unwrap());
            for i in 0..self.param_count {
                params[i] = ((rp1[i] + rp1b[i]) + (rp2[i] - rp3[i]) + (rp4[i] - rp5[i])) >> 1;
            }
        }
        self.tmp = params;
    }

    fn generate_sol2b(&mut self) {
        let ps = self.pop.cur_pop_size;
        let ps1 = self.pop.cur_pop_size1;
        let si1 = self.get_min_sol_index(2, ps);
        let si2 = self.rnd.get_int(ps as i32) as usize;
        let alt = self.select_alt_pop(0);
        let si4 = self.rnd.get_int(ps as i32) as usize;
        let mode = self.select(sel::GEN2B_MODE);
        let si1b = (mode != 0).then(|| self.rnd.get_sqr_int(ps as i32) as usize);
        let mut params = self.take_tmp();
        let rp1 = self.pop.ordered(si1);
        let rp2 = self.pop.ordered(si2);
        let rp3 = self.pop.ordered(ps1 - si2);
        let rp4 = self.pop_ref(alt).ordered(si4);
        let rp5 = self.pop_ref(alt).ordered(ps1 - si4);
        if mode == 0 {
            for i in 0..self.param_count {
                params[i] = rp1[i] + ((rp2[i] - rp3[i]) + (rp4[i] - rp5[i]));
            }
        } else {
            let rp1b = self.pop.ordered(si1b.unwrap());
            for i in 0..self.param_count {
                params[i] = ((rp1[i] + rp1b[i]) >> 1) + (rp2[i] - rp3[i]) + (rp4[i] - rp5[i]);
            }
        }
        self.tmp = params;
    }

    fn generate_sol2c(&mut self) {
        let ps = self.pop.cur_pop_size;
        let mut params = self.take_tmp();
        let si1 = self.rnd.get_pow_int(4.0, (ps / 2) as i32) as usize;
        let pc = 7usize; // 1 + 2*PairCount, PairCount=3
        let mut pop_idx = [0usize; 7];
        pop_idx[0] = si1;
        let mut pp = 1;
        if self.pop.cur_pop_size1 <= pc {
            while pp < pc {
                pop_idx[pp] = self.rnd.get_int(ps as i32) as usize;
                pp += 1;
            }
        } else {
            while pp < pc {
                let sii = self.rnd.get_int(ps as i32) as usize;
                if !pop_idx[..pp].contains(&sii) {
                    pop_idx[pp] = sii;
                    pp += 1;
                }
            }
        }
        for i in 0..self.param_count {
            params[i] = (self.pop.ordered(pop_idx[1])[i] - self.pop.ordered(pop_idx[2])[i])
                + (self.pop.ordered(pop_idx[3])[i] - self.pop.ordered(pop_idx[4])[i])
                + (self.pop.ordered(pop_idx[5])[i] - self.pop.ordered(pop_idx[6])[i]);
        }
        if self.rnd.get_bit() != 0 && self.rnd.get_bit() != 0 {
            let k = self.rnd.get_int(self.param_count as i32) as usize;
            let v1 = (self.rnd.get_raw()
                & self.rnd.get_raw()
                & self.rnd.get_raw()
                & self.rnd.get_raw()
                & self.rnd.get_raw()) as i64
                & INT_MANT_MASK;
            let v2 = (self.rnd.get_raw()
                & self.rnd.get_raw()
                & self.rnd.get_raw()
                & self.rnd.get_raw()
                & self.rnd.get_raw()) as i64
                & INT_MANT_MASK;
            params[k] += v1 - v2;
        }
        let mode = self.select(sel::GEN2C_MODE);
        if mode == 0 {
            let mut si2 = si1 as i64 + self.rnd.get_bit() as i64 * 2 - 1;
            if si2 < 0 {
                si2 = 1;
            }
            for i in 0..self.param_count {
                params[i] =
                    (self.pop.ordered(si1)[i] + self.pop.ordered(si2 as usize)[i] + params[i]) >> 1;
            }
        } else {
            for i in 0..self.param_count {
                params[i] = self.pop.ordered(si1)[i] + (params[i] >> 1);
            }
        }
        self.tmp = params;
    }

    fn generate_sol2d(&mut self) {
        if self.old_pop.cur_pop_pos < 3 {
            self.generate_sol2c();
            return;
        }
        let ps = self.pop.cur_pop_size;
        let i1 = self.rnd.get_sqr_int(ps as i32) as usize;
        let i2 = self.rnd.get_int(ps as i32) as usize;
        let old_pos = self.old_pop.cur_pop_pos;
        let i3 = self.rnd.get_int(old_pos as i32) as usize;
        let mode = self.select(sel::GEN2D_MODE);
        let i1b = (mode != 0).then(|| self.rnd.get_sqr_int(ps as i32) as usize);
        let mut params = self.take_tmp();
        let rp1 = self.pop.ordered(i1);
        let rp2 = self.pop.ordered(i2);
        let rp3 = self.old_pop.ordered(i3);
        if mode == 0 {
            for i in 0..self.param_count {
                params[i] = rp1[i] + ((rp2[i] - rp3[i]) >> 1);
            }
        } else {
            let rp1b = self.pop.ordered(i1b.unwrap());
            for i in 0..self.param_count {
                params[i] = ((rp1[i] + rp1b[i]) + (rp2[i] - rp3[i])) >> 1;
            }
        }
        self.tmp = params;
    }

    fn generate_sol4(&mut self) {
        let alt = self.select_alt_pop(1);
        let par = self.select_par_pop(1);
        let use_size = [self.pop.cur_pop_size, self.cur_pop_size_of(par)];
        let km = 5 + (self.select(sel::GEN4_MIX_FAC) << 1);
        let mut p = self.rnd.get_bit() as usize;
        let idx = self.rnd.get_sqr_int(use_size[p] as i32) as usize;
        let mut params = self.take_tmp();
        params.copy_from_slice(self.pop_ref(if p == 0 { alt } else { par }).ordered(idx));
        for _ in 1..km {
            p = self.rnd.get_bit() as usize;
            let idx = self.rnd.get_sqr_int(use_size[p] as i32) as usize;
            let rp = self.pop_ref(if p == 0 { alt } else { par }).ordered(idx);
            for i in 0..self.param_count {
                params[i] ^= rp[i];
            }
        }
        self.tmp = params;
    }

    fn generate_sol5(&mut self) {
        let par = self.select_par_pop(2);
        let par_ps = self.cur_pop_size_of(par);
        let si1 = self.rnd.get_sqr_int(par_ps as i32) as usize;
        let cp1 = self.ordered_of(par, si1);
        let alt = self.select_alt_pop(2);
        let si2 = self.rnd.get_sqr_int(self.pop.cur_pop_size as i32) as usize;
        let cp2 = self.ordered_of(alt, si2);
        let mut params = self.take_tmp();
        for i in 0..self.param_count {
            let crpl = (self.rnd.get_raw() as i64) & INT_MANT_MASK;
            params[i] = (cp1[i] & crpl) | (cp2[i] & !crpl);
            let bshift = self.rnd.get_int(INT_MANT_BITS as i32);
            params[i] +=
                ((self.rnd.get_bit() as i64) << bshift) - ((self.rnd.get_bit() as i64) << bshift);
        }
        self.tmp = params;
    }

    fn generate_sol5b(&mut self) {
        let par = self.select_par_pop(3);
        let par_ps = self.cur_pop_size_of(par);
        let i0 = self.rnd.get_sqr_int(par_ps as i32) as usize;
        let cp0 = self.ordered_of(par, i0);
        let alt = self.select_alt_pop(3);
        let ps = self.pop.cur_pop_size;
        let ps1 = self.pop.cur_pop_size1;
        let cp1 = if self.rnd.get_bit() != 0 {
            let i1 = ps1 - self.rnd.get_sqr_int(ps as i32) as usize;
            self.ordered_of(alt, i1)
        } else {
            let i1 = self.rnd.get_sqr_int(ps as i32) as usize;
            self.ordered_of(alt, i1)
        };
        let mode = self.select(sel::GEN5B_MODE);
        let mut params = self.take_tmp();
        if mode == 0 {
            for i in 0..self.param_count {
                params[i] = if self.rnd.get_bit() != 0 {
                    cp1[i]
                } else {
                    cp0[i]
                };
            }
        } else {
            let i2 = self.rnd.get_sqr_int(par_ps as i32) as usize;
            let cp2 = self.ordered_of(par, i2);
            let i3 = self.rnd.get_sqr_int(ps as i32) as usize;
            let cp3 = self.ordered_of(alt, i3);
            let cps = [cp0, cp1, cp2, cp3];
            for i in 0..self.param_count {
                let sel = ((self.rnd.get_bit() << 1) | self.rnd.get_bit()) as usize;
                params[i] = cps[sel][i];
            }
        }
        self.tmp = params;
    }

    fn generate_sol6(&mut self) {
        let ps = self.pop.cur_pop_size;
        let r = self.rnd.get_pow(4.0);
        let si = (r * ps as f64) as usize;
        let mut v = [0.0f64; 2];
        let k0 = self.rnd.get_int(self.param_count as i32) as usize;
        let use_second = self.rnd.get_bit() != 0;
        let k1 = use_second.then(|| self.rnd.get_int(self.param_count as i32) as usize);
        let row = self.pop.ordered(si);
        v[0] = self.real_value(row, k0);
        if let Some(k1) = k1 {
            v[1] = self.real_value(row, k1);
        } else {
            v[1] = v[0];
        }
        let m = 1.0 - r * r;
        v[0] *= m;
        v[1] *= m;
        let mut params = self.take_tmp();
        for i in 0..self.param_count {
            let pick = v[self.rnd.get_bit() as usize];
            params[i] = ((pick - self.min_values[i]) * self.diff_values_i[i]) as i64;
        }
        self.tmp = params;
    }

    fn generate_sol7(&mut self) {
        let ps = self.pop.cur_pop_size;
        let use_old = self.old_pop.cur_pop_pos > 2;
        const P: [f64; 4] = [1.5, 1.75, 2.0, 2.25];
        let pwr = P[self.select(sel::GEN7_POW_FAC) as usize];
        let mut params = self.take_tmp();
        for i in 0..self.param_count {
            let rv = self.rnd.get_pow(pwr);
            if use_old && self.rnd.get_bit() != 0 && self.rnd.get_bit() != 0 {
                let idx = (rv * self.old_pop.cur_pop_pos as f64) as usize;
                params[i] = self.old_pop.ordered(idx)[i];
            } else {
                let idx = (rv * ps as f64) as usize;
                params[i] = self.pop.ordered(idx)[i];
            }
        }
        self.tmp = params;
    }

    fn generate_sol8(&mut self) {
        let ps = self.pop.cur_pop_size;
        let mode = self.select(sel::GEN8_MODE);
        let num_sols = 5 + self.select(sel::GEN8_NUM) as usize;
        let mut rp = [0usize; 8];
        let first = self.rnd.get_sqr_int(ps as i32) as usize;
        rp[0] = first;
        let mut params = self.take_tmp();
        params.copy_from_slice(self.pop.ordered(first));
        for slot in rp.iter_mut().take(num_sols).skip(1) {
            let idx = self.rnd.get_sqr_int(ps as i32) as usize;
            *slot = idx;
            let r0 = self.pop.ordered(idx);
            for i in 0..self.param_count {
                params[i] = params[i].wrapping_add(r0[i]);
            }
        }
        let m = 1.0 / num_sols as f64;
        let mut cent = std::mem::take(&mut self.real_tmp);
        cent.resize(self.param_count, 0.0);
        for i in 0..self.param_count {
            cent[i] = params[i] as f64 * m;
            params[i] = cent[i] as i64;
        }
        if mode == 0 {
            const SPANS: [f64; 4] = [1.5, 2.5, 3.5, 4.5];
            let gm = SPANS[self.select(sel::GEN8_SPAN) as usize] * m.sqrt();
            for &index in &rp[..num_sols] {
                let r = self.rnd.get_gaussian() * gm;
                let rj = self.pop.ordered(index);
                for i in 0..self.param_count {
                    params[i] = params[i].wrapping_add(((cent[i] - rj[i] as f64) * r) as i64);
                }
            }
        } else {
            const SPANS: [f64; 4] = [0.5, 1.5, 2.5, 3.5];
            let gm = SPANS[self.select(sel::GEN8_SPAN + 1) as usize];
            for &index in &rp[..num_sols] {
                let r = self.rnd.get_gaussian() * gm;
                let rj = self.pop.ordered(index);
                for i in 0..self.param_count {
                    let delta = (params[i].wrapping_sub(rj[i]) as f64 * r) as i64;
                    params[i] = params[i].wrapping_add(delta);
                }
            }
        }
        self.real_tmp = cent;
        self.tmp = params;
    }

    fn generate_sol9(&mut self) {
        let ps = self.pop.cur_pop_size;
        let ps1 = self.pop.cur_pop_size1;
        let si1 = self.rnd.get_int(ps as i32) as usize;
        let si2 = self.rnd.get_sqr_int(ps as i32) as usize;
        let subtract = self.rnd.get_bit() != 0;
        let mut params = self.take_tmp();
        let rp1 = self.pop.ordered(si1);
        let rp2 = self.pop.ordered(ps1 - si2);
        if subtract {
            for i in 0..self.param_count {
                params[i] = rp1[i] - ((rp2[i] - rp1[i]) >> 1) * (1 - 2 * self.rnd.get_bit() as i64);
            }
        } else {
            for i in 0..self.param_count {
                params[i] = rp1[i] + ((rp2[i] - rp1[i]) >> 1) * (1 - 2 * self.rnd.get_bit() as i64);
            }
        }
        self.tmp = params;
    }

    fn generate_sol10(&mut self) {
        let ps = self.pop.cur_pop_size;
        let ps1 = self.pop.cur_pop_size1;
        let si1 = self.rnd.get_sqr_int(ps as i32) as usize;
        let si2 = self.rnd.get_sqr_int(ps as i32) as usize;
        let mut params = self.take_tmp();
        {
            let rp1 = self.pop.ordered(si1);
            let rp2 = self.pop.ordered(ps1 - si2);
            for i in 0..self.param_count {
                params[i] = (rp1[i] + rp2[i]) >> 1;
            }
        }
        let mut radius = 0.0;
        {
            let rp1 = self.pop.ordered(si1);
            let rp2 = self.pop.ordered(ps1 - si2);
            for i in 0..self.param_count {
                let v1 = (rp1[i] - params[i]) as f64;
                let v2 = (rp2[i] - params[i]) as f64;
                radius += v1 * v1 + 0.45 * v2 * v2;
            }
        }
        let mut s2 = 1e-300;
        let mut nv = std::mem::take(&mut self.real_tmp);
        nv.resize(self.param_count, 0.0);
        for n in nv.iter_mut() {
            *n = self.rnd.get() - 0.5;
            s2 += *n * *n;
        }
        let d = (radius / s2).sqrt();
        for i in 0..self.param_count {
            params[i] += (nv[i] * d) as i64;
        }
        self.real_tmp = nv;
        self.tmp = params;
    }

    fn generate_sol3(&mut self) {
        let ps = self.pop.cur_pop_size;
        let si1 = self.get_min_sol_index(3, ps);
        let si2 = self.rnd.get_sqr_int(ps as i32) as usize;
        let mode = self.select(sel::GEN3_MODE);
        if mode != 0 && self.pop.need_cent {
            self.pop.update_centroid();
        }
        let mut params = self.take_tmp();
        let rp1 = self.pop.ordered(si1);
        let rp2 = self.pop.ordered(si2);
        if mode == 0 {
            for i in 0..self.param_count {
                params[i] = rp1[i] + (rp1[i] - rp2[i]);
            }
        } else {
            const CENT_PROB: [f64; 4] = [0.0, 0.25, 0.5, 0.75];
            let prob = CENT_PROB[mode as usize];
            for i in 0..self.param_count {
                params[i] = if self.rnd.get() < prob {
                    self.pop.cent[i]
                } else {
                    rp1[i] + (rp1[i] - rp2[i])
                };
            }
        }
        self.tmp = params;
    }

    /// Generator that draws from an independently-running parallel optimizer.
    /// Returns the already-evaluated `(cost, real_values)`.
    fn generate_sol_par(&mut self, obj: &dyn Objective) -> (f64, Vec<f64>) {
        if self.use_par_opt == 1 {
            self.use_par_opt = self.select(sel::PAR_OPT2);
        }
        let step;
        let which_pop;
        if self.use_par_opt == 0 {
            step = self.spher.optimize(&mut self.rnd, obj);
            if step.stall > 0 {
                self.use_par_opt = 1;
            }
            if step.stall > self.param_count as i64 * 64 {
                let best = self.best_values.clone();
                self.spher.init(&mut self.rnd, Some(&best), 0.5);
                self.par_opt_pop.reset_cur_pop_pos();
            }
            which_pop = 0;
        } else {
            step = self.nmseq.optimize(&mut self.rnd, obj);
            if step.stall > 0 {
                self.use_par_opt = 0;
            }
            if step.stall > self.param_count as i64 * 16 {
                let best = self.best_values.clone();
                self.nmseq.init(&mut self.rnd, Some(&best), 1.0);
                self.par_opt2_pop.reset_cur_pop_pos();
            }
            which_pop = 1;
        }
        let mut tmp = self.take_tmp();
        for i in 0..self.param_count {
            tmp[i] = ((step.values[i] - self.min_values[i]) * self.diff_values_i[i]) as i64;
        }
        if which_pop == 0 {
            self.par_opt_pop.update_pop(step.cost, &tmp, false);
        } else {
            self.par_opt2_pop.update_pop(step.cost, &tmp, false);
        }
        self.tmp = tmp;
        (step.cost, step.values)
    }

    /// Route the method-selection tree to a generator. Returns a precomputed
    /// cost and exact values when the parallel-optimizer generator (SolPar)
    /// was used; `None`
    /// otherwise (the caller then evaluates). `obj` is `None` in ask/tell mode,
    /// where the method selector is resampled to an ask/tell-compatible
    /// generator without registering a second selector use.
    fn generate(&mut self, obj: Option<&dyn Objective>) -> Option<(f64, Vec<f64>)> {
        let mut method = self.select(sel::METHOD);
        while method == 3 && obj.is_none() {
            method = self.sels[sel::METHOD].select(&mut self.rnd);
        }
        if obj.is_none() {
            let final_method = self.sels[sel::METHOD].captured(sel::METHOD);
            if let Some(selection) = self
                .apply_sels
                .iter_mut()
                .find(|selection| selection.index == sel::METHOD)
            {
                *selection = final_method;
            }
        }
        match method {
            0 => self.generate_sol2(),
            1 => {
                let m1 = self.select(sel::M1);
                match m1 {
                    0 => {
                        let m1a = self.select(sel::M1A);
                        match m1a {
                            0 => self.generate_sol2b(),
                            1 => self.generate_sol2c(),
                            _ => self.generate_sol2d(),
                        }
                    }
                    1 => {
                        if self.select(sel::M1B) != 0 {
                            self.generate_sol4();
                        } else {
                            self.generate_sol5b();
                        }
                    }
                    2 => {
                        if self.select(sel::M1C) != 0 {
                            self.generate_sol5();
                        } else {
                            self.generate_sol10();
                        }
                    }
                    _ => self.generate_sol6(),
                }
            }
            2 => {
                if self.select(sel::M2) != 0 {
                    self.generate_sol1();
                } else {
                    let m2b = self.select(sel::M2B);
                    match m2b {
                        0 => self.generate_sol3(),
                        1 => self.generate_sol7(),
                        2 => self.generate_sol8(),
                        _ => self.generate_sol9(),
                    }
                }
            }
            _ => {
                let (cost, real) =
                    self.generate_sol_par(obj.expect("parallel generator requires an objective"));
                return Some((cost, real));
            }
        }
        None
    }

    fn in_init(&self) -> bool {
        !self.init_queue.is_empty()
    }

    /// Generate one candidate from the current (frozen) population state.
    /// `obj` enables the SolPar generator (parallel optimizers); `None` in
    /// ask/tell mode.
    fn gen_one(&mut self, obj: Option<&dyn Objective>) -> Candidate {
        if let Some(enc) = self.init_queue.pop_front() {
            let mut real = std::mem::take(&mut self.real_tmp);
            real.resize(self.param_count, 0.0);
            for i in 0..self.param_count {
                real[i] = self.real_value(&enc, i);
            }
            return Candidate {
                enc,
                real,
                sels: vec![],
                is_init: true,
                precomputed_cost: None,
            };
        }
        self.apply_sels.clear();
        if let Some(selection) = self.deferred_sels.pop_front() {
            self.apply_sels.push(selection);
        }
        let precomputed = self.generate(obj);
        let mut enc = std::mem::take(&mut self.tmp);
        for e in enc.iter_mut() {
            *e = wrap_param(&mut self.rnd, *e);
        }
        let (precomputed_cost, real) = match precomputed {
            Some((cost, values)) => (Some(cost), values),
            None => {
                let mut values = std::mem::take(&mut self.real_tmp);
                values.resize(self.param_count, 0.0);
                for i in 0..self.param_count {
                    values[i] = self.real_value(&enc, i);
                }
                (None, values)
            }
        };
        let sels = std::mem::take(&mut self.apply_sels);
        Candidate {
            enc,
            real,
            sels,
            is_init: false,
            precomputed_cost,
        }
    }

    fn select_deferred(&mut self, index: usize) -> i32 {
        let value = self.sels[index].select(&mut self.rnd);
        self.deferred_sels
            .push_back(self.sels[index].captured(index));
        value
    }

    /// Apply a candidate's evaluated cost, updating population and selectors.
    /// Returns whether the solution is push-worthy (inserted at
    /// `0 < p <= cur_pop_size1`) for the deep layer.
    fn apply_one(&mut self, cand: &Candidate, cost: f64, collect_push: bool) -> bool {
        let cost = sanitize_cost(cost);
        if cand.is_init {
            let p = self.pop.update_pop(cost, &cand.enc, false) as i64;
            self.update_best_cost(cost, &cand.real, p);
            if self.init_queue.is_empty() && self.pop.cur_pop_pos == self.pop_size {
                self.pop.update_centroid();
                // Seed the diverging parallel populations from the main one.
                for parallel in &mut self.par_pops {
                    parallel.copy_from(&self.pop);
                }
            }
            return false;
        }
        let do_eval = cand.precomputed_cost.is_none();
        let p = self.pop.update_pop(cost, &cand.enc, true);
        let mut push = false;
        if p > self.pop.cur_pop_size1 {
            for &selection in &cand.sels {
                self.sels[selection.index].decr_captured(selection);
            }
            self.stall_count += 1;
            // Dynamic population sizing: grow on failure.
            if do_eval
                && self.pop.cur_pop_size < self.pop_size
                && self.select_deferred(sel::POP_CHANGE_INCR) != 0
            {
                self.pop.incr_cur_pop_size();
            }
        } else {
            self.update_best_cost(cost, &cand.real, p as i64);
            let v = 1.0 - p as f64 * self.pop.cur_pop_size_i;
            for &selection in &cand.sels {
                self.sels[selection.index].incr_captured(selection, v);
            }
            self.stall_count = 0;
            if collect_push && p > 0 {
                push = true;
            }
            // Probabilistically push the current worst into OldPop.
            if self.rnd.get() < self.param_count_i {
                let w = self.pop.cur_pop_size1;
                let worst_cost = self.pop.costs[w];
                self.old_pop
                    .update_pop(worst_cost, self.pop.ordered(w), false);
            }
            // Dynamic population sizing: shrink on success.
            if do_eval
                && self.pop.cur_pop_size > self.pop_size / 2
                && self.select_deferred(sel::POP_CHANGE_DECR) != 0
            {
                self.pop.decr_cur_pop_size();
            }
        }
        // "Diverging populations" technique: feed the nearest parallel pop.
        self.update_par_pop(cost, &cand.enc);
        push
    }

    /// Push an improving solution into this optimizer (deep-mode PushOpt).
    fn push_solution(&mut self, cost: f64, enc: &[i64]) {
        if self.in_init() {
            return;
        }
        self.pop.update_pop(cost, enc, true);
        self.update_par_pop(cost, enc);
    }

    /// One optimization iteration (1 objective evaluation).
    fn optimize_step(&mut self, obj: &impl Objective) {
        let cand = self.gen_one(Some(obj));
        let cost = cand
            .precomputed_cost
            .unwrap_or_else(|| objective_cost(obj, &cand.real));
        self.apply_one(&cand, cost, false);
        self.recycle_candidate(cand);
    }

    /// Deep-mode step: generate + evaluate + apply, returning the stall count
    /// and any push-worthy `(cost, enc)` for the PushOpt.
    fn step_collect(
        &mut self,
        obj: &impl Objective,
        collect_push: bool,
    ) -> (i64, Option<(f64, Vec<i64>)>) {
        let mut cand = self.gen_one(Some(obj));
        let cost = cand
            .precomputed_cost
            .unwrap_or_else(|| objective_cost(obj, &cand.real));
        let should_push = self.apply_one(&cand, cost, collect_push);
        let push = should_push.then(|| (cost, std::mem::take(&mut cand.enc)));
        self.iterations += 1;
        self.evaluations += 1;
        self.recycle_candidate(cand);
        (self.stall_count, push)
    }

    fn best_cost(&self) -> f64 {
        self.best_cost
    }

    fn in_init_phase(&self) -> bool {
        self.in_init()
    }

    fn init_remaining(&self) -> usize {
        self.init_queue.len()
    }

    // ---- ask/tell interface (delayed-feedback batching) ----

    /// Ask for up to `batch` candidate rows (real-valued). During the initial
    /// population fill the batch is capped to the remaining init members.
    pub fn ask(&mut self, batch: usize) -> Vec<Vec<f64>> {
        if !self.asked.is_empty() {
            return self
                .asked
                .iter()
                .map(|candidate| candidate.real.clone())
                .collect();
        }
        if batch == 0 || self.stop != 0 || self.evaluations >= self.max_evaluations {
            return Vec::new();
        }
        let remaining_budget = (self.max_evaluations - self.evaluations) as usize;
        let requested = batch.min(remaining_budget);
        let n = if self.in_init() {
            requested.min(self.init_queue.len())
        } else {
            requested
        };
        for _ in 0..n {
            let cand = self.gen_one(None);
            self.asked.push(cand);
        }
        self.asked.iter().map(|c| c.real.clone()).collect()
    }

    /// Tell the costs for the batch returned by [`ask`](BiteOpt::ask).
    pub fn tell(&mut self, costs: &[f64]) -> i32 {
        if self.asked.is_empty() || costs.len() != self.asked.len() {
            return -1;
        }
        let asked = std::mem::take(&mut self.asked);
        for (cand, &cost) in asked.into_iter().zip(costs) {
            self.apply_one(&cand, cost, false);
            self.iterations += 1;
            self.evaluations += 1;
            self.recycle_candidate(cand);
        }
        self.update_stop();
        self.stop
    }

    pub fn current_batch_size(&self) -> usize {
        self.asked.len()
    }
    pub fn dim(&self) -> usize {
        self.param_count
    }
    pub fn population_size(&self) -> usize {
        self.pop_size
    }
    pub fn stop_code(&self) -> i32 {
        self.stop
    }
    pub fn result_public(&self) -> BiteResult {
        self.result()
    }

    fn update_stop(&mut self) {
        if self.best_cost < self.stopfitness {
            self.stop = 1;
        } else if self.stall_criterion > 0
            && self.stall_count > self.stall_criterion as i64 * 128 * self.param_count as i64
        {
            self.stop = 2;
        }
    }

    /// Run until the evaluation budget or a stop condition is reached.
    pub fn optimize(&mut self, obj: &impl Objective) -> BiteResult {
        while self.evaluations < self.max_evaluations && self.stop == 0 {
            self.optimize_step(obj);
            self.iterations += 1;
            self.evaluations += 1;
            self.update_stop();
        }
        self.result()
    }

    fn result(&self) -> BiteResult {
        BiteResult {
            x: self.best_values.clone(),
            y: self.best_cost,
            evaluations: self.evaluations,
            iterations: self.iterations,
            stop: self.stop,
        }
    }
}

// ---------------------------------------------------------------------------
// Deep multi-population layer (port of CBiteOptDeep / CBiteOptDeepAT)
// ---------------------------------------------------------------------------

/// `M` BiteOpt instances with solution "pushing" between them. `M == 1` is a
/// plain single-population BiteOpt.
pub struct DeepBiteOpt {
    opts: Vec<BiteOpt>,
    m: usize,
    cur_opt: usize,
    push_opt: usize,
    best_opt: usize,
    at_stall_count: i64,
    batch_cur_opt: usize,
    rnd: BiteRnd,
    max_evaluations: u64,
    stopfitness: f64,
    stall_criterion: i32,
    param_count: usize,
    evaluations: u64,
    iterations: i32,
    stop: i32,
}

impl DeepBiteOpt {
    pub fn new(lower: &[f64], upper: &[f64], init: Option<&[f64]>, p: &BiteParams, m: i32) -> Self {
        validate_bite_inputs(lower, upper, init, p, m).expect("invalid deep BiteOpt configuration");
        let m = m.max(1) as usize;
        let opts: Vec<BiteOpt> = (0..m)
            .map(|i| {
                let mut pi = p.clone();
                // Distinct per-optimizer RNG streams for diversity.
                pi.runid = p.runid.wrapping_add((i as i64).wrapping_mul(0x9E3779B1));
                BiteOpt::new(lower, upper, init, &pi)
            })
            .collect();
        let mut d = DeepBiteOpt {
            opts,
            m,
            cur_opt: 0,
            push_opt: 0,
            best_opt: 0,
            at_stall_count: 0,
            batch_cur_opt: 0,
            rnd: BiteRnd::new(p.seed.wrapping_add(p.runid as u64).wrapping_add(0xB17E)),
            max_evaluations: if p.max_evaluations > 0 {
                p.max_evaluations
            } else {
                50_000
            },
            stopfitness: p.stop_fitness,
            stall_criterion: p.stall_criterion.max(0),
            param_count: lower.len(),
            evaluations: 0,
            iterations: 0,
            stop: 0,
        };
        d.pick_push();
        d
    }

    fn pick_push(&mut self) {
        if self.m == 1 {
            self.push_opt = self.cur_opt;
        } else if self.m == 2 {
            self.push_opt = 1 - self.cur_opt;
        } else {
            loop {
                let p = self.rnd.get_int(self.m as i32) as usize;
                if p != self.cur_opt {
                    self.push_opt = p;
                    break;
                }
            }
        }
    }

    fn total_evaluations(&self) -> u64 {
        self.evaluations
    }

    fn best(&self) -> &BiteOpt {
        &self.opts[self.best_opt]
    }

    fn update_stop(&mut self) {
        if self.best().best_cost < self.stopfitness {
            self.stop = 1;
        } else if self.stall_criterion > 0
            && self.at_stall_count > self.stall_criterion as i64 * 128 * self.param_count as i64
        {
            self.stop = 2;
        }
    }

    fn track_best(&mut self, idx: usize) {
        if self.opts[idx].best_cost() <= self.opts[self.best_opt].best_cost() {
            self.best_opt = idx;
        }
    }

    /// One-shot optimization until the (total) evaluation budget or a stop.
    pub fn optimize(&mut self, obj: &impl Objective) -> BiteResult {
        while self.total_evaluations() < self.max_evaluations && self.stop == 0 {
            self.pick_push();
            let opt_idx = self.cur_opt;
            let (sc, push) = self.opts[opt_idx].step_collect(obj, self.m > 1);
            self.evaluations += 1;
            self.iterations += 1;
            if let Some((cost, enc)) = push {
                self.opts[self.push_opt].push_solution(cost, &enc);
                self.opts[opt_idx].tmp = enc;
            }
            self.track_best(opt_idx);
            if sc == 0 {
                self.at_stall_count = 0;
            } else {
                self.cur_opt = self.push_opt;
                self.at_stall_count += 1;
            }
            self.update_stop();
        }
        self.result()
    }

    /// Ask for up to `batch` candidate rows from the current optimizer.
    pub fn ask(&mut self, batch: usize) -> Vec<Vec<f64>> {
        if self.current_batch_size() != 0 {
            return self.opts[self.batch_cur_opt]
                .asked
                .iter()
                .map(|candidate| candidate.real.clone())
                .collect();
        }
        let evaluations = self.total_evaluations();
        if batch == 0 || self.stop != 0 || evaluations >= self.max_evaluations {
            return Vec::new();
        }
        let mut b = batch.min((self.max_evaluations - evaluations) as usize);
        if self.opts[self.cur_opt].in_init_phase() {
            let rem = self.opts[self.cur_opt].init_remaining();
            if b > rem {
                b = rem.max(1);
            }
        }
        self.batch_cur_opt = self.cur_opt;
        self.opts[self.cur_opt].ask(b)
    }

    /// Tell the costs for the last [`ask`](DeepBiteOpt::ask). Candidates are
    /// applied best-first, with solution pushing and optimizer switching.
    pub fn tell(&mut self, costs: &[f64]) -> i32 {
        let opt_idx = self.batch_cur_opt;
        if costs.len() != self.opts[opt_idx].asked.len() || costs.is_empty() {
            return -1;
        }
        let mut asked = std::mem::take(&mut self.opts[opt_idx].asked);
        let mut order: Vec<usize> = (0..asked.len()).collect();
        order.sort_by(|&a, &b| {
            let ca = sanitize_cost(costs[a]);
            let cb = sanitize_cost(costs[b]);
            ca.partial_cmp(&cb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for &i in &order {
            let push = self.opts[opt_idx].apply_one(&asked[i], costs[i], self.m > 1);
            self.opts[opt_idx].iterations += 1;
            self.opts[opt_idx].evaluations += 1;
            self.iterations += 1;
            self.evaluations += 1;
            let sc = self.opts[opt_idx].stall_count;
            if push {
                self.opts[self.push_opt].push_solution(sanitize_cost(costs[i]), &asked[i].enc);
            }
            self.track_best(opt_idx);
            if self.m > 1 {
                if sc == 0 {
                    self.at_stall_count = 0;
                } else {
                    self.at_stall_count += 1;
                    self.cur_opt = self.push_opt;
                    self.pick_push();
                }
            } else {
                self.at_stall_count = sc;
            }
        }
        if let Some(candidate) = asked.pop() {
            self.opts[opt_idx].recycle_candidate(candidate);
        }
        self.update_stop();
        self.stop
    }

    pub fn dim(&self) -> usize {
        self.param_count
    }
    pub fn population_size(&self) -> usize {
        self.opts[0].population_size()
    }
    pub fn current_batch_size(&self) -> usize {
        self.opts[self.batch_cur_opt].current_batch_size()
    }
    pub fn stop_code(&self) -> i32 {
        self.stop
    }

    pub fn result(&self) -> BiteResult {
        let b = self.best();
        BiteResult {
            x: b.best_values.clone(),
            y: b.best_cost,
            evaluations: self.evaluations,
            iterations: self.iterations,
            stop: self.stop,
        }
    }

    pub fn result_public(&self) -> BiteResult {
        self.result()
    }
}

/// Run BiteOpt on a bounded problem. `m` is the "deep" depth (number of
/// populations); `m <= 1` is a plain single-population run.
pub fn optimize_bite(
    obj: &impl Objective,
    lower: &[f64],
    upper: &[f64],
    init: Option<&[f64]>,
    p: &BiteParams,
    m: i32,
) -> BiteResult {
    let mut opt = DeepBiteOpt::new(lower, upper, init, p, m);
    opt.optimize(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
    fn rosen(x: &[f64]) -> f64 {
        (0..x.len() - 1)
            .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
            .sum()
    }
    fn rastrigin(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos())
                .sum::<f64>()
    }

    fn run(obj: impl Objective, dim: usize, seed: u64, evals: u64) -> f64 {
        let params = BiteParams {
            max_evaluations: evals,
            seed,
            ..Default::default()
        };
        optimize_bite(&obj, &vec![-5.0; dim], &vec![5.0; dim], None, &params, 1).y
    }

    #[test]
    fn deep_minimizes_rosenbrock() {
        let params = BiteParams {
            max_evaluations: 30000,
            seed: 4,
            ..Default::default()
        };
        // depth M = 3
        let r = optimize_bite(
            &(rosen as fn(&[f64]) -> f64),
            &[-5.0; 6],
            &[5.0; 6],
            None,
            &params,
            3,
        );
        assert!(r.y < 1e-2, "deep rosen: {}", r.y);
    }

    #[test]
    fn minimizes_sphere() {
        assert!(run(sphere as fn(&[f64]) -> f64, 5, 1, 15000) < 1e-6);
    }

    #[test]
    fn minimizes_rosenbrock() {
        let mut v: Vec<f64> = (0..5)
            .map(|s| run(rosen as fn(&[f64]) -> f64, 5, s, 30000))
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(v[2] < 1e-2, "rosen median too large: {v:?}");
    }

    #[test]
    fn minimizes_rastrigin() {
        let mut v: Vec<f64> = (0..5)
            .map(|s| run(rastrigin as fn(&[f64]) -> f64, 5, s, 30000))
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(v[2] < 5.0, "rastrigin median too large: {v:?}");
    }

    #[test]
    fn ask_tell_converges() {
        let params = BiteParams {
            max_evaluations: 15000,
            seed: 7,
            ..Default::default()
        };
        let mut opt = BiteOpt::new(&[-5.0; 5], &[5.0; 5], None, &params);
        while opt.evaluations < 15000 && opt.stop == 0 {
            let xs = opt.ask(8);
            let ys: Vec<f64> = xs.iter().map(|x| sphere(x)).collect();
            opt.tell(&ys);
        }
        assert!(
            opt.result_public().y < 1e-4,
            "ask/tell: {}",
            opt.result_public().y
        );
    }

    #[test]
    fn validates_configuration() {
        let params = BiteParams::default();
        assert!(validate_bite_inputs(&[], &[], None, &params, 1).is_err());
        assert!(validate_bite_inputs(&[0.0], &[1.0, 2.0], None, &params, 1).is_err());
        assert!(validate_bite_inputs(&[1.0], &[1.0], None, &params, 1).is_err());
        assert!(validate_bite_inputs(&[2.0], &[1.0], None, &params, 1).is_err());
        assert!(validate_bite_inputs(&[f64::NAN], &[1.0], None, &params, 1).is_err());
        assert!(validate_bite_inputs(&[0.0], &[f64::INFINITY], None, &params, 1).is_err());
        assert!(validate_bite_inputs(&[-f64::MAX], &[f64::MAX], None, &params, 1).is_err());
        assert!(validate_bite_inputs(&[0.0], &[1.0], Some(&[0.5, 0.5]), &params, 1).is_err());
        assert!(validate_bite_inputs(&[0.0], &[1.0], Some(&[f64::NAN]), &params, 1).is_err());
        assert!(validate_bite_inputs(&[0.0], &[1.0], None, &params, 37).is_err());

        let invalid_pop = BiteParams {
            popsize: 3,
            ..Default::default()
        };
        assert!(validate_bite_inputs(&[0.0], &[1.0], None, &invalid_pop, 1).is_err());
        let invalid_stop = BiteParams {
            stop_fitness: f64::NAN,
            ..Default::default()
        };
        assert!(validate_bite_inputs(&[0.0], &[1.0], None, &invalid_stop, 1).is_err());

        // The upstream API treats non-positive depth and population size as
        // requests for their defaults.
        let defaults = BiteParams {
            popsize: -1,
            ..Default::default()
        };
        assert!(validate_bite_inputs(&[0.0], &[1.0], Some(&[0.5]), &defaults, -1).is_ok());
    }

    #[test]
    fn helper_edge_paths_are_bounded_and_stable() {
        let mut rnd = BiteRnd::new(101);
        for value in [
            wrap_param(&mut rnd, -INT_MANT_MULT * 2),
            wrap_param(&mut rnd, INT_MANT_MULT * 3),
        ] {
            assert!((0..=INT_MANT_MULT).contains(&value));
        }
        for value in [wrap01(&mut rnd, -2.0), wrap01(&mut rnd, 3.0)] {
            assert!((0.0..=1.0).contains(&value));
        }
        for value in [
            wrap_param_real(&mut rnd, -30.0, -5.0, 10.0),
            wrap_param_real(&mut rnd, 30.0, -5.0, 10.0),
        ] {
            assert!((-5.0..=5.0).contains(&value));
        }

        let mut population = BitePop::new(2, 4);
        population.update_pop(f64::NAN, &[1, 2], false);
        assert_eq!(population.costs[0], BAD_COST);
        population.need_cent = true;
        assert_eq!(population.get_centroid().len(), 2);

        let mut selector = BiteSel::new(2);
        selector.reset(&mut rnd, 2);
        let fallback = SelUse {
            index: 0,
            value: selector.sels[0][0],
            position: 0,
            slot_id: selector.slot_ids[0],
            entry_id: u8::MAX,
        };
        selector.restore(fallback);
        assert_eq!(selector.sel, fallback.value);
    }

    #[test]
    fn direct_driver_initial_guess_defaults_and_getters() {
        let params = BiteParams {
            max_evaluations: 20,
            stop_fitness: f64::INFINITY,
            seed: 102,
            ..Default::default()
        };
        let mut opt = BiteOpt::new(&[-1.0; 2], &[1.0; 2], Some(&[0.2, -0.2]), &params);
        assert_eq!(opt.dim(), 2);
        assert_eq!(opt.population_size(), 15);
        assert_eq!(opt.stop_code(), 0);
        let result = opt.optimize(&(sphere as fn(&[f64]) -> f64));
        assert_eq!(result.evaluations, 1);
        assert_eq!(result.stop, 1);

        let default_budget = BiteParams {
            max_evaluations: 0,
            ..Default::default()
        };
        let opt = BiteOpt::new(&[-1.0], &[1.0], None, &default_budget);
        assert_eq!(opt.max_evaluations, 50_000);
        let mut deep = DeepBiteOpt::new(&[-1.0], &[1.0], None, &default_budget, 1);
        assert_eq!(deep.max_evaluations, 50_000);
        assert_eq!(deep.dim(), 1);
        assert_eq!(deep.population_size(), 12);
        assert_eq!(deep.stop_code(), 0);
        let asked = deep.ask(2);
        assert_eq!(deep.ask(5), asked);
    }

    #[test]
    fn ask_tell_enforces_batches_and_budget() {
        let params = BiteParams {
            popsize: 4,
            max_evaluations: 5,
            seed: 11,
            ..Default::default()
        };
        let mut opt = BiteOpt::new(&[-1.0; 2], &[1.0; 2], None, &params);

        assert!(opt.ask(0).is_empty());
        assert_eq!(opt.tell(&[0.0]), -1);

        let first = opt.ask(3);
        assert_eq!(first.len(), 3);
        assert_eq!(opt.ask(99), first);
        assert_eq!(opt.tell(&[1.0, 2.0]), -1);
        assert_eq!(opt.current_batch_size(), 3);
        assert_eq!(opt.tell(&[1.0, 2.0, 3.0]), 0);

        // Initial population fill is not mixed with generated candidates.
        let second = opt.ask(8);
        assert_eq!(second.len(), 1);
        assert_eq!(opt.tell(&[4.0]), 0);
        let last = opt.ask(8);
        assert_eq!(last.len(), 1);
        opt.tell(&[5.0]);

        assert!(opt.ask(8).is_empty());
        let result = opt.result_public();
        assert_eq!(result.evaluations, 5);
        assert_eq!(result.iterations, 5);
    }

    #[test]
    fn ask_tell_records_the_resampled_compatible_method() {
        let params = BiteParams {
            popsize: 4,
            max_evaluations: 300,
            seed: 111,
            ..Default::default()
        };
        let mut opt = BiteOpt::new(&[-1.0; 2], &[1.0; 2], None, &params);
        let initial = opt.ask(4);
        opt.tell(&vec![1.0; initial.len()]);

        for cost in 2..250 {
            let cand = opt.gen_one(None);
            let method = cand
                .sels
                .iter()
                .find(|selection| selection.index == sel::METHOD)
                .unwrap();
            assert_ne!(method.value, 3);
            opt.apply_one(&cand, cost as f64, false);
            opt.recycle_candidate(cand);
        }
    }

    #[test]
    fn deep_ask_tell_enforces_budget_and_sanitizes_costs() {
        let params = BiteParams {
            popsize: 4,
            max_evaluations: 7,
            seed: 12,
            ..Default::default()
        };
        let mut opt = DeepBiteOpt::new(&[-1.0; 2], &[1.0; 2], None, &params, 3);

        assert_eq!(opt.tell(&[0.0]), -1);
        let first = opt.ask(9);
        assert_eq!(first.len(), 4);
        assert_eq!(opt.tell(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY]), -1);
        assert_eq!(opt.current_batch_size(), 4);
        assert_eq!(
            opt.tell(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]),
            0
        );

        let second = opt.ask(9);
        assert_eq!(second.len(), 3);
        opt.tell(&[3.0, 2.0, 1.0]);
        assert!(opt.ask(1).is_empty());

        let result = opt.result_public();
        assert_eq!(result.evaluations, 7);
        assert_eq!(result.iterations, 7);
        assert!(result.y.is_finite());
        assert_eq!(result.y, 1.0);
    }

    #[test]
    fn non_finite_objective_values_are_rejected() {
        let params = BiteParams {
            max_evaluations: 25,
            seed: 13,
            ..Default::default()
        };
        let result = optimize_bite(
            &(|_: &[f64]| f64::NAN),
            &[-1.0; 2],
            &[1.0; 2],
            None,
            &params,
            2,
        );
        assert_eq!(result.y, BAD_COST);
        assert_eq!(result.evaluations, 25);
        assert!(result.x.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn stop_fitness_terminates_immediately() {
        let params = BiteParams {
            max_evaluations: 100,
            stop_fitness: f64::INFINITY,
            seed: 14,
            ..Default::default()
        };
        let result = optimize_bite(
            &(sphere as fn(&[f64]) -> f64),
            &[-1.0; 2],
            &[1.0; 2],
            None,
            &params,
            1,
        );
        assert_eq!(result.evaluations, 1);
        assert_eq!(result.stop, 1);
    }

    #[test]
    fn delayed_selector_feedback_restores_the_exact_slot() {
        let mut rnd = BiteRnd::new(15);
        let mut selector = BiteSel::new(3);
        selector.reset(&mut rnd, 4);

        selector.slot = 2;
        selector.selp = 3;
        selector.sel = selector.sels[2][3];
        selector.sel_id = selector.entry_ids[2][3];
        let captured = selector.captured(7);
        let captured_id = captured.slot_id;

        // Simulate later candidates overwriting the selector's current state.
        selector.slot = 4;
        selector.selp = 0;
        selector.sel = selector.sels[4][0];
        selector.sel_id = selector.entry_ids[4][0];
        let later_id = selector.slot_ids[4];
        selector.incr_captured(captured, 1.0);

        let captured_slot = selector
            .slot_ids
            .iter()
            .position(|&id| id == captured_id)
            .unwrap();
        let later_slot = selector
            .slot_ids
            .iter()
            .position(|&id| id == later_id)
            .unwrap();
        assert_eq!(selector.slot_accums[captured_slot], 0.5);
        assert_eq!(selector.slot_accums[later_slot], 0.0);
        selector.decr_captured(captured);
        assert_eq!(selector.slot_accums[captured_slot], 0.0);
        assert_eq!(
            selector.entry_ids[captured_slot]
                .iter()
                .position(|&id| id == captured.entry_id),
            Some(1)
        );
    }

    #[test]
    fn nelder_mead_copy_keeps_centroid_consistent() {
        let mut nm = NMSeqOpt::new(1, vec![-10.0], vec![20.0]);
        for i in 0..nm.m {
            nm.x[i][0] = i as f64;
            nm.y[i] = i as f64;
        }
        nm.calc_cent();
        assert_eq!(nm.xhi, nm.m - 1);
        nm.copy(&[-1.0], -1.0);

        let expected =
            nm.x.iter()
                .enumerate()
                .filter(|(i, _)| *i != nm.xhi)
                .map(|(_, x)| x[0])
                .sum::<f64>()
                * nm.m1i;
        assert!((nm.x0[0] - expected).abs() < 1e-14);
    }

    #[test]
    fn dynamic_population_size_stays_within_upstream_limits() {
        let params = BiteParams {
            popsize: 12,
            max_evaluations: 600,
            seed: 16,
            ..Default::default()
        };
        let mut opt = BiteOpt::new(&[-2.0; 3], &[2.0; 3], None, &params);
        let mut next_good = -1.0;
        while opt.evaluations < params.max_evaluations {
            let xs = opt.ask(1);
            let cost = if opt.evaluations.is_multiple_of(2) {
                next_good -= 1.0;
                next_good
            } else {
                BAD_COST
            };
            opt.tell(&vec![cost; xs.len()]);
            assert!(opt.pop.cur_pop_size >= opt.pop_size / 2);
            assert!(opt.pop.cur_pop_size <= opt.pop_size);
        }
    }

    #[test]
    fn secondary_optimizers_make_progress() {
        let objective = sphere as fn(&[f64]) -> f64;
        let mut rnd = BiteRnd::new(17);
        let mut spher = SpherOpt::new(3, vec![-5.0; 3], vec![10.0; 3], 17);
        spher.init(&mut rnd, None, 1.0);
        for _ in 0..2_000 {
            spher.optimize(&mut rnd, &objective);
        }
        assert!(spher.best_cost < 1e-5, "spher: {}", spher.best_cost);

        let mut nm = NMSeqOpt::new(3, vec![-5.0; 3], vec![10.0; 3]);
        nm.init(&mut rnd, None, 1.0);
        for _ in 0..2_000 {
            nm.optimize(&mut rnd, &objective);
        }
        assert!(nm.best_cost < 1e-8, "nelder-mead: {}", nm.best_cost);
    }
}
