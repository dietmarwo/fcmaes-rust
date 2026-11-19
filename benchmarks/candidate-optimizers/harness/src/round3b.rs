//! Round 3, second batch (`ROUND3B.md`): three published problems with
//! unordered categorical variables, their type tables and decoders.
//!
//! # Problems
//!
//! - `welded_beam_mixed`: the mixed-variable welded beam of K. Deb and
//!   M. Goyal (GeneAS, 1996), as printed in K. Deb and M. Goyal, "Optimizing
//!   engineering designs using a combined genetic search", Proc. 7th ICGA
//!   (1997), section 4.4 and tables 4 and 5, with the stress, deflection and
//!   buckling formulas that paper takes from Reklaitis, Ravindran and
//!   Ragsdell (1983). See [`welded_beam`] for every constant's origin.
//! - `cantilever_mixed`: `MixedCantileverBeam` of the Surrogate Modeling
//!   Toolbox (SMT 2.15.1, `smt/problems/mixed_cantilever_beam.py`,
//!   BSD-3-Clause, Copyright (c) 2017, SMT developers; problem of Saves,
//!   Diouane, Bartoli, Lefebvre and Morlier, "A mixed-categorical correlation
//!   kernel for Gaussian process", 2023). See [`cantilever`].
//! - `pest_control`: the pest-control benchmark of Oh, Tomczak, Gavves and
//!   Welling (COMBO, NeurIPS 2019; QUVA-Lab/COMBO, BSD-2-Clause) in the
//!   fixed-seed deterministic form of the Bounce suite (Papenmeier, Nardi and
//!   Poloczek, NeurIPS 2023; `bounce/util/pest_control.py`, MIT, Copyright
//!   (c) 2024 Leonard Papenmeier), identical to Casmopolitan's
//!   `test_funcs/pest.py` (MIT), with Bounce's default seed 0. The NumPy
//!   legacy `RandomState` draws it uses are ported exactly ([`pest`]).
//!
//! # Encoding
//!
//! ROUND3B.md section 2: every discrete coordinate is one real coordinate in
//! `[0, 1]` split into equal intervals, one per level, in the order of the
//! source. Level `k` of `n` covers `[k/n, (k+1)/n)`, the last level includes
//! the upper bound, as `categorical_index` of `examples/src/encoding.rs`
//! (`docs/combinatorial-encodings.md`): `min(floor(u n), n - 1)`
//! ([`level_of`]). Continuous coordinates keep the source's bounds.
//!
//! # Constraints
//!
//! The welded beam is constrained. Its cost is the objective plus 10^6 times
//! the sum of the positive parts of the four constraints of the source, each
//! written as `g(x) <= 0` and divided by its limit
//! ([`welded_beam::evaluate`]). The two other problems are unconstrained.

use std::sync::Arc;

use crate::problems::Problem;
use crate::round3::{Kind, Probe, TypeTable, continuous, equal_bins};

/// The round-3b campaign problems (`--problems round3b`), in campaign order.
pub const PROBLEMS: [&str; 3] = ["welded_beam_mixed", "cantilever_mixed", "pest_control"];

/// Every round-3b problem has an unordered categorical coordinate.
pub const CATEGORICAL_PROBLEMS: [&str; 3] = PROBLEMS;

/// Problems of ROUND3B.md section 2 that did not enter, with the reason.
pub const DROPPED: [(&str, &str); 0] = [];

/// Penalty weight of the scalarized constraints (ROUND3B.md section 2).
pub const PENALTY: f64 = 1.0e6;

/// Level of a unit coordinate among `count` equal intervals:
/// `min(floor(u count), count - 1)` after clamping to `[0, 1]`, the rule of
/// `categorical_index` in `examples/src/encoding.rs`.
pub fn level_of(value: f64, count: usize) -> usize {
    ((value.clamp(0.0, 1.0) * count as f64).floor() as usize).min(count - 1)
}

// ============================================================ welded beam

/// The mixed-variable welded beam of Deb and Goyal.
///
/// Source: K. Deb and M. Goyal, "Optimizing engineering designs using a
/// combined genetic search", Proc. 7th Int. Conf. on Genetic Algorithms
/// (1997), section 4.4 ("Welded beam design"), table 4 (materials) and
/// table 5 (GeneAS solution); the same authors' GeneAS paper (Computer
/// Science and Informatics 26(4), 1996) introduced the problem. That section
/// prints the variables, the objective, the four constraints, the material
/// table, `A` and `J` for both weld types, `F`, `L` and `delta_max`; it
/// takes `tau'`, `tau''`, `M`, `R`, `sigma`, `delta` and `P_c` from Reklaitis,
/// Ravindran and Ragsdell, "Engineering Optimization" (1983), the classic
/// welded beam of Ragsdell and Phillips (1976).
///
/// Variables in the source's order: `x1` weld type (0 two-sided, 1
/// four-sided), `x2` material (steel, cast iron, aluminum, brass), `x3 = h`,
/// `x4 = t`, `x5 = b` (discrete, multiples of 0.0625 in) and `x6 = l`
/// (continuous).
///
/// Bounds: the source gives the ranges of its initial population, `0.0625 <=
/// h <= 2`, `2 <= t <= 20`, `0.0625 <= b <= 2`, and lets `l` take any real
/// value (its range `2 <= l <= 20` is for initialization only; its own optimum
/// has `l = 1.6849`). The box takes `h`, `t` and `b` from those ranges and
/// `0.1 <= l <= 10` from Tran, Tran and Wang (2019), "Constrained
/// mixed-integer Gaussian mixture Bayesian optimization ...", which restates
/// this problem with these bounds (they are also the classic welded beam's
/// bounds of `l`).
///
/// Verification (`tests/round3b.rs`): the GeneAS solution of table 5 (weld
/// four-sided, steel, `h = 0.1875`, `t = 8.25`, `b = 0.25`, `l = 1.6849`) gives
/// the printed slacks `g1 = 380.1660` (380.1653 in double precision, 380.1660
/// in single precision), `g2 = 402.0473` (402.0472) and `g4 = 0.1621` (0.1339
/// at the printed `l`; 0.1621 at `l = 1.684907`, which prints as 1.6849); the
/// printed cost 1.9422 is reproduced with the classic steel coefficients
/// 0.10471 and 0.04811 (1.94221), of which table 4 prints the rounded values
/// 0.1047 and 0.0481 (1.94188 with those). The printed `g3 = 0.2346` is not
/// reproduced: the deflection formula gives 0.2344.
pub mod welded_beam {
    /// Load `F` (lb), Deb and Goyal (1997), section 4.4.
    pub const F: f64 = 6_000.0;
    /// Beam length `L` (in), Deb and Goyal (1997), section 4.4.
    pub const L: f64 = 14.0;
    /// Allowed end deflection `delta_max` (in), Deb and Goyal (1997).
    pub const DELTA_MAX: f64 = 0.25;
    /// Allowed shear stress as a fraction of `S` (`g4 = 0.577 S - tau`).
    pub const SHEAR_FRACTION: f64 = 0.577;
    /// Weld factor printed in `A` and `J` (1.414, as printed, not sqrt 2).
    pub const WELD: f64 = 1.414;
    /// Buckling coefficient of `P_c` (Reklaitis et al. 1983).
    pub const BUCKLING: f64 = 4.013;
    /// Step of the discrete dimensions `h`, `t`, `b` (in).
    pub const STEP: f64 = 0.0625;
    /// Levels of `h` and `b`: 0.0625, 0.125, ..., 2.0.
    pub const THIN_LEVELS: usize = 32;
    /// Levels of `t`: 2.0, 2.0625, ..., 20.0.
    pub const WIDTH_LEVELS: usize = 289;
    /// Lowest value of `t`.
    pub const WIDTH_MIN: f64 = 2.0;
    /// Bounds of the weld length `l` (Tran, Tran and Wang 2019).
    pub const LENGTH_BOUNDS: (f64, f64) = (0.1, 10.0);

    /// One row of table 4 of Deb and Goyal (1997).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Material {
        /// Name.
        pub name: &'static str,
        /// Allowed normal stress `S` (psi; table: kpsi).
        pub s: f64,
        /// Young's modulus `E` (psi; table: Mpsi).
        pub e: f64,
        /// Shear modulus `G` (psi; table: Mpsi).
        pub g: f64,
        /// Weld cost coefficient `c1`.
        pub c1: f64,
        /// Bar cost coefficient `c2`.
        pub c2: f64,
    }

    /// Table 4 of Deb and Goyal (1997), in the order of the material code
    /// (00 steel, 01 cast iron, 10 aluminum, 11 brass). The same table is
    /// restated by Tran, Tran and Wang (2019), table 1.
    pub const MATERIALS: [Material; 4] = [
        Material {
            name: "steel",
            s: 30_000.0,
            e: 30.0e6,
            g: 12.0e6,
            c1: 0.1047,
            c2: 0.0481,
        },
        Material {
            name: "cast iron",
            s: 8_000.0,
            e: 14.0e6,
            g: 6.0e6,
            c1: 0.0489,
            c2: 0.0224,
        },
        Material {
            name: "aluminum",
            s: 5_000.0,
            e: 10.0e6,
            g: 4.0e6,
            // A published cost coefficient; its closeness to pi/6 is a coincidence.
            #[allow(clippy::approx_constant)]
            c1: 0.5235,
            c2: 0.2405,
        },
        Material {
            name: "brass",
            s: 8_000.0,
            e: 16.0e6,
            g: 6.0e6,
            c1: 0.5584,
            c2: 0.2566,
        },
    ];

    /// A decoded design.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Design {
        /// `x1`: weld on four sides (`true`) or on two sides.
        pub four_sided: bool,
        /// `x2`: index into [`MATERIALS`].
        pub material: usize,
        /// `x3 = h`, weld thickness (in).
        pub h: f64,
        /// `x4 = t`, beam width (in).
        pub t: f64,
        /// `x5 = b`, beam thickness (in).
        pub b: f64,
        /// `x6 = l`, weld length (in).
        pub l: f64,
    }

    /// Quantities of a design.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Evaluation {
        /// Cost `f = (1 + c1) h^2 (l + x1 t) + c2 t b (L + l)`.
        pub cost: f64,
        /// Bending stress `sigma`.
        pub sigma: f64,
        /// Buckling load `P_c`.
        pub pc: f64,
        /// End deflection `delta`.
        pub delta: f64,
        /// Weld shear stress `tau`.
        pub tau: f64,
        /// The source's constraints in `g >= 0` form: `g1 = S - sigma`,
        /// `g2 = P_c - F`, `g3 = delta_max - delta`, `g4 = 0.577 S - tau`.
        pub slack: [f64; 4],
        /// The same constraints as `g(x) <= 0`, divided by their limits:
        /// `sigma/S - 1`, `1 - P_c/F`, `delta/delta_max - 1`,
        /// `tau/(0.577 S) - 1`, each computed as `(value - limit) / limit`.
        pub normalized: [f64; 4],
        /// Scalarized cost: `cost + 1e6 * sum(max(0, normalized))`.
        pub penalized: f64,
    }

    /// Decode a real vector of the box: discrete coordinates by
    /// [`super::level_of`], `l` as given.
    pub fn decode(x: &[f64]) -> Design {
        use super::level_of;
        Design {
            four_sided: level_of(x[0], 2) == 1,
            material: level_of(x[1], MATERIALS.len()),
            h: STEP * (level_of(x[2], THIN_LEVELS) + 1) as f64,
            t: WIDTH_MIN + STEP * level_of(x[3], WIDTH_LEVELS) as f64,
            b: STEP * (level_of(x[4], THIN_LEVELS) + 1) as f64,
            l: x[5],
        }
    }

    /// Evaluate a design with the material table `materials` (the campaign
    /// uses [`MATERIALS`]; the tests also use the classic steel coefficients).
    pub fn evaluate_with(design: &Design, material: &Material) -> Evaluation {
        let Design {
            four_sided,
            h,
            t,
            b,
            l,
            ..
        } = *design;
        let Material {
            s, e, g, c1, c2, ..
        } = *material;
        let x1 = if four_sided { 1.0 } else { 0.0 };
        let cost = (1.0 + c1) * h * h * (l + x1 * t) + c2 * t * b * (L + l);
        // Reklaitis et al. (1983): bending stress, deflection, buckling.
        let sigma = 6.0 * F * L / (t * t * b);
        let delta = 4.0 * F * L.powi(3) / (e * t.powi(3) * b);
        let pc = BUCKLING * (e * g * t * t * b.powi(6) / 36.0).sqrt() / (L * L)
            * (1.0 - t / (2.0 * L) * (e / (4.0 * g)).sqrt());
        // Weld area and polar moment (Deb and Goyal 1997, section 4.4).
        let (area, j) = if four_sided {
            (WELD * h * (t + l), WELD * h * (h + t + l).powi(3) / 12.0)
        } else {
            (
                WELD * h * l,
                WELD * h * l * ((h + t).powi(2) / 4.0 + l * l / 12.0),
            )
        };
        // Reklaitis et al. (1983): primary and secondary shear.
        let m = F * (L + l / 2.0);
        let r = (l * l / 4.0 + ((h + t) / 2.0).powi(2)).sqrt();
        let tau1 = F / area;
        let tau2 = m * r / j;
        let tau = (tau1 * tau1 + 2.0 * tau1 * tau2 * l / (2.0 * r) + tau2 * tau2).sqrt();
        let shear = SHEAR_FRACTION * s;
        let slack = [s - sigma, pc - F, DELTA_MAX - delta, shear - tau];
        let normalized = [
            (sigma - s) / s,
            (F - pc) / F,
            (delta - DELTA_MAX) / DELTA_MAX,
            (tau - shear) / shear,
        ];
        let violation: f64 = normalized.iter().map(|v| v.max(0.0)).sum();
        Evaluation {
            cost,
            sigma,
            pc,
            delta,
            tau,
            slack,
            normalized,
            penalized: cost + super::PENALTY * violation,
        }
    }

    /// Evaluate a design with its material of [`MATERIALS`].
    pub fn evaluate(design: &Design) -> Evaluation {
        evaluate_with(design, &MATERIALS[design.material])
    }

    /// Scalar cost of a real vector (`+inf` for a non-finite input).
    pub fn objective(x: &[f64]) -> f64 {
        if x.iter().any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        evaluate(&decode(x)).penalized
    }
}

fn welded_beam_problem() -> Problem {
    let (lo, hi) = welded_beam::LENGTH_BOUNDS;
    Problem {
        name: "welded_beam_mixed",
        domain: "structures",
        lower: vec![0.0, 0.0, 0.0, 0.0, 0.0, lo],
        upper: vec![1.0, 1.0, 1.0, 1.0, 1.0, hi],
        objective: Box::new(welded_beam::objective),
    }
}

fn welded_beam_table() -> TypeTable {
    use welded_beam::{LENGTH_BOUNDS, MATERIALS, STEP, THIN_LEVELS, WIDTH_LEVELS, WIDTH_MIN};
    let sixteenths =
        |first: i64| move |i: usize| format!("{} in", STEP * (first + i as i64) as f64);
    let h_label = sixteenths(1);
    let t_label = |i: usize| format!("{} in", WIDTH_MIN + STEP * i as f64);
    TypeTable {
        problem: "welded_beam_mixed",
        source: "Deb and Goyal (1997, ICGA), section 4.4 and tables 4-5 (GeneAS welded beam, Deb and Goyal 1996); formulas of Reklaitis, Ravindran and Ragsdell (1983); harness/src/round3b.rs: welded_beam::decode",
        coordinates: vec![
            equal_bins(
                0,
                Kind::Integer,
                2,
                0,
                &|i| ["two-sided weld", "four-sided weld"][i].into(),
                "x1 weld type (binary switch, 0 two-sided, 1 four-sided)",
            ),
            equal_bins(
                1,
                Kind::Categorical,
                MATERIALS.len(),
                0,
                &|i| MATERIALS[i].name.into(),
                "x2 bar material (codes 00, 01, 10, 11 of table 4)",
            ),
            equal_bins(
                2,
                Kind::Integer,
                THIN_LEVELS,
                1,
                &h_label,
                "x3 = h weld thickness, value in sixteenths of an inch (0.0625..2)",
            ),
            equal_bins(
                3,
                Kind::Integer,
                WIDTH_LEVELS,
                32,
                &t_label,
                "x4 = t beam width, value in sixteenths of an inch (2..20)",
            ),
            equal_bins(
                4,
                Kind::Integer,
                THIN_LEVELS,
                1,
                &h_label,
                "x5 = b beam thickness, value in sixteenths of an inch (0.0625..2)",
            ),
            continuous(
                5,
                LENGTH_BOUNDS.0,
                LENGTH_BOUNDS.1,
                "x6 = l weld length (continuous, in)",
            ),
        ],
        notes: vec![
            "The weld type is a binary switch: integer (ROUND3.md section 4). The material is an \
             unordered choice among four rows of table 4: categorical. h, t and b are multiples of \
             0.0625 in (ordered levels): integer, with values in sixteenths of an inch."
                .into(),
            "Constraints of the source as g(x) <= 0 divided by their limits: (sigma - S)/S, \
             (F - Pc)/F, (delta - 0.25)/0.25, (tau - 0.577 S)/(0.577 S); cost = f + 1e6 times the \
             sum of their positive parts."
                .into(),
            "Bounds of l (0.1..10) from Tran, Tran and Wang (2019); the source lets l take any real \
             value."
                .into(),
        ],
    }
}

fn welded_beam_probe() -> Probe {
    use welded_beam::{STEP, WIDTH_MIN, decode};
    Box::new(|index, value| {
        let mut x = vec![0.5, 0.5, 0.5, 0.5, 0.5, 1.0];
        x[index] = value;
        let design = decode(&x);
        let level = match index {
            0 => usize::from(design.four_sided),
            1 => design.material,
            2 => (design.h / STEP).round() as usize - 1,
            3 => ((design.t - WIDTH_MIN) / STEP).round() as usize,
            4 => (design.b / STEP).round() as usize - 1,
            _ => {
                return Err(format!(
                    "welded_beam_mixed: coordinate {index} is continuous"
                ));
            }
        };
        Ok(level)
    })
}

// ============================================================= cantilever

/// `MixedCantileverBeam` of SMT 2.15.1 (`smt/problems/mixed_cantilever_beam.py`).
///
/// Copyright (c) 2017, SMT developers; BSD-3-Clause (license text in
/// `round3b_reference.py`). Author P. Saves; problem of Saves, Diouane,
/// Bartoli, Lefebvre and Morlier, "A mixed-categorical correlation kernel for
/// Gaussian process" (Neurocomputing 2023, arXiv:2211.08262).
///
/// Tip deflection `y = (P L^3) / (3 E S^2 I)` with `P = 50e3`, `E = 200e9`,
/// `L` in `[10, 20]`, `S` in `[1, 2]` and the normalized moment of inertia
/// `I` of the chosen cross-section, as SMT evaluates it:
/// `I = listI[int(x0) - 1]` where `x0` is the category index 0..11 of SMT's
/// design space (labels "1".."12"). For index 0 Python's index -1 selects
/// the last entry, so category index `k` gets `listI[(k - 1) mod 12]`: SMT's
/// labels are shifted by one against the paper's numbering of the twelve
/// sections (the paper groups levels {1, 4, 7, 10}, which are entries 0, 3,
/// 6, 9 of `listI`). The categories are unordered, so the shift changes only
/// their labels; the harness follows SMT's design-space order.
pub mod cantilever {
    /// Tip load `P` (SMT option default 50e3).
    pub const P: f64 = 50e3;
    /// Young's modulus `E` (SMT option default 200e9).
    pub const E: f64 = 200e9;
    /// `self.listI` of SMT, in its order.
    pub const LIST_I: [f64; 12] = [
        0.0833, 0.139, 0.380, 0.0796, 0.133, 0.363, 0.0859, 0.136, 0.360, 0.0922, 0.138, 0.369,
    ];
    /// Bounds of `L` (SMT `FloatVariable(10.0, 20.0)`).
    pub const LENGTH_BOUNDS: (f64, f64) = (10.0, 20.0);
    /// Bounds of `S` (SMT `FloatVariable(1.0, 2.0)`).
    pub const AREA_BOUNDS: (f64, f64) = (1.0, 2.0);

    /// `I` of SMT category index `k` (0..11): `listI[k - 1]`, Python indexing.
    pub fn inertia(category: usize) -> f64 {
        LIST_I[(category + LIST_I.len() - 1) % LIST_I.len()]
    }

    /// SMT's formula, in SMT's operation order:
    /// `(P * L**3) / (3 * E * S**2 * I)`; `L**3` is `pow(L, 3.0)` and `S**2`
    /// is `S * S` (NumPy's square fast path).
    pub fn deflection(category: usize, l: f64, s: f64) -> f64 {
        (P * l.powf(3.0)) / (3.0 * E * (s * s) * inertia(category))
    }

    /// Scalar cost of a real vector `(u, L, S)` (`+inf` for a non-finite
    /// input).
    pub fn objective(x: &[f64]) -> f64 {
        if x.iter().any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        deflection(super::level_of(x[0], LIST_I.len()), x[1], x[2])
    }
}

fn cantilever_problem() -> Problem {
    use cantilever::{AREA_BOUNDS, LENGTH_BOUNDS};
    Problem {
        name: "cantilever_mixed",
        domain: "structures",
        lower: vec![0.0, LENGTH_BOUNDS.0, AREA_BOUNDS.0],
        upper: vec![1.0, LENGTH_BOUNDS.1, AREA_BOUNDS.1],
        objective: Box::new(cantilever::objective),
    }
}

fn cantilever_table() -> TypeTable {
    use cantilever::{AREA_BOUNDS, LENGTH_BOUNDS, LIST_I, inertia};
    TypeTable {
        problem: "cantilever_mixed",
        source: "SMT 2.15.1 smt/problems/mixed_cantilever_beam.py: MixedCantileverBeam._evaluate (BSD-3-Clause); harness/src/round3b.rs: cantilever::objective",
        coordinates: vec![
            equal_bins(
                0,
                Kind::Categorical,
                LIST_I.len(),
                0,
                &|k| format!("SMT category {} (I = {})", k + 1, inertia(k)),
                "cross-section shape, SMT categorical variable with labels 1..12",
            ),
            continuous(
                1,
                LENGTH_BOUNDS.0,
                LENGTH_BOUNDS.1,
                "beam length L (continuous)",
            ),
            continuous(
                2,
                AREA_BOUNDS.0,
                AREA_BOUNDS.1,
                "cross-section area S (continuous)",
            ),
        ],
        notes: vec![
            "The cross-section is an unordered choice among twelve shapes: categorical. Levels follow \
             SMT's design-space order; SMT evaluates category index k with listI[k - 1] (Python \
             indexing), a shift by one against the paper's numbering."
                .into(),
        ],
    }
}

fn cantilever_probe() -> Probe {
    Box::new(|index, value| {
        if index != 0 {
            return Err(format!(
                "cantilever_mixed: coordinate {index} is continuous"
            ));
        }
        Ok(level_of(value, cantilever::LIST_I.len()))
    })
}

// ============================================================ pest control

/// NumPy's legacy `RandomState` (MT19937 with `mt19937_seed`, the legacy
/// polar Gaussian, gamma and beta samplers), ported from NumPy 2.5.3
/// (`numpy/random/src/mt19937/mt19937.{c,h}`,
/// `numpy/random/src/legacy/legacy-distributions.c`, `_mt19937.pyx`
/// `_legacy_seeding`; BSD-3-Clause, Copyright (c) 2005-2025, NumPy
/// Developers). `RandomState(seed).beta(a, b, size=n)` draws `n` values of
/// `legacy_beta` in sequence.
pub mod legacy_random {
    const N: usize = 624;
    const M: usize = 397;
    const MATRIX_A: u32 = 0x9908_b0df;
    const UPPER_MASK: u32 = 0x8000_0000;
    const LOWER_MASK: u32 = 0x7fff_ffff;

    /// `np.random.RandomState(seed)` for an integer seed.
    #[derive(Clone)]
    pub struct RandomState {
        key: [u32; N],
        pos: usize,
        gauss: Option<f64>,
    }

    impl RandomState {
        /// `mt19937_seed`: Knuth's initialization of the reference MT19937.
        pub fn new(seed: u32) -> Self {
            let mut key = [0u32; N];
            let mut s = seed;
            for (pos, slot) in key.iter_mut().enumerate() {
                *slot = s;
                s = 1_812_433_253u32
                    .wrapping_mul(s ^ (s >> 30))
                    .wrapping_add(pos as u32 + 1);
            }
            Self {
                key,
                pos: N,
                gauss: None,
            }
        }

        fn generate(&mut self) {
            let twist = |a: u32, b: u32, c: u32| {
                let y = (a & UPPER_MASK) | (b & LOWER_MASK);
                c ^ (y >> 1) ^ ((y & 1).wrapping_neg() & MATRIX_A)
            };
            for i in 0..N - M {
                self.key[i] = twist(self.key[i], self.key[i + 1], self.key[i + M]);
            }
            for i in N - M..N - 1 {
                self.key[i] = twist(self.key[i], self.key[i + 1], self.key[i + M - N]);
            }
            self.key[N - 1] = twist(self.key[N - 1], self.key[0], self.key[M - 1]);
            self.pos = 0;
        }

        /// `mt19937_next`: next tempered 32-bit output.
        pub fn next_u32(&mut self) -> u32 {
            if self.pos == N {
                self.generate();
            }
            let mut y = self.key[self.pos];
            self.pos += 1;
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c_5680;
            y ^= (y << 15) & 0xefc6_0000;
            y ^ (y >> 18)
        }

        /// `mt19937_next_double` (`legacy_double`): 53-bit uniform in [0, 1).
        pub fn next_double(&mut self) -> f64 {
            let a = f64::from(self.next_u32() >> 5);
            let b = f64::from(self.next_u32() >> 6);
            (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
        }

        /// `legacy_gauss`: polar method, the second value cached.
        pub fn gauss(&mut self) -> f64 {
            if let Some(value) = self.gauss.take() {
                return value;
            }
            let (x1, x2, r2) = loop {
                let x1 = 2.0 * self.next_double() - 1.0;
                let x2 = 2.0 * self.next_double() - 1.0;
                let r2 = x1 * x1 + x2 * x2;
                // NumPy repeats while `r2 >= 1.0 || r2 == 0.0`; r2 is never NaN.
                if r2 < 1.0 && r2 != 0.0 {
                    break (x1, x2, r2);
                }
            };
            let f = (-2.0 * r2.ln() / r2).sqrt();
            self.gauss = Some(f * x1);
            f * x2
        }

        /// `legacy_standard_exponential`: `-log(1 - U)`.
        pub fn standard_exponential(&mut self) -> f64 {
            -(1.0 - self.next_double()).ln()
        }

        /// `legacy_standard_gamma`.
        pub fn standard_gamma(&mut self, shape: f64) -> f64 {
            if shape == 1.0 {
                return self.standard_exponential();
            }
            if shape == 0.0 {
                return 0.0;
            }
            if shape < 1.0 {
                loop {
                    let u = self.next_double();
                    let v = self.standard_exponential();
                    if u <= 1.0 - shape {
                        let x = u.powf(1.0 / shape);
                        if x <= v {
                            return x;
                        }
                    } else {
                        let y = -((1.0 - u) / shape).ln();
                        let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                        if x <= v + y {
                            return x;
                        }
                    }
                }
            }
            let b = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * b).sqrt();
            loop {
                let (x, mut v) = loop {
                    let x = self.gauss();
                    let v = 1.0 + c * x;
                    if v > 0.0 {
                        break (x, v);
                    }
                };
                v = v * v * v;
                let u = self.next_double();
                if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                    return b * v;
                }
                if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
                    return b * v;
                }
            }
        }

        /// `legacy_beta`: Johnk's algorithm when both shapes are at most 1,
        /// otherwise the ratio of two gamma draws.
        pub fn beta(&mut self, a: f64, b: f64) -> f64 {
            if a <= 1.0 && b <= 1.0 {
                loop {
                    let u = self.next_double();
                    let v = self.next_double();
                    let x = u.powf(1.0 / a);
                    let y = v.powf(1.0 / b);
                    if x + y <= 1.0 {
                        if x + y > 0.0 {
                            return x / (x + y);
                        }
                        let mut log_x = u.ln() / a;
                        let mut log_y = v.ln() / b;
                        let log_m = if log_x > log_y { log_x } else { log_y };
                        log_x -= log_m;
                        log_y -= log_m;
                        return (log_x - (log_x.exp() + log_y.exp()).ln()).exp();
                    }
                }
            }
            let ga = self.standard_gamma(a);
            let gb = self.standard_gamma(b);
            ga / (ga + gb)
        }

        /// `RandomState(seed).beta(a, b, size=count)`.
        pub fn beta_array(seed: u32, a: f64, b: f64, count: usize) -> Vec<f64> {
            let mut state = Self::new(seed);
            (0..count).map(|_| state.beta(a, b)).collect()
        }
    }
}

/// The pest-control benchmark in the deterministic form of the Bounce suite.
///
/// Source: `_pest_control_score(x, seed)` of Bounce
/// (`bounce/util/pest_control.py`, MIT, Copyright (c) 2024 Leonard
/// Papenmeier; taken there from BODi), identical to Casmopolitan's
/// `test_funcs/pest.py` (MIT, Copyright (c) 2022 Xingchen Wan et al.); the
/// model is that of COMBO (`COMBO/experiments/test_functions/
/// multiple_categorical.py`, BSD-2-Clause, Copyright (c) 2019 QUVA-Lab),
/// where every draw comes from NumPy's global generator, so COMBO's form is
/// noisy. In the fixed-seed form every beta draw is
/// `np.random.RandomState(seed).beta(...)`, a fresh generator per call, so
/// the value is a deterministic function of the 25 choices. Bounce's
/// `PestControl` uses seed 0 (its default, `configs/default.gin` sets no
/// seed); Casmopolitan's default is 8194 (`sorted(seeds)[19]` of
/// `generate_random_seed_pestcontrol`). The harness uses seed 0.
///
/// All constants are those of the reference code: 25 stages, 100
/// simulations, threshold `U = 0.1`, initial fraction `Beta(1, 30)`, spread
/// `Beta(1, 17/3)`, control `Beta(1, beta_k)` with `beta_k` starting at 2/7,
/// 3/7, 3/7, 5/7 and growing by `rate_k / 25` per use (`rate_k` = 1/7,
/// 2.5/7, 2/7, 0.5/7), prices 1.0, 0.8, 0.7, 0.5 and maximal discounts 0.2,
/// 0.3, 0.3, 0.0.
pub mod pest {
    use super::legacy_random::RandomState;

    /// Stations (stages).
    pub const STAGES: usize = 25;
    /// Choices per station: 0 no pesticide, 1..4 pesticides.
    pub const CHOICES: usize = 5;
    /// Simulated trajectories.
    pub const SIMULATIONS: usize = 100;
    /// Seed of the Bounce suite.
    pub const SEED: u32 = 0;
    const THRESHOLD: f64 = 0.1;
    const INIT_ALPHA: f64 = 1.0;
    const INIT_BETA: f64 = 30.0;
    const SPREAD_ALPHA: f64 = 1.0;
    const CONTROL_ALPHA: f64 = 1.0;

    fn control_beta0() -> [f64; 4] {
        [2.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0, 5.0 / 7.0]
    }
    fn tolerance_rate() -> [f64; 4] {
        [1.0 / 7.0, 2.5 / 7.0, 2.0 / 7.0, 0.5 / 7.0]
    }
    const PRICE: [f64; 4] = [1.0, 0.8, 0.7, 0.5];
    const MAX_DISCOUNT: [f64; 4] = [0.2, 0.3, 0.3, 0.0];

    /// The draws of one seed: they do not depend on the choices except
    /// through the number of earlier uses of each pesticide.
    pub struct PestControl {
        init: Vec<f64>,
        spread: Vec<f64>,
        /// `control[k][n]`: the control rates of pesticide `k + 1` at its
        /// `n`-th use (`beta_k` after `n` earlier uses).
        control: Vec<Vec<Vec<f64>>>,
    }

    impl PestControl {
        /// Precompute every draw of `seed`.
        pub fn new(seed: u32) -> Self {
            let init = RandomState::beta_array(seed, INIT_ALPHA, INIT_BETA, SIMULATIONS);
            let spread = RandomState::beta_array(seed, SPREAD_ALPHA, 17.0 / 3.0, SIMULATIONS);
            let rate = tolerance_rate();
            let control = control_beta0()
                .iter()
                .enumerate()
                .map(|(k, &beta0)| {
                    let mut beta = beta0;
                    (0..STAGES)
                        .map(|_| {
                            let draws =
                                RandomState::beta_array(seed, CONTROL_ALPHA, beta, SIMULATIONS);
                            beta += rate[k] / STAGES as f64;
                            draws
                        })
                        .collect()
                })
                .collect();
            Self {
                init,
                spread,
                control,
            }
        }

        /// `_pest_control_score(x, seed)` for choices `x` (each 0..=4).
        pub fn score(&self, x: &[usize]) -> f64 {
            assert_eq!(x.len(), STAGES, "pest control has {STAGES} stations");
            let mut current = self.init.clone();
            let mut next = vec![0.0; SIMULATIONS];
            let mut uses = [0usize; CHOICES];
            let mut paid = 0.0;
            let mut above = 0.0;
            for &choice in x {
                assert!(choice < CHOICES, "choice {choice} out of range");
                let price = if choice > 0 {
                    let k = choice - 1;
                    let rates = &self.control[k][uses[choice]];
                    for ((n, &c), &rate) in next.iter_mut().zip(&current).zip(rates) {
                        *n = (1.0 - rate) * c;
                    }
                    uses[choice] += 1;
                    let count = x.iter().filter(|&&v| v == choice).count() as f64;
                    PRICE[k] * (1.0 - MAX_DISCOUNT[k] / STAGES as f64 * count)
                } else {
                    for ((n, &c), &s) in next.iter_mut().zip(&current).zip(&self.spread) {
                        *n = s * (1.0 - c) + c;
                    }
                    0.0
                };
                paid += price;
                let exceeding = current.iter().filter(|&&c| c > THRESHOLD).count();
                above += exceeding as f64 / SIMULATIONS as f64;
                std::mem::swap(&mut current, &mut next);
            }
            paid + above
        }

        /// Scalar cost of a real vector of the unit box (`+inf` for a
        /// non-finite input).
        pub fn objective(&self, x: &[f64]) -> f64 {
            if x.iter().any(|v| !v.is_finite()) {
                return f64::INFINITY;
            }
            let choices: Vec<usize> = x.iter().map(|&u| super::level_of(u, CHOICES)).collect();
            self.score(&choices)
        }
    }
}

fn pest_problem() -> Problem {
    let model = Arc::new(pest::PestControl::new(pest::SEED));
    Problem {
        name: "pest_control",
        domain: "agriculture",
        lower: vec![0.0; pest::STAGES],
        upper: vec![1.0; pest::STAGES],
        objective: Box::new(move |x| model.objective(x)),
    }
}

fn pest_table() -> TypeTable {
    let labels = |i: usize| {
        if i == 0 {
            "no pesticide".to_owned()
        } else {
            format!("pesticide {i}")
        }
    };
    TypeTable {
        problem: "pest_control",
        source: "Bounce bounce/util/pest_control.py: _pest_control_score(x, seed=0) (MIT; = Casmopolitan test_funcs/pest.py; model of COMBO); harness/src/round3b.rs: pest::PestControl::objective",
        coordinates: (0..pest::STAGES)
            .map(|i| {
                equal_bins(
                    i,
                    Kind::Categorical,
                    pest::CHOICES,
                    0,
                    &labels,
                    format!("choice at station {} (0 none, 1..4 pesticide)", i + 1),
                )
            })
            .collect(),
        notes: vec![
            "Each station chooses no pesticide or one of four pesticides with their own price, \
             effect and tolerance: an unordered choice, categorical."
                .into(),
        ],
    }
}

fn pest_probe() -> Probe {
    Box::new(|_, value| Ok(level_of(value, pest::CHOICES)))
}

// ================================================================ registry

/// Build a round-3b problem by name.
pub fn by_name(name: &str) -> Option<Problem> {
    Some(match name {
        "welded_beam_mixed" => welded_beam_problem(),
        "cantilever_mixed" => cantilever_problem(),
        "pest_control" => pest_problem(),
        _ => return None,
    })
}

/// Type table of a round-3b problem (ROUND3.md section 4, ROUND3B.md
/// section 2).
pub fn type_table(name: &str) -> Option<TypeTable> {
    Some(match name {
        "welded_beam_mixed" => welded_beam_table(),
        "cantilever_mixed" => cantilever_table(),
        "pest_control" => pest_table(),
        _ => return None,
    })
}

/// Probe of a round-3b problem's own decoder: the level index the problem
/// decodes coordinate `index` to at `value` (other coordinates fixed).
pub fn level_probe(name: &str) -> Option<Probe> {
    Some(match name {
        "welded_beam_mixed" => welded_beam_probe(),
        "cantilever_mixed" => cantilever_probe(),
        "pest_control" => pest_probe(),
        _ => return None,
    })
}

/// Source of each problem's DE integer mask: the problems come from outside
/// this repository and have no own optimization code, so none passes one.
pub fn de_mask(_name: &str) -> Option<Vec<bool>> {
    None
}
