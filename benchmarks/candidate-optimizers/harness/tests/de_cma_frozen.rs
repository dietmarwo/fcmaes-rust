//! `de_cma` must stay bit-identical when the sequence adapters share code.
//!
//! The frozen values below were recorded from the round-1 `DeCma` adapter
//! (commit `ec5b849`) before the sequence code was factored out.

use std::sync::Mutex;

use candidate_harness::campaign::{Protocol, run_repetition};
use candidate_harness::problems;
use candidate_harness::solvers;
use solver_api::Guard;

const SHIFT: [f64; 10] = [1.2, -0.7, 3.1, -2.4, 0.3, 2.2, -4.1, 0.9, -1.6, 4.4];

fn shifted_rastrigin(x: &[f64]) -> f64 {
    x.iter()
        .zip(SHIFT.iter().cycle())
        .map(|(v, s)| {
            let z = v - s * 0.5;
            z * z - 10.0 * (2.0 * std::f64::consts::PI * z).cos() + 10.0
        })
        .sum()
}

/// (calls, best value bits, FNV hash of every evaluated coordinate).
fn trace(dim: usize, budget: u64, seed: u64) -> (u64, u64, u64) {
    let solver = solvers::by_name("de_cma").unwrap();
    let (lower, upper) = (vec![-5.0; dim], vec![5.0; dim]);
    let hash = Mutex::new(0xCBF2_9CE4_8422_2325_u64);
    let recorded = |x: &[f64]| {
        let mut h = hash.lock().unwrap();
        for v in x {
            *h = (*h ^ v.to_bits()).wrapping_mul(0x0000_0100_0000_01B3);
        }
        shifted_rastrigin(x)
    };
    let guard = Guard::new(&recorded, &lower, &upper, budget);
    let guarded = |x: &[f64]| guard.eval(x);
    solver.minimize(&guarded, &lower, &upper, budget, seed);
    assert_eq!((guard.over_budget(), guard.out_of_bounds()), (0, 0));
    let h = *hash.lock().unwrap();
    (guard.calls(), guard.best_y().to_bits(), h)
}

const CASES: [(usize, u64, u64); 8] = [
    (5, 1, 0),
    (5, 7, 3),
    (5, 30, 1),
    (5, 77, 2),
    (5, 1000, 17),
    (10, 20_000, 0),
    (10, 20_000, 5),
    (3, 3_001, 9),
];

#[test]
fn de_cma_sequence_is_unchanged() {
    let traces: Vec<(u64, u64, u64)> = CASES
        .iter()
        .map(|&(dim, budget, seed)| trace(dim, budget, seed))
        .collect();
    if std::env::var_os("RECORD_DE_CMA").is_some() {
        println!("{traces:#x?}");
    }
    assert_eq!(traces, FROZEN);
}

#[test]
fn de_cma_problem_rows_are_unchanged() {
    let mut values = Vec::new();
    for name in ["cassini1", "lj13"] {
        let problem = problems::by_name(name).unwrap();
        let solver = solvers::by_name("de_cma").unwrap();
        for protocol in [Protocol::S, Protocol::R] {
            let r = run_repetition(solver.as_ref(), &problem, protocol, 1, 4_000);
            values.push((r.seed, r.evaluations, r.best_value.to_bits()));
        }
    }
    if std::env::var_os("RECORD_DE_CMA").is_some() {
        println!("{values:#x?}");
    }
    assert_eq!(values, FROZEN_ROWS);
}

const FROZEN: [(u64, u64, u64); 8] = [
    (1, 0x4062_6bee_037f_9f8b, 0xbfd5_4e63_2cfb_d9a5),
    (7, 0x4056_9f13_3a8f_4461, 0xf722_9692_c803_ddd8),
    (30, 0x403d_fe65_595b_f0ba, 0x247f_b8a9_f872_ee0c),
    (62, 0x4046_ecff_986a_8a7d, 0xcbc7_9831_cf51_9b31),
    (989, 0x401b_e191_a522_3a3c, 0xfda5_d1e8_02ac_e07f),
    (12_712, 0x4027_e105_f634_b844, 0x597c_bcd1_ed3e_1c1b),
    (12_278, 0x4017_e105_f634_b847, 0x782d_7e65_2033_9dc0),
    (2_998, 0x3fff_d6b4_6062_63d0, 0x4c14_7b1b_df11_e6be),
];

const FROZEN_ROWS: [(u64, u64, u64); 4] = [
    (0x8d8c_4cfb_6858_e946, 3_987, 0x402d_69f6_eb2a_cc63),
    (0x13fe_8269_ccaa_513f, 3_832, 0x401e_f3a5_03a3_f04d),
    (0x5da7_3c6a_031c_73cf, 3_987, 0xc032_12b4_3478_577f),
    (0x24fa_fea2_0b28_6d5a, 3_832, 0xc023_3f45_d3fb_a140),
];
