//! Round 3, second batch (`ROUND3B.md`): definitions against their sources,
//! type tables against the decoders, the campaign set and `serve`.
//!
//! Reference values come from the sources: the GeneAS solution of Deb and
//! Goyal (1997, table 5), SMT 2.15.1 `MixedCantileverBeam`, Bounce's
//! `_pest_control_score` and NumPy's `RandomState` (computed with the
//! reference code and recorded here; `round3b_reference.py verify` repeats
//! the comparison at 500 and 200 random points).

use std::io::Cursor;

use candidate_harness::problems;
use candidate_harness::round3::{Coordinate, Kind, TypeTable};
use candidate_harness::round3b::{
    self, CATEGORICAL_PROBLEMS, PROBLEMS, cantilever, de_mask, legacy_random::RandomState,
    level_of, level_probe, pest, type_table, welded_beam,
};
use candidate_harness::serve::{self, decode, encode};
use candidate_harness::solvers;
use fcmaes_core::Rng;

fn table_level(c: &Coordinate, v: f64) -> usize {
    c.levels
        .iter()
        .position(|l| {
            l.intervals
                .iter()
                .any(|&(lo, hi)| (v >= lo && v < hi) || (v == hi && hi == c.upper))
        })
        .unwrap_or_else(|| panic!("coordinate {}: {v} in no interval", c.index))
}

// ------------------------------------------------------------- the set

#[test]
fn the_batch_has_three_categorical_problems_with_consistent_tables() {
    assert_eq!(
        PROBLEMS,
        ["welded_beam_mixed", "cantilever_mixed", "pest_control"]
    );
    assert_eq!(CATEGORICAL_PROBLEMS, PROBLEMS);
    assert!(round3b::DROPPED.is_empty());
    let expected = [(6, 1, 4, 1), (3, 2, 0, 1), (25, 0, 0, 25)];
    for (name, (dim, cont, int, cat)) in PROBLEMS.iter().zip(expected) {
        let problem = problems::by_name(name).unwrap();
        let table = type_table(name).unwrap();
        assert_eq!(problem.dim(), dim, "{name}");
        assert!(problem.dim() <= 200);
        assert_eq!(
            (
                table.count(Kind::Continuous),
                table.count(Kind::Integer),
                table.count(Kind::Categorical)
            ),
            (cont, int, cat),
            "{name}"
        );
        for (i, c) in table.coordinates.iter().enumerate() {
            assert_eq!(c.index, i);
            assert_eq!((c.lower, c.upper), (problem.lower[i], problem.upper[i]));
            assert_eq!(c.kind == Kind::Continuous, c.levels.is_empty());
            for l in &c.levels {
                assert_eq!(l.value.is_some(), c.kind == Kind::Integer, "{name} {i}");
            }
            if c.kind != Kind::Continuous {
                assert_eq!(
                    (c.lower, c.upper),
                    (0.0, 1.0),
                    "discrete coordinates are unit"
                );
            }
        }
        assert_eq!(table.to_json().matches("\"index\":").count(), dim);
        assert!(de_mask(name).is_none());
        assert!(candidate_harness::round3::de_mask(name).is_none());
        // Not part of round 3.
        assert!(candidate_harness::round3::type_table(name).is_none());
    }
    let welded = type_table("welded_beam_mixed").unwrap();
    let values = |i: usize| -> Vec<i64> {
        welded.coordinates[i]
            .levels
            .iter()
            .map(|l| l.value.unwrap())
            .collect()
    };
    assert_eq!(values(0), vec![0, 1]);
    assert_eq!(values(2), (1..=32).collect::<Vec<_>>());
    assert_eq!(values(3), (32..=320).collect::<Vec<_>>());
    assert_eq!(values(4), (1..=32).collect::<Vec<_>>());
    assert_eq!(welded.coordinates[1].levels.len(), 4);
    assert_eq!(
        type_table("cantilever_mixed").unwrap().coordinates[0]
            .levels
            .len(),
        12
    );
}

// ------------------------------------------------- tables and decoders

fn check_levels(table: &TypeTable) {
    let probe = level_probe(table.problem).unwrap();
    for c in table
        .coordinates
        .iter()
        .filter(|c| c.kind != Kind::Continuous)
    {
        for (level, l) in c.levels.iter().enumerate() {
            assert_eq!(l.intervals.len(), 1);
            assert_eq!(
                probe(c.index, l.center).unwrap(),
                level,
                "{} {}",
                table.problem,
                c.index
            );
            assert_eq!(table_level(c, l.center), level);
        }
        assert_eq!(probe(c.index, c.lower).unwrap(), 0);
        assert_eq!(probe(c.index, c.upper).unwrap(), c.levels.len() - 1);
        let delta = 1e-9;
        for pair in c.levels.windows(2) {
            let b = pair[0].intervals[0].1;
            assert_eq!(b, pair[1].intervals[0].0);
            let below = probe(c.index, b - delta).unwrap();
            let above = probe(c.index, b + delta).unwrap();
            assert_eq!(below + 1, above, "{} {} at {b}", table.problem, c.index);
            assert_eq!(table_level(c, b - delta), below);
            assert_eq!(table_level(c, b + delta), above);
        }
    }
}

#[test]
fn every_level_decodes_to_itself_and_boundaries_to_the_neighbors() {
    for name in PROBLEMS {
        check_levels(&type_table(name).unwrap());
    }
    // The decoding rule of docs/combinatorial-encodings.md.
    assert_eq!(level_of(0.0, 5), 0);
    assert_eq!(level_of(0.2, 5), 1);
    assert_eq!(level_of(0.199_999_999, 5), 0);
    assert_eq!(level_of(1.0, 5), 4);
    assert_eq!(level_of(1.0, 289), 288);
}

#[test]
fn moving_discrete_coordinates_to_level_centers_keeps_the_objective() {
    for name in PROBLEMS {
        let problem = problems::by_name(name).unwrap();
        let table = type_table(name).unwrap();
        let mut rng = Rng::new(0x5EED_003B);
        for _ in 0..200 {
            let x = solvers::uniform_point(&problem.lower, &problem.upper, &mut rng);
            let typed: Vec<f64> = table
                .coordinates
                .iter()
                .map(|c| match c.kind {
                    Kind::Continuous => x[c.index],
                    _ => table_level(c, x[c.index]) as f64,
                })
                .collect();
            let mapped = table.to_real(&typed);
            assert_eq!(
                (problem.objective)(&x).to_bits(),
                (problem.objective)(&mapped).to_bits(),
                "{name}"
            );
        }
    }
}

#[test]
fn every_problem_is_deterministic_and_finite() {
    for name in PROBLEMS {
        let first = problems::by_name(name).unwrap();
        let second = problems::by_name(name).unwrap();
        let mut rng = Rng::new(31);
        for _ in 0..200 {
            let x = solvers::uniform_point(&first.lower, &first.upper, &mut rng);
            let y = (first.objective)(&x);
            assert!(y.is_finite(), "{name}");
            assert_eq!(y.to_bits(), (first.objective)(&x).to_bits());
            assert_eq!(y.to_bits(), (second.objective)(&x).to_bits());
        }
        assert!((first.objective)(&vec![f64::NAN; first.dim()]).is_infinite());
    }
}

// ------------------------------------------------------- welded beam

fn geneas_design(l: f64) -> welded_beam::Design {
    welded_beam::Design {
        four_sided: true,
        material: 0,
        h: 0.1875,
        t: 8.25,
        b: 0.25,
        l,
    }
}

fn round_to(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    (value * scale).round() / scale
}

#[test]
fn welded_beam_reproduces_the_geneas_solution_of_deb_and_goyal() {
    // Table 5 of Deb and Goyal (1997): four-sided, steel, h = 0.1875,
    // t = 8.25, b = 0.25, l = 1.6849; g1 = 380.1660, g2 = 402.0473,
    // g3 = 0.2346, g4 = 0.1621, f = 1.9422.
    let at = welded_beam::evaluate(&geneas_design(1.6849));
    // g1 = S - sigma: 380.16529 in double precision; the printed 380.1660 is
    // the single-precision value (380.166015625).
    assert_eq!(round_to(at.slack[0], 4), 380.1653);
    assert_eq!(
        f64::from(30_000f32 - 504_000f32 / (8.25f32 * 8.25 * 0.25)),
        380.166_015_625
    );
    // g2 = Pc - F.
    assert!((at.slack[1] - 402.0473).abs() < 2e-4, "{}", at.slack[1]);
    // g3 = delta_max - delta: 0.23436; the printed 0.2346 is not reproduced.
    assert_eq!(round_to(at.slack[2], 4), 0.2344);
    // g4 = 0.577 S - tau is nearly active: 0.1339 at the printed l; the
    // printed 0.1621 is reached inside the rounding interval of 1.6849.
    let g4 = |l: f64| welded_beam::evaluate(&geneas_design(l)).slack[3];
    let (mut low, mut high) = (1.684_85, 1.684_95);
    assert!(
        g4(low) < 0.1621 && 0.1621 < g4(high),
        "{} {}",
        g4(low),
        g4(high)
    );
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        if g4(mid) < 0.1621 {
            low = mid
        } else {
            high = mid
        }
    }
    let l_star = low;
    assert_eq!(round_to(l_star, 4), 1.6849);
    assert_eq!(round_to(l_star, 6), 1.684_907);
    assert!((at.slack[3] - 0.1339).abs() < 1e-4);
    // Cost with the table's coefficients 0.1047 and 0.0481.
    assert_eq!(round_to(at.cost, 5), 1.94188);
    // The printed 1.9422 is the cost with the classic steel coefficients
    // 0.10471 and 0.04811, of which table 4 prints the rounded values.
    let classic = welded_beam::Material {
        c1: 0.10471,
        c2: 0.04811,
        ..welded_beam::MATERIALS[0]
    };
    let printed = welded_beam::evaluate_with(&geneas_design(1.6849), &classic);
    assert_eq!(round_to(printed.cost, 4), 1.9422);
    // Statements of the text: bending 1.3% below its limit, buckling 6.7%
    // above the load, shear 9.4e-4 % below its limit (at l = 1.68491).
    assert_eq!(round_to(-at.normalized[0] * 100.0, 1), 1.3);
    assert_eq!(round_to(-at.normalized[1] * 100.0, 1), 6.7);
    let near = welded_beam::evaluate(&geneas_design(l_star));
    assert_eq!(round_to(-near.normalized[3] * 100.0, 5), 0.00094);
    // Feasible there: no penalty.
    assert!(near.normalized.iter().all(|v| *v < 0.0));
    assert_eq!(near.penalized, near.cost);
}

// 0.5235 is a published cost coefficient, not an approximation of pi/6.
#[allow(clippy::approx_constant)]
#[test]
fn welded_beam_real_vector_decodes_to_the_source_variables() {
    let table = type_table("welded_beam_mixed").unwrap();
    let center = |i: usize, level: usize| table.coordinates[i].levels[level].center;
    // h = 0.1875 is level 2, t = 8.25 is level 100, b = 0.25 is level 3.
    let x = [
        center(0, 1),
        center(1, 0),
        center(2, 2),
        center(3, 100),
        center(4, 3),
        1.6849,
    ];
    assert_eq!(welded_beam::decode(&x), geneas_design(1.6849));
    let problem = problems::by_name("welded_beam_mixed").unwrap();
    assert_eq!(
        (problem.objective)(&x).to_bits(),
        welded_beam::evaluate(&geneas_design(1.6849))
            .penalized
            .to_bits()
    );
    // Box corners: smallest and largest levels.
    let low = welded_beam::decode(&problem.lower);
    assert_eq!(
        (low.four_sided, low.material, low.h, low.t, low.b, low.l),
        (false, 0, 0.0625, 2.0, 0.0625, 0.1)
    );
    let high = welded_beam::decode(&problem.upper);
    assert_eq!(
        (
            high.four_sided,
            high.material,
            high.h,
            high.t,
            high.b,
            high.l
        ),
        (true, 3, 2.0, 20.0, 2.0, 10.0)
    );
    // Penalty: one unit of normalized violation costs 1e6.
    let thin = welded_beam::Design {
        b: 0.0625,
        ..geneas_design(1.6849)
    };
    let e = welded_beam::evaluate(&thin);
    let violation: f64 = e.normalized.iter().map(|v| v.max(0.0)).sum();
    assert!(violation > 0.0);
    assert_eq!(e.penalized, e.cost + 1e6 * violation);
    // Material table (Deb and Goyal 1997, table 4).
    let rows: Vec<(f64, f64, f64, f64, f64)> = welded_beam::MATERIALS
        .iter()
        .map(|m| (m.s, m.e, m.g, m.c1, m.c2))
        .collect();
    assert_eq!(
        rows,
        vec![
            (30e3, 30e6, 12e6, 0.1047, 0.0481),
            (8e3, 14e6, 6e6, 0.0489, 0.0224),
            (5e3, 10e6, 4e6, 0.5235, 0.2405),
            (8e3, 16e6, 6e6, 0.5584, 0.2566),
        ]
    );
}

// --------------------------------------------------------- cantilever

#[test]
fn cantilever_equals_smt_values() {
    // SMT 2.15.1 MixedCantileverBeam()(X) for X rows (category index, L, S).
    let cases = [
        ((0, 15.0, 1.5), 0.000_338_753_387_533_875_34),
        ((3, 10.0, 2.0), 5.482_456_140_350_877e-5),
        ((11, 20.0, 1.0), 0.004_830_917_874_396_135),
        ((5, 12.5, 1.25), 0.000_783_208_020_050_125_3),
    ];
    for ((k, l, s), expected) in cases {
        assert_eq!(
            cantilever::deflection(k, l, s).to_bits(),
            f64::to_bits(expected)
        );
        let x = [(k as f64 + 0.5) / 12.0, l, s];
        let problem = problems::by_name("cantilever_mixed").unwrap();
        assert_eq!((problem.objective)(&x).to_bits(), f64::to_bits(expected));
    }
    // SMT's category index 0 uses listI[-1] = 0.369.
    assert_eq!(cantilever::inertia(0), 0.369);
    assert_eq!(cantilever::inertia(1), 0.0833);
    assert_eq!(cantilever::inertia(11), 0.138);
}

// ------------------------------------------------------- pest control

#[test]
fn legacy_random_state_matches_numpy() {
    // Reference MT19937 (init_genrand(5489)): first output 3499211612.
    assert_eq!(RandomState::new(5489).next_u32(), 3_499_211_612);
    // NumPy: RandomState(0).random_sample(), .beta(1, 30), .beta(1, 17/3),
    // .beta(1, 2/7); RandomState(123).standard_normal().
    assert_eq!(RandomState::new(0).next_double(), 0.548_813_503_927_324_8);
    assert_eq!(
        RandomState::new(0).beta(1.0, 30.0),
        0.022_943_885_093_686_226
    );
    assert_eq!(
        RandomState::new(0).beta(1.0, 17.0 / 3.0),
        0.099_091_652_285_245_93
    );
    assert_eq!(
        RandomState::new(0).beta(1.0, 2.0 / 7.0),
        0.639_508_407_879_663_5
    );
    assert_eq!(RandomState::new(123).gauss(), -1.085_630_603_300_561_2);
}

#[test]
fn pest_control_equals_the_reference_code() {
    // Bounce `_pest_control_score(x, seed)` for seeds 0 (Bounce), 8194
    // (Casmopolitan's default) and 1.
    let stairs: Vec<usize> = (0..25).map(|i| i % 5).collect();
    let none = vec![0usize; 25];
    let all3 = vec![3usize; 25];
    let random = vec![
        2, 4, 4, 1, 0, 3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 4, 0, 0, 3, 2, 3, 2, 4, 0, 3,
    ];
    let expected: [(u32, [f64; 4]); 3] = [
        (
            0,
            [
                17.92,
                22.269_999_999_999_996,
                12.320_000_000_000_002,
                16.4352,
            ],
        ),
        (
            8194,
            [
                17.700_000_000_000_003,
                22.239_999_999_999_995,
                12.380_000_000_000_003,
                16.1852,
            ],
        ),
        (
            1,
            [
                18.720_000_000_000_002,
                21.95,
                12.280_000_000_000_001,
                16.4652,
            ],
        ),
    ];
    for (seed, values) in expected {
        let model = pest::PestControl::new(seed);
        for (x, value) in [&stairs, &none, &all3, &random].into_iter().zip(values) {
            assert_eq!(model.score(x).to_bits(), value.to_bits(), "seed {seed}");
        }
    }
    let problem = problems::by_name("pest_control").unwrap();
    let x: Vec<f64> = random.iter().map(|&c| (c as f64 + 0.5) / 5.0).collect();
    assert_eq!((problem.objective)(&x), 16.4352);
}

// --------------------------------------------------------------- serve

fn request(input: &str) -> String {
    let mut output = Vec::new();
    serve::serve(Cursor::new(input.as_bytes().to_vec()), &mut output).unwrap();
    String::from_utf8(output).unwrap()
}

fn line_of(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| encode(*v))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn serve_evaluates_and_decodes_the_batch_exactly() {
    for name in PROBLEMS {
        let problem = problems::by_name(name).unwrap();
        let table = type_table(name).unwrap();
        let mut rng = Rng::new(0x5E4E_003B);
        let mut points: Vec<Vec<f64>> = (0..60)
            .map(|_| solvers::uniform_point(&problem.lower, &problem.upper, &mut rng))
            .collect();
        points.push(problem.lower.clone());
        points.push(problem.upper.clone());
        let mut input = format!("eval {name} {}\n", points.len());
        for x in &points {
            input.push_str(&line_of(x));
            input.push('\n');
        }
        input.push_str(&format!("levels {name} {}\n", points.len()));
        for x in &points {
            input.push_str(&line_of(x));
            input.push('\n');
        }
        let output = request(&input);
        let mut lines = output.lines();
        assert!(
            lines
                .next()
                .unwrap()
                .starts_with(&format!("ok {} ", points.len()))
        );
        for x in &points {
            let y = decode(lines.next().unwrap()).unwrap();
            assert_eq!(y.to_bits(), (problem.objective)(x).to_bits(), "{name}");
        }
        assert_eq!(lines.next().unwrap(), format!("ok {}", points.len()));
        for x in &points {
            let expected: Vec<String> = table
                .coordinates
                .iter()
                .map(|c| match c.kind {
                    Kind::Continuous => "-".to_owned(),
                    _ => table_level(c, x[c.index]).to_string(),
                })
                .collect();
            assert_eq!(lines.next().unwrap(), expected.join(" "), "{name}");
        }
        assert!(lines.next().is_none());
    }
}
