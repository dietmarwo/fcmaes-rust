//! Round-3 type tables (ROUND3.md section 4), checked against the problems'
//! own decoders:
//!
//! - every level of every discrete coordinate, placed at its interval
//!   center, decodes to itself;
//! - points just below and just above every interval boundary decode to the
//!   two neighboring levels;
//! - moving every discrete coordinate of a random point to the center of
//!   the level it decodes to leaves the objective bit-identical, so the
//!   typed-to-real mapping composed with the objective is the objective at
//!   the mapped point and the table captures all discrete structure.

use candidate_harness::problems;
use candidate_harness::round3::{
    self, Coordinate, ENTERED, EXAMINED, Kind, Probe, TypeTable, de_mask, level_probe, type_table,
};
use fcmaes_core::Rng;

/// Level whose intervals contain `v` according to the table.
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

fn check_levels(table: &TypeTable, probe: &Probe) {
    for c in table
        .coordinates
        .iter()
        .filter(|c| c.kind != Kind::Continuous)
    {
        assert!(c.levels.len() >= 2, "{} {}", table.problem, c.index);
        for (level, l) in c.levels.iter().enumerate() {
            assert!(l.center >= c.lower && l.center <= c.upper);
            assert_eq!(
                probe(c.index, l.center).unwrap(),
                level,
                "{} coordinate {} level {} ({}) at center {}",
                table.problem,
                c.index,
                level,
                l.label,
                l.center
            );
        }
        // Boundaries between consecutive intervals, in value order.
        let mut intervals: Vec<(f64, f64, usize)> = c
            .levels
            .iter()
            .enumerate()
            .flat_map(|(level, l)| l.intervals.iter().map(move |&(lo, hi)| (lo, hi, level)))
            .collect();
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(intervals[0].0, c.lower, "{} {}", table.problem, c.index);
        assert_eq!(intervals.last().unwrap().1, c.upper);
        let delta = 1e-9 * (c.upper - c.lower);
        for pair in intervals.windows(2) {
            let (left, right) = (pair[0], pair[1]);
            assert_eq!(left.1, right.0, "{} {}: gap", table.problem, c.index);
            let b = left.1;
            assert_eq!(
                probe(c.index, b - delta).unwrap(),
                left.2,
                "{} {} below {b}",
                table.problem,
                c.index
            );
            let above = if right.0 == right.1 {
                right.1
            } else {
                b + delta
            };
            assert_eq!(
                probe(c.index, above).unwrap(),
                right.2,
                "{} {} above {b}",
                table.problem,
                c.index
            );
        }
    }
}

fn check_snapping(name: &str, table: &TypeTable, points: usize) {
    let problem = problems::by_name(name).unwrap();
    let mut rng = Rng::new(0x5EED_0003);
    for _ in 0..points {
        let x = candidate_harness::solvers::uniform_point(&problem.lower, &problem.upper, &mut rng);
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
            "{name}: snapping discrete coordinates changed the objective"
        );
    }
}

// ------------------------------------------------------------------- tests

#[test]
fn every_examined_problem_has_a_consistent_table() {
    for name in EXAMINED {
        let table = type_table(name).unwrap();
        let problem = problems::by_name(name).unwrap();
        assert_eq!(table.dim(), problem.dim(), "{name}");
        for (i, c) in table.coordinates.iter().enumerate() {
            assert_eq!(c.index, i);
            assert_eq!(
                (c.lower, c.upper),
                (problem.lower[i], problem.upper[i]),
                "{name} {i}"
            );
            assert_eq!(
                c.kind == Kind::Continuous,
                c.levels.is_empty(),
                "{name} {i}"
            );
            for l in &c.levels {
                assert_eq!(l.value.is_some(), c.kind == Kind::Integer, "{name} {i}");
            }
        }
        // The JSON has one entry per coordinate.
        assert_eq!(table.to_json().matches("\"index\":").count(), table.dim());
    }
}

#[test]
fn entered_problems_decode_every_level_to_itself() {
    for name in ENTERED {
        let table = type_table(name).unwrap();
        let probe = level_probe(name).expect("probe for every entered problem");
        check_levels(&table, &probe);
    }
}

#[test]
fn snapping_to_level_centers_keeps_the_objective() {
    for name in ENTERED {
        let table = type_table(name).unwrap();
        check_snapping(name, &table, 50);
    }
}

#[test]
fn only_network_coverage_passes_a_de_mask() {
    for name in EXAMINED {
        let mask = de_mask(name);
        if name.starts_with("coverage_") {
            assert_eq!(mask.unwrap(), vec![true; 4_000]);
        } else {
            assert!(mask.is_none(), "{name}");
        }
    }
}

#[test]
fn truss_sections_are_sorted_by_size() {
    let sections = truss_sizing::catalogue::sections();
    assert!(sections.windows(2).all(|w| w[0].area_m2 < w[1].area_m2
        && w[0].inertia_m4 < w[1].inertia_m4
        && w[0].outer_diameter_m < w[1].outer_diameter_m));
    let tiers = energy_hub_bilevel::decode::GRID_TIERS_KW;
    assert!(tiers.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn coverage_objective_restatement_matches_the_tutorial_seed_value() {
    // The tutorial reports its cardinality seed through the same formula:
    // the selected count plus twice the uncovered edges. A full selection
    // covers every edge.
    let instance = round3::coverage_instance();
    let all = vec![true; instance.nodes()];
    assert_eq!(
        round3::coverage_objective_value(&instance, &all, false),
        instance.nodes() as f64
    );
    let none = vec![false; instance.nodes()];
    assert_eq!(
        round3::coverage_objective_value(&instance, &none, false),
        2.0 * instance.edges.len() as f64
    );
    let problem = problems::by_name("coverage_weighted").unwrap();
    let x = vec![1.5; instance.nodes()];
    assert_eq!(
        problem.objective.as_ref()(&x),
        instance.costs.iter().sum::<f64>()
    );
}
