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
    self, Coordinate, ENTERED, EXAMINED, Kind, TypeTable, de_mask, type_table,
};
use fcmaes_core::Rng;

/// Level index of `coordinate` at value `value`, read from the problem's own
/// decoder with every other coordinate at a fixed base point.
type Probe = Box<dyn Fn(usize, f64) -> usize>;

fn level_named(c: &Coordinate, label: &str) -> usize {
    c.levels
        .iter()
        .position(|l| l.label == label)
        .unwrap_or_else(|| panic!("coordinate {}: no level '{label}'", c.index))
}

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
                probe(c.index, l.center),
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
                probe(c.index, b - delta),
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
                probe(c.index, above),
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

// ------------------------------------------------------------------ probes

fn truss_probe() -> Probe {
    use truss_sizing::decode::{MIN_ACTIVE, decode, dimension};
    use truss_sizing::ground::GroundStructure;
    let ground = GroundStructure::reference();
    let members = ground.members.len();
    Box::new(move |index, value| {
        let mut x = vec![0.5; dimension(&ground)];
        if index == 0 {
            x[0] = value;
            decode(&x, &ground).unwrap().active.len() - MIN_ACTIVE
        } else {
            let member = index - 1 - members;
            x[0] = 1.0;
            x[1 + member] = 0.0; // lowest ranking key: always active
            x[index] = value;
            let decoded = decode(&x, &ground).unwrap();
            decoded
                .active
                .iter()
                .find(|a| a.member_index == member)
                .unwrap()
                .section_index
        }
    })
}

fn phased_array_probe() -> Probe {
    use phased_array_codebook::geometry::{DIMENSION, SLOTS, evaluate_geometry};
    Box::new(move |index, value| {
        let phase = index < 3 * SLOTS;
        let slot = if phase {
            index - 2 * SLOTS
        } else {
            index - 3 * SLOTS
        };
        let mut x = vec![0.5; DIMENSION];
        x[slot] = 0.0; // selected
        x[index] = value;
        let evaluation = evaluate_geometry(&x, 101).unwrap();
        // Selected: the slot (key 0) and the first 15 other slots (key 0.5,
        // ties by index), in index order.
        let position = slot.min(15);
        let codes = if phase {
            &evaluation.excitation.phase_codes
        } else {
            &evaluation.excitation.attenuator_codes
        };
        codes[position] as usize
    })
}

fn field_service_probe(table: TypeTable) -> Probe {
    use field_service_routing::decode::decode_active;
    use field_service_routing::scenarios::training;
    let instance = round3::field_service_instance();
    let scenarios = training(&instance);
    Box::new(move |index, value| {
        let mut x = vec![0.5; 2 * instance.tasks.len()];
        x[index] = value;
        let mut seen: Vec<Vec<bool>> = Vec::new();
        let mut chosen = Vec::new();
        for case in &scenarios {
            if !case.active[index] || seen.contains(&case.available) {
                continue;
            }
            seen.push(case.available.clone());
            let decoded = decode_active(&x, &case.instance, &case.active, &case.available).unwrap();
            let route = decoded
                .routes
                .iter()
                .find(|r| r.tasks.contains(&index))
                .unwrap();
            chosen.push(route.vehicle.to_string());
        }
        level_named(
            &table.coordinates[index],
            &format!("vehicle {}", chosen.join(" / ")),
        )
    })
}

fn coverage_probe() -> Probe {
    Box::new(|_, value| usize::from(network_coverage::decode::decode(&[value]).unwrap()[0]))
}

fn jobshop_probe() -> Probe {
    use candidate_harness::examples::jobshop::JobShop;
    let problem = JobShop::mini();
    Box::new(move |index, value| {
        let mut x = vec![0.5; problem.dimension()];
        x[index] = value;
        let alternatives = &problem.operations[index];
        let task = problem
            .decode(&x)
            .into_iter()
            .find(|t| t.job == alternatives[0].job && t.operation == alternatives[0].operation)
            .unwrap();
        alternatives
            .iter()
            .position(|a| a.machine == task.machine)
            .unwrap()
    })
}

fn uav_probe() -> Probe {
    use candidate_harness::examples::uav::UavProblem;
    let base = round3::uav_instance();
    // Same instance without a horizon, so every target appears in a route.
    let open = UavProblem::try_new(base.speeds().to_vec(), base.targets().to_vec(), 1e300).unwrap();
    let separators = base.vehicle_count() - 1;
    let targets = base.target_count();
    Box::new(move |index, value| {
        let mut x = vec![0.999; separators + targets];
        for t in 0..targets {
            x[separators + t] = t as f64 / targets as f64;
        }
        x[index] = value;
        open.solution(&x).assignments[0].len()
    })
}

fn material_flow_probe() -> Probe {
    use candidate_harness::examples::material_flow_planning::{DAY_SECONDS, Plant};
    let plant = Plant::original();
    Box::new(move |index, value| {
        let mut x = vec![17.5; 2];
        x[index] = value;
        let fitness = plant.fitness(&x);
        // `fitness` is the only decoder; identify the level by replaying the
        // simulation for each candidate batch size (unique match required).
        let matches: Vec<usize> = (1..=50)
            .filter(|&k| {
                let mut batches = [17, 17];
                batches[index] = k;
                let mut sim = plant.clone();
                -(sim.simulate(DAY_SECONDS, &batches) as f64) == fitness
            })
            .collect();
        let expected = (value.clamp(1.0, 50.0) as usize).max(1);
        assert!(matches.contains(&expected));
        expected - 1
    })
}

fn probe_for(name: &str, table: &TypeTable) -> Option<Probe> {
    Some(match name {
        "truss_sizing" => truss_probe(),
        "phased_array" => phased_array_probe(),
        "field_service" => field_service_probe(table.clone()),
        "coverage_cardinality" | "coverage_weighted" => coverage_probe(),
        "jobshop" => jobshop_probe(),
        "uav_assignment" => uav_probe(),
        "material_flow" => material_flow_probe(),
        _ => return None,
    })
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
        let probe = probe_for(name, &table).expect("probe for every entered problem");
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
