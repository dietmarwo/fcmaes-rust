//! The thirteen problems of protocol section 3.2 ([`all`]) and the two
//! problems examined for round 2 ([`optional_round2`]).
//!
//! - GTOP: objectives from `crates/fcmaes-gtop`; the bounds and the GTOC1
//!   shift are restated from `examples/src/problems.rs` (the examples crate
//!   pulls in tokio and a finance client, so it is not a dependency).
//! - `optical_lens`: `tutorials/optical-lens-design`, the cost of its
//!   single-objective arms (`optimize_so`):
//!   `evaluate(x, grid).scalar_score()`, `INVALID_COST` (1e12) when the
//!   design does not trace, with the publication pupil grid radius 8.
//! - `bandpass`: `tutorials/sindr-circuit-design`, the cost of its
//!   single-objective arms (`so::optimize_arm`): `evaluate_bandpass(u,
//!   points).objective`, `INVALID_COST` (1e12) on failure, with the
//!   publication sweep of 201 points, on the unit box.
//! - `phased_array`: `tutorials/phased-array-codebook`, the cost minimized by
//!   `geometry::optimize_geometry`: `evaluate_geometry(controls,
//!   points).objective`, `INVALID_COST` (1e12) on failure, with the
//!   publication cut of 4,001 points, on the unit box (D = 96).
//! - `lj13`, `lj38`: `foundations` Lennard-Jones suite, free Cartesian
//!   parameterization in the suite's box `[-1.5 N^(1/3), 1.5 N^(1/3)]^(3N)`;
//!   an evaluation error maps to 1e12, the value the suite's campaign
//!   returns for a refused evaluation. (Errors cannot occur inside the box.)
//!
//! Round 2 (`ROUND2.md`, section 3):
//!
//! - `truss_sizing`: `tutorials/truss-sizing`, the cost of its scalar arms
//!   (`so::optimize_arm`): `evaluate(controls, &GroundStructure::reference(),
//!   Scenario::TRAINING, false, &counter).objective` (penalized mass),
//!   `INVALID_COST` (1e12) when the controls do not decode, on the unit box
//!   (D = 171). The work counter only counts calls.
//! - `dispersion`: `tutorials/dispersion-source-localization`, the cost of its
//!   scalar arm (`optimize_scalar`): `scalar_objective(values,
//!   &Dataset::synthetic())`, which itself returns 1e99 for an invalid design,
//!   in the tutorial's bounds (D = 12).

use std::f64::consts::PI;
use std::sync::Arc;

use fcmaes_foundations::suites::Suite;
use fcmaes_foundations::suites::lennard_jones::{LennardJones, Parameterization};

/// Boxed scalar objective.
pub type ObjectiveFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// One benchmark problem.
pub struct Problem {
    /// Stable identifier used in result files.
    pub name: &'static str,
    /// Application domain.
    pub domain: &'static str,
    /// Lower bounds.
    pub lower: Vec<f64>,
    /// Upper bounds.
    pub upper: Vec<f64>,
    /// Objective to minimize.
    pub objective: ObjectiveFn,
}

impl Problem {
    /// Number of decision variables.
    pub fn dim(&self) -> usize {
        self.lower.len()
    }
}

const SPACE: &str = "space";

fn gtop(
    name: &'static str,
    lower: &[f64],
    upper: &[f64],
    f: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
) -> Problem {
    Problem {
        name,
        domain: SPACE,
        lower: lower.to_vec(),
        upper: upper.to_vec(),
        objective: Box::new(f),
    }
}

fn cassini1() -> Problem {
    gtop(
        "cassini1",
        &[-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
        &[0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
        fcmaes_gtop::cassini1,
    )
}

fn gtoc1() -> Problem {
    // Shifted as in the examples registry (ranks are unaffected).
    gtop(
        "gtoc1",
        &[3000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0],
        &[
            10000.0, 2000.0, 2000.0, 2000.0, 2000.0, 9000.0, 9000.0, 9000.0,
        ],
        |x| fcmaes_gtop::gtoc1(x) - 2_000_000.0,
    )
}

fn sagas() -> Problem {
    gtop(
        "sagas",
        &[
            7000.0, 0.0, 0.0, 0.0, 50.0, 300.0, 0.01, 0.01, 1.05, 8.0, -PI, -PI,
        ],
        &[
            9100.0, 7.0, 1.0, 1.0, 2000.0, 2000.0, 0.9, 0.9, 7.0, 500.0, PI, PI,
        ],
        fcmaes_gtop::sagas,
    )
}

fn messenger() -> Problem {
    gtop(
        "messenger",
        &[
            1000.0, 1.0, 0.0, 0.0, 200.0, 30.0, 30.0, 30.0, 0.01, 0.01, 0.01, 0.01, 1.1, 1.1, 1.1,
            -PI, -PI, -PI,
        ],
        &[
            4000.0, 5.0, 1.0, 1.0, 400.0, 400.0, 400.0, 400.0, 0.99, 0.99, 0.99, 0.99, 6.0, 6.0,
            6.0, PI, PI, PI,
        ],
        fcmaes_gtop::messenger,
    )
}

fn tandem() -> Problem {
    // The constrained Tandem instance with sequence [3, 2, 3, 3, 6]
    // ("Tandem 6" of the examples registry).
    gtop(
        "tandem",
        &[
            5475.0, 2.5, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05,
            1.05, -PI, -PI, -PI,
        ],
        &[
            9132.0, 4.9, 1.0, 1.0, 2500.0, 2500.0, 2500.0, 2500.0, 0.99, 0.99, 0.99, 0.99, 10.0,
            10.0, 10.0, PI, PI, PI,
        ],
        |x| fcmaes_gtop::tandem(x, &[3, 2, 3, 3, 6]),
    )
}

fn cassini2() -> Problem {
    gtop(
        "cassini2",
        &[
            -1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.05, 1.05, 1.15, 1.7, -PI, -PI, -PI, -PI,
        ],
        &[
            0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0,
            6.0, 6.5, 291.0, PI, PI, PI, PI,
        ],
        fcmaes_gtop::cassini2,
    )
}

fn rosetta() -> Problem {
    gtop(
        "rosetta",
        &[
            1460.0, 3.0, 0.0, 0.0, 300.0, 150.0, 150.0, 300.0, 700.0, 0.01, 0.01, 0.01, 0.01, 0.01,
            1.05, 1.05, 1.05, 1.05, -PI, -PI, -PI, -PI,
        ],
        &[
            1825.0, 5.0, 1.0, 1.0, 500.0, 800.0, 800.0, 800.0, 1850.0, 0.9, 0.9, 0.9, 0.9, 0.9,
            9.0, 9.0, 9.0, 9.0, PI, PI, PI, PI,
        ],
        fcmaes_gtop::rosetta,
    )
}

fn messenger_full() -> Problem {
    gtop(
        "messenger_full",
        &[
            1900.0, 3.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 1.1, 1.1, 1.05, 1.05, 1.05, -PI, -PI, -PI, -PI, -PI,
        ],
        &[
            2200.0, 4.05, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 550.0, 0.99, 0.99, 0.99,
            0.99, 0.99, 0.99, 6.0, 6.0, 6.0, 6.0, 6.0, PI, PI, PI, PI, PI,
        ],
        fcmaes_gtop::messenger_full,
    )
}

/// Pupil grid radius of the lens tutorial's publication preset.
pub const LENS_GRID_RADIUS: usize = optical_lens_design::PUBLICATION_GRID_RADIUS;

fn optical_lens() -> Problem {
    Problem {
        name: "optical_lens",
        domain: "optics",
        lower: optical_lens_design::LOWER_BOUNDS.to_vec(),
        upper: optical_lens_design::UPPER_BOUNDS.to_vec(),
        objective: Box::new(|x| {
            optical_lens_design::evaluate(x, LENS_GRID_RADIUS)
                .map(|evaluation| evaluation.scalar_score())
                .unwrap_or(optical_lens_design::INVALID_COST)
        }),
    }
}

/// AC sweep points of the circuit tutorial's publication preset.
pub const BANDPASS_POINTS: usize = sindr_circuit_design::PUBLICATION_SO_POINTS;

fn bandpass() -> Problem {
    let dim = sindr_circuit_design::BANDPASS_DIMENSION;
    Problem {
        name: "bandpass",
        domain: "circuits",
        lower: vec![0.0; dim],
        upper: vec![1.0; dim],
        objective: Box::new(|u| {
            sindr_circuit_design::so::evaluate_bandpass(u, BANDPASS_POINTS)
                .map(|evaluation| evaluation.objective)
                .unwrap_or(sindr_circuit_design::INVALID_COST)
        }),
    }
}

/// Pattern cut points of the phased-array tutorial's publication preset.
pub const PHASED_ARRAY_POINTS: usize = 4_001;

fn phased_array() -> Problem {
    use phased_array_codebook::geometry::{DIMENSION, evaluate_geometry};
    Problem {
        name: "phased_array",
        domain: "antennas",
        lower: vec![0.0; DIMENSION],
        upper: vec![1.0; DIMENSION],
        objective: Box::new(|controls| {
            evaluate_geometry(controls, PHASED_ARRAY_POINTS)
                .map(|evaluation| evaluation.objective)
                .unwrap_or(phased_array_codebook::INVALID_COST)
        }),
    }
}

/// Cost of a refused Lennard-Jones evaluation in the suite's campaign.
pub const LJ_INVALID_COST: f64 = 1.0e12;

fn lennard_jones(name: &'static str, atoms: usize) -> Problem {
    let problem = LennardJones::new(atoms, Parameterization::Free).expect("at least three atoms");
    let (lower, upper) = problem.bounds();
    Problem {
        name,
        domain: "chemistry",
        lower,
        upper,
        objective: Box::new(move |x| {
            problem
                .energy(x)
                .map(|evaluation| evaluation.energy)
                .unwrap_or(LJ_INVALID_COST)
        }),
    }
}

fn truss_sizing() -> Problem {
    use truss_sizing::decode::dimension;
    use truss_sizing::evaluate::evaluate;
    use truss_sizing::fem::{Scenario, WorkCounter};
    use truss_sizing::ground::GroundStructure;
    let ground = Arc::new(GroundStructure::reference());
    let dim = dimension(&ground);
    let counter = WorkCounter::default();
    Problem {
        name: "truss_sizing",
        domain: "structures",
        lower: vec![0.0; dim],
        upper: vec![1.0; dim],
        objective: Box::new(move |controls| {
            evaluate(controls, &ground, Scenario::TRAINING, false, &counter)
                .map_or(truss_sizing::INVALID_COST, |evaluation| {
                    evaluation.objective
                })
        }),
    }
}

fn dispersion() -> Problem {
    use dispersion_source_localization::{Dataset, lower_bounds, scalar_objective, upper_bounds};
    let dataset = Dataset::synthetic();
    Problem {
        name: "dispersion",
        domain: "environment",
        lower: lower_bounds().to_vec(),
        upper: upper_bounds().to_vec(),
        objective: Box::new(move |values| scalar_objective(values, &dataset)),
    }
}

/// All round-1 problems in protocol order.
pub fn all() -> Vec<Problem> {
    vec![
        cassini1(),
        gtoc1(),
        sagas(),
        messenger(),
        tandem(),
        cassini2(),
        rosetta(),
        messenger_full(),
        optical_lens(),
        bandpass(),
        phased_array(),
        lennard_jones("lj13", 13),
        lennard_jones("lj38", 38),
    ]
}

/// The two problems examined for round 2 (`ROUND2.md`, section 3).
pub fn optional_round2() -> Vec<Problem> {
    vec![truss_sizing(), dispersion()]
}

/// Every registered problem: [`all`] followed by [`optional_round2`].
pub fn catalog() -> Vec<Problem> {
    let mut problems = all();
    problems.extend(optional_round2());
    problems
}

/// The ten round-1 problems that enter round 2 (mean evaluation time at
/// most 25 µs, `ROUND2.md` section 3).
pub const ROUND2_REQUIRED: [&str; 10] = [
    "cassini1",
    "gtoc1",
    "sagas",
    "messenger",
    "tandem",
    "cassini2",
    "rosetta",
    "messenger_full",
    "lj13",
    "lj38",
];

/// Optional problems that passed the round-2 entry rule (timing only).
///
/// None did: in `results/round2/budgets.tsv` `truss_sizing` measured
/// 36.7 µs and `dispersion` 43.3 µs per evaluation, above the 25 µs limit.
/// Both stay registered (for `measure` and tests) but are not in the
/// round-2 set.
pub const ROUND2_ENTERED: [&str; 0] = [];

/// The round-2 problem set (`--problems round2`): [`ROUND2_REQUIRED`]
/// followed by [`ROUND2_ENTERED`].
pub fn round2_names() -> Vec<&'static str> {
    ROUND2_REQUIRED
        .iter()
        .chain(&ROUND2_ENTERED)
        .copied()
        .collect()
}

/// Look up a problem by name in the [`catalog`], then among the round-3
/// problems ([`crate::round3::by_name`]) and the round-3b problems
/// ([`crate::round3b::by_name`]), which are built only on request.
pub fn by_name(name: &str) -> Option<Problem> {
    catalog()
        .into_iter()
        .find(|problem| problem.name == name)
        .or_else(|| crate::round3::by_name(name))
        .or_else(|| crate::round3b::by_name(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fcmaes_core::Rng;

    #[test]
    fn registry_matches_protocol_dimensions() {
        let dims: Vec<(&str, usize)> = all().iter().map(|p| (p.name, p.dim())).collect();
        assert_eq!(
            dims,
            vec![
                ("cassini1", 6),
                ("gtoc1", 8),
                ("sagas", 12),
                ("messenger", 18),
                ("tandem", 18),
                ("cassini2", 22),
                ("rosetta", 22),
                ("messenger_full", 26),
                ("optical_lens", 11),
                ("bandpass", 5),
                ("phased_array", 96),
                ("lj13", 39),
                ("lj38", 114),
            ]
        );
    }

    #[test]
    fn optional_problems_have_the_tutorial_dimensions() {
        let dims: Vec<(&str, &str, usize)> = optional_round2()
            .iter()
            .map(|p| (p.name, p.domain, p.dim()))
            .collect();
        assert_eq!(
            dims,
            vec![
                ("truss_sizing", "structures", 171),
                ("dispersion", "environment", 12)
            ]
        );
        for name in round2_names() {
            assert!(by_name(name).is_some(), "{name}");
        }
        assert_eq!(round2_names().len(), 10);
    }

    #[test]
    fn every_problem_is_deterministic() {
        let mut rng = Rng::new(11);
        for problem in catalog() {
            for _ in 0..20 {
                let x = crate::solvers::uniform_point(&problem.lower, &problem.upper, &mut rng);
                let first = (problem.objective)(&x);
                let second = (problem.objective)(&x);
                assert_eq!(first.to_bits(), second.to_bits(), "{}", problem.name);
            }
        }
    }

    #[test]
    fn optional_problems_reproduce_tutorial_values() {
        // Truss: the tutorial's braced baseline evaluates to a finite mass
        // below the invalid cost.
        let ground = truss_sizing::ground::GroundStructure::reference();
        let baseline = truss_sizing::decode::baseline_controls(&ground);
        let truss = by_name("truss_sizing").unwrap();
        let value = (truss.objective)(&baseline);
        assert!(value.is_finite() && value < truss_sizing::INVALID_COST);
        // Dispersion: the generating design fits its own synthetic data
        // better than the tutorial's baseline guess.
        use dispersion_source_localization::{Dataset, Design, scalar_objective};
        let dispersion = by_name("dispersion").unwrap();
        let truth = (dispersion.objective)(Design::truth().values());
        let baseline = (dispersion.objective)(Design::baseline().values());
        assert_eq!(
            truth.to_bits(),
            scalar_objective(Design::truth().values(), &Dataset::synthetic()).to_bits()
        );
        assert!(truth < baseline);
    }

    #[test]
    fn every_problem_is_finite_at_random_points() {
        let mut rng = Rng::new(7);
        for problem in catalog() {
            assert_eq!(problem.lower.len(), problem.upper.len());
            assert!(problem.lower.iter().zip(&problem.upper).all(|(l, u)| l < u));
            for _ in 0..5 {
                let x = crate::solvers::uniform_point(&problem.lower, &problem.upper, &mut rng);
                let y = (problem.objective)(&x);
                assert!(y.is_finite(), "{} gave {y}", problem.name);
            }
        }
    }

    #[test]
    fn reference_points_reproduce_tutorial_values() {
        // The lens reference design traces and is cheaper than the invalid cost.
        let lens = by_name("optical_lens").unwrap();
        let reference = (lens.objective)(&optical_lens_design::REFERENCE_DESIGN);
        assert!(reference < optical_lens_design::INVALID_COST);
        // The phased-array seed is the tutorial's feasible start.
        let array = by_name("phased_array").unwrap();
        let seed = phased_array_codebook::geometry::geometry_seed();
        assert!((array.objective)(&seed) < 0.0);
    }
}
