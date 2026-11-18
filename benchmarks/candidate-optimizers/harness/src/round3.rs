//! Round 3 (`ROUND3.md`): mixed-variable problems, their variable-type
//! tables and the integer masks of their own DE calls.
//!
//! # Problems
//!
//! Every problem ROUND3.md section 3 names, and every other problem of the
//! repository with a single-objective arm and a discrete decoding, is built
//! here (or, for `truss_sizing`, `phased_array` and `bandpass`, in
//! [`crate::problems`]). Each objective calls the problem's own public
//! function; where the scalar wrapper of the problem's single-objective code
//! is private, it is restated here and marked as such:
//!
//! - `coverage_cardinality`, `coverage_weighted`: `objective_value` of
//!   `tutorials/network-coverage/src/so.rs` is private and is restated
//!   exactly (the decoding `decode::decode` is the tutorial's public one).
//! - `energy_hub_annual`: the invalid cost 1e6 of `annual.rs` is a private
//!   constant and is restated.
//!
//! The examples crate is not a dependency (it pulls in tokio and a finance
//! client); `jobshop`, `uav_assignment` and `material_flow` compile the
//! example modules from their source files ([`crate::examples`]).
//!
//! # Type tables
//!
//! [`type_table`] assigns each coordinate one type (continuous, integer or
//! categorical) from the decoding code alone (ROUND3.md section 4). For a
//! discrete coordinate it lists every level with the real-vector intervals
//! that decode to it and the interval center the candidate's typed point is
//! mapped to. Intervals are half-open `[lo, hi)` as the floor-type decoders
//! define them; an interval ending at the upper bound includes it. The tests
//! in `tests/round3.rs` check every level against the problem's own decoder.

use std::sync::Arc;

use crate::problems::{ObjectiveFn, Problem};

/// Variable type of one coordinate (ROUND3.md section 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The objective uses the real value.
    Continuous,
    /// Ordered discrete levels.
    Integer,
    /// Unordered choices.
    Categorical,
}

impl Kind {
    /// Name used in the JSON tables.
    pub fn name(self) -> &'static str {
        match self {
            Self::Continuous => "continuous",
            Self::Integer => "integer",
            Self::Categorical => "categorical",
        }
    }
}

/// One level of a discrete coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct Level {
    /// Numeric level value (integer coordinates): the integer the decoder
    /// computes (count, index or code).
    pub value: Option<i64>,
    /// Human-readable meaning of the level.
    pub label: String,
    /// Real-vector intervals `[lo, hi)` that decode to this level (an
    /// interval with `hi` equal to the upper bound includes it).
    pub intervals: Vec<(f64, f64)>,
    /// Real-vector point the typed candidate is mapped to: the center of the
    /// first interval.
    pub center: f64,
}

/// One coordinate of a type table.
#[derive(Clone, Debug, PartialEq)]
pub struct Coordinate {
    /// Coordinate index.
    pub index: usize,
    /// Type.
    pub kind: Kind,
    /// Lower bound of the box.
    pub lower: f64,
    /// Upper bound of the box.
    pub upper: f64,
    /// Levels (empty for continuous coordinates).
    pub levels: Vec<Level>,
    /// What the coordinate means in the decoder.
    pub meaning: String,
}

/// Per-coordinate type table of one problem.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeTable {
    /// Problem name.
    pub problem: &'static str,
    /// Source file and function the classification came from.
    pub source: &'static str,
    /// Coordinates in index order.
    pub coordinates: Vec<Coordinate>,
    /// Remarks on the classification.
    pub notes: Vec<String>,
}

impl TypeTable {
    /// Dimension.
    pub fn dim(&self) -> usize {
        self.coordinates.len()
    }

    /// Number of coordinates of `kind`.
    pub fn count(&self, kind: Kind) -> usize {
        self.coordinates.iter().filter(|c| c.kind == kind).count()
    }

    /// Map a typed point to the real vector: discrete coordinates go to the
    /// center of their level (`levels[i]` is the level index), continuous
    /// coordinates are clipped to the box. `typed[i]` is the level index for
    /// a discrete coordinate and the real value otherwise.
    ///
    /// # Panics
    /// Panics if the length differs or a level index is out of range.
    pub fn to_real(&self, typed: &[f64]) -> Vec<f64> {
        assert_eq!(typed.len(), self.dim());
        self.coordinates
            .iter()
            .zip(typed)
            .map(|(c, &t)| match c.kind {
                Kind::Continuous => t.clamp(c.lower, c.upper),
                _ => c.levels[t as usize].center,
            })
            .collect()
    }

    /// JSON representation (floats in shortest round-trip form).
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"problem\": \"{}\",\n", self.problem));
        s.push_str(&format!("  \"dimension\": {},\n", self.dim()));
        s.push_str(&format!("  \"source\": {},\n", json_string(self.source)));
        s.push_str(&format!(
            "  \"counts\": {{\"continuous\": {}, \"integer\": {}, \"categorical\": {}}},\n",
            self.count(Kind::Continuous),
            self.count(Kind::Integer),
            self.count(Kind::Categorical)
        ));
        s.push_str("  \"notes\": [");
        s.push_str(
            &self
                .notes
                .iter()
                .map(|n| json_string(n))
                .collect::<Vec<_>>()
                .join(", "),
        );
        s.push_str("],\n  \"coordinates\": [\n");
        let rows: Vec<String> = self
            .coordinates
            .iter()
            .map(|c| {
                let mut row = format!(
                    "    {{\"index\": {}, \"kind\": \"{}\", \"lower\": {}, \"upper\": {}, \"meaning\": {}",
                    c.index,
                    c.kind.name(),
                    json_f64(c.lower),
                    json_f64(c.upper),
                    json_string(&c.meaning)
                );
                if c.kind != Kind::Continuous {
                    let centers: Vec<String> = c.levels.iter().map(|l| json_f64(l.center)).collect();
                    row.push_str(&format!(", \"levels\": [{}]", centers.join(", ")));
                    if c.kind == Kind::Integer {
                        let values: Vec<String> = c
                            .levels
                            .iter()
                            .map(|l| l.value.expect("integer level value").to_string())
                            .collect();
                        row.push_str(&format!(", \"values\": [{}]", values.join(", ")));
                    }
                    let labels: Vec<String> = c.levels.iter().map(|l| json_string(&l.label)).collect();
                    row.push_str(&format!(", \"labels\": [{}]", labels.join(", ")));
                    let intervals: Vec<String> = c
                        .levels
                        .iter()
                        .map(|l| {
                            let parts: Vec<String> = l
                                .intervals
                                .iter()
                                .map(|(lo, hi)| format!("[{}, {}]", json_f64(*lo), json_f64(*hi)))
                                .collect();
                            format!("[{}]", parts.join(", "))
                        })
                        .collect();
                    row.push_str(&format!(", \"intervals\": [{}]", intervals.join(", ")));
                }
                row.push('}');
                row
            })
            .collect();
        s.push_str(&rows.join(",\n"));
        s.push_str("\n  ]\n}\n");
        s
    }
}

fn json_f64(value: f64) -> String {
    assert!(value.is_finite(), "type tables hold finite bounds only");
    format!("{value:?}")
}

fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn continuous(index: usize, lower: f64, upper: f64, meaning: impl Into<String>) -> Coordinate {
    Coordinate {
        index,
        kind: Kind::Continuous,
        lower,
        upper,
        levels: Vec::new(),
        meaning: meaning.into(),
    }
}

/// `count` equal-width bins of the unit interval (the `floor(u · count)`
/// decoders), with integer values `first, first + 1, ...` when `kind` is
/// integer.
fn equal_bins(
    index: usize,
    kind: Kind,
    count: usize,
    first: i64,
    labels: &dyn Fn(usize) -> String,
    meaning: impl Into<String>,
) -> Coordinate {
    let (lower, upper) = (0.0, 1.0);
    let width = (upper - lower) / count as f64;
    let levels = (0..count)
        .map(|i| {
            let lo = lower + i as f64 * width;
            let hi = if i + 1 == count {
                upper
            } else {
                lower + (i + 1) as f64 * width
            };
            Level {
                value: (kind == Kind::Integer).then_some(first + i as i64),
                label: labels(i),
                intervals: vec![(lo, hi)],
                center: 0.5 * (lo + hi),
            }
        })
        .collect();
    Coordinate {
        index,
        kind,
        lower,
        upper,
        levels,
        meaning: meaning.into(),
    }
}

// ------------------------------------------------------------------ problems

/// Names of every problem examined for round 3 that the harness can build,
/// in survey order. `truss_sizing`, `phased_array` and `bandpass` live in
/// [`crate::problems`].
pub const EXAMINED: [&str; 15] = [
    "truss_sizing",
    "energy_hub",
    "energy_hub_annual",
    "phased_array",
    "phased_array_beam",
    "bandpass",
    "water_network",
    "field_service",
    "coverage_cardinality",
    "coverage_weighted",
    "cassini1_minlp",
    "ml_tuning",
    "jobshop",
    "uav_assignment",
    "material_flow",
];

/// Problems that entered round 3 under the rule of ROUND3.md section 3
/// (`results/round3/problems.tsv`), in campaign order.
pub const ENTERED: [&str; 8] = [
    "truss_sizing",
    "phased_array",
    "field_service",
    "coverage_cardinality",
    "coverage_weighted",
    "jobshop",
    "uav_assignment",
    "material_flow",
];

/// Integer mask of the DE call in the problem's own single-objective code,
/// or `None` if that call passes none (ROUND3.md section 2). Only the
/// network-coverage tutorial passes one (`so.rs`: `Some(vec![true;
/// dimension])`).
pub fn de_mask(name: &str) -> Option<Vec<bool>> {
    match name {
        "coverage_cardinality" | "coverage_weighted" => {
            Some(vec![true; coverage_instance().nodes()])
        }
        _ => None,
    }
}

fn boxed(f: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) -> ObjectiveFn {
    Box::new(f)
}

fn unit_problem(name: &'static str, domain: &'static str, dim: usize, f: ObjectiveFn) -> Problem {
    Problem {
        name,
        domain,
        lower: vec![0.0; dim],
        upper: vec![1.0; dim],
        objective: f,
    }
}

fn energy_hub() -> Problem {
    use energy_hub_bilevel::config::Preset;
    use energy_hub_bilevel::decode::DIMENSION;
    use energy_hub_bilevel::evaluate::{INVALID_OBJECTIVE, evaluate_training};
    unit_problem(
        "energy_hub",
        "energy",
        DIMENSION,
        boxed(|x| {
            evaluate_training(x, Preset::Publication)
                .map_or(INVALID_OBJECTIVE, |evaluation| evaluation.objective)
        }),
    )
}

/// Invalid cost of the annual scalar arm (`const INVALID: f64 = 1.0e6` in
/// `annual.rs`, private there).
pub const ENERGY_ANNUAL_INVALID: f64 = 1.0e6;

fn energy_hub_annual() -> Problem {
    use energy_hub_bilevel::annual::evaluate_annual;
    use energy_hub_bilevel::decode::DIMENSION;
    unit_problem(
        "energy_hub_annual",
        "energy",
        DIMENSION,
        boxed(|x| {
            evaluate_annual(x, 6).map_or(ENERGY_ANNUAL_INVALID, |evaluation| evaluation.objective)
        }),
    )
}

/// Steering angle of the phased-array tutorial's scalar arm (`main.rs`).
pub const BEAM_REQUESTED_DEG: f64 = 20.0;

fn phased_array_beam() -> Problem {
    use phased_array_codebook::so::{BeamContext, DIMENSION, evaluate_beam};
    let context = Arc::new(BeamContext::stage_a(crate::problems::PHASED_ARRAY_POINTS));
    unit_problem(
        "phased_array_beam",
        "antennas",
        DIMENSION,
        boxed(move |x| {
            evaluate_beam(x, &context, BEAM_REQUESTED_DEG)
                .map_or(phased_array_codebook::INVALID_COST, |evaluation| {
                    evaluation.objective
                })
        }),
    )
}

fn water_network() -> Problem {
    use water_network_scheduling::evaluate::evaluate_training;
    use water_network_scheduling::{DIMENSION, INVALID_OBJECTIVE, network};
    let base = Arc::new(network::load().expect("checked-in water network loads"));
    unit_problem(
        "water_network",
        "water",
        DIMENSION,
        boxed(move |x| {
            evaluate_training(x, &base).map_or(INVALID_OBJECTIVE, |evaluation| evaluation.objective)
        }),
    )
}

/// The field-service instance the tutorial's scalar arm uses
/// (`instance::load_primary`).
pub fn field_service_instance() -> field_service_routing::instance::Instance {
    field_service_routing::instance::load_primary().expect("primary field-service instance loads")
}

fn field_service() -> Problem {
    use field_service_routing::INVALID_OBJECTIVE;
    use field_service_routing::instance::DIMENSION;
    use field_service_routing::scenarios::evaluate_training;
    let instance = Arc::new(field_service_instance());
    unit_problem(
        "field_service",
        "logistics",
        DIMENSION,
        boxed(move |x| {
            evaluate_training(x, &instance)
                .map_or(INVALID_OBJECTIVE, |evaluation| evaluation.objective)
        }),
    )
}

/// The publication instance of the coverage tutorial (`reference-4k`,
/// selected by the tutorial's throughput gate; `results/publication`).
pub fn coverage_instance() -> network_coverage::instance::Instance {
    use network_coverage::instance::{FIXTURES, generate};
    let config = FIXTURES
        .iter()
        .find(|config| config.name == "reference-4k")
        .expect("reference-4k fixture");
    generate(config).expect("reference-4k generates")
}

/// `objective_value` of `tutorials/network-coverage/src/so.rs` (private
/// there), restated exactly.
pub fn coverage_objective_value(
    instance: &network_coverage::instance::Instance,
    selected: &[bool],
    weighted: bool,
) -> f64 {
    let uncovered = instance
        .edges
        .iter()
        .filter(|edge| !selected[edge.u] && !selected[edge.v])
        .count();
    if weighted {
        let cost = selected
            .iter()
            .zip(&instance.costs)
            .filter(|(chosen, _)| **chosen)
            .map(|(_, cost)| cost)
            .sum::<f64>();
        cost + (instance.costs.iter().sum::<f64>() + 1.0) * uncovered as f64
    } else {
        selected.iter().filter(|value| **value).count() as f64 + 2.0 * uncovered as f64
    }
}

fn coverage(name: &'static str, weighted: bool) -> Problem {
    use network_coverage::decode::{UPPER, decode};
    let instance = Arc::new(coverage_instance());
    let dim = instance.nodes();
    Problem {
        name,
        domain: "networks",
        lower: vec![0.0; dim],
        upper: vec![UPPER; dim],
        objective: boxed(move |x| {
            decode(x).map_or(f64::INFINITY, |selected| {
                coverage_objective_value(&instance, &selected, weighted)
            })
        }),
    }
}

fn cassini1_minlp() -> Problem {
    // `examples/src/problems.rs`, `cassini1_minlp()`: the registry fixes the
    // flyby sequence to [2, 2, 3, 5].
    Problem {
        name: "cassini1_minlp",
        domain: "space",
        lower: vec![-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
        upper: vec![0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
        objective: boxed(|x| {
            let mut values = Vec::with_capacity(10);
            values.extend_from_slice(x);
            values.extend([2.0, 2.0, 3.0, 5.0]);
            fcmaes_gtop::cassini1_minlp(&values).0
        }),
    }
}

fn ml_tuning() -> Problem {
    use ml_hyperparameter_tuning::data::stream_seed;
    use ml_hyperparameter_tuning::{DIMENSION, DataConfig, Dataset, Evaluator, Preset};
    // `main.rs`, publication preset: seed 42, minimum recall 0.25, tuning
    // model seed `stream_seed(seed, 50)`.
    let dataset =
        Arc::new(Dataset::generate(DataConfig::for_preset(Preset::Publication)).expect("dataset"));
    let evaluator = Arc::new(Evaluator::new(dataset, 0.25, stream_seed(42, 50)));
    unit_problem(
        "ml_tuning",
        "machine learning",
        DIMENSION,
        boxed(move |x| evaluator.evaluate(x).scalar_fitness),
    )
}

/// Box of the jobshop example (`src/bin/jobshop.rs`).
pub const JOBSHOP_LOWER: f64 = 1e-7;
/// Box of the jobshop example (`src/bin/jobshop.rs`).
pub const JOBSHOP_UPPER: f64 = 0.999_999_9;

fn jobshop() -> Problem {
    use crate::examples::jobshop::JobShop;
    let problem = Arc::new(JobShop::mini());
    let dim = problem.dimension();
    Problem {
        name: "jobshop",
        domain: "scheduling",
        lower: vec![JOBSHOP_LOWER; dim],
        upper: vec![JOBSHOP_UPPER; dim],
        objective: boxed(move |x| problem.fitness(x)),
    }
}

/// The UAV instance of `src/bin/uav_task_assignment.rs` (defaults: 5
/// vehicles, 30 targets, map size 5,000, seed 65).
pub fn uav_instance() -> crate::examples::uav::UavProblem {
    crate::examples::uav::UavProblem::generate(5, 30, 5_000, 65).expect("UAV instance")
}

fn uav_assignment() -> Problem {
    let problem = Arc::new(uav_instance());
    unit_problem(
        "uav_assignment",
        "logistics",
        problem.dimension(),
        boxed(move |x| problem.fitness(x)),
    )
}

fn material_flow() -> Problem {
    use crate::examples::material_flow_planning::Plant;
    let plant = Arc::new(Plant::original());
    let dim = plant.dim();
    Problem {
        name: "material_flow",
        domain: "manufacturing",
        lower: vec![1.0; dim],
        upper: vec![50.0; dim],
        objective: boxed(move |x| plant.fitness(x)),
    }
}

/// Build a round-3 problem by name (also the round-1 problems it reuses).
pub fn by_name(name: &str) -> Option<Problem> {
    Some(match name {
        "energy_hub" => energy_hub(),
        "energy_hub_annual" => energy_hub_annual(),
        "phased_array_beam" => phased_array_beam(),
        "water_network" => water_network(),
        "field_service" => field_service(),
        "coverage_cardinality" => coverage("coverage_cardinality", false),
        "coverage_weighted" => coverage("coverage_weighted", true),
        "cassini1_minlp" => cassini1_minlp(),
        "ml_tuning" => ml_tuning(),
        "jobshop" => jobshop(),
        "uav_assignment" => uav_assignment(),
        "material_flow" => material_flow(),
        _ => return None,
    })
}

/// Round-3 problem names built by [`by_name`].
pub const OWN: [&str; 12] = [
    "energy_hub",
    "energy_hub_annual",
    "phased_array_beam",
    "water_network",
    "field_service",
    "coverage_cardinality",
    "coverage_weighted",
    "cassini1_minlp",
    "ml_tuning",
    "jobshop",
    "uav_assignment",
    "material_flow",
];

// --------------------------------------------------------------- type tables

fn int_label(prefix: &'static str) -> impl Fn(usize) -> String {
    move |i| format!("{prefix} {i}")
}

fn bandpass_table() -> TypeTable {
    TypeTable {
        problem: "bandpass",
        source: "tutorials/sindr-circuit-design/src/decode.rs: decode_bandpass_continuous (via so::evaluate_bandpass)",
        coordinates: (0..5)
            .map(|i| continuous(i, 0.0, 1.0, "log-decoded component value"))
            .collect(),
        notes: vec![
            "All five coordinates are log-decoded continuous component values. The E12 catalogue \
             decoding (decode_bandpass_e12) is used only by the MAP-Elites arm (qd.rs), not by a \
             single-objective arm."
                .into(),
        ],
    }
}

fn cassini1_minlp_table() -> TypeTable {
    TypeTable {
        problem: "cassini1_minlp",
        source: "examples/src/problems.rs: cassini1_minlp (fixed_sequence) over crates/fcmaes-gtop cassini1_minlp",
        coordinates: (0..6)
            .map(|i| {
                let p = cassini1_minlp();
                continuous(
                    i,
                    p.lower[i],
                    p.upper[i],
                    "launch date or leg time of flight",
                )
            })
            .collect(),
        notes: vec![
            "The repository's single-objective registry entry appends the fixed flyby sequence \
             [2, 2, 3, 5]; the planet coordinates x[6..10] of fcmaes_gtop::cassini1_minlp are not \
             decision variables of any single-objective code in the repository."
                .into(),
        ],
    }
}

fn truss_table() -> TypeTable {
    use truss_sizing::catalogue::sections;
    use truss_sizing::decode::{MAX_ACTIVE, MIN_ACTIVE, dimension};
    use truss_sizing::ground::GroundStructure;
    let ground = GroundStructure::reference();
    let members = ground.members.len();
    let dim = dimension(&ground);
    let catalogue = sections();
    let mut coordinates = vec![equal_bins(
        0,
        Kind::Integer,
        MAX_ACTIVE - MIN_ACTIVE + 1,
        MIN_ACTIVE as i64,
        &|i| format!("{} active members", MIN_ACTIVE + i),
        "active-member count (linear_integer, 8..=40)",
    )];
    for m in 0..members {
        coordinates.push(continuous(
            1 + m,
            0.0,
            1.0,
            format!("ranking key of member {m} (random key)"),
        ));
    }
    for m in 0..members {
        coordinates.push(equal_bins(
            1 + members + m,
            Kind::Integer,
            catalogue.len(),
            0,
            &|i| catalogue[i].name.to_string(),
            format!("catalogue section of member {m} (used when the member is active)"),
        ));
    }
    for k in 1 + 2 * members..dim {
        coordinates.push(continuous(k, 0.0, 1.0, "movable-node offset"));
    }
    TypeTable {
        problem: "truss_sizing",
        source: "tutorials/truss-sizing/src/decode.rs: decode (linear_integer)",
        coordinates,
        notes: vec![
            "Coordinate 0 decodes per coordinate to the active-member count. The 75 ranking keys \
             select which members are active only jointly with all other keys and the count \
             (random keys, sort and truncate): no per-coordinate discrete meaning, kept continuous."
                .into(),
            "Section coordinates decode per coordinate to a catalogue index; the twelve circular \
             hollow sections are sorted by outer diameter with non-decreasing wall thickness, so \
             area, inertia and mass are strictly increasing: sorted catalogue sizes, integer. A \
             section coordinate matters only while its member is active."
                .into(),
        ],
    }
}

fn energy_table(name: &'static str, annual: bool) -> TypeTable {
    use energy_hub_bilevel::decode::GRID_TIERS_KW;
    let mut coordinates = vec![
        continuous(0, 0.0, 1.0, "PV kWp (linear)"),
        continuous(1, 0.0, 1.0, "wind kW (linear, used when wind is included)"),
        continuous(2, 0.0, 1.0, "battery kWh (logarithmic, used when included)"),
        continuous(3, 0.0, 1.0, "battery kW (logarithmic, used when included)"),
        continuous(
            4,
            0.0,
            1.0,
            if annual {
                "electrolyser kW (linear, used when hydrogen is included)"
            } else {
                "electrolyser kW: not decoded by this objective (annual_hydrogen = false)"
            },
        ),
        continuous(
            5,
            0.0,
            1.0,
            if annual {
                "hydrogen storage kWh (linear, used when hydrogen is included)"
            } else {
                "hydrogen storage kWh: not decoded by this objective (annual_hydrogen = false)"
            },
        ),
        equal_bins(
            6,
            Kind::Integer,
            GRID_TIERS_KW.len(),
            0,
            &|i| format!("grid tier {} kW", GRID_TIERS_KW[i]),
            "grid-connection tier (categorical_index over a catalogue sorted by kW)",
        ),
        equal_bins(
            7,
            Kind::Integer,
            2,
            0,
            &|i| ["no wind", "wind"][i].into(),
            "wind on/off switch",
        ),
        equal_bins(
            8,
            Kind::Integer,
            2,
            0,
            &|i| ["no battery", "battery"][i].into(),
            "battery on/off switch",
        ),
    ];
    coordinates.push(if annual {
        equal_bins(9, Kind::Integer, 2, 0, &|i| ["no hydrogen", "hydrogen"][i].into(),
            "hydrogen on/off switch")
    } else {
        continuous(9, 0.0, 1.0,
            "hydrogen switch: not decoded by this objective (annual_hydrogen && ... short-circuits)")
    });
    TypeTable {
        problem: name,
        source: if annual {
            "tutorials/energy-hub-bilevel/src/decode.rs: decode_outer(values, true) (via annual::evaluate_annual)"
        } else {
            "tutorials/energy-hub-bilevel/src/decode.rs: decode_outer(values, false) (via evaluate::evaluate_training)"
        },
        coordinates,
        notes: vec![
            "The decoder calls the tier coordinate categorical_index, but the six grid tiers are \
             sorted by capacity (500 to 4,000 kW): sorted catalogue sizes, integer by the rule. The \
             architecture flags are binary switches: integer."
                .into(),
        ],
    }
}

fn phase_attenuator(index: usize, phase: bool, meaning: String) -> Coordinate {
    use phased_array_codebook::config::Quantization;
    let q = Quantization::default();
    if phase {
        equal_bins(
            index,
            Kind::Integer,
            1 << q.phase_bits,
            0,
            &int_label("phase code"),
            meaning,
        )
    } else {
        equal_bins(
            index,
            Kind::Integer,
            1 << q.attenuator_bits,
            0,
            &int_label("attenuator code"),
            meaning,
        )
    }
}

fn phased_array_table() -> TypeTable {
    use phased_array_codebook::geometry::SLOTS;
    let mut coordinates = Vec::new();
    for s in 0..SLOTS {
        coordinates.push(continuous(
            s,
            0.0,
            1.0,
            format!("activation key of slot {s} (random key)"),
        ));
    }
    for s in 0..SLOTS {
        coordinates.push(continuous(
            SLOTS + s,
            0.0,
            1.0,
            format!("position perturbation of slot {s}"),
        ));
    }
    for s in 0..SLOTS {
        coordinates.push(phase_attenuator(
            2 * SLOTS + s,
            true,
            format!("phase-shifter code of slot {s} (used when the slot is active)"),
        ));
    }
    for s in 0..SLOTS {
        coordinates.push(phase_attenuator(
            3 * SLOTS + s,
            false,
            format!("attenuator code of slot {s} (used when the slot is active)"),
        ));
    }
    TypeTable {
        problem: "phased_array",
        source: "tutorials/phased-array-codebook/src/geometry.rs: evaluate_geometry (selected_indices, decode::equal_width_code)",
        coordinates,
        notes: vec![
            "Activation keys choose 16 of 24 slots only jointly (sort and truncate): random keys, \
             continuous."
                .into(),
            "Phase codes are quantized values, integer by the rule; the phase is cyclic, so codes \
             0 and 63 are neighbours in the physics but not in the integer order."
                .into(),
        ],
    }
}

fn beam_table() -> TypeTable {
    use phased_array_codebook::so::ELEMENTS;
    let mut coordinates = Vec::new();
    for e in 0..ELEMENTS {
        coordinates.push(phase_attenuator(
            e,
            true,
            format!("phase-shifter code of element {e}"),
        ));
    }
    for e in 0..ELEMENTS {
        coordinates.push(phase_attenuator(
            ELEMENTS + e,
            false,
            format!("attenuator code of element {e}"),
        ));
    }
    TypeTable {
        problem: "phased_array_beam",
        source: "tutorials/phased-array-codebook/src/decode.rs: decode_beam (equal_width_code), via so::evaluate_beam",
        coordinates,
        notes: vec!["Phase codes are cyclic quantized values, integer by the rule.".into()],
    }
}

fn water_table() -> TypeTable {
    use water_network_scheduling::decode::LEVELS;
    let mut coordinates = Vec::new();
    for pump in 0..2 {
        for period in 0..12 {
            coordinates.push(equal_bins(
                pump * 12 + period,
                Kind::Integer,
                LEVELS.len(),
                0,
                &|i| format!("relative speed {}", LEVELS[i]),
                format!("speed level of pump {} in period {period}", pump + 1),
            ));
        }
    }
    coordinates.push(continuous(24, 0.0, 1.0, "low tank threshold"));
    coordinates.push(continuous(
        25,
        0.0,
        1.0,
        "high tank threshold (offset above the low one)",
    ));
    coordinates.push(continuous(26, 0.0, 1.0, "PRV setpoint"));
    coordinates.push(equal_bins(
        27,
        Kind::Categorical,
        2,
        0,
        &|i| ["pump 1 first", "pump 2 first"][i].into(),
        "pump priority under the low-tank override",
    ));
    TypeTable {
        problem: "water_network",
        source: "tutorials/water-network-scheduling/src/decode.rs: decode (category)",
        coordinates,
        notes: vec![
            "Speed levels 0, 0.8, 0.9, 1.0 are quantized values: integer.".into(),
            "The priority is a choice between two pumps, not an on/off switch: categorical. With \
             two levels the distinction is debatable."
                .into(),
        ],
    }
}

/// Compatible-vehicle lists a field-service task is decoded with, one per
/// distinct availability mask of the training scenarios in which the task
/// is active (`scenarios::training`, `decode::decode_active`).
pub fn field_service_lists(
    instance: &field_service_routing::instance::Instance,
) -> Vec<Vec<Vec<usize>>> {
    use field_service_routing::scenarios::training;
    let scenarios = training(instance);
    (0..instance.tasks.len())
        .map(|task| {
            let mut lists: Vec<Vec<usize>> = Vec::new();
            for case in &scenarios {
                if !case.active[task] {
                    continue;
                }
                let list: Vec<usize> = case
                    .instance
                    .vehicles
                    .iter()
                    .filter(|v| {
                        case.available[v.id] && v.skills & case.instance.tasks[task].skill != 0
                    })
                    .map(|v| v.id)
                    .collect();
                if !lists.contains(&list) {
                    lists.push(list);
                }
            }
            lists
        })
        .collect()
}

fn field_service_table() -> TypeTable {
    let instance = field_service_instance();
    let tasks = instance.tasks.len();
    let lists = field_service_lists(&instance);
    let mut coordinates = Vec::new();
    for (task, task_lists) in lists.iter().enumerate() {
        // Cell boundaries: every bin boundary i/n of every list.
        let mut cuts: Vec<f64> = vec![0.0, 1.0];
        for list in task_lists {
            let n = list.len();
            for i in 1..n {
                cuts.push(i as f64 / n as f64);
            }
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let levels: Vec<Level> = cuts
            .windows(2)
            .map(|w| {
                let center = 0.5 * (w[0] + w[1]);
                let chosen: Vec<String> = task_lists
                    .iter()
                    .map(|list| {
                        let n = list.len();
                        let i = ((center * n as f64).floor() as usize).min(n - 1);
                        list[i].to_string()
                    })
                    .collect();
                Level {
                    value: None,
                    label: format!("vehicle {}", chosen.join(" / ")),
                    intervals: vec![(w[0], w[1])],
                    center,
                }
            })
            .collect();
        let meaning = if task_lists.len() > 1 {
            format!(
                "vehicle of task {task}; levels are (nominal / vehicle-7-unavailable) pairs, the \
                 cells of both bin partitions"
            )
        } else {
            format!("vehicle of task {task} among its compatible vehicles")
        };
        coordinates.push(if levels.len() < 2 {
            continuous(
                task,
                0.0,
                1.0,
                format!("vehicle of task {task}: one compatible vehicle, no effect"),
            )
        } else {
            Coordinate {
                index: task,
                kind: Kind::Categorical,
                lower: 0.0,
                upper: 1.0,
                levels,
                meaning,
            }
        });
    }
    for task in 0..tasks {
        coordinates.push(continuous(
            tasks + task,
            0.0,
            1.0,
            format!("route priority key of task {task} (random key)"),
        ));
    }
    TypeTable {
        problem: "field_service",
        source: "tutorials/field-service-routing/src/decode.rs: decode_active (bin), with the masks of scenarios::training",
        coordinates,
        notes: vec![
            "An assignment coordinate selects one of the task's compatible vehicles (equal-width \
             bins): an unordered choice, categorical."
                .into(),
            "In the training scenario 'vehicle_7_unavailable' the compatible list of a task that \
             vehicle 7 can serve has one entry less, so the same coordinate is binned twice. The \
             coordinate still decodes per coordinate; its levels are the cells of the union of both \
             partitions, each a (nominal vehicle / reduced vehicle) pair. This keeps the objective \
             exactly; it is a judgment call."
                .into(),
            "Priority keys order the tasks of each route only jointly (sort): random keys, continuous."
                .into(),
        ],
    }
}

fn coverage_table(name: &'static str) -> TypeTable {
    use network_coverage::decode::UPPER;
    let dim = coverage_instance().nodes();
    let coordinates = (0..dim)
        .map(|i| Coordinate {
            index: i,
            kind: Kind::Integer,
            lower: 0.0,
            upper: UPPER,
            levels: vec![
                Level {
                    value: Some(0),
                    label: "not selected".into(),
                    intervals: vec![(0.0, 1.0)],
                    center: 0.5,
                },
                Level {
                    value: Some(1),
                    label: "selected".into(),
                    intervals: vec![(1.0, UPPER)],
                    center: 0.5 * (1.0 + UPPER),
                },
            ],
            meaning: format!("selection switch of node {i}"),
        })
        .collect();
    TypeTable {
        problem: name,
        source: "tutorials/network-coverage/src/decode.rs: decode (value >= 1.0)",
        coordinates,
        notes: vec!["Every coordinate is a binary selection switch: integer.".into()],
    }
}

fn ml_bounds_log(lo: usize, hi: usize, k: usize) -> (f64, f64) {
    let span = (hi as f64).ln() - (lo as f64).ln();
    let at = |boundary: f64| ((boundary.ln() - (lo as f64).ln()) / span).clamp(0.0, 1.0);
    (
        if k == lo { 0.0 } else { at(k as f64 - 0.5) },
        if k == hi { 1.0 } else { at(k as f64 + 0.5) },
    )
}

fn ml_integer(index: usize, lo: usize, hi: usize, log: bool, meaning: &str) -> Coordinate {
    let levels = (lo..=hi)
        .map(|k| {
            let (a, b) = if log {
                ml_bounds_log(lo, hi, k)
            } else {
                let span = (hi - lo) as f64;
                (
                    if k == lo {
                        0.0
                    } else {
                        (k as f64 - 0.5 - lo as f64) / span
                    },
                    if k == hi {
                        1.0
                    } else {
                        (k as f64 + 0.5 - lo as f64) / span
                    },
                )
            };
            Level {
                value: Some(k as i64),
                label: k.to_string(),
                intervals: vec![(a, b)],
                center: 0.5 * (a + b),
            }
        })
        .collect();
    Coordinate {
        index,
        kind: Kind::Integer,
        lower: 0.0,
        upper: 1.0,
        levels,
        meaning: meaning.into(),
    }
}

fn ml_table() -> TypeTable {
    TypeTable {
        problem: "ml_tuning",
        source: "tutorials/ml-hyperparameter-tuning/src/space.rs: decode (log_integer, linear_integer, categorical_index)",
        coordinates: vec![
            ml_integer(0, 8, 256, true, "number of trees (logarithmic, rounded)"),
            ml_integer(1, 2, 24, false, "maximum depth (linear, rounded)"),
            ml_integer(
                2,
                1,
                64,
                true,
                "minimum samples per leaf (logarithmic, rounded)",
            ),
            ml_integer(
                3,
                2,
                64,
                true,
                "minimum samples per split (logarithmic, rounded)",
            ),
            continuous(4, 0.0, 1.0, "row sample fraction"),
            continuous(5, 0.0, 1.0, "feature fraction"),
            continuous(6, 0.0, 1.0, "positive sampling weight (logarithmic)"),
            equal_bins(
                7,
                Kind::Categorical,
                2,
                0,
                &|i| ["Gini", "Entropy"][i].into(),
                "split criterion",
            ),
        ],
        notes: vec![
            "Rounded integers have half-width end intervals; logarithmic ones unequal widths."
                .into(),
        ],
    }
}

fn jobshop_table() -> TypeTable {
    use crate::examples::jobshop::JobShop;
    let problem = JobShop::mini();
    let n = problem.operations.len();
    let bins = 10 * problem.machines;
    let mut coordinates = Vec::new();
    for (i, alternatives) in problem.operations.iter().enumerate() {
        let count = alternatives.len();
        if count < 2 {
            coordinates.push(continuous(
                i,
                JOBSHOP_LOWER,
                JOBSHOP_UPPER,
                format!("machine of operation {i}: one alternative, no effect"),
            ));
            continue;
        }
        let levels = (0..count)
            .map(|level| {
                let intervals: Vec<(f64, f64)> = (0..bins)
                    .filter(|b| b % count == level)
                    .map(|b| {
                        let lo = (b as f64 / bins as f64).max(JOBSHOP_LOWER);
                        let hi = ((b + 1) as f64 / bins as f64).min(JOBSHOP_UPPER);
                        (lo, hi)
                    })
                    .collect();
                let center = 0.5 * (intervals[0].0 + intervals[0].1);
                Level {
                    value: None,
                    label: format!("machine {}", alternatives[level].machine),
                    intervals,
                    center,
                }
            })
            .collect();
        coordinates.push(Coordinate {
            index: i,
            kind: Kind::Categorical,
            lower: JOBSHOP_LOWER,
            upper: JOBSHOP_UPPER,
            levels,
            meaning: format!(
                "machine alternative of operation {i} (bin index modulo alternatives)"
            ),
        });
    }
    for i in 0..n {
        coordinates.push(continuous(
            n + i,
            JOBSHOP_LOWER,
            JOBSHOP_UPPER,
            format!("order key of operation {i} (random key)"),
        ));
    }
    TypeTable {
        problem: "jobshop",
        source: "examples/src/jobshop.rs: JobShop::decode (embedded MINI_FJS instance, the default of src/bin/jobshop.rs)",
        coordinates,
        notes: vec![
            "A machine coordinate selects one of the operation's alternative machines: unordered, \
             categorical. Each level is decoded by every other bin of width 1/(10 machines), so it \
             has several intervals; the typed point goes to the center of the first."
                .into(),
            "The instance is the four-operation MINI_FJS example; larger instances need an external \
             data file that the repository does not contain."
                .into(),
        ],
    }
}

fn uav_table() -> TypeTable {
    let problem = uav_instance();
    let separators = problem.vehicle_count() - 1;
    let targets = problem.target_count();
    let mut coordinates = Vec::new();
    for s in 0..separators {
        coordinates.push(equal_bins(
            s,
            Kind::Integer,
            targets,
            0,
            &int_label("separator position"),
            format!("route separator {s} position"),
        ));
    }
    for t in 0..targets {
        coordinates.push(continuous(
            separators + t,
            0.0,
            1.0,
            format!("visit-order key of target {} (random key)", t + 1),
        ));
    }
    TypeTable {
        problem: "uav_assignment",
        source: "examples/src/uav.rs: UavProblem::separators_and_sequence (instance of src/bin/uav_task_assignment.rs)",
        coordinates,
        notes: vec![
            "A separator coordinate decodes per coordinate to an ordered position (floor of u times \
             the target count): integer. Separators are interchangeable, which does not couple \
             their decoding."
                .into(),
            "Visit-order keys act only jointly (sort): random keys, continuous.".into(),
        ],
    }
}

fn material_flow_table() -> TypeTable {
    let coordinates = (0..2)
        .map(|i| {
            let levels = (1..=50)
                .map(|k| {
                    let (lo, hi) = if k == 50 {
                        (50.0, 50.0)
                    } else {
                        (k as f64, k as f64 + 1.0)
                    };
                    Level {
                        value: Some(k),
                        label: format!("batch size {k}"),
                        intervals: vec![(lo, hi)],
                        center: 0.5 * (lo + hi),
                    }
                })
                .collect();
            Coordinate {
                index: i,
                kind: Kind::Integer,
                lower: 1.0,
                upper: 50.0,
                levels,
                meaning: format!("batch size of product {i} (truncation)"),
            }
        })
        .collect();
    TypeTable {
        problem: "material_flow",
        source: "examples/src/material_flow_planning.rs: Plant::fitness (clamp(1, 50) as usize)",
        coordinates,
        notes: vec!["Batch sizes 1..=49 decode from [k, k + 1); 50 only from the point 50.".into()],
    }
}

/// Type table of a round-3 problem (ROUND3.md section 4).
pub fn type_table(name: &str) -> Option<TypeTable> {
    Some(match name {
        "truss_sizing" => truss_table(),
        "energy_hub" => energy_table("energy_hub", false),
        "energy_hub_annual" => energy_table("energy_hub_annual", true),
        "phased_array" => phased_array_table(),
        "phased_array_beam" => beam_table(),
        "bandpass" => bandpass_table(),
        "water_network" => water_table(),
        "field_service" => field_service_table(),
        "coverage_cardinality" => coverage_table("coverage_cardinality"),
        "coverage_weighted" => coverage_table("coverage_weighted"),
        "cassini1_minlp" => cassini1_minlp_table(),
        "ml_tuning" => ml_table(),
        "jobshop" => jobshop_table(),
        "uav_assignment" => uav_table(),
        "material_flow" => material_flow_table(),
        _ => return None,
    })
}

// -------------------------------------------------------------------- survey

/// Entry limit of ROUND3.md section 3: mean evaluation time in microseconds.
pub const ENTRY_LIMIT_US: f64 = 2_000.0;

/// Timing of one problem over the seeded 2,000-point measurement sample of
/// [`crate::campaign::measure_problem`].
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// Points evaluated.
    pub points: usize,
    /// Mean time per evaluated point in microseconds.
    pub mean_us: f64,
    /// Whether all 2,000 points were evaluated.
    pub complete: bool,
}

/// Time the measurement sample like [`crate::campaign::measure_problem`],
/// but stop as soon as the elapsed time exceeds `2,000 × limit_us`: the
/// mean over all 2,000 points then exceeds `limit_us` whatever the remaining
/// points cost, so the entry decision is the same.
pub fn measure_capped(problem: &Problem, limit_us: f64) -> Timing {
    use crate::campaign::MEASURE_POINTS;
    use crate::seeds::measure_seed;
    use std::time::Instant;
    let mut rng = fcmaes_core::Rng::new(measure_seed(problem.name));
    let points: Vec<Vec<f64>> = (0..MEASURE_POINTS)
        .map(|_| crate::solvers::uniform_point(&problem.lower, &problem.upper, &mut rng))
        .collect();
    let cap = limit_us * MEASURE_POINTS as f64 * 1e-6;
    let mut sink = 0.0;
    let started = Instant::now();
    let mut done = 0;
    for x in &points {
        sink += std::hint::black_box((problem.objective)(std::hint::black_box(x)));
        done += 1;
        if started.elapsed().as_secs_f64() > cap {
            break;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    Timing {
        points: done,
        mean_us: elapsed * 1e6 / done as f64,
        complete: done == MEASURE_POINTS,
    }
}

/// Whether the first `points` points of the measurement sample give
/// bit-identical values when evaluated twice.
pub fn deterministic_on(problem: &Problem, points: usize) -> bool {
    let mut rng = fcmaes_core::Rng::new(crate::seeds::measure_seed(problem.name));
    (0..points).all(|_| {
        let x = crate::solvers::uniform_point(&problem.lower, &problem.upper, &mut rng);
        (problem.objective)(&x).to_bits() == (problem.objective)(&x).to_bits()
    })
}

/// Examined problems the harness does not build, with the reason
/// (`problems.tsv` rows without measurements).
pub const NOT_BUILT: [(&str, &str, &str); 4] = [
    (
        "cassini2_minlp",
        "space",
        "fcmaes_gtop::cassini2_minlp decodes x[22..26] to flyby planets, but no \
         single-objective code in the repository minimizes it and the repository defines no \
         search box for the planet coordinates",
    ),
    (
        "tandem_free_sequence",
        "space",
        "fcmaes_gtop::tandem takes the flyby sequence as a separate argument, not from x; every \
         single-objective use fixes it ([3, 2, 3, 3, 6]), so no free-sequence form exists",
    ),
    (
        "dyson_scheduling",
        "space",
        "examples/src/scheduling.rs: trajectory_selection repairs collisions by moving to the next \
         unused trajectory (couples coordinates 0..10), station order and timings are sorted keys: \
         no clean typed form",
    ),
    (
        "bandpass_e12",
        "circuits",
        "tutorials/sindr-circuit-design: the E12 catalogue decoding is used only by the MAP-Elites \
         arm (qd.rs); no single-objective code minimizes it",
    ),
];
