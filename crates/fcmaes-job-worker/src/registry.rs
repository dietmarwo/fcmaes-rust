//! The models a worker solves, by problem identifier.
//!
//! A job names a registered model; it never carries code. An application
//! registers its own models as Rust closures and runs the worker with its
//! registry (`examples/custom_worker.rs`, `docs/job-api/models.md`):
//!
//! ```
//! use fcmaes_job_worker::registry::{Registry, Scalar};
//! use serde_json::json;
//!
//! let mut registry = Registry::benchmarks();
//! registry.scalar(
//!     "app.shifted-sphere",
//!     "1",
//!     json!({"dimensions": "1..10000", "parameters": {"shift": "default 1.0"}}),
//!     |model| {
//!         let shift: f64 = model.parameter("shift", 1.0)?;
//!         Ok(Scalar::new(move |x: &[f64]| {
//!             x.iter().map(|v| (v - shift) * (v - shift)).sum()
//!         }))
//!     },
//! );
//! // optimization_job_api::worker::run(fcmaes_job_worker::Adapter::new(registry));
//! ```
//!
//! A factory receives an [`Instance`]: the problem's dimension, its
//! parameters and the job's SHA-256-checked inputs. The worker records every
//! parameter value the factory used, defaults included, in the resolved job,
//! and rejects parameters and inputs the factory did not use. Factories run
//! at validation and again at execution, so they must be deterministic.
//!
//! `describe` builds every model once more for a small example problem: an
//! [`Example`] registered with [`Registry::example`], or else dimensions 2,
//! 3, 4, ... without parameters and inputs, on the model's domain or
//! `[-5, 5]`. What the factory read gives the model's `parameters_schema`,
//! and the problem gives runnable `examples`. A model that needs inputs,
//! required parameters or (multi-objective, without its own) a reference
//! point registers an `Example` to get them.
use crate::{PROFILES, evaluate, map_elites, mode, retry, surrogate};
use optimization_job_api::{JobSpec, Result, legacy};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::Arc,
};

/// A scalar objective, minimized.
pub type ScalarFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;
pub type SimulationFn = Arc<
    dyn Fn(&[f64], &crate::simulation::SimContext) -> std::result::Result<f64, String>
        + Send
        + Sync,
>;
/// Objectives followed by constraints.
pub type VectorFn = Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;
/// Fitness, minimized, and behavior descriptor.
pub type QdFn = Arc<dyn Fn(&[f64]) -> (f64, Vec<f64>) + Send + Sync>;

/// Largest decision dimension of any problem.
pub const MAX_DIMENSION: usize = 10_000;

/// The problem of every fcmaes profile: a registered model, its dimension,
/// its box and optional model parameters. Without parameters it serializes
/// like the legacy `fcmaes.job/0.1` problem, so jobs on the built-in models
/// keep their digests.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Problem {
    pub id: String,
    pub version: String,
    pub config: legacy::ProblemConfig,
    pub bounds: legacy::Bounds,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Map<String, Value>>,
}

impl Problem {
    /// The legacy problem with the same identifier, dimension and bounds.
    pub fn legacy(&self) -> legacy::Problem {
        legacy::Problem {
            id: self.id.clone(),
            version: self.version.clone(),
            config: self.config.clone(),
            bounds: self.bounds.clone(),
        }
    }
    /// Whether every coordinate of `x` is finite and inside the bounds.
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.config.dimension
            && x.iter()
                .zip(self.bounds.lower.iter().zip(&self.bounds.upper))
                .all(|(v, (l, u))| v.is_finite() && v >= l && v <= u)
    }
}

/// The box a model is defined on. Job bounds must lie inside it.
#[derive(Clone, Debug, PartialEq)]
pub struct Domain {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// A scalar model: one minimized objective.
#[derive(Clone)]
pub struct Scalar {
    /// Plain objective. For `Scalar::simulation` models it panics: call
    /// [`Scalar::value`] with a `SimContext` instead.
    pub evaluate: ScalarFn,
    pub(crate) simulation: Option<SimulationFn>,
    /// `None`: any finite box.
    pub domain: Option<Domain>,
}
impl Scalar {
    pub fn new(evaluate: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            evaluate: Arc::new(evaluate),
            simulation: None,
            domain: None,
        }
    }
    /// Fallible/cancellable simulation, called with a [`SimContext`]
    /// (`crate::simulation`). `fcmaes.surrogate/1`, `fcmaes.retry/1` and
    /// `fcmaes.evaluate/1` accept it: a failed evaluation is a failed
    /// observation, `+inf` for a retry's optimizer, or claim `none`.
    ///
    /// [`SimContext`]: crate::simulation::SimContext
    pub fn simulation(
        evaluate: impl Fn(&[f64], &crate::simulation::SimContext) -> std::result::Result<f64, String>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            // Every profile calls a simulation through `Scalar::value`; a
            // direct call is a bug, not a bad objective value.
            evaluate: Arc::new(|_| {
                panic!("a Scalar::simulation model must be called through Scalar::value")
            }),
            simulation: Some(Arc::new(evaluate)),
            domain: None,
        }
    }
    pub fn is_simulation(&self) -> bool {
        self.simulation.is_some()
    }
    pub fn value(
        &self,
        x: &[f64],
        context: &crate::simulation::SimContext,
    ) -> std::result::Result<f64, String> {
        match &self.simulation {
            Some(f) => f(x, context),
            None => Ok((self.evaluate)(x)),
        }
    }
    /// Restricts job bounds to `[lower, upper]`.
    pub fn domain(mut self, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        self.domain = Some(Domain { lower, upper });
        self
    }
}

/// A multi-objective model: `objectives` minimized values followed by
/// `constraints` values, feasible when every constraint is at most zero.
#[derive(Clone)]
pub struct MultiObjective {
    pub objectives: usize,
    pub constraints: usize,
    /// Hypervolume reference point when the job gives none; `None` makes
    /// `reference_point` a required job field.
    pub reference_point: Option<Vec<f64>>,
    pub evaluate: VectorFn,
    pub domain: Option<Domain>,
}
impl MultiObjective {
    pub fn new(
        objectives: usize,
        constraints: usize,
        evaluate: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    ) -> Self {
        Self {
            objectives,
            constraints,
            reference_point: None,
            evaluate: Arc::new(evaluate),
            domain: None,
        }
    }
    pub fn reference_point(mut self, reference_point: Vec<f64>) -> Self {
        self.reference_point = Some(reference_point);
        self
    }
    /// Restricts job bounds to `[lower, upper]`.
    pub fn domain(mut self, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        self.domain = Some(Domain { lower, upper });
        self
    }
    /// Evaluates `x`, checking the number of values.
    pub fn values(&self, x: &[f64]) -> Result<Vec<f64>> {
        let values = (self.evaluate)(x);
        let expected = self.objectives + self.constraints;
        if values.len() != expected {
            return Err(format!(
                "the model returned {} values; it declares {} objectives and {} constraints",
                values.len(),
                self.objectives,
                self.constraints
            )
            .into());
        }
        Ok(values)
    }
}

/// A quality-diversity model: minimized fitness and a behavior descriptor
/// inside fixed descriptor bounds.
#[derive(Clone)]
pub struct QualityDiversity {
    pub descriptor_lower: Vec<f64>,
    pub descriptor_upper: Vec<f64>,
    pub evaluate: QdFn,
    pub domain: Option<Domain>,
}
impl QualityDiversity {
    pub fn new(
        descriptor_lower: Vec<f64>,
        descriptor_upper: Vec<f64>,
        evaluate: impl Fn(&[f64]) -> (f64, Vec<f64>) + Send + Sync + 'static,
    ) -> Self {
        Self {
            descriptor_lower,
            descriptor_upper,
            evaluate: Arc::new(evaluate),
            domain: None,
        }
    }
    /// Restricts job bounds to `[lower, upper]`.
    pub fn domain(mut self, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        self.domain = Some(Domain { lower, upper });
        self
    }
    /// Evaluates `x`, checking the descriptor length.
    pub fn values(&self, x: &[f64]) -> Result<(f64, Vec<f64>)> {
        let (fitness, descriptor) = (self.evaluate)(x);
        if descriptor.len() != self.descriptor_lower.len() {
            return Err(format!(
                "the model returned a descriptor of length {}; it declares {}",
                descriptor.len(),
                self.descriptor_lower.len()
            )
            .into());
        }
        Ok((fitness, descriptor))
    }
}

/// A built model of any kind.
#[derive(Clone)]
pub enum Model {
    Scalar(Scalar),
    MultiObjective(MultiObjective),
    QualityDiversity(QualityDiversity),
}

/// The kind of a registered model, which decides the profile that
/// optimizes it. `fcmaes.evaluate/1` accepts every kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Scalar,
    MultiObjective,
    QualityDiversity,
}
impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Scalar => "scalar",
            Kind::MultiObjective => "multi_objective",
            Kind::QualityDiversity => "quality_diversity",
        }
    }
    /// The profile that optimizes models of this kind.
    pub fn profile(self) -> &'static str {
        match self {
            Kind::Scalar => retry::PROFILE,
            Kind::MultiObjective => mode::PROFILE,
            Kind::QualityDiversity => map_elites::PROFILE,
        }
    }
    /// Every supported profile; `profile()` remains the optimizing default.
    pub fn profiles(self) -> &'static [&'static str] {
        match self {
            Kind::Scalar => &[retry::PROFILE, surrogate::PROFILE, evaluate::PROFILE],
            Kind::MultiObjective => &[mode::PROFILE, evaluate::PROFILE],
            Kind::QualityDiversity => &[map_elites::PROFILE, evaluate::PROFILE],
        }
    }
}

/// Where a factory's inputs come from.
enum Source<'a> {
    /// The job's inputs, files relative to `base` checked against their SHA-256.
    Job { spec: &'a JobSpec, base: &'a Path },
    /// The inputs of an [`Example`], by name.
    Memory(&'a BTreeMap<String, Vec<u8>>),
}

/// What a factory knows about the problem it builds a model for.
pub struct Instance<'a> {
    id: &'a str,
    dimension: usize,
    given: Map<String, Value>,
    used: RefCell<Map<String, Value>>,
    /// Defaults of the parameters read with [`Instance::parameter`].
    defaults: RefCell<Map<String, Value>>,
    /// Parameters read with [`Instance::required`].
    required: RefCell<BTreeSet<String>>,
    source: Source<'a>,
    inputs: RefCell<BTreeSet<String>>,
}

impl Instance<'_> {
    /// The problem identifier.
    pub fn id(&self) -> &str {
        self.id
    }
    /// The decision dimension the job states (1..=10000).
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Fails unless the dimension lies in `range`.
    pub fn require_dimension(&self, range: std::ops::RangeInclusive<usize>) -> Result<usize> {
        if range.contains(&self.dimension) {
            Ok(self.dimension)
        } else if range.start() == range.end() {
            Err(format!("{} needs dimension {}", self.id, range.start()).into())
        } else {
            Err(format!(
                "{} needs a dimension in {}..={}",
                self.id,
                range.start(),
                range.end()
            )
            .into())
        }
    }
    /// The parameter `name`, or `default` if the job does not give it. The
    /// value used is recorded in the resolved job.
    pub fn parameter<T: Serialize + DeserializeOwned>(&self, name: &str, default: T) -> Result<T> {
        self.defaults
            .borrow_mut()
            .insert(name.into(), serde_json::to_value(&default)?);
        let value = match self.given.get(name) {
            Some(v) => serde_json::from_value(v.clone())
                .map_err(|e| format!("parameter {name} of {}: {e}", self.id))?,
            None => default,
        };
        self.used
            .borrow_mut()
            .insert(name.into(), serde_json::to_value(&value)?);
        Ok(value)
    }
    /// The parameter `name`, which the job must give.
    pub fn required<T: Serialize + DeserializeOwned>(&self, name: &str) -> Result<T> {
        self.required.borrow_mut().insert(name.into());
        let v = self
            .given
            .get(name)
            .ok_or_else(|| format!("{} requires parameter {name}", self.id))?;
        let value: T = serde_json::from_value(v.clone())
            .map_err(|e| format!("parameter {name} of {}: {e}", self.id))?;
        self.used
            .borrow_mut()
            .insert(name.into(), serde_json::to_value(&value)?);
        Ok(value)
    }
    /// The bytes of the job input `name`, checked against its SHA-256.
    pub fn input(&self, name: &str) -> Result<Vec<u8>> {
        let missing = || format!("{} requires the job input {name}", self.id);
        let bytes = match &self.source {
            Source::Job { spec, base } => {
                let input = spec
                    .inputs
                    .iter()
                    .find(|i| i.name == name)
                    .ok_or_else(missing)?;
                optimization_job_api::checked_input(base, input)
                    .map_err(|e| format!("input {name}: {e}"))?
            }
            Source::Memory(inputs) => inputs.get(name).cloned().ok_or_else(missing)?,
        };
        self.inputs.borrow_mut().insert(name.into());
        Ok(bytes)
    }
}

type Factory = Box<dyn Fn(&Instance) -> Result<Model>>;

struct Entry {
    version: String,
    kind: Kind,
    about: Value,
    factory: Factory,
    example: Option<Example>,
}

/// A small problem of a registered model that `describe` advertises as
/// runnable job `examples`: its dimension, optionally its bounds (else the
/// model's domain or `[-5, 5]`), parameters and the inputs it reads.
///
/// ```
/// use fcmaes_job_worker::registry::Example;
///
/// let example = Example::new(3)
///     .bounds(vec![0.0, 0.0, -10.0], vec![10.0, 5.0, 10.0])
///     .parameter("loss", "squared")
///     .input_text("data", "decay.csv", "t,y\n0,3.5\n1,2\n2,1.2\n");
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Example {
    dimension: usize,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    parameters: Map<String, Value>,
    inputs: Vec<ExampleInput>,
    reference_point: Option<Vec<f64>>,
}
/// A text input of an [`Example`]: a client writes `content_text` to `file`
/// and names it `name` with its SHA-256 in the job's `inputs`.
#[derive(Clone, Debug, PartialEq)]
pub struct ExampleInput {
    pub name: String,
    pub file: String,
    pub content_text: String,
}
impl Example {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            bounds: None,
            parameters: Map::new(),
            inputs: Vec::new(),
            reference_point: None,
        }
    }
    pub fn bounds(mut self, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        self.bounds = Some((lower, upper));
        self
    }
    /// # Panics
    ///
    /// Panics if `value` does not serialize to JSON.
    pub fn parameter(mut self, name: &str, value: impl Serialize) -> Self {
        let value = serde_json::to_value(value).expect("a JSON parameter value");
        self.parameters.insert(name.into(), value);
        self
    }
    pub fn input_text(mut self, name: &str, file: &str, content: impl Into<String>) -> Self {
        self.inputs.push(ExampleInput {
            name: name.into(),
            file: file.into(),
            content_text: content.into(),
        });
        self
    }
    /// The `reference_point` of `fcmaes.mode/1` examples, for a
    /// multi-objective model that defines none.
    pub fn reference_point(mut self, point: Vec<f64>) -> Self {
        self.reference_point = Some(point);
        self
    }
}

/// A registered model built for its example problem by `describe`.
#[derive(Clone, Debug)]
pub struct Probe {
    pub id: String,
    pub version: String,
    pub kind: Kind,
    /// The model's optimizing `profile` in `describe`.
    pub profile: &'static str,
    pub dimension: usize,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    /// The example's own parameters (none for a probed model).
    pub parameters: Map<String, Value>,
    pub inputs: Vec<ExampleInput>,
    pub simulation: bool,
    /// Whether a mode example can run: the model or the example defines a
    /// reference point.
    pub mode_ready: bool,
    /// The example's reference point, when the model has none.
    pub reference_point: Option<Vec<f64>>,
    /// JSON Schema of `problem.parameters`, from what the factory read.
    pub parameters_schema: Value,
}
impl Probe {
    /// The `problem` object of a job on this example.
    pub fn problem(&self) -> Value {
        let mut problem = json!({"id":self.id,"version":self.version,"config":{"dimension":self.dimension},
            "bounds":{"lower":self.lower,"upper":self.upper,"broadcast":false}});
        if !self.parameters.is_empty() {
            problem["parameters"] = Value::Object(self.parameters.clone());
        }
        problem
    }
}

/// The parameters a factory read and how.
struct Reads {
    used: Map<String, Value>,
    defaults: Map<String, Value>,
    required: BTreeSet<String>,
}

/// JSON Schema of the parameters a factory read when `describe` built the
/// model: the JSON type of each value, the default of those read with a
/// default, and the required ones. A parameter the factory reads only for
/// other parameter values is missing; the model checks the values.
fn parameters_schema(reads: &Reads) -> Value {
    let properties: Map<String, Value> = reads
        .used
        .iter()
        .map(|(name, value)| {
            let mut property = match value {
                Value::Bool(_) => json!({"type":"boolean"}),
                Value::Number(n) if n.is_f64() => json!({"type":"number"}),
                Value::Number(_) => json!({"type":"integer"}),
                Value::String(_) => json!({"type":"string"}),
                Value::Array(_) => json!({"type":"array"}),
                Value::Object(_) => json!({"type":"object"}),
                Value::Null => json!({}),
            };
            if let Some(default) = reads.defaults.get(name) {
                property["default"] = default.clone();
            }
            (name.clone(), property)
        })
        .collect();
    let mut schema = json!({"type":"object","additionalProperties":false});
    if !properties.is_empty() {
        schema["properties"] = Value::Object(properties);
    }
    if !reads.required.is_empty() {
        schema["required"] = json!(reads.required);
    }
    schema
}

/// Dimensions `describe` tries, in order, for a model without an [`Example`].
const PROBE_DIMENSIONS: [usize; 8] = [2, 3, 4, 6, 1, 5, 8, 10];
/// Bounds of a probed model without a domain.
const PROBE_BOX: f64 = 5.0;

/// Registered models by problem identifier.
#[derive(Default)]
pub struct Registry {
    entries: BTreeMap<String, Entry>,
}

impl Registry {
    /// A registry without models.
    pub fn new() -> Self {
        Self::default()
    }
    /// The built-in benchmark models (`crate::problems`).
    pub fn benchmarks() -> Self {
        let mut registry = Self::new();
        crate::problems::register(&mut registry);
        registry
    }

    fn add(&mut self, id: &str, version: &str, kind: Kind, about: Value, factory: Factory) {
        assert!(
            !id.is_empty() && !version.is_empty(),
            "a model needs an identifier and a version"
        );
        assert!(
            about.is_object(),
            "the description of {id} must be an object"
        );
        let entry = Entry {
            version: version.into(),
            kind,
            about,
            factory,
            example: None,
        };
        assert!(
            self.entries.insert(id.into(), entry).is_none(),
            "model {id} registered twice"
        );
    }
    /// Registers a scalar model for `fcmaes.retry/1`. `about` is a JSON object
    /// added to the model's `describe` entry (dimensions, parameters, inputs).
    ///
    /// # Panics
    ///
    /// Panics if `id` is already registered or `about` is not an object.
    pub fn scalar(
        &mut self,
        id: &str,
        version: &str,
        about: Value,
        factory: impl Fn(&Instance) -> Result<Scalar> + 'static,
    ) -> &mut Self {
        let factory: Factory = Box::new(move |i| factory(i).map(Model::Scalar));
        self.add(id, version, Kind::Scalar, about, factory);
        self
    }
    /// Registers a multi-objective model for `fcmaes.mode/1`.
    ///
    /// # Panics
    ///
    /// As [`Registry::scalar`].
    pub fn multi_objective(
        &mut self,
        id: &str,
        version: &str,
        about: Value,
        factory: impl Fn(&Instance) -> Result<MultiObjective> + 'static,
    ) -> &mut Self {
        let factory: Factory = Box::new(move |i| factory(i).map(Model::MultiObjective));
        self.add(id, version, Kind::MultiObjective, about, factory);
        self
    }
    /// Registers a quality-diversity model for `fcmaes.map-elites/1`.
    ///
    /// # Panics
    ///
    /// As [`Registry::scalar`].
    pub fn quality_diversity(
        &mut self,
        id: &str,
        version: &str,
        about: Value,
        factory: impl Fn(&Instance) -> Result<QualityDiversity> + 'static,
    ) -> &mut Self {
        let factory: Factory = Box::new(move |i| factory(i).map(Model::QualityDiversity));
        self.add(id, version, Kind::QualityDiversity, about, factory);
        self
    }

    /// Registers the example problem `describe` advertises for model `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not registered or already has an example.
    pub fn example(&mut self, id: &str, example: Example) -> &mut Self {
        let entry = self
            .entries
            .get_mut(id)
            .unwrap_or_else(|| panic!("model {id} is not registered"));
        assert!(entry.example.is_none(), "model {id} has an example");
        entry.example = Some(example);
        self
    }

    /// The profiles at least one registered model supports, in [`PROFILES`]
    /// order.
    pub fn profiles(&self) -> Vec<&'static str> {
        PROFILES
            .into_iter()
            .filter(|p| self.entries.values().any(|e| e.kind.profiles().contains(p)))
            .collect()
    }

    /// Identifier and version of every model `profile` accepts.
    pub fn models(&self, profile: &str) -> Vec<(String, String)> {
        self.entries
            .iter()
            .filter(|(_, e)| e.kind.profiles().contains(&profile))
            .map(|(id, e)| (id.clone(), e.version.clone()))
            .collect()
    }

    /// The `describe` entries of all models, with their kind and profile.
    pub fn discovery(&self) -> Value {
        self.discovery_with(&self.probes())
    }

    /// [`Registry::discovery`] with the models' probes: each entry gains the
    /// `parameters_schema` its probe found.
    pub fn discovery_with(&self, probes: &BTreeMap<String, Probe>) -> Value {
        self.entries
            .iter()
            .map(|(id, e)| {
                let mut entry = json!({"id":id,"version":e.version,"kind":e.kind.name(),
                    "profile":Self::profile_of(e), "profiles":e.kind.profiles()});
                for (k, v) in e.about.as_object().into_iter().flatten() {
                    entry[k] = v.clone();
                }
                if let Some(probe) = probes.get(id) {
                    entry["parameters_schema"] = probe.parameters_schema.clone();
                }
                entry
            })
            .collect()
    }

    /// The optimizing profile `describe` names: an expensive simulation is
    /// optimized by the surrogate by default; retry and evaluate accept it
    /// as well.
    fn profile_of(e: &Entry) -> &'static str {
        if e.about["simulation"] == true {
            surrogate::PROFILE
        } else {
            e.kind.profile()
        }
    }

    /// Every model built for its example problem; models whose factory
    /// rejects every probe are missing.
    pub fn probes(&self) -> BTreeMap<String, Probe> {
        self.entries
            .iter()
            .filter_map(|(id, e)| Some((id.clone(), self.probe(id, e)?)))
            .collect()
    }

    fn probe(&self, id: &str, entry: &Entry) -> Option<Probe> {
        match &entry.example {
            Some(example) => self.try_probe(id, entry, example),
            None => PROBE_DIMENSIONS
                .iter()
                .find_map(|&n| self.try_probe(id, entry, &Example::new(n))),
        }
    }

    fn try_probe(&self, id: &str, entry: &Entry, example: &Example) -> Option<Probe> {
        let inputs: BTreeMap<String, Vec<u8>> = example
            .inputs
            .iter()
            .map(|i| (i.name.clone(), i.content_text.as_bytes().to_vec()))
            .collect();
        let names: Vec<&str> = inputs.keys().map(String::as_str).collect();
        let n = example.dimension;
        let problem = |lower: Vec<f64>, upper: Vec<f64>| Problem {
            id: id.into(),
            version: entry.version.clone(),
            config: legacy::ProblemConfig { dimension: n },
            bounds: legacy::Bounds {
                lower,
                upper,
                broadcast: false,
            },
            parameters: (!example.parameters.is_empty()).then(|| example.parameters.clone()),
        };
        // A factory may panic on a dimension it does not support.
        let resolve = |problem: Problem| {
            catch_unwind(AssertUnwindSafe(|| {
                self.resolve_from(problem, Source::Memory(&inputs), &names, &BTreeSet::new())
            }))
            .ok()?
            .ok()
        };
        let (lower, upper) = match &example.bounds {
            Some(bounds) => bounds.clone(),
            None => {
                // Any box shows the model's domain; then the example uses it.
                let wide = problem(vec![-f64::MAX / 4.0; n], vec![f64::MAX / 4.0; n]);
                let domain = match catch_unwind(AssertUnwindSafe(|| {
                    self.build(id, entry, wide, Source::Memory(&inputs))
                })) {
                    Ok(Ok(model)) => model,
                    _ => return None,
                };
                domain.unwrap_or_else(|| (vec![-PROBE_BOX; n], vec![PROBE_BOX; n]))
            }
        };
        let (resolved, model, reads) = resolve(problem(lower, upper))?;
        let (mode_ready, reference_point) = match &model {
            Model::MultiObjective(m) if m.reference_point.is_some() => (true, None),
            Model::MultiObjective(_) => (
                example.reference_point.is_some(),
                example.reference_point.clone(),
            ),
            _ => (false, None),
        };
        Some(Probe {
            id: id.into(),
            version: entry.version.clone(),
            kind: entry.kind,
            profile: Self::profile_of(entry),
            dimension: n,
            lower: resolved.bounds.lower,
            upper: resolved.bounds.upper,
            parameters: example.parameters.clone(),
            inputs: example.inputs.clone(),
            simulation: matches!(&model, Model::Scalar(m) if m.is_simulation()),
            mode_ready,
            reference_point,
            parameters_schema: parameters_schema(&reads),
        })
    }

    /// The domain of the model `problem` builds, without the checks of a
    /// resolution.
    fn build(
        &self,
        id: &str,
        entry: &Entry,
        problem: Problem,
        source: Source,
    ) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
        let instance = Instance {
            id,
            dimension: problem.config.dimension,
            given: problem.parameters.unwrap_or_default(),
            used: RefCell::new(Map::new()),
            defaults: RefCell::new(Map::new()),
            required: RefCell::new(BTreeSet::new()),
            source,
            inputs: RefCell::new(BTreeSet::new()),
        };
        let domain = match (entry.factory)(&instance)? {
            Model::Scalar(m) => m.domain,
            Model::MultiObjective(m) => m.domain,
            Model::QualityDiversity(m) => m.domain,
        };
        Ok(domain.map(|d| (d.lower, d.upper)))
    }

    /// The kind of the registered model `id`.
    pub fn kind(&self, id: &str) -> Option<Kind> {
        self.entries.get(id).map(|e| e.kind)
    }

    /// [`Registry::resolve`] for a model of `kind`.
    pub fn resolve_as(
        &self,
        problem: Problem,
        spec: &JobSpec,
        base: &Path,
        kind: Kind,
    ) -> Result<(Problem, Model)> {
        if let Some(actual) = self.kind(&problem.id).filter(|k| *k != kind) {
            return Err(format!(
                "{} is a {} model; use {}",
                problem.id,
                actual.name().replace('_', "-"),
                actual.profile()
            )
            .into());
        }
        self.resolve(problem, spec, base)
    }

    /// Builds the model of `problem` and resolves the problem: expanded
    /// bounds inside the model's domain and the parameters the model used.
    /// `base` is the directory the job's input paths are relative to.
    pub fn resolve(
        &self,
        problem: Problem,
        spec: &JobSpec,
        base: &Path,
    ) -> Result<(Problem, Model)> {
        self.resolve_with_profile_inputs(problem, spec, base, &[])
    }

    /// Resolves a model while allowing explicitly named inputs owned and
    /// hash-checked by the calling profile (for example surrogate warm data).
    /// Normal model parameters and all other unused inputs remain rejected.
    pub fn resolve_with_profile_inputs(
        &self,
        problem: Problem,
        spec: &JobSpec,
        base: &Path,
        profile_inputs: &[&str],
    ) -> Result<(Problem, Model)> {
        let mut allowed = BTreeSet::new();
        for &name in profile_inputs {
            if !allowed.insert(name) || !spec.inputs.iter().any(|i| i.name == name) {
                return Err(format!("missing or repeated profile-owned input {name}").into());
            }
        }
        let names: Vec<&str> = spec.inputs.iter().map(|i| i.name.as_str()).collect();
        let (problem, model, _) =
            self.resolve_from(problem, Source::Job { spec, base }, &names, &allowed)?;
        Ok((problem, model))
    }

    /// Resolves `problem` with inputs from `source`; `inputs` names all of
    /// them, and each must be read by the model or be `allowed`.
    fn resolve_from(
        &self,
        mut problem: Problem,
        source: Source,
        inputs: &[&str],
        allowed: &BTreeSet<&str>,
    ) -> Result<(Problem, Model, Reads)> {
        let entry = self.entries.get(&problem.id).ok_or_else(|| {
            format!(
                "unknown problem {}; describe lists the registered models",
                problem.id
            )
        })?;
        if problem.version != entry.version {
            return Err(format!(
                "problem {} has version {}, not {}",
                problem.id, entry.version, problem.version
            )
            .into());
        }
        let n = problem.config.dimension;
        if !(1..=MAX_DIMENSION).contains(&n) {
            return Err("dimension must be 1..10000".into());
        }
        let instance = Instance {
            id: &problem.id,
            dimension: n,
            given: problem.parameters.clone().unwrap_or_default(),
            used: RefCell::new(Map::new()),
            defaults: RefCell::new(Map::new()),
            required: RefCell::new(BTreeSet::new()),
            source,
            inputs: RefCell::new(BTreeSet::new()),
        };
        let model = (entry.factory)(&instance)?;
        let used = instance.used.into_inner();
        let unknown: Vec<&str> = instance
            .given
            .keys()
            .filter(|k| !used.contains_key(*k))
            .map(String::as_str)
            .collect();
        if !unknown.is_empty() {
            return Err(format!("{} has no parameter {}", problem.id, unknown.join(", ")).into());
        }
        let read = instance.inputs.into_inner();
        if let Some(unused) = inputs
            .iter()
            .find(|i| !read.contains(**i) && !allowed.contains(*i))
        {
            return Err(format!("input {unused} is not used by {}", problem.id).into());
        }
        let reads = Reads {
            used: used.clone(),
            defaults: instance.defaults.into_inner(),
            required: instance.required.into_inner(),
        };
        let domain = match &model {
            Model::Scalar(m) => m.domain.clone(),
            Model::MultiObjective(m) => {
                if m.objectives == 0 {
                    return Err(format!("{} declares no objective", problem.id).into());
                }
                m.domain.clone()
            }
            Model::QualityDiversity(m) => {
                let (l, u) = (&m.descriptor_lower, &m.descriptor_upper);
                if l.is_empty()
                    || l.len() != u.len()
                    || l.iter()
                        .zip(u)
                        .any(|(l, u)| !l.is_finite() || !u.is_finite() || l >= u)
                {
                    return Err(format!(
                        "{} needs finite descriptor bounds with lower < upper",
                        problem.id
                    )
                    .into());
                }
                m.domain.clone()
            }
        };
        problem.bounds = problem.bounds.resolve(n)?;
        if let Some(d) = domain {
            if d.lower.len() != n || d.upper.len() != n {
                return Err(
                    format!("the domain of {} does not have dimension {n}", problem.id).into(),
                );
            }
            if !problem.bounds.within(&d.lower, &d.upper) {
                return Err(format!(
                    "{} bounds must lie inside {}",
                    problem.id,
                    describe_box(&d.lower, &d.upper)
                )
                .into());
            }
        }
        if problem.parameters.is_some() || !used.is_empty() {
            problem.parameters = Some(used);
        }
        Ok((problem, model, reads))
    }
}

/// `[l, u] in every coordinate` for a uniform box, else both vectors.
fn describe_box(lower: &[f64], upper: &[f64]) -> String {
    let uniform = |v: &[f64]| v.iter().all(|x| *x == v[0]);
    if !lower.is_empty() && uniform(lower) && uniform(upper) {
        format!("[{}, {}] in every coordinate", lower[0], upper[0])
    } else {
        format!("lower {lower:?}, upper {upper:?}")
    }
}
