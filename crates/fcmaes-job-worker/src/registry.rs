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
use crate::{evaluate, map_elites, mode, retry, surrogate};
use optimization_job_api::{JobSpec, Result, legacy};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
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

/// What a factory knows about the problem it builds a model for.
pub struct Instance<'a> {
    id: &'a str,
    dimension: usize,
    given: Map<String, Value>,
    used: RefCell<Map<String, Value>>,
    spec: &'a JobSpec,
    base: &'a Path,
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
        let input = self
            .spec
            .inputs
            .iter()
            .find(|i| i.name == name)
            .ok_or_else(|| format!("{} requires the job input {name}", self.id))?;
        let bytes = optimization_job_api::checked_input(self.base, input)
            .map_err(|e| format!("input {name}: {e}"))?;
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
}

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

    /// The `describe` entries of all models, with their kind and profile.
    pub fn discovery(&self) -> Value {
        self.entries
            .iter()
            .map(|(id, e)| {
                let mut entry = json!({"id":id,"version":e.version,"kind":e.kind.name(),
                    "profile":e.kind.profile(), "profiles":e.kind.profiles()});
                for (k, v) in e.about.as_object().into_iter().flatten() {
                    entry[k] = v.clone();
                }
                // An expensive simulation is optimized by the surrogate by
                // default; retry and evaluate accept it as well.
                if entry["simulation"] == true {
                    entry["profile"] = json!(surrogate::PROFILE);
                }
                entry
            })
            .collect()
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
        mut problem: Problem,
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
            spec,
            base,
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
        if let Some(unused) = spec
            .inputs
            .iter()
            .find(|i| !read.contains(&i.name) && !allowed.contains(i.name.as_str()))
        {
            return Err(format!("input {} is not used by {}", unused.name, problem.id).into());
        }
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
        Ok((problem, model))
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
