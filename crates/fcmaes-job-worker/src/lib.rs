//! fcmaes adapters for the common job protocol (`optimization.job/0.1`).
//!
//! - `fcmaes.retry/1`: independent BiteOpt, DE, CMA-ES (optionally with
//!   IPOP or BIPOP restarts), CR-FM-NES or PGPE retries on registered scalar
//!   models;
//! - `fcmaes.mode/1`: independent MODE runs on registered multi-objective
//!   models, each returning a re-evaluated non-dominated set;
//! - `fcmaes.map-elites/1`: independent MAP-Elites runs on registered
//!   quality-diversity models on a grid or CVT archive, optionally followed
//!   by CMA-ME or CMA-MAE emitters, each returning a re-evaluated elite
//!   archive;
//! - `fcmaes.evaluate/1`: evaluates given points on any of these models
//!   without optimizing;
//! - `fcmaes.surrogate/1`: exact-GP synchronous batches for small-budget,
//!   deterministic scalar models.
//!
//! Every scalar profile accepts `Scalar::simulation` models: cancellable,
//! fallible Rust simulations. Models come from a [`Registry`]: the built-in
//! benchmarks and whatever an application registers (`registry` module docs,
//! `examples/custom_worker.rs`). `describe` lists the profiles its models
//! support, a JSON Schema of each payload (`payload_schemas`, `input_schemas`)
//! and runnable `examples` built from the registry.
//! Every optimizing task is one seeded run with its own optimizer state; a
//! population-based run evaluates its populations on `inner_threads`
//! threads of its own and gives the same result with any number of them.
//! Every outcome carries evidence (result schema 0.2). The legacy
//! `fcmaes-native-worker` and its `fcmaes.job/0.1` contract are unchanged.
#![forbid(unsafe_code)]
pub mod evaluate;
pub mod map_elites;
pub mod mode;
pub mod problems;
pub mod registry;
pub mod retry;
pub mod schema;
pub mod simulation;
pub mod surrogate;
pub mod threads;

use registry::Probe;
pub use registry::Registry;

use optimization_job_api::{
    JobSpec, Result,
    worker::{self, Prepared, Worker},
};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

/// All profiles of this worker. `describe` lists those its registered models
/// support ([`Registry::profiles`]).
pub const PROFILES: [&str; 5] = [
    retry::PROFILE,
    mode::PROFILE,
    map_elites::PROFILE,
    evaluate::PROFILE,
    surrogate::PROFILE,
];

/// Largest `inner_threads`: the envelope's `cpu_slots` limit.
pub const MAX_INNER_THREADS: usize = 1024;

/// Cooperative stop for generation-based runs. The SDK calls
/// [`Prepared::cancel`] both for a cancel request and at the wall deadline,
/// so a stopped run reports `cancelled` either way; the job state tells the
/// two apart (`cancelled` or `timed_out`).
#[derive(Default)]
pub struct Control {
    cancelled: Arc<AtomicBool>,
}
impl Control {
    pub(crate) fn token(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    /// Why a run must stop before its next generation, if it must.
    pub fn stop(&self) -> Option<&'static str> {
        self.cancelled
            .load(Ordering::Acquire)
            .then_some("cancelled")
    }
}

/// The fcmaes worker: the five profiles on the models of a [`Registry`].
/// An application worker registers its own models and runs
/// `optimization_job_api::worker::run(Adapter::new(registry))`.
pub struct Adapter {
    registry: Registry,
}
impl Adapter {
    pub fn new(registry: Registry) -> Self {
        Self { registry }
    }
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}
/// The worker with the built-in benchmark models.
impl Default for Adapter {
    fn default() -> Self {
        Self::new(Registry::benchmarks())
    }
}
/// The payload schema of `profile` for the models of `registry`.
fn payload_schema(registry: &Registry, profile: &str) -> Value {
    let models = registry.models(profile);
    let largest = if profile == surrogate::PROFILE {
        fcmaes_surrogate::MAX_DIMENSION
    } else {
        registry::MAX_DIMENSION
    };
    let problem = schema::problem(&models, largest);
    match profile {
        retry::PROFILE => retry::payload_schema(problem),
        mode::PROFILE => mode::payload_schema(problem),
        map_elites::PROFILE => map_elites::payload_schema(problem),
        evaluate::PROFILE => evaluate::payload_schema(problem),
        _ => surrogate::payload_schema(problem),
    }
}

/// Whether `profile` can run an example of `probe`.
fn fits(probe: &Probe, profile: &str) -> bool {
    probe.kind.profiles().contains(&profile)
        && match profile {
            mode::PROFILE => probe.mode_ready,
            surrogate::PROFILE => probe.dimension <= fcmaes_surrogate::MAX_DIMENSION,
            _ => true,
        }
}

/// The examples of `profile` on `probe`.
fn profile_examples(profile: &str, probe: &Probe, representative: bool) -> Vec<Value> {
    match profile {
        retry::PROFILE => retry::examples(probe, representative),
        mode::PROFILE => vec![mode::example(probe)],
        map_elites::PROFILE => vec![map_elites::example(probe)],
        evaluate::PROFILE => vec![evaluate::example(probe)],
        _ => vec![surrogate::example(probe)],
    }
}

/// Runnable examples: for each profile, one on the first model (by
/// identifier) that needs no data inputs, else the first; and one in its
/// optimizing profile for every model whose example has inputs or
/// parameters, as its jobs take another shape.
fn examples(probes: &BTreeMap<String, Probe>, profiles: &[&'static str]) -> Vec<Value> {
    let mut examples = Vec::new();
    let mut covered = BTreeSet::new();
    for &profile in profiles {
        let first = probes
            .values()
            .filter(|p| fits(p, profile))
            .min_by_key(|p| !p.inputs.is_empty());
        if let Some(probe) = first {
            examples.extend(profile_examples(profile, probe, true));
            covered.insert((profile, probe.id.clone()));
        }
    }
    for probe in probes.values() {
        let own = !probe.inputs.is_empty() || !probe.parameters.is_empty();
        if own && fits(probe, probe.profile) && covered.insert((probe.profile, probe.id.clone())) {
            examples.extend(profile_examples(probe.profile, probe, false));
        }
    }
    examples
}

impl Worker for Adapter {
    fn describe(&self) -> Value {
        let probes = self.registry.probes();
        let profiles = self.registry.profiles();
        let mut d = worker::description(
            "fcmaes",
            // The worker inherits the workspace version, which fcmaes-core shares.
            env!("CARGO_PKG_VERSION"),
            json!(profiles),
            self.registry.discovery_with(&probes),
            MAX_INNER_THREADS,
        );
        d["resources"]["inner_threads"] = json!(
            "population evaluation threads per task for cmaes, crfmnes, pgpe, mode, map_elites, and biteopt and de with optimizer.parameters.batch (results do not depend on their number); 1 for serial biteopt and de and for fcmaes.evaluate/1"
        );
        d["inputs"] = json!(
            "files a registered model reads, by name, checked against their SHA-256; a job may name only inputs its model reads"
        );
        let mut optimizers = retry::describe();
        optimizers["mode"] = mode::describe();
        optimizers["map_elites"] = map_elites::describe();
        d["optimizers"] = optimizers;
        d["evaluation"] = evaluate::describe();
        d["surrogate"] = surrogate::describe();
        d["resources"]["surrogate"] = json!(
            "synchronous batches: concurrent * evaluation_threads <= inner_threads <= 8; GP work and nested Rayon callbacks use the same task-owned pool between/within batches"
        );
        d["inputs"] = json!(
            "model inputs are hash-checked and must be consumed; fcmaes.surrogate/1 may additionally consume its named warm-start input"
        );
        d["claims"] = json!({
            "best_found":"fcmaes.retry/1: the best point the optimizer evaluated (not a model mean or recommendation), re-evaluated and inside the bounds; evidence rechecked [objective, bounds]; no global certificate",
            "non_dominated_set":"fcmaes.mode/1: details.front is the feasible non-dominated part of the final population, re-evaluated; evidence rechecked; no Pareto-optimality certificate",
            "elite_archive":"fcmaes.map-elites/1: every elite in details.elites is re-evaluated and maps to its niche; x/objective are the best elite; evidence rechecked [objective, bounds, niche]",
            "feasible":"scalar and quality-diversity models (fcmaes.evaluate/1; fcmaes.retry/1 and fcmaes.surrogate/1 after an objective_mismatch refutation): within bounds and a usable value; no constraint check (they declare no constraints; a penalty inside the objective is not checked). Multi-objective models (fcmaes.evaluate/1): every declared constraint <= 0, rechecked",
            "none":"no usable answer (claimed evidence: a penalty value, a failed simulation, an empty front or archive); with evidence.refuted, a best_found claim that failed its re-check"
        });
        d["cancellation"] = json!({
            retry::PROFILE: "between retries; active retries run until their budget or process deadline; Scalar::simulation models stop at SimContext checkpoints, and the rest of the retry's evaluations return +inf at once",
            mode::PROFILE: "between generations; the run returns what it has",
            map_elites::PROFILE: "between generations; the run returns what it has",
            evaluate::PROFILE: "between points; Scalar::simulation models also at SimContext checkpoints"
        });
        d["cancellation"][surrogate::PROFILE] = json!(
            "stop before proposals and evaluations; Scalar::simulation also stops at SimContext checkpoints; plain callbacks must finish or the runner enforces the process deadline"
        );
        d["claims"]["best_found"] = json!(
            "fcmaes.retry/1 and fcmaes.surrogate/1: best evaluated point, no global certificate; surrogate evidence is rechecked only after exact independent deterministic re-evaluation, otherwise claimed; warm values remain historical inputs; a cancelled Scalar::simulation retry's point is not rechecked (claimed)"
        );
        let schemas: Map<String, Value> = profiles
            .iter()
            .map(|&p| (p.to_string(), payload_schema(&self.registry, p)))
            .collect();
        d["payload_schemas"] = Value::Object(schemas);
        if profiles.contains(&surrogate::PROFILE) {
            d["input_schemas"] = json!({surrogate::WARM_SCHEMA: surrogate::warm_start_schema()});
        }
        d["examples"] = json!(examples(&probes, &profiles));
        d
    }
    fn prepare(&self, s: &mut JobSpec, base: &Path) -> Result<Box<dyn Prepared>> {
        let profiles = self.registry.profiles();
        if !profiles.contains(&s.payload_schema.as_str()) {
            return Err(format!(
                "unsupported payload schema; this worker accepts {}",
                profiles.join(", ")
            )
            .into());
        }
        let registry = &self.registry;
        match s.payload_schema.as_str() {
            retry::PROFILE => retry::prepare(registry, s, base),
            mode::PROFILE => mode::prepare(registry, s, base),
            map_elites::PROFILE => map_elites::prepare(registry, s, base),
            surrogate::PROFILE => surrogate::prepare(registry, s, base),
            _ => evaluate::prepare(registry, s, base),
        }
    }
}
