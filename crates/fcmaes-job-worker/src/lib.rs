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
//!   without optimizing.
//!
//! Models come from a [`Registry`]: the built-in benchmarks and whatever an
//! application registers (`registry` module docs, `examples/custom_worker.rs`).
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
pub mod threads;

pub use registry::Registry;

use optimization_job_api::{
    JobSpec, Result,
    worker::{self, Prepared, Worker},
};
use serde_json::{Value, json};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

/// All profiles accepted by this worker.
pub const PROFILES: [&str; 4] = [
    retry::PROFILE,
    mode::PROFILE,
    map_elites::PROFILE,
    evaluate::PROFILE,
];

/// Largest `inner_threads`: the envelope's `cpu_slots` limit.
pub const MAX_INNER_THREADS: usize = 1024;

/// Cooperative stop for generation-based runs. The SDK calls
/// [`Prepared::cancel`] both for a cancel request and at the wall deadline,
/// so a stopped run reports `cancelled` either way; the job state tells the
/// two apart (`cancelled` or `timed_out`).
#[derive(Default)]
pub struct Control {
    cancelled: AtomicBool,
}
impl Control {
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

/// The fcmaes worker: the four profiles on the models of a [`Registry`].
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
impl Worker for Adapter {
    fn describe(&self) -> Value {
        let mut d = worker::description(
            "fcmaes",
            // The worker inherits the workspace version, which fcmaes-core shares.
            env!("CARGO_PKG_VERSION"),
            json!(PROFILES),
            self.registry.discovery(),
            MAX_INNER_THREADS,
        );
        d["resources"]["inner_threads"] = json!(
            "population evaluation threads per task for cmaes, crfmnes, pgpe, mode and map_elites (results do not depend on their number); 1 for biteopt, de and fcmaes.evaluate/1"
        );
        d["inputs"] = json!(
            "files a registered model reads, by name, checked against their SHA-256; a job may name only inputs its model reads"
        );
        let mut optimizers = retry::describe();
        optimizers["mode"] = mode::describe();
        optimizers["map_elites"] = map_elites::describe();
        d["optimizers"] = optimizers;
        d["evaluation"] = evaluate::describe();
        d["claims"] = json!({
            "best_found":"fcmaes.retry/1: the best point the optimizer evaluated (not a model mean or recommendation), re-evaluated and inside the bounds; evidence rechecked [objective, bounds]; no global certificate",
            "non_dominated_set":"fcmaes.mode/1: details.front is the feasible non-dominated part of the final population, re-evaluated; evidence rechecked; no Pareto-optimality certificate",
            "elite_archive":"fcmaes.map-elites/1: every elite in details.elites is re-evaluated and maps to its niche; x/objective are the best elite; evidence rechecked [objective, bounds, niche]",
            "feasible":"fcmaes.evaluate/1: the given point has a usable value (or satisfies the constraints); fcmaes.retry/1 after an objective_mismatch refutation",
            "none":"no usable answer (claimed evidence: a penalty value, an empty front or archive); with evidence.refuted, a best_found claim that failed its re-check"
        });
        d["cancellation"] = json!({
            retry::PROFILE: "between retries; active retries run until their budget or process deadline",
            mode::PROFILE: "between generations; the run returns what it has",
            map_elites::PROFILE: "between generations; the run returns what it has",
            evaluate::PROFILE: "between points"
        });
        d
    }
    fn prepare(&self, s: &mut JobSpec, base: &Path) -> Result<Box<dyn Prepared>> {
        if !PROFILES.contains(&s.payload_schema.as_str()) {
            return Err(format!(
                "unsupported payload schema; this worker accepts {}",
                PROFILES.join(", ")
            )
            .into());
        }
        let registry = &self.registry;
        match s.payload_schema.as_str() {
            retry::PROFILE => retry::prepare(registry, s, base),
            mode::PROFILE => mode::prepare(registry, s, base),
            map_elites::PROFILE => map_elites::prepare(registry, s, base),
            _ => evaluate::prepare(registry, s, base),
        }
    }
}
