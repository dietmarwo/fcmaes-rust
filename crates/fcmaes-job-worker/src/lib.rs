//! fcmaes adapters for the common job protocol (`optimization.job/0.1`).
//!
//! - `fcmaes.retry/1`: independent BiteOpt, DE or CMA-ES retries on
//!   registered scalar objectives;
//! - `fcmaes.mode/1`: independent MODE runs on registered multi-objective
//!   models, each returning a re-evaluated non-dominated set;
//! - `fcmaes.map-elites/1`: independent MAP-Elites runs on registered
//!   quality-diversity models, each returning a re-evaluated elite archive.
//!
//! Every task is one seeded run with its own optimizer state. The legacy
//! `fcmaes-native-worker` and its `fcmaes.job/0.1` contract are unchanged.
#![forbid(unsafe_code)]
pub mod map_elites;
pub mod mode;
pub mod problems;
pub mod retry;

use optimization_job_api::{
    JobSpec, Result, legacy,
    worker::{self, Prepared, Worker},
};
use serde_json::{Value, json};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

/// All profiles accepted by this worker.
pub const PROFILES: [&str; 3] = [retry::PROFILE, mode::PROFILE, map_elites::PROFILE];

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

pub struct Adapter;
impl Worker for Adapter {
    fn describe(&self) -> Value {
        let legacy = legacy::discovery();
        let (multi_objective, quality_diversity) = problems::discovery();
        let mut problems = Vec::new();
        for (profile, list) in [
            (retry::PROFILE, &legacy["problems"]),
            (mode::PROFILE, &multi_objective),
            (map_elites::PROFILE, &quality_diversity),
        ] {
            for entry in list.as_array().into_iter().flatten() {
                let mut entry = entry.clone();
                entry["profile"] = json!(profile);
                problems.push(entry);
            }
        }
        let mut d = worker::description(
            "fcmaes",
            // The worker inherits the workspace version, which fcmaes-core shares.
            env!("CARGO_PKG_VERSION"),
            json!(PROFILES),
            json!(problems),
            1,
        );
        let mut optimizers = legacy["optimizers"].clone();
        for kind in ["biteopt", "de"] {
            optimizers[kind]["profile"] = json!(retry::PROFILE);
        }
        optimizers["cmaes"] = json!({"profile":retry::PROFILE,"limit":"soft",
            "maximum_overshoot_per_retry":"population-1",
            "defaults":{"population":retry::CMAES_POPULATION,"sigma":retry::CMAES_SIGMA},
            "sigma":"initial step size in the normalized [-1,1] box, (0,2]",
            "start":"uniform random point in the bounds from the retry seed"});
        optimizers["mode"] = mode::describe();
        optimizers["map_elites"] = map_elites::describe();
        d["optimizers"] = optimizers;
        d["claims"] = json!({
            "best_found":"fcmaes.retry/1: candidate re-evaluated and inside the bounds; no global certificate",
            "non_dominated_set":"fcmaes.mode/1: details.front is re-evaluated, feasible and mutually non-dominated; no Pareto-optimality certificate",
            "elite_archive":"fcmaes.map-elites/1: every elite in details.elites is re-evaluated and maps to its niche; x/objective are the best elite"
        });
        d["cancellation"] = json!({
            retry::PROFILE: "between retries; active retries run until their budget or process deadline",
            mode::PROFILE: "between generations; the run returns what it has",
            map_elites::PROFILE: "between generations; the run returns what it has"
        });
        d
    }
    fn prepare(&self, s: &mut JobSpec, _: &Path) -> Result<Box<dyn Prepared>> {
        if !PROFILES.contains(&s.payload_schema.as_str()) {
            return Err(format!(
                "unsupported payload schema; this worker accepts {}",
                PROFILES.join(", ")
            )
            .into());
        }
        if s.resources.inner_threads != 1 || !s.inputs.is_empty() {
            return Err(format!(
                "{} requires inner_threads=1 and no inputs",
                s.payload_schema
            )
            .into());
        }
        match s.payload_schema.as_str() {
            retry::PROFILE => retry::prepare(s),
            mode::PROFILE => mode::prepare(s),
            _ => map_elites::prepare(s),
        }
    }
}
