use crate::Result;
use serde::{Deserialize, Serialize};
macro_rules! contract { ($name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
    #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    pub struct $name { $(pub $field: $ty),* }
}; }
contract!(JobSpec {
    schema_version: String,
    request_key: String,
    problem: Problem,
    optimizer: Optimizer,
    execution: Execution,
    budget: Budget,
    outputs: Outputs
});
contract!(Problem {
    id: String,
    version: String,
    config: ProblemConfig,
    bounds: Bounds
});
contract!(ProblemConfig { dimension: usize });
contract!(Bounds { lower: Vec<f64>, upper: Vec<f64>, broadcast: bool });
contract!(Optimizer {
    kind: String,
    parameters: Parameters
});
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub population: Option<usize>,
}
contract!(Execution {
    strategy: String,
    seed: u64,
    workers: usize,
    retry_wave_size: usize
});
/// Evaluation and time budget of one job.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub retries: usize,
    pub evaluations_per_retry: u64,
    pub evaluation_limit_mode: String,
    /// No new retry wave starts after this execution time.
    pub wall_time_seconds: f64,
    /// Declared interval between the wall-time deadline and killing the
    /// worker. A retry wave still running after it is lost and the job's
    /// accounting becomes incomplete.
    #[serde(default = "default_grace_seconds")]
    pub grace_seconds: f64,
}
fn default_grace_seconds() -> f64 {
    1.0
}
contract!(Outputs { progress: String });
/// Canonical Cassini1 box; job bounds must lie inside it.
pub const CASSINI1_LOWER: [f64; 6] = [-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0];
/// Canonical Cassini1 box; job bounds must lie inside it.
pub const CASSINI1_UPPER: [f64; 6] = [0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0];
impl JobSpec {
    pub fn resolve(mut self) -> Result<Self> {
        if self.schema_version != "fcmaes.job/0.1" || self.problem.version != "1" {
            return Err("unsupported schema/problem version".into());
        }
        if self.request_key.is_empty() || self.request_key.len() > 200 {
            return Err("request_key must be 1..200 bytes".into());
        }
        let n = self.problem.config.dimension;
        if !(1..=10000).contains(&n) {
            return Err("dimension must be 1..10000".into());
        }
        match self.problem.id.as_str() {
            "benchmark.sphere" => (),
            "benchmark.rosenbrock" if n >= 2 => (),
            "gtop.cassini1" if n == 6 => (),
            _ => return Err("unknown problem or invalid dimension".into()),
        }
        let b = &mut self.problem.bounds;
        if b.broadcast {
            if b.lower.len() != 1 || b.upper.len() != 1 {
                return Err("broadcast requires scalar bounds".into());
            }
            b.lower = vec![b.lower[0]; n];
            b.upper = vec![b.upper[0]; n];
            b.broadcast = false;
        }
        if b.lower.len() != n
            || b.upper.len() != n
            || b.lower
                .iter()
                .zip(&b.upper)
                .any(|(l, u)| !l.is_finite() || !u.is_finite() || l >= u || !(u - l).is_finite())
        {
            return Err("invalid bounds".into());
        }
        if self.problem.id == "gtop.cassini1"
            && (b.lower.iter().zip(CASSINI1_LOWER).any(|(l, c)| *l < c)
                || b.upper.iter().zip(CASSINI1_UPPER).any(|(u, c)| *u > c))
        {
            return Err("gtop.cassini1 bounds must lie inside the canonical Cassini1 box".into());
        }
        let e = &self.execution;
        let k = &self.budget;
        if e.strategy != "independent_retry"
            || e.workers == 0
            || e.retry_wave_size == 0
            || k.retries == 0
            || k.evaluations_per_retry == 0
        {
            return Err(
                "positive explicit workers, retries, wave size and evaluations required".into(),
            );
        }
        if !k.wall_time_seconds.is_finite()
            || k.wall_time_seconds <= 0.0
            || k.wall_time_seconds > 31_536_000.0
        {
            return Err("wall_time_seconds must be in (0,31536000]".into());
        }
        if !k.grace_seconds.is_finite() || !(0.0..=86_400.0).contains(&k.grace_seconds) {
            return Err("grace_seconds must be in [0,86400]".into());
        }
        if self.outputs.progress != "completed_retries" {
            return Err("only completed_retries progress is supported".into());
        }
        let p = self
            .optimizer
            .parameters
            .population
            .unwrap_or(if self.optimizer.kind == "de" {
                31
            } else {
                9 + 3 * n
            });
        if !(4..=1_000_000).contains(&p) {
            return Err("population must be 4..1000000".into());
        }
        match (
            self.optimizer.kind.as_str(),
            k.evaluation_limit_mode.as_str(),
        ) {
            ("biteopt", "hard") | ("de", "soft") => (),
            _ => {
                return Err(
                    "supported paths: biteopt/hard, de/soft; DE hard cap is unavailable".into(),
                );
            }
        }
        self.optimizer.parameters.population = Some(p);
        self.requested()?;
        self.maximum_evaluations()?;
        Ok(self)
    }
    pub fn requested(&self) -> Result<u64> {
        (self.budget.retries as u64)
            .checked_mul(self.budget.evaluations_per_retry)
            .ok_or_else(|| "requested evaluations overflow".into())
    }
    pub fn maximum_evaluations(&self) -> Result<u64> {
        let extra = if self.optimizer.kind == "de" {
            2 * self.optimizer.parameters.population.unwrap_or(31) as u64 - 1
        } else {
            0
        };
        self.budget
            .evaluations_per_retry
            .checked_add(extra)
            .and_then(|x| x.checked_mul(self.budget.retries as u64))
            .ok_or_else(|| "actual budget bound overflow".into())
    }
    /// Digest of the execution specification. The request key is excluded:
    /// it names a submission, not an experiment, so identical experiments
    /// submitted under different keys share a digest.
    pub fn digest(&self) -> Result<String> {
        let mut value = serde_json::to_value(self)?;
        if let Some(object) = value.as_object_mut() {
            object.remove("request_key");
        }
        Ok(crate::storage::digest(&serde_json::to_vec(&value)?))
    }
}
pub fn discovery() -> serde_json::Value {
    serde_json::json!({
        "protocol":"fcmaes.worker/0.1", "problems":[
        {"id":"benchmark.sphere","version":"1","dimensions":"1..10000"},
        {"id":"benchmark.rosenbrock","version":"1","dimensions":"2..10000"},
        {"id":"gtop.cassini1","version":"1","dimensions":6,
        "lower":CASSINI1_LOWER,"upper":CASSINI1_UPPER,"bounds":"must lie inside lower/upper"}],
        "optimizers":{"biteopt":{"limit":"hard"},"de":{"limit":"soft","maximum_overshoot_per_retry":"2*population-1"}},
        "parallelism":"independent retries; one evaluation thread per retry"
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base() -> serde_json::Value {
        json!({
            "schema_version": "fcmaes.job/0.1",
            "request_key": "k1",
            "problem": {"id": "benchmark.sphere", "version": "1", "config": {"dimension": 3},
                        "bounds": {"lower": [-5.0], "upper": [5.0], "broadcast": true}},
            "optimizer": {"kind": "biteopt", "parameters": {}},
            "execution": {"strategy": "independent_retry", "seed": 1, "workers": 2, "retry_wave_size": 2},
            "budget": {"retries": 4, "evaluations_per_retry": 100, "evaluation_limit_mode": "hard",
                       "wall_time_seconds": 10.0},
            "outputs": {"progress": "completed_retries"}
        })
    }
    fn resolve(v: serde_json::Value) -> Result<JobSpec> {
        serde_json::from_value::<JobSpec>(v)?.resolve()
    }

    #[test]
    fn resolves_defaults_and_broadcast_bounds() {
        let s = resolve(base()).unwrap();
        assert_eq!(s.problem.bounds.lower, vec![-5.0; 3]);
        assert!(!s.problem.bounds.broadcast);
        assert_eq!(s.budget.grace_seconds, 1.0);
        assert_eq!(s.optimizer.parameters.population, Some(9 + 3 * 3));
        assert_eq!(s.requested().unwrap(), 400);
        assert_eq!(s.maximum_evaluations().unwrap(), 400);
    }

    #[test]
    fn de_is_soft_with_declared_overshoot_bound() {
        let mut v = base();
        v["optimizer"] = json!({"kind": "de", "parameters": {"population": 10}});
        assert!(resolve(v.clone()).is_err(), "DE must reject a hard cap");
        v["budget"]["evaluation_limit_mode"] = json!("soft");
        let s = resolve(v).unwrap();
        assert_eq!(s.maximum_evaluations().unwrap(), 4 * (100 + 2 * 10 - 1));
    }

    #[test]
    fn rejects_zero_unknown_and_overflowing_inputs() {
        for (path, value) in [
            (vec!["execution", "workers"], json!(0)),
            (vec!["execution", "retry_wave_size"], json!(0)),
            (vec!["budget", "retries"], json!(0)),
            (vec!["budget", "evaluations_per_retry"], json!(0)),
            (vec!["budget", "grace_seconds"], json!(-1.0)),
            (vec!["budget", "wall_time_seconds"], json!(0.0)),
        ] {
            let mut v = base();
            v[path[0]][path[1]] = value;
            assert!(resolve(v).is_err(), "{path:?} accepted");
        }
        let mut v = base();
        v["budget"]["total"] = json!(5);
        assert!(resolve(v).is_err(), "unknown field accepted");
        let mut v = base();
        v["budget"]["retries"] = json!(usize::MAX);
        v["budget"]["evaluations_per_retry"] = json!(u64::MAX);
        assert!(resolve(v).is_err(), "overflow accepted");
    }

    #[test]
    fn cassini1_bounds_must_stay_in_the_canonical_box() {
        let mut v = base();
        v["problem"] = json!({"id": "gtop.cassini1", "version": "1", "config": {"dimension": 6},
            "bounds": {"lower": CASSINI1_LOWER, "upper": CASSINI1_UPPER, "broadcast": false}});
        assert!(resolve(v.clone()).is_ok());
        v["problem"]["bounds"]["upper"][5] = json!(7000.0);
        assert!(resolve(v.clone()).is_err());
        v["problem"]["bounds"] = json!({"lower": [-1e6], "upper": [1e6], "broadcast": true});
        assert!(resolve(v).is_err());
    }

    #[test]
    fn digest_ignores_request_key_but_not_the_experiment() {
        let a = resolve(base()).unwrap();
        let mut v = base();
        v["request_key"] = json!("another-key");
        let b = resolve(v).unwrap();
        assert_eq!(a.digest().unwrap(), b.digest().unwrap());
        let mut v = base();
        v["execution"]["seed"] = json!(2);
        assert_ne!(a.digest().unwrap(), resolve(v).unwrap().digest().unwrap());
    }
}
