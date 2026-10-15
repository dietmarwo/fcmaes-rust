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
contract!(Budget {
    retries: usize,
    evaluations_per_retry: u64,
    evaluation_limit_mode: String,
    wall_time_seconds: f64
});
contract!(Outputs { progress: String });
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
    pub fn digest(&self) -> Result<String> {
        Ok(crate::storage::digest(&serde_json::to_vec(self)?))
    }
}
pub fn discovery() -> serde_json::Value {
    serde_json::json!({
        "protocol":"fcmaes.worker/0.1", "problems":[
        {"id":"benchmark.sphere","version":"1","dimensions":"1..10000"},
        {"id":"benchmark.rosenbrock","version":"1","dimensions":"2..10000"},
        {"id":"gtop.cassini1","version":"1","dimensions":6,
        "lower":[-1000,30,100,30,400,1000],"upper":[0,400,470,400,2000,6000]}],
        "optimizers":{"biteopt":{"limit":"hard"},"de":{"limit":"soft","maximum_overshoot_per_retry":"2*population-1"}},
        "parallelism":"independent retries; one evaluation thread per retry"
    })
}
