use crate::{Result, coded, legacy, storage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path},
};

pub const SCHEMA: &str = "optimization.job/0.1";
/// The worker protocol of this build. It adds the result contract 0.2
/// (evidence, refutations, objective bounds and the infeasibility and
/// certified claims) to the process protocol of 0.1.
pub const PROTOCOL: &str = "optimization.worker/0.2";
/// Worker protocols the runner accepts. Every 0.1 result record is a valid
/// 0.2 record, so 0.1 workers keep working; a 0.1 runner rejects 0.2 workers
/// at registration instead of failing on their journals.
pub const PROTOCOLS: [&str; 2] = ["optimization.worker/0.1", PROTOCOL];
/// True if the runner accepts a worker that describes itself with `protocol`.
pub fn protocol_supported(protocol: &Value) -> bool {
    protocol.as_str().is_some_and(|p| PROTOCOLS.contains(&p))
}
pub const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    pub cpu_slots: usize,
    pub outer_workers: usize,
    pub inner_threads: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub wall_time_seconds: f64,
    pub grace_seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub name: String,
    pub file: String,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JobSpec {
    pub schema_version: String,
    pub request_key: String,
    pub worker: String,
    pub payload_schema: String,
    pub payload: Value,
    pub resources: Resources,
    pub limits: Limits,
    #[serde(default)]
    pub inputs: Vec<Input>,
    /// Filled by the worker during validation, never inferred by the runner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_count: Option<usize>,
}
pub fn safe_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 80
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        && s != "."
        && s != ".."
}
impl JobSpec {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA
            || self.request_key.is_empty()
            || self.request_key.len() > 200
            || !safe_name(&self.worker)
            || self.payload_schema.is_empty()
            || !self.payload.is_object()
        {
            return Err("invalid job envelope, schema, request_key or worker".into());
        }
        let r = &self.resources;
        if r.cpu_slots == 0
            || r.cpu_slots > 1024
            || r.outer_workers == 0
            || r.inner_threads == 0
            || r.outer_workers
                .checked_mul(r.inner_threads)
                .is_none_or(|n| n > r.cpu_slots)
        {
            return Err(
                "resources require 1 <= outer_workers * inner_threads <= cpu_slots <= 1024".into(),
            );
        }
        if !self.limits.wall_time_seconds.is_finite()
            || !(0.0..=31_536_000.0).contains(&self.limits.wall_time_seconds)
            || self.limits.wall_time_seconds == 0.0
            || !self.limits.grace_seconds.is_finite()
            || !(0.0..=86400.0).contains(&self.limits.grace_seconds)
        {
            return Err("invalid wall time or grace".into());
        }
        if self.task_count.is_some_and(|n| n == 0 || n > 1_000_000) || self.inputs.len() > 16 {
            return Err("invalid task/input count".into());
        }
        let mut names = BTreeSet::new();
        for i in &self.inputs {
            if !safe_name(&i.name)
                || !names.insert(&i.name)
                || i.sha256.len() != 64
                || !i
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                || i.file.is_empty()
                || Path::new(&i.file)
                    .components()
                    .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
                || !Path::new(&i.file)
                    .components()
                    .any(|c| matches!(c, Component::Normal(_)))
            {
                return Err("input requires unique safe name, relative contained path and lowercase SHA-256".into());
            }
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        let mut v = serde_json::to_value(self)?;
        v.as_object_mut().unwrap().remove("request_key");
        Ok(storage::digest(&serde_json::to_vec(&v)?))
    }
    pub fn input_bytes(&self, base: &Path, name: &str) -> Result<Vec<u8>> {
        let input = self
            .inputs
            .iter()
            .find(|i| i.name == name)
            .ok_or("missing model input")?;
        checked_input(base, input)
    }
}
pub fn checked_input(base: &Path, input: &Input) -> Result<Vec<u8>> {
    let base = fs::canonicalize(base)?;
    let path = fs::canonicalize(base.join(&input.file))?;
    if !path.starts_with(&base) || !path.is_file() || fs::metadata(&path)?.len() > MAX_INPUT_BYTES {
        return Err("input escapes directory or exceeds 64 MiB".into());
    }
    // The bounded read also covers a file that grows after the metadata check.
    use std::io::Read;
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_INPUT_BYTES || storage::digest(&bytes) != input.sha256 {
        return Err("input SHA-256 mismatch or oversized input".into());
    }
    Ok(bytes)
}
pub fn stage_inputs(s: &mut JobSpec, from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to.join("inputs"))?;
    for i in &mut s.inputs {
        let bytes = checked_input(from, i)?;
        i.file = format!("inputs/{}", i.name);
        let path = to.join(&i.file);
        fs::write(&path, bytes)?;
        fs::File::open(path)?.sync_all()?;
    }
    fs::File::open(to.join("inputs"))?.sync_all()?;
    Ok(())
}
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Request {
    Legacy(legacy::JobSpec),
    Job(JobSpec),
}
impl Request {
    /// Parses and validates a submitted job; every failure is `invalid_job`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::parse_unchecked(bytes).map_err(|e| coded("invalid_job", e.to_string()))
    }
    fn parse_unchecked(bytes: &[u8]) -> Result<Self> {
        let v: Value = serde_json::from_slice(bytes)?;
        match v["schema_version"].as_str() {
            Some("fcmaes.job/0.1") => Ok(Self::Legacy(
                serde_json::from_value::<legacy::JobSpec>(v)?.resolve()?,
            )),
            Some(SCHEMA) => {
                let s: JobSpec = serde_json::from_value(v)?;
                s.validate()?;
                Ok(Self::Job(s))
            }
            _ => Err("unsupported job schema".into()),
        }
    }
    pub fn request_key(&self) -> &str {
        match self {
            Self::Legacy(s) => &s.request_key,
            Self::Job(s) => &s.request_key,
        }
    }
    pub fn worker(&self) -> &str {
        match self {
            Self::Legacy(_) => "native",
            Self::Job(s) => &s.worker,
        }
    }
    pub fn slots(&self) -> usize {
        match self {
            Self::Legacy(s) => s.execution.workers,
            Self::Job(s) => s.resources.cpu_slots,
        }
    }
    pub fn wall(&self) -> f64 {
        match self {
            Self::Legacy(s) => s.budget.wall_time_seconds,
            Self::Job(s) => s.limits.wall_time_seconds,
        }
    }
    pub fn grace(&self) -> f64 {
        match self {
            Self::Legacy(s) => s.budget.grace_seconds,
            Self::Job(s) => s.limits.grace_seconds,
        }
    }
    pub fn digest(&self) -> Result<String> {
        match self {
            Self::Legacy(s) => s.digest(),
            Self::Job(s) => s.digest(),
        }
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        Ok(storage::write_atomic_json(path, self)?)
    }
}
