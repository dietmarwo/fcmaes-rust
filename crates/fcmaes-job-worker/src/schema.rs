//! JSON Schema (draft 2020-12) building blocks for `describe`'s
//! `payload_schemas` and `input_schemas`, and the runnable `examples`.
//!
//! Each schema accepts every payload the worker accepts. Rules a schema
//! cannot state (cross-field arithmetic, per-model dimensions, resource
//! rules) are in `$comment`s.
use crate::registry::Probe;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const DRAFT: &str = "https://json-schema.org/draft/2020-12/schema";

/// `schema` as the payload schema of `profile`.
pub fn payload(profile: &str, mut schema: Value) -> Value {
    schema["$schema"] = json!(DRAFT);
    schema["$id"] = json!(format!("urn:optimization-jobs:payload:{profile}"));
    schema
}

/// An integer in `[minimum, maximum]`.
pub fn integer(minimum: u64, maximum: u64) -> Value {
    json!({"type":"integer","minimum":minimum,"maximum":maximum})
}

/// Any `u64`, such as a seed.
pub fn u64() -> Value {
    integer(0, u64::MAX)
}

/// A number with optional bounds; `open` makes the lower one exclusive.
pub fn number(minimum: Option<f64>, maximum: Option<f64>, open: bool) -> Value {
    let mut n = json!({"type":"number"});
    if let Some(m) = minimum {
        n[if open { "exclusiveMinimum" } else { "minimum" }] = json!(m);
    }
    if let Some(m) = maximum {
        n["maximum"] = json!(m);
    }
    n
}

/// A non-empty array of numbers.
pub fn numbers() -> Value {
    json!({"type":"array","minItems":1,"items":{"type":"number"}})
}

/// `schema` or `null`: optional settings the worker reads as absent.
pub fn or_null(mut schema: Value) -> Value {
    if let Some(values) = schema.get_mut("enum").and_then(Value::as_array_mut) {
        values.push(Value::Null);
    } else if let Some(t) = schema.get("type").cloned() {
        schema["type"] = json!([t, "null"]);
    }
    schema
}

/// `null` only: a setting that does not apply.
pub fn null() -> Value {
    json!({"type":"null"})
}

/// `{"properties": {name: null, ...}}` for settings that must be absent.
pub fn absent(names: &[&str]) -> Value {
    let properties: serde_json::Map<String, Value> =
        names.iter().map(|n| (n.to_string(), null())).collect();
    json!({"properties":properties})
}

/// The `problem` of a payload: one of `models` (identifier and version) of
/// at most `max_dimension` variables.
pub fn problem(models: &[(String, String)], max_dimension: usize) -> Value {
    let mut versions: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (id, version) in models {
        versions.entry(version).or_default().push(id);
    }
    let models: Vec<Value> = versions
        .iter()
        .map(|(version, ids)| json!({"properties":{"id":{"enum":ids},"version":{"const":version}}}))
        .collect();
    json!({"type":"object","additionalProperties":false,"required":["id","version","config","bounds"],
        "properties":{"id":{"type":"string"},"version":{"type":"string"},
            "config":{"type":"object","additionalProperties":false,"required":["dimension"],
                "properties":{"dimension":integer(1, max_dimension as u64)}},
            "bounds":{"type":"object","additionalProperties":false,"required":["lower","upper","broadcast"],
                "properties":{"lower":numbers(),"upper":numbers(),"broadcast":{"type":"boolean"}},
                "if":{"properties":{"broadcast":{"const":true}}},
                "then":{"properties":{"lower":{"maxItems":1},"upper":{"maxItems":1}}}},
            "parameters":{"type":["object","null"]}},
        "oneOf":models,
        "$comment":"a registered model with its version and the dimensions its problems entry accepts; bounds: one value per variable (broadcast: one for all), finite, lower < upper, inside the model's domain; parameters: the model's parameters_schema"})
}

/// One `examples` entry: a job without `schema_version`, `request_key`,
/// `worker`, `payload_schema` and input hashes, which a client adds.
pub fn example(profile: &str, description: String, probe: &Probe, payload: Value) -> Value {
    let inputs: Vec<Value> = probe
        .inputs
        .iter()
        .map(|i| json!({"name":i.name,"file":i.file,"content_text":i.content_text}))
        .collect();
    let mut description = description;
    if !inputs.is_empty() {
        description.push_str(", with its data input");
    }
    json!({"profile":profile,"description":description,
        "resources":{"cpu_slots":1,"outer_workers":1,"inner_threads":1},
        "limits":{"wall_time_seconds":30.0,"grace_seconds":1.0},
        "payload":payload,"inputs":inputs})
}

/// The centre of a probe's box.
pub fn centre(probe: &Probe) -> Vec<f64> {
    probe
        .lower
        .iter()
        .zip(&probe.upper)
        .map(|(l, u)| 0.5 * (l + u))
        .collect()
}
