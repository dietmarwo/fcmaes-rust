# Application models

A job never carries code. It names a registered model, and the worker
evaluates that model natively: it can re-evaluate every answer before
reporting it, a job can be re-run from its record, and nothing deserializes a
function. The stock `fcmaes-job-worker` registers benchmark models. Your own
problems run in an *application worker*: a small binary that registers its
models in a `Registry` and runs the same fcmaes adapter, with every profile,
optimizer and check of the stock worker.

## A worker of your own

```toml
[dependencies]
fcmaes-job-worker = { git = "https://github.com/dietmarwo/fcmaes-rust" }
optimization-job-api = { git = "https://github.com/dietmarwo/fcmaes-rust" }
serde_json = "1"
```

```rust
use fcmaes_job_worker::{Adapter, registry::{Registry, Scalar}};
use serde_json::json;

fn main() {
    let mut registry = Registry::benchmarks(); // or Registry::new()
    registry.scalar(
        "app.shifted-sphere",
        "1",
        json!({"dimensions": "1..10000", "parameters": {"shift": "default 1.0"}}),
        |model| {
            let shift: f64 = model.parameter("shift", 1.0)?;
            Ok(Scalar::new(move |x: &[f64]| {
                x.iter().map(|v| (v - shift) * (v - shift)).sum()
            }))
        },
    );
    optimization_job_api::worker::run(Adapter::new(registry));
}
```

Build it and register the binary with the runner under a worker name of its
own:

```sh
cargo build --release
optimization-run --root ROOT workers add --id my-app --executable target/release/my-app
optimization-run --root ROOT problems describe my-app
```

`examples/custom_worker.rs` in `crates/fcmaes-job-worker` is a complete
worker that fits a curve to a data file; the
[tutorial](../../tutorials/job-api/README.md) runs it.

## Model kinds

| Registration | The closure returns | Profile |
| --- | --- | --- |
| `registry.scalar(id, version, about, factory)` → `Scalar::new(f)` | one minimized value | `fcmaes.retry/1` |
| `registry.multi_objective(...)` → `MultiObjective::new(objectives, constraints, f)` | the objectives, then the constraints (feasible at or below 0) | `fcmaes.mode/1` |
| `registry.quality_diversity(...)` → `QualityDiversity::new(descriptor_lower, descriptor_upper, f)` | `(fitness, descriptor)`, fitness minimized | `fcmaes.map-elites/1` |

`fcmaes.evaluate/1` evaluates given points on a model of any kind. A
multi-objective model may define the hypervolume reference point with
`.reference_point(...)`; otherwise jobs must give `reference_point`. A model
whose descriptor does not have two dimensions (1–100 are allowed) gets a CVT
archive. `.domain(lower,
upper)` restricts job bounds to a box, for a model that is undefined outside
it (Cassini1 is one).

`about` is a JSON object added to the model's `describe` entry, next to the
`id`, `version`, `kind` and `profile` the registry fills in. Describe there
what an agent needs to write a job: dimensions, parameters, inputs, the
meaning of the variables. Keep it short: `workers add` reads at most 64 KiB
of `describe` output, about 200 models with descriptions of this tutorial's
size.

## What a factory receives

A factory builds the model for one job from an `Instance`:

| Method | Returns |
| --- | --- |
| `dimension()` | the job's `config.dimension` (1–10,000) |
| `require_dimension(2..=2)` | that dimension, or an error naming the allowed range |
| `parameter(name, default)` | the job's `problem.parameters[name]`, or `default` |
| `required(name)` | the parameter, or an error if the job omits it |
| `input(name)` | the bytes of the job input `name`, checked against its SHA-256 |

Parameters deserialize into any `serde` type. The worker records every value
the factory used, defaults included, in the resolved job's
`problem.parameters`, so the record shows exactly what ran. A job without
parameters, for a model that reads none, keeps the legacy problem shape and
its digest.

## Inputs

Data reaches a model through the envelope's `inputs`, never inline:

```json
"inputs": [{"name": "data", "file": "decay.csv", "sha256": "d864222f..."}],
"payload": {"problem": {"id": "app.exponential-fit", "version": "1",
    "config": {"dimension": 3},
    "bounds": {"lower": [0.0, 0.0, -10.0], "upper": [10.0, 5.0, 10.0], "broadcast": false}}, ...}
```

`file` is relative to the job file. The runner checks the SHA-256 and copies
the file into the attempt, so a re-run reads the same bytes; inputs are at
most 64 MiB each and 16 per job. Parse an input once in the factory and move
the parsed data into the closure, not the bytes.

## What the worker checks

At validation, before a job is accepted:

- the problem's `id` is registered and its `version` matches;
- the factory succeeds: it rejects a wrong dimension, a bad parameter value or
  malformed data with its own message;
- every parameter the job gives is one the factory read, and every input of
  the job is one it read: nothing is silently ignored;
- the bounds are finite, `lower < upper`, and inside the model's domain;
- the model's kind fits the profile (`app.fit is a scalar model; use
  fcmaes.retry/1`).

At execution, a multi-objective model that returns a different number of
values than it declares, or a quality-diversity model with a descriptor of
the wrong length, fails the task.

## Rules for models

- **Deterministic.** The worker re-evaluates every answer before it reports
  it. A model that returns a different value for the same point produces
  `objective_mismatch` refutations, and the job cannot be reproduced.
- **Thread-safe and stateless.** Closures are `Send + Sync`. With
  `inner_threads > 1` they run concurrently on a task's population; the
  results do not depend on the number of threads only if evaluations do not
  depend on each other. A model may use Rayon itself: every task runs inside
  its own pool of `inner_threads` threads, so the model's parallel work stays
  within the job's slots.
- **Quick factories.** A factory runs at validation, which has a 30-second
  limit, and again at execution. Load and parse data there. Precomputation
  that may take longer belongs in a value the closure initializes once
  (`std::sync::OnceLock`), during execution.
- **Penalties.** A non-finite value, or a value of `1e99` or more (the fcmaes
  penalty), is no answer: a retry whose best value is a penalty claims `none`.
  MODE treats non-finite objectives as the worst value.

## Testing a model

`Worker::prepare` resolves a job exactly as `validate` does, and
`optimization_job_api::worker::execute` runs one in a directory, both in the
test process. `crates/fcmaes-job-worker/tests/registry.rs` shows a test that
stages an input, resolves the job, executes it and reads `run.json`, and the
error cases above.

## Why there is no callback or ask/tell profile

An agent could drive an optimizer step by step (ask for a population, evaluate
it elsewhere, tell the values). The job API does not offer this, on purpose:

- the objective would live outside the worker, so it could not re-check an
  answer: every claim would stay `claimed`, and a seed would no longer
  reproduce a run;
- a job would become a session with optimizer state between calls, which the
  runner would have to checkpoint and resume; jobs are recorded and re-run,
  never resumed;
- every generation would cost a round trip through the runner.

In-process ask/tell remains available in `fcmaes-core` (`Cmaes::ask`,
`Crfmnes::ask_pop`, `Mode::ask` and their `tell`) and in the Python bindings,
and service-style ask/tell tools (Optuna, Vizier, Ax) serve that interaction.
For the job API, register the model; to check candidates from elsewhere, use
`fcmaes.evaluate/1`.
