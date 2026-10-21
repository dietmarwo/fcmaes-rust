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
# The same revision as fcmaes-rust's root Cargo.toml pins; another source or
# revision would be a second, incompatible copy of the crate.
optimization-job-api = { git = "https://github.com/dietmarwo/optimization-jobs", rev = "069d87c29170c8a5a1f4641028811df1a2de0585" }
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
| `registry.scalar(id, version, about, factory)` → `Scalar::new(f)` or `Scalar::simulation(f)` | one minimized value; a simulation may fail or be cancelled | `fcmaes.retry/1`, `fcmaes.surrogate/1` |
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
`id`, `version`, `kind`, `profile`, `profiles` and `parameters_schema` the
registry fills in. Describe there what an agent needs to write a job:
dimensions, parameters, inputs, the meaning of the variables. Keep it short:
the runner reads at most 1 MiB of `describe` output in `workers add` (64 KiB
before optimization-jobs `069d87c`; keep a worker under that if it must
register with older runners). With its payload schemas and examples the
stock worker's `describe` has about 39 KB; another model adds roughly 0.5 KB
plus its `about` (the curve fit adds 1.7 KB, its example's data included), so
about 50 models of this size fit next to the benchmarks under 64 KiB, and
well over a thousand under 1 MiB. A worker that does not
need the benchmarks starts from `Registry::new()`.

## Discovery: payload schemas and examples

An agent that only sees `describe` (MCP `describe_worker`) can build and
check a job from it: `payload_schemas` holds a JSON Schema of each profile's
payload with this worker's model identifiers, and `examples` holds runnable
jobs. `profiles` lists only the profiles the registered models support, so a
worker with scalar models only offers retry, evaluate and surrogate.

`describe` builds every model once for an example problem. Without an
`Example`, it tries dimensions 2, 3, 4, 6, 1, 5, 8 and 10 with no parameters
and no inputs, on the model's domain or `[-5, 5]` in every coordinate, and
takes the first the factory accepts. A factory that panics on a dimension is
caught. What the factory read becomes the model's `parameters_schema` (each
parameter's JSON type, the default it was read with, the `required` ones);
a parameter read only for other parameter values does not appear. Each
profile gets an example on the first model, by identifier, that needs no
data inputs, and every model whose `Example` has inputs or parameters gets
one in its own `profile`. Examples are small (about a second on one slot).

Register an `Example` when the probe cannot build the model or would build a
meaningless job: the model reads data inputs, needs a parameter without a
default, needs bounds other than its domain, or is a multi-objective model
without a reference point of its own:

```rust
use fcmaes_job_worker::registry::Example;

registry.example(
    "app.exponential-fit",
    Example::new(3)
        .bounds(vec![0.0, 0.0, -10.0], vec![10.0, 5.0, 10.0])
        .parameter("loss", "squared")
        .input_text("data", "decay.csv", include_str!("decay.csv")),
);
```

Its inputs appear as `content_text` in the example; keep them small, they
count against that limit. `Example::reference_point` gives `fcmaes.mode/1`
examples a reference point. A model without an example that `describe`
cannot build has neither an example nor a `parameters_schema`; jobs on it
still work. Since `describe` runs every factory, keep factories quick: `workers
add` gives `describe` 10 seconds. `crates/fcmaes-job-worker/tests/discovery.rs`
turns every example into a job and runs it; `tutorials/job-api/run.py`
validates them against the schemas with `jsonschema`.

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
- **Constraints.** A scalar model declares none: `fcmaes.evaluate/1` claims
  `feasible` for any usable value inside the bounds, and says in
  `details.constraints` that penalties inside the objective are not
  checked. To have constraints checked, register a multi-objective model
  that returns the raw objective and the raw constraint values
  (`MultiObjective::new(1, k, ...)`, feasible at `g <= 0`) next to the
  penalized scalar one, and evaluate candidates on it.

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

## Cancellable simulation callbacks (experimental)

`Scalar::simulation` adds a fallible callback with `SimContext` cancellation
checkpoints, remaining time, per-evaluation thread allowance and a stable
run-and-point seed. `fcmaes.surrogate/1`, `fcmaes.retry/1` and
`fcmaes.evaluate/1` accept it; `about: {"simulation": true}` makes the
surrogate the model's default `profile`.

- **Retry.** `threads()` is 1, and the seed derives from the retry's
  `details.run_seed` and the point, so the recheck of the best point sees
  the same seed. A failed evaluation counts as `+inf`, which every optimizer
  ranks worst, and is counted (`work.failed_evaluations`,
  `details.failed_evaluations`, up to three messages in `details.failures`).
  A failure of the recheck refutes the claim (`objective_mismatch`). A
  cancel or the deadline stops the simulation at its next checkpoint; the
  retry's remaining evaluations then return `+inf` without calling the model
  (`cancelled_evaluations`; `evaluations` counts completed calls,
  `evaluation_attempts` all), its termination is `cancelled` and its best
  point keeps `best_found` with `claimed` evidence (`details.recheck`). A
  population that fails entirely is a flat plateau: CMA-ES widens its step,
  PGPE only drifts; start it with a `guess` where evaluations succeed.
- **Evaluate.** `threads()` is 1 and the seed derives from `run_seed`
  (default 0) and the point; a retry's `details.run_seed` reproduces its
  values. A failed point has claim `none` and `details.error`, a cancelled
  one termination `cancelled`.

Use `Scalar::new` for existing plain callbacks. Direct Scalar struct
literals must migrate to constructors. See [the surrogate API](surrogate.md),
`examples/simulation_worker.rs` and
[the RK4 tutorial](../../tutorials/surrogate/README.md).
