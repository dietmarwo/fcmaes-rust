# Experimental surrogate job profile

`fcmaes.surrogate/1` is an in-worker exact-GP optimizer in `fcmaes-job-worker`. It uses the existing `optimization.job/0.1` envelope and `optimization.result/0.2` evidence. It optimizes registered deterministic scalar Rust models, including `Scalar::simulation`. No callable code is carried in a job. It is a bounded first implementation, not the full async/TuRBO design in the supplied roadmap.

## Build and run

```sh
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
cargo build --release --locked -p fcmaes-job-worker --example simulation_worker
python3 tutorials/surrogate/run.py --root /tmp/fcmaes-surrogate-demo
```

Use a fresh root or distinct request keys for independent reruns; the runner supports idempotent request keys. The [tutorial](../../tutorials/surrogate/README.md) runs a benchmark, a warm job and an RK4 model through the real runner and checks the results. `optimization-mcp` describes, validates, submits, waits for and cancels the same profile using the existing tools, and reads its trace and snapshot artifacts with `get_artifact`; no new MCP session protocol is introduced. Discovery provides supported options and limitations.

## Current payload

```json
{
  "problem": {
    "id": "benchmark.branin", "version": "1", "config": {"dimension": 2},
    "bounds": {"lower": [-5.0, 0.0], "upper": [10.0, 15.0], "broadcast": false}
  },
  "seed": 41, "evaluations": 48, "runs": 1, "concurrent": 4,
  "evaluation_threads": 1, "log_scale": [false, false],
  "parameters": {}, "recheck_best": true
}
```

The problem shape is the existing registry shape, including optional model `parameters` and named hashed model inputs. Surrogate optimizer settings live under payload `parameters`, separately from `problem.parameters`. Unknown fields are rejected. This compact payload is implemented; the richer nested target schema in the supplied plan is not.

| Field | Meaning / bounds |
| --- | --- |
| evaluations | Required new optimization callbacks per run, 1–256. New plus retained warm observations must be <=256. Failed completed callbacks consume this budget. |
| seed | Required run seed; tasks derive independent seeds through the existing retry seed function. |
| runs | Independent tasks, default 1, max 1024. Warm inputs require runs=1. |
| concurrent | Synchronous logical batch size, default 1, max 8. |
| evaluation_threads | Allowance passed to simulation callbacks, default 1, positive. |
| log_scale | Per-coordinate flags; omitted/empty means all linear. Log bounds must be positive. |
| target | Optional finite minimization threshold, checked between batches (and on warm history). |
| recheck_best | Default true. One extra validation callback when time/cancellation permits. |
| warm_start | Optional `{"input":"previous"}` referencing one envelope input. |

Resources must satisfy `concurrent * evaluation_threads <= inner_threads <= 8` and the shared envelope's `outer_workers * inner_threads <= cpu_slots`. Example: four simulations with allowance two on an eight-thread task. The simulation must obey its allowance and avoid separate unreserved BLAS/thread pools. Nested Rayon work shares the aggregate task pool; per-evaluation allowances do not create disjoint pools. GP work happens between batches, on that same pool.

| Parameter | Default | Accepted range/options |
| --- | --- | --- |
| kernel | matern52 | matern52, rbf |
| acquisition | log_ei | log_ei, lcb (minimization) |
| output_transform | standardize | standardize, bilog, auto |
| initial_points | 8 | 1–128 |
| candidates | 256 | 32–4096 |
| fit_restarts | 2 | 1–4 |
| fit_iterations | 50 | 1–200 bounded L-BFGS iterations per restart (analytic gradient of the MAP objective in log lengthscales) |
| refit_interval | 3 | 1–64 proposal rounds |
| lcb_beta | 2 | finite 0–20 |
| minimum_distance | 1e-6 | unit-coordinate separation, 1e-12–0.01 |
| nugget | 1e-10 | 1e-14–1e-2; kernel diagonal in standardized output units. Raise it (e.g. 1e-6) for rough or numerically noisy simulations; inferred noise is not implemented. |
| trust_region | true | Two of every three proposals maximize the acquisition inside a TuRBO-1-style trust region around the incumbent (side 0.8 at start, ×2 after three successes up to 1.6, ÷2 after `ceil(max(4, d)/concurrent)` failures, reset below 2^-7, shaped by the lengthscales); every third searches the whole box. `false` searches the whole box always. |
| failure_threshold | 1e99 | positive finite; defaults to 1e10 for gtop.* models |

Auto output transformation uses bilog when nonzero observed absolute values span over six decades; it is not an algorithm selector. Configuration for noise, async, multiple trust regions, handoff, rank transforms, Sobol and Thompson sampling is unsupported and rejected.

## Register a cancellable Rust model

```rust
use fcmaes_job_worker::{Adapter, registry::{Registry, Scalar}};
use serde_json::json;

let mut registry = Registry::benchmarks();
registry.scalar("app.simulation", "1", json!({"simulation": true}), |model| {
    model.require_dimension(2..=2)?;
    Ok(Scalar::simulation(|x, ctx| {
        ctx.checkpoint()?;
        // In a real integrator, repeat checkpoint() between time steps.
        // ctx.remaining_seconds(): current time remaining
        // ctx.threads(): evaluation allowance
        // ctx.seed(): stable run-and-point seed
        Ok(x.iter().map(|v| v * v).sum())
    }))
});
// optimization_job_api::worker::run(Adapter::new(registry));
```

Factories must deterministically consume/check their model parameters and inputs, as in the existing registry. Discovery metadata `simulation: true` makes the surrogate the model's default `profile`; retry and evaluate accept simulation models too. `Scalar::new` callbacks retain their behavior. Direct external `Scalar { ... }` struct literals require migration to constructors because the new simulation storage is private.

Retry and evaluate accept simulation callbacks as well: a failed evaluation counts as `+inf` in a retry and as claim `none` in evaluate ([application models](models.md#cancellable-simulation-callbacks-experimental)). Cancellation is cooperative: callbacks must reach a checkpoint, or a plain callback must finish. The runner ultimately enforces process wall limits. A simulation `Err("cancelled")` while stop is active is partial cancelled work; other errors are failed completed evaluations. A point seed makes randomized evaluation repeatable, without implementing statistical noisy optimization.

## Files, warm starts and replay

Every task writes `surrogate-TASK.jsonl` and `surrogate-observations-TASK.json` in its attempt directory (task zero in the tutorial). The trace uses shared checksummed framing version 1 with Header/Event/Recheck records. Started/completed/cancelled events include IDs; completed observations are fsynced before waiting for stragglers. Snapshot updates are atomic. Write failures fail closed.

Both files are declared [artifacts](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#artifacts) of the job: `surrogate-TASK` (format `records`, the trace) and `surrogate-observations-TASK` (format `json`, the snapshot); `details.artifacts` names them for each task. While the job runs, `jobs get` (MCP `get_job`) lists them in `artifact_index`, and `jobs artifact JOB_ID surrogate-TASK --offset N --limit M` (MCP `get_artifact`) pages through the complete, checksum-verified trace records, so an agent can follow a long run. The sealed result records each file's size and SHA-256 in `runner.artifact_index`.

To warm start, name the snapshot of the earlier job as an envelope input with the SHA-256 from `runner.artifact_index`, and reference its input name in `warm_start`. Over MCP, pass the earlier job's `directory` as `input_dir` (optimization-jobs is adding a `put_input` tool that stages such a file); with the CLI, copy the file beside the new job file. The worker's `describe` has the snapshot's JSON Schema under `input_schemas` (`fcmaes.surrogate.observations/1`). The tutorial performs this exactly. A matching identity includes model id/version/dimension, resolved parameters, model data hashes and, for simulation models, the run seed passed to `SimContext` (`retry_run_seed(seed, task)`). A snapshot from task k of a multi-run job therefore only matches a new single-run job whose task-0 run seed is the same. Incorrect hashes, identity, trust flags, malformed records and duplicate IDs are rejected. New bounds drop out-of-box points and report a count. Warm values are historical user-supplied evidence, not independently certified. Changing model code requires a new model version.

A warm job starts new optimization state from the observations and only spends its new callback budget. It is not an exact optimizer checkpoint resume. Full traces are not accepted as warm inputs. Library `driver::replay` accepts decoded Event records and verifies proposals, state and final counters; use the shared `storage::read_checked` first for worker files. Semantic replay requires a complete Finished event. Checksum verification and decision replay do not re-evaluate the objective.

## Results and accounting

| Counter/detail | Meaning |
| --- | --- |
| work.evaluations | New completed optimization callbacks, including failures; excludes warm values and final validation. |
| work.failed_evaluations | Subset of completed callbacks with errors/panics/nonfinite/threshold values. |
| work.cancelled_evaluations | Started optimization calls stopped cooperatively; not completed observations. |
| work.evaluation_attempts | Completed plus cooperatively cancelled optimization calls. |
| work.validation_evaluations | Extra best-point validation calls started (0 or 1 per run). |
| work.cancelled_validation_evaluations | Validation calls stopped cooperatively. |
| work.model_fits | Completed hyperparameter fit operations. |
| details.fit_evaluations | Kernel-matrix factorizations by the MAP fit (accepted L-BFGS points also compute a gradient), not simulator calls. |
| details.proposal_seconds | Time in successful proposal rounds. |
| details.evaluation_seconds_sum | Sum of optimization callback elapsed wall time, including cancelled partial calls; not process CPU time and not job wall time. |

`best_found` always refers to an observed usable point, not a posterior minimizer. An exact independent deterministic recheck provides `rechecked` objective/bounds evidence; disabled, skipped or cancelled validation leaves `claimed` evidence. A completed mismatch refutes `best_found`, reports a fresh feasible value if available, and marks the snapshot untrusted. A validation cancellation does not refute an objective. No outcome certifies a global optimum. Common runner accounting can remain incomplete after a hard-killed task; do not infer complete common accounting from a surviving snapshot.

See [implementation log](../surrogate/IMPLEMENTATION_LOG.md) and [reviewed roadmap](../surrogate/PLAN.md) for exact tests and remaining gates.
