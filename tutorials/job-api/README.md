# Tutorial: durable fcmaes optimization jobs

This tutorial runs native BiteOpt searches for Sphere, Rosenbrock and Cassini1.
It uses the common job envelope and 16 independent retries over eight CPU slots.
The original `fcmaes.job/0.1` workflow remains available; see
[legacy experiment runner](../../docs/experiment-runner.md).

## Build and run the checked examples

From the repository root, on Linux with Rust 1.88+ and Python 3:

```sh
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
python3 tutorials/job-api/run.py --root /tmp/fcmaes-job-tutorial
```

The script initializes an eight-slot node, registers the executable, validates
and runs all three JSON files, then verifies the results. It prints the attempt
directories and writes `tutorial-report.json` under the node. Re-running it with
the same node and unmodified inputs returns the existing attempts.

The examples each request 16 × 1,000 optimizer evaluations. Every result has a
stable task/retry ID and seed, a checked candidate and `claim: "best_found"`.
This is not a global optimum certificate. The Sphere objective can also be
checked as `sum(x_i * x_i)`. Cassini1 uses the native GTOP model with its canonical
bounds; it does not invoke Python.

## Run one job manually

```sh
./target/release/optimization-run --root /tmp/fcmaes-manual node init --capacity 8
./target/release/optimization-run --root /tmp/fcmaes-manual workers add \
  --id fcmaes --executable ./target/release/fcmaes-job-worker
./target/release/optimization-run --root /tmp/fcmaes-manual problems describe fcmaes
./target/release/optimization-run --root /tmp/fcmaes-manual validate \
  --spec tutorials/job-api/sphere.json
./target/release/optimization-run --root /tmp/fcmaes-manual run \
  --spec tutorials/job-api/sphere.json
```

Read `run.json` in the returned `directory`, or use `optimization-run result
JOB_DIRECTORY`. `work.evaluations` sums optimizer work. The additional
`validation_evaluations` counter counts independent objective rechecks.

## Change the experiment

Copy `sphere.json`, choose a new `request_key`, and change the typed payload.
`problem` and `optimizer` reuse the legacy validated definitions. Bounds may be
broadcast from a single lower/upper pair; validation expands them. Keep
`inner_threads=1`: the parallelism here is across retries. Set
`outer_workers=1, cpu_slots=1` for a serial comparison with the same retry seeds.
Retry results are independent of scheduling and thread count for these models.

For DE, change `optimizer.kind` to `de` and `evaluation_limit_mode` to `soft`.
For population P, the declared allowance is at most `2P-1` additional evaluations
per retry. BiteOpt uses a hard limit. Unknown options and a mismatched limit
mode are rejected, not ignored. `evaluations_per_retry` and `retries` must be
positive. The job's wall limit bounds the entire batch after admission.

Cancellation is checked between retries. An active retry may continue until
its evaluation budget. The runner's process deadline remains authoritative;
forced termination preserves committed retries and marks total work unknown.

## Use existing jobs and extend the objective registry

```sh
cargo build --release --locked -p fcmaes-runner -p fcmaes-native-worker
./target/release/fcmaes-run --root /tmp/fcmaes-legacy node init --capacity 8
./target/release/fcmaes-run --root /tmp/fcmaes-legacy workers add \
  --id native --executable ./target/release/fcmaes-native-worker
./target/release/fcmaes-run --root /tmp/fcmaes-legacy run \
  --spec docs/runner-jobs/sphere-bite.json
```

The legacy manifest and `retries.jsonl` retain their original meaning. Both
workers evaluate objectives through one registry, so a new native objective
needs three changes: its validation and discovery entry in
`crates/optimization-job-api/src/legacy/spec.rs` (a shared file), its function
in `objective_for` in `crates/fcmaes-native-worker/src/lib.rs`, and tests for
bounds, budgets and deterministic seeds. Add a versioned profile if the
mathematical meaning of a job changes. Jobs select registered
models; they cannot deserialize a function or execute arbitrary source text.

See the [shared API guide](../../docs/job-api.md) for mixed-worker nodes,
lifecycle inspection, cancellation and recovery. Run `python3
scripts/job-api/test_integration.py` for local adapter acceptance, and
`python3 scripts/runner/test_integration.py` for legacy compatibility.
