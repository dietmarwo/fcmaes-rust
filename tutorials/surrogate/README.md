# Surrogate optimization through the shared job API

This executable tutorial runs three jobs: 48-call Branin minimization, an eight-new-call warm start, and a 32-call native Rust damped-oscillator calibration. It uses an eight-slot node. The oscillator has four concurrent evaluations, each with a two-thread allowance; nested work uses the same task pool.

## Prerequisites and commands

Use Rust >=1.88 and Python 3.10+; runtime scripts use only Python's standard library. Build from the repository root (Cargo fetches locked registry crates and the pinned optimization-jobs Git revision):

```sh
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
cargo build --release --locked -p fcmaes-job-worker --example simulation_worker
python3 tutorials/surrogate/run.py --root /tmp/surrogate-example
```

Use a fresh directory for an independent rerun. Request keys are intentionally stable, so an existing node may return earlier idempotent jobs. With a custom target directory, use `--bin-dir /absolute/target/release` (or export `CARGO_TARGET_DIR`). The script creates node capacity eight, registers the standard and example workers, validates/runs all jobs, independently checks Branin in Python, checks evidence/accounting, hashes the compact snapshot, and writes `tutorial-report.json` under the node root. `branin.json` and `oscillator.json` are complete editable jobs.

In the tested build the Branin result was about 0.3978908 (known optimum about 0.39788736), the warm job used exactly eight new optimization calls from 48 retained observations, and oscillator loss was about 2.42e-6. These are deterministic smoke results on this source, not a general performance guarantee. The optional final validation call is additional and separately counted.

## Inspect a job manually

```sh
target/release/fcmaes-job-worker describe
target/release/examples/simulation_worker describe
```

The report records each job directory. Use `optimization-run --root NODE result JOB_DIRECTORY` to inspect common claims, evidence, counts and details. In the attempt directory task zero writes `surrogate-0.jsonl` and `surrogate-observations-0.json`. These are local files; the existing MCP result tools do not provide paged access to them yet. The script shows the exact input-copy/hash steps needed for warm starts; never edit a snapshot and reuse its old SHA-256.

## Adapt the Rust simulation

Read [simulation_worker.rs](../../crates/fcmaes-job-worker/examples/simulation_worker.rs). It fits damping and stiffness to analytic reference observations by integrating an oscillator with RK4. Four independent replicas change initial amplitude. The model checks dimension and all parameters, and calls `ctx.checkpoint()` every 64 steps. Results collect in replica order before summing, so scheduling does not change reduction order.

- `steps`: default 2048, 64–1,000,000 and divisible by eight.
- `replicas`: default four, 1–64.
- `pause_ms`: default zero, 0–10 synthetic milliseconds per checkpoint. This helps exercise cancellation but is not a model of CPU-heavy simulation throughput.

Replace the integration body with your Rust model. Return `Err` for evaluation failures, use checkpoints in long loops, obey `ctx.threads()`, and derive reproducible simulation randomness from `ctx.seed()`. Do not start a new native pool on every evaluation. Bump the model version when its objective changes. Register factories on a deterministic registry; jobs carry model parameters and data hashes, not source code.

The first API accepts simulation models only in this profile. Global GP is limited to 32 coordinates and 256 new+retained observations. TuRBO, async queues, the overhead governor, handoff and noisy optimization remain roadmap work. The seeded simulation callback is repeatable; it is not an expected-value estimator.

## Run the actual runner/MCP tests

```sh
python3 scripts/surrogate/test_job_api.py
cargo test --release --locked -p fcmaes-surrogate -p fcmaes-job-worker
```

The Python test registers actual worker binaries, submits through MCP, checks the result, then starts a deliberately slow simulation and cancels it through the runner. It respects `CARGO_TARGET_DIR`. CI runs it and this tutorial after building both workers.

See [current API and evidence semantics](../../docs/job-api/surrogate.md), [benchmark reproduction](../../docs/surrogate/decisions.md), and [next milestones](../../docs/surrogate/PLAN.md).
