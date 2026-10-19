# Shared optimization job API 0.1

The three projects share `crates/optimization-job-api` and
`crates/optimization-runner`. Each project builds its own native worker. One
runner can register all three executables and schedule them against one CPU
budget. Each repository also builds independently; no sibling checkout or
Python extension is needed.

Start with the [runnable tutorial](../tutorials/job-api/README.md). The
[implementation plan](job-api/PLAN.md) and [validation report](job-api/VALIDATION.md)
record the implementation and its acceptance gates.

## Architecture and compatibility

| Layer | Responsibility |
| --- | --- |
| `optimization-job-api` | Strict versioned envelope, resources, input hashes, checked journals, result contract and worker SDK |
| `optimization-runner` / `optimization-run` | Worker discovery, staging, idempotency, admission, process supervision, cancellation, deadlines and journal recovery |
| `fcmaes-job-worker` | `fcmaes.retry/1` (BiteOpt, DE or CMA-ES retries on registered scalar objectives), `fcmaes.mode/1` (MODE Pareto fronts) and `fcmaes.map-elites/1` (MAP-Elites archives): deterministic task seeds and exact or declared evaluation budgets |
| `lp-ip-job-worker` | `lp-ip.solve/1`: sparse LP/MIP artifacts, candidate starts, node/pivot budgets and numerical checks |
| `sqp-job-worker` | `sqp.solve/1`: registered smooth Rust models, initial points, active-set/Clarabel backends and independent KKT checks |

The shared crates and this guide are identical in the three repositories.
They are source packages, version 0.1.1, with publishing disabled; 0.1.1 is
the 0.1.0 delivery with the review fixes of all three repositories and the
fcmaes set and archive claims. A future
shared repository or published crate can replace these synchronized copies
without changing the job protocol. Solver dependencies remain in the individual
worker crates; the runner never links all three engines.

In fcmaes-rust, `fcmaes-job-api` re-exports the original public API from the
shared crate's `legacy` module. `fcmaes-run` is a compatibility entry point into
the common runner. The original `fcmaes-native-worker`, `fcmaes.job/0.1` inputs,
retry journals, evaluation accounting, comparison rules and benchmark command
remain supported. Register that worker as `native`. Register the new adapter
as `fcmaes` for the common envelope. These are separate profiles; old jobs are
not silently converted.

The new runner defaults to `.optimization-jobs`; `fcmaes-run` retains
`.fcmaes-jobs`. `--root` takes precedence over `OPTIMIZATION_RUN_ROOT`, then the
legacy `FCMAES_RUN_ROOT`. Always use the same absolute root when sharing a node.

## Build and discover

Linux and Rust 1.88 or newer are required for the job layer. Python 3 is needed
only for the tutorials and integration tests. The solver libraries retain their
own requirements. Use the repository's lockfile:

```sh
# Replace WORKER_PACKAGE with the worker in this repository.
cargo build --release --locked -p optimization-runner -p WORKER_PACKAGE
./target/release/optimization-run --root /tmp/optimization-node node init --capacity 8
./target/release/optimization-run --root /tmp/optimization-node workers add \
  --id WORKER_ID --executable /absolute/path/to/worker
./target/release/optimization-run --root /tmp/optimization-node workers list
./target/release/optimization-run --root /tmp/optimization-node problems describe WORKER_ID
```

`describe` on a worker prints its accepted profiles, registered models, engine
version, resource limits and recovery capabilities. Executables are trusted
local programs: registration is explicit, and jobs cannot supply executable
paths or shell commands. This is a local single-user runner, not a multi-tenant
sandbox or remote service.

## Envelope

Every common job has these fields; see each tutorial for a complete valid file:

| Field | Meaning |
| --- | --- |
| `schema_version` | Exactly `optimization.job/0.1` |
| `request_key` | Caller-chosen idempotency key, 1–200 bytes |
| `worker` | Registered worker ID |
| `payload_schema` | Versioned worker profile |
| `payload` | Strict solver-specific model/algorithm/budget options |
| `resources` | Positive `cpu_slots`, `outer_workers`, `inner_threads` |
| `limits` | Positive `wall_time_seconds`, nonnegative `grace_seconds` |
| `inputs` | Named relative files with mandatory SHA-256 hashes; empty for registered models |
| `task_count` | Normally omitted; the worker fills it during validation |

Unknown envelope and payload fields, unsupported profiles and unsupported thread
widths are errors. Validation resolves defaults through the selected worker;
it does not run an optimization. `validate` does not reserve slots. `run` and
`submit` also check the requested slots against the node capacity.

Input paths are relative to the submitted specification (`model.json` or
`./model.json`). Absolute paths, parent traversal, duplicate names and symlinks
escaping that directory are rejected. Each file is limited to 64 MiB, with at
most 16 inputs. The runner checks hashes, copies inputs into the attempt
directory and validates using a staged copy of the registered executable, in a
`.staging-*` directory under the node root; the root must allow execution,
`TMPDIR` need not. The LP worker checks the model hash again when it starts. The
model is loaded once per worker execution and shared immutably by its tasks.
Worker validation output is capped at 4 MiB. Validation prints the resolved
specification, so this also bounds the specification itself: about 600,000
one-dimensional SQP starts fit, not the 1,000,000 tasks the envelope allows.

The JSON parser enables exact f64 round trips, preserving small numerical
values and reproducible content digests across journal/manifest reads.

The normalized specification and staged executable SHA-256 determine an
attempt's identity. Repeating a key with the same identity returns the existing
job; it never starts a second attempt. A conflicting identity is rejected. A new
experiment needs a new key. The experimental `spec_digest` excludes the request
key but includes the model hashes, solver settings and resource budget. The
worker binary digest and original path are recorded separately in provenance.

## Eight CPU slots

The runner requires:

```text
outer_workers * inner_threads <= cpu_slots <= 1024
```

For eight independent solves or fcmaes runs, use `8 / 8 / 1`. For one LP or
SQP solve with up to eight numerical workers, use `8 / 1 / 8`. `8 / 2 / 4` is
also valid. `8 / 8 / 8` is rejected. fcmaes currently requires
`inner_threads=1` in all three profiles. LP-IP and SQP support 1–8 inner
workers.

Admission reserves `cpu_slots` for the whole attempt, not just the currently
busy tasks. Concurrent jobs on the same root cannot reserve more than the node
capacity. LP's component, packing-neighborhood and frontier workers are forced
to one; `Options::threads` controls its numerical kernels. SQP models and QP
backend instances are private to each task. The Clarabel engine itself remains
serial; an eight-thread allowance is not an eightfold speedup promise.

Slots describe numerical work, not the exact operating-system thread count.
Supervisor, I/O and cancellation-monitor threads also exist. This is cooperative
accounting for these adapters, not CPU affinity or a kernel cgroup quota.
Small kernels may stay serial. No performance improvement is claimed from
adding this API.

## Run, inspect and stop

```sh
optimization-run --root NODE validate --spec JOB.json
optimization-run --root NODE run --spec JOB.json
optimization-run --root NODE jobs list
optimization-run --root NODE jobs get JOB_ID
optimization-run --root NODE jobs events JOB_ID
optimization-run --root NODE jobs wait JOB_ID --timeout 60
optimization-run --root NODE result JOB_DIRECTORY
optimization-run --root NODE jobs artifact JOB_ID results
optimization-run --root NODE jobs cancel JOB_ID
optimization-run --root NODE jobs cancel JOB_ID --now
```

The examples use the executable name for brevity; use `./target/release/optimization-run`
unless it is on your PATH. `run` blocks, prints a final JSON job object, and
captures bounded worker logs. Its exit code reports runner/protocol errors;
a normal cancellation, timeout or solver failure can still return a job object.
Always inspect job `state`, then each mathematical outcome. With a request key
that already names a job, `run` and `submit` start nothing and print that job's
current state, which need not be final; use `jobs wait`. A signal during
validation also stops the validator.

Lifecycle states are `submitted`, `starting` (detached only), `waiting`,
`running`, then `completed`, `cancelled`, `timed_out`, `failed` or `interrupted`.
The wall clock starts after CPU admission; queue time is separate. LP and SQP
also receive the remaining cooperative time limit. A graceful stop sets a
shared cancellation token in LP/SQP; fcmaes finishes active retries, while its
MODE and MAP-Elites runs stop after the current generation and return their
front or archive. Workers stopped at the wall deadline may report
`termination: "cancelled"`; the job state says `timed_out`. `--now`
allows at most one further second before process-group termination. The wall
limit plus its declared grace always bounds process supervision.

A completed job says that its tasks returned normally. It does not assert an
optimal solution. An infeasible LP can be a completed job. SQP can complete with
a local restoration failure. A time/iteration limit can still retain a feasible
candidate. Deadline classification is a lifecycle decision, independent of a
solver's termination reason.

## Results and durable recovery

`results.jsonl` commits one checked envelope per completed task. Each envelope
has a schema version and SHA-256 checksum of the exact `payload_json` bytes.
The common worker serializes appends and syncs them before reporting completion.
`events.jsonl` records durable lifecycle transitions. Atomic JSON snapshots
include `job.json`, `resolved.json`, `provenance.json` and final `run.json`.
`worker-process.json` records the process group for conservative liveness checks.

The result schema is `optimization.result/0.1`. `results` are ordered by
`task_id` even when tasks finish in a different order. Each contains an outcome
and named work counters. Counters from different solver profiles must not be
added as if they measured the same work.

| Claim | Meaning |
| --- | --- |
| `none` | No accepted candidate claim; inspect termination/details |
| `feasible` | Independently checked candidate within declared tolerances |
| `best_found` | fcmaes candidate checked for objective/bounds; no global certificate |
| `non_dominated_set` | fcmaes MODE: `details.front` is re-evaluated, feasible and mutually non-dominated; no single `x` or `objective`; no Pareto-optimality certificate |
| `elite_archive` | fcmaes MAP-Elites: every elite in `details.elites` is re-evaluated and maps to its niche; `x`/`objective` are the best elite; no per-niche optimality certificate |
| `numerical_optimum` | LP/MIP native optimal status with independent candidate checks; numerical, not an exact certificate |
| `first_order` | SQP convergence plus independently recomputed original-unit KKT residuals; no global/second-order certificate |

LP details preserve objective sense, incumbent, best bound, gaps, warm-start
status and available LP residuals. Infinite gap values are explicit strings
`+infinity` / `-infinity`, not silently converted to JSON null. SQP details keep
KKT acceptance tolerances, QP backend and raw status. fcmaes MODE details carry
the front and its exact hypervolume for a declared reference point; MAP-Elites
details carry the elites, coverage and QD score. Native validation work is
counted separately from solver evaluations/iterations. A non-dominated set is
the only claim without a single point: its outcome has `feasible: true` and no
`x` or `objective`. Runners built before these two claims existed reject them
when replaying a journal; use one runner build per node.

If a worker is killed, active work is unknown. `work` is null,
`runner.accounting_complete` is false, and `known_work_lower_bound` covers only
committed tasks. A normal cooperative stop accounts for its returned tasks.
No failed or truncated attempt receives fabricated zero-cost totals.

```sh
optimization-run --root NODE jobs reconcile
optimization-run --root NODE jobs repair JOB_ID
```

Reconciliation only releases slots when process termination is established.
Repair refuses a live worker or unknown liveness. A launcher that dies while its
job is still queued never started a worker: repair then seals the attempt as
interrupted with zero work and complete accounting, without a liveness proof.
Detached `submitted`/`starting` attempts may still be started by their unit and
stay unknown. Repair tolerates a torn final journal fragment, preserves a
consistent final worker manifest if available, and otherwise seals the attempt
as interrupted. A journal with earlier corruption or duplicate task IDs is never
trusted: such an attempt, and any attempt whose worker wrote one, is sealed
`failed` with no results, `work` null and `runner.journal_error`. `jobs get` then
reports the journal error, and `jobs list` shows it for that job only. Appends
end a torn final fragment first (a valid record gets its newline, anything else
is dropped), so a later record never joins it. After launcher loss, elapsed time
may be unknown. Repair never resumes solver state. Use a new request key for a
fresh solve. If the launcher dies between process creation and durable
process-group recording, reservations are conservatively retained; reboot
reconciliation can clear the old boot. Only unfinished jobs are flagged after a
reboot.

`compare` supports the legacy fcmaes behavior. Common results require identical
resolved experiment digests and payload profiles and never select a winner
automatically. Compare mathematical claims and work counters explicitly when
changing algorithms or resource widths.

## Detached operation and current limits

`submit` uses a real Linux systemd user manager with `KillMode=control-group`
and `Restart=no`, in unit `optimization-job-ID.service` (`fcmaes-ID.service` for
legacy fcmaes jobs); run `node doctor` first. Logout survival requires host-specific
user-manager lingering. Foreground execution and the refusal without a systemd
user manager are tested; detached execution has not yet been accepted on a real
service host. There is no simulated background-success fallback.

This release has no HTTP/MCP server, remote simulator plugins, arbitrary Python
callbacks, distributed scheduler, solver checkpoint/resume or cross-solver
pipeline engine. SQP starts with three native registered models and a documented
extension seam. LP accepts the documented CSR JSON profile, not arbitrary MPS
or every native tuning option. Numerical cores and their established scope are
unchanged by this integration.

## Extending and testing

Implement `worker::Worker::prepare` to deserialize and validate a versioned
payload and resolve defaults. Return a `Prepared` object with task count,
cooperative cancellation and a task-local solve operation. A task must return
its numerical outcome and work counters. The SDK handles concurrency, journal
commits and final manifests. Do not put solver-specific assumptions in the
runner. Bump a profile version when changing mathematical input/result meaning.

```sh
cargo test --release --locked -p optimization-job-api -p optimization-runner
python3 scripts/job-api/test_integration.py
python3 scripts/job-api/test_review.py
# Add independently built workers to test a mixed node:
python3 scripts/job-api/test_integration.py \
  --fcmaes /path/to/fcmaes-job-worker \
  --lp-ip /path/to/lp-ip-job-worker --sqp /path/to/sqp-job-worker
```

The harness auto-detects the local worker and skips unavailable solver-specific
cases. `test_review.py` holds the runner regression tests of the 0.1.1 review; it
drives a protocol fixture (`fake_worker.py`) and the repository's own worker.
The 0.1 acceptance run supplied all three workers; see the
[validation report](job-api/VALIDATION.md). `job-api/SHARED-SOURCES.sha256`
records the content of the files shared by the three repositories, and each
repository's CI checks it. Change a shared file in all three repositories
together.
