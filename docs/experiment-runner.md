# Native experiment runner (source release 0.1)

`fcmaes-run` runs native optimization experiments as numbered, reproducible
jobs that a person or a shell-capable agent can launch, inspect, stop and
compare. The objective and the optimizer run together in a small native
worker; the CLI coordinates jobs through per-job files. There is no daemon,
database or RPC server. Numerical algorithms in `fcmaes-core` are unchanged.

Status:

- The foreground runner is covered by the integration tests in
  `crates/fcmaes-runner/tests` and `crates/fcmaes-native-worker/tests`.
- The detached systemd backend (`submit`) is implemented but has **not** been
  accepted on a host with a systemd user manager yet. Run the manual checks at
  the end of this page before relying on unattended detached jobs.
- No 16-core performance measurement has been made. Correctness with 16
  retry workers says nothing about scaling.

## Build and first experiment

Linux and a Rust toolchain supporting edition 2024 (minimum 1.88) are
required. The three crates (`fcmaes-job-api`, `fcmaes-runner`,
`fcmaes-native-worker`) are source-only (`publish = false`). The worker
depends on `fcmaes-core` and `fcmaes-gtop`, never on `fcmaes-examples`.

From the repository root:

```bash
cargo build --release --locked -p fcmaes-runner -p fcmaes-native-worker
export FCMAES_RUN_ROOT="$PWD/experiments"
RUN="$PWD/target/release/fcmaes-run"
WORKER="$PWD/target/release/fcmaes-native-worker"

# Capacity is the number of retry workers this node may run at once.
# Use the machine's physical-core count; this example uses 2.
"$RUN" node init --capacity 2
"$RUN" workers add --id native --executable "$WORKER"
"$RUN" problems list
"$RUN" problems describe benchmark.rosenbrock
"$RUN" validate --spec docs/runner-jobs/sphere-bite.json
"$RUN" run --spec docs/runner-jobs/sphere-bite.json --output results/sphere-bite
"$RUN" result results/sphere-bite
"$RUN" run --spec docs/runner-jobs/sphere-de.json --output results/sphere-de
"$RUN" compare results/sphere-bite results/sphere-de
```

All commands print JSON on stdout (`--json` is accepted for agent workflows
and changes nothing). Diagnostics go to stderr. Runner errors exit 2 with
`error.code = "runner_error"`; malformed CLI syntax uses clap's usage error;
the worker exits 1 on errors. A successful CLI call can report a failed or
timed-out job, so automation must inspect `state` / `run.json` `status`, not
only the exit code.

## Job specification

Inputs use schema `fcmaes.job/0.1` (`crates/fcmaes-job-api/src/spec.rs`;
examples in `docs/runner-jobs/`). Every nested object rejects unknown fields.

- Problems: `benchmark.sphere`, `benchmark.rosenbrock` (dimension >= 2) and
  `gtop.cassini1` (dimension 6, bounds inside the canonical Cassini1 box).
- Optimizers: `biteopt` with `evaluation_limit_mode: "hard"`, and `de` with
  `"soft"`. `population` is optional; the resolved default is recorded.
- Budget: positive `retries` and `evaluations_per_retry`; the requested total
  is their checked product, never a separate, possibly conflicting input.
- `wall_time_seconds`: no new retry wave starts after this execution time.
  `grace_seconds` (optional, default 1, at most 86400): the worker is killed
  this long after the wall-time deadline; a wave still running then is lost.
- Workers, retry-wave size, retries and evaluations must be positive. Zero is
  never forwarded to the core, where it would mean "all CPUs".

Budget semantics, verified by the worker tests:

| Path | Limit | Actual evaluations per retry |
| --- | --- | --- |
| BiteOpt | hard | exactly `evaluations_per_retry` (or fewer on early stop) |
| DE, population P | soft | at most `evaluations_per_retry + 2P - 1` |

`validate` prints the requested total and the maximum actual total. DE with a
hard limit is rejected.

The spec digest recorded in journals and manifests covers the whole resolved
specification except `request_key`, so identical experiments submitted under
different keys share a digest.

## Seeds and determinism

The worker ignores the core retry coordinator's schedule-dependent `seed` and
`sdev`. Retry `i` uses `retry_run_seed(root_seed, i)`, uniform initialization
in the resolved bounds and solver `runid = 0`, across all waves. The worker
tests check that the per-retry results (x, y, evaluations) are identical for
1, 2 and 3 workers and different wave sizes, for both optimizers, on
Rosenbrock and Cassini1. A wall-time cut can still complete different
subsets of retries.

Each completed retry is serialized, checksummed, appended, flushed and
`sync_data()`-acknowledged before the optimizer closure returns. The normal
execution path has no shared per-evaluation counter.

## Registry, idempotency and executable identity

Registering a worker is a trusted local administrative action; only the
native worker protocol is accepted. Jobs cannot name executables, shell
snippets or environment variables. Every attempt stages a copy of the
registered binary, checks its SHA-256 and runs that copy. Rebuilding the
registered path requires registering it again.

A request key maps to a normalized spec plus worker digest. Repeating the
same key and content returns the existing job without executing anything;
different content under the same key is refused. A new run needs a new key.
The request is recorded durably before launch. The job ID is derived from the
request key, and the attempt ID currently equals the job ID.

## Files and recovery

| File | Meaning |
| --- | --- |
| `submitted.json` | Original request |
| `resolved.json` | Expanded bounds and resolved defaults |
| `provenance.json` | Attempt, worker SHA-256, build metadata, boot/unit identity |
| `worker` | Staged executable for this attempt |
| `job.json` | Lifecycle state, queue and execution timestamps |
| `run.json` | Schema-v1 scientific manifest (see `tutorials/RESULT_SCHEMA.md`) |
| `retries.jsonl` | Durable checksummed completed-retry records |
| `progress.json` | Completed-retry snapshot, rewritten after each wave |
| `cancel.request` | Stop request (`mode`: `graceful` or `immediate`) |
| `stdout.log`, `stderr.log` | Each capped at 1 MiB; excess output is discarded |
| `containment.unknown` | Group shutdown could not be verified; slots retained |
| `job.lock`, `worker.lock` | Held by the launcher and the worker for their lifetimes |

The journal keeps the route-archive v2 envelope fields and SHA-256. Retry
journals use envelope version 1 with semantic validation (duplicate IDs, spec
digest, bounds, budget). A malformed newline-terminated record is fatal; only
a malformed unterminated final record is recoverable. A valid final record
without a newline is kept and its newline restored.

```bash
"$RUN" jobs list
"$RUN" jobs get JOB_ID
"$RUN" jobs wait JOB_ID --timeout 30
"$RUN" jobs cancel JOB_ID          # finish the active wave, then stop
"$RUN" jobs cancel JOB_ID --now    # kill after 1 s; the active wave is lost
"$RUN" jobs reconcile
"$RUN" jobs repair JOB_ID
"$RUN" jobs artifact JOB_ID retries
```

**Stopping.** The worker checks `cancel.request` between retry waves.

- A graceful cancel, or the first SIGINT/SIGTERM/SIGHUP to a foreground
  `run`, lets the active wave finish. The job ends `cancelled` with exact
  accounting.
- `--now`, or a second signal, kills the worker group one second later.
- The wall-time deadline stops new waves; the worker is killed at the deadline
  plus `grace_seconds`.

A killed wave makes accounting incomplete: `actual_evaluations` is `null`, and
`runner.known_evaluations_lower_bound` counts the journaled retries. Optimizer
state of active retries is not checkpointed. The execution deadline starts
after slot admission and includes start-up; time spent waiting for slots does
not count, and there is no separate queue timeout.

**Lost launcher.** If a foreground launcher dies without cleaning up (SIGKILL,
crash), the worker keeps running until its current wave ends. `jobs get` then
reports `observed_state: "launcher_lost"`. Because the launcher and the
worker hold `job.lock` and `worker.lock` for their lifetimes, and the kernel
releases those locks when a process exits, the runner can prove both are
gone:

- slot reconciliation releases the reservation as soon as both locks are free;
- `jobs repair` (refused while either process lives) finalizes the job. If
  the worker finished on its own, its exact result is kept; otherwise the job
  becomes `interrupted` with unknown total evaluations and duration.

Corruption is never silently discarded. Limitation: if the launcher dies in
the milliseconds between starting the worker and the worker taking its lock,
the reservation can be released while the worker still starts.

## Detached Linux user services

```bash
"$RUN" node doctor
"$RUN" submit --spec docs/runner-jobs/rosenbrock-bite.json
"$RUN" jobs list
```

Use a new request key if the example already ran in the same registry. The
backend uses `systemd-run --user`, a per-attempt `.service`,
`KillMode=control-group`, `Restart=no`, an explicit working directory and a
small per-job launcher. Worker output never stays attached to the submitting
client. There is no `nohup` fallback: if the user manager is unavailable,
submission is refused, and an ambiguous start acknowledgement is retained for
inspection rather than relaunched.

The slot ledger is protected by one stable advisory lock and updated
atomically. One slot is one retry worker; BLAS/OpenMP thread counts are set
to 1. Reservations coordinate this registry's jobs, not unrelated processes.
A reservation is released on a changed boot ID, an inactive or failed unit
with an empty cgroup, or free launcher and worker locks. Unknown liveness
keeps the capacity reserved.

Acceptance checks on the target host:

1. Run `python3 scripts/runner/test_systemd.py` (after the release build). It
   runs three jobs under two slots (two run, one waits), repeats a submit
   (idempotent), lets the submitting clients exit, cancels one waiting and one
   active job, kills a launcher, then reconciles and repairs. Exit 77 means no
   user manager was found and nothing was verified.
2. If logout survival is required, enable user-manager lingering according to
   local policy (the runner changes no privileged settings). Submit a long job,
   close the SSH session, reconnect and check that the journal keeps growing.
3. Submit a long service, reboot, then inspect, reconcile and repair. Completed
   work must be preserved and nothing must resume.
4. Make the user manager unavailable and check that unresolved reservations
   stay allocated; restore it and reconcile.
5. Run on the intended local filesystem. The tests cover process crashes and
   an ENOSPC device, not power loss or storage failure.

## Comparison

`compare` reads `run.json` manifests (or job directories) and refuses arms
with different problems, bounds, objectives or formulations. Its output
starts with a per-arm `summary` (optimizer, limit mode, status, requested,
actual and lower-bound evaluations, completed retries, best value, elapsed
time), followed by the full manifests. `equal_requested_work`,
`equal_actual_work` and `equal_wall_time_budget` state which comparison
protocol the arms satisfy. Warnings flag differing seeds, worker counts,
worker binaries, evaluation-limit modes and incomplete accounting.
Failed and zero-feasible arms are kept. No statistical winner and no
hypervolume comparison is inferred. Legacy tutorial manifests without the
runner profile are readable as descriptive arms only.

## Verification

```bash
cargo test --locked -p fcmaes-job-api -p fcmaes-native-worker -p fcmaes-runner
cargo clippy --locked -p fcmaes-job-api -p fcmaes-native-worker -p fcmaes-runner --all-targets -- -D warnings
```

The runner tests need the worker binary next to `fcmaes-run`; testing both
packages together (or `cargo test --workspace`, as CI does) builds it.

The real-binary integration suite uses only the Python standard library and
the release binaries; CI runs it after the Rust tests:

```bash
cargo build --release --locked -p fcmaes-runner -p fcmaes-native-worker
python3 scripts/runner/test_integration.py
```

It covers awkward budgets (1 to 101 evaluations per retry), determinism with
up to 16 workers, comparison of real tutorial manifests, journal corruption,
artifact path escape, a killed worker and unknown slot liveness.

`fcmaes-native-worker benchmark SPEC OUTPUT_DIR none|shared_atomic|journal`
is a fixed-work diagnostic for telemetry overhead. It does not enforce the
wall-time cutoff, and its `benchmark.json` is a timing report, not a
scientific `run.json`. `python3 scripts/runner/benchmark_matrix.py --output
DIR` pairs workers 1 to 16, Sphere and Cassini1, short and long retries and
the three telemetry modes; `--quick` is a smoke run only. Meaningful numbers
need jobs of at least 30 seconds, paired repetitions and the actual 16-core
target machine.

## Shared utilities in the route-search tutorial

The GTOC1 route-search tutorial delegates its JSONL, atomic-snapshot,
checksummed-archive and bounded-subprocess helpers to `fcmaes-job-api`, so
there is one implementation. It keeps `fcmaes-core = "=0.1.3"` from the
registry but takes `fcmaes-job-api` by repository path, so the tutorial now
builds only inside a repository checkout. Two error details changed: JSON
and checksum errors arrive as `RouteSearchError::Io`, and the subprocess
helper also stops descendants of an agent command after it exits.

## Deferred

- No stop token in `fcmaes-core`, CMA-ES or coordinated retry, Python
  objective loader, simulator plugin, distributed scheduler, auto-resume or MCP
  frontend. MCP stays behind the plan's named-client value gate.
- No registry publication.
