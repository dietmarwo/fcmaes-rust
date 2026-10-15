# Native experiment runner: source release 0.1

This implementation follows the revised 15 October 2026 plan against
`3ef267342c01b5ba4dee60d81501884b7e1807c0`. Numerical algorithms are unchanged.
The optimizer and objective execute together in a small native worker.
The CLI uses per-job files; it has no daemon, database or RPC server.

The foreground release is implemented and integration-tested. The systemd
backend is implemented but **not host-accepted in this environment**, which
has no systemd user manager. Run the supplied host gate before relying on
unattended detached jobs. The target 16-physical-core performance gate remains
open; correctness with 16 retry workers does not establish scaling.

## Build and first experiment

Linux, a Rust toolchain supporting edition 2024 (manifest minimum 1.88), and
Python 3 for the integration scripts are required. This delivery was compiled
with Rust 1.99.1; the minimum version was not separately tested. The three new
crates are source-only and deliberately set `publish = false`. The worker
uses `fcmaes-core` and `fcmaes-gtop`, never `fcmaes-examples`.

From the repository root:

```bash
cargo build --release --locked -p fcmaes-runner -p fcmaes-native-worker
export FCMAES_RUN_ROOT="$PWD/experiments"
RUN="$PWD/target/release/fcmaes-run"
WORKER="$PWD/target/release/fcmaes-native-worker"

# Choose physical-core capacity explicitly for this machine. This example uses 2.
"$RUN" node init --capacity 2
"$RUN" workers add --id native --executable "$WORKER"
"$RUN" problems list --json
"$RUN" problems describe benchmark.rosenbrock --json
"$RUN" validate --spec docs/runner-jobs/sphere-bite.json --json
"$RUN" run --spec docs/runner-jobs/sphere-bite.json --output results/sphere-bite
"$RUN" result results/sphere-bite --json
"$RUN" run --spec docs/runner-jobs/rosenbrock-bite.json --output results/rosenbrock-bite
"$RUN" run --spec docs/runner-jobs/cassini1-bite.json --output results/cassini1-bite
```

All commands print JSON on stdout (the `--json` flag is accepted for agent
workflows). Diagnostics go to stderr. Command/data errors exit 2 and contain
`error.code = "runner_error"`; malformed CLI syntax uses clap's usage error.
The worker exits 1 on errors. A successfully executed CLI operation may report
a failed/timed-out job: automation must inspect `job.state` / `run.status`,
not just the CLI exit code.

To test DE, copy the Sphere example, change `request_key`, set
`optimizer.kind` to `de`, `parameters.population` to e.g. 31, and
`budget.evaluation_limit_mode` to `soft`. The validator displays the maximum
actual evaluation allowance. DE with `hard` is rejected. Compare matching
Sphere results using:

```bash
"$RUN" compare results/sphere-bite/run.json results/sphere-de/run.json --json
```

Never compare Sphere with Rosenbrock as competing solutions to one problem.
The comparator refuses mismatched native problem/bounds/objective definitions.
It retains failed and zero-feasible arms, distinguishes requested and actual
work, and warns for differing seeds, workers, binary provenance or incomplete
legacy metadata. It makes no statistical-winner or hypervolume claim.
Existing route-search manifests without an `objectives` array remain readable
as descriptive legacy arms; their scientific definitions require domain review.

## Input and seed contract

Inputs use `fcmaes.job/0.1`; see the complete JSON examples and
`crates/fcmaes-job-api/src/spec.rs`. Every nested struct rejects unknown fields.
Only independent retry parallelism is supported. Workers, retry-wave size,
retry count and evaluations per retry must be positive; bounds must be finite,
strictly increasing and dimensionally consistent. Explicit scalar broadcast is
normalized to full bounds. The total budget is checked multiplication, never
an independent conflicting input. DE's full conservative bound is also checked
for overflow.

Supported objectives: Sphere, Rosenbrock (dimension >= 2), Cassini1 (dimension
6). Supported optimizer parameters: optional explicit population, with the
resolved default recorded. Other optimizer options and algorithms are rejected.
The native worker ignores the core retry coordinator's schedule-dependent
`sdev` and `seed`. It uses `retry_run_seed(root_seed, global_retry_id)`, uniform
initialization in the resolved bounds, and solver `runid = 0` across waves.
Sorted retry results are identical across the tested worker/wave configurations
on this build/platform. Wall-time cuts can complete different subsets.

Each retry completion is serialized, checksummed, appended, flushed and
`sync_data()`-acknowledged before the optimizer closure returns. No shared
per-evaluation atomic is used in the normal execution path. Progress counts
are maintained in memory after acknowledged append, avoiding an O(retries²)
journal replay during successive wave snapshots. Finalization validates the
complete journal once.

## Registry, idempotency and executable identity

Registering a worker is a trusted local administrative action. Jobs cannot
supply executable names, shell snippets or arbitrary environment variables.
The initial registry accepts the verified native capability profile only.
Every attempt stages a copy of the registered binary, checks its SHA-256, and
runs that copy. Rebuilding the registered path requires explicit re-registration.

A request key maps to a normalized spec plus worker digest. Repeating the same
key/content returns the existing attempt without execution. Different content
under the same key fails. A new scientific run needs a new request key. A
request is durably recorded before launch. Lost launch acknowledgement leaves
the recorded unit in `starting`; repeating submit does not blindly relaunch it.
The current attempt ID equals the job ID. Explicit new attempts use new keys;
continuation of missing retry IDs is intentionally not an implicit feature.

## Files and recovery

| File | Meaning |
| --- | --- |
| `submitted.json` | Original request |
| `resolved.json` | Expanded bounds and resolved defaults |
| `provenance.json` | Attempt, worker SHA-256, build metadata, boot/unit identity |
| `worker` | Staged executable for the immutable attempt |
| `job.json` | Atomic lifecycle state, queue and execution timestamps |
| `run.json` | Canonical schema-v1 scientific manifest |
| `retries.jsonl` | Durable checksummed completed-retry payloads |
| `progress.json` | Atomic completed-retry projection |
| `cancel.request` | Cooperative stop flag |
| `stdout.log`, `stderr.log` | Each capped at 1 MiB; excess output drained/discarded |
| `containment.unknown` | Group shutdown could not be verified; slots retained |
| `job.lock`, `worker.lock` | Stable lock inodes; never unlink while active |

The generic journal retains the route archive v2 envelope fields and SHA-256
algorithm. Retry journals use envelope version 1, a distinct typed payload,
and semantic validation including duplicate IDs, spec digest, vector/budget
validity and full-record checksum. Malformed newline-terminated records are
fatal. Only a malformed unterminated final tail is recoverable. A valid final
record without a newline is retained and repaired with a newline.

`jobs get` is read-only and replays verified records even without `run.json`.
`jobs artifact ID retries` resolves registered artifact names, rejects symlink
or path escape, and refuses inline content larger than 1 MiB. For larger logs,
use the reported job directory.

```bash
"$RUN" jobs list
"$RUN" jobs get JOB_ID
"$RUN" jobs wait JOB_ID --timeout 30
"$RUN" jobs cancel JOB_ID
"$RUN" jobs reconcile
"$RUN" jobs repair JOB_ID
```

Cancellation is a request until the worker actually stops. It is checked
between waves. The launcher permits a fixed 1-second grace period, then kills
the process group, reaps the child and verifies group termination. The active
wall deadline starts after slot admission, including startup/initialization.
Queue waiting does not consume this execution deadline; there is currently no
separate queue timeout. A timeout can therefore take the configured execution
time plus grace and cleanup/finalization time.

Normal cooperative termination records exact evaluations; forced termination
records null actual evaluations and a lower bound from durable completions.
Core optimizer state in active retries is not checkpointed. `jobs repair`
requires exclusive launcher and worker locks and proven released reservation;
it repairs an incomplete tail and publishes `interrupted`, with unknown total
evaluations/duration. Corruption is never silently discarded.

A stale foreground reservation on the same boot is deliberately retained when
liveness cannot be proven. Do not delete the lock file or edit a live ledger.
After a host reboot, old boot identities can be reconciled automatically.
For a same-boot foreground launcher crash, inspect and terminate the remaining
trusted processes before administrative recovery; this release has no unsafe
force-release command. Foreground mode does not promise disconnect survival.

## Detached Linux user services

```bash
"$RUN" node doctor
"$RUN" submit --spec docs/runner-jobs/sphere-bite.json
"$RUN" jobs list
python3 scripts/runner/test_systemd.py
```

Use a new request key if the example has already run in the same registry.
The backend uses `systemd-run --user`, a predictable per-attempt `.service`,
`KillMode=control-group`, `Restart=no`, an explicit working directory, and a
small per-job launcher. Native stdout/stderr never remain attached to the
submitting client. There is no `nohup` fallback. Manager failure refuses new
submission; ambiguous start acknowledgement is retained for inspection.

The slot ledger is protected by one stable advisory lock. Reservations are
atomic and synced. The lock covers admission/update only, not execution. One
slot represents one retry worker; BLAS/OpenMP thread counts are explicitly 1.
Reservations coordinate this registry's jobs, not unrelated host processes.
A stale reservation is released only on a changed boot identity or a proven
inactive/failed unit with empty cgroup. Unknown liveness retains capacity.
Worker-process verification accounts for nested PID namespaces. Foreground
process-group containment cannot catch intentionally escaped sessions; jobs
are trusted research programs, not hostile sandbox workloads.

Real-host gate (`test_systemd.py`): three jobs under two slots, two running and
one waiting; idempotent repeated submit; submitting clients exit; waiting and
active cancellation; launcher kill; cgroup reconciliation; explicit repair.
A missing user manager exits 77 with an explicit SKIP, never success.

Manual remaining acceptance checks:

1. Enable/verify user-manager lingering according to local host administration
   policy if OS-user logout survival is required. No privileged settings are
   changed by the runner. Submit a long job, close the actual SSH/user session,
   reconnect, and verify continuing journal entries.
2. Submit a long service, reboot, inspect the old boot record, reconcile and
   repair. Confirm old completed work is preserved and no execution resumes.
3. Deliberately make the manager unavailable and verify unresolved reservations
   stay allocated. Restore access and reconcile.
4. Run on the intended local filesystem. Fsync tests here cover process crashes
   and an ENOSPC device, not arbitrary power-loss/storage-failure guarantees.

## Verification and profiling

```bash
cargo test --locked -p fcmaes-job-api
cargo test --locked --manifest-path tutorials/gtoc1-route-search/Cargo.toml --lib route_
python3 scripts/runner/test_integration.py
cargo clippy -p fcmaes-job-api -p fcmaes-runner -p fcmaes-native-worker --no-deps -- -D warnings
python3 scripts/runner/benchmark_matrix.py --quick --output /tmp/fcmaes-smoke
python3 scripts/runner/benchmark_matrix.py --output /path/to/new/target-results
```

The matrix records CPU affinity/quota, toolchain, actual work, wall time and
throughput. It pairs workers 1/2/4/8/16, Sphere/Cassini1, short/long equal-work
retry campaigns, and no extra telemetry/shared atomic counter/durable retry
journal modes. It also measures direct versus managed worker launches. The
benchmark mode is an explicit fixed-work diagnostic, does not enforce the job
wall cutoff, and never pretends its `benchmark.json` timing report is a
scientific `run.json` result.

The shared counter is a focused coherence-cost diagnostic; the full examples
`ProgressState` incumbent-CAS path, a sharded alternative, expensive simulator
adapter, measured CPU time and detached timing arm remain additional profiling
work. Fixed-work measurements need jobs lasting at least 30 seconds, paired
repetitions and the actual 16-core target before applying the proposed <=2%
median overhead gate. Adjust budgets if the default runs are too short. Tiny
smoke timings are functionality evidence only.

## Scope deliberately deferred

No numerical-core stop-token changes, CMA/coordinated retry, Python objective
loader, simulator plugin, distributed scheduler, auto-resume, or MCP frontend.
MCP remains behind the plan's named-client/value gate, which has not been met.
Registry publication and registry-only tutorial resolution are not claimed.
The existing route tutorial keeps `fcmaes-core = "=0.1.3"` and imports only the
new shared utilities via an explicit repository path dependency.
