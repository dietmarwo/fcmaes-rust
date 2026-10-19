# Job API 0.1 validation

## Continuous checks in this repository

CI runs these on every push and pull request (`.github/workflows/ci.yml`):

| Check | Command |
| --- | --- |
| Rust 1.88 minimum version | `cargo check --workspace --locked` |
| Format, lint, docs | `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo doc` |
| Rust tests | `cargo test --workspace` (shared contracts and durability, worker SDK adapter, legacy and common CLI lifecycle) |
| Legacy runner integration | `python3 scripts/runner/test_integration.py` |
| Common job API integration | `python3 scripts/job-api/test_integration.py` |
| Runnable tutorial | `python3 tutorials/job-api/run.py` |
| Shared-source identity | `sha256sum -c docs/job-api/SHARED-SOURCES.sha256` |
| Route-search tutorial (uses the shared crate by path) | tutorial matrix job: fmt, clippy, tests |

The Rust tests include these cross-profile checks. Every BiteOpt or DE retry
of a common `fcmaes.retry/1` job equals a direct call of the shared native
solver from the same `retry_run_seed`, and every CMA-ES retry equals the
library's CMA-ES retry. A legacy `fcmaes.job/0.1` job and a common job with the
same problem, optimizer, seed and budget agree retry by retry, at different
worker counts. Every `fcmaes.mode/1` front equals the feasible non-dominated
part of a `fcmaes_core::Mode` run driven directly from the same seed, with the
same hypervolume; every `fcmaes.map-elites/1` archive equals the archive of
`fcmaes_core::map_elites_batch` from the same seed, for both variations.
Generation runs stop cooperatively at the deadline, on `cancel.request` and
through `jobs cancel`, with complete accounting.

The detached systemd backend is not exercised in CI: hosted runners have no
user manager. Run `python3 scripts/runner/test_systemd.py` on the target host.

## Delivery acceptance (18 October 2026)

The 0.1 delivery was accepted on Linux x86_64 with Rust 1.99.1, against
fcmaes-rust `c2703e7`, LP-IP-craft 0.28.0 (`af9a9a4`) and SQP-craft 0.16.0
(`7bc997f`). Its raw logs were not committed.

| Gate | Observed result |
| --- | --- |
| Shared API/runner and native adapters | Release builds passed in all three independent workspaces |
| Shared contract/durability tests | 18 passed, including legacy validation and float round-trip regression |
| Legacy fcmaes runner/worker tests | 11 passed, including 6 real CLI lifecycle tests |
| Original fcmaes Python integration suite | 14 passed unchanged |
| Mixed integration with all three native workers | 10 passed, none skipped |
| LP-IP / SQP local integration | 8 passed each; tests for other solvers skipped |
| LP-IP core/native/adapter Rust selection | 196 passed |
| SQP core/solver/QP/parallel/adapter Rust selection | 197 passed; 1 optional documentation census ignored |
| Tutorial executions | 9 examples completed; numerical/termination assertions passed |
| systemd detached acceptance | Not verified: no systemd user manager; explicit skip (exit 77) |

The mixed harness uses deterministic retry seeds, independent objective/bound
checks, known LP/MIP optima, infeasible and unbounded LPs, analytic derivative
checks and independent SQP KKT acceptance for both QP backends. It compares
serial and eight-task layouts; LP/SQP also run with one task and eight inner
threads. These are correctness/resource checks, not performance benchmarks.

Fixes found during acceptance:

- The default JSON parser could shift small f64 values by one bit on a
  repeated parse, so manifest validation rejected a completed journal. The
  shared crate now enables `float_roundtrip`; a regression test covers small
  and large exponents, stable spec digests and the journal-to-manifest path.
- An LP assertion required bit-exact equality between a conservative bound
  and the objective; it now checks tolerance agreement and the bound's side.

## Review fixes (fcmaes-rust)

- **Queued launcher loss.** A job whose launcher died while it waited for
  slots could never be finalized: it had no worker process record, so
  `jobs repair` refused it as "liveness unknown" indefinitely. Repair now
  recognizes an attempt that was never admitted (free locks, no process
  record, no reservation) and seals it as interrupted with zero work.
  Covered by `test_launcher_lost_while_queued_is_sealed_with_zero_work` in
  the shared harness. This changes `crates/optimization-runner/src/lib.rs`
  and `scripts/job-api/test_integration.py`, both shared files.
- The route-search tutorial's `Cargo.lock` lacked the new
  `optimization-job-api` dependency, so `--locked` builds of the tutorial
  failed.
- `fcmaes-job-worker` duplicated the native worker's solver dispatch and
  objective registry; both workers now use one library. Its reported engine
  version follows the workspace version instead of a hard-coded string.

## CMA-ES, MODE and MAP-Elites profiles (19 October 2026)

The common worker gained CMA-ES in `fcmaes.retry/1` and the new
`fcmaes.mode/1` and `fcmaes.map-elites/1` profiles. Checked locally with Rust
1.97 on Linux x86_64, in addition to the CI set above:

| Check | Observed result |
| --- | --- |
| Worker tests (`cargo test -p fcmaes-job-worker`) | 7 passed: discovery, validation and defaults of every path, equality with direct `fcmaes-core` runs, budgets, deadline and cancel stops |
| Shared contract tests | 7 passed, including the rules of the two new claims |
| CLI lifecycle (`fcmaes-runner`) | 9 passed, including a graceful `jobs cancel` of a MODE job |
| Shared harness | 10 passed, 2 skipped (LP/SQP workers not supplied), including the new black-box test of all three paths |
| Tutorial | 7 examples completed; Python rechecks of every front point, elite and candidate passed |

Observed on the tutorial examples: CMA-ES reached Rosenbrock-10 values
below 1e-14 in all 16 retries and stopped on its own after about 7,400
evaluations on average; MODE's ZDT1 fronts reached a hypervolume of about
0.866 (the true front has 0.8767 for reference point 1.1, 1.1); MAP-Elites
covered 60–80% of 400 niches of the 10-D Rastrigin projection benchmark.
These are checks of the adapters, not performance benchmarks.

An independent review of the change found three defects, now fixed and
tested. CMA-ES stores the unclamped decode of its best sample, which can lie
an ulp outside a bound, so a box-edge optimum failed the worker's bounds
check and the whole job; the worker now reports the clamped point, which is
exactly the evaluated one. The exact three-objective hypervolume costs
O(F³) after the last cancellation check, so MODE populations for three
objectives are limited to 1,000. A CMA-ES budget sum could overflow in
debug builds before its checked rejection.

## Reproduce with sibling workers

To run the mixed acceptance, build each repository's worker and pass the
absolute paths:

```sh
python3 scripts/job-api/test_integration.py \
  --fcmaes /path/to/fcmaes-job-worker \
  --lp-ip /path/to/lp-ip-job-worker --sqp /path/to/sqp-job-worker
```
