# Job API validation

## Continuous checks in this repository

CI runs these on every push and pull request (`.github/workflows/ci.yml`):

| Check | Command |
| --- | --- |
| Rust 1.88 minimum version | `cargo check --workspace --locked` |
| Format, lint, docs | `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo doc` |
| Rust tests | `cargo test --workspace` (shared contracts and durability, worker SDK adapter, legacy and common CLI lifecycle) |
| Legacy runner integration | `python3 scripts/runner/test_integration.py` |
| Common job API integration | `python3 scripts/job-api/test_integration.py` |
| Shared runner regressions, error codes and schema checks | `JOB_API_REQUIRE_SCHEMA=1 python3 scripts/job-api/test_review.py` |
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

## Synchronization of the shared crates as 0.1.1 (19 October 2026)

After the 0.1.0 delivery, each repository's review changed the shared files
independently: this repository fixed the queued-launcher repair, SQP-craft
fixed it differently and made several further runner and journal fixes, and
LP-IP-craft kept the delivered code. Version 0.1.1 unifies them. It takes
SQP-craft's API and runner changes, keeps this repository's zero-work repair of
a queued attempt with SQP-craft's rule that detached attempts with a unit stay
unknown, and adds the fcmaes set and archive claims. The same 20 files are
now in all three repositories, recorded by `SHARED-SOURCES.sha256`.

Checked here with Rust 1.97 on Linux x86_64: the workspace tests (290 passed),
the shared harness (10 passed, 2 skipped without LP/SQP workers), the review regressions
(7 passed, the reboot case with the fcmaes worker), the legacy runner suite
(14 passed), the tutorial and the route-search tutorial (53 tests). The
other two repositories ran their own suites; see their validation reports. The
shared harness with all three workers on one node passed (12 tests, none
skipped) under each repository's runner build, and every worker reports
`build.job_api_version: "0.1.1"`.

## Result contract 0.2 (19 October 2026)

The shared crates moved to 0.2.0 with the result contract 0.2: five new
claims, evidence (levels, checks, refutations, certificate slot), objective
bounds, worker protocol 0.2 with 0.1 still accepted, coded runner errors,
`already_terminal` on cancelling a finished job, and JSON Schemas for the job
envelope and the result manifest. The shared set grew to 23 files with the
two schemas and `scripts/job-api/outcome_cases.json`. CI installs `jsonschema`
and sets `JOB_API_REQUIRE_SCHEMA=1`, so the schema checks cannot be skipped
there.

An independent review before the merge reported 13 defects; all were fixed
and are covered by tests (see the [plan](PLAN.md#6-result-contract-02-october-2026)).
After the fixes, 1.2 million generated outcomes found no disagreement between
the JSON Schema and `Outcome::check`, and every case of `outcome_cases.json`
behaves the same as a journal record through the runner.

In this repository the fcmaes worker reports evidence for all three profiles,
records a retry whose re-check fails as a refutation instead of failing the
job, labels MODE fronts as `final_population` with an unnormalized
hypervolume, and adds `fcmaes.evaluate/1`. Checked with Rust 1.97 on Linux
x86_64:

| Check | Observed result |
| --- | --- |
| `cargo fmt --check`, Clippy `-D warnings`, `cargo doc` | Passed |
| Workspace tests (`cargo test --workspace`) | 298 passed, including 26 shared (5 legacy spec, 13 contracts, 8 durability) and 10 fcmaes worker tests |
| Legacy runner suite (`scripts/runner/test_integration.py`) | 14 passed |
| Shared harness | 13 passed, 2 skipped without LP/SQP workers |
| Shared review and schema tests (`JOB_API_REQUIRE_SCHEMA=1`) | 15 passed |
| Tutorial | 8 examples completed; results and jobs validated against the schemas |
| Route-search tutorial | 56 passed |
| Mixed node with all three workers, under each repository's runner build | 13 passed, none skipped, three times |
| `sha256sum -c docs/job-api/SHARED-SOURCES.sha256` | 23 files OK |

MSRV 1.88 is checked by CI; the toolchain could not be installed on the
review host.

## Application models, parallel populations and more algorithms (19 October 2026)

The fcmaes worker gains a model registry for application workers,
`inner_threads > 1` for its population-based optimizers, CR-FM-NES and PGPE
retries, IPOP/BIPOP restarts, CVT archives and CMA-ME/CMA-MAE emitter phases
([plan](PLAN.md#7-application-models-parallel-populations-and-more-algorithms-october-2026),
[application models](models.md)). Checked with Rust 1.97 on Linux x86_64:

| Check | Observed result |
| --- | --- |
| `cargo fmt --check`, Clippy `-D warnings`, mdBook link check | Passed |
| Workspace tests (`cargo test --workspace`) | 313 passed, including 24 fcmaes worker tests: 9 profile, 6 algorithm and 5 registry integration tests, 3 unit tests and a doctest |
| Legacy runner suite | 14 passed |
| Shared harness | 14 passed, 2 skipped without LP/SQP workers; 14 passed, none skipped, on a mixed node under each repository's runner build |
| Shared review and schema tests (`JOB_API_REQUIRE_SCHEMA=1`) | 15 passed |
| Tutorial | 12 examples completed, including an application worker reading a hashed data file; results and jobs validated against the schemas |
| The 8 tutorial jobs of `main` on `main`'s and this worker | Identical resolved payloads; identical results apart from the new `details.archive_kind` of MAP-Elites |
| `sha256sum -c docs/job-api/SHARED-SOURCES.sha256` | 23 files OK |

The tests run every population-based profile with one and with several
inner threads and require identical results; compare CR-FM-NES, PGPE and
CMA-ES restart retries, and grid and CVT archives, with direct library runs
from the same seeds; check that emitter phases spend exact budgets and only
add or improve elites; and exercise the registry's rejections (unknown or
mistyped parameters, missing, unused or altered inputs, versions, kinds,
domains, a model returning the wrong number of values).

An independent review before the merge found four defects, all fixed and
tested: building a CVT archive with `inner_threads=1` ran on Rayon's global
pool (every task now runs inside its own pool, also for one thread); a CVT
archive and the emitter phase had no memory bound, so a job that passed
validation could abort at execution (sample, descriptor and
`emitters × dimension²` limits, and `dimension ≤ 3162` and `population ×
dimension ≤ 10⁷` for the evolution strategies); and the default CMA-ME
emitter did not rank candidates that fill an empty niche first, as Fontaine
et al. do. With the paper's ranking, CMA-ME on the 20-dimensional sphere
projection (400 niches, 4 × 16 × 400 emitter evaluations) reaches 82–98%
coverage instead of 38–48%.

At equal budgets of 20,000 evaluations per run (four runs, 20 dimensions,
400 grid niches), coverage was:

| Emitter | Sphere projection | Rastrigin projection |
| --- | --- | --- |
| SBX MAP-Elites | 36% | 45% |
| Iso+LineDD MAP-Elites | 50% | 57% |
| 100 SBX generations, then CMA-ME | 66% | 58% |
| 100 SBX generations, then CMA-MAE (`threshold_init` 300 or 600) | 66% | 78% |

## Reproduce with sibling workers

To run the mixed acceptance, build each repository's worker and pass the
absolute paths:

```sh
python3 scripts/job-api/test_integration.py \
  --fcmaes /path/to/fcmaes-job-worker \
  --lp-ip /path/to/lp-ip-job-worker --sqp /path/to/sqp-job-worker
```
