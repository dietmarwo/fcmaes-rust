# Job API 0.1 validation

Tested on Linux x86_64 with Rust 1.99.1 (2026-10-13). The API layer declares
Rust 1.88 as its minimum; that exact toolchain was not separately tested.
All commands use the delivered lockfiles. Solver cores are unchanged from the
base commits recorded in `SOURCE-REVISION`.

| Gate | Observed result |
| --- | --- |
| Shared API/runner and native adapters | Release builds passed in all three independent workspaces |
| Job layer format / Clippy | Passed; Clippy used all targets and `-D warnings` |
| Shared contract/durability tests | 18 passed, including legacy validation and float round-trip regression |
| fcmaes selected Rust tests, release | 30 passed (includes shared contracts, facade and native worker tests) |
| Legacy fcmaes runner/worker tests, debug | 11 passed, including 6 real CLI lifecycle tests |
| Original fcmaes Python integration suite | 14 passed unchanged |
| Mixed integration with all three native workers | 10 passed, none skipped |
| LP-IP local integration | 8 passed; 2 tests for other solvers intentionally skipped |
| SQP local integration | 8 passed; 2 tests for other solvers intentionally skipped |
| LP-IP core/native/adapter Rust selection | 196 passed |
| SQP core/solver/QP/parallel/adapter Rust selection | 197 passed; 1 optional documentation census ignored |
| Tutorial executions | 9 examples completed; numerical/termination assertions passed |
| systemd detached acceptance | Not verified: no systemd user manager; explicit skip (exit 77) |

Counts above describe test invocations and overlap; they are not a claim of
unique test cases. Python extension suites were not re-run for unchanged
bindings. Build targets, toolchains and node executables are not bundled.

## Numerical and lifecycle coverage

The mixed harness uses deterministic retry seeds, independent objective/bound
checks, known LP/MIP optima, infeasible and unbounded LPs, analytic derivative
checks and independent SQP KKT acceptance for both QP backends. It compares
serial and eight-task layouts; LP/SQP also run with one task and eight inner
threads. These are correctness/resource checks, not performance benchmarks.

Lifecycle cases include strict schema rejection, unchanged-key idempotency,
conflicting keys, over-budget resource rejection, model checksum failure,
artifact escape, duplicate journals, queued cancellation under a shared
capacity of eight, forced timeout with unknown active-work totals, killed
launcher with a live worker, conservative slot retention and repair without
solver resume. Native snapshots and checked journals are retained for all
nine tutorial examples under `validation/job-api/tutorial-runs/` in their
respective repository archives.

## Fixes found during acceptance

The first LP assertion incorrectly required bit-exact equality between a
conservative bound and objective 7. It now verifies tolerance agreement and
that the bound remains on the conservative side; no solver values were changed.

The first fcmaes tutorial revealed that the default JSON parser could shift
small f64 values by one bit on a repeated parse, causing manifest validation to
reject a completed journal. The shared crate now enables `float_roundtrip`.
The regression test covers small/large exponents, stable spec digests and the
complete journal-to-manifest-to-runner path. All tutorial examples were rerun.

Extracting the legacy runner also required retaining its original test-only
dependencies. All-target compilation and the unchanged CLI tests verified that
compatibility surface, including the original live-worker repair diagnostic.

## Reproduce

Use the commands in `docs/job-api.md` and the repository's
`tutorials/job-api/README.md`. Each ZIP builds independently. To reproduce the
mixed test, build each repository's worker and pass all three absolute paths to
`scripts/job-api/test_integration.py`. The exact native test selections are:

```sh
# fcmaes-rust
cargo test --release --locked -p optimization-job-api -p optimization-runner \
  -p fcmaes-job-api -p fcmaes-job-worker -p fcmaes-native-worker
cargo test --locked -p fcmaes-runner -p fcmaes-native-worker
python3 scripts/runner/test_integration.py

# LP-IP-craft
cargo test --release --locked -p lp-ip-craft -p lp-ip-craft-native -p lp-ip-job-worker

# SQP-craft
cargo test --release --locked -p sqp-craft -p sqp-craft-solver \
  -p sqp-craft-activeset -p sqp-craft-clarabel -p sqp-craft-parallel -p sqp-job-worker
```

`validation/job-api/` contains raw logs, tutorial results, test environment and
binary hashes in the recorded attempt provenance. Those paths refer to the
validation host and are evidence, not portable execution configuration.
`docs/job-api/SHARED-SOURCES.sha256` verifies identical shared implementation
files across the three archives. `MANIFEST.sha256` covers every delivered file
except itself and can be checked with `sha256sum -c MANIFEST.sha256` after unzip.
