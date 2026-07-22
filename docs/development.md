# Development guide

## Required checks

Run formatting, linting, and tests from the repository root:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Exercise the optional PyO3 API through an installed extension:

```bash
python -m venv .venv
.venv/bin/python -m pip install "maturin[patchelf]>=1.7,<2" numpy scipy pytest
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/maturin develop --release \
  --manifest-path crates/fcmaes-py/Cargo.toml
.venv/bin/python -m pytest crates/fcmaes-py/python_tests
```

SciPy is needed only by the retry binding tests, where the extension constructs
the public `scipy.optimize.Bounds` object passed to an optimizer callback.

Run `git diff --check` before handing off changes.

## Source map

| Area | Source |
|---|---|
| Public Rust exports | `crates/fcmaes-core/src/lib.rs` |
| Objective and bounds machinery | `crates/fcmaes-core/src/fitness.rs` |
| Optimizers | `crates/fcmaes-core/src/{de,cmaes,crfmnes,pgpe,da,biteopt,mode}.rs` |
| Retry coordinators | `crates/fcmaes-core/src/{retry,moretry}.rs` |
| Quality diversity | `crates/fcmaes-core/src/mapelites.rs` |
| Python registration | `crates/fcmaes-py/src/lib.rs` |
| Optional PyO3 surface | `crates/fcmaes-py/src/` |
| GTOP implementation | `examples/src/gtop.rs` |
| GTOP names and bounds | `examples/src/problems.rs` |
| CLI and DE→CMA runner | `examples/src/runner.rs` |
| Native binary entry points | `examples/src/bin/` |
| Mazda adapter and objectives | `examples/src/mazda.rs` |
| Trading model and Yahoo cache | `examples/src/trading.rs` |
| Factory material-flow objective | `examples/src/material_flow_planning.rs` |
| Job-shop and harvesting objectives | `examples/src/{jobshop,harvesting}.rs` |
| Harmonic and transfer-scheduling objectives | `examples/src/{tdesign,scheduling}.rs` |
| ODE/control objectives | `examples/src/{damp,f8,lotka,integration}.rs` |

## Generate rustdoc

```bash
cargo doc --workspace --no-deps
```

Add rustdoc comments to public types and methods when extending the API. Use
intra-doc links such as ``[`De::optimize`]`` where the target is in the same
crate. Keep this directory focused on workflows and architecture rather than
duplicating every generated signature.

## Testing strategy

Optimizer parity is statistical, not bit-exact across languages. Fixed-seed
Rust tests should be deterministic, while cross-language acceptance should
compare distributions, success rates, and budget use.

The test layers are:

- Core unit tests beside each Rust module.
- GTOP reference and helper tests in `fcmaes-examples`.
- PyO3 compilation as part of the workspace build and Python-level integration
  tests under `crates/fcmaes-py/python_tests/`.
- Reproducible native performance workloads in `benchmarks/`.

When adding a public parameter, test its default, a non-default path, invalid
input, result accounting, and stop behavior. Ask/tell interfaces also need
call-order and batch-length tests.

## Coverage

If `cargo-llvm-cov` is installed:

```bash
cargo llvm-cov --workspace --all-targets --summary-only
```

Coverage is a diagnostic, not a substitute for convergence and parity tests.
Optimization code needs tests that exercise update behavior and result quality,
not only line execution.

## Adding an optimizer

1. Implement numerics in `fcmaes-core` without Python dependencies.
2. Define explicit parameter and result structs and re-export them from
   `fcmaes-core/src/lib.rs`.
3. Add one-shot convergence tests and stateful contract tests when applicable.
4. Add an optional PyO3 wrapper only after the core API is stable.
5. Register that binding in `crates/fcmaes-py/src/lib.rs`.
6. Add deterministic convergence and API-contract evidence.
7. Update rustdoc and these guides.

## Adding a GTOP problem or CLI option

Add objective numerics to `gtop.rs`, then add a named `Problem` with validated
bounds in `problems.rs`. Add name aliases to `by_name`, reference-value tests,
and catalog coverage. Shared CLI options belong in `runner::Cli`; both binaries
should continue to use the same parser.

Progress output belongs on stderr so stdout remains usable by benchmark and
automation code.

## Performance checks

- Always build with `--release` before timing.
- Exclude compilation from timed samples.
- Report workers, retries, configured budgets, and actual evaluations.
- Distinguish retry threads from optimizer population threads.
- Match optimizer configuration, evaluation budgets, retries, workers, stop
  conditions, and input data before comparing two runs.
- Store raw samples next to the benchmark report.
