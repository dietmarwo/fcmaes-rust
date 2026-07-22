# Getting started

## Prerequisites

The native library requires a current stable Rust toolchain with Cargo. Python
is optional and needed only when building the PyO3 extension crate.

Check the toolchain:

```bash
rustc --version
cargo --version
```

## Build and test the Rust workspace

Run these commands from the repository root:

```bash
cargo build --workspace
cargo test --workspace
```

Use an optimized build for performance work and real optimization runs:

```bash
cargo build --release --workspace
```

The release profile enables thin LTO and one code-generation unit. Debug
builds are suitable for tests but are not representative for optimizer timing.

## Generate the API reference

The source contains rustdoc comments for public types and methods:

```bash
cargo doc --workspace --no-deps --open
```

Without `--open`, the main pages are generated below `target/doc/`, including
`target/doc/fcmaes_core/index.html` and
`target/doc/fcmaes_examples/index.html`.

## Use `fcmaes-core` from Rust

Inside this workspace, depend on the core crate with:

```toml
[dependencies]
fcmaes-core = { path = "../crates/fcmaes-core" }
```

The following program minimizes a five-dimensional sphere with Differential
Evolution:

```rust
use fcmaes_core::{De, DeParams, Fitness};

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum()
}

fn main() {
    let dim = 5;
    let lower = vec![-5.0; dim];
    let upper = vec![5.0; dim];
    let fitness = Fitness::bounded(dim, 1, &lower, &upper);
    let params = DeParams {
        max_evaluations: 20_000,
        seed: 1,
        ..Default::default()
    };
    let mut optimizer = De::new(fitness, &[], &[], None, &params);
    let result = optimizer.optimize(&sphere);

    println!("value={} evaluations={} x={:?}",
             result.y, result.evaluations, result.x);
}
```

Objectives are ordinary `Fn(&[f64]) -> f64 + Sync` functions or closures.
The blanket `Objective` implementation avoids an adapter type for the common
single-objective case.

## Build the Python extension

Create a virtual environment and install the extension build tools:

```bash
python -m venv .venv
.venv/bin/python -m pip install "maturin[patchelf]>=1.7,<2"
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/maturin develop --release \
  --manifest-path crates/fcmaes-py/Cargo.toml
```

Verify that Python loaded the Rust backend:

```bash
.venv/bin/python -c \
  'from _fcmaes_ext import phase1_build_info; print(phase1_build_info())'
```

The returned dictionary should contain `"backend": "rust"`. The extension is
low-level; this repository does not bundle a Python facade package.

## Run native examples

List the example CLI options:

```bash
cargo run --release -p fcmaes-examples --bin gtop-examples -- --help
cargo run --release -p fcmaes-examples --bin gtop-advexamples -- --help
```

The first binary uses independent retry; the second uses coordinated retry
with adaptive budgets, crossover guesses, and diversity filtering. See
[Examples](examples.md) for the full problem list and commands.

The application ports are separate binaries. For example, run the embedded
flexible job-shop problem with:

```bash
cargo run --release -p fcmaes-examples --bin jobshop -- --evals 2000
```

The complete binary catalog, optional data inputs, and commands for Mazda,
trading, material flow, job-shop/harvesting, t-design, transfer scheduling,
damp control, F-8, and Lotka-Volterra are in
[Native Rust examples](examples.md#binaries).

## Common mistakes

- Do not benchmark a debug build. Optimizer numerics are substantially slower
  without `--release`.
- Retry workers and optimizer population workers are separate levels of
  parallelism. Avoid enabling both without considering oversubscription.
- Bounds must be finite, non-empty, equal-length vectors with `lower[i] <
  upper[i]`.
- Population optimizers can finish their current population after crossing an
  evaluation limit, so reported evaluations can slightly exceed the request.
- A low `value_limit` filters retained retry results; it is not a target that
  stops the run. `stop_fitness` is the early-stop control.
