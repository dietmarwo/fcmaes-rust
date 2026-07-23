# fcmaes-rust

![Pure Rust optimizer core](https://img.shields.io/badge/optimizer%20core-100%25%20Rust-brightgreen)
![No C++ backend](https://img.shields.io/badge/C%2B%2B%20backend-none-brightgreen)

`fcmaes-rust` is a native Rust implementation of fast, parallel,
gradient-free optimization algorithms and selected fcmaes application
examples. The optimizer implementation in `fcmaes-core` is 100% Rust: it does
not compile, link, load, or call the original fast-cma-es C++ implementation.
Optimizer numerics, retry coordination, random-number generation, fitness
evaluation, and parallel execution all run in Rust.

In this project, “port” means that algorithms were translated, reimplemented,
and tested in Rust. It does not mean an FFI wrapper around the old C++ code;
C++ references in comments record provenance and behavioral comparisons only.

The repository is a standalone Cargo workspace. GTOP models and example
objective functions also execute in Rust. This includes the Mazda mass and
constraint response surfaces: their compact model data is embedded in the
example crate and evaluated by native Rust code.

## Implementation facts

| Feature | Implementation |
|---|---|
| Optimizer core | 100% native Rust in `fcmaes-core` |
| Legacy C++ optimization backend | None; no C++ library is compiled, linked, loaded, or invoked |
| Core build | Standard Cargo build; no project `build.rs`, CMake, or C/C++ compiler |
| Parallelism | Native multithreading with Rayon plus independent retry workers |
| Objective functions | Native Rust closures and batch evaluators |
| Python integration | Optional PyO3 extension that exposes the Rust core; Python is not an optimizer backend |

To build only the reusable optimizer library, a Rust toolchain is sufficient:

```bash
cargo build --release -p fcmaes-core
```

This statement deliberately applies to the optimizer core. Building every
optional workspace component can additionally require Python for `fcmaes-py`
and native tooling pulled in by data-compression or network dependencies used
by examples. Those integrations do not contain or restore the historical C++
optimizer backend.

## Workspace

| Crate | Purpose |
|---|---|
| `fcmaes-core` | Optimizers, fitness handling, RNG, retry, multi-objective optimization, and quality diversity |
| `examples` (`fcmaes-examples`) | Native GTOP problems, application objectives, benchmarks, and executable examples |
| `fcmaes-py` | Optional PyO3 extension for embedding the Rust implementation in a Python package |

Implemented algorithms include Differential Evolution, active CMA-ES,
CR-FM-NES, PGPE, Dual Annealing, BiteOpt, MODE, CVT-MAP-Elites, the
Diversifier, independent retry, coordinated retry, and weighted
multi-objective retry.

The example crate includes GTOP mission optimization, Mazda factory-design
objectives, stock-strategy optimization, material-flow planning, flexible
job-shop and harvesting, spherical t-design, transfer scheduling, damped
control, F-8 aircraft control, and Lotka-Volterra control.

## Quick start

Install a current stable Rust toolchain, then run from this directory:

```bash
cargo test --workspace
cargo build --release --workspace
```

Run a small native optimization:

```bash
cargo run --release -p fcmaes-examples --bin jobshop -- --evals 2000
```

Run a GTOP retry workload:

```bash
cargo run --release -p fcmaes-examples --bin gtop-examples -- \
  --problem cassini1 --retries 16 --evaluations 5000 --workers 16 --seed 1
```

## Documentation

- [AI problem-solving context](ai-context.md)
- [Getting started](docs/getting-started.md)
- [Architecture and implementation boundaries](docs/architecture.md)
- [Optimizer guide](docs/optimizers.md)
- [Retry and multi-objective retry](docs/retry.md)
- [Native examples and benchmarks](docs/examples.md)
- [Optional PyO3 bindings](docs/python-bindings.md)
- [Development and testing](docs/development.md)
- [Recorded native benchmark results](benchmarks/README.md)

Generate the complete API reference with:

```bash
cargo doc --workspace --no-deps --open
```

## Data-backed examples

The examples are self-contained by default. The trading example includes an
offline adjusted-close cache and can optionally refresh it through Yahoo
Finance. The Mazda decision table and compact response-surface data are bundled
under `examples/data/`; neither Mazda binary accepts or needs an external model
path. See the [Mazda data notice](examples/data/MAZDA_NOTICE.md) for provenance
and the benchmark's acknowledgement request.

Both Mazda drivers accept `--workers N` for ordered parallel objective batches;
use `--workers 16` for sixteen evaluation threads or `--workers 0` to select
available parallelism.

This public workspace intentionally contains only the Rust port and its
related documentation, native examples, benchmark results, and optional Rust
bindings. Historical Python/C++ implementations and port-planning material are
not part of this repository.

## License

The Rust source and documentation are MIT licensed; see [LICENSE](LICENSE).
The embedded Mazda benchmark data retains its recorded provenance and
acknowledgement request; see [the Mazda data notice](examples/data/MAZDA_NOTICE.md).
