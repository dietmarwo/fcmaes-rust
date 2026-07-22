# fcmaes-rust

`fcmaes-rust` is a native Rust port of the gradient-free optimization
algorithms and selected application examples from fcmaes. The repository is a
standalone Cargo workspace: optimizer numerics, retry coordination, GTOP
models, and most example objective functions execute entirely in Rust. The
Mazda examples are the one deliberate exception: they call an optional
external response-surface model through its published C ABI.

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
adapters, stock-strategy optimization, material-flow planning, flexible
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

## External data and models

Most examples are self-contained. The trading example includes an offline
adjusted-close cache and can optionally refresh it through Yahoo Finance. The
Mazda examples are Rust optimizers around the published Mazda model ABI; the
generated model library and decision table are not bundled and must be passed
with `--library` and `--decisions`.

This public workspace intentionally contains only the Rust port and its
related documentation, native examples, benchmark results, and optional Rust
bindings. Historical Python/C++ implementations and port-planning material are
not part of this repository.

## License

MIT. See [LICENSE](LICENSE).
