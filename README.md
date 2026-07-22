# fcmaes-rust

`fcmaes-rust` is a native Rust port of the gradient-free optimization
algorithms and selected application examples from fcmaes. The repository is a
standalone Cargo workspace: optimizer numerics, retry coordination, GTOP
models, and example objective functions execute entirely in Rust. This now
includes the Mazda mass and constraint response surfaces: their compact model
data is embedded in the example crate and evaluated by native Rust code.

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
