<p align="center">
  <img src="docs/images/fcmaes-rust.png" alt="fcmaes-rust logo">
</p>

# fcmaes-rust

![Pure Rust optimizer core](https://img.shields.io/badge/optimizer%20core-100%25%20Rust-brightgreen)
![No C++ backend](https://img.shields.io/badge/C%2B%2B%20backend-none-brightgreen)
[![crates.io](https://img.shields.io/crates/v/fcmaes-core.svg?cacheSeconds=300)](https://crates.io/crates/fcmaes-core)
[![docs.rs](https://docs.rs/fcmaes-core/badge.svg)](https://docs.rs/fcmaes-core)
[![PyPI](https://img.shields.io/pypi/v/fcmaes_rust.svg?cacheSeconds=300)](https://pypi.org/project/fcmaes-rust/)

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
example crate and evaluated by native Rust code. The Buckingham–Pi example
also implements dimension-matrix enumeration, numerical nullspace analysis,
regression, and continuous-exponent optimization directly in Rust; it has no
BuckinghamPy dependency.

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
| `fcmaes-gtop` | Internal native GTOP objective library shared by the examples and Python package |
| `examples` (`fcmaes-examples`) | Native GTOP problems, application objectives, benchmarks, and executable examples |
| `fcmaes-py` | Optional PyO3 extension for embedding the Rust implementation in a Python package |
| `tutorials/*` | Standalone application workspaces using NeXosim, Rapier, ReBop, Brahe, RustPower, SmartCore, native atmospheric dispersion, and a custom educational ventilation backend |

Only two registry artifacts are published: `fcmaes-core` on crates.io and the
`fcmaes-rust` binding distribution on PyPI. `fcmaes-gtop` is an internal
source dependency marked `publish = false`; `examples/` and `tutorials/` are
available only from this GitHub repository and are not included in either
registry package.

The application tutorials are intentionally not root workspace members. Each
keeps its application-specific dependencies, artifacts, and lockfile isolated;
run its Cargo commands from that tutorial directory.

All eight retain multi-objective optimization. MAP-Elites campaigns are
implemented and recorded for NeXosim, Rapier, ReBop, Brahe, atmospheric source
localization, room ventilation, and RustPower; the SmartCore hyperparameter
tutorial additionally contains an explicitly provisional QD pilot. That
tutorial demonstrates an often missed part of optimizer benchmarking:
fixed-fold tuning, disjoint model selection, frozen final evaluation, and
probability-aware metrics must be part of the objective protocol. RustPower
additionally records a descriptor case study: its first pair of descriptors
were decision variables and reached only 4% coverage, while emergent behavior
coordinates measured from the solved scenarios reached 68% mean coverage over
three seeds at the identical 100k-evaluation budget. Constrained MODE stays the
primary formulation there, because the near-unique asset architecture the
original pilot found survives the descriptor fix. The
[tutorial index](tutorials/README.md) includes commands, figures, validation
results and the common result schema. Compact canonical result directories are
version-controlled with generated SVGs so raw-evidence links and deterministic
rendering work from a clean clone. Schema-driven tutorials use `run.json`;
room ventilation adds an aggregate CSV/plot check spanning three seeds, CFD
fields, and resolution evidence.

Implemented algorithms include Differential Evolution, active CMA-ES,
CR-FM-NES, PGPE, Dual Annealing, BiteOpt, MODE, CVT-MAP-Elites, the
Diversifier, independent retry, coordinated retry, and weighted
multi-objective retry.

The example crate includes GTOP mission optimization, Mazda factory-design
objectives, stock-strategy optimization, material-flow planning, flexible
job-shop and harvesting, multi-UAV task assignment, spherical t-design,
transfer scheduling, Buckingham–Pi dimensional analysis, damped control, F-8
aircraft control, and Lotka-Volterra control.

## Quick start

### Install the released packages

Inside a Rust application, add the published optimizer library:

```bash
cargo add fcmaes-core
```

The crate is imported in Rust as `fcmaes_core`.

Python users install the published binding distribution with:

```bash
python -m pip install fcmaes-rust
```

It imports as `fcmaes_rust`; its optimizer backend is the same native Rust
implementation:

```python
import fcmaes_rust

print(fcmaes_rust.__version__)
print(fcmaes_rust.phase1_build_info())
```

A compatible prebuilt wheel requires neither a local Rust toolchain nor a
C/C++ compiler. Published wheels support CPython 3.11 through 3.13. Building
the Python package from its source distribution does require Rust.

### Build and run the repository

Install Rust 1.88 or newer, then run from this directory. Rust 1.88 is the
tested minimum for the edition-2024 source and current locked dependency set;
the CI workflow checks it explicitly.

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

With `fcmaes-rust` installed, run the repository's active CMA-ES Python
example with:

```bash
python examples/python/test_cma.py
```

## Documentation

- [AI problem-solving context](ai-context.md)
- [Getting started](docs/getting-started.md)
- [Architecture and implementation boundaries](docs/architecture.md)
- [Optimizer guide](docs/optimizers.md)
- [Retry and multi-objective retry](docs/retry.md)
- [Native examples and benchmarks](docs/examples.md)
- [Buckingham–Pi dimensional-analysis example](docs/buckingham-pi.md)
- [Native Rust application-optimization tutorials](tutorials/README.md)
- [ML hyperparameter-optimization tutorial](tutorials/ml-hyperparameter-tuning/README.md)
- [Room-ventilation optimization tutorial](tutorials/cfd-room-ventilation/README.md)
- [Optional PyO3 bindings](docs/python-bindings.md)
- [Release history](CHANGELOG.md)
- [Publishing checklist](RELEASING.md)
- [Development and testing](docs/development.md)
- [Recorded native benchmark results](benchmarks/README.md)

Generate the complete API reference with:

```bash
cargo doc --workspace --no-deps --open
```

## Optimizer comparison

The reproducible [GTOP optimizer comparison](benchmarks/optimizer-comparison/comparison.md)
uses 100 experiments per problem and a common 240,000-evaluation cap. fcmaes
has the best mean optimum on six of seven problems and the lowest mean wall
time on five of seven. The exception in mean solution quality is Tandem, where
the adaptive BIPOP-CMA-ES restart strategy leads the equal-budget table but
does not reach the `-1493` target.

In the pre-registered Tandem stress test, BIPOP-CMA-ES reached 0/1,000 targets:
its best result was -1410.050665 after 9,466,290,846 actual evaluations. In
contrast, fcmaes coordinated DE→CMA retry reached the target in 85/100
experiments with a mean of 230,727,025 evaluations. These are separate budget
regimes: the first comparison is equal-budget, while the latter result
demonstrates the benefit of adaptive retry coordination on a hard problem.
The original Python/C++ fcmaes performance table reports a similar 81/100
Tandem success rate; the linked report records both results and their exact
parallel execution models.

## Data-backed examples

The examples are self-contained by default. The trading example includes an
offline adjusted-close cache and can optionally refresh it through Yahoo
Finance. The Mazda decision table and compact response-surface data are bundled
under `examples/data/`; neither Mazda binary accepts or needs an external model
path. See the [Mazda data notice](examples/data/MAZDA_NOTICE.md) for provenance
and the benchmark's acknowledgement request. The
[Multi-UAV data and compatibility notice](examples/data/UAV_NOTICE.md)
documents the native task-assignment port and its source benchmark.
The [Buckingham notice](examples/data/BUCKINGHAM_NOTICE.md) records the
dimension-matrix catalog's provenance and the numerical port's deliberately
narrower scope than BuckinghamPy.

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
