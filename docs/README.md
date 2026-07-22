# fcmaes-rust documentation

This directory documents the standalone Rust implementation. Generated API
documentation comes from the Rust sources; these guides focus on architecture,
configuration, workflows, and runnable examples.

## Documentation map

| Document | Read this for |
|---|---|
| [Getting started](getting-started.md) | Building, testing, generating rustdoc, and running a first optimizer |
| [Architecture](architecture.md) | Workspace layout, execution paths, concurrency, and scope |
| [Optimizers](optimizers.md) | Pure-Rust optimizer APIs, defaults, one-shot operation, and ask/tell operation |
| [Retry](retry.md) | Basic, coordinated, and multi-objective retry |
| [Optional Python bindings](python-bindings.md) | Direct PyO3 extension surface and GIL considerations |
| [Examples](examples.md) | Every native binary, data input, GTOP problem, monitor, and benchmark |
| [Development](development.md) | Formatting, linting, tests, coverage, rustdoc, and extension points |

## Implemented Rust surface

- Bounded fitness handling, normalization, scalar and population evaluation,
  evaluation counting, and PCG-based random generation.
- Differential Evolution, active CMA-ES, CR-FM-NES, PGPE, Dual Annealing,
  BiteOpt, MODE, CVT-MAP-Elites, and the Diversifier.
- Independent retry, coordinated advanced retry, and weighted
  multi-objective retry.
- Native GTOP and Mazda objectives plus application drivers for factory design,
  stock trading, material flow, flexible job-shop/harvesting, spherical
  t-design, transfer scheduling, damped control, F-8, and Lotka-Volterra.
- An optional PyO3 extension crate. No Python facade package is bundled.

## Fast path

From the repository root:

```bash
cargo test --workspace
cargo build --release --workspace
cargo doc --workspace --no-deps
```

Run a small native GTOP workload:

```bash
cargo run --release -p fcmaes-examples --bin gtop-examples -- \
  --problem cassini1 --retries 16 --evaluations 5000 --workers 16 --seed 1
```

Run the hard Messenger Full workload with live progress:

```bash
cargo run --release -p fcmaes-examples --bin gtop-advexamples -- \
  --problem messenger-full --retries 50000 --evaluations 1500 \
  --workers 16 --seed 1 --value-limit 12 \
  --max-eval-fac 50 --check-interval 100 --progress-interval 10
```
