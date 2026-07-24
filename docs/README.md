# fcmaes-rust documentation

This directory documents the standalone Rust implementation. Generated API
documentation comes from the Rust sources; these guides focus on architecture,
configuration, workflows, and runnable examples.

## Documentation map

| Document | Read this for |
|---|---|
| [AI problem-solving context](../ai-context.md) | Selecting algorithms, parameters, budgets, encodings, and validation for a new user problem |
| [Getting started](getting-started.md) | Building, testing, generating rustdoc, and running a first optimizer |
| [Choosing an optimizer](choosing-an-optimizer.md) | Deciding whether fcmaes-rust, a structured solver, gradients, a surrogate, or another search representation fits the problem |
| [Architecture](architecture.md) | Workspace layout, execution paths, concurrency, and scope |
| [Optimizers](optimizers.md) | Pure-Rust optimizer APIs, defaults, one-shot operation, and ask/tell operation |
| [Retry](retry.md) | Basic, coordinated, and multi-objective retry |
| [Optional Python bindings](python-bindings.md) | Direct PyO3 extension surface and GIL considerations |
| [Examples](examples.md) | Every native binary, data input, GTOP problem, monitor, and benchmark |
| [Buckingham–Pi](buckingham-pi.md) | Numerical dimensional analysis, holdout scoring, BiteOpt retry, and MODE |
| [Application tutorials](../tutorials/README.md) | Nine native optimization applications, including simulation, policy search, ML hyperparameter tuning, and custom-backend verification |
| [Development](development.md) | Formatting, linting, tests, coverage, rustdoc, and extension points |

## Implemented Rust surface

- Bounded fitness handling, normalization, scalar and population evaluation,
  evaluation counting, and PCG-based random generation.
- Differential Evolution, active CMA-ES, CR-FM-NES, PGPE, Dual Annealing,
  BiteOpt, MODE, CVT-MAP-Elites, and the Diversifier.
- Independent retry, coordinated advanced retry, and weighted
  multi-objective retry.
- Native GTOP and Mazda objectives plus application drivers for factory design,
  stock trading, material flow, flexible job-shop/harvesting, multi-UAV task
  assignment, Buckingham–Pi analysis, spherical t-design, transfer scheduling,
  damped control, F-8, and Lotka-Volterra.
- Nine standalone native application tutorials, including a PGPE/CR-FM-NES
  neural policy-search showcase, validation-aware
  SmartCore hyperparameter optimization and robust
  room-ventilation optimization with a purpose-built D2Q9/D2Q5 backend,
  held-out releases, MODE, and MAP-Elites.
- An optional PyO3 extension distributed through the `fcmaes_rust` Python
  facade.

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
