# fcmaes-core

`fcmaes-core` provides fast, parallel, gradient-free optimization algorithms
implemented entirely in Rust. It is the reusable optimizer core of the
[fcmaes-rust](https://github.com/dietmarwo/fcmaes-rust) project and does not
compile, link, load, or call the historical C++ optimizer backend.

## Installation

```bash
cargo add fcmaes-core
```

The minimum supported Rust version is 1.88.

## Minimal example

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

    println!(
        "value={} evaluations={} x={:?}",
        result.y, result.evaluations, result.x
    );
}
```

Use optimized builds for real workloads:

```bash
cargo run --release
```

## Capabilities

- Differential Evolution, active CMA-ES, CR-FM-NES, PGPE and Dual Annealing
- BiteOpt and batched ask/tell optimization
- MODE multi-objective optimization
- independent, coordinated and weighted multi-objective retry
- CVT MAP-Elites, quality-diversity search and the Diversifier
- native multithreading and parallel batch evaluation

## Documentation

- [API reference](https://docs.rs/fcmaes-core)
- [Getting started](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/getting-started.md)
- [Optimizer selection](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/optimizers.md)
- [Retry guide](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/retry.md)

Avoid CPU oversubscription when combining retry-level and
population-evaluation parallelism. Debug builds are not representative of
optimizer performance.

## License

MIT
