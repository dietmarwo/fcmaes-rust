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

## Parallel design: retry and ask/tell

fcmaes-core uses two complementary forms of parallelism to improve the result
returned within a fixed wall time:

- `retry` runs independent optimizer instances on separate workers and keeps
  the best results. Its closure can wrap fcmaes algorithms or compatible
  single-threaded optimizers from other crates and FFI libraries.
- ask/tell and batch APIs expose one population for concurrent evaluation.
  This is especially effective when objective calls are expensive simulations,
  training jobs, hardware tests, or remote requests.

Every population-based core optimizer has ask/tell or an equivalent batch
boundary: DE, active CMA-ES, CR-FM-NES, PGPE, BiteOpt, MODE, MAP-Elites, and
Diversifier. Dual Annealing is sequential and has no population ask/tell
interface; use retry to parallelize independent Dual Annealing searches.

The controlled
[GTOP equal-wall campaign](https://dietmarwo.github.io/fcmaes-rust/benchmarks/gtop-cmaes-retry/)
uses retry around an external serial CMA-ES implementation. On its fixed
16-core machine and four-second protocol, retry improves mean objective on all
seven tested problems over 100 paired seeds per problem. This is scoped
experimental evidence, not a guarantee for every workload. When combining
outer retry with inner population evaluation, divide the physical cores
between the two layers to avoid oversubscription.

## Capabilities

- Differential Evolution, active CMA-ES, CR-FM-NES, PGPE and Dual Annealing
- BiteOpt and batched ask/tell optimization
- MODE multi-objective optimization
- exact and sampled hypervolume, IGD/IGD+, GD/GD+, epsilon, spacing, spread,
  non-dominated sorting, and crowding distance
- independent, coordinated and weighted multi-objective retry
- CVT MAP-Elites, quality-diversity search and the Diversifier
- native multithreading and parallel batch evaluation

## API map

| Problem | Start here |
|---|---|
| General bounded scalar optimization | [`De`](https://docs.rs/fcmaes-core/latest/fcmaes_core/de/struct.De.html), [`Cmaes`](https://docs.rs/fcmaes-core/latest/fcmaes_core/cmaes/struct.Cmaes.html), or [`BiteOpt`](https://docs.rs/fcmaes-core/latest/fcmaes_core/biteopt/struct.BiteOpt.html) |
| High-dimensional or noisy search | [`Pgpe`](https://docs.rs/fcmaes-core/latest/fcmaes_core/pgpe/struct.Pgpe.html) or [`Crfmnes`](https://docs.rs/fcmaes-core/latest/fcmaes_core/crfmnes/struct.Crfmnes.html) |
| Independent or adaptive restarts | [`retry`](https://docs.rs/fcmaes-core/latest/fcmaes_core/retry/index.html) |
| Several competing objectives | [`mode`](https://docs.rs/fcmaes-core/latest/fcmaes_core/mode/index.html) and [`moretry`](https://docs.rs/fcmaes-core/latest/fcmaes_core/moretry/index.html) |
| Audit a multi-objective front | [`indicators`](https://docs.rs/fcmaes-core/latest/fcmaes_core/indicators/index.html) |
| Diverse high-quality solutions | [`mapelites`](https://docs.rs/fcmaes-core/latest/fcmaes_core/mapelites/index.html) |
| External, GPU, service, or custom batch evaluator | the ask/tell methods on DE, CMA-ES, CR-FM-NES, PGPE, BiteOpt, and MODE |

Every public item is documented. Module pages provide runnable examples,
algorithm references, parameter semantics, stopping behavior, and execution
notes. The crate rejects undocumented public additions and broken rustdoc links
at compile time.

## Documentation

- [API reference](https://docs.rs/fcmaes-core)
- [Rendered user guide](https://dietmarwo.github.io/fcmaes-rust/)
- [Getting started](https://dietmarwo.github.io/fcmaes-rust/docs/getting-started.html)
- [Choosing an optimizer](https://dietmarwo.github.io/fcmaes-rust/docs/choosing-an-optimizer.html)
- [Optimizer scope and interoperation](https://dietmarwo.github.io/fcmaes-rust/docs/optimizer-boundary.html)
- [Optimizer guide](https://dietmarwo.github.io/fcmaes-rust/docs/optimizers.html)
- [Retry guide](https://dietmarwo.github.io/fcmaes-rust/docs/retry.html)

Avoid CPU oversubscription when combining retry-level and
population-evaluation parallelism. Debug builds are not representative of
optimizer performance.

## License

MIT
