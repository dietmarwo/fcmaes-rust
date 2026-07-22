# Parallel retry

## Basic retry

`fcmaes_core::retry` runs independent optimizer restarts. A fixed worker pool
claims retry IDs atomically, builds a `RetryContext`, executes the caller's
optimizer closure outside the shared-store lock, and retains bounded best
results. Each logical worker owns a persistent PCG stream spawned from the
root seed with a distinct PCG stream selector.

`RetryConfig` defaults:

| Field | Default | Meaning |
|---|---:|---|
| `num_retries` | 1,024 | Maximum restart count |
| `workers` | 0 | Available parallelism; capped by retry count |
| `capacity` | 500 | Maximum retained result entries |
| `value_limit` | positive infinity | Retain only better values |
| `stop_fitness` | negative infinity | Stop claiming work after this target is reached |
| `max_evaluations` | 50,000 | Base budget passed to each restart |
| `seed` | 0 | Root seed used to spawn independent worker streams |
| `statistic_num` | 0 | Maximum completed-retry improvement samples |

The optimizer closure receives the objective and a context, and returns
`RetryRunResult`:

```rust
use fcmaes_core::{
    retry, De, DeParams, Fitness, RetryBounds, RetryConfig, RetryRunResult,
};

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum()
}

let bounds = RetryBounds::new(vec![-5.0; 5], vec![5.0; 5]).unwrap();
let config = RetryConfig {
    num_retries: 32,
    workers: 8,
    max_evaluations: 10_000,
    seed: 1,
    ..Default::default()
};

let result = retry(&sphere, &bounds, &config, |objective, context| {
    let fitness = Fitness::bounded(
        context.bounds.dim(),
        1,
        context.bounds.lower(),
        context.bounds.upper(),
    );
    let params = DeParams {
        max_evaluations: context.max_evaluations,
        seed: context.seed,
        runid: context.run_id as i64,
        ..Default::default()
    };
    let mut optimizer = De::new(fitness, &[], &[], None, &params);
    let run = optimizer.optimize(objective);
    RetryRunResult {
        x: run.x,
        y: run.y,
        evaluations: run.evaluations,
    }
});

assert_eq!(result.runs, 32);
```

## Weighted multi-objective retry

`fcmaes_core::moretry` coordinates independent scalarizations of a
multi-objective problem. Each retry samples and p-norm-normalizes an
independent weight vector, maps it through `weight_lower..weight_upper`, and
presents a `WeightedObjective` to a scalar optimizer. The last `ncon` returned
values are constraints; every positive constraint also adds its weight as a
violation penalty.

```rust
use fcmaes_core::{
    moretry, De, DeParams, Fitness, MoRetryConfig, RetryBounds,
    RetryRunResult,
};

let objective = |x: &[f64]| vec![x[0].powi(2), (x[1] - 1.0).powi(2)];
let bounds = RetryBounds::new(vec![-2.0; 2], vec![2.0; 2])?;
let mut config = MoRetryConfig::new(vec![0.0; 2], vec![1.0; 2]);
config.retry.num_retries = 32;
config.retry.workers = 8;
config.retry.capacity = 32;

let result = moretry(&objective, &bounds, &config, |weighted, context| {
    let fitness = Fitness::bounded(
        context.bounds.dim(), 1,
        context.bounds.lower(), context.bounds.upper(),
    );
    let parameters = DeParams {
        max_evaluations: context.max_evaluations,
        seed: context.seed,
        ..Default::default()
    };
    let run = De::new(fitness, &[], &[], None, &parameters).optimize(weighted);
    RetryRunResult { x: run.x, y: run.y, evaluations: run.evaluations }
})?;
# Ok::<(), &'static str>(())
```

`MoRetryResult.entries` retains the decision vector, original vector-valued
result, sampled weights, and scalar result. `value_limits`, when set, must have
one strict upper limit per objective and constraint. `pareto_indices` returns
the non-dominated row indices for a requested objective prefix.

The coordinator creates its worker streams before scheduling. Weight draws,
step-size draws, and optimizer seeds all come from the owning persistent
stream, so logical workers remain statistically independent and reproducible
for a fixed configuration.

## Coordinated advanced retry

`advanced_retry` adds:

- A linearly increasing evaluation factor from 1 to `max_eval_fac` across
  claimed retry IDs.
- Elite-parent crossover with local bounds and a donor guess.
- Step-size estimates based on parent distance.
- Parent-value acceptance limits for crossover results.
- Diversity filtering using normalized distance.

`AdvancedRetryConfig` wraps a `RetryConfig` and defaults to 5,000 retries,
1,500 initial evaluations, factor 50, checkpoint interval 100, crossover
probability 0.5, and diversity threshold 0.15.

An advanced optimizer closure should consume every relevant context field:

- Always use `context.bounds`, which may be a crossover-local box.
- Use `context.guess` when present.
- Interpret `context.sdev` as per-coordinate step information.
- Use `context.max_evaluations`, not only the base configuration.
- Return the actual evaluation count.

Ignoring the guess and local bounds makes advanced retry behave much more like
expensive independent retry. The DE→CMA implementation in
`examples/src/runner.rs` is the canonical in-repository example.

## Results and statistics

`RetryResult` contains:

| Field | Meaning |
|---|---|
| `x`, `y` | Best retained point and objective value |
| `evaluations` | Sum reported by completed optimizer runs |
| `runs` | Number of completed retries |
| `success` | Whether a finite, dimensionally valid result was retained |
| `entries` | Retained result population, bounded by capacity |
| `improvements` | Best-value improvements observed when retries completed |

The built-in improvement history is not an every-evaluation trace. The native
GTOP CLI adds an atomic objective wrapper and reporting thread when
`--progress-interval` is enabled; see [Examples](examples.md#live-progress).

## Worker behavior

The active retry worker count is:

```text
min(requested workers or available parallelism, num_retries)
```

At least as many retries as workers are required to use every worker. The
result store is locked only while creating coordinated contexts and adding
completed results; optimizer execution remains outside the lock.

Worker RNGs are created once, before their retry loops, and are never shared.
Each worker stream supplies its retry step-size draws, crossover decisions,
parent selections, and child optimizer seeds. A root-seed mixer and distinct
native PCG stream selectors give every logical worker an independently seeded
persistent generator.

A single-worker run is exactly repeatable for a fixed root seed. With multiple
workers, operating-system scheduling can change how many retries each worker
stream consumes. The streams remain independent, but repeated parallel runs
need not be bit-for-bit identical.

Native objective functions can evaluate concurrently on all retry worker
threads. Objectives supplied through the optional PyO3 extension must acquire
the Python GIL for each callback, so cheap callbacks may not scale with the
worker count.

## Value limit versus stop fitness

These controls are intentionally different:

- `value_limit` filters which completed runs can enter the result store.
- `stop_fitness` stops new work after a sufficiently good stored result.

For Messenger Full, `--value-limit 12` means that early results above 12 are
not retained. It does not terminate when 12 is reached unless stop fitness is
also configured to that target.
