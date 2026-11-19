# Optimizers

## Choosing an optimizer

| Optimizer | Style | One-shot | Stateful | Useful starting point |
|---|---|---:|---:|---|
| Differential Evolution | Population, global | `De::optimize` | `ask` / `tell` | Robust bounded search and retry pipelines |
| jSO | Adaptive DE with population size reduction | `Jso::optimize` | `ask` / `tell` | First stage before CMA-ES, as an alternative to DE |
| Active CMA-ES | Distribution, local/global | `Cmaes::optimize` | `ask`, `tell`, `tell_x` | Continuous problems and DE→CMA sequences |
| CR-FM-NES | Natural evolution strategy | `optimize_batch` | `ask_pop` / `tell_pop` | Higher-dimensional continuous problems |
| PGPE | Mirrored policy-gradient search | `optimize_batch` | `ask_pop` / `tell_pop` | Batch evaluation and distribution search |
| Dual Annealing | Annealing plus optional local search | `optimize_da` | No | Global exploration with bounded local refinement |
| BiteOpt | Adaptive mixed generator family | `optimize_bite` / `optimize` | `ask` / `tell` | Difficult bounded black-box problems |
| MODE | Constrained multi-objective DE/NSGA-II | No | `ask` / `tell` | Pareto-front search |
| MAP-Elites / Diversifier | Quality diversity | `map_elites` / `diversify` | Archive updates | Diverse niche elites |

All optimizers minimize. Scalar optimizer result objects contain a best point
`x`, objective value `y`, evaluation and iteration counts, and an integer stop
code. `ModeResult` instead contains its population matrices plus iteration and
stop state; MAP-Elites retains its results in `Archive`.

## Parallel evaluation: ask/tell and retry

fcmaes-core offers two independent parallelism boundaries, both intended to
improve the result returned after a fixed user-visible wait:

1. `retry` runs complete optimizer instances concurrently. It is especially
   useful for multimodal problems and for adapting otherwise serial external
   optimizers.
2. Ask/tell separates candidate generation from evaluation. Evaluate one
   population concurrently, preserve its order, then tell the scores back to
   the same optimizer. This is especially useful when objective calls are
   expensive.

| Algorithm | Population-evaluation interface | Notes |
|---|---|---|
| Differential Evolution | `ask` / `tell`; `optimize_delayed(objective, workers)` | One ordered DE population batch; `optimize_delayed` keeps `workers` candidates in flight |
| jSO | `ask` / `tell`; `optimize_parallel(objective, workers)` | One generation; the batch shrinks with the population and the last one is cut at the budget |
| Active CMA-ES | `ask` / `tell`; `tell_x`; parallel one-shot evaluation | One CMA generation |
| CR-FM-NES | `ask_pop` / `tell_pop`; `optimize_batch` | One mirrored population |
| PGPE | `ask_pop` / `tell_pop`; `optimize_batch` | One mirrored population |
| BiteOpt / DeepBiteOpt | `ask(batch)` / `tell`; `BiteOpt::optimize_batch` | Delayed-feedback candidate batch |
| MODE | `ask` / `tell` | Objective-and-constraint rows for one population |
| MAP-Elites / Diversifier | `*_batch` evaluators and ordered archive updates | Equivalent batch boundary rather than a literal optimizer `tell` |
| Dual Annealing | None | Sequential proposal dependency; parallelize complete runs with retry instead |

Thus every population-based optimizer in the core has direct ask/tell or an
equivalent batch interface. Dual Annealing is not population-based and is the
only listed optimizer without one. The external `cmaes` 0.2.2 crate used in
the [GTOP retry benchmark](../benchmarks/gtop-cmaes-retry/README.md) likewise
has no public ask/tell boundary, although it does offer internal Rayon
`run_parallel()` and `next_parallel()` methods.

For native CPU evaluation, `parallel_batch` preserves candidate order:

```rust
use fcmaes_core::parallel_batch;

let xs = optimizer.ask();
let ys = parallel_batch(&xs, 16, |x| expensive_objective(x));
optimizer.tell(&ys);
```

The same split permits GPU batches, asynchronous services, or external
simulator pools, as long as results are restored to ask order before `tell`.

`tell` checks the batch it receives against the pending one, the batch
returned by the most recent `ask` and not told yet. DE, jSO, CMA-ES,
CR-FM-NES, PGPE and MODE reject a `tell` without a pending batch or with a
different number of values: `tell` (`tell_pop` for CR-FM-NES and PGPE) panics,
and `try_tell` (`try_tell_pop`) returns the error instead. A rejected call
changes no optimizer state. The optimizers differ in what a second `ask`
before `tell` does: jSO and MODE reject it, BiteOpt returns the pending batch
again, and DE, CMA-ES, CR-FM-NES and PGPE draw a new batch that replaces the
pending one. BiteOpt's `tell` returns `-1` for a wrong batch instead of
panicking.
When combining retry and population evaluation, budget threads explicitly:
for example, four retry lanes with four evaluation workers each on 16 physical
cores. Using 16 lanes each with 16 inner workers normally oversubscribes the
machine and can make wall time worse.

## Shared objective and bounds

An ordinary synchronized scalar closure implements `Objective` automatically:

```rust
let objective = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();
```

Most stateful optimizers receive a `Fitness` object:

```rust
use fcmaes_core::Fitness;

let mut fitness = Fitness::bounded(4, 1, &[-5.0; 4], &[5.0; 4]);
fitness.set_normalize(true);
```

`Fitness::new` also supports unbounded problems when both bound vectors are
empty. Retry and BiteOpt require finite box bounds.

## Differential Evolution

`De` implements DE/best/1 plus temporal locality, age-based population
replacement, oscillating mutation/crossover parameters, optional sampling
around a guess, and optional integer-coordinate mutation.

Key `DeParams` defaults:

| Field | Default |
|---|---:|
| `popsize` | 31 |
| `max_evaluations` | 100,000 |
| `keep` | 200 |
| `stop_fitness` | negative infinity |
| `f` / `cr` | 0.5 / 0.9 |
| `min_mutate` / `max_mutate` | 0.1 / 0.5 |

Pass empty guess and sigma slices for uniform initialization. With a guess,
provide a same-dimensional sigma vector. The optional integer mask must match
the problem dimension.

`ask` returns `popsize` rows; `tell` takes exactly one value per row and
rejects any other length (`try_tell` returns the error instead of panicking).

## jSO

`Jso` implements jSO (Brest, Maučec and Bošković, CEC 2017): differential
evolution with a memory of successful `F` and `CR` values, weighted
current-to-pbest mutation with an archive of replaced parents, and a
population that shrinks linearly from `round(25 ln(D) √D)` to four over the
evaluation budget. It needs finite bounds, samples its initial population
uniformly in the box and takes no guess. The [module
documentation](https://docs.rs/fcmaes-core/latest/fcmaes_core/jso/index.html)
lists the details the paper leaves open and how they are resolved.

```rust
use fcmaes_core::{Fitness, Jso, JsoParams};

let fitness = Fitness::bounded(4, 1, &[-5.0; 4], &[5.0; 4]);
let params = JsoParams {
    max_evaluations: 20_000,
    seed: 1,
    ..Default::default()
};
let mut optimizer = Jso::new(fitness, &params);
let result = optimizer.optimize(&objective);
```

Key `JsoParams` defaults are the published values:

| Field | Default |
|---|---:|
| `popsize` | 0, meaning `round(25 ln(D) √D)` |
| `max_evaluations` | 100,000; must be positive |
| `stop_fitness` | negative infinity |
| `min_popsize` / `memory_size` | 4 / 5 |
| `p_max` / `p_min` / `archive_rate` | 0.25 / 0.125 / 1.0 |

The budget drives the population size, the p-best fraction and the parameter
stages, so set `max_evaluations` to the budget the run really has. It is a
hard limit. `optimize_parallel(objective, workers)` evaluates each generation
as one batch and gives the same result as `optimize`. One `ask`/`tell` pair
is one generation; `tell` rejects a batch of the wrong length (`try_tell`
returns the error instead of panicking).

Use jSO as the first stage of a CMA-ES sequence, as DE is used in DE→CMA:
`jso_cma_run` in `examples/src/runner.rs` gives it 40% of each run and starts
CMA-ES from its best point. [The algorithm portfolio](algorithm-portfolio.md)
has the benchmark evidence.

## Active CMA-ES

`Cmaes` implements the active covariance update. A guess is required, and
`input_sigma` is either one value (broadcast to all dimensions) or one value
per coordinate.

Key `CmaesParams` defaults:

| Field | Default |
|---|---:|
| `popsize` | 31 |
| `mu` | 0, meaning half the population |
| `max_evaluations` | 100,000 |
| `accuracy` | 1.0 |
| `stop_fitness` | negative infinity |
| `stop_tol_hist_fun` / `update_gap` | -1 / -1, automatic |
| `repair_z` | `true` |

`repair_z` matters only when samples reach the bounds. A sample that was
moved into the bounds enters the update with the `z` of the point that was
evaluated (`true`), or with the `z` of the unmoved sample (`false`, the
update of 0.3.0). `true` keeps the step size from growing when samples keep
reaching the bounds; on some problems `false` does better, for example in
some coordinated-retry runs, whose bounds narrow around good solutions.
Runs that stay inside the bounds, unbounded runs and `tell_x` give the same
results with either value.

`Cmaes::optimize(objective, workers)` controls population-evaluation
parallelism. Use `1` when outer retry already supplies parallelism.

`ask` returns `popsize` decoded rows and `tell` takes exactly one value per
row. `tell_x(ys, xs)` tells `popsize` values for points supplied by the caller,
for example repaired versions of the asked rows; it needs no pending `ask`,
consumes one that is pending, and rejects `xs` unless it has one `dim`-wide
point per value. `try_tell` and `try_tell_x` return these errors instead of
panicking.

## CR-FM-NES

`Crfmnes` evaluates complete populations through a batch closure:

```rust
let result = optimizer.optimize_batch(|rows| {
    rows.iter().map(|x| objective(x)).collect()
});
```

Defaults include population 32, 100,000 evaluations, penalty coefficient
`1e5`, and constraint-violation handling enabled. `ask_pop` returns decoded,
in-bound rows; `tell_pop` consumes exactly one value per row and rejects
any other length or a call without a pending `ask_pop` (`try_tell_pop`
returns the error instead of panicking).

## Staged CR-FM-NES and BiteOpt

`crfmnes_bite` composes CR-FM-NES and serial DeepBiteOpt within one
`RetryContext`. It passes the best evaluated decision vector to BiteOpt and
returns the best finite observation across both stages. Use `retry` to run
the sequence independently on several workers.

| `CrfmnesBiteParams` field | Default |
|---|---|
| `crfmnes_share` | 0.30 of each retry's evaluation cap |
| `popsize` | 500 for both stages, rounded up to even for CR-FM-NES |
| `sigma` | Seeded draw in `[0.05, 0.10)`, a fraction of each variable's range |
| `bite_depth` | 4 |
| `stop_fitness` | Negative infinity |

CR-FM-NES evaluates only whole populations that fit its share. BiteOpt gets
every remaining call, including any remainder from that share. A small cap
that cannot fit one CR-FM-NES population goes entirely to BiteOpt. A target
stops CR-FM-NES after the current population and BiteOpt after the current
evaluation. Deriving the sequence's random streams from `RetryContext::run_seed`
makes each retry reproducible across worker counts.

CR-FM-NES works in normalized coordinates, so `sigma` is a fraction of each
variable's range and one default suits a unit box of random keys, a box of
item indices `[0, 10)` and a box of launch dates alike. On a unit box this
is the step of the Python recipe in the OneForAll tutorial, which measures it
in decision units. CR-FM-NES's bound penalty is still measured in decision
units.

```rust
use fcmaes_core::{CrfmnesBiteParams, RetryBounds, RetryConfig, crfmnes_bite, retry};

let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
let bounds = RetryBounds::new(vec![-1.0; 4], vec![1.0; 4]).unwrap();
let settings = CrfmnesBiteParams { popsize: 16, ..Default::default() };
settings.validate(&bounds, None).unwrap();
let config = RetryConfig {
    num_retries: 4,
    workers: 2,
    max_evaluations: 2_000,
    seed: 42,
    ..Default::default()
};
let result = retry(&sphere, &bounds, &config, |objective, context| {
    crfmnes_bite(objective, context, &settings).unwrap()
});
assert_eq!(result.evaluations, 8_000);
assert!(result.y < 1e-6);
```

The same example is a doctest of `crfmnes_bite`.

`crfmnes_bite_until` runs the sequence on explicit bounds, guess, seed and
budget instead of a `RetryContext`, and checks a stop predicate before every
evaluation; once it fires, the run returns the best observation so far. The
job worker uses it for `fcmaes.retry/1` with `kind: crfmnes_bite`, which takes
`crfmnes_share`, `bite_depth`, `population` and `sigma` (in the normalized
box there, twice the fraction used here); see the
[job tutorial](../tutorials/job-api/README.md#scalar-retries-biteopt-de-jso-cma-es-cr-fm-nes-pgpe-and-cr-fm-nes--biteopt).

See the [routing evidence](algorithm-portfolio.md#cr-fm-nes-to-biteopt-for-routing)
and the [example CLI](examples.md#benchmarks).

## PGPE

`Pgpe` uses mirrored sampling, ranking, and ADAM updates for the distribution
center. Defaults include population 32, ranking enabled, center learning rate
0.15, standard-deviation learning rate 0.1, and 100,000 evaluations.

The batch and ask/tell interfaces mirror CR-FM-NES. The optimizer tracks the
true objective best even when ranking is used for updates.

The
[neural-controller policy-search tutorial](../tutorials/neural-controller-policy-search/)
uses these interfaces to compare PGPE and CR-FM-NES on the same 118-parameter
fixed-topology policy, rollout budget, scenario schedule, and disjoint
validation protocol.

## Dual Annealing

Use the free function:

```rust
use fcmaes_core::{optimize_da, DaParams};

let params = DaParams {
    max_evaluations: 50_000,
    use_local_search: true,
    seed: 1,
    ..Default::default()
};
let result = optimize_da(
    &objective,
    &[0.0; 4],
    vec![-5.0; 4],
    vec![5.0; 4],
    &params,
);
```

The optional local search is a self-contained bounded projected L-BFGS
implementation using finite-difference gradients. Empty lower and upper
vectors select the unbounded path. The run never exceeds `max_evaluations`; a
gradient or line search that does not fit into the remaining budget ends it
with the best point evaluated.

## BiteOpt

The Rust implementation includes the adaptive selector tree, primary
generators, diverging populations, spherical and sequential Nelder-Mead
secondary optimizers, dynamic population sizing, deep multi-population mode,
and delayed-feedback ask/tell.

Key `BiteParams` defaults:

| Field | Default |
|---|---:|
| `popsize` | 0, automatic |
| `max_evaluations` | 100,000 |
| `stop_fitness` | negative infinity |
| `stall_criterion` | 0, automatic |

One-shot use:

```rust
use fcmaes_core::{optimize_bite, BiteParams};

let params = BiteParams {
    max_evaluations: 100_000,
    seed: 1,
    ..Default::default()
};
let result = optimize_bite(
    &objective,
    &[-5.0; 4],
    &[5.0; 4],
    None,
    &params,
    1,
);
```

The final argument is deep-mode population depth. Values at most one select a
plain run; the validated maximum is 36. Bounds must be finite and strict.

Ask/tell freezes the selector decisions associated with every candidate until
feedback arrives. Do not call `tell` without a pending batch or with the wrong
number of costs. Non-finite costs are converted to a large finite rejection
value. Batching changes the feedback timing and can trade some convergence
quality for wall-clock throughput on expensive objectives.

## MODE

`Mode` is a bounded ask/tell optimizer for `nobj` minimized objectives followed
by `ncon` constraints. A constraint is feasible when its value is at most zero.
`Mode::try_new` validates the objective width, finite strict bounds, population
size, probabilities, and optional integer mask.

```rust
use fcmaes_core::{Fitness, Mode, ModeParams};

let fitness = Fitness::bounded(4, 2, &[0.0; 4], &[1.0; 4]);
let parameters = ModeParams {
    popsize: 64,
    nsga_update: true,
    seed: 1,
    ..Default::default()
};
let mut mode = Mode::try_new(fitness, 2, 0, None, &parameters)?;
let xs = mode.ask();
let ys: Vec<Vec<f64>> = xs
    .iter()
    .map(|x| vec![x[0], x.iter().map(|v| (v - 1.0).powi(2)).sum()])
    .collect();
mode.tell(&ys);
# Ok::<(), &'static str>(())
```

NSGA-II updates use normalized crowding distance across every objective. Odd
population sizes are supported. Non-finite told values are treated as rejected
candidates. The DE update requires at least four population members.

Key `ModeParams` defaults:

| Field | Default |
|---|---:|
| `popsize` | 64 |
| `f` / `cr` | 0.5 / 0.9 |
| `pro_c` / `dis_c` | 0.5 / 15 |
| `pro_m` / `dis_m` | 0.9 / 20 |
| `nsga_update` | true |
| `pareto_update` | 0 |
| `min_mutate` / `max_mutate` | 0.1 / 0.5 |

The final population need not hold every non-dominated point a run evaluated.
For two objectives, insert each feasible told row into a separate
`ParetoArchive2D` to keep that exact front and its hypervolume; see
[the Pareto archive guide](../crates/fcmaes-core/PARETO_ARCHIVE.md) and
`cargo run --release -p fcmaes-core --example pareto_mode`.

## CVT-MAP-Elites and Diversifier

`Archive::try_new` builds a CVT behavior archive from descriptor bounds.
K-means++ initialization and Lloyd assignment use Rayon for the dominant
distance kernels. Seed parent solutions with `seed_uniform`, evaluate an
initial population, then call `map_elites`; `diversify` uses niche improvement
as the objective seen by CMA-ES.

Expensive native objectives can implement `QdBatchFitness` and use
`map_elites_batch` / `diversify_batch`. `Archive::update_evaluated` separates
parallel evaluation from ordered archive mutation. The general
`parallel_batch` helper uses the same cached, explicitly sized Rayon pools as
population fitness evaluation.

For two descriptor dimensions, `samples_per_niche = 0` selects a rectangular
grid with O(1) niche lookup. Positive values select k-means CVT centers and a
nearest-center scan; use that path when irregular Voronoi niches are required.

The archive minimizes fitness. It exposes occupancy, best fitness, descriptors,
counts, and a Python-compatible QD score. For all-positive fitness the score is
the sum of reciprocals; when any negative value exists it is the sum of
negated negative elites.

Key driver defaults:

| Configuration | Field | Default |
|---|---|---:|
| `MapElitesParams` | `generations` / `chunk_size` | 100 / 20 |
| `MapElitesParams` | `use_sbx` | true |
| `MapElitesParams` | `dis_c` / `dis_m` | 20 / 20 |
| `MapElitesParams` | `iso_sigma` / `line_sigma` | 0.02 / 0.2 |
| `MapElitesParams` | `cma_generations` | 0 |
| `DiversifierParams` | `max_evaluations` | 100,000 |
| `DiversifierParams` | `popsize` / `stall_criterion` | 31 / 20 |

## Evaluation limits and reproducibility

Most population optimizers check budgets at generation boundaries, so actual
evaluation totals can overshoot the configured limit by part of a population.
BiteOpt, jSO, Dual Annealing and `De::optimize_delayed` never exceed it. Use
reported evaluations for accounting.

The retry root seed spawns one persistent, independent PCG stream per worker;
each retry draws its optimizer seed from its worker's stream. A single-worker
run is exactly repeatable. In a multi-worker run, scheduling can change how
many retries each worker stream consumes, so repeated results need not be
bit-identical. Rust is also not expected to reproduce historical C++ or Python
streams exactly; port parity is statistical.

## Cooperative stopping (since 0.3.0)

Use `BiteOpt::optimize_until`, `DeepBiteOpt::optimize_until`, `De::optimize_until` or `Jso::optimize_until` with a borrowed `Fn() -> bool`. The predicate runs before each evaluation; DE also checks before a temporal-locality trial. The result's `stopped` flag distinguishes that exit, while `evaluations`, `x` and `y` describe actual completed work. A stop before the first evaluation has no usable best value.

`BiteOpt::optimize_batch_until` checks before each batch and incorporates an in-flight batch in full. `De::optimize_delayed_until` stops new dispatch and drains all in-flight rows in ask order; its predicate is `Sync`. Their latency is bounded by in-flight work, not by a single callback. CMA-ES has `optimize_until`; CR-FM-NES and PGPE have `optimize_batch_until`, and jSO has `optimize_parallel_until`, each checking between populations. Existing entry points use a never-firing predicate and preserve their seeded arithmetic and budget semantics.

The core cannot interrupt an arbitrary callback mid-evaluation. Long simulations should use checkpoints in their job model. Stopped job retries skip revalidation and preserve claimed evidence.
