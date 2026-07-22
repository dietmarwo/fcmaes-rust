# Optimizers

## Choosing an optimizer

| Optimizer | Style | One-shot | Stateful | Useful starting point |
|---|---|---:|---:|---|
| Differential Evolution | Population, global | `De::optimize` | `ask` / `tell` | Robust bounded search and retry pipelines |
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

`Cmaes::optimize(objective, workers)` controls population-evaluation
parallelism. Use `1` when outer retry already supplies parallelism.

## CR-FM-NES

`Crfmnes` evaluates complete populations through a batch closure:

```rust
let result = optimizer.optimize_batch(|rows| {
    rows.iter().map(|x| objective(x)).collect()
});
```

Defaults include population 32, 100,000 evaluations, penalty coefficient
`1e5`, and constraint-violation handling enabled. `ask_pop` returns decoded,
in-bound rows; `tell_pop` consumes one value per row.

## PGPE

`Pgpe` uses mirrored sampling, ranking, and ADAM updates for the distribution
center. Defaults include population 32, ranking enabled, center learning rate
0.15, standard-deviation learning rate 0.1, and 100,000 evaluations.

The batch and ask/tell interfaces mirror CR-FM-NES. The optimizer tracks the
true objective best even when ranking is used for updates.

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
vectors select the unbounded path.

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

Population optimizers check budgets at generation boundaries, so actual
evaluation totals can overshoot the configured limit by part of a population.
Use reported evaluations for accounting.

The retry root seed spawns one persistent, independent PCG stream per worker;
each retry draws its optimizer seed from its worker's stream. A single-worker
run is exactly repeatable. In a multi-worker run, scheduling can change how
many retries each worker stream consumes, so repeated results need not be
bit-identical. Rust is also not expected to reproduce historical C++ or Python
streams exactly; port parity is statistical.
