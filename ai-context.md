# AI context for solving user optimization problems with fcmaes-rust

This file is operational context for an AI that must design and implement an
optimization solution with this repository. It is not a claim that one
optimizer is universally best. Use the problem structure, evaluation cost, and
the user's desired output to choose a small set of defensible candidates, then
compare them under the same evaluation budget.

The repository is a Rust 2024 Cargo workspace. `fcmaes-core` is a 100% native
Rust optimizer implementation; it does not wrap, link, load, or invoke the
original fast-cma-es C++ backend. `examples` contains native Rust objective
functions and executable applications; `fcmaes-py` is an optional low-level
PyO3 extension exposing the Rust core, not an alternative implementation.
All optimizers minimize.

## Required workflow for the AI

Before selecting an algorithm, create a problem card containing the following
facts. Inspect the user's code and data where possible. Ask only for facts that
cannot be inferred and would materially change the solution.

| Question | Why it matters |
|---|---|
| What is the decision dimension? | Population size, covariance cost, and useful budgets depend strongly on dimension. |
| What is each variable's lower and upper bound? | Most global algorithms need a finite, meaningful search box. |
| Which variables are continuous, integer, ordinal, categorical, or structured? | DE and MODE have integer mutation masks, but categorical and structured variables still need decoding or repair. |
| Is there one objective, several objectives, or a quality-diversity archive? | This selects scalar optimization, MODE/weighted retry, or MAP-Elites. |
| Which directions are optimized? | Convert maximization to minimization by negating the value. |
| What are the constraints and feasibility convention? | MODE and `moretry` require constraints after objectives, feasible at `g(x) <= 0`. |
| Is a good initial point known? | CMA-ES, CR-FM-NES, PGPE, and local refinement benefit from it. |
| Is the objective smooth, discontinuous, noisy, stochastic, or multimodal? | Distribution search, ranking, restart strategy, and stopping tolerances differ. |
| How long does one evaluation take, and is it thread-safe? | This determines inner batch parallelism versus outer retry parallelism. |
| What is the evaluation, wall-time, memory, and core budget? | An evaluation budget is the fairest algorithm comparison unit. |
| Is a target value known? | Set `stop_fitness`; do not confuse it with `value_limit`. |
| Does the user need one solution, a Pareto front, or diverse behavior niches? | The output requirement is part of the algorithm choice. |

Do not proceed with arbitrary placeholder bounds. If bounds are genuinely
unknown, derive physically meaningful limits, transform the variables, or use
an explicitly supported unbounded path. An unnecessarily wide box can cost
orders of magnitude more evaluations.

For every completed user solution, state the resulting problem card, chosen
algorithm and rejected alternatives, objective/constraint convention,
non-default parameters, evaluation and worker budgets, seeds, validation
method, and reported quality statistics. Code without this configuration
record is not a reproducible optimization solution.

## Fast algorithm-selection decision tree

1. If the user wants the best solution in many behavior niches rather than
   one global optimum, use `Archive` plus MAP-Elites. Add Diversifier after the
   archive has useful coverage.
2. If there are two or more competing objectives:
   - Use MODE when the user wants a Pareto population from one coordinated run,
     especially with explicit constraints or integer-decoded variables.
   - Use `moretry` when many independent scalar optimizer runs are desirable,
     when outer parallelism is important, or when a DE/CMA/BiteOpt pipeline is
     already effective for weighted scalar objectives.
3. For one scalar objective:
   - Start with BiteOpt for a difficult bounded, nonsmooth, multimodal, or
     poorly characterized black box.
   - Start with DE for robust bounded exploration, mixed continuous/integer
     decoding, and DE-to-CMA pipelines.
   - Use active CMA-ES when a useful guess exists, variables are continuous,
     and correlations or ill-conditioning are important.
   - Use CR-FM-NES for higher-dimensional continuous distribution search where
     a full CMA covariance update is unattractive.
   - Use PGPE for high-throughput mirrored batches, diagonal distribution
     search, or noisy objectives where rank-based updates help.
   - Use Dual Annealing for low-to-moderate-dimensional global exploration,
     optionally followed by its bounded local search.
4. If global structure is uncertain or local optima are likely, use independent
   retry. Use coordinated advanced retry when retained elites can usefully
   generate local crossover boxes and increasing budgets.
5. Benchmark at least two plausible choices with identical bounds, objective,
   total evaluations, seeds, and worker resources. A good general comparison
   is BiteOpt versus DE followed by CMA-ES.

## Algorithm comparison

| Algorithm | Best fit | Main limitation | Practical initial configuration |
|---|---|---|---|
| Differential Evolution (`De`) | Bounded global search, discontinuities, mixed decoded variables, restarts | One-shot `optimize` is serial; convergence near a smooth optimum can be slower than CMA-ES | Default population 31; uniform initialization when no guess; give roughly 40% of a DE-to-CMA budget to DE; use ask/tell for external batching |
| Active CMA-ES (`Cmaes`) | Continuous correlated or ill-conditioned variables, refinement from a guess | Full covariance work grows roughly quadratically with dimension; requires a guess and step size | Default population 31; normalized coordinates; initial sigma commonly 0.05–0.3 in normalized optimizer units |
| CR-FM-NES (`Crfmnes`) | Medium/high-dimensional continuous search with complete batch evaluation | Needs a meaningful guess and scalar sigma; not a discrete optimizer | Even population, default 32; use `optimize_batch` and parallelize expensive batches |
| PGPE (`Pgpe`) | High-dimensional mirrored sampling, noisy objectives, parallel batches | Diagonal search distribution does not learn a full covariance | Even population, default 32; ranking on; tune center and standard-deviation learning only after establishing a baseline |
| Dual Annealing (`optimize_da`) | Low/moderate-dimensional global exploration, optional local refinement | No built-in population batching; often less attractive in very high dimensions | Keep local search enabled for deterministic objectives; disable it for very noisy or discontinuous objectives |
| BiteOpt (`BiteOpt`, `DeepBiteOpt`) | Hard bounded black boxes, nonsmoothness, multimodality, minimal tuning | No native general constraint or categorical model; batched ask/tell changes feedback timing | Automatic population; depth 1 first, then try small deep values if the budget is large |
| MODE (`Mode`) | Constrained multi-objective Pareto search | Requires an ask/tell driver and objective-plus-constraint rows | Population 64–256 initially; NSGA-II update; parallelize each asked population |
| Weighted MO retry (`moretry`) | Parallel multi-objective scalarizations using any scalar optimizer | Quality depends on objective scaling and sampled weights; the retained set is not one evolving Pareto population | At least tens of retries; normalize objective magnitudes before selecting weight ranges |
| MAP-Elites (`map_elites`) | Quality-diversity coverage over chosen descriptors | Requires meaningful behavior descriptors and fixed descriptor bounds | Capacity based on desired resolution; chunk at least the worker count; use the 2-D grid fast path when applicable |
| Diversifier (`diversify`) | CMA-ME-style improvement of/filling an existing archive | More expensive than simple emitters and needs an initialized archive | Run MAP-Elites first; default CMA population 31 and stall criterion 20 |

These are starting hypotheses. Objective landscapes dominate generic rules.

## Scalar optimizer parameters

All scalar objective values must be minimized and finite when possible. The
fitness layer replaces non-finite scalar results with `NAN_REPLACEMENT = 1e99`,
but an explicit, scaled penalty is easier to diagnose.

`Fitness::set_normalize(true)` normalizes decision coordinates to `[-1,1]`;
it does not normalize objective or constraint values. Scale objective and
constraint terms explicitly when their magnitudes differ materially.

### Differential Evolution

`DeParams` defaults are:

| Field | Default | Tuning guidance |
|---|---:|---|
| `popsize` | 31 | Increase for broad multimodal exploration or large dimensions; ensure the budget still permits many generations. |
| `max_evaluations` | 100,000 | Set from the user's real budget. |
| `keep` | 200 | Usually leave unchanged; it controls temporal/history behavior. |
| `stop_fitness` | negative infinity | Set only when a genuine target is known. |
| `f` | 0.5 | Higher explores farther; lower takes smaller differential steps. Tune after population and budget. |
| `cr` | 0.9 | Lower can help nearly separable problems; high values allow correlated coordinate changes. |
| `min_mutate` / `max_mutate` | 0.1 / 0.5 | Controls the fraction used by optional integer mutation. |
| `min_sigma` | 0 | Minimum adaptive sampling spread. Usually leave at zero. |
| `seed` / `runid` | 0 / 0 | Set from the experiment or `RetryContext`. |

`De::new(fitness, guess, sigma, integer_mask, params)` uses uniform box
initialization when `guess` and `sigma` are empty. With a guess, supply one
sigma per dimension. Keep `Fitness` normalization disabled for the standard
DE path and provide the guess and sigma in physical variable units; the
canonical runner follows this convention. If normalized fitness is required
by custom ask/tell integration, encode every working point consistently. The
integer mask improves integer-coordinate mutation, but other DE operations
remain continuous: the objective must still perform
authoritative rounding, truncation, lookup, and repair.

### Active CMA-ES

`CmaesParams` defaults are:

| Field | Default | Tuning guidance |
|---|---:|---|
| `popsize` | 31 | Increase for noise or multimodality; decrease cautiously for very expensive objectives. |
| `mu` | 0 | Zero selects half the population. Usually leave automatic. |
| `max_evaluations` | 100,000 | Give enough budget for multiple covariance updates. |
| `accuracy` | 1.0 | Leave at default unless stopping behavior has been validated. |
| `stop_fitness` | negative infinity | Use a known target. |
| `stop_tol_hist_fun` | -1 | Negative selects the automatic history tolerance. Avoid tight tolerances on noise. |
| `update_gap` | -1 | Negative selects the automatic covariance update interval. |
| `seed` / `runid` | 0 / 0 | Use distinct values for independent runs. |

The guess must match the dimension. `input_sigma` has length one (broadcast)
or one value per coordinate. With `Fitness::set_normalize(true)`, the mean is
encoded to `[-1,1]`; provide sigma in that optimizer space. A sigma of 0.1 is
roughly 5% of the full physical span. With normalization off, sigma is in raw
variable units. `Cmaes::optimize(objective, workers)` supports population
parallelism.

### CR-FM-NES

`CrfmnesParams` defaults are population 32, 100,000 evaluations,
`stop_fitness = -infinity`, `penalty_coef = 1e5`, bound-violation handling on,
and seed/run ID zero. The population is forced to at least two. Supply a guess
and scalar sigma. With normalized `Fitness`, supply the guess in physical
coordinates and sigma in normalized optimizer coordinates. The “constraint
violation” option concerns box-bound violations; arbitrary user constraints
still require a penalty or another
constrained method. Use `optimize_batch` for expensive objectives.

### PGPE

`PgpeParams` defaults are:

| Field | Default |
|---|---:|
| `popsize` | 32, rounded up to even |
| `max_evaluations` | 100,000 |
| `stop_fitness` | negative infinity |
| `lr_decay_steps` | 1,000 |
| `use_ranking` | true |
| `center_learning_rate` | 0.15 |
| `stdev_learning_rate` | 0.1 |
| `stdev_max_change` | 0.2 |
| `b1` / `b2` / `eps` | 0.9 / 0.999 / 1e-8 |
| `decay_coef` | 1.0 |
| `seed` / `runid` | 0 / 0 |

Supply a guess and either one standard deviation or one per coordinate. Keep
the guess in physical coordinates; when `Fitness` normalization is enabled,
the standard deviation is in normalized optimizer coordinates. Keep ranking
enabled for a robust first experiment. Increase the even population for noisy
objectives or to expose more parallel work. Change learning rates
one at a time and validate across multiple seeds.

### Dual Annealing

`DaParams` contains `max_evaluations = 100_000`, `use_local_search = true`,
and seed/run ID zero. It supports finite bounds and an explicitly unbounded
path when both bound vectors are empty. Local search uses bounded projected
L-BFGS with finite-difference gradients; objective noise makes those
differences unreliable.

### BiteOpt

`BiteParams` defaults are automatic population (`0`, resolving to
`9 + 3*dimension`), 100,000 evaluations, no finite stop target, automatic
stall criterion, and seed/run ID zero. The deep-mode argument to
`optimize_bite` is separate from `BiteParams`: values at most one select a
plain run; the validated maximum is 36.

Start with depth 1. Test depths 2–6 only when the problem is demonstrably
multimodal and the evaluation budget is large enough to feed multiple internal
populations. Ask/tell batching enables parallel evaluation but delays selector
feedback; compare its convergence with the one-at-a-time path on cheap test
versions.

## Recommended scalar recipes

### Unknown bounded black box

Run two fair baselines:

1. BiteOpt with automatic population and depth 1.
2. DE for approximately 40% of each run budget, then normalized CMA-ES from
   the DE result for the remaining approximately 60%.

The repository's canonical DE-to-CMA restart implementation is
`examples/src/runner.rs`. It consumes every `RetryContext` field, converts
advanced-retry fractional standard deviations to DE's physical units, and
uses normalized standard deviations for CMA-ES.

### Known good continuous guess

Use normalized CMA-ES first. Compare against CR-FM-NES for larger dimensions.
Choose initial per-coordinate sigma from uncertainty in the guess, not from a
fixed constant. If the guess is known within about 10% of each range, a
normalized sigma around 0.1–0.2 is a reasonable first test.

### Expensive objective

Prefer one population run with inner batch evaluation when every generation
is expensive and useful. Choose a population at least as large as the worker
count, normally a small multiple of it. If global restarts are more important,
use outer retry workers and keep inner optimizer workers at one.

### Noisy objective

Use PGPE ranking, larger populations, or repeated evaluations. Do not use a
tight `stop_fitness` or convergence tolerance based on one noisy observation.
If the objective averages replications internally, report both optimizer calls
and the true number of simulations; fcmaes can count only calls it makes.

## Constraints and variable encoding

Use these conventions consistently:

- All objectives are minimized. Negate quantities that must be maximized.
- MODE and `moretry` rows are `[objective_0, ..., objective_m,
  constraint_0, ..., constraint_k]`.
- A constraint is feasible when `constraint <= 0`.
- Convert an equality `h(x) = 0` to `abs(h(x)) - tolerance <= 0` when a
  tolerance is meaningful.
- For scalar optimizers, use decoding/repair for hard structural rules and a
  scaled penalty for residual violations. A common starting form is
  `objective + rho * sum(max(0, g_i(x))^2)` after normalizing terms.
- Return a large finite rejection value for invalid simulations. Reserve NaN
  for truly exceptional paths and test those paths explicitly.
- Encode a bounded integer as a real coordinate, round or truncate it in the
  objective, clamp it, and optionally provide DE/MODE's integer mask.
- Encode a categorical variable as an index into a fixed list. Do not treat
  the numerical distance between unrelated categories as physically real.
- Repair permutations, schedules, and mutually exclusive choices
  deterministically. See the job-shop, scheduling, harvesting, multi-UAV, and
  Mazda examples for decoding patterns.

If a scalar penalty dominates everywhere, normalize the base objective and
constraint residuals before tuning `rho`. If it is too small, infeasible points
win; if it is too large, the landscape becomes nearly flat away from the
feasible boundary.

## Multi-objective optimization

### MODE

Use MODE for a coordinated population. `Mode::try_new` receives a bounded
`Fitness`, number of objectives, number of trailing constraints, an optional
integer mask, and `ModeParams`. Evaluate every row returned by `ask`, preserve
row order, then call `tell` once.

`ModeParams` defaults are population 64, `f = 0.5`, `cr = 0.9`, crossover
probability/index `pro_c = 0.5` / `dis_c = 15`, mutation probability/index
`pro_m = 0.9` / `dis_m = 20`, NSGA-II update enabled,
`pareto_update = 0`, integer mutation range 0.1–0.5, and seed/run ID zero.

Parameter guidance:

- Start with NSGA-II update.
- Use at least population 64. Try 128–512 for difficult fronts, more
  objectives, or cheap objectives; keep enough budget for many generations.
- The DE update requires at least four members.
- Scale objective magnitudes before judging front quality. Crowding distance
  itself is normalized across objectives, but simulation and reporting scales
  still matter.
- Filter feasible rows before calling `pareto_indices` for the objective
  prefix.
- Use `parallel_batch` to evaluate each asked population. Ordered collection
  makes one- and multi-worker seeded runs identical for deterministic
  objectives.

### Weighted multi-objective retry

`MoRetryConfig::new(weight_lower, weight_upper)` defaults to ordinary retry,
zero constraints, p-norm exponent 2, and no value limits. It samples a weight
vector for every retry and retains the unscalarized row and weights.

Use it when scalar runs are effective and independent weight vectors can run
in parallel. Set `ncon` to the number of trailing constraints. A positive
constraint adds its weight as a violation penalty. `value_limits`, when used,
contains one strict upper limit for every objective and constraint.

Normalize or shift objectives to comparable, preferably nonnegative scales
before selecting weight bounds. With a non-integer `value_exp`, negative
weighted values can make the p-norm undefined. Use `pareto_indices` on retained
unscalarized rows; do not report the scalarized retry value as a Pareto
objective.

## Quality diversity

Use MAP-Elites only when behavior descriptors express diversity the user
actually values. A descriptor is not another minimized objective.

1. Define decision bounds and descriptor bounds.
2. Construct `Archive::try_new`.
3. Seed parent candidates with `seed_uniform`.
4. Evaluate that initial population and update the archive.
5. Run `map_elites` or `map_elites_batch`.
6. Optionally run `diversify` or `diversify_batch` to improve/fill niches.
7. Report occupancy, coverage, best fitness, `qd_score`, and representative
   elites—not only the single best point.

`MapElitesParams` defaults are 100 generations, chunk 20, SBX enabled,
`dis_c = 20`, `dis_m = 20`, Iso+LineDD sigmas 0.02/0.2, and zero CMA-emitter
generations. `DiversifierParams` defaults are 100,000 evaluations, population
31, and stall criterion 20.

Archive guidance:

- In two descriptor dimensions, `samples_per_niche = 0` selects the fast
  rectangular grid with O(1) lookup.
- Positive `samples_per_niche` selects k-means CVT centers and nearest-center
  lookup. Use it for non-grid or higher-dimensional descriptor spaces.
- Capacity controls resolution and memory. More niches also require more
  evaluations to obtain useful coverage.
- Set `chunk_size >= workers`; 4–16 times the worker count is a useful
  throughput range for expensive objectives, subject to evaluation cost.
- SBX is a good bounded default. Try Iso+LineDD when local variation along
  elite differences is more appropriate.
- `map_elites_batch` and `diversify_batch` evaluate concurrently, then mutate
  the archive serially in candidate order. `QdBatchFitness` must return exactly
  one `(fitness, descriptor)` pair per input in the same order.

## Retry selection and parameters

### Basic retry

Use `retry` for independent restarts. `RetryConfig` defaults are 1,024 retries,
available parallelism (`workers = 0`), retained capacity 500, no value filter,
no finite stop target, 50,000 evaluations per retry, root seed zero, and no
improvement-history samples.

Set these fields explicitly in production:

- `num_retries`: number of independent runs, at least the worker count.
- `workers`: zero for available cores or a specific outer worker count.
- `max_evaluations`: budget for each run, not the global total.
- `capacity`: number of distinct results to retain; one is enough if only the
  best point matters, but advanced retry and diagnostics benefit from more.
- `value_limit`: store only completed results strictly better than this filter.
- `stop_fitness`: stop claiming new runs after the stored best reaches this
  target.
- `seed`: root of independently spawned persistent worker RNG streams.
- `statistic_num`: maximum retained best-improvement samples.

Total configured work is approximately `num_retries * max_evaluations`, unless
early stopping or algorithm termination reduces it. Always return the actual
evaluation count in `RetryRunResult`.

### Coordinated advanced retry

Use `advanced_retry` when elite crossover and adaptive run budgets are likely
to help. Defaults are 5,000 retries, 1,500 starting evaluations, maximum budget
factor 50, checkpoint interval 100, crossover probability 0.5, and normalized
diversity threshold 0.15.

The optimizer closure must use:

- `context.bounds`, which may be a local crossover box;
- `context.guess`, when present;
- `context.sdev`, as per-coordinate fractional step information;
- `context.max_evaluations`, which grows with retry progress;
- `context.seed` and `context.run_id`;
- `context.value_limit`, which for crossover may require beating a parent.

Ignoring these fields defeats coordinated retry. Start with basic retry for a
new objective, then compare advanced retry under equal total evaluations.

## Parallelism rules

There are two distinct levels:

1. Outer parallelism: `retry`, `advanced_retry`, or `moretry` runs independent
   optimizers on worker threads.
2. Inner parallelism: `Cmaes::optimize`, `Fitness::eval_population*`,
   `parallel_batch`, `map_elites_batch`, or `diversify_batch` evaluates one
   population concurrently.

Worker semantics for inner batches are `1 = serial`, positive values = exactly
that many cached Rayon threads, and non-positive values = the global Rayon
pool. Retry uses `workers = 0` for available parallelism and caps active
workers by the number of retries.

Do not normally enable full outer and full inner parallelism simultaneously.
For 16 cores choose one of these:

- 16 retry workers and one inner worker for many independent searches;
- one optimizer with 16 inner workers for an expensive population batch;
- a deliberate split such as four retries with four inner workers when both
  levels have enough work.

The objective must be `Sync`. Keep read-only model data shared, avoid a mutex
around the expensive computation, and give each stochastic evaluation an
independent deterministic seed. Python callbacks must reacquire the GIL and
may not scale for cheap Python objective bodies.

Single-worker retry is exactly repeatable. Multi-worker retry owns independent
PCG streams, but operating-system scheduling can change which logical worker
claims later retries, so repeated runs need not be bit-identical. Ordered
parallel population batches are deterministic for a deterministic objective.

## Budget and parameter-setting procedure

Tune in this order:

1. Validate objective values, signs, constraints, decoding, and bounds.
2. Transform variables with extreme scale differences. Use log coordinates
   for positive variables spanning orders of magnitude.
3. Choose the algorithm family from the desired result and problem structure.
4. Set the evaluation budget and worker topology.
5. Set population/chunk size so several generations fit in the budget and all
   workers receive work.
6. Set initialization and sigma from real uncertainty.
7. Only then tune algorithm-specific mutation, crossover, or learning rates.

Use a staged budget:

- Smoke test: a few populations, enough to exercise decoding and reporting.
- Pilot: several seeds for two or three candidate algorithms.
- Production: allocate the measured wall-time/evaluation budget to the best
  robust configuration, retaining independent validation seeds.

Population algorithms check limits at generation boundaries, so reported
evaluations can exceed the configured limit by part of a population. Use the
reported evaluation count for comparisons.

## Minimal implementation patterns

### Bounded scalar DE

```rust
use fcmaes_core::{De, DeParams, Fitness};

let dim = 8;
let lower = vec![-5.0; dim];
let upper = vec![5.0; dim];
let objective = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
let fitness = Fitness::bounded(dim, 1, &lower, &upper);
let parameters = DeParams {
    max_evaluations: 50_000,
    seed: 1,
    ..Default::default()
};
let result = De::new(fitness, &[], &[], None, &parameters).optimize(&objective);
assert!(result.y.is_finite());
```

### Parallel MODE population

```rust
use fcmaes_core::{parallel_batch, Fitness, Mode, ModeParams};

let dim = 6;
let fitness = Fitness::bounded(dim, 2, &[0.0; 6], &[1.0; 6]);
let mut mode = Mode::try_new(
    fitness,
    2, // minimized objectives
    0, // trailing constraints
    None,
    &ModeParams { popsize: 128, seed: 1, ..Default::default() },
)?;
let xs = mode.ask();
let ys = parallel_batch(&xs, 16, |x| {
    vec![x.iter().sum(), x.iter().map(|v| (v - 0.5).powi(2)).sum()]
});
mode.tell(&ys);
# Ok::<(), &'static str>(())
```

### Batch MAP-Elites

```rust
use fcmaes_core::{
    map_elites_batch, parallel_batch, Archive, MapElitesParams, Rng,
};

let mut rng = Rng::new(1);
let lower = vec![-1.0; 4];
let upper = vec![1.0; 4];
let mut archive = Archive::try_new(4, &[-1.0; 2], &[1.0; 2], 256, 0, &mut rng)?;
archive.seed_uniform(&lower, &upper, &mut rng);
let mut batch = |xs: &[Vec<f64>]| {
    parallel_batch(xs, 16, |x| {
        let fitness = x.iter().map(|v| v * v).sum();
        (fitness, vec![x[0], x[1]])
    })
};
let initial = archive.xs().to_vec();
archive.update_batch(&initial, &mut batch)?;
archive.argsort();
map_elites_batch(
    &mut archive,
    &mut batch,
    &lower,
    &upper,
    &MapElitesParams { generations: 1_000, chunk_size: 128, ..Default::default() },
    &mut rng,
)?;
# Ok::<(), &'static str>(())
```

For retry and the DE-to-CMA sequence, copy the maintained patterns from
`docs/retry.md` and `examples/src/runner.rs` rather than rebuilding seed and
context handling ad hoc.

## Validation and reporting checklist
Before declaring success, the AI should:

- Unit-test decoding, objective signs, constraint signs, bounds, and known
  reference points.
- Confirm every optimizer result is decoded and re-evaluated independently.
- Run `cargo test --workspace` and use `--release` for timings.
- Record algorithm, all non-default parameters, bounds, seed, workers,
  configured budget, actual evaluations, and wall time.
- Compare algorithms using the same objective calls and resource limits.
- Use multiple seeds and report at least best, median or mean, standard
  deviation, and success/feasibility rate.
- For multi-objective runs, report feasible Pareto points and a suitable front
  quality measure; do not compare only one selected point.
- For QD runs, report capacity, occupied niches, coverage, best fitness,
  `qd_score`, and descriptor ranges.
- Check scaling by testing workers 1 and N. Deterministic ordered batches should
  produce the same values; parallel retry may be statistically rather than
  bitwise reproducible.
- Preserve a small deterministic smoke configuration in tests.

## Failure diagnosis

| Symptom | Likely cause | First action |
|---|---|---|
| No improvement from the initial population | Wrong sign, invalid decoding, huge flat penalty, or bounds too broad | Print and test several decoded points and individual objective terms. |
| All results are infeasible | Constraint sign error or penalty scale too weak/strong | Verify feasible means `<= 0` and test a known feasible point. |
| CMA-ES immediately hits bounds | Guess or sigma is in the wrong coordinate scale | Check whether `Fitness` normalization is enabled and rescale sigma. |
| DE consumes budget without refinement | Population too large for the budget or smooth local convergence is needed | Reduce population or hand the best point to CMA-ES. |
| Parallel run is slower | Objective is too cheap, batches are too small, or worker pools are nested | Use one worker level and increase work per batch. |
| Multi-objective front covers only one extreme | Objective scaling/weight ranges are poor or population/retries are too small | Normalize objectives and increase MODE population or scalarization diversity. |
| MAP-Elites coverage stays low | Descriptor bounds are wrong, capacity is too high, or invalid descriptors are returned | Inspect descriptor distributions and reduce capacity for the pilot. |
| Noisy runs stop inconsistently | Target/tolerance is tighter than noise | Increase replication/population and use robust aggregate reporting. |

## Repository references

- `docs/getting-started.md`: building and basic Rust use.
- `docs/optimizers.md`: public optimizer interfaces and defaults.
- `docs/retry.md`: basic, advanced, and weighted retry.
- `docs/architecture.md`: objective flow, normalization, and concurrency.
- `docs/examples.md`: native application and benchmark commands.
- `examples/src/runner.rs`: canonical DE-to-CMA retry integration.
- `examples/src/bin/mazda_mo.rs`: parallel constrained MODE driver.
- `examples/src/bin/mazda_qd.rs`: parallel MAP-Elites/Diversifier driver.
- `examples/src/uav.rs`: random-key decoding for mixed assignment, ordering,
  scalar, and multi-objective optimization.
- Generated rustdoc: `cargo doc --workspace --no-deps --open`.

When code and this guide disagree, treat the current public Rust API and tests
as authoritative, update the guide with the implementation, and document the
reason for the change.
