# AI context for solving user optimization problems with fcmaes-rust

Last reviewed 2026-10-19 against `main` at `67bd698`, which includes the
shared agent job API 0.1 (#5), the CMA-ES, MODE and MAP-Elites job profiles,
the 0.1.1 synchronization of the shared crates and the result contract 0.2
(evidence, refutations, evaluation profiles, error codes and JSON Schemas),
plus application models, parallel population evaluation, CR-FM-NES, PGPE,
CMA-ES restarts, CVT archives and CMA-ME/CMA-MAE emitters in the fcmaes
worker. In another checkout, check `Cargo.toml`, the available worker's
discovery output, and the linked guides before using a command or claiming a
capability.

This file is operational context for an AI that must design and implement an
optimization solution with this repository. It is not a claim that one
optimizer is universally best. Use the problem structure, evaluation cost, and
the user's desired output to choose a small set of defensible candidates, then
compare them under the same declared objective-call and compute-resource
protocol.

The repository is a Rust 2024 Cargo workspace. `fcmaes-core` is a 100% native
Rust optimizer implementation; it does not wrap, link, load, or invoke the
original fast-cma-es C++ backend. `examples` contains native Rust objective
functions and executable applications; `fcmaes-py` is an optional low-level
PyO3 extension exposing the Rust core, not an alternative implementation.
All optimizers minimize.

The public registry surfaces are deliberately smaller than the GitHub
repository:

- Rust users depend on the published `fcmaes-core` crate. Rust 1.88 is the
  minimum supported toolchain; CI checks the whole workspace with it.
- Python users install the `fcmaes-rust` distribution and import
  `fcmaes_rust`. Published wheels support CPython 3.11 through 3.13.
- `examples`, `tutorials`, and the internal native `fcmaes-gtop` source are
  GitHub-only and are not separate published packages.

Keep distribution and API names distinct:

```text
cargo add fcmaes-core          -> use fcmaes_core
python -m pip install fcmaes-rust -> import fcmaes_rust
```

Installing a compatible CPython wheel requires neither a Rust toolchain nor a
C/C++ compiler. Building the Python package from its source distribution does
require Rust. Do not tell a Python user to install the historical `fcmaes`
package when they asked specifically for this Rust implementation.

The workspace version is 0.1.5, which is also the current published release
of `fcmaes-core` (crates.io) and `fcmaes-rust` (PyPI) at this review. The
job-layer crates (`optimization-job-api`, `optimization-runner`,
`fcmaes-job-worker`, `fcmaes-job-api`, `fcmaes-runner`,
`fcmaes-native-worker`) are source-only (`publish = false`) and are part of
neither package. Schema versions such as `optimization.job/0.1` and
`fcmaes.retry/1` are protocol identifiers, independent of crate versions.
Use `CHANGELOG.md`, package manifests, and the actual registry metadata when
discussing releases; a workspace version alone does not establish publication.

The existing optimizer guidance includes the 0.1.4 additions: audited
multi-objective quality indicators, schedule-independent retry seeds, exact
ragged MAP-Elites grid layouts, non-dominated-sort/crowding utilities, the
optimizer-boundary evidence, and the Foundations and Lennard-Jones studies.
The twenty-two application tutorials, Foundations, and the new operational
[job API tutorial](tutorials/job-api/README.md) serve different purposes; do not
count the job tutorial as another application benchmark.

The 0.1.5 changelog records two implementation diagnostics that affect algorithm
advice. The equal-wall GTOP campaign wraps the external serial `cmaes` crate
with generic parallel retry: at the same four-second wait, retry improves mean
quality on all seven problems and wins 86–96 of 100 paired seeds, while using
the otherwise idle cores for more independent starts. This is evidence for
retry's execution model, not proof that every problem benefits equally. The
controlled active-CMA-ES comparison is an implementation diagnostic rather
than a general CMA-ES performance benchmark.

Population ask/tell is the complementary way to spend cores. Parallelize one
asked population when objective evaluation is expensive and feedback can wait
for the batch; use outer retry when independent starts and early completed
runs are more valuable. Every fcmaes population optimizer exposes an ask/tell
boundary; Dual Annealing is the sequential non-population exception. External
optimizers can use generic retry without becoming `fcmaes-core` dependencies.

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
| For a stochastic objective, can simulations accept explicit seeds? | Fixed common-random-number seed sets make candidate comparisons fair; disjoint seeds are needed for validation. |
| How long does one evaluation take, and is it thread-safe? | This determines inner batch parallelism versus outer retry parallelism. |
| What is the evaluation, wall-time, memory, and core budget? | Equal calls alone can favor a sequential method or hide surrogate overhead; declare both call and resource accounting. |
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

## First decide whether fcmaes is appropriate

Use the human-facing
[`Choosing an optimizer`](docs/choosing-an-optimizer.md) guide to route
structured, differentiable, evaluation-scarce, real-time, and
structure-evolution problems to more suitable tools before selecting an
fcmaes algorithm.

Do not select a gradient-free optimizer merely because the model is a
simulation. Prefer a gradient-based optimizer when all decision variables are
continuous, the end-to-end objective is differentiable, reliable analytical,
automatic-differentiation, adjoint, or sensitivity gradients are available,
and the required result is one optimum or a smooth Pareto trade-off. Those
gradients usually carry much more information per evaluation.

fcmaes is a strong candidate when one or more of these conditions holds:

- gradients are unavailable, unreliable, or dominated by simulation noise;
- event appearance/disappearance, contact, failure, clipping, repair, routing,
  integer choices, or categorical decoding makes the objective nonsmooth;
- the landscape is multimodal and needs global exploration or restarts;
- the result must be a repertoire of behaviorally diverse solutions rather
  than one optimum; or
- a robust aggregate such as a worst case, quantile, or failure rate creates
  nonsmooth outer logic.

Events alone do not invalidate gradients. Diffsol, for example, provides
forward and adjoint sensitivities, so its smooth parameter-fitting tutorials
are naturally paired with L-BFGS or another gradient-based optimizer. Consider
fcmaes around such a model only when discrete policies, resets, robust maxima,
solver failures, or other end-to-end discontinuities make those sensitivities
misleading or unavailable. See
[`tutorials/README.md#23-diffsol-why-gradients-are-the-better-default`](tutorials/README.md#23-diffsol-why-gradients-are-the-better-default).

The core boundary is intentional. Standalone local derivative-free methods,
Bayesian optimization, and gradient solvers remain external and can be driven
through retry's optimizer closure. The corrected 20-seed
[optimizer-boundary experiment](docs/optimizer-boundary.md) found no general
DE→Nelder–Mead advantage and only a narrow, small-budget BO regime. Do not add
an optimizer dependency to core merely because one application needs it.

## Fast algorithm-selection decision tree

1. If valid, economical gradients exist for the complete decision-to-objective
   map, start with a gradient-based method and use fcmaes only as a justified
   global, discrete, robust, or QD outer layer.
2. If the user wants the best solution in many behavior niches rather than
   one global optimum, use `Archive` plus MAP-Elites. Add Diversifier after the
   archive has useful coverage.
3. If there are two or more competing objectives:
   - Use MODE when the user wants a Pareto population from one coordinated run,
     especially with explicit constraints or integer-decoded variables.
   - Use `moretry` when many independent scalar optimizer runs are desirable,
     when outer parallelism is important, or when a DE/CMA/BiteOpt pipeline is
     already effective for weighted scalar objectives.
4. For one scalar objective:
   - Start with BiteOpt for a difficult bounded, nonsmooth, multimodal, or
     poorly characterized black box.
   - Start with DE for robust bounded exploration, mixed continuous/integer
     decoding, and DE-to-CMA pipelines.
   - Use active CMA-ES when a useful guess exists, variables are continuous,
     and correlations or ill-conditioning are important.
   - Use CR-FM-NES for higher-dimensional continuous distribution search where
     a full CMA covariance update is unattractive.
   - Use PGPE for high-throughput mirrored batches, diagonal distribution
     search, or noisy objectives where rank-based updates help. See the
     fixed-topology
     [neural-controller tutorial](tutorials/neural-controller-policy-search/)
     for an equal-protocol PGPE/CR-FM-NES comparison.
   - Use Dual Annealing for low-to-moderate-dimensional global exploration,
     optionally followed by its bounded local search.
5. If global structure is uncertain or local optima are likely, use independent
   retry. Use coordinated advanced retry when retained elites can usefully
   generate local crossover boxes and increasing budgets.
6. Benchmark at least two plausible choices with identical bounds, objective,
   seeds, and a declared call/wall-resource protocol. A good general
   comparison is BiteOpt versus DE followed by CMA-ES. When algorithms have
   materially different sequential overhead or parallelism, equal objective
   calls are not an equal wall-time comparison.

## Choose the integration surface

| Need | Entry point | Scope |
|---|---|---|
| Custom Rust objective, full optimizer selection, ask/tell, MODE or QD | `fcmaes-core` | Direct in-process numerical API; the recipes below apply here. |
| Durable agent-controlled experiments with registered models | `optimization-run` and `fcmaes-job-worker`, or an application worker built on it | Common jobs, CPU admission, journals, cancellation and result manifests. |
| Existing `fcmaes.job/0.1` jobs | `fcmaes-run` and `fcmaes-native-worker` | Preserved legacy contract; keep its input and result schemas. |
| Python code calling Rust optimizers | `fcmaes_rust` | Optional PyO3 API, with the callback/GIL constraints below. |

Use the common job API for agent-driven experiments when its worker supports
the problem and algorithm. Its fcmaes profiles expose independent BiteOpt, DE,
CMA-ES (with optional IPOP/BIPOP restarts), CR-FM-NES and PGPE retries,
independent MODE runs and independent MAP-Elites runs (grid or CVT archives,
optional CMA-ME/CMA-MAE emitters), each on registered native models. An
application objective becomes a registered model in an application worker
([application models](docs/job-api/models.md)): a Rust closure with
parameters and SHA-256-checked data inputs, never a callback in the job. Dual
Annealing, advanced retry, `moretry` and the Diversifier remain direct-API
features.

Prefer a native Rust objective when evaluation throughput matters. Put
read-only model data behind shared references or `Arc`, give each evaluation
isolated mutable state, and let fcmaes distribute candidates. This avoids
Python callback and serialization overhead and is the design demonstrated by
the application examples and native application tutorials.

The Buckingham–Pi example is also a useful pattern for optimized feature
discovery. It parameterizes every dimensionally valid exponent matrix as
`E = Ns * C`, scores regressions on a disjoint holdout set, rejects overflowing
log-features, and penalizes nearly dependent groups. Use this pattern when
algebraic structure can make all optimizer trials valid by construction.
Do not call an in-sample regression score “cross-validation,” and distinguish
the coefficient of variation from cross-validation in output and reports. See
`docs/buckingham-pi.md`.

Python users can still use the Rust optimizers through `fcmaes_rust`.
One-shot functions include `optimize_de`, `optimize_acma`,
`optimize_crfmnes`, `optimize_pgpe`, `optimize_da`, and `optimize_bite`;
stateful `DE`, `ACMA`, `CRFMNES`, `PGPE`, `Bite`, `MODE`, and `Archive`
classes expose the corresponding ask/tell or archive workflows. Retry is
available through `minimize_retry`, `minimize_advanced_retry`, and
`minimize_moretry`. Treat this as a low-level NumPy-oriented API and consult
`docs/python-bindings.md` for the authoritative signatures and result layouts.

The native optimizer loop releases the GIL, but every Python objective callback
must reacquire it. Cheap objectives written in Python therefore rarely scale
across threads. Python callbacks dominated by NumPy, another native extension,
or external work may scale only if that work releases the GIL. When the
objective itself can be implemented efficiently in Rust, prefer the direct
`fcmaes-core` path.

## Shared agent job API

Start with [the common API guide](docs/job-api.md) and
[the fcmaes tutorial](tutorials/job-api/README.md). The agent selects a supported
model, writes a typed job, validates it, runs it, and inspects numerical results.
The objective and optimizer stay together in the native worker process; there
is no agent, Python, or network round trip for each objective evaluation.

### Architecture, discovery and compatibility

| Component | Responsibility |
|---|---|
| `crates/optimization-job-api` | Strict envelopes, result contracts, hashes, durable journals and worker SDK. |
| `crates/optimization-runner` / `optimization-run` | Worker registration, staging, idempotency, CPU admission, process supervision and recovery. |
| `crates/fcmaes-job-worker` | Native adapter for `fcmaes.retry/1` (BiteOpt, DE, CMA-ES with restarts, CR-FM-NES, PGPE retries), `fcmaes.mode/1` (MODE), `fcmaes.map-elites/1` (MAP-Elites, CVT, CMA-ME/CMA-MAE) and `fcmaes.evaluate/1` (given points), on the models of a `Registry`; a library, a thin binary with the benchmarks, and `examples/custom_worker.rs` for application models. |
| `crates/fcmaes-job-api` | Compatibility facade re-exporting the original API from the shared crate's `legacy` module. |
| `crates/fcmaes-runner` / `fcmaes-run` | Compatibility CLI entry point into the common runner. |
| `crates/fcmaes-native-worker` | Original worker for `fcmaes.job/0.1` specifications. |

[LP-IP-craft](https://github.com/dietmarwo/LP-IP-craft) and
[SQP-craft](https://github.com/dietmarwo/SQP-craft) carry their own copies of
the shared crates and their own workers: `lp-ip-job-worker` with
`lp-ip.solve/1` and `lp-ip.evaluate/1`, and `sqp-job-worker` with `sqp.solve/1`
and `sqp.evaluate/1`. Each has its own
`ai-context.md` for its solver. Build those workers in their respective
repositories and register their executables with one runner using one absolute
node root. The fcmaes checkout builds independently and does not link either
other solver.

The copies are byte-identical: each repository records them in
`docs/job-api/SHARED-SOURCES.sha256`, and every repository's CI checks its
manifest ([this repository's](docs/job-api/SHARED-SOURCES.sha256)). Version
0.1.1 of the shared crates synchronized the three copies, which had drifted
after the 0.1.0 delivery. It carries every review fix: the zero-work repair of
a job whose launcher died while queued, staging under the node root instead of
`TMPDIR`, sealing an attempt with an unreadable journal `failed` instead of
leaving it `running`, appends that end a torn journal tail first, validators
stopped by a signal, `./`-relative input paths, `optimization-job-ID.service`
unit names, and no reboot flag on finished jobs. It also carries the fcmaes
`non_dominated_set` and `elite_archive` claims. `scripts/job-api/test_review.py`
covers the runner fixes in all three repositories.

Version 0.2.0 of the shared crates adds the result contract
`optimization.result/0.2` and worker protocol `optimization.worker/0.2`: five
new claims (`infeasible`, `unbounded`, `infeasible_or_unbounded`,
`locally_infeasible`, `certified_optimum`), `evidence` with checks,
refutations and a certificate slot, `objective_bounds`, an `imprecise` flag,
coded runner errors and JSON Schemas in `docs/job-api/schemas/`. The job
envelope stays `optimization.job/0.1`. A 0.2 runner still accepts 0.1
workers; a 0.1 runner rejects 0.2 workers at registration. The workers report
the shared version as `build.job_api_version`. Use one runner build for all
jobs that share a node root. A cross-project change must update all three
copies and their manifests.

The new runner defaults to `.optimization-jobs`; `fcmaes-run` retains
`.fcmaes-jobs`. Root precedence is `--root`, `OPTIMIZATION_RUN_ROOT`, then
`FCMAES_RUN_ROOT`, then the entry point's default. Separate roots have separate
CPU ledgers. Use the same explicit absolute root for all jobs sharing a budget.

Linux is required for the job layer. CI checks the whole workspace, including
the job layer, with Rust 1.88 and tests it with current stable. Python 3 is
only needed for tutorial/test drivers. Use the lockfile:

```sh
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
cargo build --release --locked -p fcmaes-job-worker --example custom_worker
python3 tutorials/job-api/run.py --root /tmp/fcmaes-job-tutorial
```

The tutorial runs BiteOpt on Sphere, Rosenbrock and Cassini1, CMA-ES on
Rosenbrock, BIPOP-CMA-ES on Cassini1, CR-FM-NES on 50-dimensional Rosenbrock,
MODE on ZDT1 and the constrained Binh–Korn problem, MAP-Elites on the
Rastrigin projection benchmark and, on a CVT archive with CMA-MAE, on the
sphere projection, and a curve fit to a data file in the example application
worker. It checks every result independently in Python and writes
`tutorial-report.json` under the node root. For manual
control, initialize a fresh node once and register its worker:

```sh
JOB_NODE="$PWD/.optimization-node"
./target/release/optimization-run --root "$JOB_NODE" node init --capacity 8
./target/release/optimization-run --root "$JOB_NODE" workers add \
  --id fcmaes --executable ./target/release/fcmaes-job-worker
./target/release/optimization-run --root "$JOB_NODE" workers list
./target/release/optimization-run --root "$JOB_NODE" problems describe fcmaes
./target/release/optimization-run --root "$JOB_NODE" validate \
  --spec tutorials/job-api/sphere.json
./target/release/optimization-run --root "$JOB_NODE" run \
  --spec tutorials/job-api/sphere.json
```

`problems describe fcmaes` describes the registered worker ID. Use discovery to
check supported profiles, models, optimizer parameters and resource limits;
do not infer job options from the broader core API. Each problem entry names
its profile:

| Profile | Registered models of the stock worker |
|---|---|
| `fcmaes.retry/1` | `benchmark.sphere`, `benchmark.rosenbrock`, `gtop.cassini1` |
| `fcmaes.mode/1` | `benchmark.zdt1` (2 objectives), `benchmark.dtlz2` (3 objectives), `benchmark.binh-korn` (2 objectives, 2 constraints) |
| `fcmaes.map-elites/1` | `benchmark.sphere-projection`, `benchmark.rastrigin-projection` (2-D linear-projection descriptors) |

An application worker lists its own models next to these, each with its
`kind`, `profile` and description (dimensions, parameters, inputs).

### Job specifications and repeatability

This complete specification has the same model, budget and resource layout as
the shipped Sphere tutorial, with its own request key:

```json
{
  "schema_version": "optimization.job/0.1",
  "request_key": "context.fcmaes.sphere.v1",
  "worker": "fcmaes",
  "payload_schema": "fcmaes.retry/1",
  "resources": {"cpu_slots": 8, "outer_workers": 8, "inner_threads": 1},
  "limits": {"wall_time_seconds": 30.0, "grace_seconds": 1.0},
  "inputs": [],
  "payload": {
    "problem": {
      "id": "benchmark.sphere",
      "version": "1",
      "config": {"dimension": 10},
      "bounds": {"lower": [-5.0], "upper": [5.0], "broadcast": true}
    },
    "optimizer": {"kind": "biteopt", "parameters": {"population": null}},
    "seed": 42,
    "retries": 16,
    "evaluations_per_retry": 1000,
    "evaluation_limit_mode": "hard"
  }
}
```

Unknown fields and unsupported profile/options are errors. Leave `task_count`
unset; the worker resolves it from `retries` (or `runs`). `validate` resolves
defaults but does not optimize or reserve slots. `run` and `submit` also check
node capacity.

The other profiles share the envelope and the `problem` shape and differ in the
payload. The tutorial files `zdt1-mode.json`, `binh-korn-mode.json` and
`rastrigin-map-elites.json` are complete examples:

```json
{"problem": {"id": "benchmark.zdt1", "version": "1", "config": {"dimension": 10},
             "bounds": {"lower": [0.0], "upper": [1.0], "broadcast": true}},
 "optimizer": {"kind": "mode", "parameters": {"population": 64}},
 "seed": 42, "runs": 8, "generations": 250, "reference_point": [1.1, 1.1]}
```

```json
{"problem": {"id": "benchmark.rastrigin-projection", "version": "1", "config": {"dimension": 10},
             "bounds": {"lower": [-5.12], "upper": [5.12], "broadcast": true}},
 "archive": {"capacity": 400},
 "optimizer": {"kind": "map_elites", "parameters": {"chunk_size": 20, "variation": "sbx"}},
 "seed": 42, "runs": 8, "generations": 1000}
```

MODE takes `population` (4–10,000, or 4–1,000 with three objectives, where the
exact hypervolume grows cubically; default 64), `nsga_update`,
`pareto_update`, `f`, `cr`, `pro_c`, `dis_c`, `pro_m` and `dis_m`, defaulting to
`ModeParams`; `reference_point` defaults to the problem's discovery entry.
MAP-Elites takes `chunk_size` (even, default 20) and `variation` (`sbx` with
`dis_c`/`dis_m`, or `iso_dd` with `iso_sigma`/`line_sigma` in raw variable
units); the settings of the other variation are rejected. Its archive has
`capacity` niches (1–10,000): a regular grid for two-dimensional descriptors
(`archive.kind: grid`, the default there) or a CVT archive (`kind: cvt`, the
default for other descriptor dimensions, with `samples_per_niche` and a
`seed` shared by all runs). With `cma_generations > 0`, a second phase runs
`emitters` CMA-ES emitters of `population` each: `emitter: cma_me` (default,
CMA-ME's improvement ranking with new niches first) or `cma_mae` with
`archive_learning_rate` and a required `threshold_init`.
Bounds of these models must lie inside their canonical boxes.

A request key is an idempotency key, not a label for unlimited reruns. The same
key with the same resolved specification and staged worker binary returns the
existing attempt. Changed content or binary identity under that key is a
conflict. Use a new key for a new attempt, including a fresh run after failure.
Keep resolved specifications and worker hashes with every reported result.

Objectives are compiled Rust models: JSON cannot carry an executable path,
function body or Python callback. A registered model may read named `inputs`
(data files with relative contained paths and required SHA-256 hashes, staged
by the runner) and `problem.parameters`; the stock models read neither, so
their jobs keep `inputs: []`. The worker rejects parameters and inputs the
model does not read, and records every parameter value it used in the
resolved job. Use the model's discovery entry rather than assuming any file
or parameter is supported.

### Eight-slot execution, seeds and evaluation budgets

The common envelope requires positive integers and
`outer_workers * inner_threads <= cpu_slots <= 1024`. `outer_workers` runs
tasks concurrently; `inner_threads` evaluates one task's populations in
parallel, on threads of that task's own:

| Mode | `cpu_slots` | `outer_workers` | `inner_threads` |
|---|---:|---:|---:|
| Eight concurrent fcmaes retries or runs | 8 | 8 | 1 |
| Two runs with expensive evaluations | 8 | 2 | 4 |
| Serial fcmaes comparison | 1 | 1 | 1 |

CMA-ES, CR-FM-NES, PGPE, MODE and MAP-Elites accept `inner_threads > 1`, and
their results do not depend on it: populations are evaluated in parallel and
returned in order, while the optimizer state stays serial. BiteOpt, DE and
`fcmaes.evaluate/1` evaluate one point at a time and require
`inner_threads=1`. Core ask/tell remains available through the direct
Rust/Python APIs; the job API deliberately has no ask/tell profile (see
[application models](docs/job-api/models.md#why-there-is-no-callback-or-asktell-profile)).

The node reserves all `cpu_slots` for an admitted attempt, even while fewer
tasks are active. Slots account for cooperative numerical concurrency; they
are not CPU affinity, cgroup enforcement or the exact OS thread count. The
supervisor and I/O may use additional threads. Do not claim a speedup from slot
allocation alone. Compare actual wall time, completed work and solution quality.

Each task (retry or run) uses `retry_run_seed(root_seed, task_id)`. For
deterministic registered models, the same completed task has the same result
regardless of worker count or scheduling. Deadlines/cancellation can change which tasks complete. Record
task IDs and seeds when comparing serial and parallel runs. The common and
legacy workers share one solver and objective registry and the same seed
derivation, so the same problem, optimizer, seed and budget give identical
per-retry results in both profiles; the CLI tests check this.

`retries` and `evaluations_per_retry` must be positive. BiteOpt requires `hard`
evaluation limits. DE requires `soft`: with population P, the declared allowance
is at most `2P-1` extra evaluations per retry. Switching from BiteOpt to DE means
changing both `optimizer.kind` to `de` and `evaluation_limit_mode` to `soft`.
CMA-ES (`kind: cmaes`) also requires `soft`: it evaluates whole generations, so
it may spend at most `population-1` extra evaluations, and it can stop earlier
on its own convergence tests (`termination: "optimizer_stop"`,
`details.stop_code`). Its parameters are `population` (default 31) and `sigma`
(default 0.3, the initial step size in the box normalized to `[-1, 1]`); each
retry starts at a uniform random point drawn from its seed. With
`restarts: ipop` or `bipop`, CMA-ES restarts from new random points with
doubled (IPOP) or alternating large and small (BIPOP) populations until the
budget is spent; the overshoot is then at most `max_population-1` (default
`max_population`: 512 × `population`, at most 10⁶ and 10⁷ / dimension).
CMA-ES accepts at most 3,162 variables; CR-FM-NES and PGPE scale further. CR-FM-NES (`kind: crfmnes`, for high
dimensions) and PGPE (`kind: pgpe`) also need `soft` and an even
`population` (default 32), with `sigma` defaults 0.3 and 0.1; PGPE also takes
`center_learning_rate` and `stdev_learning_rate`. BiteOpt and DE payloads are
unchanged by these additions, so their job digests are too.

MODE and MAP-Elites budgets are exact: `generations × population` and
`generations × chunk_size + cma_generations × emitters × population`
evaluations per run. Validation then re-evaluates
every evaluated member of MODE's final population, or every MAP-Elites elite,
once.
Read actual `work.evaluations`; independent candidate checks are counted
separately as `validation_evaluations`. The wall limit covers the whole attempt
after CPU admission, not each retry, and excludes time waiting for admission.

### Inspect, cancel and recover

`run` is foreground execution. Read its final JSON object, including `id`,
`directory` and lifecycle `state`. A successful CLI exit alone does not establish
successful optimization: cancellations, timeouts and solver failures can still
produce a normal job response. Replace the placeholders below with returned IDs
and paths, and keep using the same `JOB_NODE`:

```sh
./target/release/optimization-run --root "$JOB_NODE" jobs list
./target/release/optimization-run --root "$JOB_NODE" jobs get JOB_ID
./target/release/optimization-run --root "$JOB_NODE" jobs events JOB_ID
./target/release/optimization-run --root "$JOB_NODE" jobs wait JOB_ID --timeout 60
./target/release/optimization-run --root "$JOB_NODE" result JOB_DIRECTORY
./target/release/optimization-run --root "$JOB_NODE" jobs artifact JOB_ID results
./target/release/optimization-run --root "$JOB_NODE" jobs cancel JOB_ID
```

fcmaes retries check cancellation between retries; an active retry can continue
to its evaluation budget. MODE and MAP-Elites runs check it between generations
and return their current front or archive with `termination: "cancelled"`; at
the wall deadline the SDK uses the same signal, so the job state says whether
it was `cancelled` or `timed_out`. `jobs cancel JOB_ID --now` permits at most one further
second before process-group termination. The runner also enforces the wall
deadline and grace period. Forced termination retains committed task records,
but the work spent in an unfinished retry is unknown.

Use `jobs reconcile` to reconcile reservations and `jobs repair JOB_ID` to
recover a stopped attempt from checked journals. Repair refuses live workers
or unknown liveness. It can tolerate a torn final journal fragment. A journal
with earlier corruption or duplicate task IDs is never trusted: the attempt is
sealed `failed` with no results, `work` null and `runner.journal_error`, and
`jobs list` reports the error for that job only. Repair preserves a consistent
final worker manifest when available; otherwise it seals the attempt as
interrupted.
A job whose launcher died while it was still queued never started a worker:
repair seals it as interrupted with zero work, without a liveness proof.
Neither repair nor rerunning a request key resumes solver state. Start a fresh
attempt under a new key when that is required.

Detached `submit --spec JOB.json` requires a real Linux systemd user manager;
run `node doctor` first. Logout survival depends on host configuration, including
lingering. Detached execution has not yet been accepted on a real service host;
CI runners have no systemd user manager. Do not claim that an offline or
overnight job will survive disconnects until the actual host passes
`scripts/runner/test_systemd.py` (a legacy-profile job through the same runner)
and its logout behavior has been checked. Foreground `run` is the validated
starting path.

### Interpret results and compare experiments

Common `run.json` uses `optimization.result/0.2`; its `status` and the job's
`state` describe execution, while each `results` entry contains `task_id`,
`outcome` and `work`. Inspect all three layers. A completed job is not a proof
of optimality. Each outcome separates `termination` (why the task stopped),
`claim` (what is asserted) and `evidence` (`claimed`, `rechecked` with the
list of recomputed `checks`, or `certified`). fcmaes claims are always
`rechecked`; `none` without a refutation (a penalty value, an empty front or
archive) is `claimed`. The fcmaes claims are:

- `best_found` (retries): the best point the optimizer evaluated, inside the
  bounds and re-evaluated; no global certificate. If the re-evaluation
  disagrees, the claim is not reported: `evidence.refuted` names `best_found`
  with verdict `infeasible_point` (claim `none`; the point is not evaluated
  outside the box) or `objective_mismatch` (claim `feasible` with the
  recomputed value, or `none` if that value is unusable). Treat a refutation
  as a defect report, not as an optimization result.
- `non_dominated_set` (MODE): `details.front` lists the feasible, mutually
  non-dominated final-population members, each re-evaluated, with `x`,
  `objectives` and `constraints`; `details.hypervolume` is the exact
  hypervolume for the declared reference point (raw objectives,
  `normalization: "none"`), counting points outside it instead of clipping
  them. `front_source` is `final_population`, not an archive of every
  evaluated point. The outcome has no single `x` or `objective`. No
  Pareto-optimality certificate.
- `elite_archive` (MAP-Elites): `details.elites` lists every occupied niche,
  re-evaluated and mapped back to its niche; `x` and `objective` are the best
  elite. Report `coverage`, `qd_score`, `mean_fitness` and representative
  elites, not only the best one.

- `feasible` (`fcmaes.evaluate/1`): the given point has a usable value, or for
  a multi-objective model satisfies every constraint; objectives and
  constraints are in `details`.

Use `fcmaes.evaluate/1` when you, the agent, propose candidates: the worker
evaluates them with the registered model instead of you computing or
estimating values. Points must lie inside the problem's bounds and are
inline in the job, which must stay below the runner's 4 MiB validation
output.

Other workers have different claims (`numerical_optimum`, `infeasible` and
`unbounded` for LP/MIP; `first_order` and `locally_infeasible` for SQP). Do not
equate those meanings across solvers. Runner errors are JSON on stderr with a
`code` and `retryable`; only `attempt_active` is worth repeating unchanged, and
`request_key_conflict` means the key already names different content.

`results.jsonl` commits checksummed task records; `events.jsonl` records lifecycle
transitions. `resolved.json` and `provenance.json` preserve execution inputs and
worker identity. Final results are ordered by `task_id`, not completion time.
Preserve the provided journals and manifests rather than inventing another
result schema or replacing missing work with zero.

If work was forcibly interrupted, `work` is null,
`runner.accounting_complete` is false, and `known_work_lower_bound` counts only
committed tasks. Cooperative stops can account for the tasks that returned.
Report these limits explicitly, including when computing throughput.

The common `compare` command requires matching resolved experiment digests and
payload profiles and does not select a winner. Changing algorithms, budgets or
resource widths changes that digest. For such comparisons, use an explicit
benchmark harness with matched models, seeds and the chosen call/wall-resource
protocol; inspect worker hashes, completed tasks, quality and accounting
completeness. Counters from different solver profiles are not interchangeable.

### Preserve legacy jobs and extend the worker

Existing `fcmaes.job/0.1` specs, legacy manifests and `retries.jsonl` keep their
meaning; they are not silently converted to the common envelope. Build
`fcmaes-runner` and `fcmaes-native-worker`, register that executable as `native`,
and follow [the legacy runner guide](docs/experiment-runner.md). Register the
new adapter as `fcmaes` for the examples above. Legacy comparison and benchmark
tools remain separate from common-profile comparison rules.

To add a scalar objective, change three places: its validation and discovery
entry in `crates/optimization-job-api/src/legacy/spec.rs` (a shared file, so
update the manifest and the sibling copies), its function in `objective_by_id`
in `crates/fcmaes-native-worker/src/lib.rs` (used by both fcmaes workers), and
deterministic tests for bounds, objective values, budget accounting and
per-task seeds. Multi-objective and quality-diversity models live in
`crates/fcmaes-job-worker/src/problems.rs`, used only by the common worker;
add the model, its canonical box and discovery entry, and tests that compare a
job run with a direct `fcmaes-core` run from the same seed. To add a new
solver profile, implement `worker::Worker::prepare` and `Prepared`, return task
outcomes with evidence and work counters (record a claim that fails your
independent check with `Evidence::refuting`), and let the SDK own concurrency
and journal commits.
Version profiles when mathematical input or output semantics change.

Current scope: local single-user trusted executables; no HTTP/MCP service
(the guide maps the runner onto the MCP Tasks extension), distributed
scheduler, arbitrary Python callback transport, solver checkpoint resume or
automatic cross-solver pipeline. Direct integration can compose
algorithms, but that is not a capability of this job protocol yet.

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

If the simulator also owns a thread pool, benchmark two explicit topologies:
many serial simulator evaluations under fcmaes versus fewer evaluations using
the simulator's internal parallelism. Keep the total worker allowance equal
and do not enable both pools at full size.

### Noisy objective

Use PGPE ranking, larger populations, or repeated evaluations. Do not use a
tight `stop_fitness` or convergence tolerance based on one noisy observation.
If the objective averages replications internally, report both optimizer calls
and the true number of simulations; fcmaes can count only calls it makes.

Use a fixed, named training seed set for every candidate so that optimizer
comparisons use common random numbers. Re-evaluate finalists, Pareto points,
and QD elites with a disjoint holdout seed set. For noisy QD, validate both
quality and descriptor location because an elite can migrate to another niche
under new stochastic paths.

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
  Mazda examples for decoding patterns. Use
  `docs/combinatorial-encodings.md` for the tested recipes, representation
  trade-offs, and repair-versus-constraint decision.

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
  regular grid with O(1) lookup. Non-factorable capacities have ragged rows;
  use `Archive::grid_layout()` for rendering and `Archive::capacity()` for the
  coverage denominator rather than multiplying `grid_shape()` dimensions.
- Positive `samples_per_niche` selects k-means CVT centers and nearest-center
  lookup. Use it for non-grid or higher-dimensional descriptor spaces. The
  centers depend only on the random stream, not on the number of Rayon
  threads, and `Archive` is `Clone`, so one tessellation can serve several
  runs.
- Capacity controls resolution and memory. More niches also require more
  evaluations to obtain useful coverage.
- Set `chunk_size >= workers`; 4–16 times the worker count is a useful
  throughput range for expensive objectives, subject to evaluation cost.
- SBX is a good bounded default. Try Iso+LineDD when local variation along
  elite differences is more appropriate.
- `map_elites_batch` and `diversify_batch` evaluate concurrently, then mutate
  the archive serially in candidate order. `QdBatchFitness` must return exactly
  one `(fitness, descriptor)` pair per input in the same order.
- `qd_score` is higher-is-better but depends on the fitness convention: the
  implementation sums reciprocal fitness for an all-positive archive and
  negated negative elites otherwise. Compare it only between archives with the
  same quality definition, descriptor bounds, and niche geometry.

## Lessons from the application tutorials

The twenty-two standalone tutorials are implementation references for expensive
native objectives. They keep application dependencies outside the root
workspace and demonstrate these transferable choices:

| Tutorial | Main problem property | Recommended lesson |
|---|---|---|
| NeXosim production line | Stochastic discrete events and mixed controls | Compare outer candidate parallelism with simulator-internal parallelism; use common random numbers and holdout seeds. |
| Rapier trebuchet | Contact and release discontinuities | Use BiteOpt retry for one target, MODE for engineering trade-offs, and QD only for meaningful trajectory behaviors. |
| ReBop oscillator | Intrinsic stochastic simulation noise | Fix candidate-comparison seeds, report true simulation counts, and validate Pareto/QD results on disjoint paths. |
| Oscillator topology search | Discrete signed graphs around variable-dimensional stochastic inner problems | Keep the agent or evolutionary proposer outside deterministic grammar, optimization, persistence, and held-out motif scoring; match random and evolutionary controls. |
| Brahe constellation | Access-window discontinuities and worst-gap aggregation | Keep feasibility explicit and assign either fcmaes or the simulator ownership of parallelism. |
| RustPower voltage control | Mixed-integer controls, contingencies, and solver failures | Return calibrated constraint violations for failed power flows; reject a QD formulation when descriptors do not produce an informative archive. |
| Atmospheric source localization | Censored inverse inference, model mismatch, and non-identifiability | Use robust residuals and disjoint sensors/weather; keep MODE for error/emission trade-offs and interpret a source-centroid QD map as alternative hypotheses, not a confidence region. |
| Room ventilation | Custom numerical backend, variable geometry, and grid sensitivity | A purpose-built backend can keep objective state isolated and fast, but then solver verification, held-out scenarios, constraint margins, and resolution sensitivity are part of the optimization evidence. |
| SmartCore hyperparameter tuning | Mixed variables, nested stochastic fitting, and validation overfitting | Use probability-aware objectives, common folds and model seeds, disjoint candidate selection, and a frozen final test; report model-fit cost as well as optimizer calls. |
| Neural controller policy search | 118-dimensional fixed-topology policy and randomized rollouts | Use PGPE or CR-FM-NES when full covariance is unattractive; use common per-population scenarios, rotate them deterministically, validate disjoint plants, and reserve a frozen final test. |
| GTOC1 “Save the Earth” | Narrow low-thrust closure constraints and model-fidelity differences | Use a staged fidelity ladder, repropagate the final continuous-thrust solution independently, and keep ephemeris/model qualifications attached to every score claim. |
| GTOC1 route search | Agent-proposed variable planet orders with unequal fidelity costs | Let deterministic Rust own grammar and equal-budget controls; archive failures and promote candidates through predeclared L0/L1/L2 gates before making trajectory claims. |
| sindr circuit design | Smooth circuit features plus discrete catalogue realization | Interpolate peaks and crossings instead of optimizing sampled arg-max staircases; validate the final catalogue under component tolerances. |
| thevenin gate driver | Transient thresholds, ringing, and simulator disagreement | Interpolate time-domain metrics and require timestep refinement plus an independent ngspice comparison before accepting a front. |
| Optical lens design | Multimodal ray tracing with hard ray-loss regions | Verify the native ray tracer independently, retain typed invalid designs, and benchmark global search rather than assuming a local simplex can cross penalty cliffs. |
| Rapier quadruped gait | Contact-driven behavior repertoires and terrain overfitting | Treat QD as the primary output only when descriptors remain meaningful and elites survive disjoint terrain replay. |
| Phased-array codebook | Quantized phase/attenuation registers and element failures | Cross-check direct and FFT kernels, optimize hardware codes rather than ideal phases, and apply the descriptor gate on the archive's exact native layout. |
| Bilevel energy hub | Discontinuous sizing around a convex dispatch problem | Solve the inner LP to optimality, optimize only the outer nonconvex choices, and report candidate calls, LP solves, and pivots separately. |
| Field-service routing | Assignment/permutation plateaus and disrupted task sets | Prove decoder invariants, validate hard feasibility on disrupted scenarios, and skip QD when coverage or retention fails. |
| Water-network scheduling | Quantized controls with hydraulic state memory | Separate DDA optimization from PDA validation, record hydraulic failures, and reject repertoires that migrate under unseen demand. |
| Truss topology and sizing | Exact-cardinality topology plus mechanisms and conditioning | Decode exact-k structures, classify FEM failures, and test removal robustness instead of replacing mechanisms with arbitrary large stresses. |
| Network coverage | Large binary submodular structure with available certificates | Run exact tiny oracles and specialist greedy/certificate baselines before generic search; publish the specialist win when MODE is dominated. |

MODE and MAP-Elites are complementary, not substitutes. Keep MODE when the
user needs objective trade-offs, and add MAP-Elites only when descriptor-space
coverage is itself useful. A QD pilot that runs successfully but has poor or
misleading coverage should remain a documented negative result rather than be
promoted into the main formulation.

Do not treat a custom simulator as independently validated merely because its
optimizer integration is tested. Separate three questions: whether the
numerical kernel satisfies reference properties, whether selected designs
survive held-out scenarios and resolution changes, and whether the physical
model is adequate for the user's decision. The room-ventilation tutorial
demonstrates the first two while explicitly declining engineering claims about
the third.

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
- `context.run_seed` when a run must reproduce independently of worker
  scheduling; `context.seed` retains the older worker-stream behavior;
- `context.run_id`;
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

These are core-library capabilities. The fcmaes job adapter maps them onto
the envelope: `outer_workers` independent retries or runs, each evaluating its
populations on `inner_threads` threads of its own (CMA-ES, CR-FM-NES, PGPE,
MODE, MAP-Elites; BiteOpt and DE stay serial). Its resource contract and
examples are in [Shared agent job API](#shared-agent-job-api).

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
PCG streams, so the compatibility `context.seed` depends on which logical
worker claims a run. `context.run_seed`, by contrast, depends only on the root
seed and `run_id`; use it for worker-count-independent objective randomness.
Timing-dependent early stopping can still change which run IDs are started.
Ordered parallel population batches are deterministic for a deterministic
objective.

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

Some population algorithms check limits at generation boundaries, so reported
evaluations can exceed the configured limit. Use the reported count for
comparisons. In `fcmaes.retry/1`, BiteOpt has a hard cap while DE and CMA-ES
have the explicit soft allowances described above; do not apply a generic
overshoot rule to all three.

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
- Compare algorithms using the same declared objective-call and resource
  limits; use equal wall deadlines when sequential overhead or parallel slot
  utilization differs materially.
- Use multiple seeds and report at least best, median or mean, standard
  deviation, and success/feasibility rate.
- For multi-objective runs, report feasible Pareto points and a suitable front
  quality measure; do not compare only one selected point.
- For QD runs, report capacity, occupied niches, coverage, best fitness,
  `qd_score`, descriptor ranges, and—when noisy—holdout niche migration.
- Check scaling by testing workers 1 and N. Deterministic ordered batches should
  produce the same values; parallel retry may be statistically rather than
  bitwise reproducible.
- Preserve a small deterministic smoke configuration in tests.

For job-layer changes, also test schema/default resolution, request-key
idempotency/conflicts, shared CPU admission, serial/eight-worker task seeds,
independent candidate checks, cancellation, forced timeout accounting and
journal recovery. These are the commands CI runs; build the release binaries
before the Python harnesses:

```sh
cargo test --workspace --locked
cargo build --release --locked -p fcmaes-runner -p fcmaes-native-worker \
  -p optimization-runner -p fcmaes-job-worker
python3 scripts/runner/test_integration.py
python3 scripts/job-api/test_integration.py
python3 scripts/job-api/test_review.py
python3 tutorials/job-api/run.py
sha256sum -c docs/job-api/SHARED-SOURCES.sha256
```

Use the workspace test run rather than a package subset: the runner's CLI
tests drive both worker binaries, which only a workspace build guarantees.
For a shared protocol change, run the mixed harness with independently built
LP-IP and SQP workers as documented in `docs/job-api.md`. Report missing-worker
and systemd skips explicitly. [The validation report](docs/job-api/VALIDATION.md)
records the 0.1 delivery's acceptance gates and later review fixes; it is
evidence for those sources, not a test result for subsequent edits.
Resource/correctness tests are not performance benchmarks.

Treat silent candidate/result misassociation, out-of-bounds returned points,
reversed feasibility semantics, or materially false deterministic
seed/budget accounting as potential numerical-integrity security issues.
Follow [`SECURITY.md`](SECURITY.md) and report credible cases privately through
GitHub's security-advisory form. Ordinary stochastic variation and failure to
reach a global optimum are not security defects.

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

- [docs/job-api.md](docs/job-api.md): common envelope, worker profiles, node
  admission, lifecycle, result claims, recovery and current limitations.
- [tutorials/job-api/README.md](tutorials/job-api/README.md): runnable BiteOpt,
  CMA-ES, MODE and MAP-Elites jobs, eight-slot setup and legacy compatibility.
- [docs/experiment-runner.md](docs/experiment-runner.md): original
  `fcmaes.job/0.1` CLI, manifests, comparison and benchmark contract.
- [docs/job-api/PLAN.md](docs/job-api/PLAN.md) and
  [docs/job-api/VALIDATION.md](docs/job-api/VALIDATION.md): implementation scope,
  acceptance evidence and unverified host behavior.
- `crates/fcmaes-job-worker/src/`: the common-profile adapter (`retry.rs`,
  `mode.rs`, `map_elites.rs`) and its multi-objective and quality-diversity
  models (`problems.rs`).
- `crates/fcmaes-native-worker/src/lib.rs`: objective registry and solver
  dispatch shared by both fcmaes workers.
- [docs/job-api/SHARED-SOURCES.sha256](docs/job-api/SHARED-SOURCES.sha256):
  the files shared with LP-IP-craft and SQP-craft and their content hashes.
- `crates/optimization-job-api/src/spec.rs` and
  `crates/optimization-job-api/src/result.rs`: authoritative common input and
  result contracts.
- `scripts/job-api/test_integration.py`: local or mixed-worker acceptance.
- `scripts/job-api/test_review.py` and `fake_worker.py`: shared runner
  regression tests against a protocol fixture.
- `scripts/runner/test_integration.py`: original runner compatibility checks.
- `tutorials/RESULT_SCHEMA.md`: established tutorial and legacy result formats;
  use the explicitly versioned common result contract for common jobs.
- `docs/getting-started.md`: building and basic Rust use.
- `docs/choosing-an-optimizer.md`: deciding whether fcmaes-rust or a
  structured, gradient-based, surrogate-based, real-time, or
  genotype-aware alternative fits the complete problem.
- `docs/optimizers.md`: public optimizer interfaces and defaults.
- `docs/retry.md`: basic, advanced, and weighted retry.
- `docs/architecture.md`: objective flow, normalization, and concurrency.
- `docs/examples.md`: native application and benchmark commands.
- `docs/combinatorial-encodings.md`: bounded integer, categorical, Boolean,
  random-key permutation, exact-cardinality subset, partition, ordering, and
  deterministic repair patterns.
- `docs/buckingham-pi.md`: dimensionally valid continuous feature search,
  holdout validation, conditioning safeguards, and MODE objectives.
- `docs/python-bindings.md`: CPython 3.11–3.13 package, callable signatures,
  result layouts, GIL behavior, and Python examples.
- `examples/src/runner.rs`: canonical DE-to-CMA retry integration.
- `examples/src/bin/mazda_mo.rs`: parallel constrained MODE driver.
- `examples/src/bin/mazda_qd.rs`: parallel MAP-Elites/Diversifier driver.
- `examples/src/uav.rs`: random-key decoding for mixed assignment, ordering,
  scalar, and multi-objective optimization.
- `examples/src/encoding.rs`: dependency-free reference decoders and invariant
  tests supporting the combinatorial encoding cookbook.
- `examples/src/buckingham.rs`: nullspace parameterization that makes every
  continuous optimizer trial dimensionally valid.
- `foundations/README.md`: standard scalar/multi-objective suites, indicator
  evidence, Lennard-Jones scaling, and the seven-step lesson ladder.
- `tutorials/README.md`: all twenty-two native application-optimization
  tutorials, MODE/MAP-Elites selection, stochastic validation, parallelism
  ownership, split-brain protocols, and the Diffsol gradient counterexample.
- `tutorials/ml-hyperparameter-tuning/`: native probability forests,
  mixed-variable decoding, fixed-fold tuning, disjoint selection, frozen final
  evaluation, fair baselines, constrained MODE, and a MAP-Elites pilot.
- `tutorials/cfd-room-ventilation/`: custom native simulation state,
  worst-case training releases, held-out validation, three-grid sensitivity,
  MODE, and MAP-Elites.
- `tutorials/neural-controller-policy-search/`: PGPE, CR-FM-NES, active
  CMA-ES, and BiteOpt under an equal direct-policy-search protocol with
  common scenarios, disjoint validation, a frozen test, and scaling evidence.
- `benchmarks/gtop-cmaes-retry/README.md`: equal-wall evidence for applying
  generic parallel retry to an external serial optimizer.
- `benchmarks/cmaes-implementation/README.md`: controlled implementation
  diagnostic, protocol boundaries, and publication evidence.
- `docs/optimizer-boundary.md`: measured reason local, Bayesian, gradient, and
  structured solvers stay behind the adapter boundary.
- `SECURITY.md`: supported-release policy and private reporting process for
  security and silent optimization-integrity failures.
- `CHANGELOG.md`: synchronized release history and current-main changes.
- `RELEASING.md`: Trusted Publishing, artifact, tag, and clean-consumer checks.
- Generated rustdoc: `cargo doc --workspace --no-deps --open`.

When code and this guide disagree, treat the current public Rust API and tests
as authoritative, update the guide with the implementation, and document the
reason for the change.
