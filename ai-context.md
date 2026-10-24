# AI context for solving user optimization problems with fcmaes-rust

Last reviewed 2026-10-20 against `main` at `5a4eff2`. That state has the
exact two-objective `ParetoArchive2D` in `fcmaes-core`; the fcmaes job
worker's retry (BiteOpt, DE, CMA-ES with IPOP/BIPOP restarts, CR-FM-NES,
PGPE), MODE, MAP-Elites (CVT, CMA-ME/CMA-MAE) and evaluation profiles under
result contract 0.2; and the experimental `fcmaes-surrogate` crate with the
`fcmaes.surrogate/1` profile and `benchmarks/surrogate-comparison` (#14).
Since `67bd698`, the job crates `optimization-job-api` and
`optimization-runner` (0.3.0 since the pin move after their first crates.io
release) come from
[optimization-jobs](https://github.com/dietmarwo/optimization-jobs) (#11; the
in-repo `fcmaes-job-api` crate is gone), now at the pinned revision `70e210f` (after tag `v0.3.0`), which adds
`node capabilities` and `jobs execution JOB_ID`. The pins up to `v0.3.0`
added worker-declared artifacts (#14), streaming artifact pages, the MCP
tools `put_input` and `get_guide`, a host report in `node doctor`, a 1 MiB
limit for `describe` output, and the corrected Pareto guidance that
`get_guide` serves; `optimization-mcp` serves the jobs
to agents (#12); and review fixes bound BiteOpt/DE populations, name MODE
hypervolume methods and add the tutorial's `--bin-dir` (#13). The
agent-discovery review then added the worker's payload schemas and runnable
examples in `describe`, a `guess` for `fcmaes.retry/1`, simulation models in
retry and evaluate, and the scalar constraint rule below. In another
checkout, check `Cargo.toml`, the available worker's discovery output, and the
linked guides before using a command or claiming a capability.

## Start here

If the solver is not chosen yet, start with the
[cross-repository entry point](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)
in optimization-jobs. It routes between fast-cma-es, fcmaes-rust,
LP-IP-craft and SQP-craft and covers installation, the job API and the MCP
server. In this file:

| Task | Sections |
|---|---|
| Decide whether fcmaes fits and choose an algorithm | [First decide](#first-decide-whether-fcmaes-is-appropriate), [decision tree](#fast-algorithm-selection-decision-tree), [comparison](#algorithm-comparison) |
| Python, direct Rust or a durable job | [Choose the integration surface](#choose-the-integration-surface) |
| One objective | [Scalar parameters](#scalar-optimizer-parameters), [recipes](#recommended-scalar-recipes), [retry](#retry-selection-and-parameters) |
| Constraints and penalties | [Constraints and variable encoding](#constraints-and-variable-encoding) |
| A Pareto front (MODE: 0.2.0 or later; 0.1.5 lacks its fixes) | [MODE](#mode), [weighted retry](#weighted-multi-objective-retry), [exact archive](#exact-two-objective-archive) |
| Diverse behaviors | [Quality diversity](#quality-diversity) |
| An expensive simulation with at most 32 variables | [Surrogate chooser rule](#experimental-surrogate-chooser-rule) |
| Jobs, discovery, MCP and results | [Shared agent job API](#shared-agent-job-api), [MCP](#agents-the-mcp-server), [results](#interpret-results-and-compare-experiments) |
| Your own model as a job | [Application models](docs/job-api/models.md), [extend the worker](#preserve-legacy-jobs-and-extend-the-worker) |
| Threads and cores | [Parallelism rules](#parallelism-rules) |
| CFD experiment through prompts | [Room-ventilation workshop](tutorials/cfd-room-ventilation/agentic/README.md): direct native CLI, fixed budgets, held-out replay and grid/setup sensitivity; no registered CFD job profile |
| Checking a solution | [Validation checklist](#validation-and-reporting-checklist), [failure diagnosis](#failure-diagnosis) |

## Experimental surrogate chooser rule

`fcmaes.surrogate/1` is exact-GP Bayesian optimization inside the fcmaes job
worker. Use it only for deterministic scalar continuous box problems with
1–32 variables and at most 256 new plus warm observations, on any registered
scalar model. It fits a Matérn-5/2 (or RBF) ARD GP, maximizes LogEI (or LCB)
inside a single TuRBO-style trust region for two of three proposals
(`trust_region`, default true) and proposes synchronous Kriging-believer
batches of `concurrent` (1–8) points. The resource rule is
`concurrent * evaluation_threads <= inner_threads <= 8`, within the common
`outer_workers * inner_threads <= cpu_slots`.

Evidence: in [`benchmarks/surrogate-comparison`](benchmarks/surrogate-comparison/README.md)
(equal call budgets, ten seeds) it had the lowest median final regret of the
GP methods (BoTorch, egobox, scikit-optimize, Optuna) on Hartmann-3/6, the
RK4 oscillator and Styblinski–Tang-10, egobox was better on Branin, and its
proposal time was 1–3 orders of magnitude lower; at half budget Optuna's GP
sampler and noiseless BoTorch led on two and three variables. BiteOpt and
CMA-ES were orders of magnitude worse on the smooth problems at these
budgets, but BiteOpt beat every GP method on the separable, many-modal
Styblinski–Tang-10 (median regret 17 against 57). One proposal round with a
fresh fit took about 1 ms to 1 s on one thread, growing with dimension,
history and batch ([validation report](docs/surrogate/validation/REPORT.md)).

Choose it when one evaluation costs much more than that overhead, few
evaluations can run at once, and the budget is a few hundred calls at most.
Prefer `fcmaes.retry/1` (BiteOpt first; PGPE for noise) for separable or
many-modal problems with ten or more variables, more than 32 variables or 256
evaluations, noisy objectives, or many cores evaluating in parallel: at equal
wall time a parallel population can finish before a sequential loop has made
a few calls ([optimizer boundary](docs/optimizer-boundary.md#how-expensive-is-costly-enough)).
Constraints, mixed variables, noise inference, asynchronous scheduling and
handoff to retry are not implemented. Raise `nugget` (default 1e-10) for
rough or solver-noisy simulations. Emit only options that discovery lists;
an `auto` algorithm selector, `turbo`, async, noise, Sobol, Thompson sampling
or governor settings are rejected (`output_transform: auto` exists and only
picks standardize or bilog). Source-only means it is absent from the
published core and Python packages.

`Scalar::simulation` takes a fallible callback that receives a `SimContext`
with `checkpoint()`, `remaining_seconds()`, a point-derived `seed()` and a
`threads()` allowance. The surrogate, `fcmaes.retry/1` and
`fcmaes.evaluate/1` accept it, so one model can be compared under the
surrogate and BiteOpt. In a retry, a failed evaluation counts as `+inf`
(every fcmaes-core optimizer ranks it worst) and is counted in
`failed_evaluations`; `threads()` is 1 there and in evaluate. A population
that fails entirely is a flat plateau: CMA-ES widens its step there, PGPE
only drifts, so start it with a `guess` where evaluations succeed. Describe
such models with `simulation: true`, which makes the surrogate their default
`profile`. `Scalar::new` models are unchanged; use constructors, not `Scalar`
struct literals. A stable point seed is not noisy GP support.

Only independently re-evaluated exact agreement yields `rechecked` evidence;
otherwise the best historical point remains `claimed`. Final validation is an
extra separately counted call. Cancellation is partial work, not an evaluation
failure; a cancelled recheck does not refute the objective. The trace and the
snapshot of each task are declared artifacts (`surrogate-TASK`,
`surrogate-observations-TASK`): `jobs get` and MCP `get_job` list them while
the job runs, `jobs artifact` and MCP `get_artifact` page through the trace,
and the result's `runner.artifact_index` records their SHA-256. Warm starts
are new jobs from a hashed snapshot input, not exact checkpoint resumes.

Read [the current API](docs/job-api/surrogate.md),
[the executable tutorial](tutorials/surrogate/README.md), and
[the reviewed plan/status ledger](docs/surrogate/PLAN.md) before using this
profile. The rest of this file mentions it only where it changes a choice.

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

The workspace version is 0.2.1, published as `fcmaes-core` 0.2.1
(crates.io) and `fcmaes-rust` 0.2.1 (PyPI) on 23 October 2026. Check the
registries for the latest published version before relying on it. This repository's job-layer crates (`fcmaes-job-worker`,
`fcmaes-runner`, `fcmaes-native-worker`) and the experimental
`fcmaes-surrogate` 0.1.0 are source-only (`publish = false`) and are part
of neither package. The shared `optimization-job-api` and
`optimization-runner` come from optimization-jobs through git at the pinned
revision; both are also on crates.io from 0.3.0. Core additions after
0.1.5, such as `ParetoArchive2D`, need 0.2.0 or later. The Python binding
fixes (sequence arguments, `ValueError` instead of panics, logged callback
failures) and the Dual Annealing start-point fix need 0.2.1. Changes after
the latest release need a git dependency on `main`.

**MODE warning.** `fcmaes-core` and `fcmaes-rust` 0.1.5 and earlier predate
the October 2026 MODE fixes (#1: the SBX expanding branch, polynomial
mutation's upward branch, non-dominated sorting and feasibility-first
constraint handling), with which MODE no longer collapses to one extreme
point or stalls, and `ParetoArchive2D`. Use 0.2.0 or later
(`fcmaes-core = "0.2"`, `pip install "fcmaes-rust>=0.2"`); if that version
is not yet on the registries, use a git dependency
(`fcmaes-core = { git = "https://github.com/dietmarwo/fcmaes-rust" }`) or a
[source build](./README.md#install-from-source) of the Python package. The job
worker always builds with the checkout's core. The other core changes of
0.2.0 (CHANGELOG) are `De::optimize_delayed`, `BiteOpt::optimize_batch`,
thread-independent CVT archives and the `moretry` constraint
scalarization. Schema versions
such as `optimization.job/0.1` and `fcmaes.retry/1` are protocol identifiers,
independent of crate versions.
Use `CHANGELOG.md`, package manifests, and the actual registry metadata when
discussing releases; a workspace version alone does not establish publication.

The existing optimizer guidance includes the 0.1.4 additions: audited
multi-objective quality indicators, schedule-independent retry seeds, exact
ragged MAP-Elites grid layouts, non-dominated-sort/crowding utilities, the
optimizer-boundary evidence, and the Foundations and Lennard-Jones studies.
The twenty-two application tutorials, Foundations, and the operational
[job API](tutorials/job-api/README.md) and
[surrogate](tutorials/surrogate/README.md) tutorials serve different
purposes; do not count the operational tutorials as application benchmarks.

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
| Is a good initial point known? | CMA-ES, CR-FM-NES, PGPE, and local refinement benefit from it; `fcmaes.retry/1` takes it as `guess`. |
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
Bayesian optimization, and gradient solvers remain outside `fcmaes-core` and
can be driven through retry's optimizer closure; the experimental
`fcmaes-surrogate` crate is a separate workspace member, not part of core.
The corrected 20-seed
[optimizer-boundary experiment](docs/optimizer-boundary.md) found no general
DE→Nelder–Mead advantage and only a narrow, small-budget BO regime. Do not add
an optimizer dependency to core merely because one application needs it.

### Other solvers on the same job node

[LP-IP-craft](https://github.com/dietmarwo/LP-IP-craft) and
[SQP-craft](https://github.com/dietmarwo/SQP-craft) build workers for the same
job API; one `optimization-run` node and one `optimization-mcp` server serve
all three against one CPU budget. Route by structure:

| Problem | Repository, worker | Profile |
|---|---|---|
| LP or MIP, given as a model file (the documented CSR JSON input) | LP-IP-craft, `lp-ip-job-worker` | `lp-ip.solve/1` |
| Smooth constrained NLP with analytic gradient and Jacobian, at most 256 variables, a model registered in the worker | SQP-craft, `sqp-job-worker` | `sqp.solve/1` |
| Black-box, nonsmooth, noisy or multimodal scalar objective | fcmaes-rust, `fcmaes-job-worker` | `fcmaes.retry/1` |
| Expensive deterministic scalar simulation, 1–32 variables, at most 256 evaluations (experimental) | fcmaes-rust | `fcmaes.surrogate/1` |
| Pareto front or quality-diversity archive | fcmaes-rust | `fcmaes.mode/1`, `fcmaes.map-elites/1` |

Each worker also evaluates given points (`*.evaluate/1`). Build each worker in
its own repository and read its `ai-context.md` first. Claims and work
counters differ between solvers; never compare or add them across profiles.

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
   - With exactly two objectives, add `ParetoArchive2D` when the front of
     everything evaluated matters: MODE returns its final population, which
     need not contain every non-dominated point it evaluated. See
     [Exact two-objective archive](#exact-two-objective-archive).
4. For one scalar objective:
   - Start with BiteOpt for a difficult bounded, nonsmooth, multimodal, or
     poorly characterized black box.
   - Start with DE for robust bounded exploration, mixed continuous/integer
     decoding, and DE-to-CMA pipelines.
   - Use active CMA-ES when a useful guess exists, variables are continuous,
     and correlations or ill-conditioning are important (in
     `fcmaes.retry/1`, pass it as `guess` with a matching `sigma`). A plain CMA-ES
     retry that converges early leaves its budget unspent; in
     `fcmaes.retry/1`, `restarts: ipop` or `bipop` spends it on restarts.
     Compare that with more plain retries at the same total budget.
   - Use CR-FM-NES for higher-dimensional continuous distribution search where
     a full CMA covariance update is unattractive.
   - Use PGPE for high-throughput mirrored batches, diagonal distribution
     search, or noisy objectives where rank-based updates help. See the
     fixed-topology
     [neural-controller tutorial](tutorials/neural-controller-policy-search/)
     for an equal-protocol PGPE/CR-FM-NES comparison.
   - Use Dual Annealing for low-to-moderate-dimensional global exploration,
     optionally followed by its bounded local search.
   - For an expensive deterministic continuous simulation with at most 32
     variables and a few hundred evaluations, include the experimental
     `fcmaes.surrogate/1` as a candidate next to BiteOpt
     ([chooser rule](#experimental-surrogate-chooser-rule)).
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
| Durable agent-controlled experiments with registered models | `optimization-run` and `fcmaes-job-worker`, or an application worker built on it; agents connect through `optimization-mcp` | Common jobs, CPU admission, journals, cancellation and result manifests. |
| Existing `fcmaes.job/0.1` jobs | `fcmaes-run` and `fcmaes-native-worker` | Preserved legacy contract; keep its input and result schemas. |
| Python code calling Rust optimizers | `fcmaes_rust` | Optional PyO3 API, with the callback/GIL constraints below. |

Use the common job API for agent-driven experiments when its worker supports
the problem and algorithm. Its fcmaes profiles expose independent BiteOpt
and DE (serial or in parallel delayed-feedback batches), CMA-ES (with
optional IPOP/BIPOP restarts), CR-FM-NES and PGPE retries,
independent MODE runs and independent MAP-Elites runs (grid or CVT archives,
optional CMA-ME/CMA-MAE emitters), and experimental surrogate runs, each on
registered native models. An
application objective becomes a registered model in an application worker
([application models](docs/job-api/models.md)): a Rust closure with
parameters and SHA-256-checked data inputs, never a callback in the job. Dual
Annealing, advanced retry, `moretry`, the Diversifier and `ParetoArchive2D`
remain direct-API features. The IPOP/BIPOP restarts and the CMA-ME/CMA-MAE
emitter ranking, conversely, are implemented in the job worker, not as
`fcmaes-core` functions.

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
From 0.2.1, array arguments accept lists and tuples as well as NumPy arrays.
0.2.0 and earlier accepted only NumPy arrays of the exact dtype (`float64`,
`bool` for `ints`), so pass `np.asarray(..., dtype=...)` when the installed
version may be older. Invalid bounds, guesses, sigmas and masks raise
`ValueError` (0.2.0 raised pyo3's `PanicException` for several of them). An
objective callback that raises is not re-raised: the call counts as a failed
evaluation, as in fcmaes. From 0.2.1, the first failure of a run is logged
once as a warning on the `fcmaes_rust` logger; still test the objective on its
own first. The `evaluations` field is the number of calls actually made:
`optimize_de` can exceed `max_evaluations` by up to `2 * popsize - 1`, ACMA,
CR-FM-NES and PGPE by up to `popsize - 1`; BiteOpt and Dual Annealing never
exceed it.

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
| `optimization-job-api` (from optimization-jobs) | Strict envelopes, result contracts, hashes, durable journals and worker SDK. |
| `optimization-runner` / `optimization-run` (from optimization-jobs) | Worker registration, staging, idempotency, CPU admission, process supervision and recovery. |
| `optimization-runner` / `optimization-mcp` (from optimization-jobs) | MCP server over stdio: tools and tasks over `optimization-run` for agents. |
| `crates/fcmaes-job-worker` | Native adapter for `fcmaes.retry/1` (BiteOpt, DE, CMA-ES with restarts, CR-FM-NES, PGPE retries), `fcmaes.mode/1` (MODE), `fcmaes.map-elites/1` (MAP-Elites, CVT, CMA-ME/CMA-MAE), `fcmaes.evaluate/1` (given points) and `fcmaes.surrogate/1` (experimental exact GP), on the models of a `Registry`; a library, a thin binary with the benchmarks, and `examples/custom_worker.rs` and `examples/simulation_worker.rs` for application models. |
| `crates/fcmaes-surrogate` | Experimental GP optimization library behind `fcmaes.surrogate/1`; independent of the job API. |
| `crates/fcmaes-runner` / `fcmaes-run` | Compatibility CLI entry point into the common runner. |
| `crates/fcmaes-native-worker` | Original worker for `fcmaes.job/0.1` specifications. |

The shared crates live in
[optimization-jobs](https://github.com/dietmarwo/optimization-jobs), with the
[common guide](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md),
the plan, the JSON Schemas and the shared test scripts; this
repository depends on them at the git revision its root `Cargo.toml` pins,
and `cargo build -p optimization-runner` still builds `optimization-run`
here, together with `optimization-mcp`. LP-IP-craft and SQP-craft depend on
the same crates and build their own workers
([routing](#other-solvers-on-the-same-job-node)). Build those workers in their
respective repositories and register their executables with one runner using
one absolute node root. The fcmaes checkout builds independently and does not
link either other solver.

Version 0.2.0 of the shared crates has the result contract
`optimization.result/0.2` and worker protocol `optimization.worker/0.2`
(claims such as `infeasible` and `locally_infeasible`, `evidence` with checks
and refutations, `objective_bounds`, coded runner errors and JSON Schemas in
`docs/job-api/schemas/` of optimization-jobs); the job envelope stays
`optimization.job/0.1`. A 0.2 runner still accepts 0.1 workers; a 0.1 runner
rejects 0.2 workers at registration. Workers report the shared version as
`build.job_api_version`. Use one runner build for all jobs that share a node
root. A cross-project change lands in optimization-jobs first; the worker
repositories then move their pinned revision.

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
`tutorial-report.json` under the node root. After
`cargo build --release --locked -p fcmaes-job-worker --example simulation_worker`,
`python3 tutorials/surrogate/run.py --root DIR` does the same for the
surrogate profile on Branin, a warm start and the RK4 oscillator. For manual
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

`problems describe fcmaes` (MCP `describe_worker`) describes the registered
worker ID. Use discovery to check supported profiles, models, optimizer
parameters and resource limits; do not infer job options from the broader
core API. The worker's `describe` lists JSON Schemas of each payload
(`payload_schemas`, draft 2020-12, strict, with `$comment` for rules a
schema cannot state, such as `inner_threads` limits) and runnable examples
(`examples`), plus `input_schemas` for the surrogate's warm-start snapshot
and each model's `parameters_schema`. Build a job from an example by adding
`schema_version`, a new `request_key`, the worker ID and `payload_schema`
(the example's `profile`), writing each input's `content_text` to its `file`
and adding its SHA-256; validate payloads against `payload_schemas` before
`validate_job`. The examples are generated from the worker's registry, so an
application worker advertises its own models, data inputs included.
The MCP tool `put_input` stages an input file for a job (the CSV of an
application model, or a surrogate warm-start snapshot, also copied directly
from an earlier job's artifact) and returns its `file`, `sha256` and the
`input_dir` to validate and submit the job with. `profiles` lists only
the profiles the registered models support. Each model entry names its
optimizing `profile` and all accepted `profiles`:

| Profiles | Registered models of the stock worker |
|---|---|
| `fcmaes.retry/1`, `fcmaes.surrogate/1` | `benchmark.sphere`, `benchmark.rosenbrock`, `gtop.cassini1`, `benchmark.branin` (2-D), `benchmark.hartmann3` (3-D) |
| `fcmaes.mode/1` | `benchmark.zdt1` (2 objectives), `benchmark.dtlz2` (3 objectives), `benchmark.binh-korn` (2 objectives, 2 constraints) |
| `fcmaes.map-elites/1` | `benchmark.sphere-projection`, `benchmark.rastrigin-projection` (2-D linear-projection descriptors) |
| `fcmaes.evaluate/1` | every model above |

An application worker lists its own models next to these, each with its
`kind`, `profile` and description (dimensions, parameters, inputs). A model
described with `simulation: true` has `fcmaes.surrogate/1` as its `profile`
and accepts retry and evaluate too. Only the first three scalar models also
exist in the legacy `fcmaes.job/0.1` profile.

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

MODE takes `population` (4–10,000, or 4–1,000 with three or more objectives,
where the exact hypervolume grows cubically; default 64), `nsga_update`,
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
`archive_learning_rate` and a required `threshold_init`, a fitness worth
keeping in an empty niche. No measured comparison of the two emitters exists
here; compare coverage and `qd_score` at the same exact budget.
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
returned in order, while the optimizer state stays serial. BiteOpt and DE
accept it with `optimizer.parameters.batch`: BiteOpt evaluates batches of
`batch` candidates, DE keeps `batch` candidates in flight (the delayed
update of the Python fcmaes `de.minimize(workers=...)`, told in ask order),
both with delayed feedback that costs some convergence per evaluation.
Serial BiteOpt and DE and `fcmaes.evaluate/1` require `inner_threads=1`. Core ask/tell remains available through the direct
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
derivation, so serial BiteOpt or DE on the same legacy model, seed and budget
give identical per-retry results in both profiles; the CLI tests check this.

`retries` and `evaluations_per_retry` must be positive. BiteOpt requires `hard`
evaluation limits. DE requires `soft`: with population P, the declared allowance
is at most `2P-1` extra evaluations per retry (none with `batch`). Switching
from BiteOpt to DE means changing both `optimizer.kind` to `de` and
`evaluation_limit_mode` to `soft`.
CMA-ES (`kind: cmaes`) also requires `soft`: it evaluates whole generations, so
it may spend at most `population-1` extra evaluations, and it can stop earlier
on its own convergence tests (`termination: "optimizer_stop"`,
`details.stop_code`). Its parameters are `population` (default 31) and `sigma`
(default 0.3, the initial step size in the box normalized to `[-1, 1]`); each
retry starts at a uniform random point drawn from its seed, or at the
payload's `guess` (one value per variable, inside the bounds): every retry
then starts its CMA-ES, CR-FM-NES or PGPE distribution there with `sigma`,
and retries differ by seed only. BiteOpt builds its initial population
around a guess, DE samples around it with `sigma` (default 0.3, only with a
guess). With `restarts: ipop` or `bipop`, CMA-ES restarts (after a first run
from the guess, if given) from new random points with
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
to its evaluation budget. On a `Scalar::simulation` model it stops at the
next `SimContext` checkpoint, its remaining evaluations return `+inf` at once
(`cancelled_evaluations`) and its best point keeps `claimed` evidence without
a recheck. MODE and MAP-Elites runs check it between generations
and return their current front or archive with `termination: "cancelled"`; at
the wall deadline the SDK uses the same signal, so the job state says whether
it was `cancelled` or `timed_out`. Surrogate runs stop before the next
proposal or evaluation, and `Scalar::simulation` models also at their
`SimContext` checkpoints. `jobs cancel JOB_ID --now` permits at most one further
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

### Agents: the MCP server

`optimization-mcp` is an MCP server over stdio for one node; the build above
makes it next to `optimization-run`. The node's owner initializes the node
and registers workers with the CLI; the server can do neither. Connect a
client with absolute paths, for example Claude Code:

```sh
claude mcp add fcmaes-jobs -- "$PWD/target/release/optimization-mcp" --root "$JOB_NODE"
```

Its thirteen tools (`describe_node`, `describe_worker`, `validate_job`,
`submit_job`, `get_job`, `wait_job`, `get_result`, `get_events`,
`get_artifact`, `list_jobs`, `cancel_job`, `kill_job`, `repair_job`) drive the
runner CLI, whose rules apply unchanged. `get_result` returns a summary
without points; pass `tasks` (task IDs) for complete records.
`get_artifact` reads [artifacts](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#artifacts),
also while the job runs: `results` and the files a worker declared (the
surrogate's trace and snapshot); `get_job` lists them in `artifact_index`,
and `offset` and `limit` page through `records` artifacts. With the MCP
Tasks extension, `submit_job` returns a task that completes with the result.
With the default `--launch process`, a job outlives the server but not a
reboot. See [Agents and MCP](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#agents-and-mcp)
for options, limits and error codes.

### Interpret results and compare experiments

Common `run.json` uses `optimization.result/0.2`; its `status` and the job's
`state` describe execution, while each `results` entry contains `task_id`,
`outcome` and `work`. Inspect all three layers. A completed job is not a proof
of optimality. Each outcome separates `termination` (why the task stopped),
`claim` (what is asserted) and `evidence` (`claimed`, `rechecked` with the
list of recomputed `checks`, or `certified`). fcmaes claims are `rechecked`,
except a surrogate `best_found` whose independent recheck was disabled,
skipped or cancelled, which stays `claimed`; `none` without a refutation (a
penalty value, an empty front or archive) is `claimed`. The fcmaes claims are:

- `best_found` (retries and surrogate runs): the best point the optimizer
  evaluated, inside the bounds and re-evaluated; no global certificate. If the re-evaluation
  disagrees, the claim is not reported: `evidence.refuted` names `best_found`
  with verdict `infeasible_point` (claim `none`; the point is not evaluated
  outside the box) or `objective_mismatch` (claim `feasible` with the
  recomputed value, or `none` if that value is unusable). Treat a refutation
  as a defect report, not as an optimization result.
- `non_dominated_set` (MODE): `details.front` lists the feasible, mutually
  non-dominated final-population members, each re-evaluated, with `x`,
  `objectives` and `constraints`; `details.hypervolume` is the hypervolume
  for the declared reference point (raw objectives, `normalization: "none"`),
  counting points outside it instead of clipping them: exact through four
  objectives (`method: "exact"`), a deterministic Monte Carlo estimate above
  four (`method: "monte_carlo"`, `exact: false`, with `samples`, `seed` and
  `standard_error`); compare sampled values within their standard errors.
  `front_source` is `final_population`, not an archive of every evaluated
  point (in direct Rust code, `ParetoArchive2D` provides that for two
  objectives). The outcome has no single `x` or `objective`. No
  Pareto-optimality certificate.
- `elite_archive` (MAP-Elites): `details.elites` lists every occupied niche,
  re-evaluated and mapped back to its niche; `x` and `objective` are the best
  elite. Report `coverage`, `qd_score`, `mean_fitness` and representative
  elites, not only the best one.

- `feasible` (`fcmaes.evaluate/1`): for a scalar or quality-diversity model,
  within bounds and a usable value; no constraint check (`details.constraints`:
  "none declared; penalties inside the objective are not checked"). For a
  multi-objective model, every declared constraint is at most zero
  (rechecked `constraints`); objectives and constraints are in `details`.

**A scalar `feasible` is not application feasibility.** If the objective
adds a penalty for violated constraints, a point with a small violation has
a small penalty and still gets `feasible`. Keep the raw objective and the
raw constraint values, and check application feasibility separately: for
`cost + 1e3 * max(0, g)^2`, also register
`MultiObjective::new(1, 1, |x| vec![cost(x), g(x)])` and evaluate the
candidates on it, which checks `g <= 0` (`evidence.checks` lists
`constraints`), or recompute `g` in the application before accepting a
point.

Use `fcmaes.evaluate/1` when you, the agent, propose candidates: the worker
evaluates them with the registered model instead of you computing or
estimating values. Points must lie inside the problem's bounds and are
inline in the job, which must stay below the runner's 4 MiB validation
output. A `Scalar::simulation` model that fails at a point gives claim
`none` with `details.error`; pass a retry's `details.run_seed` as `run_seed`
to re-evaluate its points with the same simulation seeds.

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

A user's model belongs in an application worker
([application models](docs/job-api/models.md)), not in this repository. A new
stock benchmark is registered in `crates/fcmaes-job-worker/src/problems.rs`
with its canonical box and discovery description, plus deterministic tests
for bounds, objective values, budget accounting and per-task seeds that
compare a job run with a direct `fcmaes-core` run from the same seed. Only a
scalar model that must also run under legacy `fcmaes.job/0.1` needs two more
places: its validation and discovery entry in
`crates/optimization-job-api/src/legacy/spec.rs` of optimization-jobs (changed
there first, then this repository's pinned `rev` moves) and its function in
`objective_by_id` in `crates/fcmaes-native-worker/src/lib.rs`. To add a new
solver profile, implement `worker::Worker::prepare` and `Prepared`, return task
outcomes with evidence and work counters (record a claim that fails your
independent check with `Evidence::refuting`), and let the SDK own concurrency
and journal commits.
Version profiles when mathematical input or output semantics change.

Current scope: local single-user trusted executables; no HTTP service,
distributed scheduler, arbitrary Python callback transport, solver checkpoint
resume or automatic cross-solver pipeline. Direct integration can compose
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
| Exact GP (`fcmaes.surrogate/1`, experimental) | Expensive deterministic continuous boxes, 1–32 variables, at most 256 calls | Proposal overhead grows with history and dimension; no noise, constraints or async; lost to BiteOpt on separable many-modal Styblinski–Tang-10 | Defaults first; `concurrent` = evaluations that can run at once; raise `nugget` for solver noise |

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
| `max_evaluations` | 100,000 | Set from the user's real budget. `optimize` checks it before each generation and can spend up to `2 * popsize - 1` more; `optimize_delayed` never exceeds it. Report `DeResult::evaluations`, the actual count. |
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

`De::optimize_delayed(objective, workers)` is the delayed update of the Python
fcmaes `de.minimize(workers=...)` on the current Rayon pool, told in ask
order, within `max_evaluations` and independent of timing.

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
versions. `BiteOpt::optimize_batch(batch, eval_batch)` runs that loop with a
hard evaluation limit and results independent of how `eval_batch` schedules
its work.

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
use outer retry workers and keep inner optimizer workers at one. For only a
few hundred mostly sequential deterministic evaluations, see the
[surrogate chooser rule](#experimental-surrogate-chooser-rule).

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
- **Rule: a penalized value hides its constraints.** Keep the raw objective
  and every raw `g_i(x)` next to the penalized value, and check feasibility
  of the reported point separately: with `fcmaes.evaluate/1` on a
  multi-objective model with declared constraints (for example
  `MultiObjective::new(1, k, |x| [objective, g_1, ..., g_k])`, whose evaluate
  checks every `g_i <= 0`), or in the application. A scalar model's
  `feasible` only means a usable value inside the bounds.
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

Run MODE with `fcmaes-core`/`fcmaes-rust` 0.2.0 or later, a git dependency,
a source build or the job worker: 0.1.5 and earlier predate the October 2026
MODE fixes (#1), with which MODE no longer collapses to one extreme point;
see the MODE warning with the package facts above.

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
in parallel. Set `ncon` to the number of trailing constraints. Only a
violation `max(c, 0)` enters the p-norm, and a violated constraint adds its
weight as a penalty; a satisfied constraint adds nothing, so a large
constraint weight enforces feasibility without distorting the trade-off. `value_limits`, when used,
contains one strict upper limit for every objective and constraint.

Normalize or shift objectives to comparable, preferably nonnegative scales
before selecting weight bounds. With a non-integer `value_exp`, negative
weighted values can make the p-norm undefined. Use `pareto_indices` on retained
unscalarized rows; do not report the scalarized retry value as a Pareto
objective.

### Exact two-objective archive

`ParetoArchive2D<P>` (`crates/fcmaes-core/src/pareto.rs`, derived from
BiteOpt's `bitehv2.h`) keeps every unique non-dominated pair of two minimized
objectives below a fixed reference point, each with a payload such as the
decision vector, and adds the exact hypervolume change on every insertion.
It does not change the optimizer: insert every evaluated feasible row, for
example each row told to MODE. Use it when the deliverable is the front of
everything evaluated rather than MODE's final population, or when an exact
hypervolume trace is needed during the run. Filter infeasible rows yourself;
points beyond the reference return `ParetoUpdate::Outside` and are not kept
(as `OutsidePolicy::Exclude`); non-finite objectives are errors; capacity is
unbounded. `preferred()` and `ranked()` order entries by a score that never
changes membership (normalize scales with `ParetoBounds2D`). It has no
Python binding and needs `fcmaes-core` 0.2.0 or later. See
[`PARETO_ARCHIVE.md`](crates/fcmaes-core/PARETO_ARCHIVE.md) and
`cargo run --release -p fcmaes-core --example pareto_mode` (a constrained MODE
run with a separate archive).

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
- `MapElitesParams::cma_generations` adds CMA-ES emitter generations after
  the variation phase. The CMA-ME and CMA-MAE ranking variants and multiple
  emitters with exact budgets exist only in `fcmaes.map-elites/1`.
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
   `parallel_batch`, `map_elites_batch`, `diversify_batch`,
   `De::optimize_delayed`, or `BiteOpt::optimize_batch` evaluates one
   population or batch concurrently.

These are core-library capabilities. The fcmaes job adapter maps them onto
the envelope: `outer_workers` independent retries or runs, each evaluating its
populations on `inner_threads` threads of its own (CMA-ES, CR-FM-NES, PGPE,
MODE, MAP-Elites, and BiteOpt and DE with `batch`; serial BiteOpt and DE need
`inner_threads=1`; the surrogate needs
`concurrent * evaluation_threads <= inner_threads <= 8`). Its resource
contract and examples are in [Shared agent job API](#shared-agent-job-api).

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
comparisons. In `fcmaes.retry/1`, BiteOpt and batched DE never exceed the
budget, while serial DE, CMA-ES, CR-FM-NES and PGPE have the explicit soft
allowances described above; do not apply a generic overshoot rule to all of
them.

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
# The tutorials validate tutorial payloads and describe's examples against
# payload_schemas, and the shared tests use the published JSON Schemas.
python3 -m pip install "jsonschema>=4.18,<5"
export JOB_API_REQUIRE_SCHEMA=1
cargo build --release --locked -p fcmaes-runner -p fcmaes-native-worker \
  -p optimization-runner -p fcmaes-job-worker
cargo build --release --locked -p fcmaes-job-worker --example custom_worker
python3 scripts/runner/test_integration.py
SHARED=$(python3 scripts/job-api/shared_root.py)
export JOB_API_TARGET="$PWD/target/release" JOB_API_TUTORIALS="$PWD/tutorials/job-api"
python3 "$SHARED/scripts/job-api/test_integration.py"
python3 "$SHARED/scripts/job-api/test_review.py"
python3 "$SHARED/scripts/job-api/test_mcp.py"
python3 tutorials/job-api/run.py
cargo build --release --locked -p fcmaes-job-worker --example simulation_worker
python3 scripts/surrogate/test_job_api.py
python3 tutorials/surrogate/run.py
```

Use the workspace test run rather than a package subset: the runner's CLI
tests drive both worker binaries, which only a workspace build guarantees.
A shared protocol change belongs in optimization-jobs, whose CI runs the
mixed harness with all three workers; locally, pass independently built LP-IP
and SQP workers to the shared harness (`docs/job-api/VALIDATION.md`). Report missing-worker
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

- [docs/job-api.md](docs/job-api.md): links into the
  [shared guide](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md)
  of optimization-jobs (envelope, admission, lifecycle, claims, recovery,
  Agents and MCP, limits) and how to run the shared tests here;
  [docs/job-api/models.md](docs/job-api/models.md): application models.
- [tutorials/job-api/README.md](tutorials/job-api/README.md): runnable jobs for
  every non-surrogate profile and optimizer, an application worker, an MCP
  connection, eight-slot setup and legacy compatibility.
- Surrogate: [API](docs/job-api/surrogate.md),
  [tutorial](tutorials/surrogate/README.md), [plan](docs/surrogate/PLAN.md)
  and [comparison](benchmarks/surrogate-comparison/README.md) with other BO
  libraries.
- [crates/fcmaes-core/PARETO_ARCHIVE.md](crates/fcmaes-core/PARETO_ARCHIVE.md):
  `ParetoArchive2D` semantics, preference scores and complexity.
- [docs/experiment-runner.md](docs/experiment-runner.md): original
  `fcmaes.job/0.1` CLI, manifests, comparison and benchmark contract.
- [docs/job-api/VALIDATION.md](docs/job-api/VALIDATION.md) and the
  [plan](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api/PLAN.md)
  in optimization-jobs: implementation scope, acceptance evidence and
  unverified host behavior.
- `crates/fcmaes-job-worker/src/`: the common-profile adapter (`retry.rs`,
  `mode.rs`, `map_elites.rs`, `evaluate.rs`, `surrogate.rs`), the model
  registry (`registry.rs`, `simulation.rs`), the discovery schemas and
  examples (`schema.rs`, `tests/discovery.rs`,
  `scripts/job-api/payload_schemas.py`) and the built-in models
  (`problems.rs`).
- [AGENTS.md](AGENTS.md): build, test and documentation-check commands.
- `crates/fcmaes-native-worker/src/lib.rs`: objective registry and solver
  dispatch shared by both fcmaes workers.
- In optimization-jobs: `crates/optimization-job-api/src/spec.rs` and
  `crates/optimization-job-api/src/result.rs` (authoritative common input and
  result contracts), `crates/optimization-runner/src/mcp/` (the MCP server),
  `scripts/job-api/test_integration.py` (local or mixed-worker acceptance),
  `scripts/job-api/test_review.py` and `fake_worker.py` (runner regression
  tests against a protocol fixture) and `scripts/job-api/test_mcp.py` (the
  MCP server). Here, `scripts/job-api/shared_root.py` prints the path of
  Cargo's checkout.
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

## Optional application tutorial workers

For prompt-driven trebuchet, production-line, energy-hub and GTOC1 experiments, see [application tutorial workers](docs/job-api/application-tutorials.md). Build the selected standalone tutorial with `--features job-worker`, register its separate binary, then use discovery. These application registrations reuse the generic worker API; they are not models bundled into the stock worker. The linked guide explains parameter validation, parity tests and the differences from the original search campaigns.
