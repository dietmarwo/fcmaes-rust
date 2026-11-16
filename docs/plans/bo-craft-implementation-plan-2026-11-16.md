# BO-craft: advanced Bayesian optimization implementation plan

**Date:** 16 November 2026  
**Status:** proposed new repository and integration work; no implementation or new performance results claimed  
**Proposed repository:** dietmarwo/BO-craft  
**Proposed package/import:** bo-craft / bo_craft, subject to package-name availability  
**Companion:** [Native surrogate improvement plan](fcmaes-surrogate-improvement-plan-2026-11-16.md)

## 1. Product decision

Create a Python package for advanced Bayesian optimization, with BoTorch as the first numerical backend. Integrate it with experiment-craft for adaptive campaigns and optimization-jobs for execution. Add Optuna and optional external-service adapters in later increments.

The purpose is to optimize expensive experiments that exceed the existing native surrogate's scope: observation noise, meaningful replications, categorical/conditional parameters, unknown constraints, multiple objectives and eventually variable fidelity or evaluation cost.

The first usable release is deliberately narrower: noisy scalar continuous optimization, explicit observation uncertainty, asynchronous evaluations, durable campaign continuation and a reproducible comparison. This vertical slice must work before the project advertises the broader roadmap.

Use existing numerical libraries. BO-craft's main engineering contribution is a correct, transparent connection between optimization decisions, typed observations, application evaluators and the existing execution/campaign contracts.

## 2. Verified baseline and missing capabilities

| Repository/service | Inspected source | Relevant state |
|---|---|---|
| fcmaes-rust | 4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b | Native deterministic GP surrogate; scalar evaluate profile and simulation run_seed |
| Prepared fcmaes-rust patch | 5942b80ecfbab543e8078105244e158aea80c4c1 | Earlier local delivery; cooperative-stop improvements; not on remote main when checked |
| optimization-jobs | a0697ed5c0143d39f2a2ec954c32bea0a2fc4f34 | Shared execution, native jobs, trusted commands, artifacts and MCP |
| experiment-craft | 5b13edf8a8bfc5dbea29f19dc2e2aee5facff4a9 | Python 3.12+, static sweeps/dependencies, SQLite campaign decisions, reports and local dashboard |
| External BO-MCP | 56d590b91ac120d0f22a3e41e770843d271808cd | Independent service with BoTorch/BayBE backends, campaigns and MCP/REST interfaces |

Three boundaries affect the implementation:

1. **experiment-craft currently dispatches campaign steps sequentially.** Adaptive candidate creation and bounded concurrent in-flight jobs are new features. Existing transactional decisions are a foundation, not an already implemented BO scheduler.
2. **Generic command execution is CLI-only in the inspected shared contract.** The existing optimization MCP submission tools reject command jobs. The first BO integration is CLI/library based; a proposal command must not be advertised as available through MCP.
3. **Remote backend commissioning remains open.** Local tests and simulated managers do not establish survival across real logout, reboot or hardware failure. Production fleet placement remains gated.

The existing native surrogate's TuRBO-style region, artifacts and simulation callbacks already exist. BO-craft does not duplicate or replace that implementation.

Before starting, fetch current main in every affected repository, record its exact commit and recheck discovery. Reconcile the earlier prepared patch with merges. Never move experiment-craft's baseline-lock.json to an unmerged commit.

## 3. Ownership and dependency rules

| Component | Owns | Must delegate |
|---|---|---|
| BO-craft | Search-space/observation schemas, backend capabilities, models, acquisitions, recommendations, optimizer snapshots and numerical diagnostics | Execution, host admission, campaign job reconciliation |
| experiment-craft | Adaptive experiment policy, trial IDs, accepted suggestions, observation ingestion, scientific budgets, continuation and comparative reports | Numerical model implementation and host scheduling |
| optimization-jobs | Trusted execution, CPU admission, process deadlines/cancellation, staged inputs, native handles and artifact transport | BO decisions and statistical interpretation |
| Domain repositories | Objective/constraint/fidelity meanings, numerical model versions, units and evaluator implementation | General BO algorithms and orchestration |
| fcmaes-rust | Native deterministic surrogate and population-method baselines; registered evaluation models | Advanced Python BO dependencies |

BO-craft must also be usable as an ordinary library without experiment-craft. That mode accepts caller-owned observations and returns suggestions; persistence and execution remain the caller's responsibility. A small demonstration loop is acceptable, but there is no second production campaign database or custom fleet runner.

When integrated, experiment-craft is the authoritative trial ledger. BoTorch model files and adapter state are backend artifacts/projections, with a recorded source-history hash. External Optuna/Vizier/BO-MCP state needs explicit reconciliation; it never silently overrides the campaign ledger.

## 4. Backend strategy

### 4.1 BoTorch first

Use BoTorch/GPyTorch directly for the initial numerical adapter. Start with float64 CPU execution, normalized continuous inputs and affine output standardization. Support known per-observation variances and inferred scalar observation noise as separate modes. Prefer numerically stable logarithmic acquisitions, initially qLogNEI for noisy scalar objectives.

Direct BoTorch integration fits the existing ownership split: campaign management already has a home. BoTorch itself recommends Ax for many end-user workflows; BO-craft takes on the integration responsibilities deliberately and must test them accordingly. Do not describe direct use as universally better than Ax.

### 4.2 Optuna second

Add a native Optuna adapter for TPE, categorical/conditional search spaces and later single-objective pruning. Preserve its trial/sampler semantics. TPE is a complementary method, not a GP model hidden behind a common name.

Pin sampler options explicitly, including startup count, seed, multivariate/group behavior and handling of running trials. Defaults change across releases. The adapter must not assume that every Optuna sampler supports every combination of constraints, dynamic spaces, multiple objectives and pruning.

### 4.3 OSS Vizier later

Add OSS Vizier as an optional study-service adapter after the local lifecycle is stable. Use a selected, pinned algorithm and advertise only its tested capabilities; the service API accepting a space does not prove that every designer supports it.

Keep study identity and suggestion/result IDs. The service chooses candidates; optimization-jobs executes the user's evaluators. Reuse an existing user deployment when configured rather than making a new server mandatory.

### 4.4 BO-MCP interoperability

Treat BO-MCP as an optional external service adapter and comparison target. It is not synonymous with BoTorch and is not the proposed name of our repository.

The inspected BO-MCP documentation describes multiple backends and campaign services. It also distinguishes standalone advanced modules from features routed through campaign suggestions; for example, the listed SAASBO/multifidelity/transfer module availability must not be converted into a claim that all those campaign options work. Query capabilities and run contract tests for the exact pinned revision.

Do not fork BO-MCP as the starting point. It brings campaign and service responsibilities that already belong to other repositories here. Reassess reuse through a small adapter if an existing deployment or feature provides a concrete advantage.

### 4.5 Planned capability matrix

This table is a delivery target, not a statement that these adapters exist today.

| Capability | First owner/route | Gate |
|---|---|---|
| Noisy continuous scalar BO | BoTorch | B2–B5 |
| Known observation variances and independent repeats | BO-craft schema + domain evaluator | B1–B3 |
| Pending-point-aware async proposals | BoTorch + experiment-craft | B4–B5 |
| Categorical/conditional domains | Optuna TPE first | B6 |
| Known input constraints | Explicit feasible-space validation | B7 |
| Unknown black-box constraints | BoTorch constrained acquisition | B7 |
| Noisy multiobjective BO | BoTorch qLogNEHVI | B8 |
| Fidelity/cost-aware BO | Explicit BoTorch adapter | B9 |
| Early stopping/pruning | Optuna plus intermediate-result evaluator contract | B9 |
| External Vizier / BO-MCP studies | Optional service adapters | B10 |
| Higher-dimensional BO | TuRBO/SAAS-style experiments | B11 |
| Agent workflow | CLI plus existing evaluator/job tools | B12; broader MCP support separate |

Every backend exposes a machine-readable capability description that covers combinations, not just independent booleans. Reject an unsupported combination before fitting or spending evaluations.

## 5. Proposed package and public surface

Use a src-layout Python package, with optional dependency groups. The names below are proposals to implement, not existing commands.

| Location | Purpose |
|---|---|
| src/bo_craft/specs/ | Strict, versioned space/objective/constraint/observation definitions |
| src/bo_craft/backends/ | BoTorch, Optuna, Vizier and BO-MCP adapters |
| src/bo_craft/transforms/ | Coordinate/metric transformations and inverse/report mappings |
| src/bo_craft/proposals.py | Candidate validation, deduplication, pending-point treatment |
| src/bo_craft/state.py | Backend snapshot identity, safe serialization and replay metadata |
| src/bo_craft/diagnostics.py | Fit health, uncertainty and proposal-cost diagnostics |
| src/bo_craft/cli.py | capabilities, validate, propose, recommend and inspect operations |
| tests/ | Numerical, schema, adapter, fault-injection and integration tests |
| benchmarks/ | Frozen workloads, external controls and analysis |
| examples/ and docs/ | Local library use, campaign examples, agent guide and limitations |

Initial Python baseline: 3.12, matching experiment-craft. Lock tested dependency versions in the implementation; do not install every backend together by default. Suggested extras are botorch, optuna, vizier and bo-mcp. CPU BoTorch is sufficient for the first release. GPU acceleration requires a measured workload benefit and a real resource-admission contract; it is not implicitly covered by CPU slots.

Proposed library contract:

~~~python
backend.capabilities()
backend.validate(problem_spec)
backend.propose(request)       # explicit observed history, pending trials, seed/state
backend.recommend(request)     # recommendation with explicit uncertainty semantics
~~~

An external adapter may also need observe/reconcile operations. Keep those adapter internals separate from the public optimization job profiles. In particular, do not change fcmaes.surrogate/1 into a session-oriented ask/tell service.

For experiment-craft, run expensive numerical proposal work as a trusted command job so its CPU/time use is admitted and recorded. A proposal command consumes immutable request/observation artifacts and returns a validated proposal artifact. Benchmark Python/Torch cold-start overhead. A persistent numerical worker is a later optimization with its own lifecycle tests, not a reason to omit startup cost.

## 6. Data model and scientific semantics

### 6.1 Search spaces and metric definitions

Version the problem definition with:

- Named continuous, integer, categorical and conditional parameters; type, bounds, units and log scaling.
- Activation conditions as a validated acyclic dependency graph. An inactive parameter is absent, not assigned a meaningful zero.
- Objective names, directions and units; internally chosen minimization/maximization mapping recorded explicitly.
- Known input constraints and observed black-box constraints as distinct types. Use g(x) <= 0 as the public feasibility convention.
- Optional fidelity definitions, target fidelity, costs and known resource upper bounds.
- Evaluator identity, executable/model version, parameter defaults and hashed inputs.
- Backend and resolved algorithm settings, schema version and numerical precision.

Integer rounding and categorical encoding must happen through a supported mixed-space method. Optimizing a continuous relaxation and rounding afterward is an explicit heuristic requiring its own validation.

### 6.2 Observations and noise

A logical trial contains one candidate and its intended fidelity; it can have multiple evaluation/replication attempts. Give each level a distinct immutable ID.

An observation records named raw metric values, constraint values, replicate seed, status, cost, fidelity, runtime/native handle, source digest and available uncertainty. Retain raw replicates when possible.

For aggregated replicates, store the count and the variance semantics explicitly. Variance of individual outcomes, variance of the estimated mean, standard deviation and standard error are different quantities. For independent equal-weight replicates, estimated variance of the mean is sample variance divided by replicate count; record this assumption. Correlated replicates need a different estimator.

A known per-observation variance model is not automatically an inferred heteroskedastic-noise model. Keep known-variance and learned-noise modes separate, with independent tests. Initially allow only affine output transforms for uncertain outcomes. Nonlinear transforms require a justified distributional treatment, not a casual rescaling of variance.

For current Scalar::simulation evaluators, fcmaes.evaluate/1 supports run_seed, with a seed derived from run and point. Independent replications therefore require distinct documented run seeds. Reusing the same run_seed and point reproduces a sample path, not a new independent observation. Batch or separate evaluation jobs accordingly and preserve the exact seeds.

Current evaluate simulations use one thread per point. A future evaluator requiring more threads needs a documented application/worker extension; BO-craft cannot assume that changing inner_threads creates that capability.

### 6.3 Candidate, pending and failure rules

Canonicalize physical parameter values after decoding. Deduplicate ordinary suggestions against completed and pending trials. Intentional replicates are exempt only when declared and assigned new replicate IDs/seeds.

Keep at least these outcomes distinct: completed usable, evaluator failure, invalid input, pruned, cancelled, and unknown execution. A process timeout is not a measured poor objective or certified infeasibility. Do not train the GP on an arbitrary 1e99 placeholder.

A model may legitimately return an infeasible observation with finite constraints. That differs from a simulator crash. A later failure classifier needs its own evidence and must preserve raw outcomes.

Pending candidates are inputs to the acquisition/backend. Register them when their accepted dispatch intent is durable; remove them only on a known terminal/abandoned transition. An unknown launch remains pending/reserved.

### 6.4 Recommendations and claims

Distinguish:

1. Best observed replicate or empirical aggregate.
2. Model-recommended point/posterior mean.
3. Independently validated estimate with a stated sampling protocol.

For noise, the most favorable sample is generally not a reliable incumbent estimate. Reserve a validation budget and use fresh seeds. Confidence/credible intervals must state which uncertainty they describe. A Bayesian credible interval is not automatically a calibrated frequentist guarantee.

Preserve native evaluation claims exactly. A native rechecked objective/bounds record concerns that concrete seeded evaluation; it does not certify the expectation of a stochastic objective. Statistical summaries live in a separate versioned BO artifact and campaign report. Do not reinterpret deterministic equality recheck as a noisy validation rule.

For multiple objectives, distinguish empirical and posterior Pareto sets, and freeze objective directions, normalization and hypervolume reference points before comparison. No BO result certifies a global optimum.

## 7. Adaptive campaign lifecycle and recovery

### 7.1 One authoritative ledger

Add a new versioned adaptive campaign format and state transitions to experiment-craft. Preserve the semantics of existing static campaign formats.

The campaign stores trial identity, suggestion request, accepted candidate, evaluation intent, native handle, ingestion receipt and backend-update status. Backend snapshots are bound to an exact observation prefix and pending-set digest. Immutable input snapshots remain immutable; append-only accepted decisions define the evolving campaign.

### 7.2 Decision/dispatch sequence

1. Reconcile existing native handles and accept verified new observations.
2. Persist a proposal-computation intent with history/pending hashes, resolved backend, seed and remaining scientific budget.
3. Submit the proposal command through optimization-jobs with a stable request key.
4. Collect and verify its artifact; validate space membership, support, uniqueness, cardinality and provenance.
5. Atomically accept selected suggestions, allocate trial/replicate IDs and reserve their scientific budget.
6. Persist each evaluation intent before submitting it to the runtime.
7. Observe the returned or predicted native handle. On uncertain acknowledgement, reconcile that handle without resubmitting the scientific work.
8. Ingest each verified terminal observation once using trial/attempt identity and digest.
9. Record any external-backend update intent and its receipt; then permit the next dependent decision.

There is no distributed transaction across all services. The guarantee is idempotent decision/ingestion with conservative reconciliation, not universal exactly-once physical execution.

### 7.3 Concurrency, budgets and stop

Extend experiment-craft to support bounded in-flight evaluation jobs and at most one proposal job initially. The runtime remains the capacity authority. Campaign concurrency is scientific policy, not a second CPU-slot ledger.

Maintain separate limits for accepted candidates, evaluation attempts, completed usable observations, replications, validation, proposal compute, declared slot-seconds and wall deadline. Failed/cancelled attempts still consumed work. Unknown work keeps reservations. A declared upper bound and actual measured cost are reported separately.

Persist stop before requesting cancellation. Stop during proposal collection prevents new evaluation intents. Stop during execution retains late results and their accounting. Only a confirmed terminal outcome releases the relevant in-flight reservation according to the versioned policy.

The campaign deadline is not an always-running group-enforcement service. When the campaign client is absent, native per-job limits are the existing enforcement boundary. Set proposal/evaluator wall limits from the remaining deadline, document overhead and retain the unresolved backend commissioning gate.

### 7.4 External backend reconciliation

For Optuna/Vizier/BO-MCP, persist study identity and operation intent before mutation. Use a documented idempotency mechanism where available. If an API does not provide one, reconcile by durable IDs or halt an uncertain mutation for inspection; do not blindly ask for another suggestion.

Detect duplicate results, missing receipts, orphan external trials, divergent study contents and incompatible backend upgrades. A backend database is a projection/required engine state, not a replacement for campaign evidence. Snapshot and restore its consistent state with the campaign using a tested procedure.

Switching backend is a new campaign or explicit warm-start lineage with a new identity. Do not silently change the optimizer inside an existing study.

## 8. Incremental implementation program

Every iteration delivers a focused patch/source update, tests, evidence and a decision log. Research iterations may finish with a rejected variant; they must not end with unsupported capabilities presented as implemented.

| Iteration | Deliverable | Repository changes |
|---|---|---|
| B0 | Scope, source audit, dependency lock and comparison protocol | BO-craft scaffolding |
| B1 | Strict specs, observations, capabilities and provenance | BO-craft |
| B2 | Synchronous noisy scalar BoTorch adapter | BO-craft |
| B3 | Real evaluator bridge and replication semantics | BO-craft + application examples |
| B4 | Adaptive concurrent campaign prototype | experiment-craft + BO-craft |
| B5 | Recovery, stop, idempotency and first-release acceptance | Both; shared gaps upstream only |
| B6 | Optuna and mixed/conditional domains | BO-craft + campaign examples |
| B7 | Known/unknown constraints | BO-craft + domain examples |
| B8 | Noisy multiobjective optimization | BO-craft + reports |
| B9 | Fidelity, costs and single-objective pruning | BO-craft + evaluator contracts |
| B10 | External Vizier and BO-MCP adapters | BO-craft |
| B11 | Higher-dimensional experiments and method selection | BO-craft + shared benchmarks |
| B12 | Documentation, distribution, discovery and operator walkthrough | Relevant repositories |

### B0 — Freeze the first release and audit dependencies

**Implement:** repository skeleton, license/provenance policy, Python 3.12 environment, optional extras, lint/type/test setup and a benchmark manifest. Recheck package-name availability before publishing. Pin BoTorch, Torch, GPyTorch, NumPy, SciPy and numerical threading settings as a tested set.

Read current experiment-craft state/manifest/runtime code and its integration contract. Inventory which changes are generic runtime gaps and which are adaptive scientific policy. Recheck evaluator schemas and source commits.

**Gate:** install/import/test on a clean CPU environment; a minimal dependency install does not import Torch or require Vizier; the first-release scope is B0–B5 plus the relevant B12 documentation.

### B1 — Schemas and capability validation

**Implement:** strict problem, observation, pending-trial, proposal and backend-state schemas; objective-direction transforms; noise semantics; typed errors; explicit canonical serialization and hashes. Reject unknown fields and unsupported capability combinations.

**Tests:** invalid variance/units/counts, duplicate identities, unknown categories, conditional cycles, nonfinite JSON, inconsistent metric directions, changed evaluator inputs, mismatched pending points and unsafe/unsupported state loads.

**Gate:** all acceptance/rejection fixtures are versioned. No package requires arbitrary code embedded in a job payload or untrusted Python pickle. BoTorch tensor state uses a safe, versioned serialization route.

### B2 — Synchronous noisy scalar BoTorch

**Implement:** continuous boxes, known observation variance or inferred scalar noise, qLogNEI, explicit initial design, finite batch sizes and pending-point inputs. Use normalized inputs, affine outputs and double precision. Handle model-fit failures with a recorded bounded policy; default to a typed failure unless an explicit random-design fallback is enabled.

Provide a library example and proposed propose/recommend CLI operations with immutable JSON/artifact inputs. Record effective priors/noise bounds, fit retries, optimizer settings, seed, candidate validation and proposal compute time.

**Tests:** small analytical/reference posterior cases, duplicate locations with independent observations, zero/tiny variance, offset/scale invariance within tolerances, candidate bounds, fixed-seed same-environment behavior and fit cancellation/time cap.

**Gate:** results match an independently constructed pinned BoTorch reference within declared tolerances. Initial quality is no worse than that reference under the same modeling choices. No async or mixed-space claim yet.

### B3 — Real objective execution and replications

**Implement:** an adapter from existing fcmaes.evaluate/1 results for registered scalar simulations, plus one trusted command evaluator for Python or external models. Bind point, run/replicate seed, model version, input hashes and output units.

Start with a seeded noisy synthetic function and an existing domain model only after verifying that its stochastic quantity and replicate protocol are meaningful. Deterministic numerical solver noise and physical stochasticity must be identified separately.

**Tests:** independent seeds at identical x, reproducibility of one seed, aggregate variance semantics, corrupted/missing outputs, cancellation, nonfinite values and provenance mismatch. Count training and validation callbacks separately.

**Gate:** a complete noisy optimization and fresh-seed validation run produces auditable observations through the real runtime. No evaluator capability is inferred solely from its repository name.

### B4 — Adaptive concurrent campaigns

**Implement:** a versioned adaptive manifest, dynamic accepted-trial records, pending-set updates and bounded multiple in-flight jobs in experiment-craft. Reuse its advancement lock, WAL/FULL durability, head tracking, immutable inputs and native handles.

First use a virtual-time/fake-runtime harness, then the real runner. Require one proposal request at a time; dispatch ready accepted candidates while evaluations are active. Refitting/acquisition sees a recorded snapshot. Results arriving during proposal computation enter the next decision unless the proposal is explicitly invalidated.

**Tests:** identical and variable durations, result races, proposal arriving after stop, budget exhaustion with pending work, duplicate candidates, partial batch capacity and failure of one evaluator. Existing static campaigns must remain unchanged.

**Gate:** no over-dispatch, no duplicate observation commit and no loss of late outcomes. Show actual overlap using real evaluator timing records; merely issuing concurrent futures does not prove concurrent jobs.

### B5 — Fault tolerance and first-release gate

**Implement:** continuation and recovery across every boundary in section 7. Verify backup/restore consistency for the adaptive ledger. Add an observation-only progress report covering pending, completed, failed, cancelled, unknown and validation work.

**Fault matrix:** terminate the client before/after intent persistence, during submission, after native acceptance, before ingestion, after ingestion but before model-state publication, during proposal compute and during stop. Exercise corrupted artifacts, disk/write errors and changed executable/input identities.

**Gate:** repeated continuation never blindly re-executes uncertain work; native results are ingested once; budgets survive restart; unknown state remains explicit. Test two independent campaign workspaces with the same scientific spec so they never share handles or cancel one another.

Complete B12's installation/CLI/agent documentation for the scalar release. This is the first release candidate; later capabilities remain visibly unsupported.

### B6 — Optuna and mixed/conditional spaces

**Implement:** Optuna TPE adapter with explicit sampler configuration and a tested storage/reconciliation approach. Use one canonical logical space with backend-specific encoding. Validate inactive parameters, integer ranges, categorical equality and log transforms.

Add a mixed engineering example and a genuinely conditional model/configuration problem. Exhaustive small spaces establish correctness; random/TPE/direct-Optuna baselines establish quality.

**Tests:** valid conditional branches, duplicate suggestions after decoding, resumed study consistency, cancelled/unknown trial handling and external-study divergence. Do not require a GP to treat arbitrary category IDs as continuous distances.

**Gate:** supported spaces are never silently relaxed. Direct Optuna and the adapter produce consistent valid trial semantics; all unsupported combinations fail before execution.

### B7 — Known and black-box constraints

**Implement:** exact known-input filtering/parameterization where supported; separate modeled outputs for observed g(x). Start with scalar constrained acquisition and a documented feasibility-first initial phase when no feasible observation exists.

Use joint/independent modeling assumptions explicitly. Fit constraint noise only under a supported noise model. A probabilistic feasible recommendation carries a probability/model statement; it is not a deterministic feasibility certificate.

**Tests:** sign conventions, equality tolerance, no feasible points, objective unavailable at infeasible points, independent constraint noise, evaluator failure distinct from infeasibility and decoded candidate violations.

**Gate:** independently evaluated constraint checks agree with reports. Compare time/calls to first feasible point and validated feasible quality, retaining all violations and failures.

### B8 — Noisy multiobjective BO

**Implement:** explicit objective directions, model selection, qLogNEHVI, bounded batch/candidate sizes, empirical/Pareto recommendation artifacts and predeclared reference points. Keep scalarized methods as explicit comparison arms.

**Tests:** mixed minimize/maximize directions, duplicate/dominated points, two-objective exact hypervolume fixtures, constraints, noise, reference-point changes and validation of the final recommended set.

**Gate:** adapters agree with independent small-problem Pareto/hypervolume calculations and pinned direct BoTorch. Reports keep observed, posterior and independently validated fronts separate. Higher objective counts need separate runtime/quality evidence.

### B9 — Fidelity, evaluation cost and pruning

**Implement:** a declared fidelity variable or rung structure, a target fidelity, cost accounting and a pinned cost-aware/multifidelity acquisition. Validate that the proposed low-fidelity model is related to the target rather than treating any cheaper simulation as interchangeable.

Add pruning only when the evaluator emits durable, attributable intermediate metrics and can stop cooperatively. Use Optuna's tested single-objective pruning path first. Keep pruned observations distinct from completed target-fidelity objectives.

**Tests:** misleading low-fidelity rankings, changed cost models, partial cost on cancellation, minimum target-fidelity validation, resource limits, duplicated rung results and stop races.

**Gate:** compare at equal total declared/measured cost and equal target-fidelity validation. A reduction in full-fidelity calls alone does not prove improvement. Unsupported multiobjective pruning remains rejected.

### B10 — Optional external services

**Implement:** one adapter at a time, OSS Vizier first unless an actual BO-MCP deployment is the immediate need. Pin service/client versions, discover capabilities and use stable study/suggestion/result identifiers.

Keep credentials and endpoint configuration outside exported experiment artifacts. Preserve enough nonsecret endpoint/study identity to reconcile. Include network time and suggestion-service compute where measurable; report unknown costs honestly.

**Tests:** lost responses, service timeout after mutation, duplicate result submission, changed capabilities, unsupported search spaces and restoration with outstanding suggestions. Reconcile orphan trials without issuing new scientific work by guesswork.

**Gate:** one real disposable local service test for each advertised adapter. Mock-only support is labeled experimental/unvalidated. All external protocol drift is contained inside that optional adapter.

### B11 — Larger-dimensional and larger-history experiments

**Implement:** controlled trials of local trust regions, SAAS-style sparsity assumptions, bounded training sets and/or approximate models as supported by pinned backends. Keep strategy choice explicit before considering any automatic selector.

Benchmark actual training/proposal cost and evaluation savings. Increasing a dimension field does not establish practical high-dimensional optimization. Compare with native BiteOpt/DE/CMA-ES, native surrogate where supported and Optuna TPE.

**Gate:** a declared workload class benefits at equal wall/cost with maintained validation quality. Record negative results and routing guidance. Do not advertise general superiority on permutations, routing or arbitrary high-dimensional spaces.

### B12 — Documentation, packaging and agent workflow

**Implement:** README, AGENTS.md, ai-context.md, architecture guide, user manual, capability matrix, schema reference and runnable examples. Provide a quick CPU setup with only the selected backend.

Add experiment-craft walkthroughs for noisy scalar optimization, mixed conditional search, constrained optimization and multiobjective validation only as their gates pass. Update the shared ecosystem solver chooser with precise links and limitations.

The initial agent recipe uses shell/CLI operations and existing job observation tools. Document the current command/MCP gap. Any later generic command support in optimization-mcp is a separate optimization-jobs change with capability/policy tests; do not build a second MCP server or claim this support automatically.

If dashboard views change, test rendered phone and desktop layouts, including pending/unknown work and uncertain objective estimates. Viewing a report must never dispatch evaluations.

**Gate:** a fresh environment follows the instructions successfully; capabilities match executed examples; package metadata and licenses are checked; source and cross-repository patch ZIPs plus a complete implementation log are ready. Tagging/publishing requires a separate release action.

## 9. Benchmark and validation campaign

### 9.1 Panels and baselines

| Panel | Workloads | Essential comparisons |
|---|---|---|
| Deterministic continuity | Branin/Hartmann and existing RK4 example | Native surrogate, direct BoTorch, random/design |
| Noisy scalar | Known synthetic mean plus homoskedastic and known varying observation variances | Direct BoTorch, random/design, Optuna where supported |
| Replication | Same x with independent seeds and a known mean | Fixed and adaptive replication policies |
| Async | Virtual duration distributions plus real variable-duration callbacks | Synchronous same backend, async same backend |
| Mixed/conditional | Finite cases with enumerated truth; a versioned conditional application | Direct Optuna TPE, random, appropriate population baseline |
| Constraints | Known-feasible synthetic cases and one domain evaluator | Direct constrained BoTorch, feasible random baseline |
| Multiobjective | Noisy Branin–Currin-style and constrained synthetic cases | Direct qLogNEHVI, scalarization, MODE where appropriate |
| Multifidelity | A documented correlated low/high-fidelity pair plus a deliberately misleading pair | Target-fidelity-only BO and random |
| Failure/recovery | Kill, timeout, corrupt output, duplicate/lost service response | Contract invariants, not objective ranking |

Native deterministic BO may be included on a fixed-seed deterministic objective, but must not be presented as a statistically equivalent noisy optimizer. Population methods need their own declared noise/replication policy. Match the scientific target, not just the parameter bounds.

### 9.2 Comparison design

Freeze problem definitions, reference values, initial observations, seeds, resources and stopping rules before confirmation. Use the same observations for compatible initial designs and retain a native-default-start arm if initialization differs.

Report both evaluation budget and elapsed/cost budget. Include fitting, acquisition, Python import, service/network time, job startup, durability and validation. When external-service CPU cost is unavailable, mark it unknown and avoid a total-compute superiority claim.

Use 10 development and 30 held-out seeds for cheap synthetic panels as an initial target. Use at least five independent real-simulation pilot seeds; expand before claiming a broad improvement. Statistical power depends on effect size and variation, not just the nominal seed count.

For noisy synthetic cases, compare recommendations using the known latent mean or a large independent validation sample. For real models, use fresh held-out replicates and report their uncertainty. Avoid selecting and reporting the same optimistic replicate.

Record best validated quality, target success, feasible success, independent hypervolume, time-to-target with censoring, evaluation/failure counts, repeated evaluations, proposal latency, peak RSS and resource settings. Keep all failed or incomplete arms in the report. Aggregate normalized outcomes only under a predeclared rule.

### 9.3 Suggested acceptance thresholds

These are proposed engineering gates to freeze in B0, not achieved results.

- **Adapter correctness:** schema, transformation, reference numerics and state-machine tests pass.
- **Wrapper fidelity:** with identical supported backend settings, results match the direct backend within declared tolerances; integration adds no hidden algorithm change.
- **Recovery:** zero duplicate ingestions or blind resubmissions in the fault matrix; unknown work remains explicit.
- **Resource bounds:** no unbounded queue/model/history growth and no unreserved thread pools.
- **Async benefit:** at least 15% median time-to-target reduction on the variable-latency confirmation stratum, with no material quality regression at equal cost.
- **Quality promotion:** per-stratum upper 95% paired-bootstrap bound on normalized quality-loss ratio <=1.05 where meaningful, and no material target/feasibility-success decline. Define the exact metric before running the panel.
- **Scalar first release:** B0–B5 functionality and documentation may ship as an experimental integration even if BO does not beat every alternative; publish the measured routing limits.
- **New recommended default:** requires held-out benefit, no unresolved material regression and explicit uncertainty reporting.

For an initial decision rule, treat a greater-than-five-percentage-point loss in target/feasibility success as material. Insufficient evidence leaves a feature opt-in. Multiple objectives and fidelity need their own declared quality metric; do not reuse scalar regret mechanically.

### 9.4 Runtime safety for experiments

Default smoke tests use millisecond or virtual-time evaluators and small observation caps. Begin numerical benchmark workers conservatively, with explicit Torch/BLAS thread counts and peak-memory observation.

Use per-run timeout plus an independent watchdog, 30-minute shards and a predeclared total allowance (initially at most four hours per development campaign). Persist each cell's completed result immediately. Resume only unfinished immutable cells; retain failed/censored cells rather than silently rerunning until success.

Do not run expensive fully Bayesian/high-dimensional methods across the whole suite before a single-run memory/time check. GPU, long simulator and external-service panels are separate profiles. This plan can be developed in the current environment without waiting for the home machines.

## 10. Integration and merge order

1. Implement BO-craft's strict numerical/package surface and local examples.
2. Make any demonstrated generic execution/protocol fixes in optimization-jobs first; keep BO-specific schemas out of its generic envelope.
3. Add application-owned evaluator improvements where the current contracts are insufficient.
4. Add versioned adaptive campaigns to experiment-craft, pinned to merged runtime/evaluator changes.
5. Run the real combined integration and scientific acceptance panels in a fresh workspace.
6. Update ecosystem discovery, examples and reports.
7. Prepare release artifacts only after the capability matrix matches the tested implementation.

Where work crosses repositories, deliver separate patches with exact bases and a combined manifest. Existing static campaigns, native solver profiles, result claims, backup/restore and dashboard behavior retain regression coverage.

## 11. Now versus January scope

**Now:** Python package, schemas, BoTorch and Optuna adapters, numerical fixtures, real local evaluator jobs, bounded local async campaigns, fault injection, CPU benchmarks, documentation and phone-friendly downloadable deliverables.

**January/home lab:** sustained 16-core configurations, real seven-machine operation where commissioned, GPU crossover measurements, actual logout/reboot/resource-enforcement tests and longer simulation panels.

The home-lab schedule does not block B0–B8 local development. It does block claims that untested production fleet behavior is complete. Keep planned hosts disabled until their runtime commissioning evidence exists.

## 12. Definition of completion and deliverables

The project is complete at a declared milestone when every advertised capability has a test, a runnable example, a resolved schema and a published limitation. Optional research that failed a gate is explicitly rejected/deferred with evidence.

Each implementation delivery should include:

- Source ZIP for the new repository and patches for existing repositories.
- Exact base/final commits, dependency lock, environment and artifact checksums.
- An iteration log stating what changed, tests run, measured results, failures and remaining gates.
- Compact raw JSONL/CSV evidence and scripts that regenerate summary tables.
- README, architecture guide, user manual and agent instructions.
- A capability matrix distinguishing implemented, experimental, rejected and planned behavior.
- Real-integration results and a clear statement of hardware/service tests still pending.

The recommended first release boundary is the coherent B0–B5 vertical slice plus its documentation. A full-plan implementation continues through B6–B12, retaining experimental status where gates fail. Later increments expand capabilities without obscuring that initial acceptance boundary.

## 13. Sources and checked entry points

Sources were reviewed on 16 November 2026. Pin external package/service versions again at implementation start.

- [Native surrogate baseline](https://github.com/dietmarwo/fcmaes-rust/blob/4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b/crates/fcmaes-surrogate/README.md)
- [Existing evaluation model/run-seed semantics](https://github.com/dietmarwo/fcmaes-rust/blob/4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b/docs/job-api/models.md)
- [experiment-craft campaign semantics](https://github.com/dietmarwo/experiment-craft/blob/5b13edf8a8bfc5dbea29f19dc2e2aee5facff4a9/docs/campaigns.md)
- [experiment-craft execution integration contract](https://github.com/dietmarwo/experiment-craft/blob/5b13edf8a8bfc5dbea29f19dc2e2aee5facff4a9/docs/integration-contract.md)
- [experiment-craft contribution/ownership rules](https://github.com/dietmarwo/experiment-craft/blob/5b13edf8a8bfc5dbea29f19dc2e2aee5facff4a9/AGENTS.md)
- [optimization-jobs shared API](https://github.com/dietmarwo/optimization-jobs/blob/a0697ed5c0143d39f2a2ec954c32bea0a2fc4f34/docs/job-api.md)
- [BoTorch repository and intended integration surface](https://github.com/meta-pytorch/botorch)
- [BoTorch multiobjective acquisitions](https://archive.botorch.org/docs/multi_objective)
- [BoTorch constrained multiobjective tutorial](https://botorch.org/docs/tutorials/constrained_multi_objective_bo)
- [BoTorch TuRBO tutorial, versioned reference](https://botorch.org/docs/v0.17.0/tutorials/turbo_1)
- [Optuna conditional search-space definitions](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html)
- [Optuna algorithms and pruning](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
- [Optuna TPE sampler reference](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [OSS Vizier repository](https://github.com/google/vizier)
- [BO-MCP inspected revision](https://github.com/AccelerationConsortium/bo-mcp/tree/56d590b91ac120d0f22a3e41e770843d271808cd)

These sources support capability and architecture descriptions. Proposed integration choices and acceptance thresholds are design decisions in this document, not promises made by the external projects.
