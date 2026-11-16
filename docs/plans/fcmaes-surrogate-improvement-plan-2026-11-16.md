# Native fcmaes-surrogate improvement plan

**Date:** 16 November 2026  
**Status:** implementation plan; proposed work and acceptance thresholds, not measured new results  
**Owner:** dietmarwo/fcmaes-rust  
**Companion:** [BO-craft implementation plan](bo-craft-implementation-plan-2026-11-16.md)

## 1. Objective and product boundary

Improve the existing pure Rust surrogate for deterministic, scalar, continuous, box-bounded simulation optimization. The product target is a useful answer in less elapsed time, including model fitting, proposals, evaluation, logging, startup and validation. Maintain a small native dependency set and a usable in-process Rust API.

The work stays in fcmaes-rust, primarily crates/fcmaes-surrogate and its existing job adapter. Retain the native implementation as a practical choice for evaluations costing roughly 1–60 seconds, while measuring the crossover against BiteOpt, DE and CMA-ES. Fast test functions are numerical and scheduling fixtures, not evidence of an advantage on expensive simulations.

The companion BO-craft plan owns statistical noisy optimization, categorical and conditional spaces, general black-box constraints, multiobjective BO, multifidelity BO, and adapters to BoTorch, Optuna and external services. A larger nugget or a repeatable random seed does not turn this deterministic method into statistical noisy BO.

Incremental Rust async work is confined to evaluations within one already admitted optimization task. experiment-craft owns adaptive campaigns across jobs; optimization-jobs owns admission and execution. There is no new native ask/tell service, fleet scheduler or MCP server in this plan.

## 2. Baseline and evidence already available

The following source states were inspected on 16 November 2026:

| Item | Inspected state | Meaning |
|---|---|---|
| fcmaes-rust remote main | 4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b | Published repository source baseline for this plan |
| Prepared local fcmaes-rust patch | 5942b80ecfbab543e8078105244e158aea80c4c1 | Earlier delivery in this conversation; not present on remote main when checked |
| Surrogate source comparison | No changes in crates/fcmaes-surrogate between those two states | This plan is grounded in the same existing numerical implementation |
| optimization-jobs remote main | a0697ed5c0143d39f2a2ec954c32bea0a2fc4f34 | Shared runner reference; workers may pin an older revision |
| Existing surrogate package | fcmaes-surrogate 0.1.0, source-only | Not part of the published core/Python distributions |

Before implementation, fetch main, record the exact new base, inspect merges and read AGENTS.md. Apply or rebase the previous stop-control patch only if its behavior is still absent; never assume a local delivery has been merged. Use an isolated worktree when local work exists.

### 2.1 Current capabilities

| Area | Implemented baseline |
|---|---|
| Domain | 1–32 continuous coordinates; optional log-scaled bounds |
| Limits | At most 256 new plus warm observations; logical batches 1–8 |
| GP | Exact ARD Matérn-5/2 and RBF; unit signal variance; configurable deterministic nugget; bounded jitter escalation |
| Fitting | Analytic-gradient MAP fit with a compact bounded L-BFGS implementation |
| Acquisition | Stable LogEI and minimization LCB; candidate search plus coordinate refinement |
| Initial design | Centre plus maximin Latin hypercube design |
| Local search | One TuRBO-style region; two of every three proposals local, one global |
| Batch policy | Synchronous Kriging-believer batches; pending-point fantasies at the library proposal interface |
| Linear algebra | Packed row-major Cholesky and fixed reduction order; fantasy factor append |
| Integration | Existing fcmaes.surrogate/1 profile; registered scalar and simulation models |
| Reliability | Fallible completion logging, checked traces, artifacts, warm snapshots, cancellation and independent deterministic recheck |
| Reproducibility | Ordered synchronous decisions; tested same-host thread-count consistency; semantic trace replay |

The present trust-region implementation deliberately differs from published TuRBO: it uses LogEI, fits global history, retains a periodic global proposal and resets its region around the incumbent. It must continue to be described as TuRBO-style unless a distinct implementation matches a documented reference.

### 2.2 Findings to preserve

Existing comparisons favor the native method on several small smooth problems, but also show important limits. BiteOpt outperformed the GP methods on the tested separable many-modal Styblinski–Tang-10 task. Some external methods performed better at intermediate budgets. Ten-seed experiments and small functions do not establish general superiority.

The following variants were already tried and not adopted: acquisition-gradient refinement and a hybrid variant, a fantasy-adjusted incumbent, several larger candidate/refinement settings, alternate global-proposal frequencies, and more frequent fitting. Revisit them only with a new hypothesis, a controlled ablation and a broader held-out suite.

Some historical PLAN.md ledger entries are stale: live artifact paging and simulation support in retry/evaluate already exist. N0 reconciles the documentation against source before adding work.

## 3. Decisions fixed for this program

1. Preserve existing synchronous behavior as an explicit compatibility mode. New search and scheduling policies start opt-in.
2. Separate the complete observation archive from the set used to fit one GP. Increasing a history limit must not silently create unbounded cubic work.
3. Keep deterministic numerical regularization separate from observation noise.
4. Retain all failed, cancelled, pending and rejected work in the audit trail. Do not manufacture finite training values for failures.
5. Keep callback truth and optimizer diagnostics separate. A replay checks decisions; an exact recheck checks a deterministic evaluation; neither proves a global optimum.
6. Keep a plain, bounded first implementation of each feature. More sophisticated variants require a measured reason.
7. Treat a failed performance gate as a legitimate result. Retain the evidence and keep the feature experimental or remove it.
8. Do not block cloud-executable numerical work on access to the home lab. Reserve actual 16-core, multi-host, reboot and resource-containment claims for the appropriate environment.

## 4. Architecture and contracts

### 4.1 Module ownership

| Existing location | Planned responsibility |
|---|---|
| src/gp.rs and src/linalg.rs | Numerical diagnostics, factor reuse, posterior sampling support, bounded model construction |
| src/strategy.rs | Explicit policy/state, trust-region variants, active training-set selection |
| src/design.rs | Verified optional designs and sparse perturbation policies |
| src/driver.rs | Synchronous compatibility driver, bounded asynchronous driver and handoff accounting |
| src/space.rs and src/transform.rs | Coordinate/transform identity and cache invalidation |
| crates/fcmaes-job-worker/src/surrogate.rs | Payload validation, resources, artifacts, warm inputs and native evidence |
| scripts/surrogate and benchmarks/surrogate-comparison | Reproducible experiments and reports |
| docs/surrogate and docs/job-api/surrogate.md | Status ledger, behavior, limitations and examples |

Suggested new modules are archive.rs, sampling.rs and async_driver.rs if they simplify ownership. Their names are implementation choices. Keep the numerical crate independent of optimization-job-api.

### 4.2 Versioning and compatibility

The current event records and warm schemas reject unknown fields. Decide versions before changing serialized structures.

- Keep a decoder and fixtures for events/1 and observations/1.
- Use new event/snapshot versions for expanded policy state and async semantics.
- Preserve fcmaes.surrogate/1 behavior, defaults and limits. Introduce an opt-in fcmaes.surrogate/2 profile when new scheduling, archive or accounting semantics need it.
- A compatible additive /1 change is acceptable only if old readers, strict validation and old semantics remain demonstrably valid; document that decision.
- Prefer new advanced Rust configuration/run types where adding fields would break external struct literals.
- Resolve every effective option and algorithm version in discovery and run provenance. Reject unsupported combinations.
- Warm-from-observations remains a new run. Exact optimizer checkpoint/resume is outside this program unless separately designed and validated.
- Shared envelope or evidence changes land in optimization-jobs first. Do not invent common fields in worker result details and present them as shared guarantees.

### 4.3 Numerical cache identity

A reusable factor/model is identified by ordered training IDs and values, normalized coordinates, kernel and hyperparameters, nugget, actual numerical jitter, output-transform definition, and model/policy version.

Appending a deterministic observation is not automatically the same as appending a fantasy. Re-estimating standardization changes transformed targets. A changed bilog/auto transform may require reconstruction; changed kernel parameters or training selection invalidate the factor. With an unchanged covariance but changed affine targets, recompute the right-hand side/solves correctly. Verify all fast paths against reconstruction.

For a prototype that freezes the output transform between refits, make that a distinct model policy. Do not silently change the baseline semantics to enable caching.

### 4.4 Archive and training limits

Initially retain the current maximum training size of 256. Add a separate advanced archive budget, first validated at 1,000 observations and then 10,000. Keep raw observations durable; an active GP may use only a documented subset.

Training selection must always retain the usable incumbent, account for local geometry and boundary information, and include a deterministic global coverage quota. Record the selected IDs. Do not use the held-out objective or benchmark optimum in selection. Training subsets and duplicate suppression use distinct definitions; old points outside the GP are still evaluated points.

Only raise the training cap to 512 or 1,024 after direct memory, latency and numerical measurements. Limit fantasy/pending augmentation separately. Report O(n_train²) model storage, O(n_train³) refactorization work and archive cost, including constants measured on the actual host.

### 4.5 Resources, stopping and evidence

Synchronous mode can alternate model work and evaluations in one bounded pool. Concurrent model fitting and evaluations need explicit aggregate reservations:

    concurrent_evaluations * evaluation_threads + surrogate_threads <= inner_threads
    outer_workers * inner_threads <= cpu_slots

A coordinator that also consumes CPU is included in the reservation policy. The implementation must not create a full-size private pool for every model fit or callback. SimContext.threads() is an allowance, not per-call OS enforcement.

Record separate optimization attempts, successful/failed completions, cancelled partial work, validation attempts, fitting work, proposal time, logging time and idle/straggler time. Pending proposals consume dispatch reservation, not completed-evaluation count. On stop, abandon undispatched proposals, stop new dispatch and drain or cooperatively cancel active work. Never recycle the budget of uncertain work.

Preserve best_found only for an actually observed usable point. Rechecks follow deterministic exact-agreement rules. Penalty objectives remain scalar objectives; their low value does not certify separately declared constraints.

## 5. Incremental implementation program

Each numbered iteration ends with a focused commit, tests, benchmark records, updated documentation and a concise accepted/rejected/deferred decision. The sequence is dependency-driven; elapsed-time estimates should be made after N0, not inferred from these iteration numbers.

| Iteration | Main result | Depends on |
|---|---|---|
| N0 | Reconciled baseline and frozen experiment protocol | — |
| N1 | Diagnostics and bounded failure behavior | N0 |
| N2 | Safe GP reuse and numerical-kernel measurements | N1 |
| N3 | Improved candidate/design experiments | N0, N1 |
| N4 | Genuine restart and local-data policies | N1, N3 |
| N5 | Correlated posterior sampling | N1, N2 |
| N6 | Larger archive with bounded fitting | N2, N4 |
| N7 | Event-driven async simulator and trace contract | N1, N4 |
| N8 | Native asynchronous evaluation and background fitting | N6, N7 |
| N9 | Measured overhead governor | N2, N8 |
| N10 | Budget-correct handoff to population optimization | N0, N6 |
| N11 | Held-out comparisons and decision on defaults | Applicable earlier iterations |
| N12 | Documentation, packaging and ecosystem integration | N11 |

### N0 — Establish the actual baseline

**Implement**

- Record commits, dependency locks, toolchains, CPU quota/affinity, memory and existing limits.
- Reconcile the current source, README, status ledger and previous prepared patch.
- Freeze common objective definitions, initial observations, seeds, budgets, output conventions and comparison scripts.
- Capture full baseline trajectories and the current options-off behavior; retain the old binaries or source manifest.
- Add a bounded campaign harness with per-run watchdogs, shard manifests and incremental records.

**Verify**

Run the surrogate numerical/driver tests, real worker tutorial, shared schema checks and a small equal-call/equal-wall comparison. Confirm that missing external dependencies are reported, not silently skipped in a required comparison.

**Gate**

The same input can be rerun, every result has provenance, and interrupted benchmark execution resumes only unfinished experiment cells. No new algorithm claims are made.

### N1 — Numerical diagnostics and failure boundaries

**Implement**

Expose effective jitter, conditioning indicators where available, fit failures, fallback reason, predictive-variance clamping, selected hyperparameters, training count and proposal termination. Bound line searches, factor retries, candidate retries and logging queues.

Check stop signals around long fitting, candidate and matrix blocks. Document the maximum uninterruptible operation. Keep Error::Invalid, Error::Numerical, Error::Stopped and Error::Log distinct.

**Verify**

Independent fixtures for constant and duplicate data, near-singular kernels, log-space extremes, nonfinite values, ill-scaled outputs, saturated LogEI tails, invalid warm state and logger failure. Compare posterior means, variances and MAP gradients with independent high-precision/reference computations.

**Gate**

No silent NaNs, fabricated observations or swallowed callback/log errors. Correctness holds under the existing limits. Diagnostics remain bounded and options-off trajectories remain unchanged except for demonstrated bug fixes with dedicated evidence.

### N2 — Cache reuse and linear algebra

**Implement**

Prototype real-observation factor append, prediction workspace reuse and reusable kernel distance calculations. Benchmark fixed-hyperparameter rounds separately from refit rounds. Establish invalidation rules from section 4.3.

Compare the existing packed kernel with a blocked alternative only at measured sizes. Optional faer/other dependencies remain feature-gated until they improve whole-round time and preserve the required numerical behavior. Do not optimize a microkernel at the expense of fitting or proposal latency.

**Verify**

For randomized and pathological observation sequences, compare every update with fresh reconstruction. Cover transformed outputs, jitter changes, training-window changes, failed append, posterior variance and cancellation.

**Gate**

Target at least 20% lower median non-refit proposal latency on the intended reuse panel, with no more than 5% regression in the predeclared small-problem stratum. The numerical tolerance is established from independent fixtures before performance tuning. If the speed gate fails, retain only diagnostics or low-risk improvements.

### N3 — Candidate generation and acquisition search

**Implement**

Ablate one change at a time: verified scrambled Sobol design, sparse coordinate perturbations, dimension-aware candidate budgets, hybrid global/local candidates and bounded acquisition refinement. Retain the current LHS/coordinate implementation as a baseline.

Use a vetted source and license for Sobol direction numbers. Test actual sequence properties; do not claim general net discrepancy guarantees from a handful of projections. Revisit gradients only with a hypothesis explaining the earlier quality regression.

**Verify**

Bounds and log-coordinate round trips, duplicate detection after physical-coordinate canonicalization, all-fixed/degenerate accepted-space rules, deterministic seeds and pending-point avoidance.

**Gate**

Better equal-call or equal-wall performance in a declared stratum, with held-out confirmation. Faster acquisition maximization alone is insufficient. Keep new policies opt-in until N11.

### N4 — Trust regions and useful restarts

**Implement**

Keep the existing hybrid region unchanged as a policy. Add a separate policy with a real restart: reserve a new initial design, record a new region identity, and reset local model data explicitly. Separate global incumbent history from region-local observations.

Specify success thresholds robust to objective offset/scale. For async experiments, attribute each result to its proposal region/version and update counters in declared evaluation-count windows; do not apply a batch-size formula blindly to completion order.

Prototype multiple regions only after one-region restarts pass. Bound region count and total model work. Charge restart evaluations to the same global budget.

**Verify**

Offset/scaling cases, constant functions, tiny improvements, clipped regions, restart during cancellation, stale region completions and archive deduplication.

**Gate**

Restart behavior is auditable and budget-correct, and improves the designated trapping/multimodal panel. Label deviations from the published method. A negative result leaves the current hybrid as default.

### N5 — Correlated posterior sampling

**Implement**

First add a small, exact finite-candidate posterior sampler as a correctness oracle. Then prototype decoupled/pathwise sampling using a kernel-appropriate prior approximation plus conditioning. Validate Matérn spectral sampling independently.

Posterior Thompson samples must preserve cross-candidate correlation. Independent draws from marginal means/variances are not an acceptable substitute. Preserve a sampled path when scoring its candidates; regenerating it per candidate changes the algorithm.

**Verify**

Empirical posterior means, variances and selected covariances against the exact oracle, separating finite Monte Carlo error from approximation bias. Test repeated evaluation of one path, feature-count convergence, deterministic stream assignment and cancellation.

**Gate**

Predeclared moment/correlation tolerances and bounded memory pass; whole-proposal cost is competitive at the intended candidate count. Keep feature count explicit. No claim of exact posterior sampling for an approximate RFF construction.

### N6 — Larger budgets through bounded training sets

**Implement**

Add separate archive and training caps, deterministic local selection and coverage retention. Maintain an efficient all-history duplicate index with a correctness fallback. Bound warm-input size and parsing.

Start with archive sizes 1,000 and 10,000 while keeping n_train <=256. Test optional training sizes 512/1,024 only after memory/latency profiling. A local subset must not discard records from the durable archive.

**Verify**

Full history versus selected-history accounting, reproducible selection, incumbent retention, bound changes, malformed warm inputs and memory scaling. Measure both a locally smooth problem and a case where global information matters.

**Gate**

Memory and model work obey declared caps; the larger archive improves at least a targeted simulation stratum without concealed proposal overhead. Raising constants alone does not complete this iteration.

### N7 — Simulate asynchronous decisions before using threads

**Implement**

A deterministic virtual-time evaluator with scripted durations, failures, stragglers, cancellation and callback ordering. Define proposal, model-adoption, dispatch, completion, commit, abandonment and finish events.

Persist the exact history/pending snapshot and model version used by each proposal. Commit durable completions once; late completions remain valid observations. Define queued-point invalidation and tie-breaking.

**Verify**

Exhaustive small schedules plus generated event sequences, stop/finish races, budget exhaustion, stale proposals and sink errors. Replay recorded order and decisions.

**Gate**

Zero duplicate commits or budget overshoot. Replaying an event sequence reproduces decisions. Explicitly distinguish event-order reproducibility from seed-only reproducibility.

### N8 — Native asynchronous driver and bounded background refits

**Implement**

Connect the tested state machine to existing scalar/simulation callbacks. Keep evaluation and surrogate capacity within the aggregate reservation. Allow only a bounded number of pending/queued points and one background refit initially.

Build immutable model snapshots. Adopt a completed fit at a recorded safe boundary after checking training prefix, transform, region and configuration compatibility. Continue using the previous model only under an explicit stale-model policy.

Start with two or four evaluations plus one reserved surrogate slot when the host quota permits. Keep the old eight-thread synchronous cap until a separate larger-pool validation.

**Verify**

Real callbacks with variable latency, deadline in model work, deadline in evaluation, callback failure, stop with queued work, logger failure and nested Rayon simulation work. Record peak thread count and actual callback counts.

**Gate**

Scheduling invariants pass with the real worker. Target at least 15% lower median time to target on the predeclared variable-latency stratum; no more than 5% equal-wall loss on the uniform-latency stratum. Simulated durations establish scheduler behavior, not real solver speed.

### N9 — Overhead governor

**Implement**

An opt-in policy that can adjust refit interval, candidate budget, training cap and proposal queue length within validated bounds. Base decisions on separate observed model/proposal, evaluation, startup and durability costs.

Begin with transparent thresholds and hysteresis. Record measurements, controller version, action and reason. Keep a fixed-policy control arm. Time measurements make the run timing-dependent; record them for replay.

**Verify**

Oscillating costs, cold starts, abrupt evaluator cost changes, tiny remaining budgets and stale timing windows. Ensure controller work itself is accounted for.

**Gate**

Target proposal/model overhead below 10% of campaign elapsed time on the declared costly-evaluation stratum, with equal-wall performance no worse than the fixed policy. This is a workload-specific target, not a promise for cheap objectives.

### N10 — Budget-correct handoff

**Implement**

Add an explicit one-way handoff to BiteOpt or CMA-ES first; consider DE only after those pass. Use an observed incumbent and a documented initial scale/population policy. All optimizers share one evaluator, stop context, trace and remaining budget.

Integrate the prepared cooperative-stop changes if absent upstream. Count optimizer initialization and partial final populations. Return the best of all observed evaluations, including an improvement made immediately before stop.

The first trigger is an explicit evaluation fraction or stagnation count. A learned/automatic trigger requires a later benchmark decision.

**Verify**

Remaining budgets 0, 1, smaller than a population and exact-boundary cases; nonfinite evaluations; cancellation; log bounds; no usable incumbent; recheck after handoff.

**Gate**

No duplicated initial evaluations unless explicitly requested and charged; no lost improvements or budget overshoot. Hybrid improvement must exceed both a native-surrogate control and the population-only baseline on a declared target class.

### N11 — Freeze and evaluate

Freeze all candidate settings after development runs. Run held-out comparisons, include negative results, and produce the routing rule: which dimensions, evaluation costs, budgets and concurrency favor each method.

Promote options only through section 7. Keep compatibility mode available even if a new recommended policy emerges. Do not replace settings based solely on the same problems/seeds used to tune them.

### N12 — Deliver and document

Update the crate README, ai-context.md, job-profile guide, PLAN.md ledger, decisions.md and benchmark reports. Add examples for local training, async stragglers, deterministic replay, cancellation and population handoff.

Provide installation/discovery examples that distinguish released packages from source-only components. Optional Python bindings are a separate packaging subtask, with a documented scope and wheel tests; they are not a reason to move the numerical implementation.

Prepare a patch ZIP and implementation log with source hashes, per-iteration decisions, raw records and reproduction commands. Do not tag or publish as part of implementation unless separately requested.

## 6. Benchmark protocol

### 6.1 Workload strata

| Stratum | Suggested cases | What it tests |
|---|---|---|
| Small smooth | Branin, Hartmann-3/6, shifted/scaled variants | Precision, early progress and regression |
| Moderate dimension | Ackley 10/20/32, rotated/shifted Rosenbrock | Local models and search geometry |
| Many-modal/separable | Styblinski–Tang and additional frozen cases | Known BO weakness and handoff |
| Numerical stress | Duplicate/constant data, deterministic ripple, extreme scaling | Stability and failure policy |
| Registered simulations | Existing RK4 oscillator; a bounded trajectory case; an engineering/CRN/CFD objective with fixed seed and verified determinism | Actual evaluation/proposal crossover |
| Scheduler fixtures | Equal, bimodal and heavy-tailed virtual service times | Async utilization and accounting |
| Archive scaling | Histories 256/1,000/10,000; training subsets 64/128/256 | Memory and bounded fitting |

Select actual simulator models by discovery, not by repository name alone. A stochastic model with one fixed seed is a deterministic sample-path objective; label it accordingly. Do not claim optimization of its expectation.

Use official BoTorch TuRBO/LogEI implementations, egobox where applicable, Optuna GP/TPE, random or low-discrepancy search and native population methods as external controls. Pin working versions at N0; an old benchmark pin is historical evidence, not an instruction to install the latest version blindly.

### 6.2 Fairness and statistical reporting

- Run equal-completed-call and equal-elapsed-time protocols separately.
- Count failed completions, initial designs, intentional repeats and final validation explicitly.
- Use shared initial observations when supported; include a separate native-default-start comparison so initialization advantages are visible.
- Keep bounds, transforms, objective direction, constraints and reference values identical.
- Report per-problem/seed trajectories, median/IQR, target success, censored time-to-target, proposal overhead, peak RSS and work counters.
- Use development seeds distinct from confirmation seeds: default 10 development and 30 held-out seeds for cheap functions; at least 5 real-simulation seeds for a pilot, increasing before a broad default claim.
- Bootstrap paired comparisons within problems/seeds and report the unit of analysis. Do not pool unlike raw objectives or call small pilots statistically decisive.
- For exact known optima, use normalized regret with a predeclared scale and floor. For simulations without known optima, use independently checked fixed targets and quality distributions.
- Retain failed/time-limited runs. Never compute a speedup only on the subset where the new method succeeded.

### 6.3 Bounded execution

Provide smoke, development and confirmation profiles. Default smoke runs use virtual-time or millisecond fixtures, two seeds and small budgets. Start development with at most two heavy processes and a declared memory ceiling; measure before increasing parallelism.

Use a per-run wall cap, an independent watchdog and a maximum shard duration of about 30 minutes. Choose simulation budgets so the declared per-run cap is meaningful; record censoring. Persist each completed run atomically. A failed cell receives a typed result and traceback/log, not endless retry.

Set a total campaign allowance before launch, initially at most four hours per development campaign. Resume by immutable experiment-cell identity and skip already completed cells. Check progress between bounded shards. Actual long 10–60 second simulations are a separately budgeted confirmation panel.

## 7. Acceptance and promotion rules

These are proposed decision thresholds to freeze at N0, not observed results.

| Gate | Requirement |
|---|---|
| Correctness | Numerical fixtures, budget/accounting, cancellation, log failures and identity tests pass |
| Compatibility | Legacy mode retains recorded behavior, excluding explicitly documented numerical bug fixes |
| Resources | No growing queue, archive leak or unreserved pool; memory/latency caps measured |
| Repeatability | Sync same-host thread-count tests; async event-order replay; platform limits explicit |
| Quality noninferiority | In each predeclared stratum, the upper 95% paired-bootstrap bound on the geometric-mean final normalized-regret ratio is <=1.05; target success must not show a material decline |
| Practical improvement | At least one declared target stratum shows >=15% median time-to-target gain or >=20% lower proposal overhead with maintained quality |
| Default promotion | All applicable gates pass on held-out data; no unresolved material stratum regression |
| Inconclusive evidence | Keep opt-in and report uncertainty; add evidence before promotion |

Define a material target-success decline as more than five percentage points for the initial decision rule, and report its uncertainty. Increase sample size when the interval cannot distinguish benefit from that loss. For simulation cases without reference optima, predeclare an equivalent target/quality noninferiority statistic instead of manufacturing regret.

The 5%, 15% and 20% values are engineering decision thresholds, not universal properties of BO. Record any revision before running confirmation data.

## 8. Validation commands and delivery checklist

Existing checks to retain, adapting paths only when source discovery requires it:

~~~bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --release --locked -p fcmaes-surrogate --examples
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
JOB_API_REQUIRE_SCHEMA=1 python3 tutorials/surrogate/run.py --root /absolute/new/run
python3 scripts/check_doc_consistency.py
python3 scripts/build_book.py
python3 scripts/check_book_links.py
~~~

For each iteration retain: base/final commit, exact options, build identity, tests, raw JSONL, environment, exclusions, budget/time accounting, plots/tables derived from raw data and a decision note. Proposed experiment manifests and CLI switches must be implemented and tested before appearing as runnable commands in user documentation.

Final implementation delivery contains per-repository patches, a manifest, an apply-check helper, checksums, the complete log and compact evidence. Large raw artifacts may be separately downloadable with hashes. Upstream pins name merged commits only; local experimental commits remain clearly marked.

## 9. Now versus January validation

**Executable now:** numerical fixtures, candidate/trust-region experiments, virtual-time async schedules, modest real callbacks, native worker integration, single-host crash tests, raw-data reporting and downloadable patches.

**January/home-lab validation:** sustained 16-physical-core configurations, wider batch/thread limits, repeated long simulations, power-loss/reboot behavior and cross-host operating evidence. Multi-node placement itself remains optimization-jobs/experiment-craft work. Cloud or simulated tests cannot close those hardware gates.

## 10. Relationship to BO-craft

Both projects can consume the same benchmark specifications and comparison reports. They should share observation terminology and provenance rules where practical, without forcing a dependency on one another.

The native surrogate is the low-overhead deterministic baseline. BO-craft is the advanced Python integration. A future cross-method campaign can compare or deliberately hand off between them through experiment-craft, with a new explicit campaign decision and common budget; this is separate from N10's in-task native handoff.

## 11. Sources and implementation entry points

Repository sources are pinned where possible. Literature/tutorials establish algorithm definitions, not performance promises for this project.

- [fcmaes-surrogate README at inspected main](https://github.com/dietmarwo/fcmaes-rust/blob/4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b/crates/fcmaes-surrogate/README.md)
- [Current strategy implementation](https://github.com/dietmarwo/fcmaes-rust/blob/4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b/crates/fcmaes-surrogate/src/strategy.rs)
- [Reviewed roadmap and status ledger](https://github.com/dietmarwo/fcmaes-rust/blob/4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b/docs/surrogate/PLAN.md)
- [Recorded implementation decisions and rejected variants](https://github.com/dietmarwo/fcmaes-rust/blob/4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b/docs/surrogate/decisions.md)
- [Existing cross-library comparison](https://github.com/dietmarwo/fcmaes-rust/blob/4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b/benchmarks/surrogate-comparison/README.md)
- [Existing worker profile and evidence semantics](https://github.com/dietmarwo/fcmaes-rust/blob/4ad8b7dc6fbe2b59c1abf2c2e403721ccf02be8b/docs/job-api/surrogate.md)
- [BoTorch TuRBO-1 reference tutorial](https://botorch.org/docs/v0.17.0/tutorials/turbo_1)
- [Wilson et al., Efficiently sampling functions from Gaussian process posteriors, ICML 2020](https://proceedings.mlr.press/v119/wilson20a.html)
- [BoTorch implementation repository](https://github.com/meta-pytorch/botorch)
- [Shared optimization-jobs contract](https://github.com/dietmarwo/optimization-jobs/blob/a0697ed5c0143d39f2a2ec954c32bea0a2fc4f34/docs/job-api.md)

External package APIs and default settings must be rechecked and pinned during N0. No new benchmark results or completed implementation are claimed by this document.
