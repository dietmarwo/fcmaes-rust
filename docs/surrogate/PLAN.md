# Surrogate optimization: reviewed plan and implementation status

Reviewed 2026-10-20. This plan supersedes the implementation recommendations in the supplied second plan. The unedited sources are [supplied-plan-v2.md](supplied-plan-v2.md), [original-plan.md](original-plan.md), and [application-areas.md](application-areas.md). The second plan's emphasis on **Rust simulations taking 1–60 seconds**, larger budgets, cancellation, and measured proposal overhead is retained. This document distinguishes the experimental implementation delivered now from the full product still to build.

## 1. Product decision

Build a source-only Rust crate, `fcmaes-surrogate`, and an in-worker profile, `fcmaes.surrogate/1`, in `fcmaes-rust`. Keep the numerical library independent of the shared job API. Keep task scheduling, resource reservations, common claims, and process containment in `optimization-jobs`. Do not add duplicate surrogate engines to LP-IP-craft or SQP-craft. Those solvers may later be registered components of application objectives or acquisition subproblems.

The intended production product serves box-bounded simulation optimization. It must win on **time to a useful answer including proposals, simulations, durability, startup, and validation**, rather than only on number of evaluations. The first version is an experimental baseline for establishing that crossover. It does not implement the full 15–19 week program in the supplied plan.

### Delivered milestone M0

- Deterministic scalar minimization, 1–32 continuous coordinates, optional log-scaled positive bounds.
- At most 256 total new plus retained observations; at most eight concurrent evaluations and eight inner threads per task.
- Exact ARD Matérn-5/2 or RBF GP, fixed unit signal variance, deterministic nugget, bounded L-BFGS MAP fit with analytic gradients, LogEI or minimization LCB.
- Centre/maximin Latin hypercube design, seeded candidate generation, local coordinate refinement, synchronous Kriging-believer batches.
- Ordered decisions independent of installed Rayon thread count; pending-point fantasies at the library proposal interface.
- Fallible completion logging, checksummed durable worker traces, atomic compact warm snapshots, semantic decision replay, independent best-point re-evaluation.
- `Scalar::simulation` with cancellation checkpoints, remaining time, point-derived seed, and per-evaluation thread allowance. **Only the surrogate profile accepts these callbacks in M0.**
- Tested CLI, shared runner, MCP, warm-start, and RK4 simulation tutorial.

The profile rejects unimplemented options. `auto`, `turbo`, asynchronous scheduling, observation noise, rank transforms, Sobol, Thompson sampling, constraints, and handoff are not aliases for simpler algorithms. Existing `fcmaes.retry/1` remains the available choice for larger-budget conventional scalar objectives. Simulation models requiring that profile need the M1 API extension below.

## 2. Corrections to the supplied design

| Topic | Revised requirement and reason |
| --- | --- |
| First release scope | Ship M0 as an experimental numerical/integration baseline; reserve “first usable simulation product” for M1–M4 gates. F9/F10/F19/F21/F25/F27 are substantial independent algorithms. |
| Failure values | Classify errors, panics, non-finite values and threshold penalties before transformations. Preserve failure records. M0 excludes failures from GP fitting and avoids already evaluated points; it does not fabricate precise bad observations. Compare imputation, feasibility models and exclusion radii before selecting a production policy. |
| Noisy simulations | M0 requires a deterministic function, potentially made repeatable with a point seed. Repeating one seeded realization does not establish its expected value or uncertainty. Noise inference, replications, confidence intervals and noisy incumbent selection belong to a separate validated milestone. |
| Transform tests | Standardization and bilog have usable inverses; a tied-rank copula generally does not. Test rank ordering, tie handling and predictive semantics, without claiming an invertible objective transform. Store raw outcomes. |
| MAP objective | Define whether priors are densities on lengthscales or on log parameters. M0 uses the former, without a change-of-variables Jacobian; dropping parameter-independent constants is explicit in fixtures. Do not compare absolute NLP numbers from different conventions. |
| Linear algebra | M0 uses its own packed row-major Cholesky/solves (`linalg.rs`, fixed summation order) after nalgebra's dominated proposal time in review. Revisit faer or blocked factorizations only with numerical and whole-round measurements at n=500–1000. Define covariance nugget, numerical jitter, latent variance and observation variance separately. |
| Search algorithms | M0 fits hyperparameters by bounded L-BFGS with analytic gradients, refines the acquisition by coordinate pattern search (a gradient refinement was measured and not adopted, see decisions.md) and uses maximin LHS. Full L-BFGS-B and Sobol remain candidate improvements. Avoid unverified direction-number tables. |
| Sobol acceptance | A two-dimensional dyadic square of side `2^-m` has expected occupancy `2^(k-2m)` in a balanced `2^k` net when the applicable net properties hold and `2m <= k`; it is not `2^(k-m)`. Test the actual chosen net parameters. |
| Batch guarantees | A greedy `1-1/e` statement requires a specified set function and the theorem's assumptions. It is not a blanket guarantee for adaptive GP refits, local acquisition optimization, or Kriging-believer batches. |
| Pathwise complexity | Count feature construction, dimension, covariance updates and triangular solves. Per-candidate RFF evaluation costs include `features * dimension`; an exact correction generally includes training-point kernel work. Report measured end-to-end complexity, not dimension-free `O(m+n)`. |
| Governor determinism | A governor using measured service times is timing-dependent. Replay must record cost-model version, service-time inputs, decisions, refit adoption, and queue invalidation. Seed-only determinism applies to synchronous mode with governor off. |
| Resource containment | Synchronous: `q * evaluation_threads <= inner_threads`. Future async: reserve evaluation and surrogate capacity together. A shared Rayon pool is an aggregate bound; `SimContext.threads()` is an allowance to obey, not an OS quota for each callback. |
| Cancellation | Check before work and inside cooperative simulations. No promise of instant preemption. Count completed callbacks, cancelled partial calls, and validation calls separately; include partial elapsed work. |
| Durability | The event sink returns `Result`. Propagate disk/write failures and stop work that has not started. Flush each completion before waiting for stragglers. A snapshot is a new-job warm input, not optimizer-state resume. |
| Job artifacts | M0 leaves trace/snapshot files in the local attempt directory. Shared artifact registration, remote retrieval and live paging require a versioned `optimization-jobs` extension; do not claim they already work through MCP. |
| Fair baselines | Use valid CMA-ES populations (at least four), common initial observations where applicable, exact callback accounting, partial-generation accounting, and equal resource/wall protocols. Count rechecks separately and report both optimization and total calls. |

## 3. Feature ledger against F1–F27

“Partial” means a narrower implementation is delivered, not that the full feature's gate passed.

| Feature | M0 status | Remaining work / destination |
| --- | --- | --- |
| F1 Space | Implemented | Mixed/discrete domains remain outside this profile. |
| F2 Transforms | Partial | Standardize, bilog, auto implemented; rank/ties and transformed noisy semantics deferred. |
| F3 GP | Partial | Exact kernels, ARD and deterministic nugget implemented; inferred noise deferred. |
| F4 Priors/fitting | Partial | MAP lognormal lengthscale prior, deterministic restarts, bounded L-BFGS with analytic gradients (checked against finite differences); MLE option and signal-variance/noise hyperparameters not implemented. |
| F5 Factorization | Implemented with explicit constants | Nugget 1e-8, initial jitter 1e-10, eight attempts with tenfold jitter; O(n²) fantasy append, fresh fit/rebuild per round. Verify larger caps before raising limits. |
| F6 Acquisition | Partial | LogEI and LCB implemented; candidate/pathwise Thompson sampling in M2. |
| F7 Acquisition search | Partial | LHS/global and sparse local candidates, four local starts; Sobol/Boltzmann/L-BFGS-B/CMA option deferred. |
| F8 Batches | Partial | Synchronous Kriging believer only; no constant-liar or TS batch option. |
| F9 Async | Deferred, P1 | M3, with a reproducible event-driven simulator before real workers. |
| F10 TuRBO | Deferred, P1 | M2: trust-region state, restart rules, local-data selection, TS, then multiple regions. |
| F11 Designs | Partial | Seeded centre/maximin LHS; no Sobol; configurable initial_points default eight. |
| F12 Failures | Partial | Error/panic/threshold classification, durable records, no GP imputation or three-failure neighbourhood rule. |
| F13 Stopping | Partial | New-evaluation budget, target, wall/cancel checks implemented; no TuRBO restart budget. |
| F14 Logs/warm | Partial | Durable traces and hashed compact snapshots implemented; warm-from-trace and remote/paged artifacts not implemented. |
| F15 Determinism | Partial | Same-machine 1/2/3/8-thread tests and full 1/8-thread benchmark equality; async replay and cross-machine/CPU evidence deferred. |
| F16 Incumbent | Partial | Best observed usable deterministic value, exact independent recheck; no noisy posterior-mean recommendation. |
| F17 Constraints | Deferred, P2 | Feasibility GP, PoF and SCBO after scalar campaign. |
| F18 qLogNEI | Deferred, P2 | Noise/replication model and fixed QMC samples first. |
| F19 Handoff | Deferred, P1 | M2: CMA-ES/BiteOpt with shared accounting and measured switch policy. |
| F20 RBF/DYCORS | Deferred, P2 | M5 large-budget cheap-surrogate comparison. |
| F21 Pathwise TS | Deferred, P1 | M2: exact posterior moments and path-correlation checks, then measured candidate throughput. |
| F22 Stateless propose profile | Deferred, P3 | No session service in this project; library proposal calls already exist. |
| F23 lq-CMA-ES | Deferred, P2 | M5; ranking acceptance, rejected-model behavior and equal-wall comparisons. |
| F24 Breadth | Deferred, P3 | Multiobjective, mixed integer, known constraints, sparse GP. |
| F25 Background refit | Deferred, P1 | M3: immutable model snapshots; logged adoption boundaries. |
| F26 Simulation API | Partial | Surrogate-only context implemented. M1 extends retry/evaluate and defines errors for each profile. |
| F27 Queue/governor | Deferred, P1 | M3, with measured latency and event replay; not a fixed sampling shortcut. |

## 4. Architecture and contracts

### Numerical core

`space`, `design`, `transform`, `special`, `gp`, `strategy` and `driver` are separate modules in `crates/fcmaes-surrogate`. `propose` receives explicit history, pending points, configuration, seed and strategy state, returning points plus new state. An aborted proposal does not mutate its input state. Independent streams use semantic indices; parallel work collects in index order. GP solve/reduction order stays serial within each model. Floating-point JSON round trips preserve replay inputs.

GP models use normalized coordinates and transformed usable outcomes. They expose latent posterior variance. A materially negative variance is a numerical error; rounding-sized negatives are clamped. Duplicate/constant data, nearly singular matrices, extreme tails and failed optimization must not silently return NaNs. Numerical fit failures fall back to a logged design proposal; invalid input and stopped work do not become numerical fallbacks. Exhausted distinct candidates are an explicit error in M0.

### Driver and storage

Each round logs the proposal and allocated IDs before execution. Each callback logs a Started event and its completion immediately under a fallible serialized sink. Batch decisions commit completed observations in proposal order. Timing and completion arrival order do not influence synchronous proposals. Final counts and a Finished event close the trace. Replay regenerates proposals and checks state, IDs, event consistency and counters; it does not run the objective again or turn its assertions into mathematical certificates.

The worker wraps these events with shared SDK checksum framing and identity metadata, fsyncs completions, and atomically updates a compact snapshot. A torn last record can be discarded by the shared checked reader; corruption in complete records is rejected. Incomplete traces cannot pass semantic replay. A process crash can leave durable observations even when common task accounting remains incomplete: recovery must respect the shared runner's accounting rules.

Warm snapshots bind model ID, version, dimension, resolved model parameters, model-input hashes, and the simulation run seed where relevant. New bounds can discard out-of-box observations; they cannot silently clip them. Warm jobs use one run, allocate new IDs, and count only new calls against the new budget. They refit from observations rather than restoring an interrupted GP/optimizer state. Deliberately reusing a model version after changing code invalidates these assumptions; application authors must bump versions.

### Worker, resources and claims

One existing shared task is one seeded optimization run. `outer_workers` controls simultaneous runs; `inner_threads` bounds each task's pool. In M0 the pool is used for GP work between evaluation batches and callbacks during batches. `concurrent * evaluation_threads <= inner_threads <= 8`; the shared envelope also enforces `outer_workers * inner_threads <= cpu_slots`. Simulations must not create unbudgeted native/BLAS pools. Ordered nested Rayon work shares the same aggregate task pool; it does not assign disjoint subpools to each evaluation.

A callback is either `Scalar::new(Fn(&[f64])->f64)` or `Scalar::simulation(Fn(&[f64], &SimContext)->Result<f64,String>)`. `checkpoint()` returns `Err("cancelled")` when stopped. The seed derives from run seed and physical point, with signed zero canonicalized; it makes repeatable randomized objectives possible, but provides no noisy-BO uncertainty treatment. Remaining time is advisory and decreases to zero. Process deadlines remain the final containment mechanism.

Only an actually evaluated usable point may carry `best_found`. Exact independent re-evaluation upgrades objective/bounds evidence to `rechecked`; it never certifies global optimality. Disabled, skipped or cancelled rechecks leave historical evidence `claimed`. A completed disagreeing recheck refutes `best_found`, reports the fresh feasible value if usable, and marks the snapshot untrusted. `validation_evaluations` is separate from optimization callbacks. A cancelled validation attempt is not an objective mismatch.

The current compact payload is documented in [the API guide](../job-api/surrogate.md). The richer nested payload in the supplied plan is a future design. Freeze schema compatibility deliberately before a public release; do not replace `/1` semantics silently when adding noise, asynchronous recommendations or changed accounting.

## 5. Execution plan and gates

### M0 — experimental baseline (delivered)

Plan: establish small exact-GP numerics, deterministic synchronous decisions, fail-closed durability, and a real job-profile path before introducing async timing. Implement the ledger subset above. Test independent high-precision fixtures, fantasy updates, thread equality, logger errors, identity checks, cancellation and strict options. Benchmark four methods at the same objective budget.

Acceptance evidence is in [validation/REPORT.md](validation/REPORT.md) and [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md). M0 does not pass the future n=500/1000 latency targets or simulation crossover gates simply because small tests pass.

### M1 — shared simulation semantics and baseline campaign

1. Extend `Scalar::simulation` handling to retry and evaluate, threading context through each algorithm's evaluation path, warm starts and final validation. Define whether generation cancellation yields partial results; ensure it cannot be counted as a penalty or a completed failure. Keep incompatible vector-model contexts out until separately designed.
2. Add deterministic integration fixtures for simulation factory failures, malformed parameters, nested parallel evaluations, deadlines during proposal/evaluation/recheck, and cancellation under active retries. Preserve `Scalar::new` behavior. Document the source API effect of private simulation storage on direct struct literals.
3. Register representative existing Rust tutorial simulations with fixed model versions and deterministic seeds. Include at least one trajectory model, one engineering/dynamics model and one real error-producing simulator.
4. Separate development workloads/seeds from a frozen held-out confirmation set. Measure both cheap kernels and real 1–60 s evaluations. Record hardware, cgroup quota, affinity, compiler, resource layout, startup/fsync/recheck costs and all failures.

Gate: runner/MCP cancellation returns consistent counts and claims for both profiles; zero unbudgeted pool growth; no regression on existing retry/evaluate tests; publish equal-call and equal-wall baseline results. No performance claim before measurements.

### M2 — scalable local BO and handoff

1. Implement pathwise posterior sampling using an RFF prior plus exact conditioning. Check empirical means, variances and correlations against a small exact GP, fixed-seed paths and kernel convergence across feature counts. Benchmark total proposal cost including feature construction/refits.
2. Implement TuRBO-1 with explicit counters, length bounds, restart behavior and deterministic local-data selection. Add TuRBO-m only after one-region behavior is correct. Check published constants against the primary implementation and document deliberate deviations.
3. Add in-task CMA-ES and BiteOpt handoff from a feasible incumbent, preserving original coordinates, shared budget, failure policy, callback context and trace. Use valid population sizes, deterministic seeds and real calls for partial final generations.
4. Compare local cap/feature count/handoff fraction on development sets, then freeze settings for held-out tests. Do not make `auto` choose an unavailable algorithm.

Gate: no global n³ growth beyond a documented local cap; posterior-sampling fixtures pass; trust-region transition tests pass; bounded cancellation latency; handoff never duplicates or fabricates evaluations. Compare time to target against retry at budgets of hundreds and thousands, dimensions 5/10/20/50 and beyond where supported. Publish losses as well as gains.

### M3 — asynchronous execution, background fits and overhead governor

1. Build a deterministic event-driven simulator with controllable evaluation-duration distributions, failures and cancellation. Assign monotonically ordered event IDs; distinguish candidate creation, queue admission, evaluation start/completion and model adoption.
2. Split resource capacity explicitly: evaluation slots times evaluation threads plus surrogate capacity must fit inside reserved inner threads. Bound queued and in-flight work and snapshot memory. Eliminate oversubscription from background fitting.
3. Use immutable GP snapshots. A completed refit is adopted at an explicit logged boundary; proposals record model version and pending/queued set. Deduplicate and invalidate stale queued points under documented rules.
4. Add an overhead governor using measured service times and a versioned cost model. Record input statistics, thresholds and decisions. Reduce candidates/restarts or cap/refit frequency only through tested policies, then hand off when warranted. Preserve a governor-off synchronous reference.
5. Replay an asynchronous run from its recorded event order, cost inputs and adoption decisions. Do not promise identical seed-only async trajectories under different timing schedules.

Gate: variable-duration simulation shows measured utilization gains over synchronous batches without oversubscription; logs replay exactly; cancellation drains or marks in-flight work correctly; proposal/refit latency fits a declared budget such as 10% of median evaluation service time divided by evaluation slots on the measured operating range. Report queue latency, utilization, p50/p95 proposals and quality, not only aggregate throughput.

### M4 — job artifacts and product readiness

1. Propose a small common `optimization-jobs` artifact extension with immutable identity, hashes, relative-path validation, byte/page limits and retrieval after completion. Define live-tail consistency separately. Implement SDK/runner/MCP tests in that repository before claiming the feature here.
2. Keep existing attempt-local files as the fallback. Add paged access without loading an entire long-running trace into an agent response. Test corruption, path traversal, symlinks, truncated writes, unavailable files and version mismatch.
3. Add schema/version migration tests, a mixed-node multi-worker harness and cross-machine determinism measurements on documented CPUs. Test real systemd or equivalent process containment on a host that supports it.
4. Conduct an independent review before merging a public usable release; close findings, freeze compatibility and publish full campaign evidence.

Gate: tutorial, shared runner/MCP, artifact retrieval and accounting work on real deployment nodes; all numerical and resource gates pass; supported operating region and chooser guidance follow measured evidence. No independent review or service-containment result is claimed for this delivery.

### M5 — extensions after crossover evidence

Prioritize RBF/DYCORS and surrogate-assisted CMA-ES for larger budgets (F20/F23). Implement noisy objectives/replications before qLogNEI and noisy recommendations (F18/F16). Add feasibility GP/SCBO constraints (F17). Only then consider multiobjective, mixed domains, sparse GP and external stateless proposals. Every extension gets its own hypothesis, implementation, numerical gate, development comparison and held-out confirmation.

## 6. Benchmark protocol

For each campaign freeze source commits, lockfile, model versions, seed lists, domains, thresholds and transformations. Use random search, fcmaes CMA-ES and BiteOpt, with DE/other specialists where relevant. Share initialization when scientifically appropriate and report its cost; avoid granting one method free model evaluations. Show both equal-objective-call results and equal-wall results under the same eight-thread reservation. Warm-started campaigns must disclose reused calls and cost.

Required grids progress from d=2/3 smoke to d=5/10/20/50, budgets 48/128/256 and, after scaling support, 1000/10000. Compare q=1/2/4/8 under a fixed resource envelope. Evaluate pure library decisions separately from runner startup, fsync and recheck. Measure real simulations in addition to artificial delays; sleeps exercise scheduling but are not CPU-heavy scientific workloads. Use more seeds and confidence intervals for product conclusions; five seeds on two development functions are only a smoke test.

Gate retained changes on correctness first, then declared quality/latency metrics. Performance need not improve monotonically on every problem. Keep an experiment ledger with rejected patches and measured regressions; avoid relabeling a worse result as progress. The automatic chooser is justified only by a measured crossover map and safe fallback.

## 7. Sources and unresolved decisions

Primary references for the implemented numerical ideas:

- Rasmussen & Williams, GPML chapter 2: https://gaussianprocess.org/gpml/chapters/RW2.pdf
- Ament et al., LogEI: https://arxiv.org/abs/2310.20708
- BoTorch analytic acquisition source (numerical comparison, not a runtime dependency): https://botorch.readthedocs.io/en/stable/_modules/botorch/acquisition/analytic.html
- Wilson et al., acquisition maximization: https://arxiv.org/abs/1805.10196

The supplied second plan retains the wider bibliography and ecosystem survey. Crate availability/performance assertions about faer, L-BFGS-B, Sobol and other future dependencies remain phase-specific verification tasks. Do not turn an unverified survey into a claim that no Rust implementation exists.

Open decisions: optimizer for analytic MAP gradients; local GP data policy and cap; posterior path feature count; measured crossover/handoff defaults; failure/feasibility policy; simulation seed semantics across retries; artifact compatibility; and production limits above eight threads. Resolve them with small documented experiments at their milestone rather than speculative API commitments now.
