# Gap 6: surrogate and batch Bayesian optimization, plan

*20 October 2026. Plan for closing gap 6 of `claude/job-api-application-areas.md`:
"surrogate and batch Bayesian optimization for evaluations that cost minutes or
hours".*

*It rests on four research notes written today (Rust ecosystem, Python
ecosystem, literature, benchmarks) and on the current code of fcmaes-rust
(`92aaf03`), optimization-jobs (`eb1c9de`), SQP-craft and LP-IP-craft. An
independent review of the first draft found three critical, eight important
and several minor problems. This version fixes them.*

*Facts dated after mid-2026 come from those notes and are re-checked in phase
0 before anything depends on them: egobox 1.0.0, basin 2.0.0, Optuna 5.0,
Rustuna, faer 0.24 and rand 0.10.*

## 1. Summary

**What to build: a Rust library, `fcmaes-surrogate`.**
- An exact Gaussian-process surrogate: Matérn-5/2 or RBF, ARD, the
  dimension-scaled lengthscale prior, a MAP fit.
- Robust output transforms, so penalty values do not break the model.
- LogEI, UCB and Thompson-sampling acquisitions.
- An acquisition optimizer: quasi-random starts, sparse local perturbations,
  multi-start L-BFGS-B, with fcmaes CMA-ES as an alternative.
- Batch selection with fantasies (Kriging believer, constant liar) and
  Thompson sampling.
- Asynchronous scheduling of pending evaluations.
- TuRBO-1 and TuRBO-m trust regions.
- Later: a hand-off to CMA-ES/BiteOpt, RBF/DYCORS, constraints and qLogNEI.

**How it reaches the job API: a profile `fcmaes.surrogate/1` of the fcmaes
job worker.**
- The profile runs the whole loop inside the worker, on registered scalar
  models, so every application worker gets Bayesian optimization for its
  models.
- The "no ask/tell profile" decision stands.
- Every completed evaluation goes to a durable log.
- A later job continues from that log as a hashed input (warm start), never
  by resuming.
- Two small changes to fcmaes-job-worker are needed (section 6.5): inputs
  that a profile reads itself, and more than one profile per model kind.
- Three small changes to optimization-jobs are needed: artifacts declared by
  the worker, artifacts readable while a job runs, and paged artifact reads
  in MCP.

**Parallelism, on three levels:**
- independent runs (`outer_workers`);
- `concurrent` evaluations per run, synchronous or asynchronous, inside the
  task's own thread pools;
- Rayon inside the surrogate: hyperparameter restarts, candidate scoring,
  acquisition restarts and trust regions.

**Determinism:**
- A proposal is a pure function of (history, pending points, carried state,
  seed).
- On the same binary and CPU model, results do not depend on the thread
  count.
- An asynchronous run is reproducible by replaying its log.

**What the evidence says:**
- Bayesian optimization beats CMA-ES mainly at small budgets (≲ 10–20·d
  evaluations) and moderate dimension.
- From d ≈ 40, or with larger budgets, CMA-ES catches up, and trust regions
  are the most robust Bayesian variant (Santoni et al. 2024).
- The benchmark plan measures this crossover against fcmaes, including the
  GTOP problems. No published Bayesian-optimization results on GTOP were
  found.

**Effort:** about 14–18 weeks for one developer in six phases (section 10).
Phases 0–3 (9–13 weeks) give a usable, tested profile.

## 2. Goal, regime and non-goals

| Property | Range | Consequence |
| --- | --- | --- |
| Cost per evaluation | Seconds to hours (simulations, CAD/CFD, lab-like external runs) | Proposal overhead of seconds is negligible; per-evaluation durability matters |
| Budget | 10·d + 50 up to about 2,000 evaluations per run; up to 10,000 with TuRBO | Exact GP with Cholesky; local GPs for long runs |
| Dimension | 1–50 for global GP-BO (limit 100), up to 1,000 with TuRBO | Dimension-scaled priors; no embeddings in v1 |
| Parallel evaluations | 1–15 per run in a job on one 16-core node; more in the library's simulator | Batch and asynchronous selection; independent runs across nodes |
| Noise | Deterministic to moderately noisy | Inferred noise with a floor; a posterior-mean recommendation for noisy models |
| Failures | Crashes, non-finite values, penalty values (fcmaes-gtop returns 1e10, fcmaes 1e99) | Failures are data, not job failures (F12) |

All formulas assume **minimization**: z = (best − μ)/σ, Thompson sampling
takes the argmin of a sample, and a TuRBO success is a decrease.

**Not in the first release:**
- categorical and conditional HPO spaces (Optuna, SMAC3 and Ax serve them);
- multi-fidelity;
- high-dimensional embeddings (BAxUS, SAASBO);
- GPUs;
- one run spread across several nodes;
- an ask/tell session in the job API (section 6.6).

## 3. Research summary

### 3.1 Rust ecosystem

| Crate | State (notes, 2026-10-20) | Use here |
| --- | --- | --- |
| egobox 1.0.0 (egobox-ego, -gp, -moe, -doe) | The only mature Rust EGO stack (Apache-2.0, one maintainer, breaking changes almost every minor release). Has LogEI (default), WB2, Kriging-believer and constant-liar q-points, TREGO, constraints, hidden constraints, mixed-integer relaxation, sparse GP, KPLS, incremental GP update and experimental multi-objective. Built on ndarray, linfa-linalg, rand_xoshiro 0.6, COBYLA | Reference implementation and benchmark baseline, not a dependency |
| friedrich 0.6, gprx 0.2 | Small GP crates (nalgebra; faer); gprx is a day old | Reference |
| faer 0.24 | Pure-Rust dense LLT with rank-r update and downdate, triangular solves, per-call `Par::Seq` or `Par::Rayon(n)`; selects SIMD kernels at run time; results documented as non-portable across platforms | Linear algebra (decided in phase 0) |
| nalgebra 0.33–0.35 | Cholesky with rank-one update and column insertion; single-threaded; already in fcmaes-core | Fallback |
| basin 2.0 | Pure-Rust L-BFGS-B port, COBYLA, BOBYQA, SLSQP and more; seeded, result-identical parallel mode; released the day before the notes; uses rand 0.10 | Candidate for L-BFGS-B |
| lbfgsb-rs-pure 0.1 | Pure-Rust port of L-BFGS-B 3.0, BSD-3 | Candidate for L-BFGS-B |
| argmin 0.11 | No L-BFGS-B | Not used |
| sobol 1.0, sobol_burley 0.5 | Joe–Kuo directions (unscrambled) / Owen-scrambled but f32, ≤ 256 dimensions, ≤ 2¹⁶ points | Own f64 scrambled Sobol on Joe–Kuo directions |
| libm 0.2 (already in fcmaes-core), puruspe 0.4 | Portable libm; puruspe has erfcx with about 1e-14 relative error | Special functions on libm; puruspe optional |
| num-dual 0.15 | Forward-mode automatic differentiation | Gradient checks in tests only |
| Rustuna (PFN) | Optuna in Rust, with Python/JS bindings only; TPE, NSGA-II, QMC; no GP | Not relevant |

Gaps the new library fills:
- No Rust crate has Monte Carlo q-acquisitions with reparameterization for
  one objective.
- None has first-class asynchronous pending-point handling.
- None states that its results do not depend on the thread count.

### 3.2 Python ecosystem

- **BoTorch 0.18 with Ax 1.3 is the reference stack.**
  - The GP is RBF-ARD without outputscale, with lengthscales ~ LogNormal(√2
    + ½·ln d, √3) as a density on ℓ (mode ≈ 0.205·√d), at least 0.025.
  - Noise variance ~ LogNormal(−4, 1) with floor 1e-4; outputs standardized;
    MAP fit.
  - The acquisition is qLogNEI, maximized by multi-start L-BFGS-B from QMC
    raw samples.
  - Ax starts with the box centre plus 4 Sobol points.
  - Batches are built by sequential greedy selection over Monte Carlo
    acquisitions with pending points.
- **Optuna 5.0** (2026-09-07) made GPSampler stable (Matérn-5/2 ARD). It now
  integrates over running trials with qLogEI. TPE with constant liar is the
  default sampler.
- **Three ways to handle pending points:**
  - joint Monte Carlo integration with greedy selection: BoTorch/Ax, and
    Optuna's GP sampler since 5.0;
  - imputation by constant liar or Kriging believer: egobox, SMT, skopt,
    Optuna TPE, bayes_opt;
  - local penalization: GPyOpt, Trieste, Emukit.
- **Inactive projects:** GPyOpt is archived. pySOT, Dragonfly and skopt are
  dormant. HEBO, Nevergrad and Vizier have had no release for at least 18
  months.
- **Comparisons:**
  - Ax paper (AutoML 2025, 47 problems): Ax leads on mixed, multi-objective,
    noisy, constrained and high-dimensional problems. HEBO (81.7) and SMAC-BB
    (79.1) lead on BBOB 20-D with 50 trials.
  - carps (2025): many optimizers are statistically indistinguishable.
- **The opening:** Ax says it does not handle the thousands of trials of
  TuRBO-scale runs efficiently. A fast Rust engine can.

### 3.3 Literature

| Topic | Finding | Source |
| --- | --- | --- |
| GP defaults | Exact GP, ARD Matérn-5/2 or RBF, unit-cube inputs, standardized outputs, MAP with multi-start L-BFGS | Snoek et al. 2012; Rasmussen & Williams 2006 |
| High dimension | Lengthscales scaled with √d (prior or initialization) make vanilla BO competitive; constant short lengthscales cause vanishing gradients | Hvarfner et al. 2024; Xu et al. 2025; Papenmeier et al. 2025 |
| Acquisition | EI underflows; LogEI and qLogNEI fix it (three-branch log h(z)) | Ament et al. 2023; BoTorch code |
| Batch | Monte Carlo q-acquisitions are submodular, so greedy batches reach 1 − 1/e of the optimum; Kriging believer and constant liar are cheap fantasies; local penalization is the geometric alternative | Wilson et al. 2018; Ginsbourger et al. 2010; González et al. 2016 |
| Asynchrony | Asynchronous Thompson sampling is about as good per evaluation and better under a time budget | Kandasamy et al. 2018; Alvi et al. 2019 |
| Trust regions | TuRBO: L₀ 0.8, L_min 2⁻⁷, L_max 1.6, τ_succ 3, τ_fail ⌈max(4/q, d/q)⌉; lengthscale-shaped box; Thompson sampling on min(5000, max(2000, 200·d)) candidates with perturbation probability min(20/d, 1) | Eriksson et al. 2019; BoTorch tutorial |
| Candidates | Sparse local perturbations (DYCORS/RAASP) matter as much as the model in high dimension | Regis & Shoemaker 2013; Papenmeier et al. 2025 |
| RBF | SRBF/DYCORS: cubic RBF, 100·d candidates, weights [0.3, 0.5, 0.8, 0.95], radius adaptation | Regis & Shoemaker 2007/2013; pySOT |
| Surrogate-assisted CMA-ES | lq-CMA-ES: a global quadratic model, accepted when Kendall τ ≥ 0.85 | Hansen 2019 |
| BO vs. ES | BO wins at small budgets and moderate d; CMA-ES catches up from 10–20·d evaluations and wins from d ≈ 40–60; trust regions are the most promising BO | Santoni et al. 2024 |
| Numerics | Cholesky O(n³), append O(n²) with fixed hyperparameters; exact GPs are fine to n ≈ 2,000–5,000 | Rasmussen & Williams 2006 |
| Acquisition optimization | QMC raw samples, Boltzmann-chosen restarts, batched L-BFGS-B; flat acquisition regions need local candidates | BoTorch; Papenmeier et al. 2025 |

### 3.4 Benchmarks and methodology

**Tier A, closed form:**
- Branin; Hartmann-3/6; Ackley, Levy, Rastrigin, Griewank, Styblinski-Tang
  and Rosenbrock at 10 and 30 dimensions; Michalewicz-10.
- Branin and Hartmann-6 embedded in 100 and 500 dimensions; Ackley-200D.
- BBOB f1–f24 at 10, 20 and 40 dimensions (Santoni et al.).
- Later: the constrained and multi-objective sets.
- BoTorch's `synthetic.py` gives the domains and optima.

**Tier B, small simulations in Rust:**
- Rover 60-D.
- The GTOP problems in `fcmaes-gtop`: Cassini1, Cassini2, Messenger and
  Messenger full, Rosetta, SAGAS, GTOC1 and TandEM.

**Tier C, external:** MOPTA08, SVM-388, LassoBench, YAHPO and Bayesmark.

**Methodology:**
- 20–30 seeds.
- Median and IQR of log regret against evaluations, fixed-target ECDFs,
  Moré–Wild data profiles.
- Wilcoxon tests with Holm correction.
- Fit and acquisition time reported separately.
- Asynchronous runs: simulated durations (half-normal with mean 1,
  exponential, lognormal); regret against simulated time, time-to-target
  speedup and idle time.
- Fair ES baselines: population equal to the batch size, and the same
  initial design (TuRBO's protocol).

## 4. Constraints from the existing projects

**Job API (optimization-jobs `eb1c9de`):**
- `Worker::prepare` returns a `Prepared` with `tasks()`,
  `solve(task, remaining_seconds)` and `cancel()`.
- The SDK runs the tasks inside `thread::scope` (`worker.rs`), so every
  thread a task starts must end before `solve` returns.
- A task is the journal's commit unit. An `Err` from a task fails the job;
  mathematical failures are refutations.
- `"recovery": "journal_only_no_resume"`.
- `Outcome::check` enforces claims, evidence and objective bounds.
- Resources: `outer_workers × inner_threads ≤ cpu_slots`.
- Inputs: hashed inputs only, at most 16 per job and 64 MiB each.
- Artifacts: the manifest lists only `results.jsonl` (`result.rs`) and
  exists only after the job ends. MCP `get_artifact` returns at most 1 MiB.

**fcmaes job worker:**
- `Registry` models: `Scalar::new(f)` where `f: Fn(&[f64]) -> f64`, so a
  model cannot report an error and signals failure by a penalty value.
- `Registry::resolve` rejects any job input the model factory did not read.
  `Kind::profile` maps scalar models to `fcmaes.retry/1` only.
- `threads.rs`: each outer worker owns one Rayon pool, and a task runs
  entirely inside it, so a model's own Rayon calls stay inside the
  reservation.
- `judge` refutes a best point whose re-evaluation differs at all
  (`value != r.y`). Values ≥ 1e99 or non-finite count as unusable.

**Decision in `claude/job-api-gaps-1-3.md`: no ask/tell profile.** The
reasons:
- the objective would leave the worker;
- a job would become a session;
- every generation would cost a round trip.

Section 6.6 explains why a stateless proposal profile would not reopen it.
It is offered as an option only.

**fcmaes-core:**
- rand 0.8, rand_pcg 0.3, rayon, nalgebra 0.33 and libm.
- `Cmaes` and `BiteOpt` have ask/tell; `retry_run_seed` gives task seeds.

**fcmaes-gtop:**
- `publish = false`, so it can only be a dev-dependency of the library.
- Only `gtop.cassini1` is registered in the worker today, and the other
  problems' bounds are not in the crate.

**SQP-craft:** `NlpProblem` solves smooth problems with constraints. It is
relevant later, for acquisition maximization under known constraints.

## 5. Required features

Priorities: **P1** in the first usable profile (phases 1–3), **P2** in the
first extension round (phase 5), **P3** later.

| # | Feature | Priority | Why |
| --- | --- | --- | --- |
| F1 | Box-bounded continuous space; unit-cube transform; per-coordinate `log_scale` (payload field) | P1 | GP defaults assume unit-cube inputs |
| F2 | Output transforms: `standardize` (default for deterministic, well-scaled models), `bilog` (sign·log(1+\|y\|), then standardize), `rank` (Gaussian copula of ranks, as SCBO uses); `auto` picks bilog when observed y spans more than 6 decades | P1 | Heavy tails and penalties otherwise break a GP with fixed signal variance |
| F3 | Exact GP: Matérn-5/2 and RBF, ARD, signal variance fixed at 1, inferred noise with floor 1e-4, or `noise: "none"` with a 1e-8 nugget for deterministic models | P1 | Literature default |
| F4 | Priors as densities on the parameters (BoTorch convention, no Jacobian term): ℓᵢ ~ LogNormal(√2 + ½·ln d, √3), ℓᵢ ≥ 0.025; noise variance ~ LogNormal(−4, 1). MAP by multi-start L-BFGS-B in u = log θ with analytic gradients. MLE with ℓ₀ = 0.1·√d as an option | P1 | Hvarfner 2024; BoTorch since 0.12 |
| F5 | Cholesky with jitter (1e-8 × mean diagonal, ×10 per retry, at most 6); O(n²) append within one proposal (fantasies); a fresh factorization at the start of every proposal | P1 | Numerics; cheap fantasies; replayable rounds |
| F6 | LogEI, UCB, Thompson sampling on candidates | P1 | Default plus a scalable batch choice |
| F7 | Acquisition optimizer: Sobol raw samples, RAASP local candidates, Boltzmann-chosen starts, parallel L-BFGS-B; fcmaes CMA-ES as an option | P1 | BoTorch recipe; flat regions |
| F8 | Batch: sequential greedy with Kriging believer (default), constant liar (min/mean/max), Thompson sampling without replacement | P1 | Standard, cheap, handles pending points |
| F9 | Asynchronous mode: a free slot gets one proposal given the pending points; hyperparameters refit in the background (F25) | P1 | Wall-clock efficiency |
| F10 | TuRBO-1 and TuRBO-m with candidate Thompson sampling and the published constants | P1 | Most robust BO beyond small budgets |
| F11 | Initial designs: scrambled Sobol (f64, Joe–Kuo), maximin LHS, centre point; default `min(max(10, 2·d), evaluations/5)` points | P1 | Reproducible, space-filling |
| F12 | Failures: a model panic (caught), a non-finite value, or a value ≥ `failure_threshold` (default 1e99, the fcmaes penalty; 1e10 for fcmaes-gtop models) is a failure. It is kept in the history and imputed for the GP as worst usable y + 0.1·range(usable y) after the output transform. A point within 1e-3 (unit cube) of three failures is excluded from proposals | P1 | Real simulators fail; GTOP returns 1e10 |
| F13 | Stopping: evaluation budget, wall time, target value, TuRBO restarts, cooperative cancellation between evaluations | P1 | Job API deadlines |
| F14 | Evaluation log (checksummed JSONL) and warm start from a hashed log, checked against the problem | P1 | Durability without resuming |
| F15 | Determinism: per-item random streams, ordered reductions, `Par::Seq` linear algebra, libm special functions, replayable asynchronous order | P1 | fcmaes contract; tests |
| F16 | Recommendation: best observed value for deterministic models; best posterior mean among evaluated points for noisy ones | P1 | Noisy incumbents |
| F17 | Constraints: expected improvement × probability of feasibility (one GP per unknown constraint); SCBO in TuRBO | P2 | Engineering problems |
| F18 | Monte Carlo qLogNEI with fixed QMC base samples, incremental over pending points | P2 | Noisy objectives; principled batches |
| F19 | Hand-off: after a budget fraction or a TuRBO collapse, continue with fcmaes CMA-ES or BiteOpt from the incumbent, with σ from the trust-region length or posterior spread | P2 | Santoni: ES wins at larger budgets; natural for fcmaes |
| F20 | RBF engine: cubic RBF with linear tail, SRBF/DYCORS | P2 | Cheap baseline for large budgets |
| F21 | Pathwise Thompson sampling (RFF prior plus exact update) | P2 | Large asynchronous batches; replaces the candidate covariance |
| F22 | Stateless proposal profile `fcmaes.surrogate-propose/1` | P2, decision Q2 | External evaluators |
| F23 | Surrogate-assisted CMA-ES in fcmaes-core (lq-CMA-ES model, Kendall-τ acceptance) | P3 | Seconds-scale objectives, d ≥ 40 |
| F24 | Multi-objective (ParEGO, then qLogNEHVI); mixed integer by relaxation; local penalization; known cheap constraints via SQP-craft; sparse GPs | P3 | Breadth |
| F25 | Background refit in asynchronous mode: proposals use the last fitted hyperparameters; a refit runs on the surrogate pool and is adopted at the next proposal after it finishes. The adoption is logged, so replay is exact | P1 | The surrogate must not become the bottleneck |

## 6. Design

### 6.1 Placement

**Recommendation:** a library crate `crates/fcmaes-surrogate` in fcmaes-rust,
plus a profile module `surrogate.rs` in `fcmaes-job-worker`.

- The profile works on the same `Registry`, so application workers get it
  without new code.
- It reuses fcmaes-core: CMA-ES/BiteOpt for acquisition optimization, the
  hand-off and the baselines, plus `Rng`, `retry_run_seed` and libm.
- The library does not depend on the job API, so it could move to its own
  repository later, as the job API did.
- `fcmaes-gtop` is only a dev-dependency of the library, for the benchmarks.
- Python bindings can come later through `fcmaes-py`.

The alternative is a separate repository in the style of SQP-craft. That
would duplicate the registry and the application models (Q1).

### 6.2 Dependencies

| Need | Choice | Phase-0 check |
| --- | --- | --- |
| Dense linear algebra | faer 0.24 with `Par::Seq`, behind an internal `linalg` module; nalgebra 0.33 as fallback | Cholesky, append and solve at n = 100–3,000; bit-identical results across runs and across the cluster's CPU models (faer selects SIMD kernels at run time) |
| Bound-constrained quasi-Newton | basin (L-BFGS-B, pinned) or a vendored lbfgsb-rs-pure, behind a `BoxMinimizer` trait | Correctness with active bounds; licence; MSRV ≤ 1.88; dependency weight (basin brings rand 0.10) |
| Random numbers | The workspace's rand 0.8 with rand_chacha 0.3 (`ChaCha8Rng::set_stream`) | Stream independence |
| Special functions | Own module on `libm` (portable, unlike `std`'s system libm): log Φ, erfcx, log1mexp, log h | Accuracy against offline high-precision tables |
| Sobol | Own f64 generator on a vendored Joe–Kuo table (first 1,111 dimensions), with linear matrix scrambling and a digital shift | First points against the published sequence |
| Serialization | serde, and serde_json with `float_roundtrip` enabled by the library itself | — |
| Benchmarks and tests | criterion, proptest, num-dual (dev-dependencies) | — |

### 6.3 Library modules

```
crates/fcmaes-surrogate/src/
  lib.rs          public API
  space.rs        Space (bounds, log-scaled coordinates), unit-cube transform
  transform.rs    standardize, bilog, rank (copula); inverse for reporting
  design.rs       ScrambledSobol, maximin LHS, initial designs
  special.rs      log_ndtr, erfcx, log1mexp, log_h, normal pdf/cdf (libm)
  linalg.rs       Cholesky with jitter, append, solves, log-determinant
  kernel.rs       Matern52, Rbf (ARD): values, gradients in x and θ
  prior.rs        LogNormal densities on parameters; MAP objective terms
  gp.rs           ExactGp: fit, posterior, gradients, fantasies, LOO diagnostics
  acquisition/    log_ei.rs, ucb.rs, thompson.rs; (P2) q_log_nei.rs, pof.rs
  optimize.rs     raw samples, RAASP, starts, L-BFGS-B, CMA-ES option
  batch.rs        KrigingBeliever, ConstantLiar, Thompson
  turbo.rs        TrustRegion state machine; TuRBO-1, TuRBO-m
  history.rs      Observation, History, Pending, failures, recommendation
  strategy.rs     Strategy trait; GpBo, Turbo; (P2) Rbf, HandOff
  driver.rs       synchronous and asynchronous loops; log events; replay
  sim.rs          discrete-event simulator for benchmarks
  problems/       Tier-A functions and Rover (GTOP through the dev-dependency)
```

### 6.4 API sketch

State is data. A strategy proposes from an explicit, serializable input and
returns its new carried state. Nothing else persists between calls.

```rust
pub struct Space { lower: Vec<f64>, upper: Vec<f64>, log_scale: Vec<bool> }

pub enum Value { Finite(f64), Failed(String) }   // panics, non-finite, >= failure_threshold

pub struct Observation { pub id: u64, pub x: Vec<f64>, pub value: Value }

pub struct Round<'a> {
    pub space: &'a Space,
    pub history: &'a [Observation],   // completed, in completion order
    pub pending: &'a [(u64, Vec<f64>)],// proposed, not completed (ids)
    pub state: &'a StrategyState,     // hyperparameters, trust regions, counters
    pub seed: u64,
    pub round: u64,                   // proposal calls so far
    pub count: usize,                 // points to propose
}

pub struct Proposal {
    pub points: Vec<Vec<f64>>,
    pub state: StrategyState,         // logged with the proposal
    pub diagnostics: Diagnostics,     // times, lengthscales, TR lengths, ...
}

pub trait Strategy: Send + Sync {
    /// A pure function of `round` on a given binary and CPU model, for any
    /// number of threads (the caller installs the pool). `stop` aborts:
    /// an aborted call returns `Err(Stopped)`, never a partial proposal.
    fn propose(&self, round: &Round, stop: &dyn Fn() -> bool) -> Result<Proposal>;
}
```

Model-level traits:

```rust
pub trait Surrogate: Sized {
    fn fit(data: &Data, config: &FitConfig, warm: Option<&Hyper>, rng: Streams) -> Result<Self>;
    fn with_hyper(data: &Data, hyper: &Hyper) -> Result<Self>;          // factorization only
    fn posterior(&self, x: &[f64]) -> (f64, f64);                       // mean, variance
    fn posterior_grad(&self, x: &[f64], dm: &mut [f64], dv: &mut [f64]) -> (f64, f64);
    fn fantasize(&self, x: &[f64], y: f64) -> Self;                     // O(n²)
    fn joint_sample(&self, xs: &[Vec<f64>], rng: &mut Stream) -> Result<Vec<f64>>;
}

pub trait Acquisition: Sync {
    /// log a(x) (−inf where a vanishes), with its gradient for L-BFGS-B.
    fn log_value(&self, x: &[f64], grad: Option<&mut [f64]>) -> f64;
}
```

The driver:

```rust
pub trait Evaluator: Sync {
    fn evaluate(&self, x: &[f64]) -> std::result::Result<f64, String>;
}

pub enum Schedule {
    Synchronous { concurrent: usize },
    Asynchronous { concurrent: usize },
}

pub fn run(strategy: &dyn Strategy, evaluator: &dyn Evaluator, space: &Space,
           schedule: Schedule, budget: Budget, seed: u64,
           start: Start,                    // warm-start observations and state
           pools: &Pools,                   // evaluation pool, surrogate pool
           sink: &mut dyn FnMut(&Event),    // every proposal and completion
           control: &Control) -> Result<RunResult>;

pub fn replay(strategy: &dyn Strategy, events: &[Event]) -> Result<()>; // re-proposes, compares
```

Events and replay:
- Each proposal event records:
  - the history length it saw;
  - the pending ids;
  - the hyperparameters it used;
  - the new state.
- Each completion event records the observation and its duration.
- **Synchronous mode:**
  1. Propose `concurrent` points.
  2. Evaluate them in parallel.
  3. Append them in proposal order.
  A run is reproducible from its seed.
- **Asynchronous mode:**
  1. Keep `concurrent` evaluations in flight.
  2. Append each completion when it arrives.
  3. Propose one point with the rest pending.
  A run is reproducible by `replay` of its log.

### 6.5 Job API integration: `fcmaes.surrogate/1`

**Payload:**

```json
{
  "problem": {"id": "app.wing", "version": "1", "config": {"dimension": 12},
              "bounds": {"lower": [...], "upper": [...], "broadcast": false}},
  "method": {"kind": "gp", "parameters": {"kernel": "matern52", "acquisition": "log_ei"}},
  "space": {"log_scale": [false, true, ...]},
  "model": {"noise": "inferred", "output_transform": "auto", "failure_threshold": 1e99},
  "initial_design": {"kind": "sobol", "points": 24},
  "schedule": {"mode": "asynchronous", "concurrent": 8, "batch_strategy": "kriging_believer"},
  "evaluations": 300,
  "target": null,
  "seed": 7,
  "runs": 1,
  "recheck": {"best": true, "tolerance": 0.0},
  "warm_start": {"input": "previous"}
}
```

**Valid combinations:**

| `method.kind` | `acquisition` | `batch_strategy` | `concurrent` |
| --- | --- | --- | --- |
| `gp` | `log_ei`, `ucb` | `kriging_believer`, `constant_liar_min`/`_mean`/`_max` | 1–15 |
| `gp` | `thompson` | `thompson` (implied) | 1–15 |
| `turbo` (`regions` 1–8) | `thompson` (implied) | `thompson` (implied) | 1–15 |

Other combinations are rejected. Defaults are resolved into the normalized
job.

**Resources and threads** (replacing the plain threads of the first draft):
- Both pools are task-owned Rayon pools, kept per outer worker as in
  `threads.rs`, so a model's own Rayon calls stay inside the reservation.
  - The evaluation pool has `concurrent` threads; evaluations run as pool
    tasks.
  - The surrogate pool has `surrogate_threads` threads.
- Synchronous mode: `inner_threads ≥ concurrent`. The surrogate pool is the
  evaluation pool, used between batches when it is idle.
- Asynchronous mode: `inner_threads ≥ concurrent + surrogate_threads`
  (`surrogate_threads` defaults to 1). Proposals and background refits (F25)
  run while evaluations do, and stay inside the reservation.
- With `outer_workers × inner_threads ≤ cpu_slots`, a 16-slot node allows
  asynchronous `concurrent` ≤ 15 per run. Larger concurrencies are studied
  in the library's simulator (section 8), not as jobs.

**Tasks, seeds and recommendation:**
- One task per run, seeded by `retry_run_seed`.
- The task returns the recommendation of F16.

**Evaluation log:**
- Each task writes `surrogate-<task>.jsonl` (checksummed records via
  `optimization_job_api::storage::append_checked`, synced after each
  completion), with three record kinds:
  - observations `{id, x, value, start, end}`;
  - proposals with their carried state;
  - refit adoptions.
- Size: at d = 200 and 10,000 evaluations the log is about 100 MB.
  - A compact `observations-<task>.jsonl` (completed points only) stays
    below the 64 MiB input limit up to roughly d·n ≤ 2.5·10⁶. It is what
    warm starts read.
  - The full log is for replay and diagnosis.

**Changes needed in optimization-jobs (phase 3 prerequisites):**
1. Worker-declared artifacts: a worker lists its extra files
   (`TaskResult::artifacts` or an attempt-directory `artifacts.json`). The
   manifest includes them with their SHA-256.
2. Artifacts readable while a job runs: `jobs artifact` serves declared
   append-only artifacts during the run, reading only checksummed complete
   records. A run of days can then show progress.
3. MCP `get_artifact` with `offset` and `limit` (records), beyond the 1 MiB
   cap.
4. Documentation of how an agent passes an earlier job's artifact as an
   input: `input_dir` pointing into the attempt directory, the file hashed
   as usual.

**Changes needed in fcmaes-job-worker:**
- Profile-owned inputs: a profile declares the inputs it reads itself
  (`warm_start.input`). `Registry::resolve` exempts those from its "input
  not used by the model" rule.
- `Kind::profiles()` returns every profile of a kind: scalar models get
  `fcmaes.retry/1`, `fcmaes.surrogate/1` and `fcmaes.evaluate/1`. Discovery
  lists them.

**Warm start:**
- `warm_start.input` names a hashed observations file or full log of an
  earlier run of this profile (`runs` must be 1).
- Checks: the problem id, version, parameters and dimension must match.
  - Points outside the new bounds are dropped and counted.
  - The failure threshold is re-applied.
- Completed observations enter the history; proposals never completed are
  dropped.
- `evaluations` counts new evaluations only.
- The initial design is skipped when at least its size of usable points is
  reused; otherwise it is topped up.
- A full log also restores the carried state (hyperparameters, trust
  regions) of its last completed synchronous batch.
- Interrupted jobs are never resumed: a new job with the old log as input
  continues, and its identity includes that input.
- Reused values count as `claimed` (`details.warm_start_points`) until
  re-evaluated.

**Result, one outcome per task:**
- `claim: "best_found"` with `x` and `objective`.
- Recheck:
  - With `recheck.best`, the recommended point is evaluated once more after
    the loop, if the deadline leaves time for one evaluation at the
    largest observed duration. Otherwise it is skipped, evidence is
    `claimed`, and `details.recheck` says why.
  - Agreement within `recheck.tolerance` (absolute; 0 for deterministic
    models, required > 0 when `noise` is not `none`) gives evidence
    `rechecked` (`objective`, `bounds`).
  - A disagreement refutes the claim (`objective_mismatch`), as
    `fcmaes.retry/1` does.
  - For noisy models the reported objective is the recheck value, and the
    posterior mean is in `details.predicted_objective`.
- `details`:
  - counts: evaluations, failures and failure reasons, warm-start points,
    initial design, TuRBO restarts;
  - the final hyperparameters;
  - times: fit, acquisition, evaluation, idle;
  - the log artifacts.
- Work counters: `evaluations`, `failed_evaluations`,
  `validation_evaluations`, `model_fits`.

**Cancellation and deadlines** (fixed for the SDK's `thread::scope`):
- Graceful `cancel()`, or a remaining time too short for the observed mean
  duration, stops new proposals and new evaluations at once.
- The task waits for evaluations in flight. A scoped thread cannot be
  abandoned, and an evaluation cannot be stopped through `Fn(&[f64]) -> f64`.
- That wait is bounded by the job's wall limit plus grace. The runner then
  kills the whole process group, including simulator child processes, and
  the run's committed state is its log.
- `--now` kills the group after at most one second. In-flight evaluations
  are lost and appear in the log as started but not completed. The runner
  marks the work counts as incomplete.
- Discovery states this.
- A cancellable model API (`Scalar::cancellable(Fn(&[f64], &Cancel) ->
  Result<f64, String>)`, which kills a simulator's child process) belongs to
  the external-simulator protocol, gap 1's open follow-up. Until then, users
  size `wall_time_seconds` for one evaluation more than they need.

**Limits at validation:**

| Limit | Value |
| --- | --- |
| New evaluations plus warm-start points per run | ≤ 10,000 |
| Global GP | ≤ 3,000 observations; dimension ≤ 100 |
| TuRBO local GP | ≤ 2,000 points per region; dimension ≤ 1,000 |
| Raw samples | min(1,024 + 32·d, 1e7/d) |
| Thompson-sampling candidates per region | min(5,000, max(2,000, 200·d)), further capped so that m × candidates² × 8 B ≤ 1 GiB |

### 6.6 The optional stateless profile (F22, Q2)

`fcmaes.surrogate-propose/1` takes:
- a hashed observations file;
- the space and the method;
- the pending points;
- `count` and the seed.

It returns `count` proposals in `details.points`, with claim `none` and
evidence `claimed`, since a proposal asserts nothing about the objective. The
caller evaluates them anywhere (a lab, a licensed tool, or
`fcmaes.evaluate/1` for a registered model), appends the results and submits
the next job.

It does not reopen the no-ask/tell decision:
- there is no session state, because everything is in the hashed input and
  the same input gives the same proposals;
- no claim is made about values the worker did not compute;
- a round trip per batch costs nothing when evaluations take hours.

It is still a second way to use the library, so it waits for a real
external evaluator.

### 6.7 Parallelization

| Level | What runs in parallel | Mechanism | Determinism |
| --- | --- | --- | --- |
| Jobs and nodes | Problems or seeds on the 7 nodes | One runner per node | Per job |
| Runs | `runs` independent BO runs | `outer_workers` | Seeds from `retry_run_seed` |
| Evaluations | `concurrent` evaluations of a run | Task-owned evaluation pool | Synchronous: exact. Asynchronous: by replay |
| Hyperparameter fit | 8–16 MAP restarts (Sobol starts in u-space plus the previous optimum) | Surrogate pool, `par_iter`, `Par::Seq` inside | Ordered collect; best by value, ties by index |
| Raw-sample scoring | Raw samples and RAASP candidates, Thompson-sampling candidates | Fixed chunks of 64 with one blocked solve each | Chunking independent of thread count |
| Acquisition restarts | 10–20 L-BFGS-B runs | Surrogate pool | Ordered; deterministic argmin |
| Batch selection | Greedy, so sequential; parallel inside each step | — | — |
| TuRBO-m | m regions fitted and scored | Surrogate pool | Ordered |
| Background refit (asynchronous) | Refit next to proposals | Surrogate pool; adoption logged | By replay |

**Rules:**
- Each work item draws from its own stream, derived from (run seed, round,
  purpose, index) with `set_stream`.
- No floating-point reductions across threads: collect in order, then
  reduce sequentially.
- `Par::Seq` linear algebra wherever a result feeds a decision.
- Special functions come from libm.
- Results are guaranteed bit-identical for any thread count on the same
  binary and CPU model.
- Agreement across the cluster's nodes is measured in phase 0 and
  documented, not assumed.
- Tests compare pools of 1, 2, 3 and 8 threads.

### 6.8 Algorithms in detail (P1)

**Fit:**
- Transform y (F2); map x to the unit cube.
- Minimize, over u = (log ℓ₁..log ℓ_d, log σ_n²):

  L(u) = ½ yᵀK⁻¹y + ½ log|K| + (n/2) log 2π − Σᵢ log p_LN(ℓᵢ; μ_ℓ, σ_ℓ) − log p_LN(σ_n²; −4, 1)

  - p_LN is the LogNormal density on the parameter (BoTorch convention, no
    Jacobian term). Its mode is exp(μ − σ²), about 0.205·√d for the
    lengthscales.
  - Gradient: ∂L/∂uⱼ = θⱼ·[−½ tr((ααᵀ − K⁻¹) ∂K/∂θⱼ) + (1/θⱼ)(1 + (log θⱼ − μ)/σ²)], with α = K⁻¹y.
- Bounds: ℓ ∈ [0.025, 1e3·√d]; noise variance ∈ [1e-4, 1] when inferred.
- Start from the previous optimum, plus Sobol restarts in u-space.
- **Refit cadence:**
  - Synchronous mode: refit every round while n ≤ 500, then every
    ⌈n/100⌉ rounds.
  - Between refits, each proposal factorizes K afresh with the carried
    hyperparameters.
  - Appends are used only for fantasies inside one proposal.
- **Asynchronous mode:** F25.

**Log EI:**
- log EI = log h(z) + log σ, with z = (y_best − μ)/σ and h(z) = φ(z) + zΦ(z).
- Three branches:
  - z > −1: directly;
  - −1/√ε < z ≤ −1: log φ(z) + log1mexp(log(erfcx(−z/√2)·|z|) + ½·log(π/2));
  - otherwise: log φ(z) − 2·log|z|.
- log1mexp switches at −ln 2.
- y_best is the best transformed usable value; for noisy models, the best
  posterior mean at evaluated points.

**Acquisition optimization.** Raw-sample count, RAASP sizes and start
counts are our starting values, tuned in phase 1.
1. Score Sobol raw samples (section 6.5 limit) plus RAASP candidates:
   - centres: the 5% best points;
   - perturbation: each coordinate with probability min(20/d, 1), at least
     one, from a truncated normal with σ ∈ {1e-3, 1e-2, 1e-1}.
2. Choose 10 starts by Boltzmann weights exp(η·zscore), η = 1, always
   including the best.
3. Run L-BFGS-B, at most 200 iterations each.
4. Take the best result, ties broken by index.
5. Reject points within 1e-6 (unit cube) of observed or pending points, and
   points excluded by F12; use the next best.

**Kriging believer:**
1. Add the pending points as fantasies (x, μ(x)) by appends.
2. Then, for each of the k points:
   - optimize the acquisition;
   - append (x, μ(x)).
- The incumbent stays unchanged.
- The constant liar appends the minimum, mean or maximum of the observed
  transformed y instead.

**Thompson sampling:**
1. Generate the candidates (Sobol and RAASP).
2. Draw one joint sample per point from the posterior, conditioned on the
   pending fantasies.
3. Take the argmin without replacement.
- If the candidate covariance's Cholesky factor fails after the jitter
  schedule, fall back to fewer candidates (halving, down to 500). In P2,
  pathwise sampling (F21) replaces this step.

**TuRBO:**
- **State per region:**
  - L;
  - success and failure counts;
  - restart count;
  - `region_best`, the best value since the region's last restart.
- **Success:** a batch improves `region_best` by more than 1e-3·|region_best|.
  The global best kept across restarts never counts for this.
- **Data:** the local GP uses only the observations made since the region's
  last restart (as in TuRBO; earlier data is discarded for the model), at
  most the 2,000 most recent.
- **Hyperparameters:** fitted per region with the same priors, lengthscales
  bounded to [0.005, 4] as in the BoTorch tutorial.
- **Box:** w_i = λ_i / (Π λ_j)^{1/d}, centred at the region's best and
  clipped to the unit cube.
- **Constants:** L₀ 0.8, L_min 2⁻⁷, L_max 1.6, τ_succ 3, τ_fail
  ⌈max(4/q, d/q)⌉.
- **Restart:** below L_min, with a fresh Sobol design.
- **TuRBO-m:** samples are converted back to the original y scale before
  regions are compared.

**Failures (F12):** imputation as defined there, after the output
transform. Failures count as non-improvements for TuRBO.

## 7. Tests

**Unit and numerical (fast, in CI):**
- Special functions, against offline high-precision tables (checked-in
  JSON):
  - log Φ(z): relative error ≤ 1e-14 for z ∈ [−1e6, 5]; absolute error ≤
    1e-16 for z ∈ (5, 40];
  - log h(z): relative error ≤ 1e-12 for z ≥ −1e3, ≤ 1e-6 near the branch
    boundary at −1e6;
  - continuity at z = −1 to 1e-12;
  - erfcx and log1mexp.
- Sobol:
  - before scrambling, the first 1,024 points of dimensions 1–40 match the
    Joe–Kuo sequence;
  - every 1-D projection, and the first two coordinates, of 2ᵏ points have
    exactly 2^(k−m) points in each dyadic box of side 2^−m (t = 0);
  - for higher dimensions, the counts match the Joe–Kuo t-values;
  - the same seed gives the same points.
- Kernels:
  - symmetric, PSD for random inputs (proptest), and k(x, x) = 1;
  - gradients in x and θ against num-dual and central differences, within
    1e-6 relative error.
- GP:
  - the 1-D posterior in closed form;
  - noise-free interpolation;
  - L(u) and its gradient against finite differences;
  - the prior's mode at exp(μ − σ²);
  - an append followed by a solve equals a fresh factorization (1e-10);
  - a fantasy equals a fixed-hyperparameter refit;
  - duplicate points are handled by the jitter.
- BoTorch fixtures, computed offline with GPyTorch using exact Cholesky (a
  large `max_cholesky_size`) and its MLL multiplied by n:
  - posterior and MLL at fixed hyperparameters agree to 1e-8 and 1e-6;
  - a MAP fit from the same start agrees in lengthscales to 1e-3 relative,
    and our objective is not worse than BoTorch's by more than 1e-6.
- Transforms:
  - bilog and rank are monotone and invertible;
  - `auto` picks bilog on a sample with 1e10 penalties;
  - a GP on the transformed values gives a usable posterior there.
- Acquisitions:
  - LogEI equals log EI wherever EI is representable;
  - it is finite and monotone far in the tail;
  - its gradients pass the checks;
  - basic checks of UCB and Thompson sampling.
- Optimizer:
  - finds the maxima of analytic test acquisitions and respects the bounds;
  - Boltzmann selection is deterministic for a seed;
  - the mean number of coordinates the RAASP mask perturbs matches
    min(20/d, 1)·d within three standard errors over 10⁴ draws, and is
    never 0.
- Batches:
  - a Kriging believer with k = 1 equals the sequential proposal;
  - the k points are distinct and away from pending points;
  - Thompson sampling never repeats a candidate;
  - over 100 random instances, the mean pairwise distance orders constant
    liar min < mean < max.
- TuRBO:
  - a table-driven state machine: success against `region_best`, failure,
    expansion, shrinking, restart, clipping;
  - τ_fail for q ∈ {1, 4, 32} and d ∈ {2, 50};
  - a restarted region does not see older data.
- Failures and robustness:
  - panics are caught;
  - non-finite values, the 1e99 and 1e10 thresholds, imputation, and
    exclusion after three failures;
  - constant y; y of scale 1e-12 and 1e12;
  - bounds of very different scales;
  - d = 1 and d = 100 for the global GP;
  - runs where every evaluation fails.

**Determinism (in CI):**
- The same `Round` gives bit-identical proposals on pools of 1, 2, 3 and 8
  threads, for each strategy.
- The state survives a JSON round trip.
- An asynchronous run with randomized delays, replayed from its log,
  re-proposes every point exactly, including the refit adoptions.
- An aborted proposal returns `Stopped` and changes nothing.

**Convergence** (a `slow` CI job, fixed seeds, absolute thresholds taken
from published behaviour, not from our own baseline):
- Branin: regret < 1e-3 within 50 evaluations in at least 9 of 10 seeds.
- Hartmann-6: below −3.0 within 100 evaluations in at least 8 of 10.
- GP-BO beats Sobol search at 10·d + 50 evaluations on Branin, Hartmann-6,
  Ackley-10 and Levy-10 (domains as in BoTorch): one-sided Mann–Whitney,
  α = 0.01 with Holm correction over the four, 20 seeds.
- TuRBO on Ackley-200D (domain [−5, 10]²⁰⁰, as in TuRBO): below 5 within
  10,000 evaluations with q = 100 in the simulator. Overhead stays within
  the budget set in phase 0.
- Regression guard: the medians of the frozen phase-2 baseline on the
  Tier-A smoke set must not get significantly worse (Wilcoxon signed-rank
  across problems, α = 0.01).

**Job level** (fcmaes-job-worker tests and the shared harness):
- Validation:
  - rejects unknown parameters, invalid combinations, too small an
    `inner_threads`, exceeded limits and non-scalar models;
  - rejects a warm start for a different problem, version or parameters;
  - accepts the profile-owned `warm_start` input;
  - resolves defaults.
- The log is valid checksummed JSONL after every completion, and a torn last
  line is ignored.
- Warm start (synchronous, refit every round): continuing from a log that
  ends at a batch boundary reproduces the uninterrupted run's next
  proposals exactly. Points outside new bounds are dropped and counted.
- Recheck:
  - a mismatch beyond the tolerance is refuted;
  - a noisy model within the tolerance is `rechecked`;
  - the recheck is skipped near the deadline, with evidence `claimed`;
  - every outcome passes `Outcome::check`;
  - the work counters are exact.
- Cancellation, with a model that sleeps 2 s per evaluation:
  - graceful cancel starts nothing new and returns after the evaluations
    in flight;
  - `--now` kills the process group and keeps the log;
  - a deadline stops new evaluations.
- Threads:
  - identical results for `inner_threads` 2 and 8 in synchronous mode;
  - a model that uses Rayon stays inside the task's pools (a test probe on
    `rayon::current_thread_index`).
- Integration:
  - a sleepy-model case in `scripts/job-api/test_integration.py`;
  - the MCP real-worker smoke with `fcmaes.surrogate/1`;
  - `get_artifact` paging over a running job's log;
  - a tutorial example: Cassini1 with 150 evaluations, 4 concurrent, plus
    an asynchronous sleepy application model.

## 8. Benchmarks

The quality campaign runs in the library's harness with the discrete-event
simulator, so concurrencies above the job limits (q up to 100) and budgets
above 10,000 are possible. Each benchmark cell is still a job on the
cluster: a small benchmark profile or a fcmaes-surrogate binary under
`optimization-run`. That way the job API's journals hold the raw data.
Real-time job runs are used only where they fit the profile's limits
(GTOP with q ≤ 15).

**Micro-benchmarks** (criterion, on one dedicated 16-core node, tracked per
release):
- GP fit (MAP, 12 restarts) for n ∈ {50, 100, 200, 500, 1,000, 2,000} and
  d ∈ {2, 10, 50}.
- Posterior throughput, append, LogEI value and gradient.
- One full proposal for q ∈ {1, 8, 32}.
- Thompson sampling on 2,000 and 5,000 candidates.
- Thread scaling from 1 to 16, with bit-identity checked.
- faer against nalgebra; the L-BFGS-B candidates.
- The overhead budgets (for example the proposal time at n = 500, d = 10,
  and the total for Ackley-200D) are set from the phase-0 prototype.
  Later phases must stay within 1.5× of them.

**Quality:**

| Suite | Problems | Budget | q | Seeds |
| --- | --- | --- | --- | --- |
| A1 classic | Branin, Hartmann-3/6, Ackley/Levy/Rastrigin/Griewank/Styblinski-Tang/Rosenbrock 10-D and 30-D, Michalewicz-10 (BoTorch domains, stated per problem) | 10·d + 50; 50·d | 1, 8 | 30 |
| A2 BBOB | f1–f24, d ∈ {10, 20, 40}, instances 1–3 (port of COCO's definitions or FFI to its C code, validated against COCO values) | 10·d + 50; 50·d | 1, 8 | 3 × 10 |
| A3 high-d | Branin and Hartmann-6 embedded in 100 dimensions (GP-BO and TuRBO) and 500 dimensions (TuRBO), permuted; Ackley-200D on [−5, 10] | 1,000; 10,000 | 8, 100 | 20 |
| B1 Rover | Rover 60-D (port, validated against the Python reference) | 2,000; 20,000 | 50, 100 | 20 |
| B2 GTOP | Cassini1 (6), GTOC1 (8), SAGAS (12), TandEM (18), Messenger (18), Cassini2 and Rosetta (22), Messenger full (26), with bounds from the GTOP definitions (added to `fcmaes-gtop`) | 10·d + 50 and 1,000 (GP-BO and TuRBO); 5,000 (TuRBO only) | 1, 8, 15 | 20 |
| C external | MOPTA08, SVM-388, LassoBench through the job API | Per paper | Per paper | 15 |

**Methods compared:**
- Ours: GP-BO (Kriging believer, Thompson sampling), TuRBO-1, TuRBO-m; in
  phase 5, the hand-off and RBF/DYCORS.
- fcmaes baselines:
  - CMA-ES with population q and the same initial design;
  - BiteOpt and DE with `batch` = q;
  - BIPOP-CMA-ES at the larger budgets.
- Random and Sobol search.
- External references, run separately in Python with imported result
  files, never in CI: Ax 1.3 / BoTorch 0.18 (qLogNEI), Optuna 5 GPSampler
  and TPE, and egobox 1.0.

**Batch and asynchronous study (simulator):**
- Batch sizes: q ∈ {1, 2, 4, 8, 16, 32, 64} at equal total evaluations.
- Durations: half-normal (mean 1), exponential and lognormal (σ = 1), drawn
  from the run's seed.
- Optimizer overhead enters the simulated clock through a deterministic cost
  model, a fitted function of n, d and q from the micro-benchmarks, scaled
  by 1/τ for evaluation-to-overhead ratios τ ∈ {10, 100, 10⁴}. Measured
  overhead is reported separately. The simulated runs are therefore
  reproducible.
- Synchronous batches wait for their slowest member; asynchronous slots
  refill at once.
- Report:
  - regret against evaluations and against simulated time;
  - time to target T(q) and speedup T(1)/T(q);
  - idle fraction;
  - the q from which asynchronous mode wins.

**Statistics and reporting:**
- Median and IQR, and mean ± SE, of log₁₀ regret.
- Fixed-target ECDFs (10 log-spaced targets).
- Moré–Wild data profiles at τ = 1e-1, 1e-3, 1e-5.
- ERT with simulated restarts for BBOB.
- Two-sided Mann–Whitney per problem, on final values and on the area under
  the log-regret curve.
- Wilcoxon signed-rank across problems at α = 0.01 with Holm correction.
- Fit and acquisition time, and peak memory.
- Full logs (x, y, t_start, t_end, slot) for every run.
- An IOHanalyzer-compatible export for BBOB.

**The question the campaign answers:** for which combinations of dimension,
budget, q and evaluation time does the surrogate profile beat fcmaes's
population methods? Discovery and the documentation can then tell an agent
which profile to use.

Expected answer, from Santoni et al. 2024:
- the surrogate wins below about 20·d evaluations and d ≲ 20;
- TuRBO stays competitive up to a few hundred dimensions;
- CMA-ES and BiteOpt win at large budgets.

GTOP is new ground.

## 9. Documentation

- `docs/job-api/surrogate.md`:
  - when to use the profile, with a regime table;
  - the payload and its valid combinations;
  - threads and resources;
  - the log, warm start and recheck;
  - cancellation semantics;
  - limits;
  - the benchmark results.
- A tutorial example in `tutorials/job-api`.
- `ai-context.md`: the rule for choosing a profile.
- The architecture section of the library.
- A benchmark report generated from the logs by a script in the repository.

## 10. Phases, effort and acceptance gates

| Phase | Content | Effort | Gate |
| --- | --- | --- | --- |
| 0. Decisions | Placement (Q1); faer vs. nalgebra; L-BFGS-B crate; cross-node determinism on the cluster; special-function and Sobol prototypes; a GP-fit prototype that sets the overhead budgets; re-check of the late-2026 crate facts | 1–2 weeks | A decision record `docs/surrogate/decisions.md` with numbers |
| 1. GP core, sequential BO | F1–F7, F11, F13, F15, F16 with q = 1; `problems/` Tier A; BoTorch fixtures | 3–4 weeks | Unit, fixture and determinism tests; Branin and Hartmann convergence; overhead budgets |
| 2. Batch, asynchronous, TuRBO, log | F8–F10, F12, F25; driver, events, replay, simulator; library-level log (F14) | 3–4 weeks | Exact replay; q-scaling curves on A1; Ackley-200D within budget; baseline frozen |
| 3. Job profile | Job-API prerequisites (artifacts, paging); registry changes; `fcmaes.surrogate/1` with warm start, recheck, cancellation, discovery; docs, tutorial, harness and MCP cases | 2–3 weeks | Job-level tests; mixed-node harness; tutorial in CI; an independent review, with its findings fixed before the merge |
| 4. Campaign | A1–A3, B1–B2 on the cluster; report | 2 weeks (mostly machine time) | Report with statistics; the profile-choice rule in discovery and docs |
| 5. Extensions | F17–F22 by priority; then F23–F24 | 3+ weeks | Per feature: tests and a benchmark cell that shows its benefit |

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| GP overhead beyond n ≈ 2,000 or d ≈ 100 | TuRBO's local GP cap; refit cadence; background refit; phase-0 budgets |
| Ill-conditioning (duplicates, noise-free data, small TuRBO regions) | Jitter schedule, noise floor, duplicate guard, candidate fallback |
| Penalties and heavy tails | Failure thresholds, the bilog and rank transforms, tests with 1e10 penalties |
| Asynchronous runs not reproducible from the seed | Documented; exact replay tested; synchronous mode is the reproducible default |
| Determinism across CPU models (faer's SIMD dispatch) | Measured in phase 0; claim limited to binary plus CPU model |
| Very fresh dependencies (basin 2.0.0, faer API churn) | Traits around them, pinned versions, a vendored fallback |
| Overclaiming: BO loses to CMA-ES beyond small budgets | The campaign measures the crossover; discovery says which profile to use; the hand-off (F19) |
| Hour-long evaluations vs. wall limits and cancellation | Deadline-aware starts; evaluations in flight bounded by wall limit plus grace; warm start continues; a cancellable model API with the external-simulator protocol |
| Logs too large for inputs and MCP | Compact observation files; artifact paging |
| Scope creep | Phase gates; P2 and P3 only after the campaign |

## 12. Open questions

- **Q1.** Placement: a crate in fcmaes-rust (recommended) or a repository of
  its own?
- **Q2.** Build the stateless proposal profile (F22) for evaluators outside a
  worker?
- **Q3.** Should `recheck.best` default to on? It costs one more expensive
  evaluation and gives `rechecked` evidence.
- **Q4.** For the target applications: typical d, budget, concurrency,
  evaluation time and noise? These decide the order of phase 5.
- **Q5.** Should Python access through fcmaes-py come with phase 3 or later?
- **Q6.** Should the external-simulator protocol, with a cancellable model
  API, be built before or with this profile?

## 13. References

- Ament, Daulton, Eriksson, Balandat, Bakshy (2023). Unexpected improvements to expected improvement for Bayesian optimization. [arXiv:2310.20708](https://arxiv.org/abs/2310.20708)
- Alvi, Ru, Calliess, Roberts, Osborne (2019). Asynchronous batch Bayesian optimisation with improved local penalisation. [arXiv:1901.10452](https://arxiv.org/abs/1901.10452)
- Balandat et al. (2020). BoTorch: a framework for efficient Monte-Carlo Bayesian optimization. [arXiv:1910.06403](https://arxiv.org/abs/1910.06403)
- Eriksson, Pearce, Gardner, Turner, Poloczek (2019). Scalable global optimization via local Bayesian optimization (TuRBO). [arXiv:1910.01739](https://arxiv.org/abs/1910.01739)
- Eriksson, Poloczek (2021). Scalable constrained Bayesian optimization (SCBO). [arXiv:2002.08526](https://arxiv.org/abs/2002.08526)
- Ginsbourger, Le Riche, Carraro (2010). Kriging is well-suited to parallelize optimization. [doi:10.1007/978-3-642-10701-6_6](https://doi.org/10.1007/978-3-642-10701-6_6)
- González, Dai, Hennig, Lawrence (2016). Batch Bayesian optimization via local penalization. [arXiv:1505.08052](https://arxiv.org/abs/1505.08052)
- Hansen (2019). A global surrogate assisted CMA-ES. [doi:10.1145/3321707.3321842](https://doi.org/10.1145/3321707.3321842)
- Hvarfner, Hellsten, Nardi (2024). Vanilla Bayesian optimization performs great in high dimensions. [arXiv:2402.02229](https://arxiv.org/abs/2402.02229)
- Kandasamy, Krishnamurthy, Schneider, Póczos (2018). Parallelised Bayesian optimisation via Thompson sampling. [PMLR 84](https://proceedings.mlr.press/v84/kandasamy18a.html)
- Le Riche, Picheny (2021). Revisiting Bayesian optimization in the light of the COCO benchmark. [arXiv:2103.16649](https://arxiv.org/abs/2103.16649)
- Olson et al. (2025). Ax: a platform for adaptive experimentation. [PMLR 293](https://proceedings.mlr.press/v293/olson25a.html)
- Papenmeier, Poloczek, Nardi (2025). Understanding high-dimensional Bayesian optimization. [PMLR 267](https://proceedings.mlr.press/v267/papenmeier25a.html)
- Rasmussen, Williams (2006). Gaussian processes for machine learning. [gaussianprocess.org/gpml](http://gaussianprocess.org/gpml/)
- Regis, Shoemaker (2007). A stochastic radial basis function method for the global optimization of expensive functions. [doi:10.1287/ijoc.1060.0182](https://doi.org/10.1287/ijoc.1060.0182)
- Regis, Shoemaker (2013). Combining radial basis function surrogates and dynamic coordinate search in high-dimensional expensive black-box optimization. [doi:10.1080/0305215X.2012.687731](https://doi.org/10.1080/0305215X.2012.687731)
- Santoni, Raponi, De Leone, Doerr (2024). Comparison of high-dimensional Bayesian optimization algorithms on BBOB. [arXiv:2303.00890](https://arxiv.org/abs/2303.00890)
- Snoek, Larochelle, Adams (2012). Practical Bayesian optimization of machine learning algorithms. [arXiv:1206.2944](https://arxiv.org/abs/1206.2944)
- Wilson, Hutter, Deisenroth (2018). Maximizing acquisition functions for Bayesian optimization. [arXiv:1805.10196](https://arxiv.org/abs/1805.10196)
- Wilson, Borovitskiy, Terenin, Mostowsky, Deisenroth (2020). Efficiently sampling functions from Gaussian process posteriors. [arXiv:2002.09309](https://arxiv.org/abs/2002.09309)
- Xu, Wang, Phillips, Zhe (2025). Standard Gaussian process is all you need for high-dimensional Bayesian optimization. [arXiv:2402.02746](https://arxiv.org/abs/2402.02746)
- Software and data:
  - [BoTorch](https://github.com/pytorch/botorch): gpytorch_modules.py, logei.py, safe_math.py and the TuRBO tutorial;
  - [Optuna 5.0](https://github.com/optuna/optuna/releases/tag/v5.0.0);
  - [egobox](https://github.com/relf/egobox);
  - [pySOT](https://github.com/dme65/pySOT);
  - [pycma fitness_models.py](https://github.com/CMA-ES/pycma/blob/master/cma/fitness_models.py);
  - [faer](https://docs.rs/faer);
  - [basin](https://docs.rs/basin);
  - [COCO performance assessment](https://numbbo.github.io/coco-doc/perf-assessment/);
  - [ESA GTOP](https://www.esa.int/gsp/ACT/projects/gtop/).
