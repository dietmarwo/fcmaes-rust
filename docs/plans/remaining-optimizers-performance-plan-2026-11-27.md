# Optimizing the remaining fcmaes-rust solvers

**Release 0.5.0 · 27 November 2026 · 19 tracks · 20 proposed experiments per track**

This plan uses **477 timed baseline runs in 53 configurations** and **63 additional phase-timed runs in 7 configurations**. The experiments below are proposals. The measurements describe the original release at commit `ccb9f8b1e4153c1ce2069961715cbe4b951e7806`. [Pinned source tree](https://github.com/dietmarwo/fcmaes-rust/tree/ccb9f8b1e4153c1ce2069961715cbe4b951e7806)

## Where to start

| First work | Evidence from this machine | First experiments |
|---|---|---|
| jSO proposal generation | At 128 dimensions, ask took 4.21 ms per 4,096-call run and tell took 0.52 ms. Only 13 generations completed at the default population schedule. | J02/J04 for copy and allocation cost. J15 separately tests whether smaller initial populations improve search at that budget. |
| CR-FM-NES distribution update | At 256 dimensions, tell took 16.96 ms and ask took 5.46 ms. At 8 dimensions tell still took 2.43 ms. | N05, N07 and N11, covering ranked rows, conditional distance weights and gradient temporaries. |
| PGPE proposal generation | At 256 dimensions ask took 3.19 ms and tell took 1.50 ms. At 128 dimensions the median clean loss was 44.52 on Sphere and 44.88 after noisy observations. | P01–P03 for throughput. P16 and P20 separately test whether better noise handling improves policy quality. |
| ArchiveDE survivor selection | On the two-objective front, tell took 1.26 ms and ask took 0.48 ms. The end-to-end median hypervolume was 3.4979 against a theoretical maximum of 3.5. | A01 and A05 first. Preserve tied rows and use the three-objective IGD check to detect loss of spread. |
| Evaluation accounting | At a request of 33 calls, CR-FM-NES, PGPE and Diversifier each used 64. MoRetry used 2,112 physical calls for 2,048 reported inner calls. Both CMA handoff sequences used a median 4,087 of 4,096 available calls. Three L-SRTDE handoffs on Cassini1 stopped earlier, at 2,909–3,343 calls. | N16, P15, V01 and M01–M02 before comparing call efficiency. K12/Z19 test whether the unused tail has value. |
| Expensive-objective and surrogate paths | Uniform Cassini1 objective replay cost 6.15 μs/call. jSO with Cassini1 took 6.43 μs/call. On Hartmann-3, the synchronous GP spent a median 11.61 of 11.70 ms proposing points. | Judge Cassini1 changes by loss at a deadline. Start GP work with G01–G03 and record proposal time alongside regret. |

The cost of one objective evaluation changes the order of these priorities. For the three retry coordinators the four-worker cheap-callback runs took longer than the single-worker runs. Every callback also incremented one shared counter. Verify the crossover with realistic callbacks and a lower-contention counter before changing dispatch defaults.

## Scope and entry points

The inventory starts from the public core exports, the two surrogate worker registrations, the example sequence runner, and the repository's candidate and boundary benchmark registries. The five exclusions apply to their standalone implementations. Separately exported solvers such as jSO and ArchiveDE have their own mutation, selection and budget behavior. Hybrid tracks change how their stages are connected. [Core exports](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/lib.rs) · [Worker profiles](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-job-worker/src/lib.rs) · [Portfolio and prior inclusion decisions](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/docs/algorithm-portfolio.md)

| Repository surface | Tracks with distinct 20-experiment schedules |
|---|---|
| Core scalar solvers | J jSO, N CR-FM-NES, P PGPE, D Dual Annealing |
| Core multi-objective and quality-diversity optimizers | A ArchiveDE, M weighted multi-objective retry, V Diversifier |
| Core scalar meta-optimizers | H CR-FM-NES → BiteOpt, I independent retry, R advanced coordinated retry |
| Source-built surrogate engines | G synchronous exact GP (surrogate/1), T advanced GP (surrogate/2) |
| Candidate and boundary benchmark implementations | L L-SHADE, S L-SRTDE (both differences), F AMaLGaM-IDEA, E iAMaLGaM-IDEA, B bounded Nelder–Mead |
| Named staged solvers in the candidate harness | K jSO → CMA-ES (also an example runner using core jSO), Z L-SRTDE → CMA-ES |

AMaLGaM's two modes share a file but use different population sizes and moment-update rules, so each gets its own 20 trials. Candidate jSO also serves as a parity oracle for the public jSO track. DeepBiteOpt, CMA restarts, MODE switches, MAP-Elites emitters and the DE→CMA sequence fall within the named exclusions. Archive and indicator data structures appear in the tracks that consume them. Tutorial-local grid/random controls, application heuristics and externally invoked comparison packages fall outside the reusable solver inventory. The research tracks preserve the repository's existing distinction between candidate and shipped implementations. [Candidate registry](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/benchmarks/candidate-optimizers/harness/src/solvers.rs) · [Example runner](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/examples/src/runner.rs) · [Boundary experiment](https://github.com/dietmarwo/fcmaes-rust/tree/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/benchmarks/optimizer-boundary)

## Baseline protocol

Linux x86-64 on one shared AMD EPYC 9V74 host, CPUs 0–3 available to each process, Rust/Cargo 1.100.0, release build with thin LTO and one codegen unit. The driver links the pinned checkouts using a dedicated lockfile seeded from the release workspace. This selects rand 0.8.7 for both core and the candidates. The candidate workspace's separate lockfile selects rand 0.8.8. Use the supplied driver lockfile for comparisons to these numbers.

Each configuration had a discarded run at seed 100 and nine measured runs at seeds 101–109, randomized across configurations. Each run started a fresh process. The timer covers solver construction, setup inside the measured path and the callbacks. Process startup, preparing the synthetic input bounds and computing archive quality metrics are outside the timer. Short-run numbers include pool creation where the API performs it. Objective-only replay uses pre-generated uniformly distributed rows. Cassini1 evaluation cost depends on the actual trajectory, so its control is a cost reference and cannot be subtracted from solver time to estimate overhead.

- **Scalar panel.** Shifted Sphere minimizes Σ(xᵢ−0.23)², ripple adds 0.12(1−cos(18(xᵢ−0.23))) per coordinate, and Rosenbrock uses the standard adjacent-coordinate formula. All three use [−2,2]ᴰ. The noisy PGPE panel adds reproducible, evaluation-indexed noise in [−0.1,0.1] and scores the returned point on clean Sphere. Cassini1 uses the repository's published six-variable box and objective. CR-FM-NES and PGPE use P32, normalized coordinates and an initial sigma of 0.25. NES scores use its reported, penalty-aware value. The direct scalar CR-FM-NES→BiteOpt sequence uses its defaults of a 30% first stage, P500 and Bite depth 4. The budget-33 case skips its first stage.
- **Retry panel.** 64 retries, capacity 32, 32 inner evaluations per run. The small uniform-search callback uses the run-ID seed, honors each context's box and initial guess, and exposes coordinator cost. Coordinated retry checks every eight runs and caps the budget multiplier at 1. MoRetry uses two weights within [0.05,1] and the default exponent 2. Its quality column is the union hypervolume of retained results. Worker assignment, weights and coordinated parent selection can vary with scheduling.
- **Archive panel.** ArchiveDE uses P64. The two-objective case returns (x₀+g, 1−x₀+g), with g the mean squared distance of remaining unit coordinates from 0.5. Hypervolume uses reference (2,2). DTLZ2 has three objectives, eight unit coordinates and a 121-direction angular IGD reference, including repeated pole directions. Keep that reference fixed. Diversifier uses a uniform-seeded 256-cell grid, P32 and stall criterion 8. Each evaluated quality is 1 + shifted Sphere. Grid behavior is the first two coordinates. Arm behavior is the scaled endpoint of a unit-link planar arm. Quality is the repository's reciprocal-fitness QD score. Occupancy is recorded separately.
- **GP panel.** Hartmann-3 uses 32 calls on the unit cube and q=1. Six-dimensional ripple uses 48 calls and q=2. Both engines use 8 initial points, 128 acquisition candidates, one MAP restart and 12 fit iterations. Advanced GP uses synchronous scheduling and caps training at 16 observations to exercise subset selection. Proposal time is recorded by the engine. Native callbacks run in process. A full worker invocation with serialization, event logging and simulator costs belongs in confirmation.
- **Comparability.** The controls include population initialization and tiny-budget cases. The scalar, surrogate, Pareto and QD score columns have different units. Compare methods only within a common objective, budget, dimension and scoring rule. All timing ranges below describe variability among nine seeds on this host. They are descriptive and do not establish a portable speedup.

## Measured performance

The table reports medians. The bracketed interval spans the 25th–75th percentiles of elapsed time using inclusive quartiles. μs/call divides the run time by physical objective calls. The quality column gives final loss (lower), hypervolume or HV (higher), inverted generational distance or IGD (lower), or QD score (higher). O marks an objective-only control. Post-run quality diagnostics are excluded from wall time.

| Track | Workload | D | Workers or q | Budget → actual | Wall ms [IQR] | μs/call | Quality |
|---|---|---:|---:|---:|---:|---:|---:|
| J | Cassini1 | 6 | 1 | 4,096 → 4,096 | 26.35 [25.41–28.52] | 6.4339 | loss 11.143 |
| J | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 1.16 [1.15–1.23] | 0.28201 | loss 1.9987e-07 |
| J | shifted Sphere | 128 | 1 | 4,096 → 4,096 | 5.78 [5.63–5.83] | 1.4103 | loss 12.474 |
| J | shifted Sphere | 8 | 1 | 33 → 33 | 0.022 [0.019–0.023] | 0.65858 | loss 3.4624 |
| N | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 2.97 [2.95–3.02] | 0.72401 | loss 2.0793e-25 |
| N | shifted Sphere | 128 | 1 | 4,096 → 4,096 | 13.17 [12.45–13.24] | 3.2162 | loss 0.0087966 |
| N | shifted Sphere | 256 | 1 | 4,096 → 4,096 | 23.03 [22.79–24.19] | 5.6227 | loss 1.3833 |
| N | shifted Sphere | 8 | 1 | 33 → 64 | 0.084 [0.081–0.087] | 1.316 | loss 0.64189 |
| P | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 0.543 [0.521–0.563] | 0.13251 | loss 0.013112 |
| P | noisy Sphere | 128 | 1 | 4,096 → 4,096 | 2.95 [2.90–3.01] | 0.72045 | loss 44.882 |
| P | shifted Sphere | 128 | 1 | 4,096 → 4,096 | 2.91 [2.88–2.98] | 0.71124 | loss 44.524 |
| P | shifted Sphere | 256 | 1 | 4,096 → 4,096 | 5.54 [5.42–5.61] | 1.3531 | loss 99.342 |
| P | shifted Sphere | 8 | 1 | 33 → 64 | 0.030 [0.028–0.032] | 0.46773 | loss 0.7842 |
| D | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 1.64 [1.63–1.67] | 0.40148 | loss 6.163e-33 |
| D | Sphere, no local search | 16 | 1 | 4,096 → 4,096 | 3.07 [3.02–3.12] | 0.74952 | loss 0.0046194 |
| D | Rosenbrock | 16 | 1 | 4,096 → 4,096 | 0.834 [0.824–0.898] | 0.20355 | loss 3.1873e-17 |
| A | DTLZ2 | 8 | 1 | 4,096 → 4,096 | 2.48 [2.35–2.50] | 0.60428 | IGD 0.040857 |
| A | linear front | 8 | 1 | 4,096 → 4,096 | 1.84 [1.75–1.95] | 0.44858 | HV 3.4979 |
| A | linear front | 8 | 1 | 33 → 33 | 0.019 [0.017–0.020] | 0.583 | HV 3.1238 |
| H | Cassini1 | 6 | 1 | 4,096 → 4,096 | 29.88 [29.38–31.55] | 7.2943 | loss 19.374 |
| H | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 4.37 [4.31–4.44] | 1.0666 | loss 0.0017251 |
| H | shifted Sphere | 128 | 1 | 4,096 → 4,096 | 29.08 [27.65–33.04] | 7.0993 | loss 0.092976 |
| H | shifted Sphere | 8 | 1 | 33 → 33 | 1.56 [1.56–1.65] | 47.325 | loss 1.7672 |
| I | shifted Sphere | 16 | 1 | 2,048 → 2,048 | 0.255 [0.240–0.269] | 0.12459 | loss 6.7952 |
| I | shifted Sphere | 16 | 4 | 2,048 → 2,048 | 0.299 [0.286–0.331] | 0.14576 | loss 6.7952 |
| R | shifted Sphere | 16 | 1 | 2,048 → 2,048 | 0.297 [0.263–0.303] | 0.14481 | loss 2.0976 |
| R | shifted Sphere | 16 | 4 | 2,048 → 2,048 | 0.360 [0.346–0.423] | 0.17593 | loss 2.5946 |
| M | linear front | 8 | 1 | 2,048 → 2,112 | 0.368 [0.357–0.375] | 0.17409 | HV 3.2428 |
| M | linear front | 8 | 4 | 2,048 → 2,112 | 0.400 [0.388–0.413] | 0.18939 | HV 3.2159 |
| V | grid | 8 | 1 | 4,096 → 4,096 | 2.14 [2.13–2.16] | 0.52152 | QD 89.207 |
| V | planar arm | 16 | 1 | 4,096 → 4,096 | 5.28 [5.27–5.40] | 1.2884 | QD 97.171 |
| V | shifted Sphere | 8 | 1 | 33 → 64 | 0.146 [0.132–0.160] | 2.277 | QD 0.35668 |
| G | Hartmann-3 | 3 | 1 | 32 → 32 | 11.70 [11.53–12.04] | 365.59 | loss -3.8628 |
| G | ripple | 6 | 2 | 48 → 48 | 23.63 [22.79–25.01] | 492.34 | loss 0.51284 |
| T | Hartmann-3 | 3 | 1 | 32 → 32 | 15.74 [15.22–17.35] | 491.95 | loss -3.8549 |
| T | ripple | 6 | 2 | 48 → 48 | 31.34 [26.71–31.61] | 652.83 | loss 0.64519 |
| L | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 1.03 [1.01–1.06] | 0.25106 | loss 9.1415e-06 |
| L | Rosenbrock | 16 | 1 | 4,096 → 4,096 | 1.45 [1.43–1.46] | 0.35329 | loss 16.574 |
| S | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 1.04 [1.01–1.15] | 0.25486 | loss 0.0057441 |
| S | ripple | 16 | 1 | 4,096 → 4,096 | 2.65 [2.61–2.77] | 0.64578 | loss 0.46699 |
| F | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 0.478 [0.459–0.489] | 0.11673 | loss 8.133e-11 |
| F | Rosenbrock | 16 | 1 | 4,096 → 4,096 | 0.924 [0.904–0.950] | 0.22569 | loss 16.405 |
| E | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 0.504 [0.486–0.519] | 0.12304 | loss 1.3738e-15 |
| E | Rosenbrock | 16 | 1 | 4,096 → 4,096 | 0.974 [0.964–0.987] | 0.23771 | loss 24.179 |
| B | shifted Sphere | 8 | 1 | 4,096 → 4,096 | 0.193 [0.188–0.207] | 0.047088 | loss 3.8519e-33 |
| B | Rosenbrock | 16 | 1 | 4,096 → 4,096 | 0.590 [0.585–0.603] | 0.14393 | loss 8.0873 |
| K | Cassini1 | 6 | 1 | 4,096 → 4,087 | 24.19 [24.06–24.39] | 5.9176 | loss 10.997 |
| K | shifted Sphere | 8 | 1 | 4,096 → 4,087 | 1.52 [1.45–1.60] | 0.37159 | loss 1.2571e-12 |
| Z | Cassini1 | 6 | 1 | 4,096 → 4,087 (2,909–4,087) | 24.06 [19.92–24.43] | 5.9579 | loss 14.707 |
| Z | shifted Sphere | 8 | 1 | 4,096 → 4,087 | 1.52 [1.50–1.53] | 0.37266 | loss 3.1232e-12 |
| O | Hartmann-3 | 3 | 1 | 4,096 → 4,096 | 0.182 [0.176–0.184] | 0.044327 | loss -3.8441 |
| O | Cassini1 | 6 | 1 | 4,096 → 4,096 | 25.21 [25.06–25.38] | 6.154 | loss 22.945 |
| O | shifted Sphere | 128 | 1 | 4,096 → 4,096 | 0.353 [0.345–0.362] | 0.086253 | loss 128.35 |

Grid and arm occupancy medians were 237/256 and 184/256. The corresponding ArchiveDE archive sizes were 583 for the linear-front full-budget case and 308 for DTLZ2. The file of raw measurements includes these counts, GP proposal time and applicable solver iteration counters. The Diversifier return type has no evaluation count, so both count fields for V come from callback instrumentation.

### Public ask/tell phase timings

A second driver used the public ask/tell APIs at 4,096 calls with separate timers around each phase. It excludes construction, final reporting and cleanup between timed sections. Ask includes creation of owned output where the API returns it, and evaluation includes making its score vector. These timings explain where time is spent under this calling convention. Complete solver times are in the preceding table.

| Track | Workload | D | Ask ms | Evaluate ms | Tell ms | Largest share of timed phases |
|---|---|---:|---:|---:|---:|---:|
| A | linear front | 8 | 0.478 | 0.018 | 1.26 | tell 72% |
| J | shifted Sphere | 8 | 0.784 | 0.016 | 0.316 | ask 70% |
| J | shifted Sphere | 128 | 4.21 | 0.347 | 0.523 | ask 83% |
| N | shifted Sphere | 8 | 0.497 | 0.015 | 2.43 | tell 82% |
| N | shifted Sphere | 256 | 5.46 | 0.820 | 16.96 | tell 73% |
| P | shifted Sphere | 8 | 0.346 | 0.015 | 0.139 | ask 70% |
| P | shifted Sphere | 256 | 3.19 | 0.794 | 1.50 | ask 58% |

## Running the 20 experiments per track

One ID means one change against the pinned baseline or a documented, already accepted prerequisite. Each row specifies an observable that could disprove the change. When two candidates share code, compare them separately, then ablate each in the integrated build. Keep candidate-only changes in their own crates. For example, K07 measures the candidate-to-core substitution before any change to the handoff rules.

1. **Reproduce the reference.** Build the checked-in revision with the supplied lockfile and run the two panels. Count objective calls in the callback. Capture elapsed time, achieved objective/front/archive score, and the returned point or front. For a candidate claiming a copy, sort or allocation saving, add that counter in a separate profiling build and time the ordinary release build. Warm a persistent process in a separate test if the caller reuses pools.
2. **Screen one change.** Use development seeds 201–212 in randomized A/B or B/A order on the same host and CPUs. For sub-millisecond runs, time blocks long enough to exceed 100 ms, with a fresh solver and matched seeds for each invocation. Bracket expensive objective timings with an objective-only control. Store the entire candidate trace for at least two representative seeds.
3. **Confirm finalists.** Freeze the candidate and all thresholds, then use 30 unseen seed pairs (1001–1030). Include the baseline's cheap cell plus one held-out application family. Compare equal *physical evaluations* and a fixed wall deadline calibrated from the baseline's median time for the same workload. For whole-population APIs, use a common divisible budget or an identical truncation policy. For parallel workers, record CPU time, deadlines and in-flight/drained calls. Schedule-sensitive methods need repeated schedules within seeds.
4. **Decide.** Compute per-seed time ratios (reference/candidate), median paired differences in quality and a 95% paired bootstrap interval with 10,000 resamples. Resample seeds, then schedules within seeds when applicable. Predeclare the primary cell, quality target and tolerances before seeing the holdout. Apply Holm correction across finalists within each track when making multiple superiority claims. Record inconclusive results as such.
5. **Integrate and verify.** Rerun the same panels against both the pinned reference and each track's accepted changes. Use leave-one-change-out comparisons for interacting survivors. Run the repository's formatting, workspace Clippy, tests and Rust 1.88 compatibility checks. Exercise the Python binding and job-worker entry point wherever the changed behavior is exposed. For candidate-specific changes, run its standalone workspace tests too. [Repository build and verification instructions](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/AGENTS.md)

| Row type | Default acceptance rule for the future campaign |
|---|---|
| **E — implementation efficiency** | Preserve seeded candidate order, evaluation count, stop and public result when exact arithmetic/order is intended. For numerical kernels, compare intermediate arrays against reference tolerances fixed before screening, then apply the quality gate below if any ranking or trajectory changes. Seek ≥5% primary median speedup with the interval above 1.00, or ≥20% less allocated memory with runtime within a 3% noninferiority margin. |
| **Q — search policy or changed work** | Seek ≥10% lower median nonnegative regret or time-to-target, or ≥5% higher Pareto/QD quality, with paired uncertainty supporting improvement. On a second family exclude a loss worse than 5% using the predeclared quality scale. For near-zero or signed objectives use an absolute target/regret margin instead of ratios. Preserve coverage/extreme points and feasibility as applicable. A workload-specific trade-off is a candidate for an opt-in setting. |
| **C — correctness or API contract** | Verify the exact contract directly, including counts, feasibility, observed best and ordering. Require the existing behavior for existing callers where a new mode is opt-in. Publish any unavoidable cost separately. A correction earns credit for satisfying the contract. |

**Additional confirmation workloads.** For J/H/K/Z, use held-out GTOP trajectories such as Tandem or Cassini2. For N/P, use a rotated ill-conditioned problem and the repository's neural controller with independent episode seeds. For D/B/F/E, use Lennard–Jones and a narrow bounded Rosenbrock. For A/M, use ZDT3 and DTLZ2 with M=3/5, reference HV agreed in advance and an external feasible reference set for IGD. For V, use a more saturated grid and a CVT arm archive and report QD score, coverage, best elite and evaluations. For I/R/M and G/T, repeat with deterministic 1, 10 and 100 ms callbacks, then introduce controlled stragglers. For L/S, include the repository's candidate benchmark functions and a rotated high-dimensional case.

**Result ledger.** Keep one row per (experiment, build, workload, seed, schedule). Record base and candidate SHAs, configuration, compiler and CPU, worker counts, nominal/reported/physical evaluations, stop reason, elapsed and CPU time, quality, and the row-specific diagnostic. Record accepted, opt-in, rejected or inconclusive with the comparison, interval, guard result and reason. Store outputs and commands before testing the next change.

## The 380 proposed changes

Each track has 20 independent trial IDs. The type column gives the acceptance rule above. The final column specifies the measurement or regression check that distinguishes that trial. Numbers such as D128 and P32 denote dimension and population size.

### J · jSO

The adaptive population shrinks with the consumed budget. Measure initial, mid-run and final generations separately. Keep the candidate crate's jSO as an independent trace oracle where the parameters match. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/jso.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| J01 | E | Generate only the remaining trial prefix on the final ask | At B=P+1 count generated rows and replay the last accepted parent. |
| J02 | E | Add a borrowed pending-row view for in-process scoring | At D128 count copied bytes and compare owned ask/tell with optimize. |
| J03 | E | Move surviving rows during population reduction using a stable permutation | Compare post-shrink population order and bytes copied at every shrink. |
| J04 | E | Reuse the mutant scratch vector across targets | Count allocations per member at D8 and D128. |
| J05 | E | Fuse midpoint repair into mutant construction | Replay out-of-bounds mutants and the forced crossover coordinate. |
| J06 | E | Cache the post-shrink rank index until fitness changes | Count full sorts and verify p-best IDs on ties. |
| J07 | E | Use partial selection plus a sorted prefix for p-best on generations without shrinkage | Compare donor IDs to full stable ranking on plateaus. |
| J08 | E | Recycle the three success-history arrays between tells | Stress all-success and no-success generations and track allocated bytes. |
| J09 | E | Compute both Lehmer means in one traversal of the success records | Check terminal CR, infinite improvement weights and empty successes. |
| J10 | E | Remove the initialization clone by tracking the best row index before copying | B below P and early target must return the observed pair. |
| J11 | E | Specialize internal serial evaluation to consume borrowed rows directly | Compare direct versus batch objective traces and calls. |
| J12 | E | Refill an existing archive slot from a displaced parent on replacement | Test zero, fractional and full archive capacities and RNG advancement. |
| J13 | E | Keep generation-level progress conditions out of the inner CR/F loop | Replay one generation either side of 20%, 25%, 40%, 50% and 60%. |
| J14 | E | Reuse the score vector in parallel evaluation | Benchmark 1/4 workers with cheap and delayed objective costs. |
| J15 | Q | Test a bounded initial population of min(default,128) | D128 at 4k and 40k calls, compare regret and generations. |
| J16 | Q | Raise minimum population from 4 to 8 | Measure late stagnation and Cassini1 target hits. |
| J17 | Q | Halve the archive ratio | Compare memory high-water mark and diversity on ripple and rotated ellipsoids. |
| J18 | Q | Enlarge the success memory from 5 to 10 | Record memory turnover and loss under an ill-conditioned objective. |
| J19 | Q | End the p-best schedule at 0.08 | Plot last-quarter improvement and early-target failures. |
| J20 | Q | Reverse the stage budget pressure using a convex population-reduction schedule | Compare front-loaded exploration and holdout Cassini1 at equal actual calls. |

### N · CR-FM-NES

The update materializes several population-by-dimension arrays. Separate ask, decode, penalty/rank and natural-gradient timings. Use a rotated ellipsoid and a bounded controller for confirmation. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/crfmnes.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| N01 | E | Fill existing normal sample columns in place | Track allocations per ask and exact seeded draws at D256. |
| N02 | E | Write the mirrored half directly from its positive partner | Compare every paired coordinate and the next RNG draw. |
| N03 | E | Fuse rank-one sample construction and mean/diagonal transform | Replay z, y, x and first decoded callback rows. |
| N04 | E | Decode and clip into one reusable output row | Stress bounded normalized, narrow boxes and identity decoding. |
| N05 | E | Replace physical reordering of z/y/x with a stable rank permutation | Count bytes moved per tell and compare tied ranks. |
| N06 | E | Keep original population access through row IDs instead of cloning xs_no_sort | Check population() ordering before and after tell. |
| N07 | E | Build distance weights only in the movement regime | Count norm/exp calls and check both sides of chi thresholds. |
| N08 | E | Reuse squared z norms for distance weights and sigma accumulation | D256 norm calls and trajectory drift under tight FP tolerance. |
| N09 | E | Accumulate weighted displacements directly from x and the old mean | Validate the m and pc vectors for a full generation. |
| N10 | E | Treat pc/D as a final borrowed or scratch column in ex_y | Compare heap high-water in tell and the final-column contribution. |
| N11 | E | Stream squared-y and y-times-vbar terms into the first gradient step | Track removal of two D×P intermediates. |
| N12 | E | Recycle t and s workspaces between natural-gradient steps | Compare all selected intermediate vectors, including div==0 exit. |
| N13 | E | Accumulate the final v and diagonal updates together by sorted column | Measure tell time and minimum diagonal on rotated problems. |
| N14 | E | Cache eta/alpha scalars keyed by finite evaluation count | Include overflow penalties and all-finite batches. |
| N15 | E | Use a persistent reusable sort-index vector | Tie scores, infinite-penalty tie-breaks and P=2/32/512. |
| N16 | C | Add an opt-in hard cap that records a partial generation without updating its distribution | B=1,33,63 and external stop must return the best observed point. |
| N17 | C | Offer an observed feasible incumbent alongside the penalized rank value | Boundary optimum, re-evaluate returned point and preserve the ranking trace. |
| N18 | Q | Scale population with dimension using the nearest even value to 4√D | Compare time-to-target and calls at D32/128/256. |
| N19 | Q | Initialize sigma at 0.1 of normalized half-width versus 0.25 | Boundary-start and anisotropic-controller holdouts. |
| N20 | Q | Scale penalty by coordinate range and measure bound recovery | Track infeasible fraction, raw best and feasibility under stretched boxes. |

### P · PGPE

The estimator traverses mirrored perturbations and uses rank utilities by default. In noisy cases, compare the returned policy on fresh common noise streams. Record clean loss separately from the smallest observed noisy reward. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/pgpe.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| P01 | E | Reuse the encoded mirrored population across asks | Track allocation count and point order. |
| P02 | E | Emit plus and minus candidates into adjacent flat rows | Verify pair identity and callback row order at D256. |
| P03 | E | Fuse normalized decoding with bound clamping per candidate | Test clipped and unclipped coordinates on heterogeneous bounds. |
| P04 | E | Store scaled noises in a contiguous pair-major buffer | Compare cache misses or bytes touched and gradients at D128/256. |
| P05 | E | Traverse perturbation pairs outside coordinates in reinforce | Compare per-coordinate gradient accumulation order and runtime. |
| P06 | E | Precompute the score difference per perturbation pair | Count repeated subtractions on the D256 panel. |
| P07 | E | Precompute pair mean minus global mean | Include raw and ranked utilities, and all-equal scores. |
| P08 | E | Cache stdev-squared for each gradient update | Check very small positive standard deviations and finite steps. |
| P09 | E | Rank losses directly in descending order without the temporary negated-score vector | Check stable equal-score ranks and raw-score mode. |
| P10 | E | Reuse rank-index storage across tells | Measure allocation count with P32 and P256 ties. |
| P11 | E | Update Adam moments and center in place | Replay moments and the center across a decay boundary. |
| P12 | E | Apply relative stdev clipping into the current vector | Record bound hits for positive and negative gradients. |
| P13 | E | Remove the extra center copy by giving the Adam state one authoritative center | Verify ask immediately after tell and result(). |
| P14 | E | Reuse the best decoded row's destination on improvement | Monotonically improving callbacks, then no improvement. |
| P15 | C | Offer an explicit last-partial-batch policy with an observed incumbent | B=1,31,33 and odd requested P, preserve complete-pair gradient semantics. |
| P16 | Q | Compare ranked utilities with raw utilities under matched noise | Fresh-rollout loss, failures and target probability. |
| P17 | Q | Reduce the center learning rate from 0.15 to 0.075 | Test high-dimensional correlated and noisy policy returns. |
| P18 | Q | Halve the permitted relative stdev change | Measure exploration collapse and late loss. |
| P19 | Q | Test a larger perturbation cohort at equal physical evaluations | P32 versus P64 on noisy controller runs. |
| P20 | Q | Supply common random numbers to both members through an opt-in paired evaluator | Compare gradient variance and held-out returns at equal rollout cost. |

### D · Dual Annealing

Distinguish global visits, accepted points, finite-difference calls and line-search calls. Archive the local-search-on/off controls. Termination on a hard budget must include every physical evaluation. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/da.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| D01 | E | Write bounded clamp and decode into one buffer | Check encoded and decoded points at both bounds. |
| D02 | E | Reuse the two Gaussian draw arrays in visit_fn | Compare draw order and visits for full and single-coordinate steps. |
| D03 | E | Specialize one-coordinate visiting without allocating length-one vectors | Count allocations in the second half of each chain. |
| D04 | E | Cache the temperature-only visiting coefficient once per chain | Replay accepted moves over a complete cooling cycle. |
| D05 | E | Reuse the full-dimensional proposal row across visits | Record bytes allocated in global-only D16/64. |
| D06 | E | Keep a reusable incumbent buffer in value | Improving stream and non-finite callback cases. |
| D07 | E | Mutate a single finite-difference probe row and restore each coordinate | Bound-edge gradients and the exact evaluation order. |
| D08 | E | Reuse central gradient storage across line-search iterations | Check incomplete gradient when the last call spends the budget. |
| D09 | E | Ring-buffer L-BFGS correction pairs | Compare oldest-pair eviction and descent directions. |
| D10 | E | Reuse the two-loop recursion's temporary coefficient and direction storage | Count allocations per local update on Rosenbrock. |
| D11 | E | Fuse projected trial construction and check for a stationary projected step | Verify step length and line-search exit at a boundary optimum. |
| D12 | E | Reuse local line-search candidate rows | Check that rejection restores the previous point, including a budget exit. |
| D13 | Q | Cache an already known base value during finite differences for deterministic callbacks | Compare unique calls, returned quality and explicit evaluation accounting. |
| D14 | C | Report completed global iterations alongside actual calls | Check counts on initialization failure, partial chain and ordinary termination. |
| D15 | Q | Schedule local descent only after a minimum relative global improvement | Lennard-Jones and Rosenbrock, count local calls saved per target. |
| D16 | Q | Scale finite-difference steps by the coordinate's local magnitude | Compare numerical gradients on a stretched box and final loss. |
| D17 | Q | Try one-sided differences near active bounds | Gradient error, physical calls and stagnation at corners. |
| D18 | Q | Shorten L-BFGS history | Compare 3 versus current history on smooth and rough controls. |
| D19 | Q | Reduce the starting visiting temperature | Measure basin changes and fixed-budget multimodal loss. |
| D20 | Q | Trigger reannealing on a fixed stagnation window | Compare early and late target rates against the existing reset policy. |

### A · ArchiveDE

This independently exported multi-objective solver has an archive gate and a distinct survivor policy. Compare against the bundled MT19937-compatible reference traces. Preserve equal nondominated rows and payload order. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/archivede.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| A01 | E | Reuse the tell survivor-pool buffers between generations | Count row allocations and verify admitted child IDs. |
| A02 | E | Specialize fixed decision arrays beside dynamic Vec rows | D8/M2 and D32/M3, compare ordered archive and wall time. |
| A03 | E | Reuse scratch for objective-rank permutations | Tie-heavy fronts, signed zero and exact champion choice. |
| A04 | E | Partially select the rank-sum tail before filling survivor slots | Match the full reference tie-break at every cut. |
| A05 | E | Replace repeated shifting on removal from the spread candidate list with stable live slots | Compare every farthest winner on exact ties. |
| A06 | E | Cache rank-space distances to new seeds in contiguous scratch | Check survivor order and bytes read at P64/256. |
| A07 | E | Reuse axis-specific champion IDs when building rank sums | Count comparisons with duplicate axis minima. |
| A08 | E | Move winning population rows into the next buffer | Record row copies, preserve incumbent and offspring ordering. |
| A09 | E | Reuse dynamic offspring row storage across asks | Check repeated ask and partial-final pairs at B=33/65. |
| A10 | E | Add a reusable result-row buffer in minimize | Benchmark direct closure versus ask/tell at D8. |
| A11 | E | Batch MT19937 draws into fixed scratch preserving consumption | Compare raw draws, duplicates and donor index rejections to the reference. |
| A12 | E | Delay cloning admitted decision payloads until the survivor pool needs ownership | All-rejected and all-admitted offspring streams. |
| A13 | E | Tune the existing 3-objective block-promotion threshold on isolated updates | Sweep 128/256/512 and replay each insertion against brute force. |
| A14 | E | Reuse 3D block capacity after compaction | Measure resident memory and exact equal-row multiplicity on DTLZ2. |
| A15 | E | Accelerate repeated 2D skyline-key lookups in same-x streams | Preserve signed zero and insertion order within each x bucket. |
| A16 | E | Fast-path fixed decision coordinates during scaling | Nonunit ranges and heterogeneous fixed/free coordinates. |
| A17 | Q | Disable archive gating only | Compare HV/IGD, admission fraction and wall time on ZDT/DTLZ. |
| A18 | Q | Replace tiered survivors with the existing rank-sum mode only | Check extremal coverage and sparse regions. |
| A19 | Q | Scale population with objective count | P32/64/128 on M2/3/5 with a fixed physical budget. |
| A20 | Q | Weight the rank-space spread by empirical objective redundancy | Use independently rotated and correlated-objective fronts to test lost extremes. |

### H · CR-FM-NES → BiteOpt

Measure stage boundaries, transitions and per-call observation costs. Default P500 and a 30% first-stage share mean that small budgets can contain zero or very few NES generations. Treat the BiteOpt kernel as fixed. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/crfmnes_bite.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| H01 | E | Add a serial incumbent observer that avoids the mutex in the serial path | Confirm exact observed best after every proposal. Count lock operations. |
| H02 | E | Combine best-fitness and already-reached-target reads at each stage boundary | Use a target reached by the last NES row. |
| H03 | E | Keep the first-stage score buffer across generations | Record allocations per whole population at P500. |
| H04 | E | Borrow decoded NES rows for the stage adapter | Measure copy traffic and verify feasible callback points. |
| H05 | E | Fuse coordinate finiteness validation with decoding while retaining pre-batch checks | Inject a nonfinite coordinate late in the generation. |
| H06 | E | Transfer ownership of the incumbent at handoff | Count D-vector clones with D128/512. |
| H07 | E | Defer construction of the NES state and its fitness bounds until a generation fits | Test shares 0, 0.3 and a budget just below P/share. |
| H08 | E | Skip second-stage construction after cancellation at handoff using one latched state | Check a stop on the last first-stage callback. |
| H09 | E | Skip the locked target read during the Bite stage when the threshold is negative infinity | Retain the external stop check and compare the full sequence when first-stage share is zero. |
| H10 | E | Precompute the normalized box transform for repeated retry contexts | Measure cold one-run versus 64-retry setup time. |
| H11 | C | Enforce target checks immediately after each first-stage evaluation if requested | Count excess calls for a target crossed mid-population. |
| H12 | C | Record stage-specific actual calls in optional diagnostics | Counts must sum to the returned total for cancellation and invalid observations. |
| H13 | Q | Lower the shared population to 128 | Compare number of NES generations, Cassini1 loss and total wall time. |
| H14 | Q | Choose population from first-stage budget to permit at least four generations | Sweep tiny and medium budgets at equal total calls. |
| H15 | Q | Move the fixed first-stage share to 0.5 | Compare regret at handoff and at the final deadline. |
| H16 | Q | Hand off early after a rolling window of stagnant NES updates | Preserve exact residual calls and measure false early switches. |
| H17 | Q | Initialize the second stage with a bounded jitter around the incumbent | Recheck quality on plateaus and large-variable boxes. |
| H18 | Q | Compare fixed 0.08 sigma with the seeded random width | Normalize widths and measure seed-to-seed dispersion across problem scales. |
| H19 | Q | Vary Bite depth only at the sequence boundary | Depth 1 versus default 4, fixed total calls on Cassini1 and ripple. |
| H20 | Q | Decouple Bite population size from NES population size at handoff | Fix NES at P500, compare Bite P64/128/500 on the same residual calls. |

### I · Independent retry

Benchmark the coordinator with 64 very short runs and again with a real scalar optimizer. If callback time dominates, use a fixed wall deadline to test whether a scheduling change helps. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/retry.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| I01 | E | Run the workers=1 loop on the calling thread to avoid constructing the Rayon pool | Compare 1, 2 and 64 retries with the same contexts. |
| I02 | E | Share immutable global bounds across retry contexts | Count bound-vector clones and check retained context ownership. |
| I03 | E | Reuse worker-local allocation for the uniform sdev row | Inspect RNG draw count and aliasing if a caller clones the context. |
| I04 | E | Pre-reserve progress history to the enabled sample limit only when useful | Disabled, one-sample and large-capacity memory cases. |
| I05 | E | Avoid reading the elapsed clock when a new minimum cannot produce a stored sample | Check the capped progress record's final value. |
| I06 | Q | Partition a target-free schedule into contiguous run-ID ranges | Compare dispatch cost and coverage. Classify changed legacy worker streams as Q. |
| I07 | E | Claim small chunks of run IDs for expensive runs | Recheck cancellation before every run and count unstarted claims. |
| I08 | E | Accumulate completed-call counters locally before publication when statistics are disabled | No lost counts after stop or short runs. |
| I09 | E | Move validity checks ahead of store locking when they depend only on the returned row | Test invalid lengths, NaNs and value limits. |
| I10 | E | Replace periodic full retention sorting by selection plus sorted kept prefix | Compare retained entries, including ties, at capacities 1/32/500. |
| I11 | E | Delay best-x cloning until a retained winner needs a second owner | Count copies on an improvement-heavy stream. |
| I12 | E | Add a disabled-statistics fast path to shared.add | Replay the same completion order and compare entries. |
| I13 | E | Reuse a bounded pool across repeated top-level retry calls | Independent concurrent callers, no nested oversubscription. |
| I14 | Q | Use per-worker completion slots with an ordered reducer for deterministic mode | Compare run-ID coverage and 1/4-worker result replay. |
| I15 | E | Precompute the root-seed contribution shared by all worker RNG initializations | Replay every worker stream at 1/4/8 workers and measure short-run startup. |
| I16 | C | Add a hard global call reservation for uneven inner budgets as an opt-in coordinator API | 63/64/65 retries and early exit, reconcile with callback counts. |
| I17 | Q | Allocate longer runs only to surviving basins after a short pilot | Equal actual calls and deadline success on multimodal problems. |
| I18 | Q | Choose run width from an initial timing sample within a fixed CPU limit | Cheap callbacks and expensive uneven simulations. |
| I19 | Q | Replace the common per-run budget with a predeclared short/long mix | Keep aggregate calls fixed and compare target-hit probability. |
| I20 | Q | Use stratified initial guesses across independent retries | Test an optimizer honoring guesses on rotated and disconnected basins. |

### R · Advanced coordinated retry

Benchmark advanced_context separately from result insertion and diversity pruning. Preserve its comparison with the last two accepted entries when testing a mechanical change. Test both serial deterministic order and real concurrent completion. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/retry.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| R01 | E | Precompute inverse global widths used by diversity pruning | Compare retained IDs around the threshold on uneven bounds. |
| R02 | E | Compare squared normalized distance to a precomputed squared threshold | Include nextafter threshold cases and negative configuration normalization. |
| R03 | E | Short-circuit a distance once its lower bound exceeds the rejection radius | Count coordinates visited for D128 spaced solutions. |
| R04 | E | Reuse the temporary diverse-entry vector across checkpoints | Count allocation peaks at capacities 32/500. |
| R05 | E | Avoid a second sort when the store is already ordered after a checkpoint | Compare checkpoint and final retention with ties. |
| R06 | E | Cache the elite prefix length when the store size is unchanged | Verify the two parent distributions and collision resolution. |
| R07 | E | Move crossover row construction outside the store lock using an owned parent snapshot | Measure lock wait, invalidation and parent consistency. |
| R08 | E | Pack parent/donor snapshots in one contiguous buffer | Compare bytes cloned per crossover at high D. |
| R09 | E | Cache the global delta per coordinate for crossover box construction | Compare lo/hi, guess and sdev to baseline on narrow intervals. |
| R10 | E | Fill crossover lower, upper, guess and sdev using one reused workspace per worker | Check no context outlives its buffer. |
| R11 | E | Avoid reconstructing global bounds on the no-crossover path | Compare fresh starts to ordinary retry contexts. |
| R12 | E | Precompute the adaptive budget increment for monotonic run IDs | Exact integer rounding near half-ties, last run and one-run config. |
| R13 | E | Add a limited ordered insertion path for tiny capacities | Compare threshold-triggered truncation at 1, 8 and 32. |
| R14 | C | Give advanced_retry an external cooperative stop predicate | Stop before claims, during a crossover and during a running optimizer. Check counts. |
| R15 | Q | Select parents from a truly sorted elite prefix between checkpoints | Compare parent ranks, diversity and final loss against existing ordering. |
| R16 | Q | Decrease checkpoint interval adaptively as the store fills | Hold CPU limit and physical calls fixed. |
| R17 | Q | Scale diversity radius by the observed coordinate spread | Reevaluate multi-basin retention and best scalar value. |
| R18 | Q | Bias parent selection toward an underrepresented basin | Check escape rate and useful crossovers on rotated multimodal functions. |
| R19 | Q | Replace the linear retry-budget ramp with a geometric ramp of equal total work | Compare early target hits and final retained quality. |
| R20 | Q | Adapt crossover probability to the recent rate of beating the selected parent | Measure failed crossovers and deadline success on held-out basins. |

### M · Weighted multi-objective retry (MoRetry)

The baseline invokes the original objective once more per completed valid-width retry to store its vector. Use physical callback counts for comparisons. Assess the returned front across different weights using hypervolume and feasibility. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/moretry.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| M01 | C | Add an optional result carrier for the already-evaluated original vector | Compare 64 validation calls saved, strict validation and stochastic-objective semantics. |
| M02 | C | Expose inner calls and post-run reevaluations as separate counters | B=2048, 64 completions, invalid-width cases and failed callbacks. |
| M03 | E | Specialize scalarization at exponent two using multiplication and sqrt | Check near-overflow, signed inputs, penalties and NaN replacement. |
| M04 | E | Specialize the exponent-one scalarization | Signed values, two positive violations and early validation. |
| M05 | E | Combine constraint penalty accumulation with the weighted power pass | Compare negative, zero and positive constraint boundary values. |
| M06 | E | Validate immutable weights/exponent once per WeightedObjective | Preserve the public scalarize function's validation behavior. |
| M07 | E | Add caller-provided scratch for vector-valued objectives | Compare per-candidate allocations for 2/10 objectives. |
| M08 | E | Compute scaled weights in the sampled raw-weight vector | Preserve normalization and all-zero fallback. |
| M09 | E | Cache constant weight spans for a campaign | Compare every sampled weight under fixed and zero-width ranges. |
| M10 | E | Defer copies of final best x/y until result materialization | Capacity 1 with a worse late retry and mixed return widths. |
| M11 | E | Optimize retention pruning by selecting the kept fraction before ordering | Equal scalar values, signed zero and capacity boundaries. |
| M12 | E | Reuse the worker's bounds/sdev context storage | Confirm weights stay attached to their winning x/y. |
| M13 | E | Move finite/limit checks on returned vectors outside the store lock | Count rejected runs exactly, stress four workers. |
| M14 | E | Use a 2D skyline pass for pareto_indices postprocessing | Return baseline input indices including duplicate nondominated rows. |
| M15 | E | Precompute dominated candidates using sorted first objective before full multi-column checks | Verify brute force on ties and 3/8 objectives. |
| M16 | Q | Draw simplex-stratified retry weights | Hypervolume and extreme-point hits at 64 retries. |
| M17 | Q | Reserve a fixed fraction of runs for near-axis weights | Compare extreme coverage and central-front gaps. |
| M18 | Q | Normalize objective scales from a fixed calibration set | Include calibration calls and keep constraint units explicit. |
| M19 | Q | Retain solutions by nondominance/spread under capacity pressure | Evaluate constrained feasibility and front quality at capacity 8/32. |
| M20 | Q | Choose the next weights from gaps in the current feasible front | Compare gapped and disconnected fronts at fixed calls. |

### V · Diversifier

Time diversify and diversify_batch, especially emitter starts, improvement scoring, stall detection and archive reranking. Reuse the existing archive and CMA-ES engines. Keep grid and planar-arm descriptors as separate endpoints. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-core/src/mapelites.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| V01 | C | Add a strict-budget mode that evaluates the last prefix and never tells an incomplete population | B=1,31,33 and 4097, check actual calls, changes to niches and returned best. |
| V02 | E | Borrow the emitter's decoded batch through the flat ask API | Compare public callback order and population copy bytes. |
| V03 | E | Recycle real-quality and improvement score buffers across generations | Track allocation count at P31/128. |
| V04 | E | Carry the real-quality minimum through archive update | Verify it still represents every valid evaluated point, including non-elite losers. |
| V05 | E | Reuse two sorted stall-score buffers by swapping them | Check the exact stall generation with ties and infinities. |
| V06 | E | Share sorted improvement indices between stall detection and emitter ranking when ordering agrees | Verify both consumers' stable ties and infinity handling. |
| V07 | E | Skip sorting for stall detection on the first iteration if its sorted result can be computed lazily | Test one-generation emitters and those reaching iteration two. |
| V08 | E | Keep repeated Fitness bounds setup outside the restart loop | Compare normalized transforms and teardown at many short restarts. |
| V09 | E | Borrow the selected elite row during emitter initialization | Ensure selected archive entry can be replaced immediately afterward. |
| V10 | E | Track dirty niche-rank entries between emitter restarts | Validate sampled elite ranks against complete argsort after collisions. |
| V11 | E | Specialize ordered scalar callbacks to update a reusable result batch | Compare against serial adapter with wrong descriptors and zero occupancy. |
| V12 | E | Add an optional fixed-width two-descriptor batch view | Measure allocations on grid and arm and verify niche IDs. |
| V13 | E | Precompute niche IDs in parallel for costly CVT searches, then commit in input order | Colliding candidates must see earlier updates from the same batch. |
| V14 | E | Cache reusable emitter workspace at same D and P, reinitialize every state field | Exact seeds, means, stopping counters and restart quality. |
| V15 | Q | Seed emitters by under-visited occupied niches | Measure occupancy, QD score and best elite on arm. |
| V16 | Q | Sample restart sigma log-uniformly across the current range | Coverage and per-niche improvement on narrow behavior manifolds. |
| V17 | Q | Make the stall window proportional to population size | Measure restarts per 4k calls and coverage. |
| V18 | Q | Normalize the improvement signal by robust within-niche fitness scale | Check coverage and quality for positive and mixed-sign archives. |
| V19 | Q | Mix elite starts with a predeclared fraction of uniform starts | Score sparse, freshly seeded and nearly full archives. |
| V20 | Q | Trigger restarts from rolling improvement in occupied-cell count | Confirm quality on saturated archives where coverage cannot increase. |

### G · Synchronous exact-GP Bayesian optimization (surrogate/1)

Use Hartmann and the cheap ripple control, then a delayed simulator. Record proposal, fit, acquisition and callback time separately. Compare GP means, posterior variances, jitter and selected candidates to numerical fixtures. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-surrogate/src/strategy.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| G01 | E | Reuse per-thread cross-covariance and triangular-solve buffers in posterior | Measure allocation count per acquisition candidate. |
| G02 | E | Pass the already-computed top-start acquisition score into coordinate refinement | Remove one posterior call per start and replay winners. |
| G03 | E | Replace full candidate score sorting by a deterministic four-best selection | Verify stable original-index ordering including nonfinite scores. |
| G04 | E | Cache normalized training coordinates over rounds | Warm-history replay with invalid or changed coordinates. |
| G05 | E | Add a value-only kernel path to acquisition prediction, which currently calls value_and_slope | Check RBF and Matérn values at tiny distances and time posterior-only queries. |
| G06 | E | Batch several posterior right-hand sides against one Cholesky factor | Candidate counts 128/1024 and memory versus scalar posterior. |
| G07 | E | Reuse analytic-gradient scratch between accepted L-BFGS steps | Compare gradient fixtures and MAP optimum for n=32/128. |
| G08 | E | Give the single-restart fit a serial dispatch path | Measure tiny Hartmann and high-n refits in a bounded pool. |
| G09 | E | Tile the lower-triangle covariance assembly across training rows | Compare packed matrix and gradient slopes, memory reads and near-duplicate points. |
| G10 | E | Use a stable bounded top-k to generate initial seeds for acquisition refinement | Apply to objective-ranked history distinct from G03's candidate pool and preserve top-history IDs. |
| G11 | E | Let a disposable fantasy model append in place while preserving the fitted real model | q=1/2/8, verify posterior against independent rebuild. |
| G12 | E | Cache fixed coordinates of a coordinate-refinement trial's normalized distances | Check clipping and exact winner on both bounds. |
| G13 | E | Cache the held-out observation exclusion structure within each synchronous batch | Verify minimum distance for duplicates and near-duplicates. |
| G14 | E | Pick a serial-versus-Rayon candidate threshold from measured cost | Check deterministic order at 1/2/4 threads. |
| G15 | Q | Refit every six proposal rounds instead of three | Time per proposal, final Hartmann regret and fallback rate. |
| G16 | Q | Halve coordinate-refinement sweeps | Count posterior calls and final clean objective at matched budgets. |
| G17 | Q | Compare LCB with LogEI under common initial designs | Branin, Hartmann and multimodal controls, fixed new observations. |
| G18 | Q | Test one versus two MAP hyperparameter restarts | Numerical failure rate and equal-wall regret on scarce data. |
| G19 | Q | Allocate global proposals from a fixed every-third schedule to a stall-based schedule | Measure escape rate and trust-region collapse. |
| G20 | Q | Change synchronous batch width using measured callback latency | Compare wall-to-target under 1/10/100 ms costs and a fixed CPU budget. |

### T · Advanced GP Bayesian optimization (surrogate/2)

Measure archive preparation, active-set selection, GP reuse, async coordination and proposal latency under skewed callback cost. Benchmark the synchronous default first. Preserve event replay for every completed, failed and cancelled ID. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/crates/fcmaes-surrogate/src/advanced.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| T01 | E | Cache unit-space coordinates by observation ID in Engine | Warm IDs, failed observations and appended pending points. |
| T02 | E | Maintain an incremental ID-ordered usable-observation view | Shuffled synchronous completion must produce the same next point. |
| T03 | E | Record the history min/max with incremental updates on append | Region progress and threshold comparisons after large outliers. |
| T04 | E | Precompute distances to the incumbent before sorting active training candidates | Report number of distance evaluations at archive sizes 128/1024/10000. |
| T05 | E | Replace selected-membership linear searches with a bitmap in farthest-point selection | Exact selected IDs near the training cap and on ties. |
| T06 | E | Cache nearest-selected distances when the training prefix is unchanged | Compare full recomputation after a new best or region restart. |
| T07 | E | Retain a persistent pending-point exclusion index | Mixed pending/completed transitions and duplicates at minimum distance. |
| T08 | E | Avoid cloning a factor when only the observation values are refreshed | Compare posterior fixtures and cache hits under a fixed transform. |
| T09 | E | Extend Cholesky reuse to multi-row append-only batches via verified successive updates | Compare full refits for q=2/8, fall back on poor pivots. |
| T10 | E | Preallocate asynchronous completion-message storage within concurrency limits | Track p95 scheduler latency with 1 versus 8 slots. |
| T11 | E | Replace per-proposal immutable history copies with versioned read-only snapshots | Check replay and late completions across a restart. |
| T12 | E | Reuse random-feature Thompson scratch across proposals within the same fitted model | Re-seed path draws independently and compare posterior samples. |
| T13 | E | Batch approximate Thompson path evaluations over candidate blocks | Measure features×candidates throughput and replay tie handling. |
| T14 | E | Share a bounded executor across repeated independent advanced runs | Enforce phase-specific thread budgets and drain on cancellation. |
| T15 | Q | Compare synchronous and asynchronous scheduling at matched worker reservation | Count stale proposals, actual callbacks and deadline regret on 5/30 ms costs. |
| T16 | Q | Reduce training cap when proposal cost exceeds callback cost | Evaluate 16/32/64 on n=128+ and log hidden refits. |
| T17 | Q | Replace the binary governor switch with a filtered proposal/evaluation-cost ratio | Timing oscillations, proposal count, clean final loss. |
| T18 | Q | Compare Sobol with Latin-hypercube initial design | Common external seeds, low budgets and nonseparable objectives. |
| T19 | Q | Change the region restart failure count from 32 to a dimension-scaled threshold | Restart frequency, survival of prior useful regions and held-out regret. |
| T20 | Q | Set BO-to-Bite handoff by estimated marginal improvement per second | Compare fixed 50% and all-GP controls, enforce remaining-population guard. |

### L · L-SHADE

This is the candidate-crate implementation. Follow its memory-cell, archive and selection conventions and compare on both an equal-call smooth function and a difficult multimodal one. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/benchmarks/candidate-optimizers/cand-de/src/lshade.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| L01 | E | Generate a mutant into retained scratch | Count allocations per accepted and rejected trial. |
| L02 | E | Apply forced-coordinate crossover directly into reserved trial storage | Verify RNG position and the selected coordinate mask. |
| L03 | E | Fuse midpoint repair with the mutant write | Check exact repaired coordinates outside both box edges. |
| L04 | E | Cache the p-best rank prefix through generation creation | Compare ordered IDs including tied fitness at the cutoff. |
| L05 | E | Stop building a population as soon as initialization exhausts the budget | Check best row and allocations for B≪18D. |
| L06 | E | Reuse pending trial storage after population reduction | Peak bytes late in a 40k-evaluation run. |
| L07 | E | Move surviving row ownership during linear population reduction | Preserve index order and next generation trace. |
| L08 | E | Compute archive capacity once per selection pass | Replay pushes and random evictions at fractional capacities. |
| L09 | E | Recycle displaced-parent storage when replacing a full archive slot | Track alloc/free operations with high acceptance. |
| L10 | E | Replace repeated memory-weight normalization with one shared scale pass | Handle infinite improvement weights identically. |
| L11 | E | Represent successes as a contiguous array of F/CR/delta records | Compare memory update with all ties, strict wins and infinities. |
| L12 | E | Reuse rank and survivor index arrays across reductions | Compare early large-P and late small-P timing. |
| L13 | E | Keep incumbent output storage and copy into it on strict improvement | Verify first finite and all-nonfinite behavior. |
| L14 | E | Specialize fixed-width D8 rows in the candidate harness | Compare public solver outputs and each objective call. |
| L15 | Q | Reduce initial 18D population to 12D | Compare regret at 1k/4k/40k on all held-out landscapes. |
| L16 | Q | Lower archive ratio from 2.6 to 1.3 | Explore memory/quality trade-off on multimodal and rotated cases. |
| L17 | Q | Increase memory size from 6 to 12 | Compare success-history variance and final loss. |
| L18 | Q | Sweep p-best fraction from 0.11 to 0.18 | Measure diversity and rate of converging to a poor basin. |
| L19 | Q | Narrow the scale-factor Cauchy distribution | Count redraws and post-midrun population improvements. |
| L20 | Q | Use a nonlinear population-reduction curve | Equal calls, examine whether late convergence or first-target rate pays for it. |

### S · L-SRTDE

Measure both implemented differences, current-target and base-target. The baseline uses the default on Sphere and the base form on ripple. Future A/B comparisons must run both forms on the same problems. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/benchmarks/candidate-optimizers/cand-de/src/lsrtde.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| S01 | E | Reuse the rank selector when size and pressure are unchanged | Count CDF rebuilds and replay selected rows. |
| S02 | E | Reorder newest population in place using a rank permutation | Preserve nc's positional meaning and tied sort order. |
| S03 | E | Retain a sorted top list and merge sorted winners | Compare top IDs when incumbent and successful trial tie. |
| S04 | E | Reuse new-population survivor indices during shrink | Measure row copies with original order preserved. |
| S05 | E | Avoid copying fit arrays on generations where they are already rank-ordered | Test monotone and reversed synthetic fitness. |
| S06 | E | Fill a reusable mutation row | Check both published difference forms under identical seeds. |
| S07 | E | Fuse crossover and midpoint repair into the trial buffer | Verify actual crossover fraction and repair order. |
| S08 | E | Move successful trial rows through reusable storage shared by pool and top selection | Ensure cyclic overwrites cannot corrupt top candidates. |
| S09 | E | Cache success-rate dependent scalar parameters per generation | Compare mean F and p-best range across zero/all-success boundaries. |
| S10 | Q | Materialize a compact lookup for eligible rank draws at N=4 | Compare distributions. If seeded draws change apply Q gates. |
| S11 | E | Replace modulo wrap of nc by a checked end-of-ring branch | Stress shrinking at wrap and full-success cycles. |
| S12 | E | Stream the weighted CR update with special handling for infinite improvements | Compare memory cell advancement even with zero wins. |
| S13 | E | Cache generation crossover-mask scratch | Compare accepted-coordinate count on forced crossover. |
| S14 | E | Partition the merged top pool before sorting only surviving members | Match total order on equal fitness and partial last generations. |
| S15 | Q | Compare the two difference equations on the same Cassini and ripple seeds | Final loss, median success rate and donor diversity. |
| S16 | Q | Vary exponential rank pressure from 3 to 2 | Measure donor positions and stagnation near N_min. |
| S17 | Q | Lower the initial newest/top population factor | Equal calls and memory on D16/64. |
| S18 | Q | Change the minimum top/newest size to eight | Probe late-cycle redraw rates and target probability. |
| S19 | Q | Smooth the success rate across two generations | Test stability of F and p-best range on alternating landscapes. |
| S20 | Q | Reduce the CR memory learning step | Track memory turnover and adaptation after long plateaus. |

### F · AMaLGaM-IDEA, full-update variant

The implementation already preallocates population buffers and batches covariance products and Gaussian sampling. Measure selection, covariance formation, factorization, resampling and objective time separately. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/benchmarks/candidate-optimizers/cand-eda/src/lib.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| F01 | E | Copy full-update covariance directly into the model when eta_sigma=1 | Compare Cholesky factors and remove a redundant scale/add pass. |
| F02 | E | Write the current mean shift directly when eta_shift=1 | Check previous-mean ownership and the first generation. |
| F03 | E | Use sorted top-selection only for the retained truncation fraction | Verify elite and selected-row order on flat fitness. |
| F04 | E | Fill centered samples in matrix storage order | Compare covariance and cache misses at D32/64. |
| F05 | E | Calculate the lower triangle of ML covariance with a symmetric rank-k kernel | Confirm mirrored values and factorization residual. |
| F06 | E | Feed an owned scratch covariance to factorization, preserving the unfactored model for retries | Singular, regularized and ordinary populations. |
| F07 | E | Use triangular multiplication for the batched Gaussian transform | Compare samples and nominal/repair cost with the existing dense product. |
| F08 | E | Combine bound checking with mapped sample output | Track failed samples and replay subsequent redraws. |
| F09 | E | Chunk sample matrix multiplication to lower peak Z/Y memory | D64, large default population, equal Gaussian stream. |
| F10 | E | Reuse the Cholesky ridge-retry matrix | Track allocation and jitter-retry count for nearly singular samples. |
| F11 | E | Cache selected means for variance-stop diagnostics without copying rows | Preserve flat/Inf termination and restart count. |
| F12 | E | Accumulate the improving-sample sum using fixed-size coordinate blocks | Verify SDR on no, one and many improving points and measure large-D throughput. |
| F13 | E | Restore the shuffled AMS index array by undoing its prefix swaps instead of refilling every index | Verify selected IDs and RNG stream for tiny and large AMS fractions. |
| F14 | E | Specialize the common no-boundary-redraw path for wide boxes | Measure branch rate and confirm narrow-box fallback. |
| F15 | Q | Vary truncation fraction from 0.35 to 0.25 | Evaluate conditioning, new basin count and final loss. |
| F16 | Q | Reduce the default full covariance population factor | Compare covariance failures and fixed-call quality at D16/64. |
| F17 | Q | Lower the allowed Gaussian redraw count | Measure clamp fraction, real calls and boundary bias. |
| F18 | Q | Test diagonal shrinkage before retrying a singular covariance | Count ridge retries and quality on correlated narrow valleys. |
| F19 | Q | Increase AMS shift distance on smooth valleys | Compare improvements against multimodal boundary controls. |
| F20 | Q | Reduce the restart population growth factor | Count restarts and first target under 4k/40k calls. |

### E · iAMaLGaM-IDEA, incremental variant

Use incremental=true. This mode draws a smaller initial population and blends moments across generations. Audit first-generation initialization, blended covariance and the aged mean shift independently of the full-update variant. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/benchmarks/candidate-optimizers/cand-eda/src/lib.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| E01 | E | Fuse covariance decay and addition into a single in-place element pass | Compare blended covariance for several selection sizes. |
| E02 | E | Exploit symmetry during the incremental covariance blend | Count written elements and verify lower-triangle factorization. |
| E03 | E | Reuse previous mean storage with a swap | Track copies across many small-population generations. |
| E04 | E | Fuse mean-difference formation with the aged shift update | Check first step and long-memory accumulation. |
| E05 | E | Specialize small selected-set cross-products when selected rows are fewer than D | Run D32/128 and validate against dense multiplication. |
| E06 | E | Compare rank-one Cholesky updates against refactorizing the blended second moment | Reset on unstable downdates, check residual and jitter events. |
| E07 | E | Scale the previous factor once before accumulating low-rank contributions | Compare cost versus E06 on a fresh base, do not count a dependent-only gain. |
| E08 | E | Use a direct small-batch sampling kernel for incremental populations | Benchmark default 10√D and preserve normal draws. |
| E09 | E | Avoid allocating full factor-retry work when the first Cholesky succeeds | Verify regularization path receives the original blended matrix. |
| E10 | E | Precompute effective selected weights at restart construction | Check integer rounding and recomputation when P doubles. |
| E11 | E | Use a flat fast path for selected rows already contiguous after ranking | Replay ties and compare against gathered mean/covariance. |
| E12 | E | Specialize the existing selected-fitness variance stop for its default zero tolerance, retaining a fallback near underflow | Compare stopping generations on equal, almost equal and all-nonfinite fitness. |
| E13 | E | Reuse the accepted-sample mean in the SDR triangular solve | Preserve the scale convention and AVS multiplier sequence. |
| E14 | E | Stream narrow-box redraws through one reusable perturbation row | Verify both redraw limit and clamp distributions. |
| E15 | Q | Scale the covariance learning rate by 0.5 | Compare covariance lag on rotated and separable tasks. |
| E16 | Q | Double the mean-shift response while keeping covariance learning unchanged | Verify late learning after a plateau and large-step stability. |
| E17 | Q | Raise the first incremental population from 10√D to 14√D | Evaluate estimation variance and speed to target. |
| E18 | Q | Warm-start incremental covariance across restarts with bounded regularization | Compare collapse/recovery and equal-budget tails. |
| E19 | Q | Introduce a diagonal-only warm-up for the first few generations | Test high-D rotated holdouts where lost covariance matters. |
| E20 | Q | Trigger a reset from persistently poor covariance conditioning | Measure fewer failed generations and potential loss of learned correlations. |

### B · Bounded Nelder–Mead

This is the bounded local refiner in the boundary experiment. Include cold starts and incumbents supplied by an outer solver. Measure reflection, expansion, contraction, shrink and residual-budget branches. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/benchmarks/optimizer-boundary/src/nelder_mead.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| B01 | E | Reuse centroid scratch | Compare centroid coordinates and bytes allocated on the normal iteration. |
| B02 | E | Replace re-sorting the whole simplex after a single replacement with sorted insertion | Preserve best, second-worst and stable equal losses. |
| B03 | E | Update the centroid from the replaced vertex | Compare accumulated drift over long runs. Reset if precision gate fails. |
| B04 | E | Borrow the worst vertex during proposal construction | Exercise reflection, expansion and outside contraction. |
| B05 | E | Write reflection into a reusable row | Verify box clamping occurs before the callback. |
| B06 | E | Reuse a separate expansion/contracted-candidate row | Check which evaluated row wins on equal values and budget exit. |
| B07 | E | Shrink vertices in place using the fixed anchor | Compare best seen when budget ends halfway through shrinking. |
| B08 | E | Cache per-vertex sorted IDs for an indirect simplex | Compare data movement and branch IDs at D2/16/64. |
| B09 | E | Reuse simplex buffers across deterministic restarts | Check that all vertices receive fresh scores. |
| B10 | E | Avoid redundant clamping for vertices already projected and validated | Include fixed axes and NaN callback outputs. |
| B11 | E | Precompute initial scaled axis steps per restart | Count per-coordinate multiplies, verify shrinking scales. |
| B12 | C | Keep an independently observed best so a partially built simplex cannot discard a win | Budget ends during initialization or replacement. Compare objective trace. |
| B13 | C | Offer an early-exit policy when pending calls would only repeat the incumbent | Keep existing exact-budget mode and verify noisy/deterministic semantics. |
| B14 | Q | Choose an inward first-axis step when the outward step would clamp to the origin | Count duplicate evaluations and compare boundary minima. |
| B15 | Q | Scale initial simplex by individual box widths | Stretched boxes and rotation control. |
| B16 | Q | Compare Gao–Han coefficients with classical coefficients | D2/16 and fixed-budget late convergence. |
| B17 | Q | Rebuild a degenerate simplex when projected vertices coincide | Test corners, fixed dimensions and line-like feasible sets. |
| B18 | Q | Adapt restart shrink scale from the last accepted displacement | Local improvement at equal remaining calls. |
| B19 | Q | Evaluate shrink vertices through an ordered batch for expensive callbacks | Preserve partial-budget and evaluation error order. |
| B20 | Q | Use a rotated initial simplex based on supplied incumbent directions | Validate quality from outer-solver handoffs and random orientation controls. |

### K · jSO → CMA-ES handoff

Benchmark the named sequence and confirm changes through the example runner using core jSO. The timing panel calls the candidate-harness sequence. Attribute changes to the stage boundary and leave standalone CMA-ES fixed. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/benchmarks/candidate-optimizers/harness/src/solvers.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| K01 | E | Replace the benchmark's serial per-evaluation mutex with a serial observer | Verify best of both stages by replay and test remaining(). |
| K02 | E | Avoid a clone when promoting the first-stage best into the second stage | Check the returned x after a zero-length second stage. |
| K03 | E | Build the second stage's scalar sigma input without a length-D temporary where supported | Check the first asked population. |
| K04 | E | Reuse normalized bounds for repeated sequence invocations | Profile short independent starts and confirm encoding. |
| K05 | E | Pass a stage-specific remaining-call counter directly into the stop closure | Compare exact callback IDs on partial CMA tails. |
| K06 | E | Precompute the largest whole second-stage budget at construction | Test below one population and at the cap. |
| K07 | E | Dispatch the core jSO implementation through the benchmark adapter | Compare candidate/core jSO traces, actual count and first-stage time. |
| K08 | E | Factor out best-observation copying from the wrapper when a stage provides equivalent evidence | Validate nonfinite and identical-fitness tie paths. |
| K09 | E | Recycle per-stage reporting scratch across coordinated retries | Peak memory with 1/4 workers. No run sees stale bounds or state. |
| K10 | C | Record first-stage calls, handoff loss and second-stage best as separate measurements | Sum counts and independently re-evaluate the chosen result. |
| K11 | C | Add per-evaluation external-stop propagation through the stage handoff | Stop exactly before phase two or at a partially evaluated population. |
| K12 | Q | Choose the split that leaves an exact multiple of second-stage population | Compare use of the nine baseline spare calls and holdout quality. |
| K13 | Q | Set jSO share to 25% of the budget | Compare early/late target hits on Cassini1 and Tandem. |
| K14 | Q | Set stage share dynamically from jSO's recent success rate | Freeze thresholds before held-out GTOP. |
| K15 | Q | Trigger handoff from normalized population diameter | Include premature collapse cases and fixed-total-call controls. |
| K16 | Q | Estimate per-coordinate second-stage scale from surviving first-stage points | Compare heterogeneous boxes and rotated functions. |
| K17 | Q | Cap stage-two initial scale using the incumbent's distance to the box wall | Track clipped proposals and late-stage improvements. |
| K18 | Q | Start from a robust elite representative instead of the single jSO winner | Re-evaluate noisy/plateau winners fairly and charge additional calls. |
| K19 | Q | Test stage-two population 16 at the sequence level | Compare whole generation utilization against P31. |
| K20 | Q | Toggle second-stage repair_z using the existing JsoCmaZ0 sequence arm | Compare boundary-hit rates, deadline loss and both stages’ incumbent records on GTOP. |

### Z · L-SRTDE → CMA-ES handoff

This research sequence uses L-SRTDE's base-difference form. Track the first stage's newest and top populations, handoff loss and residual-generation utilization. Keep the second-stage numerical kernel fixed. [Implementation](https://github.com/dietmarwo/fcmaes-rust/blob/ccb9f8b1e4153c1ce2069961715cbe4b951e7806/benchmarks/candidate-optimizers/harness/src/solvers.rs)

| Trial | Type | Change to try | Measurement or regression check |
|---|:---:|---|---|
| Z01 | E | Use a sequence-local scalar observer for the benchmark's serial adapter | Count per-call synchronization and match the physical callback trace. |
| Z02 | E | Tie stage-one limit directly to its exact call counter | Compare truncated and full generations just before handoff. |
| Z03 | E | Move the returned top candidate to the second-stage initializer | Check that late newest-population replacement cannot invalidate it. |
| Z04 | E | Reuse the stage-two setup buffers for repeated sequence runs with identical boxes | Measure amortized setup under restarts. |
| Z05 | E | Defer stage-two preparation when fewer than P calls remain | Validate same physical count, seed use and returned best. |
| Z06 | E | Compute the sequence split using overflow-safe integer arithmetic | Test the share calculation at tiny and large synthetic budgets without running them. |
| Z07 | E | Cache stage-boundary dimensions and normalized-box metadata | Compare K/Z initialization time and check nonuniform bounds. |
| Z08 | E | Add a borrowed handoff record containing the top candidate and its objective | Preserve its lifetime and skip extra evaluations for deterministic objectives. |
| Z09 | E | Reuse first-stage allocation pools across retries after fully resetting newest/top/cyclic state | Compare trace of two fresh runs versus two reused runs. |
| Z10 | C | Distinguish front-stage improvements from second-stage improvements in optional results | Reconcile stage minima with the reported global minimum. |
| Z11 | C | Add cancellation while L-SRTDE is still in the donor loop | No new objective starts after cancellation. Quantify polling overhead. |
| Z12 | Q | Compare target-difference and base-difference first stages under identical handoff rules | Equal calls and second-stage seed on sphere, ripple and Cassini1. |
| Z13 | Q | Set first-stage share to 55% for the base-difference form | Confirm the extra exploration benefits multimodal controls. |
| Z14 | Q | Choose handoff when the newest population's acceptance rate collapses | Compare trigger position, late improvement and time to target. |
| Z15 | Q | Choose handoff when top and newest objective distributions converge | Check repeated plateaus against genuine convergence. |
| Z16 | Q | Select the second-stage center from the most recent improving newest row | Compare against top incumbent on rugged functions. |
| Z17 | Q | Estimate initial sigma from the newest-to-top normalized separation | Track clamp fraction and second-stage success. |
| Z18 | Q | Start two smaller local searches from distant top rows under one residual budget | Compare basin coverage and final objective at equal CPU and calls. |
| Z19 | Q | Assign the first stage the second-stage generation remainder while preserving its planned decay schedule | Record down-to-the-call use and compare to full-share rescaling. |
| Z20 | Q | Set a minimum remaining number of full CMA generations before switching | Test budgets around the threshold and report cases that stay wholly in L-SRTDE. |

## Reproducing the audit

The companion archive contains the benchmark drivers, exact Cargo lockfile, configuration schedule, all 477 end-to-end rows and all 63 phase rows. Its README gives the checkout and release-build commands. A fresh extraction and locked release build replayed all 540 measured rows. All 459 deterministic end-to-end scores passed the supplied numerical tolerance. The 18 schedule-sensitive rows passed the evaluation-count checks. The optimizer checkout is clean. Timings reflect one Linux host, the stated test panels and callback costs. The 380 experimental changes and their holdout measurements are future work.
