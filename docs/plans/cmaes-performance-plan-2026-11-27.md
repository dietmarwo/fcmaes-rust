# fcmaes-rust active CMA-ES · performance check and 30 optimization experiments

27 November 2026 · Baseline commit `d7ff66aebc388ac19b698742c7678986e7a48d58` · Comparator `cmaes = 0.2.2` with active (negative) weights.

**Start with the symmetric eigensolver.** On this host, at dimension 100 and population 17, the current public ask/tell loop spent 86.9% of elapsed time inside `symmetric_eigen()`. It called the routine in all 128 generations. On the independent, uninstrumented serial benchmark, fcmaes delivered 28.9% of the comparator’s throughput on Sphere and 28.6% on Ellipsoid. Switching to an eigensolver that does the same work is the first experiment. Changing how often the solver refreshes its basis is the second, with a separate test of convergence per objective evaluation.

**Success means getting a useful solution sooner.** With exactly 2,176 calls to the 100-D problems, fcmaes had the better final value on 14 of 20 Ellipsoid seeds and 5 of 20 Sphere seeds. At eight workers, fcmaes already led on two of the three independent-multistart cases checked. Preserve those strengths while reducing the serial cost. The measurements below are complete. The 30 numbered changes are proposals for a subsequent implementation campaign.

## 1. What was measured

### Current commit, elapsed-time comparison

Both solvers used the repository’s release harness, the same reflected and decoded objective, initial distribution, population and deadline. Arm A measures serial public ask/tell against the external solver’s serial loop. The shared objective includes coordinate reflection, timing and call accounting. A throughput ratio is the **median of paired ratios** of actual evaluations per active wall-second, fcmaes divided by external. The two throughput columns are separate medians. A ratio above 1 favors fcmaes. A 100 µs injection sets a floor on evaluation time.

| Serial problem | Population | fcmaes eval/s, natural cost | External eval/s, natural cost | Paired ratio, natural cost | Paired ratio, 100 µs |
|---|---:|---:|---:|---:|---:|
| Sphere 10 | 10 | 940,132 | 1,436,880 | 0.654 | 0.998 |
| Rosenbrock 40 | 15 | 128,384 | 238,737 | 0.538 | 0.950 |
| Sphere 100 | 17 | 15,933 | 55,217 | 0.289 | 0.712 |
| Ellipsoid 100 | 17 | 14,838 | 50,127 | 0.286 | 0.723 |

Seven paired seeds per cell, 400 ms allocations, root seed 4242, 112 library runs, 4,359,025 evaluations. All 56 fcmaes runs reached the deadline. The external solver stopped early in 13 of 56 runs (seven positive-definiteness stops, five no-effect-axis, one no-effect-coordinate). On cheap Sphere 100 the seven paired ratios ranged from 0.276 to 0.295. At 100 µs a substantial gap persisted on both 100-D cases.

| Eight workers, natural cost | Arm B · one population in parallel | Arm C · eight independent serial starts |
|---|---:|---:|
| Sphere 10 | 1.126 | 1.270 |
| Sphere 100 | 0.385 | 0.311 |
| Cassini1 | 0.979 | 1.311 |

Five paired seeds per cell, 300 ms allocations, 60 library runs. All 30 fcmaes runs reached the deadline. Fifteen external rows ended early, including ten multistart rows where all eight instances stopped. These short runs check where the serial deficit survives parallelism. They provide little evidence about final application quality. Host for both tables was a shared AMD EPYC 9V74 Linux x86-64 environment with Rust 1.100.0, default portable release flags, thin LTO and one codegen unit. Serial timings used one logical CPU and parallel timings eight. The pinned harness alternates which library runs first.

### Current commit, quality at equal evaluation counts

Each run completed 128 generations with the same harness objectives, initial means, population formula and options as Arm A. The two solvers used the same numeric seeds 8100–8119 and their respective RNGs. All 200 runs completed the stated call count. Lower is better. W/T/L is from the fcmaes perspective, with a tie when the difference is at most `1e-10 × max(1, |a|, |b|)`.

| Problem | Calls per run | Median best, fcmaes | Median best, external | W/T/L |
|---|---:|---:|---:|---:|
| Sphere 10 | 1,280 | 7.89e−8 | 8.21e−8 | 10 / 1 / 9 |
| Rosenbrock 40 | 1,920 | 48.45 | 46.52 | 9 / 0 / 11 |
| Sphere 100 | 2,176 | 5.256 | 4.748 | 5 / 0 / 15 |
| Ellipsoid 100 | 2,176 | 2.433e6 | 2.532e6 | 14 / 0 / 6 |
| Cassini1 | 1,152 | 17.679 | 17.085 | 6 / 0 / 14 |

The 400 ms cheap Ellipsoid run tells a different story from the equal-calls row. The median best at equal time was 270,900 for fcmaes and 17,070 for the external solver. Improving generation throughput has a plausible application even where the current update uses evaluations well. Both comparisons are early-budget snapshots.

### Current commit, time inside the Rust implementation

The phase probe used an unbounded, unreflected Sphere in one thread, mean 0.4 in every coordinate, scalar initial sigma 0.3, default update gap, `repair_z=false`, zero accuracy and historical tolerance, and a nominal budget of exactly 128 populations. Seed 0 warmed up the binary. The table reports medians for seeds 1–7. Timers surround `ask_all`, `update_covariance` and only the call to `sym.symmetric_eigen()`. Their scopes do not overlap. The total includes evaluation, update and remaining API work, excludes construction, and comes from an instrumented build. The raw records include the final objective and stop code. All completed without an internal stop.

| Dimension / population | Entry point | µs per generation | Sample (`ask_all`) | Covariance arithmetic | Eigensolve only | Eigensolves / 128 |
|---|---|---:|---:|---:|---:|---:|
| 10 / 10 | direct optimize | 8.69 | 14.4% | 16.2% | 50.5% | 128 |
| 10 / 10 | public ask/tell | 9.96 | 13.0% | 13.7% | 46.3% | 128 |
| 40 / 15 | direct optimize | 113.95 | 7.0% | 6.4% | 81.0% | 128 |
| 40 / 15 | public ask/tell | 114.28 | 7.0% | 6.6% | 78.0% | 128 |
| 100 / 17 | direct optimize | 1,060.24 | 3.8% | 6.7% | 86.0% | 127 |
| 100 / 17 | public ask/tell | 1,081.01 | 3.7% | 5.5% | 86.9% | 128 |
| 100 / 31 | direct optimize | 1,139.01 | 6.4% | 7.4% | 82.5% | 128 |
| 100 / 31 | public ask/tell | 1,160.13 | 6.4% | 7.2% | 81.4% | 128 |

At 100/17 on ask/tell, eliminating **all** work outside the eigensolve would give at most about 1.15× on this trace. Halving the eigensolve cost with everything else held fixed would give about 1.77×. Those are arithmetic bounds under stated assumptions. Differences between direct and public paths also include their different initial iteration counters. Compare each entry point against its own baseline.

### The repository’s longer comparison

I recomputed the following medians from the checked-in 7,920-row campaign on the 16-core Ryzen 9 9950X. These are the 10-second endpoints, 20 pairs per case. The natural-cost column includes Cassini1. The injected-cost columns contain the seven analytic cases.

| Arm | Natural cost | 1 µs floor | 100 µs floor |
|---|---:|---:|---:|
| A · serial | 0.630 | 0.664 | 0.983 |
| B · parallel population, 16 workers | 0.733 | 0.690 | 0.860 |
| C · 16 independent starts | 1.179 | 1.096 | 1.278 |

Cheap serial Sphere 100 scored 0.287 there and 0.289 here. The aggregate serial 0.983 at 100 µs should be read alongside the individual dimensions. The older campaign recorded 2,434 external internal-stop rows out of 3,960 and zero for fcmaes. Its 10-second Arm A final-objective tally was 96 fcmaes wins, 78 external wins, 266 ties. Preserve both active runtime and allocated wall time in every future comparison. [Benchmark protocol and interpretation](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/benchmarks/cmaes-implementation/README.md) · [Publication raw data](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/benchmarks/cmaes-implementation/results/implementation-v1/paired.csv)

## 2. Where to make changes

- **Factorization and refresh policy.** In `update_cma`, the `last_update + lazy_update_gap as i32` check gates covariance adaptation and some stop tests. Inside that block, `update_bd` applies a second, floating-remainder test before averaging the two triangles, decomposing, conditioning the eigenvalues and rebuilding `BD`. Increasing the public `update_gap` also skips covariance updates. For an eigen-only experiment, leave the outer check intact and count each stage independently. [CMA update](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/cmaes.rs#L554-L683)
- **Generation data movement.** `ask_all` allocates the Gaussian matrix and transforms owned columns. The positive and negative covariance terms construct diagonal weight matrices. Public `ask` decodes to an internal matrix and then produces owned rows. `try_tell` validates the batch, clones the encoded matrix and copies the columns back. `tell_x` can additionally form the inverse of `BD`. Each has a different frequency and caller. [Sampling and updates](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/cmaes.rs#L356-L553) · [Public batch API](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/cmaes.rs#L767-L907)
- **Native and Python callers.** Direct `optimize_until` already uses flat candidate storage and `eval_population_scalar_flat`. The unbounded, non-normalized evaluation path can borrow the point. Explicit Rayon pools are cached by worker count. Python `ACMA.ask` currently goes through owned Rust rows and flattens them for NumPy. The default core population is 31. Setting a nonpositive population selects the logarithmic formula used in the comparator harness. Include both. [Fitness evaluation](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/fitness.rs) · [Python ACMA](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-py/src/acma.rs) · [Comparison adapters](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/benchmarks/cmaes-implementation/src/adapters.rs)

## 3. Protocol before iteration 1

1. **Freeze three comparators.** Retain a clean executable from the pinned commit, the current best accepted build, and the locked external crate. Tag the candidate with its parent SHA and experiment number. Use identical compiler, target flags, allocator, affinity and worker limits. Pin a release harness before collecting results. Keep the repository’s portable release as the primary build. Record native-CPU results separately.
2. **Add two modes to the existing diagnostic in the implementation campaign.** A fixed-generation/actual-call mode must exercise both the direct Rust and public ask/tell paths. A sampled phase mode records ask, tell, positive/negative update, eigensolve, decode, callback and setup; count eigensolves, skipped updates, covariance floors, clipped columns, allocations/bytes and peak memory. Instrument in a separate build. Ship decisions use timings from the uninstrumented build. Calibrate the objective wrapper per cell.
3. **Use a small panel every round.** Start with dimensions 10, 40 and 100 at both logarithmic population and 31, unbounded Sphere, rotated ellipsoid with condition number 10⁶, and one clipped box. Use fixed orthogonal rotations independent of solver seeds. Add 1, 2, 6, 20, 64, 128, 256 and populations 8, 64, 128 for their relevant experiments. Disclose dimension/population/mu per cell. Use a cold-construction row for short retries and a warm row for long runs.
4. **Measure useful work.** For fast screening use 128 and 512 completed generations, recording actual calls, stop and the best value. For promotion use matched calls at approximately 100d and 1,000d (round down to whole populations for the single-run pair), wall budgets 0.1, 1 and 10 s, plus first calls and first wall time to prespecified targets. For analytic controls use best minus known optimum at 1e−2 and 1e−6, reporting cases with zero hits. Take GTOP reference targets from the pinned suite and lock them before seeing candidates. Charge the full allocation for unsuccessful target runs. Report hits/attempts, failures, median best at budget and total call/wall cost divided by hits, reported as undefined for zero hits. Independently reevaluate returned best points outside timed and budgeted runs.
5. **Exercise the applications and the cores.** Keep native bounded Cassini1 and one of Cassini2/Rosetta as quality controls; include shifted optima on and beyond a box face. Compare `repair_z` false/true in separate native runs. Test standalone CMA and the existing DE→CMA and jSO→CMA sequences with matching first-stage inputs or replayed handoffs. Measure 1×1, 1×8 population evaluation and 8×1 independent retries; add 16×1 on a host with 16 physical cores. One inner thread per outer lane. Use zero, 1 µs and 100 µs controls plus natural simulator cost. [GTOP retry protocol](https://github.com/dietmarwo/fcmaes-rust/tree/d7ff66aebc388ac19b698742c7678986e7a48d58/benchmarks/gtop-cmaes-retry) · [Coordinated sequence protocol](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/benchmarks/coordinated-jso-cma/PROTOCOL.md)
6. **Run one change at a time.** Alternate baseline/candidate order within each seed block. Start with 7 paired timing blocks, warm first and repeat each cheap cell to at least 200 ms per timing sample. Confirm promising changes with ≥20 new paired seeds, and reserve 40 further seeds plus unseen rotations/application starts for a single final replay. Bootstrap paired seed blocks (10,000 resamples) and publish median paired speedup with a 95% interval. Give each primary cell equal weight in a cross-cell geometric mean. Never multiply speedups from separate branches. Keep the reference binaries, raw rows and rejected candidates’ summaries.

### Decision rules

- **Engineering change.** Predeclare one target entry point and workload for each round. Ordinarily keep a change when it is ≥5% faster end to end there and the paired 95% interval excludes 1.00. For an allocation-focused change, accept ≥25% fewer allocated bytes in the target path if the paired time interval bounds slowdown to 2% and peak memory at eight retries stays within 5%. Require no confirmed slowdown above 5% on a protected cell; rerun ambiguous comparisons and leave unresolved changes off the default. A kernel-only win is a reason to investigate.
- **Numerics and search quality.** For storage-only changes require identical seeded populations, scores, stops and counters on the same compiler/CPU. For reordered arithmetic, alternative eigenbases and policy changes, compare reconstruction residuals and rollout quality. Predeclare a positive scale per problem (e.g. baseline median initial regret, or a GTOP objective scale) and a small floor for resolved optima. Normalize fixed-budget median best deterioration by that scale. For default promotion require the one-sided 95% upper bound ≤0.02, target success-rate loss ≤5 percentage points and time-to-target improvement on the target workload. Apply the success gate at informative, prespecified target/budget pairs. If 20 pairs cannot resolve a margin, extend independent confirmation up to 100 pairs or retain the candidate as experimental. Use the final held-out replay once for the chosen combination.
- **Numerical invariants.** Compare symmetry, positive finite sampling scales, the floor/condition correction and finite `sigma`. For an actual decomposition compare normalized Frobenius reconstruction and orthogonality errors against the pinned solver on the same symmetric input, including repeated eigenvalues, nearly singular matrices and tiny negative eigenvalues. Use residuals before and after the existing diagonal correction against their corresponding reference matrices. Fix tolerance from baseline residuals by dimension and conditioning, and retain an exact full-refresh fallback. Eigenvector signs and bases in repeated eigenspaces can differ.
- **Interface invariants.** Test stable fitness ties, nonfinite external scores and native sanitization, scalar/vector sigma, normalized and raw bounds, `repair_z` default false, zero-norm negative-update fallback, public `ask` replacement semantics, invalid `try_tell*` leaving state intact, `tell_x` with/without pending ask, and Python contiguous/strided input. Test cancellation at the existing generation boundary and actual whole-batch evaluation counts (permitted overshoot up to λ−1). Keep the current Rust 1.88 minimum, safe Rust in the core, and the portable pure-Rust default. [Repository conventions](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/AGENTS.md) · [Regression tests](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/cmaes.rs#L943-L1321)

## 4. The 30 iterations

Run each candidate against its current accepted parent and the frozen original. Where two candidates solve the same problem, retain both branches for a direct comparison. The benchmark setup above is prerequisite work. Each numbered row below tests a distinct implementation change.

### Iterations 1–6 · factorization, refresh and startup

| # | Change to try | Deciding experiment |
|---:|---|---|
| 01 | Replace only the full symmetric eigensolver in `update_bd` with a single-threaded, pure-Rust alternative (evaluate an MSRV-compatible `faer` release). Preserve the existing refresh decisions, averaging, spectral floor and `BD` construction. | Replay captured matrices at d=10, 40, 100, 256, including conversion and workspace costs. Check residuals, actual factorization counts, and whole-generation time on Sphere and rotated Ellipsoid. Record any eigenbasis-induced divergence even when residuals agree. |
| 02 | Add an experimental counter that refreshes the basis on every second or fourth **eligible** refresh. Continue updating covariance, running stopping checks at the original points and tracking factor age. Test each interval independently against interval one. | Fixed-call rotated Ellipsoid, Rosenbrock and boundary panels. Count covariance updates and eigensolves separately. Compare first-hit time, hit rate and stop codes. Roll back any policy that buys time by losing targets. |
| 03 | Make refresh depend on accumulated covariance drift since the last factorization, with a hard maximum age and an immediate full refresh when numerical checks fail. Compare to the best fixed interval from 02 at similar factorization counts. | Run slow-changing Sphere and quickly adapting rotated quadratics, including an anisotropic start. Include drift-estimation overhead and maximum factor age. Choose a threshold on development seeds only. |
| 04 | Warm-start a symmetric Jacobi eigensolve from the previous basis, cap the number of sweeps and fall back to the full solver when the residual exceeds tolerance. Keep baseline refresh dates. | Replay a sequence of real covariance matrices with a clustered spectrum and a sudden change of orientation. Charge basis transformations, residual validation and fallbacks. Keep only if 100/256-D end-to-end times improve with verified factors. |
| 05 | Remove the extra symmetric-matrix clone by symmetrizing `self.c` in place and handing a single owned solver input to the consuming decomposition. Preserve the exact pairwise average and the matrix later used for diagonal correction. | Use original nalgebra first to isolate copying. Measure bytes per refresh, peak memory and generation time on d=40/100/256. Compare seeded state snapshots with bit equality. |
| 06 | Initialize `C` directly as identity and `BD` directly from the initial diagonal scales. The constructor currently multiplies identity by its transpose for `C`. Preserve current initial `C` for unequal input sigmas too. | Time construction separately and runs of 1, 8 and 128 generations at d=10/100/256. Test scalar and vector sigma, first populations, first update and eight concurrent cold starts. |

### Iterations 7–13 · active covariance and recombination

| # | Change to try | Deciding experiment |
|---:|---|---|
| 07 | Replace multiplication by the dense diagonal weight matrix with explicit column weighting before the positive and negative products. Include the positive-only guard path. | Compare fixed `arpos`, `artmp`, weights and resulting `C` over λ=10/31/64/128. Check zero-norm fallback and both whole-generation and kernel times. |
| 08 | Accumulate the scaled old covariance and rank-one `pc pcᵀ` term directly into the output matrix. Keep the rank-μ products as in the parent. | Record full d×d allocations avoided, cache footprint and per-generation time. Exercise both `hsig` branches and the positive-only path. |
| 09 | Compute both weighted rank-μ products using a symmetric, single-triangle accumulation and mirror once before code needing a full matrix. | Compare the full updated covariance and post-floor factor against the reference over 100 consecutive updates. Benchmark d=40/100/256, small and large μ, and early exit between refreshes. |
| 10 | Replace reversed-index, ratio, inverse-permutation and reordered-ratio allocations for worst-sample norms with one sorted permutation and a direct scatter into rank order. | Use hand-computed tied, zero, extreme and repeated norms. Verify worst-column association, weights, guard and reduction order. Measure allocation count and end-to-end time on small d and large μ. |
| 11 | Retain the sampled, pre-clipping displacement `BD z` so the negative update can reuse valid columns instead of computing `BD × arzneg` again. Invalidate cached columns on repair or external tell. | Cross 0%, sparse and frequent clipping, `repair_z` on/off and `tell_x`. Verify cache-hit counts, negative steps and memory per retry lane. Reject where the extra population buffer costs more than the saved product. |
| 12 | Gather the weighted positive-parent means directly from indexed `arx` and `arz`. Remove the temporary selected-`z` matrix and retain only positive-point storage that covariance really needs. | Test identical points and tied ranks, odd/even λ and custom μ. Compare means, paths and finalist quality; time small populations and λ=64. |
| 13 | Allocate positive/negative work matrices once per optimizer and refill their active columns for each update. Keep the arithmetic and chosen update kernel fixed. | Count allocation calls and bytes/generation and measure peak resident memory at eight simultaneous starts. Include the first generation and a long run; check reuse following a skipped negative update. |

### Iterations 14–20 · sampling and generation ownership

| # | Change to try | Deciding experiment |
|---:|---|---|
| 14 | Fill the existing `arz` buffer in column-major order, drawing the same Gaussian stream, in place. | Hash the initial Gaussian matrix and subsequent asked populations for multiple seeds and run IDs. Measure allocations and small-d ask time. Include fallback sampling after an uninvertible external update. |
| 15 | Transform all sampled columns in one matrix product `BD × Z`, using the same Gaussian draws, in place of λ matrix-vector products. | Sweep d=6–256 and λ=8–128. Time the transformation and whole ask, check numerical residuals and record where batched multiplication loses to the scalar path. Use that evidence in 30. |
| 16 | Write the affine sample and its projection to the box directly into its encoded column, detecting whether a coordinate moved during the same pass. Skip flag tracking when `repair_z=false` in the production path. | Validate unbounded, raw bounded and normalized samples, exact endpoints and both repair settings. Compare all flags that are consumed, RNG order and the population. Time cheap bounded problems independently of the reflected harness. |
| 17 | Transfer an already sampled encoded batch into `arx` once at generation completion. Factor batch finalization out of repeated `tell_one` copies for internal optimize and ordinary public tell, preserving path-specific counters. | Measure saved full-matrix and column copies. Replay double ask, invalid tell/retry and early stop. Compare direct and public paths independently for evaluation count, best point, iterations and stops. |
| 18 | Reuse a decode/clamp buffer for normalized native objective calls, once per serial lane or parallel work chunk. Keep the existing borrowing fast path for unbounded/raw feasible points. | Test raw and normalized boxes with one and eight evaluation workers. Record allocations per call, values delivered to callbacks and native end-to-end time. Check per-worker scratch isolation. |
| 19 | Construct each owned public `ask()` and `population()` row directly from the stored contiguous column slice. Eliminate the intermediate owned `DVector` while preserving the independently owned rows. | Measure allocations in a Rust client repeatedly calling `ask`, `population`, `tell`. Mutate returned rows and verify internal snapshots remain independent. Direct-optimize timing acts as a control. |
| 20 | Add a flat decoded-population interface that lets a native caller read contiguous candidates with one borrow, alongside the existing owned API. Retain the same pending-batch lifecycle. | Test a realistic single-thread consumer and a borrowed-slice parallel evaluator. Compare new API vs owned API at identical scores and state, including replacement of an untold batch. Report both paths explicitly in the comparison harness. |

### Iterations 21–26 · external populations, small dimensions and Python

| # | Change to try | Deciding experiment |
|---:|---|---|
| 21 | In `tell_x`, recover external coordinates via `D⁻¹ Bᵀ ((X−m)/σ)` instead of materializing and inverting `BD`. Retain a checked fallback for tiny/invalid scales or loss of orthogonality. | Test supplied points different from the last ask, with/without pending state and unequal scales. Measure forward-reconstruction error, fallback behavior, RNG consumption on failure and time at d=10/100/256. |
| 22 | For `repair_z=true`, pack the actually repaired columns and batch their inverse transforms when many columns moved. Keep a sparse per-column path for a few repairs. | Benchmark exactly 0, 1, half and all repaired candidates; measure packing as part of the total cost. Verify repaired coordinates, untouched columns and the zero-norm guard. Protect the default false path. |
| 23 | Specialize the fitness-index sort for small populations, comparing stable insertion sorting on reusable index storage against the current stable sort; retain the large-λ path. | Sweep λ=8/10/17/31/64/128 with continuous, tied, all-equal and sanitized nonfinite scores. Require identical rank order by original index and measure the entire update on small d. |
| 24 | Replace the fitness-history shift with a circular buffer and compute its extrema in one pass. Preserve initial sentinels and the point in the generation when each stop reads history. | Replay a plateau, monotone improvement, oscillating values and stops 4/5. Check the exact stopping generation and sigma adjustments. Prioritize small λ where history can be longer. |
| 25 | Reuse evolution-path vectors and update `ps` and `pc` in place, preserving the order of mean replacement, norm evaluation and `hsig` calculation. | Measure per-generation allocations and time at d=6/10/40. Compare every path component against the reference, including the `hsig=false` branch and threshold-adjacent cases. |
| 26 | Have Python `ACMA.ask` construct one independently owned C-contiguous NumPy population from the decoded flat storage, bypassing intermediate Rust row vectors. If 20 was rejected, use an internal copying accessor. | Benchmark Python ask/tell with a vectorized Sphere and an externally repaired NumPy batch. Check shape, dtype, row order and that an old NumPy result remains valid after the next ask or optimizer destruction. |

### Iterations 27–30 · eight-worker execution and final choice

| # | Change to try | Deciding experiment |
|---:|---|---|
| 27 | Resolve the explicit cached Rayon pool once for a direct `optimize_until` invocation, instead of taking the process-wide pool-map lock for every generation. | Compare 1×8 cheap population evaluation and multiple optimizer invocations requesting pools concurrently. Check the existing `workers=1`, `workers<=0` and positive-worker behavior, cancellation latency and actual concurrency. |
| 28 | Group contiguous candidates into explicit parallel chunks with ordered score collection. Test grain sizes 1, 2 and 4 in the same pool, keeping worker limits and the candidate-to-score mapping. | Use λ=10/31/128, 8 workers and cheap, 1 µs, 100 µs and variable-latency objectives. Record wall time, worker utilization and straggler tails. Protect expensive/skewed populations against load imbalance. |
| 29 | Reuse CMA-owned scratch storage across serial restarts within an outer retry lane. Reset distribution state, counters, pending batches, history, RNG and repair flags exactly as fresh construction would. | Compare fresh/reused paired restart sequences, including changed means, sigma, seeds and IPOP/BIPOP population sizes. Measure 8×1 starts/s, allocations, retained memory, best-at-deadline and target success on bounded Cassini1. |
| 30 | Select per-column or batched sampling multiplication using a measured crossover in d and λ. Compare a single documented threshold against always-scalar and always-batched on the accepted build. | Fit on the development grid from 15. Test held-back dimensions/populations and both CPU families when available. Keep a single kernel if dispatch cannot earn its complexity. Then perform the independent release replay below. |

## 5. Decisions and deliverables for the implementation campaign

- **Order of investment.** Run 01 and 02 on separate branches from the frozen solver to distinguish eigensolver speed from fewer factorizations. After 06, 13, 20 and 26, reprofile the accepted combination and repeat cheap serial and eight-worker controls. Reorder remaining independent rows if the dominant cost moves. Log the substitution or order change. Keep all 30 trial records, including rejected changes and the workload that defeated them.
- **Performance targets.** Aim to raise the cheap serial paired ratio to the external crate from about 0.29 to at least 0.50 on **each** 100-D control. On the same machine that would require roughly 1.7× the current throughput. A second target is at least 15% less elapsed time to an attained, predeclared target on a bounded or rotated workload. Protect 8×1 throughput and target success on Cassini1 and Sphere 10. These targets guide investment; release claims use the measurements actually obtained.
- **Acceptance record per iteration.** Save `id, parent SHA, candidate SHA, hypothesis, changed functions, build flags, workload, seed, actual evaluations, generations, phase counts, time, allocated time, allocation bytes, peak memory, best, target hit, stop, decision, reason`. Quote the median paired speedup and its 95% paired-bootstrap interval, resampling whole seed blocks; record cold/warm and thread counts. Keep raw rows for both parent and candidate and use the original commit to measure cumulative gain. Never multiply individual improvements from overlapping kernels.
- **Final verification.** Rebuild without instrumentation and rerun the accepted changes against the frozen baseline and external solver on fresh seeds, interleaved on the same host. Repeat the publication preset, population-31 native direct/ask-tell runs, sequence handoffs and an eight-worker retry application. For a possible policy change report its best-operating region and its worst regression; leave it opt-in if confirmation is inconclusive. Ablate each retained high-impact change from the final combination to check that it still pays for itself. Run workspace fmt/clippy/tests with the lockfile, Python binding contracts when touched, and the Rust 1.88 build. Publish the exact benchmark commands, paired results, decisions and patch with defaults explicitly identified.

## 6. Reproduce the uninstrumented wall-time checks

Run from a clean checkout on Linux. Pick CPU IDs within the local affinity mask; the original check used CPU 1 and CPUs 0–7. Use fresh output directories. The longer publication dataset is already checked into the repository.

```bash
git checkout --detach d7ff66aebc388ac19b698742c7678986e7a48d58
cd benchmarks/cmaes-implementation
cargo build --release --locked
bench=target/release/cmaes-implementation-comparison
"$bench" --mode verify
taskset -c 1 "$bench" --mode campaign --preset pilot \
  --problems sphere10,rosenbrock40,sphere100,ellipsoid100 --arms a \
  --cost-ns 0,100000 --deadlines-ms 400 --seeds 7 --seed 4242 --workers 1 \
  --output results/nov27-serial
taskset -c 0-7 "$bench" --mode campaign --preset pilot \
  --problems sphere10,sphere100,cassini1 --arms b,c \
  --cost-ns 0 --deadlines-ms 300 --seeds 5 --seed 4242 --workers 8 \
  --output results/nov27-eight
```

The equal-call probe reused the pinned harness’s `fcmaes_optimizer`, `cmaes_options` and `SharedObjective`; it replaced the Arm A deadline with 128 public ask/tell or external `next()` calls, retained all stop checks and read the harness’s actual call count and best value. The phase probe timed calls in a separate scratch copy of the same core. Neither probe changed solver equations. Full baseline rows from this check are embedded below so the comparisons can be recalculated without access to this host.

<details>
<summary>Uninstrumented serial comparison · 112 raw rows (CSV)</summary>

```csv
preset,problem,dimension,optimum,injected_cost_ns,calibration_ns_per_eval,arm,library,seed,workers,population,deadline_ms,wall_seconds,allocated_seconds,cpu_seconds,active_cores,allocated_cores,evaluations,best,optimizer_runs,termination
pilot,sphere10,10,0.0,0,98.69655,a,fcmaes,4242,1,10,400,0.400010001,0.400010001,0.399865426,0.9996385715366152,0.9996385715366152,382470,0.0,1,deadline
pilot,sphere10,10,0.0,0,98.69655,a,cmaes,4242,1,10,400,0.028501332,0.4,0.028499311,0.9999290910333594,0.0712482775,40240,0.0,1,internal-stop-[PosDefCov]
pilot,sphere10,10,0.0,0,98.69655,a,cmaes,4243,1,10,400,0.023281196,0.4,0.023280091,0.9999525368026625,0.05820022749999999,33760,0.0,1,internal-stop-[PosDefCov]
pilot,sphere10,10,0.0,0,98.69655,a,fcmaes,4243,1,10,400,0.400090663,0.400090663,0.395069113,0.987448969785131,0.987448969785131,363320,0.0,1,deadline
pilot,sphere10,10,0.0,0,98.69655,a,fcmaes,4244,1,10,400,0.400009991,0.400009991,0.399962844,0.9998821354439619,0.9998821354439619,377640,0.0,1,deadline
pilot,sphere10,10,0.0,0,98.69655,a,cmaes,4244,1,10,400,0.038511566,0.4,0.038509625,0.9999495995566631,0.0962740625,53020,0.0,1,internal-stop-[PosDefCov]
pilot,sphere10,10,0.0,0,98.69655,a,cmaes,4245,1,10,400,0.030023373,0.4,0.029973144,0.9983270034316265,0.07493285999999999,43140,0.0,1,internal-stop-[PosDefCov]
pilot,sphere10,10,0.0,0,98.69655,a,fcmaes,4245,1,10,400,0.400007237,0.400007237,0.399988612,0.9999534383424168,0.9999534383424168,376060,0.0,1,deadline
pilot,sphere10,10,0.0,0,98.69655,a,fcmaes,4246,1,10,400,0.400013577,0.400013577,0.39999461,0.9999525841094138,0.9999525841094138,375210,0.0,1,deadline
pilot,sphere10,10,0.0,0,98.69655,a,cmaes,4246,1,10,400,0.034310955,0.4,0.034309421,0.9999552912473583,0.08577355249999999,49720,0.0,1,internal-stop-[PosDefCov]
pilot,sphere10,10,0.0,0,98.69655,a,cmaes,4247,1,10,400,0.023940548,0.4,0.023938781,0.9999261921656931,0.059846952499999995,35040,0.0,1,internal-stop-[PosDefCov]
pilot,sphere10,10,0.0,0,98.69655,a,fcmaes,4247,1,10,400,0.400012344,0.400012344,0.399933449,0.9998027685865614,0.9998027685865614,382290,0.0,1,deadline
pilot,sphere10,10,0.0,0,98.69655,a,fcmaes,4248,1,10,400,0.400259401,0.400259401,0.382729998,0.9562048937359,0.9562048937359,352250,0.0,1,deadline
pilot,sphere10,10,0.0,0,98.69655,a,cmaes,4248,1,10,400,0.033053021,0.4,0.031645424,0.9574139683026249,0.07911355999999999,42900,7.888609052210118e-31,1,internal-stop-[PosDefCov]
pilot,sphere10,10,0.0,100000,101474.61,a,fcmaes,104245,1,10,400,0.40097658,0.40097658,0.400665378,0.9992238898341643,0.9992238898341643,3840,8.405857259154483e-25,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,cmaes,104245,1,10,400,0.400340734,0.400340734,0.400302729,0.9999050683660885,0.9999050683660885,3840,4.252565335455558e-24,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,cmaes,104246,1,10,400,0.400244068,0.400244068,0.400197322,0.9998832062640339,0.9998832062640339,3870,6.13822294455513e-24,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,fcmaes,104246,1,10,400,0.401006055,0.401006055,0.400951936,0.9998650419380826,0.9998650419380826,3840,2.864968469502661e-24,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,fcmaes,104247,1,10,400,0.401089636,0.401089636,0.383062694,0.95505507900982,0.95505507900982,3510,1.114374023682604e-23,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,cmaes,104247,1,10,400,0.400751041,0.400751041,0.398687304,0.9948503265397632,0.9948503265397632,3670,1.874557547302207e-24,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,cmaes,104248,1,10,400,0.400531039,0.400531039,0.400516797,0.9999644422064379,0.9999644422064379,3850,2.5561450445545583e-23,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,fcmaes,104248,1,10,400,0.400490679,0.400490679,0.400462086,0.9999286050799698,0.9999286050799698,3870,3.3401790676687038e-25,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,fcmaes,104249,1,10,400,0.400160202,0.400160202,0.400059839,0.9997491929494776,0.9997491929494776,3880,1.089269393120941e-24,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,cmaes,104249,1,10,400,0.400579101,0.400579101,0.392472386,0.979762511374751,0.979762511374751,3800,4.922884506879461e-23,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,cmaes,104250,1,10,400,0.401003911,0.401003911,0.400904185,0.999751309158678,0.999751309158678,3920,6.140640014368727e-24,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,fcmaes,104250,1,10,400,0.400904071,0.400904071,0.400889883,0.9999646099877095,0.9999646099877095,3890,3.783702700993829e-24,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,fcmaes,104251,1,10,400,0.40015254,0.40015254,0.400131995,0.9999486570796228,0.9999486570796228,3880,4.245018303175309e-26,1,deadline
pilot,sphere10,10,0.0,100000,101474.61,a,cmaes,104251,1,10,400,0.400346121,0.400346121,0.400260497,0.9997861250665147,0.9997861250665147,3820,2.1475178724404606e-22,1,deadline
pilot,rosenbrock40,40,0.0,0,222.95655,a,fcmaes,1004245,1,15,400,0.400114543,0.400114543,0.395038934,0.9873146100565507,0.9873146100565507,51180,0.7492274374684761,1,deadline
pilot,rosenbrock40,40,0.0,0,222.95655,a,cmaes,1004245,1,15,400,0.285862805,0.4,0.285791962,0.9997521783220451,0.714479905,69405,1.6784435708206424e-25,1,internal-stop-[NoEffectAxis]
pilot,rosenbrock40,40,0.0,0,222.95655,a,cmaes,1004246,1,15,400,0.305423842,0.4,0.305406263,0.9999424439170012,0.7635156575,73560,1.4274359193883687e-25,1,internal-stop-[NoEffectAxis]
pilot,rosenbrock40,40,0.0,0,222.95655,a,fcmaes,1004246,1,15,400,0.400028273,0.400028273,0.400010164,0.9999547306997474,0.9999547306997474,51240,0.4310351733415433,1,deadline
pilot,rosenbrock40,40,0.0,0,222.95655,a,fcmaes,1004247,1,15,400,0.400061643,0.400061643,0.399937646,0.9996900552648081,0.9996900552648081,51930,4.465742332345664,1,deadline
pilot,rosenbrock40,40,0.0,0,222.95655,a,cmaes,1004247,1,15,400,0.29731534,0.4,0.297298611,0.9999437331420572,0.7432465275,69825,1.0094816345841722e-25,1,internal-stop-[NoEffectCoord]
pilot,rosenbrock40,40,0.0,0,222.95655,a,cmaes,1004248,1,15,400,0.293104288,0.4,0.293098514,0.999980300527026,0.7327462849999999,69975,1.3237795964423277e-25,1,internal-stop-[NoEffectAxis]
pilot,rosenbrock40,40,0.0,0,222.95655,a,fcmaes,1004248,1,15,400,0.40004954,0.40004954,0.394793824,0.9868623370995503,0.9868623370995503,51360,0.5363582185349499,1,deadline
pilot,rosenbrock40,40,0.0,0,222.95655,a,fcmaes,1004249,1,15,400,0.400061293,0.400061293,0.400027169,0.9999147030702619,0.9999147030702619,53940,0.0050125773547760755,1,deadline
pilot,rosenbrock40,40,0.0,0,222.95655,a,cmaes,1004249,1,15,400,0.302271894,0.4,0.302265862,0.999980044456267,0.7556646549999999,72405,1.64524041531989e-25,1,internal-stop-[NoEffectAxis]
pilot,rosenbrock40,40,0.0,0,222.95655,a,cmaes,1004250,1,15,400,0.400072389,0.400072389,0.39998175,0.9997734435004962,0.9997734435004962,95445,3.9866238543009307,1,deadline
pilot,rosenbrock40,40,0.0,0,222.95655,a,fcmaes,1004250,1,15,400,0.400023626,0.400023626,0.400001892,0.9999456682091072,0.9999456682091072,51780,0.755390701995734,1,deadline
pilot,rosenbrock40,40,0.0,0,222.95655,a,fcmaes,1004251,1,15,400,0.400067221,0.400067221,0.394797216,0.9868272012217668,0.9868272012217668,50460,2.327472621561874,1,deadline
pilot,rosenbrock40,40,0.0,0,222.95655,a,cmaes,1004251,1,15,400,0.300765943,0.4,0.300745664,0.9999325754778028,0.75186416,71715,1.8202886501884325e-25,1,internal-stop-[NoEffectAxis]
pilot,rosenbrock40,40,0.0,100000,104452.94,a,fcmaes,1104248,1,15,400,0.40100338,0.40100338,0.400976663,0.9999333746264183,0.9999333746264183,3600,37.37397021096997,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,cmaes,1104248,1,15,400,0.400088152,0.400088152,0.4000142,0.9998151607348773,0.9998151607348773,3780,38.31121305093992,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,cmaes,1104249,1,15,400,0.400993866,0.400993866,0.400937144,0.9998585464646486,0.9998585464646486,3840,38.68060119965605,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,fcmaes,1104249,1,15,400,0.400480308,0.400480308,0.395237004,0.9869074611279014,0.9869074611279014,3645,37.27576092023103,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,fcmaes,1104250,1,15,400,0.400383317,0.400383317,0.400317441,0.9998354676700977,0.9998354676700977,3600,39.300063418426554,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,cmaes,1104250,1,15,400,0.400436007,0.400436007,0.400420574,0.9999614595098087,0.9999614595098087,3810,38.46983353738425,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,cmaes,1104251,1,15,400,0.400947977,0.400947977,0.400665045,0.9992943423680125,0.9992943423680125,3780,36.41811592109987,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,fcmaes,1104251,1,15,400,0.401215034,0.401215034,0.390902462,0.9742966461221889,0.9742966461221889,3555,38.70340151949091,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,fcmaes,1104252,1,15,400,0.400706884,0.400706884,0.394237319,0.9838546197773832,0.9838546197773832,3600,39.30655691577084,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,cmaes,1104252,1,15,400,0.400868387,0.400868387,0.40066689,0.9994973487395503,0.9994973487395503,3660,37.19567614889096,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,cmaes,1104253,1,15,400,0.40046474,0.40046474,0.399281878,0.9970462767833193,0.9970462767833193,3735,37.22388028258989,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,fcmaes,1104253,1,15,400,0.40058118,0.40058118,0.400564633,0.9999586925177064,0.9999586925177064,3660,38.07276874057577,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,fcmaes,1104254,1,15,400,0.401613292,0.401613292,0.401595714,0.999956231528313,0.999956231528313,3645,38.043131001781525,1,deadline
pilot,rosenbrock40,40,0.0,100000,104452.94,a,cmaes,1104254,1,15,400,0.401271227,0.401271227,0.395284146,0.9850797151723016,0.9850797151723016,3615,39.113200995744435,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,fcmaes,2004248,1,17,400,0.400461959,0.400461959,0.400426072,0.9999103859949905,0.9999103859949905,6086,0.0009114345915119971,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,cmaes,2004248,1,17,400,0.400093364,0.400093364,0.400074228,0.9999521711637286,0.9999521711637286,22049,1.290262014579983e-18,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,cmaes,2004249,1,17,400,0.400221116,0.400221116,0.400201799,0.9999517341808623,0.9999517341808623,22134,9.486534480388282e-19,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,fcmaes,2004249,1,17,400,0.400207165,0.400207165,0.400183882,0.9999418226307868,0.9999418226307868,6358,0.0007322723641209291,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,fcmaes,2004250,1,17,400,0.400272594,0.400272594,0.394917634,0.9866217170991228,0.9866217170991228,6154,0.000777286655565366,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,cmaes,2004250,1,17,400,0.400621187,0.400621187,0.400586241,0.9999127704646336,0.9999127704646336,22253,9.33561959686631e-19,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,cmaes,2004251,1,17,400,0.40084785,0.40084785,0.40084005,0.999980541245263,0.999980541245263,22134,1.4238670605409974e-18,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,fcmaes,2004251,1,17,400,0.40005776,0.40005776,0.40004086,0.9999577561000192,0.9999577561000192,6409,0.0005338090418231443,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,fcmaes,2004252,1,17,400,0.40009758,0.40009758,0.400050694,0.9998828135876252,0.9998828135876252,6375,0.0007387283276532507,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,cmaes,2004252,1,17,400,0.400117641,0.400117641,0.400109882,0.9999806082031758,0.9999806082031758,22066,1.948856505106544e-18,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,cmaes,2004253,1,17,400,0.400083749,0.400083749,0.394726154,0.9866088162456206,0.9866088162456206,21743,2.452467074641289e-18,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,fcmaes,2004253,1,17,400,0.400616511,0.400616511,0.400595574,0.9999477380501673,0.9999477380501673,6426,0.0007423060880512255,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,fcmaes,2004254,1,17,400,0.401025716,0.401025716,0.401018034,0.9999808441212285,0.9999808441212285,6596,0.0009135174045662574,1,deadline
pilot,sphere100,100,0.0,0,461.6162,a,cmaes,2004254,1,17,400,0.400838786,0.400838786,0.400788952,0.999875675703698,0.999875675703698,22610,3.27839818605519e-19,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,fcmaes,2104251,1,17,400,0.400768059,0.400768059,0.400759058,0.9999775406253122,0.9999775406253122,2465,3.1896393798443006,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,cmaes,2104251,1,17,400,0.400648359,0.400648359,0.395386259,0.9868660388048663,0.9868660388048663,3230,0.3871441239169704,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,cmaes,2104252,1,17,400,0.400530923,0.400530923,0.400513663,0.9999569071974002,0.9999569071974002,3332,0.3832648348725484,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,fcmaes,2104252,1,17,400,0.401999492,0.401999492,0.40195644,0.9998929053373032,0.9998929053373032,2380,2.8362064821004407,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,fcmaes,2104253,1,17,400,0.402621179,0.402621179,0.402556714,0.9998398867139574,0.9998398867139574,2431,3.9246805634174438,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,cmaes,2104253,1,17,400,0.401070646,0.401070646,0.401062671,0.9999801157225553,0.9999801157225553,3366,0.29424471626035065,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,cmaes,2104254,1,17,400,0.400710675,0.400710675,0.395628643,0.9873174529228601,0.9873174529228601,3264,0.31853656633716365,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,fcmaes,2104254,1,17,400,0.402585326,0.402585326,0.402565758,0.9999513941548878,0.9999513941548878,2278,4.028286054019921,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,fcmaes,2104255,1,17,400,0.40272208,0.40272208,0.40271446,0.9999810787628034,0.9999810787628034,2397,2.714491372074743,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,cmaes,2104255,1,17,400,0.401989198,0.401989198,0.401942934,0.9998849123304054,0.9998849123304054,3332,0.4467227124449842,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,cmaes,2104256,1,17,400,0.401902578,0.401902578,0.4018617,0.9998982887838057,0.9998982887838057,3332,0.3962859039417309,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,fcmaes,2104256,1,17,400,0.401405791,0.401405791,0.345348415,0.8603473660398686,0.8603473660398686,2040,7.201956825067942,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,fcmaes,2104257,1,17,400,0.401699222,0.401699222,0.401671236,0.9999303309579225,0.9999303309579225,2227,3.4854449336886164,1,deadline
pilot,sphere100,100,0.0,100000,100337.445,a,cmaes,2104257,1,17,400,0.401570368,0.401570368,0.401562685,0.9999808676122237,0.9999808676122237,3264,0.5622599449805655,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,fcmaes,3004251,1,17,400,0.402373326,0.402373326,0.40231157,0.9998465206413808,0.9998465206413808,6035,240439.3417990182,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,cmaes,3004251,1,17,400,0.401027821,0.401027821,0.401020177,0.9999809389782959,0.9999809389782959,19635,17071.64319996735,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,cmaes,3004252,1,17,400,0.400785523,0.400785523,0.395468927,0.9867345607690526,0.9867345607690526,19754,13958.609725313792,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,fcmaes,3004252,1,17,400,0.400059765,0.400059765,0.400032637,0.999932190131642,0.999932190131642,5644,270926.2207093575,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,fcmaes,3004253,1,17,400,0.400972579,0.400972579,0.400964655,0.9999802380501436,0.9999802380501436,5950,265234.22270883474,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,cmaes,3004253,1,17,400,0.400178973,0.400178973,0.400130653,0.9998792540256732,0.9998792540256732,19431,18399.835610044065,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,cmaes,3004254,1,17,400,0.400482631,0.400482631,0.400247617,0.9994131730521917,0.9994131730521917,21029,16536.76645971317,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,fcmaes,3004254,1,17,400,0.400062484,0.400062484,0.394297795,0.9855905284035581,0.9855905284035581,5729,279163.64010149613,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,fcmaes,3004255,1,17,400,0.400792417,0.400792417,0.400775433,0.9999576239487585,0.9999576239487585,6409,175814.4942827279,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,cmaes,3004255,1,17,400,0.400267968,0.400267968,0.400259604,0.9999791039986493,0.9999791039986493,21539,16655.469079226885,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,cmaes,3004256,1,17,400,0.400643431,0.400643431,0.400619871,0.9999411945930545,0.9999411945930545,21420,18502.22836080725,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,fcmaes,3004256,1,17,400,0.400679134,0.400679134,0.400647082,0.9999200058169239,0.9999200058169239,6052,319829.2888722634,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,fcmaes,3004257,1,17,400,0.400799726,0.400799726,0.395732046,0.9873560791805531,0.9873560791805531,5474,295182.1248168979,1,deadline
pilot,ellipsoid100,100,0.0,0,1418.01045,a,cmaes,3004257,1,17,400,0.401196794,0.401196794,0.401166516,0.9999245308027062,0.9999245308027062,20111,19998.133073741763,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,fcmaes,3104254,1,17,400,0.400225242,0.400225242,0.400206776,0.9999538609811123,0.9999538609811123,2397,2185455.173963129,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,cmaes,3104254,1,17,400,0.402491822,0.402491822,0.40246675,0.9999377080511216,0.9999377080511216,3332,1034484.827364345,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,cmaes,3104255,1,17,400,0.400256799,0.400256799,0.400238143,0.9999533899235525,0.9999533899235525,3264,1002420.5526459867,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,fcmaes,3104255,1,17,400,0.400438046,0.400438046,0.395283601,0.9871279838379793,0.9871279838379793,2278,3521548.340853012,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,fcmaes,3104256,1,17,400,0.400332008,0.400332008,0.400290948,0.9998974351308926,0.9998974351308926,2397,1510022.9086688892,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,cmaes,3104256,1,17,400,0.401373009,0.401373009,0.401365382,0.9999809977257339,0.9999809977257339,3281,2651308.7270825994,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,cmaes,3104257,1,17,400,0.401513425,0.401513425,0.401468781,0.9998888106916973,0.9998888106916973,3281,1133897.8246394661,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,fcmaes,3104257,1,17,400,0.40112102,0.40112102,0.401088529,0.9999189995079292,0.9999189995079292,2448,1257952.8653809999,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,fcmaes,3104258,1,17,400,0.400931363,0.400931363,0.395567536,0.9866215829066982,0.9866215829066982,2397,1602493.0698161968,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,cmaes,3104258,1,17,400,0.400679195,0.400679195,0.400650418,0.9999281794503955,0.9999281794503955,3366,1219574.0745636262,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,cmaes,3104259,1,17,400,0.401640941,0.401640941,0.40163323,0.9999808012599991,0.9999808012599991,3349,1623949.5504098462,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,fcmaes,3104259,1,17,400,0.40053657,0.40053657,0.400511369,0.9999370818999125,0.9999370818999125,2329,3511402.4739527474,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,fcmaes,3104260,1,17,400,0.401670077,0.401670077,0.401638738,0.9999219782558012,0.9999219782558012,2431,2064780.3199431838,1,deadline
pilot,ellipsoid100,100,0.0,100000,101061.185,a,cmaes,3104260,1,17,400,0.400738795,0.400738795,0.395911022,0.9879528184936525,0.9879528184936525,3247,902224.5076854394,1,deadline
```

</details>

<details>
<summary>Uninstrumented eight-worker comparison · 60 raw rows (CSV)</summary>

```csv
preset,problem,dimension,optimum,injected_cost_ns,calibration_ns_per_eval,arm,library,seed,workers,population,deadline_ms,wall_seconds,allocated_seconds,cpu_seconds,active_cores,allocated_cores,evaluations,best,optimizer_runs,termination
pilot,sphere10,10,0.0,0,103.21285,b,cmaes,4242,8,10,300,0.300256672,0.300256672,0.878843681,2.9269746951701374,2.9269746951701374,14900,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,b,fcmaes,4242,8,10,300,0.300170787,0.300170787,0.923031055,3.075019605422163,3.075019605422163,16780,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,c,fcmaes,4242,8,10,300,0.300037792,0.300037792,2.395642524,7.984469249793705,7.984469249793705,1630960,0.0,8,deadline
pilot,sphere10,10,0.0,0,103.21285,c,cmaes,4242,8,10,300,0.066330484,0.3,0.443424831,6.685083603490667,1.47808277,367260,0.0,8,8-of-8-internal-stops
pilot,sphere10,10,0.0,0,103.21285,b,fcmaes,4243,8,10,300,0.300054197,0.300054197,0.818296726,2.7271630731430827,2.7271630731430827,13680,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,b,cmaes,4243,8,10,300,0.300206456,0.300206456,0.918435618,3.059346658420963,3.059346658420963,15670,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,c,cmaes,4243,8,10,300,0.072319238,0.3,0.371209102,5.13292330320184,1.2373636733333335,307880,0.0,8,8-of-8-internal-stops
pilot,sphere10,10,0.0,0,103.21285,c,fcmaes,4243,8,10,300,0.303691206,0.303691206,2.394754339,7.885491221632542,7.885491221632542,1641510,0.0,8,deadline
pilot,sphere10,10,0.0,0,103.21285,b,cmaes,4244,8,10,300,0.31417864,0.31417864,0.927932373,2.9535183327548937,2.9535183327548937,14920,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,b,fcmaes,4244,8,10,300,0.300122905,0.300122905,0.943417099,3.143435850056163,3.143435850056163,17420,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,c,fcmaes,4244,8,10,300,0.300057086,0.300057086,2.3983124670000002,7.992853956463472,7.992853956463472,1684810,0.0,8,deadline
pilot,sphere10,10,0.0,0,103.21285,c,cmaes,4244,8,10,300,0.1125703,0.3,0.45535802,4.045099106958052,1.5178600666666666,373410,0.0,8,8-of-8-internal-stops
pilot,sphere10,10,0.0,0,103.21285,b,fcmaes,4245,8,10,300,0.300055595,0.300055595,0.952297822,3.1737379267998653,3.1737379267998653,17420,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,b,cmaes,4245,8,10,300,0.300156799,0.300156799,0.925896864,3.084710614867665,3.084710614867665,15700,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,c,cmaes,4245,8,10,300,0.075506431,0.3,0.402690389,5.333193261379286,1.3423012966666668,363480,0.0,8,8-of-8-internal-stops
pilot,sphere10,10,0.0,0,103.21285,c,fcmaes,4245,8,10,300,0.300043488,0.300043488,2.391120247,7.9692456014909405,7.9692456014909405,1605030,0.0,8,deadline
pilot,sphere10,10,0.0,0,103.21285,b,cmaes,4246,8,10,300,0.300133574,0.300133574,0.957862851,3.191455185216966,3.191455185216966,15720,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,b,fcmaes,4246,8,10,300,0.300284496,0.300284496,0.926735745,3.086192451973944,3.086192451973944,18530,0.0,1,deadline
pilot,sphere10,10,0.0,0,103.21285,c,fcmaes,4246,8,10,300,0.300053959,0.300053959,2.387866766,7.958124511864882,7.958124511864882,1672650,0.0,8,deadline
pilot,sphere10,10,0.0,0,103.21285,c,cmaes,4246,8,10,300,0.091521592,0.3,0.390028476,4.261600650478195,1.30009492,336980,0.0,8,8-of-8-internal-stops
pilot,sphere100,100,0.0,0,455.3057,b,cmaes,1004245,8,17,300,0.301485633,0.301485633,0.498812663,1.6545155337468438,1.6545155337468438,10829,4.045292469378821e-8,1,deadline
pilot,sphere100,100,0.0,0,455.3057,b,fcmaes,1004245,8,17,300,0.301396049,0.301396049,0.378960297,1.2573499163554065,1.2573499163554065,3655,0.1816549459680866,1,deadline
pilot,sphere100,100,0.0,0,455.3057,c,fcmaes,1004245,8,17,300,0.303345889,0.303345889,2.282315475,7.523805522876231,7.523805522876231,28628,0.05144615765380494,8,deadline
pilot,sphere100,100,0.0,0,455.3057,c,cmaes,1004245,8,17,300,0.301167865,0.301167865,2.39434612,7.950204514681538,7.950204514681538,118830,9.799552477006266e-14,8,deadline
pilot,sphere100,100,0.0,0,455.3057,b,fcmaes,1004246,8,17,300,0.300420616,0.300420616,0.386716251,1.2872493777191378,1.2872493777191378,3434,0.515850334794488,1,deadline
pilot,sphere100,100,0.0,0,455.3057,b,cmaes,1004246,8,17,300,0.300349099,0.300349099,0.505148572,1.6818714412058215,1.6818714412058215,10880,6.457690294016243e-8,1,deadline
pilot,sphere100,100,0.0,0,455.3057,c,cmaes,1004246,8,17,300,0.301243845,0.301243845,2.364980167,7.8507169731550865,7.8507169731550865,103768,3.1305324259945575e-13,8,deadline
pilot,sphere100,100,0.0,0,455.3057,c,fcmaes,1004246,8,17,300,0.30122706,0.30122706,2.396152931,7.95464036663904,7.95464036663904,37706,0.011966095098849164,8,deadline
pilot,sphere100,100,0.0,0,455.3057,b,cmaes,1004247,8,17,300,0.30050653,0.30050653,0.537850341,1.7898124909298976,1.7898124909298976,9843,2.0962232906345542e-7,1,deadline
pilot,sphere100,100,0.0,0,455.3057,b,fcmaes,1004247,8,17,300,0.300784344,0.300784344,0.385147588,1.280477510491703,1.280477510491703,3791,0.12653333240033585,1,deadline
pilot,sphere100,100,0.0,0,455.3057,c,fcmaes,1004247,8,17,300,0.30113735,0.30113735,2.371962216,7.876678917444149,7.876678917444149,36108,0.011016069748220322,8,deadline
pilot,sphere100,100,0.0,0,455.3057,c,cmaes,1004247,8,17,300,0.301797949,0.301797949,2.384549857,7.901146660874094,7.901146660874094,116365,1.5552749472216321e-12,8,deadline
pilot,sphere100,100,0.0,0,455.3057,b,fcmaes,1004248,8,17,300,0.30089852,0.30089852,0.392579462,1.3046905714258747,1.3046905714258747,3910,0.09657384463282288,1,deadline
pilot,sphere100,100,0.0,0,455.3057,b,cmaes,1004248,8,17,300,0.300088417,0.300088417,0.532762683,1.7753523722310147,1.7753523722310147,9911,6.613548125180723e-7,1,deadline
pilot,sphere100,100,0.0,0,455.3057,c,cmaes,1004248,8,17,300,0.301970889,0.301970889,2.393327819,7.925690542309195,7.925690542309195,107780,1.364715226481651e-13,8,deadline
pilot,sphere100,100,0.0,0,455.3057,c,fcmaes,1004248,8,17,300,0.302353295,0.302353295,2.399590239,7.936378662584113,7.936378662584113,35037,0.010809886468439766,8,deadline
pilot,sphere100,100,0.0,0,455.3057,b,cmaes,1004249,8,17,300,0.300846457,0.300846457,0.525664573,1.7472852372663974,1.7472852372663974,9639,1.3316901431521593e-6,1,deadline
pilot,sphere100,100,0.0,0,455.3057,b,fcmaes,1004249,8,17,300,0.300365202,0.300365202,0.387581761,1.290368386281977,1.290368386281977,4029,0.10145878756697918,1,deadline
pilot,sphere100,100,0.0,0,455.3057,c,fcmaes,1004249,8,17,300,0.301417883,0.301417883,2.39837726,7.956983959043996,7.956983959043996,24293,0.1756746671201071,8,deadline
pilot,sphere100,100,0.0,0,455.3057,c,cmaes,1004249,8,17,300,0.301490496,0.301490496,2.392580216,7.935839596084647,7.935839596084647,108409,8.993421023825163e-13,8,deadline
pilot,cassini1,6,4.9307,0,5664.5372,b,cmaes,2004248,8,9,300,0.154612546,0.3,0.5036283,3.257357265173035,1.6787610000000002,8100,22.900470102313154,1,internal-stop-[NoEffectCoord]
pilot,cassini1,6,4.9307,0,5664.5372,b,fcmaes,2004248,8,9,300,0.300025285,0.300025285,0.973312427,3.2441013329926514,3.2441013329926514,15381,14.706531329916885,1,deadline
pilot,cassini1,6,4.9307,0,5664.5372,c,fcmaes,2004248,8,9,300,0.300118386,0.300118386,2.391341003,7.9679923475264856,7.9679923475264856,371934,5.303421055695461,8,deadline
pilot,cassini1,6,4.9307,0,5664.5372,c,cmaes,2004248,8,9,300,0.080531358,0.3,0.43449148,5.3953079991523305,1.4483049333333333,68688,11.03195190173271,8,8-of-8-internal-stops
pilot,cassini1,6,4.9307,0,5664.5372,b,fcmaes,2004249,8,9,300,0.300150056,0.300150056,1.008992951,3.361628394965217,3.361628394965217,13824,11.031951727800621,1,deadline
pilot,cassini1,6,4.9307,0,5664.5372,b,cmaes,2004249,8,9,300,0.205420775,0.3,0.663539915,3.230149993349017,2.2117997166666665,10719,17.267209187234847,1,internal-stop-[NoEffectCoord]
pilot,cassini1,6,4.9307,0,5664.5372,c,cmaes,2004249,8,9,300,0.073487148,0.3,0.425705871,5.792929547354321,1.4190195700000001,63702,13.66605661753476,8,8-of-8-internal-stops
pilot,cassini1,6,4.9307,0,5664.5372,c,fcmaes,2004249,8,9,300,0.300578421,0.300578421,2.380713088,7.920439132255605,7.920439132255605,349911,5.303426870959536,8,deadline
pilot,cassini1,6,4.9307,0,5664.5372,b,cmaes,2004250,8,9,300,0.211066391,0.3,0.675194599,3.198967849883784,2.2506486633333336,10053,17.267243173412353,1,internal-stop-[NoEffectCoord]
pilot,cassini1,6,4.9307,0,5664.5372,b,fcmaes,2004250,8,9,300,0.300017968,0.300017968,0.98071326,3.268848417772098,3.268848417772098,14112,16.71285774470656,1,deadline
pilot,cassini1,6,4.9307,0,5664.5372,c,fcmaes,2004250,8,9,300,0.300134172,0.300134172,2.3894589760000002,7.9613026403404685,7.9613026403404685,380007,10.998107668135466,8,deadline
pilot,cassini1,6,4.9307,0,5664.5372,c,cmaes,2004250,8,9,300,0.079236341,0.3,0.479792645,6.055209502922403,1.5993088166666667,76518,12.528792643703508,8,8-of-8-internal-stops
pilot,cassini1,6,4.9307,0,5664.5372,b,fcmaes,2004251,8,9,300,0.300097287,0.300097287,0.932609523,3.10769061700981,3.10769061700981,15867,16.845069259088902,1,deadline
pilot,cassini1,6,4.9307,0,5664.5372,b,cmaes,2004251,8,9,300,0.156696228,0.3,0.50173916,3.2019862022460424,1.6724638666666667,8532,11.031958297500475,1,internal-stop-[NoEffectCoord]
pilot,cassini1,6,4.9307,0,5664.5372,c,cmaes,2004251,8,9,300,0.086951472,0.3,0.529413825,6.088612565408899,1.7647127500000002,83340,11.031958297500475,8,8-of-8-internal-stops
pilot,cassini1,6,4.9307,0,5664.5372,c,fcmaes,2004251,8,9,300,0.300116147,0.300116147,2.391374861,7.968164608617343,7.968164608617343,359406,10.996492222763926,8,deadline
pilot,cassini1,6,4.9307,0,5664.5372,b,cmaes,2004252,8,9,300,0.226658512,0.3,0.689935047,3.0439405999453486,2.29978349,12222,17.43197213152984,1,internal-stop-[PosDefCov]
pilot,cassini1,6,4.9307,0,5664.5372,b,fcmaes,2004252,8,9,300,0.300089965,0.300089965,0.967545639,3.224185250579772,3.224185250579772,16641,14.706424870273619,1,deadline
pilot,cassini1,6,4.9307,0,5664.5372,c,fcmaes,2004252,8,9,300,0.300119055,0.300119055,2.350457402,7.8317499766884175,7.8317499766884175,355725,11.031952681673676,8,deadline
pilot,cassini1,6,4.9307,0,5664.5372,c,cmaes,2004252,8,9,300,0.097483704,0.3,0.590482695,6.057245167869287,1.9682756499999998,90720,10.996483066971306,8,8-of-8-internal-stops
```

</details>

<details>
<summary>Equal-call quality probe · 200 raw rows (CSV)</summary>

```csv
problem,pop,seed,lib,gens,calls,best,stop
sphere10,10,8100,fcmaes,128,1280,0.00000003473484979723968,0
sphere10,10,8100,cmaes,128,1280,0.000000548294482096765,0
sphere10,10,8101,cmaes,128,1280,0.000000038952686839921666,0
sphere10,10,8101,fcmaes,128,1280,0.00000005170334642176656,0
sphere10,10,8102,fcmaes,128,1280,0.0000000747878712866684,0
sphere10,10,8102,cmaes,128,1280,0.00000009787346016050461,0
sphere10,10,8103,cmaes,128,1280,0.00000005409936058746085,0
sphere10,10,8103,fcmaes,128,1280,0.0000004163244796621973,0
sphere10,10,8104,fcmaes,128,1280,0.00000016939873117884515,0
sphere10,10,8104,cmaes,128,1280,0.0000003124723993455093,0
sphere10,10,8105,cmaes,128,1280,0.00000017723484748705867,0
sphere10,10,8105,fcmaes,128,1280,0.000000013734466091193918,0
sphere10,10,8106,fcmaes,128,1280,0.00000005373793499947912,0
sphere10,10,8106,cmaes,128,1280,0.000001155178886673045,0
sphere10,10,8107,cmaes,128,1280,0.00000007443945192436996,0
sphere10,10,8107,fcmaes,128,1280,0.0000002570489337060804,0
sphere10,10,8108,fcmaes,128,1280,0.00000006573712394368568,0
sphere10,10,8108,cmaes,128,1280,0.00000014448108523702962,0
sphere10,10,8109,cmaes,128,1280,0.0000001431587818009344,0
sphere10,10,8109,fcmaes,128,1280,0.00000018632975630450164,0
sphere10,10,8110,fcmaes,128,1280,0.00000038618800752541773,0
sphere10,10,8110,cmaes,128,1280,0.0000004743442605732723,0
sphere10,10,8111,cmaes,128,1280,0.00000001675408846403307,0
sphere10,10,8111,fcmaes,128,1280,0.0000003613574963922903,0
sphere10,10,8112,fcmaes,128,1280,0.0000003286165080833077,0
sphere10,10,8112,cmaes,128,1280,0.0000000898302997652997,0
sphere10,10,8113,cmaes,128,1280,0.00000006761037693646096,0
sphere10,10,8113,fcmaes,128,1280,0.00000015474034548465828,0
sphere10,10,8114,fcmaes,128,1280,0.0000005050900977866863,0
sphere10,10,8114,cmaes,128,1280,0.000000008367264973379157,0
sphere10,10,8115,cmaes,128,1280,0.00000006091466692403676,0
sphere10,10,8115,fcmaes,128,1280,0.00000008309900215402266,0
sphere10,10,8116,fcmaes,128,1280,0.000000046712374921308606,0
sphere10,10,8116,cmaes,128,1280,0.0000000524297112504988,0
sphere10,10,8117,cmaes,128,1280,0.0000000378431516633672,0
sphere10,10,8117,fcmaes,128,1280,0.00000003787449013421815,0
sphere10,10,8118,fcmaes,128,1280,0.000000006581290396085192,0
sphere10,10,8118,cmaes,128,1280,0.00000006056467517566191,0
sphere10,10,8119,cmaes,128,1280,0.00000012656613711937343,0
sphere10,10,8119,fcmaes,128,1280,0.00000002768629312855827,0
sphere100,17,8100,fcmaes,128,2176,5.652601019512037,0
sphere100,17,8100,cmaes,128,2176,4.899167876921801,0
sphere100,17,8101,cmaes,128,2176,7.197881016562253,0
sphere100,17,8101,fcmaes,128,2176,5.781862713785347,0
sphere100,17,8102,fcmaes,128,2176,4.236525534567407,0
sphere100,17,8102,cmaes,128,2176,3.089390273358624,0
sphere100,17,8103,cmaes,128,2176,5.545881948551861,0
sphere100,17,8103,fcmaes,128,2176,5.931485786789372,0
sphere100,17,8104,fcmaes,128,2176,4.7350131183787205,0
sphere100,17,8104,cmaes,128,2176,4.597174001618897,0
sphere100,17,8105,cmaes,128,2176,6.67137297451105,0
sphere100,17,8105,fcmaes,128,2176,4.4938900747549795,0
sphere100,17,8106,fcmaes,128,2176,5.039883530966999,0
sphere100,17,8106,cmaes,128,2176,3.648163682209511,0
sphere100,17,8107,cmaes,128,2176,4.244925099637457,0
sphere100,17,8107,fcmaes,128,2176,7.7229140748136755,0
sphere100,17,8108,fcmaes,128,2176,4.251513481619792,0
sphere100,17,8108,cmaes,128,2176,5.300576300008263,0
sphere100,17,8109,cmaes,128,2176,4.054877805512335,0
sphere100,17,8109,fcmaes,128,2176,7.210161850450272,0
sphere100,17,8110,fcmaes,128,2176,5.320174447252109,0
sphere100,17,8110,cmaes,128,2176,7.996157805018718,0
sphere100,17,8111,cmaes,128,2176,5.481179545850532,0
sphere100,17,8111,fcmaes,128,2176,5.873376155775147,0
sphere100,17,8112,fcmaes,128,2176,4.9212295878509575,0
sphere100,17,8112,cmaes,128,2176,3.8774193054727384,0
sphere100,17,8113,cmaes,128,2176,4.009738543762895,0
sphere100,17,8113,fcmaes,128,2176,4.542706689999985,0
sphere100,17,8114,fcmaes,128,2176,5.052233431814258,0
sphere100,17,8114,cmaes,128,2176,3.418106587909171,0
sphere100,17,8115,cmaes,128,2176,3.021524868485199,0
sphere100,17,8115,fcmaes,128,2176,5.88543012591856,0
sphere100,17,8116,fcmaes,128,2176,5.899327115920692,0
sphere100,17,8116,cmaes,128,2176,5.177550738546946,0
sphere100,17,8117,cmaes,128,2176,4.991025656633741,0
sphere100,17,8117,fcmaes,128,2176,5.127660200218229,0
sphere100,17,8118,fcmaes,128,2176,5.458874134379904,0
sphere100,17,8118,cmaes,128,2176,5.540606766337171,0
sphere100,17,8119,cmaes,128,2176,4.59154902196485,0
sphere100,17,8119,fcmaes,128,2176,5.1912551650734375,0
rosenbrock40,15,8100,fcmaes,128,1920,42.11757054084339,0
rosenbrock40,15,8100,cmaes,128,1920,42.60849502355656,0
rosenbrock40,15,8101,cmaes,128,1920,56.85950533732567,0
rosenbrock40,15,8101,fcmaes,128,1920,107.44786154662798,0
rosenbrock40,15,8102,fcmaes,128,1920,46.930196174315284,0
rosenbrock40,15,8102,cmaes,128,1920,53.03996821458066,0
rosenbrock40,15,8103,cmaes,128,1920,50.062766300761766,0
rosenbrock40,15,8103,fcmaes,128,1920,53.97278522087995,0
rosenbrock40,15,8104,fcmaes,128,1920,120.77778736619544,0
rosenbrock40,15,8104,cmaes,128,1920,44.51076778361952,0
rosenbrock40,15,8105,cmaes,128,1920,56.301095214334026,0
rosenbrock40,15,8105,fcmaes,128,1920,43.744647032081495,0
rosenbrock40,15,8106,fcmaes,128,1920,54.50175667413122,0
rosenbrock40,15,8106,cmaes,128,1920,45.87951503398687,0
rosenbrock40,15,8107,cmaes,128,1920,49.319024910329006,0
rosenbrock40,15,8107,fcmaes,128,1920,117.75322174466457,0
rosenbrock40,15,8108,fcmaes,128,1920,52.82186194401138,0
rosenbrock40,15,8108,cmaes,128,1920,50.240528161931856,0
rosenbrock40,15,8109,cmaes,128,1920,46.86754993739135,0
rosenbrock40,15,8109,fcmaes,128,1920,47.95623169971516,0
rosenbrock40,15,8110,fcmaes,128,1920,66.56776484117194,0
rosenbrock40,15,8110,cmaes,128,1920,44.52884159664933,0
rosenbrock40,15,8111,cmaes,128,1920,46.182373073324435,0
rosenbrock40,15,8111,fcmaes,128,1920,61.22342431697283,0
rosenbrock40,15,8112,fcmaes,128,1920,44.753568862156044,0
rosenbrock40,15,8112,cmaes,128,1920,43.85086708973244,0
rosenbrock40,15,8113,cmaes,128,1920,53.62206662818052,0
rosenbrock40,15,8113,fcmaes,128,1920,46.72339130977167,0
rosenbrock40,15,8114,fcmaes,128,1920,52.22087264709802,0
rosenbrock40,15,8114,cmaes,128,1920,58.41167299746747,0
rosenbrock40,15,8115,cmaes,128,1920,46.96717619011208,0
rosenbrock40,15,8115,fcmaes,128,1920,44.350437325174404,0
rosenbrock40,15,8116,fcmaes,128,1920,48.944670335767654,0
rosenbrock40,15,8116,cmaes,128,1920,44.15086960530167,0
rosenbrock40,15,8117,cmaes,128,1920,45.45493862464954,0
rosenbrock40,15,8117,fcmaes,128,1920,42.29648417173247,0
rosenbrock40,15,8118,fcmaes,128,1920,41.979489208237645,0
rosenbrock40,15,8118,cmaes,128,1920,45.44898265947394,0
rosenbrock40,15,8119,cmaes,128,1920,45.94351622695908,0
rosenbrock40,15,8119,fcmaes,128,1920,43.13869432817103,0
ellipsoid100,17,8100,fcmaes,128,2176,2811989.1161878575,0
ellipsoid100,17,8100,cmaes,128,2176,2696606.9660609444,0
ellipsoid100,17,8101,cmaes,128,2176,4105933.4063204154,0
ellipsoid100,17,8101,fcmaes,128,2176,2778894.1127929348,0
ellipsoid100,17,8102,fcmaes,128,2176,2301626.780260968,0
ellipsoid100,17,8102,cmaes,128,2176,2126868.301577189,0
ellipsoid100,17,8103,cmaes,128,2176,4181935.3067686823,0
ellipsoid100,17,8103,fcmaes,128,2176,3228090.2691515973,0
ellipsoid100,17,8104,fcmaes,128,2176,5572038.173403418,0
ellipsoid100,17,8104,cmaes,128,2176,1469264.4964606732,0
ellipsoid100,17,8105,cmaes,128,2176,2856142.259663985,0
ellipsoid100,17,8105,fcmaes,128,2176,1807694.2099431688,0
ellipsoid100,17,8106,fcmaes,128,2176,1659015.9043655798,0
ellipsoid100,17,8106,cmaes,128,2176,2018539.40883082,0
ellipsoid100,17,8107,cmaes,128,2176,1957084.3384536481,0
ellipsoid100,17,8107,fcmaes,128,2176,1594721.7434256065,0
ellipsoid100,17,8108,fcmaes,128,2176,2583906.1462940243,0
ellipsoid100,17,8108,cmaes,128,2176,3376132.924922135,0
ellipsoid100,17,8109,cmaes,128,2176,2492813.652363923,0
ellipsoid100,17,8109,fcmaes,128,2176,3083682.360114909,0
ellipsoid100,17,8110,fcmaes,128,2176,2003341.3044184393,0
ellipsoid100,17,8110,cmaes,128,2176,2570283.5077640656,0
ellipsoid100,17,8111,cmaes,128,2176,1612947.279480794,0
ellipsoid100,17,8111,fcmaes,128,2176,2848608.2755830805,0
ellipsoid100,17,8112,fcmaes,128,2176,2012862.6114509732,0
ellipsoid100,17,8112,cmaes,128,2176,2273735.1671762564,0
ellipsoid100,17,8113,cmaes,128,2176,2934517.0092339194,0
ellipsoid100,17,8113,fcmaes,128,2176,2155767.7758143228,0
ellipsoid100,17,8114,fcmaes,128,2176,1538327.6137174657,0
ellipsoid100,17,8114,cmaes,128,2176,2299860.2222936396,0
ellipsoid100,17,8115,cmaes,128,2176,2463082.5229786113,0
ellipsoid100,17,8115,fcmaes,128,2176,2345847.2508410006,0
ellipsoid100,17,8116,fcmaes,128,2176,2952639.967091986,0
ellipsoid100,17,8116,cmaes,128,2176,3512362.291550238,0
ellipsoid100,17,8117,cmaes,128,2176,3193193.091975264,0
ellipsoid100,17,8117,fcmaes,128,2176,2383461.0847630682,0
ellipsoid100,17,8118,fcmaes,128,2176,2482719.576203512,0
ellipsoid100,17,8118,cmaes,128,2176,2296702.9470055555,0
ellipsoid100,17,8119,cmaes,128,2176,3457279.7189778974,0
ellipsoid100,17,8119,fcmaes,128,2176,3066920.3589835186,0
cassini1,9,8100,fcmaes,128,1152,6.417842462834999,0
cassini1,9,8100,cmaes,128,1152,15.902487179710086,0
cassini1,9,8101,cmaes,128,1152,17.689398678620783,0
cassini1,9,8101,fcmaes,128,1152,55.63400449584263,0
cassini1,9,8102,fcmaes,128,1152,17.491856251424878,0
cassini1,9,8102,cmaes,128,1152,16.92054297338378,0
cassini1,9,8103,cmaes,128,1152,14.729243449522658,0
cassini1,9,8103,fcmaes,128,1152,27.378565189570445,0
cassini1,9,8104,fcmaes,128,1152,72.57312927457326,0
cassini1,9,8104,cmaes,128,1152,14.724214489642105,0
cassini1,9,8105,cmaes,128,1152,17.85099522167018,0
cassini1,9,8105,fcmaes,128,1152,5.393279227692252,0
cassini1,9,8106,fcmaes,128,1152,17.846798460042514,0
cassini1,9,8106,cmaes,128,1152,17.248643023796145,0
cassini1,9,8107,cmaes,128,1152,14.809445496874462,0
cassini1,9,8107,fcmaes,128,1152,17.55153581901216,0
cassini1,9,8108,fcmaes,128,1152,14.918827145714687,0
cassini1,9,8108,cmaes,128,1152,24.03941981232657,0
cassini1,9,8109,cmaes,128,1152,17.35799184272198,0
cassini1,9,8109,fcmaes,128,1152,17.805653351262762,0
cassini1,9,8110,fcmaes,128,1152,25.909876616834513,0
cassini1,9,8110,cmaes,128,1152,24.50112961744138,0
cassini1,9,8111,cmaes,128,1152,15.736423064978165,0
cassini1,9,8111,fcmaes,128,1152,13.985623640959988,0
cassini1,9,8112,fcmaes,128,1152,57.44824754592526,0
cassini1,9,8112,cmaes,128,1152,17.740051312228452,0
cassini1,9,8113,cmaes,128,1152,14.780877281943555,0
cassini1,9,8113,fcmaes,128,1152,17.32287043795633,0
cassini1,9,8114,fcmaes,128,1152,16.99570484747666,0
cassini1,9,8114,cmaes,128,1152,15.819821357782928,0
cassini1,9,8115,cmaes,128,1152,14.758125466883643,0
cassini1,9,8115,fcmaes,128,1152,11.291029773233696,0
cassini1,9,8116,fcmaes,128,1152,19.21284784193804,0
cassini1,9,8116,cmaes,128,1152,17.62100981669742,0
cassini1,9,8117,cmaes,128,1152,11.065788104928753,0
cassini1,9,8117,fcmaes,128,1152,5.494876998172627,0
cassini1,9,8118,fcmaes,128,1152,28.13064497854141,0
cassini1,9,8118,cmaes,128,1152,17.64624831679037,0
cassini1,9,8119,cmaes,128,1152,17.69171073619082,0
cassini1,9,8119,fcmaes,128,1152,17.81306522967508,0
```

</details>

<details>
<summary>Native phase probe · 56 raw rows (CSV)</summary>

```csv
seed,d,pop,path,evals,stop,wall_ns,ask_ns,cov_ns,eig_ns,asks,covs,eigs,best
1,10,10,direct,1280,0,1175309,177856,234559,564003,128,128,128,0.00000000016397068444922818
1,10,10,ask_tell,1280,0,1585115,159885,214412,559770,128,128,128,0.000000000514690590361654
1,40,15,direct,1920,0,14843231,992892,920050,12165887,128,128,128,0.000125755577181291
1,40,15,ask_tell,1920,0,14440655,1008011,948776,11259842,128,128,128,0.00010963388853152829
1,100,17,direct,2176,0,132415539,4967583,6996908,116803947,128,127,127,0.013216108010121947
1,100,17,ask_tell,2176,0,138336370,5180788,8124835,119961484,128,128,128,0.01267633741649507
1,100,31,direct,3968,0,191001386,13176470,15044964,153423706,128,128,128,0.0011795792337911824
1,100,31,ask_tell,3968,0,145109225,9372797,10946798,118125077,128,128,128,0.0011194479603657166
2,10,10,direct,1280,0,1107888,159212,180329,558256,128,128,128,0.0000000013909651875913238
2,10,10,ask_tell,1280,0,1216211,158125,168787,572828,128,128,128,0.0000000019804216497654386
2,40,15,direct,1920,0,14585272,1032791,937532,11812941,128,128,128,0.00014373128639044413
2,40,15,ask_tell,1920,0,14520926,1024634,971211,11279047,128,128,128,0.0002792869465262099
2,100,17,direct,2176,0,149993325,6119269,9599544,129317438,128,127,127,0.008838865942034492
2,100,17,ask_tell,2176,0,151123846,5058674,7650031,133415477,128,128,128,0.016869982200991306
2,100,31,direct,3968,0,150582591,9637487,11590735,123376590,128,128,128,0.0004184882242082636
2,100,31,ask_tell,3968,0,155466442,9528479,9664408,129336539,128,128,128,0.001192227785252744
3,10,10,direct,1280,0,1111683,175039,180219,561019,128,128,128,0.0000000009410913470333284
3,10,10,ask_tell,1280,0,1274308,217364,172420,576038,128,128,128,0.00000000337798293609908
3,40,15,direct,1920,0,14212933,992453,903338,11575342,128,128,128,0.0002397228585227572
3,40,15,ask_tell,1920,0,14762690,987521,976000,11559548,128,128,128,0.00011260102455193228
3,100,17,direct,2176,0,134151241,4968292,8947756,115361266,128,127,127,0.013015649421114439
3,100,17,ask_tell,2176,0,138549801,5294734,7452454,120992123,128,128,128,0.020088177538340968
3,100,31,direct,3968,0,149174443,9509379,11107511,123368022,128,128,128,0.0010592578448883983
3,100,31,ask_tell,3968,0,148496242,9821826,10723401,120924325,128,128,128,0.0012263034958973206
4,10,10,direct,1280,0,1110111,159143,179347,560880,128,128,128,0.0000000022314285125679725
4,10,10,ask_tell,1280,0,1300838,172072,227825,597037,128,128,128,0.000000004559053820388291
4,40,15,direct,1920,0,14331220,1036244,1016623,11498211,128,128,128,0.00013596947465409757
4,40,15,ask_tell,1920,0,15061199,1230472,940770,11619089,128,128,128,0.00015546422042393907
4,100,17,direct,2176,0,134019673,5085511,9634885,115003534,128,127,127,0.022493129018729574
4,100,17,ask_tell,2176,0,135062473,5250048,7538384,117395263,128,128,128,0.015059981315584605
4,100,31,direct,3968,0,143377799,9165009,9763242,119684916,128,128,128,0.0011862359541321922
4,100,31,ask_tell,3968,0,167541817,10286246,16883914,127017942,128,128,128,0.0017363364209328877
5,10,10,direct,1280,0,1167917,159110,181344,600618,128,128,128,0.0000000012929754405466046
5,10,10,ask_tell,1280,0,1204553,161028,183178,558051,128,128,128,0.00000000011810300148362793
5,40,15,direct,1920,0,14856270,1073525,1042795,11970842,128,128,128,0.0001137370720161023
5,40,15,ask_tell,1920,0,14628257,1013235,924035,11486880,128,128,128,0.0003858431850144967
5,100,17,direct,2176,0,136397472,5048941,9391326,117314545,128,127,127,0.006364405296595672
5,100,17,ask_tell,2176,0,135910399,5049499,7753017,118108900,128,128,128,0.013053690717475882
5,100,31,direct,3968,0,145690129,9272506,10808044,120258702,128,128,128,0.0010499914153240526
5,100,31,ask_tell,3968,0,145117667,9304951,9872833,119002560,128,128,128,0.0011049346782926073
6,10,10,direct,1280,0,1153256,161304,183594,611459,128,128,128,0.0000000016141789350719476
6,10,10,ask_tell,1280,0,1234878,158126,168836,573877,128,128,128,0.000000003206040560729664
6,40,15,direct,1920,0,14328896,993563,921888,11611259,128,128,128,0.0002719838687171375
6,40,15,ask_tell,1920,0,14296218,1020655,920615,11152021,128,128,128,0.0001550199081492702
6,100,17,direct,2176,0,135710959,5374033,9256110,116497146,128,127,127,0.013863252052084829
6,100,17,ask_tell,2176,0,138369760,5097263,7570324,120495093,128,128,128,0.02653376297345926
6,100,31,direct,3968,0,145793544,9396017,11252050,120017976,128,128,128,0.0008651924371117917
6,100,31,ask_tell,3968,0,148959420,9460904,9860075,122927307,128,128,128,0.0009336877833610989
7,10,10,direct,1280,0,1111864,159570,179903,579633,128,128,128,0.0000000017015864220497834
7,10,10,ask_tell,1280,0,1395080,160675,179201,702158,128,128,128,0.0000000008502032679228933
7,40,15,direct,1920,0,18412642,1022037,940910,15440558,128,128,128,0.0002541778236384379
7,40,15,ask_tell,1920,0,15996977,1010503,1108431,12569684,128,128,128,0.0002167638959868367
7,100,17,direct,2176,0,135819041,5347928,7308523,119067447,128,127,127,0.008926204211156079
7,100,17,ask_tell,2176,0,140428530,5083941,7432853,121299184,128,128,128,0.019575786802953077
7,100,31,direct,3968,0,143407472,9149876,10187075,119203987,128,128,128,0.0009018590069545316
7,100,31,ask_tell,3968,0,146245374,9636149,10572624,118751126,128,128,128,0.0011896219847815465
```

</details>
