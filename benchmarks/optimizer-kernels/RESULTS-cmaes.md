# fcmaes-rust active CMA-ES performance implementation log

27 November 2026 · Base `d7ff66aebc388ac19b698742c7678986e7a48d58` · 30 candidate changes screened.

## Result

The patch improved serial cheap-objective throughput **1.74× on 100-D Sphere** and **1.81× on 100-D Ellipsoid** in 20 interleaved pairs at 400 ms. It passed the 0.50× external-throughput goal on both workload medians.

- At dimension 64 and above with automatic `update_gap`, reuse the basis only when it was computed in the previous generation and the live covariance diagonal is positive and finite. Keep the baseline covariance-update and eigen-eligibility calculations. An explicit gap preserves the previous refresh decisions.
- Initialize identity covariance directly and transfer an encoded generation into the update after validating its scores. Keep the score sanitation, ordered slots, and evaluation accounting at the original entry points.
- Expose a borrowed contiguous decoded batch with Rust `Cmaes::ask_flat()`. Python `ACMA.ask()` returns an independently owned NumPy array from that batch. Both ordinary and external tell paths use the existing checks.
- For external populations at dimension 64 or above, compute the inverse transform from the stored orthogonal basis and scales. Validate the reconstructed displacement. Invalid scales or a failed reconstruction use the original inverse and its sampling fallback.

The 100-D Sphere target of 0.01 took less wall time on all 20 timing pairs overall, with every run reaching the target. The long rotated 100-D holdout ended at a median of 4.29 versus 3.72 for the baseline after 99,975 calls. The earlier 9,996-call rotated endpoint favored the patch (4,126 versus 4,802). These are different prespecified rotations and budgets.

## Final elapsed-time comparison

Comparator is `cmaes` 0.2.2 with negative weights. Ratios are medians of per-seed quotients. Brackets show 95% paired bootstrap intervals with 10,000 seed-block resamples. The external ratio uses evaluations per active second. The JSONL includes allocated wall time and stop counts so early numerical stops can be examined separately.

| Objective and topology | Wall allocation | Pairs | Patch / pinned baseline | Patch / external |
|---|---:|---:|---:|---:|
| sphere100, serial | 400 ms | 20 | 1.740× [1.604, 1.931] | 0.513× [0.478, 0.537] |
| ellipsoid100, serial | 400 ms | 20 | 1.809× [1.679, 1.913] | 0.543× [0.484, 0.573] |
| sphere10, parallel pop. ×8 | 200 ms | 20 | 0.988× [0.823, 1.196] | 1.249× [0.973, 1.368] |
| sphere10, independent ×8 | 200 ms | 20 | 1.085× [0.975, 1.199] | 1.093× [0.964, 1.187] |
| cassini1, parallel pop. ×8 | 200 ms | 20 | 1.117× [0.730, 1.633] | 1.136× [0.963, 1.434] |
| cassini1, independent ×8 | 200 ms | 20 | 0.995× [0.859, 1.080] | 1.165× [0.964, 1.292] |
| sphere10, serial, 1 µs floor | 100 ms | 7 | 0.947× [0.839, 1.215] | 0.789× [0.687, 0.890] |
| sphere10, serial, 100 µs floor | 100 ms | 7 | 0.792× [0.490, 1.097] | 0.597× [0.481, 1.009] |
| rosenbrock40, serial, 1 µs floor | 100 ms | 7 | 0.929× [0.816, 1.161] | 0.631× [0.419, 0.783] |
| rosenbrock40, serial, 100 µs floor | 100 ms | 7 | 1.063× [0.968, 1.225] | 1.026× [0.935, 1.109] |
| sphere100, serial, 1 µs floor | 100 ms | 7 | 1.743× [1.549, 1.774] | 0.525× [0.448, 0.634] |
| sphere100, serial, 100 µs floor | 100 ms | 7 | 1.335× [1.046, 1.436] | 0.836× [0.799, 0.906] |
| ellipsoid100, serial, 1 µs floor | 100 ms | 7 | 1.724× [1.365, 1.832] | 0.514× [0.470, 0.642] |
| ellipsoid100, serial, 100 µs floor | 100 ms | 7 | 1.177× [1.023, 1.189] | 0.774× [0.721, 0.877] |
| sphere100, serial | 1000 ms | 7 | 1.762× [1.678, 2.293] | 0.500× [0.470, 0.536] |
| sphere100, serial | 10000 ms | 2 | 2.048× | 0.539× |
| ellipsoid100, serial | 1000 ms | 7 | 1.739× [1.637, 2.002] | 0.524× [0.448, 0.642] |
| ellipsoid100, serial | 10000 ms | 2 | 1.727× | 0.536× |

The comparison contains 582 rows. Internal-stop rows numbered 68 for the external crate, 0 for the original fcmaes build, and 0 for the patch. The 10-second rows use two exploratory pairs per problem.

### Protected control repeats

A short, seven-pair Sphere 10 run with a 100 µs floor was slow during the shared-host campaign. The repeat uses 20 new, alternating pairs. The repeat compared both elapsed and process CPU time.

| Cell | Patch / baseline by wall | Patch / baseline by process CPU |
|---|---:|---:|
| sphere10/a/250ms/100000ns | 1.003× [0.990, 1.009] | 1.003× [0.990, 1.009] |
| sphere10/a/250ms/0ns | 1.053× [1.042, 1.105] | 1.053× [1.042, 1.105] |
| rosenbrock40/a/250ms/1000ns | 1.020× [0.976, 1.088] | 1.016× [0.998, 1.060] |

### Time to target

These native serial tests time the first attaining callback and include construction, with equal maximum calls. The rotated case and its target were specified before the timing run. Rechecks run after the solve. The target path has per-point tracing overhead on both sides.

| Workload | Target | Baseline hits / 20 | Patch hits / 20 | Baseline / patch first-hit time |
|---|---:|---:|---:|---:|
| sphere100 | 0.01 | 20 | 20 | 1.819× [1.764, 2.010] |
| rotated-100-target | 8000 | 20 | 20 | 1.975× [1.552, 2.403] |

### What changed inside a generation

| Public ask/tell, 128 generations | Median original eigensolves | Median patch eigensolves | Covariance updates per build | Original µs/gen | Patch µs/gen |
|---|---:|---:|---:|---:|---:|
| d=10, λ=10 | 128 | 128 | 128 / 128 | 10.0 | 9.0 |
| d=40, λ=15 | 128 | 128 | 128 / 128 | 114.3 | 126.1 |
| d=100, λ=17 | 128 | 64 | 128 / 128 | 1081.0 | 579.3 |
| d=100, λ=31 | 128 | 64 | 128 / 128 | 1160.1 | 687.7 |

Removing basis reuse gave 0.538× [0.434, 0.563] of the patch’s performance on the 100-D Sphere generation probe. Removing population transfer gave 0.942× [0.718, 1.067] of the patch’s performance on the small native probe. The single-change confirmation for transfer is the stronger evidence for its individual effect.

## Fixed-budget and application checks

Forty held-out paired seeds per cell. The table records actual callback counts, with natural stop codes retained. Rotated ellipsoids use a deterministic Householder reflection and condition number 10⁶. Target counts use the displayed numeric thresholds, not across-library iteration counts.

| Workload | Target | Median actual calls, original / patch | Median best, original / patch | Hits, original / patch |
|---|---:|---:|---:|---:|
| sphere100 | 0.01 | 9996 / 9996 | 1.05686e-09 / 1.08206e-09 | 40 / 40 |
| rotated-100-budget-100d | 8000 | 9996 / 9996 | 4801.9 / 4125.68 | 39 / 40 |
| rotated-100-budget-1000d | 0.01 | 99975 / 99975 | 3.72205 / 4.2861 | 0 / 0 |
| rosen40-1000d | 1e-06 | 39990 / 39990 | 14.602 / 14.602 | 0 / 0 |
| box64-100d | 34 | 6386 / 6386 | 33.5194 / 33.4457 | 40 / 40 |
| box64-1000d | 34 | 7703.5 / 7734.5 | 33.8707 / 33.6488 | 31 / 33 |
| cassini1-independent | 4.95535 | 7936 / 7936 | 16.7115 / 16.7115 | 0 / 0 |
| cassini2 | 8.42491 | 7936 / 7936 | 25.8756 / 25.8756 | 0 / 0 |
| raw-boundary64 | 34 | 2511 / 2542 | 47.4593 / 47.253 | 0 / 0 |
| external100 | 0.01 | 7936 / 7936 | 9.23527e-16 / 9.25119e-16 | 40 / 40 |
| external100-noask | 0.01 | 899 / 961 | 8.47349 / 8.47349 | 0 / 0 |

The corrected Cassini2 fixture uses the repository’s published bounds on both sides. Initial Cassini2 rows were discarded after a bounds mismatch in the benchmark executable was found. The corrected Cassini2 final values, points and counts agree on all 40 pairs. The raw release record includes both the excluded fixture and the corrected replay.

Cassini1 DE→CMA and jSO→CMA replayed the actual first stages at a 40/60 budget split and identical handoffs in each pair. The 80 pairs span 20 seeds, two first-stage solvers and both `repair_z` settings. All 80 matched in first-stage score, final score, actual calls and stop code. Independent rescoring confirmed the best values.

## All 30 iterations

Each candidate was built from the same pinned tree for attribution. The guarded refresh and the successful implementation changes then faced a combined repeat and ablations. Entries use the trial’s main cell or its independent confirmation. A ratio above one favors the candidate. The ZIP keeps experimental diffs separate from the production patch.

| # | Change | Paired speedup [95% interval] | Decision and evidence |
|---:|---|---:|---|
| 01 | faer full symmetric eigensolver | 1.663× [1.567, 1.827] | **Retain trial evidence.** The 256-D advantage did not extend to the protected smaller dimensions. Maximum reconstruction residual across the replay was 7.3e−15 for faer. |
| 02 | Refresh every second or fourth eligible generation | 1.750× [1.642, 1.799] | **Retain guarded interval two.** Interval four reached 3.07× on this panel. The patch reuses only an immediately preceding factor, only with automatic gap at d≥64, and checks the live diagonal. This also preserves the shorter-dimension restart trajectories. |
| 03 | Refresh after measured covariance drift | 4.383× [3.872, 5.081] | **Reject.** Fastest development policy on this cell. Its age cap was eight eligible generations. The fixed-call Rosenbrock check reached 13/20 targets versus the baseline's 15/20, so this trial did not justify the broader policy. |
| 04 | Warm-start the previous basis with three checked Jacobi sweeps | 0.174× [0.156, 0.186] | **Reject.** The basis transforms, sweeps and validation cost more than a fresh solve, even with the full-solve fallback. |
| 05 | Symmetrize the live covariance before cloning once | 0.928× [0.895, 0.962] | **Reject.** The copy removal saved allocation, with a confirmed slowdown on this cell. |
| 06 | Initialize identity C and scaled identity BD directly | 12.292× [11.872, 12.747] | **Retain.** Construction alone was 12× faster on the unequal-sigma control. The independent audit preserved the seeded result and the eight-lane construction check also improved. |
| 07 | Scale weight columns instead of multiplying by a dense diagonal | 1.046× [0.993, 1.111] | **Reject.** The paired interval overlapped parity and total allocated bytes changed little on small populations. |
| 08 | Accumulate old C and the rank-one term into C | 1.019× [0.982, 1.162] | **Reject.** Heap bytes improved by about 18% on this cell, with an inconclusive end-to-end time change. |
| 09 | Accumulate and mirror one triangle of weighted rank-μ products | 1.025× [0.875, 1.116] | **Reject.** Uncertain throughput on the intended large populations, with different seeded arithmetic. |
| 10 | Scatter negative norm ratios using a single permutation | 1.032× [0.972, 1.049] | **Reject.** The promising first screen fell below the threshold on the new seed block. |
| 11 | Cache pre-clipping displacements for the negative update | 0.992× [0.925, 1.088] | **Reject.** The extra storage and reparations removed the expected gain. Clipped and externally supplied cases also showed little benefit. |
| 12 | Accumulate positive-parent means from indexed columns | 0.986× [0.908, 1.040] | **Reject.** The initial small-dimension, large-population speedup did not repeat. |
| 13 | Reuse positive/negative covariance work matrices | 0.973× [0.889, 1.012] | **Reject.** Little net saving after the unchanged products. The Cassini1 eight-lane check was inconclusive. |
| 14 | Fill Gaussian matrix in place, in the original draw order | 1.221× [0.913, 1.314] | **Reject.** Seeded traces agreed. The interval was wide and the allocation savings fell short of the alternative acceptance rule. |
| 15 | Use a matrix product for the entire sampling batch | 1.150× [1.069, 1.318] | **Feed crossover trial.** Promising at large dimension/population on the development grid. The resulting trajectories differed, and the final held-out grid governed the default. |
| 16 | Write affine and clipped samples into the destination column | 1.007× [0.975, 1.133] | **Reject.** The repeat run narrowed the effect toward parity. It preserved the seeded results in the serial controls. |
| 17 | Transfer encoded populations into generation state | 1.121× [1.078, 1.200] | **Retain.** Both native paths avoid repeated column copies. Public tell checks its inputs before swapping the batch, and the seeded trace matched. |
| 18 | Reuse native evaluation decode/clamp storage | 1.030× [1.011, 1.127] | **Reject.** The serial bounded win was small and other paths showed regressions or substantial noise. |
| 19 | Create owned rows from a direct column slice | 0.953× [0.866, 1.097] | **Fold into flat API.** The standalone owned-row speed estimate was inconclusive. Direct copying is used by the shared owned/borrowed implementation. |
| 20 | Borrow a flat decoded batch for Rust ask/tell | 1.163× [1.129, 1.273] | **Retain.** The serial caller improved with identical point and value hashes. The eight-worker interval was wider. |
| 21 | Use checked D⁻¹ Bᵀ for large externally supplied tells | 1.242× [1.096, 1.374] | **Retain at d≥64.** The 100-D external-population path improved. The patch checks a forward reconstruction and falls back to the original inverse when needed. |
| 22 | Pack and transform many repaired columns together | 0.990× [0.887, 1.054] | **Reject.** Packing cost absorbed the savings at d=20, including the high-clipping workload. |
| 23 | Stable small-population insertion sort | 1.135× [0.995, 1.176] | **Reject.** Kept tie order and matched audited traces. The interval included parity on the intended small case. |
| 24 | Circular fitness history with a combined extrema pass | 0.998× [0.987, 1.001] | **Reject.** The end-to-end result was statistically indistinguishable from parity. |
| 25 | Reuse scratch for the evolution-path update | 0.998× [0.872, 1.423] | **Reject.** The saving was too small to resolve, with no evidence of a durable throughput gain. |
| 26 | Build Python's owned NumPy array from the contiguous batch | 1.141× [1.016, 1.324] | **Retain.** The public shape, dtype, strided-score behavior and lifetime checks passed. Separate tell_x and small-dimension measurements are included. |
| 27 | Resolve the explicit Rayon pool once per direct run | 1.065× [0.774, 1.338] | **Reject.** The observed effect was noisy, with no robust benefit on the 1 µs objective. |
| 28 | Group work in Rayon by minimum grain two or four | 2.197× [1.905, 2.492] | **Reject.** The cheap ten-candidate gain coincided with regressions on the 1 µs and skewed 30 µs cases. The two-candidate grain also lacked a general gain. |
| 29 | Reuse shape-compatible CMA state buffers between restarts | 0.991× [0.962, 1.023] | **Reject.** The eight independent Cassini1 lanes ran at parity. The small cold-sphere improvement was uncertain. |
| 30 | Dispatch between per-column and batched sampling at d≥96 and λ≥48 | 1.040× [0.997, 1.247] | **Reject.** The crossover was fitted on the development grid. Scalar, dispatched and always-batched paths were rerun against the same held-out shapes on the integrated build. |

The selected build with trial 30 added reached 1.295× [1.200, 1.715] on d=128, λ=64. Smaller and sparse-population held-back cells were uncertain and the path changes seeded arithmetic. The current production combination keeps per-column sampling.

### Diagnostics behind the decisions

- The faer matrix replay used three actual covariance snapshots and two analytic near-singular or tiny-negative fixtures at each of d=10, 40, 100 and 256. The baseline and faer max orthogonality errors stayed below 4e−15 in normalized Frobenius norm. The 10-D near-singular baseline reconstruction residual reached 1.18e−10, while faer’s maximum over all fixtures was 7.3e−15. The initial replay bound of 2e−12 excluded this baseline case and was replaced with reporting both residuals. In the full-generation screen faer at d=100 was 0.929×, despite a favorable isolated kernel median.
- Interval-two and interval-four prototypes used the original covariance adaptation and raw eigen eligibility. In the 20-pair development panel, interval two reached the Rosenbrock 40 target on 19 pairs versus 15 for the original. Interval four reached it on 17. Four gave a worse median than the baseline on the 64-D rotated development problem. The deployed policy limits deferral to one adjacent generation.
- The exact repair-column replay covered 0, 1, 15 and 31 repaired rows. A separate reset check matched fresh construction over six starts with repeated and changed population sizes. Both tests passed in their candidate builds.
- For numerical replays the replacement eigensolver’s eigenvectors were compared by residual and orthogonality. Seeded exactness was checked by hashing evaluated coordinates, scores and the best point for storage-only changes. Parallel objective visitation order can differ on the same candidates, so the parallel hash is diagnostic.
- Heap bytes and allocation calls use a separate glibc instrumented process on serial cells. The multithreaded allocator instrumentation proved unstable during the campaign, so eight-lane memory records use process peak RSS. Peak RSS includes the executable and runtime.

## Verification and scope

- Rust 1.100.0, Linux x86-64 on AMD EPYC 9V74, portable release profile with thin LTO and one codegen unit. Timings use CPU 1 for serial, CPUs 0–7 for eight workers, and CPU 8 for the Python or sampling crossover replay when isolated. Quality-only replays ran concurrently on pinned CPUs and use call counts for comparisons.
- All 431 workspace tests and doctests passed. Workspace formatting and Clippy with warnings denied passed. The logs include pre-existing Cargo manifest warnings about unused dependencies. The locked workspace checked successfully on Rust 1.88.0.
- All 123 Python binding tests passed for the final release extension under Python 3.12 and NumPy 2.3.5. Independently verified contiguous batches, distinct allocations across successive asks, lifetime after optimizer destruction, and strided scores in trial 26.
- The full 7,920-row publication schedule would require 16 physical workers for its original topology and substantially more machine time. This run used the same comparator protocol on a reduced matrix with eight workers, 100/200/250/400 ms controls, seven pairs at 1 s on both 100-D problems and two pairs at 10 s on each. Fresh verification on the 9950X remains outside this dataset.
- Fixed-budget confirmations reused the shared harness and native direct or public ask/tell loops. Reference slots, complete scores and result snapshots stayed aligned in the separate API tests. The tests cover the new flat API, the conditional refresh and its diagonal guard, and rejection of an inaccurate external-coordinate reconstruction.

## Reproduce

The ZIP contains `0001-cmaes-performance.patch`, independent trial diffs, source for the measurement drivers, and raw fixed-call, fixed-wall, phase, allocation and Python rows. The production patch applies at the pinned commit.

```bash
git switch --detach d7ff66aebc388ac19b698742c7678986e7a48d58
git apply --check /path/to/0001-cmaes-performance.patch
git apply /path/to/0001-cmaes-performance.patch
cargo test --workspace --locked
cd benchmarks/cmaes-implementation
cargo run --release --locked -- --mode verify
cargo run --release --locked -- --mode campaign --preset pilot --problems sphere100,ellipsoid100 --arms a --cost-ns 0 --deadlines-ms 400 --seeds 20 --workers 1 --output results/cma-patch-check
```

Use the same command in a baseline checkout and compare the paired results. For the fixed-generation driver, copy `replay/cma_campaign.rs` from the ZIP into `benchmarks/cmaes-implementation/examples/` in both checkouts and build the `cma_campaign` example. Verified round-by-round settings and precise paired seeds are in the JSONL files. The experiment replay directory includes the exact case definitions and instrumentation code.

[Pinned benchmark protocol](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/benchmarks/cmaes-implementation/README.md) · [GTOP targets](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/benchmarks/benchmark_gtop.md) · [faer 0.22.6 manifest](https://docs.rs/crate/faer/0.22.6/source/Cargo.toml.orig)
