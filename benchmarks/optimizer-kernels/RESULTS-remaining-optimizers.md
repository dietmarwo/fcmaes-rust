# fcmaes-rust remaining optimizer implementation log

**27 November 2026 · release 0.5.0 · base `ccb9f8b1e4153c1ce2069961715cbe4b951e7806`**

The campaign tested the performance plan against the pinned 0.5.0 source.
The ledger closes all **380** plan rows across **19 tracks**. Independent A/B variants ran for **374** plan rows. Six rows describe behavior already present in the base revision. An additional surrogate/2 trial removed a redundant scoring pass uncovered during profiling.

The retained changes preserve normal public entry points. Hard-cap batch calls, explicit cooperative stopping and weighted-retry accounting have separate opt-in Rust entry points. The five original standalone solver exclusions apply throughout. The experimental changes in L-SHADE, L-SRTDE, AMaLGaM and Nelder–Mead are confined to the candidate and boundary benchmarks.

[Base revision](https://github.com/dietmarwo/fcmaes-rust/tree/ccb9f8b1e4153c1ce2069961715cbe4b951e7806)

## Largest measured gains

The speed factor is the median of paired reference/candidate times. Intervals are 95% paired bootstrap intervals. All rows here use the final patch from this archive, compiled with release thin LTO, on 30 seeds held out from screening.

| Optimizer | Workload, dimension, budget, workers or batch | Speed [95% interval] | Physical calls | Median quality |
|---|---|---:|---:|---:|
| jSO | sphere 128D / 4,096 / 1 | 1.11× [1.10, 1.12] | 4096 → 4096 | 10.847 → 10.847 |
| Dual Annealing | rosen 16D / 4,096 / 1 | 1.08× [1.06, 1.10] | 4096 → 4096 | 3.2019e-17 → 3.2019e-17 |
| ArchiveDE | front 8D / 4,096 / 1 | 1.07× [1.06, 1.08] | 4096 → 4096 | 3.498 → 3.498 |
| ArchiveDE | zdt3 16D / 8,192 / 1 | 1.12× [1.11, 1.14] | 8192 → 8192 | 4.8138 → 4.8138 |
| ArchiveDE | dtlz2 8D / 4,096 / 1 | 1.07× [1.04, 1.08] | 4096 → 4096 | 0.039106 → 0.039106 |
| Independent retry | sphere 16D / 2,048 / 1 | 1.69× [1.67, 1.85] | 2048 → 2048 | 7.4614 → 7.4614 |
| Weighted multi-objective retry | front 8D / 2,048 / 1 | 2.57× [2.49, 2.81] | 2112 → 2112 | 3.209 → 3.209 |
| Weighted multi-objective retry | zdt3 16D / 16,384 / 1 | 1.69× [1.64, 1.72] | 16448 → 16448 | 1.3943 → 1.3943 |
| Surrogate/2 | hartmann 3D / 32 / 1 | 1.35× [1.33, 1.39] | 32 → 32 | -3.8626 → -3.8626 |
| Surrogate/2 | branin 2D / 48 / 1 | 1.36× [1.35, 1.41] | 48 → 48 | 0.39789 → 0.39789 |
| Surrogate/2 | ripple 6D / 48 / 2 | 1.26× [1.20, 1.34] | 48 → 48 | 0.74549 → 0.74549 |
| Diversifier | grid 8D / 4,096 / 1 | 1.06× [1.04, 1.09] | 4096 → 4096 | 90.464 → 90.464 |
| Diversifier | cvt 16D / 8,192 / 1 | 1.02× [1.00, 1.04] | 8192 → 8192 | 112.3 → 112.3 |
| L-SHADE | rosen 16D / 4,096 / 1 | 1.03× [1.01, 1.06] | 4096 → 4096 | 17.575 → 17.575 |
| L-SHADE | rotated 48D / 12,288 / 1 | 1.01× [0.99, 1.04] | 12288 → 12288 | 10.552 → 10.552 |
| L-SHADE | sphere 8D / 4,096 / 1 | 1.19× [1.12, 1.25] | 4096 → 4096 | 4.7363e-06 → 4.7363e-06 |
| L-SRTDE | sphere 8D / 4,096 / 1 | 1.12× [1.07, 1.18] | 4096 → 4096 | 0.0051674 → 0.0051674 |
| L-SRTDE | rotated 48D / 12,288 / 1 | 1.09× [1.06, 1.12] | 12288 → 12288 | 89.365 → 89.365 |
| Bounded Nelder–Mead | rosen 16D / 4,096 / 1 | 1.14× [1.02, 1.21] | 4096 → 4096 | 5.7002 → 5.7002 |

Quality is final loss for scalar solvers, hypervolume for ArchiveDE's two-objective cases and MoRetry, IGD on DTLZ2, and QD score for Diversifier. The last two columns report reference → patch. Schedulers with multiple workers can change the order of completions and the corresponding result.

## Changes in the patch

| Area | What changed |
|---|---|
| jSO | Borrow the pending generation during serial evaluation and reuse its score destination. Move selected rows when the population shrinks. |
| CR-FM-NES | Move the encoded population into the original-order snapshot, rank shallow row references and borrow the extended gradient columns. Round odd populations up for mirrored sampling. Add a capped batch entry point with an independently observed incumbent. |
| PGPE | Precompute each mirrored pair's score difference and deviation once per update. Add the capped batch API, which scores a final prefix and updates the distribution only after a complete population. |
| Dual Annealing and ArchiveDE | Reuse the decoded annealing trial row. Move ArchiveDE's selected parent rows. |
| Retry and MoRetry | Inline single-worker coordination, avoid the progress clock when stats are disabled, add cooperative cancellation to advanced retry, specialize p=2 scalarization and add observed-result and post-run-accounting entry points. |
| Diversifier | Reuse decoded rows across complete generations and emitter restarts. Add an exact-cap batch entry point that can commit a final partial batch without an incomplete CMA update. |
| Surrogate/2 | Skip the preliminary scoring scan when fitted LogEI scores the final acquisition pool. |
| Benchmark candidates | Select only the needed ranking prefix in full-update AMaLGaM, avoid copying donor and survivor rows where possible in L-SHADE and L-SRTDE, and reuse bounded Nelder–Mead reflection and shrink storage. |

CR-FM-NES, PGPE and Diversifier still finish full populations by default. Their new opt-in Rust paths consume only the exact number of physically available candidates. The contract tests exercise caps of 1, 31, 33 and nearby population boundaries, callback length errors, observed incumbents and the relay of cancellation to the coordinator. For MoRetry, the accounting return value isolates coordinator-side calls from the inner solver's reported count. Supplying an already observed objective vector removes the redundant physical reevaluation while keeping vector and constraint validation.

## Measurement protocol

The screening build uses release optimization, eight codegen units and incremental compilation. The final build uses the repository's one codegen unit and thin LTO. Rust 1.100.0 on Linux x86-64 and an AMD EPYC 9V74 were used for timings. Single-worker cases used CPU 0. Higher worker counts could use CPUs 0–3. Build and test work used other CPUs. The report records the earlier integrated run separately from the accepted patch.

There are **9,012 paired screening rows**, **1,320 accepted-patch fixed-evaluation rows** and **140 accepted-patch deadline rows**. The screening uses seeds 201–212, randomized A/B order and at least one public entry point. Confirmation uses seeds 1251–1280. Extra controls use identical harness edits in both arms, such as enabling Thompson scoring when measuring T12/T13. Tables are descriptive per-case summaries. Timings shorter than a few milliseconds remain sensitive to host scheduling. Some slots use a narrow prototype of the planned change and are identified in the ledger.

Quality metrics are kept alongside elapsed time and physical callbacks, including post-run multi-objective reevaluations. Full order-sensitive callback traces were replayed at the two endpoint seeds where order is deterministic. The default synchronous GP paths use proposal order for confirmation, independent of callback arrival order. Final multiworker retry work can change scheduling. It is evaluated by total work and quality instead of requiring the same completion order.

A single shared Linux host supplied the timings. Native and research workloads used the bundled lockfile. The per-candidate workload coverage is given in the ledger and raw results. Hardware counters and cross-host reproducibility remain unverified.

## Full accepted-patch confirmation

| Optimizer | Workload, dimension, budget, workers or batch | Speed [95% interval] | Physical calls | Median quality |
|---|---|---:|---:|---:|
| jSO | sphere 128D / 4,096 / 1 | 1.11× [1.10, 1.12] | 4096 → 4096 | 10.847 → 10.847 |
| jSO | cassini2 22D / 8,192 / 1 | 1.01× [1.00, 1.02] | 8192 → 8192 | 30.193 → 30.193 |
| CR-FM-NES | sphere 256D / 4,096 / 1 | 1.05× [1.04, 1.06] | 4096 → 4096 | 1.2666 → 1.2666 |
| CR-FM-NES | rotated 128D / 8,192 / 1 | 1.05× [1.04, 1.06] | 8192 → 8192 | 6.1788 → 6.1788 |
| PGPE | sphere 256D / 4,096 / 1 | 1.04× [0.99, 1.08] | 4096 → 4096 | 100.88 → 100.88 |
| PGPE | rotated_noise 64D / 8,192 / 1 | 1.04× [1.02, 1.06] | 8192 → 8192 | 336.69 → 336.69 |
| PGPE | noise 128D / 4,096 / 1 | 1.06× [1.04, 1.11] | 4096 → 4096 | 45.65 → 45.65 |
| Dual Annealing | rosen 16D / 4,096 / 1 | 1.08× [1.06, 1.10] | 4096 → 4096 | 3.2019e-17 → 3.2019e-17 |
| Dual Annealing | lj 18D / 8,192 / 1 | 1.03× [1.01, 1.05] | 8192 → 8192 | -12.303 → -12.303 |
| ArchiveDE | front 8D / 4,096 / 1 | 1.07× [1.06, 1.08] | 4096 → 4096 | 3.498 → 3.498 |
| ArchiveDE | zdt3 16D / 8,192 / 1 | 1.12× [1.11, 1.14] | 8192 → 8192 | 4.8138 → 4.8138 |
| ArchiveDE | dtlz2 8D / 4,096 / 1 | 1.07× [1.04, 1.08] | 4096 → 4096 | 0.039106 → 0.039106 |
| CR-FM-NES → BiteOpt | sphere 128D / 4,096 / 1 | 1.00× [0.97, 1.02] | 4096 → 4096 | 0.040239 → 0.040239 |
| CR-FM-NES → BiteOpt | cassini2 22D / 8,192 / 1 | 1.00× [0.98, 1.01] | 8192 → 8192 | 28.325 → 28.325 |
| Independent retry | sphere 16D / 2,048 / 1 | 1.69× [1.67, 1.85] | 2048 → 2048 | 7.4614 → 7.4614 |
| Independent retry | ripple 24D / 16,384 / 1 | 1.00× [0.98, 1.02] | 16384 → 16384 | 14.297 → 14.297 |
| Independent retry | ripple 32D / 8,192 / 4 | 1.06× [0.94, 1.24] | 8192 → 8192 | 23.974 → 23.974 |
| Coordinated retry | sphere 16D / 2,048 / 1 | 1.59× [1.54, 1.65] | 2048 → 2048 | 2.6216 → 2.6216 |
| Coordinated retry | ripple 24D / 16,384 / 1 | 1.00× [0.98, 1.04] | 16384 → 16384 | 6.4721 → 6.4721 |
| Weighted multi-objective retry | front 8D / 2,048 / 1 | 2.57× [2.49, 2.81] | 2112 → 2112 | 3.209 → 3.209 |
| Weighted multi-objective retry | zdt3 16D / 16,384 / 1 | 1.69× [1.64, 1.72] | 16448 → 16448 | 1.3943 → 1.3943 |
| Diversifier | grid 8D / 4,096 / 1 | 1.06× [1.04, 1.09] | 4096 → 4096 | 90.464 → 90.464 |
| Diversifier | cvt 16D / 8,192 / 1 | 1.02× [1.00, 1.04] | 8192 → 8192 | 112.3 → 112.3 |
| Surrogate/1 | hartmann 3D / 32 / 1 | 1.00× [0.99, 1.01] | 32 → 32 | -3.8628 → -3.8628 |
| Surrogate/1 | branin 2D / 48 / 1 | 0.99× [0.98, 1.01] | 48 → 48 | 0.39789 → 0.39789 |
| Surrogate/1 | ripple 6D / 48 / 2 | 0.99× [0.97, 1.03] | 48 → 48 | 0.62164 → 0.62164 |
| Surrogate/2 | hartmann 3D / 32 / 1 | 1.35× [1.33, 1.39] | 32 → 32 | -3.8626 → -3.8626 |
| Surrogate/2 | branin 2D / 48 / 1 | 1.36× [1.35, 1.41] | 48 → 48 | 0.39789 → 0.39789 |
| Surrogate/2 | ripple 6D / 48 / 2 | 1.26× [1.20, 1.34] | 48 → 48 | 0.74549 → 0.74549 |
| L-SHADE | rosen 16D / 4,096 / 1 | 1.03× [1.01, 1.06] | 4096 → 4096 | 17.575 → 17.575 |
| L-SHADE | rotated 48D / 12,288 / 1 | 1.01× [0.99, 1.04] | 12288 → 12288 | 10.552 → 10.552 |
| L-SHADE | sphere 8D / 4,096 / 1 | 1.19× [1.12, 1.25] | 4096 → 4096 | 4.7363e-06 → 4.7363e-06 |
| L-SRTDE | sphere 8D / 4,096 / 1 | 1.12× [1.07, 1.18] | 4096 → 4096 | 0.0051674 → 0.0051674 |
| L-SRTDE | rotated 48D / 12,288 / 1 | 1.09× [1.06, 1.12] | 12288 → 12288 | 89.365 → 89.365 |
| AMaLGaM-IDEA | rosen 16D / 4,096 / 1 | 1.03× [1.00, 1.05] | 4096 → 4096 | 15.427 → 15.427 |
| AMaLGaM-IDEA | lj 18D / 8,192 / 1 | 1.03× [1.00, 1.04] | 8192 → 8192 | -9.1746 → -9.1746 |
| iAMaLGaM-IDEA | rosen 16D / 4,096 / 1 | 0.98× [0.91, 1.01] | 4096 → 4096 | 25.09 → 25.09 |
| iAMaLGaM-IDEA | lj 18D / 8,192 / 1 | 0.99× [0.98, 1.02] | 8192 → 8192 | -12.303 → -12.303 |
| Bounded Nelder–Mead | rosen 16D / 4,096 / 1 | 1.14× [1.02, 1.21] | 4096 → 4096 | 5.7002 → 5.7002 |
| Bounded Nelder–Mead | lj 18D / 8,192 / 1 | 1.12× [1.06, 1.14] | 8192 → 8192 | -12.303 → -12.303 |
| jSO → CMA-ES | sphere 8D / 4,096 / 1 | 1.00× [0.99, 1.01] | 4087 → 4087 | 1.614e-12 → 1.614e-12 |
| jSO → CMA-ES | cassini2 22D / 8,192 / 1 | 1.00× [0.98, 1.04] | 8174 → 8174 | 21.24 → 21.24 |
| L-SRTDE → CMA-ES | sphere 8D / 4,096 / 1 | 1.02× [1.01, 1.05] | 4087 → 4087 | 1.9527e-12 → 1.9527e-12 |
| L-SRTDE → CMA-ES | cassini2 22D / 8,192 / 1 | 1.01× [1.00, 1.01] | 8174 → 8174 | 21.226 → 21.226 |

### Equal wall-clock deadlines

Each budget is the reference median of its primary report case. The solver can finish the current batch once the stop predicate fires. The two implementations receive the same evaluation ceiling and wall-clock deadline within a pair.

| Optimizer | Deadline ms | Physical calls reference → patch | Usable runs reference → patch | Median finite quality reference → patch |
|---|---:|---:|---:|---:|
| jSO | 6.117 | 3811 → 4759 | 20/20 → 20/20 | 92.728 → 79.1 |
| CR-FM-NES | 22.438 | 4032 → 4304 | 20/20 → 20/20 | 1.3404 → 1.0494 |
| PGPE | 5.756 | 4368 → 4576 | 20/20 → 20/20 | 100.78 → 100.78 |
| CR-FM-NES → BiteOpt | 27.867 | 4000 → 5194.5 | 20/20 → 20/20 | 14.778 → 12.699 |
| Independent retry | 0.189 | 2064 → 3648 | 15/20 → 19/20 | 7.3757 → 6.5557 |
| Surrogate/1 | 10.902 | 29 → 30 | 20/20 → 20/20 | -3.8628 → -3.8628 |
| Surrogate/2 | 12.749 | 32 → 40 | 20/20 → 20/20 | -3.8627 → -3.8628 |

### Leave-one-change-out checks

These comparisons use the release screening build and unseen seeds 1001–1030. Each reference omits the named change from the integrated implementation.

| Change in the accepted patch | Workload | Speed [95% interval] |
|---|---|---:|
| J11 | sphere 128D / 4,096 / 1 | 1.09× [1.08, 1.10] |
| J11 | cassini2 22D / 8,192 / 1 | 1.00× [0.98, 1.02] |
| M03 | front 8D / 2,048 / 1 | 1.89× [1.85, 1.90] |
| M03 | zdt3 16D / 16,384 / 1 | 1.66× [1.61, 1.71] |
| V14 | grid 8D / 4,096 / 1 | 1.09× [1.07, 1.11] |
| V14 | cvt 16D / 8,192 / 1 | 1.04× [1.03, 1.04] |

## Trial ledger

Each speed factor is the paired screening median for the cases in the same order. Each row includes the reference and candidate's actual median call counts in the evidence JSON. A trace mismatch can indicate changed floating-point order, search policy, or callback arrival order. A stable trace from screening only describes the recorded cases and seeds.

### jSO

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| J01 | Limit generation construction to the number of trials that can still be evaluated. | 0.95× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| J02 | Add a borrowed pending-row iterator and use it for serial evaluation. | 1.09× / 0.99× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| J03 | Move survivors through the existing stable ranking without cloning row contents. | 1.05× / 1.00× | Kept | Selected for the integrated patch after deterministic replay. |
| J04 | Reuse one mutant row across every target in a generation. | 1.03× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| J05 | Apply midpoint repair as each mutant coordinate is written, with unchanged RNG. | 1.02× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| J06 | Recognize already rank-ordered fitness after shrinkage and avoid sorting it. | 0.99× / 0.98× | Held | This candidate tests detecting an already ranked population before sorting. |
| J07 | Order only the p-best prefix, using rank and original index as the total key. | 0.99× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| J08 | Reuse F, CR and improvement lists on the optimizer state. | 1.01× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| J09 | Accumulate the two weighted Lehmer ratios with shared weight resolution. | 1.00× / 0.95× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| J10 | Use the initial row index and copy only when that row is an observed incumbent. | 1.01× / 1.05× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| J11 | Specialize the internal borrowed serial loop and pass reusable scores across generations. | 1.22× / 1.01× | Kept | Selected for the integrated patch after deterministic replay. |
| J12 | Transfer the displaced parent into an archive slot and reuse that slot's old storage across overwrites. | 1.01× / 1.00× | Held | Overwriting an existing parent allocation performed a copy where the reference moves an owned row. |
| J13 | Hoist generation-level F and crossover policy tests outside the target loop. | 0.99× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| J14 | Write scores through indexed in-order Rayon zip into a reused vector. | 1.00× / 1.05× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| J15 | Cap initial jSO population at 128 | 1.04× / 1.00× | Setting tested | Population size is already configurable. Compare the low-budget, high-dimensional result separately. |
| J16 | Minimum population eight | 0.98× / 0.99× | Setting tested | Uses the existing minimum-population setting. |
| J17 | Archive ratio 0.5 | 1.03× / 1.00× | Setting tested | Uses the existing archive-rate setting. |
| J18 | Success memory length ten | 0.99× / 1.02× | Setting tested | Uses the existing memory-size setting. |
| J19 | Final p-best fraction 0.08 | 0.98× / 0.99× | Setting tested | Uses the existing p-best schedule settings. |
| J20 | Test a convex decrease in population size against linear population reduction. | 0.99× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
### CR-FM-NES

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| N01 | Repopulate allocated normal columns for the first half before constructing mirrored columns. | 1.01× / 1.03× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N02 | Use each newly drawn half-column directly and allocate its mirror without cloning the positive column. | 1.01× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N03 | Create y and x in one per-column pass without nalgebra expression temporaries. | 0.80× / 0.84× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N04 | Reuse Fitness's fused decoded and clipped evaluation function. | 1.02× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N05 | Gather references in rank order and traverse ranked vectors indirectly. | 0.99× / 1.05× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N06 | Move encoded rows into the original-order population and reuse shallow row IDs for the ranked view. | 1.03× / 1.08× | Kept | Selected for the integrated patch after deterministic replay. |
| N07 | Compute expensive distance weights only when the movement regime selects them. | 1.00× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N08 | Share squared z norms across distance weighting and the global sigma estimator. | 1.02× / 1.03× | Held | Changing the norm reduction order changed the seeded trajectory. |
| N09 | Stream weighted displacement from ranked x rows while the old mean is intact. | 0.91× / 0.92× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N10 | Borrow y columns and append only the pc/D column to the temporary list. | 1.00× / 1.06× | Kept | Selected for the integrated patch after deterministic replay. |
| N11 | Stream y² and y·vbar directly into the first natural-gradient step. | 0.96× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N12 | Reuse the unconstrained s-step matrix by updating each column for the final stage in place. | 0.93× / 0.97× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N13 | Accumulate paired v and diagonal updates in one column loop. | 0.84× / 0.89× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N14 | Precompute the two coefficients used by distance weights once per finite-count regime. | 0.99× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N15 | Reuse the rank permutation storage while preserving stable finite ties. | 1.00× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| N16 | An opt-in capped batch API consumes a final prefix and returns the best physical observation. | 2.56× / 1.38× / 1.01× | Opt-in | Added as an explicit API. Default calling convention retains the documented behavior. |
| N17 | Return the decoded feasible point associated with the penalized winner, for independent checking. | 0.98× / 1.04× | Held | The prototype clipped the penalized winner. The adopted capped API reports the best raw physical observation instead. |
| N18 | Select an even population nearest to four times sqrt(D). | 1.01× / 1.01× | Setting tested | Population sizing is already selectable. |
| N19 | Start normalized sigma at 0.1. | 1.01× / 1.00× | Setting tested | Uses the existing initial sigma input. |
| N20 | Scale the normalized penalty by decision-coordinate widths relative to their mean. | 1.00× / 0.98× | Held | Search or scheduling policy. No default was selected from this screen. |
### PGPE

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| P01 | Store encoded rows for repeated mirrored asks and write pairwise into existing allocations. | 0.91× / 0.93× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P02 | Construct decoded plus/minus rows in adjacent positions without storing an intermediate generation of encoded rows. | 0.93× / 0.91× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P03 | Fuse clipping in working coordinates and decoding into one pass. | 0.84× / 0.85× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P04 | Store perturbations in a single contiguous pair-major buffer. | 0.93× / 0.92× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P05 | Process one contiguous mirrored perturbation first and accumulate each coordinate in the same pair order. | 1.04× / 1.04× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P06 | Precompute the score difference once for each mirrored pair. | 1.11× / 1.12× | Kept | Selected for the integrated patch after deterministic replay. |
| P07 | Precompute pair-average deviation from the common mean before the coordinate loop. | 1.12× / 1.08× | Kept | Selected for the integrated patch after deterministic replay. |
| P08 | Hoist the standard deviation and its square into each coordinate's inner loop setup. | 1.01× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P09 | Rank descending raw losses directly and construct negated values only in raw-utility mode. | 1.01× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P10 | Retain an index vector across calls to process_scores. | 0.99× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P11 | Update both Adam moment vectors and the center in one pass with its original bias corrections. | 1.01× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P12 | Update the relative stdev clipping into the existing state vector. | 0.98× / 1.09× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P13 | Remove PGPE's duplicate center row and use the Adam center as the sole source for proposals. | 1.00× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P14 | Copy a new incumbent into the already allocated best row. | 1.04× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| P15 | An opt-in capped PGPE batch loop never updates its perturbation distribution from an incomplete pair batch. | 1.28× / 1.04× / 1.00× | Opt-in | Added as an explicit API. Default calling convention retains the documented behavior. |
| P16 | Raw PGPE utilities | 0.97× / 1.00× | Setting tested | Uses the existing ranking switch. |
| P17 | Lower PGPE center learning rate | 0.97× / 0.99× | Setting tested | Uses the existing center learning rate. |
| P18 | Halve the relative stdev limit | 1.02× / 1.01× | Setting tested | Uses the existing deviation-update limit. |
| P19 | PGPE population 64 | 1.01× / 1.00× | Setting tested | Uses the existing population setting. |
| P20 | Score both adjacent PGPE perturbations with the same noise draw in the test callback. | 1.00× / 1.00× | Held | This is a paired-noise callback experiment. The core public evaluator protocol is unchanged. |
### Dual Annealing

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| D01 | Fuse the clamp and decode passes and reuse the decoded destination. | 1.06× / 1.09× | Kept | Selected for the integrated patch after deterministic replay. |
| D02 | Retain the x draw buffer and stream the y draws after all x draws. | 0.95× / 0.96× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D03 | Specialize single-coordinate visiting as two scalar Gaussian draws. | 1.02× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D04 | Compute the temperature-dependent visiting coefficient once per chain. | 1.03× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D05 | Use the full-dimensional visit vector as the proposal instead of making a second allocation. | 0.96× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D06 | Reuse the incumbent buffer in value on every improvement. | 1.01× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D07 | Restore a single mutable finite-difference candidate after each coordinate. | 1.03× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D08 | Use one gradient destination passed into finite differences and swap it with the incumbent gradient each step. | 1.00× / 1.03× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D09 | Store the bounded L-BFGS history in deques so oldest-pair eviction is constant time. | 1.00× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D10 | Use reusable direction and alpha storage for the two-loop recursion. | 1.01× / 1.06× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| D11 | Stop backtracking on exact projected stationarity under deterministic callbacks. | 0.98× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D12 | Use the same allocated candidate row on every Armijo backtrack and swap accepted storage. | 1.01× / 1.05× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| D13 | Reuse the accepted Armijo objective value for the finite-difference base evaluation. | 0.90× / 1.01× | Held | Search or scheduling policy. No default was selected from this screen. |
| D14 | Populate the reserved iteration count with fully completed visiting chains. | 0.99× / 1.01× | Held | Measured contract prototype. The integrated API and regression tests cover the adopted interfaces. |
| D15 | Run local descent after a minimum relative global improvement, while retaining the normal initial descent. | 1.00× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| D16 | Scale normalized finite-difference displacement by the local coordinate magnitude. | 0.99× / 1.01× | Held | Search or scheduling policy. No default was selected from this screen. |
| D17 | Share the base value among one-sided coordinates lying exactly on active bounds. | 1.01× / 1.01× | Held | Search or scheduling policy. No default was selected from this screen. |
| D18 | Reduce projected L-BFGS correction history from six to three. | 1.02× / 0.97× | Held | Search or scheduling policy. No default was selected from this screen. |
| D19 | Use half of the starting visiting temperature. | 1.00× / 1.08× | Held | Search or scheduling policy. No default was selected from this screen. |
| D20 | Reanneal when global visits stagnate for 64 chains. | 1.00× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
### ArchiveDE

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| A01 | Retain survivor decision and objective destinations and swap them after selection. | 1.00× / 1.03× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A02 | Use the existing fixed-width decision-row form at D8. | 1.22× / 1.16× | Setting tested | Fixed decision rows already exist in ArchiveDe. Observed benefit uses the public array type. |
| A03 | Reuse rank permutation storage in each optimizer instance. | 1.03× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A04 | Partially select the needed rank-sum prefix on the large-objective path using the stable index tie-break. | 1.00× / 1.00× | Held | Replayed the large-objective rank-sum branch with ten objectives and tiered survivors disabled. |
| A05 | Replace shifts of active rows with a live bit and preserve earliest ties. | 0.92× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A06 | Use adjacent scratch arrays for active objective ranks and distance cache. | 0.94× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A07 | Choose the unclaimed axis champion during the existing rank-sum traversal. | 1.00× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A08 | Move selected decision/objective rows, preserving survivor rank order. | 0.97× / 0.97× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A09 | Recycle owned dynamic rows from rejected or displaced parents. | 1.12× / 1.12× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| A10 | Retain one result-row Vec through all minimize generations. | 1.02× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A11 | Draw crossover uniforms into a small fixed scratch chunk and consume them in their original order. | 1.00× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A12 | Defer obtaining a parent-row clone until the survivor choice, and preserve archive payload ownership. | 1.07× / 1.06× | Kept | Selected for the integrated patch after deterministic replay. |
| A13 | Promote the three-objective archive to blocks at 128 entries. | 1.03× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A14 | Retain up to two emptied block-row allocations for future skyline splits. | 0.99× / 0.90× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A15 | Look up duplicate-x skyline members once and retain their insertion order. | 0.95× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| A16 | Decode fixed axes directly from their identical bounds. | 0.97× / 0.99× | Held | Replayed non-unit bounds with one fixed coordinate in the targeted control. |
| A17 | Disable archive admission gate | 0.95× / 1.13× | Setting tested | Uses the existing gate switch. |
| A18 | Compare rank-sum-only survivors | 1.14× / 1.04× | Setting tested | Uses the existing tiered-survivor switch. |
| A19 | Scale the population by the number of objectives | 1.01× / 1.09× | Setting tested | Uses the existing population setting. |
| A20 | Downweight pairwise rank separation along redundant objective directions by rank association. | 1.00× / 0.99× | Held | Search or scheduling policy. No default was selected from this screen. |
### CR-FM-NES → BiteOpt

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| H01 | Fast-path serial observed losers with atomic count and incumbent read, and lock only for a new record. | 1.01× / 1.04× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| H02 | Combine the two incumbent-state reads after NES into a single guarded snapshot. | 1.00× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| H03 | Reuse the entire NES score vector between populations. | 0.99× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| H04 | Expose an internal batched decode callback for NES with borrowed reusable rows. | 0.98× / 1.00× | Held | The reused decoded buffer still needs the original batch delivery interface. |
| H05 | Validate decoded coordinates while copying each NES row into the handoff batch. | 1.00× / 1.00× | Held | Validation was folded into per-row preparation. The shape checks happen before starting evaluations. |
| H06 | Test whether an owned incumbent handoff reduces boundary overhead with the existing borrow-taking Bite constructor. | 0.99× / 1.02× | Held | This borrow avoids copying the row but lengthens the mutex scope across stage construction. |
| H07 | Delay the normalized Fitness transform until the NES budget fits a population. | 0.99× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| H08 | Check cooperative cancellation before constructing the next-stage solver. | 0.98× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| H09 | Remove the incumbent lock from target checks when stop_fitness is negative infinity. | 0.99× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| H10 | Avoid retaining a large fit transform for retries that only run the Bite stage. | 0.95× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| H11 | Stop NES immediately on the first finite target and skip the partially evaluated tell. | 0.96× / 0.99× | Held | Immediate target termination violated the existing full-population test. The original contract was preserved. |
| H12 | Record counts before and after the handoff and verify their sum against actual calls. | 1.01× / 1.04× | Held | Stage-count assertions passed. Adding a second diagnostic API grew the surface for little operational gain. |
| H13 | Use a shared population of 128. | 1.40× / 1.03× | Setting tested | Uses the existing shared population input. The two stages both change. |
| H14 | Choose population from the stage budget for four complete NES generations. | 1.16× / 1.02× | Setting tested | Selects the already configurable population from the stage budget. |
| H15 | Assign one-half of each retry's budget to CR-FM-NES. | 1.05× / 1.02× | Setting tested | Uses the existing split option. |
| H16 | Hand off after two NES generations without any new observed incumbent. | 1.00× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| H17 | Jitter a nontrivial handoff center by one percent of coordinate range within bounds. | 1.02× / 0.99× | Held | Search or scheduling policy. No default was selected from this screen. |
| H18 | Use fixed normalized scale 0.08. | 1.00× / 0.99× | Setting tested | Uses the existing sigma option. |
| H19 | Use Bite depth one. | 1.56× / 1.03× | Setting tested | Uses the existing Bite depth option. |
| H20 | Set the Bite population to 128 while retaining the NES population and budget split. | 1.41× / 1.02× | Held | Search or scheduling policy. No default was selected from this screen. |
### Independent retry

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| I01 | Inline the one-worker task and avoid constructing a one-thread Rayon pool. | 1.83× / 0.95× | Kept | Selected for the integrated patch after deterministic replay. |
| I02 | Put the validated lower and upper slices behind one shared allocation for one refcount operation. | 0.93× / 0.89× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| I03 | Reuse the uniform deviation vector across independent run contexts unless the caller owns a clone. | 0.94× / 1.04× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| I04 | Allocate progress capacity lazily and keep default disabled path allocation-free. | 1.01× / 0.94× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| I05 | Avoid reading the elapsed clock when a new minimum cannot produce a stored sample | — | Already in base | The elapsed clock is used only on a new best with progress enabled. At capacity the contract still stores the latest sample, including its timestamp. |
| I06 | In a target-free schedule distribute contiguous run-ID ranges among workers. | 0.98× / 0.74× | Held | Search or scheduling policy. No default was selected from this screen. |
| I07 | Claim four consecutive IDs at once, checking stop before starting every individual retry. | 0.97× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| I08 | Publish exact worker-local counts once per worker when event samples are disabled. | 1.00× / 1.11× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| I09 | Validate width and fitness before locking, while incrementing counts for invalid results. | 1.02× / 0.91× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| I10 | Partition by score before sorting only the retained fraction. | 0.98× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| I11 | Record the incumbent by shared allocation and materialize owned best and entries only once at return. | 0.95× / 0.86× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| I12 | Use a no-progress branch that skips the progress clock and cold helper entirely. | 1.10× / 1.15× | Kept | Selected for the integrated patch after deterministic replay. |
| I13 | Reuse a bounded Rayon pool for repeated retry calls at a fixed worker count. | 0.98× / 1.11× | Inconclusive coverage | The cold process screen includes first pool construction. It cannot establish hot repeated-call reuse gains. |
| I14 | Reduce target-free independent retry results in run-ID order from local worker slots. | 0.99× / 1.08× | Held | Search or scheduling policy. No default was selected from this screen. |
| I15 | Precompute the high stream word that is shared by all worker RNGs. | 1.02× / 0.94× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| I16 | Partition a hard global physical budget into immutable bounded run reservations. | 0.94× / 0.92× | Held | The driver allocated a bounded prefix from exact run IDs. This is a user-closure experiment and adds no new public reservation API. |
| I17 | Pilot half the runs with half budgets, then seed the long runs from retained winners. | 1.00× / 0.94× | Held | Search or scheduling policy. No default was selected from this screen. |
| I18 | Charge a single calibration evaluation, and use one worker for measured sub-50 μs calls. | 1.51× / 0.90× | Held | Search or scheduling policy. No default was selected from this screen. |
| I19 | Use a fixed short/long retry mix of half and one-and-a-half times the nominal per-run budget. | 1.01× / 1.06× | Held | Search or scheduling policy. No default was selected from this screen. |
| I20 | Provide a different stratified guess for each independent run. | 1.01× / 1.08× | Held | Search or scheduling policy. No default was selected from this screen. |
### Coordinated retry

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| R01 | Precompute the inverse widths used across all diversity comparisons within one checkpoint. | 1.05× / 1.24× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| R02 | Compare the normalized squared distance to the squared radius. | 1.00× / 1.54× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| R03 | Stop a monotonic distance sum after it already exceeds the current pruning threshold. | 0.94× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| R04 | Retain the diversity result vector across repeated checkpoints. | 0.94× / 1.30× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| R05 | Skip the final sort when a diversity checkpoint has already put every result in order. | 0.93× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| R06 | Use integer elite-length calculation, preserving ceil(0.2 * count) at ordinary capacities. | 0.99× / 1.03× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| R07 | Snapshot both parents under the store lock and construct crossover vectors outside it. | 1.00× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| R08 | Copy the two selected parents into contiguous scratch outside their original entries. | 0.98× / 1.08× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| R09 | Record coordinate widths in the shared validated global bounds for crossover setup. | 0.95× / 1.12× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| R10 | Reuse crossover guess and standard-deviation destinations per worker. | 0.94× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| R11 | Reuse a worker's existing validated shared bounds for crossover-free starts. | 1.01× / 1.14× | Inconclusive coverage | The baseline already uses two Arc-backed bounds slices. The prototype only moved the cheap clone to a local binding. |
| R12 | Compute the adaptive budget by precomputed slope times run ID. | 1.02× / 1.01× | Inconclusive coverage | The normal screen uses a flat retry multiplier. Half-integer rounding boundaries require a separate replay. |
| R13 | Use insertion rather than a full sort for a small already-ordered prefix. | 0.92× / 0.96× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| R14 | Add cooperative external cancellation between successive coordinated retry claims. | 1.02× / 0.98× | Opt-in | Added as an explicit API. Default calling convention retains the documented behavior. |
| R15 | Keep a ranked elite prefix current before drawing parents between checkpoints. | 0.97× / 1.22× | Held | Search or scheduling policy. No default was selected from this screen. |
| R16 | Check diversity every four claims instead of eight. | 0.96× / 0.91× | Setting tested | Uses the existing checkpoint interval. |
| R17 | Cut the diversity radius from 0.15 to 0.075. | 0.97× / 1.11× | Setting tested | Uses the existing diversity radius. |
| R18 | Prefer the less represented parent neighborhood as the second crossover source. | 1.04× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| R19 | Geometrically allocate per-run budgets while normalizing for the same planned total work. | 0.85× / 0.95× | Held | The rerun used a 3× final ramp. Rounding per-run quotas explains small differences in physical counts. |
| R20 | Adapt crossover probability to a rolling fraction of successful crossovers. | 1.08× / 1.12× | Held | Search or scheduling policy. No default was selected from this screen. |
### Weighted multi-objective retry

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| M01 | Accept an explicitly supplied original-vector observation and avoid a redundant callback. | 1.01× / 1.04× | Opt-in | Added as an explicit API. Default calling convention retains the documented behavior. |
| M02 | Expose the post-run callback count alongside the original result. | 1.01× / 0.93× | Opt-in | Added as an explicit API. Default calling convention retains the documented behavior. |
| M03 | Use multiplication and sqrt for the default p=2 weighted scalarization. | 1.50× / 1.60× | Kept | Selected for the integrated patch after deterministic replay. |
| M04 | Specialize p=1 to a direct weighted sum. | 1.56× / 1.31× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| M05 | Collect positive constraint penalties while computing the powered sum. | 0.95× / 0.93× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| M06 | Validate the immutable weights and exponent when constructing the per-run scalar view. | 0.90× / 1.03× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| M07 | Add optional mutable result scratch to the weighted scalar view and use it in the native evaluator. | 1.06× / 1.06× | Held | The scratch-enabled callback was prototyped with a fixed-size objective row. A public trait change needs confirmation on real vector-valued callbacks. |
| M08 | Normalize and scale each weight in the mutable sampled vector. | 0.99× / 0.95× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| M09 | Compute the constant spans once before dispatching weighted retries. | 1.00× / 0.97× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| M10 | Defer copies of final best x/y until result materialization | — | Already in base | MoStore::into_result already copies the winning decision and objective vector once, at result materialization. |
| M11 | Partition by scalar score before sorting only the kept prefix. | 1.02× / 0.85× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| M12 | Reuse the uniform-deviation buffer for each worker's serial sequence. | 1.00× / 0.89× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| M13 | Validate returned vector shape, finiteness and value limits before taking the shared store. | 1.02× / 1.05× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| M14 | Use a 2D skyline pass for pareto_indices postprocessing | — | Already in base | pareto_indices already uses its stable two-objective skyline pass and preserves equal nondominated rows. |
| M15 | Limit higher-dimensional dominance checks to points whose first objective can dominate the query. | 1.01× / 0.90× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| M16 | Sample two-objective directions on successive strata of the positive simplex. | 0.99× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| M17 | Reserve every eighth run for one of two extreme weight vectors. | 1.00× / 1.01× | Held | Search or scheduling policy. No default was selected from this screen. |
| M18 | Calibrate objective ranges using 64 charged deterministic points before scalarized retries. | 0.92× / 1.03× | Held | Search or scheduling policy. No default was selected from this screen. |
| M19 | Use nondominance before scalar score when pruning a full store. | 0.89× / 0.92× | Held | Search or scheduling policy. No default was selected from this screen. |
| M20 | Assign alternating direction weights to large current gaps in a two-objective result set. | 0.93× / 0.94× | Held | Search or scheduling policy. No default was selected from this screen. |
### Diversifier

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| V01 | Add an opt-in capped batch entry point and commit its last prefix without an incomplete CMA tell. | 1.87× / 0.92× / 0.99× | Opt-in | Added as an explicit API. Default calling convention retains the documented behavior. |
| V02 | Read flat decoded CMA rows and adapt each row once at the owned batch boundary. | 1.00× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V03 | Reuse both produced score buffers in the Diversifier and keep the public archive adapter. | 0.97× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V04 | Find the best in the generation by index and copy its row at most once. | 1.00× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V05 | Retain two score buffers for consecutive sorted stall comparisons. | 0.98× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V06 | Share a stable improvement ranking between the Diversifier stall test and CMA tell. | 0.98× / 0.97× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V07 | Skip stall-score preparation on the final generation when no later comparison can consume it. | 1.01× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V08 | Construct the normalized Fitness bounds once, then clone the template per emitter. | 1.01× / 1.05× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V09 | Borrow a selected elite row to initialize the emitter. reuse that choice in the owned public sampler. | 1.03× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V10 | Enable the existing changed-niche merge path for 256-cell archives. | 1.00× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V11 | Reuse the scalar-to-batch vector in the Diversifier adapter by taking it back after commit. | 1.09× / 1.02× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| V12 | Test fixed two-descriptor rows and materialize Vec only at the public batch interface. | 1.00× / 1.07× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V13 | Compute CVT niche IDs in parallel, then apply updates in original input order. | 0.74× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| V14 | Reuse the decoded row workspace across emitter restarts while constructing each distribution independently. | 1.11× / 1.06× | Kept | Adopted reuse of decoded row storage and fresh per-emitter distribution state. The full covariance state is reconstructed. |
| V15 | Use a two-niche tournament to favor the less visted start niche. | 1.02× / 0.98× | Held | Search or scheduling policy. No default was selected from this screen. |
| V16 | Draw a log-uniform restart sigma over the prior squared-width range. | 0.98× / 0.97× | Held | Search or scheduling policy. No default was selected from this screen. |
| V17 | Double the stall window from 8 to 16 at P32. | 1.00× / 1.02× | Setting tested | Uses the existing stall setting. |
| V18 | Scale same-niche improvement by a stable positive relative loss reference. | 0.99× / 1.01× | Held | The trial uses a relative improvement against the prior elite value as a scale proxy. |
| V19 | Choose one uniform search start out of ten, otherwise use an archived elite. | 1.03× / 1.01× | Held | Search or scheduling policy. No default was selected from this screen. |
| V20 | Let coverage growth reset the stall timer in addition to improvement scores. | 0.84× / 1.01× | Held | Search or scheduling policy. No default was selected from this screen. |
### Surrogate/1

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| G01 | Use bounded stack scratch for scaled coordinates, cross-covariance and forward substitution. | 1.03× / 1.03× | Held | The small posterior stack workspace was replayed. The measured gain was below the screening target. |
| G02 | Carry acquisition values into refinement along with start indices. | 1.01× / 1.04× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G03 | Find and sort the four best acquisition starts without sorting the full pool. | 1.00× / 1.04× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| G04 | Maintain validated, normalized physical observations in an incremental per-run cache. | 0.99× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G05 | Use a kernel value-only path for GP predictions. | 1.00× / 0.94× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G06 | Predict blocks of eight posterior right-hand sides against the same Cholesky factor. | 1.02× / 0.96× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G07 | Reuse inverse-row scratch for the serial analytic gradient and accumulate each row immediately. | 1.00× / 1.04× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G08 | Call the optimizer directly for one MAP restart and preserve parallel restarts when requested. | 1.00× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G09 | Fill packed covariance in blocks of eight training rows with the same arithmetic. | 1.00× / 0.95× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G10 | Order only the three required history centers. | 0.99× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G11 | Append fantasies in place when the caller owns the disposable local GP. | 0.98× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G12 | Reuse scaled squared-distance contributions outside the active coordinate during both refinement directions. | 0.98× / 1.10× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| G13 | Short-circuit squared-distance exclusion once the candidate is already far enough. | 0.99× / 1.12× | Held | The screening gain was absent in the integrated thin-LTO confirmation. The accepted patch retains the original exclusion scan. |
| G14 | Switch between Rayon and a serial candidate scan at a measured size cutoff. | 1.03× / 0.89× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| G15 | Refit every sixth round | 1.04× / 1.09× | Setting tested | Uses the existing refit interval. |
| G16 | Cut coordinate refinement from 12 sweeps to six. | 1.24× / 1.33× | Held | Search or scheduling policy. No default was selected from this screen. |
| G17 | Use lower confidence bound instead of log-EI | 1.07× / 1.04× | Setting tested | Uses the existing acquisition setting. |
| G18 | Run two MAP restarts | 0.88× / 1.04× | Setting tested | Uses the existing MAP restart count. |
| G19 | Let stalled trust regions spend the next slot globally while preserving a periodic global fallback. | 0.97× / 1.03× | Held | Search or scheduling policy. No default was selected from this screen. |
| G20 | Run serial batches on cheap objectives and use a latency-aware q limit on skewed callbacks. | 0.82× / 0.82× | Setting tested | Uses the existing batch width. Latency governs the benchmark's choice. |
### Surrogate/2

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| T01 | Cache each unit observation by ID and compare the physical row before reusing it. | 0.99× / 0.91× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T02 | Fast-path already ID-ordered usable histories and sort only out-of-order arrivals. | 1.02× / 1.01× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T03 | Reduce the usable range in a single observation pass instead of allocating a loss vector. | 1.01× / 1.01× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T04 | Compute each normalized distance to the incumbent a single time before ordering training rows. | 1.00× / 0.96× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T05 | Use an explicit bitmap for training sample membership. | 1.00× / 1.16× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T06 | Seed minimum separation with the precomputed incumbent distance and update only live candidates. | 0.98× / 0.98× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T07 | Memoize unit-space pending rows in a small coordinate-keyed index across proposals. | 0.99× / 0.98× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T08 | Share immutable packed Cholesky storage between cloned GP factors and copy only for an append. | 0.99× / 0.99× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T09 | Apply the existing multi-observation Cholesky append loop to one owned mutable model. | 1.00× / 1.01× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T10 | Use a bounded completion queue with exact capacity for messages in flight. | 1.00× / 0.93× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T11 | Give synchronous and asynchronous proposal workers an immutable Arc snapshot of the history. | 0.98× / 1.00× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| T12 | Optimize random-feature path evaluation by validating coordinates without a redundant GP solve. | 0.92× / 0.92× | Held | Rebenchmarked with the Thompson path enabled in both arms. |
| T13 | Evaluate each retained Thompson candidate block in parallel and keep original scoring order. | 0.90× / 1.50× | Held | Rebenchmarked with Thompson and two numerical threads. The result depends on proposal size and thread dispatch. |
| T14 | Reuse pools for explicit advanced-run thread counts within a bounded eight-thread workspace. | 0.92× / 1.00× | Inconclusive coverage | The cold process screen includes first pool construction. Pool sharing needs nested and concurrent caller validation. |
| T15 | Asynchronous versus synchronous execution under skewed evaluation latency. | 1.47× | Setting tested | Uses the existing concurrent async path and an equal thread reservation for the latency test. |
| T16 | Raise the active training cap from 16 to 32. | 0.86× / 0.75× | Setting tested | Uses the existing training cap. |
| T17 | Use a smoothed proposal-to-evaluation overhead ratio in the adaptive governor. | 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| T18 | Use the Sobol digital-shift initial design. | 0.98× / 1.01× | Setting tested | Uses the existing initial-design setting. |
| T19 | Restart after four times the dimension without improvement, minimum four failures. | 1.00× / 1.00× | Setting tested | Uses the existing region restart threshold. |
| T20 | Adjust BO-to-BiteOpt handoff after a prespecified half-budget point with a shorter remaining-population gate. | 2.16× / 2.00× | Setting tested | This evaluates a fixed handoff threshold with a smaller remaining-population minimum. The adaptive rule remains untested. |
### L-SHADE

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| L01 | Reuse the mutant scratch row across targets and generations. | 1.04× / 1.04× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L02 | Initialize the trial directly from the crossover mask, with one write for each coordinate. | 0.99× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L03 | Apply bound repair as each mutant coordinate is written. | 0.99× / 0.95× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L04 | Order only the p-best prefix to construct the generation. | 0.99× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L05 | Reserve initial row metadata only for the initial population prefix that fits the budget. | 1.10× / 1.20× | Kept | Selected for the integrated patch after deterministic replay. |
| L06 | Retain pending trial metadata allocation across shrinking generations. | 1.01× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L07 | Move surviving owned rows across the rank permutation. | 1.07× / 1.01× | Kept | Selected for the integrated patch after deterministic replay. |
| L08 | Hoist archive capacity calculation outside the accepted-trial loop. | 1.00× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L09 | Overwrite full archive entries using their provided allocation. | 0.99× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L10 | Use shared normalization decisions and a single Lehmer iteration for both parameters. | 1.01× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L11 | Store successful F, CR and improvements in a single contiguous record vector. | 0.98× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L12 | Reuse sorted rank-index storage at generation boundaries. | 0.98× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L13 | Copy only incumbent coordinates into the candidate evaluator's reusable result row. | 1.00× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| L14 | Specialize the fixed-width eight-coordinate mutation workspace. | 1.31× / 1.19× | Kept | Selected for the integrated patch after deterministic replay. |
| L15 | Decrease L-SHADE's initial population from 18D to 12D. | 1.02× / 1.03× | Setting tested | Uses the existing population multiplier. |
| L16 | Halve the archive ratio. | 1.00× / 1.00× | Setting tested | Uses the existing archive size input. |
| L17 | Double memory cells from six to twelve. | 0.98× / 1.00× | Setting tested | Uses the existing memory-size input. |
| L18 | Increase p-best fraction to 0.18. | 0.99× / 1.01× | Setting tested | Uses the existing p-best fraction. |
| L19 | Narrow the scale-factor Cauchy spread. | 1.01× / 1.03× | Setting tested | Uses the existing scale-factor input. |
| L20 | Use a convex reduction schedule for the L-SHADE population. | 0.96× / 0.94× | Held | Search or scheduling policy. No default was selected from this screen. |
### L-SRTDE

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| S01 | Retain the rank selector between generations while its size stays the same. | 1.00× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S02 | Use swap cycles to apply the sorted permutation without moving values through a second population. | 0.95× / 0.96× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S03 | Merge stable sorted winners with the already sorted top list and stop after n_next rows. | 1.03× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S04 | Move newest-population survivors from their old slots when shrinking. | 1.11× / 1.05× | Kept | Selected for the integrated patch after deterministic replay. |
| S05 | Avoid sorting or moving rows when all fitness values already appear in total order. | 1.00× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S06 | Fill a reusable mutant row across trials. | 1.06× / 1.05× | Kept | Selected for the integrated patch after deterministic replay. |
| S07 | Repair each crossover-selected mutant in its destination row before evaluation. | 1.02× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S08 | Build the two owned population entries from old storage whenever a cyclic replacement accepts. | 1.01× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S09 | Reuse the success-rate derived F mean and p-best fraction when the previous feedback remains unchanged. | 1.00× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S10 | Enumerate eligible donor triples at N=4 and sample their conditional L-SRTDE rank weights. | 0.98× / 1.02× | Held | Search or scheduling policy. No default was selected from this screen. |
| S11 | Replace repeated modulo in the cyclic newest-population insertion path with an equality check. | 1.00× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S12 | Reuse success-history arrays across generations. | 1.01× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S13 | Generate and reuse a bit mask for binomial crossover before filling the target row. | 0.90× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S14 | Partition the top survivor pool to its final size before applying its stable rank order. | 0.86× / 0.94× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| S15 | Swap the L-SRTDE difference form at matched problems and seeds. | 0.89× / 1.03× | Setting tested | Uses both shipped research variants with the same workloads and seeds. |
| S16 | Reduce rank pressure from three to two. | 1.01× / 1.02× | Setting tested | Uses the existing rank-pressure input. |
| S17 | Use initial newest/top populations of 12D. | 1.02× / 1.00× | Setting tested | Uses the existing population multiplier. |
| S18 | Raise minimum population from four to eight. | 1.01× / 1.01× | Setting tested | Uses the existing minimum-population input. |
| S19 | Smooth the success-rate update across two successive generation statistics. | 0.98× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| S20 | Reduce CR memory response by half. | 1.01× / 1.01× | Held | Search or scheduling policy. No default was selected from this screen. |
### AMaLGaM-IDEA

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| F01 | Specialize full covariance updates at eta_sigma=1 while preserving incremental semantics. | 1.00× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F02 | Use direct mean displacement when the mean-shift learning rate is exactly one. | 0.96× / 1.04× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F03 | Select and sort the truncation prefix while retaining equal-fitness index order. | 1.06× / 1.06× | Kept | Selected for the integrated patch after deterministic replay. |
| F04 | Fill centered values in column-major order to improve writes. | 1.02× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F05 | Use a symmetric lower-triangle covariance kernel and mirror its result. | 1.05× / 1.03× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| F06 | Move an owned covariance clone into Cholesky while preserving the original input for ridge retries. | 0.96× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F07 | Apply the already implemented lower triangular multiply to each sample column. | 0.76× / 0.86× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| F08 | Check direct sample coordinates against the box while writing them. | 1.03× / 1.08× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| F09 | Cap the doubled sample workspace by multiplying in blocks of 64 columns. | 1.01× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F10 | Reuse a ridge retry input before each Cholesky fallback. | 0.96× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F11 | Skip the initial variance estimate when the selected set has one finite value. | 1.02× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F12 | Use contiguous blocks of eight coordinates when accumulating successful sample means. | 1.00× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F13 | Remember AMS prefix swaps and undo just those after selecting the sampled rows. | 1.03× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F14 | Use a direct box-compliance flag for the common wide-box draw and keep the old redraw routine. | 1.00× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| F15 | Reduce the fraction selected from 0.35 to 0.25. | 1.02× / 1.10× | Setting tested | Uses the existing selection fraction. |
| F16 | Reduce the full-update start population. | 1.01× / 1.27× | Setting tested | Uses the existing population override. |
| F17 | Limit Gaussian redraws to four. | 1.03× / 1.18× | Setting tested | Uses the existing redraw limit. |
| F18 | Try covariance shrinkage toward its diagonal before the existing ridge retries. | 1.03× / 1.05× | Held | Search or scheduling policy. No default was selected from this screen. |
| F19 | Increase the AMS displacement from two to three. | 0.99× / 1.01× | Setting tested | Uses the existing anticipated mean shift. |
| F20 | Use 1.5 rather than 2 as restart-population growth. | 0.98× / 1.08× | Setting tested | Uses the existing restart-population multiplier. |
### iAMaLGaM-IDEA

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| E01 | Blend covariance memories in one in-place element pass. | 0.97× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E02 | Blend the lower triangle and reflect it before factorization. | 0.99× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E03 | Swap already allocated current and previous mean buffers. | 1.02× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E04 | Zip the current mean, previous mean and aged mean-shift slices in one loop. | 1.04× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E05 | Compute small-sample covariance column dots directly when selection uses fewer rows than dimensions. | 1.09× / 1.10× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| E06 | Update the factor for each selected residual using the previous factor when its pivots are safe. | 0.72× / 0.74× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| E07 | Factor the scaled previous covariance once, then incorporate rank-one selected sample contributions. | 0.78× / 0.80× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| E08 | Apply a simple small-cohort triangular sampler when the matrix has at most 64 columns. | 0.79× / 0.79× | Held | At least one replay digest differed. Parallel callback arrival can change the screening digest. |
| E09 | Delay diagonal inspection until unregularized Cholesky has actually failed. | 1.01× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E10 | Precompute the normalized sample weight used in maximum-likelihood covariance across a restart. | 1.02× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E11 | Copy contiguous selected rows without repeatedly multiplying their original indices. | 0.99× / 1.02× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E12 | Check whether all selected finite scores are identical before the variance calculation. | 1.02× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E13 | Prepare the SDR right-hand side from the sample mean and accumulator slices in one traversal. | 0.89× / 0.96× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E14 | Reuse the perturbation row itself for narrow-box redraws. | 0.94× / 0.95× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| E15 | Halve the covariance learning rate while holding the mean response fixed. | 1.02× / 0.89× | Held | Search or scheduling policy. No default was selected from this screen. |
| E16 | Double the incremental mean-shift response with a cap at one. | 0.94× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| E17 | Increase incremental start population from 10√D to 14√D. | 1.07× / 1.03× | Setting tested | Uses the existing population override. |
| E18 | Blend a regularized share of the previous restart covariance into the first incremental update. | 0.98× / 0.99× | Held | The restart covariance is retained only when the restart actually occurs. These short runs gave little restart coverage. |
| E19 | Use a diagonal covariance for the first incremental sampling generation. | 0.99× / 0.79× | Held | Search or scheduling policy. No default was selected from this screen. |
| E20 | Restart if a Cholesky pivot ratio signals persistent covariance collapse. | 0.98× / 1.01× | Held | Search or scheduling policy. No default was selected from this screen. |
### Bounded Nelder–Mead

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| B01 | Reuse centroid coordinate scratch across simplex steps. | 1.03× / 1.05× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| B02 | Keep the simplex ordered by moving only a newly accepted worst vertex through the ordered prefix. | 1.04× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| B03 | Update centroid from the last replaced vertex and the new worst after ordering. | 1.64× / 1.08× | Held | Centroid accumulation changed rounding and candidate traces. This variant needs a separate trajectory-quality gate. |
| B04 | Borrow the worst vertex during proposal construction. | 1.11× / 1.04× | Kept | Selected for the integrated patch after deterministic replay. |
| B05 | Reuse the reflection slot and exchange ownership with the replaced vertex. | 1.16× / 1.02× | Kept | Selected for the integrated patch after deterministic replay. |
| B06 | Keep a second proposal row for expansion and contraction across simplex iterations. | 1.03× / 1.04× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| B07 | Write shrunken vertices in place after checking that an evaluation slot remains. | 0.98× / 1.15× | Kept | Selected for the integrated patch after deterministic replay. |
| B08 | Keep a separate ordered rank index during a simplex iteration, moving row storage only once. | 0.63× / 0.73× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| B09 | Reuse the simplex vector when a bounded descent needs another restart. | 0.98× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| B10 | Skip clamp writes for probes already inside the bounds. | 0.98× / 0.94× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| B11 | Allocate the scaled per-axis restart step once and refill it. | 0.99× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| B12 | Keep a separate record of the best physical evaluation even on partial simplex construction. | 0.94× / 0.94× | Held | Measured contract prototype. The integrated API and regression tests cover the adopted interfaces. |
| B13 | Add an experimental early exit when the residual cannot form another simplex. | 1.04× / 0.99× | Held | The proposed early return would change the refiner's exact-budget contract. |
| B14 | Reverse a clamped axis step before evaluating the coincident start vertex. | 1.00× / 1.02× | Held | Search or scheduling policy. No default was selected from this screen. |
| B15 | Use an initial step of 8% of each coordinate's range. | 0.98× / 1.01× | Setting tested | Uses a different caller-supplied initial simplex step. |
| B16 | Use the classical expansion, contraction and shrink constants instead of Gao–Han. | 0.97× / 1.13× | Held | Search or scheduling policy. No default was selected from this screen. |
| B17 | Restart when projection creates an exactly duplicate initial simplex vertex. | 1.09× / 0.98× | Held | Search or scheduling policy. No default was selected from this screen. |
| B18 | Adapt the shrink scale between restarts to the magnitude of the last accepted displacement. | 0.99× / 0.98× | Held | Search or scheduling policy. No default was selected from this screen. |
| B19 | Evaluate ordered shrink candidates collected as one batch of budget-fitting rows. | 0.92× / 0.81× | Held | Search or scheduling policy. No default was selected from this screen. |
| B20 | Use an orthogonal two-axis rotation for initial simplex directions. | 1.01× / 0.99× | Held | Search or scheduling policy. No default was selected from this screen. |
### jSO → CMA-ES

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| K01 | Use a no-allocation pass-through for already feasible rows in the sequence objective guard. | 1.04× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| K02 | Avoid a clone when promoting the first-stage best into the second stage | — | Already in base | The first-stage Vec already moves into the sequence. The second constructor borrows its slice. |
| K03 | Pass one broadcast sigma to the unchanged CMA constructor. | 1.00× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| K04 | Reuse cached normalized bound metadata up to the CMA initialization. | 1.00× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| K05 | Avoid CMA construction if the residual cannot fill its default population. | 0.99× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| K06 | Calculate the 2/5 split without overflowing u64. | 1.05× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| K07 | Substitute the library jSO for the candidate first stage, using the same first-stage seed and budget. | 0.97× / 0.99× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| K08 | Move the observer's owned final row instead of cloning it for a stage-boundary result. | 0.99× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| K09 | Reuse a thread-local stage reporting row over successive coordinated solves. | 1.00× / 0.98× | Inconclusive coverage | This reuses reporting space in a process-local cache. The screen is a cold-process measurement. |
| K10 | Capture actual first/second-stage counts and both observed minima at the wrapper boundary. | 1.01× / 1.01× | Held | The stage-count prototype checked that first- and second-stage physical counts sum to the result. |
| K11 | Propagate a deterministic external stop through jSO asks and second-stage callback admission. | 1.19× / 2.73× | Held | This is a deterministic callback-count stop probe across both stages. It changed physical calls. |
| K12 | Move the 2/5 split to the nearest earlier split leaving a whole number of CMA populations. | 1.00× / 1.05× | Held | Search or scheduling policy. No default was selected from this screen. |
| K13 | Give jSO 25 percent of the total budget. | 1.00× / 0.99× | Held | Search or scheduling policy. No default was selected from this screen. |
| K14 | Set handoff from a rolling fraction of first-stage children improving on the observed record. | 0.96× / 1.01× | Held | Search or scheduling policy. No default was selected from this screen. |
| K15 | Trigger handoff when the reference-width normalized full population diameter collapses. | 0.91× / 1.02× | Held | Search or scheduling policy. No default was selected from this screen. |
| K16 | Estimate the CMA initial diagonal scales from the boundary population spread. | 0.97× / 0.99× | Held | Search or scheduling policy. No default was selected from this screen. |
| K17 | Limit the second-stage sigma by normalized distance of the incumbent to the nearest bound. | 0.99× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| K18 | Start the refinement at the coordinatewise median of the best five evaluated first-stage rows. | 0.95× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| K19 | Try CMA population 16 only when called through the sequence. | 0.94× / 1.04× | Held | Search or scheduling policy. No default was selected from this screen. |
| K20 | Run the existing JsoCmaZ0 arm with repair_z disabled. | 1.00× / 0.96× | Setting tested | Uses the harness's existing repair_z off arm. |
### L-SRTDE → CMA-ES

| Trial | Treatment | Median speed | Decision | Reason |
|---|---|---:|---|---|
| Z01 | Use a no-allocation pass-through for already feasible rows in the sequence objective guard. | 1.04× / 1.01× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| Z02 | Tie stage-one limit directly to its exact call counter | — | Already in base | The candidate evaluator already reserves calls under the exact first-stage cap, including the final partial generation. |
| Z03 | Move the returned top candidate to the second-stage initializer | — | Already in base | LSrtdeCma already moves the first stage's returned Vec into the handoff. |
| Z04 | Cache normalized bounds across repeated sequence initializations on one worker. | 0.95× / 0.97× | Inconclusive coverage | The cold process test measures the cost of installing the cache. A hot repeated-call benchmark is needed to judge amortization. |
| Z05 | Avoid CMA construction if the residual cannot fill its default population. | 1.02× / 1.19× | Held | Promising in a screen. The measured scope or integration trade-off did not justify shipping it. |
| Z06 | Calculate the 2/5 split without overflowing u64. | 0.99× / 0.98× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| Z07 | Reuse cached normalized bound metadata up to the CMA initialization. | 1.01× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| Z08 | Move the observer's owned final row instead of cloning it for a stage-boundary result. | 0.98× / 1.00× | Held | No repeatable 5% gain in the primary screening cells with the expected trace. |
| Z09 | Reuse cleared first-stage success-history buffers across independent L-SRTDE runs. | 1.02× / 0.99× | Inconclusive coverage | The trial reuses first-stage success buffers. Population and cyclic state were reconstructed. |
| Z10 | Capture actual first/second-stage counts and both observed minima at the wrapper boundary. | 1.00× / 1.00× | Held | The stage-count prototype checked that first- and second-stage physical counts sum to the result. |
| Z11 | Propagate cancellation into the L-SRTDE donor loop and stop before starting more callbacks. | 1.17× / 2.83× | Held | This is a deterministic callback-count stop probe including a check in the donor loop. Concurrent stops during sampling remain unverified. |
| Z12 | Use current-target instead of base-target L-SRTDE differences before the unchanged CMA stage. | 1.03× / 1.06× | Held | Search or scheduling policy. No default was selected from this screen. |
| Z13 | Give L-SRTDE a 55 percent first-stage share. | 1.03× / 1.02× | Held | Search or scheduling policy. No default was selected from this screen. |
| Z14 | Shift to stage two when newest-population acceptance stays below 5%. | 0.99× / 1.02× | Held | Search or scheduling policy. No default was selected from this screen. |
| Z15 | Shift to stage two when top and newest median fitness is effectively equal. | 1.01× / 1.00× | Held | Search or scheduling policy. No default was selected from this screen. |
| Z16 | Measure handoff centered on the last strictly improving newest-population trial. | 0.97× / 0.99× | Held | Search or scheduling policy. No default was selected from this screen. |
| Z17 | Set local sigmas using the handoff displacement from the center of each normalized interval. | 1.23× / 0.90× | Held | The trial uses displacement from the box midpoint as a scale proxy. Access to newest/top separation would require additional state. |
| Z18 | Split the final budget into two CMA runs from separated bounded centers. | 1.00× / 0.97× | Held | The prototype split the residual between the incumbent and a reflected start. The original top-row selection remains untested. |
| Z19 | Shift the first-stage endpoint to use the second-stage whole-population remainder. | 1.03× / 0.95× | Held | Search or scheduling policy. No default was selected from this screen. |
| Z20 | Keep the first solver for the full budget if the second stage would run fewer than four complete generations. | 6.27× / 1.02× | Held | Search or scheduling policy. No default was selected from this screen. |

### Additional measured change

T00 removed repeated LogEI candidate scoring in surrogate/2.  The screening speed factors were 1.28× [1.10, 1.38] and 1.25× [1.07, 1.47]. The change is present in the accepted patch.

## Verification of the packaged patch

| Check | Result |
|---|---|
| Patch application | Applied to a fresh checkout of the exact base with whitespace errors enabled. All 17 changed files matched the tested worktree byte for byte. |
| Rust workspace | 445 passed, including six new focused contract regressions. |
| Research crates | 49 passed in the focused candidate workspace. Three additional boundary harness tests passed. |
| Installed Python binding | 125 passed on CPython 3.12 with a rebuilt and installed wheel. |
| Real job worker | The job API and both surrogate tutorials passed. Surrogate/2 checked a 16-call native simulation, a four-call warm start and a timed-out job. |
| Toolchain and repository checks | Formatting, workspace Clippy with warnings denied, Rust 1.88 workspace check, documentation consistency and Git diff checks all passed. |
| Replayed final inputs | All 86 deterministic endpoint replay comparisons matched, including synchronous acquisition order and result quality. Multithreaded callback ordering was excluded where it can vary. |

The patch SHA-256 is `b008f3e661cec2e629d5f9dafa0c626215b688b6fef6a5d752321c8ea7db558a`. The archive provides a SHA-256 manifest, the exact plan, final and intermediate timings, reproducible runner sources and a CSV version of the trial ledger.
