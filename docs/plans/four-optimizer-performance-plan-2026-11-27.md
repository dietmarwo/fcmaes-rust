# fcmaes-rust performance check and optimization plan

**27 November 2026 · BiteOpt, DE, MODE and MAP-Elites · 20 experiments each**

I built and timed the native implementations at `d7ff66aebc388ac19b698742c7678986e7a48d58`, which matched the remote `main` at the time of the check. The baseline covers native execution cost and the quality achieved at the measured evaluation budget. The 80 proposed changes begin after the baseline tables.

| Optimizer | Measured signal | First experiment |
|---|---|---|
| BiteOpt | At d=96, raising P from 32 to 297 took time per shifted-Sphere call from 1.702 to 4.319 µs in the explicit-P sweep. | B01/B02. Replay accepted insertions and dirty centroid rebuilds. Measure how much of the live-run change each explains. |
| DE | A constant objective of 7 yielded `f64::MAX` in serial, batch and delayed DE. On 32-D shifted Sphere, direct DE took 0.242 µs/call versus 7.450 µs/call with eight in flight on eight threads. | D01 isolates and corrects best-ever tracking. D02 targets row copying. D13–D16 target ordered dispatch overhead for cheap callbacks. |
| MODE | At P=256 on ZDT1, `tell` consumed 95% of end-to-end time with NSGA-II-style reproduction and 91% with DE reproduction. | M01/M02. Reduce survivor copies and test exact two-objective **all-front** ranking against the current quadratic kernel. |
| MAP-Elites | At K=1,024, a prepared uniform descriptor lookup cost 22 ns on the regular grid and 1,797 ns with CVT. Sorting a frozen K=4,096 grid cost 51.6 µs per sort. | Q01 for ranking occupied grid cells. Q03/Q04 for contiguous CVT scans and an exact nearest-center index with a measured crossover. |

## What ran

The release build used rustc 1.100.0, LLVM 23.1.5, x86-64 Linux on a virtualized AMD EPYC 9V74, the repository benchmark lockfile, thin LTO and one codegen unit. All calls used native Rust objectives. Each configuration had one discarded warm-up seed (100), five measured seeds (101–105), and three repetitions per seed in different seed orders. Sequential processes used affinity CPU 0 and a single-thread Rayon pool. The eight-thread cases used CPUs 0–7. No candidate optimization was applied. The benchmark source, raw rows and build/run instructions are in the appendices.

Each reported value is the median of the five per-seed medians. Time per call divides each run's wall time by its **actual** evaluations before aggregation. MODE's phase percentages use within-run ratios. Scalar timing covers construction, optimization, result and teardown within the measured block. MODE times the generation loop. MAP-Elites reports archive construction plus uniform parent seeding separately from the full emitter loop. Scoring and diagnostic replay are outside that loop. This was a shared VM. Across the three sweeps, the per-sweep median for 32-D direct BiteOpt ranged from 1.36–1.49 µs/call, and for eight-thread delayed DE on the same objective it ranged from 6.71–9.84 µs/call. These numbers locate large costs. Small differences require interleaved A/B confirmation.

### BiteOpt and DE

Direct BiteOpt used the native default P=9+3d. DE used P=32, uniform box sampling, default mutation/age settings and a nominal 8,192-evaluation limit. Shifted Sphere uses an eight-coordinate off-center pattern repeated to the required dimension. Cassini1 uses the repository's six-dimensional GTOP function and published bounds, with one optimization per repetition and no target stop. The objective's computation is included in both methods. The preliminary centered-Sphere test reached zero with BiteOpt in all five seeds. The first exact zero at d=32 appeared at evaluations 2,590–3,246. Every first evaluation was nonzero. Shifted Sphere is the scalar diagnostic used below.

| Problem | BiteOpt P | BiteOpt µs/call | DE µs/call | BiteOpt median reported best | DE median reported best | Actual DE calls (range) |
|---|---:|---:|---:|---:|---:|---:|
| Shifted Sphere d=8 | 33 | 0.641 | 0.106 | 3.65e-30 | 0 | 8,193–8,244 |
| Shifted Sphere d=32 | 105 | 1.460 | 0.242 | 0.000967 | 5.85e-07 | 8,193–8,245 |
| Shifted Sphere d=96 | 297 | 4.132 | 0.628 | 15.4 | 33 | 8,205–8,238 |
| Cassini1 | 27 | 6.234 | 5.682 | 11 | 15.3 | 8,204–8,223 |

Direct DE can finish its current generation after crossing the nominal limit. The 20 independently audited direct runs on these problems used 8,193–8,245 calls. BiteOpt used exactly 8,192. The quality columns describe the returned results at those counts. The gap between methods on Cassini1 was smaller in time per call than on cheap objectives. The study did not time Cassini1 alone to derive an optimizer-only percentage.

| 32-D shifted Sphere entry point | Rayon threads | µs/actual call | Median reported best |
|---|---:|---:|---:|
| BiteOpt direct | 1 | 1.460 | 0.000967 |
| DeepBiteOpt depth 4 | 1 | 1.877 | 0.0887 |
| BiteOpt batch 16, serial callback | 1 | 1.616 | 0.000666 |
| DE direct | 1 | 0.242 | 5.85e-07 |
| DE population ask/tell | 1 | 0.253 | 0.564 |
| DE delayed, window 8 | 1 | 6.527 | 0.00117 |
| DE delayed, window 8 | 8 | 7.450 | 0.00117 |

On Cassini1, holding delayed DE's look-ahead fixed at eight gave 9.435 µs/call with one Rayon thread and 9.943 µs/call with eight. Both pool sizes returned identical printed costs for each paired seed on both objectives. Serial and delayed DE have different feedback and repair rules. Wall time across those two entry points includes that difference.

| Explicit BiteOpt population | d=32 µs/call | d=32 median best | d=96 µs/call | d=96 median best |
|---:|---:|---:|---:|---:|
| 32 | 0.979 | 0.000211 | 1.702 | 3.89 |
| 105 | 1.407 | 0.000967 | 2.385 | 7.51 |
| 297 | 2.822 | 0.019 | 4.319 | 15.4 |

Population size changed both runtime and solution quality on this fixed budget. At d=32, P=105 reproduces the native-default final results; at d=96, P=297 does the same. The independently timed duplicates provide a timing-noise check. A population change belongs in the search-quality experiments.

**DE regression, reproduced.** With dimension 8, P=32, nominal budget 32 and an objective that always returns 7, direct DE made 64 calls. Population ask/tell and delayed DE made 32. Each returned `1.7976931348623157e308` for all five seeds. On Cassini1, seed 105, direct DE returned 15.678160421199632 after 8,205 calls. The callback saw 15.678160134166063. Evaluating the returned x reproduced the higher value. The other 19 direct runs in this audit matched the recorded minimum. In both update loops the code writes `pop_y[p]` before comparing against `pop_y[best_i]`. When p is already the best slot, the new value compares with itself. D01 tests the observed consequences on ties, true improvements and stop targets.

### MODE

ZDT1 used 12 decision variables, two objectives, no constraints, a [0,1] box and exactly 8,192 evaluated offspring per run. HV is the exact dominated area of the final returned population inside reference point (1.1, 1.1). Larger is better. P changes the number of generations as well as per-generation work.

| P | Generations | NSGA-II µs/call | NSGA-II tell share | NSGA-II HV | DE-mode µs/call | DE-mode tell share | DE-mode HV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 256 | 1.021 | 94.5% | 0.8541 | 0.830 | 78.0% | 0.8589 |
| 64 | 128 | 1.261 | 87.5% | 0.8645 | 1.090 | 75.9% | 0.8675 |
| 128 | 64 | 1.760 | 90.9% | 0.8615 | 1.630 | 83.6% | 0.8699 |
| 256 | 32 | 2.890 | 94.9% | 0.7607 | 2.913 | 91.2% | 0.8415 |

Both paths pay survivor-selection costs. The DE reproduction path also spends time generating donors during `ask`. Larger populations were more expensive per evaluation here and the largest population had only 32 updates. Compare timing and quality at matched P when evaluating a code change.

### MAP-Elites

The pilot used 12 decision coordinates in [0,1], descriptors (x₀,x₁), minimized quality 1 + Σxᵢ², 256 generations of 32 offspring and no CMA emitter. Every archive confirmed 8,192 visits. The 2D grid used zero samples/niche. CVT used two samples/niche and a cold fit. Positive quality makes the QD-score the sum of reciprocal occupied qualities. A new archive used a single seeded RNG for construction, parent seeding and search, so changing geometry also changes the later RNG stream. Coverage and score are descriptive and depend on capacity and geometry.

| Archive / emitter | K | Cold setup ms | Warm µs/call | Occupied / K | QD-score | Lookup ns/query | Full sort µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grid / SBX | 256 | 0.04 | 0.502 | 256/256 | 163.0 | 22.1 | 3.18 |
| Grid / SBX | 1,024 | 0.08 | 0.911 | 982/1,024 | 468.7 | 22.1 | 15.69 |
| Grid / SBX | 4,096 | 0.27 | 2.495 | 2,498/4,096 | 780.6 | 23.5 | 51.59 |
| Grid / Iso+LineDD | 1,024 | 0.07 | 0.852 | 1,019/1,024 | 385.1 | 22.3 | 15.96 |
| CVT / SBX | 256 | 3.22 | 0.956 | 255/256 | 159.3 | 467.1 | 3.26 |
| CVT / SBX | 1,024 | 32.38 | 2.660 | 948/1,024 | 432.5 | 1,797.4 | 15.22 |

The lookup test used 8,192 prepared uniform descriptors, and the sort test repeated `argsort` 128 times on the final frozen archive. Each sort starts from canonical niche IDs in the current implementation. These tests isolate operations. Their inputs, occupancy and cache state differ from live generations, so the timings cannot be subtracted from the live total to attribute an exact percentage. In particular, a grid already has constant-time lookup. Grid gains at large K should first be sought in ranking the archive.

## How to run the 80 experiments

There are 20 distinct change proposals per implementation. The setup below and the final confirmation run sit outside that count. Within each track, change one thing at a time. Compare it to the current accepted version using the same build and inputs. Every fifth experiment also compare the accumulated winners to the pinned original. Keep rejected attempts and raw rows in the experiment log. Where two proposals replace the same data structure, branch both from the same parent and test their interaction only after each has a result.

1. **Freeze the reference.** Build the pinned commit in release mode with its lockfile, thin LTO and one codegen unit. Record git SHA, rustc/LLVM, lock hash, CPU, affinity, compiler flags, thread pool and objective version. Use separate output directories for baseline and candidate. Run untimed trace and allocation builds separately from timing builds.
2. **Screen cheaply.** Use seeds 101–105 from the baseline. Pre-generate ranking, lookup and archive-update fixtures for microbenchmarks. Time at least seven interleaved baseline/candidate pairs in alternating order after a discarded warm-up. Extend short cases to at least 100 ms per timed sample by repeating complete runs. Log startup, steady-state and total time. For parallel tests, keep pool construction outside steady-state timing.
3. **Require a measurable end-to-end effect.** For implementation changes, use paired time ratios on the declared target. A proposed default gate is at least 5% lower median time, a 95% paired bootstrap interval excluding parity and no confirmed regression over 3% in another primary workload. Pair by seed and run block, resampling whole seeds for cross-seed conclusions. If noise is larger than the gate, increase repeats or confirm on a dedicated machine. Specialized paths can dispatch only in the region where they win. Record allocation count, bytes copied or peak memory alongside timing when that is the mechanism under test.
4. **Validate trajectories and quality.** Except for D01 and the rows marked **Search**, require exact replay. Compare candidate coordinates, actual objective calls, returned solutions and stop codes in order against the appropriate reference for that entry point. Digest large traces, then locate the first difference on a mismatch. For explicitly labeled **search** rows, expect changed trajectories. Use 30 held-out paired seeds (1001–1030), fixed evaluation limits and a second fixed-wall test. Reevaluate returned solutions independently. Record failures and censored runs. A proposed promotion gate is a 10% improvement in time/evaluations to an agreed target or a 5% improvement in the predeclared quality metric, with an interval supporting improvement and no material loss on a held-out family. For near-zero losses use a target or a log-error floor. For success rates report counts and paired uncertainty.
5. **Preserve the relevant API.** Exercise direct Rust, stateful ask/tell, the Python wrapper and the registered job-worker route where they touch the changed code. Keep the repository's safe-Rust and Rust 1.88 requirements. Include shapes, nonfinite results, bounds, seeded reproducibility, budgets, cooperative stops and panic behavior in regression tests for the affected path. Run the pinned workspace tests and binding/worker smoke tests before combining winners.
6. **Use eight threads for two different questions.** Measure eight independent seeded runs with one thread each for retry throughput. Separately test one batch/ordered window on an eight-thread pool for latency. For pool scaling, keep batch width or look-ahead fixed at both thread counts. Fix total evaluations across retry configurations, limit nested parallelism, and record callback time, pool overhead, CPU time and completed evaluations. Include a cheap function, Cassini1 where applicable and a deterministic more expensive callback. Repeat finalists on a 16-physical-core machine to establish scaling there.

### Workloads and independent checks

| Track | Screening set | Hold-out and decision metric |
|---|---|---|
| BiteOpt | Shifted Sphere at dimensions 8, 32 and 96, fixed population and native default. Rotate/shift a quadratic, add shifted Rastrigin, Rosenbrock and Cassini1. Single, deep and delayed-batch variants. | Cassini2 or Rosetta plus a multimodal held-out function. Record best at 1k/8k/32k actual evaluations, target success, time to target, constructor cost, bytes/evaluation and elapsed time per retry. Cassini1's repository target is 4.95535. |
| DE | Same scalar functions. Add flat and stepped plateaus, bounds-heavy trials, sparse/all-false/dense integer masks, normal sampling from a guess, and an objective with skewed evaluation latency. | Compare direct, population ask/tell and delayed update within their own semantics. Audit every actual call, including temporal-locality trials. Serial DE checks its budget at generation boundaries. At equal budgets compare the best seen through the first N callbacks, and report final reported and actual counts separately. Confirm standalone DE before using a DE→CMA retry as a downstream check. |
| MODE | ZDT1/2/3, populations 32/64/128/256, odd sizes 31/65, both reproduction modes. Add an all-nondominated front, a dominance chain, duplicated rows, and feasible-only/mixed/infeasible synthetic matrices. | DTLZ2 with 3 and 5 objectives, constrained Binh–Korn, and a truss-sizing MO run. Record feasible fraction, first-front size, feasible hypervolume (HV), IGD+ to a frozen analytic/reference front, and survivor identities. Fix scaling and reference box per problem across all builds. For sampled higher-dimensional HV, reuse identical sample points and record error. |
| MAP-Elites | Fixed descriptor stream, plus a live grid/CVT archive. Sweep capacity 60/256/1,024/4,096 and descriptor dimension 2/4/8. Test uniform, skewed and same-niche batches, 0/10/50/100% occupancy, chunks 1/16/32/128, SBX and Iso+LineDD. | Same centroids for layout-preserving comparisons. Record setup, warm updates, occupied/capacity, core QD-score, best and median occupied quality, evaluations to fixed coverage, and age of the archive. Use network-coverage or optical-lens-design QD as an application case. Freeze descriptor bounds and score convention. Re-bin into a common reference archive if an experimental search arm changes tessellation. |

## BiteOpt — 20 iterations

The direct path can call the embedded spherical and sequential Nelder–Mead optimizers. The public ask/tell path resamples generators that require an objective during generation. Compare each with its own reference. Defaults couple population size to dimension through `9 + 3d`. Use both the fixed-dimension population sweep and the fixed-population dimension sweep before attributing scaling to a kernel. Most generators already borrow rows, the RNG already pools bits, and the direct path already recycles one candidate.

| ID | Change to try | Discriminating measurement and guard |
|---|---|---|
| B01 | Replace adjacent row swaps in `BitePop::update_pop` with a slice rotation of row handles and one bulk cost shift. | Replay reject, tied, best, middle and worst inserts at P=32/105/297. Measure ns/accepted insert and total run time. Preserve insertion before equal costs and reuse the evicted row. |
| B02 | Change dirty centroid recomputation to row-major accumulation into reusable `i128` column sums. | Sweep dimension independently of P. Digest centroids after every resize and update. Sum over the full stored population and retain the original division and cast. |
| B03 | Add a strict lower-bound exit to nearest parallel-population distance accumulation. | Count coordinates visited on clustered and uniform streams. Equal distances select the later population under the current `<=` rule. Resolve dirty centroids with the same state effects. |
| B04 | Remove the unconditional clear in `take_tmp` for generators proven to write every coordinate. | Poison buffers with a test pattern, exercise every affected generator and compare encoded candidates. Report cleared bytes per trial at d=96. Keep initialization on partial-write paths. |
| B05 | Replace owning `ordered_of` reads in Sol5/Sol5b with parent indices and short-lived borrows. | Count allocations in these generators and calls per 8k trials. Keep selector and RNG calls in order. Confirm the live run improves at their measured frequency. |
| B06 | Try flat encoded population storage plus sorted slot indices as an alternative to B01's row-handle layout. | Branch independently from pre-B01. Measure cache behavior, insertion, donor gathers and peak bytes. Confirm public results and centroid invalidation on repeated identical rows. |
| B07 | Put a fast identity check in `BiteSel::restore` before scanning the entry IDs for a delayed selection. | Count fast hits, restore time and ID searches at batch 1/16/64. On any miss run the existing lookup. Exercise slot promotion, demotion and duplicate selector values. |
| B08 | Pack selector values and entry IDs into one moveable entry per slot so that increment/decrement shifts one slice. | Replay long feedback sequences and compare selected values, IDs and logical slot order after every update. Check constructor bytes as well as update time. |
| B09 | Reuse one candidate buffer set **per outstanding batch row** across ask/tell cycles. Bound the pool by observed batch size and retain the single-candidate direct path. | Track steady-state allocations at batches 1/16/64 and retained memory after a short final batch. Verify pending re-ask, invalid tell and the exact init/budget cap. |
| B10 | For `optimize_batch_until`, present a reusable evaluation batch without building an additional owned public ask snapshot each generation. | Time a cheap callback and a batched application callback. Keep public owned `ask` snapshots independent and apply exactly one feedback per evaluated row. |
| B11 | In deep batched `tell`, reuse sort-index storage and sanitize costs once into reusable keys. | Compare index order for ties, signed zero and nonfinite costs. Match best-first feedback, pushes and active-population switches. Measure depth 1/4 at batch 8/64. |
| B12 | In direct deep mode, borrow or recycle the encoded payload handed from `step_collect` to the receiving optimizer. | Count allocations only on pushes. Check source/destination ownership, init guards, RNG state, switching and best history at depths 2/4/8. |
| B13 | Eliminate state-vector clones in `NMSeqOpt::optimize` and `copy` by splitting borrows around evaluation and replacement. | Replay reflection, expansion, contraction and reduction. Compare state, evaluation order and centroid bits. Measure end-to-end on direct runs where the branch is reached. |
| B14 | Reuse the spherical optimizer's normalized trial and real-value buffers across calls to `SpherOpt::optimize`. | Track bytes per spherical step and confirm the real coordinates passed to the objective and returned to the main optimizer agree. Exercise alternating secondary optimizers. |
| B15 | Cache spherical update weights using the exact `curem`, population size and selected exponents as keys. Keep the existing `powf` calculation on a miss. | Log hit rate, retained bytes and update time. Compare center/radius bits, including changed `curem`. Reject caching if misses outweigh saved arithmetic. |
| B16 | Fill the initialization queue directly and reuse selector reset storage for short retries. | Time construction plus the first P evaluations at depth 1/4, with and without a user guess. Preserve Gaussian draw order, the first evaluated rows and subsequent selector state. |
| B17 | Fuse the ordered wrap and real conversion passes for ordinary candidates in `gen_one`. | Test in-range, out-of-range and secondary precomputed trials. Check exact encoded/decoded pairs, repair draws and objective call counts. Measure d=8/96. |
| B18 | **Search.** Test a capped default-size population for short runs against native P and two fixed sizes. | Use the pilot sweep to nominate one rule before hold-out. Compare 8k and 32k budgets on shifted, rotated and GTOP problems. Count initialization as part of the budget. |
| B19 | **Search.** Test a delayed-batch width schedule that starts small and expands after the population fill. | Compare with constant widths 1/8/16 at equal calls on one and eight threads. Measure both target success and wall time, including the serial direct reference as a separate arm. |
| B20 | **Search.** Try a restart portfolio mixing fixed depth-1 and depth-4 retries under a fixed total evaluation and eight-thread budget. | Compare to all-depth-1 and all-depth-4 retries, with preassigned seeds and equal per-retry limits. Report time to first target, success and tail outcomes on Cassini1/2 and Rastrigin. |

## Differential evolution — 20 iterations

D01 establishes the correctness baseline for the remaining 19 attempts. Serial DE evaluates the temporal-locality trial immediately. Ask/tell queues improvements, and delayed DE refills an ordered window. The serial donor can randomly resample an out-of-bounds coordinate where the ask path clamps. Preserve these distinctions when sharing storage or kernels.

| ID | Change to try | Discriminating measurement and guard |
|---|---|---|
| D01 | **Correctness.** Decouple the best-ever update from replacement of the current best population slot. Use the corrected update to remove duplicate best-observation work where equivalent. | Test constant 7, seed 105 Cassini1, a first-candidate winner, a tied best, all-nonfinite observations, target stop and cooperative cancellation. Compare reported `x,y` with an independent callback ledger on all three APIs. Preserve tie policy and record any legitimate change in early stopping. Take a new speed baseline. |
| D02 | In serial `optimize_until`, build the trial from borrowed target, best and donor rows instead of cloning all four per member. | Keep the mutate/repair RNG interleaving unchanged even at the forced crossover index. Count row bytes copied per call at d=8/32/96, and compare trial traces. |
| D03 | Recycle a serial trial buffer by swapping it with a displaced population row on acceptance. | Count allocations separately by accepted/rejected trials. Reevaluate the incumbent and confirm its stored vector cannot be overwritten by a later trial. |
| D04 | Write the temporal-locality step directly into its final buffer with the current feasibility rule applied coordinate by coordinate. | Exercise frequent improvements, clipping and integer mutation. Match the extra evaluation's coordinates and whether the first or second trial is retained. |
| D05 | In `ask_one/next_x`, pass parent indices and borrow donor rows after RNG choices instead of cloning the target, best and two donors. | Test population ask and delayed windows 1/8. Compare ordered candidate hashes, including the generation boundary where `update_mean` runs. |
| D06 | Fuse ask-path donor selection, crossover and clamp to a single output pass. | Sweep crossover rates and bound activity. Keep forced-coordinate short-circuit RNG behavior and original comparison polarity, including exact-bound values. |
| D07 | Precompute integer-coordinate indices and their count at construction. Visit them in original order inside `modify`. | Compare masks with zero, one, half and all coordinates flagged. Audit truncation, repair and random draw counts. Record throughput for continuous and mixed cases. |
| D08 | Fill normally sampled replacement rows in place, eliminating the raw row followed by a separate clamped allocation. | Test normal and uniform sampling, aggressive aging and narrow bounds. Compare reinitialized rows, survival ages and bounds across seeds. |
| D09 | Store normal-sampling history in a reusable block and reuse `sigma_new` scratch in `update_mean`. | Benchmark normal starts at d=32/96/256. Match each sigma, mean and history wrap, including both expand and blend branches. |
| D10 | Introduce an internal owned-row variant of `tell_one` to move pending candidates into history/population after acceptance. | Compare repeated updates to the same target in one asked population. Check best/history and pending rows, plus allocations on rejected and accepted tells. |
| D11 | Replace the two improvement queues with one pre-sized deque of (target index, trial buffer). | Induce long and empty queues. Verify FIFO order, pending indices and allocations per temporal-locality follow-up. |
| D12 | Add an additive borrowed-generation view to the Rust batch evaluator while maintaining owned snapshots for existing callers. | Benchmark bare Rust and the Python copy boundary separately. Re-asking DE replaces pending work. An invalid tell must leave the pending batch usable. |
| D13 | Replace delayed DE's completion `BTreeMap` with a bounded ring keyed by dispatch sequence. | Force reversed completions, wraparound, window larger than remaining budget, early stop and objective panic. Confirm exact ask-order tells and maximum in-flight evaluations. |
| D14 | Add an inline execution path for a one-thread Rayon pool with a fixed delayed window that simulates the same ask-ahead/refill sequence. | Compare window 1/8/32 with the delayed reference. Require the same candidate order for pure deterministic objectives. Verify callback/stop/panic semantics before enabling for any caller. |
| D15 | For multi-thread delayed execution, give evaluators reusable decode storage and transfer owned buffers through a bounded return queue. | Test unnormalized/normalized coordinates, cancellation and panic draining. Measure allocation, peak bytes and throughput at 8 workers, including a costly callback. |
| D16 | Try persistent evaluators in the caller's pool, fed by a bounded queue, as an alternative to spawning one task per objective call. | Test pool sizes 1/2/8 and windows 1/8/32 with skewed costs. Check progress when the caller occupies a pool slot, ordered feedback, stop/drain and propagated panic. |
| D17 | Try contiguous row-major population/history storage with indexed row views, retaining owned external results. | Compare donor gather and accepted-write throughput at low/high P. Measure RSS and end-to-end time. Keep safe split borrows when indices alias the best or target. |
| D18 | **Search.** Replace rejection sampling in `pick_two` with a uniform exclusion-index mapping. | Enumerate legal pairs for P=4/5, including `p == best_i`. Check frequencies, then evaluate small/large P. Draw-count changes require the held-out quality gate. |
| D19 | **Search.** Gate the optional temporal-locality trial using a per-run rolling success estimate. | Log the cost of extra trials and their marginal improvements. Compare always-on versus the predeclared rule on shifted Rosenbrock, plateau and GTOP at actual-call parity. |
| D20 | **Search.** Test one age-reinitialization policy that scales the existing `keep` by population diversity against the fixed-`keep` implementation. | Record resets, diversity, target attainment and tails on rotated multimodal functions and Cassini. Preserve a fixed-`keep` control for the stand-alone DE and the DE phase in a DE→CMA sequence. |

## MODE — 20 iterations

At population P, survivor selection considers 2P rows. The existing ranker compares every pair, constructs dominance adjacency lists and peels successive fronts. A first-front archive cannot supply all ranks required to refill the population. The constrained path also ranks violations. In the NSGA-II-style path, the time charged to `tell` includes generating future offspring. Instrument sorting, gathering, crowding and variation separately before claiming a ranking speedup.

| ID | Change to try | Discriminating measurement and guard |
|---|---|---|
| M01 | In `pop_update`, rank by row index and gather only the selected P survivors into alternate buffers. Eliminate full x/y clones before ranking and extra copies after gathering. | Measure bytes per generation at P=64/256, d=12/96. Compare survivor identities and order, including the single-objective pre-sort, a split front, identical points and in-place overwrite hazards. |
| M02 | Add an exact two-objective all-front ranker using a lexicographic sweep and compressed second-coordinate prefix maxima. Delay insertion for each identical (x,y) group. | Differential-test every rank on ties in either axis, signed zeros, sanitized sentinels, chains and flat fronts. Distinct rows equal on one axis can strictly dominate. Reconstruct the current high-is-good level numbering and stable survivor order. Measure crossover versus the quadratic oracle. |
| M03 | Break the general pairwise comparison when each point is already better on at least one component. | Compare d-objective operation counts and time at 2/3/5/8 objectives on independent, correlated and conflicting data. Preserve numerical dominance and all ties after sanitization. |
| M04 | Replace per-point dominance-vector allocations with reusable packed bit rows at moderate P. | Benchmark sparse and dense dominance graphs including a chain and a broad front. Compare memory and front-peeling time to adjacency lists, and cap the quadratic memory footprint. |
| M05 | Bucket row indices by rank in one stable pass instead of rescanning all 2P rows for each level. | Sweep one front through 2P successive fronts, constrained and unconstrained. Compare exact ordering of full fronts and truncation of the first overflowing front. |
| M06 | Reuse front queues, counts and rank scratch between generations. Reset only live ranges. | Count allocations for alternating feasible counts and front depths. Poison scratch on resize and compare each complete vector of ranks. |
| M07 | Compute crowding on a list of survivor indices with a reusable objective-column buffer, avoiding cloned rows for the split front. | Test one/two rows, repeated extremes, constant columns and signed zeros. Compare the exact distance vector and selected indices under MODE's own tie conventions. |
| M08 | Select the needed k indices of an overflowing front by a partial selection, then sort just the winners by the existing total key. | Sweep k/front-size. Match reversed stable-sort ties explicitly using the original local row position. Retain a full sort when it wins. |
| M09 | Replace constrained objective/violation row clones with views over original y rows and a reusable feasible index list. | Replay all-feasible, mixed and all-infeasible Binh–Korn batches. Compare ranks, survivor order and count of points satisfying every constraint at zero tolerance. |
| M10 | Remove the dense temporary rank matrix in `ranks` by precomputing the number of violated constraints per row, then accumulating sorted columns in their original order. | Compare floating-point rank sums exactly, including ties around zero and multiple simultaneous violations. Vary constraints 1/4/16 and fraction feasible. |
| M11 | Sanitize and copy each externally supplied y row into its existing offspring buffer in `try_tell`. | Benchmark 2/5/16 output columns. Test nonfinite input, incorrect last-row width and a valid retry after error. Validate shape before committing reusable storage. |
| M12 | Add an internal borrowed offspring view for the worker evaluate loop. Keep owned `try_ask` as a compatibility wrapper. | Measure core and job-worker wall time with a cheap registered problem. Match pending-call errors, output independence, one-based progress where applicable and ordered tells. |
| M13 | In DE reproduction, produce each child from borrowed parent rows directly in its destination, folding crossover and clipping into the existing RNG sequence. | Test P=4/65 and mixed-integer masks with `pareto_update` zero/positive. Match donor exclusions, forced coordinate, mutation order and final feasible rows. |
| M14 | In NSGA-II variation, compute both children as each pair's beta values are drawn and keep the complete mutation pass after all crossover draws. | Eliminate the beta matrix. Compare RNG stream, sibling output positions and the unpaired parent at odd P. Measure emitted bytes and time per child. |
| M15 | Clamp NSGA-II offspring in place after mutation instead of allocating another row in `closest_feasible`. | Test out-of-range SBX children and boundary values with mutation on/off. Compare final child bits and allocation counts separately from M14. |
| M16 | Specialize all-front ranking for a single objective using ordered groups of equal numeric values. | Compare the current one-objective pre-sort/reversal and the feasible-subset route. Guard duplicates, signed zeros, finite sentinel and population order after selection. |
| M17 | Try cache-blocked objective-major dominance comparisons for the general 3+ objective path. | Branch from the M03 comparator. Sweep P=64–512 and objective count. Charge transposition each generation. Require a measured dispatch threshold and equality of every rank. |
| M18 | Try parallel blocks for generic ranking above that threshold, merging local dominance counts/edges in fixed order. | Compare identical P with 1/8 threads and with eight concurrent independent MODE runs. Check memory growth, nesting, rank and survivor ordering. Retain serial dispatch for smaller P. |
| M19 | **Search.** Try one predeclared `tell_switch` schedule alternating NSGA-II-style reproduction and the DE path at a fixed generation fraction. | Compare against both fixed modes using identical evaluations. Report HV, IGD+, feasibility and the worst held-out ZDT/DTLZ/constrained result. Verify the switch actually changes the intended subsequent generation. |
| M20 | **Search.** Try objective-space duplicate suppression only when filling an overflowing front. Refill unused slots in stable order if uniqueness is exhausted. | Compare to unmodified crowding on quantized and smooth objectives, including fronts with fewer distinct points than P. Preserve population size and feasible-first behavior. Report diversity together with HV and elapsed time. |

## MAP-Elites — 20 iterations

The two-dimensional regular grid already indexes a niche arithmetically. A CVT archive searches the normalized centers. The ordinary core loop and the job worker both sort by fitness after each committed generation. An optimization limited to the core driver can miss worker users. Preserve seeded, unevaluated parent rows, ordered updates for colliding candidates, counts, the winner on equal quality and the post-commit progress callback.

| ID | Change to try | Discriminating measurement and guard |
|---|---|---|
| Q01 | Sort only occupied niches and append the untouched empty indices in canonical order. Use an exact (fitness total order, niche ID) key and retain a dense fallback. | Replay 1–100% occupancy, equal-quality elites and `best_n` larger than occupancy. Time `argsort`, full 32-row generations and the worker route at K=4,096. |
| Q02 | Repair the previously sorted order from the batch's changed niche IDs, with a threshold for rebuilding from scratch. | Compare with Q01 on sparse improvements and collision-heavy batches. Compare the entire ranked list to a fresh canonical sort after every update, including filled and newly occupied cells. |
| Q03 | Store normalized CVT centers contiguously and specialize the exact squared-distance scan for 2D/4D, with a generic fallback. | Measure per-query time and center bytes at K=256/1,024/4,096. Keep summation order and the lowest original center ID at equal distance. Differentiate from ragged grid indexing. |
| Q04 | Build an exact nearest-center k-d tree once for static CVT centers, choosing tree or scan by K and descriptor dimension. | Benchmark build, memory, query and full-run break-even against Q03. Search equal-distance branches when a lower ID could win. Include boundary, duplicate-center, out-of-range and 8D cases. |
| Q05 | Cache regular grid layout and normalize the two coordinates in local scalars instead of allocating an encoded descriptor. | Test every ragged row at K=60 and 1,025 plus 1, square grids and points on/just outside edges. Preserve division/clamping and exact niche IDs. Measure ns/query and end-to-end change. |
| Q06 | Omit regular-grid center-vector materialization when arithmetic layout suffices. Preserve any observable archive semantics. | Measure constructor time, clone cost and peak bytes at K=4,096/65,536. Compare every index, grid shape, `grid_layout` and seeded parent sequence. |
| Q07 | Share immutable CVT geometry and any query index between archive clones while keeping elite state private to each run. | Exercise many worker retries from one template. Measure cold versus per-retry setup and RSS. Mutate both clones independently and compare counts, elites and descriptor bounds. |
| Q08 | Reuse encoded descriptor scratch for the generic lookup path, including batches with descriptor dimension above two. | Measure allocs/query and warm throughput for dimensions 3/8/16. Verify public queries remain reentrant and thread-safe, with no per-query lock. |
| Q09 | Compute niche IDs for a batch in parallel above a work threshold, then commit candidates in original order. | Test K=256/4,096 CVT with chunks 16/128/1,024 and 1/8 threads. Deliberately collide consecutive candidates. Require identical improvements, real costs, counts and elite identities. |
| Q10 | Add an internal commit-only update for ordinary emitters, avoiding creation of two return vectors discarded by the driver. | Compare core and worker allocations and wall time. Keep ordered public `update_evaluated` results for CMA/diversifier users. Test invalid rows and batch-length error recovery. |
| Q11 | Sample parent niche indices and borrow their rows during ordinary emission instead of cloning parent vectors in `random_xs`. | Preserve independent owned output for public calls and keep the initial parent pool. Compare sampled IDs and child bits across sparse/full archives for both emitters. |
| Q12 | Reuse offspring row storage across ordinary generations with private SBX and Iso fill-into kernels. | Measure allocations and RSS at d=12/96. Preserve SBX's current odd-chunk behavior, Iso output count, objective order and batch failure semantics. |
| Q13 | For SBX in `variation`, defer the expensive beta power until after drawing the existing sign/mask, skipping its evaluation when the mask overwrites beta with one. | Consume the same random numbers in the same order. Compare offspring bits at fixed seeds, boundary repairs and time per emitted candidate, including a live grid run. |
| Q14 | Replace repeated allocation of Lloyd partial sums/counts with reusable per-chunk storage, cleared deterministically. | Measure CVT constructor time and peak memory for samples/niche 2/16/64. Preserve fixed chunk boundaries, reduction order and center bits with 1 and 8 threads. |
| Q15 | Introduce a serial-size cutoff in the k-means++ minimum-distance update, retaining parallel pointwise updates for larger sample sets. | Sweep K and training-sample count. Measure seeding separately from Lloyd iterations. Compare center indices, RNG continuation and total setup time at 1/8 threads. |
| Q16 | In Lloyd assignment, rebuild a deterministic exact center index per iteration when the predicted query saving can repay the rebuild. | Compare construction plus all Lloyd iterations against the scan on low/high descriptor dimensions. Preserve old-center IDs on distance ties, assignments, empty clusters, stopping iteration and final centers. |
| Q17 | Remove temporary finite-quality collection in `qd_score` using an allocation-free pass or two passes over stored qualities. | Measure a progress callback that reads QD-score every generation versus logging disabled. Preserve the reciprocal-positive and negative-only rules, signed zero, empty cells and arithmetic order. |
| Q18 | Reuse sorted-improvement and previous-generation buffers in the CMA emitter and diversifier restart loops. | On a run with CMA emission enabled, count allocations/generation. Preserve sorted comparison, stall decision, per-niche improvement and actual eval count. Check the job worker's separate emitter implementation. |
| Q19 | **Search.** Try a fixed small fraction of parents drawn from all seeded niches during the occupied-only selection phase. | Compare exploration on sparse archives against existing selection and pure uniform controls. Report coverage-to-budget, QD-score and elite quality on shared geometry, including late generations. |
| Q20 | **Search.** Try an emitter portfolio alternating SBX and Iso+LineDD with a predeclared schedule. Compare to each emitter alone. | Use equal actual evaluations with even chunks and identical centroids. Compare coverage, QD-score and time to joint coverage/quality targets on the synthetic task and one application. Test one and eight evaluator threads. |

## Execution order and stopping decisions

1. Run D01 first and freeze a corrected DE reference. In the other tracks, start with B01–B05, M01–M03 and Q01/Q03. These distinguish expensive population maintenance, data movement and algorithmic growth. For large CVT archives, run Q04 as soon as Q03 supplies a fair scan comparator.
2. Complete all 20 attempts per track. If profiling rules out the proposed mechanism, run a minimal isolating prototype or a measured upper-bound test, record the evidence, and move on. Label a skipped or unbuildable attempt explicitly. Track repeats/interaction runs separately, so they cannot be counted as distinct attempts.
3. Measure the combined winners against the pinned commit, and for DE against D01 as well. Ablate the top two changes in each combination. Test cold/warm, small/large and 1/8-thread workloads. A regression confined to a small-input regime is a reason to choose a measured crossover or reject the change.
4. Keep a row per (experiment, parent SHA, candidate SHA, problem, input hash, seed, repetition, algorithm mode, P/K, d, thread count, requested/actual calls). Include phase and total times, objective-only control where applicable, validation hash, quality metrics, pass/fail, disposition and reason. Record raw samples even when the candidate loses. Deliver a patch containing accepted changes and a log listing all 80 decisions when the campaign is executed.

### Source map

All links below point to the benchmarked commit. The recently added `archivede`, `rankindex` and front-indicator utilities are separate from the four production optimizers studied here.

- Single-objective implementation and test oracles. [BiteOpt](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/biteopt.rs) · [DE](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/de.rs) · [Fitness and decoding](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/fitness.rs)
- Selection, diversity and validation. [MODE](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/mode.rs) · [MAP-Elites](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/mapelites.rs) · [Quality indicators](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-core/src/indicators.rs)
- Entry points. [Python wrappers](https://github.com/dietmarwo/fcmaes-rust/tree/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-py/src) · [MODE worker](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-job-worker/src/mode.rs) · [MAP-Elites worker](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/crates/fcmaes-job-worker/src/map_elites.rs)
- Existing application baselines. [GTOP experiments and budget definitions](https://github.com/dietmarwo/fcmaes-rust/blob/d7ff66aebc388ac19b698742c7678986e7a48d58/benchmarks/benchmark_gtop.md) · [Optimizer-comparison harness](https://github.com/dietmarwo/fcmaes-rust/tree/d7ff66aebc388ac19b698742c7678986e7a48d58/benchmarks/optimizer-comparison) · [Job tutorial payloads](https://github.com/dietmarwo/fcmaes-rust/tree/d7ff66aebc388ac19b698742c7678986e7a48d58/tutorials/job-api)

## Reproduce the performance check

The native probe below was added only to a scratch copy of the existing standalone benchmark crate so it could reuse the checked-in lockfile. The production core source was byte-for-byte the pinned commit. For a new implementation campaign, give a permanent multi-optimizer benchmark its own home. The following instructions reproduce this diagnostic.

1. Check out the commit above. Under `benchmarks/cmaes-implementation/src/bin/`, save **Appendix A** as `four_probe.rs`. Save **Appendix B** as `run_probe.py` at the checkout root.
2. Build and run from the root with the following commands. On a machine with a different allowed CPU set, replace the affinity values in the runner. An offline build works with a populated Cargo cache.

```bash
export CARGO_TARGET_DIR="$PWD/target-four"
cargo build --release --locked --manifest-path benchmarks/cmaes-implementation/Cargo.toml --bin four_probe
python3 run_probe.py "$CARGO_TARGET_DIR/release/four_probe"
for seed in 101 102 103 104 105; do
  RAYON_NUM_THREADS=1 taskset -c 0 "$CARGO_TARGET_DIR/release/four_probe" "$seed" audit
  RAYON_NUM_THREADS=1 taskset -c 0 "$CARGO_TARGET_DIR/release/four_probe" "$seed" edge
done
```

3. Group `probe.csv` by arm, dimension, size and thread count. Within each row divide `wall_ns` by actual calls `n`. Take medians over repetitions per seed and then over seeds. For MODE, compute `phase2_ns / wall_ns` before aggregation. For MAP micro rows, divide `phase1_ns` by 8,192 for ns/lookup and `phase2_ns` by 128,000 for µs/sort. In the explicit-P sweep, the population is encoded in the arm name. The Rust audit prints seed, dimension, actual calls, reported y, lowest observed y, and the reevaluated y at returned x.


<details>
<summary>Appendix A · Rust probe</summary>

```rust
use fcmaes_core::{
    Archive, BiteOpt, BiteParams, De, DeParams, DeepBiteOpt, Fitness, MapElitesParams, Mode,
    ModeParams, Rng, map_elites,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

const N: usize = 8192;
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}
fn ns(t: Duration) -> u128 {
    t.as_nanos()
}

// Exact 2-D minimization HV after clipping the front to the reference box.
fn hv(mut points: Vec<Vec<f64>>) -> f64 {
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let (mut top, mut area) = (1.1, 0.0);
    for y in points {
        if y[0] < 1.1 && y[1] < top {
            area += (1.1 - y[0]) * (top - y[1]);
            top = y[1];
        }
    }
    area
}

fn shifted(x: &[f64]) -> f64 {
    const OFF: [f64; 8] = [-2.71, 1.13, 0.49, -1.77, 3.01, -0.97, 2.21, -3.25];
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - OFF[i % 8]).powi(2))
        .sum()
}

fn problem(d: usize) -> (Vec<f64>, Vec<f64>, fn(&[f64]) -> f64) {
    if d == 6 {
        (
            vec![-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
            vec![0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
            fcmaes_gtop::cassini1,
        )
    } else {
        (vec![-5.0; d], vec![5.0; d], shifted)
    }
}

fn scalar(name: &str, d: usize, seed: u64) {
    let (lo, hi, obj) = problem(d);
    let bp = BiteParams {
        popsize: name
            .strip_prefix("bite-p")
            .and_then(|p| p.parse().ok())
            .unwrap_or(0),
        seed,
        max_evaluations: N as u64,
        ..Default::default()
    };
    let dp = DeParams {
        seed,
        popsize: 32,
        max_evaluations: N as u64,
        ..Default::default()
    };
    let start = Instant::now();
    let (n, y) = match name {
        "bite" | "bite-p32" | "bite-p105" | "bite-p297" => {
            let r = BiteOpt::new(&lo, &hi, None, &bp).optimize(&obj);
            (r.evaluations, r.y)
        }
        "bite-depth4" => {
            let r = DeepBiteOpt::new(&lo, &hi, None, &bp, 4).optimize(&obj);
            (r.evaluations, r.y)
        }
        "bite-batch16" => {
            let r = BiteOpt::new(&lo, &hi, None, &bp)
                .optimize_batch(16, |xs| xs.iter().map(|x| obj(x)).collect());
            (r.evaluations, r.y)
        }
        _ => {
            let mut de = De::new(Fitness::bounded(d, 1, &lo, &hi), &[], &[], None, &dp);
            let r = match name {
                "de" => de.optimize(&obj),
                "de-window8" => de.optimize_delayed(&obj, 8),
                "de-ask" => {
                    for _ in 0..N / 32 {
                        let xs = de.ask();
                        de.tell(&xs.iter().map(|x| obj(x)).collect::<Vec<_>>());
                    }
                    de.result()
                }
                _ => unreachable!(),
            };
            (r.evaluations, r.y)
        }
    };
    let wall = ns(start.elapsed());
    println!("{name},{d},0,{seed},{n},{wall},0,0,0,{y:.16e},0");
}

fn mode(p: usize, sbx: bool, seed: u64) {
    let d = 12;
    let mut m = Mode::new(
        Fitness::bounded(d, 2, &vec![0.0; d], &vec![1.0; d]),
        2,
        0,
        None,
        &ModeParams {
            seed,
            popsize: p as i32,
            nsga_update: sbx,
            ..Default::default()
        },
    );
    let (mut ask, mut tell) = (Duration::ZERO, Duration::ZERO);
    let start = Instant::now();
    for _ in 0..N / p {
        let t = Instant::now();
        let xs = m.ask();
        ask += t.elapsed();
        let ys: Vec<_> = xs
            .iter()
            .map(|x| {
                let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (d - 1) as f64;
                vec![x[0], g * (1.0 - (x[0] / g).sqrt())]
            })
            .collect();
        let t = Instant::now();
        m.tell(&ys);
        tell += t.elapsed();
    }
    let wall = ns(start.elapsed());
    let quality = hv(m.result().y);
    let name = if sbx { "mode-sbx" } else { "mode-de" };
    println!(
        "{name},{d},{p},{seed},{N},{wall},0,{},{},{quality:.16e},0",
        ns(ask),
        ns(tell)
    );
}

fn map(k: usize, samples: usize, sbx: bool, seed: u64) {
    let d = 12;
    let mut rng = Rng::new(seed);
    let (lo, hi) = ([0.0; 12], [1.0; 12]);
    let start = Instant::now();
    let mut a = Archive::new(d, &[0.0; 2], &[1.0; 2], k, samples, &mut rng);
    a.seed_uniform(&lo, &hi, &mut rng);
    let setup = ns(start.elapsed());
    let f = |x: &[f64]| (1.0 + sphere(x), vec![x[0], x[1]]);
    let start = Instant::now();
    map_elites(
        &mut a,
        &mut { f },
        &lo,
        &hi,
        &MapElitesParams {
            generations: N / 32,
            chunk_size: 32,
            use_sbx: sbx,
            ..Default::default()
        },
        &mut rng,
    );
    let wall = ns(start.elapsed());
    let n = a.counts().iter().sum::<u64>();
    assert_eq!(n, N as u64);
    let (score, occupied) = (a.qd_score(), a.occupied());
    let name = match (samples, sbx) {
        (0, true) => "map-grid-sbx",
        (0, false) => "map-grid-iso",
        _ => "map-cvt-sbx",
    };
    println!("{name},{d},{k},{seed},{n},{wall},{setup},0,0,{score:.16e},{occupied}");
    // Diagnostics outside the full-run timer: frozen archive, prepared uniform queries.
    let mut qr = Rng::new(99);
    let queries: Vec<[f64; 2]> = (0..N).map(|_| [qr.uniform01(), qr.uniform01()]).collect();
    let start = Instant::now();
    for q in &queries {
        black_box(a.index_of_niche(black_box(q)));
    }
    let lookup = ns(start.elapsed());
    let start = Instant::now();
    for _ in 0..128 {
        black_box(&mut a).argsort();
    }
    let sort = ns(start.elapsed());
    println!("{name}-micro,{d},{k},{seed},{N},0,0,{lookup},{sort},0,0");
}

fn audit(seed: u64) {
    for d in [6, 8, 32, 96] {
        let (lo, hi, obj) = problem(d);
        let dp = DeParams {
            seed,
            popsize: 32,
            max_evaluations: N as u64,
            ..Default::default()
        };
        let mut de = De::new(Fitness::bounded(d, 1, &lo, &hi), &[], &[], None, &dp);
        let best = std::sync::Mutex::new((f64::INFINITY, 0u64));
        let r = de.optimize(&|x: &[f64]| {
            let y = obj(x);
            let mut seen = best.lock().unwrap();
            seen.0 = seen.0.min(y);
            seen.1 += 1;
            y
        });
        let (seen, n) = *best.lock().unwrap();
        assert_eq!(r.evaluations, n);
        println!("{seed},{d},{n},{:.17e},{seen:.17e},{:.17e}", r.y, obj(&r.x));
    }
}

fn edge(seed: u64) {
    let params = DeParams {
        seed,
        popsize: 32,
        max_evaluations: 32,
        ..Default::default()
    };
    let flat = |_: &[f64]| 7.0;
    for arm in ["serial", "batch", "window"] {
        let mut de = De::new(
            Fitness::bounded(8, 1, &[-5.0; 8], &[5.0; 8]),
            &[],
            &[],
            None,
            &params,
        );
        let r = match arm {
            "serial" => de.optimize(&flat),
            "batch" => {
                let xs = de.ask();
                de.tell(&vec![7.0; xs.len()]);
                de.result()
            }
            _ => de.optimize_delayed(&flat, 8),
        };
        println!("flat,{arm},{seed},{},{:.17e},7", r.evaluations, r.y);
    }
    let (mut first, mut seen) = (f64::NAN, None);
    let counter = std::sync::Mutex::new((0usize, &mut first, &mut seen));
    let params = BiteParams {
        seed,
        max_evaluations: N as u64,
        ..Default::default()
    };
    BiteOpt::new(&[-5.0; 32], &[5.0; 32], None, &params).optimize(&|x: &[f64]| {
        let y = sphere(x);
        let mut c = counter.lock().unwrap();
        c.0 += 1;
        if c.0 == 1 {
            *c.1 = y;
        }
        if y == 0.0 && c.2.is_none() {
            *c.2 = Some(c.0);
        }
        y
    });
    println!("centered,bite,{seed},{first:.17e},{seen:?}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed = args[1].parse::<u64>().unwrap();
    let _ = rayon::current_num_threads();
    if args.get(2).is_some_and(|s| s == "edge") {
        edge(seed);
        return;
    }
    if args.get(2).is_some_and(|s| s == "pop") {
        for d in [32, 96] {
            for name in ["bite-p32", "bite-p105", "bite-p297"] {
                scalar(name, d, seed);
            }
        }
        return;
    }
    if args.get(2).is_some_and(|s| s == "audit") {
        audit(seed);
        return;
    }
    if args.get(2).is_some_and(|s| s == "window") {
        for d in [6, 32] {
            scalar("de-window8", d, seed);
        }
        return;
    }
    for d in [6, 8, 32, 96] {
        for name in ["bite", "de"] {
            scalar(name, d, seed);
        }
    }
    for name in ["bite-depth4", "bite-batch16", "de-ask"] {
        scalar(name, 32, seed);
    }
    for p in [32, 64, 128, 256] {
        for sbx in [true, false] {
            mode(p, sbx, seed);
        }
    }
    for k in [256, 1024, 4096] {
        map(k, 0, true, seed);
    }
    map(1024, 0, false, seed);
    for k in [256, 1024] {
        map(k, 2, true, seed);
    }
}
```

</details>

<details>
<summary>Appendix B · Timing runner</summary>

```python
import os, subprocess, csv, sys
from pathlib import Path
exe = Path(sys.argv[1]).resolve()
cols="rep threads arm dim size seed n wall_ns setup_ns phase1_ns phase2_ns quality occupied".split()
with open("probe.csv","w",newline="") as out:
    w=csv.writer(out)
    w.writerow(cols)
    for rep,seeds in enumerate([[100], [101,102,103,104,105], [105,104,103,102,101], [103,104,105,101,102]]):
        for seed in seeds:
            variants=[(1,[]),(1,["window"]),(8,["window"]),(1,["pop"])]
            if (seed+rep)%2: variants.reverse()
            for threads,extra in variants:
                env={**os.environ,"RAYON_NUM_THREADS":str(threads)}
                aff="0" if threads==1 else "0-7"
                result=subprocess.run(["taskset","-c",aff,exe,str(seed),*extra],capture_output=True,text=True,check=True,env=env,timeout=90)
                for row in csv.reader(result.stdout.splitlines()):
                    if rep: w.writerow([rep,threads,*row])
                out.flush()
```

</details>

<details>
<summary>Appendix C · Raw data from all 615 measured timing/diagnostic rows</summary>

```csv
rep,threads,arm,dim,size,seed,n,wall_ns,setup_ns,phase1_ns,phase2_ns,quality,occupied
1,1,bite,6,0,101,8192,49367364,0,0,0,1.7284696929097187e1,0
1,1,de,6,0,101,8204,46468500,0,0,0,1.4706560283588807e1,0
1,1,bite,8,0,101,8192,5086260,0,0,0,3.6977854932234928e-29,0
1,1,de,8,0,101,8211,851132,0,0,0,0.0000000000000000e0,0
1,1,bite,32,0,101,8192,11157393,0,0,0,1.1826333068296398e-3,0
1,1,de,32,0,101,8193,1982105,0,0,0,5.8523236436527079e-7,0
1,1,bite,96,0,101,8192,33765328,0,0,0,9.3274857814853984e0,0
1,1,de,96,0,101,8205,5118649,0,0,0,4.1982722582275272e1,0
1,1,bite-depth4,32,0,101,8192,15822258,0,0,0,8.2921804631106488e-2,0
1,1,bite-batch16,32,0,101,8192,12765584,0,0,0,1.4072638407990753e-3,0
1,1,de-ask,32,0,101,8192,1987043,0,0,0,5.6395006211537480e-1,0
1,1,mode-sbx,12,32,101,8192,7991915,0,190706,7561997,8.5413229059125528e-1,0
1,1,mode-de,12,32,101,8192,6551807,0,1190340,5117213,8.5889918485858974e-1,0
1,1,mode-sbx,12,64,101,8192,10156275,0,1020673,8882910,8.6447191155192660e-1,0
1,1,mode-de,12,64,101,8192,8853072,0,1866980,6743223,8.6709909331076906e-1,0
1,1,mode-sbx,12,128,101,8192,14395018,0,1020829,13073496,8.5784757927062072e-1,0
1,1,mode-de,12,128,101,8192,13216023,0,1910575,11045462,8.7031199968739070e-1,0
1,1,mode-sbx,12,256,101,8192,23381922,0,926043,22165354,7.6070505972010283e-1,0
1,1,mode-de,12,256,101,8192,23421311,0,1777806,21355651,8.4611453519658486e-1,0
1,1,map-grid-sbx,12,256,101,8192,4131352,43185,0,0,1.6296890243802014e2,256
1,1,map-grid-sbx-micro,12,256,101,8192,0,0,191799,399632,0,0
1,1,map-grid-sbx,12,1024,101,8192,7458843,77186,0,0,4.3370849534401907e2,977
1,1,map-grid-sbx-micro,12,1024,101,8192,0,0,180552,1950989,0,0
1,1,map-grid-sbx,12,4096,101,8192,19005572,271158,0,0,7.8716486204844477e2,2500
1,1,map-grid-sbx-micro,12,4096,101,8192,0,0,181073,8044560,0,0
1,1,map-grid-iso,12,1024,101,8192,10360191,111478,0,0,3.8514157682986286e2,1019
1,1,map-grid-iso-micro,12,1024,101,8192,0,0,316146,3227821,0,0
1,1,map-cvt-sbx,12,256,101,8192,12355477,5223176,0,0,1.6381405863139813e2,255
1,1,map-cvt-sbx-micro,12,256,101,8192,0,0,6945420,640825,0,0
1,1,map-cvt-sbx,12,1024,101,8192,21751446,43280779,0,0,4.5626875961058892e2,948
1,1,map-cvt-sbx-micro,12,1024,101,8192,0,0,14786127,1886581,0,0
1,1,de-window8,6,0,101,8192,76484202,0,0,0,1.1032465680287711e1,0
1,1,de-window8,32,0,101,8192,53428029,0,0,0,3.9601237166690626e0,0
1,8,de-window8,6,0,101,8192,70176342,0,0,0,1.1032465680287711e1,0
1,8,de-window8,32,0,101,8192,9592797,0,0,0,3.9601237166690626e0,0
1,1,bite-p32,32,0,101,8192,10346381,0,0,0,5.0581867134273466e-4,0
1,1,bite-p105,32,0,101,8192,14541037,0,0,0,1.1826333068296398e-3,0
1,1,bite-p297,32,0,101,8192,27720035,0,0,0,1.9015635428459281e-2,0
1,1,bite-p32,96,0,101,8192,19375839,0,0,0,3.6352241649416750e0,0
1,1,bite-p105,96,0,101,8192,25405551,0,0,0,6.2246554574358610e0,0
1,1,bite-p297,96,0,101,8192,43416737,0,0,0,9.3274857814853984e0,0
1,1,bite-p32,32,0,102,8192,9951820,0,0,0,1.8942881724483020e-5,0
1,1,bite-p105,32,0,102,8192,13165380,0,0,0,5.3091597129978287e-4,0
1,1,bite-p297,32,0,102,8192,24713525,0,0,0,8.7859648007388904e-3,0
1,1,bite-p32,96,0,102,8192,18952437,0,0,0,3.3135735461576306e0,0
1,1,bite-p105,96,0,102,8192,23637414,0,0,0,5.0583707442814765e0,0
1,1,bite-p297,96,0,102,8192,38650347,0,0,0,1.4257464708625225e1,0
1,8,de-window8,6,0,102,8192,78134225,0,0,0,1.8058767639340438e1,0
1,8,de-window8,32,0,102,8192,30792862,0,0,0,1.1749152610350377e-3,0
1,1,de-window8,6,0,102,8192,76417044,0,0,0,1.8058767639340438e1,0
1,1,de-window8,32,0,102,8192,80650350,0,0,0,1.1749152610350377e-3,0
1,1,bite,6,0,102,8192,63180605,0,0,0,1.2535114393026227e1,0
1,1,de,6,0,102,8223,46723117,0,0,0,1.5295226918582406e1,0
1,1,bite,8,0,102,8192,5942431,0,0,0,4.6049755342276564e-29,0
1,1,de,8,0,102,8203,859945,0,0,0,0.0000000000000000e0,0
1,1,bite,32,0,102,8192,15045011,0,0,0,5.3091597129978287e-4,0
1,1,de,32,0,102,8245,1975455,0,0,0,3.5417816658361076e-7,0
1,1,bite,96,0,102,8192,47029599,0,0,0,1.4257464708625225e1,0
1,1,de,96,0,102,8237,5450911,0,0,0,7.7820990521143955e1,0
1,1,bite-depth4,32,0,102,8192,15214677,0,0,0,2.1855863915533205e-1,0
1,1,bite-batch16,32,0,102,8192,12514620,0,0,0,3.1664345725242865e-4,0
1,1,de-ask,32,0,102,8192,2070228,0,0,0,2.3318430133627190e0,0
1,1,mode-sbx,12,32,102,8192,8381425,0,217968,7881511,8.5066788230651591e-1,0
1,1,mode-de,12,32,102,8192,6623235,0,1204906,5153539,8.5941842596006190e-1,0
1,1,mode-sbx,12,64,102,8192,10146393,0,1010714,8846419,8.6567344498245069e-1,0
1,1,mode-de,12,64,102,8192,8776780,0,1891874,6620407,8.6778101890713677e-1,0
1,1,mode-sbx,12,128,102,8192,16241343,0,1134343,14783627,8.6618259087932337e-1,0
1,1,mode-de,12,128,102,8192,13356749,0,1865420,11214283,8.6993730918882961e-1,0
1,1,mode-sbx,12,256,102,8192,23582140,0,887469,22300398,7.4875737458304514e-1,0
1,1,mode-de,12,256,102,8192,23769383,0,1808504,21662321,8.4145472733249149e-1,0
1,1,map-grid-sbx,12,256,102,8192,4097792,41753,0,0,1.6339945498594162e2,256
1,1,map-grid-sbx-micro,12,256,102,8192,0,0,180352,409066,0,0
1,1,map-grid-sbx,12,1024,102,8192,7604675,75914,0,0,4.6872253192078153e2,990
1,1,map-grid-sbx-micro,12,1024,102,8192,0,0,180702,2008415,0,0
1,1,map-grid-sbx,12,4096,102,8192,18543872,270788,0,0,6.8462888326304187e2,2425
1,1,map-grid-sbx-micro,12,4096,102,8192,0,0,203036,6590487,0,0
1,1,map-grid-iso,12,1024,102,8192,6920884,87732,0,0,4.3908336568440893e2,1017
1,1,map-grid-iso-micro,12,1024,102,8192,0,0,182134,2001856,0,0
1,1,map-cvt-sbx,12,256,102,8192,7986100,3118097,0,0,1.5930603514977449e2,255
1,1,map-cvt-sbx-micro,12,256,102,8192,0,0,3861576,402527,0,0
1,1,map-cvt-sbx,12,1024,102,8192,22058947,36943266,0,0,4.3251631500471092e2,938
1,1,map-cvt-sbx-micro,12,1024,102,8192,0,0,14693783,1853972,0,0
1,1,bite,6,0,103,8192,51067871,0,0,0,1.0999741945697014e1,0
1,1,de,6,0,103,8213,45701988,0,0,0,1.7562742444003412e1,0
1,1,bite,8,0,103,8192,5177178,0,0,0,9.8607613152626476e-32,0
1,1,de,8,0,103,8244,1109852,0,0,0,0.0000000000000000e0,0
1,1,bite,32,0,103,8192,11134051,0,0,0,9.6700773427792447e-4,0
1,1,de,32,0,103,8236,1976287,0,0,0,6.4976378846799303e-6,0
1,1,bite,96,0,103,8192,34302099,0,0,0,1.7080460560883850e1,0
1,1,de,96,0,103,8228,5145340,0,0,0,3.2991186458384973e1,0
1,1,bite-depth4,32,0,103,8192,15297882,0,0,0,7.4900827748511700e-2,0
1,1,bite-batch16,32,0,103,8192,12860591,0,0,0,6.6599948517022639e-4,0
1,1,de-ask,32,0,103,8192,2011990,0,0,0,6.7383132201002264e-3,0
1,1,mode-sbx,12,32,103,8192,8026540,0,201311,7585577,8.5662308536505916e-1,0
1,1,mode-de,12,32,103,8192,6798791,0,1218420,5142526,8.5740578795406197e-1,0
1,1,mode-sbx,12,64,103,8192,10328718,0,1038806,9039478,8.6379249087850196e-1,0
1,1,mode-de,12,64,103,8192,9604097,0,2063729,7285253,8.6751750346627632e-1,0
1,1,mode-sbx,12,128,103,8192,15147025,0,1158923,13674973,8.5993842821149036e-1,0
1,1,mode-de,12,128,103,8192,13243188,0,1870014,11115836,8.6904321852994504e-1,0
1,1,mode-sbx,12,256,103,8192,24115283,0,857711,22955746,7.7564015459984392e-1,0
1,1,mode-de,12,256,103,8192,24101163,0,1776573,22019292,8.4832454346164532e-1,0
1,1,map-grid-sbx,12,256,103,8192,4119595,42123,0,0,1.6253954254166814e2,256
1,1,map-grid-sbx-micro,12,256,103,8192,0,0,180732,437269,0,0
1,1,map-grid-sbx,12,1024,103,8192,7302820,76435,0,0,4.7882488110767110e2,949
1,1,map-grid-sbx-micro,12,1024,103,8192,0,0,180322,1863617,0,0
1,1,map-grid-sbx,12,4096,103,8192,21993728,245039,0,0,7.7079310435303319e2,2478
1,1,map-grid-sbx-micro,12,4096,103,8192,0,0,202935,9355072,0,0
1,1,map-grid-iso,12,1024,103,8192,6939012,70706,0,0,4.0567600386125605e2,1019
1,1,map-grid-iso-micro,12,1024,103,8192,0,0,181594,2336929,0,0
1,1,map-cvt-sbx,12,256,103,8192,7613929,3123644,0,0,1.5921972648803401e2,256
1,1,map-cvt-sbx-micro,12,256,103,8192,0,0,3710139,414865,0,0
1,1,map-cvt-sbx,12,1024,103,8192,21896072,31827931,0,0,4.4868395719066211e2,968
1,1,map-cvt-sbx-micro,12,1024,103,8192,0,0,14715456,1886741,0,0
1,1,de-window8,6,0,103,8192,77034033,0,0,0,1.6371480183340484e1,0
1,1,de-window8,32,0,103,8192,58833909,0,0,0,3.5541467556733648e-3,0
1,8,de-window8,6,0,103,8192,85878124,0,0,0,1.6371480183340484e1,0
1,8,de-window8,32,0,103,8192,58961210,0,0,0,3.5541467556733648e-3,0
1,1,bite-p32,32,0,103,8192,7612816,0,0,0,4.5361790549789932e-5,0
1,1,bite-p105,32,0,103,8192,11253891,0,0,0,9.6700773427792447e-4,0
1,1,bite-p297,32,0,103,8192,22329024,0,0,0,2.4928796965558945e-2,0
1,1,bite-p32,96,0,103,8192,13780968,0,0,0,3.8914182234087442e0,0
1,1,bite-p105,96,0,103,8192,18729391,0,0,0,8.9081295314135751e0,0
1,1,bite-p297,96,0,103,8192,33764398,0,0,0,1.7080460560883850e1,0
1,1,bite-p32,32,0,104,8192,8017356,0,0,0,2.4975711573075037e-4,0
1,1,bite-p105,32,0,104,8192,11103545,0,0,0,3.6169467529794422e-4,0
1,1,bite-p297,32,0,104,8192,22756137,0,0,0,3.4445033272558122e-2,0
1,1,bite-p32,96,0,104,8192,14841986,0,0,0,6.5898883207883365e0,0
1,1,bite-p105,96,0,104,8192,19541754,0,0,0,1.1062224273248001e1,0
1,1,bite-p297,96,0,104,8192,42285593,0,0,0,1.5378600611905433e1,0
1,8,de-window8,6,0,104,8192,68990667,0,0,0,1.4405043239464188e1,0
1,8,de-window8,32,0,104,8192,54945161,0,0,0,4.6363254789597410e-5,0
1,1,de-window8,6,0,104,8192,88144198,0,0,0,1.4405043239464188e1,0
1,1,de-window8,32,0,104,8192,57983317,0,0,0,4.6363254789597410e-5,0
1,1,bite,6,0,104,8192,53686836,0,0,0,1.0996495809146211e1,0
1,1,de,6,0,104,8204,46212247,0,0,0,8.6494382297011967e0,0
1,1,bite,8,0,104,8192,5273985,0,0,0,2.0707598762051560e-30,0
1,1,de,8,0,104,8193,927918,0,0,0,1.9721522630525295e-31,0
1,1,bite,32,0,104,8192,12112855,0,0,0,3.6169467529794422e-4,0
1,1,de,32,0,104,8209,2169318,0,0,0,6.9691400111331344e-6,0
1,1,bite,96,0,104,8192,36011633,0,0,0,1.5378600611905433e1,0
1,1,de,96,0,104,8238,5556760,0,0,0,2.3235472737756442e1,0
1,1,bite-depth4,32,0,104,8192,17002490,0,0,0,8.8687204951218082e-2,0
1,1,bite-batch16,32,0,104,8192,13601948,0,0,0,1.0070149233337318e-3,0
1,1,de-ask,32,0,104,8192,2224410,0,0,0,4.5809930341223978e-2,0
1,1,mode-sbx,12,32,104,8192,8758231,0,193647,8277085,8.4985952082610272e-1,0
1,1,mode-de,12,32,104,8192,7175168,0,1317106,5619082,8.5943608478619982e-1,0
1,1,mode-sbx,12,64,104,8192,10562671,0,1088824,9216115,8.6426875377162304e-1,0
1,1,mode-de,12,64,104,8192,9030782,0,1905534,6869313,8.6749248667162970e-1,0
1,1,mode-sbx,12,128,104,8192,14822657,0,1047266,13498194,8.6333261533921268e-1,0
1,1,mode-de,12,128,104,8192,12970717,0,1906892,10805585,8.6994462669792416e-1,0
1,1,mode-sbx,12,256,104,8192,24815328,0,970170,23543097,7.8136118545204170e-1,0
1,1,mode-de,12,256,104,8192,25497076,0,1878898,23306113,8.3083285176903532e-1,0
1,1,map-grid-sbx,12,256,104,8192,4146566,42624,0,0,1.6383043090580401e2,256
1,1,map-grid-sbx-micro,12,256,104,8192,0,0,180712,455686,0,0
1,1,map-grid-sbx,12,1024,104,8192,7643784,75223,0,0,4.5418514443695187e2,984
1,1,map-grid-sbx-micro,12,1024,104,8192,0,0,182035,2168686,0,0
1,1,map-grid-sbx,12,4096,104,8192,29957984,271428,0,0,7.8055646566654934e2,2601
1,1,map-grid-sbx-micro,12,4096,104,8192,0,0,207092,8239552,0,0
1,1,map-grid-iso,12,1024,104,8192,10212753,187473,0,0,3.8094611255853226e2,1020
1,1,map-grid-iso-micro,12,1024,104,8192,0,0,232921,2352203,0,0
1,1,map-cvt-sbx,12,256,104,8192,11271858,4414329,0,0,1.5906982500649806e2,255
1,1,map-cvt-sbx-micro,12,256,104,8192,0,0,4940692,515216,0,0
1,1,map-cvt-sbx,12,1024,104,8192,27886404,57892010,0,0,4.3104334099115653e2,933
1,1,map-cvt-sbx-micro,12,1024,104,8192,0,0,15206475,2139172,0,0
1,1,bite,6,0,105,8192,51735106,0,0,0,1.0996503871417433e1,0
1,1,de,6,0,105,8205,48203617,0,0,0,1.5678160421199632e1,0
1,1,bite,8,0,105,8192,6136533,0,0,0,3.6484816866471796e-30,0
1,1,de,8,0,105,8214,922770,0,0,0,0.0000000000000000e0,0
1,1,bite,32,0,105,8192,11956419,0,0,0,1.4422417837912976e-3,0
1,1,de,32,0,105,8220,2069557,0,0,0,4.2111277752988254e-8,0
1,1,bite,96,0,105,8192,33848374,0,0,0,2.3004113003140873e1,0
1,1,de,96,0,105,8221,5620295,0,0,0,3.1530023253077029e1,0
1,1,bite-depth4,32,0,105,8192,15965639,0,0,0,2.6325073062184623e-1,0
1,1,bite-batch16,32,0,105,8192,13352442,0,0,0,5.6815964655758063e-4,0
1,1,de-ask,32,0,105,8192,2270159,0,0,0,1.9316429981120489e1,0
1,1,mode-sbx,12,32,105,8192,9570525,0,197746,9132063,8.5439131660484036e-1,0
1,1,mode-de,12,32,105,8192,6892212,0,1202950,5353274,8.5821199322334074e-1,0
1,1,mode-sbx,12,64,105,8192,12867531,0,1064366,11538330,8.6591002229428138e-1,0
1,1,mode-de,12,64,105,8192,9390184,0,2158991,6941298,8.6737926438746404e-1,0
1,1,mode-sbx,12,128,105,8192,14855406,0,1078189,13503621,8.6145784182114615e-1,0
1,1,mode-de,12,128,105,8192,14174160,0,2265564,11643817,8.7006091594969159e-1,0
1,1,mode-sbx,12,256,105,8192,25930929,0,928939,24630827,7.3911761109383511e-1,0
1,1,mode-de,12,256,105,8192,25607141,0,1866124,23433877,8.3601760157556892e-1,0
1,1,map-grid-sbx,12,256,105,8192,4345906,45879,0,0,1.6238762222760559e2,256
1,1,map-grid-sbx-micro,12,256,105,8192,0,0,181243,406402,0,0
1,1,map-grid-sbx,12,1024,105,8192,8108133,145048,0,0,4.6953163196095193e2,982
1,1,map-grid-sbx-micro,12,1024,105,8192,0,0,183367,2175987,0,0
1,1,map-grid-sbx,12,4096,105,8192,21600817,317318,0,0,8.0646902045290881e2,2498
1,1,map-grid-sbx-micro,12,4096,105,8192,0,0,180752,6684719,0,0
1,1,map-grid-iso,12,1024,105,8192,7677835,70166,0,0,3.7309797616382815e2,1017
1,1,map-grid-iso-micro,12,1024,105,8192,0,0,182375,1939872,0,0
1,1,map-cvt-sbx,12,256,105,8192,8314504,2873949,0,0,1.6276406563596043e2,255
1,1,map-cvt-sbx-micro,12,256,105,8192,0,0,4124943,404910,0,0
1,1,map-cvt-sbx,12,1024,105,8192,23638912,31056058,0,0,4.2611129480401723e2,955
1,1,map-cvt-sbx-micro,12,1024,105,8192,0,0,18523121,2108887,0,0
1,1,de-window8,6,0,105,8192,79586648,0,0,0,3.7293553202786079e1,0
1,1,de-window8,32,0,105,8192,54384416,0,0,0,1.9867167906193206e-5,0
1,8,de-window8,6,0,105,8192,88001282,0,0,0,3.7293553202786079e1,0
1,8,de-window8,32,0,105,8192,66303090,0,0,0,1.9867167906193206e-5,0
1,1,bite-p32,32,0,105,8192,11873650,0,0,0,2.1101785874328078e-4,0
1,1,bite-p105,32,0,105,8192,11634694,0,0,0,1.4422417837912976e-3,0
1,1,bite-p297,32,0,105,8192,23115959,0,0,0,1.1544909245503823e-2,0
1,1,bite-p32,96,0,105,8192,13529469,0,0,0,5.3934937726627510e0,0
1,1,bite-p105,96,0,105,8192,18403730,0,0,0,7.5083041076306554e0,0
1,1,bite-p297,96,0,105,8192,32223336,0,0,0,2.3004113003140873e1,0
2,1,bite-p32,32,0,105,8192,7614319,0,0,0,2.1101785874328078e-4,0
2,1,bite-p105,32,0,105,8192,11522756,0,0,0,1.4422417837912976e-3,0
2,1,bite-p297,32,0,105,8192,23080706,0,0,0,1.1544909245503823e-2,0
2,1,bite-p32,96,0,105,8192,13560566,0,0,0,5.3934937726627510e0,0
2,1,bite-p105,96,0,105,8192,18430962,0,0,0,7.5083041076306554e0,0
2,1,bite-p297,96,0,105,8192,32377849,0,0,0,2.3004113003140873e1,0
2,8,de-window8,6,0,105,8192,76347349,0,0,0,3.7293553202786079e1,0
2,8,de-window8,32,0,105,8192,80643224,0,0,0,1.9867167906193206e-5,0
2,1,de-window8,6,0,105,8192,75477208,0,0,0,3.7293553202786079e1,0
2,1,de-window8,32,0,105,8192,53472122,0,0,0,1.9867167906193206e-5,0
2,1,bite,6,0,105,8192,59734846,0,0,0,1.0996503871417433e1,0
2,1,de,6,0,105,8205,46082102,0,0,0,1.5678160421199632e1,0
2,1,bite,8,0,105,8192,5169376,0,0,0,3.6484816866471796e-30,0
2,1,de,8,0,105,8214,861718,0,0,0,0.0000000000000000e0,0
2,1,bite,32,0,105,8192,11384238,0,0,0,1.4422417837912976e-3,0
2,1,de,32,0,105,8220,1969297,0,0,0,4.2111277752988254e-8,0
2,1,bite,96,0,105,8192,32635487,0,0,0,2.3004113003140873e1,0
2,1,de,96,0,105,8221,5206423,0,0,0,3.1530023253077029e1,0
2,1,bite-depth4,32,0,105,8192,15690984,0,0,0,2.6325073062184623e-1,0
2,1,bite-batch16,32,0,105,8192,13077008,0,0,0,5.6815964655758063e-4,0
2,1,de-ask,32,0,105,8192,1956577,0,0,0,1.9316429981120489e1,0
2,1,mode-sbx,12,32,105,8192,7992599,0,190572,7560444,8.5439131660484036e-1,0
2,1,mode-de,12,32,105,8192,6603237,0,1186675,5180714,8.5821199322334074e-1,0
2,1,mode-sbx,12,64,105,8192,10283700,0,998518,9029871,8.6591002229428138e-1,0
2,1,mode-de,12,64,105,8192,8791862,0,1857305,6685893,8.6737926438746404e-1,0
2,1,mode-sbx,12,128,105,8192,14165237,0,1045317,12801077,8.6145784182114615e-1,0
2,1,mode-de,12,128,105,8192,13106371,0,1902619,10938289,8.7006091594969159e-1,0
2,1,mode-sbx,12,256,105,8192,24075313,0,936411,22841933,7.3911761109383511e-1,0
2,1,mode-de,12,256,105,8192,23674440,0,1793390,21591515,8.3601760157556892e-1,0
2,1,map-grid-sbx,12,256,105,8192,4068668,42404,0,0,1.6238762222760559e2,256
2,1,map-grid-sbx-micro,12,256,105,8192,0,0,180622,428526,0,0
2,1,map-grid-sbx,12,1024,105,8192,7772808,76776,0,0,4.6953163196095193e2,982
2,1,map-grid-sbx-micro,12,1024,105,8192,0,0,180512,2083328,0,0
2,1,map-grid-sbx,12,4096,105,8192,20198905,244929,0,0,8.0646902045290881e2,2498
2,1,map-grid-sbx-micro,12,4096,105,8192,0,0,180762,6357076,0,0
2,1,map-grid-iso,12,1024,105,8192,6954175,70125,0,0,3.7309797616382815e2,1017
2,1,map-grid-iso-micro,12,1024,105,8192,0,0,193211,1916457,0,0
2,1,map-cvt-sbx,12,256,105,8192,7575220,3113930,0,0,1.6276406563596043e2,255
2,1,map-cvt-sbx-micro,12,256,105,8192,0,0,3721755,421404,0,0
2,1,map-cvt-sbx,12,1024,105,8192,21771744,28300358,0,0,4.2611129480401723e2,955
2,1,map-cvt-sbx-micro,12,1024,105,8192,0,0,14736917,2069517,0,0
2,1,bite,6,0,104,8192,51046629,0,0,0,1.0996495809146211e1,0
2,1,de,6,0,104,8204,44184192,0,0,0,8.6494382297011967e0,0
2,1,bite,8,0,104,8192,5222496,0,0,0,2.0707598762051560e-30,0
2,1,de,8,0,104,8193,865344,0,0,0,1.9721522630525295e-31,0
2,1,bite,32,0,104,8192,11156545,0,0,0,3.6169467529794422e-4,0
2,1,de,32,0,104,8209,1991870,0,0,0,6.9691400111331344e-6,0
2,1,bite,96,0,104,8192,33163933,0,0,0,1.5378600611905433e1,0
2,1,de,96,0,104,8238,5140463,0,0,0,2.3235472737756442e1,0
2,1,bite-depth4,32,0,104,8192,15379265,0,0,0,8.8687204951218082e-2,0
2,1,bite-batch16,32,0,104,8192,16764301,0,0,0,1.0070149233337318e-3,0
2,1,de-ask,32,0,104,8192,1993182,0,0,0,4.5809930341223978e-2,0
2,1,mode-sbx,12,32,104,8192,8046239,0,196466,7596381,8.4985952082610272e-1,0
2,1,mode-de,12,32,104,8192,6545589,0,1207883,5103502,8.5943608478619982e-1,0
2,1,mode-sbx,12,64,104,8192,10024690,0,997875,8779181,8.6426875377162304e-1,0
2,1,mode-de,12,64,104,8192,8684822,0,1858401,6572983,8.6749248667162970e-1,0
2,1,mode-sbx,12,128,104,8192,14139848,0,1015279,12852923,8.6333261533921268e-1,0
2,1,mode-de,12,128,104,8192,13359954,0,1923787,11158517,8.6994462669792416e-1,0
2,1,mode-sbx,12,256,104,8192,23593388,0,947234,22326654,7.8136118545204170e-1,0
2,1,mode-de,12,256,104,8192,38582795,0,3161052,34444660,8.3083285176903532e-1,0
2,1,map-grid-sbx,12,256,104,8192,6027059,291109,0,0,1.6383043090580401e2,256
2,1,map-grid-sbx-micro,12,256,104,8192,0,0,258199,571902,0,0
2,1,map-grid-sbx,12,1024,104,8192,13603775,433143,0,0,4.5418514443695187e2,984
2,1,map-grid-sbx-micro,12,1024,104,8192,0,0,418310,7387768,0,0
2,1,map-grid-sbx,12,4096,104,8192,51346111,8056715,0,0,7.8055646566654934e2,2601
2,1,map-grid-sbx-micro,12,4096,104,8192,0,0,301905,8089184,0,0
2,1,map-grid-iso,12,1024,104,8192,7189149,116435,0,0,3.8094611255853226e2,1020
2,1,map-grid-iso-micro,12,1024,104,8192,0,0,260282,2090138,0,0
2,1,map-cvt-sbx,12,256,104,8192,8070226,3872713,0,0,1.5906982500649806e2,255
2,1,map-cvt-sbx-micro,12,256,104,8192,0,0,3993845,412752,0,0
2,1,map-cvt-sbx,12,1024,104,8192,24340512,34416070,0,0,4.3104334099115653e2,933
2,1,map-cvt-sbx-micro,12,1024,104,8192,0,0,15581860,1913272,0,0
2,1,de-window8,6,0,104,8192,91001291,0,0,0,1.4405043239464188e1,0
2,1,de-window8,32,0,104,8192,55243952,0,0,0,4.6363254789597410e-5,0
2,8,de-window8,6,0,104,8192,81449469,0,0,0,1.4405043239464188e1,0
2,8,de-window8,32,0,104,8192,71865864,0,0,0,4.6363254789597410e-5,0
2,1,bite-p32,32,0,104,8192,8453102,0,0,0,2.4975711573075037e-4,0
2,1,bite-p105,32,0,104,8192,11403777,0,0,0,3.6169467529794422e-4,0
2,1,bite-p297,32,0,104,8192,25531086,0,0,0,3.4445033272558122e-2,0
2,1,bite-p32,96,0,104,8192,14666001,0,0,0,6.5898883207883365e0,0
2,1,bite-p105,96,0,104,8192,22367291,0,0,0,1.1062224273248001e1,0
2,1,bite-p297,96,0,104,8192,35379069,0,0,0,1.5378600611905433e1,0
2,1,bite-p32,32,0,103,8192,8375355,0,0,0,4.5361790549789932e-5,0
2,1,bite-p105,32,0,103,8192,11462476,0,0,0,9.6700773427792447e-4,0
2,1,bite-p297,32,0,103,8192,24621777,0,0,0,2.4928796965558945e-2,0
2,1,bite-p32,96,0,103,8192,14508343,0,0,0,3.8914182234087442e0,0
2,1,bite-p105,96,0,103,8192,19108802,0,0,0,8.9081295314135751e0,0
2,1,bite-p297,96,0,103,8192,35745753,0,0,0,1.7080460560883850e1,0
2,8,de-window8,6,0,103,8192,67464288,0,0,0,1.6371480183340484e1,0
2,8,de-window8,32,0,103,8192,85243664,0,0,0,3.5541467556733648e-3,0
2,1,de-window8,6,0,103,8192,79353311,0,0,0,1.6371480183340484e1,0
2,1,de-window8,32,0,103,8192,52832418,0,0,0,3.5541467556733648e-3,0
2,1,bite,6,0,103,8192,49907333,0,0,0,1.0999741945697014e1,0
2,1,de,6,0,103,8213,44321139,0,0,0,1.7562742444003412e1,0
2,1,bite,8,0,103,8192,5156357,0,0,0,9.8607613152626476e-32,0
2,1,de,8,0,103,8244,847777,0,0,0,0.0000000000000000e0,0
2,1,bite,32,0,103,8192,10952787,0,0,0,9.6700773427792447e-4,0
2,1,de,32,0,103,8236,1985991,0,0,0,6.4976378846799303e-6,0
2,1,bite,96,0,103,8192,33508412,0,0,0,1.7080460560883850e1,0
2,1,de,96,0,103,8228,5164199,0,0,0,3.2991186458384973e1,0
2,1,bite-depth4,32,0,103,8192,15067946,0,0,0,7.4900827748511700e-2,0
2,1,bite-batch16,32,0,103,8192,12874353,0,0,0,6.6599948517022639e-4,0
2,1,de-ask,32,0,103,8192,1988545,0,0,0,6.7383132201002264e-3,0
2,1,mode-sbx,12,32,103,8192,8073531,0,188142,7629340,8.5662308536505916e-1,0
2,1,mode-de,12,32,103,8192,6557097,0,1215996,5102935,8.5740578795406197e-1,0
2,1,mode-sbx,12,64,103,8192,10892307,0,1009780,9571633,8.6379249087850196e-1,0
2,1,mode-de,12,64,103,8192,8807106,0,1862888,6693795,8.6751750346627632e-1,0
2,1,mode-sbx,12,128,103,8192,14219989,0,1024781,12892011,8.5993842821149036e-1,0
2,1,mode-de,12,128,103,8192,13220083,0,1874173,11091004,8.6904321852994504e-1,0
2,1,mode-sbx,12,256,103,8192,23836204,0,860026,22672774,7.7564015459984392e-1,0
2,1,mode-de,12,256,103,8192,23311813,0,1785816,21242778,8.4832454346164532e-1,0
2,1,map-grid-sbx,12,256,103,8192,4100447,42263,0,0,1.6253954254166814e2,256
2,1,map-grid-sbx-micro,12,256,103,8192,0,0,181353,429537,0,0
2,1,map-grid-sbx,12,1024,103,8192,7091641,76835,0,0,4.7882488110767110e2,949
2,1,map-grid-sbx-micro,12,1024,103,8192,0,0,181964,1850888,0,0
2,1,map-grid-sbx,12,4096,103,8192,21648168,246091,0,0,7.7079310435303319e2,2478
2,1,map-grid-sbx-micro,12,4096,103,8192,0,0,180392,9523835,0,0
2,1,map-grid-iso,12,1024,103,8192,6977720,70035,0,0,4.0567600386125605e2,1019
2,1,map-grid-iso-micro,12,1024,103,8192,0,0,180592,2053363,0,0
2,1,map-cvt-sbx,12,256,103,8192,7611003,3219720,0,0,1.5921972648803401e2,256
2,1,map-cvt-sbx-micro,12,256,103,8192,0,0,3722326,400363,0,0
2,1,map-cvt-sbx,12,1024,103,8192,21565233,31588050,0,0,4.4868395719066211e2,968
2,1,map-cvt-sbx-micro,12,1024,103,8192,0,0,14529305,1948445,0,0
2,1,bite,6,0,102,8192,49670626,0,0,0,1.2535114393026227e1,0
2,1,de,6,0,102,8223,45343599,0,0,0,1.5295226918582406e1,0
2,1,bite,8,0,102,8192,5178250,0,0,0,4.6049755342276564e-29,0
2,1,de,8,0,102,8203,835488,0,0,0,0.0000000000000000e0,0
2,1,bite,32,0,102,8192,11297487,0,0,0,5.3091597129978287e-4,0
2,1,de,32,0,102,8245,1962947,0,0,0,3.5417816658361076e-7,0
2,1,bite,96,0,102,8192,32910752,0,0,0,1.4257464708625225e1,0
2,1,de,96,0,102,8237,5147113,0,0,0,7.7820990521143955e1,0
2,1,bite-depth4,32,0,102,8192,15115147,0,0,0,2.1855863915533205e-1,0
2,1,bite-batch16,32,0,102,8192,12444875,0,0,0,3.1664345725242865e-4,0
2,1,de-ask,32,0,102,8192,1981725,0,0,0,2.3318430133627190e0,0
2,1,mode-sbx,12,32,102,8192,8079640,0,194315,7646722,8.5066788230651591e-1,0
2,1,mode-de,12,32,102,8192,6593832,0,1206434,5134535,8.5941842596006190e-1,0
2,1,mode-sbx,12,64,102,8192,10218342,0,1022550,8944855,8.6567344498245069e-1,0
2,1,mode-de,12,64,102,8192,8809579,0,1905504,6662786,8.6778101890713677e-1,0
2,1,mode-sbx,12,128,102,8192,14128350,0,1020277,12837436,8.6618259087932337e-1,0
2,1,mode-de,12,128,102,8192,13056717,0,1870097,10930697,8.6993730918882961e-1,0
2,1,mode-sbx,12,256,102,8192,23109328,0,898978,21923895,7.4875737458304514e-1,0
2,1,mode-de,12,256,102,8192,23913591,0,1753082,21857536,8.4145472733249149e-1,0
2,1,map-grid-sbx,12,256,102,8192,4052113,42104,0,0,1.6339945498594162e2,256
2,1,map-grid-sbx-micro,12,256,102,8192,0,0,180152,394744,0,0
2,1,map-grid-sbx,12,1024,102,8192,7428339,90426,0,0,4.6872253192078153e2,990
2,1,map-grid-sbx-micro,12,1024,102,8192,0,0,180642,1937278,0,0
2,1,map-grid-sbx,12,4096,102,8192,17927043,275064,0,0,6.8462888326304187e2,2425
2,1,map-grid-sbx-micro,12,4096,102,8192,0,0,195775,6546902,0,0
2,1,map-grid-iso,12,1024,102,8192,6858710,89254,0,0,4.3908336568440893e2,1017
2,1,map-grid-iso-micro,12,1024,102,8192,0,0,194363,2049717,0,0
2,1,map-cvt-sbx,12,256,102,8192,7571194,2913429,0,0,1.5930603514977449e2,255
2,1,map-cvt-sbx-micro,12,256,102,8192,0,0,3733904,399482,0,0
2,1,map-cvt-sbx,12,1024,102,8192,21484582,35161161,0,0,4.3251631500471092e2,938
2,1,map-cvt-sbx-micro,12,1024,102,8192,0,0,14724048,1882716,0,0
2,1,de-window8,6,0,102,8192,77437750,0,0,0,1.8058767639340438e1,0
2,1,de-window8,32,0,102,8192,52734701,0,0,0,1.1749152610350377e-3,0
2,8,de-window8,6,0,102,8192,106952920,0,0,0,1.8058767639340438e1,0
2,8,de-window8,32,0,102,8192,91000294,0,0,0,1.1749152610350377e-3,0
2,1,bite-p32,32,0,102,8192,10722107,0,0,0,1.8942881724483020e-5,0
2,1,bite-p105,32,0,102,8192,13190494,0,0,0,5.3091597129978287e-4,0
2,1,bite-p297,32,0,102,8192,27029005,0,0,0,8.7859648007388904e-3,0
2,1,bite-p32,96,0,102,8192,13875466,0,0,0,3.3135735461576306e0,0
2,1,bite-p105,96,0,102,8192,20906224,0,0,0,5.0583707442814765e0,0
2,1,bite-p297,96,0,102,8192,41624986,0,0,0,1.4257464708625225e1,0
2,1,bite-p32,32,0,101,8192,7668638,0,0,0,5.0581867134273466e-4,0
2,1,bite-p105,32,0,101,8192,11302281,0,0,0,1.1826333068296398e-3,0
2,1,bite-p297,32,0,101,8192,22420444,0,0,0,1.9015635428459281e-2,0
2,1,bite-p32,96,0,101,8192,13712641,0,0,0,3.6352241649416750e0,0
2,1,bite-p105,96,0,101,8192,18352088,0,0,0,6.2246554574358610e0,0
2,1,bite-p297,96,0,101,8192,32988780,0,0,0,9.3274857814853984e0,0
2,8,de-window8,6,0,101,8192,77567565,0,0,0,1.1032465680287711e1,0
2,8,de-window8,32,0,101,8192,65198147,0,0,0,3.9601237166690626e0,0
2,1,de-window8,6,0,101,8192,77292260,0,0,0,1.1032465680287711e1,0
2,1,de-window8,32,0,101,8192,52583529,0,0,0,3.9601237166690626e0,0
2,1,bite,6,0,101,8192,52136426,0,0,0,1.7284696929097187e1,0
2,1,de,6,0,101,8204,46982214,0,0,0,1.4706560283588807e1,0
2,1,bite,8,0,101,8192,5107602,0,0,0,3.6977854932234928e-29,0
2,1,de,8,0,101,8211,884642,0,0,0,0.0000000000000000e0,0
2,1,bite,32,0,101,8192,11158224,0,0,0,1.1826333068296398e-3,0
2,1,de,32,0,101,8193,1983558,0,0,0,5.8523236436527079e-7,0
2,1,bite,96,0,101,8192,38094570,0,0,0,9.3274857814853984e0,0
2,1,de,96,0,101,8205,5193501,0,0,0,4.1982722582275272e1,0
2,1,bite-depth4,32,0,101,8192,15879194,0,0,0,8.2921804631106488e-2,0
2,1,bite-batch16,32,0,101,8192,13255112,0,0,0,1.4072638407990753e-3,0
2,1,de-ask,32,0,101,8192,2150538,0,0,0,5.6395006211537480e-1,0
2,1,mode-sbx,12,32,101,8192,8239318,0,195328,7804386,8.5413229059125528e-1,0
2,1,mode-de,12,32,101,8192,6622403,0,1229677,5159069,8.5889918485858974e-1,0
2,1,mode-sbx,12,64,101,8192,10199611,0,1075949,8874420,8.6447191155192660e-1,0
2,1,mode-de,12,64,101,8192,8936629,0,1913528,6781728,8.6709909331076906e-1,0
2,1,mode-sbx,12,128,101,8192,14385674,0,1040568,13076635,8.5784757927062072e-1,0
2,1,mode-de,12,128,101,8192,13500762,0,1932577,11307782,8.7031199968739070e-1,0
2,1,mode-sbx,12,256,101,8192,23575625,0,926717,22351822,7.6070505972010283e-1,0
2,1,mode-de,12,256,101,8192,23864910,0,1861334,21711768,8.4611453519658486e-1,0
2,1,map-grid-sbx,12,256,101,8192,4235288,43206,0,0,1.6296890243802014e2,256
2,1,map-grid-sbx-micro,12,256,101,8192,0,0,182404,403949,0,0
2,1,map-grid-sbx,12,1024,101,8192,7323629,89746,0,0,4.3370849534401907e2,977
2,1,map-grid-sbx-micro,12,1024,101,8192,0,0,182114,1917658,0,0
2,1,map-grid-sbx,12,4096,101,8192,19037670,273632,0,0,7.8716486204844477e2,2500
2,1,map-grid-sbx-micro,12,4096,101,8192,0,0,204368,6396022,0,0
2,1,map-grid-iso,12,1024,101,8192,6983317,69775,0,0,3.8514157682986286e2,1019
2,1,map-grid-iso-micro,12,1024,101,8192,0,0,193311,2036217,0,0
2,1,map-cvt-sbx,12,256,101,8192,7694568,3611989,0,0,1.6381405863139813e2,255
2,1,map-cvt-sbx-micro,12,256,101,8192,0,0,3797929,404610,0,0
2,1,map-cvt-sbx,12,1024,101,8192,21355440,32222456,0,0,4.5626875961058892e2,948
2,1,map-cvt-sbx-micro,12,1024,101,8192,0,0,14477032,1859340,0,0
3,1,bite,6,0,103,8192,51588872,0,0,0,1.0999741945697014e1,0
3,1,de,6,0,103,8213,44351352,0,0,0,1.7562742444003412e1,0
3,1,bite,8,0,103,8192,5154693,0,0,0,9.8607613152626476e-32,0
3,1,de,8,0,103,8244,852094,0,0,0,0.0000000000000000e0,0
3,1,bite,32,0,103,8192,11181429,0,0,0,9.6700773427792447e-4,0
3,1,de,32,0,103,8236,1964799,0,0,0,6.4976378846799303e-6,0
3,1,bite,96,0,103,8192,33830448,0,0,0,1.7080460560883850e1,0
3,1,de,96,0,103,8228,5119991,0,0,0,3.2991186458384973e1,0
3,1,bite-depth4,32,0,103,8192,14821671,0,0,0,7.4900827748511700e-2,0
3,1,bite-batch16,32,0,103,8192,12750503,0,0,0,6.6599948517022639e-4,0
3,1,de-ask,32,0,103,8192,1998300,0,0,0,6.7383132201002264e-3,0
3,1,mode-sbx,12,32,103,8192,8142082,0,191474,7707162,8.5662308536505916e-1,0
3,1,mode-de,12,32,103,8192,7050698,0,1313022,5468323,8.5740578795406197e-1,0
3,1,mode-sbx,12,64,103,8192,10276847,0,989402,9036487,8.6379249087850196e-1,0
3,1,mode-de,12,64,103,8192,9223150,0,1916801,7053058,8.6751750346627632e-1,0
3,1,mode-sbx,12,128,103,8192,14203710,0,1026198,12909422,8.5993842821149036e-1,0
3,1,mode-de,12,128,103,8192,13529966,0,2008190,11243681,8.6904321852994504e-1,0
3,1,mode-sbx,12,256,103,8192,24004721,0,882740,22832156,7.7564015459984392e-1,0
3,1,mode-de,12,256,103,8192,23389394,0,1789897,21311444,8.4832454346164532e-1,0
3,1,map-grid-sbx,12,256,103,8192,4116189,42083,0,0,1.6253954254166814e2,256
3,1,map-grid-sbx-micro,12,256,103,8192,0,0,204698,414845,0,0
3,1,map-grid-sbx,12,1024,103,8192,7053804,75704,0,0,4.7882488110767110e2,949
3,1,map-grid-sbx-micro,12,1024,103,8192,0,0,180602,1833481,0,0
3,1,map-grid-sbx,12,4096,103,8192,22234935,257658,0,0,7.7079310435303319e2,2478
3,1,map-grid-sbx-micro,12,4096,103,8192,0,0,181534,9815873,0,0
3,1,map-grid-iso,12,1024,103,8192,7238141,70546,0,0,4.0567600386125605e2,1019
3,1,map-grid-iso-micro,12,1024,103,8192,0,0,191999,2121826,0,0
3,1,map-cvt-sbx,12,256,103,8192,7738243,3237236,0,0,1.5921972648803401e2,256
3,1,map-cvt-sbx-micro,12,256,103,8192,0,0,3725661,400223,0,0
3,1,map-cvt-sbx,12,1024,103,8192,21869116,31826280,0,0,4.4868395719066211e2,968
3,1,map-cvt-sbx-micro,12,1024,103,8192,0,0,14566286,1948364,0,0
3,1,de-window8,6,0,103,8192,76271765,0,0,0,1.6371480183340484e1,0
3,1,de-window8,32,0,103,8192,52000181,0,0,0,3.5541467556733648e-3,0
3,8,de-window8,6,0,103,8192,83669235,0,0,0,1.6371480183340484e1,0
3,8,de-window8,32,0,103,8192,61032545,0,0,0,3.5541467556733648e-3,0
3,1,bite-p32,32,0,103,8192,7644903,0,0,0,4.5361790549789932e-5,0
3,1,bite-p105,32,0,103,8192,10934527,0,0,0,9.6700773427792447e-4,0
3,1,bite-p297,32,0,103,8192,22370139,0,0,0,2.4928796965558945e-2,0
3,1,bite-p32,96,0,103,8192,13524528,0,0,0,3.8914182234087442e0,0
3,1,bite-p105,96,0,103,8192,18579080,0,0,0,8.9081295314135751e0,0
3,1,bite-p297,96,0,103,8192,33702265,0,0,0,1.7080460560883850e1,0
3,1,bite-p32,32,0,104,8192,7616320,0,0,0,2.4975711573075037e-4,0
3,1,bite-p105,32,0,104,8192,11133477,0,0,0,3.6169467529794422e-4,0
3,1,bite-p297,32,0,104,8192,22757522,0,0,0,3.4445033272558122e-2,0
3,1,bite-p32,96,0,104,8192,13790459,0,0,0,6.5898883207883365e0,0
3,1,bite-p105,96,0,104,8192,18546601,0,0,0,1.1062224273248001e1,0
3,1,bite-p297,96,0,104,8192,31885640,0,0,0,1.5378600611905433e1,0
3,8,de-window8,6,0,104,8192,93699556,0,0,0,1.4405043239464188e1,0
3,8,de-window8,32,0,104,8192,91786570,0,0,0,4.6363254789597410e-5,0
3,1,de-window8,6,0,104,8192,79125762,0,0,0,1.4405043239464188e1,0
3,1,de-window8,32,0,104,8192,53378148,0,0,0,4.6363254789597410e-5,0
3,1,bite,6,0,104,8192,50971502,0,0,0,1.0996495809146211e1,0
3,1,de,6,0,104,8204,44079854,0,0,0,8.6494382297011967e0,0
3,1,bite,8,0,104,8192,5434324,0,0,0,2.0707598762051560e-30,0
3,1,de,8,0,104,8193,841186,0,0,0,1.9721522630525295e-31,0
3,1,bite,32,0,104,8192,14522080,0,0,0,3.6169467529794422e-4,0
3,1,de,32,0,104,8209,1988655,0,0,0,6.9691400111331344e-6,0
3,1,bite,96,0,104,8192,45980546,0,0,0,1.5378600611905433e1,0
3,1,de,96,0,104,8238,5175474,0,0,0,2.3235472737756442e1,0
3,1,bite-depth4,32,0,104,8192,15049214,0,0,0,8.8687204951218082e-2,0
3,1,bite-batch16,32,0,104,8192,12868340,0,0,0,1.0070149233337318e-3,0
3,1,de-ask,32,0,104,8192,2202096,0,0,0,4.5809930341223978e-2,0
3,1,mode-sbx,12,32,104,8192,8365088,0,191035,7935451,8.4985952082610272e-1,0
3,1,mode-de,12,32,104,8192,7043857,0,1342481,5462635,8.5943608478619982e-1,0
3,1,mode-sbx,12,64,104,8192,10448906,0,1051091,9130225,8.6426875377162304e-1,0
3,1,mode-de,12,64,104,8192,8817970,0,1886216,6678605,8.6749248667162970e-1,0
3,1,mode-sbx,12,128,104,8192,14416931,0,1021619,13116679,8.6333261533921268e-1,0
3,1,mode-de,12,128,104,8192,13127120,0,1900685,10963027,8.6994462669792416e-1,0
3,1,mode-sbx,12,256,104,8192,23671859,0,893605,22487127,7.8136118545204170e-1,0
3,1,mode-de,12,256,104,8192,26219786,0,1904791,23976686,8.3083285176903532e-1,0
3,1,map-grid-sbx,12,256,104,8192,4675221,44387,0,0,1.6383043090580401e2,256
3,1,map-grid-sbx-micro,12,256,104,8192,0,0,233342,458010,0,0
3,1,map-grid-sbx,12,1024,104,8192,9216400,78929,0,0,4.5418514443695187e2,984
3,1,map-grid-sbx-micro,12,1024,104,8192,0,0,180432,2164529,0,0
3,1,map-grid-sbx,12,4096,104,8192,18841596,285419,0,0,7.8055646566654934e2,2601
3,1,map-grid-sbx-micro,12,4096,104,8192,0,0,180532,7712485,0,0
3,1,map-grid-iso,12,1024,104,8192,7518122,88353,0,0,3.8094611255853226e2,1020
3,1,map-grid-iso-micro,12,1024,104,8192,0,0,296176,2145922,0,0
3,1,map-cvt-sbx,12,256,104,8192,8565809,3552420,0,0,1.5906982500649806e2,255
3,1,map-cvt-sbx-micro,12,256,104,8192,0,0,4129909,426403,0,0
3,1,map-cvt-sbx,12,1024,104,8192,23500883,34945778,0,0,4.3104334099115653e2,933
3,1,map-cvt-sbx-micro,12,1024,104,8192,0,0,17179402,2310279,0,0
3,1,bite,6,0,105,8192,53189283,0,0,0,1.0996503871417433e1,0
3,1,de,6,0,105,8205,49794751,0,0,0,1.5678160421199632e1,0
3,1,bite,8,0,105,8192,5574665,0,0,0,3.6484816866471796e-30,0
3,1,de,8,0,105,8214,963531,0,0,0,0.0000000000000000e0,0
3,1,bite,32,0,105,8192,12213603,0,0,0,1.4422417837912976e-3,0
3,1,de,32,0,105,8220,2111640,0,0,0,4.2111277752988254e-8,0
3,1,bite,96,0,105,8192,34984196,0,0,0,2.3004113003140873e1,0
3,1,de,96,0,105,8221,5553744,0,0,0,3.1530023253077029e1,0
3,1,bite-depth4,32,0,105,8192,16444445,0,0,0,2.6325073062184623e-1,0
3,1,bite-batch16,32,0,105,8192,13929098,0,0,0,5.6815964655758063e-4,0
3,1,de-ask,32,0,105,8192,2092962,0,0,0,1.9316429981120489e1,0
3,1,mode-sbx,12,32,105,8192,8679161,0,200012,8217567,8.5439131660484036e-1,0
3,1,mode-de,12,32,105,8192,6969426,0,1246333,5447724,8.5821199322334074e-1,0
3,1,mode-sbx,12,64,105,8192,11003110,0,1087110,9583613,8.6591002229428138e-1,0
3,1,mode-de,12,64,105,8192,9593308,0,2006027,7333425,8.6737926438746404e-1,0
3,1,mode-sbx,12,128,105,8192,15343327,0,1080488,13977208,8.6145784182114615e-1,0
3,1,mode-de,12,128,105,8192,15437939,0,2352673,12803002,8.7006091594969159e-1,0
3,1,mode-sbx,12,256,105,8192,23955346,0,938175,22725916,7.3911761109383511e-1,0
3,1,mode-de,12,256,105,8192,23719611,0,1810142,21606158,8.3601760157556892e-1,0
3,1,map-grid-sbx,12,256,105,8192,4081206,42774,0,0,1.6238762222760559e2,256
3,1,map-grid-sbx-micro,12,256,105,8192,0,0,193401,389216,0,0
3,1,map-grid-sbx,12,1024,105,8192,7145151,77026,0,0,4.6953163196095193e2,982
3,1,map-grid-sbx-micro,12,1024,105,8192,0,0,180893,1914743,0,0
3,1,map-grid-sbx,12,4096,105,8192,20441104,259962,0,0,8.0646902045290881e2,2498
3,1,map-grid-sbx-micro,12,4096,105,8192,0,0,180582,6446108,0,0
3,1,map-grid-iso,12,1024,105,8192,6699109,73991,0,0,3.7309797616382815e2,1017
3,1,map-grid-iso-micro,12,1024,105,8192,0,0,180963,1919340,0,0
3,1,map-cvt-sbx,12,256,105,8192,7466004,2879156,0,0,1.6276406563596043e2,255
3,1,map-cvt-sbx-micro,12,256,105,8192,0,0,3733182,430348,0,0
3,1,map-cvt-sbx,12,1024,105,8192,21791128,28024534,0,0,4.2611129480401723e2,955
3,1,map-cvt-sbx-micro,12,1024,105,8192,0,0,14617233,2067273,0,0
3,1,de-window8,6,0,105,8192,73397046,0,0,0,3.7293553202786079e1,0
3,1,de-window8,32,0,105,8192,52684102,0,0,0,1.9867167906193206e-5,0
3,8,de-window8,6,0,105,8192,84027074,0,0,0,3.7293553202786079e1,0
3,8,de-window8,32,0,105,8192,73559941,0,0,0,1.9867167906193206e-5,0
3,1,bite-p32,32,0,105,8192,7585483,0,0,0,2.1101785874328078e-4,0
3,1,bite-p105,32,0,105,8192,11524246,0,0,0,1.4422417837912976e-3,0
3,1,bite-p297,32,0,105,8192,24084241,0,0,0,1.1544909245503823e-2,0
3,1,bite-p32,96,0,105,8192,14130010,0,0,0,5.3934937726627510e0,0
3,1,bite-p105,96,0,105,8192,19093504,0,0,0,7.5083041076306554e0,0
3,1,bite-p297,96,0,105,8192,32941720,0,0,0,2.3004113003140873e1,0
3,1,bite,6,0,101,8192,49835462,0,0,0,1.7284696929097187e1,0
3,1,de,6,0,101,8204,47780306,0,0,0,1.4706560283588807e1,0
3,1,bite,8,0,101,8192,5177056,0,0,0,3.6977854932234928e-29,0
3,1,de,8,0,101,8211,944603,0,0,0,0.0000000000000000e0,0
3,1,bite,32,0,101,8192,11409042,0,0,0,1.1826333068296398e-3,0
3,1,de,32,0,101,8193,1961814,0,0,0,5.8523236436527079e-7,0
3,1,bite,96,0,101,8192,33839743,0,0,0,9.3274857814853984e0,0
3,1,de,96,0,101,8205,5773535,0,0,0,4.1982722582275272e1,0
3,1,bite-depth4,32,0,101,8192,16480760,0,0,0,8.2921804631106488e-2,0
3,1,bite-batch16,32,0,101,8192,13236344,0,0,0,1.4072638407990753e-3,0
3,1,de-ask,32,0,101,8192,2024850,0,0,0,5.6395006211537480e-1,0
3,1,mode-sbx,12,32,101,8192,8371327,0,193207,7918406,8.5413229059125528e-1,0
3,1,mode-de,12,32,101,8192,6975375,0,1262624,5475162,8.5889918485858974e-1,0
3,1,mode-sbx,12,64,101,8192,10438791,0,1019957,9113987,8.6447191155192660e-1,0
3,1,mode-de,12,64,101,8192,8931300,0,1945102,6741183,8.6709909331076906e-1,0
3,1,mode-sbx,12,128,101,8192,14724735,0,1045803,13403772,8.5784757927062072e-1,0
3,1,mode-de,12,128,101,8192,13491038,0,1911299,11259213,8.7031199968739070e-1,0
3,1,mode-sbx,12,256,101,8192,24029558,0,978323,22750609,7.6070505972010283e-1,0
3,1,mode-de,12,256,101,8192,24442210,0,1814047,22120630,8.4611453519658486e-1,0
3,1,map-grid-sbx,12,256,101,8192,4220085,43025,0,0,1.6296890243802014e2,256
3,1,map-grid-sbx-micro,12,256,101,8192,0,0,182645,413513,0,0
3,1,map-grid-sbx,12,1024,101,8192,7684653,87472,0,0,4.3370849534401907e2,977
3,1,map-grid-sbx-micro,12,1024,101,8192,0,0,338160,2003597,0,0
3,1,map-grid-sbx,12,4096,101,8192,20735798,315445,0,0,7.8716486204844477e2,2500
3,1,map-grid-sbx-micro,12,4096,101,8192,0,0,192289,6603305,0,0
3,1,map-grid-iso,12,1024,101,8192,7085942,71267,0,0,3.8514157682986286e2,1019
3,1,map-grid-iso-micro,12,1024,101,8192,0,0,185149,2042756,0,0
3,1,map-cvt-sbx,12,256,101,8192,7831654,3565109,0,0,1.6381405863139813e2,255
3,1,map-cvt-sbx-micro,12,256,101,8192,0,0,3826232,416778,0,0
3,1,map-cvt-sbx,12,1024,101,8192,21993763,32376799,0,0,4.5626875961058892e2,948
3,1,map-cvt-sbx-micro,12,1024,101,8192,0,0,14551985,2029577,0,0
3,1,de-window8,6,0,101,8192,92620896,0,0,0,1.1032465680287711e1,0
3,1,de-window8,32,0,101,8192,73931821,0,0,0,3.9601237166690626e0,0
3,8,de-window8,6,0,101,8192,73627032,0,0,0,1.1032465680287711e1,0
3,8,de-window8,32,0,101,8192,44755342,0,0,0,3.9601237166690626e0,0
3,1,bite-p32,32,0,101,8192,9114206,0,0,0,5.0581867134273466e-4,0
3,1,bite-p105,32,0,101,8192,12966878,0,0,0,1.1826333068296398e-3,0
3,1,bite-p297,32,0,101,8192,27457891,0,0,0,1.9015635428459281e-2,0
3,1,bite-p32,96,0,101,8192,14169390,0,0,0,3.6352241649416750e0,0
3,1,bite-p105,96,0,101,8192,20825543,0,0,0,6.2246554574358610e0,0
3,1,bite-p297,96,0,101,8192,45498821,0,0,0,9.3274857814853984e0,0
3,1,bite-p32,32,0,102,8192,7523180,0,0,0,1.8942881724483020e-5,0
3,1,bite-p105,32,0,102,8192,11101649,0,0,0,5.3091597129978287e-4,0
3,1,bite-p297,32,0,102,8192,22623481,0,0,0,8.7859648007388904e-3,0
3,1,bite-p32,96,0,102,8192,13940154,0,0,0,3.3135735461576306e0,0
3,1,bite-p105,96,0,102,8192,18483516,0,0,0,5.0583707442814765e0,0
3,1,bite-p297,96,0,102,8192,33323956,0,0,0,1.4257464708625225e1,0
3,8,de-window8,6,0,102,8192,73211982,0,0,0,1.8058767639340438e1,0
3,8,de-window8,32,0,102,8192,41789565,0,0,0,1.1749152610350377e-3,0
3,1,de-window8,6,0,102,8192,82122195,0,0,0,1.8058767639340438e1,0
3,1,de-window8,32,0,102,8192,54268508,0,0,0,1.1749152610350377e-3,0
3,1,bite,6,0,102,8192,53187009,0,0,0,1.2535114393026227e1,0
3,1,de,6,0,102,8223,47898064,0,0,0,1.5295226918582406e1,0
3,1,bite,8,0,102,8192,5248253,0,0,0,4.6049755342276564e-29,0
3,1,de,8,0,102,8203,835237,0,0,0,0.0000000000000000e0,0
3,1,bite,32,0,102,8192,13233159,0,0,0,5.3091597129978287e-4,0
3,1,de,32,0,102,8245,1974935,0,0,0,3.5417816658361076e-7,0
3,1,bite,96,0,102,8192,35045518,0,0,0,1.4257464708625225e1,0
3,1,de,96,0,102,8237,5113692,0,0,0,7.7820990521143955e1,0
3,1,bite-depth4,32,0,102,8192,16442672,0,0,0,2.1855863915533205e-1,0
3,1,bite-batch16,32,0,102,8192,13031746,0,0,0,3.1664345725242865e-4,0
3,1,de-ask,32,0,102,8192,2769411,0,0,0,2.3318430133627190e0,0
3,1,mode-sbx,12,32,102,8192,8620072,0,233506,8145898,8.5066788230651591e-1,0
3,1,mode-de,12,32,102,8192,6751998,0,1225927,5289603,8.5941842596006190e-1,0
3,1,mode-sbx,12,64,102,8192,10139730,0,1011875,8878100,8.6567344498245069e-1,0
3,1,mode-de,12,64,102,8192,8738490,0,1909181,6589317,8.6778101890713677e-1,0
3,1,mode-sbx,12,128,102,8192,14494799,0,1130090,13091325,8.6618259087932337e-1,0
3,1,mode-de,12,128,102,8192,14116459,0,2382984,11419705,8.6993730918882961e-1,0
3,1,mode-sbx,12,256,102,8192,23569295,0,872164,22405973,7.4875737458304514e-1,0
3,1,mode-de,12,256,102,8192,24069489,0,1751312,22017138,8.4145472733249149e-1,0
3,1,map-grid-sbx,12,256,102,8192,4234096,42314,0,0,1.6339945498594162e2,256
3,1,map-grid-sbx-micro,12,256,102,8192,0,0,181574,394854,0,0
3,1,map-grid-sbx,12,1024,102,8192,7402178,75915,0,0,4.6872253192078153e2,990
3,1,map-grid-sbx-micro,12,1024,102,8192,0,0,181353,2037529,0,0
3,1,map-grid-sbx,12,4096,102,8192,18156493,269455,0,0,6.8462888326304187e2,2425
3,1,map-grid-sbx-micro,12,4096,102,8192,0,0,203507,6667120,0,0
3,1,map-grid-iso,12,1024,102,8192,6952260,70797,0,0,4.3908336568440893e2,1017
3,1,map-grid-iso-micro,12,1024,102,8192,0,0,180552,2038991,0,0
3,1,map-cvt-sbx,12,256,102,8192,7943754,2887438,0,0,1.5930603514977449e2,255
3,1,map-cvt-sbx-micro,12,256,102,8192,0,0,4153655,452702,0,0
3,1,map-cvt-sbx,12,1024,102,8192,21557385,35947787,0,0,4.3251631500471092e2,938
3,1,map-cvt-sbx-micro,12,1024,102,8192,0,0,14942263,1871448,0,0
```

</details>

<details>
<summary>Appendix D · Independent DE quality audit, flat-objective controls and centered-Sphere checks</summary>

```text
seed,dim,n,reported,lowest_observed,reevaluated
101,6,8204,1.47065602835888072e1,1.47065602835888072e1,1.47065602835888072e1
101,8,8211,0.00000000000000000e0,0.00000000000000000e0,0.00000000000000000e0
101,32,8193,5.85232364365270788e-7,5.85232364365270788e-7,5.85232364365270788e-7
101,96,8205,4.19827225822752723e1,4.19827225822752723e1,4.19827225822752723e1
102,6,8223,1.52952269185824061e1,1.52952269185824061e1,1.52952269185824061e1
102,8,8203,0.00000000000000000e0,0.00000000000000000e0,0.00000000000000000e0
102,32,8245,3.54178166583610760e-7,3.54178166583610760e-7,3.54178166583610760e-7
102,96,8237,7.78209905211439548e1,7.78209905211439548e1,7.78209905211439548e1
103,6,8213,1.75627424440034119e1,1.75627424440034119e1,1.75627424440034119e1
103,8,8244,0.00000000000000000e0,0.00000000000000000e0,0.00000000000000000e0
103,32,8236,6.49763788467993034e-6,6.49763788467993034e-6,6.49763788467993034e-6
103,96,8228,3.29911864583849734e1,3.29911864583849734e1,3.29911864583849734e1
104,6,8204,8.64943822970119669e0,8.64943822970119669e0,8.64943822970119669e0
104,8,8193,1.97215226305252951e-31,1.97215226305252951e-31,1.97215226305252951e-31
104,32,8209,6.96914001113313440e-6,6.96914001113313440e-6,6.96914001113313440e-6
104,96,8238,2.32354727377564423e1,2.32354727377564423e1,2.32354727377564423e1
105,6,8205,1.56781604211996317e1,1.56781601341660632e1,1.56781604211996317e1
105,8,8214,0.00000000000000000e0,0.00000000000000000e0,0.00000000000000000e0
105,32,8220,4.21112777529882535e-8,4.21112777529882535e-8,4.21112777529882535e-8
105,96,8221,3.15300232530770295e1,3.15300232530770295e1,3.15300232530770295e1

flat: fixture,entry,seed,n,reported,known_value
centered: fixture,entry,seed,first_value,first_zero_evaluation
flat,serial,101,64,1.79769313486231571e308,7
flat,batch,101,32,1.79769313486231571e308,7
flat,window,101,32,1.79769313486231571e308,7
centered,bite,101,2.65784568252620602e2,Some(2835)
flat,serial,102,64,1.79769313486231571e308,7
flat,batch,102,32,1.79769313486231571e308,7
flat,window,102,32,1.79769313486231571e308,7
centered,bite,102,1.92390573588701955e2,Some(2590)
flat,serial,103,64,1.79769313486231571e308,7
flat,batch,103,32,1.79769313486231571e308,7
flat,window,103,32,1.79769313486231571e308,7
centered,bite,103,1.21077815988775981e2,Some(2777)
flat,serial,104,64,1.79769313486231571e308,7
flat,batch,104,32,1.79769313486231571e308,7
flat,window,104,32,1.79769313486231571e308,7
centered,bite,104,1.71421388786361888e2,Some(2881)
flat,serial,105,64,1.79769313486231571e308,7
flat,batch,105,32,1.79769313486231571e308,7
flat,window,105,32,1.79769313486231571e308,7
centered,bite,105,1.68669724814175936e2,Some(3246)
```

</details>
