# fcmaes-rust optimization implementation log

**27 November 2026 · BiteOpt, DE, MODE and MAP-Elites**

I ran all 80 change proposals and retained 25 in the integrated patch. The patch applies to `d7ff66aebc388ac19b698742c7678986e7a48d58`. It contains changes to the four core optimizer modules, the changelog and the DE guidance.

DE now reports the best finite point observed across its drivers. The constant-7 regression returned `f64::MAX` before the change. It now returns 7 and meets a stop target of 8. On the pinned production comparison, two DE rows reported better incumbents for the same seeded workload. The optimization-only comparisons against the corrected DE reference matched the recorded traces and results.

## Patched crate versus pinned main

The standalone executable calls each checkout's compiled `fcmaes-core` API. The millisecond columns are medians of the per-seed medians. Speedup is the median of paired time ratios. Brackets contain descriptive 95% bootstrap intervals over seed medians. The eight-retry rows measure throughput for independent seeded optimizations in a single eight-thread pool. The delayed DE rows use one ordered window of eight candidates.

| Optimizer / workload | Threads | Pinned ms | Patched ms | Speedup | Matching replay pairs |
|---|---:|---:|---:|---:|---:|
| BiteOpt · shifted Sphere, d=96, P=297 | 1 | 33.415 | 19.773 | 1.75× [1.69, 1.91] | 7/7 |
| BiteOpt · Rastrigin, d=32, direct | 1 | 24.674 | 20.460 | 1.23× [1.15, 1.34] | 7/7 |
| BiteOpt · Cassini1, d=6, direct | 1 | 206.326 | 200.063 | 1.03× [0.92, 1.06] | 7/7 |
| BiteOpt · shifted Sphere, d=32, batch 64 | 1 | 67.028 | 39.032 | 1.60× [1.38, 1.82] | 7/7 |
| BiteOpt · eight Rosenbrock runs | 8 | 27.323 | 22.424 | 1.49× [1.22, 1.66] | 7/7 |
| DE · shifted Sphere, d=32, direct | 1 | 18.198 | 16.176 | 1.12× [1.07, 1.18] | 7/7 |
| DE · shifted Sphere, d=96, population ask/tell | 1 | 31.741 | 27.537 | 1.14× [1.07, 1.24] | 6/7 |
| DE · shifted Sphere, d=32, ordered window 8 | 8 | 25.624 | 11.658 | 2.82× [0.96, 7.07] | 6/7 |
| DE · uneven callback costs, ordered window 8 | 8 | 88.294 | 63.874 | 1.43× [1.14, 4.76] | 7/7 |
| DE · eight shifted Sphere runs | 8 | 5.062 | 4.944 | 1.04× [0.95, 1.09] | 7/7 |
| MODE · ZDT1, SBX, P=32 | 1 | 17.538 | 9.819 | 1.74× [1.71, 1.98] | 7/7 |
| MODE · ZDT1, SBX, P=256 | 1 | 51.765 | 11.285 | 4.54× [4.36, 5.45] | 7/7 |
| MODE · ZDT1, DE, P=256 | 1 | 43.049 | 8.525 | 5.22× [4.94, 6.26] | 7/7 |
| MODE · four constraints, two objectives, P=128 | 1 | 27.230 | 11.862 | 2.31× [2.28, 2.63] | 7/7 |
| MODE · DTLZ2, five objectives, P=256 | 1 | 58.538 | 48.118 | 1.23× [1.15, 1.72] | 7/7 |
| MODE · eight ZDT1 runs | 8 | 22.905 | 13.116 | 1.84× [1.37, 2.28] | 7/7 |
| MAP-Elites · 60-cell ragged grid, SBX | 1 | 5.822 | 5.255 | 1.14× [1.09, 1.28] | 7/7 |
| MAP-Elites · 60-cell grid with CMA emitter | 1 | 4.178 | 4.192 | 1.00× [0.97, 1.04] | 7/7 |
| MAP-Elites · 4,096-cell grid, SBX | 1 | 45.774 | 10.703 | 4.13× [3.56, 5.82] | 7/7 |
| MAP-Elites · 1,024-center 2D CVT with cold setup | 1 | 75.671 | 21.836 | 3.56× [2.51, 4.07] | 7/7 |
| MAP-Elites · 256-center 8D CVT construction | 1 | 19.978 | 19.981 | 1.01× [0.98, 1.61] | 7/7 |
| MAP-Elites · eight 1,024-cell grid runs | 8 | 23.542 | 16.034 | 1.72× [1.16, 2.36] | 7/7 |

The two changed DE results came from population ask/tell at seed 104 (7.11124878299244 → 7.10820060004822) and ordered-window DE at seed 105 (3.05579481534345e-10 → 3.01029094659270e-10). The actual callback counts matched on both. Live delayed DE timing had the widest intervals. Against D01, returning an owned row gave a 1.36× median with uneven callback latencies at eight threads and a 1.03× median on the costly callback. That distinction matters when choosing inner parallelism.

The two-objective MODE winner calculates every front, including ties in either component. The general pairwise path remains in use for three or more objectives. MAP-Elites uses an exact tree for 2D CVT with at least 256 niches. The isolated 1,024-center lookup comparison was 3.87× versus the optimized flat scan, and the 1,024-center Lloyd setup comparison was 3.58× with one thread versus the same code with its fit-time tree disabled. The 8D setup comparison supports keeping the scan for high descriptor dimensions.

## Search changes on held-out seeds

Every search proposal received a 30-seed hold-out on at least two workloads, with three workload comparisons for most. B18 and D18 also ran fixed-wall checks. The equal-evaluation DE hold-outs used each run's best observed value through the first requested N callbacks to account for serial generation overshoot. MODE measures feasible hypervolume for two objectives and IGD+ for higher-dimensional fixtures. MAP-Elites keeps the same geometry and measures its native QD-score. Larger hypervolume and QD-score are preferred.

| Proposal / held-out workload | Reference → experiment | Decision |
|---|---:|---|
| BiteOpt cap P=105, 96-D shifted Sphere, 8,192 calls | 16.762 → 8.4577 (Median loss) | Held |
| BiteOpt cap P=105, 96-D shifted Sphere, 40 ms | 17.981 → 0.73885 (Median loss) | Held |
| BiteOpt cap P=105, 96-D Rastrigin, 40 ms | 583.39 → 441.85 (Median loss) | Held |
| BiteOpt expanding batch, expensive callback, versus fixed width 16 | 34.563 → 34.563 (Median loss) | Rejected |
| BiteOpt mixed-depth retries, Cassini1, versus all depth 4 | 5.4159 → 5.7174 (Median loss) | Rejected |
| DE exclusion sampling, Cassini1, 40 ms | 14.665 → 15.268 (Median loss) | Rejected |
| DE adaptive temporal locality, Cassini1 | 12.51 → 13.455 (Median prefix loss) | Rejected |
| DE diversity-based aging, Cassini1 | 12.51 → 11.049 (Median prefix loss) | Held |
| MODE alternating reproduction, ZDT2, versus fixed SBX | 0.53151 → 0.53387 (Feasible HV) | Rejected |
| MODE duplicate suppression, tied objective rows | 1.21 → 1.21 (Feasible HV) | Rejected |
| MAP-Elites seeded-parent injection, 4,096-cell grid | 782.53 → 778.26 (QD-score) | Rejected |
| MAP-Elites alternating emitters, 1,024-cell grid, versus SBX | 447.46 → 430.26 (QD-score) | Rejected |

For a short high-dimensional BiteOpt run, the cap is a promising setting to try explicitly. Its largest observed gains come from 96-dimensional workloads. A dimension and budget sweep would be needed to generalize that rule.

## All 80 iterations

### BiteOpt

| ID | Primary screen speedup | Decision | Observed result |
|---|---:|---|---|
| B01 | 1.68× | Retained | Slice rotation and bulk cost movement. The fixed-population win survived against the actual patched crate and the insertion oracle checks equal costs and the evicted slot. |
| B02 | 1.02× | Held | Row-major i128 sums matched the reference. The timing interval included parity and storage grew by one vector per population. |
| B03 | 0.89× | Rejected | Strict partial-distance exit kept tie behavior. The extra branch slowed the primary replay. |
| B04 | 1.15× | Held | Skipped zeroing on complete-overwrite paths. A poison-buffer run reproduced the trace, with negligible measured benefit. |
| B05 | 1.02× | Held | Borrowed Sol5 and Sol5b donors preserved trajectories. End-to-end gain was below the threshold. |
| B06 | 1.80× | Rejected | Flat encoded rows beat the original at large P. Direct comparisons with bulk insertion did not establish an incremental win across the four tested workloads. |
| B07 | 1.04× | Held | The captured-position identity fast path stayed exact. The median gain was below 5%. |
| B08 | 1.01× | Rejected | Packed selector entries reproduced the values and IDs. End-to-end performance was effectively unchanged. |
| B09 | 1.12× | Retained | Per-row recycling improved the 64-candidate batch. Pending and final short batches remained exact. |
| B10 | 1.03× | Held | Driver-private reused callback rows matched the public ask/tell trace. The improvement was too small and noisy. |
| B11 | 0.99× | Rejected | Deep batch sort scratch and one-time cost sanitization preserved feedback order, with essentially the same runtime. |
| B12 | 0.99× | Rejected | The pinned implementation already returns the pushed encoded vector to its original population. Recycling the entire candidate after the push showed parity. |
| B13 | 1.05× | Held | Borrowed Nelder–Mead trials removed clones. The live benefit was uncertain. |
| B14 | 1.00× | Rejected | Spherical normalized and real scratch reduced transient work. The live timing stayed at parity. |
| B15 | 1.01× | Rejected | Exact spherical weight caching preserved all sampled results. The lookup and retained weights provided no reliable end-to-end gain. |
| B16 | 0.96× | Rejected | Direct initialization queue construction and selector reuse kept the random stream but slowed short retries. |
| B17 | 1.02× | Held | The fused wrap/decode loop matched the stream. The end-to-end gain was below threshold. |
| B18 | 1.78× | Held | Capping P at 105 had a strong 96-D fixed-wall result. Wider dimension, objective and budget evidence is needed to choose a default rule. |
| B19 | 0.98× | Rejected | An expanding batch beat width 8 with the costly callback. Width 16 already delivered the same trajectories and similar timing. |
| B20 | 0.95× | Rejected | A 4+4 depth-1/depth-4 portfolio lost to the all-depth-4 control on Cassini1 and to depth 1 on 32-D Rastrigin. |

### DE

| ID | Primary screen speedup | Decision | Observed result |
|---|---:|---|---|
| D01 | 0.97× | Retained | Track the best finite observation independently. The constant-7 fixture now reports 7 in every driver and meets the stop target. |
| D02 | 0.95× | Rejected | Borrowed donor rows for the immediate-update path preserved the trace and had a slower median. |
| D03 | 1.03× | Held | Swapping in the serial trial buffer recovered part of D02's loss. It did not establish a win from the corrected base. |
| D04 | 1.03× | Held | The temporal-locality single-pass clamp preserved the trace. Timings were inconclusive. |
| D05 | 0.97× | Held | Borrowed ask-path donors matched the corrected reference. The measured gain was inconclusive. |
| D06 | 0.98× | Rejected | Fused ask crossover/clamping failed to improve on D05 and retained its dependence on that prototype. |
| D07 | 1.05× | Held | Precomputed integer indices helped the mixed-integer screen. The gain shrank on top of the integrated change. |
| D08 | 0.92× | Rejected | In-place normal sampling matched results but slowed the normal-start workload. |
| D09 | 0.98× | Held | The contiguous history and reused sigma buffer preserved samples. The measured gain was inconclusive. |
| D10 | 1.15× | Retained | Move accepted owned rows into the population and move the displaced row into history. Public ask/tell measurements showed a clear combined gain. |
| D11 | 1.21× | Held | The single improvement deque maintained FIFO semantics. Confirmation on top of accepted changes showed around 1% median gain. |
| D12 | 0.78× | Held | A borrowed batch view removed an owned Rust snapshot. Timing scatter and the additional public API outweighed the evidence. |
| D13 | 1.09× | Retained | Sequence-indexed completion slots preserve ordered tells with bounded storage. Integrated into the candidate ownership transfer. |
| D14 | 1.46× | Held | Inline delayed execution was fast for the cheap callback. Without explicit callback purity the wider-window path can change objective invocation order. A one-slot version was also timed and omitted. |
| D15 | 3.00× | Retained in modified form | The decode-buffer recycling experiment motivated moving the owned trial to the worker and returning it to the coordinator. The usual callback borrows that same buffer. |
| D16 | 0.65× | Rejected | Persistent workers with bounded feeds were slower on the isolated pool and would reserve pool slots across nested callers. |
| D17 | 1.05× | Held | Contiguous population/history slabs matched traces, with an uncertain end-to-end gain. |
| D18 | 1.13× | Rejected | Exclusion-index sampling helped some equal-evaluation cases. Fixed-wall Cassini1 and the rotated example deteriorated. |
| D19 | 1.07× | Rejected | An adaptive temporal-locality gate gave mixed hold-out quality. Cassini1 deteriorated. |
| D20 | 1.00× | Held | Diversity-scaled aging helped some hold-out outcomes. The plateau was unchanged and the confidence across workloads was insufficient. |

### MODE

| ID | Primary screen speedup | Decision | Observed result |
|---|---:|---|---|
| M01 | 1.27× | Retained | Rank by index and reuse two survivor buffers. Exact output across normal, constrained and odd-population cases. |
| M02 | 1.68× | Retained | Exact all-front Fenwick ranking for two objectives. The strongest MODE speedup, with duplicates and signed zeros verified against an independent oracle. |
| M03 | 0.69× | Rejected | Early exit from the pair comparator slowed both DTLZ2 and the five-objective case. |
| M04 | 1.05× | Held | Packed adjacency helped the dominance chain but offered an uncertain improvement on DTLZ2 and adds a dense memory cost. |
| M05 | 1.06× | Retained | Bucket by rank with stable input order. This removes per-rank full scans after the faster ranker. |
| M06 | 0.93× | Held | Reused front and adjacency scratch preserved ranks. Timing and retained graph memory favored the current fallback. |
| M07 | 1.03× | Held | Indexed crowding scratch agreed with the reference. Full-run gain was inconclusive. |
| M08 | 0.92× | Rejected | Partial split-front selection preserved the sort key and reversed tie order. The wide-front median was slower. |
| M09 | 1.16× | Retained | Borrow objective and constraint slices. The constrained replay preserved survivors and improved timing. |
| M10 | 1.08× | Retained | Replace the constraint-rank matrix with columnwise accumulation in the original order. |
| M11 | 1.00× | Retained with survivor buffers | Shape-check the full batch before copying sanitized values into reusable rows. The isolated timing was neutral. |
| M12 | 1.04× | Held | The borrowed offspring view matched the owned API. Gains were below the threshold for a new exported interface. |
| M13 | 1.08× | Retained | Fill the existing DE offspring row from borrowed donors. Standard and mixed-integer odd-sized cases matched the reference. |
| M14 | 1.13× | Held | Fused SBX generation eliminated the beta matrix but added only about 4% in the combined ablation, with an interval spanning parity. |
| M15 | 0.98× | Held | In-place clamping matched the emitted candidates. Neither the isolated test nor the combined variation ablation established a reliable gain. |
| M16 | 1.95× | Retained | Exact single-objective grouping preserved the existing reversed pre-sort and ties, with a clear speedup in that mode. |
| M17 | 0.97× | Rejected | Blocked, objective-major comparison was slower than M03, which had already lost to the original fallback. |
| M18 | 1.01× | Held | Parallel graph construction gained over M03 at eight threads. The gain versus the original fallback was uncertain and the one-thread case regressed. |
| M19 | 1.01× | Rejected | The schedule gave small hypervolume gains versus fixed SBX. The fixed-DE control and 3-objective IGD+ did not support changing the schedule. |
| M20 | 0.93× (hold-out replay) | Rejected | The duplicate-suppression prototype exposed a zero-slots corner case, then passed a fixed replay. It slowed tied and many-objective cases without a meaningful quality gain. |

### MAP-Elites

| ID | Primary screen speedup | Decision | Observed result |
|---|---:|---|---|
| Q01 | 1.09× | Retained for large archives | Use the occupied-only sort as a sparse rebuild path inside Q02. Small archives use the existing stable sort with reused index storage. |
| Q02 | 2.81× | Retained | Merge changed niche IDs with the prior order and rebuild for dense changes. The large-grid gain survived the full driver test. |
| Q03 | 1.20× | Retained | Store CVT centers in contiguous rows and specialize 2D/4D scan distance. The low-dimensional lookup replay matched. |
| Q04 | 1.16× | Retained for 2D | Use exact nearest-center search from 256 centers. A 4D test lost to the scan and the shipped index dispatch is 2D. |
| Q05 | 3.04× | Retained | Cache the ragged grid layout and normalize in scalars. Every niche center and endpoint was checked. |
| Q06 | 1.33× | Retained | Compute regular grid membership without allocating center rows. |
| Q07 | 1.39× | Retained | Share immutable CVT geometry and the tree across archive clones. Captured elite state remains independently owned. |
| Q08 | 1.01× | Held | Stack descriptor encoding matched the generic scan. The 8D timing showed little benefit. |
| Q09 | 1.28× | Held | Parallel prelookup and ordered commit preserved collisions. The scan-path speedup was noisy and the retained tree reduces the work available to parallelize. |
| Q10 | 1.00× | Held | The commit-only native driver preserved visits and order. Timing did not justify exposing another archive update API. |
| Q11 | 1.05× | Held | Borrowed parent rows helped Iso+LineDD. SBX evidence and integration benefit were uncertain. |
| Q12 | 1.03× | Held | Reusable offspring storage preserved the odd chunk behavior. Runtime was effectively unchanged. |
| Q13 | 1.11× | Retained | Skip SBX power work when the randomly chosen mask overwrites beta. Draw order and rows matched. |
| Q14 | 1.00× | Rejected | Reusable Lloyd partials kept deterministic reduction. The eight-thread timing was near parity or slower. |
| Q15 | 1.69× | Retained | Serial assignment of k-means++ distances at small sample counts avoids excessive pool work. Center assignment and later RNG continuation were exact. |
| Q16 | 3.22× | Retained for 2D | Build an exact center search tree for large 2D Lloyd assignments. The 8D alternative was slower and uses the scan path. |
| Q17 | 1.40× | Retained | QD-score traverses the stored qualities without collecting them. The positive and negative rules and measured values matched. |
| Q18 | 0.95× | Held | Reused CMA emitter comparison scratch preserved stopping and QD-score. Live end-to-end timing stayed near parity. |
| Q19 | 1.17× | Rejected | A 5% seeded-parent injection gained in the screening grid. Held-out coverage/score evidence did not support a default change. |
| Q20 | 0.90× | Rejected | The SBX/Iso portfolio improved over Iso alone and lost to SBX in the primary and held-out SBX comparisons. |

## Verification

- The final workspace passed **435 tests** including doctests. Locked workspace Clippy and formatting checks passed. `fcmaes-core` checked on Rust 1.88.0. The documentation check resolved 938 source links.
- Native regression tests exercise DE ties, nonfinite values, exact delayed budgets and cancellation before a temporal-locality call. MODE was compared with an independent pairwise front oracle. MAP-Elites checks tree ties against a scan and compares incremental archive rankings against a fresh stable sort.
- The Python extension smoke exercised the direct and batch DE bug fix, BiteOpt's hard limit, odd constrained MODE and both regular-grid and CVT archives. The schema-enabled job smoke completed 14 example jobs and exercised the registered MODE and MAP-Elites profiles, including the CVT CMA-MAE path.
- The truss-sizing MODE tutorial returned the same eight Pareto points after 2,048 calls. Field-service-routing QD returned the same six occupied niches of 60 and matching repertoire files after 1,024 calls. Each relevant CSV matched byte for byte between baseline and patch and across one and eight worker threads. I used these local tutorial drivers for the application checks. The plan's network-coverage and optical-lens candidates at this revision expose no QD driver.
- The Git diff passed `git apply --check` and applied in a fresh pinned checkout. The resulting Git diff matched byte for byte. Patch SHA-256 is `92bece9c5ead84d43b4aab9d734fde81f68d65409d26d5b0a4ffd22eba03ab3b`.

## Measurement scope

I used rustc 1.100.0 with thin LTO on an AMD EPYC 9V74 VM. Screening used seven interleaved pairs over five seeds with a nominal 35–37 ms sample target for typical cases. The integrated timing run used a 100 ms target for the slower measured arm. The faster arm can have a shorter timed block at the same repetition count. Search hold-outs measured 30 fresh seeds and used independently checked final values or a callback prefix, according to the metric. The ZIP includes **2,698 paired rows**, the workloads and the replay code.

I screened each change on a targeted workload or kernel and used additional shapes, ablations and entry points for the leading changes. The exhaustive cross-product of the plan, hardware counters, allocator instrumentation and dedicated 16-core measurements were outside this run. For timing differences near parity, use the raw paired spread and the listed workload. The application checks validated outcomes and thread-count consistency without using their one-off timings as speed claims.
