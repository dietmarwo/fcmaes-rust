# Review of the two kernel-performance deliveries

27 November 2026. Two patches against `d7ff66a` were reviewed and merged in
#83: `fcmaes-rust-four-optimizers.patch` (BiteOpt, DE, MODE, MAP-Elites; 25
of 80 screened changes) and `0001-cmaes-performance.patch` (active CMA-ES; 6
of 30). The delivered logs are
[RESULTS-four-optimizers.md](RESULTS-four-optimizers.md) and
[RESULTS-cmaes.md](RESULTS-cmaes.md), unchanged. This page records what the
review reran, what it found and what it changed.

**The speedups reproduce, and the seeded runs that were promised unchanged
are unchanged.** Three changes to seeded results were not stated, or not
stated fully, in the deliveries. The review documents them and fixes one
sign-of-zero regression.

## Combining the patches

Each patch applies unchanged to `d7ff66a`. Together they touch only in two
changelog hunks; both entries were kept. The tree of the CMA-ES patch alone
is `caa0a9e`, as its manifest records. The combined tree passes 438 tests
(435 and 431 were reported for the patches alone) and 439 after the review.

## Replay against 0.4.0

`replay_digest`, an example program added by the review, runs 2,770 seeded
cases and prints a digest of every evaluated point and of the result. It was
built unchanged against `d7ff66a` and against the merged tree.

| Optimizer | Runs | Same evaluated points | Same result |
|---|---:|---:|---:|
| BiteOpt (direct, deep, batches of 1, 7 and 64) | 480 | 480 | 480 |
| MODE (1 to 5 objectives, constraints, odd populations, ties, NaN) | 600 | 600 | 600 |
| MAP-Elites (grids, CVT in 2 to 8 descriptor dimensions, CMA emitter) | 106 | 106 | 106 |
| DE, finite objectives | 432 | 432 | 405 |
| DE, objective with NaN and infinity: `ask`/`tell` | 36 | 36 | 33 |
| DE, objective with NaN and infinity: `optimize`, `optimize_delayed` | 108 | 14 | 15 |
| CMA-ES below dimension 64 | 672 | 672 | 672 |
| CMA-ES at 64 and 100, explicit gap: `optimize`, `ask`/`tell` | 96 | 96 | 96 |
| CMA-ES at 64 and 100, explicit gap: `tell_x` | 48 | 0 | 0 |
| CMA-ES at 64 and 100, automatic gap | 192 | 0 | 0 |

The DE result differences on finite objectives are the fix: the reported
value improved in 24 runs and never got worse, and in 3 runs on a plateau
another point of the same value is reported. All 432 new reports equal the
lowest value the callback saw; 408 of the old ones did.

## Speed

The four-optimizer plan contains a probe program in full. The review built
it against both trees and ran five seeds in three interleaved repetitions on
one pinned core. The host has two cores, so the eight-thread rows of the
delivery were not rerun.

| Workload (probe) | Delivered | Here |
|---|---:|---:|
| BiteOpt, shifted sphere, d=96 | 1.75 | 1.47 |
| BiteOpt, d=32, direct / batch | 1.23 (Rastrigin) / 1.60 | 1.32 / 1.50 |
| BiteOpt, Cassini1 | 1.03 | 1.00 |
| DE, d=32, direct | 1.12 | 0.99 |
| DE, population ask/tell | 1.14 (d=96) | 1.12 (d=32) |
| MODE, ZDT1, P=32, SBX | 1.74 | 1.69 |
| MODE, ZDT1, P=256, SBX / DE | 4.54 / 5.22 | 4.41 / 5.91 |
| MAP-Elites, 4,096-cell grid | 4.13 | 4.81 |
| MAP-Elites, 1,024-center CVT | 3.56 (run with cold setup) | 4.05 (run), 3.47 (setup) |
| CMA-ES, sphere 100, serial throughput | 1.74 | 1.78 and 1.81 |
| CMA-ES, ellipsoid 100, serial throughput | 1.81 | 1.75 and 1.75 |
| CMA-ES throughput relative to `cmaes` 0.2.2, sphere 100 / ellipsoid 100 | 0.51 / 0.54 | 0.53 / 0.54 (0.4.0: 0.30 / 0.30) |

The workloads differ in budget and in some dimensions, so the columns agree
in size, not digit for digit. The CMA-ES rows use the repository's own
comparison harness, two rounds per build.

The delivered tables were also checked against the archived rows. Seven
rows of the four-optimizer table were recomputed from `raw/production.csv`
as the median of per-seed medians and match to the printed digit. The
CMA-ES headline ratios (1.740, 1.809, 0.513, 0.543) follow from
`final-wall.jsonl`. The archive holds the stated 2,698 paired rows and 80
decisions, 25 of them retained.

## Does the basis reuse cost evaluations?

At dimension 64 or higher with the automatic gap, CMA-ES now decomposes the
covariance on every second eligible generation. The delivery shows equal or
mixed fixed-budget values. The review ran both builds to a target of 1e-8
with `cma_target`, an example program it added, and counted evaluations.

| Problem | Dimension | Seeds | Hits, 0.4.0 / merged | Median evaluations, 0.4.0 → merged | Time |
|---|---:|---:|---:|---:|---:|
| Sphere | 64 | 12 | 12 / 12 | 7,008 → 7,024 | 1.70× |
| Sphere | 100 | 12 | 12 / 12 | 10,514 → 10,404 | 1.81× |
| Rosenbrock | 64 | 24 | 19 / 23 | 143,104 → 142,656 | 1.72× |
| Rosenbrock | 100 | 24 | 22 / 20 | 341,479 → 339,634 | 1.81× |
| Rosenbrock | 128 | 12 | 10 / 10 | 554,499 → 553,545 | 1.82× |
| Rotated ellipsoid, condition 1e6 | 64 | 12 | 12 / 12 | 103,984 → 103,704 | 1.70× |
| Rotated ellipsoid, condition 1e6 | 100 | 8 | 8 / 8 | 238,484 → 237,847 | 1.67× |
| Rotated ellipsoid, population 31 | 100 | 6 | 6 / 6 | 230,144 → 231,586 | 1.62× |
| Rotated cigar | 100 | 8 | 8 / 8 | 35,717 → 35,870 | 1.77× |
| Rotated ellipsoid | 200 | 3 | 3 / 3 | 1,220,408 → 1,220,408 | 0.99× |

Evaluations to target are unchanged: the paired median ratio is between
0.99 and 1.01 in every row. The Rosenbrock misses are runs that stop in its
second minimum near 3.99; over the 60 Rosenbrock runs 0.4.0 hit 51 times and
the merged tree 53 times.

The last row shows the range of the change. A reuse needs two consecutive
eligible generations. With the default population that happens up to about
dimension 160 (runs at 110 to 160 differ from 0.4.0; runs at 180 and 200 are
identical in every evaluation and take the same time).

## Findings

1. **DE changes its handling of non-finite values.** Up to 0.4.0
   `optimize` and `optimize_delayed` stored a NaN or infinite value as 1e99.
   A slot without a result accepted it, and the serial driver then spent a
   temporal-locality evaluation on it. The patch evaluates through a new
   path without that replacement, so such a value is now a failed trial, as
   it always was in `tell`. This is the cleaner rule, and it is neutral in
   quality: in 94 changed runs the best value was better 44 times and worse
   49 times. The delivery did not
   state it: its trace comparisons used finite objectives. The changelog,
   `docs/optimizers.md` and `ai-context.md` now say so.
2. **The QD-score of an archive without elites became -0.0.** The new
   allocation-free sum has no term for such an archive, and a sum without
   terms is negative zero. 18 of 106 replay cases showed it. The score is
   0.0 again, with a test.
3. **`tell_x` changes at dimension 64 or higher with any gap.** The factor
   transform replaces the matrix inverse there. The delivered documentation
   tied changed seeded runs to the automatic gap only.
4. **The DE direct speedup did not repeat.** The delivery reports 1.12 on
   the 32-dimensional sphere; the probe measured 0.99 to 1.00 here. Ask/tell
   gained 1.12. The delayed driver's eight-thread gain (2.82 with an interval
   from 0.96 to 7.07) could not be checked on two cores.
5. **The gain of the CMA-ES change has an upper dimension.** See the table
   above. The delivery tested 100 dimensions and did not state where the
   effect ends.

## Changes made by the review

- The QD-score fix and its test.
- Documentation of findings 1, 3 and 5; rewritten changelog entries with the
  measured effects.
- `replay_digest` and `cma_target` as example programs of `fcmaes-core`.
  They compile in CI and compare two checkouts
  ([usage](README.md#reproduce-the-review)).
- This directory, the two plans under `docs/plans/` and their entries in
  the benchmark index and the book.

## Not verified

- The rows with eight threads, and the 16-core confirmation both plans ask
  for.
- The search hold-outs of the four-optimizer delivery (30 seeds). Their
  rejected and held proposals are not in the patch.
- The application tutorials (truss sizing, field-service routing) and the
  DE→CMA and jSO→CMA handoff replays of the CMA-ES delivery.
- Digests on another platform. The replay compares two builds on one host.

## Open points

1. **Population cap for short high-dimensional BiteOpt runs** (B18, held).
   The 96-dimensional gain is large; a rule needs a dimension and budget
   sweep.
2. **Faster eigensolver** (CMA-ES trial 01, kept as evidence). It won at
   dimension 256 and lost at 100 inside a full generation. With the reuse
   ending near dimension 160, larger problems would need it.
3. **Delayed DE on many cores.** The gain depends on callback cost and has
   a wide interval; a 16-core rerun with a costly callback would settle
   whether it matters for applications.
