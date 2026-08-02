# Frozen foundations benchmark specification

This file is the comparison contract. All objectives are minimized, all
decision bounds are closed, and an optimizer receives the same requested
evaluation count as its controls. Checked-in conformance artifacts contain
full-precision values; Markdown rounds only for display. The `publication`
preset is an artifact-size label, not a claim of statistical benchmarking.

## Single-objective suites

The classic functions use their conventional unshifted, unrotated forms.
Dimensions 2, 10, and 40 are conformance cases; publication uses dimension 10.
Each function's bounds and known solution are defined in `src/suites/classic.rs`.
The baseline is the first 31 points of the seeded uniform-random control, not
the box center (which is the exact optimum of several unshifted functions).

## Lennard-Jones scaling family

The reduced-unit potential is `4 Σ(r^-12-r^-6)` over unordered atom pairs.
Study sizes are N=13, 38, 55, 75, and 98. The free encoding has `3N`
coordinates. The fixed-frame encoding places atom 0 at the origin, atom 1 on
positive `x`, and atom 2 in the `xy` half-plane with positive `y`, leaving
`3N-6` coordinates. Degenerate anchors are errors.

Every Cartesian coordinate is bounded by a cube of half-width `1.5 N^(1/3)`.
Initial candidates are rejection samples in a sphere of radius `0.8 N^(1/3)`
with pair separation at least `r_min = 0.75`. Every arm, including random,
uses this compact generator. Optimizer candidates crossing the box are
projected and counted.

For squared pair distance `s < s_min = 0.75²`, the physical pair term is
continued as

```text
q(s) = V(s_min) + V'(s_min)(s-s_min)
       + 0.5 |V'(s_min)|/s_min (s-s_min)².
```

This is finite, C1 at the boundary, and monotonically worse toward `s=0`.
Guarded pairs are counted. Exact coincidence has zero directional gradient,
which is unavoidable for a differentiable finite radial guard; the separated
initializer prevents it.

The primary budget is pair traversals. Energy-only evaluation and one combined
value/gradient evaluation each cost one traversal. Objective calls, gradient
calls, pair time, wall time, overlap pairs, and projections are separate
fields. The four derivative-free arms use four single-threaded basic retries;
the outer case layer owns parallelism. `argmin` L-BFGS multistart and fixed
basin hopping are mandatory reference arms. Their optional dependency is
enabled by `gradient-reference` and never enters `fcmaes-core`.

The population policies are algorithm-specific and explicit rather than
matched: DE uses its core default `15d`, CMA-ES uses
`floor(4 + 3 ln(d))`, and CR-FM-NES uses 16. Each retry receives one quarter of
the arm budget. A disjoint three-seed sensitivity grid over normalized
CR-FM-NES sigma `{0.05, 0.15, 0.50}` and population `{16, 32, 64}` selected
`sigma=0.05`, population 16 before the ten-seed primary rerun. The sensitivity
table is diagnostic and cannot support a general optimizer ranking.

Source-cited *putative-minimum* target energies are -44.326801, -173.928427,
-279.248470, -397.492331, and -543.665361 in ascending size order. Supplying
`--reference-directory` audits separately obtained Cambridge point files named
`13`, `38`, `55`, `75`, and `98`; the resulting hashes and evaluator residuals
are recorded without redistributing coordinates. Exact success is
`best_energy <= target + 1e-3`. Target-relative gaps and attainment within 1%,
5%, and 10% are secondary diagnostics. Publication uses ten seeds and 20,000
full pair-loop traversals per arm; smoke uses two seeds and 600 and cannot
support rankings. `pair_terms_evaluated` additionally reports the traversal
count multiplied by `N(N-1)/2`, so scaling across atom counts is explicit.

The QD pilot uses N=38 fixed-frame candidates, normalized radius of gyration
in `[0.25,0.75]`, mean coordination in `[0,12]` at cutoff 1.35, and a 12×12
grid. Its holdout adds deterministic independent `N(0,0.01)` coordinate
perturbations. Gates are clipping ≤5%, absolute Spearman correlation ≤0.90,
mean coverage ≥8%, fine holdout retention ≥60%, 6×6 retention ≥75%, and
cutoff ±0.01 retention ≥60%. A failed gate writes a QD skip manifest.

## Multi-objective suites

ZDT uses 30 variables except ZDT4, which uses 10. DTLZ uses three objectives
with conventional `k`: 5 for DTLZ1, 10 for DTLZ2–6, and 20 for DTLZ7. The
publication campaign reports ZDT1–4 and ZDT6 plus DTLZ1–7. It never mixes
training points into the analytic reference set.

The MODE arm uses population 64 and `nsga_update=true`: MODE's default
NSGA-II-style population update. `nsga_update=false` selects MODE's supported
DE update, but that alternative is outside this conformance campaign and no
result in the Foundations tables is a DE-versus-NSGA-II comparison. Lessons
L4–L6 likewise retain `nsga_update=true` through `ModeParams::default()`.
The analytic MODE populations are evaluated sequentially
(`evaluation_workers=1`): at this cost scale, dispatching 64 individual suite
calls to worker threads would measure scheduler overhead rather than useful
objective parallelism. The campaign's `--workers` argument is exercised only
by lesson L3's schedule-independence check. Applications with costly
independent objectives can pass each ordered ask batch through
`fcmaes_core::parallel_batch` before telling MODE.

Objective-space normalization is fixed per problem and uses the extrema of its
deterministic analytic reference set. DTLZ1 therefore uses `[0, 0, 0]` and
`[0.5, 0.5, 0.5]`; DTLZ7 uses the extrema of its deterministic disconnected
oversample. Every arm records the exact extrema it used in
`mo/indicators.csv`.

## Indicators

- Primary hypervolume uses one shared reference per problem, computed from the
  union of every published normalized arm/checkpoint front. Each coordinate is
  the union maximum plus 10% of `max(observed range, 1)`. The rule is frozen;
  the resulting coordinates are recorded in every row. This metric compares
  arms within a run and is not comparable across problems or campaigns.
- Fixed-box hypervolume at `[1.1; m]` after analytic-front normalization is a
  secondary cross-run field. It is computed on the complete front only when
  every point is inside. Otherwise its value is null, its kind is
  `not-applicable-outside-reference`, and the outside count is recorded.
- Exact hypervolume is used through four objectives. Higher-dimensional
  estimates are labeled Monte Carlo and record seed, samples, and standard
  error.
- IGD, IGD+, GD, and GD+ are arithmetic means of nearest Euclidean or modified
  Euclidean distances, rather than the alternative root-mean-square variant.
- Additive epsilon is `max_r min_a max_j(a_j-r_j)` for minimized objectives.
- Exact duplicates collapse before hypervolume. Dominated unique points are
  removed. Both counts are reported.
- Non-finite points and mismatched dimensions are errors. Points outside the
  primary reference are errors. No front is clipped or filtered for either
  hypervolume convention.

ZDT3 reference points come only from its five nondominated intervals. DTLZ5
and DTLZ6 use their one-dimensional degenerate manifold for `m > 2`. DTLZ7's
disconnected reference set is a deterministic Pareto filter of a Halton
oversample. Reference generation is seed-free.

## Fairness and replay

Every table contains `initial`, `random`, and optimizer rows. The `random`
control receives exactly the optimizer's requested budget; the `initial` row
consumes only its declared population and is never mislabeled as an optimizer
result. Requested-budget equality does not imply identical actual counts. In
the checked publication artifacts, scalar DE completes its current population
batch and records 4,006–4,029 actual evaluations against random's exact 4,000,
while MODE and random both record exactly 4,096. Every retained decision is
evaluated again before
indicators are written; the deterministic recheck count and maximum absolute
discrepancy are recorded. This same-evaluator check can detect nondeterminism
or bookkeeping corruption, not independent model error. Requested and actual
optimizer evaluations, seed, workers, dimensions, normalization, both
reference conventions, deterministic rechecks, and wall time are part of the
machine-readable artifacts. Rechecks do not consume optimizer budget.
