# Publication evidence

Recorded on 2026-07-24 with the release build. This document supports an
educational simulation-optimization tutorial. It does not validate the model
for ventilation engineering.

## Experimental design

| Setting | Value |
|---|---:|
| Decision variables | 9 continuous |
| Training releases | 3 |
| Held-out releases | 3 |
| Grid | 40 × 24 |
| Room | 5 m × 3 m |
| Maximum flow steps | 800 |
| Pollutant steps | 600 |
| Flow residual limit | `5e-4` |
| Interior flux mismatch limit | 5% |
| Optimizer seeds | 42, 43, 44 |
| Objective workers | 16 |
| Requested evaluations per run | 20,000 |
| Actual evaluations per run | 20,096 |

Every objective evaluation solves the flow once and reuses it for all three
training releases. Each pollutant objective is the worst of those releases.
The three held-out releases are evaluated only after the optimizer selects a
representative.

MODE and MAP-Elites both use 157 complete batches of 128, giving equal search
budgets. MODE receives four objectives and four constraints. MAP-Elites
minimizes the reporting scalar inside each flow/low-velocity niche and rejects
infeasible candidates.

## Three-seed results

| Method | Seed | Training quality | Held-out quality | Search time | Pareto points / niches | Coverage | QD-score |
|---|---:|---:|---:|---:|---:|---:|---:|
| MODE | 42 | 1.120668 | 1.491460 | 35.205 s | 128 | — | — |
| MODE | 43 | 1.127893 | 1.488041 | 35.427 s | 128 | — | — |
| MODE | 44 | 1.118472 | 1.497132 | 35.607 s | 126 | — | — |
| MAP-Elites | 42 | 1.143124 | 1.490528 | 33.383 s | 309 | 77.25% | 222.561 |
| MAP-Elites | 43 | 1.205762 | 1.628958 | 33.338 s | 297 | 74.25% | 215.040 |
| MAP-Elites | 44 | 1.204846 | 1.479038 | 32.778 s | 306 | 76.50% | 220.510 |

Summary values use sample standard deviation:

| Method | Training quality | Held-out quality | Search time | Result size |
|---|---:|---:|---:|---:|
| MODE | 1.122344 ± 0.004929 | 1.492211 ± 0.004592 | 35.413 ± 0.202 s | 127.3 ± 1.2 |
| MAP-Elites | 1.184577 ± 0.035902 | 1.532841 ± 0.083438 | 33.166 ± 0.337 s | 304.0 ± 6.2 |

The fixed baseline has training quality 1.598712 and held-out quality
1.791533. Relative to that baseline, mean held-out reporting quality improves
by approximately 16.7% for MODE and 14.4% for the best MAP-Elites elites.
These are descriptive results from three seeds, not confidence intervals.

MODE is more stable under this reporting scalar. MAP-Elites has a different
purpose: its archive retains hundreds of distinct flow behaviors rather than
only driving one scalar or preserving an objective Pareto set.

Search time covers the parallel optimization loop. Final population/archive
reproduction, three held-out evaluations, CSV output, and plotting are outside
that timer. Throughput is machine-specific and is not presented as a
cross-machine benchmark.

The machine-readable source is
[`replication-summary.csv`](replication-summary.csv).

## Seed-42 representatives

### MODE

```text
training quality       = 1.120668469
held-out quality       = 1.491459638
training exposure      = 0.566551495
training max receptor  = 0.505681717
fan-power proxy        = 0.921422963
training final mass    = 0.233983046
held-out final mass    = 0.532366349
flow rate              = 1.932319 m²/s
low-velocity fraction  = 0.057604
flux mismatch          = 0.010783
flow residual          = 2.951909e-4

design = [
  0.2624119813644679,
  0.4348990083590912,
  0.2633892871687757,
  0.3914347733372533,
  1.4810477598402265,
  0.6099844574061106,
  0.6394449558711917,
  0.6449705049924965,
 -0.8515328431697532
]
```

### MAP-Elites

```text
training quality       = 1.143124392
held-out quality       = 1.490527843
training exposure      = 0.570405041
training max receptor  = 0.507570305
fan-power proxy        = 0.990489130
training final mass    = 0.241672745
held-out final mass    = 0.498903302
flow rate              = 2.025000 m²/s
low-velocity fraction  = 0.027586
flux mismatch          = 0.019638
flow residual          = 4.144476e-4

design = [
  0.2967337660939455,
  0.4500000000000000,
  0.3664460873093897,
  0.3962056821558659,
  1.5000000000000000,
  0.6109417595177361,
  0.6745091930459820,
  0.6128473721673339,
 -1.0168194384482883
]
```

Both representatives generalize less well to the held-out release set than to
the training set. Reporting that gap prevents the favorable training score
from being mistaken for general ventilation performance.

## Straight-channel reference

The independent 48 × 20 full-height straight channel uses 1,200 allowed flow
steps and converges after 980:

| Property | Result |
|---|---:|
| Axial-profile relative symmetry error | `5.832812e-15` |
| Maximum transverse lattice velocity | `3.253404e-5` |
| Maximum/mean axial velocity | 1.492363 |
| Relative flux mismatch | 0.004140 |
| Velocity residual | `9.897290e-7` |

This verifies symmetry, low transverse flow, profile development, flux
conservation, and residual convergence for a simple case. It is a numerical
property test, not validation against measurements. Raw output is in
[`verification/channel-reference.csv`](verification/channel-reference.csv).

## Three-grid sensitivity

The scalar horizon scales linearly with grid width and the flow limit scales
approximately with width squared:

| Grid | Scalar steps | Flow steps |
|---|---:|---:|
| 30 × 18 | 450 | 450 |
| 40 × 24 | 600 | 800 |
| 60 × 36 | 900 | 1,800 |

Reporting qualities:

| Design | Grid | Training | Held out | Feasible |
|---|---:|---:|---:|---|
| Baseline | 30 × 18 | 1.594236 | 1.723675 | yes |
| Baseline | 40 × 24 | 1.598712 | 1.791533 | yes |
| Baseline | 60 × 36 | 1.632744 | 1.831336 | yes |
| MODE seed 42 | 30 × 18 | 1.517356 | 1.769364 | **no** |
| MODE seed 42 | 40 × 24 | 1.120668 | 1.491460 | yes |
| MODE seed 42 | 60 × 36 | 1.208667 | 1.553329 | yes |
| MAP-Elites seed 42 | 30 × 18 | 1.191322 | 1.451241 | yes |
| MAP-Elites seed 42 | 40 × 24 | 1.143124 | 1.490528 | yes |
| MAP-Elites seed 42 | 60 × 36 | 1.173980 | 1.510022 | yes |

The coarse-grid MODE representative has 5.326% interior flux mismatch against
the 5% limit. Its scalar therefore contains a constraint penalty. The 40 × 24
and 60 × 36 evaluations are feasible, but their remaining quality differences
show that the study establishes sensitivity rather than formal grid
convergence. Raw data is in
[`verification/resolution-study.csv`](verification/resolution-study.csv).

## Figures

![MODE Pareto projection and convergence](../images/mode-results.svg)

![MAP-Elites archive and convergence](../images/qd-results.svg)

![Baseline and optimized fields](../images/flow-fields.svg)

![Resolution and held-out evidence](../images/verification-results.svg)

All optimization figures use seed 42, while the summary table and right-hand
verification panel retain all three seeds.

## Artifact map

For every `mode-seed-*` directory:

- `pareto.csv`: feasible final non-dominated points and decisions;
- `convergence.csv`: search trace;
- `validation.csv`: selected design on held-out releases;
- `selected-field.csv`: final field for the worst-exposure training release.

For every `qd-seed-*` directory:

- `archive.csv`: occupied niches, descriptors, quality, visits, and decisions;
- `convergence.csv`: coverage, QD-score, and best-quality trace;
- `validation.csv`: best scalar elite on held-out releases;
- `selected-field.csv`: its worst-exposure training field.

`verification/` contains the reference, resolution study, and consistently
reproduced baseline/MODE/MAP-Elites fields used by the field figure.

## Reproduction cautions

- Optimizer seeds control candidate generation; objective simulations are
  deterministic.
- A Rust objective evaluation includes three pollutant solves, so evaluation
  counts are directly comparable between MODE and MAP-Elites.
- Held-out and verification evaluations are additional and excluded from the
  optimizer budget.
- QD-score depends on archive capacity and quality scaling and should only be
  compared under the recorded configuration.
- Three seeds reveal variability but are too few for strong statistical
  claims.
