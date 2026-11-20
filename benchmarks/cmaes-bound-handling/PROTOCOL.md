# CMA-ES bound handling: three modes

Declared on 19 November 2026, before the third mode was implemented and
before any run of this test.

## 1. Question

Active CMA-ES in fcmaes-core moves a sample that leaves the box onto the
bounds. Two ways to update from such a sample exist:

- **clip** (`repair_z: false`, the default): the update uses the `z` of the
  unmoved sample.
- **repair** (`repair_z: true`): the update uses the `z` of the evaluated
  point.

Two studies of 2,400 matched pairs each found clip better than repair in
most workflows, and repair better where samples keep being clipped and the
step size inflates. The clearest such case is the 118-variable neural
controller of `tutorials/neural-controller-policy-search`: a median loss of
1.13 with clip against 0.137 with repair at 110,000 evaluations.

Other CMA-ES implementations use a third method:

- **transform**: CMA-ES samples and updates in unbounded internal
  coordinates with untouched `z`; only the evaluated point is mapped into
  the box, by a smooth, periodic transformation made of linear and
  quadratic pieces (the `BoundTransform` of pycma).

This test asks whether transform should replace repair as the mode to
choose when clip fails. **The default stays clip whatever the outcome.**

## 2. Implementation under test

Transform is implemented in `fcmaes-core` behind an experimental setting on
the branch of this test. The transformation acts in the normalized
coordinates of `Fitness`, where the box is `[-1, 1]` in every coordinate.
Its values are checked against pycma on random points. With the setting
off, results must be bit-identical to `main`.

## 3. Comparisons

Three modes, the same seeds for all three, single-threaded runs, compared by
evaluations. Lower is better.

| Id | Problem and workflow | Budget | Repetitions |
|---|---|---|---:|
| A | Neural controller, plain CMA-ES as the job worker runs it (population 31, sigma 0.3, uniform random start) | 110,000 evaluations | 10, seeds 1 to 10 |
| B | `truss_sizing` of the candidate benchmark, jSO→CMA as one run, protocol T | 1,000,000 evaluations, root seed 7 | 10 |
| C | Messenger (reduced), coordinated DE→CMA retry (`gtop-advexamples`: 300 retries, 1,500 initial evaluations, factor 50, check interval 100, one worker) | as the schedule gives | 12, seeds 1 to 12 |
| D | Cassini2, the same coordinated retry | as the schedule gives | 12, seeds 1 to 12 |

In B the transform applies to the CMA-ES stage of the sequence; in C and D
to the CMA-ES stage of every retry.

Informational, without a role in the decision:

- A at 900,000 evaluations, 5 seeds.
- Two boundary-start probes with plain CMA-ES, one variable on `[0, 1]`
  started at 0 with sigma 0.15, population 31, seeds 1 to 5: `(x - 0.3)^2`,
  and the first-duration restriction of the damped spring in
  `examples/src/damp.rs`.
- The evaluations each mode spends in C and D.

## 4. Decision

Tests are two-sided Mann–Whitney U tests at 0.05, uncorrected.

Transform **keeps the controller gain** if in A its median is at most 0.2
and it is significantly better than clip.

Transform **replaces repair** as the non-default mode, in fcmaes-rust and in
the C++ and Python implementations of fast-cma-es, if

1. it keeps the controller gain, and
2. in B, C and D it is nowhere significantly worse than repair, and its
   median is better than repair's in at least two of the three.

Otherwise transform is not merged and repair stays.

The report also states, for every comparison, how transform stands against
clip. That does not change the default.

`decide.py` computes the tables and the decision from the raw files in
`results/` and writes `results/report.md`.

## 5. Limits known in advance

- Four problems, 10 to 12 repetitions, one host.
- A was chosen because clip fails on it. It shows whether transform helps
  there, not how often such problems occur.
- The transformation is known to slow convergence when the optimum of a
  coordinate lies exactly on a bound. B, C and D are problems of that kind.
- pycma's other method, the adaptive penalty, is not tested.
