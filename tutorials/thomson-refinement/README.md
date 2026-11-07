# X4: Points on a sphere: global and local search

This chapter runs native Rust solvers through their Python APIs. Its model and independent checks live in this directory; the shared driver in optimization-jobs adds figures and command-job execution.

## Model and decision

Thomson energy sums 1/distance over every pair of unit-sphere charges. SQP uses Cartesian equality constraints; BiteOpt normalizes triples before evaluating the same energy. Compare three local starts, global search and a declared global/local split.

## Work through the lesson

Minimize the Coulomb energy of $N$ unit charges $x_1,\dots,x_N\in\mathbb{R}^3$:

$$E(x)=\sum_{1\le i<j\le N}\frac{1}{\|x_i-x_j\|}\quad\text{subject to}\quad\|x_i\|^2=1,\qquad i=1,\dots,N.$$

SQP uses Cartesian coordinates and sphere equalities. The black-box arm normalizes triples onto the sphere before evaluating the same energy.

1. Compare three SQP starts, a global BiteOpt search, and a declared global-then-local split. Smoke uses 12 charges and the icosahedral reference 49.1652530576; study uses 60 charges and putative reference 1543.830401.
2. Read each arm's actual search pair interactions. A shared maximum number of callbacks does not imply equal consumed work; early convergence and partial evaluations change counts. Derivative and Hessian audits are outside these arm counts.
3. Compare only independently checked sphere points. If a local attempt is infeasible, retain its raw native result and audit, then explicitly project it onto the sphere and recompute its feasible energy without transferring a first-order claim. Projection/check work is outside the reported search counters. A small raw objective at an infeasible point is not an improved Thomson solution.
4. Compute a finite-difference Lagrangian Hessian projected onto the sphere tangent space only when an accepted point passes the independent first-order check. Otherwise `hessian_status` records why curvature classification was not run. `curvature` records the verdict with a relative tolerance: three near-zero rotational modes and positive remaining curvature give `local_minimum`, a negative eigenvalue `not_a_local_minimum`, anything else `inconclusive`. A converged attempt that fails the independent first-order check is kept and flagged (`converged_without_independent_first_order`); it does not stop the other arms.
5. Matching a published putative reference is not a proof of global optimality. Different formulations and global/local splits remain descriptive comparisons.

## Run

Install the pinned solver packages and scientific dependencies as described in the [shared setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/structured/README.md), with the repositories it lists cloned side by side. From this repository's root:

```bash
python ../optimization-jobs/tutorials/structured/run.py --chapter X4 --output /absolute/new/x4
```

To run the chapter on its own, use `python tutorials/thomson-refinement/run.py --preset smoke --output /absolute/new/x4-direct`. It produces JSON; the shared driver additionally renders PNG/SVG figures. `--preset study` runs the larger declared cases; a chapter without them repeats the smoke case. Output may be new or empty, and an occupied directory is refused.

## Independent answer and validation

Smoke uses 12 charges, with the icosahedral reference 49.1652530576; study uses 60. Every pair interaction of the searches is counted in `search_pair_traversals`; the projection, derivative and Hessian audits are not counted there. A finite-difference Lagrangian Hessian is projected into the tangent space.

Every run keeps the native status, points and available diagnostics. Application-specific references are computed separately. All failures remain in the report.

## Read the result

Read `summary.json` before interpreting `figure.png`. The report records the actual result; expected reference values are not substituted for missing numerical outcomes. A completed command job means that the chapter's executable checks completed, while individual solver attempts can still be nonconvergent where the lesson explicitly retains them.

The arms share requested callback caps, not identical consumed work or wall time. Reaching a published putative minimum is not a global certificate. Invalid collision candidates are retained in the failure count.

## Agent prompt

> Compare SQP multistart, global search and hybrid refinement for the sphere-charge fixture. Report actual pair work and native failures. Rank only checked sphere energies, withhold first-order claims from projected fallbacks, and classify curvature only at a verified stationary point.

Use the complete [prompt](prompt.md) for the workflow and evidence requirements.

## Follow-up exercise

Compare callback caps with actual consumed pair work. Examine the rotational zero modes only at an accepted first-order point and avoid assuming every stationary point is a minimum.

## Provenance and limits

The checked-in `results/` record one run of each preset. The shared driver writes source and package identities to `environment.json` for the runs it makes. These are reproducible teaching experiments, not industrial or hardware performance certifications. Tables that compare formulations or solvers are descriptive.

On another processor, NumPy selects other vectorized kernels. The smoke preset still agrees to about 1e-6. In the study preset, the SQP starts follow those last-digit differences: their callback counts change slightly, and the first start can stop at the evaluation limit instead of with a line-search failure. The checked energies still agree to about 1e-6.

## Recorded example

Each figure plots, on a log scale, how far each arm's independently checked energy lies from the reference. Gaps inside the grey band (at most 1e−6, the cross-processor agreement above) match the reference. A filled marker means that the point passed the independent first-order check; an open marker carries no first-order claim. The tick labels give each arm's actual search pair traversals, and the arm whose point was used for the curvature check carries its verdict.

![Recorded X4 smoke result](results/smoke/figure.png)

![Recorded X4 study result](results/study/figure.png)

| Preset (charges) | Arm | Checked energy | Gap | Pair traversals | Checks |
|---|---|---:|---:|---:|---|
| smoke (12) | SQP multistart | 49.165253058 | −1.5e−11 | 20,790 | first-order passed |
| | global | 50.119420444 | +0.95 | 30,104 | no first-order claim |
| | hybrid | 49.165253058 | −3.0e−11 | 21,417 | first-order passed; local minimum |
| study (60) | SQP multistart | 1543.830400976 | +3.4e−10 | 5,552,490 | line-search failure; first-order failed |
| | global | 1560.946963233 | +17.1 | 6,144,085 | no first-order claim |
| | hybrid | 1543.835099598 | +4.7e−3 | 4,531,424 | first-order passed; local minimum |

The small negative smoke gaps are not improvements on the icosahedron. Accepted points satisfy $\|x_i\|^2=1$ only to about 1e−12, which moves the energy by roughly $E\cdot10^{-12}$.

In the study preset, only one SQP start reaches the putative reference, and its point fails the independent first-order check. The other two starts and the hybrid converge to a different local minimum, 4.7e−3 above the reference. Curvature is therefore classified at the hybrid's point, which is not the reference configuration. This is the case step 4 describes: a good energy without a verified stationary point does not get a curvature verdict.

Inspect the [smoke summary](results/smoke/summary.json) and [study summary](results/study/summary.json) alongside the figures.

The requested callback cap is 480 in smoke and 3,600 in study. Each of three local starts receives one third; the hybrid splits half to global search and half to local refinement. Every arm retains actual work and termination.
