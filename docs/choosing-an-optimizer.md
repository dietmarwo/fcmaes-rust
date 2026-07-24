# Choosing an optimizer

`fcmaes-rust` is designed for bounded black-box optimization. It is a strong
choice for many difficult simulation and engineering problems, but it is not
the right default for every function that happens to be written in Rust.

The governing rule is simple:

> Match the optimizer to the mathematical structure of the complete problem,
> including constraints, failures, noise, and the result you need.

Gradient-free population methods make few assumptions and pay for that
generality with evaluations. A solver that can exploit valid linear, convex,
combinatorial, or derivative structure will usually be more efficient and may
provide guarantees that a black-box optimizer cannot.

This page first decides whether to use `fcmaes-rust`. If the answer is yes, use
the [optimizer guide](optimizers.md) to select an algorithm inside the library.

## Decision sequence

### 1. Can a structured solver represent the problem?

Use the strongest structure that remains valid for the complete formulation:

- For linear and mixed-integer linear models, use a modelling layer such as
  [`good_lp`](https://docs.rs/good_lp/latest/good_lp/) with a suitable backend,
  or use SCIP or HiGHS directly.
- For convex conic problems, consider
  [`clarabel`](https://docs.rs/clarabel/latest/clarabel/).
- For convex quadratic programs, including many control subproblems, consider
  [`osqp`](https://docs.rs/osqp/latest/osqp/).
- For shortest paths, flows, matching, assignment, and standard routing
  variants, start with the corresponding specialized algorithm or solver.

Such solvers can exploit model structure, establish bounds, and, depending on
the problem and backend, certify optimality or infeasibility. A black-box
optimizer only reports the best candidate it found.

One nonlinear or simulated term does not necessarily invalidate the structured
model. It may be better to isolate that term in an outer loop, use a
linearization, or solve a structured subproblem for every outer candidate.
See [Hybrid and decomposed workflows](#hybrid-and-decomposed-workflows).

Backend details matter. Some Rust modelling crates can select either native
Rust solvers or C/C++-backed solvers through features. Check the selected
backend's capabilities, build requirements, and license rather than inferring
them from the modelling crate.

### 2. Are reliable end-to-end gradients available?

If all decisions are continuous and derivatives are economical and meaningful,
start with a gradient-based method. [`argmin`](https://docs.rs/argmin/latest/argmin/)
provides methods including L-BFGS, trust-region, nonlinear conjugate-gradient,
and Gauss-Newton algorithms, together with observers and checkpointing.
Analytical derivatives, automatic differentiation, and simulator sensitivities
are all worth considering.

The word *end-to-end* is important. A derivative of the simulator state is not
automatically a useful derivative of an objective that also contains:

- integer or categorical decoding;
- contact, switching, clipping, or changing event sequences;
- solver failure or rejection of infeasible candidates;
- a maximum, quantile, rank, or failure count over scenarios; or
- noise large enough to obscure local directional information.

Events alone do not invalidate gradients. Diffsol supports forward and adjoint
sensitivities, so smooth parameter fitting is naturally paired with a
gradient-based optimizer. The
[Diffsol discussion](../tutorials/README.md#11-diffsol-why-gradients-are-the-better-default)
explains the boundary. A gradient-free outer layer becomes attractive when
discrete policies, resets, robust aggregation, or failures break the complete
derivative path.

### 3. Is it a canonical combinatorial problem?

Use a domain solver when the problem is fundamentally a standard TSP, vehicle
routing, assignment, scheduling, path, or flow problem. Specialized algorithms
usually outperform a generic real-vector encoding and may provide useful
bounds.

`fcmaes-rust` becomes relevant when that combinatorial core is coupled to
simulation, nonlinear costs, continuous settings, unusual constraints, or
several objectives that the domain solver cannot express. The
[multi-UAV task-assignment](examples.md#multi-uav-task-assignment) and
[flexible job-shop](examples.md#binaries) examples illustrate such mixed
formulations.

### 4. How many evaluations are actually affordable?

Use the total campaign budget, not only the cost of one evaluation. Include
retries, replications, validation, failed simulations, and every worker.

| Evaluation regime | Usually benchmark first |
| --- | --- |
| Very scarce, expensive evaluations | Bayesian or surrogate-based optimization |
| Moderate budget | A surrogate method and a gradient-free population method |
| High-throughput, parallel evaluations | `fcmaes-rust` retry, MODE, or MAP-Elites |

[`egobox`](https://docs.rs/egobox-ego/latest/egobox_ego/) is a relevant Rust
candidate in the scarce-evaluation regime. Its current EGO implementation
supports constraints, basic mixed-variable spaces, failed evaluations as
hidden constraints, restarts, and batched q-EI evaluation.

There is no universal numerical cutoff. Dimension, smoothness, noise, the
number of constraints, batch size, and surrogate choice can shift the
crossover by orders of magnitude. A few hundred evaluations may already be a
large budget for an expensive low-dimensional model and far too small for a
noisy 50-dimensional search. When both approaches are plausible, compare them
under the same total evaluations and compute resources.

The [GTOP optimizer comparison](../benchmarks/optimizer-comparison/comparison.md)
demonstrates the high-budget retry regime, not a universal ranking. Under the
common 240,000-evaluation budget, no tested method reached the Tandem target.
In a separate coordinated DE-to-CMA campaign, `fcmaes-rust` reached it in
85 of 100 experiments using about 230.7 million actual evaluations on average.
The alternative BIPOP-CMA-ES stress test reached 0 of 1,000 targets after about
9.47 billion actual evaluations. Those unequal-budget results are evidence for
the value of application-specific adaptive coordination on Tandem, not a
general proof that one optimizer dominates another.

### 5. What result must the optimizer return?

- For one selected design, use scalar optimization. Keep genuine feasibility
  conditions separate from preferences where the API permits it.
- For competing objectives, use MODE or weighted multi-objective retry to
  approximate a Pareto set.
- For a repertoire indexed by meaningful behavior descriptors, use
  CVT-MAP-Elites and optionally the Diversifier.

Do not collapse distinct engineering goals into a weighted sum merely because
the scalar API is convenient. Weights encode a decision. If that decision has
not been made, expose the trade-off first.

Other Rust frameworks may fit different priorities. For example,
[`optirustic`](https://docs.rs/optirustic/latest/optirustic/) provides
dedicated NSGA-II and NSGA-III implementations, while
[`radiate`](https://docs.rs/radiate/latest/radiate/) provides a broader genetic
algorithm framework with multi-objective selection, generic genotypes, and
novelty search. Compare the actual execution, persistence, and parallelism
requirements rather than algorithm names alone.

### 6. Is the candidate naturally a bounded real vector?

`fcmaes-rust` optimizes fixed-length `f64` vectors within finite bounds.
Continuous values are direct. Small integer and categorical decisions can
often be decoded by rounding or indexed lookup; random keys can represent
permutations and partitions. The
[combinatorial encoding cookbook](combinatorial-encodings.md) collects tested
integer, categorical, subset, permutation, partition, ordering, and repair
patterns.

If the candidate is inherently a variable-length program, graph, expression
tree, neural topology, or other structure, forcing it into a real vector can
destroy useful locality and make most candidates invalid. Prefer a
genotype-aware evolutionary framework or a domain-specific search.

Dimension also changes the viable algorithm and budget. Full-covariance
CMA-ES work grows roughly quadratically with dimension, and every population
method needs enough evaluations to learn across all coordinates. Within
`fcmaes-rust`, CR-FM-NES and PGPE are candidates when a full covariance is
unattractive. The
[neural-controller tutorial](../tutorials/neural-controller-policy-search/)
compares both on a 118-parameter fixed-topology policy with randomized
rollouts. For extremely large differentiable models, gradients and
problem-specific parameterizations are usually the more important advantage.

### 7. Is hard real-time or `no_std` execution required?

`fcmaes-rust` does not currently promise hard real-time bounds or `no_std`
support. A population search is also usually the wrong shape for an online
control loop with a millisecond deadline.

For smooth constrained embedded optimization,
[`optimization-engine`](https://docs.rs/optimization_engine/latest/optimization_engine/)
is one candidate. Its Rust interface accepts a cost, gradient, and constraints;
its PANOC optimizer supports a maximum-duration setting intended for real-time
applications. Whether any solver meets a deadline is a system-level property:
measure worst-case latency, memory, target support, and infeasible cases on the
deployment hardware. “Embedded” does not by itself imply microcontroller,
`no_std`, or a guaranteed control frequency.

Offline optimization can still use `fcmaes-rust` to tune controller parameters
or explore designs before a smaller online controller is deployed.

### 8. When is `fcmaes-rust` a strong candidate?

It is a good fit when most of the following are true:

- the objective is bounded but nonsmooth, discontinuous, multimodal, noisy, or
  mixed-variable;
- gradients are unavailable or unreliable for the complete objective;
- no exact or specialized formulation captures the important behavior;
- evaluations are cheap enough, or parallel enough, for population search and
  retries;
- independent evaluations can be executed concurrently without nested thread
  oversubscription; and
- the required result is one robust design, a Pareto set, or a meaningful
  quality-diversity archive.

The nine [application tutorials](../tutorials/README.md) cover different
reasons for reaching this point: stochastic discrete events, mechanical
discontinuities, intrinsic simulation noise, changing access windows,
mixed-integer controls and solver failures, censored inverse inference,
variable geometry with numerical constraints, and validation-aware
hyperparameter tuning, plus high-dimensional fixed-topology policy search.

## Constraints, noise, and validation can change the choice

Algorithm selection is inseparable from the evaluation protocol:

- If almost all candidates are infeasible, improve the parameterization,
  repair candidates, or use a solver that represents the constraints directly.
- For stochastic objectives, use common random numbers while ranking candidates
  when appropriate, then validate finalists on disjoint seeds.
- For learned models or calibrated simulators, keep a holdout set outside the
  optimizer. More search can otherwise produce a better validation-set exploit
  rather than a better solution.
- Treat simulation failures consistently and report their frequency. An
  arbitrary huge scalar can distort both optimization and comparisons.
- Let one layer own parallelism. Do not combine unrestricted simulator threads
  with unrestricted optimizer workers.

These are not cleanup details. They can determine whether gradients are useful,
whether a surrogate is data-efficient, and whether the reported optimum
generalizes.

## Hybrid and decomposed workflows

The strongest solution often combines methods:

- **Structured solver inside, black-box optimizer outside.** Solve an exact
  routing, flow, or allocation subproblem for every outer design.
- **Global exploration followed by local refinement.** Use parallel retries to
  find promising basins, then refine with CMA-ES or, where locally valid,
  gradients.
- **Gradient-free acquisition optimization.** A surrogate method can use a
  bounded global optimizer for its cheap, multimodal acquisition function.
- **Relaxation before search.** Use a linear or continuous relaxation to obtain
  a bound, starting point, or diagnostic before optimizing the full model.
- **Offline search, online control.** Optimize architecture and policy
  parameters offline; deploy a deterministic real-time controller online.

The DE-to-CMA sequence in coordinated retry is an example of staged
derivative-free exploration and refinement. CMA-ES is not a gradient-based
local solver, so this should not be confused with a classical global-to-local
gradient hybrid.

## Symptoms that the choice or formulation is wrong

| Symptom | Likely interpretation |
| --- | --- |
| Every seed converges to the same point in a few hundred evaluations | The problem may be easy or locally smooth; reduce retries and benchmark a local method. |
| Nearly every candidate is infeasible | The encoding or constraint treatment needs work, or a structured solver is more suitable. |
| More retries help but longer individual runs do not | The landscape likely has multiple basins; favor restart diversity or coordinated retry. |
| Results vary across seeds and do not improve with budget | Noise may dominate ranking; improve replication and validation before tuning the optimizer. |
| Best candidates consistently lie on a bound | Revisit the bounds and physical model before changing algorithms. |
| Throughput falls as workers increase | Look for nested thread pools, shared locks, memory bandwidth, or process-start overhead. |
| Improvements disappear on holdout data or seeds | The search is overfitting the evaluation protocol. |
| Encoding the candidate is harder than defining its fitness | A fixed real vector may be the wrong representation. |

## Fair comparison checklist

Before adopting an optimizer, compare at least two plausible choices and record:

1. identical objective, bounds, constraints, failure handling, and decoding;
2. identical total objective evaluations, including initialization and
   restarts;
3. actual worker availability and which layer owns parallelism;
4. independent root seeds and, for noisy problems, the same named scenario
   sets;
5. wall time, actual evaluations, final quality, and their distributions;
6. success rate against a predeclared target when one exists; and
7. holdout or higher-fidelity validation of selected solutions.

Report mean and standard deviation across independent experiments, but retain
quantiles or the raw results when distributions are skewed. A single best run
is not an algorithm comparison.

## Quick reference

| Problem shape | Start with |
| --- | --- |
| Linear or mixed-integer linear | `good_lp` with a suitable backend, SCIP, or HiGHS |
| Convex conic | `clarabel` |
| Convex quadratic | `osqp` |
| Smooth with reliable gradients | `argmin` or simulator sensitivities |
| Scarce expensive evaluations | Bayesian/surrogate optimization such as `egobox` |
| Canonical routing, assignment, path, or flow | A specialized or exact solver |
| Hard real-time smooth constrained control | A purpose-built solver such as `optimization-engine` |
| Variable-length programs, graphs, or topologies | A genotype-aware evolutionary framework |
| Nonsmooth, noisy, multimodal, bounded scalar objective | `fcmaes-rust` scalar optimizer plus retry |
| Pareto set from bounded black-box objectives | `fcmaes-rust` MODE or weighted retry |
| Behaviorally diverse repertoire | `fcmaes-rust` CVT-MAP-Elites and Diversifier |

Third-party capabilities linked above were checked against their public
documentation on 2026-07-24. Recheck versions, features, transitive native
dependencies, licenses, and target support before committing a project to a
specific crate or backend.
