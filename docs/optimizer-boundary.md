# The optimizer boundary

`fcmaes-core` deliberately remains a small, pure-Rust library for bounded,
gradient-free global optimization. It does not try to contain every optimizer
that may be useful before or after a global search.

The practical boundary is:

| Capability | Project decision |
|---|---|
| DE, CMA-ES, CR-FM-NES, PGPE, BiteOpt, Dual Annealing | Implement in core |
| Retry, coordinated retry, MODE, MAP-Elites, Diversifier | Implement in core |
| Pareto indicators, non-dominated sorting, crowding distance | Implement in core |
| Nelder–Mead and other local derivative-free refiners | Keep external unless broader evidence establishes a generally useful parallel role |
| Bayesian and surrogate optimization | Delegate to a specialist crate such as `egobox` |
| Gradient-based optimization | Delegate to `argmin` or a domain solver with the required derivative contract |
| Gradient-aware quality diversity (CMA-MEGA and the DQD family) | Delegate; the precondition does not hold in this project's applications |
| Linear, mixed-integer, convex, routing, and other structured optimization | Use the corresponding specialist solver |

This is not a claim that the excluded methods are inferior. It is a statement
about a coherent API, dependency cost, and the problems for which this project
has evidence. External solvers can run inside the existing retry closure; a
permanent core dependency is not needed.

## Evidence behind the decision

The dependency-isolated
[optimizer-boundary experiment](../benchmarks/optimizer-boundary/README.md)
tested two tempting additions: bounded Nelder–Mead (NM) and Bayesian
optimization (BO). Twenty paired optimizer seeds were used. The raw per-seed
and per-evaluation files, exact commands, compiler, and host are checked in
with the [rendered comparison](../benchmarks/optimizer-boundary/results/decision-v2/comparison.md).

The corrected protocol addresses weaknesses in an earlier exploratory run:

- budgets are measured in wall-resource rounds, not just objective calls;
- DE and DE→NM use the same seed and bit-identical DE prefix;
- every simplex has enough calls to initialize and descend;
- 16-worker populations and multistarts actually execute concurrently;
- fixed-common-seed and freshly resampled ReBop objectives are separate;
- stochastic candidates are scored on disjoint held-out paths; and
- BO and DE are reconstructed at equal wall deadlines, charging measured
  surrogate overhead.

### Nelder–Mead result

The broad promotion case failed. Standalone serial NM had the worst median on
all eight problem/protocol blocks. A 16-way NM multistart also lost to full DE
on all four parallel blocks.

The DE→NM tail was not reliably better than spending the same resource rounds
on DE. At 16 workers it lost to full DE on 18 of 20 CFD seeds and all 20
optical-lens seeds. On fixed-seed ReBop it recorded 3 wins, 10 losses, and 7
ties; on freshly resampled ReBop it split 10/10. None of the one-worker paired
comparisons established a general advantage. A simplex often improved the
shorter DE prefix, especially on optical-lens, but could not repay the global
evaluations sacrificed to its sequential tail.

That distinction matters: “local refinement improves an incumbent” does not
imply “the hybrid beats continued global search under the same resources.” A
problem-specific NM adapter can still be reasonable after a campaign, but the
experiment does not justify a native general-purpose core algorithm.

### Bayesian result

BO showed a real but narrow use case. On the nine-dimensional CFD problem and
a nominal 25-call deadline, EGO beat DE on 17 of 20 seeds once an evaluation
was assumed to cost at least 10 ms. At a nominal 60 calls it retained an
observed advantage from 10 ms upward. By 150 calls DE had the better median at
every tested latency.

The hard-boundary optical problem behaved differently. BO was better at the
25-call deadline for assumed costs of 10 ms or more, but DE was already better
at 60 calls and dominated 18 of 20 seeds at 150 calls for 100 ms through 1 s
latencies. Measured end-of-trace optimizer overhead was about 2.96 seconds for
CFD and 4.02 seconds for optical, versus about 0.1 milliseconds for DE.

The conclusion is therefore conditional: BO deserves consideration for very
small, sequential, expensive-evaluation campaigns, but performance depends on
dimension, landscape, invalid regions, surrogate, and acquisition policy. A
specialist crate can evolve those choices without making them permanent
dependencies of `fcmaes-core`.

## Gradient-aware quality diversity

`fcmaes-core` supports CMA-ES emitters for MAP-Elites and `diversify`, which
places it in the CMA-ME family. It stops short of the differentiable extensions
— CMA-MEGA and the wider differentiable-quality-diversity family — for a
specific reason rather than a general one.

Differentiable quality diversity requires first-order differentiable
**measures**, not only a differentiable objective. Across the applications in
`tutorials/`, every behaviour descriptor is derived from simulator output:
measured oscillation period and amplitude, half-power beamwidth, hydraulic
coverage, structural redundancy. None of them is differentiable. The
precondition fails in every case, so an implementation would have no problem
here to validate against.

That is a statement about this project's problem set, not about the method.
Where measures are differentiable, [pyribs](https://pyribs.org/) is the
reference implementation of CMA-ME, CMA-MEGA and CMA-MAE. Policy-gradient
variants such as PGA-MAP-Elites relax the requirement — there the gradient
drives fitness only — but need a reinforcement-learning contract with a critic
and a replay buffer, which is a different problem shape.

This decision should be revisited if a problem with differentiable measures
enters the repository. A Lennard-Jones cluster study is the one identified
candidate: its energy differentiates analytically, and so do its natural
descriptors, radius of gyration and a smoothed Steinhardt `Q6`.

## Supported interoperation pattern

The closure accepted by [`retry`](retry.md) is the project’s optimizer adapter
point. An adapter should:

1. seed the external solver from `RetryContext::run_seed` when replay must be
   independent of worker scheduling;
2. use `bounds`, `guess`, and `sdev` where the external API supports them;
3. treat `max_evaluations` as the campaign target and document any unavoidable
   complete-batch rounding;
4. count every real objective call in `RetryRunResult::evaluations`; and
5. return a finite decoded point whose score matches the reported objective.

Keep the adapter and its dependency in an application, tutorial, or a small
integration crate. Create a shared adapter crate only after several real uses
expose stable common code. This preserves the fast, dependency-light core while
allowing local, surrogate, gradient, and domain solvers to participate in the
same retry and result-accounting workflow.

## Decision rule for users

- Start with a structured or gradient solver when the complete problem exposes
  valid structure or derivatives.
- Consider BO when calls are scarce, sequential, and costly enough to amortize
  surrogate work; benchmark it at equal wall time.
- Use fcmaes population methods when the objective is bounded, irregular, and
  parallel evaluations are available.
- Add local refinement only after a paired held-out comparison shows that it
  beats spending the same resources on the global method.

### How expensive is "costly enough"?

Cost alone does not decide it. Per-call cost multiplies a sequential surrogate
loop and a parallel population equally, so it largely cancels. Refining the same
lattice-Boltzmann room model from 14 ms to 980 ms per call — a 71-fold span —
produced no wall-time win for sequential EGO at any deadline by which parallel
DE had already finished, and none at all at the most expensive fidelity.

Cost matters relative to the optimizer's own overhead. EGO spent 1.7–5.5 s of
modelling across 120 calls: significant at 14 ms per call, negligible at 1 s.
Sample efficiency behaved as expected — at equal call counts EGO reached the
better value at three of the four fidelities — but never converted, because the
parallel arm completed its whole campaign before the sequential one had made
ten calls.

The deciding factor is concurrency. With `W` workers a sequential loop concedes
roughly a factor of `W`, and raising the per-call cost cannot recover it: at the
most expensive fidelity the measured wall-time ratio was 12.3 against `W = 16`.
Ask whether evaluations can run concurrently before asking whether they are
expensive.

The broader problem-selection sequence is in
[Choosing an optimizer](choosing-an-optimizer.md). This page records why the
library itself stops where it does.
