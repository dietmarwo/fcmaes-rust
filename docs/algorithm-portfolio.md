# The algorithm portfolio

`fcmaes-core` contains a small set of gradient-free optimizers. This page
states which algorithms are in, which were examined and left out, and the
measurements behind each decision. It complements
[The optimizer boundary](optimizer-boundary.md), which explains why whole
classes of methods (local refiners, Bayesian optimization, gradient solvers)
stay outside core.

## The evidence

The decisions rest on the
[candidate-optimizer benchmark](../benchmarks/candidate-optimizers/README.md),
a predeclared experiment in three rounds. Every solver runs behind one
interface, on one thread, under a guard that enforces the evaluation budget
and the box. Protocols, gates and departures from them were written down
before the runs; all raw rows are in the repository.

| | Round 1 | Round 2 |
|---|---|---|
| Question | Does a missing algorithm beat the core optimizers? | Does it help with the restart budgets fcmaes is normally used with, alone or as a first stage before CMA-ES? |
| Problems | 13: eight GTOP trajectories, optical lens, band-pass filter, phased array, Lennard-Jones 13 and 38 | 10: eight GTOP trajectories, Lennard-Jones 13 and 38 |
| Budget | 16,000 to 200,000 evaluations by evaluation cost; one run (S) or eight runs of one eighth each (R) | 400,000 evaluations in runs of at most 100,000, restarted until spent |
| Repetitions | 12 per cell, 26 cells | 20 per cell, 10 cells |
| Runs | 3,744 | 2,400 |
| Record | [protocol](../benchmarks/candidate-optimizers/PROTOCOL.md), [report](../benchmarks/candidate-optimizers/results/report.md) | [protocol](../benchmarks/candidate-optimizers/ROUND2.md), [report](../benchmarks/candidate-optimizers/results/round2/report.md) |

A candidate is compared with the best core optimizer of each cell by a
two-sided Mann–Whitney U test with Holm correction. It passes a gate with at
least three winning cells, or with a mean rank better than the best core
optimizer's. The [decision file](../benchmarks/candidate-optimizers/DECISION.md)
has the outcome of every round and its limits. Round 3, a screening of a
mixed-variable optimizer on nine problems at 2,000 and 20,000 evaluations
(2,880 runs; protocols [ROUND3.md](../benchmarks/candidate-optimizers/ROUND3.md)
and [ROUND3B.md](../benchmarks/candidate-optimizers/ROUND3B.md)), is
described with its candidate below.

*Mean rank* below is the average rank among the twelve solvers of a round
(1 is best). *Best in n cells* counts the cells in which an optimizer has the
lowest median among the core optimizers.

## What is in core, and why

| Optimizer | Round 1: mean rank, best in | Round 2: mean rank, best in | What the data say |
|---|---|---|---|
| DE→CMA sequence (core DE, then core CMA-ES) | 3.60, 6 of 26 | 3.9, 6 of 10 | The strongest generalist in both rounds. |
| DE | 4.13, 10 of 26 | 5.6, 2 of 10 | Best single optimizer; best in 8 of 13 single-run cells of round 1. |
| BiteOpt | 5.02, 4 of 26 | 8.1, 0 of 10 | Strong with small run budgets: three of its four best cells are restarts of 2,000 to 25,000 evaluations. Behind at 100,000 evaluations per run. |
| Active CMA-ES | 7.38, 3 of 26 | 6.6, 1 of 10 | Best core optimizer on Tandem under restarts in both rounds, and the second stage of every sequence. |
| CR-FM-NES | 6.79, 0 of 26 | 8.0, 0 of 10 | Never the best here. Its purpose, high-dimensional problems, is outside this benchmark (114 variables at most). |
| Dual Annealing | 8.08, 3 of 26 | 9.1, 1 of 10 | A specialist: best core optimizer in three of the six Lennard-Jones cells, where its gradient-based local search fits the smooth landscape, and last on average. |
| jSO | as a candidate, see below | as a candidate, see below | Included as a first stage before CMA-ES, beside DE. |
| PGPE | not run | not run | Targets high-dimensional noisy policy search, which this benchmark does not contain. |

Three observations follow from the table:

- **No single optimizer is best everywhere.** Five of the six core
  optimizers that ran are the best of a cell somewhere. That is the case for
  keeping a portfolio and for the advice in
  [Choosing an optimizer](choosing-an-optimizer.md).
- **The sequence is worth more than its parts.** DE→CMA ranks ahead of both
  DE and CMA-ES in both rounds.
- **Budget shape matters.** Among the six core optimizers alone, BiteOpt
  has a mean rank of 2.98 in round 1 and 4.2 in round 2.

BiteOpt ran with its default depth. `DeepBiteOpt` was not part of the
benchmark.

### Why jSO is in

jSO (Brest, Maučec and Bošković, 2017) is an adaptive differential evolution
with success-history parameter adaptation and a population that shrinks
along the budget. It is in core as [`Jso`](optimizers.md) for one use: as the
first stage of a sequence with CMA-ES, in the same role DE has in DE→CMA.

| Round 2, 10 problems | jSO→CMA | DE→CMA |
|---|---:|---:|
| Mean rank among twelve solvers | 3.7 | 3.9 |
| Significantly better than the other sequence | GTOC1, Tandem | Cassini1, Sagas, Messenger full |
| Lower median | 6 problems | 4 problems |

- jSO→CMA meets the mean-rank gate. The margin is small: when the
  repetitions are resampled, it stays ahead of DE→CMA in 55% of the
  resamples.
- The two sequences have different profiles. jSO→CMA reaches a median of
  −779.6 on Tandem where DE→CMA reaches −558.8; on Sagas it settles at 225.1
  where DE→CMA reaches 83.3.
- jSO **alone** does not pass: one win and six losses in round 2, one win
  and sixteen losses in round 1. Its population schedule needs the whole
  budget of a run, and CMA-ES supplies the refinement it lacks. On the
  Lennard-Jones cluster with 38 atoms, jSO alone reaches −39.7 and jSO→CMA
  −128.7.

So jSO widens the first-stage choice; it does not replace DE. DE→CMA stays
the default sequence, and jSO→CMA is a reasonable second attempt when
DE→CMA stalls.

## What is not in core, and why

### Benchmarked and left out

| Algorithm | Round 1 (wins/losses/ties, mean rank) | Round 2 (wins/losses/ties, mean rank) | Decision |
|---|---|---|---|
| L-SHADE (Tanabe and Fukunaga, 2014) | 1/17/8, 7.58 | 0/7/3, 7.6 | Out. Same family as jSO and behind it in both rounds. |
| L-SRTDE (Stanovov and Semenkin, 2024) | 1/13/12, 5.94 | alone 0/5/5, 6.2; before CMA-ES 0/3/7, 4.4 | Out. Close to DE→CMA as a first stage, never significantly better than it, and the published equation does not reproduce the published results (see the [deviations](../benchmarks/candidate-optimizers/DEVIATIONS.md)). |
| BIPOP-aCMA-ES (Hansen, 2009) | 1/5/20, 4.88 | 0/6/4, 7.7 | Out of core. Third of twelve with small budgets, ninth with large ones. A BIPOP schedule is available in the job worker. |
| AMaLGaM-IDEA (Bosman, Grahl and Thierens, 2013) | 0/15/11, 7.60 | not run | Out. No winning cell; its published population rule leaves few generations above about 100 variables. |

Wins, losses and ties are against the best core optimizer of each cell.

### Screened: CatCMA with Margin

CatCMAwM (Hamano, Nomura, Saito, Uchida and Shirakawa, 2025) optimizes
continuous, integer and categorical variables jointly. `fcmaes-core` has no
optimizer with native categorical variables, so it was screened in round 3
as its authors' Python implementation against the core solvers working on
the decoded real vector.

| Cells | Wins | Losses | Ties | Mean rank (of 8) | Best core solver |
|---|---:|---:|---:|---:|---|
| Six mixed problems of this repository | 1 | 3 | 8 | 3.33 | BiteOpt 3.79 |
| Three categorical problems from the literature | 0 | 3 | 3 | 5.00 | DE, jSO→CMA 3.75 |
| All 18 cells | 1 | 6 | 11 | 3.89 | BiteOpt 3.92 |

- **Decision: watch.** It is not implemented in core and not rejected. Over
  all cells it is level with BiteOpt on mean rank, ahead in 43% of the
  resamples where the gate asks for 95%.
- It has no real win: the one winning cell is among infeasible truss
  designs, and it never finds a feasible truss.
- At 20,000 evaluations it matches the best core solver on the phased array
  and on field-service routing.
- On the ten cells with categorical variables it has no win and four losses.
  On the welded beam with material choice, DE, CMA-ES and both sequences
  reach the published optimum in the median; CatCMAwM does not.

The repository's real-vector encodings, described in
[Combinatorial encodings](combinatorial-encodings.md), were sufficient on
every problem tested. Problems with many categorical variables coupled to
continuous ones and costlier objectives, such as hyperparameter tuning, were
out of reach of this screening and would be the place to look again.

### Surveyed and not benchmarked

| Algorithm | Reason |
|---|---|
| NL-SHADE-RSP, NL-SHADE-LBC, RDE, RDEx | Incremental over L-SHADE and jSO; worth a run only as first stages. |
| LRA-CMA-ES | Aims at noisy and multimodal problems; needs changes inside `Cmaes` and a noisy benchmark. |
| sep-CMA-ES, VD-CMA, LM-CMA, LM-MA-ES | Large-scale covariance models; CR-FM-NES covers that niche. |
| xNES, SNES | Predecessors of CR-FM-NES. |
| CMA-ES with margin, CatCMA | Predecessors of CatCMAwM, which was screened instead. |
| Particle swarm variants | Last on all seven GTOP problems in the [earlier comparison](../benchmarks/optimizer-comparison/comparison.md). |
| MADS, DIRECT, MCS | Local or low-dimensional by design; usable through the retry adapter. |
| Niching methods (HillVallEA, RS-CMSA) | A different task; MAP-Elites and the Diversifier cover diversity. |
| Metaphor-based swarm heuristics | No consistent evidence of an advantage over DE or CMA-ES baselines. |

The [survey](../benchmarks/candidate-optimizers/PROTOCOL.md) lists them with
references.

### Outside the boundary

Nelder–Mead and other local derivative-free refiners, Bayesian and surrogate
optimization, and gradient-based solvers are covered by
[The optimizer boundary](optimizer-boundary.md) and its own experiments.

## Limits of the evidence

- **Problems.** Eight of the problems are GTOP trajectory problems, on which
  the fcmaes defaults were developed. Round 2 covers two domains only.
- **Resolution.** 12 and 20 repetitions resolve large effects. A tie is not
  evidence of equality.
- **Settings.** Candidates ran with their published defaults and core
  optimizers with library defaults. Nothing was tuned on the benchmark.
- **Round-1 single runs.** CMA-ES, CR-FM-NES, Dual Annealing and DE→CMA can
  stop before the budget is spent; round 2 restarts them until it is.
- **The jSO gate.** The mean-rank gate has no margin, and jSO→CMA passes it
  narrowly.
- **Mixed-variable screening.** Round 3 ran the candidate as a Python
  reference implementation with default parameters, compared by evaluations,
  on problems of at most 171 variables and 2 ms per evaluation.
- **Not covered.** Noisy objectives, more than 171 variables,
  multi-objective optimization, coordinated retry.

## Challenging a decision

The benchmark is built to be rerun and extended:

1. Implement the algorithm behind the
   [`Solver` contract](../benchmarks/candidate-optimizers/solver-api/src/lib.rs)
   with its published defaults.
2. Declare problems, budgets, statistics and gates before running, as the
   two protocol files do.
3. Run the campaign and let the gates decide. Keep failed rows.

A candidate that passes is implemented in core with the ask/tell interface,
documentation and tests of the existing optimizers.
