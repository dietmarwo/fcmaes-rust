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

A second source is the
[eight-worker deadline benchmark](#the-eight-worker-deadline-benchmark). It
compares by wall-clock time on eight CPUs, includes external libraries, and
has its own section.

*Mean rank* below is the average rank among the twelve solvers of a round
(1 is best). *Best in n cells* counts the cells in which an optimizer has the
lowest median among the core optimizers.

## What is in core, and why

| Optimizer | Round 1: mean rank, best in | Round 2: mean rank, best in | What the data say |
|---|---|---|---|
| DE→CMA sequence (core DE, then core CMA-ES) | 3.60, 6 of 26 | 3.9, 6 of 10 | The strongest generalist in both rounds. |
| DE | 4.13, 10 of 26 | 5.6, 2 of 10 | Best single optimizer; best in 8 of 13 single-run cells of round 1. |
| BiteOpt | 5.02, 4 of 26 | 8.1, 0 of 10 | Strong with small run budgets: three of its four best cells are restarts of 2,000 to 25,000 evaluations. Behind at 100,000 evaluations per run. |
| Active CMA-ES | 7.38, 3 of 26 | 6.6, 1 of 10 | Best core optimizer on Tandem under restarts in both rounds, and the second stage of the sequences in those rounds. |
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
rounds. CR-FM-NES on a high-dimensional problem, PGPE, `DeepBiteOpt` and
coordinated retry were measured in the
[eight-worker deadline benchmark](#the-eight-worker-deadline-benchmark).

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

### CR-FM-NES to BiteOpt for routing

The [OneForAll tutorial](https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/OneForAll.adoc)
uses CR-FM-NES to find a starting point for DeepBiteOpt. The native Rust
composition is available through [`crfmnes_bite`](optimizers.md#staged-cr-fm-nes-and-biteopt)
and `--algo crfmnes_bite` in [`benchmark-biteopt-gtop`](examples.md#benchmarks),
and as `kind: crfmnes_bite` in the `fcmaes.retry/1` job profile.
The opt-in recipe uses a hard cap per retry, a 30/70 split, population 500,
and BiteOpt depth 4 by default. Each stage works on the continuous encoding
accepted by the objective, such as random keys for a permutation.

The implemented recipe was checked on four Solomon instances at 500,000
evaluations per retry with eight paired seeds per instance. All BiteOpt
arms used population 500 and depth 4. Scores are feasible medians of
Euclidean distance plus ten times the vehicle count. Replay checked fleet
size, capacity, customer time windows and return to the depot.

| Instance | CR-FM-NES → BiteOpt | CR-FM-NES | DeepBiteOpt | DE → BiteOpt |
|---|---:|---:|---:|---:|
| `c101` | **1,070.1** | 1,174.2 | 1,769.9 | 1,835.6 |
| `r101` | **2,005.8** | 2,119.7 | 2,159.4 | 2,192.7 |
| `r105` | **1,703.3** | 1,908.2 | 1,900.2 | 1,934.5 |
| `rc101` | **1,998.2** | 2,297.7 | 2,378.5 | 2,372.7 |

The sequence returned 32 of 32 feasible routes. It won all 32 pairs against
standalone DeepBiteOpt and DE → BiteOpt, and 30 against CR-FM-NES. Standalone
DeepBiteOpt returned six feasible routes on `r101`, and all other arms in
the table returned eight per instance. At 100,000 calls, standalone
DeepBiteOpt led on `c101`, and all `r101` trials exceeded the fleet limit.
On the 400-group knapsack `I13`, standalone CR-FM-NES ended 1.94% below the
supplied bound at 500,000 calls in the median, compared with 3.84% for the
sequence. These results support the sequence as an opt-in routing recipe.

The native recipe sets CR-FM-NES's initial step as a fraction of each
variable's range. On the unit random-key box of the routing table, this is
the decision-unit step of the Python recipe. A paired check compared the two
conventions on other boxes, with one 100,000-call retry for each of 16 seeds
per case. The range fraction won 15 of 16 pairs on Rosetta and 14 of 16 on
Tandem, whose intervals span hundreds to thousands of units. MMKP `I07` and
`I13` (also at 500,000 calls), Cassini2, GTOC1 and Messenger showed no clear
difference. On Cassini1 and Sagas, the decision-unit step reached the deepest
basins in more seeds (6 against 1 and 3 against 1), without a significant
paired difference.

## What is not in core, and why

### Benchmarked and left out

| Algorithm | Round 1 (wins/losses/ties, mean rank) | Round 2 (wins/losses/ties, mean rank) | Decision |
|---|---|---|---|
| L-SHADE (Tanabe and Fukunaga, 2014) | 1/17/8, 7.58 | 0/7/3, 7.6 | Out. Same family as jSO and behind it in both rounds. |
| L-SRTDE (Stanovov and Semenkin, 2024) | 1/13/12, 5.94 | alone 0/5/5, 6.2; before CMA-ES 0/3/7, 4.4 | Out. Close to DE→CMA as a first stage, never significantly better than it, and the published equation does not reproduce the published results (see the [deviations](../benchmarks/candidate-optimizers/DEVIATIONS.md)). |
| BIPOP-aCMA-ES (Hansen, 2009) | 1/5/20, 4.88 | 0/6/4, 7.7 | Out of core. Third of twelve with small budgets, ninth with large ones. A BIPOP schedule is available in the job worker; the eight-worker benchmark points at its settings, not at a missing algorithm. |
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
- A [convergence test](../benchmarks/candidate-optimizers/ROUND3C.md) with
  budgets of 20,000 to 200,000 evaluations asked whether it solves any of
  three problems better than every core solver. It does not: on the phased
  array its lead at 20,000 evaluations is gone at 100,000, where jSO→CMA is
  significantly better.

The repository's real-vector encodings, described in
[Combinatorial encodings](combinatorial-encodings.md), were sufficient on
every problem tested. Problems with many categorical variables coupled to
continuous ones and costlier objectives, such as hyperparameter tuning, were
out of reach of this screening and would be the place to look again.

### Surveyed and not benchmarked

| Algorithm | Reason |
|---|---|
| NL-SHADE-LBC, RDE, RDEx | Incremental over L-SHADE and jSO; worth a run only as first stages. NL-SHADE-RSP, of the same group, was screened in the eight-worker benchmark. |
| LRA-CMA-ES | Aims at noisy and multimodal problems; needs changes inside `Cmaes` and a noisy benchmark. |
| sep-CMA-ES, VD-CMA, LM-CMA, LM-MA-ES | Large-scale covariance models; CR-FM-NES covers that niche. |
| xNES, SNES | Predecessors of CR-FM-NES. |
| CMA-ES with margin, CatCMA | Predecessors of CatCMAwM, which was screened instead. |
| Particle swarm variants | Last on all seven GTOP problems in the [earlier comparison](../benchmarks/optimizer-comparison/comparison.md). |
| DIRECT, MCS | Local or low-dimensional by design; usable through the retry adapter. MADS was measured as NOMAD in the eight-worker benchmark. |
| Niching methods (HillVallEA, RS-CMSA) | A different task; MAP-Elites and the Diversifier cover diversity. |
| Metaphor-based swarm heuristics | No consistent evidence of an advantage over DE or CMA-ES baselines. |
| GOMEA (LT-GOMEA, permutation GOMEA, RV-GOMEA) and other linkage-learning EAs (DSMGA-II, P3) | The discrete variants need a discrete genotype, outside the real-vector contract of core; [Combinatorial encodings](combinatorial-encodings.md) sends such problems to genotype-aware libraries. On realistic permutation problems a learned linkage tree gained little over a random one. RV-GOMEA builds on AMaLGaM and gains most from partial evaluations, which the objective contract does not offer. |

The [survey](../benchmarks/candidate-optimizers/PROTOCOL.md) lists them with
references, except the GOMEA row, which was added after it. That row rests
on [Bosman, Luong and Thierens (2016)](https://doi.org/10.1145/2908812.2908917)
for permutation GOMEA, [Bouter et al. (2017)](https://ir.cwi.nl/pub/26553/)
for RV-GOMEA, and
[Guijt, Luong, Bosman and de Weerdt (2022)](https://research.tudelft.nl/en/publications/4e4a617c-2057-4009-811b-c005e2f0c673)
for the linkage tree against a random tree on flowshop, QAP and other
permutation problems. A screening would use the
[GOMEA library](https://pypi.org/project/gomea/) of the same group, on a
problem with a native binary or categorical genotype.

### Outside the boundary

Nelder–Mead and other local derivative-free refiners, Bayesian and surrogate
optimization, and gradient-based solvers are covered by
[The optimizer boundary](optimizer-boundary.md) and its own experiments.

## The eight-worker deadline benchmark

A separate campaign of 17 to 19 November 2026 asked a different question
from the three rounds: with eight CPUs and a wall-clock deadline, does an
external optimizer deliver more than fcmaes as it is used today? Solvers are
compared by time, not by evaluations, so the speed of an implementation and
its use of the eight workers count. Every policy called the same Rust
objectives.

| | Screening | Confirmation |
|---|---|---|
| Policies | 20: nine fcmaes policies and eleven external ones | 5: the fcmaes chooser, BiteOpt in eight independent searches, three finalists |
| Problems | 6 development cases | 10 cases in 7 application families |
| Deadline per run | 15 s | 60 s |
| Seeds | 5 | 30, in blocks that share problem, seed and host |
| Runs | 600 | 1,500 |

The cases are four GTOP trajectories, a Cooke lens, a circuit, room
ventilation (CFD), a reaction network, a neural controller with 118
variables and a truss with 171, at 0.01 ms to 90 ms per evaluation. Between
the stages, 522 runs at 10 s fixed how each policy divides the eight CPUs
and its settings.

The *fcmaes chooser* is one fixed policy per family, selected on the
development cases: coordinated DE→CMA retry for the trajectories and the
reaction network, CMA-ES with BIPOP restarts for the lens, CMA-ES for
ventilation, CR-FM-NES for the controller, DE for the circuit and BiteOpt
for the truss. Circuit and truss were held out of tuning, so their policies
are untuned defaults.

A gain counts as material at 1.25 times the speed to a preset target, or at
most 0.80 of the remaining loss at the deadline, each with a simultaneous
confidence interval clear of 1. A broad nomination needs material gains in
two unrelated families and risk flags in fewer than two.

The campaign ran revision `1241e52` of this repository, from before jSO
entered core: `Jso` and the jSO→CMA sequence are not in it. Its run records
are not part of this repository; the numbers here are from its final report.

### What it says about the core optimizers

Screening, 15 seconds, five seeds, six cases. The ranks are descriptive.

| fcmaes policy | Mean rank among the 20 policies |
|---|---:|
| CMA-ES | 7.10, the best of all 20 |
| Coordinated DE→CMA retry | 8.26 |
| CR-FM-NES | 8.60 |
| CMA-ES with BIPOP restarts (job worker) | 9.26 |
| BiteOpt | 9.46 |
| CMA-ES with IPOP restarts (job worker) | 9.96 |
| DeepBiteOpt | 10.86 |
| DE | 11.26 |
| PGPE | 12.48 |

- **The case for a portfolio holds under deadlines.** The chooser uses six
  different policies for seven families. Against it, BiteOpt alone has worse
  medians at 60 seconds on Cassini2 (9.47 against 8.41), Messenger (9.23
  against 8.70), Tandem (−1208 against −1364), the controller (0.287 against
  0.134) and the reaction network (1.65 against 1.46). On ventilation it is
  marginally ahead (1.594 against 1.599). Both returned a usable result in
  all 300 runs.
- **CR-FM-NES and PGPE have their place.** On the 118-variable controller
  the screening medians are CR-FM-NES 0.149, PGPE 0.157, CMA-ES 0.186, DE
  0.227 and BiteOpt 0.368. The three rounds have no such problem.
- **Coordinated retry**, which the rounds do not cover, has the best Tandem
  median of all 20 policies (−1337.6).
- **DE**, the best single optimizer of the rounds, is eighth of the nine
  fcmaes policies here. The measures differ: 15 seconds in eight independent
  searches against a fixed evaluation budget in one.

### What it says about the candidates

Three finalists went to confirmation, one per group: a modern DE, an
external CMA-ES and MADS.

| Finalist | Usable at 60 s | Material gain against the chooser | Risk flags | Outcome |
|---|---:|---|---|---|
| ARRDE (Minion 2.0.1) | 300/300 | Ventilation: 6.59 times the speed to target. Truss: 0.763 of the remaining loss. | Controller, trajectories | Deferred |
| BIPOP-CMA-ES (libcmaes) | 270/300 | Circuit: 1.72 times the speed to target. | Truss, trajectories | Deferred |
| MADS (NOMAD 4.6.0, poll only) | 240/300 | None | Five of seven families | Deferred |

No finalist passes the broad nomination.

- **ARRDE** (Muzakka, Shali, Suhendar, Möller and Finsterbusch, 2025) is a
  differential evolution in the line of L-SHADE and jSO with a nonlinear
  population reduction and restarts. On the truss its median is 396 kg
  against 520 kg. It reaches the preset target more slowly (0.058 times the
  speed), so the gain is in the final design, not in an early lead. The
  comparison is with BiteOpt only: BiteOpt is the chooser's policy for the
  truss, and no other core optimizer ran there. The ventilation gain is in
  speed against CMA-ES; against BiteOpt there is none (0.53 times, with an
  interval that reaches 1). On
  the controller it leaves 1.61 times the chooser's remaining loss and on
  the trajectories 2.52 times. Over all families it reaches targets at 0.29
  of the chooser's speed and leaves 15% more remaining loss.
- **libcmaes BIPOP** reaches the circuit target sooner than DE (1.72 times)
  and BiteOpt (1.92 times); the final loss is at the floor of the measure
  for every policy.
  It returns no feasible truss in 30 runs, and on the trajectories it leaves
  94 times the chooser's remaining loss. BIPOP restarts already exist in the
  job worker, so this is a finding about settings: on the controller at
  screening, libcmaes BIPOP reaches 0.148 where the job worker's BIPOP
  reaches 0.532 with three times the evaluations.
- **NOMAD** shows no material gain in any family. A 4 GiB memory guard
  stopped all 60 controller and truss runs, and on the trajectories it
  leaves several hundred times the chooser's remaining loss. It ran on a
  scalar penalty; its native constraint handling was not tested.

Screened only, with the mean rank among the 20 policies: libcmaes IPOP
8.07, RCMAES (Minion) 8.60, L-SRTDE (Minion) 9.30, jSO (Minion) 9.62,
L-SHADE (author's code) 10.38, L-SRTDE (author's code) 12.84, NL-SHADE-RSP
(Minion) 14.18, NOMAD with defaults 17.79.

This agrees with the rounds where the two overlap. L-SHADE, L-SRTDE and jSO
alone stay behind CMA-ES, coordinated retry and CR-FM-NES, and NL-SHADE-RSP
is third from last. One difference: jSO and L-SRTDE from Minion rank ahead
of fcmaes DE at 15 seconds, where DE is ahead of both in the rounds.

### Decision and follow-ups

**No decision changes.** Nothing is added to core and no default changes.
Two findings are open:

1. **ARRDE on structural sizing.** The truss result is a better final design
   on a real problem, measured against BiteOpt only. The test that would
   settle it runs ARRDE against every core optimizer, including jSO→CMA,
   its closest relative in core, on the truss by evaluations, under the
   rules of the candidate benchmark.
2. **BIPOP settings in the job worker.** Initial step size, bounds handling
   and restart schedule should be compared with libcmaes on the circuit and
   the controller.

A third observation needs a look: on the reaction network, with evaluations
of 90 ms and tails above a second, fcmaes CMA-ES and its IPOP and BIPOP
variants each lost one of five screening runs to the deadline. The cause is
not established.

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
- **CMA-ES and the bounds.** All rounds and the eight-worker benchmark ran
  with an active CMA-ES that updated from unrepaired samples when a sample
  was moved onto a bound (`CHANGELOG.md` describes the correction). Their
  numbers for CMA-ES, DE→CMA and jSO→CMA may not hold on problems whose
  solutions lie on or near a bound.
- **Not covered by the rounds.** Noisy objectives, more than 171 variables,
  multi-objective optimization, coordinated retry.
- **The eight-worker benchmark.** Wall-clock results depend on hosts and
  implementations: it ran on three cloud hosts, with every block on one
  host, and its conclusions hold for the tested libraries, settings and
  deadlines. Screening has five seeds. The reaction network uses a fixed
  panel and is deterministic. Longer deadlines, noisy objectives and the
  jSO→CMA sequence were not run. Its records are outside this repository.

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
