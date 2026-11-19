# Candidate optimizers: decision

Campaign of 18 November 2026: 12 solvers, 13 real-world problems, 2
protocols, 12 repetitions, 3,744 runs, no contract violation. The rules are
in [PROTOCOL.md](PROTOCOL.md), the departures from them in
[DEVIATIONS.md](DEVIATIONS.md), all numbers in
[results/report.md](results/report.md) and every run in
[results/raw.tsv](results/raw.tsv).

## Outcome

No candidate passes an inclusion gate. Nothing is added to `fcmaes-core`.

| Candidate | Wins | Losses | Ties | Mean rank (of 12) | G1 | G2 | Decision |
|---|---:|---:|---:|---:|:-:|:-:|---|
| BIPOP-aCMA-ES (`bipop`) | 1 | 5 | 20 | 4.88 | no | no | Watch |
| L-SRTDE, base difference (`lsrtde_b`) | 1 | 13 | 12 | 5.94 | no | no | Watch |
| jSO (`jso`) | 1 | 16 | 9 | 7.27 | no | no | Watch |
| L-SHADE (`lshade`) | 1 | 17 | 8 | 7.58 | no | no | Watch |
| AMaLGaM-IDEA (`amalgam`) | 0 | 15 | 11 | 7.60 | no | no | Reject |
| L-SRTDE as printed (`lsrtde`) | 0 | 21 | 5 | 9.73 | no | no | informational |

Wins, losses and ties are against the best incumbent of each of the 26
cells. Incumbent mean ranks: DE→CMA 3.60, DE 4.13, BiteOpt 5.02, CR-FM-NES
6.79, CMA-ES 7.38, Dual Annealing 8.08.

## What the cells show

- **The winning cells are isolated.** L-SHADE and jSO win Rosetta under the
  single-run protocol (median 1.99 against 3.83 for DE). `lsrtde_b` wins
  Messenger reduced, single run (10.85 against 12.29). BIPOP wins the
  band-pass filter under restarts.
- **BIPOP is the closest to the incumbents.** It loses none of the 16 GTOP
  cells and ranks third of the twelve solvers. Its five losses are the
  band-pass single run, the phased array under restarts and three of the
  four Lennard-Jones cells.
- **The restart protocol is where the adaptive DE variants lose.** Under
  restarts (one eighth of the budget per run) L-SHADE and jSO lose 11 of 13
  cells and `lsrtde_b` 8 of 13. Under a single run they lose 6, 5 and 5 of
  13 and win 1 each. Their population schedules are tied to the budget and
  start with 18·D to 20·D individuals.
- **Lennard-Jones clusters separate the families.** With 114 variables the
  best candidate median is −53.0, against −155.1 for Dual Annealing, which
  uses gradient-based local search.

## Limits

- Twelve repetitions per cell resolve large effects only; with the Holm
  correction over 26 cells a tie is not evidence of equality.
- Two budget shapes on one host. Eight of the thirteen problems are GTOP
  problems, on which the fcmaes defaults were developed.
- The phased-array restart cell gives each run 2,000 evaluations, about one
  generation for the population-reduction candidates.
- `lsrtde_b` is the form that reproduces the results reported in the
  L-SRTDE paper; the equation printed there is `lsrtde`. Which one the
  authors ran is inferred, not confirmed.
- AMaLGaM-IDEA uses population doubling on restart, not the interleaved
  multi-population scheme of the paper.

## Findings outside the question

- Core `optimize_da` could exceed `max_evaluations` in its gradient and
  line-search path at the time of the campaign; the harness truncated its
  runs at the budget (see the [README](./README.md)).
- Four incumbents stop on their own criteria before the budget is spent
  under the single-run protocol: `Cmaes` uses a median of 32% of the budget
  (minimum 0.4%), Dual Annealing 65%, CR-FM-NES 93%, and the DE→CMA sequence
  falls below 90% in 72 of 156 runs. These are the library defaults without
  a restart wrapper; the candidates and BiteOpt and DE always spend the
  whole budget. This favors the candidates in single-run cells.

## What would change the decision

- A restart protocol with at least 100,000 evaluations per run, which is
  how coordinated retry is normally configured, for `lsrtde_b`, jSO and
  L-SHADE.
- More repetitions on the cells BIPOP ties, and a decision whether the
  BIPOP schedule of `fcmaes-job-worker` should move into core.

# Round 2

Campaign of 18 November 2026 under [ROUND2.md](ROUND2.md): 12 solvers, 10
problems (8 GTOP, 2 Lennard-Jones), protocol T (400,000 evaluations per
repetition, runs of at most 100,000, restart until spent), 20 repetitions,
2,400 runs, no contract violation. Numbers are in
[results/round2/report.md](results/round2/report.md), runs in
[results/round2/raw.tsv](results/round2/raw.tsv).

## Outcome by the declared rules

| Candidate | Wins | Losses | Ties | Mean rank (of 12) | G1 | G2 | Decision |
|---|---:|---:|---:|---:|:-:|:-:|---|
| jSO→CMA (`jso_cma`) | 1 | 3 | 6 | 3.7 | no | yes | Include |
| L-SRTDE→CMA (`lsrtde_cma`) | 0 | 3 | 7 | 4.4 | no | no | Reject |
| L-SRTDE, base difference (`lsrtde_b`) | 0 | 5 | 5 | 6.2 | no | no | Reject |
| jSO (`jso`) | 1 | 6 | 3 | 7.1 | no | no | Watch |
| L-SHADE (`lshade`) | 0 | 7 | 3 | 7.6 | no | no | Reject |
| BIPOP-aCMA-ES (`bipop`) | 0 | 6 | 4 | 7.7 | no | no | Reject |

Incumbent mean ranks: DE→CMA 3.9, DE 5.6, CMA-ES 6.6, CR-FM-NES 8.0,
BiteOpt 8.1, Dual Annealing 9.1.

`jso_cma` meets gate G2 with a mean rank of 3.7 against 3.9 for DE→CMA. By
the declared rule this is an *include*: implement jSO in core and offer it
as a sequence stage.

## How strong the include is

The gate compares two mean ranks without a margin. Two checks outside the
declared rules ([posthoc_round2.py](posthoc_round2.py)) show that the
difference is within the noise of 20 repetitions:

- When the repetitions are resampled, `jso_cma` meets G2 in 55% of 10,000
  resamples. The 95% interval of its mean rank minus that of DE→CMA is
  −1.2 to +1.1.
- Head to head with DE→CMA, `jso_cma` is significantly better on GTOC1 and
  Tandem and significantly worse on Cassini1, Sagas and Messenger full.
  `lsrtde_cma` is significantly worse on the same three and significantly
  better on none.

The data therefore support this reading: a jSO first stage is about as good
as the DE first stage overall and has a different profile; it does not
replace DE.

| Problem | DE→CMA median | jSO→CMA median | L-SRTDE→CMA median |
|---|---:|---:|---:|
| GTOC1 | −1,060,676 | −1,200,625 | −1,099,773 |
| Tandem | −558.8 | −779.6 | −706.3 |
| Messenger reduced | 10.11 | 9.50 | 8.78 |
| Rosetta | 1.963 | 1.902 | 1.885 |
| Lennard-Jones 38 | −123.7 | −128.7 | −127.5 |
| Lennard-Jones 13 | −41.445 | −41.458 | −41.420 |
| Cassini2 | 14.46 | 14.86 | 14.25 |
| Cassini1 | 4.968 | 5.303 | 5.303 |
| Messenger full | 11.02 | 13.51 | 13.45 |
| Sagas | 83.3 | 225.1 | 228.7 |

## What the cells show

- **Standalone candidates do not pass with large restart budgets either.**
  L-SHADE, jSO and L-SRTDE lose 7, 6 and 5 of 10 problems; the only win is
  jSO on GTOC1.
- **The CMA-ES second stage is what makes them competitive.** On the
  Lennard-Jones cluster with 114 variables, jSO alone reaches −39.7 and
  jSO→CMA −128.7.
- **BIPOP falls back.** It ranks ninth of twelve and loses 6 of 10
  problems. In round 1 it ranked third; the rounds differ in budget and in
  that every solver now spends the whole budget.
- **Sagas, Messenger full and Cassini1 belong to the DE first stage.**
  Both candidate sequences settle at the 225 basin on Sagas in the median.

## Limits

- Two domains only. The truss-sizing (36.7 µs) and dispersion (43.3 µs)
  problems missed the 25 µs entry rule, and the three costlier round-1
  problems were out of reach at 100,000 evaluations per run.
- One protocol, one host, 20 repetitions, published defaults.
- The first stage gets 40% of each run because the core sequence does; no
  other split was tried.

# Round 3: CatCMA with Margin

Screening of 18 and 19 November 2026 under [ROUND3.md](ROUND3.md) and
[ROUND3B.md](ROUND3B.md), with the departures recorded in
[DEVIATIONS.md](DEVIATIONS.md). CatCMAwM ran as its authors' reference
implementation (`cmaes` 0.13.1, default parameters, native variable types)
against seven core solvers, on identical objectives, at 2,000 and 20,000
evaluations with 20 repetitions: 2,880 runs, no contract violation. Reports:
[round 3](results/round3/report.md), [second batch](results/round3b/report.md),
[union](results/round3b/report-union.md).

## Outcome

**Watch. CatCMAwM is not implemented in core, and it is not rejected.**

| Cells | Wins | Losses | Ties | Mean rank (of 8) | Best incumbent | Resample share | Gates |
|---|---:|---:|---:|---:|---|---:|---|
| Round 3: six repository problems (12 cells) | 1 | 3 | 8 | 3.33 | BiteOpt 3.79 | 97.3% | G2 met |
| Second batch: three categorical problems (6 cells) | 0 | 3 | 3 | 5.00 | DE, jSO→CMA 3.75 | 0.0% | none |
| **Union (18 cells), the declared basis** | 1 | 6 | 11 | 3.89 | BiteOpt 3.92 | 43.1% | **none** |

The mean-rank gate needs the candidate ahead of every incumbent in at least
95% of the resamples. Over the union it is ahead in 43.1%.

## What the cells show

- **No real win.** The single winning cell is truss sizing at 2,000
  evaluations, where CatCMAwM finds no feasible truss in any run and only
  sits lower on the infeasibility plateau. At 20,000 evaluations it finds a
  feasible truss in 0 of 20 runs; BiteOpt does in 19, jSO→CMA in 14, core
  CMA-ES in 1 and CR-FM-NES in 0.
- **Level with the best at the larger budget on two mixed problems.** Best
  median on the phased array at 20,000 evaluations (−27.77 against −27.45
  for CR-FM-NES) and second on field-service routing (1,313.8 against
  1,300.4 for Dual Annealing); both are ties. It loses both at 2,000.
- **No advantage on categorical structure.** Over the ten cells of problems
  with a categorical coordinate: no win, four losses, six ties, mean rank
  4.5 against 3.85 for Dual Annealing.
  - Welded beam with material choice: DE, CMA-ES, DE→CMA and jSO→CMA reach
    the published optimum value 1.94188 in the median at 20,000
    evaluations; CatCMAwM reaches 2.06291.
  - Pest control (25 five-way choices): CatCMAwM ties the best at 2,000
    evaluations (12.07) and loses to BiteOpt at 20,000 (12.07 against
    12.03).
- **Six cells say nothing.** On the job shop, material flow and the mixed
  cantilever beam every solver has the same median.

## Was the candidate badly configured?

A check outside the declared rules
([posthoc_round3_start.py](posthoc_round3_start.py), output in
[results/round3/posthoc-start.txt](results/round3/posthoc-start.txt);
3 to 8 seeds that are not campaign seeds) compared the library defaults
with a random start, a random start with a doubled initial spread, and
that with a population of 31:

- On truss sizing no variant finds a feasible truss.
- On field-service routing the default is the best variant at 2,000
  evaluations (median 1,932 against 12,374 or more) and level at 20,000.
  The default starts at the center of the box, which helps there; the core
  solvers start from random points.

The margin, the learning rates and other variable typings were not varied.

## Convergence test

The round-3 budgets end at 20,000 evaluations, where the candidate was level
with the best core solver on two problems. [ROUND3C.md](ROUND3C.md) asks
whether it solves any problem better once the budget is large enough for
the solvers to converge. The rule, declared before the runs: superior on a
problem if its median is lower than every core solver's and the difference
to the best core solver is significant; superior on one problem is enough
to implement it in core.

| Problem | Total | CatCMAwM median | Best core solver | Its median | Adjusted p | Superior |
|---|---:|---:|---|---:|---:|:-:|
| phased_array | 100,000 | -29.47 | jSO→CMA | -32.31 | 0.023 | no |
| field_service | 200,000 | 1298.7 | BiteOpt | 1254.8 | 0.43 | no |
| water_network | 20,000 | 97.8 | DE | 76.7 | 0.011 | no |

**Not superior on any problem; the decision stays at watch.**
([report](results/round3c/report.md), 10 repetitions.)

- On the phased array the candidate had the lowest median at 20,000
  evaluations. At 100,000 it is second, and jSO→CMA is significantly
  better: the earlier result was a faster start, not a better solution.
- On field-service routing it is third, level with BiteOpt, CMA-ES and
  Dual Annealing.
- On the water network it is seventh of eight and significantly worse than
  DE. The library fails on this problem's single two-way categorical
  variable, so the switch was typed as an integer for the candidate
  ([DEVIATIONS.md](DEVIATIONS.md)); the problem tests its integer handling
  only.

## Limits

- The candidate is a Python reference implementation, compared by
  evaluations only.
- The repository has few cheap problems with categorical variables; the
  second batch adds three from the literature, one of which (the mixed
  cantilever beam, three variables) is solved by every solver.
- The costlier problems with the richest mixed structure (energy hub, water
  network, ML tuning) were outside the 2 ms entry rule on this host.
- 20 repetitions, two budgets, default parameters, one host.

## What would change the decision

- Mixed problems with many categorical variables coupled to continuous
  ones, at budgets of a few tens of thousands of evaluations and with
  costlier objectives: hyperparameter tuning on tabular or surrogate
  benchmarks, reaction optimization, the energy hub on a larger machine.
- A win on such problems that the core solvers cannot match with the
  repository's real-vector encodings.

# Related evidence: the eight-worker deadline benchmark

A separate campaign of 17 to 19 November 2026 compared fcmaes with external
libraries by wall-clock time on eight CPUs: 20 policies screened at 15
seconds, three finalists confirmed in 1,500 runs at 60 seconds on ten cases.
It was not run under the rules of this benchmark, its records are not in
this repository, and it changes none of the outcomes above.
[The algorithm portfolio](../../docs/algorithm-portfolio.md#the-eight-worker-deadline-benchmark)
summarizes it. Where it meets this benchmark:

- **L-SHADE, L-SRTDE, jSO alone.** Screened from their authors' code or
  from Minion, all rank behind fcmaes CMA-ES, coordinated DE→CMA retry and
  CR-FM-NES. That agrees with rounds 1 and 2. jSO→CMA was not in the
  campaign, which ran a revision from before jSO entered core.
- **BIPOP.** The libcmaes implementation reaches a circuit target 1.72 times
  sooner than the fcmaes policy and returns no feasible truss. The job
  worker already has BIPOP restarts; the finding is about their settings.
- **New candidate: ARRDE.** On a 171-variable truss it reaches a median of
  396 kg where BiteOpt reaches 520 kg at the deadline. BiteOpt was the only
  core optimizer in that comparison. A run of ARRDE against all core solvers
  on `truss_sizing` under a declared protocol here would decide it.
- **MADS.** NOMAD 4.6.0 shows no gain in any of seven families.
