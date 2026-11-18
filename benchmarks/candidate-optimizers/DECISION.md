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
