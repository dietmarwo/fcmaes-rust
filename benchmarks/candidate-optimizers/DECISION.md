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

- Core `optimize_da` can exceed `max_evaluations` in its gradient and
  line-search path.
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
