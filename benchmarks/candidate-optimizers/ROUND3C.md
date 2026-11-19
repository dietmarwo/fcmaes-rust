# Candidate optimizers, round 3, convergence test for CatCMAwM

Declared on 19 November 2026, after the round-3 decision
([DECISION.md](DECISION.md): watch) and before any run of this test.
Everything not restated here is as in [ROUND3.md](ROUND3.md).

## 1. Criterion

The maintainer's acceptance rule for CatCMA with Margin:

> It is accepted if there is a single real-world problem on which it is
> superior. Superior means solving the problem better in the end; an
> advantage at an early stage does not count.

This test applies that rule with budgets well above those of round 3.

## 2. Problems and budgets

| Problem | Why | Total evaluations |
|---|---|---:|
| `phased_array` | best median of all solvers at 20,000 evaluations in round 3 (a tie) | 100,000 |
| `field_service` | second-best median at 20,000 evaluations in round 3 (a tie); 52 categorical coordinates | 200,000 |
| `water_network` | a mixed problem of the survey that was outside the 2 ms entry rule (24 four-level integers, three continuous, one two-way choice) | 20,000 |

The budgets are five and ten times the largest round-3 budget for the first
two problems; the third runs at the largest round-3 budget because one
evaluation takes about 4 ms.

## 3. Protocol

Solvers, typing, encoding, restarts until the total is spent (run cap equal
to the total, minimum run 100) and the candidate's configuration are those of
round 3. Ten repetitions per problem, root seed 5, seeds shared by all
solvers. Raw rows go to `results/round3c/<problem>.tsv`.

## 4. Decision

On a problem, CatCMAwM is **superior** if, at the total budget,

- its median is lower than the median of every core solver, and
- a two-sided Mann–Whitney U test against the core solver with the lowest
  median gives p ≤ 0.05 after Holm correction over the three problems.

If it is superior on at least one problem, it is **accepted**: CatCMAwM is
implemented in `fcmaes-core`. Otherwise the round-3 decision stands.

`convergence_round3c.py` computes the table and the decision from the raw
files and writes `results/round3c/report.md`.
