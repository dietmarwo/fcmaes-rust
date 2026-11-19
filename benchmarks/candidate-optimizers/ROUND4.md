# Candidate optimizers, round 4: ARRDE on the truss

Declared on 19 November 2026, before any run of this test.

## 1. Question

In the eight-worker deadline benchmark
([the algorithm portfolio](../../docs/algorithm-portfolio.md#the-eight-worker-deadline-benchmark))
ARRDE reached a median of 396 kg on a 171-variable truss where BiteOpt
reached 520 kg. BiteOpt was the only core optimizer in that comparison, the
comparison was by wall-clock time, and ARRDE spent about three times as many
evaluations. This round asks whether ARRDE is better than every core
optimizer on that problem at equal evaluation budgets.

It applies the maintainer's acceptance rule:

> It is accepted if there is a single real-world problem on which it is
> superior. Superior means solving the problem better in the end; an
> advantage at an early stage does not count.

## 2. Candidate

ARRDE (Muzakka, Shali, Suhendar, Möller and Finsterbusch, 2025,
arXiv:2511.18429) from its authors' library Minion, at the revision of the
eight-worker benchmark (`535504a`, version 2.0.1), built from source. If
that revision cannot be built on this host, `minionpy` 1.9.1 from PyPI is
used and recorded in [DEVIATIONS.md](DEVIATIONS.md) before any run.

- Default parameters. The evaluation budget of a run is the run's budget.
- No start point: the library draws its initial population.
- It works on the same box and the same real-vector encoding as the core
  solvers and gets no type information.
- A Python driver evaluates whole populations through `candidate-harness
  serve`, as `run_catcmawm.py` does, and writes rows in the format of the
  Rust runner. The harness guard is not in that path, so the driver counts
  evaluations itself, records the best value it has seen and checks budget
  and box.

## 3. Problem, solvers, budgets

- Problem: `truss_sizing` (171 variables, the scalar cost of
  `tutorials/truss-sizing`).
- Core solvers: `--solvers round3` (BiteOpt, DE, CMA-ES, CR-FM-NES, Dual
  Annealing, DE→CMA, jSO→CMA).
- Protocol T: restart until the total is spent, minimum run 1,000.

| Stage | Total evaluations | Shapes | Arms | Repetitions | Root seed |
|---|---:|---|---|---:|---:|
| 1 | 1,000,000 | one run (run cap = total); eight runs (run cap = total / 8) | 8 solvers × 2 shapes | 10 | 7 |
| 2 | 4,000,000 | the shape of each arm, scaled | the better ARRDE shape and the two core arms with the lowest stage-1 medians | 10 | 8 |
| Early | 100,000 | one run | 8 solvers | 10 | 9 |

The eight-run shape mirrors the eight independent searches of the
eight-worker benchmark. Stage 2 is there because ARRDE had about three
million evaluations per search in that benchmark. The early stage is
informational: it shows who leads at a tenth of the budget.

Raw rows go to `results/round4/stage1.tsv`, `stage2.tsv` and `early.tsv`.
Seeds are shared by all solvers of a stage.

## 4. Decision

An *arm* is a solver with a shape. Ties in the median are broken by the
lower mean.

At a stage, ARRDE is **superior** if

- the median of its better arm is lower than the median of every core arm
  of that stage, and
- a two-sided Mann–Whitney U test of that arm against the core arm with the
  lowest median gives p ≤ 0.05 after Holm correction over the two stages.

If it is superior at stage 1 or stage 2, it is **accepted**: ARRDE is
implemented in `fcmaes-core`, from the paper. Otherwise it is not accepted,
and the outcome is recorded in the portfolio.

A run that ends on an infeasible design keeps its penalized cost; the report
counts such runs per arm. `decide_round4.py` computes the tables and the
decision from the raw files and writes `results/round4/report.md`.

## 5. Limits known in advance

- One problem, 10 repetitions, one host.
- Equal evaluations, not equal time. ARRDE is cheaper per evaluation than
  BiteOpt and CMA-ES in 171 variables; with objectives that cost
  microseconds, equal time favors it more than this test does.
- The candidate is a library build, the core solvers are Rust.
