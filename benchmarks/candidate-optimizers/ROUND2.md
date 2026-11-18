# Candidate optimizers, round 2: protocol

Declared on 18 November 2026, after the round-1 decision
([DECISION.md](DECISION.md)) and before any round-2 run. Gates and thresholds
in this file are fixed; a deviation is recorded in
[DEVIATIONS.md](DEVIATIONS.md) and never applied silently. Everything not
restated here is as in [PROTOCOL.md](PROTOCOL.md).

## 1. Question

Round 1 gave each restart between 2,000 and 25,000 evaluations. The adaptive
differential-evolution candidates tie their population schedule to the
budget, and that is where they lost. Round 2 asks the question round 1 left
open:

> With restart budgets of the size fcmaes is normally used with, does a
> candidate add something, either as a restarted optimizer or as the first
> stage of the DE→CMA sequence?

It also gives BIPOP-aCMA-ES, which tied most round-1 cells, more
repetitions.

## 2. Solvers

Six incumbents, unchanged: BiteOpt (`bite`), DE (`de`), active CMA-ES
(`cma`), CR-FM-NES (`crfmnes`), Dual Annealing (`da`), DE→CMA (`de_cma`).

Six candidate arms:

| Arm | Content |
|---|---|
| `lshade` | L-SHADE, as in round 1 |
| `jso` | jSO, as in round 1 |
| `lsrtde_b` | L-SRTDE, base-difference form, as in round 1 |
| `bipop` | BIPOP-aCMA-ES, as in round 1 |
| `jso_cma` | jSO→CMA sequence |
| `lsrtde_cma` | L-SRTDE (base-difference form)→CMA sequence |

A sequence arm mirrors the core DE→CMA sequence exactly, with the first
stage exchanged: the first stage gets the same share of the run's budget as
DE does in `de_cma`, with its published schedule scaled to that share; core
CMA-ES then starts from the first stage's best point with the same
population size, the same initial σ rule and the same seed scheme as in
`de_cma`, and gets the remaining evaluations.

AMaLGaM-IDEA (rejected in round 1) and L-SRTDE as printed (informational in
round 1) do not run. No candidate parameter is changed from round 1.

## 3. Problems

The ten round-1 problems with a mean evaluation time of at most 25 µs:
Cassini1, GTOC1, Sagas, Messenger reduced, Tandem, Cassini2, Rosetta,
Messenger full, and the Lennard-Jones clusters with 13 and 38 atoms.

The optical lens, the band-pass filter and the phased array (0.45 to 1.0 ms
per evaluation) are left out: 100,000 evaluations per run cost 45 to 100
seconds there, which this host cannot afford for twelve solvers.

To widen the range of domains, two further problems from this repository
are examined and enter under a rule fixed here:

- the truss-sizing tutorial (structural engineering) and the
  dispersion-source-localization tutorial (environmental inverse problem),
  each with the scalar cost its single-objective arm minimizes;
- a problem enters if it builds in the isolated workspace, its objective is
  deterministic, and its mean evaluation time over 2,000 uniform random
  points is at most 25 µs. The decision uses timing only.

The campaign needs the ten problems above; the two additions are optional.

## 4. Protocol

One protocol, **T** (restarts until the total is spent):

- The total budget of a repetition is 400,000 evaluations.
- A run gets at most 100,000 evaluations. When a run ends, by its own stop
  criterion or by its budget, the next run starts with a new seed and gets
  min(100,000, remaining) evaluations.
- Restarting ends when fewer than 1,000 evaluations remain.
- The best value over all runs counts.

A solver that always spends its run budget therefore makes four runs; a
solver that stops early makes more. This removes the round-1 effect of
incumbents leaving budget unused.

A *cell* is one problem. Each solver gets 20 repetitions per cell. Seeds
derive from root seed 2, the problem, the protocol, the repetition and the
restart index, and are the same for all solvers.

**Compute rule.** If the smoke campaign projects more than 180 minutes for
the full campaign, repetitions drop uniformly from 20 to 15. This is decided
from timings alone, before any result is examined.

## 5. Statistics and gates

As in PROTOCOL.md sections 3.5 and 3.6, with cells equal to problems:

- best incumbent per cell by median (ties: lower mean);
- two-sided Mann–Whitney U test against the best incumbent, Holm-corrected
  over the candidate's cells, α = 0.05: win, loss or tie;
- mean rank over all cells among the twelve solvers.

| Gate | Condition |
|---|---|
| G1, complementarity | at least 3 winning cells |
| G2, generalist | mean rank strictly better than the best incumbent's mean rank |

| Decision | Rule |
|---|---|
| **Include** in `fcmaes-core` | G1 or G2 holds, and the guard recorded no contract violation |
| **Watch** | at least one winning cell, but neither gate |
| **Reject** for now | no winning cell |

For a sequence arm, *include* means: implement its first-stage algorithm in
core and offer it as a sequence stage.

## 6. Validation before the campaign

- The new sequence arms pass the contract tests (budgets 1, 7, 1,000 and a
  normal budget; bounds; determinism) with zero guard violations.
- The protocol-T driver is tested: the total is never exceeded, an
  early-stopping solver gets additional runs, and a rerun reproduces the
  rows.
- All round-1 tests still pass.

## 7. Reporting

Raw rows go to `results/round2/raw.tsv`, the generated report to
`results/round2/report.md`. Round-1 files are not changed.
