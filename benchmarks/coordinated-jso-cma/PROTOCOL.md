# Protocol: jSO→CMA against DE→CMA in coordinated retry

Declared on 20 November 2026, before the runner change and before any run.
Changes after this commit go into `DEVIATIONS.md` before the affected runs.

## Question

Coordinated retry runs DE→CMA in every retry. Is jSO→CMA a better sequence
for coordinated retry on the GTOP trajectory problems?

The answer decides one thing: whether jSO is worth a port to the C++
backend of fast-cma-es, whose main use is coordinated retry. In fcmaes-rust
jSO→CMA has so far been compared in single runs and in independent
retries only.

## Why the answer is open

- In the candidate benchmark jSO→CMA was the best sequence on several
  problems in single runs and independent retries.
- Coordinated retry gives every retry shrunk bounds, a start point and step
  sizes taken from two earlier results. DE uses the start point and the step
  sizes for its initial population. jSO has no start point: it samples its
  population uniformly in the retry's bounds and uses only the bounds.
- jSO's evaluation limit is hard and its population schedule is scaled to
  the budget of the stage. DE can stop before its limit.

## Arms

| Arm | Sequence in every retry |
|---|---|
| `de_cma` | DE→CMA, as `gtop-advexamples` runs it (control) |
| `jso_cma` | jSO→CMA, `jso_cma_run` of the example runner |

Both arms share the budget split (40% for the first stage), the CMA-ES
stage, the seeds and the coordinator. CMA-ES uses its default bound
handling (`repair_z: false`). The only code change is a switch in the
example runner that selects the sequence for coordinated retry. With the
switch off, `gtop-advexamples` must print the same results as before; this
is checked on at least one problem and seed against the revision before
the change.

## Problems

The eight GTOP problems of `gtop-advexamples`: `cassini1`, `cassini2`,
`rosetta`, `tandem`, `messenger`, `gtoc1`, `sagas`, `messenger-full`.
`messenger-full` runs with `--value-limit 12`, the others with the default.

## Stages

| Stage | Problems | Retries | Seeds | Runs |
|---|---|---:|---|---:|
| 1 | all eight | 300 | 1 to 20 | 320 |
| 2 | at most three, chosen by stage 1 | 1,200 | 101 to 110 | at most 60 |

Other settings in both stages: `--evaluations 1500 --max-eval-fac 50
--check-interval 100 --workers 1`. One run is one call of
`gtop-advexamples` for one problem and one seed. Recorded per run: final
value, evaluations, retries, wall time.

Stage 2 runs for the problems where `jso_cma` is significantly better in
stage 1. If there are more than three, the three with the smallest p-values
are taken. If there is none, the test ends after stage 1.

## Tests

- Two-sided Mann–Whitney U test on the final values of a problem, 20
  against 20 runs in stage 1 and 10 against 10 in stage 2.
- Holm correction over the problems of a stage; level 0.05.
- A run without a retained result (possible on `messenger-full` with the
  value limit) counts as worse than every run with a result.
- The budget unit is the number of retries, as in the released example.
  The evaluations are recorded. An arm's advantage on a problem counts only
  if its median evaluations there are at most 10% above the other arm's.

## Decision

The measure is the value at the end of the budget.

| Outcome | Condition | Consequence |
|---|---|---|
| **Port recommended** | in stage 2, `jso_cma` is significantly better on at least one problem and within the 10% evaluation margin there | jSO and a `jso_cma` sequence are proposed for the C++ backend of fast-cma-es, as an option; DE→CMA stays the default |
| **Inconclusive** | `jso_cma` is significantly better in stage 2 on at least one problem, but only outside the evaluation margin | a rerun at matched evaluations is needed before a decision |
| **No port** | anything else, including no significant advantage in stage 1 | jSO stays in fcmaes-rust only; the result is recorded in the algorithm portfolio |

Problems where `jso_cma` is significantly worse are reported with every
outcome. They do not change the outcome, because the port would add an
option and not replace DE→CMA.

## Limits known in advance

- GTOP trajectory problems only, 6 to 26 variables, all deterministic.
- One setting of the coordinator and one budget split.
- jSO is used as it is. A jSO that takes the start point of coordinated
  retry is a different algorithm and is not tested here.
- Equal retries are not equal evaluations; the 10% margin handles this
  coarsely.
- Stage 1 can miss a small advantage: with eight problems and 20 runs per
  arm the smallest p-value has to be below 0.00625.
