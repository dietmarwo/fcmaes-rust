# GTOP: single-threaded CMA-ES through parallel retry

This bundle contains secondary scheduling or target-stopping diagnostics. Every
external `cmaes` optimizer instance is serial; `fcmaes_core::retry` supplies only
outer multistart scheduling. See the experiment README and the separate
equal-wall bundle for the user-facing solution-quality comparison.


## Fixed-work scheduling diagnostic

The fixed-work phase disables target stopping. `Same work` counts paired
runs with equal completed retries, evaluations, and final best value. Speedup is
paired sequential wall time divided by parallel wall time. This is a scheduler
diagnostic, not the solution-quality comparison.

| Problem | Workers | Pairs | Same work | Mean speedup | Sdev speedup | Efficiency | Mean active cores | Sdev active cores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cassini1 | 2 | 1 | 1 | 1.41× | 0.00× | 70.6% | 1.74 | 0.00 |

### Fixed-work arm summary

Only the paired external sequential/retry rows above are an exact
scheduling comparison. This table exposes the one-start and native-CMA rows
without treating their different searches as core-count speedups.

| Problem | Arm | Workers | Runs | Success | Mean best | Sdev best | Mean wall | Sdev wall | Mean evaluations | Sdev evaluations | Mean active cores | Sdev active cores |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cassini1 | external cmaes, one serial lane | 1 | 1 | 0% | 63.756010 | 0.000000 | 0.0009s | 0.0000s | 207 | 0 | 1.00 | 0.00 |
| cassini1 | external cmaes, sequential retries | 1 | 1 | 0% | 19.719897 | 0.000000 | 0.0035s | 0.0000s | 828 | 0 | 1.00 | 0.00 |
| cassini1 | external cmaes + fcmaes retry | 2 | 1 | 0% | 19.719897 | 0.000000 | 0.0025s | 0.0000s | 828 | 0 | 1.74 | 0.00 |
| cassini1 | fcmaes-core CMA-ES + retry | 2 | 1 | 0% | 26.092117 | 0.000000 | 0.0018s | 0.0000s | 828 | 0 | 1.98 | 0.00 |

## Target-oriented results

This phase stops scheduling new retries after reaching the published GTOP
target. Already-running starts are allowed to finish, so wall time is the
user-visible call duration through worker drain for successes and time to
exhaustion for failures. Evaluation counts remain resource
accounting, not the primary outcome.

| Problem | Arm | Workers | Runs | Success | Mean best | Sdev best | Mean wall | Sdev wall | Mean evaluations | Sdev evaluations | Mean active cores | Sdev active cores |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cassini1 | external cmaes, one serial lane | 1 | 1 | 0% | 63.756010 | 0.000000 | 0.0009s | 0.0000s | 207 | 0 | 1.00 | 0.00 |
| cassini1 | external cmaes, sequential retries | 1 | 1 | 0% | 19.719897 | 0.000000 | 0.0034s | 0.0000s | 828 | 0 | 1.00 | 0.00 |
| cassini1 | external cmaes + fcmaes retry | 2 | 1 | 0% | 19.719897 | 0.000000 | 0.0018s | 0.0000s | 828 | 0 | 1.97 | 0.00 |
| cassini1 | fcmaes-core CMA-ES + retry | 2 | 1 | 0% | 26.092117 | 0.000000 | 0.0023s | 0.0000s | 828 | 0 | 1.82 | 0.00 |

## Interpretation boundary

The fixed-work comparison only isolates `fcmaes_core::retry` scheduling.
The `fcmaes-core` CMA-ES row additionally
changes the optimizer implementation and is not a pure core-count comparison.
The coordinated DE→CMA
results in the parent GTOP report use adaptive budgets and crossover, so they are
a system-level reference rather than another equal-work arm.
