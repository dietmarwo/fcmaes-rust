# GTOP: single-threaded CMA-ES through parallel retry

The primary experiment asks a user-facing question: how much better is the
solution distribution when the same wall-time allowance can use all physical
cores? Every external `cmaes` optimizer instance is serial. Because CMA-ES can
terminate protectively before the deadline, each lane immediately starts a new
CMA-ES run and retains its best result. The serial arm uses one restart lane;
the retry arm coordinates one lane per worker through `fcmaes_core::retry`. Both
arms use the same lane deadline, objective, bounds, population, sigma, and
deterministic root-seed scheme.

## Equal-wall-time solution quality

Each retry pair includes the serial seed stream as lane zero plus additional
independent streams in the parallel arm. Mean and population standard deviation (`Sdev`) of
the best objective are the primary outcomes; smaller is better. `Retry W/T/L`
is the paired win/tie/loss count for parallel retry. Measured wall time audits
deadline comparability in the separate work table.

| Problem | Workers | Pairs | Serial success | Retry success | Serial best mean | Serial best sdev | Retry best mean | Retry best sdev | Retry W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cassini1 | 16 | 3 | 0/3 | 0/3 | 11.020164 | 0.016744 | 5.303421 | 0.000000 | 3/0/0 |

The paired win count is primarily a scheduler check. Mean-start ratios range from 17.25× to 17.25×. Under iid independent restarts with no information sharing, a `k`-to-one ratio predicts a retry win probability of `k/(k+1)`, or 94.5%–94.5% here. The observed W/T/L values are consistent with that baseline; the mean and sdev columns quantify the returned solution distribution.

### Equal-wall work audit

The arms intentionally do not use equal CPU work. These counts document how
parallel retry converts otherwise idle cores into more independent search within
the same elapsed allowance.

| Problem | Workers | Deadline | Serial wall mean | Serial wall sdev | Retry wall mean | Retry wall sdev | Serial starts mean | Retry starts mean | Serial evaluations mean | Retry evaluations mean | Serial active cores | Retry active cores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cassini1 | 16 | 250 ms | 0.250119s | 0.000008s | 0.250409s | 0.000087s | 8.0 | 138.0 | 69753 | 1025208 | 1.00 | 15.97 |

## Interpretation boundary

The equal-wall experiment intentionally spends more aggregate CPU in order to
reduce user waiting time and improve the returned solution distribution. It is
not an equal-CPU or algorithm-efficiency claim. The fixed-work comparison only isolates `fcmaes_core::retry` scheduling.
The `fcmaes-core` CMA-ES row additionally
changes the optimizer implementation and is not a pure core-count comparison.
The coordinated DE→CMA
results in the parent GTOP report use adaptive budgets and crossover, so they are
a system-level reference rather than another equal-work arm.
