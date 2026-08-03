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
| cassini1 | 16 | 100 | 38/100 | 100/100 | 5.166778 | 0.191308 | 4.930709 | 0.000002 | 86/12/2 |
| cassini2 | 16 | 100 | 0/100 | 0/100 | 20.395034 | 2.585887 | 15.140516 | 1.979760 | 96/3/1 |
| gtoc1 | 16 | 100 | 0/100 | 0/100 | -1034840.688556 | 140299.400512 | -1295551.114110 | 104974.269057 | 94/6/0 |
| messenger | 16 | 100 | 0/100 | 0/100 | 13.950756 | 1.492585 | 11.300458 | 0.694519 | 95/5/0 |
| rosetta | 16 | 100 | 0/100 | 0/100 | 6.779610 | 2.263446 | 2.816424 | 0.800562 | 93/4/3 |
| sagas | 16 | 100 | 0/100 | 0/100 | 171.742982 | 35.723425 | 120.593375 | 48.152676 | 92/7/1 |
| tandem | 16 | 100 | 0/100 | 0/100 | -201.495632 | 206.305874 | -598.689263 | 216.831269 | 93/7/0 |

### Equal-wall work audit

The arms intentionally do not use equal CPU work. These counts document how
parallel retry converts otherwise idle cores into more independent search within
the same elapsed allowance.

| Problem | Workers | Deadline | Serial wall mean | Serial wall sdev | Retry wall mean | Retry wall sdev | Serial starts mean | Retry starts mean | Serial evaluations mean | Retry evaluations mean | Serial active cores | Retry active cores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cassini1 | 16 | 4000 ms | 4.000109s | 0.000112s | 4.000491s | 0.000468s | 141.9 | 2114.4 | 1115347 | 16598162 | 1.00 | 16.00 |
| cassini2 | 16 | 4000 ms | 4.000150s | 0.000039s | 4.000503s | 0.000139s | 10.1 | 151.2 | 819929 | 12177224 | 1.00 | 16.00 |
| gtoc1 | 16 | 4000 ms | 4.000147s | 0.000140s | 4.000440s | 0.000125s | 74.7 | 1106.4 | 789225 | 11711201 | 1.00 | 15.99 |
| messenger | 16 | 4000 ms | 4.000135s | 0.000091s | 4.000469s | 0.000257s | 16.9 | 252.0 | 974569 | 14512197 | 1.00 | 16.00 |
| rosetta | 16 | 4000 ms | 4.000140s | 0.000071s | 4.000523s | 0.000195s | 9.0 | 133.1 | 800001 | 11902792 | 1.00 | 16.00 |
| sagas | 16 | 4000 ms | 4.000114s | 0.000090s | 4.000479s | 0.000269s | 67.4 | 1005.6 | 1702941 | 25383663 | 1.00 | 16.00 |
| tandem | 16 | 4000 ms | 4.000146s | 0.000083s | 4.000510s | 0.000423s | 14.8 | 221.6 | 850532 | 12774524 | 1.00 | 16.00 |

## Interpretation boundary

The equal-wall experiment intentionally spends more aggregate CPU in order to
reduce user waiting time and improve the returned solution distribution. It is
not an equal-CPU or algorithm-efficiency claim. The fixed-work comparison only isolates `fcmaes_core::retry` scheduling.
The `fcmaes-core` CMA-ES row additionally
changes the optimizer implementation and is not a pure core-count comparison.
The coordinated DE→CMA
results in the parent GTOP report use adaptive budgets and crossover, so they are
a system-level reference rather than another equal-work arm.
