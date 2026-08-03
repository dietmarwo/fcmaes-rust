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
| cassini1 | 2 | 5 | 5 | 1.96× | 0.02× | 98.1% | 1.98 | 0.01 |
| cassini1 | 4 | 5 | 5 | 3.81× | 0.04× | 95.3% | 3.92 | 0.02 |
| cassini1 | 8 | 5 | 5 | 7.33× | 0.21× | 91.7% | 7.64 | 0.22 |
| cassini1 | 16 | 5 | 5 | 12.23× | 0.14× | 76.4% | 13.61 | 0.21 |
| rosetta | 2 | 5 | 5 | 1.96× | 0.01× | 97.9% | 1.98 | 0.01 |
| rosetta | 4 | 5 | 5 | 3.81× | 0.04× | 95.3% | 3.93 | 0.03 |
| rosetta | 8 | 5 | 5 | 7.26× | 0.13× | 90.7% | 7.61 | 0.11 |
| rosetta | 16 | 5 | 5 | 11.80× | 0.66× | 73.8% | 13.27 | 0.47 |
| tandem | 2 | 5 | 5 | 1.94× | 0.02× | 97.0% | 1.97 | 0.01 |
| tandem | 4 | 5 | 5 | 3.67× | 0.06× | 91.8% | 3.79 | 0.06 |
| tandem | 8 | 5 | 5 | 6.68× | 0.24× | 83.5% | 7.00 | 0.24 |
| tandem | 16 | 5 | 5 | 11.18× | 0.98× | 69.8% | 12.58 | 0.85 |

### Fixed-work arm summary

Only the paired external sequential/retry rows above are an exact
scheduling comparison. This table exposes the one-start and native-CMA rows
without treating their different searches as core-count speedups.

| Problem | Arm | Workers | Runs | Success | Mean best | Sdev best | Mean wall | Sdev wall | Mean evaluations | Sdev evaluations | Mean active cores | Sdev active cores |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cassini1 | external cmaes, one serial lane | 1 | 5 | 0% | 30.095247 | 22.462751 | 0.0077s | 0.0003s | 2007 | 0 | 1.00 | 0.00 |
| cassini1 | external cmaes, sequential retries | 1 | 5 | 0% | 5.309159 | 0.010148 | 0.2926s | 0.0028s | 80248 | 40 | 1.00 | 0.00 |
| cassini1 | external cmaes + fcmaes retry | 2 | 5 | 0% | 5.309159 | 0.010148 | 0.1491s | 0.0018s | 80248 | 40 | 1.98 | 0.01 |
| cassini1 | external cmaes + fcmaes retry | 4 | 5 | 0% | 5.309159 | 0.010148 | 0.0767s | 0.0011s | 80248 | 40 | 3.92 | 0.02 |
| cassini1 | external cmaes + fcmaes retry | 8 | 5 | 0% | 5.309159 | 0.010148 | 0.0399s | 0.0010s | 80248 | 40 | 7.64 | 0.22 |
| cassini1 | external cmaes + fcmaes retry | 16 | 5 | 0% | 5.309159 | 0.010148 | 0.0239s | 0.0000s | 80248 | 40 | 13.61 | 0.21 |
| cassini1 | fcmaes-core CMA-ES + retry | 16 | 5 | 0% | 6.573121 | 2.229488 | 0.0245s | 0.0013s | 80035 | 370 | 13.30 | 0.29 |
| rosetta | external cmaes, one serial lane | 1 | 5 | 0% | 21.154052 | 4.436151 | 0.0113s | 0.0004s | 2002 | 0 | 0.99 | 0.01 |
| rosetta | external cmaes, sequential retries | 1 | 5 | 0% | 8.938271 | 2.615869 | 0.4315s | 0.0019s | 80080 | 0 | 1.00 | 0.00 |
| rosetta | external cmaes + fcmaes retry | 2 | 5 | 0% | 8.938271 | 2.615869 | 0.2205s | 0.0016s | 80080 | 0 | 1.98 | 0.01 |
| rosetta | external cmaes + fcmaes retry | 4 | 5 | 0% | 8.938271 | 2.615869 | 0.1132s | 0.0015s | 80080 | 0 | 3.93 | 0.03 |
| rosetta | external cmaes + fcmaes retry | 8 | 5 | 0% | 8.938271 | 2.615869 | 0.0595s | 0.0010s | 80080 | 0 | 7.61 | 0.11 |
| rosetta | external cmaes + fcmaes retry | 16 | 5 | 0% | 8.938271 | 2.615869 | 0.0367s | 0.0020s | 80080 | 0 | 13.27 | 0.47 |
| rosetta | fcmaes-core CMA-ES + retry | 16 | 5 | 0% | 6.215927 | 1.047339 | 0.0396s | 0.0008s | 79953 | 230 | 13.64 | 0.28 |
| tandem | external cmaes, one serial lane | 1 | 5 | 0% | -0.591172 | 0.742058 | 0.0103s | 0.0002s | 2004 | 0 | 1.00 | 0.00 |
| tandem | external cmaes, sequential retries | 1 | 5 | 0% | -23.037258 | 8.555004 | 0.4262s | 0.0104s | 80160 | 0 | 1.00 | 0.00 |
| tandem | external cmaes + fcmaes retry | 2 | 5 | 0% | -23.037258 | 8.555004 | 0.2197s | 0.0043s | 80160 | 0 | 1.97 | 0.01 |
| tandem | external cmaes + fcmaes retry | 4 | 5 | 0% | -23.037258 | 8.555004 | 0.1160s | 0.0035s | 80160 | 0 | 3.79 | 0.06 |
| tandem | external cmaes + fcmaes retry | 8 | 5 | 0% | -23.037258 | 8.555004 | 0.0639s | 0.0031s | 80160 | 0 | 7.00 | 0.24 |
| tandem | external cmaes + fcmaes retry | 16 | 5 | 0% | -23.037258 | 8.555004 | 0.0384s | 0.0036s | 80160 | 0 | 12.58 | 0.85 |
| tandem | fcmaes-core CMA-ES + retry | 16 | 5 | 0% | -26.260133 | 10.749192 | 0.0353s | 0.0012s | 74873 | 1248 | 12.93 | 0.41 |

## Interpretation boundary

The fixed-work comparison only isolates `fcmaes_core::retry` scheduling.
The `fcmaes-core` CMA-ES row additionally
changes the optimizer implementation and is not a pure core-count comparison.
The coordinated DE→CMA
results in the parent GTOP report use adaptive budgets and crossover, so they are
a system-level reference rather than another equal-work arm.
