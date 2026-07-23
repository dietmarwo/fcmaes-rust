# GTOP optimizer comparison

The seven problems, absolute-best values, and stop targets are identical to the [native GTOP report](../benchmark_gtop.md). Each entry is based on 100 independent experiments, a maximum of 240,000 objective evaluations, root seed 1, and at most 24 evaluation workers.

Problem definitions and putative best solutions come from ESA's [GTOP database](https://www.esa.int/gsp/ACT/projects/gtop/). See ESA's [TandEM page](https://www.esa.int/gsp/ACT/projects/gtop/tandem/) for the mission model, bounds, and problem instances used by that case.

The `workers` column is the number actually available to the optimizer. `genetic_algorithms` 3.0.0 L-SHADE is correctly shown as one worker because its DE engine has no parallel population-evaluation API. BIPOP population sizes vary between restarts.

Standard deviations use the population definition (`ddof=0`). Wall time covers only the optimizer call; compilation and process startup are excluded.

## Main results

- fcmaes produces the best mean final optimum on six of seven problems and the lowest mean optimizer wall time on five of seven.
- BIPOP-CMA-ES produces the best equal-budget Tandem mean (-495.388325), but no equal-budget method reaches the `-1493` target.
- The pre-registered BIPOP-CMA-ES stress test also reaches 0/1,000 targets with 10,000,000 configured evaluations per retry; its best result is -1410.050665 after 9,466,290,846 actual evaluations.
- fcmaes coordinated DE→CMA retry reaches the Tandem target in 85/100 experiments. It uses a much larger adaptive budget—230,727,025 actual evaluations on average—so this is evidence for adaptive coordination, not an equal-budget comparison.

## Cassini1

| Library / algorithm | Parallel mode | Version | Workers | Pop/batch | Success | Mean optimum | Sdev optimum | Mean evals | Mean wall ms | Sdev wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcmaes / BiteOpt | independent-retries | 0.1.0 | 24 | 27 | 59% | 5.18260451 | 0.701499 | 236589 | 53.105 | 2.728 |
| fcmaes / DE→CMA | coordinated-retries | 0.1.0 | 24 | varies | 7% | 5.36219479 | 0.711844 | 239013 | 55.619 | 2.064 |
| fcmaes / BiteOpt | ask-tell-batch | 0.1.0 | 24 | 24 | 13% | 9.82806006 | 3.76448 | 221197 | 319.541 | 79.221 |
| cmaes / CMA-ES | parallel-population | 0.2.2 | 24 | 24 | 0% | 14.787477 | 3.17643 | 17718 | 24.406 | 5.754 |
| cmaes / BIPOP-CMA-ES | parallel-bipop | 0.2.2 | 24 | varies | 2% | 7.09423942 | 2.71647 | 235984 | 516.340 | 74.186 |
| genetic_algorithms / L-SHADE | native-serial | 3.0.0 | 1 | 48 | 0% | 6.5543373 | 1.12002 | 240000 | 908.613 | 6.266 |
| math-optimisation / DE/best/1/bin | parallel-population | 0.5.10 | 24 | 90 | 2% | 12.2918317 | 3.65338 | 235368 | 150.815 | 20.624 |
| argmin / PSO | parallel-population | 0.11.0 | 24 | 48 | 0% | 14.287633 | 4.46822 | 240000 | 257.485 | 9.837 |

## Cassini2

| Library / algorithm | Parallel mode | Version | Workers | Pop/batch | Success | Mean optimum | Sdev optimum | Mean evals | Mean wall ms | Sdev wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcmaes / BiteOpt | independent-retries | 0.1.0 | 24 | 75 | 0% | 13.9195712 | 2.24583 | 240000 | 72.888 | 2.691 |
| fcmaes / DE→CMA | coordinated-retries | 0.1.0 | 24 | varies | 0% | 14.5284315 | 2.0162 | 238991 | 76.521 | 2.960 |
| fcmaes / BiteOpt | ask-tell-batch | 0.1.0 | 24 | 24 | 0% | 20.0664915 | 2.93358 | 240000 | 601.907 | 18.568 |
| cmaes / CMA-ES | parallel-population | 0.2.2 | 24 | 24 | 0% | 18.5813831 | 3.65102 | 140566 | 430.066 | 154.935 |
| cmaes / BIPOP-CMA-ES | parallel-bipop | 0.2.2 | 24 | varies | 0% | 17.0061579 | 3.01884 | 240000 | 1088.518 | 173.782 |
| genetic_algorithms / L-SHADE | native-serial | 3.0.0 | 1 | 48 | 0% | 17.286687 | 2.10675 | 240000 | 1239.053 | 26.170 |
| math-optimisation / DE/best/1/bin | parallel-population | 0.5.10 | 24 | 330 | 0% | 19.3459712 | 3.86131 | 239910 | 131.662 | 2.288 |
| argmin / PSO | parallel-population | 0.11.0 | 24 | 48 | 0% | 27.8498303 | 6.09158 | 240000 | 353.549 | 14.068 |

## Gtoc1

| Library / algorithm | Parallel mode | Version | Workers | Pop/batch | Success | Mean optimum | Sdev optimum | Mean evals | Mean wall ms | Sdev wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcmaes / BiteOpt | independent-retries | 0.1.0 | 24 | 33 | 1% | -1092383.84 | 140020 | 239991 | 74.312 | 1.982 |
| fcmaes / DE→CMA | coordinated-retries | 0.1.0 | 24 | varies | 0% | -1128295.88 | 126658 | 239944 | 78.716 | 2.232 |
| fcmaes / BiteOpt | ask-tell-batch | 0.1.0 | 24 | 24 | 0% | -662854.873 | 363260 | 240000 | 397.132 | 14.416 |
| cmaes / CMA-ES | parallel-population | 0.2.2 | 24 | 24 | 0% | -579699.049 | 313660 | 18514 | 31.924 | 9.110 |
| cmaes / BIPOP-CMA-ES | parallel-bipop | 0.2.2 | 24 | varies | 0% | -978550.388 | 151083 | 240000 | 590.710 | 40.260 |
| genetic_algorithms / L-SHADE | native-serial | 3.0.0 | 1 | 48 | 0% | -843070.539 | 117624 | 240000 | 1307.030 | 4.971 |
| math-optimisation / DE/best/1/bin | parallel-population | 0.5.10 | 24 | 120 | 0% | -511531.964 | 343885 | 240000 | 152.663 | 3.654 |
| argmin / PSO | parallel-population | 0.11.0 | 24 | 48 | 0% | -356307.27 | 324241 | 240000 | 303.450 | 12.445 |

## Messenger

| Library / algorithm | Parallel mode | Version | Workers | Pop/batch | Success | Mean optimum | Sdev optimum | Mean evals | Mean wall ms | Sdev wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcmaes / BiteOpt | independent-retries | 0.1.0 | 24 | 63 | 0% | 11.7315986 | 0.821327 | 240000 | 58.021 | 2.564 |
| fcmaes / DE→CMA | coordinated-retries | 0.1.0 | 24 | varies | 0% | 11.8583712 | 0.770859 | 239676 | 60.936 | 2.318 |
| fcmaes / BiteOpt | ask-tell-batch | 0.1.0 | 24 | 24 | 0% | 13.2742926 | 2.26318 | 240000 | 540.782 | 27.900 |
| cmaes / CMA-ES | parallel-population | 0.2.2 | 24 | 24 | 0% | 14.876086 | 2.9605 | 95194 | 286.019 | 107.668 |
| cmaes / BIPOP-CMA-ES | parallel-bipop | 0.2.2 | 24 | varies | 0% | 12.6161603 | 1.49001 | 240000 | 984.624 | 96.962 |
| genetic_algorithms / L-SHADE | native-serial | 3.0.0 | 1 | 48 | 0% | 14.2293892 | 0.975518 | 240000 | 977.882 | 7.742 |
| math-optimisation / DE/best/1/bin | parallel-population | 0.5.10 | 24 | 270 | 0% | 14.2397847 | 2.49657 | 239760 | 134.683 | 2.755 |
| argmin / PSO | parallel-population | 0.11.0 | 24 | 48 | 0% | 19.0872625 | 4.09862 | 240000 | 332.737 | 14.719 |

## Rosetta

| Library / algorithm | Parallel mode | Version | Workers | Pop/batch | Success | Mean optimum | Sdev optimum | Mean evals | Mean wall ms | Sdev wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcmaes / BiteOpt | independent-retries | 0.1.0 | 24 | 75 | 0% | 4.31987892 | 1.26486 | 240000 | 70.008 | 2.387 |
| fcmaes / DE→CMA | coordinated-retries | 0.1.0 | 24 | varies | 0% | 3.0024645 | 0.675278 | 239940 | 74.472 | 1.843 |
| fcmaes / BiteOpt | ask-tell-batch | 0.1.0 | 24 | 24 | 0% | 5.32104893 | 2.67627 | 240000 | 593.412 | 26.393 |
| cmaes / CMA-ES | parallel-population | 0.2.2 | 24 | 24 | 0% | 4.51977051 | 2.40153 | 141104 | 422.501 | 134.065 |
| cmaes / BIPOP-CMA-ES | parallel-bipop | 0.2.2 | 24 | varies | 0% | 4.32881119 | 1.9343 | 240000 | 1097.152 | 169.559 |
| genetic_algorithms / L-SHADE | native-serial | 3.0.0 | 1 | 48 | 0% | 6.90908464 | 1.28261 | 240000 | 1216.118 | 15.307 |
| math-optimisation / DE/best/1/bin | parallel-population | 0.5.10 | 24 | 330 | 0% | 7.27098161 | 3.43071 | 239910 | 136.469 | 2.948 |
| argmin / PSO | parallel-population | 0.11.0 | 24 | 48 | 0% | 13.84546 | 5.49368 | 240000 | 372.609 | 17.722 |

## Tandem

| Library / algorithm | Parallel mode | Version | Workers | Pop/batch | Success | Mean optimum | Sdev optimum | Mean evals | Mean wall ms | Sdev wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcmaes / BiteOpt | independent-retries | 0.1.0 | 24 | 63 | 0% | -384.40993 | 129.762 | 240000 | 64.780 | 1.755 |
| fcmaes / DE→CMA | coordinated-retries | 0.1.0 | 24 | varies | 0% | -349.165192 | 166.57 | 239622 | 65.462 | 2.275 |
| fcmaes / BiteOpt | ask-tell-batch | 0.1.0 | 24 | 24 | 0% | -182.747996 | 152.29 | 240000 | 574.295 | 24.385 |
| cmaes / CMA-ES | parallel-population | 0.2.2 | 24 | 24 | 0% | -289.546806 | 215.492 | 82754 | 243.740 | 94.737 |
| cmaes / BIPOP-CMA-ES | parallel-bipop | 0.2.2 | 24 | varies | 0% | -495.388325 | 162.818 | 240000 | 983.451 | 105.756 |
| genetic_algorithms / L-SHADE | native-serial | 3.0.0 | 1 | 48 | 0% | -139.864698 | 81.6156 | 240000 | 1052.516 | 8.690 |
| math-optimisation / DE/best/1/bin | parallel-population | 0.5.10 | 24 | 270 | 0% | -157.398465 | 210.524 | 239760 | 137.154 | 3.335 |
| argmin / PSO | parallel-population | 0.11.0 | 24 | 48 | 0% | -53.1665019 | 91.0731 | 240000 | 340.443 | 16.549 |

## Sagas

| Library / algorithm | Parallel mode | Version | Workers | Pop/batch | Success | Mean optimum | Sdev optimum | Mean evals | Mean wall ms | Sdev wall ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcmaes / BiteOpt | independent-retries | 0.1.0 | 24 | 45 | 2% | 85.5599837 | 55.484 | 239996 | 34.039 | 2.278 |
| fcmaes / DE→CMA | coordinated-retries | 0.1.0 | 24 | varies | 0% | 119.656966 | 63.6099 | 232460 | 36.827 | 1.628 |
| fcmaes / BiteOpt | ask-tell-batch | 0.1.0 | 24 | 24 | 1% | 203.710282 | 51.1415 | 238155 | 410.998 | 34.813 |
| cmaes / CMA-ES | parallel-population | 0.2.2 | 24 | 24 | 0% | 235.805497 | 15.4254 | 41073 | 85.141 | 28.441 |
| cmaes / BIPOP-CMA-ES | parallel-bipop | 0.2.2 | 24 | varies | 0% | 174.341211 | 55.1338 | 240000 | 644.907 | 41.164 |
| genetic_algorithms / L-SHADE | native-serial | 3.0.0 | 1 | 48 | 0% | 136.308129 | 38.0583 | 240000 | 557.329 | 4.962 |
| math-optimisation / DE/best/1/bin | parallel-population | 0.5.10 | 24 | 180 | 0% | 228.179198 | 6.21341 | 239940 | 102.149 | 4.067 |
| argmin / PSO | parallel-population | 0.11.0 | 24 | 48 | 0% | 242.833975 | 31.5997 | 240000 | 255.259 | 11.838 |

## Relation to coordinated retry

The native coordinated DE→CMA retry results use adaptive budgets that are much larger than the common 240,000-evaluation allowance above. Six problems reached the stop value in all 100 experiments; Tandem reached it in 85:

| Problem | Coordinated success | Mean actual evals | Multiple of 240k | Exact configured ceiling |
|---|---:|---:|---:|---:|
| Cassini1 | 100% | 1,525,802 | 6.4× | 153,000,000 |
| Cassini2 | 100% | 19,890,045 | 82.9× | 229,500,000 |
| Gtoc1 | 100% | 14,729,200 | 61.4× | 382,500,000 |
| Messenger | 100% | 19,603,567 | 81.7× | 306,000,000 |
| Rosetta | 100% | 21,406,666 | 89.2× | 153,000,000 |
| Tandem | 85% | 230,727,025 | 961.4× | 765,000,000 |
| Sagas | 100% | 8,978,824 | 37.4× | 153,000,000 |

The coordinated ceilings are the exact sums of retry limits growing linearly from 1,500 to 75,000 evaluations over each problem's retry cap. Every recorded run consumed less than its theoretical ceiling. These results demonstrate the quality available from fcmaes coordination at a larger budget; they are not an equal-budget wall-time comparison.

For context, the original Python/C++ fcmaes [Performance report](https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Performance.adoc) records 81/100 Tandem successes, 166.92 s mean wall time, and 147.87 s wall-time sdev with 32 parallel Python processes. The native Rust run records 85/100, 40.207 s, and 39.105 s using 32 native retry threads. This is a historical implementation comparison, not the equal-budget crate comparison above.

## Tandem long-retry stress test

The alternative was selected *before this stress test* by the lowest mean Tandem optimum in the 100-run, 240,000-evaluation table:

| Selection candidate | Mean Tandem optimum |
|---|---:|
| cmaes/BIPOP-CMA-ES/parallel-bipop | -495.388325 |
| cmaes/CMA-ES/parallel-population | -289.546806 |
| math-optimisation/DE/best/1/bin/parallel-population | -157.398465 |
| genetic_algorithms/L-SHADE/native-serial | -139.864698 |
| argmin/PSO/parallel-population | -53.1665019 |

Selected: **cmaes / BIPOP-CMA-ES**.

The stress test ran 1,000 independent retries with 10,000,000 evaluations allowed per retry. It used 24 concurrent optimizer processes and one optimizer thread per process, avoiding nested parallelism.

| Retries | Configured total evals | Actual total evals | Successes | Best optimum | Mean optimum | Sdev optimum | Total wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 10,000,000,000 | 9,466,290,846 | 0 | -1410.05066 | -780.409885 | 188.417 | 2236.024 s |

Mean single-retry optimizer time was 53.039 s (population sdev 5.040 s).

None of the 1,000 long retries reached the Tandem stop value `-1493`. This is strong empirical evidence for this implementation and configuration, not a mathematical impossibility result.

## Interpretation constraints

- BiteOpt retry uses 24 independent 10,000-evaluation searches. BiteOpt ask/tell instead uses one 240,000-evaluation state with batches of 24.
- The equal-budget fcmaes coordinated row uses 48 DE→CMA retries on 24 workers. Retry limits grow from 2,500 to 7,500 evaluations and sum to exactly 240,000; later retries can adapt their sigma and local box from retained solutions.
- BIPOP-CMA-ES is itself an adaptive restart strategy, not plain single-population CMA-ES: it dynamically allocates restarts between small and increasingly large populations. Its conceptual counterpart is coordinated retry, while BiteOpt retry remains the fixed-budget independent-restart baseline.
- BiteOpt also adapts selectors inside each optimizer state. This is different from BIPOP population-size adaptation and coordinated retry's cross-run sigma/box adaptation; the table exposes all three rather than treating every method as straight retry.
- Population optimizers use their native population evaluation. Equal evaluation budgets and worker caps do not make their search topology identical.
- `cmaes` is unconstrained, so its adapter searches normalized coordinates and reflects out-of-range coordinates into `[0,1]` before decoding the original GTOP bounds.
- `genetic_algorithms` DE does not enforce `RangeGene` bounds during mutation. Its adapter reflects trials into the declared box before objective evaluation.
- `math-optimisation` is GPL-3.0-or-later. It is built only in the external comparison workspace and is not a dependency of fcmaes-rust.

The individual raw files and the combined `raw/all_results.tsv` contain every seed, final objective, actual evaluation count, success flag, and wall-time measurement.
