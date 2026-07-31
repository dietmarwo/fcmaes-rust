# Optimizer-boundary decision experiment

Recorded campaign: 20 independent root seeds.

## Corrected Nelder–Mead experiment

Scores below are medians over independent optimizer seeds and use held-out validation for both ReBop variants. Lower is better. `de-head` is the exact DE prefix supplied to the hybrid; it separates improvement by the tail from differences in DE random streams.

| Protocol | Problem | DE | DE head | DE→NM | NM serial | NM multistart |
|---|---|---:|---:|---:|---:|---:|
| `parallel-r60-w16` | cfd-ventilation | 1.52498 | 1.52727 | 1.52727 | 1.66876 | 1.53134 |
| `parallel-r60-w16` | optical-lens | 86.2012 | 132.923 | 131.305 | 1.751e+05 | 239.045 |
| `parallel-r60-w16` | rebop-crn | 3.04061 | 3.1878 | 3.1878 | 4.47526 | 3.21081 |
| `parallel-r60-w16` | rebop-resampled | 3.08997 | 3.06348 | 3.00804 | 4.60839 | 3.32624 |
| `serial-r160-w1` | cfd-ventilation | 1.55287 | 1.57037 | 1.56993 | 1.63327 | — |
| `serial-r160-w1` | optical-lens | 455.142 | 557.987 | 540.134 | 792.397 | — |
| `serial-r160-w1` | rebop-crn | 3.2163 | 3.32363 | 3.40135 | 4.10756 | — |
| `serial-r160-w1` | rebop-resampled | 3.45314 | 3.55764 | 3.32032 | 4.13449 | — |

### Paired DE→NM evidence

`W/L/T` compares held-out DE→NM against full-budget DE for the same root seed. The effect is the median log score ratio; negative favors DE→NM. The interval is a deterministic paired bootstrap over seeds.

| Protocol | Problem | W/L/T | sign p | median log ratio | 95% bootstrap CI | tail vs DE head W/L/T |
|---|---|---:|---:|---:|---:|---:|
| `parallel-r60-w16` | cfd-ventilation | 2/18/0 | 0.0004 | 6.986e-05 | [2.038e-05, 3.609e-04] | 7/0/13 |
| `parallel-r60-w16` | optical-lens | 0/20/0 | 0.0000 | 0.425033 | [0.233049, 0.497756] | 18/0/2 |
| `parallel-r60-w16` | rebop-crn | 3/10/7 | 0.0923 | 0.00612956 | [0, 0.0642285] | 2/0/18 |
| `parallel-r60-w16` | rebop-resampled | 10/10/0 | 1.0000 | 0.017258 | [-0.0472225, 0.0963837] | 10/9/1 |
| `serial-r160-w1` | cfd-ventilation | 8/11/1 | 0.6476 | 1.023e-04 | [-5.485e-04, 0.0025482] | 18/0/2 |
| `serial-r160-w1` | optical-lens | 9/11/0 | 0.8238 | 0.017737 | [-0.0266009, 0.0951969] | 20/0/0 |
| `serial-r160-w1` | rebop-crn | 6/7/7 | 1.0000 | 0 | [-0.0081631, 0.0044426] | 5/4/11 |
| `serial-r160-w1` | rebop-resampled | 11/8/1 | 0.6476 | -0.00418485 | [-0.0388108, 0.0494711] | 12/6/2 |

Resource accounting is exact. With one worker, a 16-member DE generation costs 16 rounds. With 16 workers it costs one round. A serial simplex costs one round per objective call; multistart runs one simplex per worker. Every NM tail receives at least `2 × (dimension + 1)` calls, so every reported simplex is initialized and has a genuine descent budget.

## Equal-wall-time Bayesian experiment

The objective landscape is evaluated normally, while each best-so-far trace records optimizer overhead with simulator time subtracted. For an assumed per-evaluation latency `c`, call `i` completes at `overhead(i) + i·c`. DE and EGO are then compared at the same deadline `nominal calls × c`; BO cannot spend its modelling time twice.

| Problem | nominal calls | latency | DE score (calls) | BO score (calls) | BO vs DE W/L/T | sign p |
|---|---:|---:|---:|---:|---:|---:|
| cfd-ventilation | 25 | 1 ms | 1.65432 (24) | 1.65411 (2) | 9/11/0 | 0.8238 |
| cfd-ventilation | 25 | 10 ms | 1.65432 (24) | 1.63271 (19) | 17/3/0 | 0.0026 |
| cfd-ventilation | 25 | 100 ms | 1.65432 (24) | 1.63103 (23) | 17/3/0 | 0.0026 |
| cfd-ventilation | 25 | 500 ms | 1.65432 (24) | 1.63103 (24) | 18/2/0 | 0.0004 |
| cfd-ventilation | 25 | 1000 ms | 1.65432 (24) | 1.63103 (24) | 18/2/0 | 0.0004 |
| cfd-ventilation | 60 | 1 ms | 1.63566 (59) | 1.63428 (17) | 13/7/0 | 0.2632 |
| cfd-ventilation | 60 | 10 ms | 1.63566 (59) | 1.6195 (36) | 15/5/0 | 0.0414 |
| cfd-ventilation | 60 | 100 ms | 1.63566 (59) | 1.60179 (55) | 17/3/0 | 0.0026 |
| cfd-ventilation | 60 | 500 ms | 1.63566 (59) | 1.60179 (58) | 17/3/0 | 0.0026 |
| cfd-ventilation | 60 | 1000 ms | 1.63566 (59) | 1.60179 (59) | 17/3/0 | 0.0026 |
| cfd-ventilation | 150 | 1 ms | 1.5562 (149) | 1.62481 (26) | 6/14/0 | 0.1153 |
| cfd-ventilation | 150 | 10 ms | 1.5562 (149) | 1.60179 (74) | 7/13/0 | 0.2632 |
| cfd-ventilation | 150 | 100 ms | 1.5562 (149) | 1.59458 (130) | 7/13/0 | 0.2632 |
| cfd-ventilation | 150 | 500 ms | 1.5562 (149) | 1.59458 (145) | 7/13/0 | 0.2632 |
| cfd-ventilation | 150 | 1000 ms | 1.5562 (149) | 1.59458 (147) | 7/13/0 | 0.2632 |
| optical-lens | 25 | 1 ms | 2.568e+05 (24) | n/a (0) | 0/0/20 | 1.0000 |
| optical-lens | 25 | 10 ms | 2.568e+05 (24) | 1.290e+05 (18) | 15/5/0 | 0.0414 |
| optical-lens | 25 | 100 ms | 2.568e+05 (24) | 1.186e+05 (23) | 15/5/0 | 0.0414 |
| optical-lens | 25 | 500 ms | 2.568e+05 (24) | 1.126e+05 (24) | 15/5/0 | 0.0414 |
| optical-lens | 25 | 1000 ms | 2.568e+05 (24) | 1.126e+05 (24) | 15/5/0 | 0.0414 |
| optical-lens | 60 | 1 ms | 21511.8 (59) | 1.368e+05 (17) | 6/14/0 | 0.1153 |
| optical-lens | 60 | 10 ms | 21511.8 (59) | 67300.3 (34) | 7/13/0 | 0.2632 |
| optical-lens | 60 | 100 ms | 21511.8 (59) | 55604.3 (54) | 7/13/0 | 0.2632 |
| optical-lens | 60 | 500 ms | 21511.8 (59) | 55170.4 (58) | 7/13/0 | 0.2632 |
| optical-lens | 60 | 1000 ms | 21511.8 (59) | 55170.4 (59) | 7/13/0 | 0.2632 |
| optical-lens | 150 | 1 ms | 455.142 (149) | 1.126e+05 (24) | 0/20/0 | 0.0000 |
| optical-lens | 150 | 10 ms | 455.142 (149) | 51802.8 (69) | 0/20/0 | 0.0000 |
| optical-lens | 150 | 100 ms | 455.142 (149) | 24423.2 (126) | 2/18/0 | 0.0004 |
| optical-lens | 150 | 500 ms | 455.142 (149) | 24423.2 (143) | 2/18/0 | 0.0004 |
| optical-lens | 150 | 1000 ms | 455.142 (149) | 24423.2 (146) | 2/18/0 | 0.0004 |

### Measured optimizer overhead at the end of the trace

| Problem | DE | EGO |
|---|---:|---:|
| cfd-ventilation | 0.0001 s | 2.9605 s |
| optical-lens | 0.0001 s | 4.0208 s |

Raw per-seed and per-evaluation artifacts are `refiner-raw.tsv` and `bo-trace.tsv`. They, rather than this rendered table, are authoritative.
