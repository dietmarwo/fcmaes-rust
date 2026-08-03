# Controlled active CMA-ES implementation diagnostic

These are paired wall-deadline measurements of `fcmaes-core` and `cmaes` 0.2.2. Both use active (negative-weight) CMA-ES, the same normalized reflected objective, explicit population size, initial mean, and sigma. This is an implementation diagnostic, not a general CMA-ES performance benchmark: its easy analytic functions isolate overhead, scaling, and stopping behavior rather than recommending CMA-ES for those functions. Matching numeric seeds label pairs but do not create matching random streams. A run may exceed its deadline by one generation.

## Bundle summary

| Raw rows | Complete pairs | Objective calls | Active wall | Process CPU |
|---:|---:|---:|---:|---:|
| 144 | 72 | 10593440 | 0.001 h | 0.011 h |

| Library | Internal-stop rows | All rows |
|---|---:|---:|
| fcmaes-core | 0 | 72 |
| cmaes | 21 | 72 |

Paired final-objective outcomes at the longest endpoint (50 ms):

| Arm | fcmaes-core wins | cmaes wins | Ties | Pairs |
|---|---:|---:|---:|---:|
| a | 2 | 5 | 5 | 12 |
| b | 1 | 3 | 8 | 12 |
| c | 5 | 1 | 6 | 12 |

## Detailed results

The table reports medians over available seeds. Wins compare final objective values within a relative `1e-10` tie band. `Residual ns/eval` is shown only for serial Arm A and subtracts the separately measured shared-objective calibration; it is diagnostic, not a library-internal profiler.

| Problem | Cost | Arm | Deadline | Pairs | Wins fc/cma/tie | Internal stops fc/cma | Median best fc | Median best cma | Median eval/s fc | Median eval/s cma | Allocated cores fc | Allocated cores cma | Residual ns/eval fc | Residual ns/eval cma |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rosenbrock10 | 0 ns | a | 10 ms | 3 | 0/0/3 | 0/3 | 4.654279e-28 | 1.226575e-23 | 1505735 | 2044916 | 1.00 | 0.37 | 597 | 422 |
| rosenbrock10 | 0 ns | a | 50 ms | 3 | 0/1/2 | 0/3 | 4.520173e-28 | 1.496247e-23 | 1584018 | 2026871 | 1.00 | 0.08 | 564 | 426 |
| rosenbrock10 | 0 ns | b | 10 ms | 3 | 0/3/0 | 0/0 | 2.176932e-1 | 9.444679e-3 | 384301 | 422603 | 14.55 | 13.87 | — | — |
| rosenbrock10 | 0 ns | b | 50 ms | 3 | 0/1/2 | 0/3 | 5.324811e-28 | 1.496247e-23 | 424756 | 472520 | 14.52 | 4.80 | — | — |
| rosenbrock10 | 0 ns | c | 10 ms | 3 | 0/0/3 | 0/3 | 4.007413e-28 | 5.335656e-24 | 16463722 | 12737375 | 15.97 | 12.62 | — | — |
| rosenbrock10 | 0 ns | c | 50 ms | 3 | 0/0/3 | 0/3 | 1.751271e-28 | 6.254638e-24 | 16477895 | 11044986 | 15.99 | 2.51 | — | — |
| rosenbrock10 | 100000 ns | a | 10 ms | 3 | 2/1/0 | 0/0 | 1.759929e3 | 1.174851e3 | 9929 | 9940 | 1.00 | 1.00 | 675 | 563 |
| rosenbrock10 | 100000 ns | a | 50 ms | 3 | 1/2/0 | 0/0 | 1.196717e1 | 9.939553e0 | 9934 | 9951 | 1.00 | 1.00 | 618 | 446 |
| rosenbrock10 | 100000 ns | b | 10 ms | 3 | 2/1/0 | 0/0 | 8.338046e0 | 9.440543e0 | 77596 | 80529 | 11.19 | 10.79 | — | — |
| rosenbrock10 | 100000 ns | b | 50 ms | 3 | 1/2/0 | 0/0 | 5.632147e-2 | 1.890527e-2 | 78308 | 77628 | 11.17 | 11.13 | — | — |
| rosenbrock10 | 100000 ns | c | 10 ms | 3 | 0/3/0 | 0/0 | 4.367944e2 | 3.516178e2 | 147876 | 153654 | 14.91 | 15.49 | — | — |
| rosenbrock10 | 100000 ns | c | 50 ms | 3 | 2/1/0 | 0/0 | 8.189657e0 | 8.709281e0 | 158038 | 158289 | 15.97 | 15.97 | — | — |
| sphere10 | 0 ns | a | 10 ms | 3 | 0/0/3 | 0/0 | 0.000000e0 | 0.000000e0 | 1472151 | 2035880 | 1.00 | 1.00 | 615 | 427 |
| sphere10 | 0 ns | a | 50 ms | 3 | 0/0/3 | 0/3 | 0.000000e0 | 0.000000e0 | 1503506 | 2165385 | 1.00 | 0.44 | 601 | 398 |
| sphere10 | 0 ns | b | 10 ms | 3 | 0/0/3 | 0/0 | 2.376049e-27 | 0.000000e0 | 399612 | 596075 | 12.93 | 13.35 | — | — |
| sphere10 | 0 ns | b | 50 ms | 3 | 0/0/3 | 0/0 | 0.000000e0 | 0.000000e0 | 515866 | 549825 | 14.35 | 14.03 | — | — |
| sphere10 | 0 ns | c | 10 ms | 3 | 0/0/3 | 0/0 | 0.000000e0 | 0.000000e0 | 15489506 | 17079734 | 15.95 | 15.80 | — | — |
| sphere10 | 0 ns | c | 50 ms | 3 | 0/0/3 | 0/3 | 0.000000e0 | 0.000000e0 | 15565638 | 15016051 | 15.98 | 12.85 | — | — |
| sphere10 | 100000 ns | a | 10 ms | 3 | 1/2/0 | 0/0 | 4.246298e0 | 6.302844e0 | 9918 | 9943 | 1.00 | 1.00 | 780 | 520 |
| sphere10 | 100000 ns | a | 50 ms | 3 | 1/2/0 | 0/0 | 1.377269e-2 | 1.352301e-2 | 9935 | 9951 | 1.00 | 1.00 | 607 | 439 |
| sphere10 | 100000 ns | b | 10 ms | 3 | 2/1/0 | 0/0 | 2.280751e-4 | 1.819043e-4 | 76416 | 81789 | 11.19 | 10.85 | — | — |
| sphere10 | 100000 ns | b | 50 ms | 3 | 0/0/3 | 0/0 | 5.745432e-26 | 2.077860e-26 | 78704 | 82127 | 11.10 | 10.95 | — | — |
| sphere10 | 100000 ns | c | 10 ms | 3 | 1/2/0 | 0/0 | 2.391887e0 | 3.663828e0 | 153666 | 148210 | 15.50 | 14.94 | — | — |
| sphere10 | 100000 ns | c | 50 ms | 3 | 3/0/0 | 0/0 | 1.868115e-3 | 2.704991e-3 | 157807 | 158039 | 15.95 | 15.94 | — | — |

## Interpretation boundary

The three arms isolate different implementation choices:

- **Arm A** is the closest same-family implementation comparison.
- **Arm B** also compares each library's population-evaluation path.
- **Arm C** gives both implementations the same independent-multistart architecture.

None of these rows compares fcmaes coordinated DE→CMA retry with `cmaes` BIPOP. That is a
different-algorithm system comparison covered by the broader optimizer benchmark.

The analytic controls do not show that CMA-ES is the right solver for them. For sufficiently
costly application objectives, the measured implementation overhead also becomes irrelevant.
Smoke and pilot presets validate the harness; they are not publication evidence.
