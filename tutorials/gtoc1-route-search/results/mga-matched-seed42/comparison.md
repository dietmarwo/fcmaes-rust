# Matched MGA route-discovery comparison

The Gemma 4, grammar-random, and evolutionary arms use the same accepted-candidate target, attempt ceiling, canonical direction policy, duplicate filter, inner DE–CMA-ES budget, workers, root seed, and top-N metric.

A route is *MGA-qualified* when the Rust optimizer returns a finite impulsive MGA score. This is a downstream candidate, not a continuous-thrust GTOC1 solution.

| Arm | Status | Accepted / target | MGA-qualified | Best score | Top-N | Top-N sum | Niches |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma4 | completed | 100 / 100 | 100 | 1164199.485 | 20 | 19676460.478 | 63 |
| random | completed | 100 / 100 | 100 | 1233733.147 | 20 | 19270002.019 | 96 |
| evolutionary | completed | 100 / 100 | 100 | 1278762.760 | 20 | 22140497.511 | 97 |
| gemma4-assisted | completed | 100 / 100 | 100 | 1509902.092 | 20 | 26964276.026 | 81 |

| Arm | Attempts / ceiling | Duplicate rejected | Diversity rejected | Transport failed | MGA evaluations | Worker-h | Wall-h | Agent tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4 | 190 / 2500 | 0 | 90 | 0 | 225902730 | 178.732 | 11.661 | 2341623 |
| random | 166 / 2500 | 17 | 49 | 0 | 189786950 | 82.912 | 5.182 | 0 |
| evolutionary | 146 / 2500 | 11 | 35 | 0 | 195104939 | 77.569 | 4.848 | 0 |
| gemma4-assisted | 206 / 2500 | 0 | 106 | 0 | 191154284 | 71.179 | 4.700 | 1083036 |

The first three rows are the blind matched comparison. `gemma4-assisted` is a separately named, prior-informed follow-up: it uses the completed random and evolutionary archives to construct a length-stratified candidate menu and therefore is not a fourth independent arm.

Relative to cold Gemma, the assisted follow-up improves the top-N sum by 37.0% and completes in 2.48× less wall time. These are one-seed protocol results, not a general model-capability estimate.

Every reported arm completed. Repeat the blind matched protocol and the explicitly prior-informed follow-up across predeclared seeds before drawing a proposer-capability conclusion.
