# Live L0 seed-42 route-search audit

The three manifests request the same accepted-candidate target, proposal ceiling, L0 inner budget, variant cap, worker allocation, root seed, and promotion policy.

This is one MiniMax-M3 seed-42 L0 audit, not an agent-capability conclusion. All three arms completed after the evolutionary baseline was repaired. L0 feasibility means only that the Lambert screen's launch and periapsis constraints pass. Scores remain surrogate diagnostics until L1/L2 validation.

| Arm | Status | Accepted / target | L0 admissible | Lowest L0 violation | Best admissible L0 diagnostic | Niches |
|---|---|---:|---:|---:|---:|---:|
| agent | completed | 40 / 40 | 0 | 0.342919 | — | 36 |
| random | completed | 40 / 40 | 15 | 0 | 658588.701 | 39 |
| evolutionary | completed | 40 / 40 | 24 | 0 | 437949.934 | 39 |

| Arm | Attempts / ceiling | Diversity rejected | Transport failed | Actual L0 evaluations | Worker-h | Wall-h | Agent tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent | 94 / 120 | 43 | 10 | 74005732 | 45.333 | 8.697 | 944365 |
| random | 53 / 120 | 12 | 0 | 75743371 | 44.190 | 1.381 | 0 |
| evolutionary | 58 / 120 | 16 | 0 | 81822640 | 51.776 | 1.618 | 0 |

The random arm's leading L0-admissible variant is `3-3-3-6-10|0000` with diagnostic estimated score `658588.701`. It is not an impulsive or continuous-thrust-feasible GTOC1 solution.

The completed evolutionary arm uses independent grammar-random bootstrap seeds and random immigrants during exploration; exploitation still mutates feasibility-first elites. The original one-route bootstrap failure and the bootstrap-only 39-route saturation run remain preserved separately as protocol evidence.

No arm ran L1, so there are no Sims–Flanagan promotions or measured surrogate gaps.
