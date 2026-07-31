# Offline route-search protocol comparison

The three manifests request the same accepted-candidate target, proposal ceiling, L0 inner budget, variant cap, worker allocation, root seed, and promotion policy.

This is a transport and protocol fixture, not an agent-capability comparison. The `agent` arm is a deterministic mock whose first three proposals are historical routes.

| Arm | Status | Accepted / target | L0 admissible | Lowest L0 violation | Best admissible L0 diagnostic | Niches |
|---|---|---:|---:|---:|---:|---:|
| agent | completed | 3 / 3 | 0 | 0.552495 | — | 3 |
| random | completed | 3 / 3 | 0 | 1e+99 | — | 3 |
| evolutionary | completed | 3 / 3 | 0 | 1e+99 | — | 3 |

| Arm | Attempts / ceiling | Diversity rejected | Transport failed | Actual L0 evaluations | Worker-h | Wall-h | Agent tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent | 3 / 40 | 0 | 0 | 3204 | 0.001 | 0.001 | 0 |
| random | 3 / 40 | 0 | 0 | 3167 | 0.000 | 0.000 | 0 |
| evolutionary | 4 / 40 | 1 | 0 | 3171 | 0.000 | 0.000 | 0 |

No arm ran L1, so there are no Sims–Flanagan promotions or measured surrogate gaps.
