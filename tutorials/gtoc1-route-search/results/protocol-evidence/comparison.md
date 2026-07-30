# Matched-budget route-search comparison

All arms use the same accepted-candidate target, L0 inner budget, variant cap, worker allocation, root seed, and promotion policy.

This is a transport and protocol fixture, not an agent-capability comparison. The `agent` arm is a deterministic mock whose first three proposals are the historical JPL, JPL2 and Jena routes. Route scores are intentionally omitted.

| Arm | Status | Accepted | L0 feasible | Lowest L0 violation | L1 promotions | L1 passing | Niches | Mean surrogate gap | Worker-s | Wall-s | Agent tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agent | completed | 3 | 0 | 0.552495 | 0 | 0 | 3 | — | 3.801 | 2.099 | 0 |
| random | completed | 3 | 0 | 1e+99 | 0 | 0 | 3 | — | 0.259 | 0.224 | 0 |
| evolutionary | completed | 3 | 0 | 1e+99 | 0 | 0 | 3 | — | 0.352 | 0.280 | 0 |

No arm ran L1, so this fixture contains no Sims–Flanagan promotions and no surrogate-gap observations.
