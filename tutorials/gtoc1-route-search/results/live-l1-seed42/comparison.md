# Targeted random-arm L1 follow-up

The controls were predeclared from the 15 L0-admissible random routes before any L1 result was inspected: rank 1, the median rank 8, and rank 15. L1 is the impulsive Sims–Flanagan approximation, not the continuous-thrust L2 validation.

| L0 rank | Variant | L0 diagnostic | L1 outcome | L1 score | L0−L1 gap | Max mismatch | Max throttle | Worker-s | Actual evals |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `3-3-3-6-10|0000` | 658588.701 | RefinementNotClosed | 289300.288 | 369288.413 | 1.074440 | 1.028699 | 1393.1 | 56052992 |
| 8 | `3-3-3-2-6-10|00011` | 212995.658 | PropagationFailure | — | — | — | — | 3738.0 | 0 |
| 15 | `3-2-2-3-2-2-2-2-3-2-3-10|00000000100` | 7005.533 | PropagationFailure | — | — | — | — | 97.3 | 0 |

No promotion passed the declared L1 threshold. The leader returned a finite diagnostic score, but its normalized endpoint mismatch remained far above `1e-7`. Both controls encountered typed Kepler propagation failures; their zero actual-evaluation fields mean the exception occurred before the retry layer returned its evaluation counter, not that the failed promotion consumed no compute. Worker-seconds retain the observed cost.
