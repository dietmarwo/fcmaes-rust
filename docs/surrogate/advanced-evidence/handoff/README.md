# Budget-aware BiteOpt handoff

Does handing off from the `/2` GP to BiteOpt beat both the GP alone
(`advanced`) and BiteOpt alone (`biteopt`), and at which share of the
budget? The [protocol](protocol.json) fixed the guard, the panels, the
selection rule and the success criterion before any of these panels ran.

## Guard

`handoff_min_populations = 3`: the handoff takes place only if three
BiteOpt populations (9 + 3d calls each) remain; otherwise the GP keeps
the whole budget and the trace records `handoff_skipped`. At 48 calls
this skips the handoff on every problem of the panel: a 25% handoff
would leave 36 calls, against 45, 54 and 81 required for d = 2, 3 and 6.

## Development: choosing the share

Seeds 0–9 at 160 and 480 calls; sphere (d = 3), Styblinski–Tang (d = 6)
and the oscillator (d = 2). For each share, the score is the worst paired
geometric loss ratio against the better of the two baselines over all
problems and budgets ([selection](development-selection.json)):

| Arm | Worst ratio |
| --- | ---: |
| handoff-25 | 1.0 (oscillator, a tie) |
| handoff-50 | 18.4 (Styblinski–Tang at 160 calls, against BiteOpt) |
| handoff-75 | 907 (Styblinski–Tang at 480 calls, against BiteOpt) |

The rule chose a handoff after 25% of the budget. At 160 calls the guard
skips the 50% and 75% handoffs on Styblinski–Tang, which then lose to
BiteOpt alone.

## Confirmation

Fresh seeds 401–430, 48, 160 and 480 calls, 990 cells, none failed or
censored. Paired geometric loss ratio of handoff-25 with its 95%
bootstrap interval (below 1 is better); `/1` allows at most 256 calls:

| Calls | Problem | vs GP alone | vs BiteOpt alone | vs `/1` | Median loss: handoff / GP / BiteOpt |
| --- | --- | ---: | ---: | ---: | ---: |
| 48 | sphere | 1 [1, 1] | 3.01e-05 [1.64e-05, 5.72e-05] | 4.49 [2.68, 7.56] | 4.16e-06 / 4.16e-06 / 0.145 |
| 48 | Styblinski–Tang | 1 [1, 1] | 1.33 [0.816, 2.22] | 1.05 [0.817, 1.36] | 50.6 / 50.6 / 53.3 |
| 48 | oscillator | 1 [1, 1] | 0.68 [0.596, 0.77] | 1.02 [1.01, 1.04] | 0.000232 / 0.000232 / 0.000316 |
| 160 | sphere | 0.0489 [0.0161, 0.128] | 0.00136 [0.000279, 0.0065] | 0.407 [0.12, 1.19] | 7.69e-08 / 1.15e-06 / 3.61e-05 |
| 160 | Styblinski–Tang | 0.094 [0.0336, 0.262] | 1.54 [0.467, 5.12] | 0.0641 [0.0195, 0.236] | 1.18 / 28.3 / 0.948 |
| 160 | oscillator | 0.999 [0.995, 1] | 0.956 [0.904, 0.995] | 1 [1, 1] | 0.000228 / 0.000228 / 0.000228 |
| 480 | sphere | 5.27e-06 [3.46e-06, 8.59e-06] | 0.522 [0.277, 0.888] | — | 6.62e-14 / 3.32e-07 / 1.98e-13 |
| 480 | Styblinski–Tang | 8.72e-07 [7.84e-08, 8.92e-06] | 6.76 [0.667, 65.9] | — | 1.09e-06 / 14.1 / 1.56e-07 |
| 480 | oscillator | 1 [1, 1] | 1 [0.999, 1] | — | 0.000228 / 0.000228 / 0.000228 |

Median run time at 160 calls: handoff 0.09 s, GP alone 0.46 s, `/1`
0.51 s; at 480 calls: handoff 0.29 s, GP alone 1.63 s. BiteOpt alone
takes about a millisecond.

## Verdict

- **Predeclared criterion: not met.** On the target class, Styblinski–Tang
  at 160 and 480 calls, the handoff beats the GP alone by large factors.
  Against BiteOpt alone the point estimates are worse (1.54 and 6.76), and
  the intervals include 1. At 480 calls both reach the global basin
  (median losses 1e-6 and 2e-7). A hybrid is not shown better than BiteOpt
  alone on a many-modal problem.
- **The guard works as intended.** At 48 calls the handoff is skipped and
  the arm equals the GP alone (ratio 1 [1, 1]) instead of losing to it by
  three orders of magnitude.
- **Not predeclared, reported as observed:**
  - On sphere the handoff beats both baselines at 160 and 480 calls, and
    on the oscillator it ties.
  - At 160 calls it is the first `/2` arm whose point estimates do not
    lose to `/1` on any of the three problems. Its sphere interval reaches
    1.19, so noninferiority to `/1` is not shown.
  - It is about five times faster than the GP alone.

  A claim would need its own predeclared panel.

The handoff stays opt-in. When it is wanted, `handoff_fraction: 0.25`
with the default guard is the measured choice.

## Files

- [protocol.json](protocol.json)
- [provenance.json](provenance.json): source commit, binary and source
  hashes.
- Development: `development-{160,480}.jsonl.gz` with their manifests,
  and [development-selection.json](development-selection.json).
- Confirmation: `confirmation-{48,160,480}.jsonl.gz` with their
  manifests, and `confirmation-B-vs-REFERENCE.json` summaries against
  `advanced`, `biteopt` and (up to 160 calls) `legacy`.

The records omit the event traces: the panel binary replayed every `/2`
trace before writing its cell, and the summaries need only the
histories. `scripts/surrogate/verify_advanced_evidence.py` recomputes the
selection and every summary.
