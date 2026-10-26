# All-arm certificate audit evidence

These records extend the [search study](../search/README.md) with independent
verification and fresh replay of every final candidate. The exact equations,
target and verifier are unchanged. The earlier reference and search evidence
is preserved.

| Problem | Independently accepted | Fresh Julia processes | Exact rational certificate checks |
| --- | ---: | ---: | ---: |
| Scalar cube root | 9/9 | 18 | 9 |
| Cahn–Hilliard | 9/9 | 18 | 9 |

All existence radii meet the exact `1/10000000000` physical target. The
[scientific summary](scientific-summary.json) records the measured radii and
collection wall costs. Per-arm costs and the acceptance observations remain
in each problem's audit files; no speed ranking is inferred.

The [workflow checks](checks.json) and [summary](summary.json) record both
real three-job campaigns and continuation without duplicate attempts.
[Contract regressions](contract-tests.txt) record 22 passing tests.
`pre-fix-regressions.txt` intentionally records assertion failures from
temporary predecessor copies, confirming that the three fixes have effective
regression tests; it is not a failing result for the delivered code.

The [source snapshot](source-snapshot.json) identifies the checked repository
bases, application source, protocols, tested binary and environment. Each
problem retains the complete input study/bundle/ledger under `search/`, and
the audit, stage ledger, human report and strict acceptance under `audit/`.
The acceptance command rechecks all digest links; every accepted certificate
also has an exact rational recomputation in the retained test results.

This is a single local Linux environment with foreground command jobs and
capacity one. Accounted times describe final candidates; unsuccessful
observations stay in the denominator. See [the audit guide](../../AUDIT.md)
for timing/censoring rules and trust boundaries. The two easy calibration
cases and two stochastic seeds do not establish optimizer advantage or an
anytime time-to-first-certificate result.

To reproduce, run the Julia and Rust checks and `test_search.py` from
[SEARCH.md](../../SEARCH.md), then `test_audit_contract.py` and `test_audit.py`
from [AUDIT.md](../../AUDIT.md). The latter validates this retained evidence
and runs new independent scientific computations. Recompute search and audit
in the same environment for new timing evidence; reusing a retained search's
costs on a different machine is not a new performance study.
