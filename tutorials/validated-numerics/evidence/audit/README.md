# All-arm certificate audit evidence

These are actual audits of the [bounded search](../../SEARCH.md), not mock
results. They were produced on 26 October 2026 by the
[validated-numerics workflow](https://github.com/dietmarwo/fcmaes-rust/actions/runs/40417534368)
on a GitHub-hosted `ubuntu-latest` runner (Linux x86-64), at commit
`1c0b977a72ec79dc87112cc56428ce884796a084`. That commit's audit collector,
schemas, protocols, search and verifier are the ones in this repository. Each
audit records the digests of `audit.py` and its schemas, and acceptance
refuses an audit made by other code. All files are retained byte for byte, so
their digest links stay valid.

In that run, `test_search.py` searched both frozen problems, and the audit
then verified and replayed all nine final candidates of each search, every
one in two fresh Julia processes.

| Problem | Accepted / declared arms | Fresh Julia processes | Exact rational rechecks | Existence radii | Collection wall time |
| --- | ---: | ---: | ---: | --- | ---: |
| Scalar cube root | 9/9 | 18 | 9 | `1.8650578117375848e-16` | 136 s |
| Cahn–Hilliard | 9/9 | 18 | 9 | `7.34330866335485e-13` to `7.343308663359019e-13` | 287 s |

All radii meet the exact physical target `1/10000000000`. The Cahn–Hilliard
radii differ only in their last digits, which shows no useful optimizer
benefit. Each arm's accounted time adds the shared search startup (13.4 s and
22.1 s), its own warm search work, and a fresh verification and replay of
about 7.5 s each (scalar) or 16 s each (Cahn–Hilliard): about 28.5 s and 54 s
per arm. These times are dominated by costs that are the same for every arm,
so no speed ranking follows from them.

## Retained data

- For each problem, `search/` holds the audited study, candidate bundle,
  evaluation ledger and selected candidate; `audit/` holds `audit.json`,
  `events.jsonl`, `report.md` and the strict `acceptance.json`; and
  `exact-check.txt` recomputes all nine certificates in exact rational
  arithmetic.
- [Workflow checks](checks.json): 18 expected exit statuses.
  [Summary](summary.json) and [scientific summary](scientific-summary.json).
- [Negative checks](negative-checks.json):
  - a Julia stand-in that fails every stage gives a complete audit with 0/9
    accepted, which `accept` rejects with exit 3;
  - a numerically bad scalar candidate is verified as inconclusive by the real
    verifier and rejected by its fresh replay;
  - SIGTERM to the collector, and Ctrl-C to its process group, stop the audit
    with exit 130, keep all nine arms, and leave no Julia child running; both
    partial audits are rejected.
- [Contract tests](contract-tests.txt): 32 passed, without Julia; they include
  the collector run against a Julia stand-in.
  [Julia tests](julia-tests.txt): 108 assertions passed.
  [Rust tests](rust-tests.txt): 15 passed.
- [Source snapshot](source-snapshot.json): the commits, the runner, and the
  digests of the audit, search and verifier sources and of the search binary.

Two real experiment-craft campaigns ran search → all-arm audit → strict
acceptance as optimization-jobs command jobs: six runtime attempts, all
successful. Continuing both completed campaigns left them at three attempts
each.

Every workflow run checks this evidence again: `accept` from the committed
paths, and the exact rational recheck of every certificate.

## Scope and limits

One runner, foreground command jobs with capacity one, one Julia compute
thread and one BLAS thread. Times include process startup and compilation and
are observations, not controlled comparisons; CPU use and peak memory were
not measured. Only final candidates were certified, so these are not times to
a first certificate during search. The two easy reference problems and two
stochastic seeds do not establish an optimizer advantage. See
[the audit guide](../../AUDIT.md) for the timing and censoring rules and the
trust boundaries, and to reproduce the checks.
