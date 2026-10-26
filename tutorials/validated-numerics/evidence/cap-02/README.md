# CAP-02 bounded search validation

Date: 26 October 2026. Base fcmaes-rust commit:
`46a6a48e6e3980f6c4319a2c3b18f2bce2876099` (merged CAP-01 with review fixes).

Nine arms were run against each frozen problem: reference, complete grid,
local pattern search, and random/CMA-ES/BiteOpt at seeds 17 and 29. Each study
performed 201 search evaluations, nine score rechecks and one warm-up estimate.
The shared Julia model was constructed once per study. Both selected proposals
passed separate rigorous verification and fresh replay.

| Problem | Selected arm | Inverse row scales | Estimated score | Verified radius |
| --- | --- | --- | --- | --- |
| Scalar cube root | Reference | `[1]` | -0.9999999998412598 | 1.8650578117375848e-16 |
| Cahn–Hilliard | Local | `[0.984375, 1]` | -0.9822359369382633 | 7.343308663301206e-13 |

The physical error target is `1e-10`. These easy calibration cases do not show
a meaningful optimizer benefit. All scalar arms retained the nominal point;
the local PDE adjustment improved only the estimated score slightly and its
verified radius was essentially unchanged. This is evidence that the bounded
search/verification pipeline works, not a held-out performance result.

## Retained data

- [Source commits](source-snapshot.json), [workflow summary](summary.json),
  [19 recorded workflow checks](checks.json), [four injected fault cases](fault-checks.json).
- [Julia tests](julia-tests.txt): 108 assertions passed, including the 81
  reference/arithmetic/library assertions inherited from the merged base.
- [Rust tests](rust-tests.txt): 11 tests passed, covering hard evaluation caps,
  complete grids, seeded trajectories, protocol failures and child reaping.
- [Independent exact-rational bound audit](exact-check.txt): selected scalar
  and PDE intervals enclosed the recomputed rational bounds and met every
  checked theorem condition.
- [Implementation log](IMPLEMENTATION_LOG.md).

The [scalar study](scalar-cuberoot/study.json) and
[PDE study](cahn-hilliard/study.json) each retain `evaluations.jsonl`, the
selected candidate, all arm candidates with exact-byte JSON strings, verification
and replay. Recorded hashes bind the protocol, executable, Julia source,
candidate, proof and evaluation ledger. Source files for the reference verifier,
its CLI, project and manifest were preserved, so the earlier evidence still
has the same verifier identity.

Two real experiment-craft campaigns ran search → verification → replay using
optimization-jobs command execution: **six runtime attempts**, all successful.
Continuing both completed campaigns left them at three attempts each.

## Failure and reproducibility checks

The scalar study was repeated in a new process and reproduced its full
estimated-score trajectory and selected parameters. Julia tests also queried
the same PDE point before and after another point and found identical values
with unchanged reference data.

Real deadline and SIGTERM tests terminated and reaped the Julia child.
Controlled test servers exercised mismatched score rechecks, nonfinite JSON
and incomplete estimates; the release CLI rejected them and emitted no candidate.
A protected, nonempty output directory retained its original candidate unchanged.
Cleanup assertions use namespace-aware signal/wait operations rather than
assuming `/proc` exposes the caller's PID namespace.

## Reproduction and limits

See [SEARCH.md](../../SEARCH.md) for builds and exact commands. The test harness
copies the executable to a frozen validation path before running studies, so
later Cargo build/test products cannot replace it during hash checks.
The Rust binary's empty unit-test harness is disabled; its actual CLI is tested
through the release executable, alongside the Rust library contract tests.

Validation used Linux x86-64, Rust 1.88.0, Julia 1.10.10 and Python 3.12.14,
one Julia compute thread and one BLAS thread. The runtime/client commits are
recorded in `source-snapshot.json`. Compiler startup and warm per-arm wall
times are observations, not controlled comparative timing measurements.
Actual aggregate CPU use and peak memory were not measured.

The backend was foreground execution on one node. Systemd, disconnect/reboot,
remote placement and multi-node throughput were not tested. The expanded CI
workflow is included for execution after applying the patch; it was not run
on GitHub before delivery. Only selected candidates were independently
certified; a future scientific comparison must certify every arm's candidate,
retain failures and measure time to an accepted certificate on harder held-out
cases.
