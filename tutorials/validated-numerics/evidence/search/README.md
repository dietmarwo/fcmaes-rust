# Bounded search validation evidence

These are actual search studies and certificates, not mock results. They were
produced on 26 October 2026 by the
[validated-numerics workflow](https://github.com/dietmarwo/fcmaes-rust/actions/runs/40394278516)
on a GitHub-hosted `ubuntu-latest` runner (Linux x86-64), at commit
`180aaf8dc403898a620173a95afc810f0568820f`. That commit's search application,
Julia server, search module and verifier are the ones in this repository; each
study records their SHA-256 digests, and every workflow run checks that the
Julia digests still match. All files are retained byte for byte, so their
digest links stay valid.

Nine arms were run against each frozen problem: reference, complete grid,
local pattern search, and random/CMA-ES/BiteOpt at seeds 17 and 29. Each study
made 201 search evaluations (one for the reference arm, 25 for each other arm),
nine score rechecks and one warm-up estimate: 211 floating estimates. The Julia
model was constructed once per study. Both selected candidates passed separate
rigorous verification and a fresh replay.

| Problem | Selected arm | Inverse row scales | Estimated score | Verified radius |
| --- | --- | --- | --- | --- |
| Scalar cube root | Reference | `[1]` | -0.9999999998412598 | 1.8650578117375848e-16 |
| Cahn–Hilliard | Local | `[0.984375, 1]` | -0.9822359369382584 | 7.34330866335485e-13 |

The physical error target is `1e-10`. These easy calibration cases show no
meaningful optimizer benefit. Every scalar arm kept the nominal point. For
Cahn–Hilliard, the local arm shrank the zero-mode row of the inverse, which
lowers the estimated `Z2` by 0.7% and the estimated score by 3e-10. The radius
is set by the `Y` bound, which this scaling leaves unchanged, so the selected
candidate verifies with exactly the radius of the
[reference certificate](../README.md). This is evidence that the bounded
search and verification pipeline works, not a held-out performance result.

## Retained data

- [Source commits and workflow run](source-snapshot.json),
  [workflow summary](summary.json), [22 recorded checks](checks.json) and
  [six injected faults](fault-checks.json).
- [Julia tests](julia-tests.txt): 108 assertions passed, 81 for the reference
  verifier and 27 for the search model and its framed input.
- [Rust tests](rust-tests.txt): 15 contract tests passed. They cover hard
  evaluation caps, complete grids, seeded trajectories, the nominal start,
  frame limits, protocol failures, cancellation, timeouts and cleanup.
- [Independent exact-rational recheck](exact-check.txt): the recorded intervals
  of both selected certificates enclose the exactly recomputed bounds, and every
  theorem condition holds exactly.

The [scalar study](scalar-cuberoot/study.json) and the
[Cahn–Hilliard study](cahn-hilliard/study.json) each come with
`evaluations.jsonl`, the selected `candidate.json`, all arm candidates as
exact-byte JSON strings in `candidates.json`, `verification.json` and
`replay.json`. Recorded hashes bind the protocol, executable, Julia sources,
candidates, certificates and evaluation ledger.

Two real experiment-craft campaigns ran search → verification → replay as
optimization-jobs command jobs: **six runtime attempts**, all successful.
Continuing both completed campaigns left them at three attempts each.

## Failure and reproducibility checks

The scalar study was repeated in a new process and reproduced its whole
estimated-score trajectory and selected parameters. The Julia tests query the
same Cahn–Hilliard point before and after another point and find identical
values with unchanged reference data.

The search candidates depend on the machine in their last bits, because the
Julia server recomputes the Newton reference with LAPACK. An earlier run of
the same studies on another Linux x86-64 machine selected the same arms and
parameters, but 3 of the 33 Cahn–Hilliard coefficients differed by one or two
units in the last place, and its radius was 7.343308663301206e-13. Its certificates
still replayed byte for byte on GitHub's runners. These retained candidates
match the coefficients of the reference evidence, which was also produced on
a GitHub-hosted runner.

Real tests stopped running studies by deadline (exit 124), by SIGTERM to the
search (exit 130) and by Ctrl-C to the whole process group (exit 130, also
ending Julia); each killed and reaped the Julia child. A protocol with an
unknown field was refused (exit 2). Fault-injection servers sent a score that
changed on recheck, a `NaN` literal, a string score and an incomplete estimate,
and interrupted the search between two calls. The release CLI rejected each
with the expected error, emitted no candidate, counted exactly the requests
the server received, and killed the server, which would otherwise have
outlived its closed input. A protected, nonempty output directory kept its
original candidate unchanged. Cleanup checks use signal and wait operations
in the caller's PID namespace rather than assuming that `/proc` shows it.

## Reproduction and limits

See [SEARCH.md](../../SEARCH.md) for builds and exact commands. The test
harness copies the executable to a frozen path before running studies, so later
Cargo build and test products cannot replace it during hash checks. The search
binary's empty unit-test harness is disabled, because Rust 1.88's thin-LTO
linking failed on it; the CLI is tested as a release executable, alongside the
library's contract tests.

Validation used Linux x86-64, Rust 1.88.0 for the build and tests (Clippy and
rustfmt from current stable), Julia 1.10.10 and Python 3.12, with one Julia
compute thread and one BLAS thread. Julia startup with the Newton construction
took 13.4 s for the scalar study and 21.4 s for Cahn–Hilliard. After that, an
arm took about 3 ms and 0.3 s respectively, and the first arm, which includes
the first candidate export, about 0.4 s. These are observations, not
controlled comparative timings. Aggregate CPU use and peak memory were not measured.

The backend was foreground execution on one node. Systemd, disconnect/reboot,
remote placement and multi-node throughput were not tested. Only the selected
candidates were certified. A scientific comparison must certify every arm's
candidate, retain failures and measure time to an accepted certificate on
harder held-out cases.
