# CAP-02 implementation log

Date: 26 October 2026

Implemented the bounded Rust–Julia search increment on the current GitHub
source. The result is a reusable persistent evaluation lane, a frozen set of
baseline/optimizer comparisons, explicit cost/failure accounting and a tested
search → rigorous verification → independent replay command campaign.

## GitHub state checked

| Repository | Checked commit | Finding |
| --- | --- | --- |
| fcmaes-rust | `46a6a48e6e3980f6c4319a2c3b18f2bce2876099` | CAP-01 merged as PR #36, including subsequent arithmetic/provenance guards, additional tests and an exact rational checker |
| optimization-jobs | `051435c325a51cc89be4ae26645d6937eed6f5e5` | Existing command execution and sealed-output transfer support the next increment |
| experiment-craft | `013077b37e917d994d3346aa0c1a36c5dfe6fec2` | Existing foreground campaigns support search, verification and replay bindings |

Fresh clones were used. The attached CAP-01 log describes the earlier
implementation; this increment starts from the merged/reviewed GitHub state.
The original reference verifier, CLI, Julia project/manifest, exact checker,
problem specifications and retained evidence were preserved.

## Delivered changes

| Component | Behavior |
| --- | --- |
| Nested Rust workspace `search/` | CLI, serial Julia bridge, bounded algorithms, protocol/result handling and a committed Cargo lock |
| `julia/src/CAPSearch.jl`, `julia/bin/search_server.jl` | Construct one immutable reference model; evaluate each parameter vector afresh; export numerical candidates over bounded JSON frames |
| `search-protocol.json` | Nine declared arms, fixed work cap, mandatory complete grid and recorded seeds/settings |
| `make_search_campaign.py` | Hash/stage the application and protocol, collect search artifacts and bind the selected candidate to the existing verifier and replay |
| `schemas/cap-search-0.1.schema.json` | Additive study, protocol, ledger and candidate-bundle formats; search explicitly reports that certification has not been performed |
| Tests and CI | Rust contract/lifecycle tests, Julia immutable-model/proof checks, release-CLI fault injection and two real command campaigns |
| Documentation/evidence | Search guide, new evidence directory, book/index/AI-context links and changelog |

Only fcmaes-rust requires a patch. No shared SDK/schema, solver-core,
optimization-jobs or experiment-craft source changes are required.

## Search and mathematical scope

The scalar case tunes one multiplier on its approximate inverse. The PDE
case tunes the zero-mode inverse row and a shared multiplier on the remaining
finite rows, each in `[0.5,1.5]`. Coefficients, exact equation, Fourier order,
norm and physical target remain fixed. Every evaluation begins with the same
reference data. The saved inverse is a numerical proposal; the unchanged
verifier independently checks the finite defect, infinite tail, injectivity,
contraction and physical accuracy.

The deterministic objective is an estimate of self-mapping/contraction
violation at the fixed epsilon. Timing is never used as that objective.
Every arm starts at the nominal parameters. The complete grid is 25 points
in one dimension or 5 × 5 in two; the stochastic arms use seeds 17 and 29.
CMA-ES uses fixed-work ask/tell and population four; BiteOpt uses population
eight and serial batches. A partial final CMA-ES population never causes
budget overshoot or invented observations.

For the default study, the reference uses one search call, the eight other
arms use 25 each, all nine winners are rechecked once, and the shared warm-up
uses one call: **211 floating estimates per study**. Startup/Newton construction,
warm arm costs, rechecks, exports and incomplete attempts are accounted for
separately. All arm candidates are retained byte-for-byte in a bundle; only
the overall selected candidate is certified by the example pipeline.

The bridge has one active request at a time, increasing IDs, echoed points,
source fingerprints, a 1 MiB response cap, one queued frame and bounded stderr.
Deadline and signal cancellation cover startup and blocked reads. Cleanup
terminates and waits for the owned Julia process; the runtime remains the
authority for node admission, process-group containment and recovery.

## Measured results

| Case | Selected arm/parameters | Rigorous verification | Fresh replay |
| --- | --- | --- | --- |
| Scalar cube root | Reference, `[1]` | Verified; radius 1.8650578117375848e-16 | Verified |
| Cahn–Hilliard | Local, `[0.984375,1]` | Verified; radius 7.343308663301206e-13 | Verified |

Both meet the exact `1/10000000000` target. The small PDE score adjustment
did not materially improve the verified radius. The calibration establishes
the interface and accounting; it does not show a meaningful stochastic
optimizer advantage or satisfy the held-out-benefit gate.

## Validation

| Check | Result |
| --- | --- |
| Existing merged Julia baseline | 81 assertions passed before extension |
| Full extended Julia suite | 108 assertions passed; arithmetic configuration and reference-proof checks preserved |
| Rust release contract tests | 11 tests passed, including budgets, seeded trajectories, complete grids and process reaping |
| Rust formatting and all-target clippy | Passed with warnings denied |
| Main workflow | 19 recorded checks passed, including both scientific search/proof/replay pipelines, schema/digest checks, exact rational audit, repeated trajectory, timeout and SIGTERM |
| Additional fault cases | Four passed: objective mismatch, nonfinite response, incomplete estimate and protected nonempty output |
| Real runtime integration | Two foreground campaigns; six actual command attempts; continuation created no duplicates |
| Python / CI definitions | Python syntax and YAML parsing checked |
| Documentation | Source inventory/link checks, pinned mdBook build and rendered links checked |
| Delivery | ZIP integrity, checksums and clean-base patch application checked |

The independently implemented Python rational checker recomputed the selected
scalar and PDE bounds and confirmed the interval enclosures and theorem
conditions. Fault injection uses controlled fake servers solely for failure
paths; scientific results and runtime campaigns use the real Julia model,
Rust optimizers, command runtime and campaign client.

## Issues found during implementation

- The runtime pre-creates output directories. The search now accepts a new
  or empty directory while refusing nonempty directories and existing files.
  Real campaign integration covers this behavior.
- `/proc` exposes host PIDs in this execution environment. Cleanup tests were
  corrected to use `waitpid`/signal operations in the caller's PID namespace.
- Build/test feature graphs can replace Cargo's target executable. The test
  harness freezes a copy before studies and verifies its digest throughout.
- An empty release binary unit-test harness hit a Rust 1.88 thin-LTO linking
  failure. The CLI is tested as a release executable; Rust contract tests retain
  their own release harness and all 11 pass.

## Limits and next work

Testing was on Linux x86-64 with one Julia compute thread and one BLAS thread.
The full unchanged Rust optimizer/worker workspace was not rerun. No systemd,
remote-host, disconnect/reboot or multi-node performance validation was done.
The expanded GitHub Actions workflow has not run remotely for this patch.
Actual CPU consumption and peak memory are not yet measured.

The current study is serial and small. Harder frozen/held-out cases, admissible
weight/order parameterizations, more seeds, per-arm independent certification
and censored time-to-certificate reporting are needed before claiming search
benefits. Generic proof-aware artifacts/comparisons, adaptive region coverage
and remote campaign routing remain following platform increments.

No push, release, publication or deployment was performed. The patch package
contains application instructions, its exact base/implementation identities,
checksums and this log.
