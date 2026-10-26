# CAP-01 implementation log

Date: 26 October 2026
Repository changed: `dietmarwo/fcmaes-rust`
Base commit: `ed70f303b7ca7c568af786c6c814b1d13b7c06ef`
Status: implemented and locally validated; no push, release or publication.

## Delivered behavior

CAP-01 is a standalone Julia reference application under
`tutorials/validated-numerics/`. It produces numerical candidates, rigorously
verifies them, and independently replays their certificates in fresh processes.
It supports the two frozen references selected in the repository-extension
assessment: a scalar cube root and a Cahn–Hilliard Neumann steady state.

The first increment uses Julia 1.10.10, RadiiPolynomial 0.11.10 and
IntervalArithmetic 1.0.12. The two numerical libraries are pinned by full git
revision, and a generated Julia manifest records the full dependency closure.
No Julia or interval dependency was added to the Rust workspace or solver core.

Artifacts include exact problem identity, lossless binary64 coefficients and
approximate inverse, recomputed interval bounds, finite/tail estimates,
existence and uniqueness radii, guarantee flags, explicit theorem checks,
source/environment digests and timings. Replay recomputes the proof without
Newton or an optimizer. The original success flag and reported bounds are not
trusted.

`make_campaign.py` creates a three-command campaign using existing
experiment-craft input/output bindings. Candidate production, verification and
replay run as three separate optimization-jobs attempts. The final replay uses
a strict success exit code; ordinary inconclusive verification is a valid
computation outcome. Existing command policy, cancellation, resource admission
and no-automatic-replay semantics remain authoritative.

## Files and integration

| Area | Delivered files/change |
| --- | --- |
| Implementation | `julia/src/CAPReference.jl`, `julia/bin/cap.jl` |
| Reproducible environment | `julia/Project.toml`, `julia/Manifest.toml` |
| Frozen scientific cases | `problems/scalar-cuberoot.json`, `problems/cahn-hilliard.json` |
| Versioned artifacts | `schemas/cap-artifacts-0.1.schema.json` |
| Campaign generation | `make_campaign.py`, copied/hashed application inputs and required output references |
| Regression and integration tests | `julia/test/runtests.jl`, `test_workflow.py` |
| User/scientific documentation | `README.md`, `THEOREM.md`, upstream MIT notice |
| Retained results | Both candidates, verification records, replays, Julia summaries and workflow observations under `evidence/` |
| CI | `.github/workflows/cap-reference.yml`, pinned Julia/runtime/client versions and real foreground campaigns |
| Discoverability | Root README, tutorial index, book summary, AI context and changelog |

`optimization-jobs`, `experiment-craft`, `LP-IP-craft` and `SQP-craft` required
no source modifications for this increment. Only fcmaes-rust needs a patch.

## Numerical results

Both independent replays succeeded at the fixed exact target `1/10000000000`.
The following values are decimal displays of the retained binary64 bounds.

| Reference | Y upper bound | Z1 upper bound | Z2 upper bound | Verified radius |
| --- | --- | --- | --- | --- |
| Scalar cube root | 9.325289058687924e-17 | 6.661338147750939e-16 | 0; first-order theorem | 1.8650578117375848e-16 |
| Cahn–Hilliard | 7.343308663301208e-14 | 0.01775750155037711 | 841.6053169211191 | 7.343308663301207e-13 |

The PDE result includes the analytic bound on unresolved Fourier modes and
the norm conversion to physical sup error. It is a local existence/uniqueness
result in the stated even function space, not a claim about stability,
parameter-region coverage, global uniqueness or blowup.

## Validation performed

| Check | Outcome |
| --- | --- |
| Pinned environment installation | Passed; Julia distribution SHA-256 verified before installation |
| Julia proof/encoding/replay suite | 68 assertions passed; 0 failures, 0 errors |
| Fresh-process workflow and schemas | Both references produced, verified and replayed successfully |
| Invalid certificate replay | Forged bound rejected with exit 3 |
| Immutable output behavior | Overwrite refused with exit 2 |
| Real command campaigns | Two campaigns completed, six actual runtime attempts, one declared CPU slot per attempt |
| Campaign continuation | Both completed campaigns remained at three attempts; no duplicate jobs |
| Workflow acceptance checks | 16 subprocess checks returned their expected exit status |
| Python syntax | Campaign generator and workflow harness compiled successfully |
| Documentation | Source-link/inventory check, pinned mdBook build and rendered-link check |
| Patch packaging | Final ZIP integrity and clean-base patch application checked before delivery |

The numerical regressions include lost guarantee despite a true library
Boolean, uncertain and strict-threshold inequalities, empty/NaI/unbounded
intervals, zero second-order constants, wrong precision, changed coefficients,
zero inverses, missing tails, malformed shapes, incorrect exact problem data,
changed bounds and environment fingerprints. A false Newton convergence flag
does not prevent a valid independent proof, and a true flag cannot create one.

The runtime was built with Rust 1.88.0 from optimization-jobs commit
`051435c325a51cc89be4ae26645d6937eed6f5e5`; the client was experiment-craft
`2f5846e4eb18008d960c30301a54395d58bdb32d`, under Python 3.12.14. Tests ran on
Linux x86-64 with one Julia compute thread and one BLAS thread. The workflow's
observed process times totalled approximately 192 seconds, including cold/cache
effects. This is a test observation, not a speed benchmark.

## Issues found and resolved

1. Exact rationals parsed as BigInt rationals initially promoted PDE interval
   calculations to BigFloat. The adapter now explicitly constructs binary64
   interval enclosures for the PDE constants. Serialization rejects non-binary64
   endpoints, preventing silent loss of enclosure precision.
2. Giving a projection an interval-valued Fourier frequency did not change
   the projection's coefficient type. Its default Float64 entries lost the
   interval guarantee flag. Both verification projections now explicitly use
   `Interval{Float64}`. The proof regressions caught this before acceptance.
3. The license Markdown link produced a missing rendered HTML page. The
   retained upstream notice is now a static `.txt` resource.

## Limits and following increment

Replay currently requires identical verifier/environment fingerprints and
identical recomputed proof records. A changed platform or implementation may
be conservatively rejected even if it could establish the same theorem.
Independent replay means independence from the numerical search and its cached
calculations; it uses the same pinned interval/theorem implementation, not a
second arithmetic library or proof assistant.

Validation used the real foreground backend on one node. It did not test
systemd service operation, remote execution, disconnection, reboot or multiple
machines. The dedicated GitHub Actions workflow is included but was not run on
GitHub before delivering the patch. The unchanged Rust optimizer workspace's
full regression suite was not rerun; the changed Julia/Python workflow and the
required real runtime were tested directly.

No optimizer comparison, persistent Julia search service, adaptive parameter
cover, generic verification API or proof-aware UI was added. Actual aggregate
CPU use and peak memory were not measured. These are later increments.

CAP-02 can now add the bounded Rust–Julia search bridge and frozen
grid/random/local/CMA-ES/BiteOpt comparisons while retaining this verifier and
replay boundary.
