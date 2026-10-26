# Validated numerics: reference verifier and replay

This pilot reproduces two RadiiPolynomial.jl reference validations: the real
root of `u^3 - 2 = 0`, and a nonconstant Neumann steady state of the
Cahn–Hilliard equation. It separates numerical approximation, rigorous
verification and a fresh replay. The [bounded search](SEARCH.md) adds a
Rust–Julia bridge and frozen comparisons for proof-parameter tuning.
This page describes the independent reference verifier.

The implementation is a standalone Julia environment. It adds no dependency
to `fcmaes-core`, changes no shared job API and needs no access to a cluster.
One CPU slot is the default. Verification includes the infinite Fourier tail;
it does not infer a differential-equation proof from a small finite residual.

## Exact scope

| Reference | Mathematical conclusion after a successful replay |
| --- | --- |
| `scalar-cuberoot` | A positive real root exists and is unique in the reported ball about the saved binary64 approximation. |
| `cahn-hilliard` | A nonconstant, even, 2-periodic solution of `u'' + (1285/32)(u-u^3) = 0` exists and is locally unique in the reported coefficient-norm ball. Restricted to `[0,1]`, it satisfies Neumann conditions. The first cosine mode is positive. |

Both fixtures require a physical sup-norm error at most the exact rational
`1/10000000000`. The PDE uses order 32 and the norm
`abs(a0) + 2*sum(abs(ak), k>=1)`. The reported uniqueness is local, in the
stated even function space. It does not assert stability, a complete branch,
parameter-region coverage, global uniqueness or blowup.

Read [THEOREM.md](THEOREM.md) for the hypotheses, tail argument, norm conversion
and trusted components. The two problem JSON files are frozen specifications;
editing an equation, constant, representation or target is rejected. General
problem loading and tuning are later work.

## Install the pinned environment

Use **Julia 1.10.10**, RadiiPolynomial **0.11.10** at commit
`704009d7aa2e621c0900c9c158e853cb84c0c253`, and IntervalArithmetic **1.0.12** at
`56701d7324292eb0de5edfe639bfa59d2e7b19b3`. The committed
[Project.toml](julia/Project.toml) and [Manifest.toml](julia/Manifest.toml)
also pin transitive dependencies. Do not run `Pkg.update()` for a reproduction.

From this tutorial directory, with Julia 1.10.10 on `PATH`:

```bash
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_PKG_USE_CLI_GIT=true
export JULIA_DEPOT_PATH="$PWD/.julia-depot"
julia --startup-file=no --project=julia -e 'using Pkg; Pkg.instantiate()'
```

The depot must be instantiated before offline command jobs. Download Julia
from its [official distribution](https://julialang.org/downloads/); the tested
Linux x86-64 1.10.10 archive has SHA-256
`6a78a03a71c7ab792e8673dc5cedb918e037f081ceb58b50971dfb7c64c5bf81`.

## Produce, verify and replay

```bash
mkdir -p results/scalar
julia --startup-file=no --project=julia julia/bin/cap.jl candidate \
  problems/scalar-cuberoot.json results/scalar/candidate.json
julia --startup-file=no --project=julia julia/bin/cap.jl verify \
  problems/scalar-cuberoot.json results/scalar/candidate.json results/scalar/verification.json
julia --startup-file=no --project=julia julia/bin/cap.jl replay \
  problems/scalar-cuberoot.json results/scalar/candidate.json \
  results/scalar/verification.json results/scalar/replay.json
```

For the PDE, use `problems/cahn-hilliard.json` and a new output directory.
Existing output files are never overwritten. Each invocation above is a
separate process; replay does not call Newton or any optimizer. A candidate
stores both the coefficient vector and the approximate inverse as exact
binary64 bit patterns. Both are untrusted numerical proposals whose validity
must survive recomputation of the rigorous bounds.

| Command | Exit semantics |
| --- | --- |
| `candidate` | `0` writes a converged numerical reference candidate; `2` reports malformed input, an unsupported environment or an approximation error. |
| `verify` | `0` writes a valid computation outcome, either `verified` or `inconclusive`; `2` reports invalid input or an operational error. Read the outcome. |
| `replay` | `0` only if the saved claim matches a freshly verified proof; `3` writes a rejected replay; `2` reports invalid input or an operational error. This is the final acceptance gate. |

An inconclusive test is not evidence of nonexistence. A command's successful
exit is not, in general, proof success. The final replay command has deliberately
stricter exit semantics so today's command-only orchestrator can gate acceptance.

The records use [the versioned artifact schema](schemas/cap-artifacts-0.1.schema.json).
`encoding: ieee754-binary64-hex` means each 16-character word is the numeric
UInt64 bit representation of a Float64, not a platform-endian byte dump.
Interval endpoints, radius and candidate entries are lossless. Mathematical
constants enter as exact rationals or `interval(pi)`, not rounded decimal
approximations. The encoder rejects BigFloat intervals rather than narrowing
their endpoints. All proof inequalities use guaranteed interval arithmetic.

The verifier checks both the radii routine's Boolean and guarantee flag,
interval admissibility, an explicit self-mapping inequality, strict contraction,
the physical error target, inverse injectivity and the branch condition. The
PDE additionally retains its finite defect and analytic tail bound. It refuses
to run unless the interval arithmetic is in the configuration it records and
the loaded libraries are the source trees the manifest pins; `cap.jl` also
requires `--startup-file=no` (see [THEOREM.md](THEOREM.md#arithmetic-and-replay)).

`elapsed_seconds` is measured inside the operation and is not whole-process
startup time or CPU use. The workflow harness separately records process wall
time. These are reproduction observations, not optimizer benchmarks.

## Use existing experiment-craft command jobs

Install an `optimization-run` that supports `execution.command/0.1`, and
experiment-craft 0.4.0. Integration was developed against runtime commit
`051435c325a51cc89be4ae26645d6937eed6f5e5` and experiment-craft commit
`2f5846e4eb18008d960c30301a54395d58bdb32d`.

```bash
python3 make_campaign.py --problem cahn-hilliard \
  --julia "$(command -v julia)" --depot "$JULIA_DEPOT_PATH" \
  --destination /tmp/cap-ch-spec
optimization-run --root /tmp/cap-node node init --capacity 1 --allow-commands
experiment-craft dry-run /tmp/cap-ch-spec/campaign.json
experiment-craft run /tmp/cap-ch-spec/campaign.json \
  --workspace /tmp/cap-ch-campaign --runner "$(command -v optimization-run)" \
  --node-root /tmp/cap-node --mode foreground
```

Use fresh paths. `make_campaign.py` only writes a plan and copied inputs; it
never launches work. The plan stages the complete Julia application source,
project, manifest and exact problem. Required, SHA-checked outputs bind
candidate → verification → replay. Each command reserves one slot, permits
600 seconds plus five seconds of termination grace, and uses `replay: never`.
The executable path and depot are local deployment choices; the depot and Julia
installation are not copied by the runner.

After completion, `experiment-craft continue --workspace /tmp/cap-ch-campaign`
observes the existing attempts. It does not repeat the scientific replay or
launch another job. Starting a fresh replay means creating a new explicit
attempt/campaign. Use runtime output collection and the campaign's records to
retain sealed artifacts; do not rely on mutable worker files.

The existing dashboard can control and observe these three steps. It does not
yet display theorem-aware metrics, compare certification times or route Python
campaigns to remote nodes. No automatic placement or background service is
introduced here; the real integration test uses foreground execution.

## Tests

```bash
julia --startup-file=no --project=julia -e 'using Pkg; Pkg.test(;allow_reresolve=false)'
python3 -m pip install 'jsonschema>=4.18,<5'
python3 test_workflow.py --julia "$(command -v julia)" \
  --depot "$JULIA_DEPOT_PATH" --output /tmp/cap-direct-evidence
```

`exact_check.py` needs only Python. It recomputes every bound of the two
retained certificates in exact rational arithmetic, independently of the Julia
interval libraries, and requires each recorded interval to enclose the exact
value and each theorem condition to hold exactly:

```bash
python3 exact_check.py
```

To exercise the actual command runtime and campaign client too:

```bash
python3 test_workflow.py --julia "$(command -v julia)" --depot "$JULIA_DEPOT_PATH" \
  --runner /absolute/path/to/optimization-run \
  --experiment-craft-source /absolute/path/to/experiment-craft \
  --output /tmp/cap-campaign-evidence
```

The Julia suite covers exact input encoding, the false-guarantee/true-success
case, uncertain and strict-boundary inequalities, zero second-order constant,
invalid intervals, wrong precision, both reference proofs, corrupt coefficients,
zero inverses, missing tails, altered bounds, provenance mismatch and overwrite
refusal. The Python harness validates the JSON schemas, runs fresh processes,
checks the strict replay exit code, replays the two retained certificates in
[`evidence/`](evidence/README.md), and optionally runs both real three-job
campaigns plus continuation without duplicate attempts.

The CI workflow (`.github/workflows/cap-reference.yml` at the repository root)
runs these checks with pinned commits on every change to this directory.
[Retained evidence](evidence/README.md) records the run that produced the
committed certificates. A change to the verifier, its command line, project or
manifest changes the recorded digests, so the retained certificates no longer
replay: run `test_workflow.py` once with `--skip-retained` and replace them
with the new outputs.

## Replay boundaries and next steps

Replay conservatively requires the same recorded verifier/environment hashes
and identical recomputed bound records. It may reject a mathematically valid
proof from a different platform or changed implementation. Portable semantic
replay is later work. Hashes detect changed inputs; they are not signatures or
proof-assistant certificates. The trusted components are documented separately.

[The bounded search](SEARCH.md) supplies a persistent Rust–Julia bridge,
deterministic estimated scores and grid/random/local/CMA-ES/BiteOpt baselines.
The candidate and verification separation remains. Later work covers generic
verification result contracts, certification-time comparisons, parameter-region
coverage and multi-node experiments.

The reference formulas and pseudo-inverse operator are adapted from
[RadiiPolynomial's scalar example](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/704009d7aa2e621c0900c9c158e853cb84c0c253/docs/src/getting_started/first_validation.md)
and [Cahn–Hilliard example](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/704009d7aa2e621c0900c9c158e853cb84c0c253/docs/src/examples/steady_states/cahn_hilliard.md).
Its [MIT notice](RADII_POLYNOMIAL_LICENSE.txt) is retained.
