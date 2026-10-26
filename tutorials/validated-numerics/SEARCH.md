# Bounded proof-parameter search

This application connects fcmaes-core to a persistent Julia evaluator. It compares a
frozen set of search methods, exports the selected numerical candidate and
runs the existing rigorous verifier and replay in separate processes. The
two exact equations, accuracy target and verifier remain those of the
[reference stage](./README.md).

## What is tuned

The coefficient vector comes from the reference Newton solve. The search
adjusts the finite approximate inverse supplied to the verifier:

| Problem | Parameters | Admissible range |
| --- | --- | --- |
| Scalar cube root | One multiplier on the scalar inverse | `[0.5, 1.5]` |
| Cahn–Hilliard | One multiplier on the zero-mode row; one shared multiplier on the nonzero rows | `[0.5, 1.5]` for each |

Each evaluation starts from the same immutable coefficients and inverse.
There is no continuation from the previously evaluated point. The spectral
order stays 32, the exact physical target stays `1/10000000000`, and the
infinite inverse-Laplacian tail is preserved. Multiplying rows produces a
new approximate inverse; its exact binary64 entries are recorded and its
injectivity/defect must still pass the rigorous verifier.

This is a small calibration study of structured preconditioner tuning. It
establishes the search/verification interface and measurement rules. These
easy reference cases and two stochastic seeds do not establish a held-out
optimizer advantage or a complete parameter-region proof.

## Frozen comparison and score

[search-protocol.json](search-protocol.json) fixes nine arms: reference,
complete grid, local coordinate pattern search, and random/CMA-ES/BiteOpt
with seeds 17 and 29. Every arm sees the nominal all-ones point first.
Strictly lower scores replace the incumbent; ties retain the first result.

For search only, Julia computes floating estimates of `Y`, `Z1`, `Z2` and
scores a point at the fixed physical epsilon with

$$
\max\left(\frac{Y+(Z_1-1)\varepsilon+\tfrac12Z_2\varepsilon^2}{\varepsilon},
Z_1+Z_2\varepsilon-1\right).
$$

The scalar estimate uses the first-order whole-ball derivative bound and
zero `Z2`. A lower estimated score means a larger estimated safety margin.
It does not certify the inequalities. Measured wall time is recorded
separately and is never the deterministic optimization objective.

The maximum search budget is 25 evaluations per arm. The complete grid has
25 points in one dimension or `5 × 5` points in two. The fixed reference
uses one evaluation; the other arms use 25. Every winner is independently
re-evaluated once before export. The one initial estimate that compiles and
warms the Julia model is counted too. For the default protocol this gives
**201 search evaluations + 9 rechecks + 1 warm-up = 211 estimates per study**.
The Newton construction cost is a separate shared startup cost, recorded once.

CMA-ES uses population four and sigma 0.2; BiteOpt uses population eight and
serial ask/tell batches. CMA-ES never evaluates beyond the hard cap; a final
partial population is recorded without padding it with invented values.
The CMA-ES arm uses fixed-work ask/tell: tolerance flags do not truncate its
externally declared evaluation count. This is part of the calibration policy.
The local pattern starts at step 0.25 and halves after a non-improving cycle.
The executable digest, core version, settings and protocol digest are retained.

## Run locally

Instantiate the pinned Julia environment as described in [README.md](./README.md).
The Rust application is its own nested workspace, keeping process/protocol
dependencies outside fcmaes-core. From this tutorial directory:

```bash
cargo build --release --locked --manifest-path search/Cargo.toml
search/target/release/cap-search \
  --julia "$(command -v julia)" --project julia --server julia/bin/search_server.jl \
  --problem problems/cahn-hilliard.json --protocol search-protocol.json \
  --output /tmp/cap-search-ch --wall-seconds 300
```

Set `JULIA_DEPOT_PATH` to the instantiated depot. The program sets one Julia
compute thread and one BLAS/OpenMP thread. An output directory may be new or
empty (the command runtime creates it before launch). A nonempty directory
and existing artifact filenames are refused.

Outputs:

| File | Meaning |
| --- | --- |
| `study.json` | Estimated scores, method settings, source/protocol/executable identities, work counts, costs, winner and termination |
| `evaluations.jsonl` | Flushed start/finish records for every search and recheck call; an unfinished call remains visible on failure |
| `candidate.json` | Selected numerical proposal in the existing `cap.candidate/0.1` format |
| `candidates.json` | All nine proposals as exact JSON strings plus digests, permitting later extraction without changing their bytes |

All arm proposals use the reference Newton coefficient vector; the explicit
approximate inverse may differ. The candidate's approximation tag describes
that coefficient construction. The search study records the inverse scaling
and selection method. Only the selected proposal is verified by the example
pipeline; the other proposals are retained for inspection or explicit later
verification.

Run the independent verifier and replay:

```bash
julia --startup-file=no --project=julia julia/bin/cap.jl verify \
  problems/cahn-hilliard.json /tmp/cap-search-ch/candidate.json /tmp/cap-search-ch/verification.json
julia --startup-file=no --project=julia julia/bin/cap.jl replay \
  problems/cahn-hilliard.json /tmp/cap-search-ch/candidate.json \
  /tmp/cap-search-ch/verification.json /tmp/cap-search-ch/replay.json
```

The search summary explicitly sets `certification_performed: false`.
Verification and replay provide the mathematical conclusions, with their own
candidate hashes and outcomes. A search score, even a negative one, does not
strengthen an execution result into a proof.

## Run as three command jobs

```bash
python3 make_search_campaign.py --problem cahn-hilliard \
  --binary search/target/release/cap-search --julia "$(command -v julia)" \
  --depot "$JULIA_DEPOT_PATH" --destination /tmp/cap-search-spec
optimization-run --root /tmp/cap-search-node node init --capacity 1 --allow-commands
experiment-craft run /tmp/cap-search-spec/campaign.json \
  --workspace /tmp/cap-search-workspace --runner "$(command -v optimization-run)" \
  --node-root /tmp/cap-search-node --mode foreground
```

Use fresh campaign/node/spec paths. The generator stages hashed Julia source,
the exact problem and frozen protocol, then binds the sealed candidate to
verification and replay. Search, verification and replay are separate admitted
attempts. The runtime owns the node slot and the process group; the bridge
owns its single Julia child inside the search attempt. Calling campaign
`continue` observes existing attempts and does not rerun search or replay.

The workflow tests the generated campaigns with optimization-jobs commit
`051435c325a51cc89be4ae26645d6937eed6f5e5` and experiment-craft
`013077b37e917d994d3346aa0c1a36c5dfe6fec2`.

## Lifecycle and accounting

The protocol uses sequential JSON lines with schema versions, increasing
request IDs, echoed parameters and source hashes. Responses are bounded to
1 MiB, the receive queue to one frame, and retained stderr to 64 KiB.
Messages have fixed operations; no expression strings or code evaluation are
accepted. RPC handlers start no additional scientific workers. Julia startup
helpers inherit the command runtime's process group.

The wall deadline covers startup, all methods, rechecks and exports. SIGINT,
SIGTERM and SIGHUP request cancellation; polling notices it while waiting for
Julia. A Ctrl-C to the whole process group also ends Julia; that is reported
as a cancellation, not a server failure. Both normal shutdown and failure kill
and reap the Julia child. Cleanup waits at most one second for its pipes to
close: descendants that inherit them, such as Julia precompilation workers, are
not the bridge's to kill. They stay in the process group, which the command
runtime ends with the attempt; a standalone run leaves them to finish. Exit codes are 0 for
a completed numerical study, 124 for timeout, 130 for cancellation and 2 for
input/protocol/operational errors. All expected failures retain the partial
ledger and terminal study record once the output directory has been initialized.

Startup failure has explicitly incomplete work accounting. An evaluation is
attempted once its request has been sent to Julia; one sent without a response
remains attempted, with no invented completion or score. Wall costs include those failures. Startup and warm per-arm times are
observations; aggregate CPU and peak memory are not currently measured. Node
slot reservation is cooperative accounting rather than CPU affinity.

## Tests and evidence

```bash
cargo fmt --manifest-path search/Cargo.toml -- --check
cargo clippy --locked --manifest-path search/Cargo.toml --all-targets -- -D warnings
cargo test --release --locked --manifest-path search/Cargo.toml
julia --startup-file=no --project=julia -e 'using Pkg; Pkg.test(;allow_reresolve=false)'
python3 test_search.py --binary search/target/release/cap-search \
  --julia "$(command -v julia)" --depot "$JULIA_DEPOT_PATH" --output /tmp/cap-search-checks
```

Add `--runner /absolute/optimization-run --experiment-craft-source /absolute/experiment-craft`
to test both generated campaigns against the real foreground backend. The CI
workflow (`.github/workflows/cap-reference.yml`) runs the reference and search
checks on every change to this directory. Tests cover exact
work caps, grid completeness, seeded trajectory reproduction, immutable Julia
model data, honest score rechecks, oversized/truncated responses, child failure,
wall timeout, SIGTERM cleanup, candidate digests, independent proof replay and
exact rational recomputation of the selected bounds.

The workflow also checks the committed search evidence: every digest link
between study, ledger, candidate bundle, selected candidate and certificates;
that it used the current protocol; an exact rational recheck of both selected
certificates (`exact_check.py evidence/search`); and a fresh replay. A change
that invalidates it needs new evidence: run `test_search.py` once with
`--skip-retained` and replace the files from its output.

[Search evidence](evidence/search/README.md) records measured outcomes and scope.
The previous reference evidence is preserved. The next scientific increment
needs harder frozen/held-out cases, more repetitions and per-arm independent
certification/time-to-certificate analysis. Native proof-aware comparisons,
generic verification-result contracts, adaptive region coverage and remote
campaign routing remain separate platform/research increments.
