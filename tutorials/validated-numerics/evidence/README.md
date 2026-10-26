# Retained validation evidence

These are actual computed reference artifacts, not mock certificates. They
were produced on 26 October 2026 by the
[validated-numerics workflow](https://github.com/dietmarwo/fcmaes-rust/actions/runs/40370339320)
on a GitHub-hosted `ubuntu-latest` runner (Linux x86-64), at commit
`d3db5c74bb8410a796d0f3b6beca7032a4da0194`. That commit's verifier, command
line, project and manifest are the ones in this directory; each artifact
records their SHA-256 digests. The candidate, verification and replay
documents are retained byte-for-byte, so their digest links stay valid.
Numeric intervals use binary64 hex words; the table gives approximate decimal
displays only.

| Reference | Verified existence/uniqueness radius | Target | Scope |
| --- | --- | --- | --- |
| Scalar cube root | 1.8650578117375848e-16 | 1e-10 | Positive root, local ball in the real line |
| Cahn–Hilliard | 7.34330866335485e-13 | 1e-10 | Full Neumann steady-state equation, including unresolved Fourier modes |

- [Scalar candidate](scalar-cuberoot/candidate.json),
  [verification](scalar-cuberoot/verification.json),
  [fresh replay](scalar-cuberoot/replay.json).
- [Cahn–Hilliard candidate](cahn-hilliard/candidate.json),
  [verification](cahn-hilliard/verification.json),
  [fresh replay](cahn-hilliard/replay.json).
- [Julia test summaries](julia-tests.txt): **81 assertions passed** in five
  test sets.
- [Workflow outcomes](summary.json) and [process observations](commands.json):
  **16 acceptance checks** returned their expected exit status, including
  expected rejection exits.

Both references were produced, verified and replayed in separate Julia
processes. The workflow then ran each reference as a real three-step
experiment-craft campaign using optimization-jobs command execution:
**six runtime attempts**, all successful. Calling `continue` on each completed
campaign reserved no additional jobs. Forged proof replay returned exit 3;
attempting to overwrite a certificate returned exit 2.

The Julia suite additionally covers:
- rejection of corrupt coefficients, zero inverses, missing tail evidence,
  changed bounds and environment digests, invalid or uncertain intervals, and
  a true library success Boolean carrying a false guarantee flag;
- refusal of round-to-nearest interval arithmetic, which still reports
  guaranteed results;
- the library behavior the PDE bound relies on: norm weights, truncation by
  the finite inverse, and every output row of the finite defect;
- a radius where self-mapping holds but contraction fails;
- the second solution `-ū`, which fails only the branch condition;
- exact rational versus rounded input semantics and the symmetry factor in
  the physical-norm bound.

Test environment: Julia 1.10.10 with one compute thread and one BLAS thread,
IntervalArithmetic's correct rounding and generic interval matrix product,
and the numerical dependency versions and hashes recorded in each artifact
and in the committed Julia manifest. The real runtime was built from
optimization-jobs `051435c325a51cc89be4ae26645d6937eed6f5e5` with Rust
1.88.0; the campaign client was experiment-craft
`2f5846e4eb18008d960c30301a54395d58bdb32d` under Python 3.12.

Wall times include compilation and cache effects and are not comparative
optimizer benchmarks; the 16 processes took about three minutes in total.
Actual aggregate CPU and peak memory were not measured. The tests used the
real foreground backend on one machine; they do not establish systemd,
remote-host, disconnect/reboot or multi-node readiness.

The workflow replays these two certificates on every change to this
directory, and [`exact_check.py`](../exact_check.py) recomputes their bounds
in exact rational arithmetic: every recorded interval encloses the exact value,
and every theorem condition holds exactly. For the Cahn–Hilliard case, the
exact Y is about 7.1e-16, so its recorded upper bound of 7.3e-14 is about 100
times wider, from interval cancellation in the residual; the radius
`R = 10 sup(Y)` is correspondingly conservative. To replay one yourself from the tutorial directory, in the pinned
environment, use a new output path:

```bash
julia --startup-file=no --project=julia julia/bin/cap.jl replay \
  problems/cahn-hilliard.json evidence/cahn-hilliard/candidate.json \
  evidence/cahn-hilliard/verification.json /tmp/cap-retained-replay.json
```

An environment or implementation mismatch causes conservative rejection.
This is repeatable interval validation, not proof-assistant certification.
After a change to the verifier, its command line, project or manifest,
produce the evidence again with
`test_workflow.py --skip-retained` and replace these files.

## Development notes

Problems found while the verifier was written:

1. Exact rationals parsed as `BigInt` rationals promoted the PDE interval
   calculations to `BigFloat`. The verifier now constructs binary64 interval
   enclosures for the PDE constants explicitly, and serialization rejects
   non-binary64 endpoints instead of rounding them.
2. Giving a projection an interval-valued Fourier frequency did not change
   the projection's coefficient type. Its default `Float64` entries lost the
   interval guarantee flag. Both verification projections now use
   `Interval{Float64}` explicitly; the proof regressions caught this.
3. A Markdown link to the upstream license produced a missing page in the
   rendered book; the notice is now a plain `.txt` file.
