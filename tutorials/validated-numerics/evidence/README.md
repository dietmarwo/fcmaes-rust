# CAP-01 retained validation evidence

Local validation date: 26 October 2026. Base fcmaes-rust commit:
`ed70f303b7ca7c568af786c6c814b1d13b7c06ef`.

These are actual computed reference artifacts, not mock certificates. The
candidate, verification and replay documents are retained byte-for-byte so
their digest links remain valid. Numeric intervals use binary64 hex words;
the table gives approximate decimal displays only.

| Reference | Verified existence/uniqueness radius | Target | Scope |
| --- | --- | --- | --- |
| Scalar cube root | 1.8650578117375848e-16 | 1e-10 | Positive root, local ball in the real line |
| Cahn–Hilliard | 7.343308663301207e-13 | 1e-10 | Full Neumann steady-state equation, including unresolved Fourier modes |

- [Scalar candidate](scalar-cuberoot/candidate.json),
  [verification](scalar-cuberoot/verification.json),
  [fresh replay](scalar-cuberoot/replay.json).
- [Cahn–Hilliard candidate](cahn-hilliard/candidate.json),
  [verification](cahn-hilliard/verification.json),
  [fresh replay](cahn-hilliard/replay.json).
- [Julia test summaries](julia-tests.txt): **68 assertions passed**.
- [Workflow outcomes](summary.json) and [process observations](commands.json):
  **16 successful acceptance checks**, including expected rejection exits.

Both references were produced, verified and replayed in separate Julia
processes. The workflow then ran each reference as a real three-step
experiment-craft campaign using optimization-jobs command execution:
**six runtime attempts**, all successful. Calling `continue` on each completed
campaign reserved no additional jobs. Forged proof replay returned exit 3;
attempting to overwrite a certificate returned exit 2.

The Julia suite additionally rejected corrupt coefficients, zero inverses,
missing tail evidence, changed bounds and environment digests, invalid/uncertain
intervals, and a true library success Boolean carrying a false guarantee flag.
It checks exact rational versus rounded input semantics and the symmetry
factor in the physical-norm bound.

Test environment: Linux x86-64, Julia 1.10.10, Julia and BLAS compute threads
set to one; full numerical dependency versions and hashes are in each artifact
and in the committed Julia manifest. The real runtime was built from
optimization-jobs `051435c325a51cc89be4ae26645d6937eed6f5e5` with Rust 1.88.0.
The campaign client was experiment-craft
`2f5846e4eb18008d960c30301a54395d58bdb32d` under Python 3.12.14.

Wall times include compilation/cache effects and are not comparative optimizer
benchmarks. Actual aggregate CPU and peak memory were not measured. The tests
used the real foreground backend on one machine; they do not establish
systemd, remote-host, disconnect/reboot or multi-node readiness. The new GitHub
Actions workflow was added but was not remotely run before patch delivery.

To replay a retained certificate from the tutorial directory, in the pinned
environment, use a new output path:

```bash
julia --startup-file=no --project=julia julia/bin/cap.jl replay \
  problems/cahn-hilliard.json evidence/cahn-hilliard/candidate.json \
  evidence/cahn-hilliard/verification.json /tmp/cap-retained-replay.json
```

An environment or implementation mismatch causes conservative rejection.
This is repeatable interval validation, not proof-assistant certification.
