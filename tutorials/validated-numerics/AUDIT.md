# Certify every final search candidate

The [bounded search](SEARCH.md) exports one candidate per predeclared arm.
`audit.py` now checks every one with the unchanged rigorous verifier and a
fresh replay. Its collector records unsuccessful attempts; a separate strict
acceptance command requires every arm to pass. The exact equations, physical
target, theorem, Julia environment and search protocol are unchanged.

This application uses existing command artifacts. It does not add a native
verification-result contract, proof-aware dashboard or generic scientific
comparison adapter to experiment-craft. Its native campaign report still
describes command execution. The application's acceptance artifact carries the
scientific conclusion.

## Run locally

Instantiate the Julia environment as in [README.md](./README.md), install
`jsonschema>=4.18,<5` in Python 3.12, and complete the search from
[SEARCH.md](SEARCH.md). From this tutorial directory:

```bash
python3 audit.py run \
  --julia "$(command -v julia)" --project julia \
  --problem problems/cahn-hilliard.json \
  --study /tmp/cap-search-ch/study.json \
  --candidates /tmp/cap-search-ch/candidates.json \
  --evaluations /tmp/cap-search-ch/evaluations.jsonl \
  --search-protocol search-protocol.json --protocol audit-protocol.json \
  --output /tmp/cap-audit-ch
python3 audit.py accept \
  --project julia --problem problems/cahn-hilliard.json \
  --study /tmp/cap-search-ch/study.json \
  --candidates /tmp/cap-search-ch/candidates.json \
  --evaluations /tmp/cap-search-ch/evaluations.jsonl \
  --search-protocol search-protocol.json --protocol audit-protocol.json \
  --audit /tmp/cap-audit-ch/audit.json --events /tmp/cap-audit-ch/events.jsonl \
  --output /tmp/cap-audit-acceptance-ch.json
```

The output directory must be new or empty; existing results are refused.
There is no resume or automatic retry. A fresh study uses fresh paths.

The [audit protocol](audit-protocol.json) fixes arm order, acceptance, timing
and separate 120-second verification/replay limits. A changed budget is a new
hashed protocol input, not a silent adjustment to a recorded study. Each arm
starts two fresh Julia processes, including when its candidate matches
another arm's. No certificate cache substitutes for that work.

| Output | Meaning |
| --- | --- |
| `audit.json` | All arms, per-stage wall costs and limits, runtime outcomes, bounded logs, and exact verification/replay JSON strings with SHA-256 |
| `events.jsonl` | Flushed start/finish records; an interrupted attempt remains visible |
| `report.md` | Human-readable all-arm table with timing scope and limitations |
| Acceptance output | Hash-bound accepted/rejected result and an event/censored observation for every predeclared arm |

The records use [cap-audit-0.1.schema.json](schemas/cap-audit-0.1.schema.json).
They are application records. Existing candidate, verification, replay,
search, job and campaign schemas retain their meanings and versions.

## Acceptance and failure semantics

Before launching Julia, the collector checks the frozen problem and protocol,
current source/environment identities, the full search ledger, per-arm work
caps, incumbent/recheck agreement and every candidate digest. It preserves
candidate bytes when extracting them from the bundle. Verification and replay
always recompute from those bytes; they never consume floating search bounds.

The collector uses the existing `cap.jl` CLI. Verification can complete with
`inconclusive`; replay can complete with `rejected`. Those are retained valid
scientific outcomes. A timeout, process failure, invalid/missing output or
interruption has no trusted proof document. Replay is unattempted when
verification has no usable document; it still runs after an inconclusive
verification. No failed arm is dropped.

`accept` rechecks every input and stage link, the exact embedded document
digests, proof/check consistency, certificate-to-replay binding and ledger
accounting. It accepts only a complete audit with every verification and
replay verified. Saved numerical records remain subject to the verifier's
trust assumptions; hashes and structural checks do not themselves prove a
theorem. The test suite also recomputes all accepted certificates in exact
rational arithmetic and runs fresh scientific computations.

`accept` inspects recorded evidence; it does not rerun Julia. The generated
campaign binds that evidence to the admitted collector through sealed native
outputs. For new evidence from an untrusted source, run the audit afresh with
the trusted verifier before using its acceptance result.

| Command | Exit semantics |
| --- | --- |
| `run` | `0`: collection completed, including unsuccessful numerical/operational results; `130`: cancelled, with partial records; `2`: invalid input or coordinator error |
| `accept` | `0`: all arms accepted; `3`: valid audit scientifically rejected or incomplete; `2`: malformed, stale, inconsistent or mismatched evidence |

An inconclusive proof attempt is not nonexistence. An accepted collection of
point certificates is not a parameter-family or branch-coverage certificate.

The collector owns at most one Julia child. Its child inherits the command
runtime's process group and one-slot reservation. SIGINT, SIGTERM and SIGHUP
stop collection, kill/reap that child and retain all unattempted arms. Logs go
to regular temporary files, avoiding a wait for pipe-reader threads after a
descendant inherits stdout. Retained stdout/stderr are capped at 64 KiB each;
individual scientific documents are capped at 1 MiB. Temporary files are
removed after each arm. The runtime owns whole-process-group cleanup,
admission, staging, collection and recovery. Standalone Julia startup helpers
can outlive the direct child until they finish; the application does not own a
second process-group supervisor.

## Timing and censored observations

For each final candidate, accounted seconds are:

`shared search startup + arm warm search/recheck/export + fresh verification + fresh replay`

The shared startup is charged in full to each arm, never divided by the number
of arms. Verification/replay include Julia process startup and their failed
work. Stages retain their actual durations, individual budgets and outcomes.
The complete collection wall time is a separate field. The timing excludes
other arms' work, scheduling, transfers, depot installation and coordinator
overhead. It is a composite serial cost, not the elapsed position of that arm
in a shared campaign. Fixed arm order and compilation effects also limit
comparisons.

An independently accepted final candidate is an event. Otherwise the
observation is right censored at that arm's observed pipeline end, with the
reason retained. Unattempted arms remain censored observations with their
already completed search cost. Stopping and failures can be informative;
these records do not justify a Kaplan–Meier estimate or a population survival
median. The report gives the full denominator and individual observations,
without a successful-only median or a speed ranking.

This is **not time to the first certificate found during search**: intermediate
proposals are not certified. An anytime study needs a predeclared certification
schedule, an explicit stopping rule and an equal resource policy. Two easy
cases and two stochastic seeds still do not establish an optimizer advantage.
Actual process-tree CPU and peak memory remain unavailable; one reserved slot
is cooperative admission accounting, not a measurement or CPU affinity.

## Existing command runtime and campaign client

```bash
python3 make_audit_campaign.py --problem cahn-hilliard \
  --binary search/target/release/cap-search --julia "$(command -v julia)" \
  --depot "$JULIA_DEPOT_PATH" --destination /tmp/cap-audit-spec
optimization-run --root /tmp/cap-audit-node node init --capacity 1 --allow-commands
experiment-craft run /tmp/cap-audit-spec/campaign.json \
  --workspace /tmp/cap-audit-workspace --runner "$(command -v optimization-run)" \
  --node-root /tmp/cap-audit-node --mode foreground
```

The generator creates three static, separately admitted command jobs:
search → all-arm collection → strict acceptance. The collector serializes its
18 Julia processes inside one admitted attempt. Source, schemas, problem,
protocol and all derived artifacts are hash-bound through existing required
outputs. The default collection wall limit is 2,220 seconds
(nine times two 120-second budgets plus 60 seconds of coordinator allowance).
Its one-slot reservation includes the five-second grace: 2,225 reserved-slot
seconds, not measured CPU consumption. Python must provide jsonschema; the
generator uses its interpreter unless `--python` supplies another executable.

The original reference and selected-candidate campaign generators remain
available. The new generator neither launches work nor moves baseline pins.
Calling campaign `continue` observes the three recorded attempts without
resubmission. The tests use optimization-jobs `051435c` and experiment-craft
`d4db33d`, in local foreground mode with capacity one. No remote routing or
January hardware commissioning is established by these results.

## Checks and next work

```bash
python3 test_audit_contract.py
python3 test_audit.py --search-evidence /tmp/cap-search-checks \
  --julia "$(command -v julia)" --depot "$JULIA_DEPOT_PATH" \
  --output /tmp/cap-audit-checks
```

`/tmp/cap-search-checks` is a successful `test_search.py` output. Add
`--binary /absolute/cap-search --runner /absolute/optimization-run
--experiment-craft-source /absolute/experiment-craft` for real campaign tests.
The suite checks each arm independently, exact rational bounds, failures,
cancellation of the collector and its group, strict acceptance, retained
evidence and continuation. Only use `--skip-retained` when producing new
[audit evidence](evidence/audit/README.md); preserve earlier evidence directories.

Next scientific work is harder frozen/held-out cases, more independent
repetitions and an anytime certification policy. Native verification
attachments/acceptance in the shared platform, proof-aware comparisons/UI,
semantic cross-environment replay, exact region coverage and multi-node
commissioning remain separate increments. This audit supplies concrete
application records and failure cases for that platform design.
