# Implementation plan: shared agent job API

## 1. Recover and review the three contracts

Plan: preserve tested numerical engines; inspect the fcmaes lifecycle, strict
payload validation, durable journal and process-group handling. Identify where
LP/MIP and SQP need different model, budget and termination semantics.

Implemented: based on fcmaes `c2703e7f0de28d8c980f7f95144d48753d35dc7e`,
LP-IP 0.28.0 `af9a9a4ff230add635360cd18912da0e864aebe8`, and SQP 0.16.0
`7bc997fed286765c1ba2597f13a2c59694a91a31`. Previous craft evidence is retained.
No numerical-core code changes are part of this job-API integration.

Gate: compile against all three existing APIs, retain legacy schemas and run
the original fcmaes integration suite unchanged.

## 2. Extract neutral contracts and generalize the runner

Plan: version the common envelope; keep solver-specific payloads; reserve one
explicit total CPU budget; stage immutable input/executable content; preserve
idempotency, cancellation, deadlines and conservative recovery.

Implemented: identical shared crates in each repository, worker registry,
versioned capability discovery and validation, SHA-256 model staging, results
and lifecycle journals, process-group observations, no implicit retry/resume.
The original fcmaes facade and CLI delegate to the extracted implementation.

Gate: strict schema/resource tests, content hashes, duplicate/corrupt journal
rejection, false-completion rejection, atomic storage and process tests.

## 3. Implement three native adapters

Plan: use the same SDK and lifecycle while retaining each numerical meaning.

Implemented: fcmaes seeded BiteOpt/DE retries; LP/MIP CSR model artifacts with
integer kinds, warm starts and node/pivot budgets; SQP registered quadratic,
simplex-constrained and Rosenbrock models with both QP backends. Recompute
candidate checks and KKT values. Keep validation calls separate from solver work.
LP hidden component/frontier/residual pools are explicitly serial, preventing
nested oversubscription of the declared inner budget.

Gate: known LP/MIP optima, infeasible/unbounded LPs, objective/bounds checks,
SQP derivative checks and first-order acceptance, fcmaes hard/soft work bounds.

## 4. Test concurrency and interruption end to end

Plan: run actual executables with one/eight slots; mix worker registrations on
one node; exercise queued cancellation, hard deadlines, abrupt launcher loss,
live-worker repair refusal and interrupted-result accounting.

Implemented: a reusable stdlib Python harness, native contract tests and the
existing numerical-core suites. Slow protocol fixtures make liveness and forced
termination tests deterministic; actual solver workers supply numerical tests.
The LP bound assertion explicitly permits its conservative outward rounding.
Executing the fcmaes tutorial exposed a lossy JSON float parse that could reject
a completed manifest. Exact f64 JSON round trips are now enabled, with a
regression test covering tiny values, large exponents and digest stability.

Gate: all selected tests, release builds, formatting and Clippy. A lack of a
systemd user manager is documented; it is not reported as a passed detached test.

## 5. Document, reproduce and package each repository

Plan: keep one protocol guide, add a separate runnable tutorial per solver and
make every ZIP build independently with its lockfile and licenses.

Implemented: README/API links, shared guide, specific JSON examples and scripts,
extension instructions, test logs, source provenance, change patches and SHA-256
archive manifests. Existing craft release evidence is included. Build targets,
private node directories and downloaded toolchains are excluded.

Gate: execute all three tutorial scripts; verify identical shared crate content,
archive integrity and every manifest entry before delivery.

## 6. Result contract 0.2 (October 2026)

Plan: a survey of existing optimization job services, agent interfaces and
result vocabularies found the job lifecycle common, but no job API that
returns typed claims with independent re-checks. Adopt what that prior art
does well: infeasibility, unboundedness and certified claim levels
(MathOptInterface, OR-Tools MathOpt), an explicit evidence tier with a
certificate slot (VIPR, LRAT, VeriPB), the solver competitions' wrong-answer
verdicts, primal and dual objective bounds, COCO's multi-objective
conventions, evaluation of given points (Timefold's score-only runs), coded
errors and a conflict on cancelling a finished job (D-Wave's 409). Publish the
envelope and the result as JSON Schemas, and record how MCP Tasks would carry
them.

Implemented: result schema `optimization.result/0.2` with five new claims,
`evidence` (level, checks, refutation, certificate), `objective_bounds` and
`imprecise`; `Outcome::check` applied to every journal record; worker
protocol 0.2, with 0.1 workers still accepted; runner error codes with a
`retryable` flag; `already_terminal` for cancel and repair of finished jobs;
JSON Schemas for the job envelope 0.1 and the result 0.2; evidence and
refutations in all three workers; `fcmaes.evaluate/1`, `lp-ip.evaluate/1` and
`sqp.evaluate/1`.

Gate: shared outcome cases checked by the Rust validator, the JSON Schema and
the runner (as journal records); contract tests of every claim rule, 0.1
compatibility and the schema vocabulary; worker tests that force each
refutation with synthetic solver results; the shared harness on a mixed node
with all three workers; tutorial results validated against the schemas.

An independent review before the merge found that LP and SQP evaluation
failed whole jobs on a point whose values overflow, that the validation
output cap bounded the advertised point counts, that the schema and the Rust
validator disagreed on a rejected certificate and on null fields, that some
evidence claimed more than was checked (SQP `locally_infeasible`, refutations
without a check), that repair returned a retryable code where waiting could
never help, and several documentation errors. All were fixed; a differential
fuzz of 1.2 million generated outcomes then found no disagreement between
the schema and `Outcome::check`.

## 7. Application models, parallel populations and more algorithms (October 2026)

Plan: close three gaps of the fcmaes worker that limited the job API to its
benchmarks. Let an application register its own models, with parameters and
hashed data inputs, instead of editing the worker. Use `inner_threads` for the
population-based optimizers, without making results depend on it. Put the
remaining fcmaes-core algorithms behind the profiles. Do not add an ask/tell
or callback profile: the objective must stay in the worker, where answers are
re-checked and runs reproduced (see the fcmaes guide to application models).

Implemented in fcmaes-rust: a model registry (`Registry`, `Instance`) in the
fcmaes worker, with the built-in benchmarks registered through it and an
example application worker; per-task evaluation pools for CMA-ES, CR-FM-NES,
PGPE, MODE and MAP-Elites; CR-FM-NES and PGPE retries; IPOP and BIPOP restarts
for CMA-ES; CVT archives and CMA-ME and CMA-MAE emitter phases with exact
budgets for MAP-Elites; a deterministic CVT construction in fcmaes-core. Jobs
on the built-in models resolve to the same payloads and give the same results
as before. The shared crates are unchanged; this guide, this plan and the
shared harness describe and test the new fcmaes behavior.

Gate: every population-based run identical with one and several inner threads;
CR-FM-NES, PGPE and CMA-ES restarts identical to direct library runs, with
re-checked best points; CVT and grid archives identical to `fcmaes-core`
MAP-Elites; emitter phases that only add or improve elites and spend exact
budgets; registry checks of unknown parameters, unused or altered inputs,
versions, kinds and domains; resolved jobs and results on the built-in models
compared with the previous worker; the tutorial with an application worker
and a hashed data input.
