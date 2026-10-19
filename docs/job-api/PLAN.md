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
