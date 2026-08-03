# Security policy

## Supported versions

Security and silent numerical-integrity fixes are made for the latest
published `fcmaes-core` crate and `fcmaes-rust` Python package. Check whether a
suspected defect reproduces on the latest release when practical, but report a
credible issue even if it was first observed on an older version. Older
releases are not normally patched separately during the pre-1.0 series.

## What is security-relevant?

For an optimization library, security includes the integrity of the search
contract. A defect is security-relevant when normal API use can silently
return a plausible but materially invalid result. Examples include:

- memory-safety, denial-of-service, Python-buffer, FFI, parser, or dependency
  vulnerabilities;
- malformed bounds, dimensions, populations, or result rows bypassing
  validation and corrupting optimizer state;
- a returned point lying outside the documented bounds or a constrained
  optimizer silently reversing its feasibility convention;
- objective/result association, ordering, or concurrent evaluation defects
  that attach a fitness value to the wrong candidate; and
- deterministic seed, budget, or stopping defects that invalidate a promised
  reproducibility or resource-accounting contract without warning.

Ordinary stochastic variation, failure to find a global optimum, documented
population-budget overshoot, expected optimizer termination, and a benchmark
model's stated approximation limits are not vulnerabilities by themselves.
They may still be reported as correctness or documentation issues.

## Reporting a vulnerability

Report vulnerabilities privately through the repository's
[GitHub security-advisory form](https://github.com/dietmarwo/fcmaes-rust/security/advisories/new).
Do not put exploit details, credentials, private keys, proprietary objective
data, or sensitive optimization results in a public issue.

If private reporting is temporarily unavailable, open a public issue
containing only a request for a private contact channel. Do not include the
vulnerability details or sensitive reproducer in that issue.

A useful report contains:

- the smallest non-sensitive objective and configuration that reproduce the
  problem;
- the affected Rust API or Python function;
- package version, feature flags, and relevant dependency versions;
- the Rust/Python version, operating system, architecture, and worker count;
- the observed result and documented expected behavior;
- whether the failure is deterministic and which seed triggers it; and
- the potential impact and whether the issue is already public.

Maintainers aim to acknowledge a private report within seven days and provide
an initial assessment within fourteen days. These are best-effort targets for
a volunteer-maintained project, not a service-level guarantee. Reporter and
maintainer should coordinate disclosure until a fix or advisory is available.
Credit will be included when requested.

## Project safeguards

Hosted checks cover formatting, Clippy, workspace tests, Rustdoc, the declared
minimum Rust version, Python bindings, clean package consumption, and the
standalone tutorial protocols selected by CI. Dependency source, advisory, and
license policies are checked with `cargo-deny` where configured.

The repository also preserves deterministic fixtures, equal-budget controls,
decoded-result re-evaluation, typed failure paths, and raw benchmark artifacts
for high-risk numerical changes. These checks reduce risk but do not
constitute formal verification or guarantee that an optimizer will find a
global optimum. Users remain responsible for independently validating bounds,
objective and constraint signs, decoded results, and application-specific
feasibility.
