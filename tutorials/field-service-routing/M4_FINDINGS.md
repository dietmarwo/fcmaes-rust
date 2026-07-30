# Independent scorer gate

Date: 2026-07-30.

`vrp-core = 1.25.0` was inspected before implementation. It is an Apache-2.0
low-level solver library with problem, insertion-context, route, and solution
domain types. It does not expose a drop-in checker that accepts this tutorial's
explicit route order and cost specification. Pragmatic serialization and
checking belong to other parts of the upstream project.

Adding its wide solver dependency tree solely for validation would therefore
not create an independent comparison. The crate is not retained.

`src/scorer2.rs` is the fallback described by the plan. It was written from
`COST_SPEC.md` and uses direct coordinate arithmetic instead of the evaluator's
distance helper and per-route metric structures. On 1,000 supplied random
routes—100 for each frozen instance—the publication study found:

```text
maximum absolute discrepancy = 0
mean absolute discrepancy    = 0
```

Bit-exact agreement is expected here: both scorers consume the same decoded
route and apply the same IEEE-754 primitive operations in the same order,
although their control flow and intermediate structures are separate. The
check catches implementation drift in distance, waiting, lateness, capacity,
shift, and cost arithmetic. It does not independently validate route decoding;
the exact-once, skill, tie, and active-mask invariants are covered by separate
decoder tests.

This is useful regression evidence, but it is still an in-repository
cross-check. Shared interpretation errors remain possible. The tutorial does
not label it external validation and does not claim equivalence to a mature
VRP engine.
