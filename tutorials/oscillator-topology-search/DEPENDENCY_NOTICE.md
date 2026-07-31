# ReBop runtime-expression compatibility notice

Upstream compatibility target:
[Armavica/rebop 0.9.7](https://github.com/Armavica/rebop)
Upstream license: MIT
Upstream copyright: 2021 Virgile Andreani

ReBop 0.9.7 defines a public `Rate::expr(Expr)` constructor and a public
`Rate::Expr(Expr)` variant, but its crate root declares `mod expr` rather than
`pub mod expr`. A dependent Rust crate therefore cannot name or construct the
argument type. The Python binding can use it because it is inside the ReBop
crate.

`vendor/rebop/` is a reduced, tutorial-owned implementation derived from the
relevant upstream behavior. It is **not** a one-line visibility patch or a
source-identical copy. It carries only:

- the runtime expression tree and evaluator;
- sparse law-of-mass-action propensities;
- seeded Gillespie direct-method stepping; and
- the species, reaction-count, time and advance methods used by this tutorial.

Compared with upstream 0.9.7, it omits the dense rate/jump APIs, unseeded
constructor, reseeding and state/time setters, macro DSL, Python binding,
parser, benchmarks and unrelated documentation. Dense jumps are always
converted to sparse storage, so the constructor's `sparse` compatibility
argument is accepted but has no effect. The local stepper also stops a path
when an individual propensity is non-finite or negative; upstream only stops
when the accumulated total is not positive.

The integration test in `tests/upstream_compatibility.rs` links both sources
under distinct crate names. At seeds 1, 42 and 12345, it requires exact species
counts and model times for the same sparse mass-action birth/conversion/decay
network at several checkpoints. This gates the shared path used for
degradation reactions. Upstream's private `Expr` type prevents an external
side-by-side test of runtime expressions; that path is instead a small
semantic re-derivation covered by analytic propensity and seeded replay tests.

The main dependency selects the local `rebop = "=0.9.7"` path explicitly; the
registry release is a dev-dependency used only by the compatibility test. The
lockfile and `run.json` cache boundary identify the compatibility revision as
`rebop=0.9.7-expr-public-v1`.

Replace the local dependency and this notice when a released ReBop version
makes the runtime expression type constructible from external Rust without
changing the model equations.
