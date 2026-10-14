# Exact 2D archive

`ParetoArchive2D<P>` is a separate, unbounded archive of unique nondominated
minimization objective pairs. It retains the first payload for duplicate pairs
and maintains hypervolume against a fixed finite reference. It does not alter
MODE's selection, scalar BiteOpt costs, QD archives or existing Pareto helpers.

```rust
use fcmaes_core::{ParetoArchive2D, ParetoBounds2D, ParetoScore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
let bounds = ParetoBounds2D::new([0.0, 0.0], [4.0, 4.0])?;
let mut archive = ParetoArchive2D::new(bounds, ParetoScore::Linear);
let accepted = archive.add_batch(
    &[[1.0, 3.0], [3.0, 1.0], [2.0, 3.0]],
    &["left", "right", "dominated"],
)?;
assert_eq!(accepted, [true, true, false]);
assert_eq!(archive.hypervolume(), 5.0);
Ok(())
}
```

Bounds apply to objective coordinates, not decision variables. Signed finite
values are supported inside declared bounds; signed zero is canonicalized.
Preference scoring affects the cached dominator choice and preferred entry,
not geometric membership. Keep lower bounds and reference fixed throughout
the run. Finite but unrepresentable area arithmetic returns an error.

Use `insert_with(point, factory)` to defer construction of accepted payloads.
`add`/`insert` accept an already constructed payload. `add_batch` clones only
accepted payloads. A length mismatch performs no updates. A numeric row error
returns `BatchUpdateError` with its zero-based row, underlying error and the
acceptance mask for the processed prefix; earlier rows remain applied.
Allocation failure is not a recoverable transactional guarantee.

`entries()` borrows sorted entries directly from the coordinate B-tree.
`front()` creates an owned payload snapshot; `clear()` resets contents and HV
while preserving bounds and preference scoring. There is no capacity pruning.
An antichain can consume memory proportional to every evaluated point.

## Cached acceptance and ownership (13 October 2026)

Append and interior-gap conditions from BiteOpt commit
`6f3a63a02da75f55242a5d51edf8f5b7e4ff680c` now skip range walks when
coordinates prove that a candidate fits without deleting entries. Only owned
coordinates are cached: the last point and the latest accepted point's surviving
successor. Every successful insertion refreshes or invalidates the gap; `clear()`
resets all state. Stable Rust's `BTreeMap::insert` still costs O(log n).

`Clone` is available for cloneable payloads and owns independent entries and
cache values. Numeric checks and the payload factory run before membership,
HV or acceptance-cache changes, including the shortcuts. Payload-factory panics
leave the archive usable and unchanged. Rust allocation exhaustion retains the
standard allocator's abort behavior; there is no recoverable OOM API.

No experimental ZDT4 scalar cost or stopping threshold is enabled. MODE selection
is unchanged. The delivery report records gains and regressions by stream order.

## Single-map design (12 October 2026)

The archive now uses one `BTreeMap` and one cached dominance witness.
The small sorted-vector path and its 64-entry promotion threshold are removed.
After validation, a cache hit rejects immediately; a miss tests the predecessor
and computes the contiguous dominated-successor interval. The reusable victim-key
buffer separates area preflight from mutation using stable Rust 1.88 APIs.
It is scratch storage, not another front.

`preferred()` and both normalized scoring policies retain their existing
semantics. Computed score ties use original-coordinate tie-breaks. The cache
starts empty and `clear()` resets it, so the reference corner is accepted in
an empty archive with zero HV. Invalid inputs still fail before cached rejection.
Numeric checks and payload creation complete before membership changes.

This follows Aleksey Vaneev's simplified cached-map design in BiteOpt commit
`02427c2f1dad87f3a146eda8b4613f01ee2eb8d8`. It preserves fcmaes's stronger numeric
contract. The companion delivery report records before/after measurements:
removing compact storage is a maintainability tradeoff, not a universal speedup.

## MODE example and checks

From the repository root:

```sh
cargo run --release -p fcmaes-core --example pareto_mode
cargo test --locked -p fcmaes-core
cargo test --release --locked -p fcmaes-core
cargo clippy --locked -p fcmaes-core --all-targets
```

The example archives finite feasible evaluated offspring before `tell`, using
constraints `<= 0`. Include evaluated initial guesses in application drivers.
Filtering invalid evaluations precedes MODE's sentinel sanitization. The
archive result is distinct from the breeding population. For parallel runs,
keep one archive owner and merge worker-local fronts in a stable ID order.

The module includes independent prefix-front/HV oracles, numerical rollback
tests, batch and payload checks, and a fixed-seed MODE trace comparison with
tracking enabled and disabled. The latter requires identical populations and
objective evaluation traces, with no extra objective calls.

The shared fixture is `tests/data/pareto_archive_2d.csv`. Build the replay:

```sh
cargo build --release -p fcmaes-core --example pareto_trace
python tests/check_pareto_cross_language.py target/release/examples/pareto_trace
```

The Python replay additionally requires the corresponding integrated
`fast-cma-es` wheel. The Rust Python archive binding, ND batch filtering,
indicator replacement and hypervolume-driven optimizer are follow-ups.

## Provenance

Adapted from the optimized `bitehv2_rust/src/lib.rs`, SHA-256:

```text
a3a1df4620d35a59705f142f3ddf3557450baa49eec66cb83b0191db1615a5ea
```

The original numerical update logic is retained. The initial compact/B-tree
strategy was replaced by one B-tree on 12 October 2026. The insertion API, batch errors, reset/snapshot methods and examples are
unchanged; cloneable archives gained an owned `Clone` implementation.
The comparison scan implementation exists only in tests. Retain the full MIT
notice in `NOTICE.pareto`, including Aleksey Vaneev's attribution.
