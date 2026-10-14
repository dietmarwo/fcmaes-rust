# Exact 2D Pareto archive

`ParetoArchive2D<P>` keeps every unique nondominated pair of two minimized
objectives inside a fixed reference box, each with a payload `P` (for example
the decision vector), and updates the dominated hypervolume with every
insertion. It is independent of MODE: use it to record everything an
optimizer evaluated, without changing the optimizer.

```rust
use fcmaes_core::{ParetoArchive2D, ParetoUpdate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut archive = ParetoArchive2D::new([4.0, 4.0])?;
    let accepted = archive.insert_batch(
        &[[1.0, 3.0], [3.0, 1.0], [2.0, 3.0], [5.0, 0.0]],
        &["left", "right", "dominated", "outside"],
    )?;
    assert_eq!(accepted, [true, true, false, false]);
    assert_eq!(archive.hypervolume(), 5.0);
    assert_eq!(archive.insert([0.5, 0.5], "best")?, ParetoUpdate::Accepted { removed: 2 });
    Ok(())
}
```

## Semantics

- **Membership.** A point is archived unless an archived point weakly
  dominates it. Archiving it removes the points it dominates. Equal objective
  pairs keep the first payload. `-0.0` is stored as `0.0`.
- **Reference box.** Points with an objective above the reference return
  `Update::Outside` and are not archived, since they add nothing to the
  hypervolume. This matches `OutsidePolicy::Exclude` of the hypervolume
  indicators. There is no lower bound: objectives may have any finite value.
- **Errors.** Only non-finite objectives (`ParetoError::InvalidPoint`) and areas
  beyond the `f64` range (`ParetoError::NonFiniteArea`) are errors. A failed
  insertion leaves the archive unchanged. `insert_batch` stops at the failing
  row; the rows before it stay applied and the error reports their flags.
- **Payloads.** `insert_with(point, || payload)` builds the payload only when the
  point is archived; `insert_batch` clones only archived payloads.
- **Hypervolume.** Each insertion adds the exact area of the newly dominated
  region, summed with Kahan compensation. After millions of insertions the
  value agrees with a full recomputation to about 1e-16 relative error.
- **Capacity.** The archive is unbounded. In the worst case (every point
  nondominated) it keeps every point.

## Preference

`preferred()` returns the archived entry with the lowest preference score and
`ranked()` sorts all entries by it. By default the score is the mean of both
objectives. For objectives on different scales, normalize them:

```rust
use fcmaes_core::{ParetoArchive2D, ParetoBounds2D, ParetoScore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let points = [[0.1, 60.0], [0.8, 10.0]];
    // Plain mean: the second objective dominates the score.
    let mut plain = ParetoArchive2D::new([1.0, 100.0])?;
    plain.insert_batch(&points, &[1, 2])?;
    assert_eq!(*plain.preferred().unwrap().payload(), 2);
    // Normalized: `lower` is only the origin, e.g. an estimated ideal point.
    let bounds = ParetoBounds2D::new([0.0, 0.0], [1.0, 100.0])?;
    let mut scaled = ParetoArchive2D::with_preference(bounds, ParetoScore::Linear);
    scaled.insert_batch(&points, &[1, 2])?;
    assert_eq!(*scaled.preferred().unwrap().payload(), 1);
    Ok(())
}
```

Points below `lower` are archived normally; their normalized coordinates are
clamped to zero. The score never changes membership or hypervolume.

## Complexity

Entries are stored in a `BTreeMap` ordered by the first objective, so the
second objective strictly decreases along the front. The entry with the lowest
score serves as a cached dominator that rejects most points in O(1). Otherwise
an insertion costs one O(log n) predecessor lookup plus O(k log n) for the k
removed points. Appending after the last entry, or inserting into the gap
behind the previous insertion, skips the lookup.

`cargo run --release -p fcmaes-core --example pareto_bench` measures three
synthetic streams; `--example pareto_mode` records a constrained MODE run.

## Provenance

The algorithm is Aleksey Vaneev's 2D archive from BiteOpt (`bitehv2.h`), adapted
with compensated summation, error handling and payloads. MIT license; keep
`NOTICE.pareto` with copies of `src/pareto.rs`.
