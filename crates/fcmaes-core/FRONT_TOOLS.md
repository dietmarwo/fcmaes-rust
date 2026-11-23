# Box indexes and archive-gated differential evolution

All objectives are minimized. These APIs are available when building this
source checkout. The rank indexes use original input row IDs. Both Pareto
archives can retain the decision associated with an accepted objective row.

| Task | Rust in `fcmaes_core` | Python in `fcmaes_rust` |
|---|---|---|
| Find rows inside closed boxes | `rankindex::BordaIndex`, `rankindex::GridIndex` | `BordaBoxIndex`, `GridBoxIndex` |
| Run archive-gated DE | `archivede::ArchiveDe<X, Y>` | `ArchiveDE` |
| Keep every strictly nondominated observation, including duplicates | `archivede::ParetoArchive<Y, P>` | `ParetoArchiveND` |
| Keep one payload per 2D objective pair and track exact hypervolume | `ParetoArchive2D<P>` | `ParetoArchive2D` |
| Measure mean distance from reference points to an approximation | `igd(front, reference)` | `igd(front, reference)` |

## Query a fixed point set

Build once from finite points. The indexes own their rank data and can be
shared among Rust readers. Query endpoints are inclusive. Infinite endpoints
give open-ended ranges. A reversed interval or a NaN endpoint produces an empty
result. The checked Rust constructors validate matrix size, dimensions and,
for the grid, the nominal query volume and representability of its cell keys.

```rust
use fcmaes_core::rankindex::{BordaIndex, GridIndex};

fn main() -> Result<(), &'static str> {
    let points = [0., 1., 0.5, 0.5, 1., 0.]; // Three rows of width two.
    let b = BordaIndex::try_new(&points, 2)?;
    let g = GridIndex::try_new(&points, 2, 0.1)?;
    let lo = [0.4, 0.4];
    let hi = [0.6, 0.6];
    assert_eq!(b.query(&lo, &hi).ids, [1]);
    assert_eq!(g.query(&lo, &hi).ids, [1]);
    Ok(())
}
```

Borda emits IDs in shortest-axis rank order and stores O(n d²) cross-ranks.
The grid emits IDs in ascending cell-key and then input-row order.
Its positive `query_volume` parameter is an expected query volume fraction
used to select the grid resolution. Changing it preserves box membership.

```python
import numpy as np
from fcmaes_rust import BordaBoxIndex, GridBoxIndex

points = np.array([[0., 1.], [.5, .5], [1., 0.]])
lo = np.array([[.4, .4], [-np.inf, -np.inf]])
hi = np.array([[.6, .6], [np.inf, np.inf]])
for index in (BordaBoxIndex(points), GridBoxIndex(points, 0.1)):
    offsets, ids, stats = index.query_batch(lo, hi)
    assert ids[offsets[0]:offsets[1]].tolist() == [1]
    assert sorted(ids[offsets[1]:offsets[2]].tolist()) == [0, 1, 2]
```

A single Python `query(low, high)` returns a `uint32` NumPy array of row IDs.
For a batch, offsets are `uint64`, IDs are `uint32`, and each query has a list
of exact integer counters. The Borda columns are `axis_candidates` and
`score_candidates`. The grid columns are `touched_cells`, `nonempty_cells`,
`full_cells`, and `point_tests`. Python preserves counts above 2⁵³.

## Optimize and retrieve the decisions

```rust
use fcmaes_core::archivede::{ArchiveDe, ArchiveDeParams};

fn main() -> Result<(), &'static str> {
    let params = ArchiveDeParams {
        population: 16, max_evaluations: 100, seed: 42, ..Default::default()
    };
    let mut de = ArchiveDe::<[f64; 2], [f64; 2]>::new([-1.; 2], [1.; 2], 2, params)?;
    de.minimize(|x| [x[0] * x[0] + x[1] * x[1], (x[0] - 1.).powi(2)])?;
    assert_eq!(de.evaluations(), 100);
    for (objectives, decision) in de.archive().ordered_entries() {
        assert_eq!(objectives[1], (decision[0] - 1.).powi(2));
    }
    Ok(())
}
```

Use `ArchiveDe<Vec<f64>, Vec<f64>>` when dimensions are known at runtime.
The population must be at least four and larger than the objective count.
Equal lower and upper bounds fix a coordinate. The evaluation budget includes
initialization and can end partway through a generation. `ask` returns the
pending batch until `tell` accepts a finite objective matrix of the same length.
The gate admits only newly archived offspring to the survivor pool. The tiered
policy selects objective champions and spreads survivors in rank space. Both
switches can be set independently.

```python
import numpy as np
from fcmaes_rust import ArchiveDE, ParetoArchive2D, igd

de = ArchiveDE(2, [-1., -1.], [1., 1.], popsize=16, max_evaluations=100, seed=42)
hv = ParetoArchive2D([3., 5.], payload_dim=2)
while len(xs := de.ask()):
    ys = np.column_stack(((xs * xs).sum(axis=1), (xs[:, 0] - 1.) ** 2))
    hv.add_batch(ys, xs)
    de.tell(ys)
decisions, front = de.front()
assert de.nfev == 100
assert hv.hypervolume() > 0
assert igd(front, front) == 0
```

Both Python archives accept `add(y, x)` or `add_batch(ys, xs)`. Omit the
payload for `payload_dim=0`. Their `front()` returns independent matrices
of shape `(rows, payload_dim)` and `(rows, nobj)`. Admission flags describe
each insertion at the time it occurred. A later insertion may remove it.
The strict archive orders snapshots by rounded objective sum and then arrival
and keeps duplicate nondominated observations. The hypervolume archive orders
the front by increasing first objective, keeps the first payload per pair and
excludes points above the reference. Rust also exposes preference scores
and lazy payload creation. See the [2D archive guide](PARETO_ARCHIVE.md).

The two index algorithms and the archive-gated DE follow the supplied
`borda_box_query-1.cpp`, `grid_point_index-1.cpp` and `archive_de.cpp`.
The source notices accompany the crate in `NOTICE.fronts` and `NOTICE.pareto`.
