# Combinatorial encoding cookbook

`fcmaes-rust` optimizes fixed-length vectors of `f64` values. It does not
provide native Boolean, categorical, permutation, or heterogeneous chromosome
types. Instead, an objective can decode a bounded real vector into a typed
application design.

This approach is useful when discrete decisions are coupled to simulation,
continuous controls, nonlinear costs, unusual constraints, or several
objectives. For a standard TSP, knapsack, assignment, flow, or
constraint-satisfaction problem, prefer a specialized solver or a
genotype-aware evolutionary library. A representation designed for the
combinatorial structure will usually have better variation operators and may
provide useful bounds.

The GitHub-only example crate contains dependency-free, tested reference
helpers in
[`examples/src/encoding.rs`](../examples/src/encoding.rs). They are not part
of the published `fcmaes-core` API; copy or adapt the small decoders to make
application-specific choices explicit.

## Choose the representation first

| Decision | Suggested encoding | Main caveat |
|---|---|---|
| Bounded ordinal integer | Equal-width normalized bins | Rounding convention must be consistent |
| Integer spanning orders of magnitude | Logarithmic transform, then rounding | Discrete bins have unequal widths |
| Unordered category | Equal-width index lookup | Numeric category distance is artificial |
| Boolean | Threshold one normalized coordinate | Produces two flat regions |
| Permutation | Sort one random key per item | Many vectors decode to the same order |
| Exactly \(k\) selected items | Take the \(k\) smallest keys | Selection changes at rank crossings |
| Ordered items split among groups | Permutation keys plus separator keys | Repeated separators can create empty groups |
| Unique selections with preferred indices | Deterministic collision repair | Repair direction introduces bias |
| Ordered times | Map coordinates to the time range and sort | Several vectors can represent the same schedule |
| Precedence-constrained schedule | Priority keys plus a precedence-aware decoder | Decoder is domain-specific |

Before implementing any encoding, check:

1. Every coordinate has finite bounds.
2. Every intended design is reachable.
3. Invalid input is rejected or repaired deterministically.
4. Ties have an explicit, reproducible rule.
5. The mapping does not introduce unacceptable bias or redundancy.
6. A structured solver is not a better match.

## Mixed typed designs

Keep the optimizer vector separate from the decoded application type:

```rust
use fcmaes_examples::encoding::{
    EncodingError, boolean, categorical_index, linear_integer,
    logarithmic_integer,
};

#[derive(Debug)]
struct Design {
    workers: usize,
    cache_size: usize,
    enabled: bool,
    strategy: &'static str,
    gain: f64,
}

fn decode(x: &[f64]) -> Result<Design, EncodingError> {
    if x.len() != 5 {
        return Err(EncodingError::InvalidDimension {
            expected: 5,
            actual: x.len(),
        });
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(EncodingError::NonFinite);
    }
    let strategies = ["fifo", "priority", "shortest-job"];
    Ok(Design {
        workers: linear_integer(x[0], 1, 32)?,
        cache_size: logarithmic_integer(x[1], 8, 4096)?,
        enabled: boolean(x[2])?,
        strategy: strategies[categorical_index(x[3], strategies.len())?],
        gain: 0.1 + 1.9 * x[4].clamp(0.0, 1.0),
    })
}
```

The objective should validate the vector length and reject non-finite values
before indexing it. The decoder becomes the authoritative definition of
rounding, clamping, category order, and repair. Re-evaluate the final decoded
design rather than trusting a printed optimizer vector.

The SmartCore tutorial uses the same pattern for continuous fractions,
linearly and logarithmically scaled integers, and a categorical split
criterion in one eight-coordinate vector. The NeXosim and RustPower tutorials
decode mixed engineering controls into typed structs.

## Integers

For a normalized coordinate \(u \in [0,1]\), equal-width bins give every
integer from `lower` through `upper` the same interval:

```rust
let staff = linear_integer(u, 1, 8)?;
```

Using `round(lower + u * (upper - lower))` is also valid, but the endpoint
integers receive half-width intervals. Pick one convention, test both
endpoints, and do not change it after recording results.

Use a logarithmic transform when ratios matter more than differences:

```rust
let population = logarithmic_integer(u, 8, 512)?;
```

DE and MODE accept an optional integer-coordinate mask. It improves how those
optimizers mutate integer coordinates, but it does not replace decoding.
Clamping and conversion in the objective remain authoritative. BiteOpt,
CMA-ES, CR-FM-NES, and PGPE see only the real coordinate.

## Categories and Booleans

Map an unordered category through equal-width bins:

```rust
let solvers = ["direct", "iterative", "hybrid"];
let solver = solvers[categorical_index(u, solvers.len())?];
```

The category order does not create physical distance. Swapping two category
labels changes the optimizer landscape even though the application choices
are unchanged. Compare several orderings when this materially affects a small
categorical problem, or use a representation-aware method when categories
dominate the search.

A Boolean is a two-category special case:

```rust
let use_cache = boolean(u)?;
```

One-hot coordinates do not automatically solve the representation problem:
they increase dimension and still need an argmin/argmax or repair step to
enforce exactly one choice.

Mazda demonstrates lookup into per-coordinate engineering choice tables. Its
decoder preserves the original truncation convention for result parity. The
normalized equal-bin helper above is the clearer default for new code.

## Random-key permutations

Assign one real key to every item and sort the item indices by key:

```rust
let order = permutation_from_keys(&[0.8, 0.1, 0.4])?;
assert_eq!(order, [1, 2, 0]);
```

Every decoded result is a valid permutation, so no duplicate or missing item
penalty is needed. The reference helper breaks equal keys by original item
index and uses `f64::total_cmp`, making ties deterministic.

Random keys are redundant: many real vectors represent the same permutation.
Small coordinate changes usually preserve the order until two keys cross.
That piecewise-constant structure is often useful for BiteOpt and DE, but it
is not equivalent to permutation-specific crossover or mutation.

The flexible job-shop example uses one coordinate block for machine choices
and another as operation priority keys. Its decoder restores within-job
precedence after sorting the priorities:

```text
[machine-choice coordinates | priority keys]
              ↓
select alternatives, sort priorities, emit each job's next operation
```

This is more informative than simply penalizing precedence violations because
every decoded schedule respects the job order by construction.

## Exact-cardinality subsets

To select exactly \(k\) of \(n\) items, take the indices of the \(k\) smallest
keys:

```rust
let selected = select_k_from_keys(&[0.7, 0.2, 0.9, 0.1], 2)?;
assert_eq!(selected, [3, 1]);
```

This guarantees cardinality and avoids a penalty for selecting too many or
too few items. If the selected items are unordered, sort the returned indices
before using them so equivalent selection orders do not leak into caching or
reporting.

Thresholding \(n\) Boolean coordinates is preferable only when cardinality is
not fixed or when deviation from the desired count has a meaningful
constraint value.

## Unique selection with collision repair

When each coordinate expresses a preferred discrete item, repair duplicates
deterministically:

```rust
let selected = unique_indices_with_repair(&[0.0, 0.0, 0.0], 4)?;
assert_eq!(selected, [0, 1, 2]);
```

The reference implementation advances cyclically to the next unused item. The
transfer-scheduling example uses this pattern to choose ten distinct
trajectories.

Repair is not neutral. A collision at item zero favors items one, two, and so
on. Use top-\(k\) random keys when all items should be symmetric, or design a
domain-specific repair whose bias reflects a real preference.

## Permutations split among routes or groups

Use one key per item for global order and one separator key per boundary:

```rust
let routes = partition_from_keys(
    &[0.4, 0.1, 0.3, 0.2], // item keys
    &[0.2, 0.8],           // two separators create three routes
)?;
```

The helper sorts the item permutation and separator positions. Repeated
separator positions deliberately create empty groups. If empty routes are
invalid, either repair the cut positions, return a meaningful constraint
violation, or use a representation that assigns at least one item per group.

The multi-UAV example uses a related formulation with `vehicles - 1`
separator coordinates plus one key per target. Every target appears exactly
once, while the separators divide the visit sequence among vehicles.

An alternative assignment representation uses one categorical vehicle index
per item plus separate within-vehicle priority keys. That makes reassignment
local but can produce empty vehicles and uses more coordinates. Benchmark
both encodings when assignment locality matters.

## Ordered times and domain repair

Sorting normalized time coordinates guarantees a chronological sequence:

```rust
let times = sorted_breakpoints(&[0.75, 0.25, 0.5], 0.0, 24.0)?;
assert_eq!(times, [6.0, 12.0, 18.0]);
```

Add fixed horizon endpoints after decoding when the application requires
them. The transfer-scheduling example sorts station order and event times
separately.

More complex feasibility often needs a domain-specific repair. The harvesting
example moves machine availability windows until a maximum-concurrency rule
is satisfied. Such a repair should:

- terminate for every accepted input;
- be deterministic;
- preserve as much of the proposed design as possible;
- expose failure rather than silently returning an invalid design; and
- be tested on collisions and boundary cases.

## Repair, constraint, penalty, or rejection?

| Situation | Preferred treatment |
|---|---|
| Feasibility is cheap to guarantee by construction | Encode it directly |
| Every vector maps deterministically to a useful feasible design | Repair |
| Violation has a useful continuous magnitude | Return an explicit MODE or `moretry` constraint |
| Scalar optimization has a graded residual violation | Add a normalized penalty |
| Simulation is undefined and no useful violation exists | Return a documented large finite rejection value |
| Repair is strongly biased, expensive, or destroys locality | Use a structured solver or genotype-aware optimizer |

For scalar penalties, normalize objective and violation magnitudes before
choosing the coefficient:

```text
score(x) = objective(x) + rho × Σ max(0, violation_i(x))²
```

If `rho` is too small, infeasible candidates win. If it is too large, most of
the search space becomes an almost indistinguishable penalty plateau. For
MODE and `moretry`, objectives come first and constraints follow; feasibility
means `constraint <= 0`.

## Optimizer guidance

- Start with BiteOpt for a difficult scalar problem dominated by decoded,
  discontinuous choices.
- Use DE when broad bounded exploration and integer masks are useful.
- Use MODE for several objectives or meaningful explicit constraint
  violations; it also accepts an integer mask.
- Random-key order coordinates can work with CMA-ES, CR-FM-NES, and PGPE, but
  categorical plateaus give their distribution updates little local
  information.
- Use retry when many discrete basins make one run unreliable.
- Compare encodings under equal objective-call budgets. A better
  representation can matter more than changing the optimizer.

## Decoder test checklist

Test the decoder independently from optimization:

- lower and upper coordinate endpoints;
- non-finite coordinates;
- every category and integer value is reachable;
- deterministic tie handling;
- permutation uniqueness and completeness;
- exact subset cardinality;
- route/group coverage and empty-group policy;
- precedence, exclusivity, capacity, and resource invariants;
- termination and idempotence of repair where applicable;
- independently decoded final results; and
- identical decoding during serial and parallel evaluation.

Property-based or exhaustive tests over small instances are especially useful.
Also sample many uniform optimizer vectors and count decoded outcomes: this
reveals category, endpoint, repair, and separator biases that are difficult to
see from the formulas alone.

## Repository examples

| Source | Pattern |
|---|---|
| [`examples/src/encoding.rs`](../examples/src/encoding.rs) | Tested reference helpers used by this cookbook |
| [`examples/src/jobshop.rs`](../examples/src/jobshop.rs) | Machine alternatives, random-key order, and precedence-preserving dispatch |
| [`examples/src/uav.rs`](../examples/src/uav.rs) | Random-key permutation plus multi-route separators |
| [`examples/src/scheduling.rs`](../examples/src/scheduling.rs) | Unique selection repair, station permutation, and sorted times |
| [`examples/src/harvesting.rs`](../examples/src/harvesting.rs) | Deterministic resource-capacity repair and residual failure penalty |
| [`examples/src/mazda.rs`](../examples/src/mazda.rs) | Per-coordinate engineering choice tables |
| [`tutorials/ml-hyperparameter-tuning/src/space.rs`](../tutorials/ml-hyperparameter-tuning/src/space.rs) | Mixed continuous, linear/log integer, and categorical decoding |
| [`tutorials/nexosim-production-line/src/model.rs`](../tutorials/nexosim-production-line/src/model.rs) | Typed mixed continuous/integer controls with an integer mask |
| [`tutorials/rustpower-voltage-control/src/lib.rs`](../tutorials/rustpower-voltage-control/src/lib.rs) | Mixed engineering controls, lookup indices, and explicit constraints |
