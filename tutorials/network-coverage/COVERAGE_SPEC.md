# Coverage specification

This file is the numerical contract shared by the optimizer, the specialist
baselines, the independent checks, and the publication figures.

## Instance

An instance contains `n` nodes with costs `w_i ∈ (0,1]`, undirected ordinary
edges `E`, and overlapping groups `C`. Node indices are zero based. Ordinary
edges are loop free and deduplicated. Every group contains at least two distinct
valid nodes.

The synthetic and CSV fixture constructors reject self-loops. The external
edge-list adapter is a declared cleaning boundary: it drops self-loops with a
counted warning before calling the strict constructor and records the count in
run metadata.

The frozen synthetic fixtures are connected stochastic-block graphs. A
backbone path guarantees connectivity; additional edges prefer equal block
labels with probability `0.82`. Group members prefer one base block with
probability `0.78`. Costs are deterministic lognormal-like samples normalized
to `(0.05,1]`. Fixture sizes and seeds are defined in `src/instance.rs`.

## Decision decoding

There is one optimizer coordinate per node:

```text
x_i ∈ [0, 1.999999999999)
selected_i = x_i >= 1
```

The `fcmaes-core` integer mask is intentional. Its two integer bins correspond
to the two physical states. Non-finite coordinates are rejected.

## Frozen extended coverage

For selection `S`, ordinary edge `(u,v)` is covered when `u ∈ S` or `v ∈ S`.
For group `c` of size `s`, let `k=|c∩S|`. Its pair weight and coverage are

```text
g(s) = s^(-1/2)
group_cov(c,S) = g(s) [C(s,2) - C(s-k,2)].
```

This is exactly the weighted number of unordered pairs in the group with at
least one selected endpoint. The production kernel computes it from the member
count and never expands the clique. A literal pair loop exists only as a tiny
test oracle.

```text
coverage(S) = covered ordinary edges + Σ group_cov(c,S)
cost(S)     = Σ_{i∈S} w_i
roi(S)      = coverage(S) / coverage(V).
```

Empty selection has positive-zero cost and ROI zero. The analytic all-selected
value defines ROI one.

## Classic vertex-cover formulations

The classic checks use ordinary edges only.

```text
cardinality objective = |S| + 2 · uncovered_edges(S)
weighted objective    = cost(S) + (Σ_i w_i + 1) · uncovered_edges(S).
```

Both penalty coefficients make repairing any uncovered edge preferable to its
selection-cost increase. Published covers are independently replayed over every
ordinary edge.

The cardinality certificate publishes a maximal-matching lower bound `|M|` and
a reverse-delete-pruned endpoint cover no larger than `2|M|`. The weighted
certificate publishes a feasible primal-dual objective `D` and a
reverse-delete-pruned tight-vertex cover of cost no larger than `2D`. These
bounds are not interchangeable. Exact binary programs are labelled exact only
after `microlp` reports `SolutionStatus::Optimal`.

## Multi-objective formulation

MODE minimizes:

```text
f1 = cost(S) / cost(V)
f2 = 1 - roi(S).
```

Its initial population contains empty and full selections, the two certified
ordinary-edge covers, and deterministic marginal-gain-per-cost greedy prefixes.
The final two slots are deterministic pseudo-random masks. Artifacts compare
retained masks against this exact initial population and label the matching
origin—endpoint, certificate, greedy, or random initial. They do not compare
origin against greedy prefixes that were never supplied, and seeding is not
presented as an optimizer discovery.

The deterministic greedy rule chooses the unselected node with maximum
incremental coverage divided by cost, with node index breaking ties. Its full
prefix sequence is a strong baseline because the frozen coverage function is
monotone submodular.

The publication protocol runs MODE first at 8,192 evaluations and then, as an
additive budget-sensitivity check, at 200,000 evaluations on `reference-4k`.
Both campaigns are compared against the same complete greedy prefix sequence.
