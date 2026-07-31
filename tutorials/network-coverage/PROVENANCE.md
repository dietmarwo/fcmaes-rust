# Provenance and data policy

## Checked-in data

`instances/tiny`, `small`, `reference-1k`, and `reference-4k` are artificial
graphs generated entirely by this tutorial. Their fixed parameters and seeds
are defined in `src/instance.rs`; `cargo run --release --locked -- --mode
generate` reproduces the CSV files. Nodes have no person, account, geographic,
medical, or demographic meaning.

The application is framed as abstract outreach and monitoring coverage. It is
not a people-targeting or political-influence application.

## External graphs

The Python source tutorial used SNAP's `ego-Facebook` graph. This repository
does not copy, transform, or redistribute that dataset because no explicit
redistribution licence was identified for the source file. A reader who has
obtained an edge list under its own terms may run:

```bash
cargo run --release --locked -- \
  --mode mo --graph /local/path/to/undirected-edge-list.txt
```

The importer accepts two zero-based integer endpoints per non-comment line,
assigns deterministic synthetic costs, and defines no native groups. Duplicate
undirected edges are deduplicated. Self-loops are dropped with a counted
warning, and the count is retained in run metadata. Results from such a local
input are not the checked-in publication evidence.

## Method provenance

The formulation was motivated by the `Media.adoc`, `fbcover.py`, and
`edgecover.py` examples in Dietmar Wolz's `fast-cma-es` repository. This
port changes the application framing, replaces external data with synthetic
fixtures, adds an exactly expandable native group score, separates
cardinality and weighted certificates, and adds exact tiny ILPs, throughput
gating, marginal-greedy comparison, reproducible artifacts, and tests.
