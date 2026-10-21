# Agent guide

fcmaes-rust is a pure-Rust gradient-free optimization library (`fcmaes-core`:
BiteOpt, DE, CMA-ES, CR-FM-NES, PGPE, Dual Annealing, MODE, MAP-Elites,
retry), its Python binding (`fcmaes-rust`, imported as `fcmaes_rust`), and a
job worker (`fcmaes-job-worker`) that runs registered models as durable jobs
through the shared job API of
[optimization-jobs](https://github.com/dietmarwo/optimization-jobs).

- Read [ai-context.md](ai-context.md) first; its "Start here" table points to
  the section for each task. If the solver is not chosen yet, start at the
  [cross-repository entry point](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md),
  which routes between fast-cma-es, fcmaes-rust, LP-IP-craft and SQP-craft.
- Jobs: a worker's `describe` (`optimization-run problems describe ID`, MCP
  `describe_worker`) lists `payload_schemas` and runnable `examples`.
  Application models: [docs/job-api/models.md](docs/job-api/models.md).

## Build and test

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
cargo build --release --locked -p fcmaes-job-worker --example custom_worker --example simulation_worker
python3 -m pip install "jsonschema>=4.18,<5"
JOB_API_REQUIRE_SCHEMA=1 python3 tutorials/job-api/run.py --root "$(mktemp -d)"
JOB_API_REQUIRE_SCHEMA=1 python3 tutorials/surrogate/run.py --root "$(mktemp -d)"
```

`.github/workflows/ci.yml` lists the rest (shared harness, Python binding).
Each application tutorial is its own Cargo workspace: build it if you change it.

## Documentation checks

```sh
python3 scripts/check_doc_consistency.py   # inventories, versions, relative links
python3 scripts/build_book.py              # mdBook into target/book (needs mdbook)
python3 scripts/check_book_links.py        # links and anchors of the built book
```

## Conventions

- Rust 2024, minimum Rust 1.88 (`cargo +1.88.0 check --workspace --locked`);
  `#![forbid(unsafe_code)]`; build with `--locked`.
- All optimizers minimize; constraints are feasible at `g(x) <= 0`.
- A change to the shared job API lands in optimization-jobs first; then the
  pinned `rev` in `Cargo.toml` moves.
- Record changes under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md), and
  update ai-context.md when behavior changes.
