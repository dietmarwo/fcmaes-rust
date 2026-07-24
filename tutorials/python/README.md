# Tutorial plotting API

This package turns the versioned JSON/CSV artifacts emitted by the native Rust
simulator tutorials into consistent Matplotlib figures. It does not put Python
callbacks in the optimization hot path.

From `public/tutorials/python`:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest
.venv/bin/python render_all.py --check
.venv/bin/python check_docs.py
```

Render one run:

```bash
.venv/bin/fcmaes-tutorial-plots \
  ../rapier-trebuchet/results/quick/qd/run.json \
  --output-dir ../rapier-trebuchet/images/quick-qd
```

The API also accepts result arrays from the optional PyO3 bindings through
`pareto_from_arrays` and `qd_from_archive`.

`render_all.py --write` regenerates every figure from discovered schema-v1
manifests. `--check` renders into temporary directories and compares bytes, so
volatile Matplotlib metadata cannot create silent documentation drift.
`requirements-lock.txt` fixes the complete plotting stack used for regeneration
and byte-for-byte validation; use it whenever checked-in figures are updated.
`check_docs.py` verifies that every local Markdown link and image target exists.
When `public/` is the Git repository root, as it is after publication and in
CI, the check also requires every target to be present in the Git index. This
prevents locally generated but uncommitted result files from masking broken
README evidence links.
