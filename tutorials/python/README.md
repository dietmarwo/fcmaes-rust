# Tutorial plotting API

This package turns the versioned JSON/CSV artifacts emitted by the native Rust
application tutorials into consistent Matplotlib figures. It does not put
Python callbacks in the optimization hot path.

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

## Rendering and validation contract

- `render_all.py --write` regenerates every figure from discovered schema-v1
  manifests and runs each tutorial-specific `plot_results.py --write` script.
  The room-ventilation and HPO tutorials use this extension because their
  publication figures combine evidence from more than one optimizer manifest.
- A tutorial whose artifact axes do not fit the common renderer may add a
  `.custom-renderer` marker. The common renderer then skips its manifests, but
  its deterministic `plot_results.py` remains part of the same write/check
  pass.
- `render_all.py --check` renders into temporary directories and compares
  bytes. This prevents volatile Matplotlib metadata from causing silent
  documentation drift.
- `requirements-lock.txt` pins the complete plotting stack used for
  regeneration and byte-for-byte validation. Use it whenever checked-in
  figures are updated.
- `check_docs.py` verifies every local Markdown link and image target. When
  `public/` is the Git repository root—as it is after publication and in CI—it
  also requires every target to be in the Git index. Locally generated but
  uncommitted files therefore cannot mask broken evidence links.
