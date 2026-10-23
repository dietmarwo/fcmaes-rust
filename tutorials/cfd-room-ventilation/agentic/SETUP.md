# Execute the CFD prompt workshop

Use a checkout of fcmaes-rust, Linux with current stable Rust, Python 3.10+ and two available CPU cores. The standalone native tutorial requires Rust 1.88+. No real-building data, external CFD package or model API key is required for numerical execution.

From the repository root:

```sh
python3 -m venv .venv-cfd
. .venv-cfd/bin/activate
python3 -m pip install -r tutorials/cfd-room-ventilation/agentic/requirements.txt
cargo build --release --locked --manifest-path tutorials/cfd-room-ventilation/Cargo.toml
python3 tutorials/cfd-room-ventilation/agentic/test_audit.py
python3 tutorials/cfd-room-ventilation/agentic/run.py \
  --bin-dir tutorials/cfd-room-ventilation/target/release \
  --output /path/to/new/cfd-results --seeds 42 43 44 \
  --evaluations 2048 --workers 2
```

If `CARGO_TARGET_DIR` is set, point `--bin-dir` at that directory's `release` instead. The wrapper needs the existing `cfd-room-ventilation` and `verification` binaries. Use a new or empty output directory; it refuses to overwrite a previous study. `--timeout` is a per-command process limit, default 600 seconds. The native command's stdout/stderr, exit status and elapsed time are retained.

Give a shell-capable agent the checkout, binary and output paths, then paste [prompt.md](prompt.md). A generic chat-only session cannot run the experiment. This native CFD model is not registered in the shared job worker; these commands are direct CLI executions, without job-API lifecycle/evidence labels or optimizer checkpoint recovery. That avoids claiming an interface which does not exist. A future job adapter would be separate work.

The default campaign runs six optimizations and one verification command. It records 12,288 search evaluations, with baseline and verification calls outside that count. Budgets must be multiples of the 64-candidate batch size; the script checks the actual count. The chosen designs are selected by training quality inside the native program. The first seed is the predetermined mesh/field representative.

Outputs include each native `pareto.csv` or `archive.csv`, convergence and held-out CSVs, 18 grid/source/design comparisons, the channel reference, three field CSVs, four figures, `summary.json`, exact `commands.json`, binary hashes and `checksums.json`. The last file covers every other output file. Logs preserve original absolute paths as provenance.

For CI, `--seeds 42 --evaluations 512` exercises the same checks with a smaller search. Its results are separate from the three-seed tutorial measurements. `test_audit.py` covers corrupted scores/selection, full four-objective dominance and constraint failures. Native model tests remain in the original standalone workspace; no Rust model, numerical method, dependency pin or lockfile is changed.
