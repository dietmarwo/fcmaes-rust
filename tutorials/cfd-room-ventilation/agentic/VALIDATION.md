# CFD workshop validation record

Date: 23 October 2026. Source base: fcmaes-rust `92e1deb266910fff69cd0964ee3dba418d4a36e1`, fetched after the preceding tutorial merges. The workshop adds prompts, Python orchestration/auditing and CI; it changes no native model, optimizer, Cargo manifest or lockfile.

## Executed campaigns

| Campaign | Seeds | Evaluations per method/seed | Actual search evaluations |
|---|---|---:|---:|
| Tutorial measurements | 42, 43, 44 | 2,048 | 12,288 |
| CI-sized integration | 42 | 512 | 1,024 |

Both use MODE and MAP-Elites, two candidate workers, the existing 40x24 model, 800 maximum flow steps, 600 scalar steps and a 600-second limit per native command. All commands exited successfully. Baseline, held-out replay and verification work is additional to search evaluation counts. The reporting scalar selects representatives using training cases only. The first requested seed is predetermined as the field/grid representative.

Each campaign retained all fronts/archives, convergence and held-out CSVs, native logs, commands/exit codes/timings, binary hashes, 18 grid/source/design comparisons, a straight-channel reference, three field CSVs, four figures and output checksums. These raw artifacts were reviewed during integration but are not published; `run.py` regenerates them. [validation.json](validation.json) is the final three-seed summary; the figures beside this page come from that campaign.

## Assertions executed by the driver

- Exactly one selected row per run; finite metrics and all nine decisions inside native bounds.
- Feasible reporting scalar recomputed from four component metrics, tolerance `1e-8`.
- All pairwise dominance relations checked in the full four-dimensional MODE front.
- Training and held-out selected decision vectors exactly equal in the CSV exports.
- All four raw selected-design constraint margins finite and nonpositive on training and held-out replay.
- Actual search evaluations equal the requested multiple-of-64 budget.
- Repeated 40x24 selected metrics agree to `1e-8` for training and held-out cases.
- Straight channel: symmetry relative L2 below `1e-10`, transverse speed below `1e-3`, maximum/mean axial speed between 1.45 and 1.55, flux mismatch below 0.01, residual below `1e-5`.
- All three exported field grids contain 960 cells with finite velocities/concentrations. Color scales are shared across designs.

The final three-seed campaign's 18 resolution comparisons were feasible. This is grid/setup sensitivity, not a demonstrated constant-Reynolds/Peclet refinement experiment. No convergence order or GCI is inferred. The plots preserve infeasible comparisons if another run produces them.

## Code and integration checks

Executed on Linux x86-64, Rust/Cargo 1.99.1 stable, Python 3.12.14, Matplotlib 3.11.2:

- Locked standalone native release build.
- Thirteen existing Rust tests: eight library, four CLI, one verification.
- Stable Clippy for all standalone targets with warnings denied, and Rust formatting check.
- Three new Python tests exercising corrupted scalar/selection, full-dimensional dominance and invalid constraint vectors.
- Documentation inventory/relative-link checker; mdBook build and built-book link checker.
- GitHub Actions workflow syntax checked with actionlint 1.7.12 (`shellcheck` unavailable locally).
- A real CI-sized campaign, separate from the tutorial measurements; no mocked simulation output.

The workflow uses stable Rust, a locked build, plain Rust diagnostics plus the repository's problem matcher, and uploads experiment evidence even on failure. All new Markdown pages are listed in the book summary to avoid source links becoming missing rendered-book pages.

## Integration re-run

When the workshop was integrated (23 October 2026), the standalone workspace was built again with `--locked` on Rust/Cargo 1.97.0, and both campaigns were rerun with matplotlib 3.11.2:
- the 13 native tests, Clippy with warnings denied, the formatting check and the three audit tests pass;
- the three-seed campaign again counted 12,288 search evaluations, and its `summary.json` equals [validation.json](validation.json) apart from binary hashes and timings;
- in both campaigns every exported CSV is byte-identical except the `elapsed_seconds` column of `convergence.csv`, and the four figures are byte-identical to those beside this page.

## Interpretation limits

Native unit tests, replay consistency, and a channel sanity check establish different things from model validation. This workshop does not claim physical calibration, production suitability, global optimality, a statistically established solver ranking, or a registered job/MCP interface. The selected designs are assessed at only three held-out release locations; the entire front/archive is not held-out validated.

No fresh-context model trial was performed. The prompts were implemented through explicit native runs and audited artifacts. A real agent trial should give an independent session only the setup and prompt, then compare its retained evidence against the same checks.
