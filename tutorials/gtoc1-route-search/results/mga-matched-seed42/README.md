# Seed-42 impulsive-MGA route discovery

This is the reviewed publication bundle for the 100-route seed-42 experiment
documented in the [route-search tutorial](../../README.md). A finite score means
only that the Rust optimizer found an impulsive MGA scaffold under the declared
budget; it is not a low-thrust-feasible or competition-valid trajectory.

`random`, `evolutionary`, and `gemma4` form the blind matched comparison.
`gemma4-assisted` is a separately named follow-up which uses the completed
random and evolutionary archives as prior evidence. It must not be presented
as an independent fourth arm.

Each arm retains:

- the terminal `run.json` and immutable `protocol.json`;
- compact and full accepted-route archives;
- convergence and proposal-rejection logs; and
- the Rust→adapter exchange log, empty by construction for non-agent controls.

Per-request response caches are deliberately omitted because they duplicate
adapter responses. The provider prompt and candidate menu are reconstructed
deterministically from these exchanges, the adapter, and any digest-pinned
prior archives; they are not stored verbatim. The original local work
directories remain outside the repository.

Regenerate or verify the comparison from the tutorial directory:

```bash
python3 compare_campaigns.py
python3 compare_campaigns.py --check
python3 plot_results.py --check
```

The experiment has one root seed and parallel coordinated retry is not
bit-reproducible. Repeat predeclared seeds before making a general proposer or
model-capability claim.
