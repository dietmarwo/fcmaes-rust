# Equal-wall publication bundle provenance

This is the authoritative 100-pair, seven-problem equal-wall result bundle.
It contains 1,400 unique protocol rows: two arms for every problem and seed.

The first campaign produced the valid Cassini1, Cassini2, GTOC1, Messenger,
Rosetta, and Tandem blocks. Its SAGAS block exposed a physically impossible
negative time-to-50-AU objective (`-823.3041728682549`). The port had preserved
the legacy `-1` unreachable-orbit sentinel check but did not reject other
negative results from the orbital time calculation.

The Rust port was corrected to map every non-finite or negative travel time to
the established `100000` failure objective and to reject a negative final
SAGAS objective. The focused unit test covers negative and non-finite travel
time. SAGAS was then rerun for the same 100 seeds and protocol:

```bash
cargo run --release --locked --manifest-path benchmarks/gtop-cmaes-retry/Cargo.toml -- \
  --mode campaign --preset publication \
  --phases equal-wall --arms external-single,external-retry \
  --problems sagas --runs 100 --workers 1,16 \
  --wall-time-ms 4000 --evaluations 1000000000 --seed 5000057 \
  --output benchmarks/gtop-cmaes-retry/results/equal-wall-sagas-corrected-v1
```

The checked-in merge utility replaced exactly the 200 SAGAS rows and rejected
schema changes, row-count changes, or duplicate protocol keys:

```bash
python3 benchmarks/gtop-cmaes-retry/merge_problem_results.py \
  --base path/to/original-results.csv \
  --replacement benchmarks/gtop-cmaes-retry/results/equal-wall-sagas-corrected-v1/results.csv \
  --problem sagas \
  --output benchmarks/gtop-cmaes-retry/results/equal-wall-100-v2/results.csv
```

Finally, `--resume` validated every row against the complete publication
matrix and generated [`run.json`](run.json) and
[`comparison.md`](comparison.md). The superseded raw campaign is intentionally
not used for statistics or target-success claims.
