# Cooperative retry-stop evidence

The 0.3.0 preparation is based on fcmaes-rust `4ad8b7d` (0.2.1). Ordinary seeded optimizer records remain unchanged. Cancellation/deadline behavior intentionally changes: running plain-model retries now return their evaluated best point and exact partial work, without rechecking that point after stopping.

## Results

`compatibility-0.2.1.txt` and `compatibility-0.3.0.txt` are byte-identical: twenty seeds for serial BiteOpt, deep BiteOpt, serial DE, delayed DE, CMA-ES, CR-FM-NES and PGPE (140 records). Each includes the point, objective bits, evaluation/iteration counts and stop code. The new `stopped=false` diagnostic is excluded from the old-format comparison.

`cli-results.json` records eight real `optimization-run` jobs: serial and batched BiteOpt/DE, each cancelled and timed out, on a plain model sleeping 30 ms per evaluation. Serial cancellation returned in about 23–30 ms in this run; batched work drained before returning. All cases retained exact evaluation counts, zero validation evaluations, zero cancelled plain-model evaluations and claimed evidence. Timeout lifecycle states are `timed_out`; the task termination is `cancelled`, since the SDK uses the same cooperative signal for both reasons.

The dedicated worker regression checks an independent atomic callback counter for plain BiteOpt/DE and all three ES engines. Core tests cover exact serial stops, in-flight batch accounting, immediate stops, best evaluated points and stopping new retry claims.

## Reproduce

```sh
cargo test --workspace --locked
cargo build --release --locked -p optimization-runner -p fcmaes-job-worker
cargo build --release --locked -p fcmaes-job-worker --example stop_worker
python scripts/job-api/test_retry_stop.py --output stop-results.json
cargo run --release --locked -p fcmaes-core --example stop_compatibility > compatibility.txt
```

Run the same compatibility example in a 0.2.1 checkout to pair the records. Host: AMD EPYC 9V74, nine exposed CPUs, Linux x86_64, CPython 3.12.14 and Rust 1.100.0. Timings are observations from a shared QA host, not an isolated throughput benchmark.

A callback cannot be preempted midway. Use simulation checkpoints for a long evaluation; plain model latency includes the current evaluation (or in-flight population). Forced process termination still cannot recover an uncommitted task record.
