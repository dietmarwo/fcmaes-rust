# Deviations and interpretations

Written before the affected runs. None of these changes a rule, budget,
seed, threshold or the decision of `PROTOCOL.md`; they fix details the
protocol leaves open.

## 1. Seeds and start point of A

"Seeds 1 to 10" are the seeds of the earlier diagnostic that reported the
medians 1.13 (clip) and 0.137 (repair): seed `s` runs the job worker's
`es_retry` with the retry seed `retry_run_seed(s, 0)`. That retry seed
drives the uniform random start (`Rng::new(retry_seed ^ 0x9E37_79B9_7F4A_7C15)`)
and is the CMA-ES seed. The controller binary uses the same mapping for all
three modes, so the reproduction of those medians is checked with the same
seeds the decision uses.

## 2. Protocol T settings of B

"jSO→CMA as one run, protocol T, 1,000,000 evaluations, root seed 7" is run
as `candidate-harness run --protocols T --problems truss_sizing --root-seed 7
--total 1000000 --run-cap 1000000 --reps 10` with the default minimum run of
1,000 evaluations. The run cap equals the total, so each repetition is one
run of the sequence; protocol T starts a further run only if the sequence
stops with 1,000 or more evaluations left, as it does for every arm of the
harness.

## 3. Damped-spring probe

The protocol names "the first-duration restriction of the damped spring in
`examples/src/damp.rs`" without the values of the other coordinates. The
probe fixes them at the reference point of that file's test
(`x = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.1, 0.9]`, four segments) and varies
`x[0]` on `[0, 1]`, started at 0 as declared. The objective is the
amplitude (`damp::fitness`).

## 4. Evaluations of C and D

The coordinated retry of `gtop-advexamples` prints one summary line per
problem with the best value, the evaluations used and the number of runs
(retries); these are the recorded values. The retries are those the schedule
completes, which can differ between modes.

## 5. Budget of the probes

The protocol gives no budget for the two boundary-start probes. They run
3,100 evaluations (100 generations of 31), the budget of the existing unit
test of the first probe in `fcmaes-core`.
