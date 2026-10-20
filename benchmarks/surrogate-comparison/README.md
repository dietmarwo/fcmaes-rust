# fcmaes-surrogate versus established Bayesian-optimization libraries

Equal call budgets on small deterministic problems with known minima, each
method with its documented defaults and eight initial points. The question is
the one the surrogate profile is for: given a fixed number of expensive
evaluations, how close does each method get, and what does it cost per run
in proposal time?

| Method | Version | Configuration |
|---|---|---|
| `fcmaes-qQ` | this branch | `fcmaes-surrogate` defaults: Matérn-5/2 ARD GP, MAP lengthscales by L-BFGS, LogEI, trust region for two of three proposals, nugget 1e-10, Kriging-believer batches of Q |
| `botorch-qQ` | BoTorch 0.18.1, GPyTorch 1.15.2, PyTorch 2.14.1 (CPU) | `SingleTaskGP` defaults (RBF ARD kernel with the dimension-scaled LogNormal lengthscale prior, inferred noise of at least 1e-4) with `Standardize`, `fit_gpytorch_mll`, `qLogExpectedImprovement`, `optimize_acqf(num_restarts=10, raw_samples=512)`, eight scrambled Sobol points |
| `botorch-noiseless-q1` | as above | as above with fixed observation noise `1e-8 * var(y)`, BoTorch's recommendation for deterministic objectives |
| `egobox-qQ` | egobox 1.0.0 (Rust, Python wheel) | `Egor(n_doe=8, infill_strategy=LOG_EI)`, `QEiConfig(batch=Q, strategy=KB)` for Q>1 |
| `skopt-q1` | scikit-optimize 0.10.2 | `Optimizer(n_initial_points=8, initial_point_generator="lhs", acq_optimizer="lbfgs")`, default `gp_hedge` |
| `optuna-gp-q1` | Optuna 5.0.0 | `GPSampler(n_startup_trials=8)` |
| `optuna-tpe-q1` | Optuna 5.0.0 | `TPESampler(n_startup_trials=8, multivariate=True)` |
| `biteopt`, `cmaes`, `random` | fcmaes-core | non-surrogate references from `examples/benchmark.rs`, started from the same eight points as `fcmaes` |

Problems: Branin (d=2, 48 calls), Hartmann-3 (d=3, 48), Hartmann-6 (d=6,
96), Styblinski–Tang (d=10, 128), Rosenbrock (d=8, 128) and the tutorial's RK4
damped-oscillator calibration (d=2, 32). `problems.py` mirrors the Rust
objectives, and `compare.py` re-evaluates the first twelve observations of
every Rust run in Python and asserts agreement. Ten seeds per problem and
method; BoTorch and Optuna GP skip Rosenbrock, noiseless BoTorch also skips
Styblinski–Tang (compute time). Every run is single-threaded; two runs share
the two CPUs of the measuring container. Simple regret is the best value
found minus the known minimum; lower is better.

## Results

Median final simple regret over ten seeds, sequential proposals (q=1); bold
marks the lowest median. Full tables with half-budget regret, geometric means
and worst runs are in [`comparison.md`](comparison.md).

| Problem (d, calls) | fcmaes | BoTorch | BoTorch noiseless | egobox | skopt | Optuna GP | Optuna TPE | BiteOpt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Branin (2, 48) | 1.5e-06 | 2.7e-04 | 6.0e-05 | **6.6e-08** | 5.1e-04 | 7.7e-05 | 0.15 | 0.13 |
| Hartmann-3 (3, 48) | **1.5e-07** | 4.7e-03 | 7.9e-03 | 4.2e-05 | 1.5e-04 | 2.8e-05 | 0.09 | 0.10 |
| Hartmann-6 (6, 96) | **5.0e-07** | 0.020 | 2.9e-03 | 0.060 | 4.7e-04 | 1.2e-03 | 0.16 | 0.65 |
| Styblinski–Tang (10, 128) | 57 | 88 | – | 82 | 92 | 117 | 81 | **17** |
| RK4 oscillator (2, 32) | **9.7e-09** | 2.5e-05 | 5.1e-06 | 6.6e-07 | 1.6e-05 | 3.4e-06 | 1.4e-03 | 1.8e-03 |

Median proposal time per run in seconds (wall time minus objective time, one
thread per run):

| Problem | fcmaes | BoTorch | BoTorch noiseless | egobox | skopt | Optuna GP | Optuna TPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Branin | 0.04 | 33 | 33 | 0.78 | 14 | 6.1 | 0.08 |
| Hartmann-3 | 0.07 | 71 | 68 | 0.91 | 13 | 5.3 | 0.07 |
| Hartmann-6 | 0.33 | 115 | 90 | 6.3 | 48 | 14 | 0.25 |
| Styblinski–Tang | 0.89 | 200 | – | 88 | 130 | 39 | 0.50 |
| RK4 oscillator | 0.02 | 11 | 20 | 0.26 | 5.2 | 3.1 | 0.05 |

Batches of four (Kriging believer for fcmaes and egobox, joint qLogEI for
BoTorch), median final regret:

| Problem | fcmaes | BoTorch | egobox |
|---|---:|---:|---:|
| Branin | 3.7e-07 | 4.9e-03 | **2.2e-07** |
| Hartmann-3 | **1.6e-07** | 9.2e-03 | 6.1e-05 |
| Hartmann-6 | **9.1e-07** | 0.19 | 0.12 |
| Styblinski–Tang | **77** | 145 | 129 |
| RK4 oscillator | 5.6e-07 | 2.2e-04 | **3.0e-07** |

fcmaes-surrogate reaches the lowest median final regret of the GP methods on
four of five problems and is within a factor of 25 of egobox on Branin, at
1–3 orders of magnitude less proposal time than the other GP libraries. At
half budget Optuna's GP sampler and noiseless BoTorch are ahead on the two-
and three-variable problems (for example 1.2e-4 and 2.5e-4 against 4.0e-4 on
the oscillator after 16 calls): two of three fcmaes proposals stay in the
trust region, which pays off later. BoTorch's defaults infer an observation
noise of at least 1e-4 (standardized), which limits precision on these
deterministic problems; fixing the noise helps on Branin and Hartmann-6 but
not on Hartmann-3, where most noiseless runs settle in the local minimum at
regret 7.9e-3.

## Weaknesses found and what changed

The first comparison (fcmaes-surrogate as reviewed in the first pass) showed
three weaknesses; the defaults were changed and re-measured on the same seeds:

1. **Final precision on smooth problems.** With the M0 nugget of 1e-8 the GP
   could not resolve the last digits near the optimum (Branin 1.9e-5 against
   egobox's 6.6e-8). The nugget is now a configuration parameter with default
   1e-10, which lowered the final regret on 7–10 of 10 seeds on Branin,
   Hartmann-3/6 and the oscillator. Objectives with solver noise may need a
   larger value.
2. **Global search in 6–10 dimensions.** Pure global LogEI spent too many
   calls far from the incumbent (Hartmann-6 median 5.9e-5, Styblinski–Tang-10
   101). A TuRBO-1-style trust region for two of three proposals, with every
   third proposal still global, improved both on every seed; together with
   the smaller nugget the medians are now 5e-7 and 57 (rows `fcmaes-pass1`
   in [`comparison.md`](comparison.md) show the build before this pass). A pure trust region trapped a Hartmann-3 run in a
   local minimum; the global third proposal prevents that.
3. **A wrong reference value.** The often quoted Hartmann-3 minimum
   -3.86278214782076 is below the true minimum of the standard constants,
   -3.8627797873326624, which put a floor of 2.4e-6 under every Hartmann-3
   regret. All tables use the corrected value.

Changes that were measured and not adopted are listed in
[the decisions record](../../docs/surrogate/decisions.md).

## Remaining limits

- Styblinski–Tang-10 is separable with 2^10 local minima. BiteOpt reaches a
  median regret of 17 in 128 calls; every GP method stays above 50. For
  separable or many-modal problems with tens of variables, prefer
  `fcmaes.retry/1` with BiteOpt.
- Rosenbrock-8 results reflect the centre point in fcmaes' initial design:
  the centre of the symmetric box has value 7, far below typical values, and
  nobody improved on it within 128 calls. The external libraries do not
  evaluate the centre.
- Small test functions are not expensive simulations, and ten seeds support
  medians, not significance tests. Proposal times are wall times on a shared
  two-CPU container.

## Reproduce

```bash
# fcmaes-surrogate and the in-run references (from the repository root)
cargo build --release --locked -p fcmaes-surrogate --example benchmark
S10=7,11,19,23,41,43,47,53,59,61
P=branin,hartmann3,hartmann6,styblinski10,rosenbrock8,oscillator
for q in 1 4; do
  target/release/examples/benchmark --threads 1 --concurrent $q --problems $P \
    --seeds $S10 --output OUT/fcmaes-q$q.json
done

# external libraries (Python 3.13 virtual environment)
pip install "botorch==0.18.1" "scikit-optimize==0.10.2" "egobox==1.0.0" "optuna==5.0.0"
cd benchmarks/surrogate-comparison
python run_external.py --method egobox --q 1 --seeds $S10 --output OUT/egobox-q1.json
python run_external.py --method botorch --q 1 --seeds $S10 \
  --problems branin,hartmann3,hartmann6,styblinski10,oscillator --output OUT/botorch-q1.json
# ... likewise for egobox --q 4, botorch --q 4, botorch-noiseless, skopt,
# optuna-gp and optuna-tpe
python compare.py OUT --output comparison.md --csv results.csv
```

`compare.py` recomputes every regret from the recorded observations and the
known minima in `problems.py`, so result files from either side agree on the
reference values. [`results.csv`](results.csv) holds one row per run and
[`comparison.md`](comparison.md) the full tables.
