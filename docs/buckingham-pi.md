# Buckingham–Pi optimization

The `buckingham-pi` example finds dimensionless groups from a dimension matrix
and searches for continuous group exponents that explain data. The complete
numerical path runs in Rust. It does not require Python, BuckinghamPy, NumPy,
SciPy, pandas, or scikit-learn.

This is intentionally narrower than a symbolic dimensional-analysis package.
The input is a variable list and an integer matrix \(A\), whose rows are base
dimensions and whose columns are variables. An exponent vector \(e\) defines
a dimensionless product when

\[
A e = 0,\qquad
\pi(x,e) = \exp(\log(x)e) = \prod_i x_i^{e_i}.
\]

The implementation provides the capabilities needed for the optimization
example:

- numerical rank and nullspace calculation;
- exhaustive full-rank repeating-variable enumeration;
- conventional π groups from \(A_r e=-a_j\);
- compact rational formatting plus reciprocal/scalar-equivalent group
  deduplication;
- continuous parameterization \(E=N_s C\), which keeps every trial
  dimensionally valid;
- standardized ordinary least squares, train/holdout \(R^2\), coefficient of
  variation, conditioning, complexity, and dimensional-residual diagnostics;
- independent BiteOpt retry and constrained MODE search.

See the [source and data notice](../examples/data/BUCKINGHAM_NOTICE.md) for
provenance and scope.

## Run it

List the seven built-in engineering problems:

```bash
cargo run --release -p fcmaes-examples --bin buckingham-pi -- \
  --list-problems
```

Run the complete cylinder workflow with the same settings as the original
Python optimization example—32 independent retries, 2,000 evaluations per
retry, and at most 16 retry workers—plus MODE:

```bash
cargo run --release -p fcmaes-examples --bin buckingham-pi -- \
  --problem cylinder --mode all --groups 2 --samples 300 \
  --workers 16 --retries 32 --evaluations 2000 \
  --mo-evaluations 20000 --popsize 128 --seed 42
```

`--workers 0` uses the available CPU parallelism. A positive value sets the
retry worker count and the MODE batch-evaluation thread count. The main modes
can also be run separately:

```bash
# Algebra only: no generated data and no optimizer
cargo run --release -p fcmaes-examples --bin buckingham-pi -- \
  --problem pipe --mode enumerate

# Rank conventional repeating-variable bases on held-out data
cargo run --release -p fcmaes-examples --bin buckingham-pi -- \
  --problem packed-bed --mode rank --rank-limit 10

# Continuous exponent search with independent BiteOpt retries
cargo run --release -p fcmaes-examples --bin buckingham-pi -- \
  --problem cylinder --mode optimize --groups 2 --workers 16

# Pareto search over predictive quality, simplicity, and independence
cargo run --release -p fcmaes-examples --bin buckingham-pi -- \
  --problem cylinder --mode multi --groups 2 --workers 16
```

The catalog slugs are `pipe`, `pump`, `flow`, `packed-bed`, `cylinder`,
`natural-convection`, and `rayleigh-benard`. Dimensionless independent columns
such as `Pr` are reported and removed from the continuous search rather than
silently discarded. Construct `BuckinghamProblem` directly to analyze another
dimension matrix.

## Data and validation

The checked example generates positive inputs independently and log-uniformly
over \(10^{-3}\) through \(10^3\). A response is generated from all columns of
the problem's nullspace basis with additive Gaussian noise equal to 2% of the
training signal's population standard deviation. Train and holdout inputs use
different deterministic PCG streams.

The holdout split corrects an issue in the exploratory Python script: that
script described cross-validation but fitted and scored the regression on the
same rows. Rust reports both training and holdout \(R^2\), and the optimizer
uses holdout \(R^2\). In the output, `mean_cv` means the mean coefficient of
variation of π features; it does not mean cross-validation.

The synthetic response makes the repository example deterministic and
self-contained. It is not evidence that a π group predicts a physical
response. For a real application, retain an untouched validation experiment
or simulation campaign and replace both generated splits with measured data.

## Objectives and safeguards

For \(k=\dim\ker(A)\) and \(m\) requested groups, the optimizer searches the
\(k m\) entries of \(C\) in `[-9, 9]`, then calculates \(E=N_sC\).

The scalar BiteOpt objective is

\[
1-R^2_{\mathrm{holdout}}
+100\,p_{\mathrm{spread}}
+p_{\mathrm{conditioning}}.
\]

`p_spread` penalizes a π feature whose coefficient of variation is below 0.1
or above 10. The conditioning term activates only when the smallest-to-largest
singular-value ratio of the standardized π-feature matrix is below 0.05. This
prevents a nominal multi-group solution from succeeding by returning duplicate
or nearly dependent features.

MODE minimizes:

1. \(1-R^2_{\mathrm{holdout}}\);
2. \(\sum_{ij}|E_{ij}|\), favoring interpretable exponents;
3. feature dependence, defined as one minus the singular-value ratio.

The spread violation is a MODE constraint and is feasible at zero. This keeps
predictive fit, simplicity, and independence visible instead of hiding their
trade-off in one arbitrary weighted sum.

Candidate log-features outside `[-80, 80]`, non-finite values, singular
regressions, and dimensionally invalid exponent matrices are rejected. OLS
features are standardized using training statistics, and those same
statistics are applied to holdout rows.

## Reproducible sample

On the development host, the full cylinder command above produced:

| Search | Actual evaluations | Holdout \(R^2\) | Train \(R^2\) | Notes |
|---|---:|---:|---:|---|
| BiteOpt retry | 64,000 | 0.999947896 | 0.999609257 | 32 completed independent retries |
| MODE | 20,096 | 0.999842624 | 0.999052457 | best predictive point among 31 feasible Pareto points |

The quality values reproduce for the stated seed and code revision; wall time
is intentionally omitted because this is an example validation, not a
controlled cross-machine benchmark. The CLI prints elapsed time, evaluation
throughput, dimensional residuals, and every final exponent vector.

## When to use which path

- Use `enumerate` when the dimension matrix is small and conventional,
  rational π groups are the desired result.
- Use `rank` when experimental data should choose between conventional
  repeating-variable bases.
- Use `optimize` when real-valued exponents and one predictive answer are
  acceptable.
- Use `multi` when interpretability and group independence matter alongside
  predictive quality.

Optimization cannot repair a wrong dimension matrix, biased measurements, or
data that do not cover the intended physical regime. Always validate the
selected groups on separately acquired data and inspect their dimensional
residual and conditioning.
