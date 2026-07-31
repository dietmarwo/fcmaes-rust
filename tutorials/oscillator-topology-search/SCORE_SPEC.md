# Oscillation-score specification

This file freezes the scalar inner objective.

## Sampling

- burn-in: 64 model-time units;
- retained samples: 128;
- interval: 1 time unit;
- target period: 24;
- DFT bins: integer bins 2 through 16, corresponding to periods 64 through 8;
- amplitude: 90th percentile minus 10th percentile;
- a gene participates if amplitude is at least 5 and spectral concentration is
  at least 0.05.

For every participating gene, the peak DFT bin is quadratically interpolated
in log power. Spectral concentration is the peak and adjacent-bin power
normalized by total centered-signal energy. Autocorrelation decay uses the
correlation at one and two measured periods.

## Replicate aggregation

Let:

- \(e_p=|\bar p-24|/24\);
- \(s\) be mean spectral concentration;
- \(a\) be mean amplitude over all three genes;
- \(d\) be mean autocorrelation decay;
- \(q\) be participating-gene fraction;
- \(c\) combine period coefficients of variation across genes and stochastic
  replications;
- \(f\) be failed-replication fraction; and
- \(m\) be mean total molecule count.

The minimized score is

\[
e_p + 2(1-s) + \max(0,(10-a)/10) + 0.5d
+ 2(1-q) + c + 5f + 0.0002m.
\]

A replicate fails if the simulation runs away above 100,000 total molecules,
does not produce all 128 samples, or has zero participating genes. Invalid
parameters receive an equivalent failed replicate.

Training uses the named `TRAINING_SEEDS`; validation uses the disjoint
`VALIDATION_SEEDS`. The validation-minus-training scalar score is the reported
generalization gap.

## Independent gates

Tests pin:

- a three-channel analytic sine's period and amplitude;
- lower score at the target period than at a shifted period;
- lower score for three participating channels than one;
- rejection of a flat trace;
- monotone activation and inhibition propensities; and
- bit-identical replay for one seeded runtime model.

The fixed Vilar tutorial is not an oracle because it simulates a different
nine-species reaction network.
