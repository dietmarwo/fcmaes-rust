# Native advanced surrogate results

Two confirmation panels, each with 30 fresh seeds per problem and variant
and 48 callbacks per run. All variants remain experimental: neither panel
passes the noninferiority gate of the
[plan](../../plans/fcmaes-surrogate-improvement-plan-2026-11-16.md#7-acceptance-and-promotion-rules)
(upper 95% bound of the paired geometric loss ratio <= 1.05 in every
stratum), so no setting was promoted.

| Panel | Seeds | Source | Records |
| --- | --- | --- | --- |
| Delivered | 201–230 | the delivered implementation ([provenance](provenance.json)) | [raw](confirmation.jsonl.gz), [manifest](confirmation-manifest.json), [summary](summary.json) |
| Trust-region review | 301–330 | after the review fixes, commit `ec9111a` ([provenance](trust-region-review/provenance.json)) | [raw](trust-region-review/confirmation.jsonl.gz), [manifest](trust-region-review/confirmation-manifest.json), [summary](trust-region-review/summary.json) |

The review found that the delivered LogEI policy never narrowed its search
(its region only grew, and refinement always covered the whole box), and
that its decisions depended on the order in which a synchronous batch's
callbacks completed. After both fixes, on fresh seeds:

| Problem | Variant | Median loss | Median seconds | Paired geometric loss ratio [95% bootstrap] |
| --- | --- | ---: | ---: | ---: |
| sphere | legacy | 9.988e-07 | 0.02535 | 1 [1, 1] |
| sphere | advanced | 4.009e-06 | 0.1233 | 3.525 [2.071, 5.985] |
| sphere | thompson | 0.0001793 | 0.297 | 193.9 [124.2, 306.5] |
| sphere | handoff | 0.001704 | 0.05156 | 1838 [1076, 3079] |
| sphere | governor | 2.478e-06 | 0.1142 | 3.059 [1.659, 5.586] |
| ackley | legacy | 4.441e-16 | 0.03625 | 1 [1, 1] |
| ackley | advanced | 4.441e-16 | 0.1106 | 1 [1, 1] |
| ackley | thompson | 4.441e-16 | 0.2492 | 1 [1, 1] |
| ackley | handoff | 4.441e-16 | 0.0538 | 1 [1, 1] |
| ackley | governor | 4.441e-16 | 0.099 | 1 [1, 1] |
| styblinski | legacy | 49.91 | 0.03724 | 1 [1, 1] |
| styblinski | advanced | 47.15 | 0.141 | 1.061 [0.8678, 1.313] |
| styblinski | thompson | 58.95 | 0.3327 | 1.352 [1.087, 1.731] |
| styblinski | handoff | 82.71 | 0.04927 | 1.769 [1.475, 2.174] |
| styblinski | governor | 48.72 | 0.1444 | 1.089 [0.9079, 1.32] |
| oscillator | legacy | 0.0002281 | 0.01978 | 1 [1, 1] |
| oscillator | advanced | 0.0002305 | 0.08115 | 1.013 [0.9984, 1.028] |
| oscillator | thompson | 0.0002363 | 0.2534 | 1.041 [1.009, 1.071] |
| oscillator | handoff | 0.0002388 | 0.04192 | 1.095 [1.032, 1.171] |
| oscillator | governor | 0.0002285 | 0.08342 | 1.002 [0.9742, 1.025] |

The delivered panel, before the fixes:

| Problem | Variant | Median loss | Median seconds | Paired geometric loss ratio [95% bootstrap] |
|---|---|---:|---:|---:|
| sphere | legacy | 1.27966e-06 | 0.02456 | 1 [1, 1] |
| sphere | advanced | 0.000229201 | 0.0907 | 231.3 [99.15, 542.3] |
| sphere | thompson | 0.0870127 | 0.3232 | 7.602e+04 [4.61e+04, 1.224e+05] |
| sphere | handoff | 0.0173237 | 0.03012 | 9713 [4377, 1.996e+04] |
| sphere | governor | 0.000287899 | 0.08427 | 287.8 [150.7, 538.5] |
| ackley | legacy | 4.44089e-16 | 0.03358 | 1 [1, 1] |
| ackley | advanced | 4.44089e-16 | 0.07718 | 1 [1, 1] |
| ackley | thompson | 4.44089e-16 | 0.2706 | 1 [1, 1] |
| ackley | handoff | 4.44089e-16 | 0.03525 | 1 [1, 1] |
| ackley | governor | 4.44089e-16 | 0.07569 | 1 [1, 1] |
| styblinski | legacy | 45.5081 | 0.03144 | 1 [1, 1] |
| styblinski | advanced | 68.3764 | 0.1112 | 1.377 [1.22, 1.569] |
| styblinski | thompson | 91.3893 | 0.3382 | 1.941 [1.654, 2.281] |
| styblinski | handoff | 74.8899 | 0.03539 | 1.604 [1.437, 1.802] |
| styblinski | governor | 64.6587 | 0.1065 | 1.378 [1.204, 1.575] |
| oscillator | legacy | 0.000227957 | 0.01552 | 1 [1, 1] |
| oscillator | advanced | 0.000239302 | 0.06076 | 1.069 [1.029, 1.115] |
| oscillator | thompson | 0.000233679 | 0.2581 | 1.062 [1.012, 1.113] |
| oscillator | handoff | 0.000238755 | 0.02668 | 1.058 [1.011, 1.116] |
| oscillator | governor | 0.000249027 | 0.06172 | 1.086 [1.041, 1.138] |

How to read them:

- Each ratio pairs a variant with `legacy` on the same seed and adds 1e-12
  to both losses, so exact optima do not divide by zero. The panels use
  different seeds, so compare each variant with its own panel's `legacy`.
- `scripts/surrogate/summarize_advanced_panel.py` computes the review summary, with a percentile bootstrap of 10,000
  resamples and a recorded seed. On the delivered data it reproduces the
  medians exactly and the ratios to 2e-15. The delivered bootstrap
  procedure was not recorded, so the delivered intervals are as reported.
  `scripts/surrogate/verify_advanced_evidence.py` checks both summaries against their raw data.
- Ackley starts at its optimum for every method (the centre of the initial
  design), so it says nothing about search.
- Times are measured on shared hosts: the delivered panel on 9 CPUs, the
  review panel on 2.
- The asynchronous pilot (synthetic 5/30 ms callback latency, ratio 0.78)
  and the factor-append microbenchmark were recorded with the delivered
  source and not rerun. The pilot is a scheduling check, not a speed claim
  for real simulations; the microbenchmark does not measure whole-proposal
  time.
