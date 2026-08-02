# Oscillator topology-search comparison

All scores are minimized (**lower is better**). Held-out reference encodings are excluded from proposal histories. A dash means a proposal arm did not exactly rediscover that topology.

| Arm | Repressilator | Goodwin-like | Positive cycle | Toggle control | Classes | Accepted | Best | Median | Score < 1 | Agent tokens (input+output) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random (complete) | — | — | — | — | 5 | 200 | 0.613988 | 2.950889 | 10 | 0 |
| evolutionary (complete) | — | — | — | — | 5 | 200 | 0.483957 | 2.636004 | 59 | 0 |
| agent (complete) | 188 | — | — | — | 5 | 200 | 0.471418 | 0.957210 | 103 | 2081201 |
