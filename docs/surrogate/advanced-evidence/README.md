# Native advanced surrogate results

30 fresh seeds per problem/variant, 48 callbacks each. All variants remain experimental.

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

Ratios use a 1e-12 numerical floor. Ackley starts at the optimum for every method. Timing is measured on a shared host. The async panel uses synthetic 5/30 ms callback latency and is a scheduling pilot, not a real-simulation speed claim. The factor-append result is a numerical microbenchmark, not whole-proposal speed.
