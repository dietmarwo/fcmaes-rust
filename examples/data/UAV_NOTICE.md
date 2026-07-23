# Multi-UAV benchmark provenance

The native Rust example implements the extended team-orienteering formulation
used by the enhanced
[Multi-UAV-Task-Assignment-Benchmark](https://github.com/dietmarwo/Multi-UAV-Task-Assignment-Benchmark).
The benchmark is based on:

K. Xiao, J. Lu, Y. Nie, L. Ma, X. Wang, and G. Wang, “A Benchmark for
Multi-UAV Task Assignment of an Extended Team Orienteering Problem,”
[arXiv:2009.00363](https://arxiv.org/abs/2009.00363), 2020.

The implementation under `examples/src/uav.rs` was written as native Rust. It
does not compile, link, or call the Python benchmark or its historical C++
fcmaes backend. No benchmark result files or images are bundled.

Generated instances follow the benchmark's vehicle-speed, position, reward,
service-time, map-size, and time-limit distributions. Rust uses the workspace's
PCG generator, so a Rust seed is repeatable but does not generate the same
instance as Python's Mersenne Twister for the same numeric seed.

The evaluator preserves the enhanced continuous formulation's random-key
encoding and arrival-horizon semantics. Objective parity is therefore
structural and statistical, not bit-for-bit seed parity.
