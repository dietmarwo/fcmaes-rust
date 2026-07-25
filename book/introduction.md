# Native optimization in Rust

`fcmaes-rust` provides parallel, gradient-free optimization algorithms whose
numerics, retry coordination, random streams, fitness evaluation, and
multithreading all run in Rust. The reusable `fcmaes-core` crate has no C or
C++ build dependency. An optional PyO3 package gives Python users the same
native implementation.

This book connects three layers that serve different needs:

- the [generated `fcmaes-core` API reference](https://docs.rs/fcmaes-core)
  gives exact Rust types, methods, runnable examples, and algorithm
  references;
- the user guides explain selection, configuration, parallelism, retry,
  ask/tell operation, encodings, and Python integration;
- the application tutorials preserve executable simulation models, commands,
  raw results, validation protocols, and deterministic figures.

If you have a concrete problem but have not chosen an optimizer, start with
[Choosing an optimizer](docs/choosing-an-optimizer.md). If you want to run
code immediately, use [Getting started](docs/getting-started.md). For a
realistic end-to-end study, choose a chapter from the tutorial navigation.

The optimizer library is published on
[crates.io](https://crates.io/crates/fcmaes-core), while examples and
tutorials are maintained in the
[GitHub repository](https://github.com/dietmarwo/fcmaes-rust).
