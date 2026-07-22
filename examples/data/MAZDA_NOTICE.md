# Mazda benchmark data notice

`mazda_model.bin` is a compact, machine-readable translation of the three-car
Mazda constrained discrete multi-objective benchmark (Mazda CdMOBP). It embeds
the benchmark's discrete decision choices, normalization ranges, regression
coefficients, radial-basis training matrices, and three published reference
rows. It contains no compiled C++ code.

The native Rust evaluator deduplicates training matrices shared by related
response surfaces. The binary can be regenerated from the retained original
benchmark distribution with `mazda/port_mazda_model.py` in the development
workspace; that porting utility and the original C++/Python sources are not
runtime dependencies of the public repository.

The original benchmark distribution asks first-time users to acknowledge
their name and affiliation to `benchmark@flab.isas.jaxa.jp`. Please follow that
request when using the Mazda benchmark in research or published comparisons.

The Rust implementation is covered by the repository license. This notice
records the provenance and acknowledgement request for the embedded benchmark
data; it does not replace any terms accompanying the original distribution.
