# Buckingham–Pi example provenance

The dimension-matrix catalog and the continuous-exponent formulation in
`examples/src/buckingham.rs` are adapted from
[dietmarwo/BuckinghamExamples](https://github.com/dietmarwo/BuckinghamExamples),
revision `f3cab5da91a80b96e82f0c55ea3ebfc2b255affd` (2025-07-03).
That project is distributed under the MIT License.

The Rust implementation is a numerical port, not a binding to BuckinghamPy.
It accepts an integer dimension matrix directly and implements matrix-rank and
nullspace calculation, repeating-variable enumeration, π-group construction,
regression diagnostics, holdout scoring, and optimization in native Rust. It
does not port BuckinghamPy's symbolic unit parser or its text-reporting API.

The built-in response data are synthetic and deterministic. They are intended
to exercise the dimensional-analysis and optimization pipeline, not to
represent experimental measurements. Users should replace the generated
train/holdout matrices with separately validated experimental or simulation
data for scientific applications.

Copyright (c) Dietmar Wolz. Licensed under the MIT License.
