# `thevenin` dependency notice

This tutorial depends exactly on `thevenin-cirq = 0.5.0` and
`thevenin-types = 0.5.0`; both pull the `thevenin 0.5.0` simulation engine.
No `thevenin` source is copied or modified here.

As checked on 2026-07-27:

- Cargo metadata declares `BSD-3-Clause`;
- the upstream repository README states `BSD-3-Clause`;
- the published crate archive does not contain a `LICENSE` file or the BSD
  notice text.

The first two items are an explicit license declaration by the publisher. The
missing notice is nevertheless a packaging and provenance defect because a
redistributor needs the applicable copyright notice to satisfy BSD retention
conditions. This repository therefore:

1. pins the exact dependency version;
2. does not vendor or redistribute its source;
3. records the omission instead of silently treating it as resolved;
4. should re-check the published archive before updating the dependency.

This notice documents engineering provenance; it is not legal advice.

