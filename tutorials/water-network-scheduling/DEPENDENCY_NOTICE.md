# Dependency notice

The tutorial's source is MIT-licensed. Its exact `Cargo.lock` contains one
file-level copyleft transitive dependency:

| Crate | Version | License | Path |
|---|---:|---|---|
| `paris` | 1.5.15 | MPL-2.0 | `epanet-rs` → `simplelog` (feature `paris`) → `paris` |

`paris` is used unmodified as part of the hydraulic backend's logging stack.
Its source and MPL-2.0 license are distributed through the normal Cargo registry
source mechanism. MPL-2.0 applies at the covered-file level; this tutorial does
not copy or modify those files.

`deny.toml` records a crate-specific `paris` exception. MPL-2.0 is not added to
the general license allow-list, so any future copyleft dependency remains a
review failure.

## Unmaintained transitive macro crate

`cargo deny` also reports `RUSTSEC-2024-0436`: `paste = 1.0.15` is archived and
unmaintained. It is reached through both pinned upstream libraries:

```text
epanet-rs → faer → gemm → paste
fcmaes-core → nalgebra → simba → paste
```

The advisory reports no safe upgrade and no vulnerability. The tutorial does
not call `paste` directly. `deny.toml` therefore names this advisory explicitly
instead of accepting all unmaintained workspace dependencies; new advisories
still fail the check.
