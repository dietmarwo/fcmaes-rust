# Dependency notice

The tutorial's source is MIT-licensed. Its exact `Cargo.lock` contains one
direct file-level copyleft dependency:

| Crate | Version | License | Path |
|---|---:|---|---|
| `pykep-core` | 0.1.4 | MPL-2.0 | direct dependency |

`pykep-core` is used unmodified through its public Rust API. Its source and
MPL-2.0 license are distributed through the normal Cargo registry source
mechanism. MPL-2.0 applies at the covered-file level; this tutorial does not
copy or modify those files.

`deny.toml` records a crate-specific `pykep-core` exception. MPL-2.0 is not
added to the general license allow-list, so any future copyleft dependency
remains a review failure.

## Unmaintained transitive macro crate

`cargo deny` also reports `RUSTSEC-2024-0436`: `paste = 1.0.15` is archived and
unmaintained. It is reached through both pinned numerical libraries:

```text
pykep-core → differential-equations → simba → paste
fcmaes-core → nalgebra → simba → paste
```

The advisory reports no safe upgrade and no vulnerability. The tutorial does
not call `paste` directly. `deny.toml` therefore names this advisory explicitly
instead of accepting all unmaintained dependencies; new advisories still fail
the check.
