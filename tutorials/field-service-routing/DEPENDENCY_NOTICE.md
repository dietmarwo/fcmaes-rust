# Dependency notice

`cargo deny` reports `RUSTSEC-2024-0436`: `paste = 1.0.15` is archived and
unmaintained. It is reached only through the pinned optimization library:

```text
fcmaes-core → nalgebra → simba → paste
```

The advisory identifies no vulnerability and reports no safe upgrade. This
tutorial does not call `paste` directly. `deny.toml` therefore names that one
advisory explicitly instead of accepting all unmaintained dependencies; new
advisories continue to fail the check.
