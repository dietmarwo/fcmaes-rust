# Suite and data provenance

The formulas are clean-room Rust implementations from the primary definitions
listed below. This directory contains no copied competition implementation,
shift vector, rotation matrix, COCO binary, or externally licensed data file.

| Suite | Definition source | Repository policy |
|---|---|---|
| Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Schwefel, Levy, Zakharov | standard mathematical definitions, stated explicitly in `classic.rs` | formulas only |
| ZDT1–4, ZDT6 | Zitzler, Deb and Thiele, *Comparison of Multiobjective Evolutionary Algorithms*, Evolutionary Computation 8(2), 2000, DOI 10.1162/106365600568202 | formulas and analytic fronts |
| DTLZ1–7 | Deb, Thiele, Laumanns and Zitzler, *Scalable Test Problems for Evolutionary Multiobjective Optimization*, 2005, DOI 10.1007/1-84628-137-7_6 | formulas and analytic fronts |
| WFG1–9 | Huband et al., *A Review of Multiobjective Test Problems and a Scalable Test Problem Toolkit*, IEEE TEC 10(5), 2006, DOI 10.1109/TEVC.2005.861417 | gated; no unverified implementation ships |
| CEC shift/rotation data | competition distributions | loader and synthetic test fixture only; users obtain data under its own terms |
| BBOB/COCO | Hansen et al., COCO/BBOB documentation | gated; no COCO binary or near-compatible implementation ships |
| Lennard-Jones clusters | University of Cambridge, [table of Lennard-Jones cluster minima](https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html), with the literature links attached to that table | clean-room pair formula and scalar targets only; no coordinates |

## Lennard-Jones target and coordinate policy

The five scalar target values and point-group labels are transcribed from the
linked Cambridge table. They are described as source-cited *putative* minima.
For the publication artifact, the linked point files were downloaded to a
temporary directory and evaluated by this implementation. All five residuals
are below `4.9e-7`; exact source URLs and SHA-256 hashes are retained in
[`reference-audit.json`](results/publication/lennard-jones/reference-audit.json).
No coordinate file, image, database dump, or structure-derived fixture is
redistributed.

The local loader accepts either conventional XYZ (`count`, comment, then
`symbol x y z`) or rows of `x y z` / `symbol x y z`. A user may run
`--reference-file` to evaluate one separately obtained structure, or
`--reference-directory` during a campaign for files named by atom count.
Artifacts keep `reference_structure_audited` distinct from target provenance;
a missing file is an error and never selects built-in coordinates.

## CEC local-data format

The loader accepts UTF-8 text with comments beginning `#`:

```text
dimension 2
shift 1.5 -2.0
rotation 1.0 0.0
rotation 0.0 1.0
```

There must be exactly one shift row and exactly `dimension` rotation rows,
each with `dimension` finite values. Missing files, malformed rows, and
non-orthogonal matrices are actionable errors; no identity fallback exists.

## Current evidence gates

`results/publication/wfg/run.json` and `results/publication/bbob/run.json`
record `status: "skipped"` and `reason: "reference-fixtures-unavailable"`.
Their schema-v2 manifests also record preset, seed, workers, replay command,
and the fact that implementation was not attempted because independent
fixtures are a prerequisite. This is an evidence result, not a promise that
plausible formulas were tested.
The gate can change only in a later reviewed commit that adds primary-source
fixed-point fixtures with clear redistribution terms.
