# Why this tutorial uses a custom flow backend

The optimization changes inlet and outlet openings, inlet velocity, and a
rasterized internal baffle for every candidate. A useful tutorial backend
therefore needs to:

- rebuild geometry cheaply from a nine-variable decision vector;
- keep all mutable solver state inside one objective evaluation;
- run deterministically for ordered parallel population batches;
- avoid nested worker pools while fcmaes owns candidate parallelism;
- expose velocity and pollutant fields for reproducible diagnostics; and
- remain small enough that readers can follow the complete
  simulation-to-objective path.

The tutorial implements that deliberately narrow backend directly in Rust.
Steady incompressible flow uses a D2Q9 lattice-Boltzmann kernel with
bounce-back walls, wall vents, and a rasterized baffle. Pollutant transport
uses a D2Q5 passive-scalar advection-diffusion kernel. One converged flow field
is reused for the three training releases or the three held-out releases.

This choice makes the example self-contained and keeps Python outside the
optimization hot path. It also has important consequences:

- the solver is part of the tutorial model, not an independently maintained
  general-purpose CFD package;
- boundary conditions, lattice scaling, and fan-power/pressure values are
  educational simplifications;
- unit tests, a straight-channel property check, held-out releases, and a
  three-grid sensitivity study provide numerical evidence but not experimental
  validation; and
- conclusions concern optimizer integration, robustness checks, behavior
  diversity, and numerical sensitivity—not real-building performance.

The backend boundary is the immutable `RoomProblem`. Every call allocates
isolated flow and scalar state, returns optimizer-facing metrics, and can
optionally retain one final field:

```text
decision vector
      |
      v
geometry + vent masks + baffle rasterization
      |
      v
D2Q9 steady flow
      |
      +---- reused for each pollutant release
      v
D2Q5 passive scalar
      |
      v
objectives + constraints + QD descriptors
```

Replacing the backend later would require reproducing the same geometry,
physical scaling, objectives, constraints, source sets, and validation
protocol. Merely obtaining similar optimizer scores would not establish
solver equivalence.
