# Data provenance

No external routing instance or customer data are redistributed.

The ten checked-in files `instances/instance-00.csv` through
`instance-09.csv` are synthetic outputs of `src/instance.rs`. Their frozen
seeds are:

```text
11, 29, 47, 71, 101, 131, 173, 211, 257, 307
```

Each file contains:

- generator metadata;
- eight vehicle records;
- 50 base task records and two reserve urgent-task records; and
- a complete feasible witness route.

The generator builds routes first and derives task windows from their arrival
times. Tests replay every witness and require zero skill, capacity, time-window,
and shift violation. The witness proves satisfiability; it is used only as a
structured optimizer seed and comparator, not as a claimed optimum.

Regenerate byte-identical files with:

```bash
cargo run --release --locked -- --mode generate
```

The coordinates, demands, windows, costs, and measured performance values are
synthetic. They must not be interpreted as a real field-service operation.
