# Application tutorials

The tutorials put native Rust objective functions around simulators and
domain models, then treat optimization as a complete experiment: define the
decision vector and constraints, choose an execution model, keep independent
validation cases, record raw evidence, and render deterministic figures.

Use the chapters in the sidebar instead of reading one long index. The
twenty-two application tutorials cover:

| Tutorial | Main optimization ideas |
|---|---|
| NeXosim production line | simulation allocation, MODE, MAP-Elites |
| Rapier trebuchet | discontinuous physics, MODE, MAP-Elites |
| ReBop oscillator | noisy kinetic rates, robust MODE, MAP-Elites |
| Oscillator topology search | runtime signed networks, deterministic parallel BiteOpt retry, held-out motifs, and optional agent proposals |
| Brahe constellation | nonsmooth access windows, MODE, MAP-Elites |
| RustPower voltage control | mixed variables, constraints, MODE, MAP-Elites |
| Atmospheric source localization | inverse modeling, advanced retry, MODE, MAP-Elites |
| Room-ventilation CFD | custom verified backend, robust design, MODE, MAP-Elites |
| ML hyperparameter tuning | leakage-free evaluation, BiteOpt, MODE, QD diagnostics |
| Neural-controller policy search | 118-dimensional PGPE and CR-FM-NES |
| GTOC1 “Save the Earth” | planet-order search, staged DE–CMA-ES, low thrust, split-brain architecture |
| Split-brain GTOC1 route search (work in progress) | provider-independent proposals, completed seed-42 L0 controls, predeclared random-arm L1 promotions, and explicit validation limits |
| sindr circuit design | smooth AC features, equal-budget retry, constrained MODE, E12 MAP-Elites |
| thevenin gate driver | transient measurement, constrained MODE, timestep and ngspice validation |
| Pure-Rust optical lens design | validated sequential ray tracing, equal-budget retry, constrained MODE |
| Rapier quadruped gait | contact-derived MAP-Elites repertoire, motor work, held-out terrain validation |
| Phased-array codebook | quantized controls, robust retry, constrained MODE, descriptor-gated MAP-Elites |
| Bilevel energy hub | embedded pure-Rust LP, discrete outer sizing, robust retry, MODE, chronological H₂ replay |
| Field-service routing | assignment and priority random keys, robust scenarios, constrained MODE, descriptor-gated MAP-Elites |
| Water-network scheduling | stepwise hydraulics, quantized controls, safety overrides, constrained MODE, descriptor validation |
| Truss topology and sizing | exact-k topology decoding, typed FEM failures, scalar retry, constrained MODE, removal-robustness gate |
| Network coverage | 4,000 binary decisions, certified graph covers, native group-pair scoring, and a measured specialist-baseline comparison |

The canonical, detailed [tutorial index](tutorials/README.md) retains the
common experiment contract, comparison table, complete commands, and Diffsol
discussion. Each chapter then supplies the model-specific rationale, source
layout, results, caveats, and reproduction instructions.
