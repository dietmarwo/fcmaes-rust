# Application tutorials

The tutorials put native Rust objective functions around simulators and
domain models, then treat optimization as a complete experiment: define the
decision vector and constraints, choose an execution model, keep independent
validation cases, record raw evidence, and render deterministic figures.

Use the chapters in the sidebar instead of reading one long index. Together
they cover:

| Tutorial | Main optimization ideas |
|---|---|
| NeXosim production line | simulation allocation, MODE, MAP-Elites |
| Rapier trebuchet | discontinuous physics, MODE, MAP-Elites |
| ReBop oscillator | noisy kinetic rates, robust MODE, MAP-Elites |
| Brahe constellation | nonsmooth access windows, MODE, MAP-Elites |
| RustPower voltage control | mixed variables, constraints, MODE, MAP-Elites |
| Atmospheric source localization | inverse modeling, advanced retry, MODE, MAP-Elites |
| Room-ventilation CFD | custom verified backend, robust design, MODE, MAP-Elites |
| ML hyperparameter tuning | leakage-free evaluation, BiteOpt, MODE, QD diagnostics |
| Neural-controller policy search | 118-dimensional PGPE and CR-FM-NES |
| GTOC1 “Save the Earth” | planet-order search, staged DE–CMA-ES, low thrust, split-brain architecture |

The canonical, detailed [tutorial index](tutorials/README.md) retains the
common experiment contract, comparison table, complete commands, and Diffsol
discussion. Each chapter then supplies the model-specific rationale, source
layout, results, caveats, and reproduction instructions.
