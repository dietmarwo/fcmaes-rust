# Descriptor-pilot verdict

- protocol revision: `2`
- decision: `rejected`
- feasible observations: 99 / 384

## Frozen generator mixture
- `structured-local`: 61 / 288 feasible; per-arm feasible [24, 17, 20] from attempts [96, 96, 96]
- `broad-uniform`: 38 / 96 feasible; per-arm feasible [12, 13, 13] from attempts [32, 32, 32]

## Registered descriptor gates
- D1 depth/survival: passed=false, bounds=[0.28, 0.0] to [0.39, 0.3], reachable=[0.3333333333333333, 0.0] to [0.37898189365106316, 0.2473806274939325], lower clipping=[0.0, 0.9595959595959596], upper clipping=[0.0, 0.0], rho=0.044268, minimum arm coverage=5.000%, holdout retention=98.990%
- D2 utilization-spread/survival: passed=false, bounds=[0.0, 0.0] to [0.3, 0.3], reachable=[0.057044638146628815, 0.0] to [0.23470395544378567, 0.2473806274939325], lower clipping=[0.0, 0.9595959595959596], upper clipping=[0.0, 0.0], rho=-0.171413, minimum arm coverage=5.000%, holdout retention=53.535%
- D3 active-count/mass control: passed=false, bounds=[8.0, 0.0] to [40.0, 5000.0], reachable=[36.0, 2149.523504023347] to [40.0, 5120.808549312233], lower clipping=[0.0, 0.0], upper clipping=[0.7474747474747475, 0.08080808080808081], rho=0.451831, minimum arm coverage=3.333%, holdout retention=100.000%
