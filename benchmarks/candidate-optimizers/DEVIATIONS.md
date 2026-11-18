# Deviations from the protocol

Every entry below was decided on 18 November 2026 before the campaign ran,
from published sources, reference implementations and synthetic test
functions only. No campaign result existed when they were recorded.

1. **Repetitions: 12 instead of 20.** The idle-host measurement gives 43 s of
   objective time per solver, protocol and repetition, which projects to more
   than 150 minutes for 20 repetitions. The compute rule of section 3.3
   applies.
2. **BIPOP-aCMA-ES initial σ: 0.2 of the box width, not 0.3.** Section 3.1
   requires the values published by the authors; the BBOB setting is σ = 2 on
   [−5, 5]. The schedule follows the DEAP example and pycma: the first run is
   not booked to a regime, a small run is capped at half the evaluations of
   the most recent large run, and ties go to the large regime.
3. **L-SRTDE runs in two forms, as two candidates.** The open descriptions
   print the mutation with the p-best difference taken from the current
   vector (`lsrtde`). That form fails a 30-variable rotated ellipsoid. The
   form of the predecessor LL-NADE, with the difference taken from the base
   vector (`lsrtde_b`), solves it. The conference paper is not openly
   available and the GPL reference code was not consulted, so the benchmark
   cannot settle which form the authors ran. Both enter the campaign and the
   gates apply to each; ranks are therefore over twelve solvers. An inclusion
   decision that holds for only one form is reported as unsettled.
4. **AMaLGaM-IDEA is plain AMaLGaM.** The model is re-estimated from scratch
   every generation, as published for the population rule 17 + 3·D^1.5; the
   incremental learning rates belong to iAMaLGaM, which uses a different
   population rule. The fraction of shifted samples is ½·τ·n/(n − 1).
5. **Dual Annealing is truncated at the budget.** Core `optimize_da` can
   exceed `max_evaluations` in its gradient and line-search path. The adapter
   counts only the first B evaluations of the library run.

Known limits that follow from the protocol and are not changed:

- On the phased-array problem (96 variables, B = 16,000), protocol R gives
  2,000 evaluations per run. That is less than one AMaLGaM population
  (2,838) and about one generation for L-SHADE (1,728) and L-SRTDE (1,920).
- L-SRTDE details that no open source confirms (spread of F, memory
  averaging, index rules) are listed in the module documentation.
