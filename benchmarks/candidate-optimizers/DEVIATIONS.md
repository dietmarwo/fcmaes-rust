# Deviations from the protocol

Every entry below was decided on 18 November 2026 from published sources,
reference implementations and synthetic test functions only. Entries 1 to 5
were recorded before the campaign ran. Entry 6 was recorded while it ran,
before any campaign result had been examined.

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
   gates apply to each; ranks are therefore over twelve solvers. Entry 6
   supersedes the rule for reading the two forms.
4. **AMaLGaM-IDEA is plain AMaLGaM.** The model is re-estimated from scratch
   every generation, as published for the population rule 17 + 3·D^1.5; the
   incremental learning rates belong to iAMaLGaM, which uses a different
   population rule. The fraction of shifted samples is ½·τ·n/(n − 1).
5. **Dual Annealing is truncated at the budget.** At the time of the
   campaign, core `optimize_da` could exceed `max_evaluations` in its
   gradient and line-search path. The adapter counts only the first B
   evaluations of the library run.

6. **L-SRTDE after the conference paper became available** (recorded while
   the campaign was running, before any campaign result had been examined).
   Both forms now follow Algorithm 1 of the CEC 2024 paper: initial success
   rate 0.5, newest population sorted every generation, crossover-rate
   spread 0.05, strict acceptance, and a top population built from
   successful trials only. The paper prints the mutation with the p-best
   difference taken from the current vector, as the open descriptions do.
   A replication check on 30-variable CEC-2017-style functions (300,000
   evaluations, 10 seeds, seeded rotations and shifts in place of the
   official data) separates the two forms:

   | Function | Paper, Table X | `lsrtde` (as printed) | `lsrtde_b` (base difference) |
   |---|---|---|---|
   | Rotated Bent Cigar, mean error | 0 | 4.3e4 | 0 |
   | Rotated Rastrigin, mean error | 8.36 | 186 | 3.9 |

   Only `lsrtde_b` reproduces the published level of performance, so it is
   the L-SRTDE candidate for the inclusion decision. `lsrtde` stays in the
   tables for information and cannot earn an inclusion. That the printed
   equation differs from what the authors ran is an inference from these
   results; the GPL reference code was not consulted. The rows of both forms
   produced before this change are kept in `results/superseded-lsrtde.tsv`
   and are not used; both forms were rerun with the same seeds and budgets.

Known limits that follow from the protocol and are not changed:

- On the phased-array problem (96 variables, B = 16,000), protocol R gives
  2,000 evaluations per run. That is less than one AMaLGaM population
  (2,838) and about one generation for L-SHADE (1,728) and L-SRTDE (1,920).
- L-SRTDE details that the paper leaves open are listed in the module
  documentation.

## Round 2

No deviation from [ROUND2.md](ROUND2.md). Both optional problems failed the
timing rule and did not enter. The compute rule kept 20 repetitions: one
full-scale timing repetition (root seed 999, not a campaign row) projected
64 minutes.

## Round 3

Recorded on 18 November 2026 after the problem survey and before any
optimization run of round 3. The survey used code inspection and timing
only.

1. **The categorical requirement is not met; the round continues as a
   narrower screening.** [ROUND3.md](ROUND3.md) requires three problems with
   an unordered categorical coordinate. Two exist under the entry rule:
   field-service routing (52 categorical coordinates) and the job shop
   (three two-way choices). The question is therefore limited to
   *mixed-integer problems with some categorical structure*. A statement
   about categorical variables rests on those two problems alone, and the
   outcome cannot be read as a verdict on CatCMAwM for categorical design
   problems in general.
2. **Dimension cap of 200 variables.** The two network-coverage problems
   (4,000 integer coordinates) are left out: the reference implementation
   performs a full eigendecomposition several times per generation, which is
   infeasible at that size. Six problems run: truss sizing, phased-array
   geometry, field-service routing, job shop, UAV assignment and material
   flow.
3. **Binary switches stay integer, as declared.** The `cmaes` library
   recommends treating binary variables as categorical. The protocol's
   typing rule is kept so that the type tables do not depend on the
   candidate.
4. **Candidate seed.** `CatCMAwM` accepts 32-bit seeds; the candidate uses
   the low 32 bits of the seed that the other solvers receive for the same
   run.

Known limits of this round, unchanged by the entries above:

- The costlier problems with the richest mixed structure (energy hub, water
  network, ML tuning) are outside the 2 ms entry rule on this host.
- The candidate is a Python reference implementation and is compared by
  evaluations only.

## Round 3, second batch

No deviation from [ROUND3B.md](ROUND3B.md). All three problems entered.

- **Welded beam.** The definition follows Deb and Goyal. Their published
  solution is reproduced for the bending and buckling constraints and, within
  the rounding of the printed weld length, for the shear constraint; the
  printed deflection slack (0.2346) and cost (1.9422) differ in the fourth
  digit (0.2344 and 1.94188). The bounds of the weld length come from Tran,
  Tran and Wang (2019), because the original gives none. Later restatements
  of the problem differ from the original in several formulas; the README
  lists the differences.
- **Pest control.** The function as first published draws unseeded random
  numbers. The batch uses the fixed-seed form of the Bounce and Casmopolitan
  suites (seed 0), which is a deterministic function of the 25 choices, and
  reproduces the reference code bit for bit.

## Round 3, convergence test

Recorded on 19 November 2026, after the Rust solvers and before the
candidate had run on `water_network`. No candidate result on that problem
existed and no result of the test had been examined.

1. **The pump priority of `water_network` is typed integer for the
   candidate.** The round-3 survey table types this two-way switch as
   categorical. With it as the only categorical variable, the reference
   implementation of CatCMAwM breaks down: its step-size adaptation for the
   categorical part leaves its valid range (a square root of a negative
   number), the category probabilities become undefined, and a sample selects
   no category. The convergence test therefore uses the table
   `results/round3c/types/water_network.json`, in which the switch is an
   integer with levels 0 and 1, as the round-3 rule for binary switches says.
   The objective and the encoding of the core solvers are unchanged. The
   table was checked on 496 points (300 random, 196 next to interval
   boundaries): moving every discrete coordinate to its level center leaves
   the objective bit-identical.

## Round 4

Recorded on 19 November 2026, after the candidate library was built and
before any run of ROUND4.md. No result of the test existed.

1. **Build route of Minion.** The library is built from source at the
   declared revision (`535504a`, version 2.0.1, MIT license) with its
   documented `pip install .` route (scikit-build and CMake, release build).
   Its CMake file fetches the Eigen headers from a source host that this
   build host cannot reach, so the build uses a local copy of the same Eigen
   release (3.4.0, the tag the library's CMake file pins) obtained from the
   conda-forge package and passed with `-DEigen3_DIR`. Nothing else differs
   from the documented build, and the `minionpy` wheel from PyPI is not
   used.
2. **Seed.** `ARRDE` takes the seed as a C++ `int`, and `-1` means "draw a
   random seed". The driver passes the low 31 bits of the derived run seed,
   `seed & 0x7FFFFFFF`, which is never `-1`. The library seeds a
   thread-local generator in the constructor; two runs with one seed
   reproduce each other exactly (checked in `test_arrde.py`).
3. **Budget.** The library overshoots its own evaluation budget by a few
   evaluations (a 20,000 evaluation run on a test function made 20,007
   calls). The driver's guard refuses every point beyond the run's
   allowance, answers it with `+inf` without evaluating it, and counts it in
   `over_budget`; the evaluation count of a row never exceeds its allowance.
   No callback is used to stop the library.
4. **Run shapes and files.** The Rust runner resumes by (solver, problem,
   protocol, repetition, total), which does not distinguish two run caps, so
   each shape has its own file, `stage1-cap1000000.tsv` and
   `stage1-cap125000.tsv` (stage 2: `stage2-cap4000000.tsv` and
   `stage2-cap500000.tsv`). The decision script reads both.
5. **Infeasible designs.** The cost of `truss_sizing` is the mass of the
   design when all constraints hold and the mass plus 10^8 times the sum of
   squared constraint violations otherwise. A run is reported as ending
   infeasible when its best cost is at least 10^4; this is a violation of
   about 1% in one constraint, and the survey of round 3 shows infeasible
   results of the order 4e8. A marginally infeasible design with a smaller
   penalty is counted as feasible. The count is informational and does not
   enter the decision.
6. **Restart on the corrected CMA-ES.** The first start of this round ran
   the early stage, the one-run shape of stage 1 and 53 core rows of the
   eight-run shape on the core as it was before active CMA-ES was corrected
   for samples moved onto the bounds (`REPAIR-Z` in
   `crates/fcmaes-core/src/cmaes.rs`). That correction changes the arms
   `cma`, `de_cma` and `jso_cma`; BiteOpt, DE, CR-FM-NES, Dual Annealing and
   ARRDE do not use the changed code. On the maintainer's instruction the
   round is restarted from the beginning on the corrected core, with the
   declared seeds, budgets, arms and decision rule unchanged. The rows of
   the first start are kept in `results/round4/before-cma-fix/` and do not
   enter the decision. No decision had been computed from them; their
   one-run medians had been looked at (jSO→CMA 453, ARRDE 481, DE→CMA 543,
   BiteOpt 560). Arms that do not use the changed code must reproduce
   their first-start rows exactly; the report states whether they do.
7. **Two more core arms at stage 2: jSO→CMA with `repair_z` off.** Recorded
   on 19 November 2026 after stage 1 and after the ten stage-2 runs of
   `jso_cma` as one run (median 420.35), before any other stage-2 run. The
   correction of item 6 has since become the option `CmaesParams::repair_z`
   (default `true`). On this problem the CMA-based arms were better without
   it at stage 1 (jSO→CMA as one run: 452.8 in the first start against
   470.5). On the maintainer's instruction stage 2 therefore also runs
   `jso_cma_z0`, which is `jso_cma` with `repair_z: false` in its CMA-ES
   stage, in both shapes, with the stage's seeds and budgets.
   - Both settings are core configurations. The declared rule is applied
     to all four core arms of stage 2: ARRDE is superior there only if the
     median of its arm is lower than the medians of `jso_cma` and
     `jso_cma_z0` in both shapes and the test against the one with the
     lowest median is significant. The report also gives the outcome
     against the two declared arms alone.
   - Stage 1 is not rerun with the added arm. Its outcome does not depend
     on it: ARRDE (480.6) is not below the declared `jso_cma` arm (470.5),
     and the first start shows `jso_cma` without the correction at 452.8.
   - `jso_cma_z0` must reproduce the first-start rows of `jso_cma` exactly
     (same seeds, the update of 0.3.0); two repetitions of the one-run
     shape of stage 1 are rerun into a scratch file to check this.
