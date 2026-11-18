# Candidate optimizers, round 3: mixed-variable screening

Declared on 18 November 2026, before any round-3 run. Gates and thresholds
in this file are fixed; a deviation is recorded in
[DEVIATIONS.md](DEVIATIONS.md) and never applied silently. Everything not
restated here is as in [PROTOCOL.md](PROTOCOL.md) and [ROUND2.md](ROUND2.md).

## 1. Question

CatCMA with Margin (CatCMAwM; Hamano, Nomura, Saito, Uchida and Shirakawa,
GECCO 2025) optimizes continuous, integer and categorical variables jointly
with a multivariate Gaussian and categorical distributions. `fcmaes-core`
has no optimizer with native categorical variables: discrete structure is
decoded inside the objective, and DE and MODE accept an integer mask.

> On the mixed-variable real-world problems of this repository, with a few
> thousand to a few tens of thousands of evaluations, does CatCMAwM beat the
> core optimizers working on the decoded real vector?

This is a **screening** round. CatCMAwM runs as its authors' reference
implementation. A Rust implementation in core is written only if the
screening passes.

## 2. Solvers

**Candidate, `catcmawm`:** class `CatCMAwM` of the Python package `cmaes`
(MIT license), version pinned in the report, default parameters, with every
variable given its native type (section 4). A run ends when the
implementation's own stop criterion fires or the budget is spent.

**Incumbents,** on the real-vector encoding each problem's own code decodes:
BiteOpt (`bite`), DE (`de`), active CMA-ES (`cma`), CR-FM-NES (`crfmnes`),
Dual Annealing (`da`), DE→CMA (`de_cma`) and jSO→CMA (`jso_cma`), configured
as in round 2. Where a problem's own optimization code passes an integer
mask to DE, `de` and the DE stage of `de_cma` use the same mask; otherwise
none.

All solvers are compared by objective evaluations, never by time: the
candidate is Python and the incumbents are Rust.

## 3. Problems

A problem enters under a rule fixed here, from code inspection and timing
only:

- it is in this repository (a tutorial, an example or `fcmaes-gtop`), with a
  deterministic scalar cost that its own single-objective code minimizes;
- its decoding gives at least one coordinate a discrete meaning: an
  integer or quantized value, a binary switch, or an unordered choice;
- its mean evaluation time over 2,000 uniform random points is at most 2 ms.

To be examined: truss sizing, the energy hub, the phased-array codebook, the
SINDR circuit designs, water-network scheduling, field-service routing, the
network-coverage problem, the mixed-integer GTOP problems (`cassini1_minlp`,
`cassini2_minlp`, Tandem with free sequence if the code offers it), and any
other problem in the repository that meets the rule.

The round needs at least five problems, at least three of them with an
unordered categorical coordinate. Otherwise it stops and reports that the
repository cannot answer the question.

## 4. Variable types

For every problem a table assigns each coordinate one type, from the
decoding code alone and before any optimization run:

| Type | Meaning | Given to CatCMAwM as |
|---|---|---|
| continuous | the objective uses the real value | a bounded continuous variable |
| integer | ordered discrete levels: counts, quantized values, sorted catalog sizes, binary switches | an integer variable with those levels |
| categorical | unordered choices | a categorical variable with that many categories |

CatCMAwM's typed point is mapped to the problem's real vector by placing
each discrete coordinate at the center of the interval that decodes to the
chosen level, so both sides evaluate exactly the same objective. The table
and the mapping are tested: every level of every discrete coordinate decodes
to itself.

## 5. Protocol

- Two budgets per problem: **2,000** and **20,000** evaluations. A *cell* is
  one problem at one budget.
- A solver runs until its own stop criterion or the budget; if evaluations
  remain, it restarts with a new seed, until fewer than 100 remain. The best
  value counts. This holds for the candidate too.
- 20 repetitions per cell, root seed 3, seeds shared by all solvers. The
  candidate derives its random stream from the same seed value.
- The budget and the box are enforced outside the solver for every arm, and
  the recorded best value is the enforcer's.

**Compute rule.** If the timing run projects more than 180 minutes,
repetitions drop uniformly from 20 to 15, decided from timings alone.

## 6. Statistics and gates

As in ROUND2.md section 5, with one change that follows from round 2, where
the mean-rank gate was met by a margin inside the noise:

| Gate | Condition |
|---|---|
| G1, complementarity | at least 3 winning cells, on at least 2 different problems |
| G2, generalist | mean rank strictly better than the best incumbent's mean rank, **and** this holds in at least 95% of 10,000 resamples of the repetitions (seeded) |

| Decision | Rule |
|---|---|
| **Include** | G1 or G2 holds and no contract violation was recorded: implement CatCMAwM in core |
| **Watch** | at least one winning cell, but neither gate |
| **Reject** for now | no winning cell |

Results are also reported separately for the cells of problems with a
categorical coordinate and for the rest. This split is descriptive and
decides nothing.

## 7. Reporting

Raw rows go to `results/round3/raw.tsv`, the type tables to
`results/round3/types/`, the generated report to `results/round3/report.md`.
Files of rounds 1 and 2 are not changed.
