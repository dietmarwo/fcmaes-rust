# Candidate optimizers, round 3, second batch: categorical problems

Declared on 18 November 2026, while the round-3 campaign was running and
before any round-3 result had been examined. Gates and thresholds are fixed;
a deviation is recorded in [DEVIATIONS.md](DEVIATIONS.md). Everything not
restated here is as in [ROUND3.md](ROUND3.md) and its recorded deviations.

## 1. Why a second batch

The round-3 survey found only two problems with an unordered categorical
coordinate in this repository. CatCMAwM's distinguishing feature is its
joint handling of categorical and continuous variables, so round 3 alone
cannot speak for it. This batch adds published real-world problems that have
such variables and are cheap to evaluate.

## 2. Problems

| Problem | Variables | Source to implement from |
|---|---|---|
| Welded beam with material and weld-type choice | weld type (two levels), bar material (four materials), beam dimensions | the mixed-variable welded-beam design problem of Deb and Goyal (1996), as restated in later mixed-variable optimization papers |
| Mixed cantilever beam | one of twelve listed cross-section shapes, length, cross-section area | `MixedCantileverBeam` of the Surrogate Modeling Toolbox (SMT, BSD-3-Clause) |
| Pest control | 25 stations, each with five choices (none or one of four pesticides) | the pest-control benchmark of Oh, Tomczak, Gavves and Welling (2019, COMBO), as used in later mixed-variable Bayesian-optimization suites |

Rules for a problem to enter:

- its definition (formulas, constants, bounds, constraint limits) is taken
  from a published source or from openly licensed reference code, and is
  checked against a second independent source or against values computed by
  the reference code at fixed points; the sources are named in the code;
- it is deterministic: the same point always gives the same value;
- a problem that cannot be verified in this way is dropped and the reason
  recorded. The batch needs at least two of the three problems, otherwise it
  stops.

An objective that exists only as Python reference code and cannot be ported
without changing its values may be evaluated through that code for all
solvers; the comparison is by evaluations, so the evaluation cost does not
matter.

**Constraints.** A constrained problem is scalarized in the same way for
every solver: the cost is the objective plus 10⁶ times the sum of the
positive parts of the constraints, each written as `g(x) ≤ 0` and divided by
its limit as in the source.

**Types and encoding.** The typing rule of ROUND3.md section 4 applies
unchanged: ordered levels and binary switches are integer, unordered choices
are categorical. For the core optimizers every discrete coordinate is a real
coordinate split into equal intervals, one per level, in the order of the
source.

## 3. Solvers, protocol, statistics

As in round 3: the same seven incumbents and `catcmawm`, totals of 2,000 and
20,000 evaluations, restarts until spent, 20 repetitions, root seed 3, the
same tests. Raw rows go to `results/round3b/raw.tsv`.

## 4. Decision

- The gates of ROUND3.md are evaluated for round 3 alone and reported as
  declared there.
- This batch is reported alone with the same tables.
- **The decision on CatCMAwM is taken on the union of the round-3 cells and
  the cells of this batch**, with the gates of ROUND3.md section 6
  (G1: at least 3 winning cells on at least 2 different problems; G2: mean
  rank strictly better than the best incumbent's, holding in at least 95% of
  10,000 resamples) and the same Include, Watch and Reject rules.
- The union is also split, descriptively, into cells of problems with a
  categorical coordinate and the rest.
