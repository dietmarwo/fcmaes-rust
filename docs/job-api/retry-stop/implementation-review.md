# Implementation review: fcmaes item 4

## Commits

The cumulative patch is based on `4ad8b7d` and prepares synchronized 0.3.0 source metadata. It contains the core methods, worker wiring, regression updates, reproducible evidence and release notes. No tag or registry publication is performed.

## Verification

Core stop tests, worker tests with an independent call counter, real CLI cancel/deadline jobs, 140 byte-identical baseline/current optimizer records, the full workspace, legacy and shared runner regressions, job/surrogate tutorials and discovery/schema checks are recorded in the combined log.

## Notes for review

The public methods delegate with a never-firing predicate. Serial DE must retain a primary or temporal-locality value even if cancellation occurs before the old population bookkeeping would promote that value; the early-stop test at one evaluation covers this case without altering ordinary seeded arithmetic. Delayed DE stops dispatch, drains in ask order and counts all dispatched evaluations. Basic `retry_until` prevents new claims; an application passes the same predicate into its inner optimizer.

The worker observes its `Control` signal and remaining deadline for every model. Plain callbacks are still evaluated normally; simulations retain checkpoint cancellation and separate failed/cancelled attempt accounting. A stop skips the final recheck and retains claimed evidence. Tests that previously assumed a cancelled retry consumed its nominal whole budget now assert exact partial work.

Testing also exposed that the ES helper's predicate had only been checked between restarts. Population-level methods were added for CMA-ES, CR-FM-NES and PGPE so the every-model stop promise holds; ordinary ES records remain byte-identical to the baseline.

## Merge order

Core methods precede worker wiring; they can merge in one change. Release 0.3.0 after the normal package gates. The maintainer creates release tags. Only after publication should CFD-craft, CRN-craft and experiment-craft pins move; the patch does not point downstream projects at an unpublished local revision.
