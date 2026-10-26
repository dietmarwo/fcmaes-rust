# Final-candidate certificate audit

Problem: `scalar-cuberoot`. Accepted: **9/9**.

| Arm | Verification | Replay | Accepted | Accounted seconds | Observation |
| --- | --- | --- | --- | ---: | --- |
| reference | verified | verified | yes | 31.050694 | event |
| grid | verified | verified | yes | 31.458761 | event |
| local | verified | verified | yes | 29.642787 | event |
| random-17 | verified | verified | yes | 30.804835 | event |
| random-29 | verified | verified | yes | 31.005649 | event |
| cmaes-17 | verified | verified | yes | 33.354937 | event |
| cmaes-29 | verified | verified | yes | 30.971874 | event |
| biteopt-17 | verified | verified | yes | 31.303157 | event |
| biteopt-29 | verified | verified | yes | 30.674011 | event |

Timing adds the shared search startup in full to each arm, that arm's warm search/recheck/export work, and its fresh Julia verification and replay (including process startup).
These are final-candidate observations, not first-found or anytime certificate times. Scheduling, transfers, other arms' work and coordinator overhead are excluded; the complete audit wall time is recorded separately.
Unsuccessful and unattempted arms stay in the denominator and are right censored at the observed pipeline end. Stopping may be informative: no survival estimate, successful-only median, ranking or optimizer-advantage claim is made.
The equations, target and proof scope are unchanged. This is not parameter-region coverage. A completed collection is not scientific acceptance.
