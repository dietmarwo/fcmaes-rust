# Frozen field-service routing cost specification

This document is the authority for both `evaluate.rs` and the independently
written `scorer2.rs`. Results under `results/publication/` were generated only
after this specification was frozen.

## Instance

The depot has a two-dimensional coordinate. Task `i` has a coordinate in km,
service duration `s_i` in seconds, demand `q_i`, service-start window
`[e_i,l_i]`, and one required skill bit. Vehicle `v` has capacity `Q_v`, shift
`[E_v,L_v]`, a skill-bit set, fixed dispatch cost `f_v`, and per-km cost `c_v`.

The publication instances have 50 nominal tasks, two inactive reserve urgent
tasks, and eight vehicles. The resulting optimizer vector always has 104
coordinates. A scenario activates or deactivates task-mask entries; it never
changes the vector length.

## Travel

The default leg distance is Euclidean:

```text
d(a,b) = sqrt((x_a-x_b)^2 + (y_a-y_b)^2) km
travel_seconds(a,b) = 3600 d(a,b) / 48
```

The traffic scenario multiplies travel seconds by `1.3`, not distance or
distance cost. The rounding holdout rounds each leg to the nearest integer km
before computing both its time and cost.

This is a transparent teaching model, not a road-network router. It omits
one-way streets, turn costs, time-dependent roads, breaks, and pickup-delivery
precedence.

## Route forward pass

For vehicle `v` serving `t_1,...,t_r`:

```text
clock = E_v
previous = depot
for task in route:
    clock += travel_seconds(previous, task)
    waiting += max(0, e_task - clock)
    clock = max(clock, e_task)
    lateness += max(0, clock - l_task)
    clock += s_task
    load += q_task
    distance += d(previous, task)
    previous = task
clock += travel_seconds(previous, depot)
distance += d(previous, depot)
duration = clock - E_v
```

Waiting is free but consumes shift time. A task arriving late starts
immediately; lateness is measured at service start. An empty route has zero
cost and does not count as a used vehicle.

## Cost, objectives, and constraints

Hard-window monetary cost is

```text
cost = sum over non-empty routes (f_v + c_v * route_distance_v)
```

Skills and exactly-once service are feasible by construction in the decoder.
The three normalized constraints are:

```text
capacity = sum_v max(0, load_v-Q_v) / 100
lateness = sum_tasks max(0, start_i-l_i) / 3600
shift = sum_v max(0, return_v-L_v) / 3600
```

Each is feasible at exactly zero, up to a `1e-9` arithmetic tolerance. The
scalar robust objective is the worst cost over the five training scenarios
plus `10,000` times the sum of their worst normalized violations.

The soft-window MODE arm minimizes nominal distance, vehicles used, makespan,
and aggregate lateness. It constrains capacity and shift but deliberately does
not constrain lateness.

MAP-Elites, when permitted by the descriptor gate, minimizes worst training
cost and excludes every hard-infeasible plan.

## Hand-check used by tests

At speed `60 km/h`, one task at `(4,3)` is 5 km from the depot. A vehicle
starting at 08:00 arrives at 08:05, waits 55 minutes for a 09:00 window, serves
for 10 minutes, and returns in 5 minutes. The route therefore has:

```text
distance = 10 km
waiting = 3300 s
duration = 4500 s
```

When the latest start is 08:59:59, lateness is exactly one second. The unit
test asserts these values without reference to `scorer2`.
