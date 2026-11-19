#!/usr/bin/env python3
"""Post-hoc check for round 3. Not part of the declared rules.

Does CatCMAwM's default start explain its round-3 results? The campaign used
the defaults of `cmaes.CatCMAwM`: the mean at the center of the box, an
initial standard deviation of one twelfth of the range (one tenth for
integers), the default population, and the same start for every restart.
This script reruns the candidate with three alternatives on seeds that are
not campaign seeds and prints medians; it decides nothing.

Usage (the harness binary as for run_catcmawm.py):
    python3 posthoc_round3_start.py PROBLEM TOTAL SEEDS [VARIANT,...]
"""
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import run_catcmawm as rc

HERE = Path(__file__).resolve().parent


def run(space, serve, problem, total, seed, variant):
    rng = np.random.RandomState(seed)
    best = math.inf; used = 0; runs = 0
    args = space.constructor_arguments()
    while total - used >= 100:
        kw = dict(args); kw["seed"] = int(rng.randint(0, 2**31 - 1))
        probe = rc.check_cmaes()(**kw)
        nmi = probe._Nmi
        if nmi > 0 and variant["random_mean"]:
            lo = []; hi = []
            if "x_space" in args: 
                xs = np.asarray(args["x_space"], float); lo += list(xs[:,0]); hi += list(xs[:,1])
            if "z_space" in args:
                for z in args["z_space"]: lo.append(min(z)); hi.append(max(z))
            lo = np.array(lo); hi = np.array(hi)
            kw["mean"] = lo + rng.rand(nmi) * (hi - lo)
        if nmi > 0 and variant["sigma"] is not None: kw["sigma"] = variant["sigma"]
        if variant["popsize"] is not None: kw["population_size"] = variant["popsize"]
        opt = rc.check_cmaes()(**kw); pop = opt.population_size
        guard = rc.Guard(space.lower, space.upper, total - used)
        while True:
            left = guard.budget - guard.calls
            if left <= 0: break
            size = min(pop, left)
            samples = [opt.ask() for _ in range(size)]
            vectors = [space.to_real(s)[0] for s in samples]
            values = guard.evaluate(vectors, lambda v: serve.evaluate(problem, v))
            if size < pop: break
            opt.tell(list(zip(samples, values)))
            if opt.should_stop(): break
        used += guard.calls; best = min(best, guard.best_y); runs += 1
        if guard.calls == 0: break
    return best, runs

VARIANTS = {
 "default":      dict(random_mean=False, sigma=None, popsize=None),
 "rand_mean":    dict(random_mean=True,  sigma=None, popsize=None),
 "rand_wide":    dict(random_mean=True,  sigma=2.0,  popsize=None),
 "rand_wide_p31":dict(random_mean=True,  sigma=2.0,  popsize=31),
}
if __name__ == "__main__":
    problem, total, nseeds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    names = sys.argv[4].split(",") if len(sys.argv) > 4 else list(VARIANTS)
    serve = rc.Serve(rc.default_harness()); info = serve.info(problem)
    space = rc.TypedSpace.load(problem, info, HERE / "results/round3/types")
    for name in names:
        t = time.time(); vals = []; rs = []
        for s in range(nseeds):
            b, r = run(space, serve, problem, total, 777000 + s, VARIANTS[name]); vals.append(b); rs.append(r)
        print(f"{problem} {total} {name:14s} median {statistics.median(vals):.6g} min {min(vals):.6g} max {max(vals):.6g} feasible(<1e8) {sum(v<1e8 for v in vals)}/{nseeds} runs {min(rs)}-{max(rs)}  [{time.time()-t:.0f}s]", flush=True)
    serve.close()
