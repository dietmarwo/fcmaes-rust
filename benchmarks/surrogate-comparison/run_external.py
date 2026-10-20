#!/usr/bin/env python3
"""Run established Bayesian-optimization libraries on the comparison problems.

Writes the JSON format of fcmaes-surrogate's `examples/benchmark.rs`, so
`compare.py` treats every method alike. Each run is single-threaded; runs are
spread over worker processes. `proposal_seconds` is wall time minus the time
spent inside objective calls.

    python run_external.py --method botorch --q 1 --seeds 7,11 --output out.json
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import PROBLEMS  # noqa: E402

INITIAL = 8


class Recorder:
    """Counts calls, stores the history and the time spent in the objective."""

    def __init__(self, objective, lower, upper, budget):
        self.objective, self.lower, self.upper, self.budget = objective, lower, upper, budget
        self.history, self.seconds = [], 0.0

    def __call__(self, x):
        x = [min(max(float(v), lo), hi) for v, lo, hi in zip(x, self.lower, self.upper)]
        began = time.perf_counter()
        value = float(self.objective(x))
        self.seconds += time.perf_counter() - began
        if len(self.history) < self.budget:
            self.history.append({"id": len(self.history), "x": x, "value": {"value": value}})
        return value

    def unit(self, u):
        return self(lo + float(v) * (hi - lo) for v, lo, hi in zip(u, self.lower, self.upper))


def run_botorch(f, d, budget, q, seed, noiseless=False):
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)
    X = sobol.draw(INITIAL)
    Y = torch.tensor([[-f.unit(x)] for x in X])
    bounds = torch.stack([torch.zeros(d), torch.ones(d)])
    while len(f.history) < budget:
        # Default: inferred noise. `noiseless`: fixed tiny observation noise,
        # BoTorch's recommendation for deterministic objectives.
        yvar = torch.full_like(Y, 1e-8 * float(Y.var())) if noiseless else None
        model = SingleTaskGP(X, Y, train_Yvar=yvar, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        k = min(q, budget - len(f.history))
        acq = qLogExpectedImprovement(model, best_f=Y.max())
        candidates, _ = optimize_acqf(acq, bounds=bounds, q=k, num_restarts=10, raw_samples=512)
        X = torch.cat([X, candidates])
        Y = torch.cat([Y, torch.tensor([[-f.unit(x)] for x in candidates])])


def run_skopt(f, d, budget, q, seed):
    from skopt import Optimizer

    opt = Optimizer([(0.0, 1.0)] * d, n_initial_points=INITIAL, initial_point_generator="lhs",
                    random_state=seed, acq_optimizer="lbfgs")
    while len(f.history) < budget:
        k = min(q, budget - len(f.history))
        xs = opt.ask(n_points=k, strategy="cl_min") if k > 1 else [opt.ask()]
        opt.tell(xs, [f.unit(x) for x in xs])


def run_egobox(f, d, budget, q, seed):
    import numpy as np
    import egobox as egx

    def batch(xs):
        return np.array([[f.unit(x)] for x in np.atleast_2d(xs)])

    options = dict(n_doe=INITIAL, seed=seed, infill_strategy=egx.InfillStrategy.LOG_EI)
    if q > 1:
        options["qei_config"] = egx.QEiConfig(batch=q, strategy=egx.QEiStrategy.KB)
    egor = egx.Egor([[0.0, 1.0]] * d, **options)
    iterations = (budget - INITIAL + q - 1) // q
    egor.minimize(batch, max_iters=iterations)


def run_optuna(f, d, budget, q, seed, gp):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if gp:
        sampler = optuna.samplers.GPSampler(seed=seed, n_startup_trials=INITIAL)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=INITIAL, multivariate=True)
    study = optuna.create_study(sampler=sampler)
    while len(f.history) < budget:
        trial = study.ask()
        u = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(d)]
        study.tell(trial, f.unit(u))


METHODS = {
    "botorch": run_botorch,
    "botorch-noiseless": lambda *a: run_botorch(*a, noiseless=True),
    "skopt": run_skopt,
    "egobox": run_egobox,
    "optuna-gp": lambda *a: run_optuna(*a, gp=True),
    "optuna-tpe": lambda *a: run_optuna(*a, gp=False),
}


def one(job):
    method, problem, seed, q = job
    objective, lower, upper, optimum, budget = PROBLEMS[problem]
    f = Recorder(objective, lower, upper, budget)
    began = time.perf_counter()
    METHODS[method](f, len(lower), budget, q, seed)
    wall = time.perf_counter() - began
    best = min(o["value"]["value"] for o in f.history)
    return {
        "problem": problem, "dimension": len(lower), "method": f"{method}-q{q}", "seed": seed,
        "evaluations": len(f.history), "best": best, "regret": max(best - optimum, 0.0),
        "wall_seconds": wall, "proposal_seconds": wall - f.seconds, "history": f.history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--problems", default=",".join(PROBLEMS))
    parser.add_argument("--seeds", default="7,11,19,23,41")
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    jobs = [(args.method, p, int(s), args.q) for p in args.problems.split(",") for s in args.seeds.split(",")]
    with multiprocessing.get_context("spawn").Pool(args.processes) as pool:
        runs = []
        for run in pool.imap(one, jobs):
            runs.append(run)
            print(f"{run['method']} {run['problem']} seed {run['seed']}: regret {run['regret']:.4g}, "
                  f"proposal {run['proposal_seconds']:.1f} s", flush=True)
    with open(args.output, "w") as out:
        json.dump({"schema_version": "fcmaes.surrogate.benchmark/1", "concurrent": args.q,
                   "seeds": [int(s) for s in args.seeds.split(",")], "runs": runs}, out)


if __name__ == "__main__":
    main()
