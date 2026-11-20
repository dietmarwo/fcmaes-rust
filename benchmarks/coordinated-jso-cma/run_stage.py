#!/usr/bin/env python3
"""Resumable driver for the coordinated jSO->CMA benchmark (standard library only).

usage: run_stage.py --stage 1 [--binary PATH] [--parallel 2] [--budget-s 480]

One output file per run in results/runs/ (written under a temporary name and
renamed at the end); finished runs are skipped. A lock file keeps a second
driver from starting. --budget-s stops starting new runs after that many
seconds, so that one call fits into a time limit; call again to continue.
The binary defaults to $CARGO_TARGET_DIR/release/gtop-advexamples.
"""
import argparse, os, subprocess, sys, time, fcntl
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "runs")
PROBLEMS = ["cassini1", "cassini2", "rosetta", "tandem", "messenger", "gtoc1", "sagas", "messenger-full"]
ARMS = {"de_cma": [], "jso_cma": ["--jso-cma"]}
STAGES = {1: dict(problems=PROBLEMS, retries=300, seeds=range(1, 21))}


def run_file(problem, arm, seed):
    return os.path.join(RUNS, f"{problem}__{arm}__{seed}.txt")


def complete(path):
    try:
        text = open(path).read()
    except OSError:
        return False
    return "value=" in text and "wall_s=" in text


def one_run(binary, problem, arm, seed, retries):
    path = run_file(problem, arm, seed)
    tmp = path + ".tmp"
    cmd = [binary, "--problem", problem, "--retries", str(retries), "--evaluations", "1500",
           "--workers", "1", "--seed", str(seed), "--max-eval-fac", "50", "--check-interval", "100"]
    if problem == "messenger-full":
        cmd += ["--value-limit", "12"]
    cmd += ARMS[arm]
    t0 = time.time()
    out = subprocess.run(["nice", "-n", "5"] + cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if out.returncode != 0 or "value=" not in out.stdout:
        sys.stderr.write(f"FAILED {problem} {arm} {seed}: rc={out.returncode} {out.stderr[:200]}\n")
        return
    with open(tmp, "w") as f:
        f.write(out.stdout.rstrip("\n") + "\n")
        f.write("cmd=" + " ".join(["gtop-advexamples"] + cmd[1:]) + f"\nwall_s={wall:.1f}\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--binary")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--budget-s", type=float, default=480)
    a = ap.parse_args()
    binary = a.binary or os.path.join(os.environ["CARGO_TARGET_DIR"], "release", "gtop-advexamples")
    os.makedirs(RUNS, exist_ok=True)
    lock = open(os.path.join(HERE, "results", ".driver.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another driver holds the lock")
    st = STAGES[a.stage]
    todo = [(p, arm, s) for s in st["seeds"] for p in st["problems"] for arm in ARMS
            if not complete(run_file(p, arm, s))]
    print(f"{len(todo)} runs to do", flush=True)
    deadline = time.time() + a.budget_s
    it = iter(todo)

    def worker():
        while time.time() < deadline:
            try:
                job = next(it)
            except StopIteration:
                return
            one_run(binary, *job, st["retries"])

    with ThreadPoolExecutor(a.parallel) as ex:
        for _ in range(a.parallel):
            ex.submit(worker)
    left = sum(1 for j in todo if not complete(run_file(*j)))
    print(f"{left} runs left", flush=True)


if __name__ == "__main__":
    main()
