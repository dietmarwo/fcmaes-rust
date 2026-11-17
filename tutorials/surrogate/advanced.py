#!/usr/bin/env python3
"""Experimental v2 jobs: async RK4, warm archive and cooperative deadline."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import jsonschema

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--runner", type=Path, required=True)
p.add_argument("--worker", type=Path, required=True)
p.add_argument("--simulation-worker", type=Path, required=True)
a = p.parse_args()
a.root.mkdir(parents=True, exist_ok=False)


def cli(*args):
    proc = subprocess.run(
        [str(a.runner), "--root", str(a.root), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


cli("node", "init", "--capacity", "8")
cli("workers", "add", "--id", "simulation", "--executable", a.simulation_worker)
d = cli("problems", "describe", "simulation")
job = {
    "schema_version": "optimization.job/0.1",
    "request_key": "advanced-rk4",
    "worker": "simulation",
    "payload_schema": "fcmaes.surrogate/2",
    "payload": {
        "problem": {
            "id": "app.damped-oscillator",
            "version": "1",
            "config": {"dimension": 2},
            "parameters": {"steps": 256, "replicas": 2},
            "bounds": {"lower": [0.05, 0.5], "upper": [1.5, 4.0], "broadcast": False},
        },
        "seed": 41,
        "evaluations": 16,
        "runs": 1,
        "concurrent": 2,
        "evaluation_threads": 1,
        "parameters": {
            "base": {
                "initial_points": 4,
                "candidates": 64,
                "fit_restarts": 1,
                "fit_iterations": 8,
            },
            "schedule": "asynchronous",
            "training_cap": 32,
            "archive_cap": 1000,
            "surrogate_threads": 1,
        },
        "recheck_best": True,
    },
    "resources": {"outer_workers": 1, "inner_threads": 3, "cpu_slots": 3},
    "limits": {"wall_time_seconds": 30, "grace_seconds": 2},
    "inputs": [],
}
reports = {}


def run(spec, name):
    jsonschema.Draft202012Validator(
        d["payload_schemas"]["fcmaes.surrogate/2"]
    ).validate(spec["payload"])
    path = a.root / (name + ".json")
    path.write_text(json.dumps(spec))
    cli("validate", "--spec", path)
    j = cli("run", "--spec", path)
    r = cli("result", j["directory"])
    reports[name] = {"job": j, "result": r}
    return j, r


j, r = run(job, "rk4")
assert j["state"] == "completed"
assert r["work"]["evaluations"] == 16
assert r["results"][0]["outcome"]["evidence"]["level"] == "rechecked"
data = cli("jobs", "artifact", j["id"], "surrogate-observations-0")["content"].encode()
snapshot = json.loads(data)
assert snapshot["schema_version"] == "fcmaes.surrogate.observations/2"
(a.root / "previous.json").write_bytes(data)
warm = copy.deepcopy(job)
warm["request_key"] = "advanced-warm"
warm["payload"]["evaluations"] = 4
warm["payload"]["warm_start"] = {"input": "previous"}
warm["inputs"] = [
    {
        "name": "previous",
        "file": "previous.json",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
]
j, r = run(warm, "warm")
assert r["work"]["evaluations"] == 4
assert r["results"][0]["outcome"]["details"]["warm_start_points"] == 16
slow = copy.deepcopy(job)
slow["request_key"] = "advanced-timeout"
slow["payload"]["evaluations"] = 500
slow["payload"]["problem"]["parameters"] = {
    "steps": 64000,
    "replicas": 2,
    "pause_ms": 1,
}
slow["limits"]["wall_time_seconds"] = 0.3
slow["payload"]["recheck_best"] = False
j, r = run(slow, "deadline")
assert j["state"] == "timed_out", j
assert r["work"].get("validation_evaluations", 0) == 0
(a.root / "advanced-report.json").write_text(json.dumps(reports, indent=2) + "\n")
print(
    json.dumps(
        {
            "passed": True,
            "rk4_evaluations": 16,
            "warm_evaluations": 4,
            "deadline_state": j["state"],
        }
    )
)
