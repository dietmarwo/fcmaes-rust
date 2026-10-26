#!/usr/bin/env python3
"""Create a frozen three-command experiment-craft campaign; never launch work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PROJECT_FILES = ("Project.toml", "Manifest.toml", "src/CAPReference.jl", "bin/cap.jl")


def generate(destination, julia, depot, problem, wall_seconds=600):
    destination, julia, depot = map(lambda p: Path(p).resolve(), (destination, julia, depot))
    if not julia.is_file() or not depot.is_dir():
        raise ValueError("Julia executable and an instantiated depot must exist")
    if problem not in ("scalar-cuberoot", "cahn-hilliard"):
        raise ValueError("unknown reference problem")
    if not 30 <= wall_seconds <= 3600:
        raise ValueError("wall_seconds must be 30..3600")
    destination.mkdir(parents=True, exist_ok=False)
    sources = [(ROOT / "julia" / name, "project/" + name) for name in PROJECT_FILES]
    sources.append((ROOT / "problems" / (problem + ".json"), "problem.json"))
    inputs = []
    for source, staged in sources:
        relative = "inputs/" + staged
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        data = target.read_bytes()
        inputs.append({"path": staged, "file": relative, "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(), "executable": False})
    environment = {"JULIA_DEPOT_PATH": str(depot), "JULIA_LOAD_PATH": "@:@stdlib",
                   "JULIA_NUM_THREADS": "1", "JULIA_PKG_OFFLINE": "true",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def binding(step, output, path):
        return {"path": path, "from": {"step": step, "kind": "output", "output": output}}

    def step(name, arguments, derived, output):
        return {"id": name, "job": {"schema_version": "execution.command/0.1",
                "argv": [str(julia), "--startup-file=no", "--history-file=no", "--project=project",
                         "project/bin/cap.jl", *arguments, "out/" + output + ".json"],
                "env": environment.copy(), "resources": {"cpu_slots": 1},
                "limits": {"wall_seconds": wall_seconds, "grace_seconds": 5}, "replay": "never",
                "inputs": inputs + derived,
                "outputs": [{"name": output, "path": "out/" + output + ".json",
                             "max_bytes": 1048576, "required": True}]}}

    candidate = binding("candidate", "candidate", "candidate.json")
    verification = binding("verify", "verification", "verification.json")
    campaign = {"schema_version": "experiment.campaign/0.1", "name": "Validated numerics " + problem,
                "allow_commands": True,
                "budget": {"max_jobs": 3, "max_wall_seconds": 3 * wall_seconds + 300,
                           "max_reserved_slot_seconds": 3 * (wall_seconds + 5)},
                "steps": [step("candidate", ["candidate", "problem.json"], [], "candidate"),
                          step("verify", ["verify", "problem.json", "candidate.json"], [candidate], "verification"),
                          step("replay", ["replay", "problem.json", "candidate.json", "verification.json"],
                               [candidate, verification], "replay")]}
    path = destination / "campaign.json"
    path.write_text(json.dumps(campaign, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True, help="new output directory")
    parser.add_argument("--julia", type=Path, required=True)
    parser.add_argument("--depot", type=Path, required=True, help="already instantiated Julia depot")
    parser.add_argument("--problem", choices=["scalar-cuberoot", "cahn-hilliard"], default="scalar-cuberoot")
    parser.add_argument("--wall-seconds", type=int, default=600)
    args = parser.parse_args()
    print(generate(args.destination, args.julia, args.depot, args.problem, args.wall_seconds))


if __name__ == "__main__":
    main()
