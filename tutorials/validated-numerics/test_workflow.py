#!/usr/bin/env python3
"""Fresh-process acceptance and optional real experiment-craft command campaigns.

Requires jsonschema. No mock verifier, worker, systemd or runtime is used.
The optional campaign integration uses the real foreground runtime on one node.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import jsonschema
from make_campaign import ROOT, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", type=Path, required=True)
    parser.add_argument("--depot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--experiment-craft-source", type=Path)
    args = parser.parse_args()
    if bool(args.runner) != bool(args.experiment_craft_source):
        parser.error("--runner and --experiment-craft-source must be supplied together")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(JULIA_DEPOT_PATH=str(args.depot.resolve()), JULIA_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    schema = json.loads((ROOT / "schemas/cap-artifacts-0.1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    commands = []

    def run(label, argv, expected=0, env=environment):
        started = time.monotonic()
        result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, env=env, timeout=2400)
        (args.output / (label + ".stdout.log")).write_text(result.stdout)
        (args.output / (label + ".stderr.log")).write_text(result.stderr)
        commands.append({"label": label, "returncode": result.returncode,
                         "expected": expected, "wall_seconds": time.monotonic() - started})
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode != expected:
            raise AssertionError(f"{label}: expected {expected}, got {result.returncode}: {result.stderr[-3000:]}")
        print(label + ": passed", flush=True)
        return result.stdout

    def artifact(path):
        value = json.loads(path.read_text())
        validator.validate(value)
        return value

    julia = [args.julia.resolve(), "--startup-file=no", "--history-file=no",
             "--project=" + str(ROOT / "julia"), ROOT / "julia/bin/cap.jl"]
    results = []
    # Each operation is a fresh process. Always exercise both scientific references.
    for problem in ("scalar-cuberoot", "cahn-hilliard"):
        folder = args.output / problem
        folder.mkdir()
        problem_file = ROOT / "problems" / (problem + ".json")
        artifact(problem_file)
        candidate, certificate, replay = (folder / name for name in ("candidate.json", "verification.json", "replay.json"))
        run(problem + "-candidate", [*julia, "candidate", problem_file, candidate])
        run(problem + "-verify", [*julia, "verify", problem_file, candidate, certificate])
        run(problem + "-replay", [*julia, "replay", problem_file, candidate, certificate, replay])
        artifact(candidate)
        verification, replayed = artifact(certificate), artifact(replay)
        assert verification["outcome"] == replayed["outcome"] == "verified"
        assert replayed["certificate_sha256"] == hashlib.sha256(certificate.read_bytes()).hexdigest()
        assert verification["candidate_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
        radius = struct.unpack(">d", bytes.fromhex(verification["proof"]["existence_radius"]))[0]
        results.append({"problem": problem, "outcome": "verified", "existence_radius": radius,
                        "proof_scope": verification["proof"]["scope"]})
        if problem == "scalar-cuberoot":
            # A forged flag/bound must fail even though it is syntactically valid JSON.
            forged = copy.deepcopy(verification)
            forged["proof"]["bounds"]["Y"]["hi"] = "0000000000000000"
            forged_file = folder / "forged.json"
            forged_file.write_text(json.dumps(forged))
            validator.validate(forged)
            rejected_file = folder / "rejected-replay.json"
            run("forged-replay", [*julia, "replay", problem_file, candidate, forged_file, rejected_file], expected=3)
            assert artifact(rejected_file)["outcome"] == "rejected"
            run("refuse-overwrite", [*julia, "verify", problem_file, candidate, certificate], expected=2)
        else:
            missing = copy.deepcopy(verification)
            del missing["proof"]["bounds"]["Z1_tail"]
            assert not validator.is_valid(missing)

    if args.runner:
        source = args.experiment_craft_source.resolve()
        env = environment.copy()
        env["PYTHONPATH"] = str(source)
        # Public CLI throughout; do not read or mutate campaign SQLite directly.
        craft = [sys.executable, "-m", "experiment_craft.cli"]
        for problem in ("scalar-cuberoot", "cahn-hilliard"):
            folder = args.output / (problem + "-campaign")
            folder.mkdir()
            manifest = generate(folder / "spec", args.julia, args.depot, problem)
            node, workspace = folder / "node", folder / "workspace"
            run(problem + "-node-init", [args.runner.resolve(), "--root", node, "node", "init", "--capacity", "1", "--allow-commands"])
            run(problem + "-campaign-compile", [*craft, "dry-run", manifest], env=env)
            output = run(problem + "-campaign-run", [*craft, "run", manifest, "--workspace", workspace,
                         "--runner", args.runner.resolve(), "--node-root", node, "--mode", "foreground"], env=env)
            status = json.loads(output)
            assert status["status"] == "completed", status
            assert status["steps"] == {"candidate": "succeeded", "verify": "succeeded", "replay": "succeeded"}
            assert status["jobs_reserved"] == 3
            # A second continue observes existing handles; it must not create attempts.
            repeated = json.loads(run(problem + "-campaign-continue", [*craft, "continue", "--workspace", workspace], env=env))
            assert repeated["status"] == "completed" and repeated["jobs_reserved"] == 3
            results.append({"problem": problem, "campaign": "completed", "attempts": 3, "continue_created_attempts": 0})
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
