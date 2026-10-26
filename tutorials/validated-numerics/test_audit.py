#!/usr/bin/env python3
"""Real all-arm verification/replay, strict acceptance, exact checks and optional command campaigns."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import audit
import exact_check
from make_audit_campaign import generate
from test_workflow import annotate

ROOT = audit.ROOT


def arguments(problem, search):
    return ["--problem", ROOT / "problems" / (problem + ".json"), "--study", search / "study.json",
            "--candidates", search / "candidates.json", "--evaluations", search / "evaluations.jsonl",
            "--search-protocol", ROOT / "search-protocol.json", "--project", ROOT / "julia",
            "--protocol", ROOT / "audit-protocol.json"]


def recheck_exact(search, record, destination):
    """Recompute every accepted arm's bounds as exact rational numbers, without Julia."""
    bundle = audit.parse((search / "candidates.json").read_bytes())
    problem = audit.parse((search / "study.json").read_bytes())["problem_id"]
    report = exact_check.Report()
    destination.mkdir(parents=True)
    with (destination / "exact-check.txt").open("w") as log, contextlib.redirect_stdout(log):
        for row in record["arms"]:
            if not audit.accepted(row):
                continue
            folder = destination / row["id"] / problem
            folder.mkdir(parents=True)
            (folder / "candidate.json").write_text(bundle["arms"][row["id"]]["candidate_json"])
            (folder / "verification.json").write_text(row["verification"]["document"]["json"])
            print("ARM " + row["id"])
            checker = exact_check.scalar if problem == "scalar-cuberoot" else exact_check.cahn_hilliard
            checker(folder.parent, report)
    assert not report.failures, report.failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("search-evidence", "julia", "depot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--campaigns-only", action="store_true", help="run only the optional real runtime campaigns")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--experiment-craft-source", type=Path)
    parser.add_argument("--skip-retained", action="store_true", help="only when generating new evidence/audit")
    args = parser.parse_args()
    if (bool(args.runner) != bool(args.experiment_craft_source) or (args.runner and not args.binary)
            or (args.campaigns_only and not args.runner)):
        parser.error("runtime checks require --runner, --experiment-craft-source and --binary")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(JULIA_DEPOT_PATH=str(args.depot.resolve()), JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    checks, summary = [], []

    def run(label, argv, expected=0, environment=env):
        started = time.monotonic()
        result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, env=environment, timeout=3600)
        (output / (label + ".stdout.log")).write_text(result.stdout)
        (output / (label + ".stderr.log")).write_text(result.stderr)
        checks.append({"check": label, "returncode": result.returncode, "expected": expected,
                       "wall_seconds": time.monotonic() - started})
        (output / "checks.json").write_bytes(audit.encoded(checks))
        assert result.returncode == expected, (label, result.returncode, result.stderr[-3000:])
        print(label + ": passed", flush=True)
        return result.stdout

    if not args.campaigns_only:
        for problem in ("scalar-cuberoot", "cahn-hilliard"):
            search = args.search_evidence.resolve() / problem
            folder = output / problem
            common = arguments(problem, search)
            run(problem + "-audit", [sys.executable, ROOT / "audit.py", "run", *common,
                                     "--julia", args.julia.resolve(), "--output", folder])
            acceptance = output / (problem + "-acceptance.json")
            run(problem + "-accept", [sys.executable, ROOT / "audit.py", "accept", *common,
                "--audit", folder / "audit.json", "--events", folder / "events.jsonl", "--output", acceptance])
            record = audit.parse((folder / "audit.json").read_bytes())
            value = audit.parse(acceptance.read_bytes())
            assert value["accepted_arms"] == value["total_arms"] == 9 and value["complete"]
            assert all(item["event"] and not item["right_censored"] for item in value["observations"])
            # 18 actual processes, including a separate replay even when candidates
            # are identical. No proof cache can silently replace an arm's work.
            events = [audit.parse(line) for line in (folder / "events.jsonl").read_bytes().splitlines()]
            assert len([e for e in events if e["event"] == "stage_started"]) == 18
            for row in record["arms"]:
                assert row["verification"]["pid"] != row["replay"]["pid"]
                for stage in ("verification", "replay"):
                    try:
                        os.kill(row[stage]["pid"], 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise AssertionError("Julia child remains alive")
            recheck_exact(search, record, output / (problem + "-exact"))
            summary.append({"problem": problem, "accepted_arms": 9, "total_arms": 9,
                            "fresh_julia_processes": 18, "exact_rational_checks": 9})
            if not args.skip_retained:
                retained = ROOT / "evidence/audit" / problem
                run(problem + "-retained-accept", [sys.executable, ROOT / "audit.py", "accept",
                    *arguments(problem, retained / "search"), "--audit", retained / "audit/audit.json",
                    "--events", retained / "audit/events.jsonl", "--output", output / (problem + "-retained-acceptance.json")])
                recheck_exact(retained / "search", audit.parse((retained / "audit/audit.json").read_bytes()),
                              output / (problem + "-retained-exact"))

        # Operational failures are still complete audit observations, never proofs.
        fake = output / "failed-julia"
        fake.write_text("#!/usr/bin/env python3\nraise SystemExit(7)\n")
        fake.chmod(0o755)
        failed = output / "failed-audit"
        common = arguments("scalar-cuberoot", args.search_evidence.resolve() / "scalar-cuberoot")
        run("all-failures-retained", [sys.executable, ROOT / "audit.py", "run", *common,
                                     "--julia", fake, "--output", failed])
        run("failures-strictly-rejected", [sys.executable, ROOT / "audit.py", "accept", *common,
            "--audit", failed / "audit.json", "--events", failed / "events.jsonl", "--output", output / "failed-acceptance.json"], expected=3)
        value = audit.parse((output / "failed-acceptance.json").read_bytes())
        assert value["accepted_arms"] == 0 and value["total_arms"] == 9
        assert all(item["right_censored"] and not item["event"] for item in value["observations"])

        # A real verifier must classify a numerically bad proposal as inconclusive;
        # its real fresh replay rejects it. Exit zero from verification is not acceptance.
        invalid_candidate = audit.parse((args.search_evidence / "scalar-cuberoot/candidate.json").read_bytes())
        invalid_candidate["coefficients"] = ["3ff0000000000000"]  # exact 1.0
        candidate = output / "inconclusive-candidate.json"
        candidate.write_bytes(audit.encoded(invalid_candidate))
        certificate, replay = output / "inconclusive-verification.json", output / "inconclusive-replay.json"
        julia = [args.julia.resolve(), "--startup-file=no", "--history-file=no", "--project=" + str(ROOT / "julia"), ROOT / "julia/bin/cap.jl"]
        problem_file = ROOT / "problems/scalar-cuberoot.json"
        run("real-inconclusive-verification", [*julia, "verify", problem_file, candidate, certificate])
        run("real-inconclusive-replay", [*julia, "replay", problem_file, candidate, certificate, replay], expected=3)
        context_args = SimpleNamespace(problem=problem_file, study=args.search_evidence / "scalar-cuberoot/study.json",
            candidates=args.search_evidence / "scalar-cuberoot/candidates.json", evaluations=args.search_evidence / "scalar-cuberoot/evaluations.jsonl",
            search_protocol=ROOT / "search-protocol.json", project=ROOT / "julia")
        context = audit.load_inputs(context_args)
        verification, replayed = audit.parse(certificate.read_bytes()), audit.parse(replay.read_bytes())
        audit.check_verification(verification, context, audit.sha(candidate.read_bytes()))
        audit.check_replay(replayed, verification, audit.sha(certificate.read_bytes()), context, audit.sha(candidate.read_bytes()))
        assert verification["outcome"] == "inconclusive" and replayed["outcome"] == "rejected"

        # Cancellation of the collector alone, and of its whole process group,
        # must retain all arms and kill/reap its one active Julia child.
        for group in (False, True):
            folder = output / ("cancelled-group" if group else "cancelled")
            process = subprocess.Popen(list(map(str, [sys.executable, ROOT / "audit.py", "run", *common,
                                        "--julia", args.julia.resolve(), "--output", folder])),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True)
            try:
                end = time.monotonic() + 30
                ledger = folder / "events.jsonl"
                while time.monotonic() < end:
                    if ledger.exists() and b"stage_started" in ledger.read_bytes():
                        break
                    assert process.poll() is None
                    time.sleep(0.02)
                else:
                    raise AssertionError("audit did not start Julia")
                os.killpg(process.pid, signal.SIGINT) if group else process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=10)
                assert process.returncode == 130, (stdout, stderr)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
            partial = audit.parse((folder / "audit.json").read_bytes())
            assert partial["complete"] is False and len(partial["arms"]) == 9
            assert partial["arms"][0]["verification"]["status"] == "cancelled"
            try:
                os.kill(partial["arms"][0]["verification"]["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError("cancelled Julia child remains alive")
            run("cancelled-group-rejected" if group else "cancelled-rejected", [sys.executable, ROOT / "audit.py", "accept", *common,
                "--audit", folder / "audit.json", "--events", folder / "events.jsonl", "--output", folder / "acceptance.json"], expected=3)

    if args.runner:
        craft_env = env.copy()
        craft_env["PYTHONPATH"] = str(args.experiment_craft_source.resolve())
        craft = [sys.executable, "-m", "experiment_craft.cli"]
        for problem in ("scalar-cuberoot", "cahn-hilliard"):
            directory = output / (problem + "-campaign")
            directory.mkdir()
            manifest = generate(directory / "spec", args.julia, args.depot, args.binary, problem)
            node, workspace = directory / "node", directory / "workspace"
            run(problem + "-node", [args.runner.resolve(), "--root", node, "node", "init", "--capacity", "1", "--allow-commands"])
            run(problem + "-compile", [*craft, "dry-run", manifest], environment=craft_env)
            value = audit.parse(run(problem + "-campaign", [*craft, "run", manifest, "--workspace", workspace,
                "--runner", args.runner.resolve(), "--node-root", node, "--mode", "foreground"], environment=craft_env))
            assert value["status"] == "completed" and value["jobs_reserved"] == 3
            assert value["steps"] == {"search": "succeeded", "audit": "succeeded", "accept": "succeeded"}
            value = audit.parse(run(problem + "-continue", [*craft, "continue", "--workspace", workspace], environment=craft_env))
            assert value["status"] == "completed" and value["jobs_reserved"] == 3
            summary.append({"problem": problem, "campaign": "completed", "runtime_attempts": 3,
                            "continued_without_duplicates": True})
    (output / "summary.json").write_bytes(audit.encoded(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        annotate("test_audit.py", traceback.format_exc())
        raise
