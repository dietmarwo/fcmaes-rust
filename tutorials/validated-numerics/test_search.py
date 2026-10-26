#!/usr/bin/env python3
"""Real search, verification and replay; reproducibility, fault and optional runtime checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import jsonschema
from make_campaign import ROOT
from make_search_campaign import generate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_gone(pid):
    # The proc filesystem may expose host IDs rather than this process namespace.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"child {pid} still exists")


def fault_checks(binary, depot, directory):
    """Fault injection against the real release CLI; these scripts are not verifiers."""
    directory.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["JULIA_DEPOT_PATH"] = str(Path(depot).resolve())
    # Deliberately lie about a floating estimate to exercise rejection branches.
    code = '''#!/usr/bin/env python3
import hashlib,json,pathlib,sys
project=pathlib.Path(sys.argv[3].split('=',1)[1]);server=pathlib.Path(sys.argv[4]);problem=pathlib.Path(sys.argv[5])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
environment=json.loads((project.parent/'evidence/scalar-cuberoot/candidate.json').read_text())['environment']
estimate=dict(score=0.0,estimated_Y=0.0,estimated_Z1=0.0,estimated_Z2=0.0,estimated_polynomial_normalized=0.0,estimated_contraction_minus_one=0.0,physical_epsilon=1e-10)
print(json.dumps(dict(schema_version='cap.search.response/0.1',request_id=0,status='ready',dimension=1,problem_id='scalar-cuberoot',problem_sha256=sha(problem),reference=estimate,environment=environment,server_sha256=sha(server),search_module_sha256=sha(project/'src/CAPSearch.jl'),construction_seconds=0.0)),flush=True)
n=0
for line in sys.stdin:
 r=json.loads(line);n+=1
 value=estimate.copy();value['score']=float(n)
 if MODE=='nonfinite':value['score']=float('nan')
 if MODE=='missing':del value['estimated_Z2']
 print(json.dumps(dict(schema_version='cap.search.response/0.1',request_id=r['request_id'],status='estimated',parameters=r['parameters'],estimate=value)),flush=True)
'''
    results = []
    for mode, expected in [("mismatch", "objective_mismatch"), ("nonfinite", "protocol_error"), ("missing", "protocol_error")]:
        script = directory / (mode + ".py")
        script.write_text(code.replace("import hashlib,json,pathlib,sys", "import hashlib,json,pathlib,sys\nMODE=" + repr(mode)))
        script.chmod(0o755)
        output = directory / (mode + "-result")
        result = subprocess.run([str(Path(binary).resolve()), "--julia", str(script.resolve()),
                    "--project", str(ROOT / "julia"), "--server", str(ROOT / "julia/bin/search_server.jl"),
                    "--problem", str(ROOT / "problems/scalar-cuberoot.json"),
                    "--protocol", str(ROOT / "search-protocol.json"), "--output", str(output),
                    "--wall-seconds", "5"], capture_output=True, text=True, env=env, timeout=10)
        record = json.loads((output / "study.json").read_text())
        assert result.returncode == 2 and record["outcome"] == expected, (result.stderr, record)
        assert not (output / "candidate.json").exists()
        assert_gone(record["server_pid"])
        results.append({"fault": mode, "outcome": expected, "child_reaped": True})
    protected = directory / "protected-output"
    protected.mkdir()
    sentinel = protected / "candidate.json"
    sentinel.write_bytes(b"existing candidate must survive\n")
    before = digest(sentinel)
    refused = subprocess.run([str(Path(binary).resolve()), "--julia", "unused",
                "--project", str(ROOT / "julia"), "--server", str(ROOT / "julia/bin/search_server.jl"),
                "--problem", str(ROOT / "problems/scalar-cuberoot.json"),
                "--protocol", str(ROOT / "search-protocol.json"), "--output", str(protected)],
                capture_output=True, text=True, env=env, timeout=10)
    assert refused.returncode == 2 and digest(sentinel) == before
    assert sorted(p.name for p in protected.iterdir()) == ["candidate.json"]
    results.append({"fault": "nonempty-output", "outcome": "refused", "existing_artifact_unchanged": True})
    (directory / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["binary", "julia", "depot", "output"]:
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--experiment-craft-source", type=Path)
    args = parser.parse_args()
    if bool(args.runner) != bool(args.experiment_craft_source):
        parser.error("supply both runtime integration arguments")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # Cargo may replace target/release/cap-search when switching build/test
    # feature graphs. Keep one immutable executable for the whole study.
    frozen = args.output / "bin/cap-search"
    frozen.parent.mkdir()
    shutil.copyfile(args.binary.resolve(), frozen)
    frozen.chmod(0o755)
    args.binary = frozen
    environment = os.environ.copy()
    environment.update(JULIA_DEPOT_PATH=str(args.depot.resolve()), JULIA_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    schema = json.loads((ROOT / "schemas/cap-search-0.1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    search_schema = jsonschema.Draft202012Validator(schema)
    artifact_schema = jsonschema.Draft202012Validator(json.loads((ROOT / "schemas/cap-artifacts-0.1.schema.json").read_text()))
    protocol = ROOT / "search-protocol.json"
    spec = json.loads(protocol.read_text())
    search_schema.validate(spec)
    observations = []

    def run(label, argv, expected=0, env=environment):
        started = time.monotonic()
        result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, env=env, timeout=2400)
        (args.output / (label + ".stdout.log")).write_text(result.stdout)
        (args.output / (label + ".stderr.log")).write_text(result.stderr)
        observations.append({"check": label, "returncode": result.returncode,
                             "expected": expected, "wall_seconds": time.monotonic() - started})
        (args.output / "checks.json").write_text(json.dumps(observations, indent=2) + "\n")
        if result.returncode != expected:
            raise AssertionError(f"{label}: exit {result.returncode}, expected {expected}: {result.stderr[-2000:]}")
        print(label + ": passed", flush=True)
        return result.stdout

    def command(problem, output, *, julia=None, seconds=300, protocol_file=protocol):
        return [args.binary.resolve(), "--julia", julia or args.julia.resolve(),
                "--project", ROOT / "julia", "--server", ROOT / "julia/bin/search_server.jl",
                "--problem", ROOT / "problems" / (problem + ".json"), "--protocol", protocol_file,
                "--output", output, "--wall-seconds", str(seconds)]

    def study(folder):
        record = json.loads((folder / "study.json").read_text())
        search_schema.validate(record)
        assert record["outcome"] == "completed" and record["certification_performed"] is False
        assert record["protocol_sha256"] == digest(protocol)
        assert record["search_binary_sha256"] == digest(args.binary.resolve())
        expected_search = sum(1 if arm["method"] == "reference" else spec["evaluation_budget"] for arm in spec["arms"])
        assert record["counts"]["search_attempted"] == record["counts"]["search_completed"] == expected_search
        assert record["counts"]["rechecks_attempted"] == record["counts"]["rechecks_completed"] == len(spec["arms"])
        assert record["counts"]["total_attempted_including_warmup"] == expected_search + len(spec["arms"]) + 1
        assert record["counts"]["startup_accounting_complete"]
        assert record["selection"]["candidate_sha256"] == digest(folder / "candidate.json")
        assert record["evaluation_log_sha256"] == digest(folder / "evaluations.jsonl")
        candidate = json.loads((folder / "candidate.json").read_text())
        artifact_schema.validate(candidate)
        bundle = json.loads((folder / "candidates.json").read_text())
        search_schema.validate(bundle)
        assert set(bundle["arms"]) == {arm["id"] for arm in spec["arms"]}
        for arm in record["arms"]:
            item = bundle["arms"][arm["id"]]
            assert hashlib.sha256(item["candidate_json"].encode()).hexdigest() == item["candidate_sha256"] == arm["candidate_sha256"]
            artifact_schema.validate(json.loads(item["candidate_json"]))
            assert arm["rechecked_score"] == arm["best"]["score"]
        events = [json.loads(line) for line in (folder / "evaluations.jsonl").read_text().splitlines()]
        for event in events:
            search_schema.validate(event)
        starts = [e for e in events if e["event"] == "evaluation_started"]
        finishes = [e for e in events if e["event"] == "evaluation_finished"]
        assert len(starts) == len(finishes) == expected_search + len(spec["arms"])
        assert [(e["arm"], e["phase"], e["parameters"]) for e in starts] == [(e["arm"], e["phase"], e["parameters"]) for e in finishes]
        return record, finishes

    julia = [args.julia.resolve(), "--startup-file=no", "--history-file=no",
             "--project=" + str(ROOT / "julia"), ROOT / "julia/bin/cap.jl"]
    summaries = []
    for problem in ("scalar-cuberoot", "cahn-hilliard"):
        folder = args.output / problem
        run(problem + "-search", command(problem, folder))
        record, _ = study(folder)
        problem_file = ROOT / "problems" / (problem + ".json")
        run(problem + "-verify", [*julia, "verify", problem_file, folder / "candidate.json", folder / "verification.json"])
        run(problem + "-replay", [*julia, "replay", problem_file, folder / "candidate.json", folder / "verification.json", folder / "replay.json"])
        for file in ["verification.json", "replay.json"]:
            value = json.loads((folder / file).read_text())
            artifact_schema.validate(value)
            assert value["outcome"] == "verified"
        summaries.append({"problem": problem, "arm_count": len(record["arms"]), "counts": record["counts"],
                          "selected_arm": record["selection"]["arm"], "verification": "verified", "replay": "verified"})

    run("independent-exact-bounds", [sys.executable, ROOT / "exact_check.py", args.output])
    repeated = args.output / "scalar-repeated"
    run("seeded-search-repeat", command("scalar-cuberoot", repeated))
    first, first_events = study(args.output / "scalar-cuberoot")
    second, second_events = study(repeated)
    fields = lambda events: [(e["arm"], e["phase"], e["parameters"], e["estimate"]) for e in events]
    assert fields(first_events) == fields(second_events)
    assert first["selection"]["parameters"] == second["selection"]["parameters"]

    bad = args.output / "invalid-protocol.json"
    bad.write_text(json.dumps({**spec, "unrecognized": True}))
    run("unknown-protocol-field", command("scalar-cuberoot", args.output / "bad-protocol-result", protocol_file=bad), expected=2)
    assert not (args.output / "bad-protocol-result").exists()
    timeout = args.output / "timeout"
    run("wall-timeout", command("scalar-cuberoot", timeout, seconds=0.05), expected=124)
    timed_out = json.loads((timeout / "study.json").read_text())
    search_schema.validate(timed_out)
    assert timed_out["outcome"] == "timed_out" and not timed_out["counts"]["startup_accounting_complete"]
    assert_gone(timed_out["server_pid"])
    assert not (timeout / "candidate.json").exists()

    interrupted = args.output / "cancelled"
    process = subprocess.Popen(list(map(str, command("scalar-cuberoot", interrupted))),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    try:
        end = time.monotonic() + 10
        while time.monotonic() < end:
            if (interrupted / "evaluations.jsonl").is_file() and (interrupted / "evaluations.jsonl").stat().st_size:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("server did not start before cancellation test")
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, (process.returncode, stderr)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    cancelled = json.loads((interrupted / "study.json").read_text())
    search_schema.validate(cancelled)
    assert cancelled["outcome"] == "cancelled"
    assert_gone(cancelled["server_pid"])
    observations.append({"check": "sigterm-cleanup", "returncode": process.returncode, "expected": 130})
    (args.output / "checks.json").write_text(json.dumps(observations, indent=2) + "\n")
    print("sigterm-cleanup: passed", flush=True)
    fault_checks(args.binary, args.depot, args.output / "fault-injection")

    if args.runner:
        env = environment.copy()
        env["PYTHONPATH"] = str(args.experiment_craft_source.resolve())
        craft = [sys.executable, "-m", "experiment_craft.cli"]
        for problem in ("scalar-cuberoot", "cahn-hilliard"):
            directory = args.output / (problem + "-campaign")
            directory.mkdir()
            manifest = generate(directory / "spec", args.julia, args.depot, args.binary, problem)
            node, workspace = directory / "node", directory / "workspace"
            run(problem + "-node", [args.runner.resolve(), "--root", node, "node", "init", "--capacity", "1", "--allow-commands"])
            run(problem + "-compile", [*craft, "dry-run", manifest], env=env)
            status = json.loads(run(problem + "-campaign", [*craft, "run", manifest, "--workspace", workspace,
                            "--runner", args.runner.resolve(), "--node-root", node, "--mode", "foreground"], env=env))
            assert status["status"] == "completed" and status["jobs_reserved"] == 3, status
            assert status["steps"] == {"search": "succeeded", "verify": "succeeded", "replay": "succeeded"}
            status = json.loads(run(problem + "-continue", [*craft, "continue", "--workspace", workspace], env=env))
            assert status["status"] == "completed" and status["jobs_reserved"] == 3
            summaries.append({"problem": problem, "campaign": "completed", "runtime_attempts": 3, "continued_without_duplicates": True})
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
