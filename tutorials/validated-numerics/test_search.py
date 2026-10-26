#!/usr/bin/env python3
"""Real search, verification and replay; reproducibility, fault and optional runtime checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback

import jsonschema
from make_campaign import ROOT
from make_search_campaign import generate
from test_workflow import annotate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_gone(pid):
    # The proc filesystem may expose host IDs rather than this process namespace.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"child {pid} still exists")


def check_outcome_codes(schema):
    """Every failure code the search can raise must be a study outcome the schema accepts."""
    codes = {"io_error"}  # Failure::io
    for source in sorted((ROOT / "search/src").glob("*.rs")):
        codes.update(re.findall(r'Failure::new\(\s*"([a-z_]+)"', source.read_text()))
    accepted = set(schema["$defs"]["study"]["properties"]["outcome"]["enum"])
    assert codes <= accepted, f"study outcomes missing from the schema: {sorted(codes - accepted)}"
    return codes


def check_study(folder, spec, protocol, search_schema, artifact_schema, binary=None):
    """Check a completed study against the files it names; return it and its finished evaluations."""
    record = json.loads((folder / "study.json").read_text())
    search_schema.validate(record)
    assert record["outcome"] == "completed" and record["certification_performed"] is False
    assert record["protocol_sha256"] == digest(protocol)
    if binary is not None:
        assert record["search_binary_sha256"] == digest(binary)
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
    assert bundle["arms"][record["selection"]["arm"]]["candidate_json"].encode() == (folder / "candidate.json").read_bytes()
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
    # Per arm, not only in total: one arm overspending while another underspends must fail.
    assert [arm["id"] for arm in record["arms"]] == [arm["id"] for arm in spec["arms"]]
    for arm, item in zip(spec["arms"], record["arms"]):
        budget = 1 if arm["method"] == "reference" else spec["evaluation_budget"]
        assert item["best"]["evaluations"] == budget, item["id"]
        assert sum(e["arm"] == arm["id"] and e["phase"] == "search" for e in finishes) == budget, item["id"]
        assert sum(e["arm"] == arm["id"] and e["phase"] == "recheck" for e in finishes) == 1, item["id"]
    return record, finishes


def check_retained_proofs(folder, artifact_schema):
    """The retained verification and replay must certify exactly the retained selected candidate."""
    candidate = digest(folder / "candidate.json")
    for name in ("verification.json", "replay.json"):
        value = json.loads((folder / name).read_text())
        artifact_schema.validate(value)
        assert value["outcome"] == "verified" and value["candidate_sha256"] == candidate, name


FAKE_SERVER = '''#!/usr/bin/env python3
import hashlib,json,os,pathlib,signal,sys,time
MODE=pathlib.Path(sys.argv[0]).stem
project=pathlib.Path(sys.argv[3].split('=',1)[1]);server=pathlib.Path(sys.argv[4]);problem=pathlib.Path(sys.argv[5])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
environment=json.loads((project.parent/'evidence/scalar-cuberoot/candidate.json').read_text())['environment']
estimate=dict(score=0.0,estimated_Y=0.0,estimated_Z1=0.0,estimated_Z2=0.0,estimated_polynomial_normalized=0.0,estimated_contraction_minus_one=0.0,physical_epsilon=1e-10)
print(json.dumps(dict(schema_version='cap.search.response/0.1',request_id=0,status='ready',dimension=1,problem_id='scalar-cuberoot',problem_sha256=sha(problem),reference=estimate,environment=environment,server_sha256=sha(server),search_module_sha256=sha(project/'src/CAPSearch.jl'),construction_seconds=0.0)),flush=True)
n=0
for line in sys.stdin:
 r=json.loads(line);n+=1
 with open(sys.argv[0]+'.requests','a') as log:log.write(r['op']+'\\n')
 value=estimate.copy();value['score']=float(n)
 if MODE=='interrupt':
  value['score']=0.0
  # Interrupt the search before it receives this answer, so the cancellation
  # is recorded before its next call.
  if n==1:os.kill(os.getppid(),signal.SIGINT)
 if MODE=='nan-literal':value['score']=float('nan')
 if MODE=='string-score':value['score']='NaN'
 if MODE=='missing':del value['estimated_Z2']
 print(json.dumps(dict(schema_version='cap.search.response/0.1',request_id=r['request_id'],status='estimated',parameters=r['parameters'],estimate=value)),flush=True)
# Outlive the closed input, so a passing cleanup check shows that the search killed this process.
time.sleep(30)
'''


def fault_checks(binary, depot, directory):
    """Fault injection against the real release CLI; these scripts are not verifiers."""
    directory.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["JULIA_DEPOT_PATH"] = str(Path(depot).resolve())
    results = []
    # Each fake server lies about a floating estimate in one way (or interrupts the
    # search) to exercise one rejection branch; the error message names the branch.
    for mode, returncode, outcome, message in [
            ("mismatch", 2, "objective_mismatch", "same point did not reproduce its floating score"),
            ("nan-literal", 2, "protocol_error", "expected value"),  # not JSON: rejected by the parser
            ("string-score", 2, "protocol_error", "incomplete or nonfinite estimate"),
            ("missing", 2, "protocol_error", "incomplete or nonfinite estimate"),
            ("interrupt", 130, "cancelled", "search cancelled")]:
        script = directory / (mode + ".py")
        script.write_text(FAKE_SERVER)
        script.chmod(0o755)
        output = directory / (mode + "-result")
        result = subprocess.run([str(Path(binary).resolve()), "--julia", str(script.resolve()),
                    "--project", str(ROOT / "julia"), "--server", str(ROOT / "julia/bin/search_server.jl"),
                    "--problem", str(ROOT / "problems/scalar-cuberoot.json"),
                    "--protocol", str(ROOT / "search-protocol.json"), "--output", str(output),
                    "--wall-seconds", "5"], capture_output=True, text=True, env=env, timeout=20)
        record = json.loads((output / "study.json").read_text())
        assert result.returncode == returncode and record["outcome"] == outcome, (mode, result.stderr, record)
        assert message in record["error"]["message"], (mode, record["error"])
        assert not (output / "candidate.json").exists()
        assert_gone(record["server_pid"])
        # Attempts are exactly the evaluation requests the server received; a
        # cancellation between two calls leaves no attempt that was never sent.
        sent = Path(str(script) + ".requests").read_text().split().count("evaluate")
        events = [json.loads(line) for line in (output / "evaluations.jsonl").read_text().splitlines()]
        started = sum(e["event"] == "evaluation_started" for e in events)
        attempted = record["counts"]["search_attempted"] + record["counts"]["rechecks_attempted"]
        assert started == attempted == sent, (mode, started, attempted, sent)
        results.append({"fault": mode, "outcome": outcome, "message": record["error"]["message"],
                        "attempted_equals_sent": True, "child_killed_and_reaped": True})
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
    parser.add_argument("--skip-retained", action="store_true",
                        help="do not check evidence/search; only to regenerate it after a change")
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
    check_outcome_codes(schema)
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
            detail = result.stderr[-2000:]
            written = Path(str(argv[-1]))
            if written.suffix == ".json" and written.is_file():
                # A rejected replay says which of its checks failed.
                try:
                    detail += "\nchecks: " + json.dumps(json.loads(written.read_text()).get("checks"))
                except (ValueError, AttributeError):
                    pass
            raise AssertionError(f"{label}: exit {result.returncode}, expected {expected}: {detail}")
        print(label + ": passed", flush=True)
        return result.stdout

    def command(problem, output, *, julia=None, seconds=300, protocol_file=protocol):
        return [args.binary.resolve(), "--julia", julia or args.julia.resolve(),
                "--project", ROOT / "julia", "--server", ROOT / "julia/bin/search_server.jl",
                "--problem", ROOT / "problems" / (problem + ".json"), "--protocol", protocol_file,
                "--output", output, "--wall-seconds", str(seconds)]

    def study(folder):
        return check_study(folder, spec, protocol, search_schema, artifact_schema, args.binary.resolve())

    julia = [args.julia.resolve(), "--startup-file=no", "--history-file=no",
             "--project=" + str(ROOT / "julia"), ROOT / "julia/bin/cap.jl"]
    summaries = []
    for problem in () if args.skip_retained else ("scalar-cuberoot", "cahn-hilliard"):
        # The committed evidence must still be the files it names, use the current
        # protocol, and replay byte-identically in a fresh process.
        retained = ROOT / "evidence/search" / problem
        check_study(retained, spec, protocol, search_schema, artifact_schema)
        check_retained_proofs(retained, artifact_schema)
        replayed = args.output / (problem + "-retained-replay.json")
        run(problem + "-retained-replay", [*julia, "replay", ROOT / "problems" / (problem + ".json"),
                                           retained / "candidate.json", retained / "verification.json", replayed])
        value = json.loads(replayed.read_text())
        artifact_schema.validate(value)
        assert value["outcome"] == "verified"
        summaries.append({"problem": problem, "retained_evidence": "verified"})
    for problem in ("scalar-cuberoot", "cahn-hilliard"):
        folder = args.output / problem
        run(problem + "-search", command(problem, folder))
        record, _ = study(folder)
        problem_file = ROOT / "problems" / (problem + ".json")
        run(problem + "-verify", [*julia, "verify", problem_file, folder / "candidate.json", folder / "verification.json"])
        run(problem + "-replay", [*julia, "replay", problem_file, folder / "candidate.json", folder / "verification.json", folder / "replay.json"])
        check_retained_proofs(folder, artifact_schema)
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

    def interrupt(label, folder, marker, send, group):
        """Signal a running real study once `marker` is in its ledger; it must end as cancelled."""
        process = subprocess.Popen(list(map(str, command("scalar-cuberoot", folder))), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, env=environment, start_new_session=group)
        try:
            ledger = folder / "evaluations.jsonl"
            end = time.monotonic() + 60
            while time.monotonic() < end:
                if ledger.is_file() and marker in ledger.read_text():
                    break
                time.sleep(0.01)
            else:
                raise AssertionError(f"{label}: {marker} not reached before the signal")
            send(process)
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 130, (label, process.returncode, stderr[-2000:])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL) if group else process.kill()
                process.communicate()
        cancelled = json.loads((folder / "study.json").read_text())
        search_schema.validate(cancelled)
        assert cancelled["outcome"] == "cancelled", (label, cancelled.get("error"))
        assert_gone(cancelled["server_pid"])
        observations.append({"check": label, "returncode": process.returncode, "expected": 130})
        (args.output / "checks.json").write_text(json.dumps(observations, indent=2) + "\n")
        print(label + ": passed", flush=True)

    # SIGTERM to the search process alone, while Julia starts.
    interrupt("sigterm-cleanup", args.output / "cancelled", "server_started",
              lambda process: process.send_signal(signal.SIGTERM), group=False)
    # A terminal Ctrl-C reaches Julia too, which then exits on its own; that is
    # still a cancellation, not a server failure.
    interrupt("group-interrupt", args.output / "group-interrupt", "evaluation_started",
              lambda process: os.killpg(process.pid, signal.SIGINT), group=True)
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
    try:
        main()
    except Exception:
        annotate("test_search.py", traceback.format_exc())
        raise
