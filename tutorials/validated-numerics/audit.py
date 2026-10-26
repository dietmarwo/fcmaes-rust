#!/usr/bin/env python3
"""Audit every final search candidate with fresh Julia verification and replay.

`run` collects numerical and operational failures without turning them into
proofs. `accept` checks the sealed inputs and every result, then exits 0 only
when all predeclared arms have independently verified and replayed. This is an
application command, not a runtime or experiment-craft result contract.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time

import jsonschema

ROOT = Path(__file__).resolve().parent
MIB = 1024 * 1024
MAX_ARMS = 16  # the search protocol's limit
DOCUMENT_LIMIT = 256 * 1024  # one stage document; real ones are a few KiB
LOG_TAIL = 16 * 1024  # retained end of each stage's stdout and stderr
# Embedded in audit.json, a document grows at most 3x (escaped quotes, \u
# escapes of non-ASCII text) and a log byte at most 6x (U+FFFD or a control
# character as a \u escape). Everything the collector can write fits.
AUDIT_LIMIT = MAX_ARMS * 2 * (3 * DOCUMENT_LIMIT + 2 * 6 * LOG_TAIL + 4096) + 65536


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite_float(text):
    value = float(text)
    require(math.isfinite(value), "nonfinite JSON number")
    return value


def unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def parse(data):
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=unique_pairs, parse_float=finite_float, parse_constant=invalid)


def read(path, maximum=MIB):
    with Path(path).open("rb") as stream:
        data = stream.read(maximum + 1)
    require(len(data) <= maximum, "input exceeds byte limit: " + str(path))
    return data


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def schema(name):
    return jsonschema.Draft202012Validator(parse(read(ROOT / "schemas" / name)))


def blob(data):
    return {"sha256": sha(data), "json": data.decode("utf-8")}


def unblob(value):
    data = value["json"].encode("utf-8")
    require(len(data) <= DOCUMENT_LIMIT and sha(data) == value["sha256"], "embedded document digest mismatch")
    return parse(data)


def implementation():
    return {name: sha(read(ROOT / name)) for name in (
        "audit.py", "schemas/cap-audit-0.1.schema.json",
        "schemas/cap-artifacts-0.1.schema.json", "schemas/cap-search-0.1.schema.json")}


def load_protocol(path):
    data = read(path, 65536)
    value = parse(data)
    schema("cap-audit-0.1.schema.json").validate(value)
    require(value["schema_version"] == "cap.audit.protocol/0.1", "expected audit protocol")
    return value, sha(data)


def same_score(a, b):
    return struct.pack(">d", a) == struct.pack(">d", b)


def load_inputs(args):
    """Cross-check the complete frozen study, not just the chosen candidate."""
    names = ("problem", "study", "candidates", "evaluations", "search_protocol")
    raw = {name: read(getattr(args, name), 8 * MIB if name == "evaluations" else MIB) for name in names}
    problem, study, bundle, protocol = (parse(raw[name]) for name in ("problem", "study", "candidates", "search_protocol"))
    scientific = schema("cap-artifacts-0.1.schema.json")
    search = schema("cap-search-0.1.schema.json")
    scientific.validate(problem)
    for value in (study, bundle, protocol):
        search.validate(value)
    require(study["schema_version"] == "cap.search.study/0.1" and study["outcome"] == "completed"
            and study["certification_performed"] is False, "expected complete uncertified search study")
    require(bundle["schema_version"] == "cap.search.candidates/0.1"
            and protocol["schema_version"] == "cap.search.protocol/0.1", "wrong search input kind")
    require(study["problem_sha256"] == sha(raw["problem"]) and study["problem_id"] == problem["id"],
            "search problem identity mismatch")
    require(study["protocol_sha256"] == sha(raw["search_protocol"])
            and encoded(study["protocol"]) == encoded(protocol), "search protocol identity mismatch")
    require(study["evaluation_log_sha256"] == sha(raw["evaluations"]), "search ledger digest mismatch")
    project = Path(args.project)
    environment = study["server"]["environment"]
    for key, file in (("project_sha256", "Project.toml"), ("manifest_sha256", "Manifest.toml"),
                      ("verifier_sha256", "src/CAPReference.jl"), ("cli_sha256", "bin/cap.jl")):
        require(environment[key] == sha(read(project / file)), "current verifier identity mismatch: " + file)
    for key, file in (("server_sha256", "bin/search_server.jl"), ("search_module_sha256", "src/CAPSearch.jl")):
        require(study["server"][key] == sha(read(project / file)), "current search source mismatch: " + file)
    ids = [a["id"] for a in protocol["arms"]]
    require(len(ids) <= MAX_ARMS, "more search arms than an audit can record")
    require(len(set(ids)) == len(ids) and [a["id"] for a in study["arms"]] == ids
            and set(bundle["arms"]) == set(ids), "missing, duplicate or reordered arm")
    events = [parse(line) for line in raw["evaluations"].splitlines()]
    for event in events:
        search.validate(event)
    require(events[-1]["event"] == "study_finished" and events[-1]["outcome"] == "completed"
            and events[-1]["counts"] == study["counts"], "search terminal event mismatch")
    require(sum(e["event"] == "server_ready" for e in events) == 1, "missing search warm-up")
    starts = [e for e in events if e["event"] == "evaluation_started"]
    finishes = [e for e in events if e["event"] == "evaluation_finished"]
    require(len(starts) == len(finishes) and all(
        encoded([a[k] for k in ("arm", "phase", "parameters")]) == encoded([b[k] for k in ("arm", "phase", "parameters")])
        for a, b in zip(starts, finishes)), "search attempt/completion mismatch")
    require(all(e["arm"] in ids for e in finishes), "foreign arm in search ledger")
    total = 0
    for declared, arm in zip(protocol["arms"], study["arms"]):
        require(all(arm[k] == declared[k] for k in ("id", "method", "seed")), "arm settings mismatch")
        budget = 1 if arm["method"] == "reference" else protocol["evaluation_budget"]
        calls = [e for e in finishes if e["arm"] == arm["id"] and e["phase"] == "search"]
        rechecks = [e for e in finishes if e["arm"] == arm["id"] and e["phase"] == "recheck"]
        require(len(calls) == arm["best"]["evaluations"] == budget and len(rechecks) == arm["recheck_evaluations"] == 1,
                "per-arm search budget mismatch: " + arm["id"])
        best = min(calls, key=lambda e: e["estimate"]["score"])
        require(same_score(best["estimate"]["score"], arm["best"]["score"])
                and encoded(best["parameters"]) == encoded(arm["best"]["parameters"])
                and same_score(rechecks[0]["estimate"]["score"], arm["rechecked_score"])
                and same_score(arm["rechecked_score"], arm["best"]["score"])
                and encoded(rechecks[0]["parameters"]) == encoded(arm["best"]["parameters"]), "search best/recheck mismatch")
        item = bundle["arms"][arm["id"]]
        candidate = item["candidate_json"].encode("utf-8")
        require(sha(candidate) == item["candidate_sha256"] == arm["candidate_sha256"], "candidate bundle digest mismatch")
        doc = parse(candidate)
        scientific.validate(doc)
        require(doc["schema_version"] == "cap.candidate/0.1" and doc["problem_id"] == problem["id"]
                and doc["problem_sha256"] == sha(raw["problem"]) and doc["environment"] == environment,
                "candidate problem/environment mismatch")
        total += budget
    counts = study["counts"]
    require(counts == {"warmup_evaluations": 1, "startup_accounting_complete": True,
        "search_attempted": total, "search_completed": total, "rechecks_attempted": len(ids),
        "rechecks_completed": len(ids), "total_attempted_including_warmup": total + len(ids) + 1}, "search total counts mismatch")
    selected = min(study["arms"], key=lambda a: a["best"]["score"])
    require(study["selection"] == {"arm": selected["id"], "parameters": selected["best"]["parameters"],
        "estimated_score": selected["best"]["score"], "candidate_sha256": selected["candidate_sha256"]}, "search selection mismatch")
    return {"problem": problem, "study": study, "bundle": bundle, "environment": environment,
            "inputs": {name: sha(data) for name, data in raw.items()}}


def check_verification(value, context, candidate_sha):
    schema("cap-artifacts-0.1.schema.json").validate(value)
    problem = context["problem"]
    require(value["schema_version"] == "cap.verification/0.1" and value["problem_id"] == problem["id"]
            and value["problem_sha256"] == context["inputs"]["problem"] and value["candidate_sha256"] == candidate_sha,
            "verification artifact identity mismatch")
    require(value["environment"] == context["environment"], "verification environment mismatch")
    proof = value["proof"]
    require(proof["theorem"] == problem["theorem"] and proof["norm"] == problem["representation"]["norm"],
            "verification target/theorem mismatch")
    reasons = sorted(k for k, ok in proof["checks"].items() if not ok)
    require(value["reasons"] == reasons and value["outcome"] == ("inconclusive" if reasons else "verified"),
            "verification outcome contradicts its checks")


def check_replay(value, verification, certificate_sha, context, candidate_sha):
    schema("cap-artifacts-0.1.schema.json").validate(value)
    require(value["schema_version"] == "cap.replay/0.1" and value["candidate_sha256"] == candidate_sha
            and value["certificate_sha256"] == certificate_sha and value["problem_id"] == context["problem"]["id"]
            and value["problem_sha256"] == context["inputs"]["problem"], "replay artifact identity mismatch")
    fresh = value["recomputed"]
    check_verification(fresh, context, candidate_sha)
    # A fresh process records its own elapsed time; an exact copy was not recomputed.
    require(encoded(fresh) != encoded(verification), "replay repeats the certificate instead of recomputing it")
    expected = {"problem_identity": verification["problem_sha256"] == fresh["problem_sha256"] and verification["problem_id"] == fresh["problem_id"],
                "candidate_identity": verification["candidate_sha256"] == fresh["candidate_sha256"],
                "environment_identity": verification["environment"] == fresh["environment"],
                "original_verified": verification["outcome"] == "verified" and verification["reasons"] == [],
                "fresh_verified": fresh["outcome"] == "verified",
                "recomputed_proof_matches": verification["proof"] == fresh["proof"]}
    require(value["checks"] == expected and value["outcome"] == ("verified" if all(expected.values()) else "rejected"),
            "replay outcome contradicts recomputed checks")


def not_run(limit, reason):
    return {"status": "not_run", "pid": None, "returncode": None, "wall_seconds": 0.0,
            "wall_limit_seconds": limit, "document": None, "error": reason, "stdout": "", "stderr": ""}


def oversized(output, directory):
    """A runaway stage: a document or a log larger than the audit may hold."""
    limits = ((output, DOCUMENT_LIMIT), (directory / "stdout.log", MIB), (directory / "stderr.log", MIB))
    return any(path.exists() and path.stat().st_size > limit for path, limit in limits)


def run_stage(argv, directory, output, limit, cancelled, validate, event):
    """One child, inherited runtime process group, bounded output; no pipe-reader threads."""
    result = not_run(limit, None)
    started = time.monotonic()
    child = None
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        try:
            if cancelled():
                result.update(status="cancelled", error="audit cancelled before launch")
                return result
            env = os.environ.copy()
            env.update(JULIA_NUM_THREADS="1", JULIA_LOAD_PATH="@:@stdlib", JULIA_PKG_OFFLINE="true",
                       JULIA_PKG_PRECOMPILE_AUTO="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            child = subprocess.Popen(list(map(str, argv)), stdout=stdout, stderr=stderr, env=env)
            result["pid"] = child.pid
            event("stage_started", pid=child.pid)
            while child.poll() is None:
                if cancelled():
                    result.update(status="cancelled", error="audit cancelled")
                    break
                if time.monotonic() - started >= limit:
                    result.update(status="timed_out", error="stage wall deadline exceeded")
                    break
                if oversized(output, directory):
                    result.update(status="invalid_output", error="stage output exceeds byte limit")
                    break
                time.sleep(0.02)
            if result["status"] == "not_run":
                result["status"] = "completed"
            if child.poll() is None:
                child.kill()
            # No timeout: a killed child is always reaped before the next stage.
            result["returncode"] = child.wait()
            if cancelled():
                result.update(status="cancelled", error="audit cancelled")
            elif result["status"] == "completed" and time.monotonic() - started > limit:
                result.update(status="timed_out", error="stage wall deadline exceeded")
            if result["status"] == "completed" and oversized(output, directory):
                result.update(status="invalid_output", error="stage output exceeds byte limit")
            if result["status"] == "completed":
                if child.returncode not in (0, 3):
                    result.update(status="process_failed", error="Julia command failed")
                else:
                    try:
                        data = read(output, DOCUMENT_LIMIT)
                        value = parse(data)
                        validate(value)
                        expected = 3 if value.get("outcome") == "rejected" else 0
                        require(child.returncode == expected, "exit code contradicts document outcome")
                        result["document"] = blob(data)
                    except Exception as error:  # any defect of the untrusted output, e.g. RecursionError
                        result.update(status="invalid_output", document=None, error=(str(error) or type(error).__name__)[:2000])
        except Exception as error:
            result.update(status="process_failed", document=None, error=(str(error) or type(error).__name__)[:2000])
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                result["returncode"] = child.wait()
            result["wall_seconds"] = time.monotonic() - started
            if result["status"] == "completed" and result["wall_seconds"] > limit:
                result.update(status="timed_out", document=None, error="stage wall deadline exceeded during output validation")
    for name in ("stdout", "stderr"):
        # The end of a log says why a process failed; its size is bounded.
        with (directory / (name + ".log")).open("rb") as stream:
            stream.seek(max(0, (directory / (name + ".log")).stat().st_size - LOG_TAIL))
            result[name] = stream.read(LOG_TAIL).decode("utf-8", errors="replace")
    return result


def accepted(row):
    return all(row[s]["status"] == "completed" and unblob(row[s]["document"])["outcome"] == "verified"
               for s in ("verification", "replay"))


def observation(row, study):
    arm = next(a for a in study["arms"] if a["id"] == row["id"])
    seconds = study["startup_wall_seconds"] + arm["warm_wall_seconds"] + sum(row[s]["wall_seconds"] for s in ("verification", "replay"))
    event = accepted(row)
    return {"event": event, "right_censored": not event, "observed_seconds": seconds,
            "reason": "accepted" if event else "/".join(row[s]["status"] if row[s]["document"] is None else unblob(row[s]["document"])["outcome"]
                                                       for s in ("verification", "replay"))}


def markdown(record, context):
    rows = record["arms"]
    count = sum(accepted(row) for row in rows)
    lines = ["# Final-candidate certificate audit", "", f"Problem: `{context['problem']['id']}`. Accepted: **{count}/{len(rows)}**.", "",
        "| Arm | Verification | Replay | Accepted | Accounted seconds | Observation |",
        "| --- | --- | --- | --- | ---: | --- |"]
    for row in rows:
        outcomes = [unblob(row[s]["document"])["outcome"] if row[s]["document"] else row[s]["status"] for s in ("verification", "replay")]
        item = observation(row, context["study"])
        lines.append(f"| {row['id']} | {outcomes[0]} | {outcomes[1]} | {'yes' if item['event'] else 'no'} | {item['observed_seconds']:.6f} | {'event' if item['event'] else 'right censored'} |")
    lines += ["", "Timing adds the shared search startup in full to each arm, that arm's warm search/recheck/export work, and its fresh Julia verification and replay (including process startup).",
              "These are final-candidate observations, not first-found or anytime certificate times. Scheduling, transfers, other arms' work and coordinator overhead are excluded; the complete audit wall time is recorded separately.",
              "Unsuccessful and unattempted arms stay in the denominator and are right censored at the observed pipeline end. Stopping may be informative: no survival estimate, successful-only median, ranking or optimizer-advantage claim is made.",
              "The equations, target and proof scope are unchanged. This is not parameter-region coverage. A completed collection is not scientific acceptance.", ""]
    return "\n".join(lines).encode()


def validate_audit(record, context, protocol, protocol_sha, events):
    schema("cap-audit-0.1.schema.json").validate(record)
    require(record["schema_version"] == "cap.audit/0.1" and record["inputs"] == context["inputs"]
            and record["protocol_sha256"] == protocol_sha and record["protocol"] == protocol,
            "audit input/protocol identity mismatch")
    require(record["implementation"] == implementation(), "audit implementation identity mismatch")
    require(record["events_sha256"] == sha(events), "audit event ledger digest mismatch")
    require([r["id"] for r in record["arms"]] == [a["id"] for a in context["study"]["arms"]], "audit arm set/order mismatch")
    ledger = [parse(line) for line in events.splitlines()]
    expected_events = []
    documents = [row[stage]["document"]["sha256"] for row in record["arms"]
                 for stage in ("verification", "replay") if row[stage]["document"] is not None]
    require(len(set(documents)) == len(documents), "a stage document repeats another stage's document")
    for row in record["arms"]:
        require(None in (row["verification"]["pid"], row["replay"]["pid"])
                or row["verification"]["pid"] != row["replay"]["pid"], "verification and replay share a process")
        item = context["bundle"]["arms"][row["id"]]
        require(row["candidate_sha256"] == item["candidate_sha256"], "audit candidate identity mismatch")
        for stage, limit_key in (("verification", "verification_wall_seconds"), ("replay", "replay_wall_seconds")):
            result = row[stage]
            require(result["wall_limit_seconds"] == protocol[limit_key], "stage budget mismatch")
            if result["pid"] is not None:
                expected_events.append({"event": "stage_started", "arm": row["id"], "stage": stage, "pid": result["pid"]})
            if result["status"] != "not_run":
                expected_events.append({"event": "stage_finished", "arm": row["id"], "stage": stage, "status": result["status"]})
            if result["status"] == "completed":
                require(result["pid"] is not None and result["document"] is not None and result["error"] is None
                        and result["wall_seconds"] <= result["wall_limit_seconds"], "incomplete completed stage")
                value = unblob(result["document"])
                if stage == "verification":
                    check_verification(value, context, row["candidate_sha256"])
                    require(result["returncode"] == 0, "verification exit mismatch")
                else:
                    require(row["verification"]["status"] == "completed", "replay without completed verification")
                    verification = unblob(row["verification"]["document"])
                    check_replay(value, verification, row["verification"]["document"]["sha256"], context, row["candidate_sha256"])
                    require(result["returncode"] == (0 if value["outcome"] == "verified" else 3), "replay exit mismatch")
            else:
                require(result["document"] is None and result["error"] is not None, "failed stage contains a trusted document")
                if result["status"] == "not_run":
                    require(result["pid"] is None and result["returncode"] is None and result["wall_seconds"] == 0,
                            "unattempted stage carries work")
                # A launched process was reaped and has an exit status. Only a
                # cancellation before launch or a failed launch has neither.
                require((result["pid"] is None) == (result["returncode"] is None), "stage process without exit status")
                require(result["pid"] is not None or result["status"] in ("not_run", "cancelled", "process_failed"),
                        "stage outcome needs a launched process")
                require(result["status"] != "timed_out" or result["wall_seconds"] >= result["wall_limit_seconds"],
                        "timed-out stage ended before its deadline")
        # Replay runs only after a completed verification; a complete audit
        # replays exactly those.
        require(row["replay"]["status"] == "not_run" or row["verification"]["status"] == "completed",
                "replay attempted without a completed verification")
        require(not record["complete"] or row["verification"]["status"] not in ("not_run", "cancelled")
                and row["replay"]["status"] != "cancelled"
                and (row["replay"]["status"] == "not_run") == (row["verification"]["status"] != "completed"),
                "audit completion mismatch: " + row["id"])
    require(ledger == expected_events, "audit stage ledger/accounting mismatch")
    stages = sum(row[stage]["wall_seconds"] for row in record["arms"] for stage in ("verification", "replay"))
    require(record["total_wall_seconds"] >= stages, "audit wall time shorter than its stages")


def empty_directory(path):
    path = Path(path).absolute()
    require(not path.is_symlink(), "output directory is a symlink")
    if path.exists():
        require(path.is_dir() and not any(path.iterdir()), "output directory must be new or empty")
    else:
        path.mkdir(parents=True)
    return path


def run(args):
    started = time.monotonic()
    context = load_inputs(args)
    protocol, protocol_sha = load_protocol(args.protocol)
    output = empty_directory(args.output)
    record = {"schema_version": "cap.audit/0.1", "inputs": context["inputs"], "protocol": protocol,
              "protocol_sha256": protocol_sha, "implementation": implementation(), "complete": False,
              "total_wall_seconds": 0.0, "events_sha256": sha(b""), "arms": []}
    for arm in context["study"]["arms"]:
        record["arms"].append({"id": arm["id"], "candidate_sha256": arm["candidate_sha256"],
            "verification": not_run(protocol["verification_wall_seconds"], "not reached"),
            "replay": not_run(protocol["replay_wall_seconds"], "not reached")})
    cancelled = [False]
    handlers = {}
    documents = set()

    def cancel(signum, frame):
        cancelled[0] = True

    def snapshot():
        record["total_wall_seconds"] = time.monotonic() - started
        record["events_sha256"] = sha(read(output / "events.jsonl", MIB))
        data = encoded(record)
        require(len(data) <= AUDIT_LIMIT, "audit record exceeds its byte limit")
        temporary = output / "audit.tmp"
        temporary.write_bytes(data)
        temporary.replace(output / "audit.json")
        (output / "report.md").write_bytes(markdown(record, context))

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, cancel)
    try:
        with (output / "events.jsonl").open("xb") as ledger:
            snapshot()
            for row in record["arms"]:
                if cancelled[0]:
                    break
                def event(kind, **values):
                    ledger.write(encoded({"event": kind, "arm": row["id"], "stage": current_stage, **values}))
                    ledger.flush()
                candidate = context["bundle"]["arms"][row["id"]]["candidate_json"].encode("utf-8")
                with tempfile.TemporaryDirectory(prefix="stage-", dir=output) as work:
                    work = Path(work)
                    (work / "candidate.json").write_bytes(candidate)
                    julia = [Path(args.julia).resolve(), "--startup-file=no", "--history-file=no",
                             "--project=" + str(Path(args.project).resolve()), Path(args.project).resolve() / "bin/cap.jl"]
                    for current_stage, opcode, limit_key in (("verification", "verify", "verification_wall_seconds"),
                                                             ("replay", "replay", "replay_wall_seconds")):
                        if current_stage == "replay" and row["verification"]["status"] != "completed":
                            row["replay"] = not_run(protocol[limit_key], "verification unavailable")
                            break
                        stage_dir = work / current_stage
                        stage_dir.mkdir()
                        target = stage_dir / "result.json"
                        argv = [*julia, opcode, Path(args.problem).resolve(), work / "candidate.json"]
                        if current_stage == "verification":
                            validate = lambda v: check_verification(v, context, row["candidate_sha256"])
                        else:
                            certificate = row["verification"]["document"]
                            (work / "verification.json").write_bytes(certificate["json"].encode("utf-8"))
                            argv.append(work / "verification.json")
                            validate = lambda v: check_replay(v, unblob(certificate), certificate["sha256"], context, row["candidate_sha256"])
                        result = run_stage([*argv, target], stage_dir, target, protocol[limit_key], lambda: cancelled[0], validate, event)
                        if result["document"] is not None and result["document"]["sha256"] in documents:
                            # Each stage is a separate process; a byte-identical
                            # document is a cached or copied result, not new work.
                            result.update(status="invalid_output", document=None,
                                          error="document repeats an earlier stage's document")
                        elif result["document"] is not None:
                            documents.add(result["document"]["sha256"])
                        row[current_stage] = result
                        event("stage_finished", status=result["status"])
                        snapshot()
                        if cancelled[0]:
                            break
                if cancelled[0]:
                    break
            record["complete"] = not cancelled[0]
            snapshot()
        validate_audit(record, context, protocol, protocol_sha, read(output / "events.jsonl", MIB))
        # The written record decides: a signal after completion changes nothing.
        return 0 if record["complete"] else 130
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def accept(args):
    context = load_inputs(args)
    protocol, protocol_sha = load_protocol(args.protocol)
    data, events = read(args.audit, AUDIT_LIMIT), read(args.events, MIB)
    record = parse(data)
    validate_audit(record, context, protocol, protocol_sha, events)
    observations = [{"arm": row["id"], **observation(row, context["study"])} for row in record["arms"]]
    count = sum(item["event"] for item in observations)
    result = {"schema_version": "cap.audit.acceptance/0.1", "audit_sha256": sha(data),
              "protocol_sha256": protocol_sha, "inputs": context["inputs"],
              "outcome": "accepted" if record["complete"] and count == len(observations) else "rejected",
              "complete": record["complete"], "accepted_arms": count, "total_arms": len(observations),
              "observations": observations}
    schema("cap-audit-0.1.schema.json").validate(result)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded(result))
    print(json.dumps({"outcome": result["outcome"], "accepted_arms": count, "total_arms": len(observations)}))
    return 0 if result["outcome"] == "accepted" else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "accept"):
        sub = commands.add_parser(command)
        for name in ("problem", "study", "candidates", "evaluations", "search-protocol", "project", "protocol", "output"):
            sub.add_argument("--" + name, type=Path, required=True)
        if command == "run":
            sub.add_argument("--julia", type=Path, required=True)
        else:
            sub.add_argument("--audit", type=Path, required=True)
            sub.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args) if args.command == "run" else accept(args)
    except Exception as error:  # every failure is exit 2, never a traceback with exit 1
        message = str(error) or type(error).__name__
        print(json.dumps({"error": {"code": "invalid_audit", "message": message[:2000]}}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
