#!/usr/bin/env python3
"""Protocol fixture for scripts/job-api/test_review.py: a worker whose
behaviour is chosen by payload["mode"]. It is not a solver. It speaks worker
protocol 0.1, so it also checks that the runner still accepts 0.1 workers."""
import fcntl
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

command = sys.argv[1]
if command == "describe":
    print(json.dumps({"protocol": "optimization.worker/0.1", "profiles": ["test.fake/1"]}))
    sys.exit(0)
if command == "validate":
    spec = json.loads(Path(sys.argv[2]).read_text())
    time.sleep(spec["payload"].get("validate_sleep", 0))
    spec["task_count"] = 1
    print(json.dumps({"resolved": spec}))
    sys.exit(0)

directory = Path(sys.argv[2])
lock = (directory / "worker.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX)
payload = json.loads((directory / "resolved.json").read_text())["payload"]
mode = payload.get("mode", "sleep")
if mode == "sleep":
    time.sleep(payload.get("seconds", 120))
elif mode == "garbage_journal":
    # A faulty worker: one complete line that is no journal envelope.
    (directory / "results.jsonl").write_bytes(b"this is not an envelope\n")
    sys.exit(3)
elif mode == "crash":
    signal.raise_signal(signal.SIGSEGV)
elif mode == "complete":
    # One 0.1 result record (no evidence), committed like the Rust SDK does,
    # then the worker manifest the runner cross-checks against the journal.
    spec = json.loads((directory / "resolved.json").read_text())
    spec.pop("request_key")
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    outcome = payload.get("outcome", {"termination": "evaluation_limit", "claim": "best_found",
                                      "feasible": True, "objective": 1.0, "x": [1.0], "details": {}})
    record = {"task_id": 0, "spec_digest": digest, "elapsed_seconds": 0.0, "work": {"evaluations": 1},
              "outcome": outcome}
    payload = json.dumps(record)
    envelope = {"schema_version": 1, "checksum": hashlib.sha256(payload.encode()).hexdigest(),
                "payload_json": payload}
    (directory / "results.jsonl").write_text(json.dumps(envelope) + "\n")
    (directory / "run.json").write_text(json.dumps({
        "schema_version": "optimization.result/0.1", "status": "completed", "elapsed_seconds": 0.0,
        "results": [record], "work": {"evaluations": 1},
        "runner": {"spec_digest": digest, "accounting_complete": True}}))
sys.exit(0)
