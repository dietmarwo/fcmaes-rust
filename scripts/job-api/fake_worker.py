#!/usr/bin/env python3
"""Protocol fixture for scripts/job-api/test_review.py: a worker whose
behaviour is chosen by payload["mode"]. It is not a solver."""
import fcntl
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
sys.exit(0)
