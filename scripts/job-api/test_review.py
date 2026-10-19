#!/usr/bin/env python3
"""Runner regression tests, shared by fcmaes-rust, LP-IP-craft and SQP-craft
(see docs/job-api.md): the review of the job API 0.1.0, the error codes of
0.2 and the JSON Schemas of the job envelope and the result manifest.

The tests run the release runner against `fake_worker.py`, a protocol fixture
that still speaks protocol 0.1, and against the repository's own worker where
a solver result is needed. Build first, with the worker package of this
repository:

    cargo build --release --locked -p optimization-runner -p WORKER_PACKAGE
    python3 scripts/job-api/test_review.py

The schema tests need the `jsonschema` package; without it they are skipped,
unless JOB_API_REQUIRE_SCHEMA=1 (as in CI) makes them fail.
"""
import copy
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None

REPO = Path(__file__).resolve().parents[2]
SCHEMAS = REPO / "docs/job-api/schemas"
CASES = Path(__file__).resolve().parent / "outcome_cases.json"
RUN = REPO / "target/release/optimization-run"
FAKE = Path(__file__).resolve().parent / "fake_worker.py"
TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted"}
# One small, quickly completed job per real worker: (binary, profile, payload).
LP_MODEL = json.dumps({
    "schema_version": "lp-ip.model/1", "sense": "minimize", "objective": [2.0, 3.0],
    "objective_offset": 5.0, "variable_lower": [0.0, 0.0], "variable_upper": [1.0, 1.0],
    "kinds": ["binary", "binary"], "matrix": {"indptr": [0, 2], "indices": [0, 1], "values": [1.0, 1.0]},
    "row_lower": [1.0], "row_upper": [1.0],
}).encode()
REAL = {
    "sqp": ("sqp-job-worker", "sqp.solve/1", {
        "problem": "benchmark.simplex", "dimension": 4, "starts": [[0.1, 0.2, 0.3, 0.4]],
        "backend": "active_set", "max_iterations": 100, "max_callback_calls": 2000,
        "feasibility_tolerance": 1e-7, "stationarity_tolerance": 1e-6}),
    "lp-ip": ("lp-ip-job-worker", "lp-ip.solve/1", {
        "model_input": "model", "starts": [None], "node_limit": 1000, "iteration_limit": 10000,
        "feasibility_tolerance": 1e-7, "integrality_tolerance": 1e-7, "presolve": True}),
    "fcmaes": ("fcmaes-job-worker", "fcmaes.retry/1", {
        "problem": {"id": "benchmark.sphere", "version": "1", "config": {"dimension": 2},
                    "bounds": {"lower": [-5.0], "upper": [5.0], "broadcast": True}},
        "optimizer": {"kind": "biteopt", "parameters": {"population": None}},
        "seed": 1, "retries": 1, "evaluations_per_retry": 100, "evaluation_limit_mode": "hard"}),
}
WORKER = next((name for name, (binary, _, _) in REAL.items()
               if (REPO / "target/release" / binary).exists()), None)


def envelope(worker, payload, wall=30.0, grace=0.2):
    return {
        "schema_version": "optimization.job/0.1",
        "request_key": "k",
        "worker": worker,
        "payload_schema": REAL[worker][1] if worker in REAL else "test.fake/1",
        "payload": payload,
        "resources": {"cpu_slots": 8, "outer_workers": 8, "inner_threads": 1},
        "limits": {"wall_time_seconds": wall, "grace_seconds": grace},
        "inputs": [],
    }


def job_id(key):
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class Node(unittest.TestCase):
    """A fresh node with the fake worker and this repository's worker."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "node"
        self.specs = 0
        self.processes = []
        self.addCleanup(self.stop_processes)
        self.call("node", "init", "--capacity", 8)
        self.call("workers", "add", "--id", "fake", "--executable", FAKE)
        if WORKER:
            binary = REPO / "target/release" / REAL[WORKER][0]
            self.call("workers", "add", "--id", WORKER, "--executable", binary)

    def stop_processes(self):
        for process, directory in self.processes:
            group = directory / "worker-process.json"
            if group.exists():
                try:
                    os.killpg(json.loads(group.read_text())["pgid"], signal.SIGKILL)
                except (ProcessLookupError, json.JSONDecodeError):
                    pass
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

    def raw(self, *args, env=None):
        return subprocess.run(
            [str(RUN), "--root", str(self.root), *map(str, args)],
            capture_output=True, text=True, timeout=60, env=env,
        )

    def call(self, *args, **kw):
        result = self.raw(*args, **kw)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def error(self, *args):
        """The runner's JSON error for a command that must fail."""
        result = self.raw(*args)
        self.assertEqual(result.returncode, 2, result.stdout)
        return json.loads(result.stderr.strip().splitlines()[-1])["error"]

    def real_job(self):
        """A small job of this repository's worker that completes quickly."""
        spec = envelope(WORKER, copy.deepcopy(REAL[WORKER][2]))
        if WORKER == "lp-ip":
            (self.base / "model.json").write_bytes(LP_MODEL)
            spec["inputs"] = [{"name": "model", "file": "model.json",
                               "sha256": hashlib.sha256(LP_MODEL).hexdigest()}]
        return spec

    def write(self, spec):
        self.specs += 1
        spec = copy.deepcopy(spec)
        spec["request_key"] = f"review-{self.specs}"
        path = self.base / f"spec-{self.specs}.json"
        path.write_text(json.dumps(spec))
        return path, spec["request_key"]

    def start(self, spec):
        path, key = self.write(spec)
        directory = self.root / "jobs" / job_id(key)
        process = subprocess.Popen(
            [str(RUN), "--root", str(self.root), "run", "--spec", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.processes.append((process, directory))
        return process, job_id(key), directory

    def wait_for(self, condition, timeout=15):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if condition():
                return
            time.sleep(0.01)
        self.fail("condition not reached")

    def state(self, directory):
        try:
            return json.loads((directory / "job.json").read_text())["state"]
        except (FileNotFoundError, json.JSONDecodeError):
            return None

class Review(Node):
    def test_unreadable_journal_ends_failed_and_keeps_listing(self):
        path, key = self.write(envelope("fake", {"mode": "garbage_journal"}))
        # `run` reports the journal error; the delivery also left the job
        # `running` and every `jobs list` failing.
        result = self.raw("run", "--spec", path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("results.jsonl", result.stderr)
        directory = self.root / "jobs" / job_id(key)
        self.assertEqual(self.state(directory), "failed")
        run = json.loads((directory / "run.json").read_text())
        self.assertIsNone(run["work"])
        self.assertFalse(run["runner"]["accounting_complete"])
        self.assertIn("journal_error", run["runner"])
        # `jobs get` still rejects the corrupt journal; `jobs list` reports
        # that per job instead of failing for every job on the node.
        self.assertNotEqual(self.raw("jobs", "get", job_id(key)).returncode, 0)
        path, _ = self.write(envelope("fake", {"mode": "sleep", "seconds": 0}))
        self.assertIn(self.call("run", "--spec", path)["state"], TERMINAL)
        listed = {job["id"]: job for job in self.call("jobs", "list")}
        self.assertIn("error", listed[job_id(key)])
        self.assertEqual(len(listed), 2)
        self.assertEqual(json.loads((self.root / "slots.json").read_text()), {})

    def test_queued_attempt_whose_launcher_died_can_be_repaired(self):
        hold, hold_id, hold_dir = self.start(envelope("fake", {"mode": "sleep"}, wall=60))
        self.wait_for(lambda: self.state(hold_dir) == "running")
        queued, queued_id, queued_dir = self.start(envelope("fake", {"mode": "sleep"}, wall=60))
        self.wait_for(lambda: self.state(queued_dir) == "waiting")
        queued.kill()
        queued.communicate(timeout=5)
        self.call("jobs", "cancel", hold_id, "--now")
        hold.communicate(timeout=10)
        self.assertFalse((queued_dir / "worker-process.json").exists())
        repaired = self.call("jobs", "repair", queued_id)
        self.assertEqual(repaired["state"], "interrupted")
        # No worker ever ran: zero work with complete accounting.
        run = json.loads((queued_dir / "run.json").read_text())
        self.assertEqual(run["work"], {})
        self.assertTrue(run["runner"]["accounting_complete"])
        self.assertEqual(json.loads((self.root / "slots.json").read_text()), {})

    @unittest.skipUnless(WORKER, "no solver worker built")
    def test_finished_job_is_not_interrupted_by_a_later_reboot(self):
        path, key = self.write(self.real_job())
        job = self.call("run", "--spec", path)
        self.assertEqual(job["state"], "completed")
        record = Path(job["directory"]) / "job.json"
        state = json.loads(record.read_text())
        state["boot_id"] = "00000000-0000-0000-0000-000000000000"
        record.write_text(json.dumps(state))
        self.assertIsNone(self.call("jobs", "get", job_id(key)).get("observed_state"))

    def test_validation_does_not_need_an_executable_tmpdir(self):
        path, _ = self.write(envelope("fake", {"mode": "sleep", "seconds": 0}))
        env = dict(os.environ, TMPDIR=str(self.base / "missing"))
        self.assertTrue(self.call("validate", "--spec", path, env=env)["valid"])
        self.assertEqual([p for p in self.root.iterdir() if p.name.startswith(".staging-")], [])

    def test_dot_slash_input_path_is_accepted(self):
        (self.base / "model.json").write_text("{}")
        spec = envelope("fake", {"mode": "sleep", "seconds": 0})
        spec["inputs"] = [{"name": "model", "file": "./model.json",
                           "sha256": hashlib.sha256(b"{}").hexdigest()}]
        path, _ = self.write(spec)
        self.assertTrue(self.call("validate", "--spec", path)["valid"])

    def test_interrupted_validation_stops_the_validator(self):
        path, _ = self.write(envelope("fake", {"mode": "sleep", "validate_sleep": 30}))
        runner = subprocess.Popen(
            [str(RUN), "--root", str(self.root), "run", "--spec", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.wait_for(lambda: any(p.name.startswith(".staging-") for p in self.root.iterdir()))
        staging = next(p for p in self.root.iterdir() if p.name.startswith(".staging-"))
        time.sleep(0.3)
        runner.send_signal(signal.SIGINT)
        runner.communicate(timeout=10)
        self.assertNotEqual(runner.returncode, 0)
        validators = subprocess.run(
            ["pgrep", "-f", f"{staging}/worker validate"], capture_output=True, text=True,
        ).stdout.split()
        self.assertEqual(validators, [])
        self.assertFalse(staging.exists())

    def test_slots_are_released_after_a_worker_crash(self):
        path, _ = self.write(envelope("fake", {"mode": "crash"}))
        job = self.call("run", "--spec", path)
        self.assertEqual(job["state"], "failed")
        self.assertIsNone(self.call("result", job["directory"])["work"])
        self.assertEqual(json.loads((self.root / "slots.json").read_text()), {})


    def test_cancel_after_the_end_is_a_conflict_and_repair_of_a_live_job_is_retryable(self):
        process, ident, directory = self.start(envelope("fake", {"mode": "sleep"}, wall=60))
        self.wait_for(lambda: self.state(directory) == "running")
        self.wait_for(lambda: (directory / "worker-process.json").exists())
        error = self.error("jobs", "repair", ident)
        self.assertEqual((error["code"], error["retryable"]), ("attempt_active", True))
        self.call("jobs", "cancel", ident, "--now")
        process.communicate(timeout=10)
        self.assertEqual(self.state(directory), "cancelled")
        (directory / "cancel.request").unlink()
        # Like HTTP 409: nothing is left to stop, and nothing is written.
        error = self.error("jobs", "cancel", ident)
        self.assertEqual((error["code"], error["retryable"]), ("already_terminal", False))
        self.assertIn("cancelled", error["message"])
        self.assertFalse((directory / "cancel.request").exists())
        self.assertEqual(self.error("jobs", "repair", ident)["code"], "already_terminal")

    def test_request_errors_have_stable_codes(self):
        path, key = self.write(envelope("fake", {"mode": "sleep", "seconds": 0}))
        self.call("run", "--spec", path)
        # Same key, different content: rejected, never a second job.
        spec = json.loads(path.read_text())
        spec["payload"]["seconds"] = 1
        path.write_text(json.dumps(spec))
        error = self.error("run", "--spec", path)
        self.assertEqual((error["code"], error["retryable"]), ("request_key_conflict", False))
        self.assertIn(job_id(key), error["message"])
        big = envelope("fake", {"mode": "sleep", "seconds": 0})
        big["resources"] = {"cpu_slots": 16, "outer_workers": 16, "inner_threads": 1}
        error = self.error("run", "--spec", self.write(big)[0])
        self.assertEqual(error["code"], "capacity_exceeded")
        self.assertIn("cpu_slots 16 exceed the node capacity 8", error["message"])
        self.assertEqual(self.error("jobs", "get", "0" * 32)["code"], "unknown_job")
        self.assertEqual(self.error("jobs", "get", "nonsense")["code"], "unknown_job")
        nobody = envelope("fake", {})
        nobody["worker"] = "nobody"
        self.assertEqual(self.error("validate", "--spec", self.write(nobody)[0])["code"], "unknown_worker")
        other = envelope("fake", {})
        other["payload_schema"] = "test.other/1"
        self.assertEqual(self.error("validate", "--spec", self.write(other)[0])["code"], "unsupported_profile")
        broken = self.base / "broken.json"
        broken.write_text("{")
        self.assertEqual(self.error("validate", "--spec", broken)["code"], "invalid_job")
        unknown = envelope("fake", {})
        unknown["priority"] = 1
        self.assertEqual(self.error("validate", "--spec", self.write(unknown)[0])["code"], "invalid_job")

    def test_a_protocol_0_1_worker_still_runs(self):
        self.assertEqual(self.call("workers", "list")["fake"]["description"]["protocol"],
                         "optimization.worker/0.1")
        path, _ = self.write(envelope("fake", {"mode": "complete"}))
        job = self.call("run", "--spec", path)
        self.assertEqual(job["state"], "completed")
        run = self.call("result", job["directory"])
        # The runner seals the attempt with its own manifest; the 0.1 record
        # keeps its exact content (no evidence is invented for it).
        self.assertEqual(run["schema_version"], "optimization.result/0.2")
        self.assertTrue(run["runner"]["accounting_complete"])
        self.assertEqual(run["results"][0]["outcome"]["claim"], "best_found")
        self.assertNotIn("evidence", run["results"][0]["outcome"])


class Schemas(Node):
    """The published JSON Schemas agree with the runner and the workers."""

    def setUp(self):
        if jsonschema is None:
            if os.environ.get("JOB_API_REQUIRE_SCHEMA") == "1":
                self.fail("the jsonschema package is required (JOB_API_REQUIRE_SCHEMA=1)")
            self.skipTest("the jsonschema package is not installed")
        super().setUp()
        self.job_schema = json.loads((SCHEMAS / "optimization.job-0.1.schema.json").read_text())
        self.result_schema = json.loads((SCHEMAS / "optimization.result-0.2.schema.json").read_text())
        for schema in [self.job_schema, self.result_schema]:
            jsonschema.Draft202012Validator.check_schema(schema)

    def valid(self, schema, value):
        errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(value), key=str)
        self.assertEqual(errors, [], [e.message for e in errors[:3]])

    def test_outcome_cases_agree_with_the_runner_rules(self):
        # The Rust contract tests check the same cases against Outcome::check.
        outcome = {key: value for key, value in self.result_schema.items()
                   if key not in ("type", "required", "properties", "additionalProperties")}
        outcome["$ref"] = "#/$defs/outcome"
        validator = jsonschema.Draft202012Validator(outcome)
        for case in json.loads(CASES.read_text()):
            self.assertEqual(validator.is_valid(case["outcome"]), case["valid"], case["name"])

    def test_job_envelopes_and_tutorial_jobs_are_valid(self):
        self.valid(self.job_schema, envelope("fake", {"mode": "sleep"}))
        for path in sorted((REPO / "tutorials/job-api").glob("*.json")):
            spec = json.loads(path.read_text())
            if spec.get("schema_version") == "optimization.job/0.1":
                self.valid(self.job_schema, spec)
        bad = envelope("fake", {})
        bad["resources"]["cpu_slots"] = 0
        self.assertFalse(jsonschema.Draft202012Validator(self.job_schema).is_valid(bad))

    def test_result_manifests_are_valid(self):
        path, _ = self.write(envelope("fake", {"mode": "garbage_journal"}))
        self.raw("run", "--spec", path)
        failed = json.loads(next((self.root / "jobs").iterdir()).joinpath("run.json").read_text())
        self.valid(self.result_schema, failed)
        # A protocol 0.1 worker's results, sealed by this runner.
        job = self.call("run", "--spec", self.write(envelope("fake", {"mode": "complete"}))[0])
        self.valid(self.result_schema, self.call("result", job["directory"]))
        if WORKER:
            job = self.call("run", "--spec", self.write(self.real_job())[0])
            run = self.call("result", job["directory"])
            self.valid(self.result_schema, run)
            for record in run["results"]:
                self.assertIn("evidence", record["outcome"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
