#!/usr/bin/env python3
"""Audit contract regressions. Fixture documents are not new scientific executions."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import audit
from make_audit_campaign import generate


def fixture_args(folder=None):
    root = audit.ROOT
    folder = folder or root / "evidence/search/scalar-cuberoot"
    return SimpleNamespace(problem=root / "problems/scalar-cuberoot.json", study=folder / "study.json",
        candidates=folder / "candidates.json", evaluations=folder / "evaluations.jsonl",
        search_protocol=root / "search-protocol.json", project=root / "julia", protocol=root / "audit-protocol.json")


def fixture_record(context, protocol, protocol_sha):
    """Repeat the scalar fixture's identical candidate; no fake proof engine is used."""
    folder = audit.ROOT / "evidence/search/scalar-cuberoot"
    documents = {s: audit.blob((folder / f"{s}.json").read_bytes()) for s in ("verification", "replay")}
    rows, events = [], []
    for arm in context["study"]["arms"]:
        row = {"id": arm["id"], "candidate_sha256": arm["candidate_sha256"]}
        for i, stage in enumerate(("verification", "replay")):
            row[stage] = {"status": "completed", "pid": 100 + i, "returncode": 0, "wall_seconds": 2.0,
                "wall_limit_seconds": protocol[stage + "_wall_seconds"], "document": copy.deepcopy(documents[stage]),
                "error": None, "stdout": "", "stderr": ""}
            events += [{"event": "stage_started", "arm": arm["id"], "stage": stage, "pid": 100 + i},
                       {"event": "stage_finished", "arm": arm["id"], "stage": stage, "status": "completed"}]
        rows.append(row)
    ledger = b"".join(audit.encoded(e) for e in events)
    return {"schema_version": "cap.audit/0.1", "inputs": context["inputs"], "protocol": protocol,
        "protocol_sha256": protocol_sha, "implementation": audit.implementation(), "complete": True,
        "total_wall_seconds": 36.0, "events_sha256": audit.sha(ledger), "arms": rows}, ledger


class Contract(unittest.TestCase):
    def setUp(self):
        self.args = fixture_args()
        self.context = audit.load_inputs(self.args)
        self.protocol, self.protocol_sha = audit.load_protocol(self.args.protocol)
        self.record, self.ledger = fixture_record(self.context, self.protocol, self.protocol_sha)

    def validate(self, record=None, ledger=None):
        audit.validate_audit(record or self.record, self.context, self.protocol, self.protocol_sha,
                             self.ledger if ledger is None else ledger)

    def test_fixture_links_and_all_arm_acceptance(self):
        self.validate()
        self.assertEqual(sum(audit.accepted(r) for r in self.record["arms"]), 9)

    def test_missing_duplicate_reordered_or_foreign_arm(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.append(rows[0]),
                         lambda rows: rows.reverse(), lambda rows: rows[0].update(id="foreign")):
            with self.subTest(mutation=mutation):
                record = copy.deepcopy(self.record)
                mutation(record["arms"])
                with self.assertRaises((ValueError, audit.jsonschema.ValidationError)):
                    self.validate(record)

    def test_changed_input_or_protocol_or_implementation(self):
        for key in ("inputs", "implementation"):
            record = copy.deepcopy(self.record)
            record[key][next(iter(record[key]))] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(record)
        record = copy.deepcopy(self.record)
        record["protocol"]["verification_wall_seconds"] += 1
        with self.assertRaises(ValueError):
            self.validate(record)

    def test_document_digest_and_stale_certificate(self):
        record = copy.deepcopy(self.record)
        record["arms"][0]["verification"]["document"]["json"] += " "
        with self.assertRaisesRegex(ValueError, "digest"):
            self.validate(record)
        record = copy.deepcopy(self.record)
        document = audit.unblob(record["arms"][0]["replay"]["document"])
        document["certificate_sha256"] = "0" * 64
        record["arms"][0]["replay"]["document"] = audit.blob(audit.encoded(document))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.validate(record)

    def test_forged_verified_flag_is_rejected(self):
        record = copy.deepcopy(self.record)
        document = audit.unblob(record["arms"][0]["verification"]["document"])
        document["proof"]["checks"]["physical_accuracy"] = False
        record["arms"][0]["verification"]["document"] = audit.blob(audit.encoded(document))
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.validate(record)
        record = copy.deepcopy(self.record)
        document = audit.unblob(record["arms"][0]["replay"]["document"])
        document["checks"]["recomputed_proof_matches"] = False
        record["arms"][0]["replay"]["document"] = audit.blob(audit.encoded(document))
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.validate(record)

    def test_budget_exit_and_ledger_mismatches(self):
        for key, value in (("returncode", 3), ("wall_limit_seconds", 1), ("wall_seconds", 1000)):
            record = copy.deepcopy(self.record)
            record["arms"][0]["verification"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(record)
        with self.assertRaisesRegex(ValueError, "ledger digest"):
            self.validate(ledger=self.ledger[:-1])
        record = copy.deepcopy(self.record)
        shortened = self.ledger.split(b"\n", 1)[1]
        record["events_sha256"] = audit.sha(shortened)
        with self.assertRaisesRegex(ValueError, "accounting"):
            self.validate(record, shortened)

    def test_timeout_kept_in_denominator_and_censored(self):
        row = self.record["arms"][0]
        row["verification"] = {**audit.not_run(120, "deadline"), "status": "timed_out", "pid": 100,
                               "returncode": -9, "wall_seconds": 120.02}
        row["replay"] = audit.not_run(120, "verification unavailable")
        events = [audit.parse(line) for line in self.ledger.splitlines()]
        events[1]["status"] = "timed_out"
        events = events[:2] + events[4:]
        ledger = b"".join(audit.encoded(e) for e in events)
        self.record["events_sha256"] = audit.sha(ledger)
        self.validate(ledger=ledger)
        self.assertEqual(len(self.record["arms"]), 9)
        self.assertEqual(sum(audit.accepted(r) for r in self.record["arms"]), 8)
        item = audit.observation(row, self.context["study"])
        self.assertTrue(item["right_censored"])
        self.assertFalse(item["event"])
        self.assertGreaterEqual(item["observed_seconds"], 120.02)

    def test_valid_inconclusive_and_rejected_replay_are_not_accepted(self):
        row = self.record["arms"][0]
        verification = audit.unblob(row["verification"]["document"])
        verification["proof"]["checks"]["physical_accuracy"] = False
        verification.update(outcome="inconclusive", reasons=["physical_accuracy"])
        row["verification"]["document"] = audit.blob(audit.encoded(verification))
        replay = audit.unblob(row["replay"]["document"])
        replay["recomputed"] = copy.deepcopy(verification)
        replay["checks"].update(original_verified=False, fresh_verified=False)
        replay.update(outcome="rejected", certificate_sha256=row["verification"]["document"]["sha256"])
        row["replay"]["document"] = audit.blob(audit.encoded(replay))
        row["replay"]["returncode"] = 3
        self.validate()
        self.assertFalse(audit.accepted(row))
        self.assertIn("inconclusive/rejected", audit.observation(row, self.context["study"])["reason"])

    def test_strict_acceptance_cli_rejects_partial_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.record["complete"] = False
            (path / "audit.json").write_bytes(audit.encoded(self.record))
            (path / "events.jsonl").write_bytes(self.ledger)
            args = SimpleNamespace(**vars(self.args), audit=path / "audit.json", events=path / "events.jsonl", output=path / "accepted.json")
            self.assertEqual(audit.accept(args), 3)
            result = audit.parse(args.output.read_bytes())
            self.assertEqual(result["outcome"], "rejected")
            self.assertEqual(result["total_arms"], 9)
            before = args.output.read_bytes()
            with self.assertRaises(FileExistsError):
                audit.accept(args)
            self.assertEqual(args.output.read_bytes(), before)

    def test_search_ledger_and_per_arm_counts_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("study.json", "candidates.json", "evaluations.jsonl"):
                (folder / name).write_bytes((self.args.study.parent / name).read_bytes())
            args = fixture_args(folder)
            study = audit.parse(args.study.read_bytes())
            study["arms"][0]["best"]["evaluations"] += 1
            args.study.write_bytes(audit.encoded(study))
            with self.assertRaisesRegex(ValueError, "per-arm"):
                audit.load_inputs(args)
            args.study.write_bytes(self.args.study.read_bytes())
            args.evaluations.write_bytes(args.evaluations.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "ledger digest"):
                audit.load_inputs(args)

    def test_search_candidate_swap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("study.json", "candidates.json", "evaluations.jsonl"):
                (folder / name).write_bytes((self.args.study.parent / name).read_bytes())
            args = fixture_args(folder)
            bundle = audit.parse(args.candidates.read_bytes())
            bundle["arms"]["reference"]["candidate_json"] += " "
            args.candidates.write_bytes(audit.encoded(bundle))
            with self.assertRaisesRegex(ValueError, "bundle digest"):
                audit.load_inputs(args)

    def test_malformed_evidence_is_exit_two_not_a_traceback(self):
        # Deep nesting (RecursionError) and an integer too large for binary64
        # (OverflowError) are malformed evidence, reported with exit 2.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("study.json", "candidates.json", "evaluations.jsonl"):
                (folder / name).write_bytes((self.args.study.parent / name).read_bytes())
            args = fixture_args(folder)
            nested = folder / "audit.json"
            nested.write_text("[" * 200000 + "]" * 200000)
            (folder / "events.jsonl").write_bytes(self.ledger)
            common = ["--problem", args.problem, "--study", args.study, "--candidates", args.candidates,
                      "--evaluations", args.evaluations, "--search-protocol", args.search_protocol,
                      "--project", args.project, "--protocol", args.protocol,
                      "--audit", nested, "--events", folder / "events.jsonl"]
            command = [sys.executable, audit.ROOT / "audit.py", "accept", *common]
            result = subprocess.run(list(map(str, command + ["--output", folder / "a.json"])), capture_output=True, text=True)
            self.assertEqual((result.returncode, result.stdout), (2, ""), result.stderr[-500:])
            self.assertIn("invalid_audit", result.stderr)
            study = audit.parse(args.study.read_bytes())
            study["arms"][0]["best"]["score"] = 10 ** 400
            args.study.write_bytes(audit.encoded(study))
            result = subprocess.run(list(map(str, command + ["--output", folder / "b.json"])), capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr[-500:])
            self.assertIn("invalid_audit", result.stderr)

    def test_nonfinite_duplicate_and_oversize_inputs(self):
        for text in ('{"a":1,"a":2}', '[NaN]', '[Infinity]', '[1e999]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.parse(text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_bytes(b" " * 20)
            with self.assertRaisesRegex(ValueError, "byte limit"):
                audit.read(path, 19)

    def test_output_directory_refuses_nonempty_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "sentinel").write_text("keep")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                audit.empty_directory(path)
            link = path / "link"
            link.symlink_to(path, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                audit.empty_directory(link)
            self.assertEqual((path / "sentinel").read_text(), "keep")

    def test_generated_campaign_keeps_three_admitted_jobs_and_hashed_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = generate(Path(directory) / "spec", sys.executable, directory, sys.executable, "scalar-cuberoot")
            plan = audit.parse(path.read_bytes())
            self.assertEqual([s["id"] for s in plan["steps"]], ["search", "audit", "accept"])
            self.assertEqual(plan["budget"]["max_jobs"], 3)
            self.assertEqual(plan["steps"][1]["job"]["limits"]["wall_seconds"], 2220)
            self.assertEqual(plan["steps"][1]["job"]["argv"][0], str(Path(sys.executable).absolute()))
            self.assertTrue(all(s["job"]["replay"] == "never" for s in plan["steps"]))
            for step in plan["steps"]:
                for item in step["job"]["inputs"]:
                    if "file" in item:
                        data = (path.parent / item["file"]).read_bytes()
                        self.assertEqual(audit.sha(data), item["sha256"])
                        self.assertEqual(len(data), item["bytes"])


# A stand-in for `julia ... cap.jl verify|replay` in collector tests: it writes
# the retained scalar documents with this process's own elapsed time, so each
# document is distinct, and binds each replay to the certificate it was given.
FAKE_JULIA = """
import hashlib, json, os, sys, time
from pathlib import Path
evidence = Path(os.environ["FAKE_EVIDENCE"])
operation, output = sys.argv[5], Path(sys.argv[-1])
for stream in (sys.stdout, sys.stderr):
    stream.buffer.write(b"\\xff" * int(os.environ.get("FAKE_LOG_BYTES", "0")))
    stream.flush()
unique = lambda: time.time_ns() % 10**12 / 1e9 + os.getpid() / 1e3
if operation == "verify":
    value = json.loads((evidence / "verification.json").read_text())
    value.update(candidate_sha256=hashlib.sha256(Path(sys.argv[7]).read_bytes()).hexdigest(), elapsed_seconds=unique())
else:
    certificate = Path(sys.argv[8]).read_bytes()
    value = json.loads((evidence / "replay.json").read_text())
    value.update(certificate_sha256=hashlib.sha256(certificate).hexdigest(), elapsed_seconds=unique(),
                 recomputed=dict(json.loads(certificate), elapsed_seconds=unique()))
output.write_text(json.dumps(value))
"""


class Runs(unittest.TestCase):
    """The collector end to end, with a stand-in for the Julia CLI."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.julia = self.folder / "julia"
        self.julia.write_text("#!" + sys.executable + "\n" + FAKE_JULIA)
        self.julia.chmod(0o755)
        self.environment = {"FAKE_EVIDENCE": str(audit.ROOT / "evidence/search/scalar-cuberoot")}

    def run_audit(self, name="out", **environment):
        args = SimpleNamespace(**vars(fixture_args()), julia=self.julia, output=self.folder / name)
        saved = os.environ.copy()
        os.environ.update(self.environment, **environment)
        try:
            return audit.run(args), args
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def accept(self, args, name="acceptance.json"):
        args = SimpleNamespace(**{**vars(args), "output": self.folder / name},
                               audit=args.output / "audit.json", events=args.output / "events.jsonl")
        return audit.accept(args)

    def test_stand_in_collection_is_accepted(self):
        code, args = self.run_audit()
        self.assertEqual(code, 0)
        self.assertEqual(self.accept(args), 0)

    def test_a_signal_after_completion_does_not_contradict_the_record(self):
        # A SIGTERM arriving after the last stage, while the finished record is
        # being validated, must not turn a complete audit into exit 130.
        original = audit.validate_audit
        def late_signal(*arguments):
            os.kill(os.getpid(), signal.SIGTERM)
            return original(*arguments)
        audit.validate_audit = late_signal
        try:
            code, args = self.run_audit()
        finally:
            audit.validate_audit = original
        record = audit.parse((args.output / "audit.json").read_bytes())
        self.assertTrue(record["complete"])
        self.assertEqual(code, 0)


class Processes(unittest.TestCase):
    def stage(self, code, limit=5, cancellation=False, validation_delay=0):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder = Path(self.directory.name)
        events = []
        cancelled = [False]
        self.validated = False
        def validate(value):
            self.validated = True
            time.sleep(validation_delay)
        def event(kind, **values):
            events.append((kind, values))
            if cancellation:
                cancelled[0] = True
        code = code.replace("OUTPUT", repr(str(folder / "result.json")))
        record = audit.run_stage([sys.executable, "-c", code], folder, folder / "result.json", limit,
                                 lambda: cancelled[0], validate, event)
        if record["pid"] is not None:
            with self.assertRaises(ProcessLookupError):
                os.kill(record["pid"], 0)
        return record, events

    def test_timeout_kills_and_reaps_child(self):
        record, events = self.stage("import time; time.sleep(30)", limit=0.05)
        self.assertEqual(record["status"], "timed_out")
        self.assertLess(record["wall_seconds"], 2)
        self.assertEqual(len(events), 1)

    def test_cancellation_kills_and_reaps_child(self):
        record, _ = self.stage("import time; time.sleep(30)", cancellation=True)
        self.assertEqual(record["status"], "cancelled")
        self.assertIsNone(record["document"])

    def test_missing_output_after_exit_zero_is_invalid(self):
        record, _ = self.stage("pass")
        self.assertEqual(record["status"], "invalid_output")
        self.assertIn("result.json", record["error"])

    def test_process_failure_is_not_a_proof(self):
        record, _ = self.stage("raise SystemExit(7)")
        self.assertEqual(record["status"], "process_failed")
        self.assertEqual(record["returncode"], 7)

    def test_large_output_is_bounded(self):
        record, _ = self.stage("import time; print('a'*1100000,flush=True); time.sleep(30)")
        self.assertEqual(record["status"], "invalid_output")
        self.assertEqual(len(record["stdout"]), 65536)

    def test_fast_large_output_is_rejected_after_process_exit(self):
        record, _ = self.stage("print('a'*1100000,flush=True)")
        self.assertEqual(record["status"], "invalid_output")
        self.assertIn("exceeds byte limit", record["error"])

    def test_deadline_includes_output_validation(self):
        record, _ = self.stage("from pathlib import Path; Path(OUTPUT).write_text('{\"outcome\":\"verified\"}')",
                               limit=0.5, validation_delay=0.6)
        self.assertTrue(self.validated, "the test must reach output validation")
        self.assertEqual(record["status"], "timed_out")
        self.assertIsNone(record["document"])
        self.assertIn("during output validation", record["error"])

    def test_deeply_nested_output_is_invalid_not_a_crash(self):
        record, _ = self.stage("from pathlib import Path; Path(OUTPUT).write_text('['*200000+']'*200000)")
        self.assertEqual(record["status"], "invalid_output")
        self.assertIsNone(record["document"])

    def test_inherited_pipes_do_not_delay_cleanup(self):
        # Regular log files, not pipes: a descendant retaining stdout cannot
        # keep the parent's reader threads alive. It exits on its own shortly.
        record, _ = self.stage("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.5)']); time.sleep(30)", limit=0.05)
        self.assertEqual(record["status"], "timed_out")
        self.assertLess(record["wall_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
