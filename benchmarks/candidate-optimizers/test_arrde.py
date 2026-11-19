#!/usr/bin/env python3
"""Tests of the round-4 driver `run_arrde.py` and of `decide_round4.py`.

Needs the release harness binary (found through $CARGO_TARGET_DIR) and the
Minion build of README, section Round 4. Run from this directory:

    python3 -m pytest -q test_arrde.py
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_arrde as driver
import run_catcmawm as base

HARNESS = base.default_harness()
PROBLEM = "truss_sizing"


class Bridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.serve = base.Serve(HARNESS)
        cls.info = cls.serve.info(PROBLEM)
        cls.optimizer = driver.check_minion()

    @classmethod
    def tearDownClass(cls):
        cls.serve.close()

    def evaluate(self, vectors):
        return self.serve.evaluate(PROBLEM, vectors)

    def test_round_trip_matches_harness_objective(self):
        rng = np.random.default_rng(1)
        lo, hi = self.info["lower"], self.info["upper"]
        vectors = [lo + rng.random(len(lo)) * (hi - lo) for _ in range(20)]
        values = self.evaluate(vectors)
        self.assertEqual(len(values), 20)
        self.assertEqual(self.serve.check(PROBLEM, vectors, values), 0)  # mismatches
        again = self.evaluate(vectors)
        self.assertTrue(np.array_equal(values.view(np.uint64), again.view(np.uint64)))

    def repetition(self, total, run_cap, root=99, rep=0):
        settings = {"root_seed": root, "total": total, "run_cap": run_cap, "min_run": 1000}
        return driver.run_repetition(self.optimizer, PROBLEM, self.info, rep, settings,
                                     self.evaluate), settings

    def test_budget_never_exceeded_and_best_is_seen(self):
        for total, cap in ((6000, 6000), (6500, 2000)):
            result, _ = self.repetition(total, cap)
            self.assertLessEqual(result["evaluations"], total)
            self.assertEqual(result["out_of_bounds"], 0)
            self.assertTrue(math.isfinite(result["best_value"]))
            self.assertGreaterEqual(result["library_calls"], result["evaluations"])
        result, _ = self.repetition(6500, 2000)
        self.assertGreaterEqual(result["runs"], 3)
        # The best point re-evaluates to the recorded value.
        again = self.evaluate([result["best_x"]])[0]
        self.assertEqual(again, result["best_value"])

    def test_seed_determinism(self):
        a, _ = self.repetition(5000, 5000, rep=0)
        b, _ = self.repetition(5000, 5000, rep=0)
        c, _ = self.repetition(5000, 5000, rep=1)
        self.assertEqual(a["best_value"], b["best_value"])
        self.assertEqual(a["seeds"], b["seeds"])
        self.assertNotEqual(a["best_value"], c["best_value"])

    def test_seed_matches_harness_derivation(self):
        out = subprocess.run([HARNESS, "seeds", "--root-seed", "7", "--problems", PROBLEM,
                              "--totals", "1000000", "--reps", "2", "--restarts", "3"],
                             check=True, capture_output=True, text=True).stdout
        rows = [line.split("\t") for line in out.strip().split("\n")]
        found = 0
        for row in rows:
            if PROBLEM in row:
                nums = [int(v) for v in row if v.isdigit()]
                seed = nums[-1]
                self.assertIn(seed, {base.run_seed(7, PROBLEM, r, k)
                                     for r in range(2) for k in range(3)})
                found += 1
        self.assertGreater(found, 0)
        self.assertEqual(driver.library_seed(2**40 + 12345), 12345)
        self.assertNotEqual(driver.library_seed(0x7FFFFFFF), -1)

    def test_row_format_is_the_runner_format(self):
        result, settings = self.repetition(4000, 4000)
        line = driver.format_row(PROBLEM, self.info["domain"], self.info["dim"], 0,
                                 settings, result)
        fields = line.rstrip("\n").split("\t")
        self.assertEqual(len(fields), len(base.RAW_HEADER_T.split("\t")))
        self.assertEqual(fields[0], "arrde")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.tsv"
            base.load_or_create(path)
            base.append_line(path, line)
            rows = base.load_or_create(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["solver"], "arrde")
        self.assertEqual(float(rows[0]["best_value"]), result["best_value"])


if __name__ == "__main__":
    unittest.main()
