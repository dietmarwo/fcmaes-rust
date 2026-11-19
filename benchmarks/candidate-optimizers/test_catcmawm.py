#!/usr/bin/env python3
"""Tests of the round-3 candidate driver `run_catcmawm.py`, of
`analyze.py --round 3`, `--round 3b` and `--round 3u`, and of the round-3b
reference check `round3b_reference.py`.

Needs the release harness binary (`cargo build --release -p candidate-harness
--features candidates`; found through $CARGO_TARGET_DIR or target/). Run from
this directory:

    python3 -m unittest -v test_catcmawm
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze
import round3b_reference as reference
import run_catcmawm as driver

HERE = Path(__file__).resolve().parent
HARNESS = driver.default_harness()


def harness(*args: str) -> str:
    return subprocess.run([HARNESS, *args], check=True, capture_output=True,
                          text=True).stdout


def toy_table() -> dict:
    """Two continuous, one integer (values 2, 5, 9) and one three-way
    categorical coordinate."""
    return {
        "problem": "toy", "dimension": 4,
        "coordinates": [
            {"index": 0, "kind": "continuous", "lower": -1.0, "upper": 1.0},
            {"index": 1, "kind": "integer", "lower": 0.0, "upper": 3.0,
             "levels": [0.5, 1.5, 2.5], "values": [2, 5, 9]},
            {"index": 2, "kind": "categorical", "lower": 0.0, "upper": 1.0,
             "levels": [1 / 6, 0.5, 5 / 6]},
            {"index": 3, "kind": "continuous", "lower": 0.0, "upper": 2.0},
        ],
    }


def toy_objective(vectors):
    return [float((x[0] - 0.3) ** 2 + (x[1] - 1.5) ** 2 + abs(x[2] - 0.5) + x[3])
            for x in vectors]


class Seeds(unittest.TestCase):
    def test_python_seeds_equal_the_harness_seeds(self):
        for root in (3, 999):
            text = harness("seeds", "--root-seed", str(root), "--problems", "round3",
                           "--totals", "2000,20000", "--reps", "3", "--restarts", "4")
            lines = text.strip().split("\n")
            self.assertEqual(lines[0], "problem\ttotal\trep\trestart\tseed")
            self.assertEqual(len(lines) - 1, 6 * 2 * 3 * 4)
            for line in lines[1:]:
                problem, _total, rep, restart, seed = line.split("\t")
                self.assertEqual(driver.run_seed(root, problem, int(rep), int(restart)),
                                 int(seed), line)

    def test_candidate_gets_the_low_32_bits(self):
        seed = driver.run_seed(3, "jobshop", 0, 0)
        self.assertEqual(seed, 8837266590916294613)
        self.assertEqual(driver.candidate_seed(seed), seed % 2**32)


class Wire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.serve = driver.Serve(HARNESS)

    @classmethod
    def tearDownClass(cls):
        cls.serve.close()

    def special(self) -> np.ndarray:
        return np.array([0.0, -0.0, 1.0, -2.5e-7, 1e12, 1e99, 2.2250738585072014e-308,
                         5e-324, 1.7976931348623157e308, math.inf, -math.inf, 0.1 + 0.2,
                         1e-7, 0.9999999, 4.933385123709584])

    def random_bits(self, count: int) -> np.ndarray:
        rng = np.random.default_rng(11)
        return rng.integers(0, 2**64 - 1, size=count, dtype=np.uint64,
                            endpoint=True).view(np.float64)

    def test_vectors_round_trip_bit_for_bit(self):
        values = np.concatenate([self.special(), self.random_bits(4000)])
        nan = np.array([np.uint64(0x7FF8000000000001)]).view(np.float64)
        vectors = [values[i:i + 40] for i in range(0, len(values), 40)] + [nan]
        echoed = self.serve.echo(vectors)
        for a, b in zip(vectors, echoed):
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))

    def test_float_format_equals_rust(self):
        values = np.concatenate([self.special(), self.random_bits(3000),
                                 np.array([math.nan])])
        values = values[~np.isnan(values)].tolist() + [math.nan]
        rust = self.serve.fmt(values)
        for value, expected in zip(values, rust):
            self.assertEqual(driver.format_float(value), expected, repr(value))


class GuardSemantics(unittest.TestCase):
    def test_guard_refuses_over_budget_and_out_of_bounds(self):
        guard = driver.Guard(np.array([-1.0]), np.array([1.0]), 2)
        calls = []

        def batch(vectors):
            calls.extend(vectors)
            return [float(v[0] ** 2) for v in vectors]

        values = guard.evaluate([np.array([0.5]), np.array([2.0]), np.array([0.1]),
                                 np.array([0.0]), np.array([math.nan]),
                                 np.array([0.2, 0.2])], batch)
        self.assertEqual(values[0], 0.25)
        self.assertEqual(values[2], 0.1 * 0.1)
        self.assertTrue(all(math.isinf(v) for v in (values[1], values[3], values[4],
                                                     values[5])))
        self.assertEqual((guard.calls, guard.over_budget, guard.out_of_bounds), (2, 1, 3))
        self.assertEqual(len(calls), 2)
        self.assertEqual(guard.best_y, 0.1 * 0.1)
        np.testing.assert_array_equal(guard.best_x, [0.1])


class FakeStop:
    """CatCMAwM whose stop criterion fires after `generations` tells."""
    generations = 3

    def __init__(self, **kwargs):
        self.inner = driver.check_cmaes()(**kwargs)
        self.tells = 0

    @property
    def population_size(self):
        return self.inner.population_size

    def ask(self):
        return self.inner.ask()

    def tell(self, solutions):
        self.tells += 1
        self.inner.tell(solutions)

    def should_stop(self):
        return self.tells >= self.generations


class Protocol(unittest.TestCase):
    def setUp(self):
        self.space = driver.TypedSpace(toy_table())
        self.calls = 0

    def evaluate(self, vectors):
        self.calls += len(vectors)
        for x in vectors:
            if not (np.all(x >= self.space.lower) and np.all(x <= self.space.upper)):
                raise AssertionError("evaluated outside the box")
        return toy_objective(vectors)

    def test_a_partial_last_generation_is_evaluated_and_ends_the_run(self):
        optimizer_class = driver.check_cmaes()
        popsize = self.space.default_popsize()
        self.assertEqual(popsize, 8)
        settings = driver.settings_for(3, 3 * popsize + 1, 1)
        result = driver.run_repetition(optimizer_class, self.space, 0, settings, self.evaluate)
        self.assertEqual(result["evaluations"], 3 * popsize + 1)
        self.assertEqual(self.calls, 3 * popsize + 1)
        self.assertEqual(result["runs"], 1)
        self.assertEqual((result["over_budget"], result["out_of_bounds"]), (0, 0))
        self.assertEqual(result["seeds"], [driver.run_seed(3, "toy", 0, 0)])
        self.assertTrue(math.isfinite(result["best_value"]))

    def test_early_stops_restart_with_new_seeds_until_less_than_min_run_remains(self):
        settings = driver.settings_for(3, 1000, 100)
        result = driver.run_repetition(FakeStop, self.space, 2, settings, self.evaluate)
        # Each run: 3 generations of 8 = 24 evaluations; restarts while at
        # least 100 remain: 38 runs leave 88 < 100.
        self.assertEqual(result["runs"], 38)
        self.assertEqual(result["evaluations"], 38 * 24)
        self.assertEqual(result["stopped_by_criterion"], 38)
        self.assertEqual(result["seeds"],
                         [driver.run_seed(3, "toy", 2, k) for k in range(38)])
        self.assertEqual(len(set(result["seeds"])), 38)
        self.assertFalse(result["stopped_empty"])
        self.assertEqual(result["seed"], result["seeds"][0])
        self.assertEqual(self.calls, result["evaluations"])

    def test_the_run_allowance_is_never_exceeded(self):
        optimizer_class = driver.check_cmaes()
        for total in (1, 7, 8, 9, 101, 1000):
            self.calls = 0
            settings = driver.settings_for(5, total, 1)
            result = driver.run_repetition(optimizer_class, self.space, 1, settings,
                                           self.evaluate)
            self.assertEqual(result["evaluations"], total)
            self.assertEqual(self.calls, total)
            self.assertEqual((result["over_budget"], result["out_of_bounds"]), (0, 0))

    def test_runs_reproduce(self):
        optimizer_class = driver.check_cmaes()
        settings = driver.settings_for(3, 500, 50)
        a = driver.run_repetition(optimizer_class, self.space, 4, settings, self.evaluate)
        b = driver.run_repetition(optimizer_class, self.space, 4, settings, self.evaluate)
        self.assertEqual(a["best_value"], b["best_value"])
        self.assertEqual(a["evaluations"], b["evaluations"])


class TypeMapping(unittest.TestCase):
    def test_toy_mapping(self):
        space = driver.TypedSpace(toy_table())
        self.assertEqual(space.constructor_arguments(), {
            "x_space": [[-1.0, 1.0], [0.0, 2.0]], "z_space": [[2, 5, 9]], "c_space": [3]})
        solution_class = driver.check_cmaes().Solution
        for z, level in ((2, 0), (5, 1), (9, 2)):
            for k in range(3):
                c = np.zeros((1, 3), dtype=bool)
                c[0, k] = True
                x, chosen = space.to_real(solution_class(np.array([0.25, 1.75]),
                                                         np.array([z]), c))
                self.assertEqual(list(x), [0.25, [0.5, 1.5, 2.5][level],
                                           [1 / 6, 0.5, 5 / 6][k], 1.75])
                self.assertEqual(chosen, [level, k])
        with self.assertRaises(driver.DriverError):
            space.to_real(solution_class(np.array([0.0, 0.0]), np.array([3]),
                                         np.eye(1, 3, dtype=bool)))
        with self.assertRaises(driver.DriverError):
            space.to_real(solution_class(np.array([0.0, 0.0]), np.array([2]),
                                         np.zeros((1, 3), dtype=bool)))

    def test_round_three_spaces_match_the_type_tables(self):
        import json
        optimizer_class = driver.check_cmaes()
        serve = driver.Serve(HARNESS)
        try:
            for problem in driver.PROBLEMS:
                info = serve.info(problem)
                space = driver.TypedSpace.load(problem, info)
                table = json.loads((driver.TYPES_DIR / f"{problem}.json").read_text())
                counts = table["counts"]
                self.assertEqual((len(space.continuous), len(space.integer),
                                  len(space.categorical)),
                                 (counts["continuous"], counts["integer"],
                                  counts["categorical"]))
                optimizer = optimizer_class(seed=1, **space.constructor_arguments())
                self.assertEqual(optimizer.population_size, space.default_popsize())
                coordinates = table["coordinates"]
                for j, index in enumerate(space.integer):
                    c = coordinates[index]
                    self.assertEqual(space.z_space[j], c["values"])
                    for value, center in zip(c["values"], c["levels"]):
                        self.assertEqual(space.z_centers[j][value], center)
                for j, index in enumerate(space.categorical):
                    self.assertEqual(space.c_centers[j], coordinates[index]["levels"])
                for j, index in enumerate(space.continuous):
                    c = coordinates[index]
                    self.assertEqual(space.x_space[j], [c["lower"], c["upper"]])
                # Library samples stay inside the box (it clips x itself).
                for _ in range(200):
                    x, _ = space.to_real(optimizer.ask())
                    self.assertTrue(np.all(x >= space.lower) and np.all(x <= space.upper))
        finally:
            serve.close()


class Bridge(unittest.TestCase):
    def test_bridge_is_exact(self):
        failures = driver.verify_bridge(HARNESS, driver.PROBLEMS, 1000, 200, verbose=False)
        self.assertEqual(failures, [])

    def test_a_wrong_mapping_is_detected(self):
        serve = driver.Serve(HARNESS)
        try:
            space = driver.TypedSpace.load("jobshop", serve.info("jobshop"))
            space.c_centers[0] = list(reversed(space.c_centers[0]))
            solution_class = driver.check_cmaes().Solution
            c = np.zeros((3, 2), dtype=bool)
            c[:, 0] = True
            x, chosen = space.to_real(solution_class(np.full(5, 0.5), None, c))
            levels = serve.levels("jobshop", [x])[0]
            decoded = [int(levels[i]) for i in space.discrete_order()]
            self.assertNotEqual(decoded, chosen)
        finally:
            serve.close()


def read_rows(path: Path) -> list[list[str]]:
    lines = path.read_text().split("\n")
    if lines[0] != driver.RAW_HEADER_T or lines[-1] != "":
        raise AssertionError("bad file")
    return [line.split("\t") for line in lines[1:-1]]


class DriverFile(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="catcmawm-test-"))
        self.out = self.directory / "raw.tsv"

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def drive(self, *extra, out=None, check=True):
        command = [sys.executable, str(HERE / "run_catcmawm.py"), "run",
                   "--out", str(out or self.out), "--problems", "jobshop,material_flow",
                   "--totals", "40,90", "--min-run", "10", "--reps", "2", *extra]
        return subprocess.run(command, check=check, capture_output=True, text=True)

    def rust(self, *extra, check=True):
        command = [HARNESS, "run", "--protocols", "T", "--out", str(self.out),
                   "--problems", "jobshop,material_flow", "--solvers", "bite,cma",
                   "--totals", "40,90", "--min-run", "10", "--root-seed", "3",
                   "--reps", "2", *extra]
        return subprocess.run(command, check=check, capture_output=True, text=True)

    def test_driver_and_runner_fill_one_file_with_one_resume_key(self):
        self.drive()
        rows = read_rows(self.out)
        self.assertEqual(len(rows), 8)
        keys = {(r[0], r[1], r[4], r[5], r[16]) for r in rows}
        self.assertEqual(len(keys), 8)
        self.assertIn("0 runs to do", self.drive().stdout)
        # An interrupted append is dropped and recomputed.
        text = self.out.read_text()
        cut = text.rstrip("\n").rfind("\n") + 1
        self.out.write_text(text[:cut] + "catcmawm\tjobshop\t")
        self.assertIn("1 runs to do", self.drive().stdout)
        self.assertEqual(len(read_rows(self.out)), 8)
        # The Rust runner appends its rows to the same file and resumes.
        self.rust()
        self.assertIn("0 runs to do", self.rust().stdout)
        rows = read_rows(self.out)
        self.assertEqual(len(rows), 8 + 2 * 2 * 2 * 2)
        # Same seed, budget, domain and dimension in the shared columns.
        by_key = {(r[0], r[1], r[5], r[16]): r for r in rows}
        for (solver, problem, rep, total), row in by_key.items():
            if solver != "catcmawm":
                continue
            rust_row = by_key[("bite", problem, rep, total)]
            self.assertEqual(row[1:9], rust_row[1:9])
            self.assertEqual(row[15:], rust_row[15:])
            self.assertEqual((row[11], row[12]), ("0", "0"))
            self.assertLessEqual(int(row[9]), int(total))
            self.assertLess(int(total) - int(row[9]), 10)
        # The analysis reads the file (cells problem x total, 3 solvers).
        report = self.directory / "report.md"
        analyze.main(["--round", "3", "--raw", str(self.out), "--out", str(report),
                      "--reps", "2", "--solvers", "bite,cma,catcmawm",
                      "--problems", "jobshop,material_flow"])
        text = report.read_text()
        self.assertIn("this is a smoke test", text)
        self.assertIn("complete: 4", text)

    def test_two_workers_write_the_same_rows(self):
        other = self.directory / "two.tsv"
        self.drive()
        self.drive("--workers", "2", out=other)

        def strip(path):
            return sorted([r[:13] + r[14:] for r in read_rows(path)])
        self.assertEqual(strip(self.out), strip(other))

    def test_lock_and_settings_are_enforced(self):
        lock = driver.lock_path(self.out)
        # A live process holds the lock (this test process): both refuse.
        lock.write_text(f"{os.getpid()}\n")
        result = self.drive(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("locked", result.stderr)
        result = self.rust(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("locked", result.stderr)
        self.assertFalse(self.out.exists())
        # A stale lock is replaced, and released at the end.
        lock.write_text("4194999\n")
        self.drive()
        self.assertFalse(lock.exists())
        # Rows of other settings are refused by both writers.
        result = self.drive("--root-seed", "4", check=False)
        self.assertIn("settings", result.stderr)
        result = self.drive("--totals", "40", check=False)
        self.assertNotEqual(result.returncode, 0)
        result = self.rust("--min-run", "11", check=False)
        self.assertNotEqual(result.returncode, 0)


class Round3b(unittest.TestCase):
    """The second round-3 batch (ROUND3B.md) through the driver and `serve`."""

    def test_python_seeds_equal_the_harness_seeds(self):
        text = harness("seeds", "--root-seed", "3", "--problems", "round3b",
                       "--totals", "2000,20000", "--reps", "3", "--restarts", "4")
        lines = text.strip().split("\n")[1:]
        self.assertEqual(len(lines), 3 * 2 * 3 * 4)
        self.assertEqual({line.split("\t")[0] for line in lines}, set(driver.PROBLEMS_3B))
        for line in lines:
            problem, _total, rep, restart, seed = line.split("\t")
            self.assertEqual(driver.run_seed(3, problem, int(rep), int(restart)), int(seed))

    def test_sets_and_type_tables(self):
        import json
        self.assertEqual(driver.expand_problems("round3b"), driver.PROBLEMS_3B)
        self.assertEqual(driver.expand_problems("round3"), driver.PROBLEMS)
        self.assertEqual(analyze.PROBLEMS_R3B, driver.PROBLEMS_3B)
        self.assertEqual(set(analyze.CATEGORICAL_R3B), set(driver.PROBLEMS_3B))
        listed = harness("list")
        self.assertIn("round-3b problems: " + ",".join(driver.PROBLEMS_3B), listed)
        optimizer_class = driver.check_cmaes()
        serve = driver.Serve(HARNESS)
        try:
            for problem in driver.PROBLEMS_3B:
                self.assertEqual(driver.default_types_dir(problem), driver.TYPES_DIR_3B)
                info = serve.info(problem)
                space = driver.TypedSpace.load(problem, info)
                table = json.loads((driver.TYPES_DIR_3B / f"{problem}.json").read_text())
                self.assertGreater(table["counts"]["categorical"], 0)
                self.assertEqual((len(space.continuous), len(space.integer),
                                  len(space.categorical)),
                                 (table["counts"]["continuous"], table["counts"]["integer"],
                                  table["counts"]["categorical"]))
                optimizer = optimizer_class(seed=1, **space.constructor_arguments())
                self.assertEqual(optimizer.population_size, space.default_popsize())
                for _ in range(100):
                    x, _ = space.to_real(optimizer.ask())
                    self.assertTrue(np.all(x >= space.lower) and np.all(x <= space.upper))
        finally:
            serve.close()
        for problem in driver.PROBLEMS:
            self.assertEqual(driver.default_types_dir(problem), driver.TYPES_DIR)
        # The welded beam: weld type integer {0, 1}, material categorical,
        # h, t, b integer in sixteenths of an inch.
        space = driver.TypedSpace.load("welded_beam_mixed")
        self.assertEqual(space.z_space[0], [0, 1])
        self.assertEqual(space.z_space[1], list(range(1, 33)))
        self.assertEqual(space.z_space[2], list(range(32, 321)))
        self.assertEqual(space.c_space, [4])
        self.assertEqual(space.x_space, [[0.1, 10.0]])

    def test_bridge_is_exact(self):
        failures = driver.verify_bridge(HARNESS, driver.PROBLEMS_3B, 1000, 200, verbose=False)
        self.assertEqual(failures, [])

    def test_rust_ports_equal_the_reference_code(self):
        self.assertEqual(reference.verify(HARNESS, 500, 200, verbose=False), [])

    def test_driver_runs_the_batch(self):
        directory = Path(tempfile.mkdtemp(prefix="catcmawm-3b-"))
        try:
            out = directory / "raw.tsv"
            command = [sys.executable, str(HERE / "run_catcmawm.py"), "run", "--out", str(out),
                       "--problems", "round3b", "--totals", "60,150", "--min-run", "20",
                       "--reps", "1"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run([HARNESS, "run", "--protocols", "T", "--out", str(out),
                            "--problems", "round3b", "--solvers", "bite,de", "--totals",
                            "60,150", "--min-run", "20", "--root-seed", "3", "--reps", "1"],
                           check=True, capture_output=True, text=True)
            rows = read_rows(out)
            self.assertEqual(len(rows), 3 * 2 * 3)
            for row in rows:
                self.assertEqual((row[11], row[12]), ("0", "0"))
                self.assertLessEqual(int(row[9]), int(row[16]))
                self.assertTrue(math.isfinite(float(row[10])))
        finally:
            shutil.rmtree(directory, ignore_errors=True)


SOLVERS_R3 = analyze.INCUMBENTS_R3 + [analyze.CANDIDATE_R3]


def synthetic_raw(path: Path, problems: list[str], reps: int, candidate_gain: dict,
                  seed: int, root_seed: int = 3, totals=(2000, 20000), skip=None):
    """A protocol-T raw file with normal values: solver i has mean i/10; the
    candidate has its listed gain below 0 on a listed problem and mean 0.3,
    behind the best incumbents, on every other problem."""
    rng = np.random.default_rng(seed)
    lines = [driver.RAW_HEADER_T]
    for problem in problems:
        for total in totals:
            for index, solver in enumerate(SOLVERS_R3):
                for rep in range(reps):
                    if skip == (solver, problem, total, rep):
                        continue
                    loc = index / 10.0
                    if solver == analyze.CANDIDATE_R3:
                        loc = -candidate_gain[problem] if problem in candidate_gain else 0.3
                    value = rng.normal(loc, 0.2)
                    lines.append("\t".join(map(str, [
                        solver, problem, "x", 5, "T", rep, 1000 + rep, total, "1e0", total,
                        driver.format_float(value), 0, 0, "1.000", 1, root_seed, total, total,
                        100])))
    path.write_text("\n".join(lines) + "\n")


class UnionAnalysis(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="analyze-3u-"))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def analyze(self, round_id: str, raw: list[Path], name: str, reps: int = 20) -> str:
        out = self.directory / name
        analyze.main(["--round", round_id, "--raw", ",".join(map(str, raw)),
                      "--out", str(out), "--reps", str(reps)])
        return out.read_text()

    @staticmethod
    def decision(text: str) -> list[str]:
        line = next(line for line in text.splitlines() if line.startswith("| catcmawm |"))
        return [cell.strip() for cell in line.strip("|").split("|")]

    def test_union_gates_use_the_cells_of_both_files(self):
        # Round 3: a clear win on one problem only; round 3b: a clear win on
        # one more problem. G1 needs 3 winning cells on 2 problems: neither
        # file alone meets it, the union does.
        r3 = self.directory / "r3.tsv"
        r3b = self.directory / "r3b.tsv"
        synthetic_raw(r3, analyze.PROBLEMS_R3, 20, {"jobshop": 2.0}, 1)
        synthetic_raw(r3b, analyze.PROBLEMS_R3B, 20, {"pest_control": 2.0}, 2)
        alone3 = self.decision(self.analyze("3", [r3], "r3.md"))
        alone3b = self.decision(self.analyze("3b", [r3b], "r3b.md"))
        union_text = self.analyze("3u", [r3, r3b], "union.md")
        union = self.decision(union_text)
        # Columns: candidate, wins, losses, ties, problems won, mean rank,
        # share, G1, G2, no violation, decision.
        self.assertEqual((alone3[1], alone3[4], alone3[7]), ("2", "1", "no"))
        self.assertEqual((alone3b[1], alone3b[4], alone3b[7]), ("2", "1", "no"))
        self.assertEqual((union[1], union[4], union[7], union[10]), ("4", "2", "yes", "Include"))
        self.assertIn("complete: 18", union_text)
        # Holm over all 18 union cells.
        rows = [line for line in union_text.split("## Win / loss / tie")[1].split("##")[0]
                .splitlines() if line.startswith("| ") and not line.startswith("| Cell")]
        self.assertEqual(len(rows), 18)
        self.assertIn("Categorical: cells of field_service, jobshop, welded_beam_mixed, "
                      "cantilever_mixed, pest_control", union_text)
        self.assertIn("| categorical | 10 |", union_text)
        self.assertIn("| other | 8 |", union_text)
        self.assertNotIn("smoke test", union_text)

    def test_the_union_needs_the_resampling_share(self):
        # Mean rank 1 on every cell gives G2 with share 1.
        r3 = self.directory / "r3.tsv"
        r3b = self.directory / "r3b.tsv"
        gains3 = {p: 0.05 for p in analyze.PROBLEMS_R3}
        gains3b = {p: 0.05 for p in analyze.PROBLEMS_R3B}
        synthetic_raw(r3, analyze.PROBLEMS_R3, 20, gains3, 3)
        synthetic_raw(r3b, analyze.PROBLEMS_R3B, 20, gains3b, 4)
        union = self.decision(self.analyze("3u", [r3, r3b], "u.md"))
        self.assertEqual(union[8], "yes" if float(union[6]) >= 0.95 else "no")

    def test_batch_report_and_refusals(self):
        r3 = self.directory / "r3.tsv"
        r3b = self.directory / "r3b.tsv"
        synthetic_raw(r3, analyze.PROBLEMS_R3, 2, {}, 5)
        synthetic_raw(r3b, analyze.PROBLEMS_R3B, 2, {}, 6, root_seed=999)
        text = self.analyze("3b", [r3b], "b.md", reps=2)
        self.assertTrue(text.startswith("# Candidate optimizers, round 3, second batch"))
        self.assertIn("this is a smoke test", text)
        self.assertIn("The decision on CatCMAwM is taken on the union", text)
        self.assertIn("| other | 0 |", text)
        with self.assertRaises(analyze.AnalysisError):     # mixed root seeds
            self.analyze("3u", [r3, r3b], "x.md", reps=2)
        with self.assertRaises(analyze.AnalysisError):     # one file only
            self.analyze("3u", [r3], "x.md", reps=2)
        with self.assertRaises(analyze.AnalysisError):     # same problems twice
            self.analyze("3u", [r3, r3], "x.md", reps=2)
        with self.assertRaises(analyze.AnalysisError):     # two files for 3b
            self.analyze("3b", [r3b, r3b], "x.md", reps=2)

    def test_round_three_report_is_unchanged_by_the_batch(self):
        # `--round 3` ignores round-3b rows and keeps its texts.
        r3 = self.directory / "r3.tsv"
        synthetic_raw(r3, analyze.PROBLEMS_R3, 3, {}, 7)
        text = self.analyze("3", [r3], "r3.md", reps=3)
        self.assertTrue(text.startswith(
            "# Candidate optimizers, round 3: mixed-variable screening report"))
        self.assertIn("Categorical: cells of field_service, jobshop; other", text)
        self.assertNotIn("union", text)


class Resampling(unittest.TestCase):
    def test_share_is_one_for_a_clear_leader_and_low_for_a_coin_flip(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(6, 3, 20))
        data[:, 0, :] -= 10.0
        self.assertEqual(analyze.resample_share(data, 0, [1, 2], 2000, 3), 1.0)
        same = rng.normal(size=(6, 3, 20))
        share = analyze.resample_share(same, 0, [1, 2], 2000, 3)
        self.assertLess(share, 0.95)
        self.assertEqual(share, analyze.resample_share(same, 0, [1, 2], 2000, 3))


if __name__ == "__main__":
    unittest.main()
