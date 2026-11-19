#!/usr/bin/env python3
"""Round-3 candidate driver: CatCMA with Margin on the round-3 problems.

Runs `cmaes.CatCMAwM` (the authors' reference implementation, pinned below,
default parameters) on the round-3 problems under protocol T and appends rows
with solver name `catcmawm` to the raw file of the Rust runner, in the same
format and with the same resume key (ROUND3.md sections 2, 4 and 5;
DEVIATIONS.md, round 3).

Subcommands:

    python3 run_catcmawm.py run --out results/round3/raw.tsv [--reps 20]
        [--problems round3|a,b] [--totals 2000,20000] [--root-seed 3]
        [--min-run 100] [--workers 1] [--harness PATH]
    python3 run_catcmawm.py verify [--vectors 1000] [--typed 200]
    python3 run_catcmawm.py spaces

How it works:

- Variable spaces come from `results/round3/types/<problem>.json`:
  continuous coordinates become `x_space` bounds, integer coordinates
  `z_space` rows with the integer values the decoder computes (`values`),
  categorical coordinates `c_space` category counts.
- A typed sample is mapped to the problem's real vector: a discrete
  coordinate goes to the center of the interval that decodes to the chosen
  level (`levels` of the table), a continuous coordinate takes the sampled
  value. `CatCMAwM.ask` already clips continuous values to `x_space` (the raw
  Gaussian sample can leave the box; the solution's `x` cannot), so the
  driver does not clip.
- Objectives are evaluated by the harness binary (`candidate-harness
  serve`, started once per driver process), one generation per request, with
  every value sent as its IEEE-754 bit pattern.
- A `Guard` per run enforces the run allowance and the box, as
  `solver_api::Guard` does, and records the best value itself.
- Protocol T as in the Rust runner: run k gets min(run cap, remaining)
  evaluations and seed `run_seed(root, problem, 'T', rep, k)`, of which
  `CatCMAwM` gets the low 32 bits (DEVIATIONS.md, round 3, entry 4). A run
  ends when `should_stop()` fires after a generation or when the allowance is
  spent; if a whole population does not fit into what remains, only the
  samples that fit are evaluated (in ask order) and the run ends.
  Restarting ends when fewer than `min_run` evaluations remain, or after a
  run without evaluations.

The Rust runner and this driver must not write one file at the same time.
Both take the lock file `<out>.lock` (holding the process id) before they
read the file and release it at the end; a lock of a process that no longer
exists is replaced. With `--workers 2` the driver runs repetitions in two
worker processes, each with its own `serve`; only the parent process writes
the file.

Besides the raw rows, the driver appends one informational line per row to
`<out stem>.catcmawm.tsv` next to the raw file: population size, runs ended
by `should_stop()`, by the allowance, objective time inside `serve`, time in
the bridge, and wall time. The analysis does not read it.

Every check raises an exception; the script never uses `assert`, so it
behaves identically under `python -O`.
"""

from __future__ import annotations

import os

# One BLAS thread per process: the driver may run two worker processes on two
# cores, and the eigendecompositions of CatCMAwM must not oversubscribe them.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from decimal import Decimal  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CMAES_VERSION = "0.13.1"
SOLVER = "catcmawm"
HERE = Path(__file__).resolve().parent
TYPES_DIR = HERE / "results" / "round3" / "types"

# Round-3 sets and declared settings (ROUND3.md section 5; DEVIATIONS.md,
# round 3). They mirror `harness/src/round3.rs`.
PROBLEMS = ["truss_sizing", "phased_array", "field_service", "jobshop",
            "uav_assignment", "material_flow"]
TOTALS = [2_000, 20_000]
MIN_RUN = 100
ROOT_SEED = 3

RAW_HEADER_T = "\t".join([
    "solver", "problem", "domain", "dim", "protocol", "rep", "seed", "budget",
    "budget_scale", "evaluations", "best_value", "over_budget", "out_of_bounds",
    "wall_ms", "runs", "root_seed", "total", "run_cap", "min_run",
])
DETAIL_HEADER = "\t".join([
    "problem", "total", "rep", "popsize", "runs", "stopped_by_criterion",
    "ended_by_allowance", "evaluations", "objective_ms", "bridge_ms", "wall_ms",
])

MASK64 = (1 << 64) - 1


class DriverError(Exception):
    """A consistency check failed."""


# ------------------------------------------------------------------ library

def check_cmaes():
    """Return `cmaes.CatCMAwM` after checking the pinned version and the MIT
    license of the installed package."""
    try:
        version = metadata.version("cmaes")
    except metadata.PackageNotFoundError as error:
        raise DriverError(f"cmaes is not installed; pip install cmaes=={CMAES_VERSION}") \
            from error
    if version != CMAES_VERSION:
        raise DriverError(f"cmaes {version} is installed; this driver is pinned to "
                          f"{CMAES_VERSION} (pip install cmaes=={CMAES_VERSION})")
    meta = metadata.metadata("cmaes")
    license_text = (meta.get("License-Expression") or meta.get("License") or "").strip()
    classifiers = meta.get_all("Classifier") or []
    if license_text != "MIT" and "License :: OSI Approved :: MIT License" not in classifiers:
        raise DriverError(f"cmaes {version}: license '{license_text}' is not MIT")
    from cmaes import CatCMAwM
    return CatCMAwM


# -------------------------------------------------------------------- seeds

def splitmix64(x: int) -> int:
    """SplitMix64 finalizer, as `harness/src/seeds.rs`."""
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def fnv1a64(text: str) -> int:
    """64-bit FNV-1a of the UTF-8 bytes."""
    h = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & MASK64
    return h


def run_seed(root: int, problem: str, rep: int, restart: int) -> int:
    """`t_run_seed` of the harness: run_seed_from(root, problem, 'T', rep,
    restart). The total does not enter the derivation."""
    h = splitmix64(root & MASK64)
    h = splitmix64(h ^ fnv1a64(problem))
    h = splitmix64(h ^ ord("T"))
    h = splitmix64(h ^ rep)
    return splitmix64(h ^ restart)


def candidate_seed(seed: int) -> int:
    """Low 32 bits of the run seed (DEVIATIONS.md, round 3, entry 4)."""
    return seed & 0xFFFFFFFF


# --------------------------------------------------------------- formatting

def format_float(value: float) -> str:
    """Rust's `{:e}` for f64: shortest round-trip digits in the form
    `d.ddde<exp>` (`1e0`, `-2.5e-7`, `inf`, `NaN`)."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    sign, digits, exponent = Decimal(repr(float(value))).as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    prefix = "-" if sign else ""
    if digits == [0]:
        return f"{prefix}0e0"
    while digits and digits[0] == 0:  # cannot occur for repr, kept for safety
        digits.pop(0)
    power = exponent + len(digits) - 1
    mantissa = str(digits[0]) + ("." + "".join(map(str, digits[1:])) if len(digits) > 1 else "")
    return f"{prefix}{mantissa}e{power}"


def format_row(problem: str, domain: str, dim: int, rep: int, settings: dict,
               result: dict) -> str:
    """One protocol-T raw row (with newline), as `format_row_t` of the
    harness."""
    fields = [
        SOLVER, problem, domain, str(dim), "T", str(rep), str(result["seed"]),
        str(settings["total"]), "1e0", str(result["evaluations"]),
        format_float(result["best_value"]), str(result["over_budget"]),
        str(result["out_of_bounds"]), f"{result['wall_ms']:.3f}", str(result["runs"]),
        str(settings["root_seed"]), str(settings["total"]), str(settings["run_cap"]),
        str(settings["min_run"]),
    ]
    return "\t".join(fields) + "\n"


def settings_for(root_seed: int, total: int, min_run: int) -> dict:
    """Protocol-T settings of one total: run cap equal to the total."""
    return {"root_seed": root_seed, "total": total, "run_cap": total, "min_run": min_run}


# ------------------------------------------------------------------- bridge

def encode_vector(values) -> str:
    """Values as space-separated 16-digit hex bit patterns."""
    raw = np.asarray(values, dtype=np.float64).astype(">f8").tobytes().hex()
    return " ".join(raw[i:i + 16] for i in range(0, len(raw), 16))


def decode_tokens(tokens: list[str]) -> np.ndarray:
    for token in tokens:
        if len(token) != 16:
            raise DriverError(f"bad value token '{token}'")
    return np.frombuffer(bytes.fromhex("".join(tokens)), dtype=">f8").astype(np.float64)


class Serve:
    """Client of `candidate-harness serve` (protocol in harness/src/serve.rs)."""

    def __init__(self, binary: str):
        self.binary = binary
        self.process = subprocess.Popen(
            [binary, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.objective_ns = 0
        self.bridge_ns = 0

    def _request(self, header: str, lines: list[str]) -> list[str]:
        payload = header + "\n" + "".join(line + "\n" for line in lines)
        try:
            self.process.stdin.write(payload.encode("ascii"))
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise DriverError(f"serve exited: {self._stderr_hint()}") from error
        first = self.process.stdout.readline().decode("ascii").rstrip("\n")
        if not first.startswith("ok"):
            raise DriverError(f"serve: {first or 'no response'} (request '{header}')")
        return first.split()

    def _stderr_hint(self) -> str:
        return f"exit status {self.process.poll()}"

    def _lines(self, count: int) -> list[str]:
        return [self.process.stdout.readline().decode("ascii").rstrip("\n")
                for _ in range(count)]

    def info(self, problem: str) -> dict:
        payload = f"info {problem}\n".encode("ascii")
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        first = self.process.stdout.readline().decode("ascii").rstrip("\n")
        if not first.startswith("ok "):
            raise DriverError(f"serve: {first or 'no response'} (info {problem})")
        _, dim, domain = first.split(" ", 2)
        lower, upper = (decode_tokens(line.split()) for line in self._lines(2))
        if len(lower) != int(dim) or len(upper) != int(dim):
            raise DriverError(f"serve info {problem}: bound lengths")
        return {"dim": int(dim), "domain": domain, "lower": lower, "upper": upper}

    def evaluate(self, problem: str, vectors: list[np.ndarray]) -> np.ndarray:
        """Objective values of one batch, bit-exact."""
        if not vectors:
            return np.empty(0)
        started = time.perf_counter_ns()
        tokens = self._request(f"eval {problem} {len(vectors)}",
                               [encode_vector(v) for v in vectors])
        if len(tokens) != 3 or int(tokens[1]) != len(vectors):
            raise DriverError(f"serve eval: unexpected header {tokens}")
        values = decode_tokens(self._lines(len(vectors)))
        self.bridge_ns += time.perf_counter_ns() - started
        self.objective_ns += int(tokens[2])
        return values

    def check(self, problem: str, vectors: list[np.ndarray], values) -> int:
        lines = [encode_vector(np.append(v, y)) for v, y in zip(vectors, values)]
        tokens = self._request(f"check {problem} {len(lines)}", lines)
        return int(tokens[1])

    def levels(self, problem: str, vectors: list[np.ndarray]) -> list[list[str]]:
        tokens = self._request(f"levels {problem} {len(vectors)}",
                               [encode_vector(v) for v in vectors])
        if int(tokens[1]) != len(vectors):
            raise DriverError("serve levels: count")
        return [line.split() for line in self._lines(len(vectors))]

    def echo(self, vectors: list[np.ndarray]) -> list[np.ndarray]:
        tokens = self._request(f"echo {len(vectors)}", [encode_vector(v) for v in vectors])
        if int(tokens[1]) != len(vectors):
            raise DriverError("serve echo: count")
        return [decode_tokens(line.split()) for line in self._lines(len(vectors))]

    def fmt(self, values) -> list[str]:
        tokens = self._request(f"fmt {len(values)}", [encode_vector([v]) for v in values])
        if int(tokens[1]) != len(values):
            raise DriverError("serve fmt: count")
        return self._lines(len(values))

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b"quit\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            self.process.wait(timeout=30)
        self.process.stdout.close()


def default_harness() -> str:
    target = os.environ.get("CARGO_TARGET_DIR") or str(HERE / "target")
    return str(Path(target) / "release" / "candidate-harness")


# ------------------------------------------------------------ typed spaces

class TypedSpace:
    """Variable spaces of one problem from its type table, and the mapping
    of a `CatCMAwM` sample to the problem's real vector."""

    def __init__(self, table: dict, info: dict | None = None):
        self.problem = table["problem"]
        coordinates = table["coordinates"]
        self.dim = int(table["dimension"])
        if len(coordinates) != self.dim:
            raise DriverError(f"{self.problem}: {len(coordinates)} coordinates, D={self.dim}")
        self.lower = np.array([float(c["lower"]) for c in coordinates])
        self.upper = np.array([float(c["upper"]) for c in coordinates])
        if info is not None:
            if info["dim"] != self.dim:
                raise DriverError(f"{self.problem}: harness D={info['dim']}, table D={self.dim}")
            same = (self.lower.view(np.uint64) == info["lower"].view(np.uint64)).all() and \
                (self.upper.view(np.uint64) == info["upper"].view(np.uint64)).all()
            if not same:
                raise DriverError(f"{self.problem}: type-table bounds differ from the harness box")
        self.continuous, self.integer, self.categorical = [], [], []
        self.x_space, self.z_space, self.c_space = [], [], []
        self.z_centers: list[dict] = []     # integer value -> real-vector center
        self.z_level: list[dict] = []       # integer value -> level index
        self.c_centers: list[list[float]] = []
        for i, c in enumerate(coordinates):
            if int(c["index"]) != i:
                raise DriverError(f"{self.problem}: coordinate {i} has index {c['index']}")
            kind = c["kind"]
            if kind == "continuous":
                if not float(c["lower"]) < float(c["upper"]):
                    raise DriverError(f"{self.problem} {i}: empty interval")
                self.continuous.append(i)
                self.x_space.append([float(c["lower"]), float(c["upper"])])
            elif kind == "integer":
                values = [int(v) for v in c["values"]]
                centers = [float(v) for v in c["levels"]]
                if len(values) != len(centers) or len(values) < 2:
                    raise DriverError(f"{self.problem} {i}: levels and values differ")
                if any(b <= a for a, b in zip(values, values[1:])):
                    raise DriverError(f"{self.problem} {i}: values not strictly increasing")
                self.integer.append(i)
                self.z_space.append(values)
                self.z_centers.append(dict(zip(values, centers)))
                self.z_level.append({v: k for k, v in enumerate(values)})
            elif kind == "categorical":
                centers = [float(v) for v in c["levels"]]
                if len(centers) < 2:
                    raise DriverError(f"{self.problem} {i}: fewer than two categories")
                self.categorical.append(i)
                self.c_space.append(len(centers))
                self.c_centers.append(centers)
            else:
                raise DriverError(f"{self.problem} {i}: unknown kind {kind}")
            for center in c.get("levels", []):
                if not float(c["lower"]) <= float(center) <= float(c["upper"]):
                    raise DriverError(f"{self.problem} {i}: level center outside the box")

    @classmethod
    def load(cls, problem: str, info: dict | None = None,
             types_dir: Path = TYPES_DIR) -> "TypedSpace":
        with open(types_dir / f"{problem}.json") as handle:
            return cls(json.load(handle), info)

    def constructor_arguments(self) -> dict:
        """`x_space`, `z_space` and `c_space` for `CatCMAwM` (omitted when
        empty, as the library expects)."""
        arguments = {}
        if self.x_space:
            arguments["x_space"] = self.x_space
        if self.z_space:
            arguments["z_space"] = self.z_space
        if self.c_space:
            arguments["c_space"] = self.c_space
        return arguments

    def default_popsize(self) -> int:
        """`CatCMAwM` default population: 4 + floor(3 ln(N)), N all
        variables."""
        return 4 + math.floor(3 * math.log(self.dim))

    def to_real(self, solution) -> tuple[np.ndarray, list[int]]:
        """Real vector of a typed sample and the chosen level index of every
        discrete coordinate (integer coordinates first, then categorical, in
        coordinate order within each kind)."""
        x = np.empty(self.dim)
        chosen: list[int] = []
        if self.continuous:
            values = np.asarray(solution.x, dtype=np.float64)
            if values.shape != (len(self.continuous),):
                raise DriverError(f"{self.problem}: continuous part of shape {values.shape}")
            x[self.continuous] = values
        if self.integer:
            values = np.asarray(solution.z)
            if values.shape != (len(self.integer),):
                raise DriverError(f"{self.problem}: integer part of shape {values.shape}")
            for j, index in enumerate(self.integer):
                value = int(values[j])
                if value != values[j] or value not in self.z_centers[j]:
                    raise DriverError(f"{self.problem} {index}: integer value {values[j]} "
                                      "is not a level")
                x[index] = self.z_centers[j][value]
                chosen.append(self.z_level[j][value])
        if self.categorical:
            onehot = np.asarray(solution.c)
            if onehot.shape[0] != len(self.categorical):
                raise DriverError(f"{self.problem}: categorical part of shape {onehot.shape}")
            for j, index in enumerate(self.categorical):
                row = onehot[j, :self.c_space[j]]
                hits = np.flatnonzero(row)
                if len(hits) != 1 or np.any(onehot[j, self.c_space[j]:]):
                    raise DriverError(f"{self.problem} {index}: one-hot row {onehot[j]} does "
                                      "not select exactly one category")
                k = int(hits[0])
                x[index] = self.c_centers[j][k]
                chosen.append(k)
        return x, chosen

    def discrete_order(self) -> list[int]:
        """Coordinate indices in the order of `to_real`'s chosen levels."""
        return self.integer + self.categorical


# -------------------------------------------------------------------- guard

class Guard:
    """Budget and box enforcement with the semantics of `solver_api::Guard`:
    a point outside the box (or of the wrong length, or with NaN) is refused
    and counted as out of bounds; a point beyond the allowance is refused and
    counted as over budget; a refused point gets +inf without evaluation. The
    guard records the best value it saw."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray, budget: int):
        self.lower, self.upper, self.budget = lower, upper, budget
        self.calls = 0
        self.over_budget = 0
        self.out_of_bounds = 0
        self.best_y = math.inf
        self.best_x = None

    def evaluate(self, vectors: list[np.ndarray], evaluate_batch) -> list[float]:
        values = [math.inf] * len(vectors)
        accepted = []
        for i, x in enumerate(vectors):
            x = np.asarray(x, dtype=np.float64)
            inside = x.shape == self.lower.shape and bool(
                np.all((x >= self.lower) & (x <= self.upper)))
            if not inside:
                self.out_of_bounds += 1
                continue
            if self.calls < self.budget:
                self.calls += 1
                accepted.append(i)
            else:
                self.over_budget += 1
        if accepted:
            ys = evaluate_batch([vectors[i] for i in accepted])
            if len(ys) != len(accepted):
                raise DriverError("evaluator returned a wrong number of values")
            for i, y in zip(accepted, ys):
                y = float(y)
                values[i] = y
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = np.array(vectors[i], dtype=np.float64)
        return values


# ----------------------------------------------------------------- protocol

def run_once(optimizer_class, space: TypedSpace, guard: Guard, seed32: int,
             evaluate_batch) -> dict:
    """One run of `CatCMAwM` behind `guard`: generations until
    `should_stop()` fires or the allowance is spent; a population that does
    not fit is evaluated only as far as it fits, and the run ends."""
    optimizer = optimizer_class(seed=seed32, **space.constructor_arguments())
    popsize = optimizer.population_size
    stopped = False
    partial = False
    while True:
        left = guard.budget - guard.calls
        if left <= 0:
            break
        size = min(popsize, left)
        samples = [optimizer.ask() for _ in range(size)]
        vectors = [space.to_real(sample)[0] for sample in samples]
        values = guard.evaluate(vectors, evaluate_batch)
        if size < popsize:
            partial = True
            break
        optimizer.tell(list(zip(samples, values)))
        if optimizer.should_stop():
            stopped = True
            break
    return {"popsize": popsize, "stopped": stopped, "partial": partial}


def run_repetition(optimizer_class, space: TypedSpace, rep: int, settings: dict,
                   evaluate_batch) -> dict:
    """One protocol-T repetition, as `run_repetition_t` of the harness."""
    result = {
        "seed": run_seed(settings["root_seed"], space.problem, rep, 0),
        "evaluations": 0, "best_value": math.inf, "over_budget": 0,
        "out_of_bounds": 0, "wall_ms": 0.0, "runs": 0, "stopped_empty": False,
        "stopped_by_criterion": 0, "ended_by_allowance": 0, "popsize": 0,
        "seeds": [],
    }
    remaining = settings["total"]
    while remaining >= max(settings["min_run"], 1):
        allowance = min(settings["run_cap"], remaining)
        seed = run_seed(settings["root_seed"], space.problem, rep, result["runs"])
        guard = Guard(space.lower, space.upper, allowance)
        started = time.perf_counter()
        outcome = run_once(optimizer_class, space, guard, candidate_seed(seed), evaluate_batch)
        result["wall_ms"] += (time.perf_counter() - started) * 1e3
        result["seeds"].append(seed)
        result["popsize"] = outcome["popsize"]
        if outcome["stopped"]:
            result["stopped_by_criterion"] += 1
        else:
            result["ended_by_allowance"] += 1
        calls = guard.calls
        result["evaluations"] += calls
        result["over_budget"] += guard.over_budget
        result["out_of_bounds"] += guard.out_of_bounds
        result["best_value"] = min(result["best_value"], guard.best_y)
        result["runs"] += 1
        remaining -= calls
        if calls == 0:
            result["stopped_empty"] = True
            break
    return result


# ---------------------------------------------------------------- raw file

def lock_path(out: Path) -> Path:
    return Path(str(out) + ".lock")


class RunLock:
    """The lock rule of the Rust runner (`campaign::RunLock`)."""

    def __init__(self, out: Path):
        self.path = lock_path(out)
        self.held = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                holder = self.path.read_text().strip() if self.path.exists() else ""
                alive = holder.isdigit() and Path(f"/proc/{holder}").exists()
                if alive or not holder.isdigit():
                    raise DriverError(
                        f"{self.path.parent / self.path.stem} is locked by process "
                        f"'{holder}' (the Rust runner or run_catcmawm.py); they must not "
                        "write one file concurrently. Remove the lock file only if no such "
                        "process runs.") from None
                print(f"warning: removing stale lock {self.path} of process {holder}",
                      file=sys.stderr)
                self.path.unlink()
                continue
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            self.held = True
            return self
        raise DriverError(f"{self.path}: could not take the lock")

    def __exit__(self, *exc):
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False
        return False


def load_or_create(path: Path, header: str = RAW_HEADER_T) -> list[dict]:
    """Rows of a protocol-T raw file, as `load_or_create_raw_t` of the
    harness: creates the file with its header, drops an incomplete last line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(header + "\n")
        return []
    data = path.read_bytes()
    if not data:
        path.write_text(header + "\n")
        return []
    if not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        print(f"warning: {path}: dropping an incomplete last line ({len(data) - keep} bytes)",
              file=sys.stderr)
        with open(path, "r+b") as handle:
            handle.truncate(keep)
        data = data[:keep]
        if not data:
            path.write_text(header + "\n")
            return []
    lines = data.decode("utf-8").split("\n")[:-1]
    if lines[0] != header:
        raise DriverError(f"{path}: unexpected header")
    names = header.split("\t")
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != len(names):
            raise DriverError(f"{path}:{number}: malformed row")
        row = dict(zip(names, fields))
        try:
            for key in ("rep", "root_seed", "total", "run_cap", "min_run"):
                if key in row:
                    row[key] = int(row[key])
        except ValueError as error:
            raise DriverError(f"{path}:{number}: {error}") from error
        rows.append(row)
    return rows


def row_key(row: dict) -> tuple:
    """Resume key (solver, problem, protocol, repetition, total)."""
    return (row["solver"], row["problem"], row["protocol"], int(row["rep"]), int(row["total"]))


def append_line(path: Path, line: str):
    """Append one complete line with a single write to a file opened for
    appending, as the Rust runner does."""
    data = line.encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        written = os.write(fd, data)
    finally:
        os.close(fd)
    if written != len(data):
        raise DriverError(f"{path}: short write ({written} of {len(data)} bytes)")


def detail_path(out: Path) -> Path:
    return out.with_name(out.stem + ".catcmawm.tsv")


# ------------------------------------------------------------------ workers

_WORKER: dict = {}


def _worker_init(harness: str, types_dir: str):
    _WORKER["class"] = check_cmaes()
    _WORKER["serve"] = Serve(harness)
    _WORKER["types_dir"] = Path(types_dir)
    _WORKER["spaces"] = {}


def _space(problem: str) -> tuple[TypedSpace, dict]:
    spaces = _WORKER["spaces"]
    if problem not in spaces:
        info = _WORKER["serve"].info(problem)
        spaces[problem] = (TypedSpace.load(problem, info, _WORKER["types_dir"]), info)
    return spaces[problem]


def _worker_task(task: tuple) -> tuple:
    problem, rep, settings = task
    serve = _WORKER["serve"]
    space, info = _space(problem)
    objective_before, bridge_before = serve.objective_ns, serve.bridge_ns
    result = run_repetition(_WORKER["class"], space, rep, settings,
                            lambda vectors: serve.evaluate(problem, vectors))
    result["objective_ms"] = (serve.objective_ns - objective_before) / 1e6
    result["bridge_ms"] = (serve.bridge_ns - bridge_before) / 1e6
    line = format_row(problem, info["domain"], info["dim"], rep, settings, result)
    detail = "\t".join(str(v) for v in [
        problem, settings["total"], rep, result["popsize"], result["runs"],
        result["stopped_by_criterion"], result["ended_by_allowance"], result["evaluations"],
        f"{result['objective_ms']:.3f}", f"{result['bridge_ms']:.3f}",
        f"{result['wall_ms']:.3f}"]) + "\n"
    return task, line, detail, result["stopped_empty"], result["runs"]


def _worker_close():
    serve = _WORKER.pop("serve", None)
    if serve is not None:
        serve.close()


# --------------------------------------------------------------- commands

def expand_problems(value: str) -> list[str]:
    names = []
    for name in (v.strip() for v in value.split(",")):
        if not name:
            continue
        names.extend(PROBLEMS if name == "round3" else [name])
    if len(set(names)) != len(names):
        raise DriverError("duplicate problem")
    return names


def parse_totals(value: str) -> list[int]:
    try:
        totals = [int(v) for v in value.split(",") if v.strip()]
    except ValueError as error:
        raise DriverError(f"--totals: {error}") from error
    if not totals or any(t <= 0 for t in totals) or len(set(totals)) != len(totals):
        raise DriverError("--totals needs distinct positive totals")
    return totals


def command_run(args) -> int:
    check_cmaes()
    out = Path(args.out)
    problems = expand_problems(args.problems)
    totals = parse_totals(args.totals)
    if args.reps <= 0 or args.workers <= 0 or args.min_run <= 0 or args.root_seed < 0:
        raise DriverError("--reps, --workers and --min-run must be positive")
    all_settings = [settings_for(args.root_seed, t, args.min_run) for t in totals]
    declared = (args.root_seed, totals, args.min_run) == (ROOT_SEED, TOTALS, MIN_RUN)
    print(f"catcmawm (cmaes {CMAES_VERSION}), protocol T, totals "
          f"{','.join(map(str, totals))}: root seed {args.root_seed}, run cap = total, "
          f"min run {args.min_run}"
          f"{' (the declared round-3 settings)' if declared else ' (not the declared round-3 settings)'}",
          flush=True)
    with RunLock(out):
        existing = load_or_create(out)
        allowed = [(s["root_seed"], s["total"], s["run_cap"], s["min_run"]) for s in all_settings]
        for row in existing:
            settings_key = (row["root_seed"], row["total"], row["run_cap"], row["min_run"])
            if settings_key not in allowed:
                raise DriverError(f"{out} holds rows with protocol-T settings {settings_key} "
                                  f"(current {allowed}); use another output file")
        done = {row_key(row) for row in existing}
        tasks = [(problem, rep, settings)
                 for problem in problems for settings in all_settings
                 for rep in range(args.reps)
                 if (SOLVER, problem, "T", rep, settings["total"]) not in done]
        print(f"{len(tasks)} runs to do ({len(done)} rows already present), "
              f"{args.workers} worker process(es)", flush=True)
        detail = detail_path(out)
        if not detail.exists():
            detail.write_text(DETAIL_HEADER + "\n")
        started = time.perf_counter()
        written = 0

        def record(item):
            nonlocal written
            task, line, detail_line, stopped_empty, runs = item
            append_line(out, line)
            append_line(detail, detail_line)
            written += 1
            problem, rep, settings = task
            if stopped_empty:
                print(f"warning: catcmawm on {problem} total {settings['total']} rep {rep}: "
                      f"run {runs - 1} made no evaluation", file=sys.stderr)
            print(f"done catcmawm {problem} {settings['total']} rep {rep} "
                  f"({written}/{len(tasks)}, elapsed {time.perf_counter() - started:.1f}s)",
                  flush=True)

        harness = args.harness or default_harness()
        if args.workers == 1:
            _worker_init(harness, str(args.types))
            try:
                for task in tasks:
                    record(_worker_task(task))
            finally:
                _worker_close()
        else:
            context = multiprocessing.get_context("spawn")
            with context.Pool(args.workers, initializer=_worker_init,
                              initargs=(harness, str(args.types))) as pool:
                for item in pool.imap_unordered(_worker_task, tasks, chunksize=1):
                    record(item)
    print(f"wrote {written} rows to {out}")
    return 0


def command_spaces(args) -> int:
    optimizer_class = check_cmaes()
    serve = Serve(args.harness or default_harness())
    try:
        print("problem\tD\tcontinuous\tinteger\tcategorical\tpopsize")
        for problem in expand_problems(args.problems):
            space = TypedSpace.load(problem, serve.info(problem), Path(args.types))
            optimizer = optimizer_class(seed=0, **space.constructor_arguments())
            if optimizer.population_size != space.default_popsize():
                raise DriverError(f"{problem}: population {optimizer.population_size}")
            print(f"{problem}\t{space.dim}\t{len(space.continuous)}\t{len(space.integer)}\t"
                  f"{len(space.categorical)}\t{optimizer.population_size}")
    finally:
        serve.close()
    return 0


def random_typed_solution(space: TypedSpace, optimizer_class, rng):
    """A uniformly drawn typed point in the library's `Solution` form."""
    x = rng.uniform(space.lower[space.continuous], space.upper[space.continuous]) \
        if space.continuous else None
    z = np.array([row[rng.integers(len(row))] for row in space.z_space]) \
        if space.z_space else None
    c = None
    if space.c_space:
        c = np.zeros((len(space.c_space), max(space.c_space)), dtype=bool)
        for j, count in enumerate(space.c_space):
            c[j, rng.integers(count)] = True
    return optimizer_class.Solution(x, z, c)


def verify_bridge(harness: str, problems: list[str], vectors: int, typed: int,
                  types_dir: Path = TYPES_DIR, verbose: bool = True) -> list[str]:
    """Bridge checks; returns the failures (empty when exact).

    1. `vectors` uniform random real vectors per problem (plus the two box
       corners): `serve eval` values equal bit for bit the in-process
       evaluation of a separately built instance (`serve check`), and the
       vectors themselves survive the wire (`serve echo`).
    2. `typed` uniform random typed points per problem plus 50 samples of a
       fresh `CatCMAwM`: the real vector of `TypedSpace.to_real` decodes, with
       the problem's own decoder (`serve levels`), to the chosen level of
       every discrete coordinate, and continuous coordinates are passed
       unchanged."""
    optimizer_class = check_cmaes()
    serve = Serve(harness)
    failures = []
    try:
        for problem in problems:
            info = serve.info(problem)
            space = TypedSpace.load(problem, info, types_dir)
            rng = np.random.default_rng(fnv1a64(problem) & 0xFFFFFFFF)
            points = [rng.uniform(info["lower"], info["upper"]) for _ in range(vectors)]
            points += [info["lower"].copy(), info["upper"].copy()]
            values = serve.evaluate(problem, points)
            mismatches = serve.check(problem, points, values)
            echoed = serve.echo(points)
            wire = sum(1 for a, b in zip(points, echoed)
                       if a.shape != b.shape or (a.view(np.uint64) != b.view(np.uint64)).any())
            if mismatches or wire:
                failures.append(f"{problem}: {mismatches} value mismatches, {wire} vectors "
                                "changed on the wire")
            samples = [random_typed_solution(space, optimizer_class, rng) for _ in range(typed)]
            optimizer = optimizer_class(seed=int(rng.integers(2**32)),
                                        **space.constructor_arguments())
            samples += [optimizer.ask() for _ in range(50)]
            mapped = [space.to_real(sample) for sample in samples]
            decoded = serve.levels(problem, [m[0] for m in mapped])
            wrong = 0
            for sample, (x, chosen), levels in zip(samples, mapped, decoded):
                for index, level in zip(space.discrete_order(), chosen):
                    if levels[index] != str(level):
                        wrong += 1
                for j, index in enumerate(space.continuous):
                    if levels[index] != "-" or x[index] != sample.x[j]:
                        wrong += 1
            if wrong:
                failures.append(f"{problem}: {wrong} typed coordinates decode wrongly")
            if verbose:
                print(f"{problem}: {len(points)} vectors bit-identical: {mismatches == 0 and wire == 0}; "
                      f"{len(samples)} typed points decode to their levels: {wrong == 0}",
                      flush=True)
    finally:
        serve.close()
    return failures


def command_verify(args) -> int:
    failures = verify_bridge(args.harness or default_harness(),
                             expand_problems(args.problems), args.vectors, args.typed,
                             Path(args.types))
    if failures:
        for failure in failures:
            print(f"FAILED {failure}", file=sys.stderr)
        return 1
    print("bridge verified")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify", "spaces"):
        p = sub.add_parser(name)
        p.add_argument("--problems", default="round3",
                       help="round3 (default) or a comma-separated list")
        p.add_argument("--harness", default=None,
                       help="candidate-harness binary (default $CARGO_TARGET_DIR/release/"
                            "candidate-harness, else target/release/candidate-harness)")
        p.add_argument("--types", default=str(TYPES_DIR), help="type-table directory")
        if name == "run":
            p.add_argument("--out", required=True, help="raw TSV shared with the Rust runner")
            p.add_argument("--totals", default=",".join(map(str, TOTALS)))
            p.add_argument("--root-seed", type=int, default=ROOT_SEED)
            p.add_argument("--min-run", type=int, default=MIN_RUN)
            p.add_argument("--reps", type=int, default=20)
            p.add_argument("--workers", type=int, default=1,
                           help="worker processes (each with its own serve)")
        if name == "verify":
            p.add_argument("--vectors", type=int, default=1000)
            p.add_argument("--typed", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return {"run": command_run, "verify": command_verify,
            "spaces": command_spaces}[args.command](args)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except DriverError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
