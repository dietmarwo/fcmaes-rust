"""Versioned result loading and optional PyO3-array adapters."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np


SCHEMA_VERSION = 1


def _column(values: Iterable[str]) -> np.ndarray:
    materialized = list(values)
    try:
        return np.asarray([float(value) for value in materialized], dtype=float)
    except ValueError:
        return np.asarray(materialized, dtype=object)


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    """Load a headered CSV without imposing a pandas dependency."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        rows = list(reader)
    return {
        name: _column(row.get(name, "") for row in rows)
        for name in reader.fieldnames
    }


@dataclass
class RunData:
    """One run manifest plus lazy-loaded artifact tables."""

    metadata: Dict[str, Any]
    root: Path
    tables: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    def artifact_path(self, name: str) -> Path:
        artifact = self.metadata.get("artifacts", {}).get(name)
        if isinstance(artifact, Mapping):
            artifact = artifact.get("path")
        if not isinstance(artifact, str) or not artifact:
            raise KeyError(f"run has no {name!r} artifact")
        path = (self.root / artifact).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"artifact escapes run directory: {artifact}")
        return path

    def table(self, name: str) -> Dict[str, np.ndarray]:
        if name not in self.tables:
            self.tables[name] = load_csv(self.artifact_path(name))
        return self.tables[name]

    def has_artifact(self, name: str) -> bool:
        return name in self.metadata.get("artifacts", {})


def load_run(path: Union[str, Path]) -> RunData:
    """Load and validate a schema-v1 ``run.json`` manifest."""

    manifest = Path(path).resolve()
    with manifest.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {metadata.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if not metadata.get("tutorial") or not metadata.get("formulation"):
        raise ValueError("run.json requires tutorial and formulation")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("run.json requires an artifacts object")
    return RunData(metadata=metadata, root=manifest.parent)


def pareto_from_arrays(
    xs: Sequence[Sequence[float]],
    ys: Sequence[Sequence[float]],
    *,
    objective_names: Optional[Sequence[str]] = None,
    constraint_count: int = 0,
) -> Dict[str, np.ndarray]:
    """Adapt MODE ask/tell arrays to the plotting table convention.

    The caller supplies the final decision matrix and its objective/constraint
    matrix. This avoids requiring additional state from the low-level PyO3
    ``MODE`` class.
    """

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("xs and ys must be row-aligned two-dimensional arrays")
    if constraint_count < 0 or constraint_count >= y.shape[1]:
        raise ValueError("constraint_count must leave at least one objective")
    nobj = y.shape[1] - constraint_count
    names = list(objective_names or [f"objective_{i}" for i in range(nobj)])
    if len(names) != nobj:
        raise ValueError("objective_names length must match objective columns")
    table: Dict[str, np.ndarray] = {
        "point_id": np.arange(x.shape[0]),
        "feasible": (
            np.ones(x.shape[0], dtype=float)
            if constraint_count == 0
            else np.all(y[:, nobj:] <= 0.0, axis=1).astype(float)
        ),
    }
    for index, name in enumerate(names):
        table[name] = y[:, index]
    for index in range(constraint_count):
        table[f"constraint_{index}"] = y[:, nobj + index]
    for index in range(x.shape[1]):
        table[f"decision_{index}"] = x[:, index]
    return table


def qd_from_archive(archive: Any) -> Dict[str, np.ndarray]:
    """Adapt the optional PyO3 ``Archive`` arrays for immediate plotting."""

    ys = np.asarray(archive.ys(), dtype=float)
    xs = np.asarray(archive.xs(), dtype=float)
    descriptors = np.asarray(archive.descriptors(), dtype=float)
    if ys.ndim != 1 or xs.ndim != 2 or descriptors.ndim != 2:
        raise ValueError("archive arrays have unexpected dimensions")
    if xs.shape[0] != ys.size or descriptors.shape[0] != ys.size:
        raise ValueError("archive arrays are not row aligned")
    occupied = np.isfinite(ys)
    table: Dict[str, np.ndarray] = {
        "niche_id": np.flatnonzero(occupied),
        "quality_train": ys[occupied],
    }
    for index in range(descriptors.shape[1]):
        table[f"descriptor_{index}_train"] = descriptors[occupied, index]
    for index in range(xs.shape[1]):
        table[f"decision_{index}"] = xs[occupied, index]
    return table
