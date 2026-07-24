"""Pareto-front figures for two to four objectives."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .io import RunData
from .style import COLORS, apply_style, save_figure


def _specs(metadata: Mapping[str, Any], table: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    configured = metadata.get("objectives", [])
    if configured:
        return [dict(item) for item in configured]
    columns = sorted(name for name in table if name.startswith("objective_"))
    return [{"column": name, "label": name.removeprefix("objective_").replace("_", " ").title()} for name in columns]


def _values(table: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray(table[spec["column"]], dtype=float)
    return result * float(spec.get("display_sign", 1.0)) * float(spec.get("display_scale", 1.0))


def _label(spec: Mapping[str, Any]) -> str:
    label = str(spec.get("label", spec["column"]))
    unit = spec.get("unit")
    return f"{label} [{unit}]" if unit else label


def _mask(table: Mapping[str, np.ndarray], name: str, default: bool) -> np.ndarray:
    count = len(next(iter(table.values())))
    if name not in table:
        return np.full(count, default)
    values = np.asarray(table[name])
    if values.dtype.kind in "fiu":
        return values.astype(float) != 0.0
    return np.asarray([str(value).lower() in {"1", "true", "yes"} for value in values])


def plot_pareto(
    run: RunData,
    *,
    output: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot feasible Pareto points with pairwise projections."""

    apply_style()
    table = run.table("pareto")
    specs = _specs(run.metadata, table)
    if len(specs) < 2:
        raise ValueError("Pareto plotting requires at least two objectives")
    if len(specs) > 4:
        raise ValueError("the tutorial plotter supports at most four objectives")
    for spec in specs:
        if spec.get("column") not in table:
            raise ValueError(f"missing Pareto column {spec.get('column')!r}")

    feasible = _mask(table, "feasible", True)
    selected = _mask(table, "selected", False) & feasible
    values = [_values(table, spec) for spec in specs]
    count = len(specs)

    if count == 2:
        figure, axis = plt.subplots(figsize=(6.8, 5.0))
        axes = np.asarray([[axis]])
        pairs = [(axis, 0, 1)]
    else:
        figure, axes = plt.subplots(
            count - 1,
            count - 1,
            figsize=(3.45 * (count - 1), 3.1 * (count - 1)),
            squeeze=False,
        )
        pairs = []
        for row in range(count - 1):
            for column in range(count - 1):
                axis = axes[row, column]
                if column > row:
                    axis.set_visible(False)
                    continue
                pairs.append((axis, column, row + 1))

    for axis, x_index, y_index in pairs:
        if np.any(~feasible):
            axis.scatter(
                values[x_index][~feasible],
                values[y_index][~feasible],
                s=14,
                color=COLORS["muted"],
                alpha=0.3,
                label="infeasible",
            )
        axis.scatter(
            values[x_index][feasible],
            values[y_index][feasible],
            s=22,
            color=COLORS["primary"],
            alpha=0.78,
            edgecolors="none",
            label="feasible Pareto",
        )
        if np.any(selected):
            axis.scatter(
                values[x_index][selected],
                values[y_index][selected],
                s=65,
                marker="*",
                color=COLORS["selected"],
                edgecolors="black",
                linewidths=0.4,
                label="selected",
                zorder=3,
            )
        axis.set_xlabel(_label(specs[x_index]))
        axis.set_ylabel(_label(specs[y_index]))

    handles, labels = pairs[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=len(handles),
        )
    figure.suptitle(
        title
        or f"{run.metadata['tutorial']}: {run.metadata.get('formulation', 'MO')} Pareto front",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(figure, output)
    return figure
