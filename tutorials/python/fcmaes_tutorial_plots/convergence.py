"""Optimization convergence figures."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .io import RunData
from .style import apply_style, save_figure


def plot_convergence(
    run: RunData,
    *,
    output: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot all documented convergence metrics against actual evaluations."""

    apply_style()
    table = run.table("convergence")
    if "evaluations" not in table:
        raise ValueError("convergence.csv requires an evaluations column")
    x = np.asarray(table["evaluations"], dtype=float)
    configured = run.metadata.get("convergence_metrics")
    if configured:
        metrics = [str(name) for name in configured]
    else:
        metrics = [
            name
            for name, values in table.items()
            if name not in {"evaluations", "elapsed_seconds", "generation"}
            and np.asarray(values).dtype.kind in "fiu"
        ]
    if not metrics:
        raise ValueError("convergence.csv contains no numeric metric")

    figure, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(7.2, max(3.2, 2.45 * len(metrics))),
        sharex=True,
        squeeze=False,
    )
    for axis, metric in zip(axes[:, 0], metrics):
        if metric not in table:
            raise ValueError(f"missing convergence metric {metric!r}")
        axis.plot(x, np.asarray(table[metric], dtype=float), linewidth=1.8)
        axis.set_ylabel(metric.replace("_", " ").title())
    axes[-1, 0].set_xlabel("Evaluations")
    figure.suptitle(
        title or f"{run.metadata['tutorial']}: {run.metadata['formulation']} convergence"
    )
    figure.tight_layout()
    save_figure(figure, output)
    return figure
