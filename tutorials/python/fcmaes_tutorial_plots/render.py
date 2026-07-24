"""Render every available common artifact for one run."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt

from .convergence import plot_convergence
from .io import RunData
from .pareto import plot_pareto
from .qd import plot_qd


def render_run(run: RunData, output_dir: Union[str, Path]) -> Dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rendered: Dict[str, Path] = {}
    if run.has_artifact("pareto"):
        path = destination / "pareto.svg"
        figure = plot_pareto(run, output=str(path))
        plt.close(figure)
        rendered["pareto"] = path
    if run.has_artifact("qd_archive"):
        path = destination / "qd-archive.svg"
        figure = plot_qd(run, output=str(path))
        plt.close(figure)
        rendered["qd_archive"] = path
        table = run.table("qd_archive")
        if "quality_validation" in table:
            path = destination / "qd-archive-validation.svg"
            figure = plot_qd(run, output=str(path), validation=True)
            plt.close(figure)
            rendered["qd_archive_validation"] = path
    if run.has_artifact("convergence"):
        path = destination / "convergence.svg"
        figure = plot_convergence(run, output=str(path))
        plt.close(figure)
        rendered["convergence"] = path
    return rendered
