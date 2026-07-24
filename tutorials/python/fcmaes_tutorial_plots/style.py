"""Shared accessible plotting style."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib as mpl
from matplotlib.figure import Figure


COLORS = {
    "primary": "#0072B2",
    "secondary": "#D55E00",
    "selected": "#CC79A7",
    "muted": "#9AA0A6",
    "empty": "#ECEFF1",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "svg.hashsalt": "fcmaes-rust-tutorials-v1",
        }
    )


def save_figure(figure: Figure, output: Optional[Union[str, Path]]) -> None:
    if output is None:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "fcmaes-tutorial-plots", "Date": None}
    figure.savefig(path, bbox_inches="tight", metadata=metadata)
