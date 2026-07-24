"""MAP-Elites archive visualization."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .io import RunData
from .style import apply_style, save_figure


def _descriptor_specs(metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(item) for item in metadata.get("descriptors", [])]


def _label(spec: Mapping[str, Any]) -> str:
    label = str(spec.get("label", spec["column"]))
    unit = spec.get("unit")
    return f"{label} [{unit}]" if unit else label


def plot_qd(
    run: RunData,
    *,
    output: Optional[str] = None,
    title: Optional[str] = None,
    validation: bool = False,
) -> Figure:
    """Plot a two-dimensional regular or CVT MAP-Elites archive."""

    apply_style()
    table = run.table("qd_archive")
    specs = _descriptor_specs(run.metadata)
    if len(specs) != 2:
        raise ValueError("QD plotting currently requires exactly two descriptors")
    suffix = "validation" if validation else "train"
    columns = [
        spec.get(f"{suffix}_column", f"{spec['column']}_{suffix}")
        for spec in specs
    ]
    quality_column = run.metadata.get("qd", {}).get(
        f"quality_{suffix}_column", f"quality_{suffix}"
    )
    required = [*columns, quality_column]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"missing QD columns: {', '.join(missing)}")

    x = np.asarray(table[columns[0]], dtype=float)
    y = np.asarray(table[columns[1]], dtype=float)
    quality = np.asarray(table[quality_column], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(quality)
    x, y, quality = x[finite], y[finite], quality[finite]

    figure, axis = plt.subplots(figsize=(7.2, 5.5))
    grid_shape = run.metadata.get("qd", {}).get("grid_shape")
    if (
        isinstance(grid_shape, list)
        and len(grid_shape) == 2
        and "grid_x" in table
        and "grid_y" in table
    ):
        columns_count, rows_count = int(grid_shape[0]), int(grid_shape[1])
        image = np.full((rows_count, columns_count), np.nan)
        gx = np.asarray(table["grid_x"], dtype=int)[finite]
        gy = np.asarray(table["grid_y"], dtype=int)[finite]
        inside = (
            (gx >= 0)
            & (gx < columns_count)
            & (gy >= 0)
            & (gy < rows_count)
        )
        image[gy[inside], gx[inside]] = quality[inside]
        bounds = [spec.get("bounds") for spec in specs]
        if all(isinstance(bound, list) and len(bound) == 2 for bound in bounds):
            extent = [
                float(bounds[0][0]),
                float(bounds[0][1]),
                float(bounds[1][0]),
                float(bounds[1][1]),
            ]
        else:
            extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
        colormap = plt.get_cmap("viridis_r").copy()
        colormap.set_bad("#ECEFF1")
        artist = axis.imshow(
            image,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            extent=extent,
            cmap=colormap,
        )
    else:
        artist = axis.scatter(
            x,
            y,
            c=quality,
            cmap="viridis_r",
            s=32,
            edgecolors="none",
        )

    color_label = run.metadata.get("qd", {}).get("quality_label", "Quality (minimized)")
    figure.colorbar(artist, ax=axis, label=color_label)
    axis.set_xlabel(_label(specs[0]))
    axis.set_ylabel(_label(specs[1]))
    axis.set_title(
        title
        or f"{run.metadata['tutorial']}: MAP-Elites ({suffix})"
    )
    figure.tight_layout()
    save_figure(figure, output)
    return figure
