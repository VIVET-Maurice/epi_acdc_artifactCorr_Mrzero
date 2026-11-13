"""Small display helpers shared by the cells of the generation 10 notebook."""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

# One fixed colour per method, used in every line plot.
METHOD_COLORS = {
    "No ACDC": "#8a8984",
    "Static shim, voxel centres": "#2a78d6",
    "Phase, voxel centres": "#eb6834",
    "Static shim, voxel volume + signal mask": "#1baf7a",
    "Phase, voxel volume + signal mask": "#eda100",
    "Image objective (gen. 08)": "#e87ba4",
}


def use_notebook_style():
    """Quiet axes: light grid, no top/right frame, readable text."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "lines.linewidth": 2.0, "legend.frameon": False,
        "axes.titlesize": 11, "font.size": 10,
    })


def table(rows, headings=("Parameter", "Value")):
    """Display a list of rows as a Markdown table."""
    def text(value):
        return f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)

    lines = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
    lines += ["| " + " | ".join(text(value) for value in row) + " |" for row in rows]
    display(Markdown("\n".join(lines)))


def image_row(phantom, images, titles, label, cmap="gray", vmin=None, vmax=None, suptitle=None):
    """Show images side by side in physical coordinates, with one shared scale.

    Returns the figure. ``images`` are arrays on the phantom grid [rows (y), columns (x)].
    """
    count = len(images)
    figure, axes = plt.subplots(1, count, figsize=(3.4 * count + 0.8, 3.9),
                                layout="constrained", squeeze=False)
    if vmin is None:
        vmin = min(np.nanmin(image) for image in images)
    if vmax is None:
        vmax = max(np.nanmax(image) for image in images)
    for axis, image, title in zip(axes[0], images, titles):
        artist = axis.pcolormesh(phantom.X_m * 1e3, phantom.Y_m * 1e3, image,
                                 cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
        axis.set(title=title.replace(", ", ",\n"), aspect="equal", xlabel="x [mm]")
        axis.set_facecolor("black" if cmap == "gray" else "#f1f0ec")
        axis.grid(False)
    axes[0, 0].set_ylabel("y [mm]")
    figure.colorbar(artist, ax=axes[0], label=label, shrink=0.85)
    if suptitle:
        figure.suptitle(suptitle)
    return figure
