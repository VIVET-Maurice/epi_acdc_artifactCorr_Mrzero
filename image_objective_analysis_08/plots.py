"""Figure functions shared by 04_figures.py and image_objective_results.ipynb.

Every function draws from the arrays saved by 02_mechanisms.py and
05_compare_variants.py and returns the Matplotlib figure. Nothing is simulated
here.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# One palette for every figure: fixed series colours, one diverging pair with
# a neutral midpoint, grey for magnitude images.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"no ACDC": "#2a78d6", "static field shim": "#eb6834",
          "image-optimised (notebook)": "#1baf7a", "perfect B0": "#52514e"}
MODES = {"dynamic": "#1baf7a", "static": "#eda100", "dynamic from best static": "#e87ba4"}
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ef8b8a", "#a82a2a"])
GRAY = "gray"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "legend.frameon": False, "figure.dpi": 150})


def load_results(out_dir=OUT):
    """Arrays and scores written by 02_mechanisms.py."""
    arrays = dict(np.load(Path(out_dir) / "02_arrays.npz"))
    scores = json.loads((Path(out_dir) / "02_mechanisms.json").read_text())
    return arrays, scores


def show(ax, image, title, cmap=GRAY, vmin=0.0, vmax=None, subtitle=None):
    handle = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(title, loc="left")
    if subtitle:
        ax.text(0, -0.04, subtitle, transform=ax.transAxes, va="top", ha="left", color=MUTED, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return handle


def mark_te(ax, te_ms, label=True):
    ax.axvline(te_ms, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    if label:
        ax.text(te_ms, 0.02, " ky = 0 (TE)", transform=ax.get_xaxis_transform(), va="bottom", ha="left",
                color=MUTED, fontsize=8)


def figure_images(a, scores):
    """Target and simulated EPI for the notebook's waveforms, one grey scale."""
    reference = a["reference"]
    vmax = float(np.percentile(reference, 99.5))
    panels = [("Target: measured GRE / gain", reference, None),
              ("Perfect B0, no ACDC", a["perfect"], scores["perfect B0, no ACDC"]["J_vs_GRE"]),
              ("No ACDC", a["image no ACDC"], scores["no ACDC"]["full model"]["J_vs_GRE"]),
              ("Static field shim", a["image static field shim"], scores["static field shim"]["full model"]["J_vs_GRE"]),
              ("Image-optimised (notebook)", a["image image-optimised (notebook)"],
               scores["image-optimised (notebook)"]["full model"]["J_vs_GRE"])]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.3), layout="constrained")
    for ax, (title, image, score) in zip(axes, panels):
        handle = show(ax, image, title, vmax=vmax,
                      subtitle=None if score is None else f"loss against the GRE target: {score:.3f}")
    fig.colorbar(handle, ax=axes, shrink=0.8, pad=0.01, label="magnitude (simulator units)")
    fig.suptitle("Simulated EPI of acquisition 029 against the optimisation target (same grey scale)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    return fig


def figure_time(a):
    """Residual field, k-space energy and slice dephasing along the EPI train."""
    t = 1e3 * a["line_centres"]
    te = t[len(t) // 2]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True, layout="constrained")

    ax = axes[0]
    for name in ("no ACDC", "static field shim", "image-optimised (notebook)"):
        ax.plot(t[:-1], a["field rms per line " + name], color=SERIES[name], label=name)
    ax.set_ylabel("Hz"); ax.set_ylim(0)
    ax.set_title("Residual field at the voxel centres, RMS over the object, line by line", loc="left")
    ax.legend(loc="upper right", ncols=3)
    mark_te(ax, te, label=False)

    ax = axes[1]
    for name, key in (("perfect B0", None), ("no ACDC", "no ACDC"), ("static field shim", "static field shim"),
                      ("image-optimised (notebook)", "image-optimised (notebook)")):
        energy = a["line energy perfect"] if key is None else a["line energy " + key]
        ax.plot(t, 100 * energy / a["line energy perfect"].sum(), color=SERIES[name], label=name)
    ax.set_ylabel("% of the perfect-B0 total"); ax.set_ylim(0)
    ax.set_title("k-space energy acquired in each EPI line", loc="left")
    ax.legend(loc="upper right", ncols=2)
    mark_te(ax, te, label=False)

    ax = axes[2]
    for name in ("static field shim", "image-optimised (notebook)"):
        qz = np.sqrt((a["qz " + name]**2).mean(axis=0))
        ax.plot(t, qz, color=SERIES[name], label=name)
    ax.axhline(1.0, color=MUTED, linewidth=0.8)
    ax.text(t[0], 1.0, "1 cycle: the slice signal cancels", va="bottom", ha="left", color=MUTED, fontsize=8)
    ax.set_ylabel("cycles across the 5 mm slice"); ax.set_ylim(0)
    ax.set_xlabel("time after the RF pulse [ms]")
    ax.set_title("Through-slice phase spread created by the ACDC coils, RMS over the object", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    mark_te(ax, te)
    return fig


def figure_echo(a):
    """Local echo time of every voxel: maps and distribution."""
    te = 1e3 * a["line_centres"][len(a["line_centres"]) // 2]
    names = ("no ACDC", "static field shim", "image-optimised (notebook)")
    fig = plt.figure(figsize=(13.5, 3.6), layout="constrained")
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.06, 1.5])
    for column, name in enumerate(names):
        ax = fig.add_subplot(grid[0, column])
        handle = show(ax, 1e3 * a["echo " + name] - te, name, cmap=DIVERGING, vmin=-30, vmax=30)
    fig.colorbar(handle, cax=fig.add_subplot(grid[0, 3]), label="local echo time minus TE [ms]")
    ax = fig.add_subplot(grid[0, 4])
    bins = np.linspace(0, 120, 61)
    for name in names:
        values = 1e3 * a["echo " + name]
        ax.hist(values[np.isfinite(values)], bins=bins, histtype="step", linewidth=2, color=SERIES[name], label=name)
    mark_te(ax, te)
    ax.set_xlabel("local echo time [ms]"); ax.set_ylabel("voxels")
    ax.set_title("Distribution over the object", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=8)
    fig.suptitle("When each voxel is in phase with its phase-encode neighbours (blue: earlier than TE, less T2* decay)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    return fig


def figure_currents(a):
    """Image-optimised commands and total current per line."""
    t = 1e3 * a["line_centres"]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True, layout="constrained",
                             gridspec_kw=dict(height_ratios=[2.2, 1]))
    extent = [t[0] - 0.44, t[-1] + 0.44, 32.5, 0.5]
    handle = axes[0].imshow(a["currents_optimised"], cmap=DIVERGING, vmin=-3, vmax=3, aspect="auto",
                            extent=extent, interpolation="nearest")
    axes[0].grid(False)
    axes[0].set_ylabel("ACDC output")
    axes[0].set_title("Image-optimised commands, one per EPI line", loc="left")
    fig.colorbar(handle, ax=axes[0], label="current [A]", pad=0.01)
    mark_te(axes[0], t[len(t) // 2])
    for name, key in (("static field shim", "currents_static"), ("image-optimised (notebook)", "currents_optimised")):
        axes[1].plot(t, np.abs(a[key]).sum(axis=0), color=SERIES[name], label=name)
    axes[1].axhline(30, color=MUTED, linewidth=0.8)
    axes[1].text(t[-1], 30, "limit 30 A", va="bottom", ha="right", color=MUTED, fontsize=8)
    axes[1].set_ylim(0, 33); axes[1].set_ylabel("sum of |current| [A]")
    axes[1].set_xlabel("time after the RF pulse [ms]")
    axes[1].set_title("Total current used", loc="left")
    axes[1].legend(loc="lower right", ncols=2)
    mark_te(axes[1], t[len(t) // 2], label=False)
    return fig


def figure_variants(a, out_dir=OUT):
    """Simulated EPI for every waveform of section 5, one grey scale."""
    path = Path(out_dir) / "05_comparison.json"
    if not path.exists():
        print("figure 5 skipped: run 05_compare_variants.py first")
        return None
    report = json.loads(path.read_text())
    images = dict(np.load(Path(out_dir) / "05_images.npz"))
    vmax = float(np.percentile(a["perfect"], 99.5))
    names = list(report)
    columns = 4
    rows = int(np.ceil((len(names) + 1) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.55 * rows), layout="constrained")
    axes = np.atleast_2d(axes)
    handle = show(axes[0, 0], a["perfect"], "Perfect B0, no ACDC", vmax=vmax,
                  subtitle="the self-consistent target")
    for ax, name in zip(axes.ravel()[1:], names):
        row = report[name]
        show(ax, images[name], name, vmax=vmax,
             subtitle=f"vs perfect {row['J_vs_perfect']:.3f} | vs GRE {row['J_vs_GRE']:.3f} | "
                      f"signal {row['signal_vs_perfect']:.2f}")
    for ax in axes.ravel()[len(names) + 1:]:
        ax.set_visible(False)
    fig.colorbar(handle, ax=axes, shrink=0.6, pad=0.01, label="magnitude (simulator units)")
    fig.suptitle("Simulated EPI for every waveform, same grey scale", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    return fig


def figure_convergence(notebook_history, out_dir=OUT):
    """Image loss against iteration for the notebook run and the re-optimisations.

    ``notebook_history`` is the ``relative_mse`` array of the saved notebook
    run. The two panels have different targets, hence different losses.
    """
    out_dir = Path(out_dir)
    files = {"gre static": "03_gre_static.npz", "perfect static": "03_perfect_static.npz",
             "perfect dynamic": "03_perfect_dynamic.npz",
             "perfect dynamic from best static": "03_perfect_dynamic_from_static.npz"}
    if not all((out_dir / name).exists() for name in files.values()):
        print("convergence figure skipped: run 03_optimise_variants.py first")
        return None
    history = {key: np.load(out_dir / name)["history_J_image"] for key, name in files.items()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")

    def curve(ax, values, mode, start=0, va="center"):
        x = start + np.arange(len(values))
        ax.plot(x, values, color=MODES[mode], label=mode)
        ax.annotate(f" {values[-1]:.3f}", (x[-1], values[-1]), color=INK, fontsize=8, va=va)

    ax = axes[0]
    curve(ax, notebook_history, "dynamic")
    curve(ax, history["gre static"], "static")
    ax.axhline(0.307, color=MUTED, linewidth=0.8)
    ax.text(1000, 0.307, "perfect shim: 0.307 ", va="bottom", ha="right", color=MUTED, fontsize=8)
    ax.set_title("Target: measured GRE (the notebook loss)", loc="left")
    ax.set_ylabel("image loss against the GRE")

    ax = axes[1]
    curve(ax, history["perfect dynamic"], "dynamic")
    curve(ax, history["perfect static"], "static", va="bottom")
    curve(ax, history["perfect dynamic from best static"], "dynamic from best static",
          start=len(history["perfect static"]) - 1, va="top")
    ax.axhline(0.329, color=MUTED, linewidth=0.8)
    ax.text(1000, 0.329, "no ACDC: 0.329 ", va="bottom", ha="right", color=MUTED, fontsize=8)
    ax.set_title("Target: perfect-B0 simulated EPI (self-consistent)", loc="left")
    ax.set_ylabel("image loss against the perfect-B0 EPI")
    for ax in axes:
        ax.set_xlabel("iteration"); ax.set_ylim(0, 0.66); ax.set_xlim(0, 1090)
        ax.legend(loc="upper right")
    return fig
