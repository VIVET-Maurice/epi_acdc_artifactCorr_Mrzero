"""Small English plotting helpers for the version-07 spiral workflow.

These helpers do not alter acquisition samples, turn boundaries or currents.
The sequence reader is responsible for validating the physical trigger timing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_spiral_trajectory(acquisition, sequence_path=None):
    """Plot actual ADC coordinates with chronological command-turn boundaries.

    ``line_index`` is the shared model's historical name for one-based command
    state labels. In the spiral pipeline those labels mean spiral turns, not
    k-space lines. Commands and ADC storage shapes are independent.
    """
    if acquisition.get("sequence_kind") != "spiral":
        raise ValueError("The spiral trajectory diagnostic requires sequence_kind='spiral'.")
    kx, ky, times = (np.asarray(acquisition[name], dtype=float).reshape(-1)
                     for name in ("kx_mInv", "ky_mInv", "time_s"))
    labels = np.asarray(acquisition["line_index"]).reshape(-1)
    starts = np.asarray(acquisition["line_start_s"], dtype=float).reshape(-1)
    ends = np.asarray(acquisition["line_end_s"], dtype=float).reshape(-1)
    if (not kx.size or ky.shape != kx.shape or times.shape != kx.shape
            or labels.shape != kx.shape or starts.shape != ends.shape
            or np.any(np.diff(times) <= 0) or np.any(ends <= starts)):
        raise ValueError("Spiral diagnostics need matching ordered samples and valid turn windows.")
    turn_ids = np.arange(1, len(starts) + 1)
    counts = np.asarray([np.count_nonzero(labels == turn) for turn in turn_ids])
    if np.any(counts == 0):
        raise ValueError("Every command turn must contain ADC samples.")
    first_sample = np.asarray([np.flatnonzero(labels == turn)[0] for turn in turn_ids])
    figure, axes = plt.subplots(1, 3, figsize=(18, 4.7), layout="constrained")
    axes[0].plot(kx, ky, color="0.7", linewidth=.45, zorder=1)
    artist = axes[0].scatter(kx, ky, c=labels, s=.5, cmap="viridis", rasterized=True)
    axes[0].scatter(kx[first_sample], ky[first_sample], s=17, marker="x",
                    color="tab:red", linewidths=.7, label="First ADC in each triggered turn")
    # A few labels show chronological order without obscuring all 65 revolutions.
    for index in np.unique(np.linspace(0, len(starts)-1, min(6, len(starts)), dtype=int)):
        sample = first_sample[index]
        axes[0].annotate(str(index+1), (kx[sample], ky[sample]),
                         xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].set(title="Actual spiral ADC trajectory", xlabel="kx [cycles/m]",
                ylabel="ky [cycles/m]", aspect="equal")
    axes[0].legend(fontsize=7, loc="lower left")
    figure.colorbar(artist, ax=axes[0], label="Chronological spiral turn", shrink=.8)
    axes[1].plot(times*1000, kx, linewidth=.6, label="kx")
    axes[1].plot(times*1000, ky, linewidth=.6, label="ky")
    for index, start in enumerate(starts):
        axes[1].axvline(start*1000, color="tab:red", alpha=.3, linewidth=.45,
                        label="Turn trigger / command boundary" if index == 0 else None)
    axes[1].set(title="ADC encoding and exact turn starts", xlabel="Time after RF [ms]",
                ylabel="k [cycles/m]")
    axes[1].legend(fontsize=7)
    axes[2].bar(turn_ids, counts, color="tab:blue", width=.8)
    axes[2].set(title="Variable ADC count per spiral turn", xlabel="Spiral turn",
                ylabel="ADC samples")
    duration_axis = axes[2].twinx()
    duration_axis.plot(turn_ids, (ends-starts)*1000, color="tab:orange", linewidth=1,
                       label="Command window")
    duration_axis.set_ylabel("Turn command-window duration [ms]", color="tab:orange")
    duration_axis.tick_params(axis="y", colors="tab:orange")
    figure.suptitle("Spiral sampling and ACDC updates — no Cartesian line approximation")
    image_shape = acquisition.get("image_shape")
    if image_shape is None:
        image_shape = np.asarray(acquisition["matrix_size"])[::-1]
    report = {
        "sequence_file": str(Path(sequence_path).resolve()) if sequence_path is not None else None,
        "trajectory_kind": "spiral", "sample_count": int(kx.size),
        "turn_count": int(len(starts)), "sample_storage_shape": list(np.asarray(acquisition["kx_mInv"]).shape),
        "image_shape": [int(value) for value in image_shape],
        "ADC_samples_per_turn": counts.tolist(), "turn_start_s": starts.tolist(),
        "turn_end_s": ends.tolist(), "ADC_range_after_RF_s": [float(times[0]), float(times[-1])],
        "trajectory_coordinates": "actual Pulseq ADC centres, cycles/m",
        "state_labels": "one-based chronological spiral turns; historical API key line_index",
        "turn_markers": "first ADC after each verified trigger; exact trigger times shown on time axis",
        "rounded_or_discarded_ADC_samples": 0,
    }
    return figure, report


def set_turn_axis_labels(live_plot):
    """Label the shared live current heatmap using spiral state semantics."""
    axis = getattr(live_plot, "_current_axis", None)
    if axis is not None:
        axis.set_xlabel("Chronological spiral turn")
        axis.set_ylabel("Calibration channel")


__all__ = ["plot_spiral_trajectory", "set_turn_axis_labels"]
