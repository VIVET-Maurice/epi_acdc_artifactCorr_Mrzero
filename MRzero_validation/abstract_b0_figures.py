"""Measured multi-echo/B0 panels for the abstract notebook.

The established Twix, Pulseq and continuous-reconstruction readers perform all
acquisition validation. No spatial smoothing, field extrapolation, frequency
recentering or image registration is added here. Paired field statistics use
one intersection mask, with signal and phase-fit quality checked in each scan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_multiecho_maps(scan_path, sequence_path, field_echo_count=6, device="cpu"):
    """Return measured magnitude echoes and an early-echo B0 estimate.

    All acquired echoes are reconstructed and retained as RSS magnitudes.
    ``field_echo_count`` selects the *first* echoes for the established B0 fit,
    so low signal at late TEs does not silently change the B0 mask. The full
    acquisition and sequence metadata, plus this explicit fit selection, are
    retained. The measured field sign follows ``exp(-2*pi*i*B0*TE)``; an absolute
    phase-unwrapping branch is assumed as described by the existing estimator.
    """
    from read_acdc_b0 import read_b0_coil_images, b0_from_coil_images
    from measured_continuous import reconstruct_b0_echoes
    from baseline_metadata_audit import _header_frequency

    if (isinstance(field_echo_count, bool) or int(field_echo_count) != field_echo_count
            or field_echo_count < 2):
        raise ValueError("field_echo_count must be an integer of at least two.")
    scan_path, sequence_path = Path(scan_path), Path(sequence_path)
    raw = read_b0_coil_images(scan_path, sequence_path)
    reconstruct_b0_echoes(raw, sequence_path, device=device)
    times = np.asarray(raw["metadata"]["TE_s"], dtype=float)
    count = int(field_echo_count)
    if count > len(times):
        raise ValueError(f"Requested {count} B0-fit echoes, but only {len(times)} were acquired.")
    coils = np.asarray(raw["coil_images"])
    field = b0_from_coil_images(coils[:count], times[:count], return_intermediates=True)
    magnitude = np.sqrt(np.sum(np.abs(coils) ** 2, axis=1))
    metadata = dict(raw["metadata"])
    metadata.update({
        "scan_path": str(scan_path.resolve()),
        "sequence_path": str(sequence_path.resolve()),
        "field_echo_count": count,
        "field_echo_indices_zero_based": list(range(count)),
        "field_TE_s": times[:count].tolist(),
        "field_fit": dict(field["metadata"]),
        "magnitude_definition": "RSS after complex continuous reconstruction of each receiver",
        "processing_policy": "No smoothing, extrapolation, field recentering or registration.",
    })
    return {
        "first_echo": magnitude[0],
        "echo_magnitude": magnitude,
        "X_m": np.array(raw["X_m"], copy=True),
        "Y_m": np.array(raw["Y_m"], copy=True),
        "b0_Hz": np.array(field["b0Hz"], copy=True),
        "mask": np.array(field["mask"], dtype=bool, copy=True),
        "support_mask": np.array(field["mask"], dtype=bool, copy=True),
        "phase_residual_rad": np.array(field["phase_residual_rad"], copy=True),
        "metadata": metadata,
        "fit_diagnostic": {
            "phase_residual_rad": np.array(field["phase_residual_rad"], copy=True),
            "magnitude_mask": np.array(field["intermediates"]["magnitude_mask"], copy=True),
            "phase_residual_mask": np.array(field["intermediates"]["phase_residual_mask"], copy=True),
            "metadata": dict(field["metadata"]),
        },
        "scanner_frequency": _header_frequency(scan_path),
    }


def _validate_pair(before, after):
    for data in (before, after):
        shape = np.asarray(data["b0_Hz"]).shape
        if len(shape) != 2 or any(np.asarray(data[key]).shape != shape
                                  for key in ("first_echo", "X_m", "Y_m", "mask")):
            raise ValueError("B0, first-echo magnitude, coordinates and mask must have the same 2D shape.")
    if np.asarray(before["b0_Hz"]).shape != np.asarray(after["b0_Hz"]).shape:
        raise ValueError("B0 pairs must have the same measured image grid; no resampling is performed.")
    for key in ("X_m", "Y_m"):
        if not np.allclose(before[key], after[key], rtol=0, atol=1e-6):
            raise ValueError("B0 pairs have different measured coordinates; no image registration is performed.")
    for key in ("slice_position_mm", "slice_thickness_mm"):
        a, b = before.get("metadata", {}).get(key), after.get("metadata", {}).get(key)
        if a is not None and b is not None and not np.allclose(a, b, rtol=0, atol=1e-3):
            raise ValueError(f"B0 pairs have different {key}; paired field statistics are not valid.")
    a = before.get("metadata", {}).get("slice_quaternion_wxyz")
    b = after.get("metadata", {}).get("slice_quaternion_wxyz")
    if a is not None and b is not None:
        if not np.isclose(abs(np.dot(a, b)), 1, rtol=0, atol=1e-6):
            raise ValueError("B0 pairs have different slice orientations.")


def compare_b0_maps(before, after, *, magnitude_fraction=.05):
    """Compute paired measured statistics without removing either map's mean.

    The returned mask is the intersection of both validated B0 masks, both
    independent high-signal masks and both finite field maps. A scanner-frequency
    mismatch is recorded rather than silently corrected. Equal declared
    frequencies do not establish equal shim fields or absence of temporal drift.
    """
    _validate_pair(before, after)
    if not np.isfinite(magnitude_fraction) or not 0 < magnitude_fraction < 1:
        raise ValueError("magnitude_fraction must lie strictly between zero and one.")
    mask = np.ones(np.asarray(before["b0_Hz"]).shape, dtype=bool)
    for data in (before, after):
        field = np.asarray(data["b0_Hz"], dtype=float)
        magnitude = np.asarray(data["first_echo"], dtype=float)
        valid_magnitude = np.isfinite(magnitude) & (magnitude >= 0)
        peak = float(np.max(magnitude[valid_magnitude])) if valid_magnitude.any() else 0.
        if peak <= 0:
            raise ValueError("A B0 map has no finite positive first-echo magnitude.")
        mask &= (np.asarray(data["mask"], dtype=bool) & np.isfinite(field)
                 & valid_magnitude & (magnitude > magnitude_fraction * peak))
        # Loader masks already enforce this, and retaining the explicit test
        # prevents uncertainty being lost in portable/reconstructed dictionaries.
        phase = data.get("phase_residual_rad")
        threshold = data.get("fit_diagnostic", {}).get("metadata", {}).get("max_phase_residual_rad")
        if phase is not None and threshold is not None:
            phase = np.asarray(phase)
            if phase.shape != mask.shape:
                raise ValueError("Phase-fit quality must share the B0 image grid.")
            mask &= np.isfinite(phase) & (phase <= float(threshold))
    if not mask.any():
        raise ValueError("No common reliable high-signal B0 voxels remain in both scans.")

    def statistics(data):
        values = np.asarray(data["b0_Hz"], dtype=float)[mask]
        return {"n_voxels": int(values.size), "mean_Hz": float(np.mean(values)),
                "spatial_SD_Hz": float(np.std(values, ddof=0)),
                "RMS_Hz": float(np.sqrt(np.mean(values ** 2))),
                "minimum_Hz": float(values.min()), "maximum_Hz": float(values.max())}

    frequency = {"before": before.get("scanner_frequency", {"available": False}),
                 "after": after.get("scanner_frequency", {"available": False}),
                 "same_declared_reference": None, "after_minus_before_Hz": None,
                 "interpretation": "Scanner reference is checked, never fitted or subtracted. Equal declared references do not measure drift or prove identical shim settings."}
    available = all(frequency[key].get("available", False)
                    and frequency[key].get("header_fields_agree", True)
                    for key in ("before", "after"))
    if available:
        difference = float(frequency["after"]["frequency_Hz"] - frequency["before"]["frequency_Hz"])
        frequency.update(after_minus_before_Hz=difference,
                         same_declared_reference=abs(difference) <= .5)
    return {"mask": mask, "before": statistics(before), "after": statistics(after),
            "scanner_frequency": frequency,
            "mask_policy": "One common intersection; each scan independently passes its B0 phase-quality mask and a first-echo high-signal threshold.",
            "magnitude_fraction": float(magnitude_fraction), "field_demeaned": False,
            "geometry_policy": "Same measured grid and slice required; no registration or resampling.",
            "causal_pairing_verified": False,
            "pairing_caveat": "Geometry and metadata checks do not verify command identity or exclude motion, repositioning, shim changes or temporal drift."}


def plot_b0_comparison(before, after, *, title="Measured B0 before and after static AC/DC shimming", limit_Hz=None,
                       display_rotation_deg=0):
    """Plot two measured maps and their distributions with one common mask.

    Map labels state the acquisition condition, not a verified causal pairing.
    The caller remains responsible for confirming which commands were applied.
    Numerical statistics are also available through :func:`compare_b0_maps`.
    ``display_rotation_deg`` applies the same counterclockwise physical display
    rotation to both coordinate grids. Field arrays, masks and statistics remain
    unchanged; no array interpolation or image registration is performed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    comparison = compare_b0_maps(before, after)
    if not np.isscalar(display_rotation_deg) or not np.isfinite(display_rotation_deg):
        raise ValueError("display_rotation_deg must be a finite angle in degrees.")
    theta = np.deg2rad(float(display_rotation_deg))
    X, Y = np.asarray(before["X_m"]), np.asarray(before["Y_m"])
    display_X_mm = (np.cos(theta) * X - np.sin(theta) * Y) * 1000
    display_Y_mm = (np.sin(theta) * X + np.cos(theta) * Y) * 1000
    mask = comparison["mask"]
    values = [np.asarray(data["b0_Hz"])[mask] for data in (before, after)]
    if limit_Hz is None:
        maximum = max(float(np.max(np.abs(v))) for v in values)
        limit_Hz = max(1., np.ceil(maximum / 25) * 25)
    if not np.isfinite(limit_Hz) or limit_Hz <= 0:
        raise ValueError("limit_Hz must be finite and positive.")
    figure = plt.figure(figsize=(13, 4.8), constrained_layout=True)
    grid = figure.add_gridspec(1, 4, width_ratios=(1, 1, .055, 1.05))
    axes = [figure.add_subplot(grid[0, index]) for index in (0, 1)]
    cax, histogram = figure.add_subplot(grid[0, 2]), figure.add_subplot(grid[0, 3])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    norm = Normalize(-float(limit_Hz), float(limit_Hz))
    for axis, data, label, key in zip(axes, (before, after),
                                    ("Before static AC/DC", "After static AC/DC"),
                                    ("before", "after")):
        image = axis.pcolormesh(display_X_mm, display_Y_mm,
                               np.ma.array(data["b0_Hz"], mask=~mask), cmap=cmap, norm=norm,
                               shading="auto", rasterized=True)
        stats = comparison[key]
        axis.set_title(f"{label}\nMean {stats['mean_Hz']:.1f} Hz | SD {stats['spatial_SD_Hz']:.1f} Hz\nRMS {stats['RMS_Hz']:.1f} Hz", fontsize=10)
        axis.set(xlabel="X [mm]", ylabel="Y [mm]", aspect="equal", facecolor="#eeeeee")
    figure.colorbar(image, cax=cax, label="B0 offset (Hz)", extend="both" if any(np.any(np.abs(v) > limit_Hz) for v in values) else "neither")
    all_values = np.concatenate(values)
    edges = np.linspace(float(all_values.min()) - .5, float(all_values.max()) + .5, 45)
    for value, label, color in zip(values, ("Before static AC/DC", "After static AC/DC"), ("#4c78a8", "#f58518")):
        histogram.hist(value, bins=edges, histtype="step", linewidth=1.8, label=label, color=color)
    histogram.axvline(0, color="0.5", linewidth=.8, linestyle="--")
    histogram.set(xlabel="B0 offset (Hz)", ylabel="Voxel count", title=f"Common reliable mask: {int(mask.sum())} voxels")
    histogram.legend(fontsize=8, loc="best")
    histogram.grid(alpha=.2)
    frequency = comparison["scanner_frequency"]
    delta = frequency["after_minus_before_Hz"]
    if delta is None:
        note = "Scanner frequency: unknown or inconsistent header metadata"
    elif frequency["same_declared_reference"]:
        note = f"Scanner frequency matched (difference {delta:+.1f} Hz)"
    else:
        note = f"WARNING: scanner-frequency difference {delta:+.1f} Hz; comparison may be confounded"
    figure.suptitle(f"{title}\n{note}; fields were not demeaned", fontsize=12)
    figure.acdc_b0_comparison = comparison
    figure.acdc_b0_display_rotation_deg = float(display_rotation_deg)
    return figure


__all__ = ["load_multiecho_maps", "compare_b0_maps", "plot_b0_comparison"]
