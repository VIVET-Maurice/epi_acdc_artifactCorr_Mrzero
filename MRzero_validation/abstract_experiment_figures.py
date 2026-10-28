"""Reproducible English figures from explicitly identified phantom acquisitions.

This module never optimizes currents, registers an EPI to its target, infers a
CSV-to-scan association, or treats different conditions as technical repeats.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import torch

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "Experiment") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "Experiment"))
from prepare_acdc_inputs import read_gre, read_scan
from read_acdc_sequence import read_sequence
from measured_continuous import reconstruct_receivers
from measured_scan_comparison import (
    _check_patient_geometry, _check_scan, _scan_grid, compare_registered_images,
)
from baseline_metadata_audit import _frequency_from_header

COLORS = ["#777777", "#2374AB", "#D55E00", "#8A4F9E"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def default_studies(project_root):
    """Explicit selections, including the nonstandard ``some-result`` folder."""
    root = Path(project_root)
    seq = "gre_epi_128x128_fov200mm_nav3_cimax_snr_t2star40ms_line_triggers_v6.seq"
    bseq = "megre_b0_s0_t2star_128x128_fov200mm_32echo.seq"
    rseq = "gre_128x128_fov200mm_flip15deg_tr15ms.seq"
    shared = dict(epi_sequence=seq, b0_sequence=bseq, reference_sequence=rseq)
    return [
        dict(shared, id="025", title="Acquisition 025",
             folder=root / "acquisition_025_(06)",
             baseline="meas_MID01838_FID49593_scan.dat",
             reference="meas_MID01840_FID49595_reference.dat",
             reference_mode="raw_gre",
             b0_before="meas_MID01839_FID49594_b0_map.dat",
             b0_static="some-result/meas_MID01848_FID49603_b0_map_staticACDC_shim.dat",
             corrected=[
                 dict(label="Static B0-based", path="results/meas_MID01845_FID49600_static_scan.dat"),
                 dict(label="Dynamic image-domain", path="results/meas_MID01843_FID49598_image_based_correct_scan.dat"),
                 dict(label="Dynamic image-domain\n(zero initialization)",
                      path="results/meas_MID01849_FID49604_image_based_correct_initzero_scan.dat"),
             ]),
        dict(shared, id="026", title="Acquisition 026 — without brace",
             folder=root / "acquisition_026_(06)_without_brace",
             baseline="meas_MID01858_FID49613_scan.dat",
             reference="meas_MID01860_FID49615_reference.dat",
             reference_mode="raw_or_archived_gre",
             reference_archive=root / "MRzero_validation/results_experiment_echo32/mrzero06_20261027_172210_654642/mrzero06_optimization.npz",
             b0_before="meas_MID01859_FID49614_b0_map.dat",
             b0_static="results/meas_MID01863_FID49618_b0_map_staticACDC_shim.dat",
             corrected=[
                 dict(label="Static B0-based", path="results/meas_MID01861_FID49616_static_correction_scan.dat"),
                 dict(label="Dynamic image-domain", path="results/meas_MID01862_FID49617_dyn_image_based_correction_scan.dat"),
             ]),
    ]


def _protocol(path):
    import twixtools
    with contextlib.redirect_stdout(io.StringIO()):
        scans = twixtools.read_twix(str(path), include_scans=[-1], parse_prot=True,
                                  parse_data=False, parse_pmu=False,
                                  parse_geometry=False, verbose=False)
    hdr = scans[-1]["hdr"]
    yaps = hdr["MeasYaps"]
    frequencies = _frequency_from_header(hdr)
    frequency = next(iter(frequencies.values()), None)
    if frequencies and max(frequencies.values()) - min(frequencies.values()) > .5:
        raise ValueError(f"Conflicting scanner-frequency header fields: {path}")
    return dict(frequency_Hz=frequency, observed_frequency_fields_Hz=frequencies,
                scanner_shim_currents=yaps.get("sGRADSPEC", {}).get("alShimCurrent"),
                receiver_gain=yaps.get("sRXSPEC", {}).get("lGain"))


def _archived_reference(study):
    """Require the exact archived baseline/B0/sequence before reusing a GRE."""
    archive = Path(study["reference_archive"])
    report_path = archive.with_suffix(".json")
    report = json.loads(report_path.read_text())
    folder = Path(study["folder"])
    required = {
        "scan": folder / study["baseline"],
        "b0": folder / study["b0_before"],
        "scan_sequence": folder / "sequences" / study["epi_sequence"],
        "b0_sequence": folder / "sequences" / study["b0_sequence"],
        "reference_sequence": folder / "sequences" / study["reference_sequence"],
    }
    for key, path in required.items():
        expected = report.get("source_sha256", {}).get(key)
        if not expected or sha256(path) != expected:
            raise ValueError(f"Archived GRE {key} provenance mismatch: {path}")
    original = report["input_selection"]["reference"]
    if Path(original).name != Path(study["reference"]).name:
        raise ValueError("Archived GRE belongs to a different requested reference.")
    checks = report.get("reference_geometry_checks", [])
    if (not report.get("reference_sequence_validation", {}).get("validated")
            or not all("reference GRE/EPI: " + key in checks for key in
                       ("slice_position_mm", "slice_thickness_mm", "slice_quaternion_wxyz"))):
        raise ValueError("Archived GRE lacks validated geometry/sequence evidence.")
    with np.load(archive, allow_pickle=False) as data:
        ref = {"image": data["measured_GRE_native_magnitude"].copy(),
               "X_m": data["measured_GRE_native_X_m"].copy(),
               "Y_m": data["measured_GRE_native_Y_m"].copy(), "label": "GRE reference"}
    if (ref["image"].ndim != 2 or any(np.asarray(ref[k]).shape != ref["image"].shape
                                     or not np.isfinite(ref[k]).all() for k in ("image", "X_m", "Y_m"))):
        raise ValueError("Invalid archived native GRE arrays.")
    provenance = dict(mode="archived measured GRE reconstruction", archive=str(archive.resolve()),
                      archive_sha256=sha256(archive), report_sha256=sha256(report_path),
                      original_raw_path=original, original_raw_sha256=report["source_sha256"]["reference"],
                      raw_reference_available=False, geometry_checks=checks,
                      protocol=report["baseline_metadata_audit"]["frequency_reference"]["acquisitions"]["GRE"])
    return ref, provenance


def load_study(study, *, reference_threshold_fraction=.1, display_rotation_deg=90,
               navigator_phase_branch=0, device="cpu"):
    """Fresh reconstruction, shared baseline-only gain, strict native geometry."""
    folder = Path(study["folder"])
    acquisition, timing = read_sequence(folder / "sequences" / study["epi_sequence"])
    records, scan_audit = [], []
    base_metadata = None
    sources = [dict(label="Without AC/DC", path=study["baseline"])] + list(study["corrected"])
    for selected in sources:
        path = folder / selected["path"]
        _, metadata, diagnostics = read_scan(path, acquisition, return_intermediates=True,
                                              navigator_phase_branch=navigator_phase_branch)
        _check_scan(metadata, acquisition, timing)
        if base_metadata is None:
            base_metadata = metadata
        else:
            _check_patient_geometry(base_metadata, metadata, selected["label"])
            if abs(metadata["slice_thickness_mm"] - base_metadata["slice_thickness_mm"]) > .001:
                raise ValueError("EPI slice thickness differs across conditions.")
        image, reconstruction = reconstruct_receivers(diagnostics, acquisition, device=device)
        X, Y = _scan_grid(image, metadata, acquisition)
        records.append(dict(image=image, X_m=X, Y_m=Y, label=selected["label"]))
        scan_audit.append(dict(label=selected["label"], path=str(path.resolve()), sha256=sha256(path),
                               metadata=metadata, protocol=_protocol(path), reconstruction=reconstruction,
                               navigator_phase_branch=int(navigator_phase_branch),
                               waveform_match_verified=False,
                               label_source="explicit acquisition selection; filename describes condition"))
    refpath = folder / study["reference"]
    if refpath.is_file() and study["reference_mode"] in ("raw_gre", "raw_or_archived_gre"):
        racq, _ = read_sequence(folder / "sequences" / study["reference_sequence"])
        gre, diag = read_gre(refpath, racq, return_intermediates=True)
        ref_metadata = gre["twix_metadata"]
        _check_patient_geometry(base_metadata, ref_metadata, "GRE reference")
        if abs(ref_metadata["slice_thickness_mm"] - base_metadata["slice_thickness_mm"]) > .001:
            raise ValueError("GRE and EPI slice thickness differs.")
        magnitude, _ = reconstruct_receivers(diag, racq, device=device)
        reference = dict(image=magnitude, X_m=gre["X_m"], Y_m=gre["Y_m"], label="GRE reference")
        reference_audit = dict(mode="raw measured GRE", path=str(refpath.resolve()),
                               sha256=sha256(refpath), raw_reference_available=True, protocol=_protocol(refpath))
    elif study["reference_mode"] == "raw_or_archived_gre":
        reference, reference_audit = _archived_reference(study)
    elif study["reference_mode"] == "multiecho_first":
        from abstract_b0_figures import load_multiecho_maps
        b0 = load_multiecho_maps(folder / study["b0_before"], folder / "sequences" / study["b0_sequence"], device=device)
        reference = dict(image=b0["first_echo"], X_m=b0["X_m"], Y_m=b0["Y_m"], label="Multi-echo GRE: echo 1")
        reference_audit = dict(mode="first measured GRE echo; anatomical reference, not necessarily optimization target",
                               TE_s=float(b0["metadata"]["TE_s"][0]), path=str(folder / study["b0_before"]),
                               protocol=_protocol(folder / study["b0_before"]))
        _check_patient_geometry(base_metadata, b0["metadata"], "First multi-echo GRE")
    else:
        raise FileNotFoundError(f"GRE reference unavailable: {refpath}")
    values = [a["protocol"].get("frequency_Hz") for a in scan_audit]
    ref_frequency = reference_audit["protocol"].get("frequency_Hz")
    if any(v is None for v in values) or ref_frequency is None:
        raise ValueError("Scanner-frequency verification is unavailable.")
    if max(values + [ref_frequency]) - min(values + [ref_frequency]) > .5:
        raise ValueError("Scanner frequency differs across conditions/reference; review acquisition pairing.")
    shim_values = [a["protocol"]["scanner_shim_currents"] for a in scan_audit]
    if any(v is None for v in shim_values):
        raise ValueError("Scanner shim settings are unavailable; identical settings cannot be verified.")
    if any(v != shim_values[0] for v in shim_values[1:]):
        raise ValueError("Scanner shim settings differ across EPI conditions.")
    gains = [a["protocol"]["receiver_gain"] for a in scan_audit]
    if any(v is None for v in gains):
        raise ValueError("Receiver gain is unavailable; identical gain cannot be verified.")
    if any(v != gains[0] for v in gains[1:]):
        raise ValueError("Receiver gain differs across EPI conditions.")
    theta = np.deg2rad(display_rotation_deg)
    frame = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    result = compare_registered_images(reference, records[0], records[1:], frame_matrix=frame,
                                       reference_threshold_fraction=reference_threshold_fraction)
    result.update(study=study, acquisition=acquisition, timing=timing,
                  scan_audit=scan_audit, reference_audit=reference_audit,
                  acquisition_ids=[re.search(r"MID\d+", a["path"]).group() for a in scan_audit])
    result["report"].update(
        study_id=study["id"], reference_provenance=reference_audit, scan_audit=scan_audit,
        same_declared_scanner_frequency=True, same_scanner_shim_settings=True,
        same_receiver_gain=True, technical_repetitions_per_condition=1,
        interpretation="Magnitude disagreement includes contrast, geometric distortion, signal loss and ghosting; not a pure geometric error.",
        label_source="explicit selections based on acquisition filenames; controller load association not verified",
        dynamic_B0_acquisition_available=False,
        EPI_ADC_range_ms=[float(np.min(acquisition["time_s"])) * 1000, float(np.max(acquisition["time_s"])) * 1000])
    return result


def _reference_outline(bundle):
    labels, n = ndimage.label(bundle["mask"])
    if not n:
        raise ValueError("Reference mask is empty.")
    largest = 1 + np.argmax(np.bincount(labels.ravel())[1:])
    return ndimage.binary_fill_holes(labels == largest)


def _physical_image(ax, values, bundle, *, cmap="gray", vmin=0, vmax=1.2):
    image = ax.pcolormesh(bundle["X_m"] * 1000, bundle["Y_m"] * 1000,
                          values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                          rasterized=True)
    ax.set(xlabel="X [mm]", ylabel="Y [mm]", aspect="equal", facecolor="black")
    return image


def plot_epi_comparison(bundle, *, vmax=1.2, show_contour=True):
    """Full FOV; one fixed EPI scale and one GRE contour, no image registration."""
    count = len(bundle["images"])
    fig, axes = plt.subplots(1, count, figsize=(3.25 * count, 4.2), layout="constrained",
                             sharex=True, sharey=True)
    outline = _reference_outline(bundle)
    for i, (ax, values, label) in enumerate(zip(axes, bundle["images"], bundle["labels"])):
        artist = _physical_image(ax, values, bundle, vmax=vmax)
        if show_contour:
            ax.contour(bundle["X_m"] * 1000, bundle["Y_m"] * 1000, outline,
                       levels=[.5], colors=["#56B4E9"], linewidths=.6, alpha=.8)
        suffix = "Measured GRE" if i == 0 else bundle["acquisition_ids"][i - 1]
        ax.set_title(f"{chr(65 + i)}  {label}\n{suffix}", fontsize=10)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.colorbar(artist, ax=list(axes), label="Magnitude [GRE-normalized units]", shrink=.82)
    fig.suptitle(bundle["study"]["title"], fontsize=13)
    return fig


def plot_difference_maps(bundle, *, vmax=.6):
    count = len(bundle["images"]) - 1
    fig, axes = plt.subplots(1, count, figsize=(3.3 * count, 4.2), layout="constrained", sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    reference = bundle["images"][0]
    for ax, image, row in zip(axes, bundle["images"][1:], bundle["rows"]):
        artist = _physical_image(ax, np.abs(image - reference), bundle, cmap="magma", vmax=vmax)
        ax.contour(bundle["X_m"] * 1000, bundle["Y_m"] * 1000, bundle["mask"], levels=[.5], colors="cyan", linewidths=.5)
        ax.set_title(f"{row['label']}\nNRMSE {row['NRMSE_percent']:.1f}%", fontsize=10)
    fig.colorbar(artist, ax=list(axes), label="Absolute magnitude difference [GRE-normalized units]", shrink=.82)
    fig.suptitle(bundle["study"]["title"] + " — supplementary magnitude disagreement", fontsize=12)
    return fig


def plot_metrics_and_profiles(bundles, *, profile_native_y_mm=0., profile_native_x_mm=0.):
    """Fixed physical cuts; descriptive NRMSE without invented error bars."""
    fig, axes = plt.subplots(len(bundles), 3, figsize=(15, 4.0 * len(bundles)),
                             layout="constrained", squeeze=False)
    for bundle, row_axes in zip(bundles, axes):
        ax, readout_ax, phase_ax = row_axes
        rows = bundle["rows"]
        values = [r["NRMSE_percent"] for r in rows]
        bars = ax.bar(np.arange(len(values)), values, color=COLORS[:len(values)], width=.65)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_xticks(np.arange(len(values)), [r["label"].replace("Dynamic image-domain", "Image-domain") for r in rows], fontsize=8)
        ax.tick_params(axis="x", rotation=18)
        ax.set(ylabel="Magnitude NRMSE versus GRE [%]", ylim=(0, max(values) * 1.27),
               title=bundle["study"]["title"] + "\nSingle acquisition per condition")
        X, Y = bundle["native_X_m"], bundle["native_Y_m"]
        iy = int(np.argmin(np.abs(Y[:, 0] * 1000 - profile_native_y_mm)))
        ix = int(np.argmin(np.abs(X[0] * 1000 - profile_native_x_mm)))
        for i, (image, label) in enumerate(zip(bundle["images"], bundle["labels"])):
            color = "black" if i == 0 else COLORS[i - 1]
            readout_ax.plot(X[iy] * 1000, image[iy], color=color, label=label.replace("\n", " "), lw=1.1)
            phase_ax.plot(Y[:, ix] * 1000, image[:, ix], color=color, lw=1.1)
        readout_ax.set(xlabel="Native readout position [mm]", ylabel="Magnitude [GRE-normalized units]",
                       title=f"Fixed profile: native y = {Y[iy, 0] * 1000:.1f} mm")
        phase_ax.set(xlabel="Native phase-encoding position [mm]", ylabel="Magnitude [GRE-normalized units]",
                     title=f"Fixed profile: native x = {X[0, ix] * 1000:.1f} mm")
        readout_ax.legend(fontsize=7, loc="upper right")
        for p in row_axes:
            p.spines[["top", "right"]].set_visible(False)
    return fig


def save_figure(fig, output_dir, stem, caption, *, dpi=300):
    """PNG for slides, PDF/SVG for publication, and an English caption."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in ("png", "pdf", "svg"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths[suffix] = path
    (output / f"{stem}_caption.txt").write_text(caption.strip() + "\n", encoding="utf-8")
    return paths


def export_study(bundle, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = "acquisition_" + bundle["study"]["id"]
    report = dict(bundle["report"], metrics=bundle["rows"])
    (output / f"{prefix}_provenance.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / f"{prefix}_measured_images.npz", images=np.stack(bundle["images"]),
                        labels=np.asarray(bundle["labels"]), mask=bundle["mask"],
                        X_m=bundle["X_m"], Y_m=bundle["Y_m"],
                        native_X_m=bundle["native_X_m"], native_Y_m=bundle["native_Y_m"],
                        gain=bundle["gain"], reference_scale=bundle["reference_scale"])
    with (output / f"{prefix}_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "RMSE", "NRMSE_percent"])
        writer.writeheader()
        writer.writerows(bundle["rows"])
    return report


def plot_echoes_and_timing(multiecho, acquisition, *, title="Measured multi-echo signal and EPI timing",
                           echo_indices=(0, 7, 14, 19, 25, 31), display_rotation_deg=90):
    """Show measured late-echo signal; explicitly show extrapolation past TE32."""
    echoes = multiecho["echo_magnitude"]
    times = np.asarray(multiecho["metadata"]["TE_s"]) * 1000
    indices = [int(i) for i in echo_indices if 0 <= i < len(times)]
    fig = plt.figure(figsize=(15, 7), layout="constrained")
    grid = fig.add_gridspec(2, len(indices), height_ratios=(1.1, 1.))
    theta = np.deg2rad(display_rotation_deg)
    X, Y = multiecho["X_m"], multiecho["Y_m"]
    coords = {"X_m": np.cos(theta) * X - np.sin(theta) * Y,
              "Y_m": np.sin(theta) * X + np.cos(theta) * Y}
    scale = float(np.percentile(echoes[0], 99.8))
    for k, index in enumerate(indices):
        ax = fig.add_subplot(grid[0, k])
        _physical_image(ax, echoes[index] / scale, coords, vmax=1.)
        ax.set_title(f"Echo {index + 1}\nTE {times[index]:.2f} ms", fontsize=10)
    curve_ax = fig.add_subplot(grid[1, :len(indices) // 2])
    support = np.asarray(multiecho["support_mask"], bool)
    curve = np.mean(echoes[:, support], axis=1)
    curve_ax.plot(times, curve / curve[0], "o-", ms=3, color="#2374AB")
    curve_ax.set(xlabel="Echo time [ms]", ylabel="Mean RSS / first-echo mean",
                 title="Measured signal in one fixed support mask")
    timing_ax = fig.add_subplot(grid[1, len(indices) // 2:])
    adc = np.asarray(acquisition["time_s"]) * 1000
    timing_ax.axvspan(times[0], times[-1], color="#56B4E9", alpha=.3, label="Measured echo-time range")
    timing_ax.hlines(1, adc.min(), adc.max(), color="#D55E00", lw=5, label="EPI imaging ADC range")
    timing_ax.scatter(times, np.zeros_like(times), marker="|", color="#2374AB")
    timing_ax.axvline(times[-1], color="#777777", ls="--", lw=1)
    timing_ax.set(xlabel="Time after excitation [ms]", yticks=[0, 1], yticklabels=["MEGRE echoes", "EPI readout"],
                  ylim=(-.6, 1.8), title="EPI samples after the final echo require extrapolation")
    timing_ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=13)
    return fig


__all__ = ["default_studies", "load_study", "plot_epi_comparison", "plot_difference_maps",
           "plot_metrics_and_profiles", "plot_echoes_and_timing", "save_figure", "export_study", "json_safe"]
