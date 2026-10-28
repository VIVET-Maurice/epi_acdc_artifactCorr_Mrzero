"""Validate bounded EPI baseline calibration and its native MRzero application.

Fit/validation/test masks are fixed spatial blocks of one measured acquisition.
They are internal holdouts, not independent repeat scans. A virtual RSS object
does not supply the scanner's complex 32-receiver k-space, so calibration uses
reconstructed magnitude images. No controller or waveform export is performed.

Example from the project root:
    python MRzero_validation/validate_baseline_calibration.py --profile cpu

The default one-point thickness sampling is sufficient for zero-current checks
on the same parents with the declared z-uniform object/B0 and kz=0. Optional
--smoke-iterations 2 tests the calibrated optimizer without claiming convergence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for directory in (PROJECT_ROOT / "Experiment", PROJECT_ROOT / "MRzero_validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from echo_shim_inputs import prepare_echo_shim_inputs
from inplane_geometry import FrameReconstruction
from twin_geometry import compare_magnitude
from twin_inputs import prepare_echo_object, read_twin_inputs
from validate_epi_fidelity import CALIBRATION_SHA256, EPI_SEQUENCE, _run_model, json_safe
from signal_disagreement import signal_disagreement, fixed_comparison_envelope


def split_metrics(measured, images, masks, *, analysis_mask=None):
    """Fit each image's scalar gain using training pixels only, then freeze it."""
    measured = np.abs(np.asarray(measured))
    fit_mask = np.asarray(masks["fit"], bool)
    roi = measured > .2 * measured.max()
    report, arrays = {}, {"measured_signal_roi": roi}
    for name, image in images.items():
        image = np.abs(np.asarray(image))
        training = compare_magnitude(measured, image, mask=fit_mask)
        gain = training["gain"]
        scaled = gain * image
        scores = {"gain_from_fit_pixels_only": gain, "fit": training}
        for split, mask in masks.items():
            scores[split] = compare_magnitude(measured, image, mask=mask, gain=gain)
        scores["full_FOV_descriptive"] = compare_magnitude(measured, image, gain=gain)
        for fraction in (.25, .5):
            label = str(int(100 * fraction))
            hole_mask = roi & (scaled < fraction * measured)
            arrays[f"{name}_holes_{label}_mask"] = hole_mask
            scores[f"holes_{label}"] = {
                split: {"pixels": int(np.count_nonzero(hole_mask & mask)),
                    "signal_ROI_pixels": int(np.count_nonzero(roi & mask))}
                for split, mask in masks.items()}
        disagreement = signal_disagreement(measured, image, gain=gain,
            analysis_mask=analysis_mask, subsets=masks, bright_fraction=.1)
        scores["signal_disagreement"] = disagreement["report"]
        arrays.update({f"{name}_disagreement_{key}": value
                       for key, value in disagreement["arrays"].items()})
        arrays[f"{name}_scaled_image"] = scaled
        report[name] = scores
    report["policy"] = {
        "gain_fit": "Training mask only; gain is frozen for validation and test.",
        "legacy_holes": "Simulation deficits only: below 25% or 50% of measured magnitude, within measured > 20%-peak signal ROI. Excludes real dark regions.",
        "signal_disagreement": "Report both simulation excess (including measured signal loss not reproduced) and simulation deficit; signed squared errors include measured zeros.",
        "full_FOV": "Descriptive; combines pixels used for fitting and holdout pixels.",
        "dataset_independence": "One acquisition; deterministic spatial holdouts, no independent baseline repeat.",
    }
    return report, arrays


def check_same_object(before, after):
    """Allow the fitted physical translation, while keeping signal mass fixed."""
    for key in ("parent_flat_index", "parent_valid_mask", "weights", "native_amplitude", "native_rate_sInv"):
        if not np.array_equal(before[key], after[key], equal_nan=True):
            raise ValueError(f"Baseline calibration changed {key}; native parity would compare different objects.")
    if not np.array_equal(before["expanded_experiment"]["m0_proxy"], after["expanded_experiment"]["m0_proxy"]):
        raise ValueError("Baseline calibration changed the weighted spin amplitudes.")
    return {"same_parents": True, "same_amplitudes": True, "same_decay_rates": True,
            "same_weights": True, "physical_translation_allowed": True,
            "calibration_maps_resampled_at_translated_positions": True}


def _plot(path, arrays, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = arrays["measured_image"]
    before, after = arrays["before_scaled_image"], arrays["applied_scaled_image"]
    X, Y = arrays["X_image_m"] * 1000, arrays["Y_image_m"] * 1000
    split = np.zeros(measured.shape)
    for number, name in enumerate(("fit", "validation", "test"), 1):
        split[arrays[f"{name}_mask"]] = number
    image_panels = [
        (measured, "EPI mesurée", "gray", 0, measured.max()),
        (before, f"Avant calibration\nNRMSE test : {metrics['before']['test']['NRMSE_percent']:.2f} %", "gray", 0, measured.max()),
        (after, f"Après calibration appliquée\nNRMSE test : {metrics['applied']['test']['NRMSE_percent']:.2f} %", "gray", 0, measured.max()),
    ]
    diagnostic_panels = [
        (split, "0 garde / 1 ajustement / 2 validation / 3 test", "viridis", 0, 3),
        (before - measured, "Avant − mesure", "RdBu_r", -measured.max(), measured.max()),
        (after - measured, "Après − mesure", "RdBu_r", -measured.max(), measured.max()),
    ]
    disagreement_panels = [
        (arrays["applied_disagreement_measured_low_simulated_high_50_mask"],
         "EPI réelle faible, simulation trop forte\nMesure < 50 % de la simulation", "Reds", 0, 1),
        (arrays["applied_disagreement_simulated_low_measured_high_50_mask"],
         "Simulation trop faible, EPI réelle forte\nSimulation < 50 % de la mesure", "Blues", 0, 1),
        (arrays["applied_disagreement_both_low_mask"],
         "Signal faible dans les deux images\nCela ne prouve pas leur accord", "Greys", 0, 1),
    ]
    for target, panels in ((path, image_panels), (path.with_stem(path.stem + "_diagnostics"), diagnostic_panels),
                           (path.with_stem(path.stem + "_signal_disagreement"), disagreement_panels)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained", sharex=True, sharey=True)
        if target.stem.endswith("_signal_disagreement"):
            fig.suptitle("Indicateurs descriptifs : luminosité ≥ 10 % du maximum mesuré ; enveloppe fixe de la référence")
        for ax, (values, title, cmap, low, high) in zip(axes, panels):
            im = ax.pcolormesh(X, Y, values, shading="auto", cmap=cmap, vmin=low, vmax=high)
            ax.set(title=title, xlabel="X [mm]", ylabel="Y [mm]", aspect="equal")
            fig.colorbar(im, ax=ax, shrink=.75)
        fig.savefig(target, dpi=140)
        plt.close(fig)


def validate(arguments):
    from baseline_calibration import fit_baseline_calibration, phase_factor
    from baseline_metadata_audit import audit_baseline_inputs
    from experiment_voxel import FiniteVoxelEchoACDCModel

    output = arguments.output.resolve() if arguments.output else (
        PROJECT_ROOT / "MRzero_validation/results_baseline_calibration"
        / datetime.now(timezone.utc).strftime("calibration_%Y%m%d_%H%M%S_%f_UTC"))
    if any((output / f"calibration.{extension}").exists() for extension in ("json", "npz", "png")):
        raise FileExistsError(f"Choose a new output directory: calibration artifacts already exist in {output}.")
    folder = arguments.input_dir.resolve()
    inputs = read_twin_inputs(folder, scan_sequence_path=arguments.epi_sequence,
        b0_sequence_path=arguments.b0_sequence, b0_scan_path=arguments.b0_scan,
        b0_echo_count=arguments.b0_field_echo_count)
    input_audit = audit_baseline_inputs(inputs)
    fit = prepare_echo_object(inputs, arguments.object_threshold_fraction, arguments.background_corner_m,
        support_min_snr=arguments.object_min_snr, allow_uncalibrated_noise=False)
    settings = {"units": "Hz/A", "registration_confirmed": True,
        "patient_to_source_m": np.eye(4).tolist(), "source_sha256": arguments.calibration_sha256,
        "registration_note": "Confirmed Bay5 placement retained from notebooks 05/06."}
    calibration_path = arguments.calibration.resolve() if arguments.calibration else folder / "ACDC_3T.npz"
    common = {"alignment_angle_deg": arguments.alignment_angle_deg,
        "alignment_mirror_x": arguments.alignment_mirror_x,
        "alignment_mirror_y": arguments.alignment_mirror_y,
        "frame_angle_deg": arguments.frame_angle_deg, "subspins_z": arguments.subspins_z,
        "rate_mode": arguments.rate_mode, "b0_mode": "regularized",
        "b0_sigma_pixels": arguments.b0_sigma_pixels, "b0_weighting": "echo_power", "inplane_voxel": True,
        "current_response": {"enabled": True, "tau_s": 133e-6 / np.log(100.)}}
    before = prepare_echo_shim_inputs(inputs, fit, calibration_path, settings, **common)
    initial_arguments = deepcopy(arguments)
    initial_arguments.smoke_iterations = 0
    print(f"Running uncalibrated native MRzero baseline: {before['report']['subspin_count']} spins.", flush=True)
    before_run, base_arrays = _run_model(before, FiniteVoxelEchoACDCModel, initial_arguments,
                                         inputs["scan_diagnostics"]["corrected_kspace"])
    # Calibration is small image-space work; no spin caches are retained.
    # Parameters are expressed on the native readout axis. Joint frame changes
    # leave the image/k-space array order intact but can reverse the kx axis.
    acquisition = inputs["acquisition"]
    reconstruction = FrameReconstruction(before["expanded_experiment"]["acquisition"], "cpu", torch.float64,
        frame_matrix=before["combined_frame_matrix"], sample_chunk=arguments.reconstruction_chunk)
    calibration = fit_baseline_calibration(acquisition, base_arrays["kdata"], base_arrays["measured_image"],
        reconstruction=reconstruction, block_size=arguments.block_size, guard_pixels=arguments.guard_pixels)
    after = prepare_echo_shim_inputs(inputs, fit, calibration_path, settings, **common,
                                     baseline_calibration=calibration["parameters"])
    shared = check_same_object(before, after)
    print("Applied baseline parameters:", json.dumps(json_safe(calibration["parameters"])), flush=True)
    applied_arguments = deepcopy(arguments)
    applied_arguments.sample_chunk = before_run["runtime"]["sample_chunk"]
    after_run, applied_arrays = _run_model(after, FiniteVoxelEchoACDCModel, applied_arguments,
                                          inputs["scan_diagnostics"]["corrected_kspace"])
    if after_run["runtime"]["sample_chunk"] != before_run["runtime"]["sample_chunk"]:
        raise RuntimeError("Available memory changed the effective native MRzero block size between models; "
                           "rerun with a smaller --sample-chunk and stable GPU memory availability.")
    factor = phase_factor(acquisition, calibration["parameters"])
    expected_kdata = base_arrays["kdata"] * factor
    expected_image = reconstruction.numpy_image(expected_kdata)
    parity = {
        "native_kdata_relative_error": float(np.linalg.norm(applied_arrays["kdata"] - expected_kdata)
                                              / np.linalg.norm(expected_kdata)),
        "native_image_relative_error": float(np.linalg.norm(applied_arrays["image"] - expected_image)
                                              / np.linalg.norm(expected_image)),
        "relative_tolerance": 5e-5,
        "same_effective_native_sample_chunk": True,
        "effective_native_sample_chunk": before_run["runtime"]["sample_chunk"],
        "policy": "Rebuilt MRzero with applied parameters versus frozen original MRzero k-space times the fitted common phase factor.",
    }
    parity["passed"] = (parity["native_kdata_relative_error"] < parity["relative_tolerance"]
                         and parity["native_image_relative_error"] < parity["relative_tolerance"])
    if not parity["passed"]:
        raise RuntimeError(f"Native baseline calibration parity failed: {parity}")
    masks = {"fit": np.asarray(calibration["fit_mask"], bool),
             "validation": np.asarray(calibration["validation_mask"], bool),
             "test": np.asarray(calibration["test_mask"], bool)}
    metrics, arrays = split_metrics(base_arrays["measured_image"],
        {"before": base_arrays["image"], "candidate": calibration["candidate_image"],
         "applied": applied_arrays["image"]}, masks,
        analysis_mask=fixed_comparison_envelope(base_arrays["ideal_image"]))
    arrays.update(measured_image=base_arrays["measured_image"], before_image=base_arrays["image"],
        candidate_image=calibration["candidate_image"], applied_image=applied_arrays["image"],
        before_kdata=base_arrays["kdata"], applied_kdata=applied_arrays["kdata"], applied_phase_factor=factor,
        expected_applied_kdata=expected_kdata, before_ideal_image=base_arrays["ideal_image"],
        applied_ideal_image=applied_arrays["ideal_image"], X_image_m=base_arrays["X_image_m"],
        Y_image_m=base_arrays["Y_image_m"], fit_mask=masks["fit"], validation_mask=masks["validation"],
        test_mask=masks["test"], parent_flat_index=before["parent_flat_index"],
        parent_valid_mask=before["parent_valid_mask"], native_amplitude=before["native_amplitude"],
        native_rate_sInv=before["native_rate_sInv"], native_B0_before_Hz=before["native_B0_Hz"],
        native_B0_applied_Hz=after["native_B0_Hz"],
        subspin_xyz_before_patient_m=before["xyz_m"], subspin_xyz_applied_patient_m=after["xyz_m"],
        subspin_basis_before_Hz_per_A=before["fieldbasis"], subspin_basis_applied_Hz_per_A=after["fieldbasis"])
    for key in ("smoke_image", "smoke_kdata", "smoke_currents_A"):
        if key in applied_arrays:
            arrays[key] = applied_arrays[key]
    for key, value in after["expanded_experiment"]["inplane_voxel"].items():
        if isinstance(value, np.ndarray):
            arrays[f"applied_voxel_{key}"] = value
    code_paths = [Path(__file__), *(Path(__file__).parent / name for name in (
        "baseline_calibration.py", "baseline_metadata_audit.py", "signal_disagreement.py", "validate_epi_fidelity.py",
        "echo_shim_inputs.py", "experiment_voxel.py",
        "experiment_echo.py", "experiment_mrzero.py", "experiment_subspins.py", "experiment_realistic.py",
        "twin_inputs.py", "echo_object.py", "b0_regularization.py", "continuous_reconstruction.py",
        "inplane_geometry.py", "native_resummation.py", "acdc_hardware.py", "experiment_optimization.py")),
        PROJECT_ROOT / "Experiment/acdc_gpu.py"]
    report = {"created_UTC": datetime.now(timezone.utc).isoformat(), "configuration": vars(arguments),
        "input_paths": inputs["paths"], "source_sha256": dict(inputs["source_sha256"],
            calibration=before["report"]["calibration_source_sha256"]),
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
        "calibration": calibration["report"], "input_metadata_audit": input_audit,
        "applied_parameters": calibration["parameters"],
        "candidate_parameters": calibration["candidate_parameters"], "metrics": metrics,
        "native_parity": parity, "shared_object_checks": shared, "echo_fit": fit["report"],
        "before_preparation": before["report"], "applied_preparation": after["report"],
        "before_model": before_run, "applied_model": after_run,
        "limitations": ["No independent repeated baseline EPI is available for acquisition_019; these are within-acquisition spatial holdouts.",
            "Test pixels are excluded from parameter fitting and parameter-count selection; any subsequent deployment gate still uses this internal test.",
            "The earlier physical-model choices already used this acquisition; this is not an untouched external validation dataset.",
            "Only low-dimensional bounded baseline parameters are calibrated; no voxelwise image residual is fitted.",
            "The virtual one-receiver model is evaluated against measured RSS magnitudes, not fitted to scanner complex receiver data.",
            "An optimizer smoke test does not validate a corrected physical acquisition or establish convergence."],
        "controller_used": False, "waveform_exported": False,
        "artifacts": {"report": "calibration.json", "arrays": "calibration.npz", "figure": "calibration.png",
                      "split_and_difference_figure": "calibration_diagnostics.png",
                      "signal_disagreement": "calibration_signal_disagreement.png"}}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "calibration.npz", **arrays)
    _plot(output / "calibration.png", arrays, metrics)
    (output / "calibration.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(json_safe({"output": output, "parameters": calibration["parameters"],
        "calibration": calibration["report"], "native_parity": parity, "metrics": metrics}),
        indent=2, ensure_ascii=False))
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "acquisition_019")
    parser.add_argument("--epi-sequence", type=Path, default=Path("sequences") / EPI_SEQUENCE)
    parser.add_argument("--b0-sequence", type=Path)
    parser.add_argument("--b0-scan", type=Path)
    parser.add_argument("--b0-field-echo-count", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-sha256", default=CALIBRATION_SHA256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subspins-z", type=int, default=1)
    parser.add_argument("--rate-mode", choices=("pooled", "guarded", "none"), default="pooled")
    parser.add_argument("--b0-sigma-pixels", type=float, default=1.)
    parser.add_argument("--object-threshold-fraction", type=float, default=.04)
    parser.add_argument("--object-min-snr", type=float, default=5.)
    parser.add_argument("--background-corner-m", type=float, default=.085)
    parser.add_argument("--alignment-angle-deg", type=float, default=180.)
    parser.add_argument("--alignment-mirror-x", action="store_true")
    parser.add_argument("--alignment-mirror-y", action="store_true")
    parser.add_argument("--frame-angle-deg", type=float, default=0.)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--guard-pixels", type=int, default=1)
    parser.add_argument("--profile", choices=("cpu", "auto", "v100", "gtx1660ti", "blackwell"), default="cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "float64"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=.35)
    parser.add_argument("--sample-chunk", type=int, default=128,
                        help="Fixed native ADC block size (default 128); effective size is pinned across the native parity check.")
    parser.add_argument("--reconstruction-chunk", type=int, default=256)
    parser.add_argument("--smoke-iterations", type=int, choices=(0, 1, 2), default=0)
    arguments = parser.parse_args(argv)
    if arguments.subspins_z < 1 or arguments.block_size < 3 or arguments.guard_pixels < 0:
        parser.error("Positive thickness count, block size >= 3 and nonnegative guard pixels are required.")
    if arguments.require_cuda and arguments.profile == "cpu":
        parser.error("--require-cuda requires --profile auto or a GPU profile.")
    return arguments


if __name__ == "__main__":
    validate(parse_arguments())
