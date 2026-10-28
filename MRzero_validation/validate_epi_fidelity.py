"""Compare measured EPI with two zero-current MRzero twins on identical parents.

The legacy model uses uniform local-plane B0 regularization and point voxels.
The candidate uses echo-power weighted B0 planes and analytic in-plane boxes.
Only an evaluation intensity gain is fitted to measured EPI. No image-driven
B0, relaxation, geometry, smoothing or voxel-size parameter search is performed.

Example, from the project root (one thickness point is exact at zero current
for the assumed z-uniform object/B0 and kz=0):

    python MRzero_validation/validate_epi_fidelity.py --subspins-z 1 --profile cpu

No controller or CSV export is used. Optional --smoke-iterations 2 exercises
the candidate optimizer locally, without claiming optimization convergence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for directory in (PROJECT_ROOT / "Experiment", PROJECT_ROOT / "MRzero_validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from echo_shim_inputs import prepare_echo_shim_inputs
from experiment_echo import EchoSubspinACDCModel
from inplane_geometry import FrameReconstruction
from twin_geometry import compare_magnitude
from twin_inputs import prepare_echo_object, read_twin_inputs
from signal_disagreement import signal_disagreement, fixed_comparison_envelope


EPI_SEQUENCE = "gre_epi_128x128_fov200mm_nav3_dwell5us_rise240us_ext1_single_trigger_first_readout_v5.seq"
CALIBRATION_SHA256 = "bace6a584e780d80af8f2ca7c12957bd5b9dc7462be0968f949cc9a99243a06c"


def json_safe(value):
    """Keep reports strict JSON even when a rejected fit has infinite T2*."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compare_baselines(measured, legacy, candidate, X_m, Y_m, *, reference_image=None):
    """Compare both directions of signal error; retain one-sided legacy counts."""
    measured, legacy, candidate = (np.abs(np.asarray(item))
                                   for item in (measured, legacy, candidate))
    if (measured.ndim != 2 or legacy.shape != measured.shape
            or candidate.shape != measured.shape
            or not all(np.isfinite(item).all() for item in (measured, legacy, candidate))):
        raise ValueError("Measured and simulated images must share one finite 2D grid.")
    roi = measured > .20 * measured.max()
    if not roi.any():
        raise ValueError("The measured 20%-of-peak ROI is empty.")
    reports, arrays = {}, {"measured_image": measured, "measured_signal_roi": roi,
                           "X_image_m": np.asarray(X_m), "Y_image_m": np.asarray(Y_m)}
    analysis_mask = None if reference_image is None else fixed_comparison_envelope(reference_image)
    for name, image in (("legacy", legacy), ("candidate", candidate)):
        full = compare_magnitude(measured, image, X_m=X_m, Y_m=Y_m)
        scaled = image * full["gain"]
        roi_metrics = compare_magnitude(measured, image, mask=roi, gain=full["gain"],
                                        X_m=X_m, Y_m=Y_m)
        holes = {}
        for fraction in (.25, .50):
            label = str(int(100 * fraction))
            mask = roi & (scaled < fraction * measured)
            holes[label] = {"threshold_fraction_of_measured": fraction,
                            "pixels": int(mask.sum()),
                            "fraction_of_measured_signal_roi": float(mask.sum() / roi.sum())}
            arrays[f"{name}_holes_{label}_mask"] = mask
        reports[name] = {"full_FOV": full, "measured_signal_ROI": roi_metrics,
                         "holes": holes}
        disagreement = signal_disagreement(measured, image, gain=full["gain"],
            analysis_mask=analysis_mask, bright_fraction=.1)
        reports[name]["signal_disagreement"] = disagreement["report"]
        arrays.update({f"{name}_disagreement_{key}": value
                       for key, value in disagreement["arrays"].items()})
        arrays[f"{name}_image"] = image
        arrays[f"{name}_scaled_image"] = scaled
    reports["comparison_policy"] = {
        "intensity_gain": "One positive least-squares gain per twin, fitted over the full image only for evaluation.",
        "ROI": "Measured RSS > 20% of the measured full-image peak; identical for both twins.",
        "legacy_holes": "Simulation deficits only: scaled simulation < 25% or < 50% of measured RSS inside the measured-signal ROI. Excludes real dark regions.",
        "signal_disagreement": "Signed full-FOV residuals include measured zeros. Two-way ratio diagnostics also identify measured low signal not reproduced by the model.",
        "registration_or_image_parameter_fit": False,
        "signal_ROI_pixels": int(roi.sum()),
        "signal_ROI_threshold": float(.20 * measured.max()),
    }
    reports["candidate_minus_legacy_NRMSE_percentage_points"] = (
        reports["candidate"]["full_FOV"]["NRMSE_percent"]
        - reports["legacy"]["full_FOV"]["NRMSE_percent"])
    for label in ("25", "50"):
        before = reports["legacy"]["holes"][label]["pixels"]
        after = reports["candidate"]["holes"][label]["pixels"]
        reports[f"relative_reduction_holes_{label}"] = (float((before - after) / before)
                                                         if before else None)
    return reports, arrays


def assert_shared_object(legacy, candidate):
    """Reject comparisons confounded by support, amplitudes or relaxation."""
    for key in ("parent_flat_index", "parent_valid_mask", "weights", "native_amplitude",
                "native_rate_sInv", "frame_X_m", "frame_Y_m"):
        if not np.array_equal(legacy[key], candidate[key], equal_nan=True):
            raise ValueError(f"The two preparations must retain identical {key}.")
    for key in ("m0_proxy", "amplitude_reference_time_s"):
        if not np.array_equal(legacy["expanded_experiment"][key],
                              candidate["expanded_experiment"][key]):
            raise ValueError(f"The two preparations changed {key}.")
    return {"same_parent_indices": True, "same_parent_mask": True,
            "same_amplitude_and_rates": True, "same_volume_weights": True,
            "same_physical_and_output_coordinates": True}


def _run_model(prepared, model_type, arguments, measured_kspace):
    frame = prepared["combined_frame_matrix"]

    def reconstruction_factory(acquisition, device, dtype):
        return FrameReconstruction(acquisition, device, dtype, frame_matrix=frame,
                                   sample_chunk=arguments.reconstruction_chunk)

    started = perf_counter()
    model = model_type(prepared["expanded_experiment"], prepared["expanded_calibration"],
        max_current_A=3., reconstruction_factory=reconstruction_factory,
        hardware={"min_update_interval_s": 8e-6, "dac_resolution_A": .0006},
        profile=arguments.profile, precision=arguments.precision,
        memory_fraction=arguments.memory_fraction, sample_chunk=arguments.sample_chunk,
        require_cuda=arguments.require_cuda)
    zeros = np.zeros((model.nc, model.nstates))
    image, kdata = model.forward(zeros)
    receivers = model.reconstruction.numpy_image(np.moveaxis(measured_kspace, 1, 0))
    measured = np.sqrt(np.sum(abs(receivers)**2, axis=0))
    arrays = {"image": np.array(image, copy=True), "kdata": np.array(kdata, copy=True),
              "ideal_image": np.array(model.ideal_image, copy=True),
              "measured_image": measured,
              "X_image_m": np.array(model.reconstruction.X_m, copy=True),
              "Y_image_m": np.array(model.reconstruction.Y_m, copy=True)}
    report = {"runtime": deepcopy(model.runtime_info), "zero_current": True,
              "construction_and_forward_seconds": perf_counter() - started,
              "model_class": model_type.__name__,
              "baseline_vs_own_zero_B0_reference_MSE": float(
                  np.sum((abs(image) - abs(model.ideal_image))**2)
                  / np.sum(abs(model.ideal_image)**2))}
    if model_type is not EchoSubspinACDCModel:
        report["finite_voxel_quadrature"] = model.validate_voxel_quadrature(zeros)
        if not report["finite_voxel_quadrature"]["passed"]:
            raise RuntimeError(f"Finite-voxel quadrature check failed: {report['finite_voxel_quadrature']}")
    if arguments.smoke_iterations and model_type is not EchoSubspinACDCModel:
        from experiment_optimization import optimize_mrzero_image
        solution = optimize_mrzero_image(model, zeros, max_iterations=arguments.smoke_iterations,
            total_current_limit_A=30., field_limit_Hz=500., learning_rate_A=.01,
            patience=arguments.smoke_iterations + 1, use_cuda_graph=False)
        report["optional_optimizer_smoke"] = solution["optimization_info"]
        report["optional_optimizer_smoke_hardware"] = model.hardware.diagnostics(
            solution["currents_A"], total_current_limit_A=30., field_limit_Hz=500.)
        arrays.update(smoke_image=solution["image"], smoke_kdata=solution["kdata"],
                      smoke_currents_A=solution["currents_A"])
    device = model.device
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report, arrays


def _plot_comparison(path, arrays, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, Y = arrays["X_image_m"] * 1000, arrays["Y_image_m"] * 1000
    measured = arrays["measured_image"]
    legacy, candidate = arrays["legacy_scaled_image"], arrays["candidate_scaled_image"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), layout="constrained", sharex=True, sharey=True)
    rows = [
        [(measured, "Measured EPI", "gray", 0, measured.max()),
         (legacy, f"Legacy point twin\nNRMSE {report['legacy']['full_FOV']['NRMSE_percent']:.2f}%", "gray", 0, measured.max()),
         (candidate, f"Weighted B0 + finite voxel\nNRMSE {report['candidate']['full_FOV']['NRMSE_percent']:.2f}%", "gray", 0, measured.max())],
        [(legacy - measured, "Legacy minus measured", "RdBu_r", -measured.max(), measured.max()),
         (candidate - measured, "Candidate minus measured", "RdBu_r", -measured.max(), measured.max()),
         (candidate - legacy, "Candidate minus legacy", "RdBu_r", -measured.max(), measured.max())],
        [(arrays["measured_signal_roi"], "Fixed measured-signal ROI", "gray", 0, 1),
         (arrays["legacy_holes_25_mask"], f"Legacy below 25% of measured\n{report['legacy']['holes']['25']['pixels']} pixels", "gray", 0, 1),
         (arrays["candidate_holes_25_mask"], f"Candidate below 25% of measured\n{report['candidate']['holes']['25']['pixels']} pixels", "gray", 0, 1)],
    ]
    for axis, (values, title, cmap, low, high) in zip(axes.ravel(), sum(rows, [])):
        artist = axis.pcolormesh(X, Y, values, shading="auto", cmap=cmap, vmin=low, vmax=high)
        axis.set(title=title, xlabel="X [mm]", ylabel="Y [mm]", aspect="equal")
        fig.colorbar(artist, ax=axis, shrink=.7)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    # A lower legacy deficit count is not evidence that real EPI signal losses
    # are reproduced: show the missing opposite direction next to it.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained", sharex=True, sharey=True)
    fig.suptitle("Indicateurs descriptifs : rapport 50 %, luminosité ≥ 10 % du maximum mesuré ; enveloppe fixe de la référence")
    for row, name in enumerate(("legacy", "candidate")):
        for axis, key, title, cmap, low, high in (
            (axes[row, 0], "difference", f"{name} : simulation − mesure", "RdBu_r", -measured.max(), measured.max()),
            (axes[row, 1], "measured_low_simulated_high_50_mask", "Signal réel faible, simulé trop fort", "Reds", 0, 1),
            (axes[row, 2], "simulated_low_measured_high_50_mask", "Signal simulé faible, réel plus fort", "Blues", 0, 1)):
            artist = axis.pcolormesh(X, Y, arrays[f"{name}_disagreement_{key}"],
                shading="auto", cmap=cmap, vmin=low, vmax=high)
            axis.set(title=title, xlabel="X [mm]", ylabel="Y [mm]", aspect="equal")
            fig.colorbar(artist, ax=axis, shrink=.7)
    fig.savefig(path.with_stem(path.stem + "_signal_disagreement"), dpi=140)
    plt.close(fig)


def validate(arguments):
    from experiment_voxel import FiniteVoxelEchoACDCModel

    output = arguments.output.resolve() if arguments.output else (
        PROJECT_ROOT / "MRzero_validation/results_epi_fidelity"
        / datetime.now(timezone.utc).strftime("fidelity_%Y%m%d_%H%M%S_%f_UTC"))
    if (output / "comparison.json").exists() or (output / "comparison.npz").exists():
        raise FileExistsError(f"Choose a new output directory; comparison artifacts already exist in {output}.")
    folder = arguments.input_dir.resolve()
    inputs = read_twin_inputs(folder, scan_sequence_path=arguments.epi_sequence,
        b0_sequence_path=arguments.b0_sequence, b0_scan_path=arguments.b0_scan,
        b0_echo_count=arguments.b0_field_echo_count)
    fit = prepare_echo_object(inputs, arguments.object_threshold_fraction,
        arguments.background_corner_m, support_min_snr=arguments.object_min_snr,
        allow_uncalibrated_noise=False)
    calibration = arguments.calibration.resolve() if arguments.calibration else folder / "ACDC_3T.npz"
    settings = {"units": "Hz/A", "registration_confirmed": True,
                "patient_to_source_m": np.eye(4).tolist(),
                "source_sha256": arguments.calibration_sha256,
                "registration_note": "Confirmed Bay5 placement retained from notebooks 05/06."}
    common = {"alignment_angle_deg": arguments.alignment_angle_deg,
        "alignment_mirror_x": arguments.alignment_mirror_x,
        "alignment_mirror_y": arguments.alignment_mirror_y,
        "frame_angle_deg": arguments.frame_angle_deg,
        "subspins_z": arguments.subspins_z, "rate_mode": arguments.rate_mode,
        "b0_mode": "regularized", "b0_sigma_pixels": arguments.b0_sigma_pixels,
        "current_response": {"enabled": True, "tau_s": 133e-6 / np.log(100.)}}
    legacy = prepare_echo_shim_inputs(inputs, fit, calibration, settings,
        **common, b0_weighting="uniform", inplane_voxel=False)
    candidate = prepare_echo_shim_inputs(inputs, fit, calibration, settings,
        **common, b0_weighting="echo_power", inplane_voxel=True)
    shared = assert_shared_object(legacy, candidate)
    runs, model_arrays = {}, {}
    for name, prepared, model_type in (("legacy", legacy, EchoSubspinACDCModel),
                                       ("candidate", candidate, FiniteVoxelEchoACDCModel)):
        print(f"Running zero-current {name}: {prepared['report']['subspin_count']} spins.", flush=True)
        runs[name], model_arrays[name] = _run_model(prepared, model_type, arguments,
                                                   inputs["scan_diagnostics"]["corrected_kspace"])
    for key in ("measured_image", "X_image_m", "Y_image_m"):
        if not np.allclose(model_arrays["legacy"][key], model_arrays["candidate"][key], rtol=0, atol=1e-12):
            raise ValueError(f"The two forward models changed the common {key}.")
    metrics, arrays = compare_baselines(model_arrays["legacy"]["measured_image"],
        model_arrays["legacy"]["image"], model_arrays["candidate"]["image"],
        model_arrays["legacy"]["X_image_m"], model_arrays["legacy"]["Y_image_m"],
        reference_image=model_arrays["candidate"]["ideal_image"])
    for name in ("legacy", "candidate"):
        for key, value in model_arrays[name].items():
            if key not in ("image", "measured_image", "X_image_m", "Y_image_m"):
                arrays[f"{name}_{key}"] = value
        arrays[f"{name}_native_B0_Hz"] = (legacy if name == "legacy" else candidate)["native_B0_Hz"]
    arrays.update(parent_flat_index=legacy["parent_flat_index"], parent_valid_mask=legacy["parent_valid_mask"],
        native_X_m=legacy["native_X_m"], native_Y_m=legacy["native_Y_m"],
        native_amplitude=legacy["native_amplitude"], native_rate_sInv=legacy["native_rate_sInv"],
        object_support=fit["object_support"], noise_mask=fit["noise_mask"],
        noise_covariance=fit["noise_covariance"], echo_times_s=inputs["echo_times_s"],
        fit_valid_mask=fit["fit_valid_mask"], target_reliable_mask=fit["target_reliable_mask"],
        subspin_weights=candidate["weights"], subspin_xyz_patient_m=candidate["xyz_m"],
        subspin_basis_Hz_per_A=candidate["fieldbasis"],
        subspin_amplitude_at_reference=candidate["expanded_experiment"]["m0_proxy"],
        kx_mInv=inputs["acquisition"]["kx_mInv"], ky_mInv=inputs["acquisition"]["ky_mInv"],
        ADC_time_s=inputs["acquisition"]["time_s"])
    for name in ("edge_vectors_m", "b0_gradient_Hz_per_pixel", "field_gradient_Hz_per_A_per_pixel"):
        arrays[f"candidate_voxel_{name}"] = candidate["expanded_experiment"]["inplane_voxel"][name]
    for name in ("field_at_corners_Hz_per_A", "corner_offsets_native_pixels"):
        if name in candidate["expanded_experiment"]["inplane_voxel"]:
            arrays[f"candidate_voxel_{name}"] = candidate["expanded_experiment"]["inplane_voxel"][name]
    code_paths = [Path(__file__), *(Path(__file__).parent / name for name in (
        "experiment_voxel.py", "experiment_echo.py", "experiment_mrzero.py", "echo_shim_inputs.py",
        "experiment_subspins.py", "experiment_realistic.py", "b0_regularization.py", "twin_inputs.py",
        "echo_object.py", "continuous_reconstruction.py", "inplane_geometry.py", "measured_continuous.py",
        "native_resummation.py", "acdc_hardware.py", "signal_disagreement.py")), *(PROJECT_ROOT / "Experiment" / name for name in (
        "acdc_gpu.py", "read_acdc_b0.py", "read_acdc_sequence.py", "prepare_acdc_inputs.py"))]
    summary = {"created_UTC": datetime.now(timezone.utc).isoformat(),
        "configuration": vars(arguments), "input_paths": inputs["paths"],
        "source_sha256": dict(inputs["source_sha256"], calibration=legacy["report"]["calibration_source_sha256"]),
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
        "shared_object_checks": shared, "echo_fit": fit["report"],
        "legacy_preparation": legacy["report"], "candidate_preparation": candidate["report"],
        "model_runs": runs, "metrics": metrics,
        "thickness_sampling": {"count": arguments.subspins_z,
            "one_point_zero_current_equivalence": "With kz=0 and the declared z-uniform amplitude/B0, thickness weights sum to one. ACDC field variation has zero phase at zero current."},
        "limitations": ["Effective GRE amplitude and R2* remain measured proxies, not intrinsic PD/T1/T2.",
            "Finite boxes assume a uniform local amplitude and locally linear B0/ACDC fields.",
            "The finite-voxel zero-B0 reference includes the voxel aperture; it differs from the point-model reference.",
            "The comparison changes B0 weighting and voxel integration together; it does not isolate their individual effects.",
            "No measured EPI parameter fitting is performed except one intensity gain per twin for evaluation."],
        "artifacts": {"report": "comparison.json", "arrays": "comparison.npz", "figure": "comparison.png",
                      "signal_disagreement": "comparison_signal_disagreement.png"},
        "controller_used": False, "waveform_exported": False}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "comparison.npz", **arrays)
    _plot_comparison(output / "comparison.png", arrays, metrics)
    (output / "comparison.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(json_safe({"output": output, "metrics": metrics}), indent=2, ensure_ascii=False))
    return summary


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "acquisition_019")
    parser.add_argument("--epi-sequence", type=Path, default=Path("sequences") / EPI_SEQUENCE,
                        help="Explicit sequence matching the measured EPI scan.")
    parser.add_argument("--b0-sequence", type=Path, help="Relative to input-dir, or absolute; default sequences/b0_map.seq.")
    parser.add_argument("--b0-scan", type=Path, help="Matching B0 Twix; default unique *b0_map.dat.")
    parser.add_argument("--b0-field-echo-count", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-sha256", default=CALIBRATION_SHA256)
    parser.add_argument("--output", type=Path, help="New local artifact directory; default results_epi_fidelity/fidelity_*_UTC.")
    parser.add_argument("--subspins-z", type=int, default=8,
                        help="Thickness points, as in notebooks (8). Set 1 explicitly for a faster zero-current check.")
    parser.add_argument("--rate-mode", choices=("pooled", "guarded", "none"), default="pooled")
    parser.add_argument("--b0-sigma-pixels", type=float, default=1.)
    parser.add_argument("--object-threshold-fraction", type=float, default=.04)
    parser.add_argument("--object-min-snr", type=float, default=5.)
    parser.add_argument("--background-corner-m", type=float, default=.085)
    parser.add_argument("--alignment-angle-deg", type=float, default=180.)
    parser.add_argument("--alignment-mirror-x", action="store_true")
    parser.add_argument("--alignment-mirror-y", action="store_true")
    parser.add_argument("--frame-angle-deg", type=float, default=0.)
    parser.add_argument("--profile", choices=("cpu", "auto", "v100", "gtx1660ti", "blackwell"), default="cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "float64"), default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=.35)
    parser.add_argument("--sample-chunk", type=int)
    parser.add_argument("--reconstruction-chunk", type=int, default=256)
    parser.add_argument("--smoke-iterations", type=int, choices=(0, 1, 2), default=0)
    arguments = parser.parse_args(argv)
    if arguments.subspins_z < 1:
        parser.error("--subspins-z must be positive.")
    if arguments.require_cuda and arguments.profile == "cpu":
        parser.error("--require-cuda requires a GPU profile or --profile auto.")
    return arguments


if __name__ == "__main__":
    validate(parse_arguments())
