"""Diagnostics against known synthetic truth, separate from pipeline inputs.

These helpers evaluate maps and the forward model after reconstruction. They
never feed truth into the measured-data object fit. Map scores are descriptive;
raw-signal consistency is checked against the declared complex noise level.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from scipy import ndimage

PROJECT_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = PROJECT_DIR / "MRzero_validation"
for directory in (PROJECT_DIR / "Experiment", VALIDATION_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from epi_twin import NativeEPITwin
from continuous_reconstruction import ContinuousReconstruction
from twin_inputs import prepare_echo_object


def serialisable(value):
    """Convert NumPy values to JSON data; represent nonfinite numbers as null."""
    if isinstance(value, dict):
        return {str(key): serialisable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialisable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return serialisable(value.tolist())
    if isinstance(value, np.generic):
        return serialisable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _nrmse_percent(prediction, reference):
    denominator = np.linalg.norm(reference)
    return float(100 * np.linalg.norm(prediction - reference) / denominator) if denominator > 0 else np.nan


def evaluate_maps(inputs, phantom):
    """Fit the native echoes and compare interior voxels with known maps.

    This diagnostic requires matching native object and B0 grids, normally
    128-square. Primary amplitude errors use physical gain one. A separately
    reported positive gain diagnoses scaling without hiding it in that error.
    Return (report, fit) for direct use in the generation notebook.
    """
    raw, field = inputs["b0_raw"], inputs["field"]
    if (phantom.proton_density.shape != raw["X_m"].shape
            or not np.allclose(phantom.X_m, raw["X_m"], rtol=1e-5, atol=2e-7)
            or not np.allclose(phantom.Y_m, raw["Y_m"], rtol=1e-5, atol=2e-7)):
        raise ValueError("Map evaluation requires the phantom on the native B0 grid; use object_matrix=128.")
    fit = prepare_echo_object(inputs)
    interior = ndimage.binary_erosion(phantom.support_mask, iterations=6)
    mask = interior & field["mask"] & np.isfinite(field["b0Hz"])
    if not mask.any():
        raise ValueError("No reliable interior B0 voxel is available for comparison.")
    b0_error = field["b0Hz"][mask] - phantom.B0_Hz[mask]
    truth_amplitude = phantom.proton_density * np.exp(-phantom.R2star_sInv * inputs["echo_times_s"][0])
    estimated = fit["first_echo_amplitude"][mask]
    truth = truth_amplitude[mask]
    gain = max(0., float(np.dot(estimated, truth) / np.dot(estimated, estimated))) if np.any(estimated) else np.nan
    rate_mask = mask & fit["fit_valid_mask"] & np.isfinite(fit["R2star_sInv"])
    rate_error = fit["R2star_sInv"][rate_mask] - phantom.R2star_sInv[rate_mask]
    report = {
        "comparison_voxels": int(mask.sum()), "interior_erosion_pixels": 6,
        "B0_RMSE_Hz": float(np.sqrt(np.mean(b0_error**2))),
        "B0_mean_error_Hz": float(np.mean(b0_error)),
        "B0_max_absolute_error_Hz": float(np.max(np.abs(b0_error))),
        "first_echo_NRMSE_percent_gain_one": _nrmse_percent(estimated, truth),
        "first_echo_diagnostic_positive_gain": gain,
        "first_echo_NRMSE_percent_after_gain": _nrmse_percent(gain * estimated, truth),
        "R2star_fit_valid_voxels": int(rate_mask.sum()),
        "R2star_RMSE_sInv": float(np.sqrt(np.mean(rate_error**2))) if rate_error.size else np.nan,
        "R2star_mean_error_sInv": float(np.mean(rate_error)) if rate_error.size else np.nan,
        "truth_used_for_echo_fit": False, "automatic_pass_fail_threshold": False,
    }
    return report, fit


def compare_truth_epi(inputs, phantom, device="auto", adc_chunk=128):
    """Compare independently generated raw EPI with the pipeline forward model.

    Known maps enter only this diagnostic. The acquisition generator does not
    use NativeEPITwin. Both paths share MRzero physics, so their agreement tests
    ordering, timing, phase, quadrature and receiver handling rather than an
    independent physical model. The comparison uses raw EPI before navigator
    phase correction, including its declared noise. A separate image score
    reports the effect of the pipeline's navigator correction.
    Return (prediction, report); the consistency flag includes the known noise.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    acquisition = inputs["acquisition"]
    object_area = abs(float(np.linalg.det(np.array([
        [phantom.X_m[0, 1] - phantom.X_m[0, 0], phantom.Y_m[0, 1] - phantom.Y_m[0, 0]],
        [phantom.X_m[1, 0] - phantom.X_m[0, 0], phantom.Y_m[1, 0] - phantom.Y_m[0, 0]],
    ]))))
    epi_area = float(np.prod(acquisition["FOV_m"]) / np.asarray(acquisition["kx_mInv"]).size)
    model = NativeEPITwin(acquisition, phantom.X_m, phantom.Y_m,
        amplitude=phantom.proton_density * object_area / epi_area,
        B0_Hz=phantom.B0_Hz, support_mask=phantom.support_mask,
        rate_sInv=phantom.R2star_sInv, amplitude_reference_time_s=0,
        receive_weights=phantom.receive_weights, device=device, sample_chunk=adc_chunk)
    prediction = model.simulate(b0_enabled=True)
    diagnostics = inputs["scan_diagnostics"]
    observed = np.moveaxis(diagnostics["kspace"], 1, 0)
    corrected = np.moveaxis(diagnostics.get("corrected_kspace", diagnostics["kspace"]), 1, 0)
    if observed.shape != prediction["kdata"].shape:
        raise ValueError("Stored EPI receiver/sample counts disagree with the truth phantom.")
    reconstruction = ContinuousReconstruction(acquisition, "cpu", torch.float64)
    raw_coil_images = reconstruction.numpy_image(observed)
    raw_image = np.sqrt(np.sum(np.abs(raw_coil_images)**2, axis=0))
    residual = observed - prediction["kdata"]
    denominator = np.linalg.norm(prediction["kdata"])
    observed_norm = np.linalg.norm(observed)
    sigma = inputs.get("manifest", {}).get("reports", {}).get("scan", {}).get("noise_std_real_imag_kspace")
    expected_norm = float(sigma * np.sqrt(2 * observed.size)) if sigma is not None else np.nan
    report = {
        "known_truth_diagnostic_only": True, "object_to_EPI_area_weight": object_area / epi_area,
        "comparison_data": "raw k-space before navigator phase correction",
        "kspace_relative_error": float(np.linalg.norm(residual) / denominator) if denominator > 0 else np.nan,
        "noise_expected_relative_error": expected_norm / denominator if denominator > 0 else np.nan,
        "residual_complex_RMS": float(np.sqrt(np.mean(np.abs(residual)**2))),
        "declared_noise_std_real_imag_kspace": sigma,
        "image_NRMSE_percent_gain_one": _nrmse_percent(prediction["image"], raw_image),
        "corrected_image_NRMSE_percent": _nrmse_percent(prediction["image"], inputs["scan"]),
        "navigator_correction_relative_signal_change": (
            float(np.linalg.norm(corrected - observed) / observed_norm) if observed_norm > 0 else np.nan),
        "automatic_assertion": False, "simulation": prediction["report"],
    }
    report["signal_noise_consistent"] = (
        bool(report["kspace_relative_error"] <= 1.2 * report["noise_expected_relative_error"] + 1e-4)
        if sigma is not None else None)
    report["signal_noise_consistency_bound"] = (
        1.2 * report["noise_expected_relative_error"] + 1e-4 if sigma is not None else None)
    prediction["measured_image"] = raw_image
    prediction["corrected_measured_image"] = inputs["scan"]
    return prediction, report


__all__ = ["evaluate_maps", "compare_truth_epi", "serialisable"]
