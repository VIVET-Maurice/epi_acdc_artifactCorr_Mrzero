"""A fixed measured GRE target for the native MRzero image objective.

Object estimation and the image optimization target are separate operations.
This module changes only the comparison image, fixed coverage and receiver
scale. It never changes spin amplitudes, relaxation, B0 or ACDC responses.
The zero-B0 simulated image remains available as ``model.ideal_image``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "Experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from resample_acdc_maps import resample_to_grid


def _binary_mask(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must be a binary mask with shape {shape}.")
    return array.astype(bool)


def _magnitude(value, shape, name):
    array = np.asarray(value)
    if (array.shape != shape or not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite image with shape {shape}.")
    if not np.iscomplexobj(array) and np.any(array < 0):
        raise ValueError(f"{name} must be a nonnegative magnitude or a complex image.")
    return np.abs(array).astype(float)


def magnitude_objective_terms(model, image, *, gradient=True):
    """Return magnitude MSE and its complex-image derivative.

    Existing models without measured-target weights retain the original
    full-image objective. The derivative obeys ``dJ = Re(sum(conj(d)*dI))``;
    its zero-magnitude generalized derivative is zero. No scale or mask is
    estimated while evaluating a candidate waveform.
    """
    amplitude = image.abs()
    residual = amplitude - model._reference_amplitude
    weights = getattr(model, "_image_target_weights", None)
    weighted = residual if weights is None else residual * weights
    value = (residual * weighted).sum() / model._reference_energy
    if not gradient:
        return value, None
    derivative = (2 * weighted * image
                  / amplitude.clamp_min(torch.finfo(model.dtype).tiny)
                  / model._reference_energy)
    return value, derivative


@torch.no_grad()
def relative_image_mse(model, image):
    """Evaluate exactly the model's current fixed image target and mask."""
    values = torch.as_tensor(image, device=model.device, dtype=model.complex_dtype)
    if values.shape != model._reference_amplitude.shape:
        raise ValueError("Image shape differs from the model's fixed comparison grid.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Image must be finite on the comparison grid.")
    return float(magnitude_objective_terms(model, values, gradient=False)[0])


@torch.no_grad()
def configure_measured_gre_target(model, reference, *, baseline_image=None,
                                  measured_baseline_image=None, frame_matrix=None,
                                  threshold_fraction=0.0, roi_mask=None, gain=None):
    """Install a measured GRE magnitude target on the EPI comparison grid.

    ``reference`` supplies its native ``magnitude``, ``X_m`` and ``Y_m``;
    its matrix may differ from the EPI matrix. Linear interpolation uses
    physical coordinates, without extrapolation or image registration.
    The inverse joint frame transform maps the model output coordinates
    back to that native frame. It is inferred from ``FrameReconstruction``
    when omitted, or may be supplied explicitly.

    The ROI is the fixed GRE coverage, optionally intersected with a binary
    EPI-grid ``roi_mask`` and a GRE intensity threshold. Zero threshold keeps
    covered background pixels as well as tissue. Missing GRE coverage is
    excluded, never replaced with invented measurements.

    With ``measured_baseline_image``, two positive least-squares gains are
    estimated once: simulated unshimmed EPI -> measured unshimmed EPI, then
    measured unshimmed EPI -> measured GRE. This matches the closed-loop
    normalization. Without it, a single baseline -> GRE gain is used.
    ``gain`` supplies an already frozen simulator -> GRE scale instead.
    The target in simulator units is GRE/gain, so the objective is exactly
    sum(mask * (gain*abs(simulation)-GRE)**2) / sum(mask*GRE**2).

    Return the installed target and a serializable report. The native GRE,
    baseline images, object arrays and ``model.ideal_image`` are preserved.
    """
    native = np.asarray(reference.get("magnitude"))
    if (native.ndim != 2 or min(native.shape) < 2 or np.iscomplexobj(native)
            or not np.issubdtype(native.dtype, np.number)
            or np.any(native[np.isfinite(native)] < 0)):
        raise ValueError("Reference GRE must be a real nonnegative 2D magnitude.")
    native = native.astype(float)
    if any(np.iscomplexobj(reference[key]) for key in ("X_m", "Y_m")):
        raise ValueError("Reference X_m/Y_m must be real physical coordinates.")
    sourceX, sourceY = (np.asarray(reference[key], dtype=float) for key in ("X_m", "Y_m"))
    source_valid = _binary_mask(reference.get("valid_mask", np.ones(native.shape, bool)),
                                native.shape, "Reference valid_mask") & np.isfinite(native)
    operator = model.reconstruction
    targetX, targetY = (np.asarray(getattr(operator, key), dtype=float)
                        for key in ("X_m", "Y_m"))
    if (targetX.ndim != 2 or min(targetX.shape) < 2 or targetY.shape != targetX.shape
            or not np.isfinite(targetX).all() or not np.isfinite(targetY).all()):
        raise ValueError("Model reconstruction must expose finite 2D X_m/Y_m coordinates.")
    matrix_value = getattr(operator, "frame_matrix", np.eye(2)) if frame_matrix is None else frame_matrix
    if np.iscomplexobj(matrix_value):
        raise ValueError("frame_matrix must be a finite real orthogonal 2x2 matrix.")
    matrix = np.asarray(matrix_value, dtype=float)
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(2), rtol=0, atol=1e-10)):
        raise ValueError("frame_matrix must be a finite real orthogonal 2x2 matrix.")
    native_targetX = matrix[0, 0] * targetX + matrix[1, 0] * targetY
    native_targetY = matrix[0, 1] * targetX + matrix[1, 1] * targetY
    # A joint orthogonal transform followed by its inverse can move an exact
    # first/last source center a few ulps outside the interpolation interval.
    # Snap only floating-point roundoff, never a physical coverage gap.
    for query, source in ((native_targetX, sourceX), (native_targetY, sourceY)):
        tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(source))))
        for bound in (float(np.min(source)), float(np.max(source))):
            query[np.abs(query-bound) <= tolerance] = bound
    sampled, covered = resample_to_grid(native, sourceX, sourceY, source_valid,
                                       native_targetX, native_targetY)
    if not covered.any():
        raise ValueError("The measured GRE does not cover any EPI comparison pixels.")
    if (isinstance(threshold_fraction, (bool, np.bool_))
            or not np.isscalar(threshold_fraction) or not np.isreal(threshold_fraction)
            or not np.isfinite(threshold_fraction) or not 0 <= threshold_fraction < 1):
        raise ValueError("threshold_fraction must be finite and lie in [0, 1).")
    peak = float(np.max(sampled[covered]))
    requested = (np.ones(targetX.shape, bool) if roi_mask is None else
                 _binary_mask(roi_mask, targetX.shape, "roi_mask"))
    if threshold_fraction > 0:
        requested &= covered & (sampled > float(threshold_fraction) * peak)
    mask = requested & covered
    if not mask.any() or np.sum(sampled[mask]**2) <= 0:
        raise ValueError("Measured GRE needs nonzero signal on a nonempty fixed ROI.")
    if baseline_image is None:
        baseline_image = model.forward(np.zeros((model.nc, model.nstates)))[0]
    baseline = _magnitude(baseline_image, targetX.shape, "Simulated unshimmed baseline")
    baseline_roi, gre_roi = baseline[mask], sampled[mask]
    simulation_to_epi_gain = experimental_gain = None
    if gain is None:
        baseline_energy = float(np.sum(baseline_roi**2))
        if baseline_energy <= 0:
            raise ValueError("Simulated baseline needs nonzero signal on the fixed GRE ROI.")
        if measured_baseline_image is not None:
            measured = _magnitude(measured_baseline_image, targetX.shape, "Measured unshimmed EPI")
            measured_roi = measured[mask]
            measured_energy = float(np.sum(measured_roi**2))
            if measured_energy <= 0:
                raise ValueError("Measured EPI needs nonzero signal on the fixed GRE ROI.")
            simulation_to_epi_gain = float(np.sum(baseline_roi * measured_roi) / baseline_energy)
            experimental_gain = float(np.sum(measured_roi * gre_roi) / measured_energy)
            gain = simulation_to_epi_gain * experimental_gain
            gain_source = "simulated unshimmed EPI -> measured unshimmed EPI -> measured GRE; two fixed least-squares gains"
        else:
            gain = float(np.sum(baseline_roi * gre_roi) / baseline_energy)
            gain_source = "simulated unshimmed EPI -> measured GRE; fixed least-squares gain"
    else:
        gain_source = "explicit fixed simulator-to-GRE gain"
    if (isinstance(gain, (bool, np.bool_)) or not np.isscalar(gain) or not np.isreal(gain)
            or not np.isfinite(gain) or gain <= 0):
        raise ValueError("The fixed simulator-to-GRE gain must be positive and finite.")
    gain = float(gain)
    measured_reference = np.where(covered, sampled, 0.0)
    reference_image = measured_reference / gain
    reference_tensor = torch.as_tensor(reference_image, device=model.device, dtype=model.complex_dtype)
    weights = torch.as_tensor(mask, device=model.device, dtype=model.dtype)
    energy = (reference_tensor.abs().square() * weights).sum()
    if not bool(torch.isfinite(energy)) or float(energy) <= 0:
        raise ValueError("The scaled measured GRE target has invalid energy.")
    report = dict(
        target_kind="measured GRE magnitude",
        objective="sum(mask * (fixed_gain * abs(MRzero_EPI_ACDC) - measured_GRE)**2) / sum(mask * measured_GRE**2)",
        native_reference_shape=list(native.shape), comparison_shape=list(targetX.shape),
        comparison_grid="native EPI output pixels, with joint display-frame coordinates",
        native_reference_spacing_mm=[float(np.hypot(sourceX[1, 0]-sourceX[0, 0], sourceY[1, 0]-sourceY[0, 0])*1000),
                                     float(np.hypot(sourceX[0, 1]-sourceX[0, 0], sourceY[0, 1]-sourceY[0, 0])*1000)],
        native_reference_resolution_preserved=True, added_measured_spatial_detail=False,
        interpolation="linear in native physical XY coordinates; no extrapolation",
        registration_fitted=False, frame_matrix=matrix.tolist(),
        threshold_fraction=float(threshold_fraction), ROI_source="full measured GRE coverage" if roi_mask is None and threshold_fraction == 0 else "fixed GRE threshold and/or explicit EPI-grid ROI",
        covered_pixels=int(covered.sum()), comparison_pixels=int(mask.sum()),
        excluded_missing_GRE_coverage_pixels=int(np.sum(requested & ~covered)),
        fixed_simulator_to_GRE_gain=gain, gain_fit_source=gain_source,
        simulation_to_baseline_EPI_gain=simulation_to_epi_gain,
        experimental_baseline_to_GRE_gain=experimental_gain,
        mask_changes_with_currents=False, gain_changes_with_currents=False,
        zero_B0_image_role="simulation diagnostic only; never the measured-GRE optimization target",
        object_parameters_modified=False,
    )
    model._reference = reference_tensor
    model._reference_amplitude = reference_tensor.abs()
    model._reference_energy = energy
    model._image_target_weights = weights
    model.reference = reference_image.copy()
    target = dict(reference_image=reference_image.copy(), measured_reference_magnitude=measured_reference,
                  mask=mask.copy(), coverage_mask=covered.copy(), gain=gain,
                  X_m=targetX.copy(), Y_m=targetY.copy(),
                  native_X_m=native_targetX, native_Y_m=native_targetY, report=report)
    model.image_target = target
    model.runtime_info["image_target"] = report
    report["baseline_relative_mse"] = relative_image_mse(model, baseline_image)
    report["baseline_NRMSE_percent"] = 100 * np.sqrt(report["baseline_relative_mse"])
    return target


__all__ = ["configure_measured_gre_target", "magnitude_objective_terms", "relative_image_mse"]
