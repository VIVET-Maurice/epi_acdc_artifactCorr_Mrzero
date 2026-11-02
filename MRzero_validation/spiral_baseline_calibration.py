"""Bounded zero-current spiral calibration with guarded spatial validation.

The only candidates are a constant frequency offset and two rigid object
translations. Spatial B0 variations, relaxation, amplitude and receive maps
remain fixed. Complex data from every receive channel are reconstructed before
forming RSS; receiver magnitudes must never be added as complex samples.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import least_squares

from baseline_calibration import (
    _jacobian_information, _metrics, _parameter_vector, spatial_split,
)
from baseline_metadata_audit import _frequency_audit
from continuous_reconstruction import ContinuousReconstruction


PARAMETER_NAMES = ("frequency_offset_Hz", "readout_shift_mm", "phase_shift_mm")
DEFAULT_BOUNDS = dict(zip(PARAMETER_NAMES, (30., 2., 2.)))


def phase_basis(acquisition):
    """Exact sample phases per Hz, native X mm and native Y mm."""
    if acquisition.get("sequence_kind") != "spiral":
        raise ValueError("Spiral calibration requires a spiral acquisition.")
    kx, ky, time = (np.asarray(acquisition[key]) for key in
                    ("kx_mInv", "ky_mInv", "time_s"))
    rf = np.asarray(acquisition.get("rf_time_s", np.zeros_like(time)))
    if (kx.ndim != 2 or kx.size < 2
            or any(x.shape != kx.shape or np.iscomplexobj(x) or not np.isfinite(x).all()
                   for x in (kx, ky, time, rf))
            or np.any(time <= rf) or np.ptp(rf) > 1e-12):
        raise ValueError("Spiral calibration requires finite matching k/time arrays and one RF history.")
    return -2 * np.pi * np.stack((time - rf, kx * 1e-3, ky * 1e-3))


def phase_factor(acquisition, parameters):
    """Multiply simulated samples before reconstruction; never warp an image."""
    vector = _parameter_vector(parameters, PARAMETER_NAMES)
    return np.exp(1j * np.einsum("p,pij->ij", vector, phase_basis(acquisition)))


def fit_baseline_calibration(acquisition, baseline_kdata, measured_image,
                             reconstruction=None, bounds=None, block_size=8,
                             guard_pixels=1, candidate_validator=None):
    """Fit nested frequency, frequency+X and frequency+X+Y candidates.

    ``baseline_kdata`` may have shape ``[sample_row, sample_column]`` for a
    single receiver, or ``[receiver, sample_row, sample_column]``. The measured
    image is always two-dimensional RSS. All receivers share the same three
    physical phase parameters. Both the RSS image and its analytic derivative
    include every supplied receiver; no receiver compression is performed.

    Applying an accepted translation requires rebuilding the object and
    querying the physical coil maps at the translated positions. The returned
    parameters are supported by ``prepare_echo_shim_inputs``. Translation is
    retained in the zero-B0 target; the fitted frequency offset is removed.
    There is no Cartesian odd/even correction in this calibration.

    An optional ``candidate_validator(parameters)`` checks a fitted candidate's
    physical rebuild without inspecting held-out measurements. It runs once
    for each nested model, not at optimizer iterations, and returns a dictionary
    with a Boolean ``accepted`` entry and descriptive checks. A rejected richer
    model does not discard an eligible simpler model. Validator exceptions are
    propagated; callers should convert expected coverage failures into explicit
    rejected check results.
    """
    basis = phase_basis(acquisition)
    supplied = np.asarray(baseline_kdata)
    measured = np.asarray(measured_image)
    single_receiver = supplied.ndim == 2
    kdata = supplied[None] if single_receiver else supplied
    output_shape = tuple(acquisition.get("image_shape", basis.shape[1:]))
    if (kdata.ndim != 3 or kdata.shape[0] < 1
            or kdata.shape[1:] != basis.shape[1:]
            or not np.iscomplexobj(kdata) or not np.isfinite(kdata).all()
            or measured.shape != output_shape or np.iscomplexobj(measured)
            or not np.isfinite(measured).all() or np.any(measured < 0)
            or not np.any(measured > 0)):
        raise ValueError("Spiral calibration requires finite complex samples [receiver, sample_row, sample_column] "
                         "or [sample_row, sample_column], and a matching nonnegative measured RSS image.")
    limits = DEFAULT_BOUNDS.copy()
    if bounds is not None:
        if not isinstance(bounds, dict) or set(bounds) - set(PARAMETER_NAMES):
            raise ValueError("bounds must contain supported spiral parameter names only.")
        limits.update(bounds)
    bound = _parameter_vector(limits, PARAMETER_NAMES)
    if np.any(bound <= 0):
        raise ValueError("All calibration bounds must be positive.")
    if candidate_validator is not None and not callable(candidate_validator):
        raise ValueError("candidate_validator must be callable or None.")
    fit_mask, validation_mask, test_mask = spatial_split(measured.shape, block_size, guard_pixels)
    masks = {"fit": fit_mask, "validation": validation_mask, "test": test_mask,
             "full_FOV": np.ones(measured.shape, bool)}
    _metrics(measured, np.zeros_like(measured), masks)
    operator = reconstruction or ContinuousReconstruction(acquisition, "cpu", torch.float64)
    if operator.dtype not in (torch.float64, torch.complex128):
        raise ValueError("Calibration reconstruction must use float64/complex128 precision.")
    measured_fit = measured[fit_mask].astype(float)
    norm = np.linalg.norm(measured_fit)
    cache = {}

    def evaluate(parameters):
        key = tuple(parameters)
        if key == cache.get("key"):
            return cache["value"]
        phase = np.exp(1j * np.einsum("p,pij->ij", parameters, basis))
        samples = kdata * phase[None]
        # Leading axes are [image-or-derivative, receiver]. Reconstruction is
        # linear and independent across those axes, while RSS couples receivers.
        batch = np.concatenate((samples[None], 1j * basis[:, None] * samples[None]))
        images = operator.numpy_image(batch.reshape((-1,) + basis.shape[1:]))
        images = images.reshape((4, kdata.shape[0]) + measured.shape)
        receiver_image, differential = images[0], images[1:]
        magnitude = np.sqrt(np.sum(abs(receiver_image) ** 2, axis=0))
        derivative = np.sum((receiver_image.conj()[None] * differential).real, axis=1)
        derivative /= np.maximum(magnitude[None], np.finfo(float).tiny)
        train, dtrain = magnitude[fit_mask], derivative[:, fit_mask]
        energy = float(train @ train)
        if energy <= np.finfo(float).tiny:
            raise ValueError("Simulated training RSS signal is zero.")
        gain = float(train @ measured_fit / energy)
        dgain = (dtrain @ measured_fit - 2 * gain * (dtrain @ train)) / energy
        residual = (gain * train - measured_fit) / norm
        jacobian = (gain * dtrain + dgain[:, None] * train).T / norm
        image = receiver_image[0] if single_receiver else receiver_image
        value = (image, magnitude * gain, gain, residual, jacobian)
        cache.update(key=key, value=value)
        return value

    zero = np.zeros(len(PARAMETER_NAMES))
    baseline_image, baseline_scaled, baseline_gain, _, _ = evaluate(zero)
    baseline_metrics = _metrics(measured, baseline_scaled, masks)
    runs, selected = [], None
    best_validation = baseline_metrics["validation"]["NRMSE_percent"]
    selection_margin = .05  # Percentage points, fixed before fitting.
    previous = zero.copy()
    for size in (1, 2, 3):
        def padded(values):
            return np.r_[values, np.zeros(len(PARAMETER_NAMES) - size)]

        def fun(values):
            return evaluate(padded(values))[3]

        def jac(values):
            return evaluate(padded(values))[4][:, :size]

        starts = [np.zeros(size), previous[:size].copy()]
        for fraction in (-.5, .5):
            start = previous[:size].copy()
            start[0] = fraction * bound[0]
            starts.append(start)
        unique_starts = list({tuple(start): start for start in starts}.values())
        fits = [least_squares(fun, start, jac=jac,
                    bounds=(-bound[:size], bound[:size]), x_scale=bound[:size],
                    max_nfev=120, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                for start in unique_starts]
        fitted = min(fits, key=lambda fit: fit.cost)
        candidate = padded(fitted.x)
        previous = candidate.copy()
        _, scaled, gain, _, jacobian = evaluate(candidate)
        # Test pixels do not affect fitting, starts, gain or model selection.
        scores = _metrics(measured, scaled,
            {"fit": fit_mask, "validation": validation_mask}, include_bright_roi=False)
        information = _jacobian_information(jacobian[:, :size], bound[:size])
        at_bound = bool(np.any(abs(fitted.x) >= .98 * bound[:size]))
        physical_check = {"accepted": True, "reason": "No additional physical rebuild validator supplied."}
        if candidate_validator is not None:
            physical_check = candidate_validator(dict(zip(PARAMETER_NAMES, candidate.tolist())))
            if (not isinstance(physical_check, dict)
                    or not isinstance(physical_check.get("accepted"), (bool, np.bool_))):
                raise ValueError("candidate_validator must return a dictionary with a Boolean accepted entry.")
            physical_check = dict(physical_check, accepted=bool(physical_check["accepted"]))
        eligible = bool(fitted.success and not at_bound and information["Jacobian_identifiable"]
                        and physical_check["accepted"])
        improves = scores["validation"]["NRMSE_percent"] < best_validation - selection_margin
        runs.append({"parameter_count": size,
            "parameters": dict(zip(PARAMETER_NAMES, candidate.tolist())),
            "gain_from_fit_pixels": gain, "metrics": scores,
            "solver_success": bool(fitted.success), "solver_message": str(fitted.message),
            "total_function_evaluations": int(sum(fit.nfev for fit in fits)),
            "near_parameter_bound": at_bound, **information,
            "physical_rebuild_validation": physical_check,
            "eligible": eligible, "selected_when_evaluated": bool(eligible and improves)})
        if eligible and improves:
            selected, best_validation = candidate.copy(), scores["validation"]["NRMSE_percent"]
    if selected is None:
        selected = zero.copy()
    candidate_image, candidate_scaled, candidate_gain, _, _ = evaluate(selected)
    candidate_metrics = _metrics(measured, candidate_scaled, masks)
    has_candidate = bool(np.any(selected != 0))
    test_improves = candidate_metrics["test"]["NRMSE_percent"] < baseline_metrics["test"]["NRMSE_percent"]
    accepted = bool(has_candidate and test_improves)
    applied = selected if accepted else zero
    parameters = dict(zip(PARAMETER_NAMES, applied.tolist()))
    report = {
            "format": "bounded_spiral_baseline_calibration_v2",
            "accepted": accepted,
            "reason": ("Validation-selected candidate improves the untouched internal test subset." if accepted else
                "No eligible validation improvement." if not has_candidate else
                "Validation-selected candidate did not improve the untouched internal test subset; original model retained."),
            "parameters": parameters,
            "candidate_parameters": dict(zip(PARAMETER_NAMES, selected.tolist())),
            "bounds_symmetric_half_width": limits, "candidate_models": runs,
            "baseline": baseline_metrics, "candidate": candidate_metrics,
            "applied": candidate_metrics if accepted else baseline_metrics,
            "baseline_gain_from_fit_pixels": baseline_gain,
            "candidate_gain_from_fit_pixels": candidate_gain,
            "gain_applied_to_object": False,
            "signal_loss_evaluation": "RSS MSE penalizes both signal excess and deficit. "
                "Bidirectional diagnostics expose measured low-signal regions missed by the simulation; "
                "they do not establish the cause of signal loss or modify the zero-B0 target.",
            "receiver_model": {"receiver_count": int(kdata.shape[0]),
                "combination": "Root sum of squared receiver image magnitudes after complex reconstruction",
                "derivative": "Exact derivative of receiver RSS and training-only fitted scalar gain",
                "receiver_compression": False},
            "selection": {"criterion": "validation NRMSE, nested 1/2/3 parameters; starts selected by fit cost only",
                "minimum_improvement_percentage_points": selection_margin,
                "test_used_for_fit_or_model_selection": False,
                "test_used_for_final_application_gate": True, "refit_on_full_image": False},
            "split": {"kind": "internal guarded spatial blocks of one measured RSS image",
                "block_size": int(block_size), "guard_pixels_per_block_edge": int(guard_pixels),
                "label_rule": "(3*(row//block)+7*(column//block))%5",
                "labels": {"fit": [0, 1, 2], "validation": [3], "test": [4]},
                "pixels": {name: int(mask.sum()) for name, mask in masks.items()},
                "independent_acquisition_validation": False},
            "fixed_quantities": ["in-plane rotation", "spatial B0 variations/gradients",
                "object support", "GRE amplitude", "relaxation", "receive model",
                "temporal ACDC current response", "nominal spiral trajectory"],
            "interpretation": "Effective residual parameters inferred from magnitude; not independently measured hardware constants.",
            "readout_shift_interpretation": "Assumed rigid GRE object/B0 registration shift; ACDC maps must be resampled. "
                "Zero-current RSS cannot distinguish it from an instrumental phase ramp or independently establish coil registration.",
            "translation_interpretation": "Rigid native-plane object/B0 translation; physical ACDC maps are resampled after acceptance.",
            "target_policy": "Rebuild native MRzero: remove fitted global B0 offset in the zero-B0 target, retain both rigid object translations.",
            "limitations": [
                "Magnitude agreement cannot establish the cause of a phase ramp or independently validate imaging-to-coil registration.",
                "Nominal Pulseq gradients do not measure gradient delays or eddy currents.",
                "Guarded spatial subsets share one acquisition and do not replace an independent scan.",
                "The local scaled Jacobian must identify all fitted parameters; a bound hit or rank failure prevents application."],
        }
    return {"parameters": parameters, "candidate_parameters": report["candidate_parameters"], "report": report,
        "baseline_image": baseline_image, "candidate_image": candidate_image,
        "calibrated_image": candidate_image if accepted else baseline_image,
        "baseline_scaled_image": baseline_scaled, "candidate_scaled_image": candidate_scaled,
        "calibrated_scaled_image": candidate_scaled if accepted else baseline_scaled,
        "fit_mask": fit_mask, "validation_mask": validation_mask, "test_mask": test_mask}


def audit_baseline_inputs(inputs):
    """Read-only scanner frequency and spiral phase-basis evidence."""
    frequency = _frequency_audit(inputs)
    scans = frequency.get("acquisitions", {})
    if "EPI" in scans:
        scans["Spiral"] = scans.pop("EPI")
    if "known_B0_to_EPI_offset_Hz" in frequency:
        frequency["known_B0_to_spiral_offset_Hz"] = frequency.pop("known_B0_to_EPI_offset_Hz")
    basis = phase_basis(inputs["acquisition"]).reshape(3, -1).T
    # Remove common phase: a spatially constant complex phase is invisible
    # to magnitude. This nominal-basis rank is evidence, not a fitted-image test.
    centered = (basis - basis.mean(axis=0)) * np.array([30., 2., 2.])
    singular = np.linalg.svd(centered, compute_uv=False)
    diagnostics = inputs.get("scan_diagnostics", {})
    return {"schema_version": 1, "calibration_parameters_fitted": False,
        "frequency_reference": frequency,
        "navigators": {"available": np.asarray(diagnostics.get("navigator_kspace", [])).size > 0,
            "applied_to_spiral_imaging": False,
            "policy": "Cartesian navigator records are retained as diagnostics; no EPI odd/even correction is applied."},
        "phase_basis": {"parameters": list(PARAMETER_NAMES),
            "nominal_centered_scaled_singular_values": singular.tolist(),
            "rank": int(np.linalg.matrix_rank(centered)),
            "policy": "Jointly test frequency and two translations only when the image Jacobian remains identifiable."},
        "reconstruction": inputs["acquisition"].get("density_compensation_report", {}),
        "validation_policy": ["Freeze the object, bounds, density weights and spatial subsets before fitting.",
            "Fit the intensity gain on training pixels only; use validation for model selection and untouched test pixels for the application gate.",
            "Rebuild native MRzero after acceptance, verify exact sample-phase parity and preserve object amplitude."]}


__all__ = ["fit_baseline_calibration", "phase_factor", "phase_basis",
           "audit_baseline_inputs", "PARAMETER_NAMES", "DEFAULT_BOUNDS"]
