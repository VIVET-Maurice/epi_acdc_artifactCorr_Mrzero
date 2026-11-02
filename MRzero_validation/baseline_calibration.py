"""Small, bounded calibration of a zero-current EPI twin against measured RSS.

The object, its spatial B0 variations, relaxation and receive amplitude are
fixed. Only a global off-resonance, a readout displacement and a CONSTANT
negative-minus-positive readout phase may vary. A phase-encode displacement
is deliberately fixed: it is nearly degenerate with frequency in this EPI.

All three effects act exactly on the simulated complex samples. No magnitude
image is warped or smoothed. Fits use spatial blocks with guard bands; model
selection uses separate validation blocks, and a final test uses a third set.
This is internal spatial validation of ONE acquisition, not an independent
scanner validation. A scalar intensity gain is fitted on training pixels only
and is used for comparison, never to modify the object in the ACDC optimizer.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import least_squares

from continuous_reconstruction import ContinuousReconstruction
from signal_disagreement import signal_disagreement


PARAMETER_NAMES = ("frequency_offset_Hz", "readout_shift_mm", "odd_even_phase_rad")
DEFAULT_BOUNDS = dict(zip(PARAMETER_NAMES, (30., 2., .15)))


def _parameter_vector(parameters, names=PARAMETER_NAMES):
    if not isinstance(parameters, dict) or set(parameters) - set(names):
        raise ValueError("Calibration parameters must be a dictionary of supported parameter names.")
    values = []
    for name in names:
        value = parameters.get(name, 0.)
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not np.isrealobj(value) or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite real scalar.")
        values.append(float(value))
    return np.array(values)


def phase_basis(acquisition):
    """Radians per Hz, per mm, and per radian of negative-minus-positive phase.

    Arrays are in ordered readout coordinates, as returned by read_sequence.
    Native MRzero uses exp(-2*pi*i*(k*r+B0*t))*exp(+i*ADCphase).
    """
    kx = np.asarray(acquisition["kx_mInv"])
    time = np.asarray(acquisition["time_s"])
    rf = np.asarray(acquisition.get("rf_time_s", np.zeros_like(time)))
    if (kx.ndim != 2 or min(kx.shape) < 2 or time.shape != kx.shape or rf.shape != kx.shape
            or any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in (kx, time, rf))
            or np.any(time <= rf)):
        raise ValueError("Calibration requires finite matching 2D kx/time/RF arrays and positive evolution.")
    if np.any(np.diff(kx, axis=1) <= 0):
        raise ValueError("Calibration expects readout samples ordered by increasing kx.")
    polarity = np.sign(time[:, -1] - time[:, 0])
    if set(np.unique(polarity)) != {-1., 1.}:
        raise ValueError("Calibration requires both positive and negative EPI readouts.")
    evolution = time - rf
    return np.stack((-2 * np.pi * evolution, -2 * np.pi * kx * 1e-3,
                     np.broadcast_to(-.5 * polarity[:, None], kx.shape)))


def phase_factor(acquisition, parameters):
    """Exact complex sample multiplier for a fixed object at zero ACDC current."""
    return np.exp(1j * np.einsum("p,pij->ij", _parameter_vector(parameters), phase_basis(acquisition)))


def spatial_split(shape, block_size=8, guard_pixels=1):
    """Deterministic 60/20/20 block labels with unused borders around EACH block."""
    for value, name, minimum in ((block_size, "block_size", 3), (guard_pixels, "guard_pixels", 0)):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    block_size, guard_pixels = int(block_size), int(guard_pixels)
    if 2 * guard_pixels >= block_size:
        raise ValueError("Guard bands leave no pixels inside a block.")
    row, column = np.indices(shape)
    label = (3 * (row // block_size) + 7 * (column // block_size)) % 5
    core = ((row % block_size >= guard_pixels) & (row % block_size < block_size - guard_pixels)
            & (column % block_size >= guard_pixels) & (column % block_size < block_size - guard_pixels))
    masks = (core & (label < 3), core & (label == 3), core & (label == 4))
    if any(np.count_nonzero(mask) < 8 for mask in masks):
        raise ValueError("Image is too small for three nonempty guarded calibration subsets.")
    return masks


def _metrics(measured, scaled, masks, *, include_bright_roi=True):
    result = {}
    for name, mask in masks.items():
        energy = float(np.sum(measured[mask] ** 2))
        if energy <= 0:
            raise ValueError(f"Measured signal is absent from the {name} subset.")
        result[name] = {"NRMSE_percent": float(100 * np.sqrt(np.sum((scaled[mask] - measured[mask])**2) / energy)),
                        "pixels": int(mask.sum())}
    if include_bright_roi:
        roi = measured > .2 * measured.max()
        result["bright_ROI"] = {"pixels": int(roi.sum()),
            "below_25_percent_pixels": int(np.sum(roi & (scaled < .25 * measured))),
            "below_50_percent_pixels": int(np.sum(roi & (scaled < .50 * measured)))}
        # Legacy counts only describe simulation deficits at bright measured
        # pixels. Include the opposite direction without discarding real zeros.
        result["signal_disagreement"] = signal_disagreement(
            measured, scaled, gain=1., subsets=masks, bright_fraction=.1)["report"]
        result["legacy_bright_ROI_count_scope"] = (
            "Simulation deficits only; excludes dark measured pixels. "
            "Not a measure of reproduction of real EPI signal losses.")
    return result


def _jacobian_information(jacobian, bounds):
    """Require actual sensitivity as well as a tolerable condition number."""
    scaled = np.asarray(jacobian) * np.asarray(bounds)[None]
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = max(1e-10, max(scaled.shape) * np.finfo(float).eps * singular[0])
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular[-1] > tolerance else None
    return {"scaled_Jacobian_rank": rank,
            "scaled_Jacobian_min_singular_value": float(singular[-1]),
            "scaled_Jacobian_condition": condition,
            "Jacobian_identifiable": bool(rank == scaled.shape[1] and condition is not None and condition < 1e6)}


def fit_baseline_calibration(acquisition, baseline_kdata, measured_image, reconstruction=None,
                             bounds=None, block_size=8, guard_pixels=1):
    """Fit nested 1/2/3-parameter models; return the accepted physical parameters.

    ``bounds`` contains positive symmetric half widths (defaults 30 Hz,
    2 mm and 0.15 rad). No frequency/PE-translation or arbitrary B0 fitting.
    Adding parameters must improve validation NRMSE by at least 0.05 percentage
    point. A bound hit, a deficient local Jacobian, failure to converge, or no
    improvement on the untouched test set prevents automatic application.
    Test pixels never enter fitting, gain estimation, starts or model selection.
    Rejection retains the original model and archives the candidate diagnostic.
    """
    return _fit_calibration(acquisition, baseline_kdata, measured_image, reconstruction,
                            bounds, block_size, guard_pixels,
                            basis=phase_basis(acquisition), parameter_names=PARAMETER_NAMES,
                            default_bounds=DEFAULT_BOUNDS)


def _fit_calibration(acquisition, baseline_kdata, measured_image, reconstruction,
                     bounds, block_size, guard_pixels, *, basis, parameter_names,
                     default_bounds, report_metadata=None):
    """Shared guarded fit engine; trajectory modules supply exact sample phases."""
    kdata = np.asarray(baseline_kdata)
    measured = np.asarray(measured_image)
    if (kdata.shape != basis.shape[1:] or not np.iscomplexobj(kdata) or not np.isfinite(kdata).all()
            or measured.shape != tuple(acquisition.get("image_shape", kdata.shape)) or np.iscomplexobj(measured)
            or not np.isfinite(measured).all() or np.any(measured < 0) or not np.any(measured > 0)):
        raise ValueError("Calibration requires complex finite simulated k-space and matching nonnegative measured RSS.")
    limits = default_bounds.copy()
    if bounds is not None:
        if not isinstance(bounds, dict) or set(bounds) - set(parameter_names):
            raise ValueError("bounds must contain supported parameter names only.")
        limits.update(bounds)
    bound = _parameter_vector(limits, parameter_names)
    if np.any(bound <= 0):
        raise ValueError("All calibration bounds must be positive.")
    fit_mask, validation_mask, test_mask = spatial_split(measured.shape, block_size, guard_pixels)
    masks = {"fit": fit_mask, "validation": validation_mask, "test": test_mask,
             "full_FOV": np.ones(measured.shape, bool)}
    # Validate signal before attempting gain/optimization.
    _metrics(measured, np.zeros_like(measured), masks)
    operator = reconstruction or ContinuousReconstruction(acquisition, "cpu", torch.float64)
    # Double precision is needed for stable parameter fitting; no native MRzero
    # recomputation or large point magnetization cache is used in this search.
    if operator.dtype not in (torch.float64, torch.complex128):
        raise ValueError("Calibration reconstruction must use float64/complex128 precision.")
    measured_fit = measured[fit_mask].astype(float)
    norm = np.linalg.norm(measured_fit)
    cache = {}

    def evaluate(parameters):
        key = tuple(parameters)
        if key == cache.get("key"):
            return cache["value"]
        samples = kdata * np.exp(1j * np.einsum("p,pij->ij", parameters, basis))
        batch = np.concatenate((samples[None], 1j * basis * samples[None]))
        images = operator.numpy_image(batch)
        image, differential = images[0], images[1:]
        magnitude = abs(image)
        derivative = (image.conj()[None] * differential).real / np.maximum(magnitude[None], np.finfo(float).tiny)
        train, dtrain = magnitude[fit_mask], derivative[:, fit_mask]
        energy = float(train @ train)
        if energy <= np.finfo(float).tiny:
            raise ValueError("Simulated training signal is zero.")
        gain = float(train @ measured_fit / energy)
        dgain = (dtrain @ measured_fit - 2 * gain * (dtrain @ train)) / energy
        residual = (gain * train - measured_fit) / norm
        jacobian = (gain * dtrain + dgain[:, None] * train).T / norm
        value = (image, magnitude * gain, gain, residual, jacobian)
        cache.update(key=key, value=value)
        return value

    zero = np.zeros(3)
    baseline_image, baseline_scaled, baseline_gain, _, _ = evaluate(zero)
    baseline_metrics = _metrics(measured, baseline_scaled, masks)
    runs = []
    selected = None
    best_validation = baseline_metrics["validation"]["NRMSE_percent"]
    selection_margin = .05  # percentage point, fixed before looking at held-out test
    previous = zero.copy()
    for size in (1, 2, 3):
        def padded(values):
            return np.r_[values, np.zeros(3 - size)]

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
        fits = [least_squares(fun, start, jac=jac, bounds=(-bound[:size], bound[:size]),
                    x_scale=bound[:size], max_nfev=120, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                for start in unique_starts]
        fitted = min(fits, key=lambda fit: fit.cost)
        candidate = padded(fitted.x)
        previous = candidate.copy()
        _, scaled, gain, _, jacobian = evaluate(candidate)
        # Never inspect test pixels during nested model selection.
        scores = _metrics(measured, scaled, {"fit": fit_mask, "validation": validation_mask}, include_bright_roi=False)
        information = _jacobian_information(jacobian[:, :size], bound[:size])
        at_bound = bool(np.any(abs(fitted.x) >= .98 * bound[:size]))
        eligible = bool(fitted.success and not at_bound and information["Jacobian_identifiable"])
        improves = scores["validation"]["NRMSE_percent"] < best_validation - selection_margin
        row = {"parameter_count": size, "parameters": dict(zip(parameter_names, candidate.tolist())),
               "gain_from_fit_pixels": gain, "metrics": scores, "solver_success": bool(fitted.success),
               "solver_message": str(fitted.message), "total_function_evaluations": int(sum(f.nfev for f in fits)),
               "near_parameter_bound": at_bound, **information,
               "eligible": eligible, "selected_when_evaluated": bool(eligible and improves)}
        runs.append(row)
        if eligible and improves:
            selected = candidate.copy()
            best_validation = scores["validation"]["NRMSE_percent"]
    if selected is None:
        selected = zero.copy()
    candidate_image, candidate_scaled, candidate_gain, _, _ = evaluate(selected)
    candidate_metrics = _metrics(measured, candidate_scaled, masks)
    has_candidate = bool(np.any(selected != 0))
    test_improves = candidate_metrics["test"]["NRMSE_percent"] < baseline_metrics["test"]["NRMSE_percent"]
    accepted = bool(has_candidate and test_improves)
    applied = selected if accepted else zero
    parameters = dict(zip(parameter_names, applied.tolist()))
    report = {
        "format": "bounded_epi_baseline_calibration_v1", "accepted": accepted,
        "reason": ("Validation-selected candidate improves the untouched internal test subset." if accepted else
                   "No eligible validation improvement." if not has_candidate else
                   "Validation-selected candidate did not improve the untouched internal test subset; original model retained."),
        "parameters": parameters, "candidate_parameters": dict(zip(parameter_names, selected.tolist())),
        "bounds_symmetric_half_width": limits, "candidate_models": runs,
        "baseline": baseline_metrics, "candidate": candidate_metrics,
        "applied": candidate_metrics if accepted else baseline_metrics,
        "baseline_gain_from_fit_pixels": baseline_gain, "candidate_gain_from_fit_pixels": candidate_gain,
        "gain_applied_to_object": False,
        "signal_loss_evaluation": "MSE already penalizes both signal excess and deficit. "
            "Additional bidirectional diagnostics expose real low-signal regions missed by the simulation. "
            "They do not infer the cause of signal loss or change the simulated zero-B0 target.",
        "selection": {"criterion": "validation NRMSE, nested 1/2/3 parameters; starts selected by fit cost only",
            "minimum_improvement_percentage_points": selection_margin,
            "test_used_for_fit_or_model_selection": False,
            "test_used_for_final_application_gate": True, "refit_on_full_image": False},
        "split": {"kind": "internal guarded spatial blocks of one measured RSS image", "block_size": int(block_size),
            "guard_pixels_per_block_edge": int(guard_pixels), "label_rule": "(3*(row//block)+7*(column//block))%5",
            "labels": {"fit": [0, 1, 2], "validation": [3], "test": [4]},
            "pixels": {name: int(mask.sum()) for name, mask in masks.items()},
            "independent_acquisition_validation": False},
        "fixed_quantities": ["phase-encode displacement", "in-plane rotation", "spatial B0 variations/gradients",
                             "object support", "GRE amplitude", "relaxation", "receive model", "temporal ACDC current response"],
        "interpretation": "Effective residual parameters inferred from magnitude; not independently measured hardware constants.",
        "readout_shift_interpretation": "Assumed rigid GRE object/B0 registration shift; ACDC spatial maps must be resampled. "
            "Zero-current RSS cannot distinguish this from an instrumental phase-ramp/origin error or establish imaging-to-ACDC registration.",
        "target_policy": "Rebuild native MRzero: zero-B0 target removes fitted frequency offset; translated object and residual ADC phase remain.",
        "limitations": ["Pixels share a reconstructed acquisition and spatial correlations; guard bands do not create independent measurements.",
            "This acquisition previously informed model development; internal test is a diagnostic, not external validation.",
            "Frequency and phase-encode displacement are almost degenerate; phase-encode displacement is fixed.",
            "Odd/even parameter is only a constant residual AFTER existing navigator correction, not a second navigator correction."],
    }
    if report_metadata is not None:
        report.update(report_metadata)
    return {"parameters": parameters, "candidate_parameters": report["candidate_parameters"], "report": report,
            "baseline_image": baseline_image, "candidate_image": candidate_image,
            "calibrated_image": candidate_image if accepted else baseline_image,
            "baseline_scaled_image": baseline_scaled, "candidate_scaled_image": candidate_scaled,
            "calibrated_scaled_image": candidate_scaled if accepted else baseline_scaled,
            "fit_mask": fit_mask, "validation_mask": validation_mask, "test_mask": test_mask}


__all__ = ["fit_baseline_calibration", "phase_factor", "phase_basis", "spatial_split", "PARAMETER_NAMES", "DEFAULT_BOUNDS"]
