"""Transfer independently measured complex receiver encoding to spiral spins.

The multi-echo fit stores a complex amplitude per receiver at its reference
echo, after noise unwhitening. Those amplitudes contain effective receive and
initial transverse phase, not isolated coil sensitivities. Restoring their
excitation-time phase lets native MRzero evolve the field exactly once.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def _receiver_order(source, target, count):
    """Map measured receiver IDs without assuming contiguous channel numbers."""
    source_ids, target_ids = source.get("receiver_channel_ids"), target.get("receiver_channel_ids")
    if source_ids is None or target_ids is None:
        return np.arange(count), {
            "channel_ID_order_verified": False,
            "channel_order_policy": "Recorded array order; channel IDs are unavailable in this archive.",
        }
    source_ids, target_ids = list(source_ids), list(target_ids)
    if (len(source_ids) != count or len(target_ids) != count
            or len(set(source_ids)) != count or len(set(target_ids)) != count
            or set(source_ids) != set(target_ids)):
        raise ValueError("B0 and spiral receiver channel IDs must identify the same distinct receivers.")
    order = np.array([source_ids.index(channel) for channel in target_ids])
    return order, {"channel_ID_order_verified": True,
        "source_receiver_channel_ids": source_ids,
        "target_receiver_channel_ids": target_ids,
        "receiver_permutation": order.tolist(),
        "channel_order_policy": "B0 complex receiver maps reordered by measured Twix channel ID into spiral data order.",
    }


def prepare_spiral_receive_model(prepared, fit, inputs, *, mode="auto"):
    """Return a prepared object with fixed, unit-RSS complex receive weights.

    ``mode='auto'`` uses the fitted complex echo amplitudes when available;
    otherwise it keeps the historical single RSS proxy. ``'complex_echo'``
    requires those measured complex data. ``'rss_proxy'`` explicitly requests
    the old model. Object support, amplitudes, B0, relaxation, physical coil
    maps and quadrature weights remain unchanged.

    The fit convention is ``A_c * exp(-2*pi*i*B0*(TE-anchor))``. MRzero evolves
    from RF, so its weight is ``A_c / RSS(A) * exp(+2*pi*i*B0*anchor)``. Omitting
    this restoration would introduce a spatially varying phase error.
    """
    if mode not in {"auto", "complex_echo", "rss_proxy"}:
        raise ValueError("Spiral receive mode must be auto, complex_echo or rss_proxy.")
    if inputs["acquisition"].get("sequence_kind") != "spiral":
        raise ValueError("Spiral receive preparation requires a spiral acquisition.")
    complex_fit = fit.get("multi_echo")
    available = isinstance(complex_fit, dict) and "complex_receiver_amplitude" in complex_fit
    enabled = mode != "rss_proxy" and available
    if mode == "complex_echo" and not available:
        raise ValueError("complex_echo receive mode requires the complex 32-echo receiver fit.")
    result = deepcopy(prepared)
    experiment = result["expanded_experiment"]
    report = {"requested_mode": mode, "mode": "complex_echo" if enabled else "rss_proxy",
        "estimated_from_spiral_scan": False, "extra_fitted_spiral_parameters": 0,
        "object_amplitude_modified": False, "support_modified": False,
        "receive_channel_is_ACDC_channel": False}
    if not enabled:
        experiment.pop("receive_weights", None)
        report.update(receiver_count=1, combination="Historical single receive-weighted RSS object",
                      reason="Explicit historical mode." if mode == "rss_proxy" else "Complex receiver fit unavailable.")
    else:
        amplitudes = np.asarray(complex_fit["complex_receiver_amplitude"])
        field = np.asarray(complex_fit.get("refined_b0_Hz"))
        parent_shape = np.asarray(result["parent_valid_mask"]).shape
        if (amplitudes.ndim != 3 or amplitudes.shape[1:] != parent_shape
                or not np.iscomplexobj(amplitudes) or field.shape != parent_shape
                or np.iscomplexobj(field)):
            raise ValueError("Complex receiver amplitudes and refined B0 must match the native parent grid.")
        count = amplitudes.shape[0]
        if count != int(inputs["scan_metadata"]["receiver_count"]):
            raise ValueError("Spiral and complex echo fit receiver counts differ.")
        order, order_report = _receiver_order(inputs["b0_raw"]["metadata"], inputs["scan_metadata"], count)
        indices = np.asarray(result["parent_flat_index"], dtype=int)
        selected = amplitudes[order].reshape(count, -1)[:, indices]
        local_field = field.ravel()[indices]
        norm = np.sqrt(np.sum(np.abs(selected)**2, axis=0))
        if (not np.isfinite(selected).all() or not np.isfinite(local_field).all()
                or not np.isfinite(norm).all() or np.any(norm <= 0)):
            raise ValueError("Every retained parent needs finite, nonzero complex receiver encoding and B0.")
        anchor = float(fit["amplitude_reference_time_s"])
        fitted_anchor = float(complex_fit.get("amplitude_reference_time_s", anchor))
        if (not np.isfinite(anchor) or anchor < 0
                or not np.isclose(anchor, fitted_anchor, rtol=0, atol=1e-12)):
            raise ValueError("Complex receiver fit and MRzero amplitude reference times disagree.")
        phase = np.exp(2j * np.pi * local_field * anchor)
        parent_weights = selected / norm[None] * phase[None]
        children = np.asarray(experiment["m0_proxy"]).shape[1]
        experiment["receive_weights"] = np.repeat(parent_weights[..., None], children, axis=2)
        report.update(order_report, receiver_count=count,
            combination="Reconstruct complex receivers independently, then root sum of squared magnitudes",
            source="Independent complex multi-echo fit, transformed back from noise-whitened receiver coordinates",
            interpretation="Effective complex receive-weighted initial magnetization; not isolated coil sensitivity maps",
            amplitude_reference_time_s=anchor,
            phase_convention="A_c / RSS(A) * exp(+2*pi*i*refined_B0*reference_TE); native MRzero evolves exp(-2*pi*i*B0*time_since_RF)",
            within_voxel_assumption="Fixed complex receiver weights within each in-plane pixel and along its thickness children",
            calibration_policy="Weights stay attached to the same native material parents during bounded rigid baseline calibration",
            between_acquisition_assumption="Complex spatial receive and initial phase encoding transfers from MEGRE to the spiral scan",
            RSS_normalization_max_error=float(np.max(np.abs(np.sum(abs(parent_weights)**2, axis=0)-1))))
    experiment["receive_model"] = deepcopy(report)
    result["report"]["receive_model"] = report
    return result


def fixed_spiral_object_checks(candidate, reference):
    """Check a physical nuisance rebuild before selecting its fitted model.

    Geometry may move and physical ACDC maps must be resampled, but a phase-only
    nuisance fit cannot change object membership, amplitude, decay, quadrature
    or complex initial magnetization. This check does not inspect scan pixels.
    """
    checks = {key: bool(np.array_equal(candidate[key], reference[key], equal_nan=True))
              for key in ("parent_flat_index", "parent_valid_mask", "weights",
                          "native_amplitude", "native_rate_sInv")}
    source, rebuilt = reference["expanded_experiment"], candidate["expanded_experiment"]
    checks["subspin_amplitude"] = bool(np.array_equal(source["m0_proxy"], rebuilt["m0_proxy"], equal_nan=True))
    checks["receive_weights_presence"] = ("receive_weights" in source) == ("receive_weights" in rebuilt)
    if "receive_weights" in source:
        checks["receive_weights"] = bool("receive_weights" in rebuilt and np.array_equal(
            source["receive_weights"], rebuilt["receive_weights"], equal_nan=True))
    return checks


__all__ = ["prepare_spiral_receive_model", "fixed_spiral_object_checks"]
