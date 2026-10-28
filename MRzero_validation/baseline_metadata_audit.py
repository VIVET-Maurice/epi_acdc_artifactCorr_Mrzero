"""Read-only evidence for a bounded EPI baseline calibration.

This audit never estimates a calibration from image agreement. It reports
scanner frequency references, the navigator correction already applied, and
the near-degeneracy of a global frequency offset with a PE displacement.
Real Twix files are reopened with protocol-only parsing, never raw-data parsing.
Missing metadata and portable synthetic acquisitions remain usable.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np


def _number(value, *, positive=False):
    try:
        array = np.asarray(value)
        if array.size != 1 or np.iscomplexobj(array):
            return None
        result = float(array.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) and (not positive or result > 0) else None


def _frequency_from_header(header):
    observed = {}
    for section in ("MeasYaps", "Phoenix"):
        try:
            nucleus = header[section]["sTXSPEC"]["asNucleusInfo"][0]
            value = _number(nucleus["lFrequency"], positive=True)
        except (KeyError, IndexError, TypeError):
            value = None
        if value is not None:
            observed[section + ".sTXSPEC.asNucleusInfo[0].lFrequency"] = value
    try:
        value = _number(header["Dicom"]["lFrequency"], positive=True)
    except (KeyError, TypeError):
        value = None
    if value is not None:
        observed["Dicom.lFrequency"] = value
    return observed


def _header_frequency(path):
    if path is None:
        return {"available": False, "reason": "No acquisition path supplied."}
    path = Path(path)
    report = {"filename": path.name, "available": False}
    if path.suffix.lower() != ".dat":
        report["reason"] = "Frequency metadata unavailable for a portable non-Twix archive."
        return report
    try:
        import twixtools
        with contextlib.redirect_stdout(io.StringIO()):
            scans = twixtools.read_twix(str(path), include_scans=[-1],
                parse_prot=True, parse_data=False, parse_pmu=False,
                parse_geometry=False, verbose=False)
        observed = _frequency_from_header(scans[-1]["hdr"])
    except Exception as error:
        # An informational audit must not block simulation when headers cannot
        # be reopened, a parser is unavailable, or a portable archive is used.
        report["reason"] = "Protocol-only frequency read unavailable: " + str(error)[:250]
        return report
    if not observed:
        report["reason"] = "No finite positive scanner frequency reference in the parsed protocol."
        return report
    values = list(observed.values())
    report.update(available=True, frequency_Hz=values[0], observed_fields_Hz=observed,
        header_fields_agree=bool(np.allclose(values, values[0], rtol=0, atol=.5)),
        read_method="Twix protocol only; parse_data=False, final measurement only")
    return report


def _frequency_audit(inputs):
    manifest = inputs.get("manifest") or {}
    if str(manifest.get("format", "")).startswith("mrzero_synthetic"):
        return {"available": False, "reason": "Synthetic acquisition; no measured scanner frequency reference."}
    paths = inputs.get("paths") or {}
    scans = {label: _header_frequency(paths.get(key))
             for label, key in (("EPI", "scan"), ("B0", "b0"), ("GRE", "reference"))}
    values = [record["frequency_Hz"] for record in scans.values() if record["available"]]
    report = {"available": bool(values), "acquisitions": scans,
        "all_three_available": len(values) == 3,
        "max_reference_difference_Hz": None if len(values) < 2 else max(values) - min(values),
        "same_declared_reference": None if len(values) < 2 else bool(max(values) - min(values) <= .5),
        "known_B0_to_EPI_offset_Hz": None,
        "caveat": "Equal declared references do not measure temporal B0 drift or prove identical shim fields."}
    if scans["EPI"]["available"] and scans["B0"]["available"]:
        report["known_B0_to_EPI_offset_Hz"] = scans["B0"]["frequency_Hz"] - scans["EPI"]["frequency_Hz"]
    return report


def _navigator_audit(inputs):
    diagnostics = inputs.get("scan_diagnostics") or {}
    fit = diagnostics.get("navigator_fit")
    if not isinstance(fit, dict):
        return {"available": False, "reason": "No measured navigator fit in scan_diagnostics.",
                "recommended_residual_odd_even_phase_bounds_rad": None}
    residual_rms = np.asarray(fit.get("fit_residual_rad", []), dtype=float)
    finite = residual_rms[np.isfinite(residual_rms)]
    if not finite.size:
        return {"available": False, "reason": "No finite fitted navigator phase residual.",
                "recommended_residual_odd_even_phase_bounds_rad": None}
    report = {"available": True, "fitted_receiver_count": int(finite.size),
        "per_receiver_residual_RMS_rad": [None if not np.isfinite(value) else float(value)
                                          for value in residual_rms.reshape(-1)],
        "residual_RMS_range_rad": [float(finite.min()), float(finite.max())],
        "median_receiver_residual_RMS_rad": float(np.median(finite)),
        "aggregate_weighted_residual_RMS_rad": None,
        "residual_percentile_5_95_rad": None,
        "already_applied_intercept_range_rad": None,
        "already_applied_slope_range_rad_per_normalized_x": None,
        "phase_branch": int(fit.get("phase_branch", 0)),
        "recommended_residual_odd_even_phase_bounds_rad": [-.15, .15],
        "bound_interpretation": "Conservative candidate bound for an additional shared residual phase, not a confidence interval or a measured correction.",
        "caveat": "Navigator fits precede imaging; small navigator residuals do not prove absence of later gradient or phase drift."}
    correction = np.asarray(diagnostics.get("navigator_phase_correction_rad", []), dtype=float)
    if correction.ndim == 2 and correction.shape[1] >= 2 and np.isfinite(correction).all():
        intercept = correction.mean(axis=1)
        slope = (correction[:, -1] - correction[:, 0]) / 2
        report["already_applied_intercept_range_rad"] = [float(intercept.min()), float(intercept.max())]
        report["already_applied_slope_range_rad_per_normalized_x"] = [float(slope.min()), float(slope.max())]
    phase = np.asarray(fit.get("phase_estimate_rad", []), dtype=float)
    mask = np.asarray(fit.get("fit_mask", []), dtype=bool)
    if correction.shape == phase.shape == mask.shape and phase.ndim == 2:
        selected = mask & np.isfinite(phase) & np.isfinite(correction)
        if selected.any():
            residual = phase - correction
            report["residual_percentile_5_95_rad"] = np.percentile(residual[selected], [5, 95]).tolist()
            navigator = np.asarray(diagnostics.get("navigator_hybrid", []))
            if navigator.shape == (3,) + phase.shape and np.isfinite(navigator).all():
                weight = np.abs(navigator[1]) * np.sqrt(np.abs(navigator[0]) * np.abs(navigator[2]))
                total = float(weight[selected].sum())
                if total > 0:
                    report["aggregate_weighted_residual_RMS_rad"] = float(np.sqrt(
                        np.sum(weight[selected] * residual[selected]**2) / total))
    return report


def _phase_encoding_audit(inputs):
    acquisition = inputs.get("acquisition") or {}
    try:
        time = np.asarray(acquisition["time_s"], dtype=float)
        ky = np.asarray(acquisition["ky_mInv"], dtype=float)
        rf = np.asarray(acquisition.get("rf_time_s", np.zeros_like(time)), dtype=float)
        if (time.ndim != 2 or min(time.shape) < 2 or ky.shape != time.shape
                or rf.shape != time.shape or not np.isfinite(time).all()
                or not np.isfinite(ky).all() or not np.isfinite(rf).all()
                or np.ptp(rf) > 1e-12):
            raise ValueError("Need finite EPI timings and ky from one RF history.")
        elapsed = (time - rf).reshape(-1)
        phase_k = ky.reshape(-1)
        if np.ptp(phase_k) <= 0:
            raise ValueError("No phase-encoding span.")
        # This is a timing identity from nominal ADC events, not a parameter
        # fitted to EPI appearance: t = intercept + coefficient * ky + residual.
        design = np.column_stack((np.ones(elapsed.size), phase_k))
        coefficient = np.linalg.lstsq(design, elapsed, rcond=None)[0]
        residual = elapsed - design @ coefficient
        centered_energy = float(np.sum((elapsed - elapsed.mean())**2))
        if centered_energy <= 0:
            raise ValueError("No time evolution in the ADC history.")
    except (KeyError, ValueError, TypeError, np.linalg.LinAlgError) as error:
        return {"available": False, "reason": str(error)}
    pixel_m = None
    fov = np.asarray(acquisition.get("FOV_m", []), dtype=float).reshape(-1)
    if fov.size >= 2 and np.isfinite(fov[1]) and fov[1] > 0:
        pixel_m = float(fov[1] / time.shape[0])
    rms = float(np.sqrt(np.mean(residual**2)))
    report = {"available": True,
        "PE_displacement_m_per_Hz": float(coefficient[1]),
        "PE_displacement_mm_per_Hz": float(coefficient[1] * 1000),
        "PE_displacement_pixels_per_Hz": None if pixel_m is None else float(coefficient[1] / pixel_m),
        "timing_ky_R_squared": float(1 - np.sum(residual**2) / centered_energy),
        "nontranslation_timing_residual_RMS_s": rms,
        "nontranslation_phase_RMS_rad_at_10Hz": float(2 * np.pi * 10 * rms),
        "kspace_power_weighted_nontranslation_phase_RMS_rad_at_10Hz": None,
        "identifiability_policy": "Do not jointly fit global frequency offset and PE translation from one magnitude EPI; keep PE translation fixed or constrain it independently.",
        "global_phase_policy": "A common complex phase does not change magnitude and is not a magnitude-calibration parameter.",
        "frame_policy": "A joint transform of spins, k and reconstruction cannot identify physical coil registration at zero current."}
    samples = np.asarray((inputs.get("scan_diagnostics") or {}).get("corrected_kspace", []))
    if samples.ndim == 3 and (samples.shape[0], samples.shape[2]) == time.shape and np.isfinite(samples).all():
        power = np.sum(np.abs(samples)**2, axis=1).reshape(-1)
        if power.sum() > 0:
            weighted_rms = float(np.sqrt(np.sum(power * residual**2) / power.sum()))
            report["kspace_power_weighted_nontranslation_phase_RMS_rad_at_10Hz"] = 2 * np.pi * 10 * weighted_rms
    return report


def audit_baseline_inputs(inputs):
    """Return JSON-serializable metadata evidence without fitting parameters."""
    return {"schema_version": 1, "calibration_parameters_fitted": False,
        "frequency_reference": _frequency_audit(inputs),
        "navigators": _navigator_audit(inputs),
        "phase_encoding_degeneracy": _phase_encoding_audit(inputs),
        "validation_policy": [
            "Keep bounds, B0 weighting and masks fixed before calibration; never fit a free field or deformation.",
            "With one scan, hold out spatial blocks or adjacent odd/even line pairs; holding out one parity alone cannot validate odd/even phase.",
            "Fit gains on training data and freeze them for validation; reject bounds-saturating or unstable parameter estimates.",
            "Internal held-out checks do not replace a second acquisition; freeze calibration before optimizing shim currents.",
        ]}


__all__ = ["audit_baseline_inputs"]
