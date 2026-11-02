"""Separate effective spiral gradient encoding from nominal reconstruction.

A physical gradient delay changes the field experienced by spins. It must not
silently move measured samples to new NUFFT locations: the measured and
simulated data are compared with the same, fixed nominal reconstruction.
Positive delays here mean gradients lag the nominal waveform.
"""

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np


DELAY_PARAMETER_NAMES = ("delay_x_s", "delay_y_s")


def _delay_parameters(parameters):
    if not isinstance(parameters, dict) or set(parameters) - set(DELAY_PARAMETER_NAMES):
        raise ValueError("Spiral encoding parameters must contain delay_x_s and/or delay_y_s only.")
    values = []
    for name in DELAY_PARAMETER_NAMES:
        value = parameters.get(name, 0.)
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not np.isrealobj(value) or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite real scalar in seconds.")
        values.append(float(value))
    return tuple(values)


def _trajectory(acquisition):
    if acquisition.get("sequence_kind") != "spiral":
        raise ValueError("Effective spiral encoding requires a spiral acquisition.")
    arrays = tuple(np.asarray(acquisition[key]) for key in ("kx_mInv", "ky_mInv", "time_s"))
    kx, ky, time = arrays
    if (kx.ndim != 2 or kx.size < 2 or any(array.shape != kx.shape
            or np.iscomplexobj(array) or not np.isfinite(array).all() for array in arrays)
            or np.any(np.diff(time.ravel()) <= 0)):
        raise ValueError("Spiral kx, ky and chronological ADC times must be matching finite real 2D arrays.")
    return kx, ky, time


def _matrix(value):
    matrix = np.asarray(value)
    if (matrix.shape != (2, 2) or np.iscomplexobj(matrix) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(2), rtol=0, atol=1e-10)):
        raise ValueError("The native-to-model frame matrix must be a finite orthogonal 2 by 2 matrix.")
    return matrix.astype(float)


def _sampled_delay(values, time, delay):
    if delay == 0:
        return values.copy()
    source_time, source = time.ravel(), values.ravel()
    query = source_time - delay
    delayed = np.interp(query, source_time, source)
    before, after = query < source_time[0], query > source_time[-1]
    delayed[before] = source[0] + (query[before] - source_time[0]) * (
        source[1] - source[0]) / (source_time[1] - source_time[0])
    delayed[after] = source[-1] + (query[after] - source_time[-1]) * (
        source[-1] - source[-2]) / (source_time[-1] - source_time[-2])
    return delayed.reshape(values.shape)


def _pulseq_delay(acquisition, delay_x_s, delay_y_s, sequence_path):
    import pypulseq as pp
    from read_spiral_sequence import read_spiral_sequence

    path = Path(sequence_path)
    nominal, timing = read_spiral_sequence(path)
    digest = timing["sequence_sha256"]
    if acquisition.get("sequence_sha256", digest) != digest:
        raise ValueError("The spiral sequence SHA256 does not match the supplied acquisition.")
    for key in ("kx_mInv", "ky_mInv", "time_s"):
        expected, supplied = np.asarray(nominal[key]), np.asarray(acquisition[key])
        tolerance = 1e-10 if key == "time_s" else 1e-8
        if supplied.shape != expected.shape or not np.allclose(supplied, expected, rtol=0, atol=tolerance):
            raise ValueError(f"The nominal spiral {key} does not match the supplied Pulseq sequence.")
    for key in ("rf_time_s", "line_index", "line_start_s", "line_end_s"):
        if key in acquisition and (np.shape(acquisition[key]) != np.shape(nominal[key])
                or not np.allclose(acquisition[key], nominal[key], rtol=0, atol=1e-12)):
            raise ValueError(f"The spiral {key} does not match the supplied Pulseq sequence.")
    if delay_x_s == delay_y_s == 0:
        return np.array(acquisition["kx_mInv"], copy=True), np.array(acquisition["ky_mInv"], copy=True)
    sequence = pp.Sequence()
    sequence.read(str(path))
    # PyPulseq's positive trajectory_delay advances the gradient waveform;
    # use the negative of our physical lag. Integration includes pre-ADC
    # gradients and RF history, avoiding sampled-k endpoint extrapolation.
    k_adc, _, excitation, refocusing, adc_times = sequence.calculate_kspace(
        trajectory_delay=[-delay_x_s, -delay_y_s, 0.])
    shape = np.shape(acquisition["kx_mInv"])
    count = int(np.prod(shape))
    actual = np.asarray(k_adc)[:, -count:]
    relative_time = np.asarray(adc_times)[-count:] - timing["rf_center_s"]
    if (actual.shape != (3, count) or not np.isfinite(actual).all()
            or len(refocusing) or len(excitation) != 1
            or not np.allclose(relative_time, np.asarray(acquisition["time_s"]).ravel(), rtol=0, atol=1e-10)
            or np.max(abs(actual[2])) > 1e-3):
        raise ValueError("Delayed Pulseq gradient integration changed the imaging ADC/RF history or dimensionality.")
    return actual[0].reshape(shape).copy(), actual[1].reshape(shape).copy()


def delayed_spiral_trajectory(acquisition, delay_x_s, delay_y_s, *, sequence_path=None):
    """Return native effective kx and ky at the unchanged imaging ADC times.

    With ``sequence_path``, delayed Pulseq gradients are integrated with their
    RF history after verifying the complete nominal imaging trajectory, timing
    and available sequence hash. Without a sequence, ``k_axis(t-delay_axis)``
    uses linear sampled-k interpolation and explicit endpoint extrapolation.
    That fallback is an effective approximation, not a measured hardware model.
    Fitting code must specify physically justified delay bounds independently.
    """
    delay_x_s, delay_y_s = _delay_parameters(dict(delay_x_s=delay_x_s, delay_y_s=delay_y_s))
    kx, ky, time = _trajectory(acquisition)
    if sequence_path is not None:
        return _pulseq_delay(acquisition, delay_x_s, delay_y_s, sequence_path)
    return _sampled_delay(kx, time, delay_x_s), _sampled_delay(ky, time, delay_y_s)


def reconstruction_acquisition(acquisition):
    """Copy an acquisition and restore the fixed nominal Fourier trajectory."""
    result = deepcopy(acquisition)
    available = [key in acquisition for key in ("nominal_kx_mInv", "nominal_ky_mInv")]
    if any(available) and not all(available):
        raise ValueError("Nominal reconstruction requires both nominal kx and ky arrays.")
    if all(available):
        for axis in ("x", "y"):
            result[f"k{axis}_mInv"] = np.array(acquisition[f"nominal_k{axis}_mInv"], copy=True)
    _trajectory(result)
    return result


def calibration_acquisition(native_acquisition, framed_forward_acquisition, frame_matrix):
    """Return native effective k for phase calibration, without moving NUFFT k.

    Use ``reconstruction_acquisition`` separately for the reconstruction
    factory. Frequency and translation phase bases must use actual forward k.
    """
    matrix = _matrix(frame_matrix)
    nominal = reconstruction_acquisition(native_acquisition)
    kx, ky, time = _trajectory(framed_forward_acquisition)
    if (time.shape != np.shape(nominal["time_s"])
            or not np.allclose(time, nominal["time_s"], rtol=0, atol=1e-12)):
        raise ValueError("Forward encoding and native acquisition must retain identical ADC times.")
    actual_native = np.stack((kx, ky), axis=-1) @ matrix
    result = deepcopy(nominal)
    result["nominal_kx_mInv"] = np.array(nominal["kx_mInv"], copy=True)
    result["nominal_ky_mInv"] = np.array(nominal["ky_mInv"], copy=True)
    result.update(kx_mInv=actual_native[..., 0], ky_mInv=actual_native[..., 1])
    return result


def apply_spiral_encoding(prepared, delay_parameters, *, sequence_path=None):
    """Copy prepared inputs, changing only forward gradient encoding.

    Delays refer to native readout/phase gradient axes even when a model uses a
    rotated or mirrored coordinate frame. ADC times, turn labels, current
    windows, RF phases, object weights and density compensation remain fixed.
    Repeated calls always start from the archived nominal trajectory.
    """
    delay_x_s, delay_y_s = _delay_parameters(delay_parameters)
    framed_nominal = reconstruction_acquisition(prepared["expanded_experiment"]["acquisition"])
    matrix = _matrix(prepared.get("combined_frame_matrix",
        framed_nominal.get("combined_frame_matrix", np.eye(2))))
    nominal_native_k = np.stack((framed_nominal["kx_mInv"], framed_nominal["ky_mInv"]), axis=-1) @ matrix
    native = reconstruction_acquisition(prepared.get("acquisition_native", framed_nominal))
    if "acquisition_native" in prepared:
        native_stored = np.stack((native["kx_mInv"], native["ky_mInv"]), axis=-1)
        if native_stored.shape != nominal_native_k.shape or not np.allclose(
                native_stored, nominal_native_k, rtol=0, atol=1e-8):
            raise ValueError("Native and framed nominal spiral trajectories disagree.")
    else:
        native.update(kx_mInv=nominal_native_k[..., 0], ky_mInv=nominal_native_k[..., 1])
    actual_x, actual_y = delayed_spiral_trajectory(native, delay_x_s, delay_y_s, sequence_path=sequence_path)
    actual_framed = (np.stack((framed_nominal["kx_mInv"], framed_nominal["ky_mInv"]), axis=-1)
        if delay_x_s == delay_y_s == 0 else np.stack((actual_x, actual_y), axis=-1) @ matrix.T)
    result = deepcopy(prepared)
    forward = result["expanded_experiment"]["acquisition"]
    for axis in ("x", "y"):
        forward[f"nominal_k{axis}_mInv"] = np.array(framed_nominal[f"k{axis}_mInv"], copy=True)
        native[f"nominal_k{axis}_mInv"] = np.array(native[f"k{axis}_mInv"], copy=True)
    forward.update(kx_mInv=actual_framed[..., 0], ky_mInv=actual_framed[..., 1])
    native.update(kx_mInv=actual_x, ky_mInv=actual_y)
    result["acquisition_native"] = native
    report = {"format": "effective_spiral_gradient_delay_v1",
        "parameters": dict(zip(DELAY_PARAMETER_NAMES, (delay_x_s, delay_y_s))),
        "positive_delay_convention": "Gradient waveform lags nominal scanner time.",
        "parameter_axes": "Native readout X and phase Y before model frame rotation/reflection.",
        "forward_encoding": ("Pulseq integration of delayed gradients including pre-ADC and RF history."
            if sequence_path is not None else "Effective sampled-k interpolation: k_axis(t-delay_axis)."),
        "endpoint_policy": ("Full Pulseq waveform integration; no sampled-k endpoint extrapolation."
            if sequence_path is not None else "Linear extrapolation using the first or last two ADC samples."),
        "sequence_path": None if sequence_path is None else str(Path(sequence_path).resolve()),
        "sequence_sha256": None if sequence_path is None else hashlib.sha256(Path(sequence_path).read_bytes()).hexdigest(),
        "reconstruction_trajectory": "Fixed nominal trajectory for measured and simulated images.",
        "adc_times_changed": False, "turn_state_indices_changed": False,
        "current_integration_windows_changed": False,
        "hardware_delay_independently_measured": False,
        "limitations": "A bounded effective gradient-delay model is not a measurement of gradient impulse response, eddy currents or the full trajectory."}
    forward["spiral_encoding_report"] = deepcopy(report)
    native["spiral_encoding_report"] = deepcopy(report)
    result.setdefault("report", {})["spiral_encoding"] = deepcopy(report)
    result["expanded_experiment"].setdefault("provenance", {})["spiral_encoding"] = deepcopy(report)
    return result


__all__ = ["delayed_spiral_trajectory", "apply_spiral_encoding",
           "reconstruction_acquisition", "calibration_acquisition", "DELAY_PARAMETER_NAMES"]
