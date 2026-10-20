"""Apply continuous-coordinate reconstruction to already ordered measured data.

Raw-data readers retain responsibility for Twix line ordering, REFLECT handling,
navigator correction and sequence validation. These helpers replace only the
final Fourier reconstruction. Measured receiver phases are never reapplied.
Use them before rotating reference images into the simulation display frame.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from continuous_reconstruction import ContinuousReconstruction


def _preserve_native(container, names):
    for name in names:
        if name in container and "native_" + name not in container:
            container["native_" + name] = np.array(container[name], copy=True)


def reconstruct_receivers(diagnostics, acquisition):
    """Update scan diagnostics and return ``(corrected_RSS, report)``.

    The input ``kspace`` and optional ``corrected_kspace`` are already in
    ``[phase, receiver, readout]`` order from ``read_scan``. Coil images keep
    that order. Original reconstructed arrays are retained with ``native_``
    prefixes. This helper requires raw k-space, rather than an image archive.
    """
    if "kspace" not in diagnostics:
        raise ValueError("Continuous receiver reconstruction requires measured raw kspace.")
    raw = np.asarray(diagnostics["kspace"])
    corrected = np.asarray(diagnostics.get("corrected_kspace", raw))
    shape = np.asarray(acquisition["kx_mInv"]).shape
    if (raw.ndim != 3 or raw.shape != corrected.shape
            or (raw.shape[0], raw.shape[2]) != shape or raw.shape[1] < 1
            or not np.isfinite(raw).all() or not np.isfinite(corrected).all()):
        raise ValueError("Receiver kspace must be finite [phase, receiver, readout] arrays matching acquisition.")
    operator = ContinuousReconstruction(acquisition, "cpu", torch.float64)
    uncorrected_coils = np.moveaxis(operator.numpy_image(np.moveaxis(raw, 1, 0)), 0, 1)
    coils = np.moveaxis(operator.numpy_image(np.moveaxis(corrected, 1, 0)), 0, 1)
    _preserve_native(diagnostics, ("coil_images", "uncorrected_coil_images", "uncorrected_image"))
    diagnostics["coil_images"] = coils
    diagnostics["uncorrected_coil_images"] = uncorrected_coils
    diagnostics["uncorrected_image"] = np.sqrt(np.sum(np.abs(uncorrected_coils)**2, axis=1))
    image = np.sqrt(np.sum(np.abs(coils)**2, axis=1))
    report = dict(operator.report)
    report.update({"receiver_count": int(raw.shape[1]),
                   "input_axes": ["phase", "receiver", "readout"],
                   "combination": "RSS after complex reconstruction per receiver",
                   "receiver_phase_reapplied": False})
    diagnostics["continuous_reconstruction"] = report
    return image, report


def reconstruct_b0_echoes(b0_raw, sequence_path):
    """Replace B0 ``coil_images`` using the validated sequence's actual k.

    ``read_b0_coil_images`` supplies k-space in ``[echo, receiver, phase,
    readout]`` order and its validated report in ``metadata.sequence_info``.
    Pulseq is evaluated once; trajectory samples are reflected and phase rows
    sorted by the same rules as ``read_b0_sequence``. Dummy RF pulses have no
    ADC samples. Return one reconstruction report per echo, retaining original
    coil images as ``native_coil_images`` for comparison.
    """
    import pypulseq as pp

    metadata = b0_raw.get("metadata", {})
    info = metadata.get("sequence_info")
    if not isinstance(info, dict):
        raise ValueError("B0 reconstruction requires validated metadata.sequence_info.")
    data = np.asarray(b0_raw["kspace"])
    if data.ndim != 4 or min(data.shape) < 1 or not np.isfinite(data).all():
        raise ValueError("B0 kspace must be finite [echo, receiver, phase, readout] data.")
    ne, receivers, ny, nx = data.shape
    if list(info["image_shape"]) != [ny, nx] or int(info["echo_count"]) != ne:
        raise ValueError("Validated B0 sequence dimensions disagree with measured kspace.")
    sequence_path = Path(sequence_path)
    digest = hashlib.sha256(sequence_path.read_bytes()).hexdigest()
    if "sequence_sha256" in info and info["sequence_sha256"] != digest:
        raise ValueError("B0 sequence has changed since its raw-data validation.")
    sequence = pp.Sequence()
    sequence.read(str(sequence_path))
    k = np.asarray(sequence.calculate_kspace()[0], dtype=np.float64)
    if k.shape != (3, ny * ne * nx) or not np.isfinite(k).all():
        raise ValueError("B0 calculated ADC trajectory disagrees with measured echo/sample counts.")
    trajectory = k.reshape(3, ny, ne, nx)
    if np.max(np.abs(trajectory[2])) > 1e-3:
        raise ValueError("B0 continuous reconstruction supports the validated 2D slice only.")
    polarity = np.sign(np.mean(np.diff(trajectory[0], axis=-1), axis=-1)).astype(int)
    expected_polarity = int(info["polarity"])
    if expected_polarity not in (-1, 1) or np.any(polarity != expected_polarity):
        raise ValueError("Calculated B0 readout polarity disagrees with the validated sequence.")
    if expected_polarity < 0:
        trajectory = trajectory[..., ::-1]
    order = np.argsort(trajectory[1, :, 0, 0])
    if ("acquired_phase_rows" in info
            and not np.array_equal(np.argsort(order), info["acquired_phase_rows"])):
        raise ValueError("Calculated B0 phase order disagrees with the validated sequence.")
    trajectory = trajectory[:, order]
    coils = np.empty(data.shape, dtype=np.complex128)
    reports = []
    for echo in range(ne):
        acquisition = {"kx_mInv": trajectory[0, :, echo],
                       "ky_mInv": trajectory[1, :, echo], "FOV_m": info["FOV_m"]}
        operator = ContinuousReconstruction(acquisition, "cpu", torch.float64)
        coils[echo] = operator.numpy_image(data[echo])
        report = dict(operator.report)
        report.update({"echo_index": echo, "receiver_count": receivers,
                       "sequence_sha256": digest, "receiver_phase_reapplied": False})
        reports.append(report)
    _preserve_native(b0_raw, ("coil_images",))
    b0_raw["coil_images"] = coils
    metadata["continuous_reconstruction"] = reports
    return reports


__all__ = ["reconstruct_receivers", "reconstruct_b0_echoes"]
