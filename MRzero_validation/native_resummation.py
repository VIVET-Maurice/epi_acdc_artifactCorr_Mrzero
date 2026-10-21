"""Condition-aware checks of native MRzero point and ADC signal sums."""

import numpy as np
import torch


RESUMMATION_RTOL = 5e-5
RESUMMATION_ROUNDOFF = 8 * torch.finfo(torch.float32).eps


def resum_native_points(point_mag, native):
    """Validate each ADC against a bounded float32 roundoff allowance.

    MRzero forms its one-coil signal as ``(Mxy @ PD) * ADC_rotation``
    but returns points as ``ADC_rotation * Mxy * PD``. Even an accurate
    resummation of those rounded points need not equal the native float32
    dot product. When spins cancel, the native signal is a poor scale for
    that discrepancy. Use the uncancelled point mass for an absolute
    allowance capped at eight float32 epsilons, independent of point count.
    This is a numerical acceptance budget, not a worst-case BLAS error bound.
    The original relative tolerance still applies separately at each ADC.
    """
    if not bool(torch.isfinite(point_mag).all() & torch.isfinite(native).all()):
        raise RuntimeError("Native MRzero point resummation failed (non-finite magnetization or signal).")
    resummed = point_mag.sum(dim=1, dtype=torch.complex128)
    native = native.to(torch.complex128)
    residual = torch.abs(resummed - native)
    mass = torch.abs(point_mag).sum(dim=1, dtype=torch.float64)
    tolerance = RESUMMATION_RTOL * torch.abs(native) + RESUMMATION_ROUNDOFF * mass
    tiny = torch.finfo(torch.float64).tiny
    relative_error = float(torch.linalg.vector_norm(residual)
                           / torch.linalg.vector_norm(native).clamp_min(torch.finfo(torch.float32).tiny))
    mass_relative_error = float(torch.linalg.vector_norm(residual)
                                / torch.linalg.vector_norm(mass).clamp_min(tiny))
    maximum_ratio = float(torch.max(residual / tolerance.clamp_min(tiny)))
    if not np.isfinite(maximum_ratio) or maximum_ratio > 1:
        raise RuntimeError(
            f"Native MRzero point resummation failed (relative error {relative_error:.3g}, "
            f"point-mass error {mass_relative_error:.3g}, tolerance ratio {maximum_ratio:.3g}).")
    return resummed, {"relative_error": relative_error, "mass_relative_error": mass_relative_error,
                      "max_tolerance_ratio": maximum_ratio}

