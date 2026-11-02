"""Patient-space geometry of the native Twix readout/phase frame."""

from __future__ import annotations

import numpy as np


AXIAL_QUATERNION_WXYZ = np.asarray([1., 0., 0., 1.]) / np.sqrt(2.)


def normalized_quaternion(value) -> np.ndarray:
    """Validate a Siemens w,x,y,z quaternion, then remove rounding drift."""
    try:
        if np.iscomplexobj(value):
            raise ValueError
        quaternion = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Twix slice quaternion must contain four finite real values with unit norm.") from None
    if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1., rtol=0., atol=1e-6)):
        raise ValueError("Twix slice quaternion must contain four finite values with unit norm.")
    return quaternion / np.linalg.norm(quaternion)


def slice_basis(metadata: dict) -> np.ndarray:
    """Return patient-space columns (readout, phase, slice normal).

    Missing orientation retains the historical axial frame for legacy image
    archives. Raw Twix readers always supply the measured quaternion.
    """
    w, x, y, z = normalized_quaternion(metadata.get("slice_quaternion_wxyz",
                                                  AXIAL_QUATERNION_WXYZ))
    basis = np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    basis[np.abs(basis) < 1e-15] = 0.
    basis[np.abs(basis - 1.) < 1e-15] = 1.
    basis[np.abs(basis + 1.) < 1e-15] = -1.
    return basis


def slice_center(metadata: dict) -> np.ndarray:
    """Return the measured centre in mm; old archives default to isocentre."""
    value = metadata.get("slice_position_mm", [0., 0., 0.])
    try:
        if np.iscomplexobj(value):
            raise ValueError
        center = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Twix slice position must contain three finite real patient coordinates.") from None
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Twix slice position must contain three finite patient coordinates.")
    return center


def encoding_origin(metadata: dict) -> np.ndarray:
    """Project the patient slice centre onto readout/phase axes, in metres."""
    origin = slice_basis(metadata)[:, :2].T @ slice_center(metadata) * 1e-3
    if "encoding_origin_m" in metadata:
        value = metadata["encoding_origin_m"]
        try:
            if np.iscomplexobj(value):
                raise ValueError
            declared = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("encoding_origin_m must contain two finite real readout/phase coordinates.") from None
        if declared.shape != (2,) or not np.isfinite(declared).all():
            raise ValueError("encoding_origin_m must contain two finite readout/phase coordinates.")
        if not np.allclose(declared, origin, rtol=0., atol=1e-6):
            raise ValueError("The encoding origin disagrees with the measured Twix slice centre/orientation.")
    return origin
