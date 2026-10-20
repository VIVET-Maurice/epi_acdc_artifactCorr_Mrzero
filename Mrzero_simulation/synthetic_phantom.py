"""Analytical 2D phantom and known truth for synthetic MRzero acquisitions.

Spatial axes are [Y row, X column]. Positions are in metres, B0 in Hz and
R2star in inverse seconds. Proton density is the relative amplitude at RF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Phantom2D:
    """Object maps independent of the acquisition matrix.

    Receiver weights have axes [receiver, Y, X] and unit RSS norm at every
    point. Apply them once to object amplitude. No T1 or steady state is
    inferred, because the acquisition simulator uses ideal independent RF.
    """

    X_m: np.ndarray
    Y_m: np.ndarray
    proton_density: np.ndarray
    R2star_sInv: np.ndarray
    B0_Hz: np.ndarray
    support_mask: np.ndarray
    receive_weights: np.ndarray
    FOV_m: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        """Amplitude at RF, before decay and without pixel-area weights."""
        return self.proton_density

    @property
    def PD(self) -> np.ndarray:
        """Conventional alias for relative proton density."""
        return self.proton_density

    @property
    def rate_sInv(self) -> np.ndarray:
        """Decay rate accepted by NativeEPITwin."""
        return self.R2star_sInv

    @property
    def T2star_s(self) -> np.ndarray:
        """Effective decay time in seconds; infinite outside the object."""
        return np.divide(
            1.0, self.R2star_sInv,
            out=np.full_like(self.R2star_sInv, np.inf),
            where=self.R2star_sInv > 0,
        )


def _elliptical_blob(x, y, center_x, center_y, radius_x, radius_y):
    """Smooth Gaussian region with center and radius expressed in FOV units."""
    radius_squared = ((x - center_x) / radius_x) ** 2
    radius_squared += ((y - center_y) / radius_y) ** 2
    return np.exp(-2.0 * radius_squared)


def _receiver_weights(x, y, ncoils):
    """Smooth complex sensitivities, normalized pointwise to unit RSS."""
    if ncoils == 1:
        return np.ones((1,) + x.shape, dtype=np.complex128)
    weights = []
    for angle in np.linspace(0.0, 2.0 * np.pi, ncoils, endpoint=False):
        direction = x * np.cos(angle) + y * np.sin(angle)
        magnitude = np.exp(1.5 * direction)
        phase_rad = 1.2 * (x * np.sin(angle) - y * np.cos(angle)) + angle
        weights.append(magnitude * np.exp(1j * phase_rad))
    weights = np.asarray(weights)
    return weights / np.sqrt(np.sum(np.abs(weights) ** 2, axis=0))[None]


def make_phantom(
    matrix_size: int = 128,
    fov_m: float = 0.2,
    *,
    b0_scale: float = 1.0,
    ncoils: int = 1,
) -> Phantom2D:
    """Create a smooth asymmetric elliptical object without patient data.

    At a 0.20 m FOV, outer radii are 73 mm along X and 80 mm along Y. Broad
    elliptical regions remain visible at GRE 32-square, EPI 64-square and
    B0 128-square resolutions. Corners contain no object signal and can be
    used to estimate noise added later. Radii scale proportionally to FOV.

    R2star varies around 9--15 inverse seconds. B0 combines a slope, a
    quadratic term and two local disturbances of a few tens of Hz.
    b0_scale is dimensionless; zero creates a homogeneous-field control.

    The coordinate origin is at matrix_size // 2, as in the continuous
    reconstruction. Pixel-area weighting is left to the caller: multiply
    amplitude once by object_pixel_area / acquisition_pixel_area.
    """
    if (isinstance(matrix_size, (bool, np.bool_))
            or not isinstance(matrix_size, (int, np.integer))
            or matrix_size < 8):
        raise ValueError("matrix_size must be an integer >= 8.")
    if (isinstance(ncoils, (bool, np.bool_))
            or not isinstance(ncoils, (int, np.integer))
            or ncoils < 1):
        raise ValueError("ncoils must be a positive integer.")
    fov_m, b0_scale = float(fov_m), float(b0_scale)
    if not np.isfinite(fov_m) or fov_m <= 0:
        raise ValueError("fov_m must be a finite positive length in metres.")
    if not np.isfinite(b0_scale):
        raise ValueError("b0_scale must be a finite dimensionless factor.")

    axis_m = (np.arange(matrix_size) - matrix_size // 2) * fov_m / matrix_size
    X_m, Y_m = np.meshgrid(axis_m, axis_m)
    x, y = X_m / fov_m, Y_m / fov_m
    radius = np.sqrt((x / 0.365) ** 2 + (y / 0.400) ** 2)
    support = radius < 1.0

    left = _elliptical_blob(x, y, -0.14, 0.02, 0.14, 0.24)
    upper_right = _elliptical_blob(x, y, 0.11, 0.14, 0.12, 0.15)
    lower = _elliptical_blob(x, y, 0.06, -0.20, 0.16, 0.10)
    pocket = _elliptical_blob(x, y, -0.04, -0.02, 0.06, 0.11)
    # A cosine taper occupies the outer 6 percent of the object's radius.
    edge_position = np.clip((radius - 0.94) / 0.06, 0.0, 1.0)
    edge_taper = 0.5 * (1.0 + np.cos(np.pi * edge_position))
    density = (0.70 + 0.18 * left + 0.30 * upper_right
               - 0.14 * lower - 0.28 * pocket) * edge_taper
    density = np.where(support, density, 0.0)
    rate = np.where(support, 9.0 + 3.0 * upper_right + 2.0 * lower + pocket, 0.0)

    local_positive = _elliptical_blob(x, y, 0.22, 0.17, 0.10, 0.15)
    local_negative = _elliptical_blob(x, y, -0.20, -0.15, 0.13, 0.10)
    field_hz = (65.0 * x - 45.0 * y + 95.0 * x * y
                + 55.0 * local_positive - 42.0 * local_negative)
    field_hz = np.where(support, b0_scale * field_hz, 0.0)

    return Phantom2D(
        X_m=X_m, Y_m=Y_m, proton_density=density,
        R2star_sInv=rate, B0_Hz=field_hz, support_mask=support,
        receive_weights=_receiver_weights(x, y, ncoils),
        FOV_m=np.array([fov_m, fov_m], dtype=np.float64),
    )


def save_ground_truth(path: str | Path, phantom: Phantom2D) -> Path:
    """Save known maps as an NPZ without Python objects.

    Ground-truth maps evaluate errors; they must not replace maps estimated
    from the simulated echoes in the pipeline being tested.
    """
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Ground truth must be saved in a .npz file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_m=phantom.X_m, Y_m=phantom.Y_m,
        proton_density=phantom.proton_density,
        R2star_sInv=phantom.R2star_sInv,
        T2star_s=phantom.T2star_s, B0_Hz=phantom.B0_Hz,
        support_mask=phantom.support_mask,
        receive_weights=phantom.receive_weights, FOV_m=phantom.FOV_m,
        amplitude_reference_time_s=np.array(0.0),
        spatial_axes=np.array(["Y", "X"]),
        phase_convention=np.array("exp(-2*pi*i*(kx*X+ky*Y+B0*t))"),
    )
    return path


__all__ = ["Phantom2D", "make_phantom", "save_ground_truth"]
