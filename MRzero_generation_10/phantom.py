"""Numerical head-like phantom for the ACDC experiments at 3 T.

The phantom reproduces the situation that ACDC is meant to correct:

* an axial slice, 5 mm thick, through a head-sized object placed in the coil
  frame of the ACDC calibration (x: left-right, y: back-front, z: along B0);
* air cavities below the slice (the nasal cavity with the sinuses, and two
  ear canals). Each one is a sphere of air in tissue and creates the exact
  dipole field of a magnetised sphere. Above the nasal cavity the field
  reaches about 200 Hz and changes quickly through the slice: without
  correction the EPI image is both distorted and locally extinguished;
* a small smooth background field, as left by an imperfect global shim;
* simple contrast (white matter, cortex, ventricles) and a regular grid of
  darker lines, which makes geometric distortion easy to see.

Because the field is an analytic function of (x, y, z), its variation across
each voxel is known in all three directions, including through the slice.
"""

from dataclasses import dataclass

import numpy as np

LARMOR_HZ = 123.25e6                 # proton frequency at 3 T
AIR_TISSUE_SUSCEPTIBILITY = 9.4e-6   # chi(air) - chi(tissue), SI units

# Head outline in the slice: semi-axes along x and y.
HEAD_SEMI_AXES_M = (0.070, 0.085)

# Air cavities: centre (x, y, depth below the slice centre) and radius, in metres.
# Sizes and positions were chosen so that the off-resonance resembles the map
# measured in acquisition 029. Measured: -87 to +251 Hz, standard deviation
# 38 Hz, 13 Hz left after a static ACDC shim. Phantom: -36 to +210 Hz,
# standard deviation 36 Hz, 10 Hz left after a static ACDC shim.
CAVITIES_BELOW_SLICE_M = (
    {"name": "nasal cavity and sinuses", "x": 0.000, "y": 0.084, "depth": 0.040, "radius": 0.0275},
    {"name": "left ear canal", "x": -0.064, "y": -0.010, "depth": 0.030, "radius": 0.014},
    {"name": "right ear canal", "x": 0.064, "y": -0.010, "depth": 0.030, "radius": 0.014},
)


@dataclass(frozen=True)
class HeadPhantom:
    """Object maps on the image grid, [rows (y), columns (x)]."""

    X_m: np.ndarray            # x coordinate of every voxel centre
    Y_m: np.ndarray            # y coordinate of every voxel centre
    slice_z_m: float           # z coordinate of the slice centre
    voxel_size_m: np.ndarray   # (dx, dy, slice thickness)
    amplitude: np.ndarray      # transverse magnetisation right after the RF pulse
    T2star_s: np.ndarray       # signal decay time
    B0_Hz: np.ndarray          # off-resonance at the voxel centre
    B0_gradient_Hz: np.ndarray  # [rows, columns, 3] change of B0 across the voxel (x, y, z)
    support: np.ndarray        # True where the object is

    @property
    def n_voxels(self):
        return int(self.support.sum())

    def signal_fraction(self, time_s):
        """Fraction of each voxel's signal left at ``time_s`` without correction.

        The variation of B0 across a voxel spreads the phase of its spins; the
        remaining signal is the product of one sinc per axis. Shape [voxels].
        """
        return np.abs(np.prod(np.sinc(self.B0_gradient_Hz[self.support] * time_s), axis=1))

    def to_image(self, voxel_values):
        """Place one value per object voxel on the image grid (NaN elsewhere)."""
        image = np.full(self.support.shape, np.nan)
        image[self.support] = voxel_values
        return image

    def voxel_centres_m(self):
        """Coordinates [voxels, 3] of the object voxels, in the coil frame."""
        return np.column_stack((self.X_m[self.support], self.Y_m[self.support],
                                np.full(self.n_voxels, self.slice_z_m)))


def air_sphere_field_Hz(x, y, z, centre_m, radius_m):
    """Off-resonance outside a sphere of air surrounded by tissue, B0 along z.

    Exact magnetostatic solution: a dipole,
        f = f_Larmor * (delta_chi / 3) * (radius / r)^3 * (3 cos^2(theta) - 1),
    where r is the distance to the centre and theta the angle with B0.
    """
    dx, dy, dz = x - centre_m[0], y - centre_m[1], z - centre_m[2]
    r2 = dx**2 + dy**2 + dz**2
    strength = LARMOR_HZ * AIR_TISSUE_SUSCEPTIBILITY / 3.0
    return strength * radius_m**3 * (3.0 * dz**2 - r2) / r2**2.5


def b0_field_Hz(x, y, z, slice_z_m):
    """Off-resonance of the phantom at any point (x, y, z), in Hz."""
    field = np.zeros(np.broadcast(x, y, z).shape)
    for cavity in CAVITIES_BELOW_SLICE_M:
        centre = (cavity["x"], cavity["y"], slice_z_m - cavity["depth"])
        field = field + air_sphere_field_Hz(x, y, z, centre, cavity["radius"])
    # Smooth background left by the scanner shim (coordinates in units of 10 cm).
    u, v = x / 0.1, y / 0.1
    return field + 10.0 * u - 15.0 * v + 20.0 * (u**2 - v**2) + 15.0 * u * v


def make_head_phantom(matrix=128, fov_m=0.2, slice_z_m=0.03, slice_thickness_m=0.005,
                      b0_scale=1.0, grid_spacing_pixels=10):
    """Create the phantom on a ``matrix`` x ``matrix`` grid covering ``fov_m``.

    ``b0_scale`` multiplies the whole off-resonance (0 gives a perfect magnet).
    ``grid_spacing_pixels`` is the period of the darker grid lines (0: no grid).
    The grid origin is the same as in the image reconstruction: pixel
    ``matrix // 2`` is at coordinate 0.
    """
    pixel = fov_m / matrix
    axis = (np.arange(matrix) - matrix // 2) * pixel
    X, Y = np.meshgrid(axis, axis)
    semi_x, semi_y = HEAD_SEMI_AXES_M

    # --- Shape: an ellipse whose edge is smoothed over one pixel -------------
    radius = np.sqrt((X / semi_x)**2 + (Y / semi_y)**2)          # 1 on the outline
    distance_to_edge = np.hypot(X, Y) * (1.0 - radius) / np.maximum(radius, 1e-9)
    inside = np.clip(0.5 + distance_to_edge / pixel, 0.0, 1.0)   # fraction of the pixel in the head
    inside[radius < 1e-6] = 1.0

    # --- Contrast ----------------------------------------------------------
    amplitude = np.full(X.shape, 0.8)        # white matter
    T2star = np.full(X.shape, 0.045)
    amplitude[radius > 0.85] = 1.0           # cortex
    for side in (-1, 1):                     # two lateral ventricles
        ventricle = ((X - side * 0.014) / 0.007)**2 + ((Y - 0.005) / 0.020)**2 < 1.0
        amplitude[ventricle] = 1.2
        T2star[ventricle] = 0.070
    if grid_spacing_pixels:
        index = np.arange(matrix) - matrix // 2
        on_line = index % grid_spacing_pixels == 0
        amplitude[on_line[:, None] | on_line[None, :]] *= 0.5
    amplitude = amplitude * inside
    support = amplitude > 0

    # --- Off-resonance at the voxel centre and across the voxel ---------------
    Z = np.full(X.shape, slice_z_m)
    size = np.array([pixel, pixel, slice_thickness_m])
    B0 = b0_scale * b0_field_Hz(X, Y, Z, slice_z_m)
    B0_gradient = np.empty(X.shape + (3,))
    for direction, (ex, ey, ez) in enumerate(np.eye(3)):
        half = 0.5 * size[direction]
        B0_gradient[..., direction] = b0_scale * (
            b0_field_Hz(X + half * ex, Y + half * ey, Z + half * ez, slice_z_m)
            - b0_field_Hz(X - half * ex, Y - half * ey, Z - half * ez, slice_z_m))
    # The scanner tunes its frequency on the object: remove the mean offset.
    B0 = B0 - np.average(B0[support], weights=amplitude[support])
    B0[~support] = 0.0
    B0_gradient[~support] = 0.0

    return HeadPhantom(X_m=X, Y_m=Y, slice_z_m=float(slice_z_m), voxel_size_m=size,
                       amplitude=amplitude, T2star_s=T2star, B0_Hz=B0,
                       B0_gradient_Hz=B0_gradient, support=support)
