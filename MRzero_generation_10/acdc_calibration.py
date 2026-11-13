"""Read the 3D ACDC calibration and sample it on the voxels of an object.

``data/ACDC_3T.npz`` contains, on a regular 2 mm grid in the coil frame:

* ``fieldmap[x, y, z, channel]``: field produced by 1 A in each of the
  32 channels, in Hz/A;
* ``mask[x, y, z]``: where the calibration was measured;
* ``xx, yy, zz``: coordinates in metres.

As in generation 08, every voxel needs the coil field at its centre and the
change of that field across the voxel along x, y and z. The centre gives the
phase of the voxel; the changes give its intravoxel dephasing.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class CoilFields:
    """ACDC fields sampled on ``voxels`` object voxels."""

    maps_Hz_per_A: np.ndarray       # [voxels, channels] field at the voxel centre
    gradients_Hz_per_A: np.ndarray  # [voxels, 3, channels] change across the voxel (x, y, z)
    covered: np.ndarray             # [voxels] True if the whole voxel lies inside the calibration


def load_calibration(path):
    """Load the calibration file and check that its grid is regular."""
    with np.load(path) as archive:
        fieldmap = archive["fieldmap"]
        mask = archive["mask"].astype(bool)
        coordinates = [archive[name] for name in ("xx", "yy", "zz")]

    axes = []
    for dimension, volume in enumerate(coordinates):
        selector = [0, 0, 0]
        selector[dimension] = slice(None)
        axis = volume[tuple(selector)]
        shape = [1, 1, 1]
        shape[dimension] = -1
        if not np.allclose(volume, axis.reshape(shape)) or np.any(np.diff(axis) <= 0):
            raise ValueError("xx, yy, zz must follow array axes 0, 1, 2 and increase.")
        axes.append(axis.copy())
    return {"axes": tuple(axes), "fieldmap": fieldmap, "mask": mask}


def sample_coil_fields(calibration, centres_m, voxel_size_m):
    """Interpolate the coil fields on voxels of size ``voxel_size_m`` (x, y, z).

    ``centres_m`` has shape [voxels, 3]. Trilinear interpolation is used and
    nothing is extrapolated: a voxel is ``covered`` only if its centre and its
    eight corners are surrounded by measured calibration points.
    """
    centres = np.asarray(centres_m, dtype=float)
    size = np.asarray(voxel_size_m, dtype=float)
    options = dict(method="linear", bounds_error=False)
    field = RegularGridInterpolator(calibration["axes"], calibration["fieldmap"],
                                    fill_value=np.nan, **options)
    measured = RegularGridInterpolator(calibration["axes"], calibration["mask"].astype(float),
                                       fill_value=0.0, **options)

    maps = field(centres)

    # Field change across the voxel: difference between two opposite faces.
    gradients = np.empty((len(centres), 3, maps.shape[-1]))
    for axis in range(3):
        half = np.zeros(3)
        half[axis] = 0.5 * size[axis]
        gradients[:, axis] = field(centres + half) - field(centres - half)

    # The interpolated mask equals 1 only if all eight neighbours were measured.
    covered = measured(centres) > 1 - 1e-9
    for corner in product((-0.5, 0.5), repeat=3):
        covered &= measured(centres + np.asarray(corner) * size) > 1 - 1e-9
    covered &= np.isfinite(maps).all(axis=1) & np.isfinite(gradients).all(axis=(1, 2))

    return CoilFields(maps_Hz_per_A=maps, gradients_Hz_per_A=gradients, covered=covered)
