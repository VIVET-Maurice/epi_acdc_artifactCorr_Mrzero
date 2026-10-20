"""In-plane geometry without rotating or interpolating image arrays.

The sequence coordinates are X=patient +y, Y=patient -x. A frame rotation
changes both coordinates and k vectors. A physical alignment rotation changes
the object's positions relative to the scanner and is a different operation.
In-plane mirrors are also supported. X/y tilts out of the slice are excluded.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from continuous_reconstruction import ContinuousReconstruction


def _angle_degrees(value):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError("angle_deg must be a finite real scalar.")
    angle = np.asarray(value, dtype=float)
    if angle.ndim != 0 or not np.isfinite(angle):
        raise ValueError("angle_deg must be a finite real scalar.")
    return float(angle)


def _xy_coordinates(X_m, Y_m):
    if np.iscomplexobj(X_m) or np.iscomplexobj(Y_m):
        raise ValueError("X_m and Y_m must be real coordinates.")
    X, Y = np.asarray(X_m, dtype=float), np.asarray(Y_m, dtype=float)
    if X.shape != Y.shape or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("X_m and Y_m must have the same shape and be finite.")
    return X, Y


def rotation_matrix_2d(angle_deg):
    """Return the counterclockwise XY rotation matrix for an angle in degrees."""
    radians = np.deg2rad(_angle_degrees(angle_deg) % 360.)
    cosine, sine = np.cos(radians), np.sin(radians)
    # Exact right-angle zeros avoid artificial k residuals on Cartesian grids.
    if abs(cosine) < 4 * np.finfo(float).eps:
        cosine = 0.
    if abs(sine) < 4 * np.finfo(float).eps:
        sine = 0.
    return np.array([[cosine, -sine], [sine, cosine]])


def inplane_matrix(angle_deg=0., mirror_x=False, mirror_y=False):
    """Return R(angle) @ diag(sign_x, sign_y): named mirrors precede rotation.

    mirror_x negates X and mirror_y negates Y. A single mirror changes the
    determinant to -1; two mirrors are equivalent to a 180-degree rotation.
    """
    if not isinstance(mirror_x, (bool, np.bool_)) or not isinstance(mirror_y, (bool, np.bool_)):
        raise ValueError("mirror_x and mirror_y must be booleans.")
    signs = np.diag([-1. if mirror_x else 1., -1. if mirror_y else 1.])
    return rotation_matrix_2d(angle_deg) @ signs


def _orthogonal_matrix(value):
    if np.iscomplexobj(value):
        raise ValueError("frame matrix must be a finite real orthogonal 2x2 matrix.")
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(2), rtol=0, atol=1e-10)):
        raise ValueError("frame matrix must be a finite real orthogonal 2x2 matrix.")
    return matrix.copy()


def rotate_points(X_m, Y_m, angle_deg=0., center_m=(0., 0.), *,
                  mirror_x=False, mirror_y=False, matrix=None):
    """Rotate positions around an explicit center, returning new X/Y arrays.

    ``matrix`` supplies an explicit orthogonal transform and takes precedence
    over angle/mirror options. Joint frame changes use the origin (0, 0).
    A nonzero center
    is available for physical object alignment: rotating k alone would then
    omit the translation-dependent Fourier phase of that center.
    """
    X, Y = _xy_coordinates(X_m, Y_m)
    if np.iscomplexobj(center_m):
        raise ValueError("center_m must contain two finite real coordinates.")
    center = np.asarray(center_m, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_m must contain two finite real coordinates.")
    matrix = (inplane_matrix(angle_deg, mirror_x, mirror_y) if matrix is None
              else _orthogonal_matrix(matrix))
    dx, dy = X - center[0], Y - center[1]
    return (center[0] + matrix[0, 0] * dx + matrix[0, 1] * dy,
            center[1] + matrix[1, 0] * dx + matrix[1, 1] * dy)


def rotate_acquisition_k(acquisition, angle_deg=0., *,
                         mirror_x=False, mirror_y=False, matrix=None):
    """Copy an acquisition and rotate its k vectors about the sequence origin.

    Every ADC keeps its time, phase, index and sample value. No sample is
    rounded, reordered, reflected or discarded. To change coordinate frame,
    use the same angle for spin coordinates and the reconstruction grid.
    """
    kx, ky = _xy_coordinates(acquisition['kx_mInv'], acquisition['ky_mInv'])
    result = deepcopy(acquisition)
    result['kx_mInv'], result['ky_mInv'] = rotate_points(
        kx, ky, angle_deg, mirror_x=mirror_x, mirror_y=mirror_y, matrix=matrix)
    return result


def frame_to_patient_xyz(X_m, Y_m, Z_patient_m, frame_angle_deg=0., *, frame_matrix=None):
    """Undo the joint XY frame rotation before querying registered 3D coils.

    Z_patient_m is the absolute patient-z coordinate of each point, including
    its actual thickness offset. It may be scalar or broadcast to X/Y shape.
    Output shape is X_m.shape + (3,), ordered patient x/y/z in metres. This
    supports the existing axial frame, not an arbitrary oblique slice.
    """
    matrix = (inplane_matrix(frame_angle_deg) if frame_matrix is None
              else _orthogonal_matrix(frame_matrix))
    X, Y = rotate_points(X_m, Y_m, matrix=matrix.T)
    if np.iscomplexobj(Z_patient_m):
        raise ValueError("Z_patient_m must be real and finite.")
    try:
        Z = np.broadcast_to(np.asarray(Z_patient_m, dtype=float), X.shape)
    except ValueError as error:
        raise ValueError("Z_patient_m must broadcast to the X/Y coordinate shape.") from error
    if not np.isfinite(Z).all():
        raise ValueError("Z_patient_m must be real and finite.")
    return np.stack((-Y, X, Z), axis=-1)


class FrameReconstruction:
    """Fourier reconstruction on the jointly rotated output coordinate grid.

    ``acquisition`` already contains k vectors transformed by ``frame_matrix``
    or, if absent, ``frame_angle_deg``. A supplied orthogonal matrix takes
    precedence and may include mirrors. This operator undoes it internally and uses the existing
    continuous-coordinate reconstruction. Its output coordinates are rotated
    by the same angle. Since (R k).(R x) = k.x, the kernels and image array
    ordering remain unchanged. ``adjoint`` is the true Hermitian adjoint.

    This is essential for angles other than right-angle array permutations:
    rotating k and spins while reconstructing on the original XY grid would
    rotate the simulated image relative to the comparison target.
    """

    def __init__(self, acquisition, device, dtype, frame_angle_deg=0., sample_chunk=256,
                 frame_matrix=None):
        self.frame_angle_deg = _angle_degrees(frame_angle_deg) if frame_matrix is None else None
        self.frame_matrix = (inplane_matrix(self.frame_angle_deg) if frame_matrix is None
                             else _orthogonal_matrix(frame_matrix))
        native_acquisition = rotate_acquisition_k(acquisition, matrix=self.frame_matrix.T)
        self.base = ContinuousReconstruction(native_acquisition, device, dtype,
                                              sample_chunk=sample_chunk)
        self.device, self.dtype = self.base.device, self.base.dtype
        self.complex_dtype = self.base.complex_dtype
        self.image_size, self.nsamples = self.base.image_size, self.base.nsamples
        self.X_m, self.Y_m = rotate_points(self.base.X_m, self.base.Y_m,
                                          matrix=self.frame_matrix)
        self.report = dict(self.base.report)
        self.report.update(
            operator="normalized Fourier adjoint on the jointly rotated k/output grid",
            frame_angle_deg=self.frame_angle_deg,
            frame_matrix=self.frame_matrix.tolist(),
            frame_determinant=float(np.linalg.det(self.frame_matrix)),
            frame_center_m=[0., 0.],
            frame_operation="joint orthogonal spin, k and output-coordinate transform; native array ordering retained",
            k_diagnostics_frame="native acquisition after inverse frame rotation",
            array_resampling=False,
            changes_physical_alignment=False,
            physical_registration_inferred=False,
        )

    def image(self, kdata):
        """Reconstruct all ADCs, retaining the native output-array ordering."""
        return self.base.image(kdata)

    def adjoint(self, image):
        """Apply the exact Hermitian adjoint, including normalization."""
        return self.base.adjoint(image)

    def numpy_image(self, kdata):
        """Return NumPy complex receiver images with optional leading batches."""
        return self.base.numpy_image(kdata)


__all__ = ['rotation_matrix_2d', 'inplane_matrix', 'rotate_points', 'rotate_acquisition_k',
           'frame_to_patient_xyz', 'FrameReconstruction']
