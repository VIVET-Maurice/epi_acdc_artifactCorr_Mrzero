"""Explicit physical geometry and comparison helpers for measured MR twins.

Array reindexing, physical rotation and periodic Fourier-origin reversal are
different operations. None of these helpers infers the correct physical
registration by minimizing image error or performs nonrigid registration.
"""

from __future__ import annotations

import numpy as np


def _spatial_axes(values, axes):
    if len(axes) != 2:
        raise ValueError("Provide exactly two distinct spatial axes.")
    result = tuple(int(axis) % values.ndim for axis in axes)
    if values.ndim < 2 or result[0] == result[1]:
        raise ValueError("Provide exactly two distinct spatial axes.")
    if any(int(axis) < -values.ndim or int(axis) >= values.ndim for axis in axes):
        raise ValueError("Spatial axis is outside the input array.")
    return result


def _image_and_grid(values, X_m, Y_m, rotation_deg, axes):
    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError("Image needs two spatial dimensions.")
    axes = _spatial_axes(values, axes)
    X, Y = np.asarray(X_m, dtype=float), np.asarray(Y_m, dtype=float)
    shape = tuple(values.shape[axis] for axis in axes)
    if (X.shape != shape or Y.shape != shape or min(shape) < 1
            or not np.isfinite(X).all() or not np.isfinite(Y).all()):
        raise ValueError("X_m and Y_m must be finite grids matching the ordered spatial axes.")
    if isinstance(rotation_deg, (bool, np.bool_)) or rotation_deg not in (0, 180):
        raise ValueError("rotation_deg must be 0 or 180.")
    return values, X, Y, axes


def reindex_image(values, X_m, Y_m, rotation_deg=0, *, axes=(-2, -1)):
    """Reorder data and coordinates together, leaving every point in place.

    Return ``(values_reindexed, X_reindexed, Y_reindexed, report)``. Descending
    coordinate axes after 180 degrees are intentional: this is a representation
    change and does not rotate the object relative to the coils or gradients.
    """
    values, X, Y, axes = _image_and_grid(values, X_m, Y_m, rotation_deg, axes)
    turns = rotation_deg // 90
    output = np.rot90(values, turns, axes=axes).copy()
    report = {"operation": "array reindexing only", "rotation_deg": rotation_deg,
              "changes_physical_placement": False, "interpolation": False,
              "boundary_wrap": False, "physical_registration_inferred": False}
    return output, np.rot90(X, turns).copy(), np.rot90(Y, turns).copy(), report


def rotate_physical_image(values, X_m, Y_m, rotation_deg=0, *,
                          center_m=(0.0, 0.0), axes=(-2, -1)):
    """Rotate physical sample positions around an explicit sequence-XY centre.

    Return ``(values_rotated, X_rotated, Y_rotated, report)`` without interpolation,
    wrapping or inventing coverage. At 180 degrees, values are reindexed and
    coordinates become ``2*center - rot90(coordinates, 2)``. For an even
    FFT-centred grid the resulting grid is displaced by one original pixel
    relative to the old grid; this displacement must not be discarded.

    This changes object placement relative to an unchanged scanner/coil frame.
    If instead the intention is a coordinate-frame transformation, all gradient
    k vectors, patient geometry and coil registration must be transformed
    consistently by the caller. A low image NRMSE cannot establish registration.
    """
    values, X, Y, axes = _image_and_grid(values, X_m, Y_m, rotation_deg, axes)
    center = np.asarray(center_m, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_m must contain finite sequence X/Y coordinates in metres.")
    output, outputX, outputY, _ = reindex_image(values, X, Y, rotation_deg, axes=axes)
    if rotation_deg == 180:
        outputX = 2 * center[0] - outputX
        outputY = 2 * center[1] - outputY
        transform = [[-1.0, 0.0, float(2 * center[0])],
                     [0.0, -1.0, float(2 * center[1])], [0.0, 0.0, 1.0]]
    else:
        transform = np.eye(3).tolist()
    report = {"operation": "physical sample rotation", "rotation_deg": rotation_deg,
              "center_sequence_XY_m": center.tolist(), "physical_transform_XY": transform,
              "changes_physical_placement": rotation_deg != 0,
              "interpolation": False, "boundary_wrap": False,
              "coordinate_frame_change_requires_joint_k_and_registration_transform": True,
              "physical_registration_inferred": False}
    return output, outputX, outputY, report


def fft_origin_reverse(values, axes=(-2, -1), *, return_report=False):
    """Reverse periodic Fourier coordinates about ``n//2`` on selected axes.

    This exact DFT permutation is ``source=(2*(n//2)-index) % n``. On even
    axes it equals a flip followed by a one-pixel roll; the Fourier origin
    stays fixed. Periodic Nyquist-boundary handling is explicit. It must not
    be used as an implicit physical resampling of a finite B0/calibration map.
    """
    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError("Image needs two spatial dimensions.")
    axes = _spatial_axes(values, axes)
    output = values
    maps = []
    for axis in axes:
        size = values.shape[axis]
        if size < 1:
            raise ValueError("Spatial axes must be nonempty.")
        source = (2 * (size // 2) - np.arange(size)) % size
        maps.append(source.tolist())
        output = np.take(output, source, axis=axis)
    output = output.copy()
    report = {"operation": "periodic DFT origin reversal",
              "fixed_origin_index": [values.shape[axis] // 2 for axis in axes],
              "source_index_maps": maps, "boundary_wrap": True,
              "boundary_convention": "periodic modulo indexing at Nyquist boundaries",
              "physical_registration_inferred": False}
    return (output, report) if return_report else output


def compare_magnitude(reference, candidate, mask=None, *, gain=None,
                      X_m=None, Y_m=None, contour_fraction=0.2):
    """Compare fixed-grid magnitudes, edges and support without registration.

    A missing ``gain`` fits one positive least-squares intensity factor on the
    supplied common mask. Reuse this returned gain for subsequent corrected
    scans; independent gain fitting can obscure relative intensity changes.
    Gradients use pixel coordinates and require valid neighbours, avoiding
    artificial edges at mask holes. Contour Dice compares thresholded support
    at the same fraction of the reference peak, after the shared gain.
    Optional physical grids add intensity-centroid displacement in millimetres;
    centroid RMS is the RMS of its two coordinate differences, not registration.
    """
    reference, candidate = np.abs(np.asarray(reference)), np.abs(np.asarray(candidate))
    if reference.ndim != 2 or candidate.shape != reference.shape or min(reference.shape) < 2:
        raise ValueError("Reference and candidate need matching two-dimensional image grids.")
    requested = np.ones(reference.shape, bool) if mask is None else np.asarray(mask, dtype=bool)
    if requested.shape != reference.shape:
        raise ValueError("Comparison mask must match the images.")
    valid = requested & np.isfinite(reference) & np.isfinite(candidate)
    if not valid.any() or np.sum(reference[valid]**2) <= 0:
        raise ValueError("Comparison needs finite pixels with nonzero reference energy.")
    fitted = gain is None
    if fitted:
        energy = float(np.sum(candidate[valid]**2))
        if energy <= 0:
            raise ValueError("Cannot fit intensity gain to a zero candidate.")
        gain = float(np.sum(reference[valid] * candidate[valid]) / energy)
    if not np.isscalar(gain) or not np.isfinite(gain) or gain <= 0:
        raise ValueError("Intensity gain must be positive and finite.")
    if not np.isfinite(contour_fraction) or not 0 < contour_fraction < 1:
        raise ValueError("contour_fraction must lie between zero and one.")
    scaled = candidate * gain
    difference = scaled[valid] - reference[valid]
    reference_rms = float(np.sqrt(np.mean(reference[valid]**2)))
    rmse = float(np.sqrt(np.mean(difference**2)))
    edge_valid = valid.copy()
    edge_valid[1:] &= valid[:-1]
    edge_valid[:-1] &= valid[1:]
    edge_valid[:, 1:] &= valid[:, :-1]
    edge_valid[:, :-1] &= valid[:, 1:]
    ref_gradient = np.stack(np.gradient(reference), axis=-1)
    cand_gradient = np.stack(np.gradient(scaled), axis=-1)
    ref_edge_energy = float(np.sum(ref_gradient[edge_valid]**2))
    edge_error = (100 * float(np.sqrt(np.sum(
        (cand_gradient[edge_valid] - ref_gradient[edge_valid])**2) / ref_edge_energy))
                  if ref_edge_energy > 0 else None)
    threshold = float(contour_fraction * np.max(reference[valid]))
    ref_contour = valid & (reference >= threshold)
    cand_contour = valid & (scaled >= threshold)
    denominator = int(ref_contour.sum() + cand_contour.sum())
    dice = float(2 * np.sum(ref_contour & cand_contour) / denominator) if denominator else None
    result = {"gain": float(gain), "gain_fitted": fitted,
              "requested_pixels": int(requested.sum()), "used_pixels": int(valid.sum()),
              "excluded_nonfinite_pixels": int(np.sum(requested & ~valid)),
              "RMSE": rmse, "NRMSE_percent": 100 * rmse / reference_rms,
              "edge_gradient_NRMSE_percent": edge_error, "gradient_coordinates": "pixels",
              "edge_pixels": int(edge_valid.sum()), "contour_Dice": dice,
              "contour_reference_peak_fraction": float(contour_fraction),
              "contour_shared_threshold": threshold,
              "nonrigid_registration_applied": False,
              "centroid_distance_mm": None, "centroid_RMS_mm": None}
    if (X_m is None) != (Y_m is None):
        raise ValueError("Physical centroid comparison needs both X_m and Y_m.")
    if X_m is not None:
        X, Y = np.asarray(X_m, dtype=float), np.asarray(Y_m, dtype=float)
        if (X.shape != reference.shape or Y.shape != reference.shape
                or not np.isfinite(X[valid]).all() or not np.isfinite(Y[valid]).all()):
            raise ValueError("Physical comparison grids must match finite comparison pixels.")
        if float(np.sum(scaled[valid])) > 0:
            coordinates = np.stack((X[valid], Y[valid]), axis=-1)
            ref_center = np.sum(coordinates * reference[valid, None], axis=0) / np.sum(reference[valid])
            cand_center = np.sum(coordinates * scaled[valid, None], axis=0) / np.sum(scaled[valid])
            delta = (cand_center - ref_center) * 1000
            result.update(reference_centroid_XY_mm=(ref_center * 1000).tolist(),
                          candidate_centroid_XY_mm=(cand_center * 1000).tolist(),
                          centroid_distance_mm=float(np.linalg.norm(delta)),
                          centroid_RMS_mm=float(np.sqrt(np.mean(delta**2))))
    return result


__all__ = ["reindex_image", "rotate_physical_image", "fft_origin_reverse", "compare_magnitude"]
