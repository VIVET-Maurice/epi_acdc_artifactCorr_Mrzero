"""Refine a measured reference before using it as a sampled Fourier object."""

import numpy as np

from resample_acdc_maps import resample_to_grid


def prepare_simulation_reference(reference, acquisition):
    """Return (reference on simulation grid, report), retaining the reference FOV.

    The MATLAB model treats each object pixel as a point source. Its spacing
    must be no larger than the scan's reconstructed pixel spacing, otherwise
    a coarse reference produces a comb in the finer inverse FFT. Refine only
    when needed, using linear interpolation in the measured physical frame.
    This does not recover spatial detail absent from the native reference.

    Uncovered reference samples are zero outside ``valid_mask``. B0 and ACDC
    must subsequently be sampled directly on this same grid with their own
    validity checks. Patient position/spacing are updated for calibration.
    """
    magnitude = np.asarray(reference["magnitude"], dtype=float)
    X, Y = (np.asarray(reference[key], dtype=float) for key in ("X_m", "Y_m"))
    if (magnitude.ndim != 2 or min(magnitude.shape) < 2 or X.shape != magnitude.shape
            or Y.shape != magnitude.shape or not np.isfinite(magnitude).all()
            or np.any(magnitude < 0) or not np.isfinite(X).all() or not np.isfinite(Y).all()):
        raise ValueError("Reference needs finite, nonnegative magnitude and matching physical X_m/Y_m.")
    coordinates = np.stack((X, Y), axis=-1)
    row_step = coordinates[1, 0] - coordinates[0, 0]
    column_step = coordinates[0, 1] - coordinates[0, 0]
    steps = np.stack((row_step, column_step))
    # Each array axis must follow exactly one physical sequence axis.
    nonzero = np.abs(steps) > 1e-12
    if not (np.all(nonzero.sum(axis=0) == 1) and np.all(nonzero.sum(axis=1) == 1)):
        raise ValueError("Simulation reference needs a rectilinear X_m/Y_m grid.")
    rows, columns = np.indices(magnitude.shape)
    expected = coordinates[0, 0] + rows[..., None] * row_step + columns[..., None] * column_step
    if not np.allclose(coordinates, expected, rtol=0, atol=1e-10):
        raise ValueError("Simulation reference needs constant physical pixel spacing.")
    scan_shape = np.asarray(acquisition["kx_mInv"]).shape
    fov = np.asarray(acquisition["FOV_m"], dtype=float)
    if len(scan_shape) != 2 or min(scan_shape) < 2 or fov.shape != (2,) or not np.isfinite(fov).all() or np.any(fov <= 0):
        raise ValueError("Scan needs a 2D trajectory and positive readout/phase FOV_m.")
    scan_spacing = fov / np.asarray(scan_shape[::-1])  # physical X, Y
    native_shape = np.asarray(magnitude.shape)
    physical_axis = nonzero.argmax(axis=1)
    native_spacing = np.abs(steps[np.arange(2), physical_axis])
    required = native_shape * native_spacing / scan_spacing[physical_axis]
    # Pulseq gradient rounding and Twix FOV headers can differ by a few ppm.
    rounded = np.rint(required)
    required = np.where(np.isclose(required, rounded, rtol=1e-4, atol=1e-8), rounded, required)
    shape = np.maximum(native_shape, np.ceil(required).astype(int))
    result = {key: reference[key] for key in
              ("position_mm", "spacing_mm", "orientation", "slice_thickness_mm") if key in reference}
    if np.array_equal(shape, native_shape):
        targetX, targetY, resampled = X.copy(), Y.copy(), magnitude.copy()
        valid = np.ones(magnitude.shape, dtype=bool)
    else:
        ratio = native_shape / shape
        target_rows, target_columns = np.indices(tuple(shape))
        center = coordinates[tuple(native_shape // 2)]
        target = (center + (target_rows - shape[0] // 2)[..., None] * row_step * ratio[0]
                  + (target_columns - shape[1] // 2)[..., None] * column_step * ratio[1])
        # Keep coordinates coinciding with native measurements exact. Floating
        # arithmetic at the first/last center must not turn coverage into NaN.
        for axis in range(2):
            native_index = ((np.arange(shape[axis]) - shape[axis] // 2) * ratio[axis]
                            + native_shape[axis] // 2)
            nearest = np.rint(native_index).astype(int)
            coincident = (np.isclose(native_index, nearest, rtol=0, atol=1e-12)
                          & (nearest >= 0) & (nearest < native_shape[axis]))
            component = physical_axis[axis]
            if axis == 0:
                target[coincident, :, component] = coordinates[nearest[coincident], 0, component, None]
            else:
                target[:, coincident, component] = coordinates[0, nearest[coincident], component]
        targetX, targetY = target[..., 0], target[..., 1]
        resampled, valid = resample_to_grid(magnitude, X, Y, np.ones(magnitude.shape, bool),
                                          targetX, targetY)
        resampled = np.where(valid, resampled, 0.0)
        if "spacing_mm" in result:
            spacing = np.asarray(result["spacing_mm"], dtype=float)
            result["spacing_mm"] = spacing * ratio
            if "position_mm" in result and "orientation" in result:
                orientation = np.asarray(result["orientation"], dtype=float)
                first_native_index = native_shape // 2 - (shape // 2) * ratio
                result["position_mm"] = (np.asarray(result["position_mm"], dtype=float)
                    + first_native_index[0] * spacing[0] * orientation[3:]
                    + first_native_index[1] * spacing[1] * orientation[:3])
    result.update(magnitude=resampled, X_m=targetX, Y_m=targetY, valid_mask=valid)
    report = {"native_reference_shape": native_shape.tolist(), "simulation_shape": shape.tolist(),
              "scan_shape": list(scan_shape), "refined": bool(np.any(shape != native_shape)),
              "native_spacing_mm": (native_spacing * 1000).tolist(),
              "simulation_spacing_mm": (native_spacing * native_shape / shape * 1000).tolist(),
              "reference_covered_pixels": int(valid.sum()),
              "interpolation": "Linear in physical coordinates; no extrapolation; native FOV retained."}
    return result, report
