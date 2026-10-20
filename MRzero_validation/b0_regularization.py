"""Regularize measured B0 with local planes, retaining the measured mask.

The bandwidth is chosen in native pixels, independently of the EPI image.
Unlike an ordinary masked Gaussian blur, a local plane preserves a linear
field next to irregular mask boundaries. It does not add measured resolution.
"""

import numpy as np
from scipy.ndimage import correlate


def regularize_b0_local_plane(b0_Hz, valid_mask, *, sigma_pixels=.75,
                             truncate=3., spacing_mm=None, preserve_mean=True):
    """Return raw/regularized fields, unchanged coverage and an explicit report.

    Fit ``a + b*dr + c*dc`` to finite measured neighbors using a Gaussian
    window. The value at the window center is ``a``. Pixels outside the input
    mask keep their original value, including NaNs. Singular neighborhoods
    also keep their raw value. Optional mean preservation adds one common
    offset to successfully fitted pixels only, leaving singular pixels intact.
    ``spacing_mm`` is row/column spacing, used for reporting the bandwidth.
    """
    raw = np.asarray(b0_Hz, dtype=float)
    mask = np.asarray(valid_mask)
    if (raw.ndim != 2 or mask.shape != raw.shape or np.iscomplexobj(b0_Hz)
            or not np.isin(mask, (0, 1)).all()):
        raise ValueError("B0 must be a real 2D array and its validity mask binary with matching shape.")
    sigma = np.asarray(sigma_pixels, dtype=float)
    if sigma.size == 1:
        sigma = np.full(2, sigma.item())
    if sigma.shape != (2,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("sigma_pixels must be positive scalar or row/column pair.")
    if not np.isscalar(truncate) or not np.isfinite(truncate) or truncate <= 0:
        raise ValueError("truncate must be positive and finite.")
    if not isinstance(preserve_mean, (bool, np.bool_)):
        raise ValueError("preserve_mean must be a boolean.")
    spacing = None if spacing_mm is None else np.asarray(spacing_mm, dtype=float)
    if spacing is not None and (spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
        raise ValueError("spacing_mm must contain positive row/column spacing.")
    valid = mask.astype(bool) & np.isfinite(raw)
    if not valid.any():
        raise ValueError("No finite valid measured B0 pixels remain.")
    radius = np.maximum(1, np.ceil(truncate * sigma).astype(int))
    dr, dc = np.meshgrid(np.arange(-radius[0], radius[0] + 1),
                         np.arange(-radius[1], radius[1] + 1), indexing="ij")
    window = np.exp(-.5 * ((dr / sigma[0])**2 + (dc / sigma[1])**2))
    window /= window.sum()
    design = (np.ones_like(dr), dr, dc)
    measured = np.where(valid, raw, 0.)
    matrix = np.empty(raw.shape + (3, 3))
    rhs = np.empty(raw.shape + (3,))
    for row in range(3):
        rhs[..., row] = correlate(measured, window * design[row], mode="constant", cval=0.)
        for column in range(row, 3):
            moment = correlate(valid.astype(float), window * design[row] * design[column],
                               mode="constant", cval=0.)
            matrix[..., row, column] = moment
            matrix[..., column, row] = moment
    eigenvalues = np.linalg.eigvalsh(matrix)
    fitted = valid & (eigenvalues[..., 0] > 1e-10 * eigenvalues[..., -1])
    output = raw.copy()
    if fitted.any():
        output[fitted] = np.linalg.solve(matrix[fitted], rhs[fitted, :, None])[:, 0, 0]
    mean_before = float(raw[valid].mean())
    mean_offset = 0.
    if preserve_mean and fitted.any():
        mean_offset = float((raw[valid].sum() - output[valid].sum()) / fitted.sum())
        output[fitted] += mean_offset
    change = np.zeros_like(raw)
    change[valid] = output[valid] - raw[valid]
    report = {"method": "Gaussian-weighted local plane [1, row offset, column offset]",
        "sigma_pixels_row_column": sigma.tolist(), "window_radius_pixels_row_column": radius.tolist(),
        "window_shape_pixels": list(window.shape), "truncate_sigma": float(truncate),
        "spacing_mm_row_column": None if spacing is None else spacing.tolist(),
        "sigma_mm_row_column": None if spacing is None else (sigma * spacing).tolist(),
        "window_width_mm_row_column": None if spacing is None else (np.asarray(window.shape) * spacing).tolist(),
        "input_valid_pixels": int(mask.sum()), "finite_valid_pixels": int(valid.sum()),
        "fitted_pixels": int(fitted.sum()), "singular_raw_fallback_pixels": int((valid & ~fitted).sum()),
        "invalid_pixels_filled": 0, "preserve_mean": bool(preserve_mean),
        "mean_before_Hz": mean_before, "mean_after_Hz": float(output[valid].mean()),
        "mean_restoring_offset_Hz": mean_offset,
        "RMS_change_Hz": float(np.sqrt(np.mean(change[valid]**2))),
        "max_abs_change_Hz": float(np.max(abs(change[valid]))),
        "bandwidth_selection": "Fixed in native pixels; no EPI image or optimization result is used.",
        "coverage_policy": "Only finite pixels in the original mask are fitted; no interpolation outside measured coverage."}
    return {"b0Hz": output, "mask": valid, "raw_b0Hz": raw.copy(),
            "change_Hz": change, "fitted_mask": fitted, "report": report}


__all__ = ["regularize_b0_local_plane"]
