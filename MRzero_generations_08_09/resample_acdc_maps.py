"""Interpolate already registered 2D measured maps without extrapolation."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def resample_to_grid(values, X, Y, mask, targetX, targetY):
    """Return (maps, valid) on the target grid; trailing axes are channels.

    Coordinates are metres in the same physical frame. This performs sampling,
    not registration. Rectilinear source grids may run along either array axis
    and in either direction. Every contributing measurement must be valid.
    """
    values = np.asarray(values)
    X, Y, targetX, targetY = [np.asarray(a, dtype=float) for a in (X, Y, targetX, targetY)]
    if (X.ndim != 2 or min(X.shape) < 2 or Y.shape != X.shape
            or values.shape[:2] != X.shape or values.ndim not in (2, 3)
            or np.iscomplexobj(values) or not np.isfinite(X).all() or not np.isfinite(Y).all()
            or targetX.ndim != 2 or targetY.shape != targetX.shape
            or not np.isfinite(targetX).all() or not np.isfinite(targetY).all()):
        raise ValueError("Measured maps need finite 2D coordinates and matching real arrays.")
    mask = np.asarray(mask)
    if mask.shape != X.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("Measured-map validity mask must be binary and match its source grid.")
    valid = mask.astype(bool) & (np.isfinite(values) if values.ndim == 2
                                else np.isfinite(values).all(axis=2))
    values = values.astype(float)
    if (X.shape == targetX.shape and np.allclose(X, targetX, atol=1e-9, rtol=0)
            and np.allclose(Y, targetY, atol=1e-9, rtol=0)):
        result = values.copy()
        result[~valid] = np.nan
        return result, valid
    # Arrange values as (physical x, physical y, optional channel).
    if (np.allclose(X, X[:, :1], atol=1e-10, rtol=0)
            and np.allclose(Y, Y[:1, :], atol=1e-10, rtol=0)):
        x, y = X[:, 0], Y[0, :]
    elif (np.allclose(X, X[:1, :], atol=1e-10, rtol=0)
          and np.allclose(Y, Y[:, :1], atol=1e-10, rtol=0)):
        x, y = X[0, :], Y[:, 0]
        values, valid = values.swapaxes(0, 1), valid.T
    else:
        raise ValueError("Source maps must use a registered rectilinear X_m/Y_m grid.")
    for axis in (x, y):
        if not (np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0)):
            raise ValueError("Source coordinate axes must be strictly monotonic.")
    points = np.stack((targetX, targetY), axis=-1)
    weights = RegularGridInterpolator((x, y), valid.astype(float), bounds_error=False,
                                      fill_value=0)(points)
    target_valid = weights >= 1 - 1e-12
    safe_values = values.copy()
    safe_values[~valid] = 0  # invalid values never contribute to an accepted sample
    result = RegularGridInterpolator((x, y), safe_values, bounds_error=False,
                                      fill_value=np.nan)(points)
    result[~target_valid] = np.nan
    return result, target_valid
