"""Prepare midpoint sub-spins without inventing measured field information.

The output is a virtual 2D array [parent voxel, sub-spin]. Its two axes are
storage axes, not image axes. The acquisition and reconstruction stay those
of notebook 04. A whole parent is excluded if any of its children lacks a
valid measured interpolation stencil; the remaining children are never
renormalized to compensate for missing measurements.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path
import sys

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "Experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from prepare_acdc_calibration import (interpolate_maps, sha256, source_grid,
                                     target_patient_grid)
from resample_acdc_maps import resample_to_grid


@lru_cache(maxsize=1)
def _load_source(path, modified_ns, size):
    """Keep one original 3D calibration in memory, shared by V2/V3 preparation."""
    with np.load(path, allow_pickle=False) as archive:
        names = ("fieldmap", "mask", "xx", "yy", "zz")
        if not all(name in archive for name in names):
            raise ValueError("Sub-spins require the original 3D ACDC NPZ, not center-sampled 2D maps.")
        source = {name: archive[name] for name in names}
    if np.iscomplexobj(source["fieldmap"]):
        raise ValueError("ACDC responses must be real Hz/A values.")
    axes = source_grid(source)
    # Invalid numerical values in any channel invalidate the entire stencil.
    source_valid = source["mask"].astype(bool) & np.isfinite(source["fieldmap"]).all(axis=-1)
    return source, axes, source_valid, sha256(Path(path))


def _full_stencil_valid(axes, mask, points):
    """Check every corner, including zero-weight corners at exact grid centers."""
    inside = np.ones(points.shape[:-1], dtype=bool)
    low_indices = []
    for dimension, axis in enumerate(axes):
        coordinate = points[..., dimension]
        inside &= (coordinate >= axis[0]) & (coordinate <= axis[-1])
        low = np.searchsorted(axis, coordinate, side="right") - 1
        low_indices.append(np.clip(low, 0, len(axis) - 2))
    for corner in product((0, 1), repeat=len(axes)):
        indices = tuple(low + step for low, step in zip(low_indices, corner))
        inside &= mask[indices]
    return inside


def _sample_native_b0(native, X, Y):
    """Sample the native measured 2D map directly in the fixed physical frame."""
    required = ("b0Hz", "X_m", "Y_m", "mask")
    if not isinstance(native, dict) or not all(key in native for key in required):
        raise ValueError("b0_mode='native' requires b0Hz, X_m, Y_m and mask on the native B0 grid.")
    values = np.asarray(native["b0Hz"], dtype=float)
    sourceX = np.asarray(native["X_m"], dtype=float)
    sourceY = np.asarray(native["Y_m"], dtype=float)
    mask = np.asarray(native["mask"])
    sampled, valid = resample_to_grid(values, sourceX, sourceY, mask, X, Y)
    # Match the explicitly validated axis handling in resample_to_grid.
    source_valid = mask.astype(bool) & np.isfinite(values)
    if (np.allclose(sourceX, sourceX[:, :1], atol=1e-10, rtol=0)
            and np.allclose(sourceY, sourceY[:1, :], atol=1e-10, rtol=0)):
        x, y = sourceX[:, 0], sourceY[0, :]
    else:
        x, y = sourceX[0, :], sourceY[:, 0]
        source_valid = source_valid.T
    if x[0] > x[-1]:
        x, source_valid = x[::-1], source_valid[::-1]
    if y[0] > y[-1]:
        y, source_valid = y[::-1], source_valid[:, ::-1]
    valid &= _full_stencil_valid((x, y), source_valid, np.stack((X, Y), axis=-1))
    sampled[~valid] = np.nan
    spacing_m = [float(np.median(np.diff(axis))) for axis in (x, y)]
    return sampled, valid, values.shape, spacing_m


def _binary_mask(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or not np.isin(result, (0, 1)).all():
        raise ValueError(f"{name} must be a binary mask matching the parent grid.")
    return result.astype(bool)


def prepare_subspin_inputs(experiment, calibration_path, calibration_settings,
                           b0_native=None, *, counts=(1, 1, 8),
                           b0_mode="parent", z_gradient_hz_per_m=0.0):
    """Return weighted sub-spins and the inputs accepted by the existing model.

    ``counts=(row, column, thickness)`` uses equally weighted midpoints of the
    parent voxel. V2 uses ``(1, 1, 8), b0_mode='parent'``: the measured 2D B0
    is constant through thickness. V3 uses e.g. ``(2, 2, 8), 'native'`` to
    sample the original 128x128 B0 at each in-plane position. Both versions
    sample the original 3D ACDC responses at every child position.

    ``z_gradient_hz_per_m`` is an explicitly synthetic additional B0 gradient
    along the slice normal. Zero is the default; no measured B0 variation
    through thickness is inferred from the 2D map. The GRE magnitude remains
    an effective transverse-amplitude proxy and is divided by child count,
    without estimating proton density or undoing any measured signal loss.
    """
    n = np.asarray(counts)
    if (n.shape != (3,) or n.dtype.kind == "b" or not np.isfinite(n).all()
            or np.any(n < 1) or np.any(n != np.floor(n))):
        raise ValueError("counts must contain three positive integers: row, column, thickness.")
    counts = tuple(n.astype(int))
    if b0_mode not in {"parent", "native"}:
        raise ValueError("b0_mode must be 'parent' or 'native'.")
    if not np.isscalar(z_gradient_hz_per_m) or not np.isfinite(z_gradient_hz_per_m):
        raise ValueError("z_gradient_hz_per_m must be a finite scalar.")
    settings = dict(calibration_settings)
    if settings.get("units") != "Hz/A" or settings.get("registration_confirmed") is not True:
        raise ValueError("Sub-spin calibration needs explicit Hz/A units and confirmed registration.")
    affine = np.asarray(settings.get("patient_to_source_m"), dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or abs(np.linalg.det(affine[:3, :3])) < 1e-10):
        raise ValueError("patient_to_source_m must be an invertible 4x4 affine in metres.")

    geometry, field = experiment["geometry"], experiment["field"]
    m0 = np.asarray(experiment["m0_proxy"], dtype=float)
    if m0.ndim != 2 or not np.isfinite(m0).all() or np.any(m0 < 0):
        raise ValueError("Parent m0_proxy must be finite, nonnegative and 2D.")
    support = _binary_mask(field["simulation_mask"], m0.shape, "simulation_mask") & (m0 > 0)
    quality = _binary_mask(field["mask"], m0.shape, "B0 quality mask")
    if not support.any():
        raise ValueError("Parent simulation support is empty.")
    patient = target_patient_grid(experiment)
    spacing = np.asarray(geometry["spacing_mm"], float).reshape(2) * 1e-3
    thickness = float(geometry["slice_thickness_mm"]) * 1e-3
    if (not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or not np.isfinite(thickness) or thickness <= 0):
        raise ValueError("Parent spacing and slice thickness must be finite and positive.")
    orientation = np.asarray(geometry["orientation"], float).reshape(6)
    column_direction, row_direction = orientation[:3], orientation[3:]
    normal = np.cross(column_direction, row_direction)
    if not np.allclose([column_direction[2], row_direction[2], abs(normal[2])],
                       [0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError("This measured 2D B0/trajectory adapter supports axial slices only; oblique encoding needs a separate geometry model.")
    fractions = [(np.arange(count) + .5) / count - .5 for count in counts]
    row, column, z = [value.ravel() for value in np.meshgrid(*fractions, indexing="ij")]
    offsets = (row[:, None] * spacing[0] * row_direction
               + column[:, None] * spacing[1] * column_direction
               + z[:, None] * thickness * normal)
    parent_index = np.flatnonzero(support)
    xyz = patient.reshape(-1, 3)[parent_index, None] + offsets[None]
    X, Y = xyz[..., 1], -xyz[..., 0]
    nchild = len(offsets)

    path = Path(calibration_path).resolve()
    stat = path.stat()
    source, axes, source_valid, source_hash = _load_source(str(path), stat.st_mtime_ns, stat.st_size)
    if settings.get("source_sha256") not in (None, source_hash):
        raise ValueError("The 3D calibration hash differs from the confirmed source_sha256.")
    query = xyz @ affine[:3, :3].T + affine[:3, 3]
    maps, coil_valid = interpolate_maps(source, axes, query)
    coil_valid &= _full_stencil_valid(axes, source_valid, query)
    maps[~coil_valid] = np.nan

    native_shape, native_spacing = None, None
    if b0_mode == "native":
        b0, b0_valid, native_shape, native_spacing = _sample_native_b0(b0_native, X, Y)
    else:
        parent_b0 = np.asarray(field["b0Hz"], dtype=float)
        if parent_b0.shape != m0.shape:
            raise ValueError("Parent B0 must match m0_proxy.")
        b0 = np.repeat(parent_b0.ravel()[parent_index, None], nchild, axis=1)
        b0_valid = np.isfinite(b0)
    b0 += float(z_gradient_hz_per_m) * (z * thickness)[None]
    parent_keep = (coil_valid & b0_valid).all(axis=1)
    parent_valid = np.zeros(m0.shape, dtype=bool)
    parent_valid.ravel()[parent_index[parent_keep]] = True
    if not parent_keep.any() or not np.any(parent_valid & quality):
        raise ValueError("No complete sub-spin parents with reliable B0 remain; inspect native B0/calibration coverage.")

    requested_amplitude = float(m0[support].sum())
    parent_index = parent_index[parent_keep]
    xyz, X, Y = xyz[parent_keep], X[parent_keep], Y[parent_keep]
    b0, maps = b0[parent_keep], maps[parent_keep]
    weights = np.full(b0.shape, 1.0 / nchild)
    amplitude = m0.ravel()[parent_index, None] * weights
    repeated_quality = np.repeat(quality.ravel()[parent_index, None], nchild, axis=1)
    t2 = np.asarray(experiment["relaxation"]["T2star_s"], dtype=float)
    if t2.size == 1:
        t2 = np.full(m0.shape, t2.item())
    if t2.shape != m0.shape or np.isnan(t2[support]).any() or np.any(t2[support] <= 0):
        raise ValueError("T2star_s must be positive scalar or parent-grid array; infinity is allowed.")
    t2 = np.repeat(t2.ravel()[parent_index, None], nchild, axis=1)

    metadata = {"parent_shape": m0.shape, "parent_flat_index": parent_index,
                "weights": weights, "counts": counts, "xyz_patient_m": xyz,
                "parent_valid_mask": parent_valid, "b0_mode": b0_mode}
    expanded = dict(experiment)
    expanded["m0_proxy"] = amplitude
    expanded["geometry"] = dict(geometry, X_m=X, Y_m=Y,
        Z_m=np.broadcast_to(z * thickness, b0.shape).copy())
    expanded["field"] = dict(field, b0Hz=b0, measured_b0Hz=b0.copy(),
        simulation_mask=np.ones(b0.shape, dtype=bool), mask=repeated_quality)
    expanded["relaxation"] = dict(experiment["relaxation"], T2star_s=t2)
    expanded["subspins"] = metadata
    calibration = {"acdc_maps": maps, "X_m": X.copy(), "Y_m": Y.copy(),
        "units": "Hz/A", "spatial_valid_mask": np.ones(b0.shape, dtype=bool),
        "registration_confirmed": True,
        "metadata": dict(settings, source_file=str(path), source_sha256=source_hash,
            source_channel_order=np.arange(1, maps.shape[-1] + 1),
            interpolation="Original 3D maps, trilinear at each child; full valid stencil; no extrapolation.")}

    b0_spread = np.ptp(b0, axis=1)
    b0_spread_map = np.full(m0.shape, np.nan)
    b0_spread_map.ravel()[parent_index] = b0_spread
    # Last quadrature dimension is thickness; inspect z variation separately.
    thickness_spread = np.ptp(maps.reshape(len(parent_index), *counts, maps.shape[-1]), axis=3)
    thickness_spread = thickness_spread.max(axis=(1, 2, 3))
    report = {"counts_row_column_thickness": list(counts), "children_per_parent": nchild,
        "requested_parent_count": int(support.sum()), "retained_parent_count": len(parent_index),
        "excluded_parent_count": int(support.sum() - len(parent_index)),
        "parents_missing_calibration": int((~coil_valid.all(axis=1)).sum()),
        "parents_missing_B0": int((~b0_valid.all(axis=1)).sum()),
        "subspin_count": int(b0.size), "slice_thickness_mm": thickness * 1000,
        "slice_normal_patient": normal.tolist(), "b0_mode": b0_mode,
        "native_B0_shape": None if native_shape is None else list(native_shape),
        "native_B0_spacing_X_Y_mm": None if native_spacing is None else (np.asarray(native_spacing) * 1000).tolist(),
        "B0_thickness_model": "Measured 2D B0 extruded through thickness, plus explicit synthetic gradient.",
        "synthetic_z_gradient_Hz_per_m": float(z_gradient_hz_per_m),
        "B0_spread_median_Hz": float(np.median(b0_spread)),
        "B0_spread_max_Hz": float(b0_spread.max()),
        "B0_spread_percentiles_Hz": np.percentile(b0_spread, [5, 50, 95]).tolist(),
        "B0_std_median_Hz": float(np.median(np.std(b0, axis=1))),
        "parents_with_B0_spread": int(np.count_nonzero(b0_spread > 1e-10)),
        "calibration_thickness_spread_median_Hz_per_A": float(np.median(thickness_spread)),
        "calibration_thickness_spread_max_Hz_per_A": float(thickness_spread.max()),
        "retained_amplitude_fraction": float(m0[parent_valid].sum() / requested_amplitude),
        "amplitude_model": "Measured GRE effective transverse amplitude divided equally between children; no additional normalization.",
        "missing_child_policy": "Exclude whole parent; never renormalize partial children.",
        "calibration_source_sha256": source_hash}
    expanded["subspins"]["report"] = report
    return {"expanded_experiment": expanded, "expanded_calibration": calibration,
        "parent_shape": m0.shape, "parent_flat_index": parent_index,
        "display_parent_index": np.repeat(parent_index[:, None], nchild, axis=1),
        "parent_valid_mask": parent_valid, "xyz_m": xyz, "offsets_m": offsets,
        "weights": weights, "subspin_b0": b0, "fieldbasis": maps,
        "b0_sample_spread_per_parent_Hz": b0_spread_map, "report": report}


__all__ = ["prepare_subspin_inputs"]
