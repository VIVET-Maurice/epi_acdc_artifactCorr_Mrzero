"""Native echo object, thickness spins and patient-registered ACDC responses.

Imaging-to-coil alignment transforms the native spin/ADC/output coordinates
into the patient coil frame. A second passive frame transforms their common
representation. Scalar image values stay attached to native array indices.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from b0_regularization import regularize_b0_local_plane
from inplane_geometry import inplane_matrix
from subspin_sampling import _load_source, _full_stencil_valid, _binary_mask
from prepare_acdc_calibration import interpolate_maps


def rotation_matrix(angle_deg):
    """Counterclockwise rotation in sequence X/Y; accept any finite angle."""
    return inplane_matrix(angle_deg)


def _slice_geometry(inputs):
    scan = inputs["scan_metadata"]
    raw = inputs["b0_raw"].get("metadata", {})
    position = np.asarray(scan["slice_position_mm"], dtype=float)
    thickness = float(scan["slice_thickness_mm"]) * 1e-3
    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("The EPI slice needs finite patient position and positive thickness.")
    expected = np.array([1., 0., 0., 1.]) / np.sqrt(2)
    for metadata in (scan, raw):
        if "slice_quaternion_wxyz" in metadata:
            quaternion = np.asarray(metadata["slice_quaternion_wxyz"], float)
            if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
                    or not np.isclose(np.linalg.norm(quaternion), 1., rtol=0, atol=1e-8)
                    or not np.allclose(abs(np.dot(quaternion, expected)), 1., rtol=0, atol=1e-8)):
                raise ValueError("Native echo shim preparation supports the validated axial slice frame only.")
    if "slice_position_mm" in raw and not np.isclose(raw["slice_position_mm"][2], position[2], rtol=0, atol=1e-3):
        raise ValueError("B0 and EPI measurements describe different slice planes.")
    return position, thickness


def prepare_echo_shim_inputs(inputs, fit, calibration_path, calibration_settings, *,
                             frame_angle_deg=0., alignment_angle_deg=180.,
                             alignment_mirror_x=False, alignment_mirror_y=False,
                             frame_mirror_x=False, frame_mirror_y=False,
                             alignment_center_m=(0., 0.), subspins_z=8,
                             rate_mode="pooled", b0_mode="regularized",
                             b0_sigma_pixels=1., quality_mask=None,
                             current_response=None):
    """Expand native object pixels into weighted thickness spins.

    Native 128x128 B0 and echo amplitudes are never reduced to a 64x64 map.
    Scalar fields remain attached to native object indices. Imaging-to-coil
    alignment Q transforms both their coordinates and ADC k. Patient queries
    are ``(-QY, QX, slice_Z + dz)``. Passive frame P transforms XY and k once
    more, giving combined matrix O=P@Q. The model must also apply O to its
    reconstruction output grid. Mirrors precede rotations: R@diag(mx,my).
    Only origin-centered transforms are supported, avoiding an unmodeled
    translation phase. Raw measured data are never reindexed/interpolated.

    ``quality_mask`` optionally restricts field-fit/field-limit points, never
    object support. By default every retained child is a field-limit point.
    Missing B0 or any missing ACDC child excludes the complete parent. No
    remaining signal is renormalized. In-plane pixel area relative to the EPI
    pixel area and thickness weights are applied exactly once to amplitude.
    """
    mirrors = (alignment_mirror_x, alignment_mirror_y, frame_mirror_x, frame_mirror_y)
    if not all(isinstance(value, (bool, np.bool_)) for value in mirrors):
        raise ValueError("Mirror controls must be booleans.")
    frame_rotation = inplane_matrix(frame_angle_deg, frame_mirror_x, frame_mirror_y)
    alignment_rotation = inplane_matrix(alignment_angle_deg, alignment_mirror_x, alignment_mirror_y)
    combined = frame_rotation @ alignment_rotation
    center = np.asarray(alignment_center_m, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("alignment_center_m must contain finite physical X/Y coordinates.")
    if np.any(center != 0):
        raise ValueError("Only origin-centered imaging-to-coil alignment is supported; a translated center needs explicit signal phase compensation.")
    if isinstance(subspins_z, (bool, np.bool_)) or int(subspins_z) != subspins_z or subspins_z < 1:
        raise ValueError("subspins_z must be a positive integer.")
    count = int(subspins_z)
    raw, field, acquisition = inputs["b0_raw"], inputs["field"], inputs["acquisition"]
    X, Y = (np.asarray(raw[key], dtype=float) for key in ("X_m", "Y_m"))
    if X.ndim != 2 or min(X.shape) < 2 or Y.shape != X.shape or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Native echo coordinates must be finite matching 2D grids.")
    coordinates = np.stack((X, Y), axis=-1)
    row_step, column_step = coordinates[1, 0] - coordinates[0, 0], coordinates[0, 1] - coordinates[0, 0]
    rows, columns = np.indices(X.shape)
    expected = coordinates[0, 0] + rows[..., None] * row_step + columns[..., None] * column_step
    object_area = abs(float(np.linalg.det(np.stack((column_step, row_step)))))
    if object_area <= 0 or not np.allclose(coordinates, expected, rtol=0, atol=1e-10):
        raise ValueError("Native coordinates must have a regular nondegenerate physical grid.")
    image_area = float(np.prod(acquisition["FOV_m"]) / np.asarray(acquisition["kx_mInv"]).size)
    if not np.isfinite(image_area) or image_area <= 0:
        raise ValueError("EPI FOV and trajectory must define positive pixel area.")
    spacing_mm = np.array([np.linalg.norm(row_step), np.linalg.norm(column_step)]) * 1000
    measured_b0 = np.asarray(field["b0Hz"], dtype=float)
    if measured_b0.shape != X.shape:
        raise ValueError("Measured B0 must be on the native echo grid.")
    b0_valid = _binary_mask(field["mask"], X.shape, "Native B0 mask") & np.isfinite(measured_b0)
    if b0_mode == "regularized":
        smoothing = regularize_b0_local_plane(measured_b0, b0_valid,
            sigma_pixels=b0_sigma_pixels, spacing_mm=spacing_mm)
        b0 = smoothing["b0Hz"]
        smoothing_report = smoothing["report"]
    elif b0_mode == "raw":
        b0 = measured_b0.copy()
        smoothing_report = {"method": "Raw native measured B0; no regularization."}
    else:
        raise ValueError("b0_mode must be 'regularized' or 'raw'.")
    if rate_mode in ("none", "pooled"):
        amplitude = np.asarray(fit["first_echo_amplitude"], dtype=float)
        accepted = bool(fit.get("pooled", {}).get("target_reliable_mask", False)) if rate_mode == "pooled" else None
        pooled_rate = float(fit["pooled"]["R2star_sInv"]) if rate_mode == "pooled" and accepted else 0.
        rate = np.full(X.shape, pooled_rate)
    elif rate_mode == "guarded":
        amplitude = np.asarray(fit["simulation_amplitude_at_reference"], dtype=float)
        rate = np.asarray(fit["simulation_R2star_sInv"], dtype=float)
        accepted = None
    else:
        raise ValueError("rate_mode must be 'none', 'pooled' or 'guarded'.")
    if amplitude.shape != X.shape or rate.shape != X.shape:
        raise ValueError("Echo amplitude and rates must match the native grid.")
    if np.any(rate < 0):
        raise ValueError("Effective echo decay rates must be nonnegative.")
    object_support = _binary_mask(fit.get("object_support", fit["simulation_support"]), X.shape, "Object support")
    baseline_support = _binary_mask(fit["simulation_support"], X.shape, "Echo simulation support")
    usable_amplitude = np.isfinite(amplitude) & (amplitude > 0)
    requested = object_support & usable_amplitude
    covered = requested & baseline_support & b0_valid & np.isfinite(rate)
    if not covered.any():
        raise ValueError("No native object pixels have reliable B0 and finite amplitude/rate.")
    quality = np.ones(X.shape, dtype=bool) if quality_mask is None else _binary_mask(quality_mask, X.shape, "Field quality mask")
    position_mm, thickness = _slice_geometry(inputs)
    dz = ((np.arange(count) + .5) / count - .5) * thickness
    physical_xy = (coordinates - center) @ alignment_rotation.T + center
    frame_xy = physical_xy @ frame_rotation.T
    parent_index = np.flatnonzero(covered)
    physical_points = physical_xy.reshape(-1, 2)[parent_index]
    xyz = np.empty((len(parent_index), count, 3))
    xyz[..., 0] = -physical_points[:, 1, None]
    xyz[..., 1] = physical_points[:, 0, None]
    xyz[..., 2] = position_mm[2] * 1e-3 + dz

    settings = dict(calibration_settings)
    if settings.get("units") != "Hz/A" or settings.get("registration_confirmed") is not True:
        raise ValueError("Native ACDC calibration requires confirmed registration and explicit Hz/A units.")
    affine = np.asarray(settings.get("patient_to_source_m"), float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or abs(np.linalg.det(affine[:3, :3])) < 1e-10):
        raise ValueError("patient_to_source_m must be an invertible affine in metres.")
    path = Path(calibration_path).resolve()
    stat = path.stat()
    source, axes, source_valid, digest = _load_source(str(path), stat.st_mtime_ns, stat.st_size)
    if settings.get("source_sha256") not in (None, digest):
        raise ValueError("Calibration differs from its confirmed source_sha256.")
    query = xyz @ affine[:3, :3].T + affine[:3, 3]
    maps, valid_children = interpolate_maps(source, axes, query)
    valid_children &= _full_stencil_valid(axes, source_valid, query)
    keep = valid_children.all(axis=1)
    parent_valid = np.zeros(X.shape, dtype=bool)
    parent_valid.ravel()[parent_index[keep]] = True
    if not parent_valid.any() or not np.any(parent_valid & quality):
        raise ValueError("No complete ACDC-covered parents with field-limit points remain.")
    parent_index, xyz, maps = parent_index[keep], xyz[keep], maps[keep]
    shape = (len(parent_index), count)
    weights = np.full(shape, 1. / count)
    area_ratio = object_area / image_area
    amplitudes = amplitude.ravel()[parent_index, None] * area_ratio * weights
    b0_children = np.repeat(b0.ravel()[parent_index, None], count, axis=1)
    rates = np.repeat(rate.ravel()[parent_index, None], count, axis=1)
    t2 = np.divide(1., rates, out=np.full(shape, np.inf), where=rates > 0)
    frame_points = frame_xy.reshape(-1, 2)[parent_index]
    child_X = np.repeat(frame_points[:, 0, None], count, axis=1)
    child_Y = np.repeat(frame_points[:, 1, None], count, axis=1)
    child_quality = np.repeat(quality.ravel()[parent_index, None], count, axis=1)
    framed_acquisition = deepcopy(acquisition)
    k = np.stack((acquisition["kx_mInv"], acquisition["ky_mInv"]), axis=-1) @ combined.T
    framed_acquisition.update(kx_mInv=k[..., 0], ky_mInv=k[..., 1],
        frame_angle_deg=float(frame_angle_deg) % 360., combined_frame_matrix=combined.copy())
    anchor = float(fit["amplitude_reference_time_s"])
    if not np.isfinite(anchor) or anchor < 0:
        raise ValueError("The echo amplitude time anchor must be finite nonnegative seconds.")
    metadata = {"parent_shape": X.shape, "parent_flat_index": parent_index,
        "parent_valid_mask": parent_valid, "weights": weights, "counts": (1, 1, count),
        "xyz_patient_m": xyz, "frame_angle_deg": float(frame_angle_deg) % 360.,
        "alignment_angle_deg": float(alignment_angle_deg) % 360.,
        "combined_frame_matrix": combined.copy(),
        "through_plane_B0": "Native measured 2D B0 extruded; no invented through-plane B0 gradient."}
    response = {"enabled": True, "tau_s": 133e-6 / np.log(100)} if current_response is None else deepcopy(current_response)
    experiment = {"m0_proxy": amplitudes, "acquisition": framed_acquisition,
        "geometry": {"X_m": child_X, "Y_m": child_Y,
            "Z_m": np.broadcast_to(dz, shape).copy(), "slice_thickness_mm": thickness * 1000,
            "spacing_mm": spacing_mm, "position_mm": position_mm.copy(),
            "orientation": np.array([0., 1., 0., -1., 0., 0.]),
            "coordinate_system": "Passive sequence XY; patient positions stored separately in subspins."},
        "field": {"b0Hz": b0_children, "measured_b0Hz": b0_children.copy(),
            "simulation_mask": np.ones(shape, dtype=bool), "mask": child_quality},
        "relaxation": {"T2star_s": t2, "amplitude_reference_time_s": anchor},
        "amplitude_reference_time_s": anchor, "frame_angle_deg": float(frame_angle_deg) % 360.,
        "combined_frame_matrix": combined.copy(),
        "current_response": response, "subspins": metadata,
        "provenance": {"source_sha256": deepcopy(inputs.get("source_sha256", {})),
            "object_model": "Native GRE echo amplitude, effective residual R2*, native measured B0.",
            "alignment_interpretation": "Imaging-to-coil registration applied jointly to spin, ADC and output coordinates.",
            "passive_frame_changes_physical_placement": False},
        "assumptions": ["GRE receive-weighted amplitude and saturation are not PD/T1 corrected.",
            "Native-voxel R2* includes unresolved dephasing; it is not intrinsic T2.",
            "Through-plane object amplitude and B0 are uniform; 3D ACDC maps vary at each thickness point."],
        "optimization_ready": True}
    calibration = {"acdc_maps": maps, "X_m": child_X.copy(), "Y_m": child_Y.copy(),
        "units": "Hz/A", "spatial_valid_mask": np.ones(shape, dtype=bool),
        "registration_confirmed": True, "metadata": dict(settings, source_file=str(path),
            source_sha256=digest, source_channel_order=np.arange(1, maps.shape[-1] + 1))}
    requested_mass = float(amplitude[requested].sum())
    baseline_mass = float(amplitude[covered].sum())
    retained_mass = float(amplitude[parent_valid].sum())
    report = {"native_shape": list(X.shape), "children_per_parent": count,
        "requested_parent_count": int(requested.sum()), "baseline_B0_parent_count": int(covered.sum()),
        "retained_parent_count": len(parent_index), "subspin_count": int(amplitudes.size),
        "excluded_missing_B0_parent_count": int((requested & ~covered).sum()),
        "excluded_missing_ACDC_parent_count": int(covered.sum() - len(parent_index)),
        "excluded_missing_B0_amplitude_fraction": 1. - baseline_mass / requested_mass,
        "excluded_ACDC_amplitude_fraction_of_baseline": 1. - retained_mass / baseline_mass,
        "retained_amplitude_fraction_of_requested": retained_mass / requested_mass,
        "native_pixel_area_m2": object_area, "EPI_pixel_area_m2": image_area,
        "amplitude_area_ratio": area_ratio, "amplitude_reference_time_s": anchor,
        "rate_mode": rate_mode, "pooled_rate_accepted": accepted,
        "rate_min_sInv": float(rates.min()), "rate_max_sInv": float(rates.max()),
        "B0_mode": b0_mode, "B0_regularization": smoothing_report,
        "slice_thickness_mm": thickness * 1000, "slice_center_patient_mm": position_mm.tolist(),
        "frame_angle_deg": float(frame_angle_deg) % 360.,
        "alignment_angle_deg": float(alignment_angle_deg) % 360., "alignment_center_sequence_m": center.tolist(),
        "alignment_mirror_x": bool(alignment_mirror_x), "alignment_mirror_y": bool(alignment_mirror_y),
        "frame_mirror_x": bool(frame_mirror_x), "frame_mirror_y": bool(frame_mirror_y),
        "alignment_matrix": alignment_rotation.tolist(), "passive_frame_matrix": frame_rotation.tolist(),
        "combined_frame_matrix": combined.tolist(),
        "field_limit_children": int(child_quality.sum()), "all_children_in_field_limits": bool(child_quality.all()),
        "quality_mask_affects_simulation_support": False,
        "missing_calibration_policy": "Exclude complete parent; never fill unknown ACDC response or renormalize retained amplitude.",
        "calibration_source_sha256": digest}
    metadata["report"] = report
    return {"expanded_experiment": experiment, "expanded_calibration": calibration,
        "parent_shape": X.shape, "parent_flat_index": parent_index, "parent_valid_mask": parent_valid,
        "baseline_support_mask": covered, "weights": weights, "xyz_m": xyz,
        "subspin_b0": b0_children, "fieldbasis": maps, "native_amplitude": amplitude.copy(),
        "native_rate_sInv": rate.copy(), "native_B0_Hz": b0.copy(),
        "native_X_m": X.copy(), "native_Y_m": Y.copy(),
        "physical_X_m": physical_xy[..., 0], "physical_Y_m": physical_xy[..., 1],
        "frame_X_m": frame_xy[..., 0], "frame_Y_m": frame_xy[..., 1],
        "combined_frame_matrix": combined,
        "acquisition_native": deepcopy(acquisition), "report": report}


__all__ = ["prepare_echo_shim_inputs", "rotation_matrix"]
