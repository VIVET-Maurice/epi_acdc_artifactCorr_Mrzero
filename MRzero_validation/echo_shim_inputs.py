"""Native echo object, thickness spins and patient-registered ACDC responses.

An optional fixed reference expresses imaging-to-coil alignment once in 3D.
Q and a second passive frame define the common Fourier representation;
scalar image values stay attached to native array indices.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from b0_regularization import regularize_b0_local_plane
from inplane_geometry import inplane_matrix
from subspin_sampling import _load_source, _full_stencil_valid, _binary_mask
from prepare_acdc_calibration import interpolate_maps
from slice_geometry import encoding_origin, slice_basis, slice_center


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
    basis = slice_basis(scan)
    if ("slice_quaternion_wxyz" in raw
            and not np.allclose(slice_basis(raw), basis, rtol=0, atol=1e-6)):
        raise ValueError("B0 and EPI measurements describe different slice orientations.")
    if "slice_position_mm" in raw:
        raw_position = np.asarray(raw["slice_position_mm"], dtype=float)
        if (raw_position.shape != (3,) or not np.isfinite(raw_position).all()
                or not np.allclose(raw_position, position, rtol=0, atol=1e-3)):
            raise ValueError("B0 and EPI measurements describe different slice planes/origins.")
    if ("slice_thickness_mm" in raw
            and not np.isclose(float(raw["slice_thickness_mm"]) * 1e-3,
                               thickness, rtol=0, atol=1e-6)):
        raise ValueError("B0 and EPI measurements describe different slice thicknesses.")
    return position, thickness, basis


def _calibration_alignment(position_mm, basis, planar_alignment, reference_pose):
    """Map native local vectors to aligned patient coordinates for coil queries.

    An explicit reference freezes the alignment in the scanner/patient frame.
    Without one, retain the historical rotation about each current slice.
    This identifies a reference convention, not a measured hardware affine.
    """
    alignment_3d = np.eye(3)
    alignment_3d[:2, :2] = planar_alignment
    if reference_pose is None:
        reference_basis = basis
        reference_center = position_mm * 1e-3
        reference = None
        scope = "slice_local"
    else:
        if (not isinstance(reference_pose, dict)
                or "slice_position_mm" not in reference_pose
                or "slice_quaternion_wxyz" not in reference_pose):
            raise ValueError("alignment_reference_pose needs slice_position_mm and slice_quaternion_wxyz.")
        reference_basis = slice_basis(reference_pose)
        reference_center = slice_center(reference_pose) * 1e-3
        reference = dict(slice_position_mm=(reference_center * 1000).tolist(),
                         slice_quaternion_wxyz=np.asarray(reference_pose["slice_quaternion_wxyz"], float).tolist())
        scope = "scanner_fixed"
    patient_alignment = np.eye(4)
    patient_alignment[:3, :3] = reference_basis @ alignment_3d @ reference_basis.T
    patient_alignment[:3, 3] = reference_center - patient_alignment[:3, :3] @ reference_center
    native_to_patient = np.eye(4)
    if reference_pose is None:
        # Keep the historical operation order for legacy notebooks.
        native_to_patient[:3, :3] = basis @ alignment_3d
        native_to_patient[:3, 3] = position_mm * 1e-3
    else:
        native_to_patient[:3, :3] = patient_alignment[:3, :3] @ basis
        native_to_patient[:3, 3] = patient_alignment[:3, :3] @ (position_mm * 1e-3) + patient_alignment[:3, 3]
    return native_to_patient, patient_alignment, reference, scope


def prepare_echo_shim_inputs(inputs, fit, calibration_path, calibration_settings, *,
                             frame_angle_deg=0., alignment_angle_deg=180.,
                             alignment_mirror_x=False, alignment_mirror_y=False,
                             frame_mirror_x=False, frame_mirror_y=False,
                             alignment_center_m=(0., 0.), alignment_reference_pose=None,
                             subspins_z=8,
                             rate_mode="pooled", b0_mode="regularized",
                             b0_sigma_pixels=1., quality_mask=None,
                             current_response=None, b0_weighting="uniform",
                             inplane_voxel=False, baseline_calibration=None):
    """Expand native object pixels into weighted thickness spins.

    Native 128x128 B0 and echo amplitudes are never reduced to a 64x64 map.
    Scalar fields remain attached to native object indices. Imaging-to-coil
    alignment Q transforms native coordinates relative to the measured slice
    center and ADC k. Legacy patient queries are ``slice_center + R @ (Qx, Qy, dz)``,
    where R is the measured native readout/phase/normal basis in patient space.
    Here x/y are local vectors, while the native X_m/Y_m grids retain their
    absolute header coordinates. Passive frame P transforms local XY and k
    once more, giving combined matrix O=P@Q. The model must also apply O to
    its centered reconstruction output grid. Mirrors precede rotations:
    R@diag(mx,my). The measured slice displacement never rotates around the
    scanner isocenter. The acquisition's receiver/Fourier origin is handled by
    centered model coordinates, without inventing a translation phase. The
    optional alignment_center_m remains an additional registration transform
    and must be zero. Raw measured data are never reindexed/interpolated.

    With ``alignment_reference_pose`` containing fixed ``slice_position_mm``
    and ``slice_quaternion_wxyz``, Q defines a single scanner-fixed alignment:
    G(p) = c_ref + R_ref @ Q_3D @ R_ref.T @ (p-c_ref). Each measured point
    p = c_slice + R_slice @ (x,y,dz), its edges and thickness normal are mapped
    by this same G, then by patient_to_source_m. The reference pose must remain
    fixed when changing the acquisition. It reproduces the legacy placement
    on the reference slice and avoids moving the coil frame with other slices.
    Q and P still define the common passive Fourier representation O=P@Q.

    ``quality_mask`` optionally restricts field-fit/field-limit points, never
    object support. By default every retained child is a field-limit point.
    Missing B0 or any missing ACDC child excludes the complete parent. No
    remaining signal is renormalized. In-plane pixel area relative to the EPI
    pixel area and thickness weights are applied exactly once to amplitude.

    ``b0_weighting='echo_power'`` weights local B0 planes by the magnitude of
    adjacent-echo complex coil cross products from the B0 echoes only.
    ``inplane_voxel=True`` supplies an analytic, locally linear finite-pixel
    model without multiplying the number of MRzero points. Native B0 slopes
    must be identifiable. ACDC gradients are measured over each full pixel;
    both gradient endpoints and all four voxel corners require valid 3D
    calibration stencils. Unknown B0 or calibration is never extrapolated.

    Optional ``baseline_calibration`` contains three finite real scalars:
    ``frequency_offset_Hz``, ``readout_shift_mm`` and ``odd_even_phase_rad``
    (missing values default to zero). The shift moves native X coordinates
    before alignment and all 3D coil queries. The frequency offset is added
    after B0/gradient estimation. The residual odd/even phase is applied to
    the ADC phase after the measured navigator correction, with convention
    -0.5 * readout_polarity * odd_even_phase_rad. It remains in the zero-B0
    reference; the frequency offset is removed with B0. No input is mutated.

    For spiral acquisitions, ``phase_shift_mm`` replaces ``odd_even_phase_rad``:
    it translates native Y before alignment and 3D coil queries. No EPI
    navigator or odd/even phase correction is applied to a spiral.

    ``rate_mode='complex'`` uses ``fit['multi_echo']``: exponential envelope,
    residual R2, refined B0, selected unresolved frequency dispersion and the
    exact fixed in-plane B0 slopes used during the complex fit. It requires
    ``inplane_voxel=True`` and never smooths those slopes again. Unresolved
    dispersion remains current independent and remains in the zero-B0 target.
    """
    baseline_parameters = None
    if baseline_calibration is not None:
        if not isinstance(baseline_calibration, dict):
            raise ValueError("baseline_calibration must be None or a dictionary of scalar parameters.")
        names = ("frequency_offset_Hz", "readout_shift_mm", "odd_even_phase_rad")
        if inputs["acquisition"].get("sequence_kind") == "spiral":
            names = ("frequency_offset_Hz", "readout_shift_mm", "phase_shift_mm")
        unknown = set(baseline_calibration) - set(names)
        if unknown:
            raise ValueError(f"Unknown baseline_calibration parameters: {sorted(unknown)}.")
        baseline_parameters = {}
        for name in names:
            value = baseline_calibration.get(name, 0.)
            scalar = np.asarray(value)
            if (isinstance(value, (bool, np.bool_)) or scalar.ndim != 0
                    or np.iscomplexobj(scalar) or not np.issubdtype(scalar.dtype, np.number)
                    or not np.isfinite(scalar)):
                raise ValueError(f"baseline_calibration.{name} must be a finite real scalar.")
            baseline_parameters[name] = float(scalar)
    mirrors = (alignment_mirror_x, alignment_mirror_y, frame_mirror_x, frame_mirror_y)
    if not all(isinstance(value, (bool, np.bool_)) for value in mirrors):
        raise ValueError("Mirror controls must be booleans.")
    if not isinstance(inplane_voxel, (bool, np.bool_)):
        raise ValueError("inplane_voxel must be boolean.")
    if b0_weighting not in ("uniform", "echo_power"):
        raise ValueError("b0_weighting must be 'uniform' or 'echo_power'.")
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
    position_mm, thickness, patient_basis = _slice_geometry(inputs)
    native_to_patient, patient_alignment, alignment_reference, alignment_scope = _calibration_alignment(
        position_mm, patient_basis, alignment_rotation, alignment_reference_pose)
    native_origin = encoding_origin(inputs["scan_metadata"])
    original_X, original_Y = (np.asarray(raw[key], dtype=float) for key in ("X_m", "Y_m"))
    X, Y = original_X, original_Y
    if baseline_parameters is not None and baseline_parameters["readout_shift_mm"] != 0:
        X = X + baseline_parameters["readout_shift_mm"] * 1e-3
    if baseline_parameters is not None and baseline_parameters.get("phase_shift_mm", 0.) != 0:
        Y = Y + baseline_parameters["phase_shift_mm"] * 1e-3
    if X.ndim != 2 or min(X.shape) < 2 or Y.shape != X.shape or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Native echo coordinates must be finite matching 2D grids.")
    coordinates = np.stack((X, Y), axis=-1)
    # A rigid origin shift cannot change pixel area, quadrature weights or B0
    # gradient geometry; derive edges before adding its floating-point offset.
    original_coordinates = np.stack((original_X, original_Y), axis=-1)
    row_step = original_coordinates[1, 0] - original_coordinates[0, 0]
    column_step = original_coordinates[0, 1] - original_coordinates[0, 0]
    rows, columns = np.indices(X.shape)
    expected = coordinates[0, 0] + rows[..., None] * row_step + columns[..., None] * column_step
    object_area = abs(float(np.linalg.det(np.stack((column_step, row_step)))))
    if object_area <= 0 or not np.allclose(coordinates, expected, rtol=0, atol=1e-10):
        raise ValueError("Native coordinates must have a regular nondegenerate physical grid.")
    image_shape = acquisition.get("image_shape", np.asarray(acquisition["kx_mInv"]).shape)
    image_area = float(np.prod(acquisition["FOV_m"]) / np.prod(image_shape))
    if not np.isfinite(image_area) or image_area <= 0:
        raise ValueError("Imaging FOV and output matrix must define positive pixel area.")
    spacing_mm = np.array([np.linalg.norm(row_step), np.linalg.norm(column_step)]) * 1000
    measured_b0 = np.asarray(field["b0Hz"], dtype=float)
    if measured_b0.shape != X.shape:
        raise ValueError("Measured B0 must be on the native echo grid.")
    b0_valid = _binary_mask(field["mask"], X.shape, "Native B0 mask") & np.isfinite(measured_b0)
    complex_fit = None
    if rate_mode == "complex":
        if not inplane_voxel:
            raise ValueError("rate_mode='complex' requires inplane_voxel=True to retain the gradients used during fitting.")
        complex_fit = fit.get("multi_echo")
        if not isinstance(complex_fit, dict):
            raise ValueError("rate_mode='complex' requires fit['multi_echo'].")
        complex_maps = {}
        for key, expected_shape in (
                ("amplitude_envelope_at_reference", X.shape),
                ("residual_R2_sInv", X.shape),
                ("refined_b0_Hz", X.shape),
                ("unresolved_width_Hz", X.shape),
                ("dispersion_kind", X.shape),
                ("known_inplane_gradient_Hz_per_pixel", X.shape + (2,))):
            value = np.asarray(complex_fit.get(key))
            if (value.shape != expected_shape or np.iscomplexobj(value)
                    or not np.issubdtype(value.dtype, np.number)):
                raise ValueError(f"multi_echo.{key} must be a real numeric array with shape {expected_shape}.")
            complex_maps[key] = value.astype(float, copy=True)
        complex_support = _binary_mask(complex_fit.get("support_mask", fit["simulation_support"]),
                                       X.shape, "Complex-fit support mask")
        kinds = complex_maps["dispersion_kind"]
        if (not np.isin(kinds, (-1, 0, 1, 2)).all()
                or not np.isin(kinds[complex_support], (0, 1, 2)).all()):
            raise ValueError("multi_echo.dispersion_kind must contain 0, 1 or 2 on support; -1 is allowed only outside it.")
        complex_maps["dispersion_kind"] = np.where(complex_support, kinds, 0)
        width = complex_maps["unresolved_width_Hz"]
        if not np.isfinite(width).all() or np.any(width < 0):
            raise ValueError("multi_echo.unresolved_width_Hz must be finite nonnegative Hz.")
        b0_valid &= complex_support & np.isfinite(complex_maps["refined_b0_Hz"])
    b0_precision = None
    weighting_echo_count = None
    if b0_weighting == "echo_power":
        coils = np.asarray(raw.get("coil_images"))
        if (coils.ndim != 4 or coils.shape[-2:] != X.shape or not np.iscomplexobj(coils)
                or coils.shape[0] < 2 or coils.shape[1] < 1 or not np.isfinite(coils).all()):
            raise ValueError("echo_power weighting requires finite complex native coil_images with at least two echoes.")
        weighting_echo_count = inputs.get("field_echo_count", field.get("metadata", {}).get("field_echo_count", coils.shape[0]))
        if (isinstance(weighting_echo_count, bool) or int(weighting_echo_count) != weighting_echo_count
                or not 2 <= weighting_echo_count <= coils.shape[0]):
            raise ValueError("field_echo_count must identify the B0 echoes used for echo_power weighting.")
        weighting_echo_count = int(weighting_echo_count)
        products = np.sum(coils[1:weighting_echo_count] * coils[:weighting_echo_count - 1].conj(), axis=1)
        b0_precision = np.mean(np.abs(products), axis=0)
    smoothing = None
    if complex_fit is not None:
        gradients = complex_maps["known_inplane_gradient_Hz_per_pixel"]
        gradient_valid = np.isfinite(gradients).all(axis=-1)
        if "known_inplane_gradient_valid_mask" in complex_fit:
            gradient_valid &= _binary_mask(complex_fit["known_inplane_gradient_valid_mask"],
                                           X.shape, "Complex-fit known in-plane gradient mask")
        smoothing = {"b0Hz": complex_maps["refined_b0_Hz"],
            "gradient_Hz_per_pixel": gradients,
            "gradient_valid_mask": gradient_valid,
            "report": {"method": "Fixed known in-plane B0 gradients and refined center B0 from complex multi-echo fit; no second smoothing.",
                "gradient_source": "fit.multi_echo.known_inplane_gradient_Hz_per_pixel",
                "weighting_source": b0_weighting, "weighting_echo_count": weighting_echo_count,
                "weighting_uses_EPI": False, "requested_B0_mode": b0_mode}}
    elif b0_mode == "regularized" or inplane_voxel:
        smoothing = regularize_b0_local_plane(measured_b0, b0_valid,
            sigma_pixels=b0_sigma_pixels, spacing_mm=spacing_mm, weights=b0_precision)
        smoothing["report"].update(weighting_source=b0_weighting,
            weighting_echo_count=weighting_echo_count,
            weighting_interpretation="Adjacent-echo cross-product magnitude is a signal-based relative phase-precision proxy, not a calibrated phase variance." if b0_precision is not None else "Uniform measured samples.",
            weighting_uses_EPI=False)
    if b0_mode not in ("regularized", "raw"):
        raise ValueError("b0_mode must be 'regularized' or 'raw'.")
    if complex_fit is not None:
        b0 = complex_maps["refined_b0_Hz"].copy()
        smoothing_report = smoothing["report"]
    elif b0_mode == "regularized":
        b0 = smoothing["b0Hz"]
        smoothing_report = smoothing["report"]
    elif b0_mode == "raw":
        b0 = measured_b0.copy()
        smoothing_report = {"method": "Raw native measured B0; no regularization.",
            "gradient_estimator": None if smoothing is None else smoothing["report"]}
    if baseline_parameters is not None and baseline_parameters["frequency_offset_Hz"] != 0:
        b0 = b0 + baseline_parameters["frequency_offset_Hz"]
    if rate_mode in ("none", "pooled"):
        amplitude = np.asarray(fit["first_echo_amplitude"], dtype=float)
        accepted = bool(fit.get("pooled", {}).get("target_reliable_mask", False)) if rate_mode == "pooled" else None
        pooled_rate = float(fit["pooled"]["R2star_sInv"]) if rate_mode == "pooled" and accepted else 0.
        rate = np.full(X.shape, pooled_rate)
    elif rate_mode == "guarded":
        amplitude = np.asarray(fit["simulation_amplitude_at_reference"], dtype=float)
        rate = np.asarray(fit["simulation_R2star_sInv"], dtype=float)
        accepted = None
    elif rate_mode == "complex":
        amplitude = complex_maps["amplitude_envelope_at_reference"]
        rate = complex_maps["residual_R2_sInv"]
        accepted = None
    else:
        raise ValueError("rate_mode must be 'none', 'pooled', 'guarded' or 'complex'.")
    if amplitude.shape != X.shape or rate.shape != X.shape:
        raise ValueError("Echo amplitude and rates must match the native grid.")
    if np.any(rate < 0):
        raise ValueError("Effective echo decay rates must be nonnegative.")
    object_support = _binary_mask(fit.get("object_support", fit["simulation_support"]), X.shape, "Object support")
    baseline_support = _binary_mask(fit["simulation_support"], X.shape, "Echo simulation support")
    usable_amplitude = np.isfinite(amplitude) & (amplitude > 0)
    requested = object_support & usable_amplitude
    baseline_covered = requested & baseline_support & b0_valid & np.isfinite(rate)
    covered = baseline_covered.copy()
    if inplane_voxel:
        covered &= smoothing["gradient_valid_mask"]
    if not covered.any():
        raise ValueError("No native object pixels have reliable B0 and finite amplitude/rate.")
    quality = np.ones(X.shape, dtype=bool) if quality_mask is None else _binary_mask(quality_mask, X.shape, "Field quality mask")
    dz = ((np.arange(count) + .5) / count - .5) * thickness
    local_coordinates = coordinates - native_origin
    aligned_local = local_coordinates @ alignment_rotation.T
    physical_xy = aligned_local + native_origin
    frame_xy = local_coordinates @ combined.T
    parent_index = np.flatnonzero(covered)
    physical_points = local_coordinates.reshape(-1, 2)[parent_index]
    local_xyz = np.empty((len(parent_index), count, 3))
    local_xyz[..., :2] = physical_points[:, None, :]
    local_xyz[..., 2] = dz
    xyz_patient_native = local_xyz @ patient_basis.T + position_mm * 1e-3
    xyz = local_xyz @ native_to_patient[:3, :3].T + native_to_patient[:3, 3]

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
    # Preserve the complete physical and local maps in every optimization.
    native_to_source = affine @ native_to_patient
    effective_patient_to_source = affine @ patient_alignment
    query = xyz @ affine[:3, :3].T + affine[:3, 3]
    maps, valid_children = interpolate_maps(source, axes, query)
    valid_children &= _full_stencil_valid(axes, source_valid, query)
    center_keep = valid_children.all(axis=1)
    field_gradient = None
    corner_linearization_error = None
    if inplane_voxel:
        # Edges follow the native row/column axes through physical alignment;
        # the passive frame affects encoding vectors, never patient queries.
        native_edges = np.stack((row_step, column_step))
        patient_edges = np.column_stack((native_edges, np.zeros(2))) @ native_to_patient[:3, :3].T
        endpoints = np.stack((-.5 * patient_edges[0], .5 * patient_edges[0],
                              -.5 * patient_edges[1], .5 * patient_edges[1]))
        endpoint_xyz = xyz[..., None, :] + endpoints
        endpoint_query = (endpoint_xyz @ affine[:3, :3].T + affine[:3, 3]).reshape(len(parent_index), count * 4, 3)
        endpoint_maps, endpoint_valid = interpolate_maps(source, axes, endpoint_query)
        endpoint_valid &= _full_stencil_valid(axes, source_valid, endpoint_query)
        endpoint_maps = endpoint_maps.reshape(len(parent_index), count, 4, maps.shape[-1])
        field_gradient = np.stack((endpoint_maps[..., 1, :] - endpoint_maps[..., 0, :],
                                   endpoint_maps[..., 3, :] - endpoint_maps[..., 2, :]), axis=-2)
        valid_children &= endpoint_valid.reshape(len(parent_index), count, 4).all(axis=-1)
        corners = np.asarray([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
        corner_xyz = xyz[..., None, :] + corners @ patient_edges
        corner_query = (corner_xyz @ affine[:3, :3].T + affine[:3, 3]).reshape(len(parent_index), count * 4, 3)
        corner_maps, corner_valid = interpolate_maps(source, axes, corner_query)
        corner_valid &= _full_stencil_valid(axes, source_valid, corner_query)
        corner_maps = corner_maps.reshape(len(parent_index), count, 4, maps.shape[-1])
        valid_children &= corner_valid.reshape(len(parent_index), count, 4).all(axis=-1)
        predicted_corners = maps[..., None, :] + np.einsum("va,pzac->pzvc", corners, field_gradient)
        corner_linearization_error = np.abs(predicted_corners - corner_maps)
    keep = valid_children.all(axis=1)
    parent_valid = np.zeros(X.shape, dtype=bool)
    parent_valid.ravel()[parent_index[keep]] = True
    if not parent_valid.any() or not np.any(parent_valid & quality):
        raise ValueError("No complete ACDC-covered parents with field-limit points remain.")
    parent_index, xyz, maps = parent_index[keep], xyz[keep], maps[keep]
    xyz_patient_native = xyz_patient_native[keep]
    retained_query = query[keep]
    if inplane_voxel:
        field_gradient = field_gradient[keep]
        corner_linearization_error = corner_linearization_error[keep]
        corner_maps = corner_maps[keep]
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
    native_acquisition = deepcopy(acquisition)
    native_acquisition["encoding_origin_m"] = native_origin.copy()
    if baseline_parameters is not None and baseline_parameters.get("odd_even_phase_rad", 0.) != 0:
        times = np.asarray(acquisition.get("time_s"), dtype=float)
        sample_shape = np.asarray(acquisition["kx_mInv"]).shape
        if (times.ndim != 2 or times.shape != sample_shape or times.shape[1] < 2
                or not np.isfinite(times).all()):
            raise ValueError("Residual odd/even phase requires finite 2D ADC times matching k-space.")
        polarity = np.sign(times[:, -1] - times[:, 0])
        if (not np.isin(polarity, (-1, 1)).all()
                or not np.all(np.sign(np.diff(times, axis=1)) == polarity[:, None])):
            raise ValueError("Residual odd/even phase requires monotone ADC time rows with polarity +/-1.")
        nominal_phase = np.asarray(acquisition.get("phase_offset_rad", np.zeros(sample_shape)), dtype=float)
        if nominal_phase.shape != sample_shape or not np.isfinite(nominal_phase).all():
            raise ValueError("Nominal phase_offset_rad must be finite and match k-space.")
        native_acquisition["phase_offset_rad"] = (nominal_phase
            - .5 * polarity[:, None] * baseline_parameters["odd_even_phase_rad"])
    framed_acquisition = deepcopy(native_acquisition)
    k = np.stack((acquisition["kx_mInv"], acquisition["ky_mInv"]), axis=-1) @ combined.T
    framed_acquisition.update(kx_mInv=k[..., 0], ky_mInv=k[..., 1],
        frame_angle_deg=float(frame_angle_deg) % 360., combined_frame_matrix=combined.copy(),
        encoding_origin_m=np.zeros(2), native_frame_origin_m=native_origin.copy())
    anchor = float(fit["amplitude_reference_time_s"])
    if not np.isfinite(anchor) or anchor < 0:
        raise ValueError("The echo amplitude time anchor must be finite nonnegative seconds.")
    metadata = {"parent_shape": X.shape, "parent_flat_index": parent_index,
        "parent_valid_mask": parent_valid, "weights": weights, "counts": (1, 1, count),
        "xyz_patient_m": xyz, "frame_angle_deg": float(frame_angle_deg) % 360.,
        "alignment_angle_deg": float(alignment_angle_deg) % 360.,
        "combined_frame_matrix": combined.copy(),
        "native_frame_origin_m": native_origin.copy(),
        "native_to_patient_basis": patient_basis.copy(),
        "xyz_patient_native_m": xyz_patient_native,
        "patient_alignment_m": patient_alignment.copy(),
        "native_local_to_aligned_patient_m": native_to_patient.copy(),
        "native_local_to_source_m": native_to_source.copy(),
        "through_plane_B0": "Native measured 2D B0 extruded; no invented through-plane B0 gradient."}
    response = {"enabled": True, "tau_s": 133e-6 / np.log(100)} if current_response is None else deepcopy(current_response)
    experiment = {"m0_proxy": amplitudes, "acquisition": framed_acquisition,
        "geometry": {"X_m": child_X, "Y_m": child_Y,
            "Z_m": np.broadcast_to(dz, shape).copy(), "slice_thickness_mm": thickness * 1000,
            "spacing_mm": spacing_mm, "position_mm": position_mm.copy(),
            "orientation": patient_basis[:, :2].T.ravel().copy(),
            "coordinate_system": "Slice-centered passive sequence XY; absolute patient positions stored separately in subspins."},
        "field": {"b0Hz": b0_children, "measured_b0Hz": (b0_children.copy()
            if baseline_parameters is None else np.repeat(measured_b0.ravel()[parent_index, None], count, axis=1)),
            "simulation_mask": np.ones(shape, dtype=bool), "mask": child_quality},
        "relaxation": {"T2star_s": t2, "amplitude_reference_time_s": anchor},
        "amplitude_reference_time_s": anchor, "frame_angle_deg": float(frame_angle_deg) % 360.,
        "combined_frame_matrix": combined.copy(),
        "current_response": response, "subspins": metadata,
        "provenance": {"source_sha256": deepcopy(inputs.get("source_sha256", {})),
            "baseline_calibration": deepcopy(baseline_parameters),
            "object_model": "Native GRE echo amplitude, effective residual R2*, native measured B0.",
            "alignment_interpretation": ("Scanner-fixed 3D map registration anchored to the declared reference pose; Q/P also define a joint passive Fourier representation."
                if alignment_scope == "scanner_fixed" else
                "Slice-local imaging-to-coil alignment; spin, ADC and output coordinates share a common representation."),
            "slice_origin_interpretation": ("Measured patient slice is mapped by a fixed 3D calibration alignment; Fourier coordinates remain relative to its receiver origin."
                if alignment_scope == "scanner_fixed" else
                "Measured patient slice center is added after local imaging-to-coil alignment; Fourier coordinates are relative to the receiver origin."),
            "passive_frame_changes_physical_placement": False},
        "assumptions": ["GRE receive-weighted amplitude and saturation are not PD/T1 corrected.",
            "Native-voxel R2* includes unresolved dephasing; it is not intrinsic T2.",
            "Through-plane object amplitude and B0 are uniform; 3D ACDC maps vary along the measured slice normal at each thickness point."],
        "optimization_ready": True}
    baseline_interpretation = ("Readout displacement is assumed to be a rigid GRE object/B0 registration shift; "
        "zero-current magnitude cannot distinguish a phase-ramp origin error or validate imaging-to-ACDC registration.")
    if acquisition.get("sequence_kind") == "spiral":
        baseline_interpretation = ("Native X/Y displacements are assumed rigid GRE object/B0 registration shifts; "
            "coil maps are resampled at translated patient positions. Zero-current magnitude alone cannot validate imaging-to-ACDC registration.")
    if baseline_parameters is not None:
        experiment["assumptions"].append(baseline_interpretation)
    calibration = {"acdc_maps": maps, "X_m": child_X.copy(), "Y_m": child_Y.copy(),
        "units": "Hz/A", "spatial_valid_mask": np.ones(shape, dtype=bool),
        "registration_confirmed": True, "metadata": dict(settings, source_file=str(path),
            source_sha256=digest, source_channel_order=np.arange(1, maps.shape[-1] + 1),
            native_local_to_source_m=native_to_source.copy(),
            effective_patient_to_source_m=effective_patient_to_source.copy(),
            alignment_reference_pose=deepcopy(alignment_reference),
            alignment_scope=alignment_scope)}
    if inplane_voxel:
        experiment["inplane_voxel"] = {
            "edge_vectors_m": np.stack((row_step, column_step)) @ combined.T,
            "b0_gradient_Hz_per_pixel": np.repeat(
                smoothing["gradient_Hz_per_pixel"].reshape(-1, 2)[parent_index, None, :], count, axis=1),
            "field_gradient_Hz_per_A_per_pixel": field_gradient,
            "field_at_corners_Hz_per_A": corner_maps,
            "corner_offsets_native_pixels": corners.copy(),
            "model": "Uniform finite native pixel; analytic Fourier integral of locally linear B0 and ACDC fields.",
            "axes": ["native row", "native column"],
            "gradient_missing_policy": "Exclude whole parent when B0 slope or any ACDC endpoint/corner stencil is unknown.",
            "ACDC_corner_coverage_verified": True,
            "relaxation_caveat": "Effective R2* may already contain unresolved dephasing; do not interpret it as intrinsic T2 or count resolved gradients twice.",
        }
        experiment["assumptions"].append("In-plane amplitude is uniform per measured GRE pixel; B0 and ACDC are locally linear there.")
    if complex_fit is not None:
        experiment["multi_echo_coherence"] = {
            "width_Hz": np.repeat(complex_maps["unresolved_width_Hz"].ravel()[parent_index, None], count, axis=1),
            "dispersion_kind": np.repeat(complex_maps["dispersion_kind"].astype(np.int8).ravel()[parent_index, None], count, axis=1),
            "spatial_assignment": "Unresolved frequency dispersion; no spatial axis inferred from 2D echoes.",
            "current_dependence": "None; unresolved dispersion cannot be assigned an ACDC response from this acquisition.",
            "zero_B0_target_policy": "Retain unresolved coherence in zero-B0 reference.",
            "width_definition": "Uniform: full width in Hz; Gaussian: standard deviation in Hz.",
            "amplitude_policy": "Envelope at explicit reference time, before known in-plane and unresolved coherence factors.",
            "report": deepcopy(complex_fit.get("report", {})),
        }
        experiment["provenance"]["object_model"] = (
            "Complex multi-echo envelope, residual R2, refined center B0, fixed known in-plane slopes and unresolved coherence.")
        experiment["assumptions"][1] = (
            "Residual exponential R2 is fitted jointly with unresolved frequency dispersion after the fixed known in-plane dephasing; it is not intrinsic T2.")
        experiment["assumptions"].append(
            "Unresolved multi-echo dispersion has no identified spatial direction or current response; its coherence stays in the zero-B0 target.")
        experiment["inplane_voxel"]["relaxation_caveat"] = (
            "Residual R2 and unresolved coherence were fitted jointly after the same known in-plane dephasing; retain these exact slopes.")
    requested_mass = float(amplitude[requested].sum())
    baseline_mass = float(amplitude[baseline_covered].sum())
    gradient_covered_mass = float(amplitude[covered].sum())
    retained_mass = float(amplitude[parent_valid].sum())
    report = {"native_shape": list(X.shape), "children_per_parent": count,
        "requested_parent_count": int(requested.sum()), "baseline_B0_parent_count": int(baseline_covered.sum()),
        "retained_parent_count": len(parent_index), "subspin_count": int(amplitudes.size),
        "excluded_missing_B0_parent_count": int((requested & ~baseline_covered).sum()),
        "excluded_missing_B0_gradient_parent_count": int((baseline_covered & ~covered).sum()),
        "excluded_missing_ACDC_gradient_or_corner_parent_count": int(np.sum(center_keep & ~keep)),
        "excluded_missing_ACDC_parent_count": int(covered.sum() - len(parent_index)),
        "excluded_missing_B0_amplitude_fraction": 1. - baseline_mass / requested_mass,
        "excluded_B0_gradient_amplitude_fraction_of_baseline": 1. - gradient_covered_mass / baseline_mass,
        "excluded_ACDC_amplitude_fraction_of_baseline": (gradient_covered_mass - retained_mass) / baseline_mass,
        "retained_amplitude_fraction_of_requested": retained_mass / requested_mass,
        "native_pixel_area_m2": object_area, "EPI_pixel_area_m2": image_area,
        "output_image_pixel_area_m2": image_area,
        "amplitude_area_ratio": area_ratio, "amplitude_reference_time_s": anchor,
        "rate_mode": rate_mode, "pooled_rate_accepted": accepted,
        "rate_min_sInv": float(rates.min()), "rate_max_sInv": float(rates.max()),
        "B0_mode": b0_mode, "B0_regularization": smoothing_report,
        "B0_weighting": b0_weighting, "inplane_voxel_enabled": bool(inplane_voxel),
        "complex_multi_echo_enabled": complex_fit is not None,
        "complex_multi_echo_report": None if complex_fit is None else deepcopy(complex_fit.get("report", {})),
        "baseline_calibration": deepcopy(baseline_parameters),
        "baseline_calibration_interpretation": baseline_interpretation,
        "baseline_calibration_phase_convention": (
            "No EPI navigator phase correction; rigid X/Y shifts act through actual Fourier coordinates and remain in the zero-B0 target."
            if acquisition.get("sequence_kind") == "spiral" else
            "phase_offset_rad += -0.5 * polarity * odd_even_phase_rad; residual after navigator correction; retained in zero-B0 target"),
        "baseline_calibration_frequency_convention": "constant Hz offset added after local B0/gradient fit; removed with B0 in zero-B0 target",
        "inplane_voxel_field_limit_corners_per_child": 4 if inplane_voxel else 0,
        "ACDC_corner_linearization_max_abs_error_Hz_per_A": None if not inplane_voxel else float(corner_linearization_error.max()),
        "slice_thickness_mm": thickness * 1000, "slice_center_patient_mm": position_mm.tolist(),
        "native_to_patient_basis": patient_basis.tolist(),
        "slice_normal_patient": patient_basis[:, 2].tolist(),
        "calibration_pose": {
            "coordinate_units": "m",
            "alignment_scope": alignment_scope,
            "alignment_reference_pose": alignment_reference,
            "patient_alignment_m": patient_alignment.tolist(),
            "patient_to_source_m": affine.tolist(),
            "native_local_to_aligned_patient_m": native_to_patient.tolist(),
            "native_local_to_source_m": native_to_source.tolist(),
            "effective_unaligned_patient_to_source_m": effective_patient_to_source.tolist(),
            "slice_center_source_m": native_to_source[:3, 3].tolist(),
            "native_patient_bounds_m": np.stack((xyz_patient_native.min(axis=(0, 1)), xyz_patient_native.max(axis=(0, 1)))).tolist(),
            "source_axes_bounds_m": [[float(axis[0]), float(axis[-1])] for axis in axes],
            "retained_patient_bounds_m": np.stack((xyz.min(axis=(0, 1)), xyz.max(axis=(0, 1)))).tolist(),
            "retained_source_bounds_m": np.stack((retained_query.min(axis=(0, 1)), retained_query.max(axis=(0, 1)))).tolist(),
            "registration_confirmed_setting": True,
            "registration_note": settings.get("registration_note", ""),
            "registration_validation": "The supplied transform is used as declared; interpolation coverage and synthetic tests do not establish physical registration.",
            "through_plane_B0": "Measured 2D B0 repeated through thickness; no measured normal gradient.",
        },
        "native_frame_origin_m": native_origin.tolist(), "model_encoding_origin_m": [0., 0.],
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
        "xyz_patient_native_m": xyz_patient_native,
        "xyz_source_m": retained_query,
        "subspin_b0": b0_children, "fieldbasis": maps, "native_amplitude": amplitude.copy(),
        "native_rate_sInv": rate.copy(), "native_B0_Hz": b0.copy(),
        "native_measured_B0_Hz": measured_b0.copy(),
        "native_B0_gradient_Hz_per_pixel": None if smoothing is None else smoothing["gradient_Hz_per_pixel"].copy(),
        "native_B0_gradient_valid_mask": None if smoothing is None else smoothing["gradient_valid_mask"].copy(),
        "native_B0_weights": None if b0_precision is None else b0_precision.copy(),
        "native_X_m": X.copy(), "native_Y_m": Y.copy(),
        "original_native_X_m": original_X.copy(), "original_native_Y_m": original_Y.copy(),
        "physical_X_m": physical_xy[..., 0], "physical_Y_m": physical_xy[..., 1],
        "frame_X_m": frame_xy[..., 0], "frame_Y_m": frame_xy[..., 1],
        "combined_frame_matrix": combined,
        "acquisition_native": native_acquisition, "report": report}


__all__ = ["prepare_echo_shim_inputs", "rotation_matrix"]
