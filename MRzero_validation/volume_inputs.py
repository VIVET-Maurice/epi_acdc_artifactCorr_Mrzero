"""Register measured 3D volumes and prepare one full-volume shim experiment.

The viewer uses patient X/Y/Z. The forward model keeps the EPI's native
readout/phase/partition grid. Selecting a plane creates a loss mask; it never
turns a 3D acquisition into a slice-selective 2D sequence.
"""

from pathlib import Path
from copy import deepcopy

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from echo_shim_inputs import _calibration_alignment
from inplane_geometry import inplane_matrix
from subspin_sampling import _load_source, _full_stencil_valid
from prepare_acdc_calibration import interpolate_maps
from volume_slice_viewer import extract_volume_slice


def with_native_pixel_signs(record, signs, *, reason):
    """Declare voxel-index directions separately from MDH gradient directions.

    Siemens Cartesian image RPS axes use the sign convention in
    twixtools.geometry: prs_to_pcs() @ rps_to_prs() = -Q[:, [1,0,2]].
    Pulseq's console OrientationMapping must be accounted for separately.
    This changes the coordinate affine, not the measured arrays, and is never
    selected by fitting anatomy. The original record remains unchanged.
    """
    signs = np.asarray(signs)
    if signs.shape != (3,) or signs.dtype.kind == 'b' or not np.isin(signs, [-1, 1]).all():
        raise ValueError('Native pixel signs must contain three values, each -1 or +1.')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('A declared voxel-axis convention needs a provenance reason.')
    result = dict(record)
    result['basis_patient'] = np.asarray(record['basis_patient']) @ np.diag(signs)
    result['metadata'] = deepcopy(record['metadata'])
    result['native_to_patient_m'] = np.eye(4)
    result['native_to_patient_m'][:3, :3] = result['basis_patient']
    result['native_to_patient_m'][:3, 3] = record['center_m']
    spacing = np.array([np.diff(axis).mean() for axis in record['axes_m']])
    voxel = result['native_to_patient_m'].copy()
    voxel[:3, :3] = result['basis_patient'] @ np.diag(spacing)
    voxel[:3, 3] -= voxel[:3, :3] @ (np.array(record['magnitude'].shape)//2)
    result['voxel_to_patient_m'] = voxel
    result['metadata'].update(native_pixel_signs=signs.tolist(),
        voxel_axis_convention_reason=reason, basis_patient=result['basis_patient'].tolist(),
        geometry_convention=record['metadata'].get('geometry_convention', '')+' '+reason,
        spatial_registration_fitted=False)
    return result


def patient_grid(record):
    """Patient positions of native voxel centres, in metres, shape XYZ3."""
    local = np.stack(np.meshgrid(*record['axes_m'], indexing='ij'), axis=-1)
    return local @ np.asarray(record['basis_patient']).T + record['center_m']


def sample_record(record, values, points_patient_m, *, valid=None, chunk=100_000):
    """Linear interpolation in physical coordinates, with no extrapolation."""
    values = np.asarray(values)
    points = np.asarray(points_patient_m, float)
    if points.shape[-1] != 3 or values.shape != tuple(map(len, record['axes_m'])):
        raise ValueError('Volume values and physical query points have incompatible shapes.')
    good = np.isfinite(values)
    if valid is not None:
        good &= np.asarray(valid, bool)
    data = RegularGridInterpolator(record['axes_m'], np.where(good, values, 0.),
                                   bounds_error=False, fill_value=0.)
    coverage = RegularGridInterpolator(record['axes_m'], good.astype(float),
                                       bounds_error=False, fill_value=0.)
    flat = points.reshape(-1, 3)
    output, covered = np.empty(len(flat), float), np.empty(len(flat), bool)
    basis = np.asarray(record['basis_patient'])
    for start in range(0, len(flat), chunk):
        sl = slice(start, start + chunk)
        native = (flat[sl] - record['center_m']) @ basis
        # Remove round-off at measured boundary centres, without extending the
        # physical field of view to a genuinely outside query.
        for dimension, axis in enumerate(record['axes_m']):
            for boundary in (axis[0], axis[-1]):
                close = np.abs(native[:, dimension]-boundary) < 1e-12
                native[close, dimension] = boundary
        covered[sl] = coverage(native) >= 1. - 1e-7
        output[sl] = data(native)
    output[~covered] = np.nan
    return output.reshape(points.shape[:-1]), covered.reshape(points.shape[:-1])


def build_volume_bundle(reference, epi, b0, *, b0_offset_Hz=0.):
    """Make the linked patient-grid viewer and a separate native EPI scene.

    B0 is resampled using its measured centre and orientation, which may differ
    from the EPI. A known global alias/frequency correction can be supplied in
    b0_offset_Hz; no image-based orientation or offset search is performed.
    """
    points = patient_grid(epi)
    ref, ref_valid = sample_record(reference, reference['magnitude'], points)
    field, field_valid = sample_record(b0, b0['b0Hz'], points, valid=b0['mask'])
    field += float(b0_offset_Hz)
    native = dict(reference_magnitude=ref, reference_valid=ref_valid,
                  epi_magnitude=np.asarray(epi['magnitude']), b0_Hz=field,
                  b0_valid=field_valid, patient_xyz_m=points,
                  axes_m=epi['axes_m'], basis_patient=epi['basis_patient'],
                  center_m=epi['center_m'],
                  FOV_m=np.asarray(epi['metadata']['FOV_m']))
    native['amplitude_reference_time_s'] = float(np.asarray(reference['metadata'].get('TE_s', 0.)).ravel()[0])
    # The patient-aligned display grid covers the measured EPI voxel centres.
    # Its resolution is the smallest native EPI spacing; it is not a new scan.
    spacing = min(float(np.diff(a).min()) for a in epi['axes_m'])
    low, high = points.min(axis=(0, 1, 2)), points.max(axis=(0, 1, 2))
    axes = tuple(np.linspace(a, b, int(np.ceil((b-a)/spacing-1e-7)) + 1)
                 for a, b in zip(low, high))
    display_points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    shown_ref, _ = sample_record(reference, reference['magnitude'], display_points)
    shown_epi, _ = sample_record(epi, epi['magnitude'], display_points)
    shown_b0, shown_valid = sample_record(b0, b0['b0Hz'], display_points, valid=b0['mask'])
    shown_b0 += float(b0_offset_Hz)
    bundle = dict(reference_magnitude=shown_ref, epi_magnitude=shown_epi,
                  b0_Hz=shown_b0, b0_valid=shown_valid, axes_m=axes)
    return bundle, native


def slice_roi(native, selection, *, thickness_m):
    """A patient-aligned slab on the full native 3D reconstruction grid."""
    if selection.get('axis') not in ('x', 'y', 'z'):
        raise ValueError('The confirmed selection needs an X, Y, or Z axis.')
    thickness = float(thickness_m)
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError('ROI thickness must be positive in metres.')
    axis = 'xyz'.index(selection['axis'])
    coordinate = float(selection['coordinate_m'])
    roi = np.abs(native['patient_xyz_m'][..., axis] - coordinate) <= thickness/2 + 1e-12
    roi &= native['reference_valid'] & np.isfinite(native['epi_magnitude'])
    if not roi.any():
        raise ValueError('The selected plane contains no common reference/EPI voxels.')
    return roi


def sample_patient_plane(values, native, volume_bundle, selection):
    """Display a native 3D result on exactly the confirmed viewer plane."""
    shown = extract_volume_slice(volume_bundle, selection['axis'], selection['index'])
    row, column = np.meshgrid(shown['row_coordinates_m'], shown['column_coordinates_m'], indexing='ij')
    points = np.empty(row.shape+(3,))
    points[..., 'xyz'.index(selection['axis'])] = selection['coordinate_m']
    points[..., 'xyz'.index(shown['orientation']['row_axis'])] = row
    points[..., 'xyz'.index(shown['orientation']['column_axis'])] = column
    return sample_record(native, values, points)[0], shown


def prepare_volume_shim(native, selection, calibration_path, calibration_settings, *,
                        alignment_reference_pose, alignment_angle_deg=90.,
                        alignment_mirror_x=False, alignment_mirror_y=False,
                        roi_thickness_m, object_threshold_fraction=.04,
                        T2star_s=None, chunk=32768):
    """Sample all 32 coil maps on the full EPI volume, then choose the loss ROI.

    Only measured field coverage is used. Missing object samples are excluded
    and reported without redistributing their amplitude. No 32-echo relaxation
    fit is manufactured from the two-echo B0 acquisition.
    """
    if not 0 <= object_threshold_fraction < 1:
        raise ValueError('object_threshold_fraction must lie in [0, 1).')
    if calibration_settings.get('units') != 'Hz/A':
        raise ValueError('The calibration must explicitly use Hz/A.')
    affine = np.asarray(calibration_settings['patient_to_source_m'], float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-10):
        raise ValueError('patient_to_source_m must be an invertible 4x4 affine in metres.')
    if alignment_reference_pose is None:
        raise ValueError('3D optimization requires a fixed alignment_reference_pose.')
    Q = inplane_matrix(alignment_angle_deg, alignment_mirror_x, alignment_mirror_y)
    _, alignment, reference_pose, _ = _calibration_alignment(
        np.zeros(3), np.eye(3), Q, alignment_reference_pose)
    effective = affine @ alignment
    path = Path(calibration_path).resolve()
    stat = path.stat()
    source, axes, valid_source, digest = _load_source(str(path), stat.st_mtime_ns, stat.st_size)
    if calibration_settings.get('source_sha256') not in (None, digest):
        raise ValueError('The coil calibration differs from its declared SHA256.')
    reference = np.asarray(native['reference_magnitude'], float)
    peak = float(np.nanmax(reference))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError('The reference volume must contain positive signal.')
    requested = native['reference_valid'] & (reference > object_threshold_fraction*peak)
    support = requested & native['b0_valid'] & np.isfinite(native['b0_Hz'])
    candidates = np.flatnonzero(support)
    b0_support = support.copy()
    points = native['patient_xyz_m'].reshape(-1, 3)
    maps = np.zeros(reference.shape + (32,), np.float32)
    flat_maps = maps.reshape(-1, 32)
    for start in range(0, len(candidates), chunk):
        indices = candidates[start:start+chunk]
        query = points[indices] @ effective[:3, :3].T + effective[:3, 3]
        sampled, valid = interpolate_maps(source, axes, query[:, None, :])
        valid = valid[:, 0] & _full_stencil_valid(axes, valid_source, query[:, None, :])[:, 0]
        flat_maps[indices[valid]] = sampled[valid, 0]
        support.ravel()[indices[~valid]] = False
    if not support.any():
        raise ValueError('No object voxels have both measured B0 and calibrated coil coverage.')
    amplitude = np.where(support, reference/peak, 0.).astype(np.float32)
    field = np.where(support, native['b0_Hz'], 0.).astype(np.float32)
    target = np.where(native['reference_valid'], reference/peak, 0.).astype(np.float32)
    roi = slice_roi(native, selection, thickness_m=roi_thickness_m)
    if not np.any(roi & support):
        raise ValueError('The selected ROI does not intersect the calibrated object.')
    if T2star_s is not None and (not np.isfinite(float(T2star_s)) or float(T2star_s) <= 0):
        raise ValueError('T2star_s must be None or a finite positive relaxation time.')
    rate = 0. if T2star_s is None else 1./float(T2star_s)
    report = dict(
        model='Full native 3D volume; selected patient plane defines the image loss only.',
        alignment_reference_pose=reference_pose,
        effective_patient_to_source_m=effective.tolist(), calibration_sha256=digest,
        requested_object_voxels=int(requested.sum()), retained_object_voxels=int(support.sum()),
        object_voxels_with_measured_B0=int(b0_support.sum()),
        excluded_B0_object_amplitude_fraction=float(np.nansum(reference[requested & ~b0_support])/
                                                     np.nansum(reference[requested])),
        excluded_calibration_amplitude_fraction_of_B0_object=float(np.nansum(reference[b0_support & ~support])/
                                                                     np.nansum(reference[b0_support])),
        excluded_object_amplitude_fraction=float(np.nansum(reference[requested & ~support])/
                                                  np.nansum(reference[requested])),
        roi_voxels=int(roi.sum()), roi_object_voxels=int((roi & support).sum()),
        excluded_selected_plane_object_amplitude_fraction=float(
            np.nansum(reference[roi & requested & ~support])/np.nansum(reference[roi & requested])),
        selection=dict(selection), roi_thickness_m=float(roi_thickness_m),
        relaxation='No inferred relaxation' if T2star_s is None else 'User-specified constant T2star',
        T2star_s=T2star_s,
        missing_response_policy='Exclude unsupported object samples; never extrapolate or renormalize.',
        field_model='Measured 3D B0, including its measured variation through every plane.')
    return dict(amplitude=amplitude, b0_Hz=field, fieldmaps_Hz_per_A=maps,
                FOV_m=native['FOV_m'], R2_sInv=np.full(reference.shape, rate, np.float32),
                amplitude_reference_time_s=native.get('amplitude_reference_time_s', 0.),
                support=support, reference_magnitude=target, roi_mask=roi, report=report)


__all__ = ['with_native_pixel_signs', 'patient_grid', 'sample_record', 'build_volume_bundle', 'slice_roi', 'sample_patient_plane', 'prepare_volume_shim']
