"""Known sphere-slice inputs for Generation 08; no scans or echo fitting.

The object is a voxelised axial cross-section of a sphere. Each selected pixel
is one uniform full-thickness box, as in Generation 08. The curved boundary
inside an edge voxel and the RF slice profile are not resolved.
"""

from itertools import product
from pathlib import Path

import numpy as np

from prepare_acdc_calibration import interpolate_maps
from subspin_sampling import _full_stencil_valid, _load_source


def spherical_bubble_field(points_m, center_m, radius_m, delta_chi_ppm=9.4,
                           main_field_T=3.0, main_field_direction=(0., 0., 1.)):
    """Exterior weak-susceptibility sphere field in Hz and gradient in Hz/m.

    Delta chi is chi_air - chi_water (SI ppm); gamma is in Hz/T, not rad/s/T.
    Formula: https://pmc.ncbi.nlm.nih.gov/articles/PMC3465725/ (Eq. 3).
    Points inside/on the air sphere are rejected: this models water outside it.
    """
    points = np.asarray(points_m, dtype=float)
    center = np.asarray(center_m, dtype=float)
    direction = np.asarray(main_field_direction, dtype=float)
    if (points.ndim < 1 or points.shape[-1] != 3 or center.shape != (3,)
            or direction.shape != (3,) or not np.isfinite(points).all()
            or not np.isfinite([*center, *direction, radius_m, delta_chi_ppm, main_field_T]).all()):
        raise ValueError('Bubble geometry and field parameters must be finite 3D values.')
    if radius_m <= 0 or main_field_T <= 0 or np.linalg.norm(direction) == 0:
        raise ValueError('Bubble radius, main field and direction norm must be positive.')
    direction = direction / np.linalg.norm(direction)
    v = points - center
    q = np.sum(v*v, axis=-1)
    if np.any(q <= radius_m**2):
        raise ValueError('The exterior bubble formula requires all points outside the air sphere.')
    s = v @ direction
    coefficient = 42.57747892e6 * main_field_T * delta_chi_ppm*1e-6 * radius_m**3 / 3
    angular = 3*s*s - q
    field = coefficient * angular / q**2.5
    gradient = coefficient * ((6*s[..., None]*direction - 2*v)/q[..., None]**2.5
                              - 5*angular[..., None]*v/q[..., None]**3.5)
    return field, gradient


def make_sphere_inputs(acquisition, calibration_path, *, radius_m=0.060,
                       slice_offset_m=0.0, slice_thickness_m=0.005,
                       proton_density=1.0, t2star_s=np.inf,
                       b0_offset_Hz=0.0, b0_gradient_Hz_per_m=(0., 0., 0.),
                       coil_center_m=(0., 0., 0.030),
                       bubble_radius_m=0.0, bubble_center_m=(.015, -.010, -.018),
                       bubble_delta_chi_ppm=9.4, main_field_T=3.0,
                       main_field_direction=(0., 0., 1.)):
    """Return experiment, ACDC calibration and known maps for one sphere slice.

    Image coordinates are centred on the sphere. ``slice_offset_m`` selects
    an axial plane relative to its centre. ``coil_center_m`` places that centre
    virtually in the 3D ACDC calibration; this is not a measured registration.
    Synthetic X/Y align with calibration X/Y, with translation only.
    B0 is a linear background plus an optional spherical air-bubble field.
    The host sphere's own susceptibility boundary is not modelled. The bubble
    must lie entirely below the selected slab; radius zero disables it.
    Its analytic local gradient supplies the same three-sinc model as 08.
    """
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('radius_m must be positive and finite.')
    if not np.isfinite(slice_offset_m) or abs(slice_offset_m) >= radius_m:
        raise ValueError('The selected slice must intersect the sphere.')
    if not np.isfinite(slice_thickness_m) or slice_thickness_m <= 0:
        raise ValueError('slice_thickness_m must be positive and finite.')
    if not np.isfinite(proton_density) or proton_density <= 0:
        raise ValueError('proton_density must be positive and finite.')
    if np.isnan(t2star_s) or t2star_s <= 0:
        raise ValueError('t2star_s must be positive; infinity is allowed.')
    gradient = np.asarray(b0_gradient_Hz_per_m, dtype=float)
    placement = np.asarray(coil_center_m, dtype=float)
    if gradient.shape != (3,) or placement.shape != (3,) or not np.isfinite([*gradient, *placement, b0_offset_Hz]).all():
        raise ValueError('B0 gradient, B0 offset and virtual coil placement must be finite.')
    bubble_center = np.asarray(bubble_center_m, dtype=float)
    bubble_direction = np.asarray(main_field_direction, dtype=float)
    if (bubble_center.shape != (3,) or bubble_direction.shape != (3,)
            or not np.isfinite([*bubble_center, *bubble_direction, bubble_radius_m,
                                bubble_delta_chi_ppm, main_field_T]).all()):
        raise ValueError('Bubble geometry and main field parameters must be finite.')
    if bubble_radius_m < 0 or main_field_T <= 0 or np.linalg.norm(bubble_direction) == 0:
        raise ValueError('Use a nonnegative bubble radius and a positive main field/direction norm.')
    bubble_direction = bubble_direction / np.linalg.norm(bubble_direction)

    ny, nx = np.asarray(acquisition['kx_mInv']).shape
    fov = np.asarray(acquisition['FOV_m'], dtype=float)
    dx, dy = fov / [nx, ny]
    if radius_m > min(fov) / 2:
        raise ValueError('The sphere radius must fit inside the image FOV.')
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    X, Y = np.meshgrid(x, y)
    support = X**2 + Y**2 + slice_offset_m**2 <= radius_m**2
    indices = np.flatnonzero(support)
    if not len(indices):
        raise ValueError('The selected sphere slice contains no voxel centres.')
    if (np.any(abs(X[support])+dx/2 > fov[0]/2+1e-12)
            or np.any(abs(Y[support])+dy/2 > fov[1]/2+1e-12)):
        raise ValueError('Some sphere voxels extend outside the FOV. Reduce radius_m by at least half a pixel.')
    shape = (len(indices), 1)
    native_X, native_Y = X[support, None], Y[support, None]
    edges = np.array([[0., dy, 0.], [dx, 0., 0.], [0., 0., slice_thickness_m]])

    # The same 3D ACDC maps as 08: check centres, six endpoints and eight corners.
    corners = np.array(list(product((-.5, .5), repeat=3)))
    offsets = np.vstack((np.zeros((1, 3)), -.5*np.eye(3), .5*np.eye(3), corners))
    centres = np.column_stack((X[support], Y[support], np.full(len(indices), slice_offset_m))) + placement
    query = centres[:, None, :] + offsets @ edges
    path = Path(calibration_path).resolve()
    stat = path.stat()
    source, axes, source_valid, source_hash = _load_source(str(path), stat.st_mtime_ns, stat.st_size)
    maps, valid = interpolate_maps(source, axes, query)
    valid &= _full_stencil_valid(axes, source_valid, query)
    covered = valid.all(axis=1)
    if not covered.all():
        raise ValueError(f'{int((~covered).sum())} sphere voxels lack complete ACDC coverage. '
                         'Use a smaller radius or change coil_center_m; the object was not cropped.')
    coil_gradients = maps[:, 4:7] - maps[:, 1:4]
    coil_corners = maps[:, 7:15]
    linear_corners = maps[:, 0, None, :] + np.einsum('va,pac->pvc', corners, coil_gradients)

    B0 = b0_offset_Hz + gradient[0]*X + gradient[1]*Y + gradient[2]*slice_offset_m
    voxel_centres = np.column_stack((X[support], Y[support], np.full(len(indices), slice_offset_m)))
    total_gradient = np.broadcast_to(gradient, (len(indices), 3)).copy()
    bubble_map = np.zeros(X.shape)
    bubble_error = 0.
    if bubble_radius_m > 0:
        if bubble_center[2]+bubble_radius_m >= slice_offset_m-slice_thickness_m/2:
            raise ValueError('The air bubble must lie completely below the selected slice slab.')
        bubble_values, bubble_gradient = spherical_bubble_field(
            voxel_centres, bubble_center, bubble_radius_m, bubble_delta_chi_ppm,
            main_field_T, bubble_direction)
        bubble_map[support] = bubble_values
        B0 = B0 + bubble_map
        total_gradient += bubble_gradient
        # Quantify the curvature neglected by 08's locally linear voxel model.
        exact_corners, _ = spherical_bubble_field(
            voxel_centres[:, None, :] + corners @ edges, bubble_center,
            bubble_radius_m, bubble_delta_chi_ppm, main_field_T, bubble_direction)
        linear_bubble = bubble_values[:, None] + np.einsum('va,pa->pv', corners @ edges, bubble_gradient)
        bubble_error = float(np.max(abs(exact_corners-linear_bubble)))
    gradient_map = np.full(X.shape+(3,), np.nan)
    gradient_map[support] = total_gradient
    pd = np.where(support, proton_density, 0.)
    experiment = dict(
        m0_proxy=np.full(shape, proton_density), acquisition=acquisition,
        amplitude_reference_time_s=0.,
        geometry=dict(X_m=native_X, Y_m=native_Y, Z_m=np.zeros(shape),
                      slice_thickness_mm=slice_thickness_m*1000),
        field=dict(b0Hz=B0[support, None], simulation_mask=np.ones(shape, bool), mask=np.ones(shape, bool)),
        relaxation=dict(T2star_s=np.full(shape, t2star_s), amplitude_reference_time_s=0.),
        current_response=dict(enabled=True, tau_s=133e-6/np.log(100.)),
        subspins=dict(parent_shape=(ny, nx), parent_flat_index=indices, weights=np.ones(shape)),
        inplane_voxel=dict(
            edge_vectors_m=edges[:, :2], voxel_edge_vectors_m=edges,
            b0_gradient_Hz_per_pixel=(total_gradient @ edges.T)[:, None, :],
            field_gradient_Hz_per_A_per_pixel=coil_gradients[:, None],
            field_at_corners_Hz_per_A=coil_corners[:, None],
            axes=['native row', 'native column', 'slice normal'],
            through_plane_B0='Known analytic background/bubble gradient; locally linear within each voxel.'),
    )
    calibration = dict(acdc_maps=maps[:, 0, None, :], X_m=native_X, Y_m=native_Y,
                       units='Hz/A', spatial_valid_mask=np.ones(shape, bool), registration_confirmed=False)
    phantom = dict(
        X_m=X, Y_m=Y, PD=pd, B0_Hz=np.where(support, B0, np.nan), support=support,
        bubble_B0_Hz=np.where(support, bubble_map, np.nan), total_B0_gradient_Hz_per_m=gradient_map,
        radius_m=float(radius_m), slice_offset_m=float(slice_offset_m),
        slice_thickness_m=float(slice_thickness_m), T2star_s=float(t2star_s),
        B0_offset_Hz=float(b0_offset_Hz), B0_gradient_Hz_per_m=gradient,
        bubble=dict(enabled=bool(bubble_radius_m > 0), radius_m=float(bubble_radius_m),
                    center_m=bubble_center, delta_chi_ppm=float(bubble_delta_chi_ppm),
                    main_field_T=float(main_field_T), main_field_direction=bubble_direction,
                    corner_linearization_max_error_Hz=bubble_error),
        coil_center_m=placement, coil_query_centres_m=centres,
        calibration_path=str(path), calibration_sha256=source_hash,
        retained_voxels=len(indices), excluded_calibration_voxels=0,
        corner_linearization_max_error_Hz_per_A=float(np.max(abs(linear_corners-coil_corners))),
        geometry='One voxelised axial cross-section of a sphere; one uniform full-thickness box per pixel.',
        amplitude='Known uniform synthetic PD at RF; no fitted amplitude or unresolved coherence.',
    )
    return experiment, calibration, phantom
