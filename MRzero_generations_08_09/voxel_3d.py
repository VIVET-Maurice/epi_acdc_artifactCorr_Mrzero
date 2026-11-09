"""Generation 08: one MRzero center and an analytic box in all three axes.

The measured B0 map is two dimensional, so its normal derivative is unknown
and is assumed zero. ACDC derivatives along the full slice thickness come
from the registered three dimensional calibration. No z sub-slices are added.
"""

from itertools import product
from pathlib import Path

import numpy as np
import torch

from echo_shim_inputs import prepare_echo_shim_inputs
from experiment_echo import EchoSubspinACDCModel
from measured_image_target import magnitude_objective_terms
from prepare_acdc_calibration import interpolate_maps
from subspin_sampling import _load_source, _full_stencil_valid


def normalized_sinc_and_derivative(q):
    """Normalized sinc and its derivative, finite at zero and sinc roots."""
    z = torch.pi * q
    small = z.abs() < .1
    z2 = z.square()
    series = 1 + z2 * (-1/6 + z2 * (1/120 + z2 * (-1/5040 + z2/362880)))
    derivative_series = torch.pi * z * (-1/3 + z2 * (1/30 + z2 * (-1/840 + z2/45360)))
    denominator = torch.where(small, torch.ones_like(z), z)
    sine, cosine = torch.sin(z), torch.cos(z)
    regular = sine / denominator
    derivative_regular = torch.pi * (z * cosine - sine) / denominator.square()
    return (torch.where(small, series, regular),
            torch.where(small, derivative_series, derivative_regular))


def _filter_parents(prepared, keep):
    """Remove uncovered parents consistently, retaining their original weights."""
    experiment = prepared["expanded_experiment"]
    calibration = prepared["expanded_calibration"]
    for key in ("parent_flat_index", "weights", "xyz_m", "xyz_patient_native_m",
                "xyz_source_m", "subspin_b0", "fieldbasis"):
        prepared[key] = prepared[key][keep]
    experiment["m0_proxy"] = experiment["m0_proxy"][keep]
    for section, names in (("geometry", ("X_m", "Y_m", "Z_m")),
                           ("field", ("b0Hz", "measured_b0Hz", "simulation_mask", "mask")),
                           ("relaxation", ("T2star_s",))):
        for name in names:
            experiment[section][name] = experiment[section][name][keep]
    for name in ("acdc_maps", "X_m", "Y_m", "spatial_valid_mask"):
        calibration[name] = calibration[name][keep]
    metadata = experiment["subspins"]
    for name in ("parent_flat_index", "weights", "xyz_patient_m", "xyz_patient_native_m"):
        metadata[name] = metadata[name][keep]
    if "multi_echo_coherence" in experiment:
        for name in ("width_Hz", "dispersion_kind"):
            experiment["multi_echo_coherence"][name] = experiment["multi_echo_coherence"][name][keep]
    valid = np.zeros(prepared["parent_shape"], dtype=bool)
    valid.ravel()[prepared["parent_flat_index"]] = True
    prepared["parent_valid_mask"] = valid
    metadata["parent_valid_mask"] = valid


def prepare_voxel_inputs(inputs, fit, calibration_path, calibration_settings, **options):
    """Prepare one full-thickness voxel per native pixel and 3D local slopes.

    The returned dictionary follows prepare_echo_shim_inputs. Its existing
    'inplane_voxel' metadata name is retained for pipeline compatibility, but
    now contains three axes and eight independently sampled physical corners.
    All endpoints and corners must be covered by valid calibration stencils.
    """
    if options.pop("subspins_z", 1) != 1:
        raise ValueError("Generation 08/09 uses one voxel through the slice: subspins_z=1.")
    if options.pop("inplane_voxel", True) is not True:
        raise ValueError("Generation 08/09 requires finite-voxel preparation.")
    prepared = prepare_echo_shim_inputs(inputs, fit, calibration_path,
        calibration_settings, subspins_z=1, inplane_voxel=True, **options)
    experiment = prepared["expanded_experiment"]
    old = experiment["inplane_voxel"]
    thickness = prepared["report"]["slice_thickness_mm"] * 1e-3
    edges = np.vstack((np.column_stack((old["edge_vectors_m"], np.zeros(2))),
                       [0., 0., thickness]))
    # Derive patient/source edges from native image geometry, before the
    # passive common-frame transform. This also supports an oblique slice.
    native_xy = np.stack((prepared["original_native_X_m"],
                          prepared["original_native_Y_m"]), axis=-1)
    native_edges = np.array([[*(native_xy[1, 0] - native_xy[0, 0]), 0.],
                             [*(native_xy[0, 1] - native_xy[0, 0]), 0.],
                             [0., 0., thickness]])
    pose = experiment["subspins"]["native_local_to_aligned_patient_m"]
    source_pose = experiment["subspins"]["native_local_to_source_m"]
    patient_edges = native_edges @ np.asarray(pose)[:3, :3].T
    source_edges = native_edges @ np.asarray(source_pose)[:3, :3].T
    path = Path(calibration_path).resolve()
    stat = path.stat()
    source, axes, source_valid, _ = _load_source(str(path), stat.st_mtime_ns, stat.st_size)
    centers = prepared["xyz_source_m"]

    def sample(offsets):
        query = (centers[..., None, :] + offsets @ source_edges).reshape(len(centers), -1, 3)
        maps, valid = interpolate_maps(source, axes, query)
        valid &= _full_stencil_valid(axes, source_valid, query)
        return maps[:, None], valid.all(axis=1)

    endpoints = np.concatenate((-.5 * np.eye(3), .5 * np.eye(3)))
    endpoint_maps, endpoint_valid = sample(endpoints)
    gradients = np.stack([endpoint_maps[..., axis + 3, :] - endpoint_maps[..., axis, :]
                          for axis in range(3)], axis=-2)
    corners = np.array(list(product((-.5, .5), repeat=3)))
    corner_maps, corner_valid = sample(corners)
    keep = endpoint_valid & corner_valid
    if not keep.any() or not np.any(experiment["field"]["mask"][keep]):
        raise ValueError("No complete voxels with valid 3D ACDC endpoints/corners remain.")
    before = float(experiment["m0_proxy"].sum())
    old_b0_gradient = old["b0_gradient_Hz_per_pixel"][keep]
    _filter_parents(prepared, keep)
    gradients, corner_maps = gradients[keep], corner_maps[keep]
    b0_gradient = np.concatenate((old_b0_gradient,
                                  np.zeros(old_b0_gradient.shape[:-1] + (1,))), axis=-1)
    linear = (prepared["fieldbasis"][..., None, :]
              + np.einsum("va,pzac->pzvc", corners, gradients))
    error = float(np.max(np.abs(linear - corner_maps)))
    experiment["inplane_voxel"] = dict(
        edge_vectors_m=edges[:, :2], voxel_edge_vectors_m=edges,
        patient_edge_vectors_m=patient_edges, source_edge_vectors_m=source_edges,
        b0_gradient_Hz_per_pixel=b0_gradient,
        field_gradient_Hz_per_A_per_pixel=gradients,
        field_at_corners_Hz_per_A=corner_maps,
        corner_offsets_native_pixels=corners,
        axes=["native row", "native column", "slice normal"],
        model="Uniform 3D box; local linear B0/ACDC fields; exact product of three sinc factors.",
        through_plane_B0="Unmeasured normal B0 derivative assumed zero; no gradient inferred from echo width.",
        ACDC_corner_coverage_verified=True)
    report = prepared["report"]
    after = float(experiment["m0_proxy"].sum())
    baseline_mass = before / (1. - report["excluded_ACDC_amplitude_fraction_of_baseline"]
        - report["excluded_B0_gradient_amplitude_fraction_of_baseline"])
    requested_mass = before / report["retained_amplitude_fraction_of_requested"]
    report.update(children_per_parent=1, subspin_count=int(experiment["m0_proxy"].size),
        retained_parent_count=int(keep.sum()),
        excluded_missing_ACDC_parent_count=report["excluded_missing_ACDC_parent_count"] + int((~keep).sum()),
        excluded_missing_ACDC_gradient_or_corner_parent_count=report["excluded_missing_ACDC_gradient_or_corner_parent_count"] + int((~keep).sum()),
        excluded_ACDC_amplitude_fraction_of_baseline=report["excluded_ACDC_amplitude_fraction_of_baseline"] + (before - after) / baseline_mass,
        retained_amplitude_fraction_of_requested=after / requested_mass,
        field_limit_children=int(experiment["field"]["mask"].sum()),
        all_children_in_field_limits=bool(experiment["field"]["mask"].all()),
        inplane_voxel_field_limit_corners_per_child=8,
        ACDC_corner_linearization_max_abs_error_Hz_per_A=error,
        voxel_axes=experiment["inplane_voxel"]["axes"],
        through_plane_B0=experiment["inplane_voxel"]["through_plane_B0"],
        through_plane_sampling="one center; exact analytic full-thickness box factor")
    pose_report = report["calibration_pose"]
    for name, points in (("retained_patient_bounds_m", prepared["xyz_m"]),
                         ("retained_source_bounds_m", prepared["xyz_source_m"]),
                         ("native_patient_bounds_m", prepared["xyz_patient_native_m"])):
        pose_report[name] = np.stack((points.min(axis=(0, 1)), points.max(axis=(0, 1)))).tolist()
    experiment["subspins"]["report"] = report
    experiment["assumptions"].append(report["through_plane_B0"])
    return prepared


class FiniteVoxel3DACDCModel(EchoSubspinACDCModel):
    """Native center histories times one analytic box factor in each axis.

    Optional multi_echo_coherence retains generation 06's fitted unresolved
    dispersion once. It has no assigned spatial axis or ACDC dependence.
    """

    def __init__(self, experiment, calibration, max_current_A=3., **options):
        metadata = experiment.get("inplane_voxel", {})
        shape = np.asarray(experiment["m0_proxy"]).shape
        nc = np.asarray(calibration["acdc_maps"]).shape[-1]
        specs = {"edge_vectors_m": (3, 2),
                 "b0_gradient_Hz_per_pixel": shape + (3,),
                 "field_gradient_Hz_per_A_per_pixel": shape + (3, nc),
                 "field_at_corners_Hz_per_A": shape + (8, nc)}
        self._voxel_numpy = {}
        for name, expected in specs.items():
            value = np.asarray(metadata.get(name))
            if (value.shape != expected or np.iscomplexobj(value)
                    or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()):
                raise ValueError(f"inplane_voxel.{name} must be finite real with shape {expected}.")
            self._voxel_numpy[name] = value.astype(float, copy=True)
        if abs(np.linalg.det(self._voxel_numpy["edge_vectors_m"][:2])) <= 0:
            raise ValueError("The row and column voxel edges must span a nonzero area.")
        coherence = experiment.get("multi_echo_coherence")
        self._coherence_numpy = None
        if coherence is not None:
            width = np.asarray(coherence.get("width_Hz"))
            kind = np.asarray(coherence.get("dispersion_kind", np.ones(shape)))
            if (width.shape != shape or np.iscomplexobj(width)
                    or not np.isfinite(width).all() or np.any(width < 0)
                    or kind.shape != shape or not np.isin(kind, (0, 1, 2)).all()):
                raise ValueError("Unresolved coherence needs nonnegative finite width and kind 0/1/2 per voxel.")
            self._coherence_numpy = (width.copy(), kind.copy())
        self._coherence_tensors = None
        self._voxel_ready = False
        super().__init__(experiment, calibration, max_current_A, **options)
        self._retain_contributions = False
        self.runtime_info.update(retain_contributions=False,
            inplane_voxel=dict(model="analytic uniform 3D box", axes=metadata["axes"],
                native_point_cache_expansion=1, native_point_dephasing="disabled",
                through_plane_B0=metadata.get("through_plane_B0"),
                through_plane_sampling="one center; analytic full-thickness integral",
                field_constraint_domain="eight linearized and eight independently sampled physical corners"),
            multi_echo_coherence=dict(enabled=coherence is not None,
                spatial_assignment="unresolved; independent of ACDC", applied_once=True))

    def _ensure_voxel_tensors(self):
        if self._voxel_ready:
            return
        self._voxel_edges = self.tensor(self._voxel_numpy["edge_vectors_m"])
        self._voxel_b0_gradient = self.tensor(self._voxel_numpy["b0_gradient_Hz_per_pixel"][self.support])
        gradients = self._voxel_numpy["field_gradient_Hz_per_A_per_pixel"]
        self._voxel_field_gradient = self.tensor(gradients[self.support])
        offsets = np.array(list(product((-.5, .5), repeat=3)))
        linear = (self.maps_numpy[self.quality_mask, None, :]
                  + np.einsum("va,pac->pvc", offsets, gradients[self.quality_mask]))
        actual = self._voxel_numpy["field_at_corners_Hz_per_A"][self.quality_mask]
        self.B_quality = self.tensor(np.concatenate((linear.reshape(-1, self.nc),
                                                     actual.reshape(-1, self.nc))))
        self._voxel_ready = True

    def _make_base(self, selected, b0_enabled=True):
        native = super()._make_base(selected, b0_enabled)
        if self._coherence_numpy is None:
            return native
        if self._coherence_tensors is None:
            width, kind = self._coherence_numpy
            self._coherence_tensors = (self.tensor(width[self.support])[:, None],
                                      torch.as_tensor(kind[self.support], device=self.device)[:, None])
        width, kind = self._coherence_tensors
        q = width * self._evolution[selected][None]
        factor = torch.where(kind == 1, torch.sinc(q),
            torch.where(kind == 2, torch.exp(-2 * torch.pi**2 * q.square()), torch.ones_like(q)))
        return native * factor

    def _voxel_integrated_currents(self, currents, selected, response_enabled):
        enabled = self.current_response["enabled"] if response_enabled is None else response_enabled
        H = self.H_response if enabled else self.H_instant
        if H is None:
            raise ValueError("Rebuild the model with the requested hardware response enabled.")
        if enabled and not self.common_tau:
            return torch.bmm(currents[:, None, :], H[:, selected, :].transpose(1, 2)).squeeze(1)
        return currents @ H[selected].T

    def _voxel_factors(self, currents, selected, response_enabled=None,
                       b0_enabled=True, derivatives=False):
        self._ensure_voxel_tensors()
        integrated = self._voxel_integrated_currents(currents, selected, response_enabled)
        factors, slopes = [], []
        for axis in range(3):
            edge = self._voxel_edges[axis]
            q = (edge[0] * self._kx[selected] + edge[1] * self._ky[selected])[None]
            q = q + self._voxel_field_gradient[:, axis] @ integrated
            if b0_enabled:
                q = q + self._voxel_b0_gradient[:, axis, None] * self._evolution[selected][None]
            value, slope = normalized_sinc_and_derivative(q) if derivatives else (torch.sinc(q), None)
            factors.append(value)
            slopes.append(slope)
        factor = factors[0] * factors[1] * factors[2]
        if not derivatives:
            return factor
        derivative = [slopes[a] * factors[(a + 1) % 3] * factors[(a + 2) % 3] for a in range(3)]
        return factor, derivative

    def _contribution(self, currents, selected, response_enabled=None, b0_enabled=True):
        point = super()._contribution(currents, selected, response_enabled, b0_enabled)
        return point * self._voxel_factors(currents, selected, response_enabled, b0_enabled)

    @torch.no_grad()
    def _objective_tensor(self, currents, gradient=True):
        image, kdata = self._forward_tensor(currents)
        value, derivative = magnitude_objective_terms(self, image, gradient=gradient)
        if not gradient:
            return value, image, kdata
        adjoint = self.reconstruction.adjoint(derivative).reshape(-1)
        result = torch.zeros_like(currents)
        for first in range(0, self.nsamples, self.sample_chunk):
            selected = slice(first, min(first + self.sample_chunk, self.nsamples))
            factor, slopes = self._voxel_factors(currents, selected, derivatives=True)
            point = super()._contribution(currents, selected)
            product_value = point * adjoint[selected].conj()
            sensitivity = (2 * np.pi) * (product_value * factor).imag
            channel_sensitivity = self.B.T @ sensitivity
            for axis in range(3):
                channel_sensitivity += self._voxel_field_gradient[:, axis].T @ (product_value.real * slopes[axis])
            if self._channel_response:
                result += torch.bmm(channel_sensitivity[:, None], self._H[:, selected]).squeeze(1)
            else:
                result += channel_sensitivity @ self._H[selected]
        return value, result, image, kdata

    def response_diagnostics(self, currents_A, field_limit_Hz=np.inf, total_current_limit_A=np.inf):
        result = super().response_diagnostics(currents_A, field_limit_Hz, total_current_limit_A)
        actual = self.tensor(result["actual_currents_at_adc_A"])
        maximum = 0.
        for first in range(0, actual.shape[1], 128):
            maximum = max(maximum, float((self.B_quality @ actual[:, first:first + 128]).abs().max()))
        result.update(maximum_actual_quality_field_Hz=maximum,
            actual_adc_field_limits_met=maximum <= field_limit_Hz + 1e-8,
            limit_check_domain="all eight linearized and eight calibrated 3D corners at every ADC")
        return result

    def validate_voxel_quadrature(self, currents_A=None, sample_count=24,
                                  point_count=64, quadrature_order=64):
        """Compare each analytic axis factor against independent box quadrature."""
        for value in (sample_count, point_count, quadrature_order):
            if isinstance(value, bool) or int(value) != value or value < 2:
                raise ValueError("Quadrature and validation sample counts must be integers >=2.")
        currents = self._check_currents(np.zeros((self.nc, self.nstates)) if currents_A is None else currents_A)
        samples = np.unique(np.linspace(0, self.nsamples - 1, min(sample_count, self.nsamples), dtype=int))
        points = np.unique(np.linspace(0, int(self.support.sum()) - 1,
                                      min(point_count, int(self.support.sum())), dtype=int))
        self._ensure_voxel_tensors()
        integrated = self.numpy(self._voxel_integrated_currents(currents, samples, None))
        k = np.stack((self.numpy(self._kx)[samples], self.numpy(self._ky)[samples]))
        elapsed = self.numpy(self._evolution)[samples]
        gradient = self._voxel_numpy["b0_gradient_Hz_per_pixel"][self.support][points]
        field = self._voxel_numpy["field_gradient_Hz_per_A_per_pixel"][self.support][points]
        phases = [((self._voxel_numpy["edge_vectors_m"][a] @ k)[None]
                   + gradient[:, a, None] * elapsed[None] + field[:, a] @ integrated) for a in range(3)]
        maximum = max(float(np.max(abs(q))) for q in phases)
        required = int(np.ceil(2 * np.pi * maximum)) + 16
        order = min(512, max(int(quadrature_order), required))
        nodes, weights = np.polynomial.legendre.leggauss(order)
        integrals = [np.sum(np.exp(-2j * np.pi * q[..., None] * nodes / 2) * weights / 2, axis=-1) for q in phases]
        expected = integrals[0] * integrals[1] * integrals[2]
        actual = self.numpy(self._voxel_factors(currents, samples))[points]
        error = float(np.max(np.abs(expected - actual)))
        tolerance = 1e-8 if self.dtype == torch.float64 else 1e-5
        return dict(passed=bool(error <= tolerance and order >= required),
            max_abs_factor_error=error, tolerance=tolerance, axes=3,
            quadrature_order=order, required_quadrature_order=required,
            sample_count=len(samples), point_count=len(points), native_point_cache_expansion=1,
            oracle="independent Gauss-Legendre box integral along all three voxel axes")


__all__ = ["prepare_voxel_inputs", "FiniteVoxel3DACDCModel"]
