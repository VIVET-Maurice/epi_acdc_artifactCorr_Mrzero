"""Native MRzero point histories with exact local linear in-plane box integrals.

The two in-plane pixel axes are integrated analytically. Existing weighted
through-plane children remain separate native MRzero point spins, so native
magnetization caches do not grow with an in-plane quadrature grid.
"""

from __future__ import annotations

import numpy as np
import torch

from experiment_echo import EchoSubspinACDCModel
from measured_image_target import magnitude_objective_terms


def normalized_sinc_and_derivative(q):
    """Return sin(pi*q)/(pi*q) and its derivative, including exact q=0.

    A Taylor polynomial avoids cancellation near zero in both precisions.
    The regular branch uses a safe denominator even when torch.where evaluates
    both branches; no division by sinc is required at its nonzero roots.
    """
    z = torch.pi * q
    small = z.abs() < .1
    z2 = z.square()
    series = 1 + z2 * (-1/6 + z2 * (1/120 + z2 * (-1/5040 + z2/362880)))
    derivative_series = torch.pi * z * (
        -1/3 + z2 * (1/30 + z2 * (-1/840 + z2/45360)))
    denominator = torch.where(small, torch.ones_like(z), z)
    sine, cosine = torch.sin(z), torch.cos(z)
    regular = sine / denominator
    derivative_regular = torch.pi * (z * cosine - sine) / denominator.square()
    return (torch.where(small, series, regular),
            torch.where(small, derivative_series, derivative_regular))


class FiniteVoxelEchoACDCModel(EchoSubspinACDCModel):
    """Integrate a uniform in-plane box with locally linear B0/ACDC fields.

    Required ``experiment['inplane_voxel']`` metadata:

    * ``edge_vectors_m[2, 2]``: full native row and column pixel edges in the
      same sequence XY frame as the spins and ADC k coordinates;
    * ``b0_gradient_Hz_per_pixel[parent, child, 2]``;
    * ``field_gradient_Hz_per_A_per_pixel[parent, child, 2, channel]``.

    Optional ``field_at_corners_Hz_per_A[parent, child, 4, channel]`` retains
    independently interpolated calibration values at the four actual corners.
    Both those values and the linear-model corners then constrain currents.

    Gradients are scalar field changes across a FULL native pixel along its
    row/column axis, independent of how those axes are rotated in XY. For an
    offset u in [-1/2, 1/2], B0(u)=B0(center)+u*gradient. Pixel volume and z
    weights already reside in m0_proxy and are not multiplied again.

    Native MRzero supplies every center-spin baseline. The additional factor
    sinc(q_row)*sinc(q_column) is the exact box integral under this explicitly
    local linear-field/uniform-amplitude model, not an extra intrinsic T2*.
    The objective includes derivatives of this factor and of center phase.
    Field limits cover the four linearized corners of every quality child.
    """

    def __init__(self, experiment, calibration, max_current_A=3., **options):
        metadata = experiment.get('inplane_voxel')
        if not isinstance(metadata, dict):
            raise ValueError("FiniteVoxelEchoACDCModel requires experiment['inplane_voxel'] metadata.")
        shape = np.asarray(experiment['m0_proxy']).shape
        channels = np.asarray(calibration['acdc_maps']).shape[-1]
        specifications = [
            ('edge_vectors_m', (2, 2)),
            ('b0_gradient_Hz_per_pixel', shape + (2,)),
            ('field_gradient_Hz_per_A_per_pixel', shape + (2, channels)),
        ]
        if 'field_at_corners_Hz_per_A' in metadata:
            specifications.append(('field_at_corners_Hz_per_A', shape + (4, channels)))
        self._voxel_numpy = {}
        for name, expected_shape in specifications:
            value = np.asarray(metadata.get(name))
            if (value.shape != expected_shape or np.iscomplexobj(value)
                    or not np.issubdtype(value.dtype, np.number)
                    or not np.isfinite(value).all()):
                raise ValueError(f'inplane_voxel.{name} must be finite real with shape {expected_shape}.')
            self._voxel_numpy[name] = value.astype(np.float64, copy=True)
        edges = self._voxel_numpy['edge_vectors_m']
        if abs(np.linalg.det(edges)) <= np.finfo(float).eps * np.linalg.norm(edges)**2:
            raise ValueError('inplane_voxel.edge_vectors_m must span a nondegenerate pixel area.')
        self._voxel_ready = False
        super().__init__(experiment, calibration, max_current_A, **options)
        self._retain_contributions = False
        self.runtime_info['retain_contributions'] = False
        self.runtime_info['inplane_voxel'] = {
            'model': 'uniform box; locally linear B0 and ACDC fields; exact normalized sinc integral',
            'edge_vectors_m': edges.tolist(),
            'axes': ['native row', 'native column'],
            'native_point_cache_expansion': 1,
            'native_point_dephasing': 'disabled; in-plane box factor applied exactly once',
            'current_derivative': 'center shim phase plus both sinc-factor derivatives; no division by sinc',
            'field_constraint_domain': 'linearized in-plane corners of every reliable z child, '
                'plus independently calibrated corners when supplied',
            'field_constraint_points': int(self.B_quality.shape[0]),
            'calibrated_corner_limits_included': 'field_at_corners_Hz_per_A' in self._voxel_numpy,
            'through_plane_sampling': 'existing weighted native point children, unchanged',
            'B0_disabled_policy': 'remove center B0 and B0 gradient; retain encoding and ACDC box phases',
            'effective_R2star_overlap_possible': bool(torch.any(self._rate > 0)),
            'effective_R2star_assumption': 'Any nonzero fitted native-voxel R2* is treated as residual decay; '
                'it can already contain unresolved B0 dephasing and overlap with the explicit box factor. '
                'The model cannot separate these effects from early magnitude echoes alone.',
        }

    def _ensure_voxel_tensors(self):
        if self._voxel_ready:
            return
        self._voxel_edges = self.tensor(self._voxel_numpy['edge_vectors_m'])
        self._voxel_b0_gradient = self.tensor(
            self._voxel_numpy['b0_gradient_Hz_per_pixel'][self.support])
        gradient = self._voxel_numpy['field_gradient_Hz_per_A_per_pixel']
        self._voxel_field_gradient = self.tensor(gradient[self.support])
        # Called during the parent's initial ideal forward, before the hardware
        # object is built. Its cached positive/negative bases therefore already
        # contain corners, including the unequal-response-tau case.
        quality_gradient = gradient[self.quality_mask]
        centers = self.maps_numpy[self.quality_mask]
        corners = [centers + u * quality_gradient[:, 0] + v * quality_gradient[:, 1]
                   for u, v in ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5))]
        if 'field_at_corners_Hz_per_A' in self._voxel_numpy:
            actual = self._voxel_numpy['field_at_corners_Hz_per_A'][self.quality_mask]
            corners.extend(actual[:,corner] for corner in range(4))
        self._voxel_corner_maps_numpy = np.concatenate(corners, axis=0)
        self.B_quality = self.tensor(self._voxel_corner_maps_numpy)
        self._voxel_ready = True

    def _voxel_integrated_currents(self, currents, selected, response_enabled):
        enabled = self.current_response['enabled'] if response_enabled is None else response_enabled
        H = self.H_response if enabled else self.H_instant
        if H is None:
            raise ValueError('Response was disabled when preparing this model; prepare it again to enable it.')
        if enabled and not self.common_tau:
            return torch.bmm(currents[:, None, :], H[:, selected, :].transpose(1, 2)).squeeze(1)
        return currents @ H[selected].T

    def _voxel_factors(self, currents, selected, response_enabled=None, b0_enabled=True,
                       derivatives=False):
        self._ensure_voxel_tensors()
        integrated = self._voxel_integrated_currents(currents, selected, response_enabled)
        factors, slopes = [], []
        for axis in (0, 1):
            edge = self._voxel_edges[axis]
            q = (edge[0] * self._kx[selected] + edge[1] * self._ky[selected])[None]
            q = q + self._voxel_field_gradient[:, axis] @ integrated
            if b0_enabled:
                q = q + self._voxel_b0_gradient[:, axis, None] * self._evolution[selected][None]
            if derivatives:
                value, slope = normalized_sinc_and_derivative(q)
                factors.append(value)
                slopes.append(slope)
            else:
                factors.append(torch.sinc(q))
        factor = factors[0] * factors[1]
        if derivatives:
            return factor, slopes[0] * factors[1], factors[0] * slopes[1]
        return factor

    def _contribution(self, currents, selected, response_enabled=None, b0_enabled=True):
        # Do not override _make_base: cached baseline remains native MRzero
        # center magnetization with no finite-voxel dephasing applied there.
        factor = self._voxel_factors(currents, selected, response_enabled, b0_enabled)
        point = super()._contribution(currents, selected, response_enabled, b0_enabled)
        return point * factor

    @torch.no_grad()
    def _objective_tensor(self, currents, gradient=True):
        # Recompute a single bounded point block after the image adjoint.
        # Retaining attenuated contributions would not recover the unattenuated
        # point at sinc roots and would add cache without avoiding that work.
        image, kdata = self._forward_tensor(currents)
        value, derivative = magnitude_objective_terms(self, image, gradient=gradient)
        if not gradient:
            return value, image, kdata
        adjoint = self.reconstruction.adjoint(derivative).reshape(-1)
        result = torch.zeros_like(currents)
        for first in range(0, self.nsamples, self.sample_chunk):
            selected = slice(first, min(first + self.sample_chunk, self.nsamples))
            factor, drow, dcolumn = self._voxel_factors(currents, selected, derivatives=True)
            point = super()._contribution(currents, selected)
            product = point * adjoint[selected].conj()
            sensitivity = (2 * np.pi) * (product * factor).imag
            channel_sensitivity = self.B.T @ sensitivity
            channel_sensitivity += self._voxel_field_gradient[:, 0].T @ (product.real * drow)
            channel_sensitivity += self._voxel_field_gradient[:, 1].T @ (product.real * dcolumn)
            if self._channel_response:
                result += torch.bmm(channel_sensitivity[:, None], self._H[:, selected]).squeeze(1)
            else:
                result += channel_sensitivity @ self._H[selected]
        return value, result, image, kdata

    def response_diagnostics(self, currents_A, field_limit_Hz=np.inf, total_current_limit_A=np.inf):
        result = super().response_diagnostics(currents_A, field_limit_Hz, total_current_limit_A)
        actual = self.tensor(result['actual_currents_at_adc_A'])
        maximum = 0.
        for first in range(0, actual.shape[1], 128):
            maximum = max(maximum, float((self.B_quality @ actual[:, first:first+128]).abs().max()))
        result.update(maximum_actual_quality_field_Hz=maximum,
            actual_adc_field_limits_met=maximum <= field_limit_Hz + 1e-8,
            limit_check_domain='all linearized and supplied calibrated in-plane corners per reliable z child at every ADC; command bounds also constrain corners',
            field_display_domain='z-weighted parent center fields; constraints additionally cover four in-plane corners')
        return result

    def validate_voxel_quadrature(self, currents_A=None, sample_count=24, point_count=64,
                                  quadrature_order=64):
        """Check bounded deterministic box factors against independent quadrature.

        Only selected scalar phases are integrated. This does not expand the
        native MRzero point cache or pretend to validate unknown field gradients.
        Native center histories retain their separate resummation checks.
        """
        for value, name, lower in ((sample_count,'sample_count',1),
                                   (point_count,'point_count',1),
                                   (quadrature_order,'quadrature_order',2)):
            if isinstance(value,bool) or int(value)!=value or value<lower:
                raise ValueError(f'{name} must be an integer >= {lower}.')
        currents = self._check_currents(np.zeros((self.nc,self.nstates))
                                       if currents_A is None else currents_A)
        samples = np.unique(np.linspace(0,self.nsamples-1,min(int(sample_count),self.nsamples),dtype=int))
        points = np.unique(np.linspace(0,int(self.support.sum())-1,
                            min(int(point_count),int(self.support.sum())),dtype=int))
        integrated = self.numpy(self._voxel_integrated_currents(currents,samples,None))
        edge = self._voxel_numpy['edge_vectors_m']
        b0gradient = self._voxel_numpy['b0_gradient_Hz_per_pixel'][self.support][points]
        fieldgradient = self._voxel_numpy['field_gradient_Hz_per_A_per_pixel'][self.support][points]
        k = np.stack((self.numpy(self._kx)[samples],self.numpy(self._ky)[samples]))
        elapsed = self.numpy(self._evolution)[samples]
        phases = [((edge[axis] @ k)[None] + b0gradient[:,axis,None]*elapsed[None]
                   + fieldgradient[:,axis] @ integrated) for axis in (0,1)]
        maximum = max(float(np.max(abs(q))) for q in phases)
        required = int(np.ceil(2*np.pi*maximum))+16
        order = min(512,max(int(quadrature_order),required))
        nodes,weights = np.polynomial.legendre.leggauss(order)
        integrals = [np.sum(np.exp(-2j*np.pi*q[...,None]*(nodes/2))*(weights/2),axis=-1)
                     for q in phases]
        expected = integrals[0]*integrals[1]
        analytic = self.numpy(self._voxel_factors(currents,samples))[points]
        error = float(np.max(abs(expected-analytic)))
        tolerance = 1e-8 if self.dtype==torch.float64 else 1e-5
        return dict(passed=bool(error<=tolerance and order>=required),
            max_abs_factor_error=error,tolerance=tolerance,sample_count=len(samples),
            point_count=len(points),quadrature_order=order,required_quadrature_order=required,
            quadrature_order_capped=order<required,max_abs_cycles_per_pixel=maximum,
            sample_indices=samples.tolist(),point_indices=points.tolist(),
            oracle='independent Gauss-Legendre integral of exp(-2*pi*i*q*u), u in [-1/2,1/2], one factor per axis',
            native_point_cache_expansion=1)


__all__ = ['FiniteVoxelEchoACDCModel', 'normalized_sinc_and_derivative']
