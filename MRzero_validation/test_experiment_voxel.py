"""Independent finite-box quadrature, exact gradients and corner constraints."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'Experiment'))
from acdc_gpu import ACDCModel
from continuous_reconstruction import ContinuousReconstruction
from experiment_echo import EchoSubspinACDCModel
from experiment_voxel import FiniteVoxelEchoACDCModel, normalized_sinc_and_derivative
from inplane_geometry import FrameReconstruction
from test_acdc_gpu import event_response
from test_experiment_subspins import expanded_fixture


def voxel_fixture(children=2):
    _, _, experiment, calibration = expanded_fixture(children=children, in_plane=True)
    experiment['amplitude_reference_time_s'] = .0007
    rng = np.random.default_rng(319)
    shape = experiment['m0_proxy'].shape
    experiment['inplane_voxel'] = {
        'edge_vectors_m': np.array([[.003, .011], [.013, -.002]]),
        'b0_gradient_Hz_per_pixel': rng.normal(size=shape + (2,)) * 110,
        'field_gradient_Hz_per_A_per_pixel': rng.normal(size=shape + (2, 32)) * 220,
    }
    return experiment, calibration


def model(experiment, calibration, **options):
    return FiniteVoxelEchoACDCModel(experiment, calibration, profile='cpu',
        precision='float64', sample_chunk=5, **options)


def quadrature_signal(experiment, calibration, commands, *, b0_enabled=True,
                      response_enabled=True, order=20):
    """Dense Gauss-Legendre sum of explicit displaced spins, without sinc.

    This oracle integrates actual offset XYZ, B0 and ACDC values and computes
    the temporal response independently through the event-ODE helper.
    """
    acq = experiment['acquisition']
    active = experiment['field']['simulation_mask'] & (experiment['m0_proxy'] > 0)
    metadata = experiment['inplane_voxel']
    edges = metadata['edge_vectors_m']
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u, v = np.meshgrid(nodes/2, nodes/2, indexing='ij')
    wu, wv = np.meshgrid(weights/2, weights/2, indexing='ij')
    u, v, volume = u.ravel(), v.ravel(), (wu*wv).ravel()
    x = experiment['geometry']['X_m'][active, None] + u * edges[0, 0] + v * edges[1, 0]
    y = experiment['geometry']['Y_m'][active, None] + u * edges[0, 1] + v * edges[1, 1]
    b0 = (experiment['field']['b0Hz'][active, None]
          + metadata['b0_gradient_Hz_per_pixel'][active, 0, None] * u
          + metadata['b0_gradient_Hz_per_pixel'][active, 1, None] * v)
    field_gradient = metadata['field_gradient_Hz_per_A_per_pixel'][active]
    maps = (calibration['acdc_maps'][active, None, :]
            + u[None, :, None] * field_gradient[:, None, 0, :]
            + v[None, :, None] * field_gradient[:, None, 1, :])
    amplitude = experiment['m0_proxy'][active, None]
    rate = 1/experiment['relaxation']['T2star_s'][active, None]
    elapsed = np.asarray(acq['time_s']) - np.asarray(acq['rf_time_s'])
    if response_enabled:
        _, integrated = event_response(commands, acq['line_start_s'], acq['line_end_s'],
                                        acq['time_s'], experiment['current_response']['tau_s'])
        _, before_rf = event_response(commands, acq['line_start_s'], acq['line_end_s'],
                                       acq['rf_time_s'], experiment['current_response']['tau_s'])
        integrated -= before_rf
    else:
        integrated = np.asarray([[sum(commands[channel, state] * max(0., min(t, end)-max(rf, start))
            for state, (start, end) in enumerate(zip(acq['line_start_s'], acq['line_end_s'])))
            for channel in range(commands.shape[0])]
            for t, rf in zip(acq['time_s'].ravel(), acq['rf_time_s'].ravel())])
    signal = np.empty(elapsed.shape, complex)
    for sample, t in enumerate(elapsed.ravel()):
        index = np.unravel_index(sample, elapsed.shape)
        cycles = x * acq['kx_mInv'][index] + y * acq['ky_mInv'][index]
        if b0_enabled:
            cycles = cycles + b0 * t
        cycles = cycles + np.einsum('pqc,c->pq', maps, integrated[sample])
        decay = np.exp(-(t-experiment['amplitude_reference_time_s']) * rate)
        signal[index] = np.sum(amplitude * volume * decay * np.exp(-2j*np.pi*cycles)
                               * np.exp(1j*acq['phase_offset_rad'][index]))
    return signal


class FiniteVoxelTests(unittest.TestCase):
    def test_normalized_sinc_and_derivative_at_zero_and_roots(self):
        for dtype in (torch.float32, torch.float64):
            q = torch.tensor([0., 1e-8, -1e-5, .015, -.015, 1., -1., 2.], dtype=dtype)
            value, derivative = normalized_sinc_and_derivative(q)
            self.assertTrue(torch.isfinite(value).all() and torch.isfinite(derivative).all())
            np.testing.assert_allclose(value.numpy(), np.sinc(q.numpy()), rtol=2e-6, atol=1e-7)
            self.assertEqual(float(value[0]), 1.)
            self.assertEqual(float(derivative[0]), 0.)
            np.testing.assert_allclose(derivative[[5,6,7]].numpy(), [-1.,1.,.5], rtol=1e-6)
            small = q[1:3].numpy()
            np.testing.assert_allclose(derivative[1:3].numpy(), -np.pi**2*small/3, rtol=2e-6)

    def test_signal_matches_independent_dense_displaced_spin_quadrature(self):
        experiment, calibration = voxel_fixture(children=3)
        currents = np.random.default_rng(320).normal(size=(32,3)) * .045
        with patch.object(ACDCModel, '_make_base', side_effect=AssertionError('analytic baseline forbidden')):
            child = model(experiment, calibration)
            for response in (True, False):
                for b0 in (True, False):
                    actual_image, actual_kdata = child.forward(currents, response_enabled=response, b0_enabled=b0)
                    expected = quadrature_signal(experiment, calibration, currents,
                        b0_enabled=b0, response_enabled=response)
                    np.testing.assert_allclose(actual_kdata, expected, rtol=4e-5, atol=5e-6)
                    expected_image = child.reconstruction.numpy_image(expected)
                    np.testing.assert_allclose(actual_image, expected_image, rtol=4e-5, atol=4e-6)
        self.assertGreater(child.runtime_info['native_blocks_simulated'], 0)
        self.assertEqual(child.runtime_info['inplane_voxel']['native_point_cache_expansion'], 1)
        self.assertEqual(child._base_cache.shape[0], int(child.support.sum()))

    def test_current_gradient_includes_field_gradient_and_unequal_response_taus(self):
        experiment, calibration = voxel_fixture(children=3)
        child = model(experiment, calibration)
        rng = np.random.default_rng(321)
        currents = rng.normal(size=(32,3)) * .05
        direction = rng.normal(size=currents.shape)
        direction /= np.linalg.norm(direction)
        _, gradient = child.objective_and_grad(currents)
        epsilon = 1e-5
        difference = (child.objective_and_grad(currents+epsilon*direction)[0]
                      - child.objective_and_grad(currents-epsilon*direction)[0])/(2*epsilon)
        np.testing.assert_allclose(np.sum(gradient*direction), difference, rtol=3e-6, atol=2e-9)
        # The independent dense-spin oracle also checks the objective derivative.
        ideal = quadrature_signal(experiment, calibration, np.zeros_like(currents), b0_enabled=False)
        reference = np.abs(child.reconstruction.numpy_image(ideal))
        def oracle_loss(candidate):
            signal = quadrature_signal(experiment, calibration, candidate)
            image = np.abs(child.reconstruction.numpy_image(signal))
            return np.sum((image-reference)**2)/np.sum(reference**2)
        oracle_derivative = (oracle_loss(currents+epsilon*direction)-oracle_loss(currents-epsilon*direction))/(2*epsilon)
        np.testing.assert_allclose(np.sum(gradient*direction), oracle_derivative, rtol=8e-5, atol=2e-6)

    def test_sinc_zero_does_not_erase_its_nonzero_current_derivative(self):
        experiment, calibration = voxel_fixture()
        experiment['acquisition']['kx_mInv'][:] = 0
        experiment['acquisition']['ky_mInv'][:] = 0
        gradient = experiment['inplane_voxel']['b0_gradient_Hz_per_pixel']
        gradient[...,0] = 1/experiment['acquisition']['time_s'].ravel()[0]
        gradient[...,1] = 0
        child = model(experiment, calibration)
        zero = child.tensor(np.zeros((32,3)))
        factor, drow, _ = child._voxel_factors(zero, slice(0,1), derivatives=True)
        np.testing.assert_allclose(factor.numpy(), 0., atol=2e-15)
        np.testing.assert_allclose(drow.numpy(), -1., atol=2e-15)
        _, derivative = child.objective_and_grad(np.zeros((32,3)))
        self.assertTrue(np.isfinite(derivative).all())
        self.assertGreater(np.linalg.norm(derivative), 1e-5)

    def test_common_response_tau_gradient_and_bounded_quadrature_validation(self):
        experiment,calibration=voxel_fixture()
        experiment['current_response']['tau_s']=30e-6
        child=model(experiment,calibration)
        rng=np.random.default_rng(328)
        currents=rng.normal(size=(32,3))*.04
        direction=rng.normal(size=currents.shape);direction/=np.linalg.norm(direction)
        _,gradient=child.objective_and_grad(currents)
        epsilon=1e-5
        difference=(child.objective_and_grad(currents+epsilon*direction)[0]
                    -child.objective_and_grad(currents-epsilon*direction)[0])/(2*epsilon)
        np.testing.assert_allclose(np.sum(gradient*direction),difference,rtol=3e-6,atol=2e-9)
        check=child.validate_voxel_quadrature(currents,sample_count=5,point_count=7,quadrature_order=12)
        self.assertTrue(check['passed'],check)
        self.assertEqual(check['sample_count'],5)
        self.assertEqual(check['point_count'],7)
        self.assertLess(check['max_abs_factor_error'],1e-10)
        self.assertEqual(check['native_point_cache_expansion'],1)
        self.assertTrue(child.runtime_info['inplane_voxel']['effective_R2star_overlap_possible'])
        # The notebook check compares the implemented factor, not two copies of
        # the same closed-form expression, and therefore detects a bad factor.
        with patch.object(child,'_voxel_factors',return_value=child.tensor(np.zeros((int(child.support.sum()),5)))):
            self.assertFalse(child.validate_voxel_quadrature(currents,5,7)['passed'])

    def test_joint_frame_rotation_and_mirror_preserve_signal_and_gradient(self):
        experiment, calibration = voxel_fixture()
        original = model(experiment, calibration)
        angle = np.deg2rad(37.)
        frame = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]) @ np.diag([-1.,1.])
        transformed, maps = deepcopy(experiment), deepcopy(calibration)
        for container in (transformed['geometry'],maps):
            points = frame @ np.stack([container['X_m'].ravel(),container['Y_m'].ravel()])
            container['X_m']=points[0].reshape(container['X_m'].shape)
            container['Y_m']=points[1].reshape(container['Y_m'].shape)
        acq = transformed['acquisition']
        k = frame @ np.stack([acq['kx_mInv'].ravel(),acq['ky_mInv'].ravel()])
        acq['kx_mInv']=k[0].reshape(acq['kx_mInv'].shape)
        acq['ky_mInv']=k[1].reshape(acq['ky_mInv'].shape)
        transformed['inplane_voxel']['edge_vectors_m'] @= frame.T
        framed = model(transformed,maps,reconstruction_factory=lambda a,d,t:FrameReconstruction(a,d,t,frame_matrix=frame))
        currents=np.random.default_rng(322).normal(size=(32,3))*.04
        for actual,expected in zip(framed.forward(currents),original.forward(currents)):
            np.testing.assert_allclose(actual,expected,rtol=5e-5,atol=4e-6)
        value,gradient=framed.objective_and_grad(currents)
        expected_value,expected_gradient=original.objective_and_grad(currents)
        np.testing.assert_allclose(value,expected_value,rtol=5e-5,atol=3e-6)
        np.testing.assert_allclose(gradient,expected_gradient,rtol=8e-5,atol=4e-6)

    def test_zero_q_normalization_and_no_extra_native_dephasing(self):
        signals=[]
        for count in (1,2,5):
            experiment,calibration=voxel_fixture(children=count)
            experiment['field']['b0Hz'][:]=0
            experiment['relaxation']['T2star_s'][:]=np.inf
            experiment['inplane_voxel']['b0_gradient_Hz_per_pixel'][:]=0
            for key in ('kx_mInv','ky_mInv','phase_offset_rad'):
                experiment['acquisition'][key][:]=0
            child=model(experiment,calibration)
            _,signal=child.forward(np.zeros((32,3)))
            np.testing.assert_allclose(signal,experiment['m0_proxy'].sum(),rtol=3e-6,atol=3e-6)
            signals.append(signal)
            point=EchoSubspinACDCModel(experiment,calibration,profile='cpu',precision='float64',sample_chunk=5)
            np.testing.assert_allclose(signal,point.forward(np.zeros((32,3)))[1],rtol=1e-12,atol=1e-12)
        for signal in signals[1:]:
            np.testing.assert_allclose(signal,signals[0],rtol=3e-6,atol=3e-6)

    def test_corner_field_limits_detect_violation_hidden_at_centers(self):
        experiment,calibration=voxel_fixture()
        calibration['acdc_maps'][:]=.1
        calibration['acdc_maps'][...,0]=1e-6
        gradients=experiment['inplane_voxel']['field_gradient_Hz_per_A_per_pixel']
        gradients[:]=0
        gradients[...,0,0]=100
        child=model(experiment,calibration)
        currents=np.zeros((32,3));currents[0]=1
        self.assertLess(np.max(abs(child.maps_numpy[child.quality_mask]@currents)),1.)
        self.assertGreater(float((child.B_quality@child.tensor(currents)).abs().max()),49.)
        self.assertFalse(child.hardware.diagnostics(currents,30.,10.)['checks']['command_field'])
        diagnostics=child.response_diagnostics(currents,10.,30.)
        self.assertFalse(diagnostics['actual_adc_field_limits_met'])
        projected=child.project_currents(currents,30.,10.)
        self.assertTrue(child.hardware.diagnostics(projected,30.,10.)['all_limits_met'])
        self.assertEqual(child.B_quality.shape[0],4*int(child.quality_mask.sum()))
        torch.testing.assert_close(child.hardware.positive_field_basis,child.B_quality.clamp_min(0))
        torch.testing.assert_close(child.hardware.negative_field_basis,child.B_quality.clamp_max(0))

    def test_missing_or_malformed_metadata_is_rejected(self):
        experiment,calibration=voxel_fixture()
        del experiment['inplane_voxel']
        with self.assertRaisesRegex(ValueError,'inplane_voxel'):
            model(experiment,calibration)
        for key,value in (('edge_vectors_m',np.zeros((2,2))),
                          ('b0_gradient_Hz_per_pixel',np.zeros((2,2))),
                          ('field_gradient_Hz_per_A_per_pixel',np.full((1,1,2,32),np.nan))):
            experiment,calibration=voxel_fixture()
            experiment['inplane_voxel'][key]=value
            with self.assertRaisesRegex(ValueError,'inplane_voxel'):
                model(experiment,calibration)

    def test_optional_calibrated_corners_add_constraints_beyond_linear_model(self):
        experiment,calibration=voxel_fixture()
        metadata=experiment['inplane_voxel']
        metadata['field_gradient_Hz_per_A_per_pixel'][:]=0
        calibration['acdc_maps'][:]=.1
        corners=np.repeat(calibration['acdc_maps'][...,None,:],4,axis=-2)
        corners[...,2,0]=100
        metadata['field_at_corners_Hz_per_A']=corners
        child=model(experiment,calibration)
        currents=np.zeros((32,3));currents[0]=1
        self.assertEqual(child.B_quality.shape[0],8*int(child.quality_mask.sum()))
        self.assertTrue(child.runtime_info['inplane_voxel']['calibrated_corner_limits_included'])
        self.assertFalse(child.hardware.diagnostics(currents,30.,10.)['all_limits_met'])
        projected=child.project_currents(currents,30.,10.)
        self.assertTrue(child.hardware.diagnostics(projected,30.,10.)['all_limits_met'])
        self.assertLess(np.max(abs(projected)),.101)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
