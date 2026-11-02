"""Independent receiver encoding, RSS adjoints and native spiral parity."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiment_multiecho import MultiEchoVoxelACDCModel
from experiment_spiral import SpiralReceiverACDCModel
from test_experiment_multiecho import fixture as voxel_fixture
from test_acdc_gpu import event_response
from acdc_gpu import ACDCModel


def fixture(receivers=3, *, turns=3, children=2):
    experiment, calibration = voxel_fixture(children=children, width=170.)
    acq = experiment['acquisition']
    samples = 4 * turns
    starts = np.linspace(0., .004, turns, endpoint=False)
    ends = np.r_[starts[1:], .004]
    states = np.repeat(np.arange(turns), 4)
    time = starts[states] + (ends[states] - starts[states]) * np.tile([.15, .35, .6, .85], turns)
    angle = np.linspace(0, 2*np.pi*turns, samples, endpoint=False)
    radius = np.linspace(.1, 30., samples)
    acq.update(sequence_kind='spiral', image_shape=(5, 6),
        kx_mInv=(radius*np.cos(angle))[None], ky_mInv=(radius*np.sin(angle))[None],
        time_s=time[None], line_index=(states+1)[None],
        rf_time_s=np.zeros((1, samples)), phase_offset_rad=np.zeros((1, samples)),
        line_start_s=starts, line_end_s=ends)
    if receivers is not None:
        rng = np.random.default_rng(351)
        shape = (receivers,) + experiment['m0_proxy'].shape
        experiment['receive_weights'] = (rng.uniform(.2, 1., shape)
                                          * np.exp(1j*rng.uniform(-np.pi, np.pi, shape)))
    return experiment, calibration


def model(experiment, calibration, **options):
    return SpiralReceiverACDCModel(experiment, calibration,
        profile='cpu', precision='float64', sample_chunk=5, **options)


def receiver_quadrature(experiment, calibration, commands, *, b0=True, response=True):
    """Explicit displaced-spin Bloch sum and ODE response, without sinc."""
    acq = experiment['acquisition']
    active = experiment['field']['simulation_mask'] & (experiment['m0_proxy'] > 0)
    metadata = experiment['inplane_voxel']
    nodes, weights = np.polynomial.legendre.leggauss(20)
    u, v = np.meshgrid(nodes/2, nodes/2, indexing='ij')
    wu, wv = np.meshgrid(weights/2, weights/2, indexing='ij')
    u, v, volume = u.ravel(), v.ravel(), (wu*wv).ravel()
    edges = metadata['edge_vectors_m']
    x = experiment['geometry']['X_m'][active, None] + u*edges[0, 0] + v*edges[1, 0]
    y = experiment['geometry']['Y_m'][active, None] + u*edges[0, 1] + v*edges[1, 1]
    gradients = metadata['field_gradient_Hz_per_A_per_pixel'][active]
    maps = (calibration['acdc_maps'][active, None]
             + u[None, :, None]*gradients[:, None, 0]
             + v[None, :, None]*gradients[:, None, 1])
    static = (experiment['field']['b0Hz'][active, None]
               + u*metadata['b0_gradient_Hz_per_pixel'][active, 0, None]
               + v*metadata['b0_gradient_Hz_per_pixel'][active, 1, None])
    if response:
        _, integral = event_response(commands, acq['line_start_s'], acq['line_end_s'],
            acq['time_s'], experiment['current_response']['tau_s'])
    else:
        integral = np.array([[sum(commands[c, s]*max(0., min(t, end)-start)
            for s, (start, end) in enumerate(zip(acq['line_start_s'], acq['line_end_s'])))
            for c in range(commands.shape[0])] for t in acq['time_s'].ravel()])
    receive = experiment['receive_weights'][:, active]
    receive = receive / np.linalg.norm(receive, axis=0)[None]
    rate = 1 / experiment['relaxation']['T2star_s'][active]
    width = experiment['multi_echo_coherence']['width_Hz'][active]
    kernel = experiment['multi_echo_coherence']['dispersion_kind'][active]
    signal = np.empty((len(receive), acq['time_s'].size), complex)
    for sample, t in enumerate(acq['time_s'].ravel()):
        cycles = x*acq['kx_mInv'].ravel()[sample] + y*acq['ky_mInv'].ravel()[sample]
        if b0:
            cycles = cycles + static*t
        cycles = cycles + np.einsum('pqc,c->pq', maps, integral[sample])
        coherence = np.where(kernel == 1, np.sinc(width*t),
            np.where(kernel == 2, np.exp(-2*np.pi**2*(width*t)**2), 1.))
        amplitude = (experiment['m0_proxy'][active]
            * np.exp(-(t-experiment['amplitude_reference_time_s'])*rate)*coherence)
        point = np.sum(volume[None]*np.exp(-2j*np.pi*cycles), axis=-1)*amplitude
        signal[:, sample] = receive @ point
    return signal.reshape((len(receive),) + acq['time_s'].shape)


class SpiralReceiverTests(unittest.TestCase):
    def test_receiver_signals_match_independent_displaced_spin_quadrature(self):
        experiment, calibration = fixture(receivers=3)
        experiment['multi_echo_coherence']['dispersion_kind'][:, 0] = 2
        currents = np.random.default_rng(352).normal(size=(32, 3))*.04
        with patch.object(ACDCModel, '_make_base', side_effect=AssertionError('native MRzero required')):
            child = model(experiment, calibration, receive_chunk=2)
            for response in (False, True):
                for b0 in (False, True):
                    receiver_images, actual = child.receiver_forward(currents,
                        response_enabled=response, b0_enabled=b0)
                    expected = receiver_quadrature(experiment, calibration, currents,
                        response=response, b0=b0)
                    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=6e-6)
                    images = child.reconstruction.numpy_image(expected)
                    np.testing.assert_allclose(receiver_images, images, rtol=5e-5, atol=6e-6)
                    combined, data = child.forward(currents, response_enabled=response, b0_enabled=b0)
                    np.testing.assert_allclose(data, actual, rtol=0, atol=0)
                    np.testing.assert_allclose(combined, np.linalg.norm(receiver_images, axis=0),
                                               rtol=2e-14, atol=2e-14)
        self.assertEqual(child.runtime_info['receive_model']['native_point_cache_expansion'], 1)
        self.assertFalse(child.runtime_info['receive_model']['receiver_point_time_cache'])

    def test_current_gradient_includes_complex_receiver_rss_and_box_derivatives(self):
        for common_tau in (False, True):
            experiment, calibration = fixture(receivers=4, children=3)
            experiment['multi_echo_coherence']['dispersion_kind'][:] = [0, 1, 2]
            if common_tau:
                experiment['current_response']['tau_s'] = 40e-6
            child = model(experiment, calibration, receive_chunk=2)
            child._image_target_weights = child.tensor(np.random.default_rng(353).uniform(.2, 1, child.image_size))
            currents = np.random.default_rng(354).normal(size=(32, 3))*.04
            direction = np.random.default_rng(355).normal(size=currents.shape)
            direction /= np.linalg.norm(direction)
            _, gradient = child.objective_and_grad(currents)
            epsilon = 1e-5
            difference = (child.objective_and_grad(currents+epsilon*direction)[0]
                - child.objective_and_grad(currents-epsilon*direction)[0])/(2*epsilon)
            np.testing.assert_allclose(np.sum(gradient*direction), difference,
                                       rtol=5e-6, atol=3e-9)

    def test_default_single_receiver_preserves_original_images_signals_and_objective(self):
        experiment, calibration = fixture(receivers=None)
        original = MultiEchoVoxelACDCModel(experiment, calibration,
            profile='cpu', precision='float64', sample_chunk=5)
        child = model(experiment, calibration)
        currents = np.random.default_rng(356).normal(size=(32, 3))*.03
        for actual, expected in zip(child.forward(currents), original.forward(currents)):
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        for actual, expected in zip(child.objective_and_grad(currents), original.objective_and_grad(currents)):
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        self.assertEqual(child.forward(currents)[1].shape, (1, 12))
        images, data = child.receiver_forward(currents)
        self.assertEqual(data.shape, (1, 1, 12))
        self.assertEqual(images.shape, (1, 5, 6))

    def test_receive_norm_does_not_count_rss_amplitude_twice(self):
        experiment, calibration = fixture(receivers=3)
        experiment['receive_weights'][:] = np.array([1, 2j, -3])[:, None, None]
        child = model(experiment, calibration)
        original_input = deepcopy(experiment)
        del original_input['receive_weights']
        original = model(original_input, calibration)
        currents = np.zeros((32, 3))
        np.testing.assert_allclose(child.forward(currents)[0], abs(original.forward(currents)[0]),
                                   rtol=2e-13, atol=2e-13)
        experiment['receive_weights'] *= 123.
        scaled = model(experiment, calibration)
        for actual, expected in zip(scaled.forward(currents), child.forward(currents)):
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_receiver_chunking_and_native_cpu_cache_do_not_change_physics(self):
        experiment, calibration = fixture(receivers=5)
        cached = model(experiment, calibration, receive_chunk=5)
        streamed = model(experiment, calibration, receive_chunk=1, memory_fraction=1e-8)
        self.assertIsNotNone(cached._base_cache)
        self.assertIsNone(streamed._base_cache)
        currents = np.random.default_rng(357).normal(size=(32, 3))*.04
        for actual, expected in zip(streamed.forward(currents), cached.forward(currents)):
            np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=6e-6)
        value, gradient = streamed.objective_and_grad(currents)
        expected_value, expected_gradient = cached.objective_and_grad(currents)
        np.testing.assert_allclose(value, expected_value, rtol=5e-5, atol=6e-6)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=8e-5, atol=6e-6)

    def test_all_65_turns_and_disabled_channel_are_retained(self):
        experiment, calibration = fixture(receivers=2, turns=65, children=1)
        limits = np.full(32, 3.)
        limits[8] = 0.
        child = model(experiment, calibration, max_current_A=limits)
        self.assertEqual(child.nstates, 65)
        self.assertEqual(child.forward(np.zeros((32, 65)))[1].shape, (2, 1, 260))
        currents = np.random.default_rng(358).normal(size=(32, 65))
        projected = child.project_currents(currents, 30.)
        np.testing.assert_array_equal(projected[8], np.zeros(65))
        self.assertEqual(child.runtime_info['command_state_unit'], 'spiral turn')

    def test_zero_rss_pixels_have_finite_generalized_derivative(self):
        experiment, calibration = fixture(receivers=2)
        child = model(experiment, calibration)
        zero_images = torch.zeros((2,) + child.image_size, dtype=child.complex_dtype)
        zero_k = torch.zeros((2,) + child.sampling_shape, dtype=child.complex_dtype)
        with patch.object(child, '_receiver_forward_tensor', return_value=(zero_images, zero_k)):
            value, gradient = child.objective_and_grad(np.zeros((32, 3)))
        self.assertTrue(np.isfinite(value))
        np.testing.assert_array_equal(gradient, np.zeros((32, 3)))

    def test_malformed_or_missing_supported_receive_weights_are_rejected(self):
        for wrong in (np.zeros((3, 7)), np.full((3, 11, 2), np.nan),
                      np.zeros((3, 11, 2))):
            experiment, calibration = fixture()
            experiment['receive_weights'] = wrong
            with self.assertRaisesRegex(ValueError, 'receive_weights'):
                model(experiment, calibration)
        experiment, calibration = fixture()
        for chunk in (True, 0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, 'receive_chunk'):
                model(experiment, calibration, receive_chunk=chunk)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
