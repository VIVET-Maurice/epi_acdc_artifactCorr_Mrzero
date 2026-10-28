"""Native multi-echo coherence, exact current gradients and input propagation."""

from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiment_multiecho import MultiEchoVoxelACDCModel
from experiment_voxel import FiniteVoxelEchoACDCModel
from echo_shim_inputs import prepare_echo_shim_inputs
from test_experiment_voxel import voxel_fixture, quadrature_signal
from test_echo_shim_inputs import fixture as input_fixture
from subspin_sampling import _load_source
from acdc_gpu import ACDCModel


def fixture(children=2, width=120., kind=1):
    experiment, calibration = voxel_fixture(children=children)
    shape = experiment['m0_proxy'].shape
    experiment['multi_echo_coherence'] = {
        'width_Hz': np.full(shape, width),
        'dispersion_kind': np.full(shape, kind, dtype=np.int8),
    }
    return experiment, calibration


def model(experiment, calibration, **options):
    return MultiEchoVoxelACDCModel(experiment, calibration,
        profile='cpu', precision='float64', sample_chunk=5, **options)


class MultiEchoModelTests(unittest.TestCase):
    def test_uniform_width_multiplies_native_quadrature_and_preserves_signed_null(self):
        experiment, calibration = fixture(width=350.)
        elapsed = np.asarray(experiment['acquisition']['time_s']) - experiment['acquisition']['rf_time_s']
        factor = np.sinc(350 * elapsed)
        self.assertTrue((factor < 0).any())
        currents = np.random.default_rng(846).normal(size=(32, 3)) * .04
        with patch.object(ACDCModel, '_make_base', side_effect=AssertionError('analytic baseline forbidden')):
            child = model(experiment, calibration)
            for response in (True, False):
                for b0 in (True, False):
                    actual_image, actual_signal = child.forward(currents,
                        response_enabled=response, b0_enabled=b0)
                    expected = quadrature_signal(experiment, calibration, currents,
                        response_enabled=response, b0_enabled=b0) * factor
                    np.testing.assert_allclose(actual_signal, expected, rtol=5e-5, atol=5e-6)
                    np.testing.assert_allclose(actual_image, child.reconstruction.numpy_image(expected),
                                               rtol=5e-5, atol=4e-6)
        self.assertEqual(child.runtime_info['multi_echo_coherence']['native_point_cache_expansion'], 1)

    def test_gaussian_width_matches_independent_frequency_integral(self):
        experiment, calibration = fixture(width=50., kind=2)
        child = model(experiment, calibration)
        elapsed = child.numpy(child._evolution)
        # Independent Gauss-Hermite average over N(0, sigma**2), without using
        # the closed form in the model under test.
        nodes, weights = np.polynomial.hermite.hermgauss(80)
        expected = np.sum(np.exp(-2j * np.pi * elapsed[:, None] * np.sqrt(2) * 50 * nodes)
                          * weights / np.sqrt(np.pi), axis=-1)
        np.testing.assert_allclose(child.numpy(child._coherence_factor(slice(None))),
            np.broadcast_to(expected.real, (int(child.support.sum()), len(elapsed))),
            rtol=1e-11, atol=2e-12)
        zero = np.zeros((32, 3))
        expected_signal = quadrature_signal(experiment, calibration, zero) * expected.reshape(child.image_size)
        np.testing.assert_allclose(child.forward(zero)[1], expected_signal, rtol=5e-5, atol=5e-6)

    def test_mixed_child_kernels_include_mono_and_do_not_infer_a_spatial_axis(self):
        experiment, calibration = fixture(children=3, width=170.)
        experiment['multi_echo_coherence']['dispersion_kind'][:] = [0, 1, 2]
        child = model(experiment, calibration)
        actual = child.numpy(child._coherence_factor(slice(None))).reshape(-1, 3, child.nsamples)
        time = child.numpy(child._evolution)
        np.testing.assert_allclose(actual[:, 0], 1.)
        np.testing.assert_allclose(actual[:, 1], np.broadcast_to(np.sinc(170*time), actual[:, 1].shape))
        np.testing.assert_allclose(actual[:, 2], np.broadcast_to(np.exp(-2*np.pi**2*(170*time)**2), actual[:, 2].shape))
        self.assertIn('no through-plane', child.runtime_info['multi_echo_coherence']['spatial_assignment'])

    def test_zero_width_or_mono_exactly_preserves_finite_voxel_model(self):
        for width, kind in ((0., 1), (0., 2), (90., 0)):
            experiment, calibration = fixture(width=width, kind=kind)
            expected = FiniteVoxelEchoACDCModel(experiment, calibration,
                profile='cpu', precision='float64', sample_chunk=5)
            actual = model(experiment, calibration)
            currents = np.random.default_rng(847).normal(size=(32, 3)) * .03
            for result, reference in zip(actual.forward(currents), expected.forward(currents)):
                np.testing.assert_allclose(result, reference, rtol=1e-13, atol=1e-13)
            for result, reference in zip(actual.objective_and_grad(currents), expected.objective_and_grad(currents)):
                np.testing.assert_allclose(result, reference, rtol=1e-13, atol=1e-13)

    def test_zero_b0_target_retains_unresolved_coherence_and_reference_uses_it(self):
        experiment, calibration = fixture(width=210.)
        child = model(experiment, calibration)
        zero = np.zeros((32, 3))
        baseline = FiniteVoxelEchoACDCModel(experiment, calibration,
            profile='cpu', precision='float64', sample_chunk=5)
        expected = baseline.ideal_kdata * np.sinc(210 * child.numpy(child._evolution).reshape(child.image_size))
        np.testing.assert_allclose(child.ideal_kdata, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(child.forward(zero, b0_enabled=False)[1], expected, rtol=1e-12, atol=1e-12)
        self.assertGreater(np.linalg.norm(child.ideal_kdata - baseline.ideal_kdata), .01)

    def test_joint_current_gradient_with_both_kernel_families_and_response_modes(self):
        for kind in (1, 2):
            experiment, calibration = fixture(children=3, width=170., kind=kind)
            for common_tau in (False, True):
                if common_tau:
                    experiment['current_response']['tau_s'] = 30e-6
                child = model(experiment, calibration)
                rng = np.random.default_rng(848)
                currents = rng.normal(size=(32, 3)) * .04
                direction = rng.normal(size=currents.shape)
                direction /= np.linalg.norm(direction)
                _, gradient = child.objective_and_grad(currents)
                epsilon = 1e-5
                difference = (child.objective_and_grad(currents + epsilon*direction)[0]
                              - child.objective_and_grad(currents - epsilon*direction)[0]) / (2*epsilon)
                np.testing.assert_allclose(np.sum(gradient*direction), difference, rtol=4e-6, atol=2e-9)

    def test_unresolved_sinc_root_does_not_make_current_gradient_nonfinite(self):
        experiment, calibration = fixture()
        first_time = np.asarray(experiment['acquisition']['time_s']).ravel()[0]
        experiment['multi_echo_coherence']['width_Hz'][:] = 1 / first_time
        child = model(experiment, calibration)
        np.testing.assert_allclose(child.numpy(child._coherence_factor(slice(0, 1))), 0, atol=2e-15)
        _, gradient = child.objective_and_grad(np.zeros((32, 3)))
        self.assertTrue(np.isfinite(gradient).all())

    def test_inplane_sinc_root_retains_current_derivative_with_unresolved_coherence(self):
        experiment, calibration = fixture(width=170.)
        experiment['acquisition']['kx_mInv'][:] = 0
        experiment['acquisition']['ky_mInv'][:] = 0
        gradient = experiment['inplane_voxel']['b0_gradient_Hz_per_pixel']
        gradient[..., 0] = 1 / experiment['acquisition']['time_s'].ravel()[0]
        gradient[..., 1] = 0
        child = model(experiment, calibration)
        zero = np.zeros((32, 3))
        factor, drow, _ = child._voxel_factors(child.tensor(zero), slice(0, 1), derivatives=True)
        np.testing.assert_allclose(child.numpy(factor), 0, atol=2e-15)
        np.testing.assert_allclose(child.numpy(drow), -1, atol=2e-15)
        _, derivative = child.objective_and_grad(zero)
        self.assertTrue(np.isfinite(derivative).all())
        self.assertGreater(np.linalg.norm(derivative), 1e-5)
        direction = np.random.default_rng(849).normal(size=zero.shape)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-5
        difference = (child.objective_and_grad(zero+epsilon*direction)[0]
                      - child.objective_and_grad(zero-epsilon*direction)[0])/(2*epsilon)
        np.testing.assert_allclose(np.sum(derivative*direction), difference, rtol=4e-6, atol=2e-9)

    def test_cpu_native_cache_is_unmodified_and_device_cache_has_factor_once(self):
        experiment, calibration = fixture(width=180.)
        cached = model(experiment, calibration)
        uncached = model(experiment, calibration, memory_fraction=1e-8)
        self.assertIsNotNone(cached._base_cache)
        self.assertIsNone(uncached._base_cache)
        self.assertTrue(uncached._native_cpu_blocks)
        zero = np.zeros((32, 3))
        original = {key: value.clone() for key, value in uncached._native_cpu_blocks.items()}
        first = uncached.forward(zero)[1]
        for _ in range(2):
            np.testing.assert_allclose(uncached.forward(zero)[1], first, rtol=0, atol=0)
        for key, value in uncached._native_cpu_blocks.items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            selected = slice(*key)
            torch.testing.assert_close(uncached._make_base(selected),
                value.to(uncached.device) * uncached._coherence_factor(selected), rtol=0, atol=0)
        # Changing the native chunk changes float32 accumulation slightly.
        np.testing.assert_allclose(first, cached.forward(zero)[1], rtol=4e-5, atol=4e-6)

    def test_missing_or_malformed_width_and_kernel_are_rejected(self):
        for key, value in (('width_Hz', np.zeros((2, 3))), ('width_Hz', -1.),
                           ('dispersion_kind', np.full((15, 2), 3))):
            experiment, calibration = fixture()
            experiment['multi_echo_coherence'][key] = value
            with self.assertRaisesRegex(ValueError, 'multi_echo_coherence'):
                model(experiment, calibration)
        experiment, calibration = fixture()
        del experiment['multi_echo_coherence']
        with self.assertRaisesRegex(ValueError, 'multi_echo_coherence'):
            model(experiment, calibration)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable in this process')
    def test_cuda_native_caches_signal_and_gradient_match_cpu(self):
        experiment, calibration = fixture(children=3, width=170.)
        experiment['multi_echo_coherence']['dispersion_kind'][:] = [0, 1, 2]
        cpu = model(experiment, calibration)
        gpu = MultiEchoVoxelACDCModel(experiment, calibration,
            profile='auto', require_cuda=True, precision='float32', sample_chunk=5)
        streamed = MultiEchoVoxelACDCModel(experiment, calibration,
            profile='auto', require_cuda=True, precision='float32', sample_chunk=1,
            memory_fraction=1e-9)
        self.assertEqual(gpu.device.type, 'cuda')
        self.assertEqual(gpu._base_cache.device.type, 'cuda')
        self.assertEqual(gpu._coherence_width.device.type, 'cuda')
        self.assertIsNone(streamed._base_cache)
        self.assertTrue(all(value.device.type == 'cpu' for value in streamed._native_cpu_blocks.values()))
        currents = np.random.default_rng(850).normal(size=(32, 3)) * .04
        expected = cpu.forward(currents)
        value, gradient = cpu.objective_and_grad(currents)
        for candidate in (gpu, streamed):
            for actual, reference in zip(candidate.forward(currents), expected):
                np.testing.assert_allclose(actual, reference, rtol=8e-5, atol=8e-6)
            actual_value, actual_gradient = candidate.objective_and_grad(currents)
            np.testing.assert_allclose(actual_value, value, rtol=8e-5, atol=8e-6)
            error = np.linalg.norm(actual_gradient-gradient) / max(np.linalg.norm(gradient), 1e-12)
            self.assertLess(error, 2e-4)


class MultiEchoPreparationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.inputs, self.fit, self.source, self.path, self.settings = input_fixture(self.folder.name)
        shape = self.fit['first_echo_amplitude'].shape
        gradients = np.broadcast_to([12., -25.], shape + (2,)).copy()
        self.fit['multi_echo'] = {
            'amplitude_envelope_at_reference': self.fit['first_echo_amplitude'] * 1.05,
            'residual_R2_sInv': np.full(shape, 3.2),
            'refined_b0_Hz': self.inputs['field']['b0Hz'] + 4.5,
            'unresolved_width_Hz': np.arange(np.prod(shape), dtype=float).reshape(shape),
            'dispersion_kind': np.arange(np.prod(shape)).reshape(shape) % 3,
            'known_inplane_gradient_Hz_per_pixel': gradients,
            'known_inplane_gradient_valid_mask': np.ones(shape, bool),
            'quality_mask': np.zeros(shape, bool),
            'report': {'fit_echoes': 32},
        }

    def tearDown(self):
        _load_source.cache_clear()
        self.folder.cleanup()

    def prepare(self, **options):
        return prepare_echo_shim_inputs(self.inputs, self.fit, self.path, self.settings,
            rate_mode='complex', inplane_voxel=True, **options)

    def test_exact_fit_fields_envelope_and_kernels_follow_retained_parents(self):
        before = deepcopy(self.fit['multi_echo'])
        with patch('echo_shim_inputs.regularize_b0_local_plane',
                   side_effect=AssertionError('complex-fit slopes must not be smoothed again')):
            result = self.prepare(subspins_z=3, baseline_calibration={'frequency_offset_Hz': 2.})
        expanded, index = result['expanded_experiment'], result['parent_flat_index']
        expected_amplitude = before['amplitude_envelope_at_reference'].ravel()[index] / 4
        np.testing.assert_allclose(expanded['m0_proxy'].sum(axis=1), expected_amplitude)
        np.testing.assert_allclose(expanded['relaxation']['T2star_s'], 1 / 3.2)
        np.testing.assert_allclose(expanded['field']['b0Hz'],
            np.repeat((before['refined_b0_Hz'].ravel()[index] + 2)[:, None], 3, axis=1))
        np.testing.assert_allclose(expanded['inplane_voxel']['b0_gradient_Hz_per_pixel'],
            np.repeat(before['known_inplane_gradient_Hz_per_pixel'].reshape(-1, 2)[index, None], 3, axis=1))
        for target, source in (('width_Hz', 'unresolved_width_Hz'), ('dispersion_kind', 'dispersion_kind')):
            np.testing.assert_array_equal(expanded['multi_echo_coherence'][target],
                np.repeat(before[source].ravel()[index, None], 3, axis=1))
        self.assertTrue((expanded['multi_echo_coherence']['width_Hz'] > 0).any())
        self.assertTrue(result['report']['complex_multi_echo_enabled'])
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_array_equal(before[key], self.fit['multi_echo'][key])

    def test_unknown_fixed_gradient_excludes_whole_parent_without_refitting(self):
        self.fit['multi_echo']['known_inplane_gradient_valid_mask'][3, 3] = False
        result = self.prepare()
        self.assertFalse(result['parent_valid_mask'][3, 3])
        self.assertEqual(result['report']['excluded_missing_B0_gradient_parent_count'], 1)

    def test_core_minus_one_sentinel_is_allowed_only_outside_its_support(self):
        core = self.fit['multi_echo']
        core['support_mask'] = self.fit['simulation_support'].copy()
        core['support_mask'][3, 3] = False
        core['dispersion_kind'][~core['support_mask']] = -1
        result = self.prepare()
        self.assertFalse(result['parent_valid_mask'][3, 3])
        self.assertTrue(np.isin(result['expanded_experiment']['multi_echo_coherence']['dispersion_kind'],
                                (0, 1, 2)).all())
        core['dispersion_kind'][4, 4] = -1
        with self.assertRaisesRegex(ValueError, 'dispersion_kind'):
            self.prepare()

    def test_complex_mode_requires_finite_voxels_and_typed_maps(self):
        with self.assertRaisesRegex(ValueError, 'requires inplane_voxel'):
            prepare_echo_shim_inputs(self.inputs, self.fit, self.path, self.settings,
                rate_mode='complex', inplane_voxel=False)
        self.fit['multi_echo']['unresolved_width_Hz'][3, 3] = np.nan
        with self.assertRaisesRegex(ValueError, 'unresolved_width_Hz'):
            self.prepare()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
