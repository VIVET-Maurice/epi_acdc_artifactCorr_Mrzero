"""Physical echo-train regressions, including signal nulls and complex noise."""

from copy import deepcopy
import unittest

import numpy as np
import torch

from complex_echo_fit import (fit_complex_echoes, _kernel, _project,
                              _normal_equations)


def echo_fixture(noise=0.):
    times = .002123 + np.arange(32) * .00182
    rates = np.array([[12., 20., 8., 4.]])
    field = np.array([[25., -40., 15., -8.]])
    width = np.array([[0., 38., 12., 0.]])
    kind = np.array([[0, 1, 2, 0]])
    gradients = np.array([[[2., 5.], [3., 1.], [2., 4.], [0., 0.]]])
    receivers = np.array([1., .5j, -.2 + .3j])
    t = times[:, None, None]
    envelope = np.exp(-rates[None] * (t - times[0]) - 2j * np.pi * field[None] * (t - times[0]))
    envelope *= np.sinc(gradients[None, ..., 0] * t) * np.sinc(gradients[None, ..., 1] * t)
    envelope *= np.where(kind[None] == 1, np.sinc(width[None] * t),
        np.where(kind[None] == 2, np.exp(-2 * np.pi**2 * width[None]**2 * t**2), 1))
    coils = envelope[:, None] * receivers[None, :, None, None]
    # Imaginary covariance terms exercise the direction of complex whitening.
    mixing = np.array([[1., .15j, .1], [.2, .9, -.1j], [.1j, .05, .8]], complex)
    covariance = max(noise, 1e-5)**2 * mixing @ mixing.conj().T
    if noise:
        rng = np.random.default_rng(155)
        z = (rng.normal(size=coils.shape) + 1j * rng.normal(size=coils.shape)) / np.sqrt(2)
        coils = coils + noise * np.einsum('cd,edyx->ecyx', mixing, z)
    kwargs = dict(noise_covariance=covariance, initial_b0_Hz=field - 2.,
        support_mask=np.ones(rates.shape, bool), known_inplane_gradient_Hz_per_pixel=gradients,
        device='cpu', max_iterations=60)
    return coils, times, kwargs, rates, field, width, kind, receivers


class ComplexEchoTests(unittest.TestCase):
    def test_exact_models_recover_signed_phase_rate_width_and_envelope(self):
        coils, times, kwargs, rate, field, width, kind, receivers = echo_fixture()
        original_coils, original_kwargs = coils.copy(), deepcopy(kwargs)
        fitted = fit_complex_echoes(coils, times, **kwargs)
        np.testing.assert_array_equal(fitted['dispersion_kind'], kind)
        np.testing.assert_allclose(fitted['residual_R2_sInv'], rate, atol=1e-5)
        np.testing.assert_allclose(fitted['refined_b0_Hz'], field, atol=1e-6)
        np.testing.assert_allclose(fitted['unresolved_width_Hz'], width, atol=1e-5)
        np.testing.assert_allclose(fitted['amplitude_envelope_at_reference'], np.linalg.norm(receivers), atol=1e-6)
        np.testing.assert_allclose(fitted['predicted_RSS'], np.linalg.norm(coils, axis=1), atol=1e-7)
        self.assertTrue(fitted['quality_mask'].all())
        self.assertFalse(fitted['report']['validation_amplitude_refitted'])
        self.assertEqual(fitted['report']['training_echo_indices'], list(range(26)))
        self.assertEqual(fitted['report']['validation_echo_indices'], list(range(26, 32)))
        self.assertTrue(fitted['report']['refit_uses_all_echoes'])
        np.testing.assert_array_equal(coils, original_coils)
        for key in ('initial_b0_Hz', 'support_mask', 'known_inplane_gradient_Hz_per_pixel', 'noise_covariance'):
            np.testing.assert_array_equal(kwargs[key], original_kwargs[key])

    def test_signal_null_and_revival_do_not_disable_fitted_decay(self):
        coils, times, kwargs, rate, _, _, _, _ = echo_fixture(noise=.001)
        fit = fit_complex_echoes(coils, times, **kwargs)
        self.assertEqual(fit['dispersion_kind'][0, 1], 1)
        self.assertGreater(fit['residual_R2_sInv'][0, 1], 10.)
        self.assertAlmostEqual(fit['unresolved_width_Hz'][0, 1], 38., delta=.5)
        rss = fit['predicted_RSS'][:, 0, 1]
        self.assertLess(rss.min(), .04 * rss[0])
        self.assertGreater(rss[-8:].max(), 3 * rss.min())
        self.assertGreater(fit['selected_gaussian_mask'].sum(), 0)
        self.assertAlmostEqual(fit['residual_R2_sInv'][0, 2], rate[0, 2], delta=1.)

    def test_poor_model_agreement_keeps_loss_instead_of_no_decay_fallback(self):
        coils, times, kwargs, *_ = echo_fixture()
        # A localized phase excursion violates every candidate envelope.
        coils[12, :, 0, 1] *= np.exp(.8j)
        fit = fit_complex_echoes(coils, times, **kwargs)
        self.assertFalse(fit['quality_mask'][0, 1])
        self.assertGreater(fit['residual_R2_sInv'][0, 1], 1.)
        self.assertGreater(fit['unresolved_width_Hz'][0, 1], 10.)
        self.assertLess(fit['predicted_RSS'][-1, 0, 1], .3 * fit['amplitude_envelope_at_reference'][0, 1])

    def test_an_exact_complex_zero_at_echo_15_remains_a_valid_observation(self):
        coils, times, kwargs, *_ = echo_fixture()
        exact_width = 1. / times[14]
        coils[:, :, 0, 1] *= (np.sinc(exact_width * times) / np.sinc(38. * times))[:, None]
        coils[14, :, 0, 1] = 0.
        fit = fit_complex_echoes(coils, times, **kwargs)
        self.assertEqual(fit['dispersion_kind'][0, 1], 1)
        self.assertAlmostEqual(fit['unresolved_width_Hz'][0, 1], exact_width, delta=1e-5)
        self.assertAlmostEqual(fit['residual_R2_sInv'][0, 1], 20., delta=1e-5)
        self.assertLess(fit['predicted_RSS'][14, 0, 1], 1e-7)

    def test_first_echo_dephasing_is_not_folded_into_positive_envelope(self):
        coils, times, kwargs, *_ = echo_fixture()
        kwargs['amplitude_reference_time_s'] = 0.
        fit = fit_complex_echoes(coils, times, **kwargs)
        expected = np.linalg.norm(coils[0], axis=0)
        self.assertEqual(fit['amplitude_reference_time_s'], 0.)
        self.assertGreater(fit['amplitude_envelope_at_reference'][0, 1], expected[0, 1])
        np.testing.assert_allclose(fit['predicted_RSS'], np.linalg.norm(coils, axis=1), atol=1e-7)

    def test_covariance_scale_changes_quality_without_changing_exact_parameters(self):
        coils, times, kwargs, rate, field, width, *_ = echo_fixture()
        kwargs['noise_covariance'] *= 1000
        fit = fit_complex_echoes(coils, times, **kwargs)
        np.testing.assert_allclose(fit['residual_R2_sInv'], rate, atol=1e-5)
        np.testing.assert_allclose(fit['refined_b0_Hz'], field, atol=1e-6)
        np.testing.assert_allclose(fit['unresolved_width_Hz'], width, atol=1e-5)

    def test_support_excludes_unmeasured_pixels_and_invalid_inputs_are_rejected(self):
        coils, times, kwargs, *_ = echo_fixture()
        kwargs['support_mask'][0, 0] = False
        coils[:, :, 0, 0] = np.nan
        fit = fit_complex_echoes(coils, times, **kwargs)
        self.assertEqual(fit['dispersion_kind'][0, 0], -1)
        self.assertFalse(fit['quality_mask'][0, 0])
        self.assertTrue(np.all(fit['predicted_RSS'][:, 0, 0] == 0))
        for change, message in (({'noise_covariance': -np.eye(3)}, 'positive semidefinite'),
                                ({'width_initializations_Hz': [0.]}, 'nonzero'),
                                ({'device': 'cpu', 'require_cuda': True}, 'CUDA')):
            arguments = dict(kwargs, **change)
            with self.subTest(change=change), self.assertRaisesRegex((ValueError, RuntimeError), message):
                fit_complex_echoes(coils, times, **arguments)

    def test_reduced_normal_equation_gradient_matches_complex_finite_difference(self):
        rng = np.random.default_rng(99)
        times = torch.linspace(.03, 1., 15, dtype=torch.float64)
        x = times - times[0]
        known = torch.sinc(.6 * times)[None]
        y = torch.tensor(rng.normal(size=(1, 15, 3)) + 1j * rng.normal(size=(1, 15, 3)))
        for kind in (0, 1, 2):
            p = torch.tensor([[[.7, -.13] if kind == 0 else [.7, -.13, 1.4]]], dtype=torch.float64)
            h, dh = _kernel(p, x, times, known, kind)
            amplitude, _, denominator = _project(y, h)
            _, right = _normal_equations(y, h, dh, amplitude, denominator)
            finite = []
            for parameter in range(p.shape[-1]):
                direction = torch.zeros_like(p)
                direction[..., parameter] = 1e-5
                plus = _project(y, _kernel(p + direction, x, times, known, kind)[0])[1]
                minus = _project(y, _kernel(p - direction, x, times, known, kind)[0])[1]
                finite.append(((plus - minus) / 2e-5).item())
            np.testing.assert_allclose(-2 * right.numpy().ravel(), finite, rtol=1e-7, atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable in this execution environment')
    def test_cuda_is_used_and_matches_cpu(self):
        coils, times, kwargs, *_ = echo_fixture(noise=.001)
        cpu = fit_complex_echoes(coils, times, **kwargs)
        kwargs.update(device='cuda', require_cuda=True)
        gpu = fit_complex_echoes(coils, times, **kwargs)
        self.assertTrue(gpu['report']['cuda_used'])
        np.testing.assert_array_equal(gpu['dispersion_kind'], cpu['dispersion_kind'])
        np.testing.assert_allclose(gpu['predicted_RSS'], cpu['predicted_RSS'], atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
