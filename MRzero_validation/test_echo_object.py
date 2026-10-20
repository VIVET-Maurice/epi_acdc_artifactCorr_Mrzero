"""Synthetic checks of amplitude/rate recovery, uncertainty and safe fallbacks."""

import unittest

import numpy as np

from echo_object import (fit_echo_object, INVALID_DATA, LOW_SNR,
                         NONEXPONENTIAL, NO_NOISE_CALIBRATION,
                         TARGET_UNCERTAIN)


TIMES = np.array([.0021231, .0039431, .0057631])


def echoes(amplitude=100., rate=25., *, shape=(12, 10), receivers=3,
           noise_std=0., seed=17):
    amplitude = np.broadcast_to(amplitude, shape)
    rate = np.broadcast_to(rate, shape)
    # Receive sensitivity is normalized to unit RSS, with nonzero phases.
    sensitivity = np.arange(1, receivers + 1) * np.exp(1j * np.arange(receivers) * .7)
    sensitivity /= np.linalg.norm(sensitivity)
    signal = amplitude[None, None] * np.exp(-TIMES[:, None, None, None] * rate)
    signal = signal * sensitivity[None, :, None, None]
    rng = np.random.default_rng(seed)
    noise = noise_std * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
    return signal + noise


class EchoObjectTests(unittest.TestCase):
    def test_recovers_known_effective_amplitude_and_rate_at_native_positions(self):
        row, column = np.indices((12, 10))
        amplitude = 100 + row + 2 * column
        rate = 15 + 2 * row + column
        result = fit_echo_object(echoes(amplitude, rate), TIMES, noise_std=1e-5, target_te_s=.098)
        np.testing.assert_allclose(result["R2star_sInv"], rate, atol=1e-7)
        np.testing.assert_allclose(result["effective_amplitude_TE0"], amplitude, rtol=1e-10)
        np.testing.assert_allclose(result["fitted_amplitude_at_reference"], amplitude * np.exp(-rate * TIMES[0]), rtol=1e-10)
        np.testing.assert_allclose(result["amplitude_at_target"], amplitude * np.exp(-rate * .098), rtol=1e-8)
        self.assertTrue(result["target_reliable_mask"].all())
        self.assertEqual(result["amplitude_reference_time_s"], TIMES[0])

    def test_far_te_extrapolation_is_rejected_despite_acceptable_short_echo_fit(self):
        coils = echoes(30, 25)
        short = fit_echo_object(coils, TIMES, noise_std=1., target_te_s=TIMES[-1])
        long = fit_echo_object(coils, TIMES, noise_std=1., target_te_s=.098)
        self.assertTrue(short["fit_valid_mask"].all())
        self.assertTrue(short["target_reliable_mask"].all())
        self.assertFalse(long["target_reliable_mask"].any())
        self.assertTrue(np.all(long["fallback_reason_flags"] & TARGET_UNCERTAIN))
        np.testing.assert_allclose(long["simulation_R2star_sInv"], 0)
        np.testing.assert_allclose(long["simulation_amplitude_at_reference"], long["first_echo_amplitude"])
        self.assertTrue(np.isnan(long["amplitude_at_target"]).all())

    def test_no_noise_calibration_does_not_create_false_confidence(self):
        result = fit_echo_object(echoes(100, 20), TIMES, target_te_s=.098)
        np.testing.assert_allclose(result["R2star_sInv"], 20., atol=1e-9)
        self.assertFalse(result["target_reliable_mask"].any())
        self.assertTrue(np.all(result["fallback_reason_flags"] & NO_NOISE_CALIBRATION))
        self.assertTrue(np.isnan(result["R2star_std_sInv"]).all())
        np.testing.assert_allclose(result["simulation_R2star_sInv"], 0)

    def test_non_exponential_echoes_are_flagged(self):
        coils = echoes(100, 25)
        coils[1] *= .7
        result = fit_echo_object(coils, TIMES, noise_std=.1)
        self.assertFalse(result["fit_valid_mask"].any())
        self.assertTrue(np.all(result["fallback_reason_flags"] & NONEXPONENTIAL))
        self.assertTrue(result["fallback_mask"].all())

    def test_bounded_fit_never_turns_increasing_echoes_into_negative_decay(self):
        result = fit_echo_object(echoes(100, -10), TIMES, noise_std=.1)
        np.testing.assert_allclose(result["R2star_sInv"], 0)
        self.assertTrue(np.all(result["unbounded_R2star_sInv"] < 0))
        self.assertTrue(np.isinf(result["T2star_s"]).all())
        self.assertFalse(result["target_reliable_mask"].any())

    def test_low_snr_and_invalid_complex_data_have_explicit_fallbacks(self):
        coils = echoes(4, 25)
        coils[:, :, 0, 0] = np.nan
        result = fit_echo_object(coils, TIMES, noise_std=1.)
        self.assertFalse(result["target_reliable_mask"].any())
        self.assertTrue(np.all(result["fallback_reason_flags"][1:] & LOW_SNR))
        self.assertTrue(result["fallback_reason_flags"][0, 0] & INVALID_DATA)
        self.assertEqual(result["simulation_amplitude_at_reference"][0, 0], 0.)

    def test_full_correlated_noise_covariance_is_used_for_rss_power(self):
        C = np.array([[2., .6 + .2j], [.6 - .2j, 3.]])
        coils = echoes(100, 25, receivers=2)
        result = fit_echo_object(coils, TIMES, noise_covariance=C)
        power = np.sum(abs(coils)**2, axis=1)
        np.testing.assert_allclose(result["debiased_RSS"]**2, power - np.trace(C).real)
        quad = np.einsum("ecyx,cd,edyx->eyx", coils.conj(), C, coils).real
        expected_var = np.maximum(2 * quad - np.sum(abs(C)**2), np.sum(abs(C)**2)) / (4 * (power - np.trace(C).real))
        np.testing.assert_allclose(result["echo_amplitude_variance"], expected_var)

    def test_background_estimates_complex_noise_covariance_without_using_object(self):
        shape, receivers = (40, 40), 2
        support = np.zeros(shape, bool)
        support[12:28, 12:28] = True
        signal = echoes(np.where(support, 100., 0.), 25, shape=shape, receivers=receivers)
        rng = np.random.default_rng(18)
        C = np.array([[2., .7 + .2j], [.7 - .2j, 3.]])
        L = np.linalg.cholesky(C)
        samples = (rng.normal(size=(receivers, 3 * 40 * 40))
                   + 1j * rng.normal(size=(receivers, 3 * 40 * 40))) / np.sqrt(2)
        noise = (L @ samples).reshape(receivers, 3, 40, 40).transpose(1, 0, 2, 3)
        result = fit_echo_object(signal + noise, TIMES, support_mask=support, noise_mask=~support)
        np.testing.assert_allclose(result["noise_covariance"], C, atol=.1)
        self.assertEqual(result["report"]["noise"]["sample_count"], 3 * int((~support).sum()))
        self.assertTrue(result["report"]["noise"]["calibrated"])
        self.assertEqual(np.count_nonzero(result["simulation_amplitude_at_reference"][~support]), 0)

    def test_statistical_rate_uncertainty_matches_independent_noisy_voxels(self):
        result = fit_echo_object(echoes(100, 35, shape=(80, 80), noise_std=1., seed=21),
                                 TIMES, noise_std=1., target_te_s=TIMES[-1])
        rate = result["R2star_sInv"]
        predicted_std = result["R2star_std_sInv"]
        self.assertAlmostEqual(float(rate.mean()), 35., delta=.2)
        # Residual inflation makes the reported std conservative; use median
        # to compare the independently known noise floor with Monte Carlo.
        ratio = float(rate.std() / np.median(predicted_std))
        self.assertGreater(ratio, .9)
        self.assertLess(ratio, 1.1)

    def test_pooling_precision_requires_explicit_effective_sample_count(self):
        coils = echoes(30, 25)
        conservative = fit_echo_object(coils, TIMES, noise_std=1., target_te_s=.098)
        pooled = fit_echo_object(coils, TIMES, noise_std=1., target_te_s=.098,
                                 pooled_effective_voxel_count=120)
        self.assertFalse(conservative["pooled"]["target_reliable_mask"])
        self.assertTrue(pooled["pooled"]["target_reliable_mask"])
        self.assertEqual(pooled["pooled"]["voxel_count"], 120)
        np.testing.assert_allclose(pooled["pooled"]["target_log_std"],
                                   conservative["pooled"]["target_log_std"] / np.sqrt(120))

    def test_rejects_unphysical_noise_inputs_and_background_overlap(self):
        coils = echoes()
        with self.assertRaises(ValueError):
            fit_echo_object(coils, TIMES, noise_covariance=np.array([[1, 2, 0], [2, 1, 0], [0, 0, 1]]))
        with self.assertRaises(ValueError):
            fit_echo_object(coils, TIMES, noise_std=1., noise_mask=np.ones((12, 10), bool))
        with self.assertRaises(ValueError):
            fit_echo_object(coils, TIMES, noise_mask=np.ones((12, 10), bool))
        with self.assertRaises(ValueError):
            fit_echo_object(coils, TIMES, noise_std=1., pooled_effective_voxel_count=121)


if __name__ == "__main__":
    unittest.main()
