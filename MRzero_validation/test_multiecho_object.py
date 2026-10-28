"""End-to-end object preparation retains real late-echo losses and units."""
from copy import deepcopy
import unittest

import numpy as np

from echo_object import fit_echo_object
from multiecho_object import prepare_complex_echo_model, multiecho_comparison, known_inplane_field


def fixture():
    times = .002123 + np.arange(32) * .00182
    row, column = np.indices((4, 4))
    field = 25. + 2*row - 3*column
    receivers = np.array([1., .5j, -.2+.3j])
    t = times[:, None, None]
    envelope = (np.exp(-(20.+2j*np.pi*field[None])*(t-times[0]))
                * np.sinc(2*t)*np.sinc(-3*t)*np.sinc(36*t))
    coils = receivers[None, :, None, None] * envelope[:, None]
    support = np.ones(field.shape, bool)
    fit = fit_echo_object(coils, times, noise_covariance=np.eye(3)*1e-8,
                          support_mask=support, target_te_s=.0780575)
    fit.update(object_support=support.copy(), simulation_support=support.copy(),
               noise_mask=np.zeros_like(support))
    X, Y = np.meshgrid((np.arange(4)-2)*.005, (np.arange(4)-2)*.005)
    inputs = dict(b0_raw=dict(coil_images=coils, X_m=X, Y_m=Y),
        echo_times_s=times, field=dict(b0Hz=field, mask=support),
        field_echo_count=6, acquisition=dict(TE_s=.0780575))
    return inputs, fit


class MultiEchoObjectTests(unittest.TestCase):
    def test_preparation_recovers_envelope_without_resetting_late_null_to_no_decay(self):
        inputs, legacy = fixture()
        original = deepcopy(legacy)
        revised = prepare_complex_echo_model(inputs, legacy, device='cpu',
            max_iterations=40, max_width_Hz=80.)
        result = revised['multi_echo']
        np.testing.assert_array_equal(result['initial_b0_Hz'], known_inplane_field(inputs)['b0Hz'])
        self.assertFalse(np.shares_memory(result['initial_b0_Hz'], inputs['field']['b0Hz']))
        np.testing.assert_allclose(result['residual_R2_sInv'], 20., atol=.02)
        np.testing.assert_allclose(result['unresolved_width_Hz'], 36., atol=.02)
        np.testing.assert_array_equal(result['dispersion_kind'], 1)
        self.assertGreater(revised['report']['multi_echo']['old_zero_decay_amplitude_fraction'], .99)
        self.assertFalse(revised['report']['multi_echo']['uses_EPI_to_fit_object'])
        self.assertIn('Preserve', revised['report']['multi_echo']['zero_B0_target'])
        for key in ('simulation_R2star_sInv', 'simulation_amplitude_at_reference',
                    'first_echo_amplitude', 'object_support', 'simulation_support'):
            np.testing.assert_array_equal(legacy[key], original[key])
        comparison = multiecho_comparison(inputs, revised)
        holes = next(row for row in comparison['rows'] if row['region'] == 'holes_at_echo15')
        self.assertEqual(holes['pixels'], 16)
        self.assertGreater(holes['legacy_RSS_RMSE_over_first_echo'], .3)
        self.assertLess(holes['complex_RSS_RMSE_over_first_echo'], .001)

    def test_uncalibrated_noise_does_not_become_a_confident_complex_model(self):
        inputs, fit = fixture()
        fit['report']['noise']['calibrated'] = False
        with self.assertRaisesRegex(ValueError, 'calibrated receiver noise'):
            prepare_complex_echo_model(inputs, fit, device='cpu')


if __name__ == '__main__':
    unittest.main()
