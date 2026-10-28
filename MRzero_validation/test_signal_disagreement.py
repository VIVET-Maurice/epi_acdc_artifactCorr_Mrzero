"""Opposite signal losses, measured zeros, frozen gains and fixed domains."""

import unittest
import numpy as np

from signal_disagreement import signal_disagreement, fixed_comparison_envelope


class SignalDisagreementTests(unittest.TestCase):
    def test_measured_zero_simulated_bright_and_opposite_case_are_both_counted(self):
        measured = np.array([[0., 1.], [1., 0.]])
        simulated = np.array([[1., 0.], [1., 0.]])
        result = signal_disagreement(measured, simulated, gain=1.)
        report = result['report']['full_FOV']
        self.assertEqual(report['SSE'], 2.)
        self.assertEqual(report['overprediction_SSE'], 1.)
        self.assertEqual(report['underprediction_SSE'], 1.)
        self.assertEqual(report['normalized_SSE'], 1.)
        self.assertEqual(report['measured_zero_pixels'], 2)
        for label in ('25', '50'):
            self.assertEqual(report['ratios'][label]['measured_low_simulated_high_pixels'], 1)
            self.assertEqual(report['ratios'][label]['simulated_low_measured_high_pixels'], 1)
        self.assertTrue(result['arrays']['measured_low_simulated_high_25_mask'][0, 0])
        self.assertTrue(result['arrays']['simulated_low_measured_high_25_mask'][0, 1])

    def test_swapping_images_swaps_signed_sse_and_ratio_directions(self):
        measured = np.array([[0., .8, 1.], [.9, .1, .6]])
        simulated = np.array([[1., .2, .9], [.1, .8, .7]])
        one = signal_disagreement(measured, simulated, gain=1., reference_scale=1.)
        other = signal_disagreement(simulated, measured, gain=1., reference_scale=1.)
        np.testing.assert_array_equal(one['arrays']['difference'], -other['arrays']['difference'])
        self.assertAlmostEqual(one['report']['full_FOV']['overprediction_SSE'],
                               other['report']['full_FOV']['underprediction_SSE'])
        np.testing.assert_array_equal(one['arrays']['measured_low_simulated_high_50_mask'],
                                      other['arrays']['simulated_low_measured_high_50_mask'])

    def test_supplied_gain_is_used_without_a_refit(self):
        result = signal_disagreement(np.ones((3, 3)), 1j * np.ones((3, 3)), gain=2.)
        self.assertFalse(result['report']['gain_fitted'])
        self.assertEqual(result['report']['full_FOV']['NRMSE_percent'], 100.)
        np.testing.assert_array_equal(result['arrays']['difference'], np.ones((3, 3)))

    def test_optional_envelope_never_changes_primary_full_fov_scores(self):
        measured = np.ones((4, 4))
        simulated = measured.copy(); simulated[0, 0] = 0.
        mask = np.zeros((4, 4), bool); mask[1:3, 1:3] = True
        result = signal_disagreement(measured, simulated, gain=1., analysis_mask=mask,
                                      subsets={'test': np.eye(4, dtype=bool)})
        self.assertEqual(result['report']['full_FOV']['underprediction_SSE'], 1.)
        self.assertEqual(result['report']['analysis_mask']['underprediction_SSE'], 0.)
        self.assertEqual(result['report']['subsets']['test']['pixels'], 2)
        self.assertFalse(result['arrays']['simulated_low_measured_high_50_mask'].any())

    def test_zero_energy_domain_keeps_absolute_overprediction_error(self):
        result = signal_disagreement(np.zeros((2, 2)), np.ones((2, 2)), gain=1., reference_scale=1.)
        report = result['report']['full_FOV']
        self.assertEqual(report['SSE'], 4.)
        self.assertEqual(report['overprediction_SSE'], 4.)
        self.assertIsNone(report['NRMSE_percent'])
        self.assertEqual(report['ratios']['25']['measured_low_simulated_high_pixels'], 4)

    def test_reference_envelope_fills_reference_holes_without_using_epi(self):
        reference = np.zeros((7, 7)); reference[1:6, 1:6] = 1.; reference[3, 3] = 0.
        mask = fixed_comparison_envelope(reference)
        self.assertEqual(int(mask.sum()), 25)
        self.assertTrue(mask[3, 3])
        self.assertFalse(mask[0, 0])
        self.assertFalse(fixed_comparison_envelope(np.zeros((3, 3))).any())

    def test_symmetric_brightness_gate_retains_dark_measurements(self):
        measured = np.array([[.01, .8], [.01, .1]])
        simulated = np.array([[.8, .01], [.1, .01]])
        result = signal_disagreement(measured, simulated, gain=1., reference_scale=1.)
        self.assertEqual(result['report']['full_FOV']['ratios']['25']['measured_low_simulated_high_pixels'], 1)
        self.assertEqual(result['report']['full_FOV']['ratios']['25']['simulated_low_measured_high_pixels'], 1)
        self.assertTrue(result['arrays']['both_low_mask'][1].all())

    def test_invalid_masks_values_and_gains_are_rejected(self):
        image = np.ones((3, 3))
        for gain in (0., -1., np.nan, True):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                signal_disagreement(image, image, gain=gain)
        for options in ({'analysis_mask': np.zeros((2, 2))}, {'ratios': [0.]},
                        {'ratios': [.5, .5]}, {'subsets': []}, {'reference_scale': -1.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                signal_disagreement(image, image, gain=1., **options)
        broken = image.copy(); broken[0, 0] = np.nan
        with self.assertRaises(ValueError):
            signal_disagreement(broken, image, gain=1.)


if __name__ == '__main__':
    unittest.main()
