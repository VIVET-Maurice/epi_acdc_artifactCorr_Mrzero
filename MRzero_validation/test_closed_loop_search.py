"""Closed-loop cost anchors, real-only selection and conservative scan budgets."""

import unittest

import numpy as np

from closed_loop_search import (
    propose_closed_loop_candidate, quantized_waveform_fingerprint, select_measured_best,
)


def _point(identifier, alpha, cost, fingerprint=None):
    return {'id': identifier, 'alpha': alpha, 'experimental_J': cost,
            'fingerprint': fingerprint or f'commands_{alpha:.2f}'}


class ClosedLoopSearchTests(unittest.TestCase):
    def setUp(self):
        self.alpha = np.linspace(0., 1., 21)
        self.fingerprints = [f'commands_{alpha:.2f}' for alpha in self.alpha]
        self.cost = .16+.64*(self.alpha-.75)**2
        self.points = [_point('baseline', 0., self.cost[0]), _point('historical', 1., self.cost[-1])]

    def run_search(self, **options):
        return propose_closed_loop_candidate(self.alpha, self.cost, self.points,
            waveform_fingerprints=self.fingerprints, **options)

    def test_affine_discrepancy_matches_both_real_anchors_and_proposes_local_gain(self):
        self.points[0]['experimental_J'] += .04
        self.points[1]['experimental_J'] += .01
        result = self.run_search()
        np.testing.assert_allclose(result['calibrated_J'], self.cost+.04*(1-self.alpha)+.01*self.alpha)
        self.assertAlmostEqual(result['calibrated_J'][0], self.points[0]['experimental_J'])
        self.assertAlmostEqual(result['calibrated_J'][-1], self.points[1]['experimental_J'])
        candidate = result['candidate']
        self.assertIsNotNone(candidate)
        self.assertGreaterEqual(candidate['alpha'], .75)
        self.assertGreaterEqual(candidate['predicted_gain_pp'], .5)
        self.assertFalse(candidate['improvement_guaranteed'])
        self.assertEqual(result['budget']['remaining_additional_scans'], 2)

    def test_third_anchor_gives_piecewise_linear_discrepancy_without_overshoot(self):
        self.points.append(_point('new_1', .5, .5))
        result = self.run_search()
        delta = result['calibrated_J']-self.cost
        expected = np.interp(self.alpha, [0, .5, 1], [0, .5-self.cost[10], 0])
        np.testing.assert_allclose(delta, expected)
        self.assertAlmostEqual(result['calibrated_J'][10], .5)
        self.assertLessEqual(delta.max(), .5-self.cost[10]+1e-14)
        self.assertEqual(result['calibration']['method'], 'piecewise linear anchored discrepancy')
        self.assertFalse(result['calibration']['extrapolation_allowed'])

    def test_actual_improvement_changes_real_best_independently_of_simulation(self):
        self.points.append(_point('new_1', .8, .10))
        result = self.run_search()
        self.assertEqual(result['measured_selection']['nominal_best']['id'], 'new_1')
        self.assertEqual(result['measured_selection']['retained_best']['id'], 'new_1')
        self.assertTrue(result['measured_selection']['robust_improvement'])

    def test_degrading_scan_never_replaces_previous_real_best(self):
        self.points.append(_point('new_1', .8, 5.))
        result = self.run_search()
        self.assertEqual(result['measured_selection']['nominal_best']['id'], 'historical')
        self.assertEqual(result['measured_selection']['retained_best']['id'], 'historical')
        self.assertEqual(result['budget']['completed_additional_scans'], 1)

    def test_tiny_improvement_is_nominal_only_and_incumbent_is_persistent(self):
        points = [_point('baseline', 0., .5), _point('historical', 1., .16),
                  _point('new_1', .8, (.399)**2)]
        selected = select_measured_best(points, uncertainty_floor_pp=.5)
        self.assertEqual(selected['nominal_best']['id'], 'new_1')
        self.assertEqual(selected['retained_best']['id'], 'historical')
        self.assertFalse(selected['robust_improvement'])
        selected_again = select_measured_best(points, incumbent_id='historical', uncertainty_floor_pp=.05)
        self.assertEqual(selected_again['retained_best']['id'], 'new_1')
        self.assertTrue(selected_again['selection_uses_experimental_cost_only'])

    def test_negative_calibration_is_flagged_and_only_explicitly_explored(self):
        self.cost = (self.alpha-.5)**2
        self.points = [_point('baseline', 0., .01), _point('historical', 1., .01)]
        result = self.run_search(trust_radius=1.)
        self.assertIsNone(result['candidate'])
        self.assertTrue(result['negative_calibrated_mask'][10])
        self.assertTrue(np.isnan(result['predicted_NRMSE_percent'][10]))
        self.assertFalse(result['ordinary_eligible_mask'][10])
        exploratory = self.run_search(trust_radius=1., allow_exploration=True)['candidate']
        self.assertTrue(exploratory['exploratory'])
        self.assertEqual(exploratory['alpha'], .5)
        self.assertLess(exploratory['calibrated_J'], 0)
        self.assertIsNone(exploratory['predicted_NRMSE_percent'])
        self.assertIsNone(exploratory['predicted_gain_pp'])

    def test_unpromising_bank_stops_unless_exploration_is_explicit(self):
        self.cost[:] = .3
        self.points = [_point('baseline', 0., .3), _point('historical', 1., .3)]
        result = self.run_search()
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['stop_reason'], 'no_promising_local_candidate')
        self.assertIsNone(self.run_search(min_predicted_gain_pp=0.)['candidate'])
        exploratory = self.run_search(allow_exploration=True)['candidate']
        self.assertTrue(exploratory['exploratory'])
        self.assertFalse(exploratory['improvement_guaranteed'])

    def test_quantized_duplicates_never_reoffer_measured_commands(self):
        self.fingerprints[15] = self.fingerprints[-1]  # attractive alpha .75 is physically already tested
        self.fingerprints[16] = self.fingerprints[17]  # .8 and .85 share DAC commands
        result = self.run_search()
        self.assertTrue(result['already_measured_mask'][15])
        self.assertTrue(result['duplicate_quantized_waveform_mask'][16])
        self.assertFalse(result['ordinary_eligible_mask'][15])
        self.assertFalse(result['ordinary_eligible_mask'][16])
        self.assertNotIn(result['candidate']['fingerprint'],
                         [point['fingerprint'] for point in self.points])

    def test_zero_experimental_anchor_is_preserved_exactly(self):
        self.points[1]['experimental_J'] = 0.
        result = self.run_search()
        self.assertEqual(result['calibrated_J'][-1], 0.)
        self.assertEqual(result['predicted_NRMSE_percent'][-1], 0.)
        self.assertTrue(result['valid_calibrated_mask'][-1])

    def test_budget_zero_one_two_and_completed_rounds(self):
        for budget in (0, 1, 2):
            with self.subTest(budget=budget):
                result = self.run_search(max_additional_scans=budget)
                self.assertEqual(result['budget']['remaining_additional_scans'], budget)
                self.assertEqual(result['candidate'] is None, budget == 0)
        self.points.extend([_point('new_1', .8, .15), _point('new_2', .9, .14)])
        result = self.run_search()
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['budget']['remaining_additional_scans'], 0)
        self.assertEqual(result['measured_selection']['nominal_best']['id'], 'new_2')

    def test_hardware_admissibility_and_local_radius_are_enforced(self):
        feasible = np.ones(21, bool); feasible[15:18] = False
        self.cost[15:18] = np.nan
        result = self.run_search(admissible=feasible, trust_radius=.1, allow_exploration=True)
        candidate = result['candidate']
        self.assertGreaterEqual(candidate['alpha'], .9-1e-12)
        self.assertTrue(candidate['hardware_admissible_as_supplied'])
        blocked = self.run_search(admissible=feasible, trust_radius=0., allow_exploration=True)
        self.assertIsNone(blocked['candidate'])

    def test_bad_anchors_costs_associations_and_budgets_are_rejected(self):
        for options in ({'max_additional_scans': True}, {'max_additional_scans': 3},
                        {'max_additional_scans': 1.5}, {'trust_radius': -1.},
                        {'allow_exploration': 'yes'}, {'uncertainty_floor_pp': -.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_search(**options)
        self.points[1]['fingerprint'] = 'wrong_applied_command'
        with self.assertRaisesRegex(ValueError, 'fingerprints disagree'):
            self.run_search()
        for points in ([self.points[0]], [_point('a', 0., .1), _point('b', .9, .2)],
                       [_point('a', 0., -.1), _point('b', 1., .2)]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                propose_closed_loop_candidate(self.alpha, self.cost, points,
                    waveform_fingerprints=self.fingerprints)

    def test_fingerprint_is_dtype_stable_and_includes_timing_and_channels(self):
        commands = np.array([[0, 2], [-2, 4]], dtype=np.int16)
        fingerprint = quantized_waveform_fingerprint(commands)
        self.assertEqual(fingerprint, quantized_waveform_fingerprint(commands.astype(float)))
        changed = commands.copy(); changed[0, 1] += 1
        self.assertNotEqual(fingerprint, quantized_waveform_fingerprint(changed))
        self.assertNotEqual(quantized_waveform_fingerprint(commands, timing_s=[0, .1]),
                            quantized_waveform_fingerprint(commands, timing_s=[0, .2]))
        self.assertNotEqual(quantized_waveform_fingerprint(commands, channel_order=[0, 1]),
                            quantized_waveform_fingerprint(commands, channel_order=[1, 0]))
        with self.assertRaises(ValueError):
            quantized_waveform_fingerprint(np.array([[np.nan]]))


if __name__ == '__main__':
    unittest.main()
