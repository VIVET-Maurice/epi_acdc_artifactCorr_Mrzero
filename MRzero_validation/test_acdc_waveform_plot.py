"""Check waveform time origins, output mapping and the existing CSV format."""

from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Experiment'))
from export_acdc_waveform import export_waveform
from acdc_waveform_plot import plot_acdc_waveform, read_waveform_csv, waveform_plot_data


class SmallWaveformModel:
    """Independent two-channel rectangular-response fixture, with no MRI/GPU."""
    nc = 2
    line_start_s = np.array([.010, .012, .014])
    line_end_s = np.array([.012, .014, .016])
    current_response = dict(enabled=True, tau_s=np.array([.0001, .0002]))

    def sample_waveform(self, commands, times_s=None, *, response_enabled=None):
        if times_s is None:
            edges = np.r_[self.line_start_s, self.line_end_s[-1]]
            times_s = np.unique(np.r_[0., edges,
                (edges[:, None] + np.linspace(0., .0016, 41)).ravel()])
        times = np.asarray(times_s)
        command = commands @ ((times[:, None] >= self.line_start_s)
                               & (times[:, None] < self.line_end_s)).T
        if response_enabled is False:
            actual = command.copy()
        else:
            actual = np.zeros_like(command)
            for channel, tau in enumerate(self.current_response['tau_s']):
                for state in range(3):
                    start_age = np.maximum(0., times - self.line_start_s[state])
                    end_age = np.maximum(0., times - self.line_end_s[state])
                    actual[channel] += commands[channel, state] * (
                        -np.expm1(-start_age / tau) + np.expm1(-end_age / tau))
        return dict(time_s=times, commanded_A=command, actual_A=actual)


class WaveformPlotTests(unittest.TestCase):
    def setUp(self):
        self.model = SmallWaveformModel()
        self.commands = np.array([[.1, .2, .15], [-.3, .8, .1]])

    def test_time_origin_changes_only_display_times_and_retains_pretrigger_history(self):
        rf = waveform_plot_data(self.model, self.commands)
        trigger = waveform_plot_data(self.model, self.commands,
            origin='first_imaging_trigger', trigger_rf_s=.010)
        np.testing.assert_allclose(trigger['time_ms'], rf['time_ms'] - 10., atol=2e-15)
        np.testing.assert_array_equal(trigger['actual_A'], rf['actual_A'])
        np.testing.assert_array_equal(trigger['commanded_A'], rf['commanded_A'])
        np.testing.assert_array_equal(trigger['zoom_time_us'], rf['zoom_time_us'])
        self.assertEqual(trigger['time_ms'][0], -10.)
        self.assertTrue(trigger['report']['pre_origin_samples_retained'])
        self.assertAlmostEqual(trigger['report']['final_zero_from_origin_ms'], 6.)

    def test_physical_output_mapping_polarity_and_largest_transition_are_preserved(self):
        original = self.model.sample_waveform(self.commands)
        data = waveform_plot_data(self.model, self.commands,
                                  channel_map=[5, 8], channel_polarity=[1, -1])
        self.assertEqual(data['report']['selected_physical_output'], 8)
        self.assertEqual(data['report']['selected_model_channel'], 2)
        self.assertAlmostEqual(data['transition_rf_s'], .012)
        np.testing.assert_array_equal(data['actual_A'][4], original['actual_A'][0])
        np.testing.assert_array_equal(data['actual_A'][7], -original['actual_A'][1])
        self.assertTrue(np.all(data['actual_A'][:4] == 0))
        self.assertTrue(np.all(data['commanded_A'][:, -1] == 0))
        np.testing.assert_array_equal(self.commands, [[.1, .2, .15], [-.3, .8, .1]])

    def test_first_order_rise_is_from_the_model_not_recomputed_by_display(self):
        commands = np.array([[.5, .4, .3], [-.3, .8, .1]])
        data = waveform_plot_data(self.model, commands, channel=1)
        # The selected largest channel-1 step occurs at 10 ms (0 -> 0.5 A).
        self.assertEqual(data['transition_rf_s'], .010)
        at_tau = int(np.argmin(abs(data['zoom_time_us'] - 100.)))
        # The zoom grid is dense but need not contain exactly tau.
        elapsed_s = data['zoom_time_us'][at_tau] * 1e-6
        expected = .5 * (1 - np.exp(-elapsed_s / .0001))
        self.assertAlmostEqual(data['zoom_actual_A'][at_tau], expected, places=13)

    def test_real_exported_CSV_commands_and_trigger_times_are_overlaid_without_quantization(self):
        settings = dict(channel_map=[5, 8], channel_polarity=[1, -1],
                        max_current_A=3., total_current_limit_A=30.)
        edges = np.r_[self.model.line_start_s, self.model.line_end_s[-1]] - .010
        with tempfile.TemporaryDirectory(prefix='waveform_plot_') as folder:
            directory = Path(folder).resolve()
            self.assertTrue(directory.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            exported = export_waveform(self.commands, edges, settings, directory / 'csv')
            data = waveform_plot_data(self.model, self.commands, origin='rf',
                channel_map=[5, 8], channel_polarity=[1, -1],
                csv_file=exported['csv_file'], csv_origin_rf_s=.010)
            np.testing.assert_array_equal(data['csv']['time_s'], edges)
            np.testing.assert_allclose(data['csv']['time_ms'], edges * 1000 + 10., atol=2e-15)
            np.testing.assert_array_equal(data['csv']['commanded_A'][4, :-1], self.commands[0])
            self.assertTrue(data['report']['csv_commands_match_within_0p6mA'])
            self.assertTrue(data['report']['csv_model_event_times_match_within_1ns'])
            self.assertTrue(data['report']['csv_final_row_is_zero'])
            fig, report = plot_acdc_waveform(self.model, self.commands,
                origin='first_imaging_trigger', trigger_rf_s=.010,
                channel_map=[5, 8], channel_polarity=[1, -1],
                csv_file=exported['csv_file'], csv_origin_rf_s=.010)
            fig.canvas.draw()
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(report['selected_physical_output'], 8)
            plt.close(fig)

    def test_zeroed_or_invalid_CSV_times_and_implicit_origins_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='waveform_plot_') as folder:
            directory = Path(folder).resolve()
            self.assertTrue(directory.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            path = directory / 'waveform.csv'
            np.savetxt(path, [[0., .1, .2], [0., .2, .1]], delimiter=',')
            with self.assertRaisesRegex(ValueError, 'zeroed'):
                read_waveform_csv(path)
            with self.assertRaisesRegex(ValueError, 'csv_origin_rf_s'):
                waveform_plot_data(self.model, self.commands, csv_file=path)
            with self.assertRaisesRegex(ValueError, 'trigger_rf_s'):
                waveform_plot_data(self.model, self.commands, origin='first_imaging_trigger')


if __name__ == '__main__':
    unittest.main()
