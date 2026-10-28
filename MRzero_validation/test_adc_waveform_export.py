"""Current-monitor sampling schedule exports without any hardware action."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from adc_waveform_export import export_adc_waveform

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Experiment'))
from read_acdc_sequence import read_sequence

LINE_SEQUENCE = ROOT / 'acquisition_024_(06)/sequences/gre_epi_64x64_fov200mm_nav3_cimax_snr_t2star40ms_line_triggers_v6.seq'
SINGLE_SEQUENCE = ROOT / 'acquisition_015/sequences/gre_epi_64x64_fov200mm_nav3_rise1ms_ext1_single_trigger_first_readout_v4.seq'


class ADCWaveformTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='waveform_adc_')
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name).resolve()
        self.assertTrue(self.directory.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.timing = dict(rf_center_s=.001, state_start_s=[.01, .01076, .01176],
                           state_end_s=[.01076, .01176, .01276], sequence_duration_s=.014)
        self.trigger = dict(trigger_rf_s=.01, trigger_sequence_s=.011,
            trigger_times_rf_s=self.timing['state_start_s'], channel='ext1',
            trigger_count=3, mode='per_line', sequence_sha256='testfixture',
            sequence_duration_s=.014)
        self.mock = patch('adc_waveform_export.sequence_trigger_timing', return_value=self.trigger)
        self.verified = self.mock.start()
        self.addCleanup(self.mock.stop)

    def export(self, **kwargs):
        return export_adc_waveform('example.seq', self.timing, self.directory, **kwargs)

    def test_irregular_trigger_segments_restart_and_keep_one_numeric_column(self):
        result = self.export()
        self.verified.assert_called_once_with('example.seq', self.timing, channel='ext1')
        self.assertEqual(Path(result['csv_file']).name, 'waveform_adc.csv')
        self.assertEqual(result['segment_sample_counts'], [16, 20, 20])
        self.assertEqual(result['samples'], 56)
        self.assertEqual(result['trigger_count'], 3)
        rows = Path(result['csv_file']).read_text().splitlines()
        self.assertTrue(all(len(row.split(',')) == 1 for row in rows))
        self.assertEqual(rows[0], '0')
        self.assertEqual(rows[16], '0')
        self.assertEqual(rows[36], '0')
        values = np.loadtxt(result['csv_file'], ndmin=2)
        self.assertEqual(values.shape, (56, 1))
        np.testing.assert_allclose(values[:16, 0], np.arange(16) * 50e-6, rtol=0, atol=1e-18)
        self.assertAlmostEqual(values[15, 0], .00075)
        self.assertAlmostEqual(result['duration_s'], .00276)
        self.assertEqual(result['duration_source'], 'last_imaging_command_end')
        metadata = json.loads(Path(result['metadata_file']).read_text())
        self.assertEqual(metadata, result)
        self.assertEqual(metadata['csv_sha256'], hashlib.sha256(Path(result['csv_file']).read_bytes()).hexdigest())
        self.assertEqual(metadata['segments'][1]['csv_row_start'], 16)
        self.assertEqual(metadata['segments'][2]['csv_row_stop'], 56)
        for segment in metadata['segments']:
            self.assertLess(segment['final_sample_local_s'], segment['duration_s'])

    def test_exact_endpoint_is_excluded_despite_absolute_clock_roundoff(self):
        self.trigger['trigger_times_rf_s'] = [.01]
        result = self.export(duration_s=.001)
        self.assertEqual(result['samples'], 20)
        self.assertAlmostEqual(np.loadtxt(result['csv_file'])[-1], .00095)

    def test_custom_interval_and_partial_duration(self):
        result = self.export(interval_s=100e-6, duration_s=.0008)
        self.assertEqual(result['segment_sample_counts'], [8, 1])
        self.assertEqual(result['trigger_count'], 2)
        self.assertEqual(result['sequence_trigger_count'], 3)
        self.assertEqual(np.loadtxt(result['csv_file'])[-1], 0.)
        self.assertEqual(result['interval_s'], .0001)

    def test_stop_exactly_at_trigger_does_not_add_an_endpoint_segment(self):
        result = self.export(duration_s=.00076)
        self.assertEqual(result['trigger_count'], 1)
        self.assertEqual(result['segment_sample_counts'], [16])

    def test_single_trigger_uses_one_continuous_clock(self):
        self.trigger.update(trigger_times_rf_s=[.01], trigger_count=1, mode='single_trigger')
        result = self.export()
        self.assertEqual(result['trigger_count'], 1)
        self.assertEqual(result['segment_sample_counts'], [56])
        values = np.loadtxt(result['csv_file'])
        self.assertEqual(np.count_nonzero(values == 0), 1)
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_custom_duration_may_extend_past_imaging_until_sequence_end(self):
        result = self.export(duration_s=.003)
        self.assertAlmostEqual(result['stop_sequence_s'], .014)
        with self.assertRaisesRegex(ValueError, 'beyond'):
            self.export(duration_s=.00301)

    def test_invalid_intervals_and_durations_are_rejected_before_files_exist(self):
        for interval in (0., -1e-6, np.nan, np.inf, True, 1j, [50e-6]):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.export(interval_s=interval)
        for duration in (0., -1., np.nan, np.inf, True, 1j, [.001]):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.export(duration_s=duration)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_no_unsupported_hardware_minimum_but_optional_limit_is_enforced(self):
        result = self.export(interval_s=10e-6, duration_s=30e-6)
        self.assertEqual(result['samples'], 3)
        self.assertIsNone(result['min_interval_s'])
        with self.assertRaisesRegex(ValueError, 'at least'):
            self.export(interval_s=10e-6, min_interval_s=25e-6)
        for minimum in (0., -1., np.nan, True, [1e-6]):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                self.export(min_interval_s=minimum)

    def test_invalid_final_imaging_ends_and_sequence_duration_are_rejected(self):
        for ends in ([.01076, .01176], [.01076, np.nan, .01276],
                     [.01076, .009, .01276], [.01076, .01176, .014],
                     [.01076, .01176, .01276 + 1j]):
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                with patch.dict(self.timing, state_end_s=ends):
                    self.export()
        for duration in (0., np.nan, True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                with patch.dict(self.timing, sequence_duration_s=duration):
                    self.export()
        with patch.dict(self.timing, sequence_duration_s=.0141):
            with self.assertRaisesRegex(ValueError, 'actual sequence duration'):
                self.export()

    def test_identical_export_reuses_files_and_changed_options_get_next_name(self):
        first = self.export()
        before = {path: path.read_bytes() for path in self.directory.iterdir()}
        again = self.export()
        self.assertTrue(again['already_present'])
        self.assertEqual(again['csv_file'], first['csv_file'])
        self.assertEqual(before, {path: path.read_bytes() for path in self.directory.iterdir()})
        changed = self.export(interval_s=100e-6)
        self.assertEqual(Path(changed['csv_file']).name, 'waveform_adc2.csv')
        self.assertFalse(changed['already_present'])
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_corrupted_existing_files_and_orphan_metadata_are_preserved(self):
        (self.directory / 'waveform_adc.csv').write_bytes(b'original data\n')
        (self.directory / 'waveform_adc.json').write_text('[]')
        (self.directory / 'waveform_adc2.json').write_text('orphan sidecar')
        before = {path: path.read_bytes() for path in self.directory.iterdir()}
        result = self.export()
        self.assertEqual(Path(result['csv_file']).name, 'waveform_adc3.csv')
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_changed_source_metadata_does_not_reuse_same_csv_times(self):
        first = self.export()
        self.trigger['sequence_sha256'] = 'changedfixture'
        second = self.export()
        self.assertNotEqual(first['csv_file'], second['csv_file'])
        self.assertEqual(Path(first['csv_file']).read_bytes(), Path(second['csv_file']).read_bytes())


@unittest.skipUnless(LINE_SEQUENCE.is_file(), 'Actual 64-line-trigger sequence is unavailable')
class ActualLineTriggerTests(unittest.TestCase):
    def test_actual_64_trigger_sequence_resets_at_every_line(self):
        original = LINE_SEQUENCE.read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, timing = read_sequence(LINE_SEQUENCE)
            with tempfile.TemporaryDirectory(prefix='waveform_adc_real_') as folder:
                result = export_adc_waveform(LINE_SEQUENCE, timing, folder)
                csv_bytes = Path(result['csv_file']).read_bytes()
                values = np.loadtxt(result['csv_file'])
        self.assertEqual(result['trigger_count'], 64)
        self.assertEqual(result['trigger']['mode'], 'per_line')
        self.assertEqual(result['samples'], 1024)
        self.assertEqual(result['segment_sample_counts'], [16] * 64)
        self.assertAlmostEqual(result['duration_s'], .04864)
        self.assertEqual(np.count_nonzero(values == 0), 64)
        self.assertEqual(len(values), sum(result['segment_sample_counts']))
        for segment in result['segments']:
            part = values[segment['csv_row_start']:segment['csv_row_stop']]
            self.assertEqual(part[0], 0.)
            self.assertLess(part[-1], segment['duration_s'])
            np.testing.assert_allclose(np.diff(part), 50e-6, rtol=0, atol=1e-17)
        self.assertEqual(hashlib.sha256(csv_bytes).hexdigest(), result['csv_sha256'])
        self.assertEqual(LINE_SEQUENCE.read_bytes(), original)


@unittest.skipUnless(SINGLE_SEQUENCE.is_file(), 'Actual single-trigger sequence is unavailable')
class ActualSingleTriggerTests(unittest.TestCase):
    def test_actual_single_trigger_sequence_keeps_one_clock(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, timing = read_sequence(SINGLE_SEQUENCE)
            with tempfile.TemporaryDirectory(prefix='waveform_adc_single_') as folder:
                result = export_adc_waveform(SINGLE_SEQUENCE, timing, folder)
                values = np.loadtxt(result['csv_file'])
        self.assertEqual(result['trigger_count'], 1)
        self.assertEqual(np.count_nonzero(values == 0), 1)
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertLess(values[-1], result['duration_s'])


if __name__ == '__main__':
    unittest.main()
