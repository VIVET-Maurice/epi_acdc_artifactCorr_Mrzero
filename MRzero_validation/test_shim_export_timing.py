"""Verify trigger rebasing and CSV boundary rows without operating hardware."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Experiment'))
from export_acdc_waveform import export_waveform
from read_acdc_sequence import read_sequence
from shim_export_timing import sequence_trigger_timing, waveform_edges_for_origin


SEQUENCE = ROOT / 'acquisition_015/sequences/gre_epi_64x64_fov200mm_nav3_rise1ms_ext1_single_trigger_first_readout_v4.seq'
LINE_SEQUENCE = ROOT / 'acquisition_024_(06)/sequences/gre_epi_64x64_fov200mm_nav3_cimax_snr_t2star40ms_line_triggers_v6.seq'


def windows():
    starts = .01037 + np.arange(64) * .00271
    return starts, np.r_[starts[1:], starts[-1] + .00271]


class ExportOriginTests(unittest.TestCase):
    def test_verified_trigger_rebases_command_zero_and_final_return(self):
        starts, ends = windows()
        saved_starts, saved_ends = starts.copy(), ends.copy()
        edges, report = waveform_edges_for_origin(starts, ends, trigger_rf_s=starts[0])
        self.assertEqual(edges[0], 0.)
        self.assertAlmostEqual(edges[-1], .17344)
        np.testing.assert_allclose(np.diff(edges), .00271, atol=4e-17)
        self.assertEqual(report['csv_rows_with_final_zero_and_preamble'], 65)
        self.assertFalse(report['includes_zero_preamble'])
        self.assertTrue(report['final_zero_required'])
        np.testing.assert_array_equal(starts, saved_starts)
        np.testing.assert_array_equal(ends, saved_ends)

    def test_RF_origin_keeps_zero_preamble_and_model_interface(self):
        starts, ends = windows()
        model = SimpleNamespace(line_start_s=starts[:, None], line_end_s=ends[:, None])
        edges, report = waveform_edges_for_origin(model, origin='rf')
        self.assertEqual(edges[0], starts[0])
        self.assertEqual(edges[-1], ends[-1])
        self.assertTrue(report['includes_zero_preamble'])
        self.assertEqual(report['csv_rows_with_final_zero_and_preamble'], 66)

    def test_trigger_may_precede_RF_but_commands_cannot_precede_CSV_origin(self):
        starts, ends = windows()
        edges, _ = waveform_edges_for_origin(starts, ends, trigger_rf_s=-.001)
        np.testing.assert_allclose(edges, np.r_[starts, ends[-1]] + .001)
        with self.assertRaisesRegex(ValueError, 'before'):
            waveform_edges_for_origin(starts, ends, trigger_rf_s=starts[0] + 1e-5)

    def test_unknown_trigger_gaps_and_short_updates_are_rejected(self):
        starts, ends = windows()
        with self.assertRaisesRegex(ValueError, 'Explicit trigger'):
            waveform_edges_for_origin(starts, ends)
        gap_ends = ends.copy()
        gap_ends[0] -= 1e-4
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            waveform_edges_for_origin(starts, gap_ends, origin='rf')
        with self.assertRaisesRegex(ValueError, 'shorter'):
            waveform_edges_for_origin([.001, .001004], [.001004, .002], origin='rf')
        with self.assertRaises(ValueError):
            waveform_edges_for_origin(starts, ends, origin='unknown')

    def test_actual_trigger_events_are_required_not_filename_or_window_assumptions(self):
        # Use the system temp capability. A sibling workspace directory may
        # be sandbox-denied when tests run from MRzero_validation; Windows
        # tempfile retries that PermissionError instead of failing promptly.
        with tempfile.TemporaryDirectory(prefix='shim_timing_') as folder:
            directory = Path(folder).resolve()
            self.assertTrue(directory.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            path = directory / 'example.seq'
            path.write_text('mock sequence content', encoding='utf-8')
            event = SimpleNamespace(type='output', channel='ext1', delay=0., duration=1e-5)
            blocks = {1: SimpleNamespace(trigger=None), 2: SimpleNamespace(trigger={1: event})}
            sequence = SimpleNamespace(block_durations={1: .020, 2: .010},
                                       get_block=lambda number: blocks[number], read=lambda value: None)
            timing = dict(sequence_kind='epi', rf_center_s=.001, state_start_s=[.019])
            with patch('pypulseq.Sequence', return_value=sequence):
                result = sequence_trigger_timing(path, timing)
                self.assertAlmostEqual(result['trigger_rf_s'], .019)
                self.assertEqual(result['mode'], 'single_trigger')
                self.assertEqual(result['trigger_count'], 1)
                np.testing.assert_allclose(result['trigger_times_rf_s'], [.019])
                self.assertEqual(result['trigger_state_indices'], [0])
                self.assertFalse(result['verified_per_line_triggers'])
                self.assertTrue(result['verified_first_imaging_trigger'])
                self.assertFalse(result['hardware_latency_measured'])
                event.delay = 2e-6
                with self.assertRaisesRegex(ValueError, 'not at'):
                    sequence_trigger_timing(path, timing)
                event.delay = 0.
                blocks[2].trigger[2] = event
                with self.assertRaises(ValueError):
                    sequence_trigger_timing(path, timing)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    sequence_trigger_timing(path, dict(timing, sequence_sha256='wrong'))


class ImagingLineTriggerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='shim_line_triggers_')
        self.addCleanup(folder.cleanup)
        directory = Path(folder.name).resolve()
        self.assertTrue(directory.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.path = directory / 'example.seq'
        self.original = b'mock sequence content with a complete trigger train'
        self.path.write_bytes(self.original)
        self.starts, _ = windows()
        self.timing = dict(sequence_kind='epi', rf_center_s=.001,
                           state_start_s=self.starts.tolist())

    def verify(self, times, *, timing=None, other_events=()):
        events = [SimpleNamespace(type='output', channel='ext1',
                    delay=float(time) + .001, duration=1e-5) for time in times]
        events.extend(other_events)
        block = SimpleNamespace(trigger=dict(enumerate(events, start=1)))
        sequence = SimpleNamespace(block_durations={1: .4},
            get_block=lambda number: block, read=lambda value: None)
        with patch('pypulseq.Sequence', return_value=sequence):
            return sequence_trigger_timing(self.path,
                self.timing if timing is None else timing)

    def test_each_of_64_imaging_starts_has_a_verified_trigger(self):
        result = self.verify(self.starts)
        self.assertEqual(result['mode'], 'per_line')
        self.assertEqual(result['trigger_count'], 64)
        self.assertTrue(result['verified_per_line_triggers'])
        self.assertTrue(result['verified_first_imaging_trigger'])
        self.assertEqual(result['trigger_state_indices'], list(range(64)))
        np.testing.assert_allclose(result['trigger_times_rf_s'], self.starts,
                                   rtol=0, atol=1e-16)
        np.testing.assert_allclose(result['trigger_times_sequence_s'], self.starts + .001,
                                   rtol=0, atol=1e-16)
        self.assertLessEqual(result['maximum_trigger_alignment_error_s'],
                             result['trigger_alignment_tolerance_s'])
        self.assertAlmostEqual(result['trigger_rf_s'], self.starts[0])
        self.assertAlmostEqual(result['trigger_sequence_s'], self.starts[0] + .001)
        self.assertFalse(result['hardware_latency_measured'])
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_single_first_imaging_trigger_remains_valid_for_64_states(self):
        result = self.verify(self.starts[:1])
        self.assertEqual(result['mode'], 'single_trigger')
        self.assertEqual(result['trigger_count'], 1)
        self.assertFalse(result['verified_per_line_triggers'])
        self.assertEqual(result['trigger_state_indices'], [0])

    def test_missing_or_additional_line_triggers_are_rejected(self):
        for times in ([], self.starts[:-1], np.r_[self.starts, self.starts[-1] + .00271]):
            with self.subTest(count=len(times)), self.assertRaises(ValueError):
                self.verify(times)

    def test_duplicate_or_misaligned_internal_triggers_are_rejected(self):
        for shift in ('duplicate', 'late'):
            times = self.starts.copy()
            times[31] = times[30] if shift == 'duplicate' else times[31] + 2e-6
            with self.subTest(error=shift), self.assertRaises(ValueError):
                self.verify(times)

    def test_sub_nanosecond_clock_rounding_is_accepted_and_reported(self):
        times = self.starts.copy()
        times[31] += .5e-9
        result = self.verify(times)
        self.assertTrue(result['verified_per_line_triggers'])
        self.assertAlmostEqual(result['maximum_trigger_alignment_error_s'], .5e-9,
                               delta=1e-16)

    def test_navigator_trigger_cannot_be_used_as_the_imaging_origin(self):
        with self.assertRaises(ValueError):
            self.verify(np.r_[self.starts[0] - .00271, self.starts])
        # Even a count of 64 is insufficient when the first output is earlier
        # and the final imaging line has no corresponding output.
        with self.assertRaises(ValueError):
            self.verify(np.r_[self.starts[0] - .00271, self.starts[:-1]])

    def test_other_channels_and_input_events_do_not_count_as_ext1_outputs(self):
        other_events = [
            SimpleNamespace(type='output', channel='ext2', delay=.002, duration=1e-5),
            SimpleNamespace(type='trigger', channel='ext1', delay=.002, duration=1e-5),
        ]
        result = self.verify(self.starts, other_events=other_events)
        self.assertEqual(result['trigger_count'], 64)
        self.assertEqual(len(result['output_events']), 65)

    def test_invalid_imaging_clock_is_rejected(self):
        for reason in ('nonfinite', 'repeated', 'decreasing'):
            starts = self.starts.copy()
            if reason == 'nonfinite':
                starts[12] = np.nan
            elif reason == 'repeated':
                starts[12] = starts[11]
            else:
                starts[11:13] = starts[11:13][::-1]
            with self.subTest(error=reason), self.assertRaises(ValueError):
                self.verify(self.starts, timing=dict(self.timing, state_start_s=starts))


@unittest.skipUnless(SEQUENCE.is_file(), 'Actual acquisition_015 sequence is unavailable')
class ActualSequenceCSVTests(unittest.TestCase):
    def test_real_sequence_trigger_and_synthetic_command_CSV_dry_run(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            acquisition, timing = read_sequence(SEQUENCE)
            trigger = sequence_trigger_timing(SEQUENCE, timing)
        self.assertAlmostEqual(trigger['trigger_sequence_s'], .01147)
        self.assertAlmostEqual(trigger['trigger_rf_s'], .01037)
        edges, report = waveform_edges_for_origin(acquisition['line_start_s'],
            acquisition['line_end_s'], trigger_rf_s=trigger['trigger_rf_s'])
        self.assertEqual(edges[0], 0.)
        self.assertAlmostEqual(edges[-1], .17344)
        currents = .2 * np.sin(np.arange(32 * 64).reshape(32, 64) * .03)
        settings = dict(channel_map=list(range(1, 33)), channel_polarity=[1] * 32,
                        max_current_A=3., total_current_limit_A=30., min_dac_interval_s=8e-6)
        with tempfile.TemporaryDirectory(prefix='shim_csv_dryrun_') as folder:
            directory = Path(folder).resolve()
            # Verify the resolved recursive-cleanup target before leaving scope.
            self.assertTrue(directory.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            exported = export_waveform(currents, edges, settings, directory / 'csv',
                                       zero_time_column=False)
            csv = np.loadtxt(exported['csv_file'], delimiter=',')
            self.assertEqual(csv.shape, (65, 33))
            np.testing.assert_array_equal(csv[:, 0], edges)
            np.testing.assert_array_equal(csv[:-1, 1:], currents.T)
            np.testing.assert_array_equal(csv[-1, 1:], np.zeros(32))
            self.assertGreater(np.diff(csv[:, 0]).min(), 8e-6)
            self.assertTrue(report['zero_time_column_must_be_false'])


@unittest.skipUnless(LINE_SEQUENCE.is_file(), 'Actual 64-line-trigger sequence is unavailable')
class ActualLineTriggerSequenceTests(unittest.TestCase):
    def test_real_64_trigger_sequence_uses_first_imaging_trigger_as_origin(self):
        original = LINE_SEQUENCE.read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            acquisition, timing = read_sequence(LINE_SEQUENCE)
            trigger = sequence_trigger_timing(LINE_SEQUENCE, timing)
        self.assertEqual(trigger['mode'], 'per_line')
        self.assertEqual(trigger['trigger_count'], 64)
        self.assertTrue(trigger['verified_per_line_triggers'])
        self.assertTrue(trigger['verified_first_imaging_trigger'])
        self.assertEqual(trigger['trigger_state_indices'], list(range(64)))
        np.testing.assert_allclose(trigger['trigger_times_rf_s'],
                                   timing['state_start_s'], rtol=0, atol=1e-12)
        self.assertAlmostEqual(trigger['trigger_rf_s'], .00421)
        edges, report = waveform_edges_for_origin(acquisition['line_start_s'],
            acquisition['line_end_s'], trigger_rf_s=trigger['trigger_rf_s'])
        self.assertEqual(edges[0], 0.)
        self.assertEqual(report['command_state_count'], 64)
        self.assertEqual(edges.size, 65)
        self.assertGreater(np.diff(edges).min(), 8e-6)
        self.assertEqual(LINE_SEQUENCE.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
