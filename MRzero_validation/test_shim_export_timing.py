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
                self.assertTrue(result['verified_first_imaging_trigger'])
                self.assertFalse(result['hardware_latency_measured'])
                event.delay = 2e-6
                with self.assertRaisesRegex(ValueError, 'not at'):
                    sequence_trigger_timing(path, timing)
                event.delay = 0.
                blocks[2].trigger[2] = event
                with self.assertRaisesRegex(ValueError, 'exactly one'):
                    sequence_trigger_timing(path, timing)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    sequence_trigger_timing(path, dict(timing, sequence_sha256='wrong'))


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


if __name__ == '__main__':
    unittest.main()
