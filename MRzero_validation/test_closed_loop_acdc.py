"""Persistent orchestration with explicit toy images/readers and REAL CSV checks.

These fixtures are synthetic software tests, never scanner measurements. Actual
command construction, signed16 quantization, continuous hardware diagnostics,
CSV export/reload, shared image metrics and state hashing run without mocks.
Only the sequence trigger parser, Twix reader/protocol classification, receiver
reconstruction and the physical forward model are replaced with toy fixtures.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import closed_loop_acdc as loop
import closed_loop_commands as commands
from measured_scan_comparison import compare_registered_images
from test_closed_loop_commands import make_context, historical


class ClosedLoopOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='closed_loop_orchestration_test_')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.context = make_context(self.folder)
        self.csv, self.npz, _ = historical(self.folder, self.context)
        self.old = commands.load_historical_commands(self.context, csv_path=self.csv,
                                                     npz_path=self.npz)['quantized_currents_A']
        self.paths = {'scan_sequence': self.folder/'example.seq'}
        for name in ('reference', 'b0', 'scan'):
            self.paths[name] = self.folder/f'{name}.dat'
            self.paths[name].write_text(f'EXPLICIT SYNTHETIC UNIT TEST FILE: {name}')
        self.previous = self.folder/'historical.dat'
        self.previous.write_text('EXPLICIT SYNTHETIC UNIT TEST FILE: historical')
        self.checker = 2*((np.indices((8, 8)).sum(axis=0) % 2).astype(float)) - 1
        self.reference = np.ones((8, 8))
        self.baseline = self.reference+.4*self.checker
        self.previous_image = self.reference+.2*self.checker
        self.X, self.Y = np.meshgrid((np.arange(8)-4)*.025, (np.arange(8)-4)*.025)
        self.mask = np.zeros((8, 8), bool); self.mask[1:7, 1:7] = True
        self.acquisition = {'FOV_m': [.2, .2], 'kx_mInv': np.zeros((8, 8)),
            'TE_s': .02, 'TR_s': .006, 'dwell_s': 5e-6,
            'polarity': [1, -1]*4, 'line_index': np.repeat(np.arange(8)[:, None], 8, axis=1)}
        self.metadata = {'slice_position_mm': [0., 0., 0.],
            'slice_quaternion_wxyz': [1., 0., 0., 0.], 'slice_thickness_mm': 5.,
            'header_flip_deg': 90., 'receiver_count': 2, 'FOV_m': [.2, .2],
            'image_shape': [8, 8], 'TE_s': .02, 'TR_s': .006, 'dwell_s': 5e-6,
            'navigator_count': 3, 'polarity': [1, -1]*4, 'acquired_phase_rows': list(range(8))}
        self.inputs = {'paths': self.paths, 'manifest': {'format': 'mrzero_synthetic_twix'},
            'source_sha256': {'scan_sequence': loop.sha256(self.paths['scan_sequence'])},
            'timing': {'state_start_s': self.context['line_start_s'],
                       'state_end_s': self.context['line_end_s'], 'navigator_count': 3},
            'scan_metadata': deepcopy(self.metadata), 'acquisition': self.acquisition,
            'gre': {'twix_metadata': deepcopy(self.metadata)}}
        self.prepared = {'report': {'calibration_source_sha256': 'b'*64}}
        # Reuse the real standalone hardware fixture as the model's command API.
        hardware = commands._hardware(self.context)
        self.model = hardware.model
        self.model.hardware = hardware
        self.model.current_response = deepcopy(self.context['metadata']['current_response'])
        self.model.reconstruction = SimpleNamespace(X_m=self.X, Y_m=self.Y)
        self.model.forward = Mock(side_effect=self._toy_forward)
        self.images = {}
        self.scan_metadata = {}
        self.kinds = {}
        for target, replacement in (
                ((commands, 'sequence_trigger_timing'), {'return_value': self.context['metadata']['trigger']}),
                ((loop, 'twix_data_kind'), {'side_effect': lambda path: self.kinds.get(str(Path(path)), 'synthetic')}),
                ((loop, 'compare_measured_scan_set'), {'side_effect': self._comparison}),
                ((loop, 'read_scan'), {'side_effect': self._read_scan}),
                ((loop, 'reconstruct_receivers'), {'side_effect': lambda diagnostics, acquisition:
                                                  (diagnostics['toy_RSS_image'], {'fixture': 'toy receiver image'})})):
            mock = patch.object(target[0], target[1], **replacement)
            started = mock.start(); self.addCleanup(mock.stop)
            if target[1] == 'compare_measured_scan_set': self.comparison_reader = started
            if target[1] == 'read_scan': self.feedback_reader = started

    def _toy_forward(self, currents):
        alpha = float(np.sum(np.asarray(currents)*self.old)/np.sum(self.old**2))
        amplitude = .1+.8*(alpha-.75)**2
        return (self.reference+amplitude*self.checker).astype(complex), np.zeros((8, 8), complex)

    def _comparison(self, inputs, paths, *, frame_matrix=None, roi_mask=None, **options):
        records = [{'image': image, 'X_m': self.X, 'Y_m': self.Y, 'label': label}
                   for image, label in ((self.reference, 'toy GRE'), (self.baseline, 'toy baseline'),
                                        (self.previous_image, 'toy historical'))]
        result = compare_registered_images(*records[:2], [records[2]], frame_matrix=frame_matrix,
            roi_mask=roi_mask, normalize=True,
            reference_threshold_fraction=options['reference_threshold_fraction'])
        result['metadata'] = [deepcopy(self.metadata) for _ in range(3)]
        return result

    def _read_scan(self, path, acquisition, **options):
        key = str(Path(path))
        metadata = deepcopy(self.scan_metadata.get(key, self.metadata))
        return None, metadata, {'toy_RSS_image': self.images[key], 'navigator_fit': {'toy': True}}

    def configuration(self, **overrides):
        result = {'previous_scan': self.previous, 'previous_csv': self.csv, 'previous_npz': self.npz,
            'association_confirmed': True, 'data_kind': 'synthetic', 'alpha_points': 21,
            'allow_exploration': True, 'roi_mask': self.mask,
            'waveform_parameters': deepcopy(self.context['metadata']['settings']),
            'csv_time_origin': 'first_imaging_trigger', 'trigger_channel': 'ext1', 'field_limit_Hz': 500.}
        return dict(result, **overrides)

    def initialize(self, name='session', **overrides):
        with redirect_stdout(io.StringIO()):
            return loop.run_round0(self.folder/name, configuration=self.configuration(**overrides),
                inputs=self.inputs, model=self.model, prepared=self.prepared, frame_matrix=np.eye(2),
                simulation_metadata={'test_fixture': 'toy forward, no MRzero/scanner validation'})

    def feedback_file(self, name, amplitude):
        path = self.folder/f'{name}.dat'
        path.write_text(f'EXPLICIT SYNTHETIC UNIT TEST FILE: {name}')
        self.images[str(path)] = self.reference+amplitude*self.checker
        return path

    def test_two_rounds_real_metric_improvement_then_degradation_retains_observed_best(self):
        initial = self.initialize()
        folder = initial['session_dir']
        self.assertEqual(len(initial['manifest']['proposals']), 1)
        normalization = deepcopy(initial['manifest']['normalization'])
        forward_count = self.model.forward.call_count
        first = loop.run_feedback_round(folder, round_index=1,
            scan_path=self.feedback_file('new_1_better', .05), association_confirmed=True)
        self.assertEqual(first['manifest']['best_measured_id'], 'round_1')
        self.assertEqual(first['manifest']['incumbent_id'], 'round_1')
        self.assertEqual(len(first['manifest']['proposals']), 2)
        second = loop.run_feedback_round(folder, round_index=2,
            scan_path=self.feedback_file('new_2_worse', .65), association_confirmed=True)
        self.assertEqual(second['manifest']['best_measured_id'], 'round_1')
        self.assertEqual(second['manifest']['incumbent_id'], 'round_1')
        self.assertIsNone(second['manifest']['pending_proposal_id'])
        self.assertEqual(second['manifest']['search']['budget']['remaining_additional_scans'], 0)
        self.assertEqual(second['manifest']['normalization'], normalization)
        np.testing.assert_array_equal(initial['arrays']['mask'], second['arrays']['mask'])
        gain = normalization['experimental_gain']/normalization['reference_scale']
        np.testing.assert_allclose(second['arrays']['image_round_2'], gain*self.images[str(self.folder/'new_2_worse.dat')])
        point = second['manifest']['measured_points'][-1]
        score = loop.relative_image_cost(second['arrays'][point['image_key']], second['arrays']['reference_image'], self.mask)
        self.assertEqual(point['experimental_J'], score['J'])
        self.assertFalse(point['diagnostics']['shared_gain_refitted'])
        self.assertFalse(point['diagnostics']['ROI_refitted'])
        self.assertEqual(self.comparison_reader.call_count, 1)
        self.assertEqual(self.model.forward.call_count, forward_count)
        self.assertEqual(len(list(Path(folder).glob('round_*/*.csv'))), 4)
        resumed = loop.load_closed_loop(folder)
        np.testing.assert_array_equal(resumed['arrays']['reference_image'], initial['arrays']['reference_image'])

    def test_resume_and_reimport_are_idempotent_without_model_or_extra_csv(self):
        initial = self.initialize()
        folder = initial['session_dir']
        self.model.forward.side_effect = AssertionError('Resume must not construct/simulate a model.')
        with redirect_stdout(io.StringIO()):
            resumed = loop.run_round0(folder, configuration=self.configuration())
        self.assertEqual(len(resumed['manifest']['proposals']), 1)
        self.assertEqual(len(list(Path(folder).glob('round_*/*.csv'))), 2)
        path = self.feedback_file('first', .1)
        first = loop.run_feedback_round(folder, round_index=1, scan_path=path, association_confirmed=True)
        reader_count = self.feedback_reader.call_count
        repeated = loop.run_feedback_round(folder, round_index=1, scan_path=path, association_confirmed=True)
        self.assertEqual(len(repeated['manifest']['measured_points']), 3)
        self.assertEqual(len(repeated['manifest']['proposals']), len(first['manifest']['proposals']))
        self.assertEqual(self.feedback_reader.call_count, reader_count)
        self.assertEqual(len(list(Path(folder).glob('round_*/*.csv'))), 4)

    def test_acquisition_waveform_folder_preserves_verified_metadata_and_resume(self):
        output = self.folder / 'acquisition_021_(06)' / 'waveform'
        initial = self.initialize(waveform_output_dir=output)
        folder = Path(initial['session_dir'])
        proposal = initial['manifest']['proposals'][0]
        csv = Path(proposal['csv']['path'])
        self.assertEqual(csv.parent.name, 'round_1')
        self.assertEqual(csv.parents[2], output / 'closed_loop')
        self.assertTrue(csv.parent.parent.name.startswith('session_'))
        self.assertFalse(list(folder.glob('round_*/*.csv')))
        self.assertEqual({path.name for path in csv.parent.iterdir()},
                         {'waveform1.csv', 'waveform1_zero_time.csv', 'commands.npz', 'commands.json'})
        self.assertEqual(proposal['csv']['sha256'], loop.sha256(csv))
        self.assertEqual(proposal['export']['csv_file'], str(csv))
        self.assertEqual(proposal['export']['source_sha256']['csv'], loop.sha256(csv))
        companion = Path(proposal['export']['csv_zero_time_file'])
        self.assertEqual(companion, csv.with_name('waveform1_zero_time.csv'))
        self.assertEqual(proposal['export']['source_sha256']['csv_zero_time'], loop.sha256(companion))
        timestamped = np.loadtxt(csv, delimiter=',')
        zero_time = np.loadtxt(companion, delimiter=',')
        np.testing.assert_array_equal(zero_time[:, 0], 0.)
        np.testing.assert_array_equal(zero_time[:, 1:], timestamped[:, 1:])
        for record in proposal['export_files'].values():
            self.assertEqual(Path(record['path']).parent, csv.parent)
            self.assertEqual(record['sha256'], loop.sha256(record['path']))
        metadata = json.loads((csv.parent / 'commands.json').read_text())
        self.assertEqual(metadata['waveform_export']['csv_file'], str(csv))
        original = {path: path.read_bytes() for path in csv.parent.iterdir()}
        self.model.forward.side_effect = AssertionError('Resume must not simulate a new model.')
        with redirect_stdout(io.StringIO()):
            resumed = loop.run_round0(folder, configuration=self.configuration(waveform_output_dir=output))
        self.assertEqual(len(resumed['manifest']['proposals']), 1)
        self.assertEqual(len(list(output.rglob('*.csv'))), 2)
        for path, content in original.items(): self.assertEqual(path.read_bytes(), content)
        first = loop.run_feedback_round(folder, round_index=1,
            scan_path=self.feedback_file('external_first', .1), association_confirmed=True)
        self.assertEqual(len(first['manifest']['proposals']), 2)
        self.assertEqual(len(list(output.rglob('*.csv'))), 4)
        self.assertFalse(list(folder.glob('round_*/*.csv')))
        loop.load_closed_loop(folder)
        for target in original:
            target.write_bytes(original[target]+b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Fichier modifié'):
                loop.load_closed_loop(folder)
            target.write_bytes(original[target])
        loop.load_closed_loop(folder)
        companion.unlink()
        # A missing companion cannot bypass verification of the canonical
        # timestamped waveform or silently regenerate commands on resume.
        with self.assertRaises((ValueError, FileNotFoundError)):
            loop.load_closed_loop(folder)
        self.assertEqual(csv.read_bytes(), original[csv])

    def test_waveform_output_directory_is_frozen_and_legacy_sessions_resume_in_place(self):
        output = self.folder / 'acquisition' / 'waveform'
        configured = self.initialize('configured', waveform_output_dir=output)
        with self.assertRaisesRegex(ValueError, 'waveform_output_dir'):
            loop.run_round0(configured['session_dir'],
                configuration=self.configuration(waveform_output_dir=self.folder / 'another_waveform'))
        legacy = self.initialize('legacy')
        del legacy['manifest']['configuration']['waveform_output_dir']
        loop._save(legacy)
        self.model.forward.side_effect = AssertionError('Legacy resume must not simulate.')
        with redirect_stdout(io.StringIO()):
            resumed = loop.run_round0(legacy['session_dir'], configuration=self.configuration())
        self.assertNotIn('waveform_output_dir', resumed['manifest']['configuration'])
        self.assertEqual(len(list(Path(legacy['session_dir']).glob('round_*/*.csv'))), 2)
        with self.assertRaisesRegex(ValueError, 'waveform_output_dir'):
            loop.run_round0(legacy['session_dir'], configuration=self.configuration(waveform_output_dir=output))

    def test_sessions_with_identical_names_have_distinct_waveform_folders(self):
        output = self.folder / 'waveform'
        first = self.initialize('one/session', waveform_output_dir=output)
        second = self.initialize('two/session', waveform_output_dir=output)
        directories = [Path(state['manifest']['proposals'][0]['csv']['path']).parent.parent
                       for state in (first, second)]
        self.assertNotEqual(*directories)
        self.assertTrue(all(path.name.startswith('session_') for path in directories))
        self.assertEqual(len(list(output.rglob('*.csv'))), 4)
        loop.load_closed_loop(first['session_dir']); loop.load_closed_loop(second['session_dir'])

    def test_session_exported_before_zero_time_companion_resumes_unchanged(self):
        state = self.initialize('legacy_no_companion')
        proposal = state['manifest']['proposals'][0]
        exported = proposal['export']
        companion = Path(exported.pop('csv_zero_time_file'))
        exported.pop('zero_time_export')
        exported['source_sha256'].pop('csv_zero_time')
        companion.unlink()
        proposal['export_files'].pop(companion.name)
        metadata_path = Path(exported['metadata_file'])
        metadata = json.loads(metadata_path.read_text())
        metadata['waveform_export'].pop('csv_zero_time_file')
        metadata['waveform_export'].pop('zero_time_export')
        metadata_path.write_text(json.dumps(metadata))
        exported['source_sha256']['metadata'] = loop.sha256(metadata_path)
        proposal['export_files'][metadata_path.name]['sha256'] = loop.sha256(metadata_path)
        loop._save(state)
        existing = {path: path.read_bytes() for path in metadata_path.parent.iterdir()}
        self.model.forward.side_effect = AssertionError('Legacy resume must not simulate.')
        with redirect_stdout(io.StringIO()):
            resumed = loop.run_round0(state['session_dir'], configuration=self.configuration())
        self.assertEqual(len(resumed['manifest']['proposals']), 1)
        self.assertFalse(companion.exists())
        self.assertNotIn('csv_zero_time_file', resumed['manifest']['proposals'][0]['export'])
        self.assertEqual(set(metadata_path.parent.iterdir()), set(existing))
        for path, content in existing.items():
            self.assertEqual(path.read_bytes(), content)

    def test_waveform_output_directory_rejects_invalid_values_before_export(self):
        regular_file = self.folder / 'not_a_directory'
        regular_file.write_text('existing file')
        for invalid in ('', '   ', 42, True, regular_file):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'waveform_output_dir'):
                self.initialize('invalid_output', waveform_output_dir=invalid)
        self.assertFalse(list((self.folder / 'invalid_output').glob('round_*/*.csv')))

    def test_zero_and_one_additional_scan_budgets(self):
        zero = self.initialize('zero', max_additional_scans=0)
        self.assertEqual(zero['manifest']['proposals'], [])
        self.assertFalse(list(Path(zero['session_dir']).glob('round_*/*.csv')))
        one = self.initialize('one', max_additional_scans=1)
        final = loop.run_feedback_round(one['session_dir'], round_index=1,
            scan_path=self.feedback_file('only_scan', .1), association_confirmed=True)
        self.assertEqual(len(final['manifest']['proposals']), 1)
        self.assertIsNone(final['manifest']['pending_proposal_id'])
        self.assertEqual(final['manifest']['search']['budget']['remaining_additional_scans'], 0)

    def test_no_csv_when_no_candidate_predicts_the_minimum_gain(self):
        self.model.forward.side_effect = lambda currents: (self.baseline.astype(complex), np.zeros((8, 8), complex))
        state = self.initialize(allow_exploration=False)
        self.assertIsNone(state['manifest']['search']['candidate'])
        self.assertEqual(state['manifest']['search']['stop_reason'], 'no_promising_local_candidate')
        self.assertEqual(state['manifest']['proposals'], [])
        self.assertFalse(list(Path(state['session_dir']).glob('round_*/*.csv')))

    def test_round_index_requires_a_nonboolean_integer(self):
        state = self.initialize()
        for invalid in (True, False, np.bool_(True), 1., 2., 0, 3, '1'):
            with self.subTest(round_index=invalid), self.assertRaises(ValueError):
                loop.run_feedback_round(state['session_dir'], round_index=invalid)

    def test_four_actual_figures_are_saved_without_model_reconstruction(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        state = self.initialize()
        state = loop.load_closed_loop(state['session_dir'])
        figures = []
        try:
            figures = loop.plot_closed_loop(state)
            self.assertEqual(len(figures), 4)
            for name in ('cost', 'images', 'residuals', 'currents'):
                path = Path(state['session_dir'])/f'round_0_{name}.png'
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)
        finally:
            for figure in figures: plt.close(figure)

    def test_unconfirmed_association_missing_file_and_duplicate_baseline_never_export(self):
        for name, options, exception in (
                ('unconfirmed', {'association_confirmed': False}, ValueError),
                ('missing', {'previous_scan': self.folder/'missing.dat'}, FileNotFoundError),
                ('duplicate', {'previous_scan': self.paths['scan']}, ValueError)):
            with self.subTest(name=name), self.assertRaises(exception): self.initialize(name, **options)
            self.assertFalse(list((self.folder/name).glob('round_*/*.csv')))

    def test_duplicate_feedback_file_wrong_association_and_out_of_order_round_are_rejected(self):
        state = self.initialize(); folder = state['session_dir']
        duplicate = self.folder/'copied_historical.dat'; duplicate.write_bytes(self.previous.read_bytes())
        path = self.feedback_file('new', .1)
        for options in (
                {'round_index': 1, 'scan_path': duplicate, 'association_confirmed': True},
                {'round_index': 1, 'scan_path': path, 'association_confirmed': False},
                {'round_index': 2, 'scan_path': path, 'association_confirmed': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                loop.run_feedback_round(folder, **options)
        final = loop.load_closed_loop(folder)
        self.assertEqual(len(final['manifest']['measured_points']), 2)
        self.assertEqual(len(final['manifest']['proposals']), 1)

    def test_frozen_configuration_and_feedback_geometry_reject_changes(self):
        state = self.initialize(); folder = state['session_dir']
        with self.assertRaisesRegex(ValueError, 'Configuration'):
            loop.run_round0(folder, configuration=self.configuration(reference_threshold_fraction=.2))
        path = self.feedback_file('shifted', .1)
        bad = deepcopy(self.metadata); bad['slice_position_mm'][0] += 1
        self.scan_metadata[str(path)] = bad
        with self.assertRaisesRegex(ValueError, 'centres differ'):
            loop.run_feedback_round(folder, round_index=1, scan_path=path, association_confirmed=True)
        self.assertEqual(len(loop.load_closed_loop(folder)['manifest']['measured_points']), 2)
        self.assertEqual(len(list(Path(folder).glob('round_*/*.csv'))), 2)

    def test_initial_model_grid_mismatch_never_exports(self):
        self.model.reconstruction.X_m = self.X+.001
        with self.assertRaisesRegex(ValueError, 'grid differs'):
            self.initialize()
        self.assertFalse(list((self.folder/'session').glob('round_*/*.csv')))

    def test_feedback_cannot_shrink_frozen_roi_to_drop_invalid_pixels(self):
        state = self.initialize(); path = self.feedback_file('missing_pixels', .1)
        self.images[str(path)][3, 3] = np.nan
        with self.assertRaisesRegex(ValueError, 'entire frozen ROI'):
            loop.run_feedback_round(state['session_dir'], round_index=1, scan_path=path, association_confirmed=True)
        self.assertEqual(len(loop.load_closed_loop(state['session_dir'])['manifest']['measured_points']), 2)

    def test_protocol_origin_mixing_is_rejected_initially_and_on_feedback(self):
        self.kinds[str(self.previous)] = 'measured'
        with self.assertRaisesRegex(ValueError, 'Data origin mismatch'):
            self.initialize('mixed_initial')
        self.kinds.clear()
        state = self.initialize(); path = self.feedback_file('copied_real_feedback', .1)
        self.kinds[str(path)] = 'measured'
        with self.assertRaisesRegex(ValueError, 'Data origin mismatch'):
            loop.run_feedback_round(state['session_dir'], round_index=1, scan_path=path, association_confirmed=True)
        self.assertEqual(self.feedback_reader.call_count, 0)

    def test_historical_csv_npz_mismatch_cannot_initialize(self):
        values = np.loadtxt(self.csv, delimiter=','); values[0, 1] += .01
        original_header = self.csv.read_text().splitlines()[0][2:]
        np.savetxt(self.csv, values, delimiter=',', fmt='%.17g', header=original_header)
        with self.assertRaises(ValueError): self.initialize()
        self.assertFalse(list((self.folder/'session').glob('round_*/*.csv')))

    def test_all_persistent_source_and_export_hashes_are_checked(self):
        # Each mutation has a fresh session so no preceding failure masks it.
        targets = ('source', 'context', 'export', 'arrays', 'manifest')
        for name in targets:
            with self.subTest(target=name):
                state = self.initialize('checksum_'+name)
                folder, manifest = Path(state['session_dir']), state['manifest']
                if name == 'source': target = self.paths['b0']
                elif name == 'context': target = Path(manifest['command_context']['path'])
                elif name == 'export': target = Path(manifest['proposals'][0]['csv']['path'])
                elif name == 'arrays': target = folder/manifest['arrays_file']
                else:
                    pointer = json.loads((folder/'session.json').read_text()); target = folder/pointer['state_file']
                original = target.read_bytes(); target.write_bytes(original+b'corrupt')
                with self.assertRaises(ValueError): loop.load_closed_loop(folder)
                target.write_bytes(original)
                loop.load_closed_loop(folder)


if __name__ == '__main__':
    unittest.main()
