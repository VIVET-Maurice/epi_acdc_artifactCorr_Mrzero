"""Continuous hardware checks and real CSV round trips for field-only shims.

The tiny models and solver outputs are explicit software fixtures. No measured
acquisition, scanner, controller or image optimization is executed here.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

from acdc_hardware import HardwareConstraints
from export_acdc_waveform import quantize_currents
from read_acdc_sequence import read_sequence
import field_shim_export as exports
from test_acdc_hardware import tiny_model


LINE_TRIGGER_SEQUENCE = (Path(__file__).resolve().parent.parent / 'acquisition_024_(06)' /
    'sequences/gre_epi_64x64_fov200mm_nav3_cimax_snr_t2star40ms_line_triggers_v6.seq')


def contiguous_model(tau=(50e-6, 150e-6), *, cycle=800e-6, **configuration):
    model = tiny_model(tau=tau)
    edges = np.array([20e-6, 80e-6, 160e-6, 320e-6])
    model.line_start_s, model.line_end_s = edges[:-1], edges[1:]
    config = dict(cycle_duration_s=cycle, **configuration)
    model.hardware = HardwareConstraints(model, config)
    return model, edges


class ContinuousFieldWaveformTests(unittest.TestCase):
    def test_common_and_unequal_tau_match_existing_continuous_hardware_metrics(self):
        currents = np.array([[1.2, -1.1, .7], [-.6, 1.7, -.8]])
        for tau in ((50e-6, 150e-6), (60e-6, 60e-6)):
            with self.subTest(tau=tau):
                model, edges = contiguous_model(tau=tau, coil_R_ohm=[.3, .5],
                    coil_L_H=[30e-6, 80e-6], max_slew_A_per_s=1e8,
                    avg_abs_current_A=2., max_voltage_V=100., max_coil_power_W=10.)
                expected = model.hardware.diagnostics(currents, 20., 100.)
                actual = exports.waveform_hardware_diagnostics(model, currents, edges,
                    total_current_limit_A=20., field_limit_Hz=100.)
                for key in ('peak_actual_channel_current_A', 'peak_slew_A_per_s',
                            'peak_total_current_bound_A', 'peak_field_bound_Hz',
                            'avg_abs_current_A', 'mean_square_current_A2', 'end_current_A',
                            'average_coil_power_W', 'peak_coil_voltage_V'):
                    np.testing.assert_allclose(actual[key], expected[key], rtol=2e-13, atol=1e-12,
                                               err_msg=f'{key}, tau={tau}')
                self.assertTrue(actual['all_limits_met'])
                for key, value in expected['checks'].items():
                    self.assertEqual(actual['checks'][key], value)

    def test_final_zero_command_at_TR_is_included_in_slew_and_voltage(self):
        model, _ = contiguous_model(tau=(100e-6, 100e-6), cycle=800e-6,
                                   coil_R_ohm=.3, coil_L_H=80e-6)
        edges = np.array([0., 80e-6, 160e-6, 800e-6])
        model.line_start_s, model.line_end_s = edges[:-1], edges[1:]
        model.hardware = HardwareConstraints(model, model.hardware.config)
        currents = np.array([[.9, .9, 1.], [.9, .9, 1.]])
        expected = model.hardware.diagnostics(currents, 20., 100.)
        actual = exports.waveform_hardware_diagnostics(model, currents, edges,
            total_current_limit_A=20., field_limit_Hz=100.)
        self.assertGreater(np.max(actual['peak_slew_A_per_s']), 9900.)
        np.testing.assert_allclose(actual['peak_slew_A_per_s'], expected['peak_slew_A_per_s'],
                                   rtol=1e-13)
        np.testing.assert_allclose(actual['peak_coil_voltage_V'], expected['peak_coil_voltage_V'],
                                   rtol=1e-13)
        np.testing.assert_allclose(actual['end_current_A'], expected['end_current_A'], rtol=1e-13)

    def test_projection_contracts_all_commands_uniformly_and_checks_optional_limits_after_DAC(self):
        model, edges = contiguous_model(tau=(15e-6, 150e-6),
            max_slew_A_per_s=[4000., 3000.], avg_abs_current_A=[.04, .03],
            coil_R_ohm=[.3, .5], coil_L_H=[30e-6, 80e-6], max_voltage_V=[.2, .15],
            max_coil_power_W=[.001, .002])
        original = np.array([[1.7, -1.5, .9], [1.7, -1.5, .9]])
        saved = original.copy()
        commands, quantized, report = exports.prepare_field_waveform(model, original, edges,
            total_current_limit_A=.3, field_limit_Hz=.5)
        self.assertLess(report['uniform_contraction'], 1.)
        self.assertFalse(report['solver_optimum_preserved'])
        np.testing.assert_allclose(commands / original, report['uniform_contraction'], rtol=1e-13)
        np.testing.assert_array_equal(original, saved)
        np.testing.assert_array_equal(quantized, quantize_currents(commands))
        self.assertTrue(report['commands']['all_limits_met'], report['commands']['checks'])
        self.assertTrue(report['nominal_DAC']['all_limits_met'], report['nominal_DAC']['checks'])
        self.assertFalse(report['raw']['all_limits_met'])
        independently_checked = model.hardware.diagnostics(quantized, .3, .5)
        self.assertTrue(independently_checked['all_limits_met'], independently_checked['checks'])

    def test_finite_voxel_corner_basis_is_enforced_even_when_center_field_is_safe(self):
        model, edges = contiguous_model(tau=(60e-6, 60e-6))
        model.B_quality = model.tensor([[1., -1.], [100., -100.]])
        currents = np.array([[1., 1., 1.], [0., 0., 0.]])
        raw = exports.waveform_hardware_diagnostics(model, currents, edges,
            total_current_limit_A=20., field_limit_Hz=5.)
        self.assertEqual(raw['peak_command_field_Hz'], 100.)
        self.assertFalse(raw['checks']['command_field'])
        commands, quantized, report = exports.prepare_field_waveform(model, currents, edges,
            total_current_limit_A=20., field_limit_Hz=5.)
        self.assertLess(report['uniform_contraction'], .05)
        self.assertLessEqual(np.max(np.abs(model.numpy(model.B_quality) @ commands)), 5.)
        self.assertLessEqual(np.max(np.abs(model.numpy(model.B_quality) @ quantized)), 5.)
        self.assertTrue(report['nominal_DAC']['all_limits_met'])

    def test_zero_and_feasible_commands_preserve_solver_optimum(self):
        model, edges = contiguous_model()
        for currents in (np.zeros((2, 3)), np.full((2, 3), .001)):
            with self.subTest(amplitude=currents[0, 0]):
                commands, quantized, report = exports.prepare_field_waveform(model, currents, edges,
                    total_current_limit_A=20., field_limit_Hz=100.)
                np.testing.assert_array_equal(commands, currents)
                self.assertEqual(report['uniform_contraction'], 1.)
                self.assertTrue(report['solver_optimum_preserved'])
                self.assertTrue(report['nominal_DAC']['all_limits_met'])

    def test_disabled_zero_channels_do_not_contract_healthy_channels(self):
        model, edges = contiguous_model()
        model.max_current_A[0] = 0.
        currents = np.array([[0., 0., 0.], [.02, -.03, .01]])
        commands, quantized, report = exports.prepare_field_waveform(model, currents, edges,
            total_current_limit_A=20., field_limit_Hz=100.)
        np.testing.assert_array_equal(commands, currents)
        np.testing.assert_array_equal(quantized[0], 0.)
        self.assertEqual(report['uniform_contraction'], 1.)
        self.assertTrue(report['nominal_DAC']['checks']['disabled_channels_zero'])

    def test_all_disabled_zero_waveform_passes_without_zero_over_zero_contraction(self):
        model, edges = contiguous_model()
        model.max_current_A[:] = 0.
        commands, quantized, report = exports.prepare_field_waveform(model, np.zeros((2, 3)), edges,
            total_current_limit_A=20., field_limit_Hz=100.)
        np.testing.assert_array_equal(commands, 0.)
        np.testing.assert_array_equal(quantized, 0.)
        self.assertEqual(report['uniform_contraction'], 1.)
        self.assertTrue(report['nominal_DAC']['all_limits_met'])

    def test_nonzero_disabled_command_requires_refit_even_below_current_tolerance(self):
        model, edges = contiguous_model()
        model.max_current_A[0] = 0.
        currents = np.array([[1e-15, 0., 0.], [.02, -.03, .01]])
        diagnostic = exports.waveform_hardware_diagnostics(model, currents, edges,
            total_current_limit_A=20., field_limit_Hz=100.)
        self.assertFalse(diagnostic['checks']['disabled_channels_zero'])
        with self.assertRaisesRegex(ValueError, 'refit'):
            exports.prepare_field_waveform(model, currents, edges,
                total_current_limit_A=20., field_limit_Hz=100.)

    def test_nominal_DAC_recheck_catches_loss_of_channel_field_cancellation(self):
        model, edges = contiguous_model(tau=(60e-6, 60e-6))
        model.B_quality = model.tensor([[1000., -1000.]])
        original = np.repeat(np.array([[.1001], [.1]]), 3, axis=1)
        commands, quantized, report = exports.prepare_field_waveform(model, original, edges,
            total_current_limit_A=20., field_limit_Hz=.1)
        # Individual DAC truncation can increase a cancellation residual even
        # though each channel current decreases. A second feasibility check
        # must therefore contract the entire waveform again.
        self.assertGreater(report['quantization_rechecks'], 0)
        self.assertTrue(report['commands']['all_limits_met'])
        self.assertTrue(report['nominal_DAC']['all_limits_met'])
        np.testing.assert_allclose(commands / original, report['uniform_contraction'], rtol=1e-13)
        self.assertLessEqual(np.max(np.abs(model.numpy(model.B_quality) @ quantized)), .1)

    def test_fine_raster_checks_do_not_construct_time_by_time_response_matrices(self):
        model, _ = contiguous_model(cycle=.08)
        edges = 20e-6 + np.arange(5001) * 10e-6
        phase = np.arange(5000) * .013
        currents = np.stack((.1*np.sin(phase), -.08*np.cos(phase)))
        with (patch('acdc_gpu.response_matrices', side_effect=AssertionError('No dense time response')),
              patch('acdc_hardware.response_matrices', side_effect=AssertionError('No dense hardware model'))):
            result = exports.waveform_hardware_diagnostics(model, currents, edges,
                total_current_limit_A=20., field_limit_Hz=100.)
        self.assertTrue(result['all_limits_met'])
        self.assertFalse(result['response_time_matrix_allocated'])
        self.assertAlmostEqual(result['minimum_command_interval_s'], 10e-6)

    def test_invalid_currents_timing_and_complex_inputs_are_rejected(self):
        model, edges = contiguous_model()
        for currents, times in ((np.ones((1, 3)), edges), (np.full((2, 3), np.nan), edges),
                                (np.ones((2, 3))*1j, edges), (np.ones((2, 3)), edges*1j),
                                (np.ones((2, 3)), edges[::-1]),
                                (np.ones((2, 3)), np.array([0., 4e-6, 80e-6, 160e-6]))):
            with self.subTest(currents_shape=currents.shape, edges=times), self.assertRaises(ValueError):
                exports.waveform_hardware_diagnostics(model, currents, times,
                    total_current_limit_A=20., field_limit_Hz=100.)
        with self.assertRaisesRegex(ValueError, 'real'):
            exports.prepare_field_waveform(model, np.ones((2, 3))*1j, edges,
                total_current_limit_A=20., field_limit_Hz=100.)


class FieldShimCSVExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='field_shim_export_test_')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.model, self.edges = contiguous_model(tau=(60e-6, 60e-6))
        self.settings = dict(channel_map=[5, 2], channel_polarity=[-1, 1],
            max_current_A=3., total_current_limit_A=20., min_dac_interval_s=8e-6)
        self.fits = {
            'static': dict(currents_A=np.array([[.4], [-.25]]),
                           state_edges_rf_s=self.edges[[0, -1]], report={'solver': 'fixture static'}),
            'dynamic': dict(currents_A=np.array([[.2, -.3, .1], [-.1, .15, -.05]]),
                            state_edges_rf_s=self.edges, report={'solver': 'fixture dynamic'}),
            'report': {'upstream_commit': 'explicit software fixture', 'dynamic_interpretation': 'constant B0'},
        }
        self.trigger = dict(trigger_rf_s=float(self.edges[0]), channel='ext1',
                            verified_first_imaging_trigger=True)
        self.source = {'reference': 'a'*64, 'b0': 'b'*64, 'calibration': 'c'*64}
        self.solver_patch = patch('field_shim_waveforms.solve_field_shim_waveforms', return_value=self.fits)
        self.solver = self.solver_patch.start(); self.addCleanup(self.solver_patch.stop)
        self.trigger_patch = patch.object(exports, 'sequence_trigger_timing', return_value=self.trigger)
        self.trigger_parser = self.trigger_patch.start(); self.addCleanup(self.trigger_patch.stop)

    def export(self):
        return exports.export_field_shim_waveforms(self.model, self.folder / 'waveform', self.settings,
            total_current_limit_A=20., field_limit_Hz=100., timing={'fixture': True},
            sequence_path=self.folder / 'fixture.seq', provenance=self.source)

    def test_named_CSVs_roundtrip_mapping_polarity_DAC_and_all_sidecar_provenance(self):
        result = self.export()
        self.assertEqual(set(result['exports']), {'static', 'dynamic'})
        self.assertFalse(hasattr(self.model, 'forward'))
        for kind, record in result['exports'].items():
            with self.subTest(kind=kind):
                path = Path(record['csv_file'])
                self.assertEqual(path.name, f'waveform_{kind}.csv')
                table = np.loadtxt(path, delimiter=',', ndmin=2)
                expected = self.fits[kind]['currents_A']
                self.assertEqual(table.shape, (expected.shape[1]+1, 9))
                np.testing.assert_array_equal(table[:-1, 5], -expected[0])
                np.testing.assert_array_equal(table[:-1, 2], expected[1])
                np.testing.assert_array_equal(table[-1, 1:], 0.)
                unused = [i for i in range(1, 9) if i not in (2, 5)]
                np.testing.assert_array_equal(table[:, unused], 0.)
                np.testing.assert_allclose(table[:, 0], self.fits[kind]['state_edges_rf_s']-self.edges[0],
                                           rtol=0, atol=1e-18)
                metadata = json.loads(Path(record['metadata_file']).read_text())
                self.assertEqual(metadata['method'], kind)
                self.assertEqual(metadata['source_sha256'], self.source)
                self.assertTrue(metadata['csv_DAC_roundtrip_verified'])
                self.assertFalse(metadata['image_optimization_required'])
                self.assertTrue(metadata['no_hardware_action_performed'])
                self.assertEqual(metadata['CSV_timing']['origin'], 'first_imaging_trigger')
                self.assertEqual(metadata['waveform_export']['csv_file'], str(path))
                actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(metadata['csv_sha256'], actual_sha)
                self.assertEqual(record['csv_sha256'], actual_sha)
                self.assertTrue(metadata['hardware']['commands']['all_limits_met'])
                self.assertTrue(metadata['hardware']['nominal_DAC']['all_limits_met'])
                with np.load(record['archive_file'], allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive['currents_A'], expected)
                    np.testing.assert_array_equal(archive['solver_currents_A'], expected)
                    np.testing.assert_array_equal(archive['nominal_DAC_currents_A'], quantize_currents(expected))
                    np.testing.assert_array_equal(archive['state_edges_rf_s'], self.fits[kind]['state_edges_rf_s'])
        self.solver.assert_called_once()
        self.trigger_parser.assert_called_once()

    def test_repeated_field_exports_preserve_CSVs_and_sidecars_independent_of_image_optimization(self):
        first = self.export()
        old = {path: path.read_bytes() for path in (self.folder / 'waveform').iterdir()}
        second = self.export()
        self.assertEqual(self.solver.call_count, 2)
        self.assertEqual(len(list((self.folder / 'waveform').iterdir())), 12)
        for kind in ('static', 'dynamic'):
            self.assertEqual(Path(first['exports'][kind]['csv_file']).name, f'waveform_{kind}.csv')
            self.assertEqual(Path(second['exports'][kind]['csv_file']).name, f'waveform_{kind}2.csv')
        for path, content in old.items(): self.assertEqual(path.read_bytes(), content)

    def test_disabled_channels_remain_exactly_zero_in_static_and_dynamic_CSVs(self):
        for all_disabled in (False, True):
            with self.subTest(all_disabled=all_disabled):
                self.model.max_current_A[0] = 0.
                if all_disabled:
                    self.model.max_current_A[:] = 0.
                for fit in self.fits.values():
                    if 'currents_A' in fit:
                        fit['currents_A'][0] = 0.
                        if all_disabled:
                            fit['currents_A'][:] = 0.
                result = self.export()
                for kind, record in result['exports'].items():
                    table = np.loadtxt(record['csv_file'], delimiter=',', ndmin=2)
                    np.testing.assert_array_equal(table[:, 5], 0.)
                    if all_disabled:
                        np.testing.assert_array_equal(table[:, 1:], 0.)
                    else:
                        np.testing.assert_array_equal(table[:-1, 2], self.fits[kind]['currents_A'][1])
                    self.assertEqual(record['uniform_contraction'], 1.)
                    with np.load(record['archive_file'], allow_pickle=False) as archive:
                        np.testing.assert_array_equal(archive['nominal_DAC_currents_A'][0], 0.)

    @unittest.skipUnless(LINE_TRIGGER_SEQUENCE.is_file(), 'Actual acquisition_024 line-trigger sequence is unavailable')
    def test_real_per_line_trigger_sequence_exports_both_field_shims_and_verified_timing(self):
        # Exercise the actual Pulseq parser and trigger verification, retaining
        # only the tiny CPU field model and explicit solver-current fixtures.
        self.trigger_patch.stop()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            _, timing = read_sequence(LINE_TRIGGER_SEQUENCE)
        starts = np.asarray(timing['state_start_s'])
        ends = np.asarray(timing['state_end_s'])
        self.assertEqual(timing['navigator_count'], 3)
        self.assertEqual(starts.size, 64)
        self.model.nstates = starts.size
        self.model.line_start_s, self.model.line_end_s = starts, ends
        self.model.hardware = HardwareConstraints(self.model,
            {'cycle_duration_s': timing['sequence_duration_s']-timing['rf_center_s']})
        edges = np.r_[starts, ends[-1]]
        phase = np.arange(starts.size)*.13
        self.fits['static']['state_edges_rf_s'] = edges[[0, -1]]
        self.fits['dynamic'].update(state_edges_rf_s=edges,
            currents_A=np.stack((.2*np.sin(phase), -.1*np.cos(phase))))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            result = exports.export_field_shim_waveforms(self.model, self.folder/'waveform', self.settings,
                total_current_limit_A=20., field_limit_Hz=100., timing=timing,
                sequence_path=LINE_TRIGGER_SEQUENCE, provenance=self.source)
        for kind, record in result['exports'].items():
            with self.subTest(kind=kind):
                metadata = json.loads(Path(record['metadata_file']).read_text())
                trigger = metadata['trigger']
                self.assertEqual(trigger['mode'], 'per_line')
                self.assertEqual(trigger['trigger_count'], 64)
                self.assertTrue(trigger['verified_first_imaging_trigger'])
                self.assertTrue(trigger['verified_per_line_triggers'])
                self.assertEqual(trigger['trigger_state_indices'], list(range(64)))
                self.assertEqual(len(trigger['output_events']), 64)
                np.testing.assert_allclose(trigger['trigger_times_rf_s'], starts, rtol=0, atol=1e-12)
                np.testing.assert_allclose(trigger['trigger_times_sequence_s'],
                    starts+timing['rf_center_s'], rtol=0, atol=1e-12)
                self.assertAlmostEqual(trigger['trigger_rf_s'], .00421)
                self.assertAlmostEqual(trigger['trigger_sequence_s'], .00531)
                self.assertLessEqual(trigger['maximum_trigger_alignment_error_s'], 1e-12)
                self.assertFalse(trigger['hardware_latency_measured'])
                table = np.loadtxt(record['csv_file'], delimiter=',', ndmin=2)
                np.testing.assert_allclose(table[:, 0],
                    self.fits[kind]['state_edges_rf_s']-starts[0], rtol=0, atol=1e-12)
                self.assertEqual(table[0, 0], 0.)
                self.assertTrue(metadata['csv_DAC_roundtrip_verified'])
                self.assertTrue(metadata['hardware']['nominal_DAC']['all_limits_met'])
        self.solver.assert_called_once()
        self.trigger_parser.assert_not_called()

    def test_existing_sidecar_is_preserved_and_new_CSV_is_removed_on_collision(self):
        output = self.folder / 'waveform'
        output.mkdir()
        existing = output / 'waveform_static.json'
        existing.write_text('{"previous": "preserve"}\n')
        with self.assertRaisesRegex(FileExistsError, 'sidecars'):
            self.export()
        self.assertEqual(existing.read_text(), '{"previous": "preserve"}\n')
        self.assertEqual(list(output.iterdir()), [existing])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
