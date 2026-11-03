"""Full 3D constrained optimization, disabled channels and timed off events."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

from volume_epi_model import VolumeEPIModel
from volume_optimization import optimize_volume_image, export_volume_waveform
from test_volume_epi_model import fixture


def tiny_model(limits=(.8, 0.)):
    volume, acquisition, _ = fixture((4, 3, 2))
    volume['b0_Hz'] = -.45*volume['fieldmaps_Hz_per_A'][..., 0]
    volume['R2_sInv'] = 0.
    volume['amplitude_reference_time_s'] = 0.
    nx, ny, nz = volume['amplitude'].shape
    fov = volume['FOV_m']
    acquisition['kx_mInv'] = np.broadcast_to((np.arange(nx)-nx//2)[None, :]/fov[0], (ny, nx)).copy()
    acquisition['kz_partition_mInv'] = (np.arange(nz)-nz//2)/fov[2]
    acquisition['phase_offset_rad'][:] = 0.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        model = VolumeEPIModel(volume, acquisition, interpolation_nodes='exact', max_current_A=limits)
    model.set_target(volume['amplitude'], gain=1.)
    return model


def export_fixture():
    currents = np.array([[.1, .2, -.1], [0., 0., 0.]])
    solution = dict(currents_A=currents, channel_max_current_A=np.array([.5, 0.]),
        total_current_limit_A=1., initial_objective=1., best_objective=.7, quantized_objective=.71,
        improved=True, quantized_objective_not_worse_than_initial=True, finite_optimization=True,
        hardware={'all_limits_met': True}, hardware_after_DAC={'all_limits_met': True},
        field_guidance_weight=.05, field_guidance_scale_Hz=100.)
    timing = dict(partition_count=2, repeated_waveform_per_partition=True, verified_per_line_triggers=True,
        waveform_event_time_first_trigger_s=[0., .01, .02, .03, .1, .11, .12, .13],
        waveform_event_state_index=[0, 1, 2, -1, 0, 1, 2, -1], sequence_sha256='test_seq_hash')
    settings = dict(channel_map=[2, 1], channel_polarity=[-1, 1], max_current_A=[.5, 0.],
                    disabled_channels=[1], total_current_limit_A=1., min_dac_interval_s=8e-6)
    metadata = dict(selection={'axis': 'z', 'index': 1}, pose={'center_m': [0., 0., .01]},
                    source_sha256={'reference': 'declared_test_reference_hash'})
    return solution, timing, settings, metadata


class VolumeOptimizationTests(unittest.TestCase):
    def test_sparse_display_snapshots_reuse_the_objective_forward_and_cannot_change_optimization(self):
        options = dict(initialization='zero', max_iterations=5, patience=10,
                       learning_rate_A=.04, total_current_limit_A=1., field_limit_Hz=100.)
        control = optimize_volume_image(tiny_model(), **options)
        model = tiny_model()
        original_forward = model.forward
        evaluations, snapshots = [], []

        def observed_forward(commands):
            image, kdata = original_forward(commands)
            evaluations.append((np.array(model.numpy(commands), copy=True),
                                np.array(model.numpy(image), copy=True)))
            return image, kdata

        def display(iteration, row, commands):
            snapshots.append((iteration, deepcopy(row), commands.copy()))
            # A plotting callback must not own views into optimizer parameters,
            # autograd's image, or the scalar result/history being accumulated.
            commands[:] = 123.
            row['image'][:] = 456.+7.j
            row['history']['image_mse'][0] = -1.
            row['objective'] = -2.

        with patch.object(model, 'forward', side_effect=observed_forward):
            result = optimize_volume_image(model, callback=display, callback_every=2, **options)

        self.assertEqual([iteration for iteration, _, _ in snapshots], [0, 2, 4, 5])
        # Every scalar iteration requires one forward; final DAC validation
        # adds exactly one. Display cadence does not add simulations.
        self.assertEqual(len(evaluations), len(result['history'])+1)
        self.assertEqual(result['history'], control['history'])
        np.testing.assert_array_equal(result['currents_A'], control['currents_A'])
        for iteration, row, commands in snapshots:
            expected_commands, expected_image = evaluations[iteration]
            np.testing.assert_array_equal(commands, expected_commands)
            np.testing.assert_array_equal(row['image'], expected_image)
            expected_loss = float(model._image_loss(torch.as_tensor(row['image'])))
            self.assertAlmostEqual(row['image_mse'], expected_loss, places=14)
            self.assertEqual(row['history']['iteration'], list(range(iteration+1)))
            self.assertEqual(row['history']['image_mse'],
                             [item['image_mse'] for item in result['history'][:iteration+1]])
            self.assertAlmostEqual(row['max_field_Hz'],
                float(np.max(np.abs(model.numpy(model.B_quality) @ commands))), places=13)
        for row in result['history']:
            self.assertTrue(all(value is None or np.isscalar(value) for value in row.values()))
            self.assertNotIn('image', row)
            self.assertNotIn('history', row)

    def test_patience_and_gradient_failure_send_one_terminal_snapshot_with_complete_history(self):
        for reason in ('patience', 'gradient'):
            with self.subTest(reason=reason):
                model = tiny_model(limits=(0., 0.)) if reason == 'patience' else tiny_model()
                original_loss = model._image_loss
                objective_evaluations = 0
                snapshots = []

                def failing_loss(image):
                    nonlocal objective_evaluations
                    value = original_loss(image)
                    if torch.is_grad_enabled():
                        objective_evaluations += 1
                        if reason == 'gradient' and objective_evaluations == 4:
                            value.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
                    return value

                with patch.object(model, '_image_loss', side_effect=failing_loss):
                    result = optimize_volume_image(model, initialization='zero', max_iterations=20,
                        patience=2 if reason == 'patience' else 20, callback_every=10,
                        field_guidance_weight=0., total_current_limit_A=1., field_limit_Hz=100.,
                        callback=lambda iteration, row, commands: snapshots.append((iteration, row, commands)))
                terminal = 2 if reason == 'patience' else 3
                self.assertEqual([item[0] for item in snapshots], [0, terminal])
                self.assertEqual(snapshots[-1][1]['history']['iteration'], list(range(terminal+1)))
                self.assertEqual(snapshots[-1][1]['history']['image_mse'],
                                 [row['image_mse'] for row in result['history']])
                self.assertEqual(len(result['history']), terminal+1)
                self.assertEqual(result['finite_optimization'], reason == 'patience')
                if reason == 'gradient':
                    self.assertEqual(result['stop_reason'], 'nonfinite or missing image gradient')

    def test_real_full_volume_optimizer_improves_combined_cost_with_disabled_channel(self):
        model = tiny_model()
        evaluated = []
        result = optimize_volume_image(model, initialization='zero', max_iterations=50,
            learning_rate_A=.04, patience=30, total_current_limit_A=1., field_limit_Hz=100.,
            callback=lambda iteration, row, commands: evaluated.append(commands.copy()), callback_every=1)
        self.assertTrue(result['improved'])
        self.assertTrue(result['quantized_objective_not_worse_than_initial'])
        self.assertLess(result['best_objective'], result['initial_objective'])
        self.assertTrue(result['hardware']['all_limits_met'])
        self.assertTrue(result['hardware_after_DAC']['all_limits_met'])
        self.assertGreater(np.max(np.abs(result['currents_A'][0])), .01)
        np.testing.assert_array_equal(result['currents_A'][1], np.zeros(model.nstates))
        for commands in evaluated:
            np.testing.assert_array_equal(commands[1], np.zeros(model.nstates))
        self.assertEqual(result['field_guidance']['sequence_kind'], 'epi3d')
        self.assertEqual(result['field_guidance']['quality_centers'], np.prod(model.shape))

    def test_all_disabled_channels_remain_exactly_zero_without_false_improvement(self):
        model = tiny_model(limits=(0., 0.))
        result = optimize_volume_image(model, initialization='static', max_iterations=3, patience=2,
                                       total_current_limit_A=1., field_limit_Hz=100.)
        np.testing.assert_array_equal(result['currents_A'], np.zeros((2, 3)))
        self.assertFalse(result['improved'])
        self.assertTrue(result['hardware_after_DAC']['all_limits_met'])

    def test_timed_export_repeats_lines_and_keeps_every_interpartition_off_event(self):
        solution, timing, settings, metadata = export_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            report = export_volume_waveform(solution, timing, settings, temporary,
                metadata=metadata, time_interpolation_report={'passed': True, 'relative_complex_kdata_error': 0.})
            csv = np.loadtxt(report['csv']['csv_file'], delimiter=',')
            np.testing.assert_allclose(csv[:, 0], timing['waveform_event_time_first_trigger_s'])
            np.testing.assert_allclose(csv[:, 2], [-.1, -.2, .1, 0., -.1, -.2, .1, 0.])
            np.testing.assert_array_equal(csv[:, 1], np.zeros(8))
            np.testing.assert_array_equal(csv[[3, 7], 1:], np.zeros((2, 8)))
            sidecar = json.loads(Path(report['sidecar_json']).read_text())
            self.assertEqual(sidecar['metadata'], metadata)
            self.assertEqual(sidecar['explicit_zero_events'], 2)
            self.assertFalse(sidecar['zero_time_export_supported'])
            self.assertEqual(len(list(Path(temporary).glob('*.csv'))), 1)

    def test_failed_convergence_quantization_hardware_or_zero_time_never_writes_csv(self):
        original, timing, settings, metadata = export_fixture()
        cases = []
        for field in ('improved', 'finite_optimization', 'quantized_objective_not_worse_than_initial'):
            changed = deepcopy(original)
            changed[field] = False
            cases.append((field, changed, {'passed': True}, False))
        changed = deepcopy(original)
        changed['hardware_after_DAC']['all_limits_met'] = False
        cases.extend([('hardware', changed, {'passed': True}, False),
                      ('interpolation', original, {'passed': False}, False),
                      ('zero_time', original, {'passed': True}, True)])
        for name, solution, convergence, zero_time in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    export_volume_waveform(solution, timing, settings, temporary, metadata=metadata,
                        time_interpolation_report=convergence, zero_time_column=zero_time)
                self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
