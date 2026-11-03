"""Execute V4's model-to-export cells on a real small 3D problem."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from volume_epi_model import VolumeEPIModel
from volume_optimization import optimize_volume_image, export_volume_waveform
from volume_slice_viewer import VolumeSliceViewer, extract_volume_slice
from volume_inputs import sample_record, sample_patient_plane
from export_acdc_waveform import validate_waveform_settings
from test_volume_epi_model import fixture


NOTEBOOK = Path(__file__).with_name('06_acdc_echo32_b0_s0_t2star_MRzeroV4 (3D).ipynb')


class VolumeNotebookTests(unittest.TestCase):
    def test_executable_notebook_uses_confirmed_roi_and_preserves_full_3d_outputs(self):
        for axis in ('x', 'y', 'z'):
            with self.subTest(axis=axis):
                self.execute_notebook(axis)

    def execute_notebook(self, axis):
        notebook = json.loads(NOTEBOOK.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(NOTEBOOK), 'exec')
        volume, acquisition, _ = fixture()
        nx, ny, nz = volume['amplitude'].shape
        acquisition['kx_mInv'] = np.broadcast_to((np.arange(nx)-nx//2)[None, :]/volume['FOV_m'][0], (ny, nx)).copy()
        acquisition['kz_partition_mInv'] = (np.arange(nz)-nz//2)/volume['FOV_m'][2]
        acquisition['phase_offset_rad'][:] = 0.
        acquisition['line_end_s'][:-1, 0] = acquisition['line_start_s'][1:, 0]
        volume['b0_Hz'] = -.45*volume['fieldmaps_Hz_per_A'][..., 0]
        volume['R2_sInv'] = 0.
        volume['amplitude_reference_time_s'] = 0.
        volume['reference_magnitude'] = volume['amplitude'].copy()
        volume['roi_mask'] = np.zeros((nx, ny, nz), bool)
        plane = [slice(None)]*3
        plane['xyz'.index(axis)] = volume['amplitude'].shape['xyz'.index(axis)]//2
        volume['roi_mask'][tuple(plane)] = True
        volume['support'] = np.ones((nx, ny, nz), bool)
        volume['report'] = {'model': 'Independent small full-volume integration test'}
        axes = tuple((np.arange(n)-n//2)*f/n for n, f in zip((nx, ny, nz), volume['FOV_m']))
        bundle = dict(axes_m=axes, reference_magnitude=volume['amplitude'],
                      epi_magnitude=volume['amplitude'], b0_Hz=volume['b0_Hz'])
        viewer = VolumeSliceViewer(bundle, axis=axis)
        with self.assertRaises(RuntimeError):
            viewer.snapshot()
        selection = viewer.confirm_selection()
        native = dict(axes_m=axes, basis_patient=np.eye(3), center_m=np.zeros(3),
                      epi_magnitude=volume['amplitude'], b0_Hz=volume['b0_Hz'])
        settings = validate_waveform_settings(dict(channel_map=[1, 2], channel_polarity=[-1, -1],
            disabled_channels=[2], max_current_A=.8, total_current_limit_A=1., min_dac_interval_s=8e-6), 2)
        starts = acquisition['line_start_s'].ravel()
        times = np.r_[np.r_[starts, acquisition['line_end_s'][-1, 0]],
                      np.r_[starts+.1, acquisition['line_end_s'][-1, 0]+.1]]-starts[0]
        timing = dict(partition_count=2, repeated_waveform_per_partition=True,
                      verified_per_line_triggers=True,
                      waveform_event_time_first_trigger_s=times,
                      waveform_event_state_index=np.tile(np.r_[np.arange(ny), -1], 2),
                      sequence_sha256='independent_test_sequence')
        with tempfile.TemporaryDirectory() as folder:
            namespace = dict(np=np, torch=torch, plt=plt, Path=Path, json=json,
                viewer=viewer, selection=selection, volume=volume, acquisition=acquisition,
                native_scene=native, volume_bundle=bundle,
                records={name:dict(magnitude=volume['amplitude'], metadata={'source_sha256':name})
                         for name in ['reference', 'epi', 'b0']},
                GPU_REQUIRED=False, MODEL_DTYPE=torch.float64, CURRENT_RESPONSE={'enabled':True, 'tau_s':.00003},
                WAVEFORM_PARAMETERS=settings, INTERPOLATION_NODES=4, INTERPOLATION_MAX_NODES=4,
                INTERPOLATION_TOLERANCE=1e-8, VolumeEPIModel=VolumeEPIModel,
                optimize_volume_image=optimize_volume_image, export_volume_waveform=export_volume_waveform,
                extract_volume_slice=extract_volume_slice, sample_record=sample_record,
                sample_patient_plane=sample_patient_plane,
                RUN_OPTIMIZATION=True, IMAGE_INITIALIZATION='zero', TOTAL_CURRENT_LIMIT_A=1., FIELD_LIMIT_HZ=100.,
                FIELD_GUIDANCE_WEIGHT=.05, FIELD_GUIDANCE_SCALE_HZ=100., MAX_ITERATIONS=50,
                LEARNING_RATE_A=.04, PATIENCE=30, HARDWARE={}, LIVE_UPDATE_EVERY=20, MAX_CURRENT_A=.8,
                OUTPUT_DIR=Path(folder)/'results', WAVEFORM_OUTPUT_DIR=Path(folder)/'waveforms',
                sequence_timing=timing, frequency_adjustment_Hz=0., B0_GLOBAL_OFFSET_HZ=0., EXPORT_WAVEFORM=True)
            def require_same_selection():
                if viewer.snapshot() != selection:
                    raise RuntimeError('Selection changed')
            namespace['require_same_selection'] = require_same_selection
            with redirect_stdout(io.StringIO()):
                for index in [11, 12, 14, 16, 18, 20]:
                    exec(''.join(notebook['cells'][index]['source']), namespace)
            saved = np.load(namespace['OUTPUT_DIR']/'volume_optimization.npz', allow_pickle=False)
            self.assertEqual(saved['optimized_image'].shape, (nx, ny, nz))
            np.testing.assert_array_equal(saved['roi_mask'], volume['roi_mask'])
            np.testing.assert_array_equal(saved['currents_A'][1], 0.)
            self.assertTrue(namespace['solution']['improved'])
            self.assertTrue(namespace['final_accuracy']['passed'])
            # V3-style display uses the actual normalized target and the same
            # physical orientation as the selected plane, with fixed scales.
            history = namespace['solution']['history']
            self.assertEqual(namespace['live'].iterations, [row['iteration'] for row in history])
            np.testing.assert_allclose(namespace['live'].losses, [row['image_mse'] for row in history])
            reference_plane, shown = sample_patient_plane(volume['reference_magnitude'], native, bundle, selection)
            np.testing.assert_allclose(namespace['image_history'].reference, reference_plane)
            archive = np.load(namespace['image_history_npz'], allow_pickle=False)
            expected_iterations = [row['iteration'] for row in history if row['iteration'] % 20 == 0]
            np.testing.assert_array_equal(archive['iterations'], expected_iterations)
            self.assertEqual(archive['images'].shape, (len(expected_iterations),) + reference_plane.shape)
            np.testing.assert_allclose(archive['images'][0], archive['baseline'], atol=1e-7)
            self.assertEqual(archive['origin'].item(), 'lower')
            html = namespace['image_history_html'].read_text()
            self.assertIn('Archived iterations:', html)
            self.assertIn('type="range"', html)
            self.assertNotIn('Itération', html)
            self.assertTrue((namespace['OUTPUT_DIR']/'optimization_progress.png').is_file())
            saved_summary = json.loads((namespace['OUTPUT_DIR']/'volume_optimization.json').read_text())
            self.assertEqual(saved_summary['image_history']['selection'], selection)
            self.assertEqual(namespace['live'].figure.axes[0].images[0].origin, 'lower')
            self.assertEqual(namespace['live'].figure.axes[0].get_xlabel(),
                             f"Patient {shown['orientation']['column_axis'].upper()} [mm]")
            export = namespace['waveform_export']
            self.assertEqual(len(np.loadtxt(export['csv']['csv_file'], delimiter=',')), 2*(ny+1))
            self.assertEqual(json.loads(Path(export['sidecar_json']).read_text())['metadata']['selection'], selection)
            viewer.select(index=0)
            with self.assertRaises(RuntimeError):
                exec(''.join(notebook['cells'][20]['source']), namespace)
            self.assertEqual(len(list(namespace['WAVEFORM_OUTPUT_DIR'].glob('*.csv'))), 1)


if __name__ == '__main__':
    unittest.main()
