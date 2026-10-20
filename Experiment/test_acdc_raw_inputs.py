"""Exercise raw B0 and NPZ calibration through the existing folder pipeline."""

import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat

import run_acdc_experiment as pipeline
import test_run_acdc_experiment as folder_tests
from test_read_acdc_b0 import coil_images, fake_twix, TE


class RawFolderTests(unittest.TestCase):
    def setUp(self):
        # Reuse the same measured-map layout and hardware fixture as the
        # existing folder tests; only the input formats differ here.
        fixture = folder_tests.FolderPipelineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.input, self.output, self.config = fixture.input, fixture.output, fixture.config
        self.config['files'].update(b0='b0.dat', b0_sequence='b0.seq', calibration='calibration.npz')
        self.config['preparation']['b0_sign'] = -1
        np.savez(self.input / 'calibration.npz', **fixture.calibration)
        (self.input / 'b0.dat').write_bytes(b'Synthetic raw-data reader fixture')
        (self.input / 'b0.seq').write_bytes(b'Synthetic sequence-reader fixture')
        pipeline.write_json(self.input / 'config.json', self.config)
        self.scans = fake_twix(coil_images(np.full((128, 128), 37.)))
        self.scans[0]['hdr']['MeasYaps']['sSliceArray']['asSlice'][0]['dPhaseFOV'] = 200
        self.sequence = dict(image_shape=[128, 128], echo_count=3, polarity=1,
                             TE_s=TE.tolist(), TR_s=.01, dwell_s=10e-6,
                             FOV_m=[.2, .2], acquired_phase_rows=list(range(128)))

    def prepare(self):
        with patch('twixtools.read_twix', return_value=self.scans), \
                patch('read_acdc_b0.read_b0_sequence', return_value=self.sequence):
            return pipeline.prepare(self.input, self.output)

    def test_raw_reconstruction_npz_calibration_sign_and_provenance(self):
        _, report = self.prepare()
        experiment = loadmat(self.output / 'prepared/experiment_input.mat', simplify_cells=True)['experiment']
        support = experiment['field']['simulation_mask'].astype(bool)
        np.testing.assert_allclose(experiment['field']['b0Hz'][support], -37, atol=1e-10)
        with np.load(self.output / 'prepared/b0_reconstructed.npz') as b0:
            np.testing.assert_allclose(b0['b0Hz'], -37, atol=1e-10)
            self.assertEqual(b0['applied_b0_sign'].item(), -1)
        self.assertEqual(report['b0_reconstruction']['echo_count'], 3)
        calibration = loadmat(self.output / 'prepared/calibration.mat', simplify_cells=True)
        self.assertEqual(calibration['acdc_maps'].shape, (128, 128, 32))
        manifest = json.loads((self.output / 'provenance.json').read_text())
        self.assertIn(str((self.input / 'b0.seq').resolve()), manifest['sources'])
        self.assertIn('prepared/b0_reconstructed.npz', manifest['prepared'])
        (self.input / 'b0.seq').write_bytes(b'Changed acquisition sequence')
        with self.assertRaisesRegex(ValueError, 'File changed since preparation'):
            pipeline.check_snapshot(self.output)

    def test_raw_b0_requires_its_multiecho_sequence(self):
        self.config['files'].pop('b0_sequence')
        pipeline.write_json(self.input / 'config.json', self.config)
        with self.assertRaisesRegex(ValueError, 'requires files.b0_sequence'):
            self.prepare()
        self.assertFalse((self.output / 'provenance.json').exists())

    def test_different_b0_slice_cannot_be_interpolated_onto_reference(self):
        # Supply physical reference metadata without changing its measured
        # amplitude or grid. The raw B0 describes a different axial plane.
        with np.load(self.input / 'gre.npz') as data:
            reference = dict(data)
        reference['position_mm'] = np.array([0, 0, 3.])
        np.savez(self.input / 'gre.npz', **reference)
        with self.assertRaisesRegex(ValueError, 'different slice planes'):
            self.prepare()
        self.assertFalse((self.output / 'provenance.json').exists())


if __name__ == '__main__':
    unittest.main()
