"""Translated acquisitions preserve encoding while retaining patient geometry."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Experiment'))
from acdc_gpu import ACDCModel
from test_acdc_gpu import fixture
from inplane_geometry import inplane_matrix
from measured_image_target import configure_measured_gre_target
from measured_scan_comparison import compare_registered_images
from measured_continuous import reconstruct_receivers
from test_measured_image_target import fake_model, grid, native_reference
import test_twin_inputs as twin_test_helpers
from twin_inputs import _read_nominal_reference, prepare_echo_object
from slice_geometry import slice_basis, encoding_origin


class OffsetGeometryIntegrationTests(unittest.TestCase):
    def test_absolute_model_coordinates_and_receiver_origin_translate_together(self):
        experiment, calibration = fixture(response=True)
        baseline = ACDCModel(experiment, calibration, profile='cpu', precision='float64')
        shifted, maps = deepcopy(experiment), deepcopy(calibration)
        origin = np.array([-.00598, .013])
        shifted['acquisition']['encoding_origin_m'] = origin
        for key, delta in zip(('X_m', 'Y_m'), origin):
            shifted['geometry'][key] += delta
            maps[key] += delta
        translated = ACDCModel(shifted, maps, profile='cpu', precision='float64')
        commands = np.random.default_rng(91).normal(0, .02, (baseline.nc, baseline.nstates))
        for actual, expected in zip(translated.forward(commands), baseline.forward(commands)):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        actual = translated.objective_and_grad(commands)
        expected = baseline.objective_and_grad(commands)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-11, atol=1e-12)

    def test_local_model_samples_absolute_measured_gre_after_rotation_and_mirror(self):
        matrix = inplane_matrix(37., mirror_y=True)
        model = fake_model((32, 32), matrix)
        origin = np.array([-.0059808583, .014])
        model.reconstruction.native_frame_origin_m = origin
        X, Y = grid(32)
        image = 2 + 3*X - 1.7*Y
        target = configure_measured_gre_target(model,
            native_reference(X+origin[0], Y+origin[1], image),
            baseline_image=np.ones(X.shape), gain=1.)
        self.assertTrue(target['mask'].all())
        np.testing.assert_allclose(target['reference_image'], image, atol=2e-14)
        np.testing.assert_allclose(target['native_X_m'], X+origin[0], atol=2e-16)
        np.testing.assert_allclose(target['native_Y_m'], Y+origin[1], atol=2e-16)

    def test_nominal_raw_reference_keeps_local_fourier_reconstruction(self):
        measured = twin_test_helpers.TwinInputTests().nominal_reference_fixture(32)
        measured[1]['slice_position_mm'] = [12., -7., -17.3]
        with patch('twin_inputs.read_scan', return_value=measured):
            gre, diagnostics, acquisition = _read_nominal_reference(Path('offset.dat'))
        X, Y = grid(32)
        np.testing.assert_allclose(gre['X_m'], X-.007, atol=1e-16)
        np.testing.assert_allclose(gre['Y_m'], Y-.012, atol=1e-16)
        actual, _ = reconstruct_receivers(diagnostics, acquisition)
        axes = (0, 2)
        coils = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(
            measured[2]['kspace'], axes=axes), axes=axes), axes=axes)
        np.testing.assert_allclose(actual, np.sqrt(np.sum(abs(coils)**2, axis=1)), atol=1e-13)

    def test_oblique_nominal_reference_and_rotated_target_share_the_measured_plane(self):
        measured = twin_test_helpers.TwinInputTests().nominal_reference_fixture(32)
        measured[1]['slice_position_mm'] = [12., -7., -17.3]
        measured[1]['slice_quaternion_wxyz'] = [
            .7070966958999634, -.0037749139592051506,
            .00040555562009103596, .7071066498756409]
        basis = slice_basis(measured[1])
        center = np.asarray(measured[1]['slice_position_mm'])
        origin = (center*1e-3) @ basis[:, :2]
        with patch('twin_inputs.read_scan', return_value=measured):
            gre, diagnostics, acquisition = _read_nominal_reference(Path('oblique.dat'))
        X, Y = grid(32)
        np.testing.assert_allclose(gre['X_m'], X+origin[0], atol=1e-16)
        np.testing.assert_allclose(gre['Y_m'], Y+origin[1], atol=1e-16)
        np.testing.assert_allclose(gre['position_mm'],
            center-100*basis[:, 0]-100*basis[:, 1], atol=1e-12)
        np.testing.assert_allclose(gre['orientation'].reshape(2, 3).T, basis[:, :2])
        np.testing.assert_allclose(acquisition['encoding_origin_m'], encoding_origin(measured[1]))
        frame = inplane_matrix(180.)
        model = fake_model((32, 32), frame)
        model.reconstruction.native_frame_origin_m = origin
        image = 2 + 3*X - 1.7*Y
        reference = dict(gre, magnitude=image)
        target = configure_measured_gre_target(model, reference,
            baseline_image=np.ones(X.shape), gain=1.)
        self.assertTrue(target['mask'].all())
        np.testing.assert_allclose(target['reference_image'], image, atol=2e-14)

    def test_noise_corners_follow_fov_center_and_measured_comparison_uses_same_frame(self):
        helper = twin_test_helpers.TwinInputTests()
        inputs = helper.noise_fixture()
        baseline = prepare_echo_object(inputs, allow_uncalibrated_noise=False)
        moved = deepcopy(inputs)
        origin = np.array([-.006, .012])
        moved['b0_raw']['X_m'] += origin[0]
        moved['b0_raw']['Y_m'] += origin[1]
        moved['b0_raw'].setdefault('metadata', {})['slice_position_mm'] = [-12., -6., -17.3]
        translated = prepare_echo_object(moved, allow_uncalibrated_noise=False)
        np.testing.assert_array_equal(translated['noise_mask'], baseline['noise_mask'])
        np.testing.assert_array_equal(translated['object_support'], baseline['object_support'])
        X, Y = grid(8)
        image = np.ones(X.shape)
        record = dict(image=image, X_m=X+origin[0], Y_m=Y+origin[1], encoding_origin_m=origin)
        frame = inplane_matrix(90.)
        comparison = compare_registered_images(record, record, [record], frame_matrix=frame)
        np.testing.assert_allclose(comparison['X_m'], -Y, atol=1e-16)
        np.testing.assert_allclose(comparison['Y_m'], X, atol=1e-16)
        self.assertAlmostEqual(comparison['rows'][0]['RMSE'], 0.)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
