"""Measured-GRE target geometry, fixed receiver scale and native gradients."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from measured_image_target import (configure_measured_gre_target,
                                   magnitude_objective_terms, relative_image_mse)
from inplane_geometry import inplane_matrix
from test_experiment_echo import cpu_model
from test_experiment_subspins import expanded_fixture
from test_experiment_voxel import model as voxel_model, voxel_fixture


def grid(ny, nx=None, fov=.2):
    nx = ny if nx is None else nx
    return np.meshgrid((np.arange(nx)-nx//2)*fov/nx,
                       (np.arange(ny)-ny//2)*fov/ny)


def fake_model(shape, frame=None):
    X, Y = grid(*shape)
    matrix = np.eye(2) if frame is None else frame
    operator = SimpleNamespace(X_m=matrix[0, 0]*X+matrix[0, 1]*Y,
                               Y_m=matrix[1, 0]*X+matrix[1, 1]*Y,
                               frame_matrix=matrix)
    return SimpleNamespace(reconstruction=operator, device=torch.device('cpu'),
        dtype=torch.float64, complex_dtype=torch.complex128,
        ideal_image=np.ones(shape, complex), runtime_info={})


def native_reference(X, Y, image=None):
    return dict(X_m=X, Y_m=Y, magnitude=2+3*X-1.7*Y if image is None else image)


class MeasuredGREGeometryTests(unittest.TestCase):
    def test_64_to_128_physical_interpolation_without_extrapolation(self):
        model = fake_model((128, 128))
        X, Y = grid(64)
        target = configure_measured_gre_target(model, native_reference(X, Y),
            baseline_image=np.ones((128, 128)), gain=1)
        expectedX, expectedY = grid(128)
        covered = (expectedX <= X.max()) & (expectedY <= Y.max())
        np.testing.assert_array_equal(target['mask'], covered)
        np.testing.assert_allclose(target['reference_image'][covered],
                                   (2+3*expectedX-1.7*expectedY)[covered], atol=1e-14)
        self.assertEqual(target['report']['comparison_pixels'], 127**2)
        self.assertFalse(target['report']['added_measured_spatial_detail'])
        self.assertFalse(target['report']['registration_fitted'])

    def test_128_reference_on_64_epi_rotation_and_single_mirror(self):
        frame = inplane_matrix(37., mirror_x=True)
        model = fake_model((64, 64), frame)
        X, Y = grid(128)
        target = configure_measured_gre_target(model, native_reference(X, Y),
            baseline_image=np.ones((64, 64)), gain=1)
        targetX, targetY = grid(64)
        self.assertTrue(target['mask'].all())
        np.testing.assert_allclose(target['reference_image'], 2+3*targetX-1.7*targetY, atol=2e-14)
        np.testing.assert_allclose(target['native_X_m'], targetX, atol=2e-16)
        self.assertEqual(np.linalg.det(frame), -1.)

    def test_fixed_two_stage_gain_matches_closed_loop_convention(self):
        model = fake_model((4, 4))
        X, Y = grid(4)
        gre = 2+np.arange(16).reshape(4, 4)*.13
        baseline = np.linspace(.4, 2, 16).reshape(4, 4)
        measured_epi = np.flipud(baseline)*2+.15
        roi = np.ones((4, 4), bool); roi[0, 0] = False
        target = configure_measured_gre_target(model, native_reference(X, Y, gre),
            baseline_image=baseline, measured_baseline_image=measured_epi, roi_mask=roi)
        first = np.sum(baseline[roi]*measured_epi[roi])/np.sum(baseline[roi]**2)
        second = np.sum(measured_epi[roi]*gre[roi])/np.sum(measured_epi[roi]**2)
        self.assertAlmostEqual(target['gain'], first*second)
        candidate = baseline.copy(); candidate[1, 2] *= 1.8
        expected = np.sum((first*second*candidate[roi]-gre[roi])**2)/np.sum(gre[roi]**2)
        self.assertAlmostEqual(relative_image_mse(model, candidate), expected)
        original_report = deepcopy(target['report'])
        for scale in (.2, 1.3, 3):
            relative_image_mse(model, scale*candidate)
        self.assertEqual(target['report'], original_report)
        np.testing.assert_array_equal(target['mask'], roi)
        candidate[0, 0] = 1e9
        self.assertAlmostEqual(relative_image_mse(model, candidate), expected)

    def test_measured_pattern_drives_objective_instead_of_zero_b0_image(self):
        model = fake_model((4, 4))
        ideal = model.ideal_image.copy()
        X, Y = grid(4)
        measured = np.eye(4)*3+.2
        target = configure_measured_gre_target(model, native_reference(X, Y, measured),
            baseline_image=np.ones((4, 4)), gain=1)
        self.assertEqual(relative_image_mse(model, measured), 0)
        self.assertGreater(relative_image_mse(model, ideal), .5)
        np.testing.assert_array_equal(model.ideal_image, ideal)
        np.testing.assert_array_equal(model.reference, measured)
        self.assertFalse(target['report']['object_parameters_modified'])

    def test_default_full_coverage_retains_dark_pixels(self):
        model = fake_model((4, 4))
        X, Y = grid(4)
        measured = np.zeros((4, 4)); measured[1:3, 1:3] = 2
        target = configure_measured_gre_target(model, native_reference(X, Y, measured),
            baseline_image=np.ones((4, 4)), gain=1)
        self.assertTrue(target['mask'].all())
        clean = measured.copy(); ghost = clean.copy(); ghost[0, 0] = 1
        self.assertGreater(relative_image_mse(model, ghost), relative_image_mse(model, clean))

    def test_missing_measurements_never_become_target_zeros(self):
        model = fake_model((4, 4))
        X, Y = grid(4)
        reference = native_reference(X, Y)
        reference['valid_mask'] = np.ones((4, 4), bool)
        reference['valid_mask'][1, 2] = False
        reference['magnitude'][1, 2] = np.nan
        target = configure_measured_gre_target(model, reference,
            baseline_image=np.ones((4, 4)), gain=1)
        self.assertFalse(target['mask'][1, 2])
        self.assertTrue(np.isfinite(target['reference_image']).all())
        first = np.ones((4, 4)); second = first.copy(); second[1, 2] = 1e6
        self.assertEqual(relative_image_mse(model, first), relative_image_mse(model, second))

    def test_invalid_reference_geometry_or_gain_is_rejected(self):
        model = fake_model((4, 4))
        X, Y = grid(4)
        for options in ({'gain': 0}, {'gain': np.nan}, {'threshold_fraction': 1},
                        {'frame_matrix': np.ones((2, 2))}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                configure_measured_gre_target(model, native_reference(X, Y),
                    baseline_image=np.ones((4, 4)), **options)

    def test_weighted_complex_image_derivative_matches_directional_difference(self):
        model = fake_model((4, 4))
        X, Y = grid(4)
        mask = np.indices((4, 4)).sum(axis=0) % 3 != 0
        configure_measured_gre_target(model, native_reference(X, Y),
            baseline_image=np.ones((4, 4)), roi_mask=mask, gain=1)
        rng = np.random.default_rng(1002)
        image = torch.tensor(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
        direction = torch.tensor(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
        _, derivative = magnitude_objective_terms(model, image)
        epsilon = 1e-6
        difference = (relative_image_mse(model, image+epsilon*direction)
                      - relative_image_mse(model, image-epsilon*direction))/(2*epsilon)
        self.assertAlmostEqual(float((derivative.conj()*direction).real.sum()), difference, places=8)


class NativeMeasuredGREGradientTests(unittest.TestCase):
    def _native_gradient(self, model):
        zero = np.zeros((model.nc, model.nstates))
        baseline = model.forward(zero)[0]
        original_ideal = model.ideal_image.copy()
        X, Y = model.reconstruction.X_m, model.reconstruction.Y_m
        gre = np.arange(baseline.size).reshape(baseline.shape)*.07+.2
        mask = np.ones(gre.shape, bool); mask[0, 0] = False
        configure_measured_gre_target(model, native_reference(X, Y, gre),
            baseline_image=baseline, measured_baseline_image=np.abs(baseline)*1.4+.1,
            roi_mask=mask)
        rng = np.random.default_rng(1003)
        currents = rng.normal(size=zero.shape)*.035
        direction = rng.normal(size=zero.shape); direction /= np.linalg.norm(direction)
        value, gradient = model.objective_and_grad(currents)
        image = model.forward(currents)[0]
        self.assertAlmostEqual(value, relative_image_mse(model, image), places=13)
        epsilon = 1e-5
        difference = (model.objective_and_grad(currents+epsilon*direction)[0]
                      - model.objective_and_grad(currents-epsilon*direction)[0])/(2*epsilon)
        np.testing.assert_allclose(np.sum(gradient*direction), difference, rtol=4e-6, atol=2e-9)
        np.testing.assert_array_equal(model.ideal_image, original_ideal)
        response = model.response_diagnostics(currents, field_limit_Hz=500, total_current_limit_A=30)
        for name in ('instantaneous', 'response'):
            self.assertAlmostEqual(response[name+'_relative_mse'],
                relative_image_mse(model, response[name+'_image']), places=13)

    def test_point_model_current_gradient_and_response_costs_use_measured_gre(self):
        _, _, experiment, calibration = expanded_fixture(children=2, in_plane=True)
        experiment['amplitude_reference_time_s'] = .003
        self._native_gradient(cpu_model(experiment, calibration))

    def test_finite_voxel_current_gradient_and_response_costs_use_measured_gre(self):
        experiment, calibration = voxel_fixture(children=2)
        self._native_gradient(voxel_model(experiment, calibration))


if __name__ == '__main__':
    unittest.main()
