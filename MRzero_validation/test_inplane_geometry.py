"""Independent Fourier and registered-coil checks for rotations and mirrors."""

from copy import deepcopy
import unittest

import numpy as np
import torch

from continuous_reconstruction import ContinuousReconstruction
from inplane_geometry import (FrameReconstruction, frame_to_patient_xyz,
                              inplane_matrix, rotate_acquisition_k,
                              rotate_points, rotation_matrix_2d)


def acquisition_fixture():
    row, column = np.indices((3, 4))
    # Deliberately off-grid and nonseparable: retaining the real coordinates
    # matters, and an FFT or rounded Cartesian oracle would be insufficient.
    return dict(kx_mInv=(column - 2) / .12 + .337 + .001 * row * column,
                ky_mInv=(row - 1) / .09 - .123 + .0007 * np.sin(row + column),
                time_s=.001 + (4 * row + column) * .0001,
                rf_time_s=np.zeros((3, 4)), phase_offset_rad=.03 * (row + column),
                FOV_m=np.asarray([.12, .09]),
                metadata={'label': 'unchanged'}, line_index=row + 1)


def direct_reconstruction(kdata, acquisition, X_m, Y_m):
    kx, ky = acquisition['kx_mInv'], acquisition['ky_mInv']
    kernel = np.exp(2j * np.pi * (kx.reshape(-1, 1) * X_m.reshape(1, -1)
                                + ky.reshape(-1, 1) * Y_m.reshape(1, -1))) / kx.size
    values = np.asarray(kdata)
    return (values.reshape(values.shape[:-2] + (-1,)) @ kernel).reshape(values.shape)


class InplaneGeometryTests(unittest.TestCase):
    def test_named_mirrors_precede_counterclockwise_rotation(self):
        np.testing.assert_allclose(rotation_matrix_2d(90), [[0., -1.], [1., 0.]], atol=0)
        Q = inplane_matrix(90, mirror_x=True)
        X, Y = rotate_points(np.array([2.]), np.array([3.]), matrix=Q)
        np.testing.assert_allclose([X[0], Y[0]], [-3., -2.], atol=0)
        for mx, my in ((False, False), (True, False), (False, True), (True, True)):
            matrix = inplane_matrix(37, mx, my)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(2), atol=5e-16)
            self.assertAlmostEqual(np.linalg.det(matrix), -1 if mx != my else 1)

    def test_joint_k_and_point_transforms_preserve_every_encoding_inner_product(self):
        acquisition = acquisition_fixture()
        rng = np.random.default_rng(9)
        X, Y = rng.normal(size=7), rng.normal(size=7)
        original = (acquisition['kx_mInv'][..., None] * X
                    + acquisition['ky_mInv'][..., None] * Y)
        for angle in (17, 37, 90, 180):
            for mirror in (False, True):
                matrix = inplane_matrix(angle, mirror_x=mirror)
                transformed = rotate_acquisition_k(acquisition, matrix=matrix)
                rx, ry = rotate_points(X, Y, matrix=matrix)
                actual = transformed['kx_mInv'][..., None] * rx + transformed['ky_mInv'][..., None] * ry
                np.testing.assert_allclose(actual, original, rtol=2e-14, atol=2e-14)

    def test_acquisition_transform_copies_without_reordering_times_or_phases(self):
        original = acquisition_fixture()
        saved = deepcopy(original)
        transformed = rotate_acquisition_k(original, 37, mirror_y=True)
        for key in ('time_s', 'rf_time_s', 'phase_offset_rad', 'line_index'):
            np.testing.assert_array_equal(transformed[key], saved[key])
        for key in ('kx_mInv', 'ky_mInv', 'time_s', 'phase_offset_rad'):
            np.testing.assert_array_equal(original[key], saved[key])
        transformed['time_s'][0, 0] = 42.
        transformed['metadata']['label'] = 'edited copy'
        self.assertEqual(original['metadata']['label'], 'unchanged')
        self.assertNotEqual(original['time_s'][0, 0], 42.)

    def test_nonzero_center_is_an_explicit_physical_alignment_operation(self):
        X, Y = rotate_points(np.array([.04]), np.array([.03]), 90, center_m=(.01, .02))
        np.testing.assert_allclose([X[0], Y[0]], [0., .05], atol=1e-17)

    def test_inverse_passive_frame_preserves_alignment_and_absolute_patient_z(self):
        X, Y = np.meshgrid(np.array([-.04, -.01, .03]), np.array([-.02, .025]))
        Z = np.array([[.020, .021, .022], [.023, .024, .025]])
        Q, P = inplane_matrix(17, mirror_y=True), inplane_matrix(37, mirror_x=True)
        physical_X, physical_Y = rotate_points(X, Y, matrix=Q)
        model_X, model_Y = rotate_points(physical_X, physical_Y, matrix=P)
        patient = frame_to_patient_xyz(model_X, model_Y, Z, frame_matrix=P)
        np.testing.assert_allclose(patient, np.stack((-physical_Y, physical_X, Z), axis=-1),
                                   atol=2e-17, rtol=2e-15)
        # Undoing Q as well would lose the physical alignment: it is separate
        # from the passive model-frame change.
        unaligned = np.stack((-Y, X, Z), axis=-1)
        self.assertGreater(np.linalg.norm(patient - unaligned), .02)

    def test_frame_alone_preserves_registered_basis_and_fitted_physical_currents(self):
        X, Y = np.meshgrid(np.linspace(-.04, .03, 4), np.linspace(-.03, .05, 3))
        target = 2. + 80 * X - 50 * Y

        def fitted_currents(alignment, passive):
            physical_X, physical_Y = rotate_points(X, Y, matrix=alignment)
            model_X, model_Y = rotate_points(physical_X, physical_Y, matrix=passive)
            patient = frame_to_patient_xyz(model_X, model_Y, .02, frame_matrix=passive)
            basis = np.column_stack((np.ones(X.size), 100 * patient[..., 0].ravel(),
                                     100 * patient[..., 1].ravel()))
            return np.linalg.lstsq(basis, -target.ravel(), rcond=None)[0]

        identity = np.eye(2)
        expected = fitted_currents(identity, identity)
        for angle in (17, 37, 90, 180):
            actual = fitted_currents(identity, inplane_matrix(angle, mirror_x=True))
            np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
        changed = fitted_currents(inplane_matrix(37), identity)
        self.assertGreater(np.linalg.norm(changed - expected), .1)

    def test_invalid_geometry_is_rejected(self):
        for bad in (True, np.nan, np.inf, 1j, [10, 20]):
            with self.subTest(angle=bad), self.assertRaises(ValueError):
                rotation_matrix_2d(bad)
        with self.assertRaises(ValueError):
            inplane_matrix(17, mirror_x=1)
        for matrix in (np.eye(3), [[2, 0], [0, 1]], [[1, 1], [0, 1]], [[np.nan, 0], [0, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                rotate_points([0.], [0.], matrix=matrix)
        with self.assertRaises(ValueError):
            frame_to_patient_xyz(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(5))


class FrameReconstructionTests(unittest.TestCase):
    def test_arbitrary_and_mirrored_frames_match_direct_fourier_on_rotated_output_grid(self):
        acquisition = acquisition_fixture()
        rng = np.random.default_rng(10)
        data = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
        native = ContinuousReconstruction(acquisition, 'cpu', torch.float64, sample_chunk=3)
        native_image = native.numpy_image(data)
        for angle in (0, 17, 37, 90, 180):
            for mirror in (False, True):
                with self.subTest(angle=angle, mirror=mirror):
                    matrix = inplane_matrix(angle, mirror_y=mirror)
                    framed_acquisition = rotate_acquisition_k(acquisition, matrix=matrix)
                    operator = FrameReconstruction(framed_acquisition, 'cpu', torch.float64,
                                                   sample_chunk=3, frame_matrix=matrix)
                    actual = operator.numpy_image(data)
                    expected = direct_reconstruction(data, framed_acquisition, operator.X_m, operator.Y_m)
                    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=3e-14)
                    np.testing.assert_allclose(actual, native_image, atol=2e-15, rtol=3e-14)
                    self.assertEqual(operator.report['rounded_or_discarded_samples'], 0)
                    self.assertAlmostEqual(operator.report['frame_determinant'], -1 if mirror else 1)

    def test_adjoint_is_hermitian_for_rotation_and_reflection(self):
        acquisition = acquisition_fixture()
        rng = np.random.default_rng(11)
        data = torch.as_tensor(rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4)))
        target = torch.as_tensor(rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4)))
        for matrix in (inplane_matrix(37), inplane_matrix(17, mirror_x=True), inplane_matrix(90, mirror_y=True)):
            operator = FrameReconstruction(rotate_acquisition_k(acquisition, matrix=matrix),
                                           'cpu', torch.float64, frame_matrix=matrix, sample_chunk=3)
            left = torch.vdot(operator.image(data).ravel(), target.ravel())
            right = torch.vdot(data.ravel(), operator.adjoint(target).ravel())
            self.assertLess(float(abs(left - right)), 2e-15)

    def test_angular_constructor_and_explicit_matrix_have_identical_output(self):
        acquisition = rotate_acquisition_k(acquisition_fixture(), 37)
        angular = FrameReconstruction(acquisition, 'cpu', torch.float64, frame_angle_deg=37)
        matrix = FrameReconstruction(acquisition, 'cpu', torch.float64, frame_matrix=inplane_matrix(37))
        data = np.arange(12).reshape(3, 4).astype(complex)
        np.testing.assert_array_equal(angular.numpy_image(data), matrix.numpy_image(data))
        np.testing.assert_array_equal(angular.X_m, matrix.X_m)
        np.testing.assert_array_equal(angular.Y_m, matrix.Y_m)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
