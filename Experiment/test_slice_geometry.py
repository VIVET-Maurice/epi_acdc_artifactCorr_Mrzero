"""Patient geometry uses measured rotations, independent of image amplitudes."""

import unittest
import numpy as np

from slice_geometry import encoding_origin, normalized_quaternion, slice_basis


class SliceGeometryTests(unittest.TestCase):
    def test_axial_float32_rounding_and_quaternion_sign(self):
        expected = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_array_equal(slice_basis({}), expected)
        np.testing.assert_array_equal(encoding_origin({}), np.zeros(2))
        q = np.asarray([1., 0., 0., 1.], dtype=np.float32) / np.sqrt(np.float32(2.))
        for sign in [-1, 1]:
            np.testing.assert_allclose(slice_basis({"slice_quaternion_wxyz": sign*q}), expected,
                                       rtol=0, atol=1e-15)

    def test_oblique_basis_and_origin_follow_independent_rotation(self):
        angle = np.deg2rad(.435)
        q = np.asarray([np.cos(angle/2), np.sin(angle/2), -np.sin(angle/2), np.cos(angle/2)]) / np.sqrt(2)
        expected = np.asarray([[0., -1., 0.], [np.cos(angle), 0., -np.sin(angle)],
                               [np.sin(angle), 0., np.cos(angle)]])
        center = np.asarray([12., -7., 40.])
        metadata = {"slice_quaternion_wxyz": q, "slice_position_mm": center}
        np.testing.assert_allclose(slice_basis(metadata), expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(encoding_origin(metadata), expected[:, :2].T@center*.001,
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(expected.T @ expected, np.eye(3), rtol=0, atol=1e-15)

    def test_malformed_quaternion_and_declared_origin_are_rejected(self):
        for value in ([1, 0, 0], [1, 0, 0, 1], [np.nan, 0, 0, 0], np.zeros(4), [[1, 0, 0, 0]]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "quaternion"):
                normalized_quaternion(value)
        for declared in ([0, 0, 0], [np.nan, 0], [.1, 0], [1j, 0]):
            with self.subTest(declared=declared), self.assertRaisesRegex(ValueError, "origin"):
                encoding_origin({"slice_position_mm": [0, 0, 0], "encoding_origin_m": declared})
        for center in ([0, 0], [np.nan, 0, 0], [1j, 0, 0]):
            with self.subTest(center=center), self.assertRaisesRegex(ValueError, "position"):
                encoding_origin({"slice_position_mm": center})


if __name__ == "__main__":
    unittest.main()
