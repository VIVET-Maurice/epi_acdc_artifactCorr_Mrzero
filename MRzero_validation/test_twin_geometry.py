"""Landmark/DFT parity and fixed-grid geometry comparison regressions."""

import unittest

import numpy as np

from twin_geometry import (compare_magnitude, fft_origin_reverse,
                           reindex_image, rotate_physical_image)


def grid(ny, nx, fov=(.2, .2)):
    return np.meshgrid((np.arange(nx) - nx // 2) * fov[0] / nx,
                       (np.arange(ny) - ny // 2) * fov[1] / ny)


class TwinGeometryTests(unittest.TestCase):
    def test_physical_landmark_rotation_agrees_at_32_64_128_without_pixel_shift(self):
        legacy_landmarks = []
        for size in [32, 64, 128]:
            X, Y = grid(size, size)
            image = np.zeros(X.shape)
            image[size // 2 + size // 8, size // 2] = 1
            rotated, newX, newY, report = rotate_physical_image(image, X, Y, 180)
            point = np.unravel_index(np.argmax(rotated), rotated.shape)
            np.testing.assert_allclose([newX[point], newY[point]], [0, -.025], atol=1e-16)
            legacy_landmarks.append([X[point], Y[point]])
            self.assertFalse(report["boundary_wrap"])
            self.assertFalse(report["interpolation"])
            # Retain the actual shifted grid instead of relabelling rotated
            # samples onto the original floor(n/2)-centred coordinates.
            np.testing.assert_allclose(newX - X, .2 / size, atol=1e-16)
            np.testing.assert_allclose(newY - Y, .2 / size, atol=1e-16)
        self.assertGreater(np.linalg.norm(np.subtract(legacy_landmarks[0], legacy_landmarks[-1])), .006)

    def test_reindexing_changes_no_physical_sample(self):
        X, Y = grid(5, 8)
        values = 4 + 3 * X - 7 * Y
        moved, newX, newY, report = reindex_image(values, X, Y, 180)
        np.testing.assert_allclose(moved, 4 + 3 * newX - 7 * newY)
        self.assertFalse(report["changes_physical_placement"])
        self.assertLess(newX[0, -1], newX[0, 0])

    def test_physical_rotation_about_offset_center_and_receiver_axis(self):
        X, Y = grid(4, 6)
        X, Y = X + .02, Y - .03
        values = np.stack([X + 2 * Y, 3 * X - Y], axis=1)
        moved, newX, newY, _ = rotate_physical_image(
            values, X, Y, 180, center_m=(.02, -.03), axes=(0, 2))
        np.testing.assert_allclose(moved[:, 0], (.04 - newX) + 2 * (-.06 - newY))
        np.testing.assert_allclose(moved[:, 1], 3 * (.04 - newX) - (-.06 - newY))
        restored, restoredX, restoredY, _ = rotate_physical_image(
            moved, newX, newY, 180, center_m=(.02, -.03), axes=(0, 2))
        np.testing.assert_allclose(restored, values)
        np.testing.assert_allclose(restoredX, X)
        np.testing.assert_allclose(restoredY, Y)

    def test_fft_origin_reversal_preserves_origin_and_is_involution(self):
        for shape in [(32, 32), (5, 8), (7, 9)]:
            image = np.zeros(shape)
            center = tuple(size // 2 for size in shape)
            image[center] = 1
            reversed_image, report = fft_origin_reverse(image, return_report=True)
            np.testing.assert_array_equal(reversed_image, image)
            data = np.arange(np.prod(shape)).reshape(shape)
            np.testing.assert_array_equal(fft_origin_reverse(fft_origin_reverse(data)), data)
            self.assertTrue(report["boundary_wrap"])

    def test_fft_origin_reversal_matches_opposite_fourier_sign(self):
        rng = np.random.default_rng(47)
        for shape in [(4, 6), (5, 7)]:
            k = rng.normal(size=shape) + 1j * rng.normal(size=shape)
            image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))
            opposite = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(k))) / k.size
            np.testing.assert_allclose(fft_origin_reverse(image), opposite, atol=5e-15)

    def test_comparison_gain_and_edges_exact_for_scaled_reference(self):
        reference = np.zeros((16, 20))
        reference[4:12, 5:15] = 1
        result = compare_magnitude(reference, reference * 4)
        self.assertAlmostEqual(result["gain"], .25)
        self.assertEqual(result["NRMSE_percent"], 0)
        self.assertEqual(result["edge_gradient_NRMSE_percent"], 0)
        self.assertEqual(result["contour_Dice"], 1)
        self.assertFalse(result["nonrigid_registration_applied"])

    def test_geometric_metrics_detect_one_pixel_shift_without_alignment(self):
        reference = np.zeros((16, 20))
        reference[4:12, 5:15] = 1
        candidate = np.roll(reference, 1, axis=1)
        X, Y = np.meshgrid(np.arange(20) * .001, np.arange(16) * .002)
        result = compare_magnitude(reference, candidate, gain=1, X_m=X, Y_m=Y)
        self.assertGreater(result["NRMSE_percent"], 0)
        self.assertGreater(result["edge_gradient_NRMSE_percent"], 0)
        self.assertLess(result["contour_Dice"], 1)
        self.assertAlmostEqual(result["centroid_distance_mm"], 1)
        self.assertAlmostEqual(result["centroid_RMS_mm"], 1 / np.sqrt(2))
        self.assertFalse(result["gain_fitted"])

    def test_common_mask_excludes_holes_from_gradient_without_false_boundary_edges(self):
        X, Y = grid(8, 10)
        reference = 2 + X + Y
        candidate = reference.copy()
        mask = np.ones(reference.shape, bool)
        mask[3, 4] = False
        reference[3, 4], candidate[3, 4] = np.nan, np.nan
        result = compare_magnitude(reference, candidate, mask, gain=1)
        self.assertEqual(result["NRMSE_percent"], 0)
        self.assertEqual(result["edge_gradient_NRMSE_percent"], 0)
        self.assertLess(result["edge_pixels"], result["used_pixels"])

    def test_invalid_geometry_and_missing_centroid_grid_are_rejected(self):
        X, Y = grid(4, 6)
        with self.assertRaisesRegex(ValueError, "rotation_deg"):
            rotate_physical_image(X, X, Y, 90)
        with self.assertRaisesRegex(ValueError, "matching"):
            rotate_physical_image(X, X.T, Y.T, 180)
        with self.assertRaisesRegex(ValueError, "both X_m"):
            compare_magnitude(np.ones(X.shape), np.ones(X.shape), X_m=X)


if __name__ == "__main__":
    unittest.main()
