"""Independent affine/noise/mask checks for native B0 regularization."""

import unittest

import numpy as np

from b0_regularization import regularize_b0_local_plane


class LocalPlaneB0Tests(unittest.TestCase):
    def test_affine_field_survives_irregular_boundaries_and_holes(self):
        row, column = np.indices((30, 35))
        field = 17. + 1.3 * row - .7 * column
        mask = ((row - 13)**2 / 130 + (column - 18)**2 / 160 < 1)
        mask[11:15, 17:21] = False
        mask[5:9, 20:25] = False
        result = regularize_b0_local_plane(field, mask, sigma_pixels=1.)
        self.assertGreater(result["report"]["fitted_pixels"], 100)
        np.testing.assert_allclose(result["b0Hz"][mask], field[mask], atol=1e-12)
        np.testing.assert_array_equal(result["b0Hz"][~mask], field[~mask])
        np.testing.assert_array_equal(result["mask"], mask)

    def test_constant_field_is_unchanged(self):
        mask = np.ones((15, 20), bool)
        mask[3:5, 7:11] = False
        result = regularize_b0_local_plane(np.full(mask.shape, -73.), mask)
        np.testing.assert_allclose(result["b0Hz"], -73., atol=1e-12)

    def test_fixed_bandwidth_reduces_known_noise_and_preserves_measured_mean(self):
        row, column = np.indices((64, 64))
        truth = 30 + .5 * row - .2 * column
        mask = (row - 32)**2 + (column - 32)**2 < 27**2
        noise = np.random.default_rng(31).normal(0, 3., truth.shape)
        measured = truth + noise
        result = regularize_b0_local_plane(measured, mask, sigma_pixels=.75, spacing_mm=[1.5, 1.5])
        before = np.sqrt(np.mean((measured[mask] - truth[mask])**2))
        after = np.sqrt(np.mean((result["b0Hz"][mask] - truth[mask])**2))
        self.assertLess(after, .5 * before)
        self.assertAlmostEqual(result["b0Hz"][mask].mean(), measured[mask].mean(), places=12)
        np.testing.assert_allclose(result["report"]["sigma_mm_row_column"], [1.125, 1.125])
        self.assertEqual(result["report"]["invalid_pixels_filled"], 0)

    def test_input_is_immutable_and_invalid_measurements_are_not_filled(self):
        measured = np.arange(100, dtype=float).reshape(10, 10)
        measured[4, 5] = np.nan
        measured[0, 0] = -999.
        mask = np.ones(measured.shape, bool)
        mask[0, 0] = False
        original, original_mask = measured.copy(), mask.copy()
        result = regularize_b0_local_plane(measured, mask)
        np.testing.assert_array_equal(measured, original)
        np.testing.assert_array_equal(mask, original_mask)
        self.assertTrue(np.isnan(result["b0Hz"][4, 5]))
        self.assertFalse(result["mask"][4, 5])
        self.assertEqual(result["b0Hz"][0, 0], -999.)
        np.testing.assert_array_equal(result["raw_b0Hz"], original)

    def test_singular_isolated_pixel_keeps_raw_value_during_mean_restore(self):
        rng = np.random.default_rng(32)
        measured = rng.normal(size=(20, 20))
        mask = np.zeros(measured.shape, bool)
        mask[3:9, 3:9] = True
        mask[17, 17] = True
        measured[17, 17] = 123.
        result = regularize_b0_local_plane(measured, mask, sigma_pixels=.75)
        self.assertEqual(result["report"]["singular_raw_fallback_pixels"], 1)
        self.assertEqual(result["b0Hz"][17, 17], 123.)
        self.assertAlmostEqual(result["b0Hz"][mask].mean(), measured[mask].mean(), places=12)

    def test_anisotropic_bandwidth_preserves_linear_field(self):
        row, column = np.indices((12, 16))
        measured = 7 + .3 * row + .7 * column
        result = regularize_b0_local_plane(measured, np.ones(measured.shape, bool),
                                            sigma_pixels=[.75, 1.2], spacing_mm=[1., 2.])
        np.testing.assert_allclose(result["b0Hz"], measured, atol=1e-12)
        np.testing.assert_allclose(result["report"]["sigma_mm_row_column"], [.75, 2.4])


if __name__ == "__main__":
    unittest.main()
