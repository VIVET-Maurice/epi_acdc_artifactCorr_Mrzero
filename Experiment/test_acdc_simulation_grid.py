"""Regression for the 32x32-reference / 64x64-scan point-source artifact."""

import unittest

import numpy as np

from prepare_acdc_calibration import target_patient_grid
from prepare_acdc_simulation_grid import prepare_simulation_reference
from resample_acdc_maps import resample_to_grid


def reference_fixture(shape=(32, 32), fov=(.2, .2)):
    ny, nx = shape
    dx, dy = np.asarray(fov) / [nx, ny]
    X, Y = np.meshgrid((np.arange(nx) - nx // 2) * dx,
                       (np.arange(ny) - ny // 2) * dy)
    return {"magnitude": np.exp(-(X**2 + Y**2) / .03**2), "X_m": X, "Y_m": Y,
            "position_mm": np.array([ny // 2 * dy, -(nx // 2) * dx, .007]) * 1000,
            "spacing_mm": np.array([dy, dx]) * 1000,
            "orientation": np.array([0, 1, 0, -1, 0, 0]), "slice_thickness_mm": 5.}


def scan_fixture(shape=(64, 64), fov=(.2, .2)):
    kx, ky = np.meshgrid((np.arange(shape[1]) - shape[1] // 2) / fov[0],
                         (np.arange(shape[0]) - shape[0] // 2) / fov[1])
    return {"kx_mInv": kx, "ky_mInv": ky, "FOV_m": np.asarray(fov)}


def zero_b0_point_source_image(reference, scan):
    # Independent separable Fourier sum, matching point-source physics.
    ex = np.exp(-2j * np.pi * scan["kx_mInv"][0, :, None] * reference["X_m"][0, None, :])
    ey = np.exp(-2j * np.pi * scan["ky_mInv"][:, 0, None] * reference["Y_m"][None, :, 0])
    kspace = ey @ reference["magnitude"] @ ex.T
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))


class SimulationGridTests(unittest.TestCase):
    def test_refinement_removes_the_zero_b0_comb(self):
        reference, scan = reference_fixture(), scan_fixture()
        old_image = zero_b0_point_source_image(reference, scan)
        off_lattice = np.ones(old_image.shape, bool)
        off_lattice[::2, ::2] = False
        self.assertLess(float(np.linalg.norm(old_image[off_lattice])), 1e-12)
        simulation, report = prepare_simulation_reference(reference, scan)
        image = zero_b0_point_source_image(simulation, scan)
        self.assertEqual(simulation["magnitude"].shape, (64, 64))
        self.assertTrue(report["refined"])
        np.testing.assert_allclose(image, simulation["magnitude"], atol=3e-14, rtol=0)
        self.assertGreater(float(np.linalg.norm(image[off_lattice])), 1)
        # Refinement retains native measurements at their original locations.
        np.testing.assert_allclose(simulation["magnitude"][::2, ::2], reference["magnitude"], atol=1e-14)

    def test_equal_or_finer_reference_is_unchanged(self):
        for shape in ((64, 64), (128, 128)):
            with self.subTest(shape=shape):
                reference = reference_fixture(shape)
                simulation, report = prepare_simulation_reference(reference, scan_fixture())
                self.assertFalse(report["refined"])
                for key in ("magnitude", "X_m", "Y_m", "position_mm", "spacing_mm"):
                    np.testing.assert_array_equal(simulation[key], reference[key])

    def test_fov_and_axis_spacing_control_refinement(self):
        reference = reference_fixture()
        simulation, _ = prepare_simulation_reference(reference, scan_fixture(fov=(.2, .4)))
        self.assertEqual(simulation["magnitude"].shape, (32, 64))
        simulation, report = prepare_simulation_reference(reference, scan_fixture(fov=(.4, .4)))
        self.assertEqual(simulation["magnitude"].shape, (32, 32))
        self.assertFalse(report["refined"])

    def test_updated_patient_geometry_matches_every_simulation_coordinate(self):
        reference = reference_fixture((25, 31), (.18, .2))
        simulation, _ = prepare_simulation_reference(reference, scan_fixture((64, 64)))
        patient = target_patient_grid({"geometry": simulation})
        np.testing.assert_allclose(patient[..., 1], simulation["X_m"], atol=1e-15)
        np.testing.assert_allclose(-patient[..., 0], simulation["Y_m"], atol=1e-15)
        np.testing.assert_allclose(patient[..., 2], .007)
        center = tuple(np.asarray(simulation["magnitude"].shape) // 2)
        self.assertEqual(simulation["X_m"][center], 0)
        self.assertEqual(simulation["Y_m"][center], 0)

    def test_b0_and_channel_maps_retain_physical_values_and_validity(self):
        reference, scan = reference_fixture(), scan_fixture()
        simulation, _ = prepare_simulation_reference(reference, scan)
        X, Y = reference["X_m"], reference["Y_m"]
        b0 = 20 + 100 * X - 50 * Y
        mask = np.ones(X.shape, bool)
        mask[16, 16] = False
        maps = b0[..., None] * np.arange(1, 33)
        b0_fine, valid = resample_to_grid(b0, X, Y, mask, simulation["X_m"], simulation["Y_m"])
        maps_fine, maps_valid = resample_to_grid(maps, X, Y, mask, simulation["X_m"], simulation["Y_m"])
        expected = 20 + 100 * simulation["X_m"] - 50 * simulation["Y_m"]
        np.testing.assert_allclose(b0_fine[valid], expected[valid], atol=1e-13)
        np.testing.assert_allclose(maps_fine[valid], expected[valid, None] * np.arange(1, 33), atol=1e-12)
        np.testing.assert_array_equal(valid, maps_valid)
        self.assertFalse(valid[32, 32])
        self.assertTrue(np.isnan(maps_fine[~valid]).all())

    def test_reference_boundary_is_excluded_without_extrapolation(self):
        simulation, _ = prepare_simulation_reference(reference_fixture(), scan_fixture())
        self.assertFalse(simulation["valid_mask"][-1].any())
        self.assertFalse(simulation["valid_mask"][:, -1].any())
        self.assertTrue(np.all(simulation["magnitude"][~simulation["valid_mask"]] == 0))

    def test_transposed_and_reversed_source_axes_are_supported(self):
        reference = reference_fixture((20, 32))
        for transpose, reverse in ((True, False), (False, True)):
            source = {key: (reference[key].T if transpose else reference[key][:, ::-1])
                      for key in ("magnitude", "X_m", "Y_m")}
            simulation, _ = prepare_simulation_reference(source, scan_fixture())
            self.assertEqual(simulation["magnitude"].shape, (64, 64))
            self.assertTrue(np.isfinite(simulation["magnitude"]).all())

    def test_header_rounding_does_not_add_an_extra_pixel(self):
        simulation, _ = prepare_simulation_reference(reference_fixture(),
                                                     scan_fixture(fov=(.1999999, .1999998)))
        self.assertEqual(simulation["magnitude"].shape, (64, 64))


if __name__ == "__main__":
    unittest.main()
