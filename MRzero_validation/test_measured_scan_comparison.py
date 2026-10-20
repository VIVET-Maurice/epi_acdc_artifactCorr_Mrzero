"""Shared-gain, native geometry and measured reader parity regressions."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from measured_scan_comparison import compare_registered_images, compare_measured_scan_set


def record(shape=(8, 10), value=None, fov=(.2, .16), label="image"):
    ny, nx = shape
    X, Y = np.meshgrid((np.arange(nx) - nx // 2) * fov[0] / nx,
                       (np.arange(ny) - ny // 2) * fov[1] / ny)
    if value is None:
        value = 2 + X + Y
    return {"image": np.asarray(value), "X_m": X, "Y_m": Y, "label": label}


class MeasuredComparisonTests(unittest.TestCase):
    def test_unique_baseline_gain_applies_unchanged_to_every_corrected_scan(self):
        reference = record(label="GRE")
        baseline = record(value=reference["image"] * .5, label="baseline")
        corrected = [record(value=reference["image"] * .25, label="85"),
                     record(value=reference["image"] * .75, label="86")]
        result = compare_registered_images(reference, baseline, corrected)
        self.assertAlmostEqual(result["gain"], 2)
        self.assertAlmostEqual(result["rows"][0]["NRMSE_percent"], 0)
        self.assertAlmostEqual(result["rows"][1]["NRMSE_percent"], 50)
        self.assertAlmostEqual(result["rows"][2]["NRMSE_percent"], 50)
        self.assertTrue(result["report"]["same_gain_for_all_corrected_scans"])

    def test_arbitrary_rotation_mirror_changes_coordinates_not_array_order_or_errors(self):
        reference, baseline = record(), record()
        angle = np.deg2rad(37)
        matrix = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ np.diag([-1, 1])
        original = compare_registered_images(reference, baseline, [record()], normalize=False)
        result = compare_registered_images(reference, baseline, [record()], frame_matrix=matrix, normalize=False)
        for actual, expected in zip(result["images"], original["images"]):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(result["X_m"], matrix[0, 0] * baseline["X_m"] + matrix[0, 1] * baseline["Y_m"])
        np.testing.assert_allclose(result["Y_m"], matrix[1, 0] * baseline["X_m"] + matrix[1, 1] * baseline["Y_m"])
        self.assertEqual(result["rows"], original["rows"])

    def test_coarse_gre_is_resampled_to_epi_without_extrapolation(self):
        reference, baseline = record((4, 4)), record((8, 8))
        result = compare_registered_images(reference, baseline, [record((8, 8))], reference_threshold_fraction=0)
        self.assertEqual(result["images"][0].shape, (8, 8))
        self.assertFalse(result["mask"][-1].any())
        self.assertFalse(result["mask"][:, -1].any())
        self.assertGreater(result["report"]["excluded_missing_coverage_pixels"], 0)
        self.assertAlmostEqual(result["rows"][0]["NRMSE_percent"], 0, places=12)

    def test_default_gre_threshold_and_full_coverage_share_one_gain(self):
        image = np.zeros((8, 10))
        image[2:6, 3:7] = 2
        reference = record(value=image)
        baseline = record(value=image * .5)
        corrected = record(value=image * .25)
        roi = compare_registered_images(reference, baseline, [corrected])
        full = compare_registered_images(reference, baseline, [corrected], reference_threshold_fraction=0)
        self.assertEqual(int(roi["mask"].sum()), 16)
        self.assertEqual(int(full["mask"].sum()), 80)
        self.assertAlmostEqual(roi["gain"], 2)
        self.assertAlmostEqual(full["gain"], 2)
        self.assertAlmostEqual(roi["rows"][1]["NRMSE_percent"], 50)
        self.assertAlmostEqual(full["rows"][1]["NRMSE_percent"], 50)

    def test_all_corrected_scans_share_intersection_mask_and_roi(self):
        reference, baseline = record(), record()
        first, second = record(), record()
        first["image"][0, 0] = np.nan
        second["image"][-1, -1] = np.nan
        roi = np.ones(reference["image"].shape, bool)
        roi[0] = False
        result = compare_registered_images(reference, baseline, [first, second], roi_mask=roi)
        self.assertFalse(result["mask"][0].any())
        self.assertFalse(result["mask"][-1, -1])
        self.assertEqual(result["report"]["common_pixels"], int(roi.sum()) - 1)
        self.assertTrue(all(row["NRMSE_percent"] == 0 for row in result["rows"]))

    def test_high_level_reader_baseline_rss_parity_and_geometry_rejection(self):
        reference = record((4, 4))
        image = reference["image"] * .5
        kdata = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
        diagnostics = {"kspace": kdata[:, None, :], "corrected_kspace": kdata[:, None, :]}
        acquisition = {"kx_mInv": np.meshgrid((np.arange(4) - 2) / .2, (np.arange(4) - 2) / .16)[0],
                       "ky_mInv": np.meshgrid((np.arange(4) - 2) / .2, (np.arange(4) - 2) / .16)[1],
                       "FOV_m": [.2, .16], "TE_s": .02, "TR_s": 1., "dwell_s": 8e-6,
                       "polarity": np.ones(4), "line_index": np.repeat(np.arange(1, 5)[:, None], 4, axis=1)}
        metadata = {"slice_position_mm": [0., 0., 3.], "slice_quaternion_wxyz": [2**-.5, 0., 0., 2**-.5],
                    "image_shape": [4, 4], "FOV_m": [.2, .16], "TE_s": .02, "TR_s": 1.,
                    "dwell_s": 8e-6, "navigator_count": 0, "polarity": np.ones(4),
                    "acquired_phase_rows": list(range(4)), "slice_thickness_mm": 5.}
        inputs = {"gre": {"magnitude": reference["image"], "X_m": reference["X_m"],
                           "Y_m": reference["Y_m"], "twix_metadata": metadata},
                  "scan": image, "scan_metadata": metadata, "scan_diagnostics": diagnostics,
                  "acquisition": acquisition, "timing": {"navigator_count": 0}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan85.dat"
            path.write_bytes(b"mock Twix")
            with patch("measured_scan_comparison.read_scan", return_value=(image, metadata, deepcopy(diagnostics))):
                result = compare_measured_scan_set(inputs, [path])
            self.assertLess(result["report"]["baseline_reader_RSS_relative_error"], 1e-14)
            self.assertAlmostEqual(result["gain"], 2)
            self.assertEqual(len(result["rows"]), 2)
            bad = dict(metadata, slice_position_mm=[0., 0., 4.])
            with patch("measured_scan_comparison.read_scan", return_value=(image, bad, deepcopy(diagnostics))):
                with self.assertRaisesRegex(ValueError, "centres differ"):
                    compare_measured_scan_set(inputs, [path])
            # A contrast/timing mismatch never reaches the continuous
            # reconstruction. The rest of a notebook can report and continue.
            wrong_timing = dict(metadata, TE_s=.08)
            with patch("measured_scan_comparison.read_scan", return_value=(image, wrong_timing, deepcopy(diagnostics))):
                skipped = compare_measured_scan_set(inputs, [path], skip_incompatible=True)
            self.assertEqual(skipped["images"], [])
            self.assertFalse(skipped["report"]["comparison_available"])
            self.assertIn("TE_s", skipped["report"]["skipped_scans"][0]["reason"])
            second = Path(directory) / "scan86.dat"
            second.write_bytes(b"another mock Twix")
            with patch("measured_scan_comparison.read_scan", side_effect=[
                    (image, wrong_timing, deepcopy(diagnostics)), (image, metadata, deepcopy(diagnostics))]):
                mixed = compare_measured_scan_set(inputs, [path, second], skip_incompatible=True)
            self.assertTrue(mixed["report"]["comparison_available"])
            self.assertEqual(mixed["labels"][-1], second.name)
            self.assertEqual(mixed["report"]["compatible_scan_count"], 1)
            self.assertEqual(len(mixed["report"]["skipped_scans"]), 1)

            denied = PermissionError(13, "Permission denied", str(path))
            with patch("measured_scan_comparison.read_scan", side_effect=denied):
                with self.assertRaises(PermissionError):
                    compare_measured_scan_set(inputs, [path])
                inaccessible = compare_measured_scan_set(inputs, [path], skip_incompatible=True)
            self.assertEqual(inaccessible["images"], [])
            self.assertFalse(inaccessible["report"]["comparison_available"])
            self.assertEqual(inaccessible["report"]["skipped_scans"][0]["error_type"], "PermissionError")
            self.assertIn(path.name, inaccessible["report"]["skipped_scans"][0]["reason"])
            with patch("measured_scan_comparison.read_scan", side_effect=[
                    denied, (image, metadata, deepcopy(diagnostics))]):
                accessible = compare_measured_scan_set(inputs, [path, second], skip_incompatible=True)
            self.assertTrue(accessible["report"]["comparison_available"])
            self.assertEqual(accessible["report"]["compatible_scan_count"], 1)
            self.assertEqual(accessible["labels"][-1], second.name)
            self.assertEqual(accessible["report"]["skipped_scans"][0]["error_type"], "PermissionError")


if __name__ == "__main__":
    unittest.main()
