"""Verify NPZ channel/coordinate handling and equivalence to the supplied MAT."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import loadmat, savemat

from read_acdc_calibration import read_calibration


ROOT = Path(__file__).resolve().parent


class CalibrationNpzTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ACDC_3T.npz"
        x, y, z = np.meshgrid(np.arange(4) * .01, np.arange(4) * .01,
                              np.arange(3) * .01, indexing="ij")
        self.source = {"xx": x, "yy": y, "zz": z, "mask": np.ones(x.shape, bool),
                       "fieldmap": 10*x[..., None] + 20*y[..., None] + 30*z[..., None]
                                   + np.arange(1, 33)}
        self.save_source()
        self.settings = {"units": "Hz/A", "registration_confirmed": True,
                         "patient_to_source_m": np.eye(4).tolist()}
        # Pixel columns follow patient +x; rows follow patient +y.
        row, col = np.indices((2, 2))
        self.experiment = {"geometry": {"X_m": .005 + row*.01, "Y_m": -(.005 + col*.01),
                            "orientation": [1, 0, 0, 0, 1, 0],
                            "position_mm": [5, 5, 5], "spacing_mm": [10, 10]}}

    def save_source(self):
        np.savez(self.path, **self.source)

    def read(self, settings=None):
        return read_calibration(self.path, self.experiment,
                                self.settings if settings is None else settings)

    def test_interpolation_preserves_patient_axes_sign_and_channel_order(self):
        result = self.read()
        geometry = self.experiment["geometry"]
        expected = (-10*geometry["Y_m"] + 20*geometry["X_m"] + 30*.005)[..., None] + np.arange(1, 33)
        np.testing.assert_allclose(result["acdc_maps"], expected, rtol=0, atol=1e-13)
        self.assertTrue(result["spatial_valid_mask"].all())
        np.testing.assert_array_equal(result["metadata"]["source_channel_order"], np.arange(1, 33))

    def test_patient_to_source_affine_is_applied_in_metres(self):
        self.settings["patient_to_source_m"][0][3] = .005
        original = self.read({**self.settings, "patient_to_source_m": np.eye(4).tolist()})
        translated = self.read()
        np.testing.assert_allclose(translated["acdc_maps"] - original["acdc_maps"], .05, atol=1e-13)

    def test_missing_source_stencil_stays_unknown(self):
        self.source["mask"][0, 0, 0] = False
        self.save_source()
        result = self.read()
        self.assertFalse(result["spatial_valid_mask"][0, 0])
        self.assertTrue(np.isnan(result["acdc_maps"][0, 0]).all())
        self.assertTrue(result["spatial_valid_mask"][1, 1])

    def test_no_extrapolation(self):
        self.settings["patient_to_source_m"][2][3] = 1
        result = self.read()
        self.assertFalse(result["spatial_valid_mask"].any())
        self.assertTrue(np.isnan(result["acdc_maps"]).all())

    def test_explicit_units_registration_and_affine_required(self):
        for key in ("units", "registration_confirmed", "patient_to_source_m"):
            settings = copy.deepcopy(self.settings)
            del settings[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.read(settings)

    def test_bad_affine_and_ambiguous_patient_geometry_rejected(self):
        self.settings["patient_to_source_m"][3][0] = .1
        with self.assertRaisesRegex(ValueError, "affine"):
            self.read()
        self.settings["patient_to_source_m"] = np.eye(4).tolist()
        del self.experiment["geometry"]["position_mm"]
        with self.assertRaisesRegex(ValueError, "patient geometry"):
            self.read()

    def test_changed_npz_cannot_reuse_confirmed_source_hash(self):
        self.settings["source_sha256"] = self.read()["metadata"]["source_sha256"]
        self.assertTrue(self.read()["registration_confirmed"])
        self.source["fieldmap"][0, 0, 0, 0] += 1
        self.save_source()
        with self.assertRaisesRegex(ValueError, "source_sha256 differs"):
            self.read()

    def test_settings_must_be_an_object(self):
        for settings in (False, [], "registered", 1):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "must be an object"):
                self.read(settings)

    def test_mat_input_keeps_existing_metadata(self):
        path = self.path.with_suffix(".mat")
        savemat(path, {"acdc_maps": np.ones((2, 2, 32)), "units": "Hz/A",
                       "registration_confirmed": False})
        calibration = read_calibration(path, self.experiment, self.settings)
        self.assertEqual(calibration["units"], "Hz/A")
        self.assertFalse(calibration["registration_confirmed"])

    def test_2d_npz_reads_embedded_metadata_without_overriding_registration(self):
        values = {"acdc_maps": np.arange(128).reshape(2, 2, 32), "units": "Hz/A",
                  "registration_confirmed": False, "spatial_valid_mask": np.ones((2, 2), bool),
                  "X_m": self.experiment["geometry"]["X_m"], "Y_m": self.experiment["geometry"]["Y_m"]}
        np.savez(self.path, **values)
        actual = self.read()
        for key, expected in values.items():
            np.testing.assert_array_equal(actual[key], expected)
        self.assertFalse(actual["registration_confirmed"])

    def test_2d_npz_requires_embedded_units_and_registration(self):
        np.savez(self.path, acdc_maps=np.ones((2, 2, 32)))
        with self.assertRaisesRegex(ValueError, "2D ACDC NPZ requires"):
            self.read()

    @unittest.skipUnless((ROOT / "acdc_input/calibration.mat").is_file()
                         and (ROOT / "ACDC_maps/ACDC_3T.npz").is_file()
                         and (ROOT / "prepared/experiment_input.mat").is_file(),
                         "Supplied Bay5 calibration and prepared DICOM geometry are absent")
    def test_supplied_bay5_npz_matches_confirmed_mat(self):
        experiment = loadmat(ROOT / "prepared/experiment_input.mat", simplify_cells=True)["experiment"]
        settings = json.loads((ROOT / "acdc_input/calibration_provenance.json").read_text())
        expected = loadmat(ROOT / "acdc_input/calibration.mat", simplify_cells=True)
        actual = read_calibration(ROOT / "ACDC_maps/ACDC_3T.npz", experiment, settings)
        np.testing.assert_array_equal(actual["spatial_valid_mask"], expected["spatial_valid_mask"])
        np.testing.assert_allclose(actual["acdc_maps"], expected["acdc_maps"], rtol=0, atol=1e-12, equal_nan=True)
        self.assertEqual(actual["metadata"]["source_sha256"], settings["source_sha256"])


if __name__ == "__main__":
    unittest.main()
