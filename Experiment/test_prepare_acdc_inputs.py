"""Input/geometry regressions and comparison with the existing measured notebook data."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from prepare_acdc_inputs import (check_physical_geometry, prepare_inputs,
                                 read_gre, read_scan, read_sequence, centered_fft,
                                 check_reference_sequence)
from resample_acdc_maps import resample_to_grid


ROOT = Path(__file__).resolve().parent
SEQUENCE = ROOT / "sequence" / "gre_epi_32x32_fov200mm_nav3_mintr.seq"


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        row, col = np.indices((128, 128))
        self.X = (row - 64) * .0015625
        self.Y = -(col - 64) * .0015625
        self.mask = np.zeros((128, 128), dtype=bool)
        self.mask[48:80, 48:80] = True
        self.gre = 1 + row / 128
        self.b0 = np.where(self.mask, 23.5, np.nan)
        self.config = {
            "files": {"gre": "gre.npz", "b0": "b0.npz", "scan": "scan.npz",
                      "sequence": str(SEQUENCE)},
            "preparation": {"geometry_confirmed": True, "b0_sign_confirmed": True,
                            "b0_units": "Hz"},
        }
        np.savez(self.folder / "gre.npz", magnitude=self.gre,
                 X_m=self.X, Y_m=self.Y, mask=self.mask)
        np.savez(self.folder / "b0.npz", b0Hz=self.b0,
                 X_m=self.X, Y_m=self.Y, mask=self.mask)
        np.savez(self.folder / "scan.npz", image=np.full((32, 32), 500.0))

    def prepare(self):
        return prepare_inputs(self.folder, self.folder / "prepared", self.config)

    def test_independent_gre_amplitude_and_no_field_extrapolation(self):
        report = self.prepare()
        experiment = loadmat(report["experiment_input"], simplify_cells=True)["experiment"]
        expected = np.where(self.mask, self.gre / self.gre[self.mask].max(), 0)
        np.testing.assert_allclose(experiment["m0_proxy"], expected)
        np.testing.assert_array_equal(experiment["images"]["scan"], 500)
        self.assertTrue(np.isnan(experiment["field"]["measured_b0Hz"][~self.mask]).all())
        np.testing.assert_allclose(experiment["field"]["b0Hz"][self.mask], 23.5)
        self.assertEqual(len(report["state_edges_s"]), 33)
        np.testing.assert_allclose(report["first_adc_delay_s"], 94e-6, atol=1e-12)
        np.testing.assert_allclose(report["first_adc_sample_center_delay_s"], 99.5e-6, atol=1e-12)

    def test_npz_reference_metadata_is_serializable_and_requires_finite_scalars(self):
        arrays = dict(magnitude=self.gre, X_m=self.X, Y_m=self.Y, mask=self.mask)
        metadata = dict(TE_s=.004, TR_s=.012, flip_deg=15)
        np.savez(self.folder / "gre.npz", **arrays, **metadata)
        report = self.prepare()
        self.assertEqual(report["reference_metadata"], metadata)
        for key in metadata:
            for invalid in (np.array([1., 2.]), np.array([.004]), np.nan, np.inf, .004j):
                with self.subTest(key=key, invalid=invalid):
                    np.savez(self.folder / "gre.npz", **arrays, **{key: invalid})
                    with self.assertRaisesRegex(ValueError, f"GRE {key} must be a finite real scalar"):
                        read_gre(self.folder / "gre.npz")

    def test_b0_hole_rejected_before_output(self):
        self.b0[60, 60] = np.nan
        np.savez(self.folder / "b0.npz", b0Hz=self.b0, X_m=self.X, Y_m=self.Y, mask=self.mask)
        with self.assertRaisesRegex(ValueError, "lack reliable measured B0"):
            self.prepare()
        self.assertFalse((self.folder / "prepared").exists())

    def test_shifted_grid_with_insufficient_coverage_rejected(self):
        np.savez(self.folder / "b0.npz", b0Hz=self.b0,
                 X_m=self.X + .001, Y_m=self.Y, mask=self.mask)
        with self.assertRaisesRegex(ValueError, "lack reliable measured B0"):
            self.prepare()

    def test_confirmation_and_units_are_required(self):
        for name, wrong in (("geometry_confirmed", False), ("b0_sign_confirmed", False),
                            ("b0_units", "rad/s")):
            config = copy.deepcopy(self.config)
            config["preparation"][name] = wrong
            with self.subTest(name=name), self.assertRaises(ValueError):
                prepare_inputs(self.folder, self.folder / "prepared", config)

    def test_npy_requires_explicit_grid_confirmation(self):
        np.save(self.folder / "b0.npy", self.b0)
        self.config["files"]["b0"] = "b0.npy"
        with self.assertRaisesRegex(ValueError, "b0_on_gre_grid"):
            self.prepare()
        self.config["preparation"]["b0_on_gre_grid"] = True
        self.assertEqual(self.prepare()["support_voxels"], int(self.mask.sum()))

    def test_no_overwrite_of_previous_preparation(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()

    def test_roi_outside_the_acquired_fov_rejected(self):
        np.savez(self.folder / "gre.npz", magnitude=self.gre,
                 X_m=self.X + .2, Y_m=self.Y, mask=self.mask)
        np.savez(self.folder / "b0.npz", b0Hz=self.b0,
                 X_m=self.X + .2, Y_m=self.Y, mask=self.mask)
        with self.assertRaisesRegex(ValueError, "outside the scan field of view"):
            self.prepare()

    def test_observable_dicom_twix_geometry_mismatch_rejected(self):
        gre = {"position_mm": [-100, -100, 0]}
        scan = {"FOV_m": [.2, .2], "slice_position_mm": [0, 0, 0],
                "slice_quaternion_wxyz": (np.asarray([1, 0, 0, 1]) / np.sqrt(2)).tolist()}
        acquisition = {"FOV_m": [.2, .2]}
        cases = [("FOV_m", [.25, .2], "field of view"),
                 ("slice_position_mm", [0, 0, 4], "different slice planes"),
                 ("slice_position_mm", [4, 0, 0], "centered Twix"),
                 ("slice_quaternion_wxyz", [1, 0, 0, 0], "orientation differs")]
        for key, value, error in cases:
            changed = {**scan, key: value}
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, error):
                check_physical_geometry(gre, changed, acquisition, self.mask, self.X, self.Y,
                                        dicom_input=True)

    def test_sequence_matches_existing_notebook_export(self):
        prepared = ROOT / "prepared" / "experiment_input.mat"
        if not prepared.is_file():
            self.skipTest("Existing notebook export is unavailable.")
        historical = loadmat(prepared, simplify_cells=True)["experiment"]["acquisition"]
        acquisition, timing = read_sequence(SEQUENCE)
        for key in ("kx_mInv", "ky_mInv", "time_s", "line_index", "line_start_s"):
            np.testing.assert_allclose(np.asarray(acquisition[key]).squeeze(), historical[key], atol=1e-12)
        self.assertAlmostEqual(timing["state_edges_s"][-1], .02114)
        self.assertAlmostEqual(timing["rf_center_s"], .0011)

    def test_refocusing_rf_is_not_a_supported_gre_model(self):
        import pypulseq as pp
        sequence = pp.Sequence()
        sequence.read(str(SEQUENCE))
        sequence.add_block(pp.make_block_pulse(np.pi, duration=.001, use="refocusing"))
        path = self.folder / "refocusing.seq"
        sequence.write(str(path))
        with self.assertRaisesRegex(ValueError, "no refocusing RF"):
            read_sequence(path)

    def test_tr_definition_must_match_single_shot_duration(self):
        import pypulseq as pp
        sequence = pp.Sequence()
        sequence.read(str(SEQUENCE))
        sequence.set_definition("TR", .05)
        path = self.folder / "wrong_tr.seq"
        sequence.write(str(path))
        with self.assertRaisesRegex(ValueError, "TR must equal"):
            read_sequence(path)

    def test_actual_dicom_twix_and_masked_measured_b0(self):
        dicom_paths = sorted((ROOT / "scan_dcm").rglob("*.dcm"))
        twix_paths = sorted((ROOT / "scan_twix").glob("*A22_1000.dat"))
        prepared = ROOT / "prepared" / "experiment_input.mat"
        if not dicom_paths or not twix_paths or not prepared.is_file():
            self.skipTest("Raw measured inputs are unavailable.")
        dicom = None
        for path in dicom_paths:
            try:
                read_gre(path)
                dicom = path
                break
            except ValueError:
                pass
        self.assertIsNotNone(dicom)
        historical = loadmat(prepared, simplify_cells=True)["experiment"]
        field, geometry = historical["field"], historical["geometry"]
        np.savez(self.folder / "b0.npz", b0Hz=field["measured_b0Hz"], mask=field["mask"],
                 X_m=geometry["X_m"], Y_m=geometry["Y_m"])
        self.config["files"]["gre"] = str(dicom)
        self.config["files"]["scan"] = str(twix_paths[0])
        self.config["preparation"]["roi_mask"] = "b0.npz"
        report = self.prepare()
        self.assertEqual(report["support_voxels"], int(field["mask"].sum()))
        scan, metadata = read_scan(twix_paths[0])
        np.testing.assert_allclose(scan, historical["images"]["perturbed"], rtol=1e-12)
        self.assertEqual(metadata["navigator_count"], 3)

    def test_reference_dimensions_independent_from_target_scan(self):
        x, y = (np.arange(10) - 5) * .02, (np.arange(12) - 6) * (.2 / 12)
        X, Y = np.meshgrid(x, y)
        mask = np.ones_like(X, dtype=bool)
        np.savez(self.folder / "gre.npz", magnitude=1 + X, X_m=X, Y_m=Y, mask=mask)
        bx, by = np.meshgrid(np.linspace(-.11, .11, 23), np.linspace(-.11, .11, 23))
        np.savez(self.folder / "b0.npz", b0Hz=12 + 3 * bx - 2 * by,
                 X_m=bx, Y_m=by, mask=np.ones_like(bx, dtype=bool))
        report = self.prepare()
        experiment = loadmat(report["experiment_input"], simplify_cells=True)["experiment"]
        self.assertEqual(experiment["m0_proxy"].shape, (12, 10))
        self.assertEqual(experiment["images"]["scan"].shape, (32, 32))
        np.testing.assert_allclose(experiment["field"]["b0Hz"], 12 + 3 * X - 2 * Y)
        self.assertFalse(report["reference_sequence_validation"]["validated"])

    def test_reference_sequence_uses_physical_axes_not_dicom_array_order(self):
        X, Y = np.meshgrid((np.arange(8)-4)*.025, (np.arange(6)-3)*.05, indexing="ij")
        gre = {"X_m": X, "Y_m": Y, "TE_s": .005}
        acquisition = {"kx_mInv": np.zeros((6, 8)), "FOV_m": [.2, .3], "TE_s": .005}
        check_reference_sequence(gre, acquisition)
        with self.assertRaisesRegex(ValueError, "reference matrix"):
            check_reference_sequence(gre, {**acquisition, "kx_mInv": np.zeros((8, 6))})
        with self.assertRaisesRegex(ValueError, "reference FOV"):
            check_reference_sequence(gre, {**acquisition, "FOV_m": [.3, .2]})

    def test_two_gre_sequences_with_distinct_reference_and_scan_matrices(self):
        from test_read_acdc_sequence import make_gre_sequence
        scan_path = make_gre_sequence(self.folder / "scan.seq", nx=8, ny=6, dummies=2)
        reference_path = make_gre_sequence(self.folder / "reference.seq", nx=10, ny=12, te=.004)
        X, Y = np.meshgrid((np.arange(10)-5)*.02, (np.arange(12)-6)*(.2/12))
        mask = np.ones_like(X, dtype=bool)
        np.savez(self.folder / "gre.npz", magnitude=np.ones_like(X), X_m=X, Y_m=Y, mask=mask)
        np.savez(self.folder / "b0.npz", b0Hz=np.zeros_like(X), X_m=X, Y_m=Y, mask=mask)
        np.savez(self.folder / "scan.npz", image=np.ones((6, 8)))
        self.config["files"].update(sequence=str(scan_path), reference_sequence=str(reference_path))
        report = self.prepare()
        experiment = loadmat(report["experiment_input"], simplify_cells=True)["experiment"]
        self.assertTrue(report["reference_sequence_validation"]["validated"])
        self.assertEqual(report["reference_image_shape"], [12, 10])
        self.assertEqual(report["image_shape"], [6, 8])
        self.assertEqual(experiment["images"]["scan"].shape, (6, 8))
        self.assertAlmostEqual(experiment["acquisition"]["TE_s"], .006)
        self.assertAlmostEqual(report["reference_sequence_validation"]["TE_s"], .004)
        self.assertEqual(len(report["state_start_s"]), 6)

    def test_conventional_gre_requires_own_reference_sequence(self):
        from test_read_acdc_sequence import make_gre_sequence
        self.config["files"]["sequence"] = str(make_gre_sequence(self.folder / "scan.seq"))
        with self.assertRaisesRegex(ValueError, "requires files.reference_sequence"):
            self.prepare()


class MeasuredMapResamplingTests(unittest.TestCase):
    def test_transposed_descending_axes_and_multiple_channels(self):
        x, y = np.array([2., 1., 0.]), np.array([-1., 0., 1., 2.])
        X, Y = np.meshgrid(x, y)
        maps = np.stack((3*X + 2*Y, X - Y), axis=2)
        tx, ty = np.meshgrid([.25, 1.75], [-.5, 1.5])
        result, valid = resample_to_grid(maps, X, Y, np.ones_like(X), tx, ty)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(result, np.stack((3*tx + 2*ty, tx - ty), axis=2))

    def test_holes_and_outside_points_are_never_extrapolated(self):
        X, Y = np.meshgrid(np.arange(3.), np.arange(3.), indexing="ij")
        values = X + Y
        values[1, 1] = np.nan
        tx, ty = np.array([[.5, 3.]]), np.array([[.5, 1.]])
        result, valid = resample_to_grid(values, X, Y, np.ones_like(X), tx, ty)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(result).all())

    def test_identical_grid_preserves_measured_values_and_mask(self):
        X, Y = np.meshgrid(np.arange(3.), np.arange(3.))
        mask = np.ones_like(X, dtype=bool)
        mask[1, 1] = False
        result, valid = resample_to_grid(X + Y, X, Y, mask, X, Y)
        np.testing.assert_array_equal(valid, mask)
        np.testing.assert_array_equal(result[mask], (X + Y)[mask])
        self.assertTrue(np.isnan(result[1, 1]))


class GenericTwixTests(unittest.TestCase):
    def fixture(self):
        ny, channels, nx = 6, 3, 8
        rng = np.random.default_rng(15)
        images = rng.normal(size=(ny, channels, nx)) + 1j*rng.normal(size=(ny, channels, nx))
        kspace = centered_fft(images, (0, 2))
        slice_data = SimpleNamespace(SlicePos=SimpleNamespace(Sag=0, Cor=0, Tra=0),
                                     Quaternion=[2**-.5, 0, 0, 2**-.5])
        lines = []
        for row in range(ny):
            block = SimpleNamespace(data=kspace[row], cLin=row, cAve=0, cSlc=0, cPar=0,
                                    cEco=0, cPhs=0, cRep=0, cSet=0,
                                    mdh=SimpleNamespace(SliceData=slice_data, CenterCol=nx//2, CenterLin=ny//2),
                                    is_image_scan=lambda: True, is_flag_set=lambda flag: False)
            lines.append(block)
        header = {"alTE": [5000], "alTR": [10000], "sRXSPEC": {"alDwellTime": [10000]},
                  "adFlipAngleDegree": [15], "sKSpace": {"lBaseResolution": nx, "lPhaseEncodingLines": ny},
                  "sSliceArray": {"asSlice": [{"dReadoutFOV": 200, "dPhaseFOV": 180, "dThickness": 5}]}}
        return {"mdb": lines, "hdr": {"MeasYaps": header}}, np.sqrt(np.sum(abs(images)**2, axis=1))

    def test_full_gre_arbitrary_matrix_receivers_and_no_double_phase_correction(self):
        twix, expected = self.fixture()
        acquisition = {"kx_mInv": np.zeros(expected.shape), "phase_offset_rad": np.ones(expected.shape)*.7}
        with patch("twixtools.read_twix", return_value=[twix]):
            result, metadata = read_scan(Path("synthetic.dat"), acquisition)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        self.assertEqual(metadata["receiver_count"], 3)
        self.assertEqual(metadata["navigator_count"], 0)
        self.assertEqual(metadata["image_shape"], [6, 8])

    def test_gre_diagnostics_preserve_default_reconstruction(self):
        twix, expected = self.fixture()
        with patch("twixtools.read_twix", return_value=[twix]):
            default, metadata = read_scan(Path("synthetic.dat"))
            result, detailed_metadata, intermediate = read_scan(
                Path("synthetic.dat"), return_intermediates=True)
        np.testing.assert_array_equal(result, default)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        self.assertEqual(metadata, detailed_metadata)
        self.assertEqual(intermediate["array_axes"], ["phase", "receiver", "readout"])
        self.assertEqual(intermediate["kspace"].shape, (6, 3, 8))
        np.testing.assert_array_equal(intermediate["corrected_kspace"], intermediate["kspace"])
        np.testing.assert_allclose(
            centered_fft(intermediate["coil_images"], (0, 2)), intermediate["kspace"], atol=1e-12)
        self.assertIsNone(intermediate["navigator_hybrid"])

    def test_epi_diagnostics_recover_known_navigator_phase_error(self):
        twix, expected = self.fixture()
        ny, channels, nx = 6, 3, 8
        clean_kspace = np.stack([block.data for block in twix["mdb"]])
        negative = np.arange(ny) % 2 == 1
        x = np.linspace(-1, 1, nx)
        phase_error = .23 + .51*x
        hybrid = centered_fft(clean_kspace, (2,), inverse=True)
        hybrid[negative] *= np.exp(1j*phase_error)[None, None, :]
        measured = centered_fft(hybrid, (2,))
        for row, block in enumerate(twix["mdb"]):
            block.data = measured[row, :, ::-1] if negative[row] else measured[row]
            block.is_flag_set = lambda flag, reflected=negative[row]: flag == "REFLECT" and reflected
        # A smooth nonzero navigator profile supplies all samples for the fit.
        positive = np.ones((channels, nx), complex) * np.arange(1, channels+1)[:, None]
        navigator_images = np.stack([positive, positive*np.exp(1j*phase_error), positive])
        navigator_kspace = centered_fft(navigator_images, (2,))
        navigators = []
        for index in range(3):
            navigator = copy.copy(twix["mdb"][0])
            navigator.data = navigator_kspace[index, :, ::-1] if index == 1 else navigator_kspace[index]
            navigator.is_image_scan = lambda: False
            navigator.is_flag_set = lambda flag, index=index: flag == "PHASCOR" or (flag == "REFLECT" and index == 1)
            navigators.append(navigator)
        twix["mdb"] = navigators + twix["mdb"]
        with patch("twixtools.read_twix", return_value=[twix]):
            default, metadata = read_scan(Path("synthetic.dat"))
            result, _, intermediate = read_scan(Path("synthetic.dat"), return_intermediates=True)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        np.testing.assert_array_equal(result, default)
        np.testing.assert_allclose(intermediate["corrected_kspace"], clean_kspace, atol=1e-12)
        np.testing.assert_allclose(intermediate["navigator_phase_correction_rad"],
                                   np.broadcast_to(phase_error, (channels, nx)), atol=1e-12)
        self.assertEqual(metadata["navigator_count"], 3)
        self.assertGreater(np.linalg.norm(intermediate["uncorrected_image"]-result), .1)

    def test_image_archive_diagnostics_mark_absent_raw_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.npz"
            np.savez(path, image=np.ones((6, 8)))
            result, metadata, intermediate = read_scan(path, return_intermediates=True)
        np.testing.assert_array_equal(result, 1)
        self.assertEqual(metadata, {})
        self.assertFalse(intermediate["has_raw_kspace"])

    def test_duplicate_rows_or_missing_rows_rejected(self):
        for kind in ("duplicate", "missing"):
            twix, _ = self.fixture()
            if kind == "duplicate":
                twix["mdb"][-1].cLin = 0
            else:
                twix["mdb"].pop()
            with self.subTest(kind=kind), patch("twixtools.read_twix", return_value=[twix]):
                with self.assertRaises(ValueError):
                    read_scan(Path("synthetic.dat"))

    def test_multiecho_or_wrong_sequence_shape_rejected(self):
        twix, _ = self.fixture()
        with patch("twixtools.read_twix", return_value=[twix]):
            with self.assertRaisesRegex(ValueError, "disagrees with Pulseq"):
                read_scan(Path("synthetic.dat"), {"kx_mInv": np.zeros((8, 8))})
            twix["mdb"][-1].cEco = 1
            with self.assertRaisesRegex(ValueError, "one slice, echo"):
                read_scan(Path("synthetic.dat"))


class DicomReferenceTests(unittest.TestCase):
    def test_rectangular_dicom_reference_geometry_and_optional_timing(self):
        fields = {"ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                  "ImagePositionPatient": [-40, -30, 0], "PixelSpacing": [10, 10],
                  "SliceThickness": 5, "EchoTime": 4.2, "RepetitionTime": 12,
                  "FlipAngle": 15}
        elements = [SimpleNamespace(keyword=key, value=value) for key, value in fields.items()]
        dataset = SimpleNamespace(ImageType=["ORIGINAL", "PRIMARY", "M"],
                                  pixel_array=np.ones((6, 8), dtype=np.uint16), iterall=lambda: iter(elements))
        with patch("pydicom.dcmread", return_value=dataset):
            gre = read_gre(Path("reference.dcm"))
        self.assertEqual(gre["magnitude"].shape, (6, 8))
        np.testing.assert_allclose(gre["X_m"][:, 0], np.arange(-3, 3)*.01)
        np.testing.assert_allclose(gre["Y_m"][0], -np.arange(-4, 4)*.01)
        self.assertAlmostEqual(gre["TE_s"], .0042)
        self.assertAlmostEqual(gre["TR_s"], .012)
        self.assertEqual(gre["flip_deg"], 15)


class TwixReferenceTests(unittest.TestCase):
    def fixture(self, ny=6, nx=8, fov=(.2, .18)):
        """Known complex coil images encoded by direct DFT, independently of FFT helpers."""
        channels = 3
        row, column = np.indices((ny, nx))
        magnitude = .3 + (row + 1) / ny + (column + 1) / nx
        images = np.stack([magnitude * (coil + 1) * np.exp(1j * (.13 * row + .21 * column + coil))
                           for coil in range(channels)], axis=1)
        phase = np.arange(ny) - ny // 2
        readout = np.arange(nx) - nx // 2
        phase_encoding = np.exp(-2j * np.pi * phase[:, None] * phase[None, :] / ny)
        read_encoding = np.exp(-2j * np.pi * readout[:, None] * readout[None, :] / nx)
        kspace = np.stack([phase_encoding @ images[:, coil, :] @ read_encoding.T
                           for coil in range(channels)], axis=1)
        slice_data = SimpleNamespace(SlicePos=SimpleNamespace(Sag=0, Cor=0, Tra=7),
                                     Quaternion=[2**-.5, 0, 0, 2**-.5])
        lines = [SimpleNamespace(data=kspace[index], cLin=index, cAve=0, cSlc=0, cPar=0,
                                 cEco=0, cPhs=0, cRep=0, cSet=0,
                                 mdh=SimpleNamespace(SliceData=slice_data, CenterCol=nx//2, CenterLin=ny//2),
                                 is_image_scan=lambda: True, is_flag_set=lambda flag: False)
                 for index in range(ny)]
        header = {"alTE": [5000], "alTR": [10000], "sRXSPEC": {"alDwellTime": [10000]},
                  "adFlipAngleDegree": [15], "sKSpace": {"lBaseResolution": nx, "lPhaseEncodingLines": ny},
                  "sSliceArray": {"asSlice": [{"dReadoutFOV": fov[0]*1000,
                                               "dPhaseFOV": fov[1]*1000, "dThickness": 5}]}}
        acquisition = {"sequence_kind": "gre", "kx_mInv": np.zeros((ny, nx)),
                       "FOV_m": list(fov), "TE_s": .005, "TR_s": .010, "dwell_s": 10e-6,
                       "polarity": np.ones(ny, int),
                       "line_index": np.repeat(np.arange(1, ny+1)[:, None], nx, axis=1)}
        return {"mdb": lines, "hdr": {"MeasYaps": header}}, acquisition, images

    def test_raw_reference_rss_and_physical_geometry_for_distinct_matrices(self):
        from prepare_acdc_calibration import target_patient_grid
        for ny, nx, fov in ((6, 8, (.2, .18)), (10, 12, (.24, .3))):
            twix, acquisition, images = self.fixture(ny, nx, fov)
            with self.subTest(ny=ny, nx=nx), patch("twixtools.read_twix", return_value=[twix]):
                reference = read_gre(Path("reference.dat"), acquisition)
                detailed, diagnostics = read_gre(Path("reference.dat"), acquisition,
                                                  return_intermediates=True)
            np.testing.assert_allclose(reference["magnitude"],
                                       np.sqrt(np.sum(abs(images)**2, axis=1)), atol=1e-12)
            np.testing.assert_array_equal(reference["magnitude"], detailed["magnitude"])
            np.testing.assert_allclose(diagnostics["coil_images"], images, atol=1e-12)
            np.testing.assert_allclose(reference["X_m"][0],
                                       (np.arange(nx)-nx//2)*fov[0]/nx, atol=1e-12)
            np.testing.assert_allclose(reference["Y_m"][:, 0],
                                       (np.arange(ny)-ny//2)*fov[1]/ny, atol=1e-12)
            np.testing.assert_allclose(reference["position_mm"], [fov[1]*500, -fov[0]*500, 7])
            np.testing.assert_allclose(reference["spacing_mm"], [fov[1]*1000/ny, fov[0]*1000/nx])
            patient = target_patient_grid({"geometry": reference})
            np.testing.assert_allclose(patient[0, 0], [fov[1]/2, -fov[0]/2, .007])
            np.testing.assert_allclose(patient[ny//2, nx//2], [0, 0, .007], atol=1e-12)
            self.assertEqual(reference["twix_metadata"]["receiver_count"], 3)

    def test_reflected_lines_and_chronological_phase_order_are_preserved(self):
        twix, acquisition, images = self.fixture()
        order = [3, 0, 5, 1, 4, 2]
        for block in twix["mdb"]:
            block.data = block.data[:, ::-1]
            block.is_flag_set = lambda flag: flag == "REFLECT"
        twix["mdb"] = [twix["mdb"][index] for index in order]
        acquisition["polarity"] = -acquisition["polarity"]
        acquisition["line_index"] = np.repeat((np.argsort(order)+1)[:, None], 8, axis=1)
        with patch("twixtools.read_twix", return_value=[twix]):
            reference = read_gre(Path("reference.dat"), acquisition)
        np.testing.assert_allclose(reference["magnitude"], np.sqrt(np.sum(abs(images)**2, axis=1)), atol=1e-12)
        self.assertEqual(reference["twix_metadata"]["acquired_phase_rows"], order)

    def test_reference_requires_own_mono_echo_gre_sequence(self):
        twix, acquisition, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "own reference_sequence"):
            read_gre(Path("reference.dat"))
        with self.assertRaisesRegex(ValueError, "mono-echo 2D GRE"):
            read_gre(Path("reference.dat"), {**acquisition, "sequence_kind": "epi"})
        with patch("twixtools.read_twix", return_value=[twix]):
            twix["mdb"][-1].cEco = 1
            with self.assertRaisesRegex(ValueError, "one slice, echo"):
                read_gre(Path("reference.dat"), acquisition)

    def test_epi_navigators_rejected_for_reference(self):
        twix, acquisition, _ = self.fixture()
        navigators = []
        for index in range(3):
            navigator = copy.copy(twix["mdb"][0])
            navigator.is_image_scan = lambda: False
            navigator.is_flag_set = lambda flag, index=index: flag == "PHASCOR" or (flag == "REFLECT" and index == 1)
            navigators.append(navigator)
        twix["mdb"] = navigators + twix["mdb"]
        with patch("twixtools.read_twix", return_value=[twix]):
            with self.assertRaisesRegex(ValueError, "without EPI navigators"):
                read_gre(Path("reference.dat"), acquisition)

    def test_reference_sequence_timing_grid_and_order_mismatches_rejected(self):
        twix, acquisition, _ = self.fixture()
        cases = [("TE_s", .006, "TE_s"), ("TR_s", .020, "TR_s"),
                 ("dwell_s", 20e-6, "dwell_s"), ("FOV_m", [.2, .2], "reference FOV"),
                 ("polarity", -np.ones(6), "polarity"),
                 ("line_index", acquisition["line_index"][::-1], "phase-row order"),
                 ("kx_mInv", np.zeros((6, 10)), "disagrees with Pulseq")]
        for key, value, error in cases:
            with self.subTest(key=key), patch("twixtools.read_twix", return_value=[twix]):
                with self.assertRaisesRegex(ValueError, error):
                    read_gre(Path("reference.dat"), {**acquisition, key: value})

    def test_unsupported_reference_orientation_and_offset_rejected(self):
        for field, value, error in (("Quaternion", [1, 0, 0, 0], "axial orientation"),
                                    ("SlicePos", SimpleNamespace(Sag=3, Cor=0, Tra=7), "centered in-plane")):
            twix, acquisition, _ = self.fixture()
            setattr(twix["mdb"][0].mdh.SliceData, field, value)
            with self.subTest(field=field), patch("twixtools.read_twix", return_value=[twix]):
                with self.assertRaisesRegex(ValueError, error):
                    read_gre(Path("reference.dat"), acquisition)

    def test_reference_and_scan_twix_plane_or_rotation_mismatch_rejected(self):
        twix, acquisition, _ = self.fixture()
        with patch("twixtools.read_twix", return_value=[twix]):
            reference = read_gre(Path("reference.dat"), acquisition)
        metadata = reference["twix_metadata"]
        support = np.ones((6, 8), bool)
        for key, value, error in (("slice_position_mm", [0, 0, 8], "different slice planes"),
                                  ("slice_position_mm", [1, 0, 7], "different slice planes"),
                                  ("slice_quaternion_wxyz", [1, 0, 0, 0], "orientations")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, error):
                check_physical_geometry(reference, {**metadata, key: value}, acquisition,
                                        support, reference["X_m"], reference["Y_m"], dicom_input=False)
        check_physical_geometry(reference, {**metadata, "slice_quaternion_wxyz":
                                           (-np.asarray(metadata["slice_quaternion_wxyz"])).tolist()},
                                acquisition, support, reference["X_m"], reference["Y_m"], dicom_input=False)

    def test_prepare_inputs_parses_reference_sequence_before_raw_reference(self):
        from test_read_acdc_sequence import make_gre_sequence
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            reference_path = make_gre_sequence(folder / "reference.seq", nx=8, ny=6, fov=(.2, .18),
                                                te=.005, tr=.010, dwell=20e-6)
            twix, _, images = self.fixture()
            twix["hdr"]["MeasYaps"]["sRXSPEC"]["alDwellTime"] = [20000]
            # The file marker only routes the explicitly mocked Twix reader.
            (folder / "reference.dat").write_bytes(b"synthetic Twix fixture")
            X, Y = np.meshgrid((np.arange(8)-4)*.025, (np.arange(6)-3)*.03)
            np.savez(folder / "b0.npz", b0Hz=np.full((6, 8), 12.), X_m=X, Y_m=Y,
                     mask=np.ones((6, 8), bool))
            np.savez(folder / "scan.npz", image=np.ones((32, 32)))
            config = {"files": {"gre": "reference.dat", "b0": "b0.npz", "scan": "scan.npz",
                                "sequence": str(SEQUENCE), "reference_sequence": str(reference_path)},
                      "preparation": {"geometry_confirmed": True, "b0_sign_confirmed": True,
                                      "b0_units": "Hz"}}
            with patch("twixtools.read_twix", return_value=[twix]):
                report = prepare_inputs(folder, folder / "prepared", config)
            experiment = loadmat(report["experiment_input"], simplify_cells=True)["experiment"]
            expected = np.sqrt(np.sum(abs(images)**2, axis=1))
            np.testing.assert_allclose(experiment["images"]["gre_reference"], expected, atol=1e-12)
            self.assertTrue(report["reference_sequence_validation"]["validated"])
            self.assertEqual(report["reference_twix_metadata"]["receiver_count"], 3)
            np.testing.assert_allclose(experiment["geometry"]["position_mm"], [90, -100, 7])


if __name__ == "__main__":
    unittest.main()
