"""Known-field checks for phase/sign, coil combination, masking and Twix layout."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from read_acdc_b0 import (b0_from_coil_images, read_b0_coil_images,
                           read_b0_sequence, read_b0_twix)


ROOT = Path(__file__).resolve().parent.parent
TE = np.asarray([.002, .004, .006])


def coil_images(field, magnitude=None, echo_times=TE):
    """Independent signal oracle with deliberately cancelling receiver phases."""
    if magnitude is None:
        magnitude = np.ones_like(field)
    sensitivity = np.asarray([1, 2j, -.7, -.4j])[:, None, None]
    return (sensitivity[None] * magnitude[None, None]
            * np.exp(-2j * np.pi * np.asarray(echo_times)[:, None, None, None] * field[None, None]))


def fake_twix(images, echo_times=TE):
    ne, channels, ny, nx = images.shape
    axes = (2, 3)
    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(images, axes=axes), axes=axes), axes=axes)
    geometry = SimpleNamespace(SlicePos=SimpleNamespace(Sag=0, Cor=0, Tra=0),
                               Quaternion=[2 ** -.5, 0, 0, 2 ** -.5])
    lines = []
    for row in range(ny):
        for echo in range(ne):
            lines.append(SimpleNamespace(
                data=k[echo, :, row], cLin=row, cEco=echo,
                cAve=0, cSlc=0, cPar=0, cPhs=0, cRep=0, cSet=0,
                is_image_scan=lambda: True, is_flag_set=lambda flag: False,
                mdh=SimpleNamespace(CenterCol=nx // 2, CenterLin=ny // 2, SliceData=geometry)))
    header = {"sKSpace": {"lBaseResolution": nx, "lPhaseEncodingLines": ny},
              "alTE": (np.asarray(echo_times) * 1e6).tolist(), "alTR": [10000], "sRXSPEC": {"alDwellTime": [10000]},
              "sSliceArray": {"asSlice": [{"dReadoutFOV": 200, "dPhaseFOV": 160, "dThickness": 5}]}}
    return [{"mdb": lines, "hdr": {"MeasYaps": header}}]


class B0PhaseTests(unittest.TestCase):
    def test_known_field_and_receiver_phase_cancellation(self):
        field = np.linspace(-100, 120, 32)[None] + np.linspace(-20, 20, 24)[:, None]
        result = b0_from_coil_images(coil_images(field), TE)
        np.testing.assert_allclose(result["b0Hz"], field, atol=1e-10)
        self.assertTrue(result["mask"].all())
        self.assertLess(result["phase_residual_rad"].max(), 1e-12)

    def test_explicit_sign_and_frequency_offset(self):
        field = np.full((8, 10), 63.)
        result = b0_from_coil_images(coil_images(field), TE, phase_sign=1, frequency_offset_hz=500)
        np.testing.assert_allclose(result["b0Hz"], -field + 500, atol=1e-10)

    def test_spatial_unwrap_keeps_smooth_field_beyond_principal_band(self):
        field = np.broadcast_to(np.linspace(-420, 420, 64), (32, 64))
        # The reliable central anchor is in the principal band; edges cross it.
        magnitude = np.broadcast_to(1 + 2 * np.exp(-np.linspace(-3, 3, 64) ** 2), field.shape)
        result = b0_from_coil_images(coil_images(field, magnitude), TE)
        np.testing.assert_allclose(result["b0Hz"], field, atol=1e-9)
        self.assertEqual(result["metadata"]["principal_frequency_band_hz"], [-250, 250])

    def test_low_signal_and_inconsistent_echoes_remain_nan(self):
        field = np.full((12, 16), 30.)
        magnitude = np.ones_like(field)
        magnitude[:2] = 0
        images = coil_images(field, magnitude)
        images[1, :, 5:7, 6:8] *= np.exp(1j)
        result = b0_from_coil_images(images, TE)
        self.assertFalse(result["mask"][:2].any())
        self.assertFalse(result["mask"][5:7, 6:8].any())
        self.assertTrue(np.isnan(result["b0Hz"][~result["mask"]]).all())
        np.testing.assert_allclose(result["b0Hz"][result["mask"]], 30, atol=1e-10)

    def test_diagnostic_products_explain_masks_without_changing_field(self):
        field = np.full((12, 16), 30.)
        magnitude = np.ones_like(field)
        magnitude[:2] = 0
        images = coil_images(field, magnitude)
        images[1, :, 5:7, 6:8] *= np.exp(1j)
        default = b0_from_coil_images(images, TE)
        result = b0_from_coil_images(images, TE, return_intermediates=True)
        np.testing.assert_array_equal(result["b0Hz"], default["b0Hz"])
        np.testing.assert_array_equal(result["mask"], default["mask"])
        intermediate = result["intermediates"]
        self.assertEqual(intermediate["echo_magnitude"].shape, (3, 12, 16))
        self.assertEqual(intermediate["echo_cross_products"].shape, (2, 12, 16))
        # For a known 30 Hz field the wrapped phase is -2*pi*30*DeltaTE.
        np.testing.assert_allclose(intermediate["wrapped_phase_increment_rad"][8:, :],
                                   -2*np.pi*30*.002, atol=1e-12)
        self.assertFalse(intermediate["magnitude_mask"][:2].any())
        self.assertFalse(intermediate["phase_residual_mask"][5:7, 6:8].any())
        self.assertTrue((intermediate["component_labels"] > 0)[result["mask"]].all())

    def test_small_disconnected_components_excluded(self):
        magnitude = np.zeros((16, 16))
        magnitude[2:12, 2:12] = 1
        magnitude[14, 14] = 2
        result = b0_from_coil_images(coil_images(np.zeros_like(magnitude), magnitude), TE)
        self.assertEqual(result["mask"].sum(), 100)
        self.assertFalse(result["mask"][14, 14])

    def test_unequal_echo_spacing_and_no_signal_rejected(self):
        images = coil_images(np.zeros((8, 8)))
        with self.assertRaisesRegex(ValueError, "equally spaced"):
            b0_from_coil_images(images, [.002, .004, .007])
        with self.assertRaisesRegex(ValueError, "No reliable"):
            b0_from_coil_images(images * 0, TE)


class B0SequenceGridTests(unittest.TestCase):
    def sequence_fixture(self, echo_drift_samples=.013):
        ny, nx, ne = 4, 8, 32
        dx, dy, dwell = 5., 5., 7.8e-6
        centers = .001 + np.arange(ny) * .065
        rf = [{"center": center, "end": center + .0002, "phase": 0.} for center in centers]
        adc = [{"samples": nx, "dwell": dwell, "phase": 0.} for _ in range(ny * ne)]
        echo_times = .0021231 + np.arange(ne) * .00182
        offsets = np.linspace(0, echo_drift_samples, ne) * dx
        trajectory = np.zeros((3, ny, ne, nx))
        trajectory[0] = (np.arange(nx) - nx // 2) * dx + offsets[:, None]
        trajectory[1] = ((np.arange(ny) - ny // 2) * dy)[:, None, None]
        times = (centers[:, None, None] + echo_times[None, :, None]
                 + (np.arange(nx) - nx // 2)[None, None, :] * dwell)
        sequence = SimpleNamespace(
            read=lambda path: None, definitions={"Matrix": [nx, ny], "TE": echo_times},
            calculate_kspace=lambda: (trajectory.reshape(3, -1), None, centers, [], times.reshape(-1)))
        return sequence, rf, adc, trajectory

    def read_fixture(self, sequence, rf, adc):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.seq"
            path.write_text("synthetic ADC fixture", encoding="utf-8")
            with patch("pypulseq.Sequence", return_value=sequence):
                with patch("read_acdc_b0._events", return_value=(rf, adc, .26)):
                    return read_b0_sequence(path)

    def test_long_echo_train_accepts_small_constant_echo_shifts_without_rounding_te(self):
        sequence, rf, adc, _ = self.sequence_fixture()
        result = self.read_fixture(sequence, rf, adc)
        self.assertEqual(result["echo_count"], 32)
        # An independent zero-crossing calculation: positive k offset crosses
        # the origin earlier by offset / step * dwell.
        expected_te = .0021231 + np.arange(32) * .00182 - np.linspace(0, .013, 32) * 7.8e-6
        np.testing.assert_allclose(result["TE_s"], expected_te, atol=1e-14)

    def test_long_train_does_not_allow_large_offsets_or_nonuniform_sampling(self):
        with self.subTest(defect="large echo offset"):
            sequence, rf, adc, _ = self.sequence_fixture(echo_drift_samples=.05)
            with self.assertRaisesRegex(ValueError, "uniform Cartesian"):
                self.read_fixture(sequence, rf, adc)
        with self.subTest(defect="nonuniform readout"):
            sequence, rf, adc, trajectory = self.sequence_fixture()
            trajectory[0, :, :, 1] += .05
            with self.assertRaisesRegex(ValueError, "uniform Cartesian"):
                self.read_fixture(sequence, rf, adc)


class B0TwixTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "synthetic.dat"
        self.path.write_bytes(b"Synthetic fixture, not scanner data.")
        self.field = np.linspace(-70, 90, 16)[None] + np.linspace(-10, 10, 12)[:, None]
        self.scans = fake_twix(coil_images(self.field))

    def read(self):
        with patch("twixtools.read_twix", return_value=self.scans):
            return read_b0_twix(self.path)

    def test_full_reconstruction_and_physical_grid(self):
        result = self.read()
        np.testing.assert_allclose(result["b0Hz"], self.field, atol=1e-10)
        np.testing.assert_allclose(result["X_m"][0], (np.arange(16) - 8) * .2 / 16)
        np.testing.assert_allclose(result["Y_m"][:, 0], (np.arange(12) - 6) * .16 / 12)
        self.assertEqual(result["metadata"]["echo_count"], 3)

    def test_echo_stage_and_diagnostic_wrapper_match_independent_signal_oracle(self):
        with patch("twixtools.read_twix", return_value=self.scans):
            echoes = read_b0_coil_images(self.path)
            result = read_b0_twix(self.path, return_intermediates=True)
        expected = coil_images(self.field)
        np.testing.assert_allclose(echoes["coil_images"], expected, atol=1e-12)
        np.testing.assert_array_equal(result["intermediates"]["kspace"], echoes["kspace"])
        np.testing.assert_allclose(result["intermediates"]["coil_images"], expected, atol=1e-12)
        np.testing.assert_allclose(result["b0Hz"], self.field, atol=1e-10)
        self.assertEqual(echoes["metadata"]["TE_s"], TE.tolist())

    def test_echo_stage_remains_inspectable_when_field_estimation_fails(self):
        self.scans = fake_twix(coil_images(self.field) * 0)
        with patch("twixtools.read_twix", return_value=self.scans):
            echoes = read_b0_coil_images(self.path)
            with self.assertRaisesRegex(ValueError, "No reliable"):
                read_b0_twix(self.path)
        self.assertEqual(echoes["coil_images"].shape, (3, 4, 12, 16))
        np.testing.assert_array_equal(echoes["coil_images"], 0)

    def test_duplicate_or_missing_echo_line_rejected(self):
        self.scans[0]["mdb"][-1] = self.scans[0]["mdb"][0]
        with self.assertRaisesRegex(ValueError, "unique complete"):
            self.read()

    def test_multiple_slices_rejected(self):
        self.scans[0]["mdb"][-1].cSlc = 1
        with self.assertRaisesRegex(ValueError, "one slice"):
            self.read()

    def test_offset_preserved_and_unsupported_rotation_rejected(self):
        geometry = self.scans[0]["mdb"][0].mdh.SliceData
        geometry.SlicePos.Sag, geometry.SlicePos.Cor = 12, -7
        result = self.read()
        self.assertAlmostEqual(result["X_m"][6, 8], -.007)
        self.assertAlmostEqual(result["Y_m"][6, 8], -.012)
        geometry.Quaternion = [1, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "axial"):
            self.read()

    def test_bipolar_echoes_rejected(self):
        self.scans[0]["mdb"][1].is_flag_set = lambda flag: flag == "REFLECT"
        with self.assertRaisesRegex(ValueError, "monopolar"):
            self.read()

    def test_sequence_and_twix_disagreement_rejected(self):
        info = {"image_shape": [12, 16], "echo_count": 3, "polarity": 1, "TE_s": [.002, .004, .008]}
        with patch("read_acdc_b0.read_b0_sequence", return_value=info):
            with patch("twixtools.read_twix", return_value=self.scans):
                with self.assertRaisesRegex(ValueError, "TE_s disagrees"):
                    read_b0_twix(self.path, Path("synthetic.seq"))

    def test_three_echo_twix_is_not_relabelled_as_thirty_two_echoes(self):
        info = {"image_shape": [12, 16], "echo_count": 32}
        with patch("read_acdc_b0.read_b0_sequence", return_value=info):
            with patch("twixtools.read_twix", return_value=self.scans):
                with self.assertRaisesRegex(ValueError, "echo_count.*expected 32, found 3"):
                    read_b0_coil_images(self.path, Path("32echo.seq"))

    def test_thirty_two_echo_twix_preserves_echo_order_and_known_field(self):
        echo_times = .0021231 + np.arange(32) * .00182
        expected = coil_images(self.field, echo_times=echo_times)
        scans = fake_twix(expected, echo_times=echo_times)
        with patch("twixtools.read_twix", return_value=scans):
            echoes = read_b0_coil_images(self.path)
        self.assertEqual(echoes["coil_images"].shape, (32, 4, 12, 16))
        np.testing.assert_allclose(echoes["coil_images"], expected, atol=1e-12)
        result = b0_from_coil_images(echoes["coil_images"], echoes["metadata"]["TE_s"])
        np.testing.assert_allclose(result["b0Hz"], self.field, atol=1e-10)

    def test_supplied_real_b0_sequence_uses_actual_adc_not_stale_echo_samples(self):
        candidates = [ROOT / "Experiment/sequence/megre_b0_128x128_fov200mm_3echo.seq",
                      ROOT / "acquisition_019/sequences/b0_map.seq"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            self.skipTest("Local acquisition is not installed")
        result = read_b0_sequence(path)
        self.assertEqual(result["matrix_size"], [128, 128])
        self.assertEqual(result["echo_count"], 3)
        self.assertEqual(result["dummy_rf_count"], 32)
        np.testing.assert_allclose(result["TE_s"], [.0021231, .0039431, .0057631], atol=1e-8)
        np.testing.assert_allclose(result["FOV_m"], [.2, .2], atol=1e-6)

    def test_supplied_thirty_two_echo_sequence_retains_actual_timing_and_grid_drift(self):
        path = ROOT / "acquisition_019/sequences/megre_b0_s0_t2star_128x128_fov200mm_32echo.seq"
        if not path.exists():
            self.skipTest("Local 32-echo sequence is not installed")
        result = read_b0_sequence(path)
        self.assertEqual(result["matrix_size"], [128, 128])
        self.assertEqual(result["echo_count"], 32)
        self.assertEqual(result["dummy_rf_count"], 32)
        self.assertEqual(result["polarity"], 1)
        self.assertAlmostEqual(result["dwell_s"], 7.8e-6)
        self.assertAlmostEqual(result["TR_s"], .06197)
        self.assertAlmostEqual(result["sequence_duration_s"], 9.9152)
        # Zero-k crossing differs slightly from the rounded TE definitions.
        np.testing.assert_allclose(result["TE_s"], .0021231 + np.arange(32) * .00182, atol=1.6e-7)
        np.testing.assert_allclose(result["FOV_m"], [.2, .2], atol=1e-6)
        self.assertGreater(result["echo_readout_center_error_samples"][-1], .012)
        self.assertLess(result["echo_readout_center_error_samples"][-1], .014)


if __name__ == "__main__":
    unittest.main()
