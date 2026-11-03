"""Synthetic 3D k-space verifies cropping, B0 sign and declared world geometry."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from volume_twix import read_twix_volume


class _Block:
    def __init__(self, data, line, partition, echo, shape, geometry, reflect=False):
        self.data = np.asarray(data, dtype=np.complex64)
        self.cLin, self.cPar, self.cEco = line, partition, echo
        self.cAve = self.cSlc = self.cPhs = self.cRep = self.cSet = self.cSeg = 0
        self.mdh = SimpleNamespace(UsedChannels=len(data), SamplesInScan=shape[0],
            CenterCol=shape[0] // 2, CenterLin=shape[1] // 2, CenterPar=shape[2] // 2,
            SliceData=deepcopy(geometry))
        self._reflect = reflect

    def is_image_scan(self):
        return True

    def is_flag_set(self, flag):
        return flag == "REFLECT" and self._reflect


def _scan(field=67., receiver_scale=(1., 2.), *, reflect=False, phase_wrapped=False, support_radius=None):
    raw_shape = (16, 12, 10)
    shape = (8, 8, 8)
    te = [.002, .00446]
    xyz = np.meshgrid(*(np.arange(n) - n // 2 for n in raw_shape), indexing="ij")
    # A larger connected cuboid covers the full nominal reconstructed FOV.
    support = ((abs(xyz[0]) <= 5) & (abs(xyz[1]) <= 5) & (abs(xyz[2]) <= 4))
    if support_radius is not None:
        support = np.logical_and.reduce([abs(axis) <= support_radius for axis in xyz])
    amplitude = support.astype(float)
    amplitude[8, 6, 5] = 2.
    field_values = xyz[0] * 80. if phase_wrapped else np.full(raw_shape, field)
    quaternion = Rotation.from_euler("xyz", [13., -7., 31.], degrees=True).as_quat()[[3, 0, 1, 2]]
    geometry = SimpleNamespace(SlicePos=SimpleNamespace(Sag=11., Cor=-17., Tra=-41.),
                               Quaternion=quaternion)
    blocks = []
    coils = []
    for echo, time in enumerate(te):
        images = np.asarray([scale * amplitude * np.exp(.31j * channel)
                             * np.exp(-2j * np.pi * field_values * time)
                             for channel, scale in enumerate(receiver_scale)])
        coils.append(images)
        kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(images, axes=(1, 2, 3)),
                                           axes=(1, 2, 3)), axes=(1, 2, 3))
        for z in range(raw_shape[2]):
            for y in range(raw_shape[1]):
                reverse = reflect and y % 2 == 1
                data = kspace[:, :, y, z]
                if reverse:
                    data = data[:, ::-1]
                blocks.append(_Block(data, y, z, echo, raw_shape, geometry, reverse))
    scan = {"hdr": {"MeasYaps": {
        "sSliceArray": {"asSlice": [{"dReadoutFOV": 8., "dPhaseFOV": 8., "dThickness": 8.}]},
        "sKSpace": {"ucDimension": 4, "lBaseResolution": 8, "lImagesPerSlab": 8,
                    "dSliceOversamplingForDialog": .25}, "alTE": [2000, 4460],
        "sTXSPEC": {"asNucleusInfo": [{"lFrequency": 123221547}]}},
        "Config": {"N0FImageColumns": 8, "N0FImageLines": 8,
                   "ReadoutOversamplingFactor": 2., "phaseOversampling": .5},
        "Meas": {"flReadoutOSFactor": 2.}}, "mdb": blocks}
    crop = (slice(4, 12), slice(2, 10), slice(1, 9))
    return scan, np.asarray(coils)[(slice(None), slice(None)) + crop], field_values[crop], quaternion


class VolumeTwixTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "synthetic.dat"
        self.path.write_bytes(b"synthetic Twix metadata mocked by test")

    def tearDown(self):
        self.directory.cleanup()

    def _read(self, scan, kind="b0", **options):
        with mock.patch("twixtools.read_twix", return_value=[scan]):
            return read_twix_volume(self.path, kind, **options)

    def test_all_three_oversampling_crops_and_rss_match_encoded_object(self):
        scan, coil_images, field, _ = _scan(reflect=True)
        result = self._read(scan)
        self.assertEqual(result["magnitude"].shape, (8, 8, 8))
        np.testing.assert_allclose(result["magnitude"],
            np.sqrt(np.sum(abs(coil_images[0]) ** 2, axis=0)), rtol=3e-7, atol=1e-6)
        self.assertEqual(result["metadata"]["image_crop_start"], [4, 2, 1])
        self.assertEqual(result["metadata"]["image_crop_stop"], [12, 10, 9])
        np.testing.assert_allclose(result["metadata"]["encoded_FOV_m"], [.016, .012, .010])
        for axis in result["axes_m"]:
            np.testing.assert_allclose(axis, (np.arange(8) - 4) * .001)

    def test_b0_uses_negative_echo_product_phase_with_receive_phase_cancelled(self):
        scan, _, expected, _ = _scan(field=67.)
        result = self._read(scan)
        self.assertTrue(result["mask"].all())
        np.testing.assert_allclose(result["b0Hz"], expected, rtol=0, atol=1e-4)
        np.testing.assert_allclose(result["echo_pair_coherence"], 1., rtol=0, atol=2e-7)
        self.assertFalse(result["metadata"]["absolute_frequency_branch_known"])
        self.assertFalse(result["metadata"]["applied_ACDC_current_known"])
        self.assertAlmostEqual(result["metadata"]["frequency_alias_period_Hz"], 1 / .00246)
        self.assertEqual(result["metadata"]["frequency_Hz"], 123221547)
        self.assertEqual(result["metadata"]["TE_s"], .002)
        self.assertIsNone(result["metadata"]["sequence_signature_md5"])

    def test_3d_unwrap_crosses_alias_boundary_without_inventing_background_field(self):
        scan, _, expected, _ = _scan(phase_wrapped=True)
        result = self._read(scan)
        self.assertTrue(result["mask"].all())
        self.assertGreater(abs(expected).max(), result["metadata"]["principal_frequency_interval_Hz"][1])
        np.testing.assert_allclose(result["b0Hz"], expected, rtol=0, atol=2e-4)
        self.assertLessEqual(np.nanmax(result["wrapped_b0Hz"]), .5 / .00246)
        self.assertGreaterEqual(np.nanmin(result["wrapped_b0Hz"]), -.5 / .00246)

    def test_patient_affine_follows_prs_to_rps_contract_without_data_fitting(self):
        scan, _, _, quaternion = _scan()
        result = self._read(scan)
        prs = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        expected_basis = prs[:, [1, 0, 2]]
        np.testing.assert_allclose(result["basis_patient"], expected_basis, rtol=0, atol=5e-16)
        np.testing.assert_allclose(result["center_m"], [.011, -.017, -.041])
        for index in ([4, 4, 4], [0, 3, 7]):
            coordinate = (np.asarray(index) - 4) * .001
            expected = expected_basis @ coordinate + result["center_m"]
            actual = result["voxel_to_patient_m"] @ np.r_[index, 1.]
            np.testing.assert_allclose(actual[:3], expected, rtol=0, atol=2e-17)
        self.assertFalse(result["metadata"]["spatial_registration_fitted"])

    def test_unmeasured_background_stays_nan_and_never_connects_signal_components(self):
        scan, _, _, _ = _scan(support_radius=1.)
        result = self._read(scan)
        self.assertEqual(int(result["mask"].sum()), 27)
        self.assertTrue(np.isnan(result["b0Hz"][~result["mask"]]).all())
        self.assertTrue(np.isnan(result["wrapped_b0Hz"][~result["mask"]]).all())
        np.testing.assert_allclose(result["b0Hz"][result["mask"]], 67., rtol=0, atol=1e-4)

    def test_geometry_override_is_explicit_and_changes_no_image_or_b0_values(self):
        scan, _, _, _ = _scan()
        baseline = self._read(scan)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [43., -16., 17.], degrees=True).as_matrix()
        transform[:3, 3] = [-.02, .03, .05]
        changed = self._read(scan, patient_from_native_m=transform)
        np.testing.assert_array_equal(changed["magnitude"], baseline["magnitude"])
        np.testing.assert_array_equal(changed["b0Hz"], baseline["b0Hz"])
        np.testing.assert_array_equal(changed["native_to_patient_m"], transform)
        self.assertTrue(changed["metadata"]["geometry_override"])
        with self.assertRaises(ValueError):
            self._read(scan, patient_from_native_m=np.ones((4, 4)))

    def test_receiver_subset_and_safe_cache_preserve_native_arrays(self):
        scan, images, _, _ = _scan()
        cache = Path(self.directory.name) / "cache"
        result = self._read(scan, receiver_indices=[1], cache_dir=cache)
        np.testing.assert_allclose(result["magnitude"], abs(images[0, 1]), rtol=3e-7, atol=1e-6)
        self.assertEqual(result["metadata"]["receiver_indices"], [1])
        with mock.patch("twixtools.read_twix", side_effect=AssertionError("cache must avoid Twix parsing")):
            cached = read_twix_volume(self.path, "b0", receiver_indices=[1], cache_dir=cache)
        self.assertTrue(cached["metadata"]["cache_hit"])
        for key in ("magnitude", "b0Hz", "mask", "voxel_to_patient_m"):
            np.testing.assert_array_equal(cached[key], result[key])
        for invalid in ([], [2], [-1], [0, 0], [True], [0.5]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self._read(scan, receiver_indices=invalid)

    def test_missing_and_duplicate_cartesian_rows_are_rejected(self):
        scan, _, _, _ = _scan()
        incomplete = deepcopy(scan)
        incomplete["mdb"].pop(17)
        duplicate = deepcopy(scan)
        duplicate["mdb"][17] = duplicate["mdb"][18]
        for case in (incomplete, duplicate):
            with self.subTest(case=len(case["mdb"])), self.assertRaisesRegex(ValueError, "Missing or duplicate"):
                self._read(case)

    def test_actual_echo_count_ignores_stale_protocol_echo_times_for_reference(self):
        scan, images, _, _ = _scan()
        scan["mdb"] = [block for block in scan["mdb"] if block.cEco == 0]
        scan["hdr"]["MeasYaps"]["alTE"] = [2000, 4460, 20000, 25000]
        # A stale Config readout factor must not override actual Meas=1.
        scan["hdr"]["Config"]["N0FImageColumns"] = 16
        scan["hdr"]["MeasYaps"]["sSliceArray"]["asSlice"][0]["dReadoutFOV"] = 16.
        scan["hdr"]["Meas"]["flReadoutOSFactor"] = 1.
        result = self._read(scan, kind="reference")
        self.assertEqual(result["metadata"]["echo_times_s"], [.002])
        self.assertEqual(result["magnitude"].shape, (16, 8, 8))
        self.assertNotIn("b0Hz", result)

    def test_pulseq_md5_comes_from_sequence_variant_not_the_wip_integer(self):
        scan, _, _, _ = _scan()
        signature = "ee3c6dfcfc7c799bb60c8cbecf73bc92"
        scan["hdr"]["Meas"]["tSequenceVariant"] = signature
        scan["hdr"]["MeasYaps"]["tSequenceFileName"] = "%CustomerSeq%\\pulseq_v152p1"
        scan["hdr"]["MeasYaps"]["sWipMemBlock"] = {"alFree": [0] * 63 + [-1770430438]}
        result = self._read(scan)
        self.assertEqual(result["metadata"]["sequence_signature_md5"], signature)
        self.assertEqual(result["metadata"]["pulseq_signature_word32"], -1770430438)
        self.assertFalse(result["metadata"]["sequence_identity_verified"])


if __name__ == "__main__":
    unittest.main()
