"""Frequency sign, protocol-only I/O, navigator residuals and PE ambiguity."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from baseline_metadata_audit import audit_baseline_inputs


def fixture():
    ky = np.broadcast_to(((np.arange(4) - 2) * 5.)[:, None], (4, 6)).copy()
    readout = np.linspace(-.0003, .0003, 6)[None] * (-1.)**np.arange(4)[:, None]
    time = .02 + .000224 * ky + readout
    x = np.linspace(-1, 1, 6)
    correction = np.array([.1, .12])[:, None] + np.array([-.03, .04])[:, None] * x
    residual = np.broadcast_to([-.02, .01, -.01, .02, -.01, .01], (2, 6)).copy()
    return {"manifest": {"format": "mrzero_synthetic_acquisition"},
        "acquisition": {"time_s": time, "ky_mInv": ky, "rf_time_s": np.zeros_like(time), "FOV_m": [.2, .2]},
        "scan_diagnostics": {
            "navigator_fit": {"fit_residual_rad": np.sqrt(np.mean(residual**2, axis=1)),
                "fit_mask": np.ones((2, 6), bool), "phase_estimate_rad": correction + residual, "phase_branch": 0},
            "navigator_phase_correction_rad": correction,
            "navigator_hybrid": np.ones((3, 2, 6), complex),
            "corrected_kspace": np.ones((4, 2, 6), complex)}}


class BaselineMetadataAuditTests(unittest.TestCase):
    def test_empty_inputs_and_synthetic_acquisitions_do_not_require_twix(self):
        report = audit_baseline_inputs({})
        self.assertFalse(report["frequency_reference"]["available"])
        self.assertFalse(report["navigators"]["available"])
        self.assertFalse(report["phase_encoding_degeneracy"]["available"])
        json.dumps(report, allow_nan=False)
        inputs = fixture()
        inputs["paths"] = {"scan": "synthetic.dat", "b0": "synthetic.dat", "reference": "synthetic.dat"}
        with patch("twixtools.read_twix", side_effect=AssertionError("Synthetic data must not reopen Twix")):
            report = audit_baseline_inputs(inputs)
        self.assertFalse(report["frequency_reference"]["available"])
        json.dumps(report, allow_nan=False)

    def test_frequency_reads_only_protocol_and_reports_B0_to_EPI_sign(self):
        inputs = {"paths": {"scan": "scan.dat", "b0": "b0.dat", "reference": "reference.dat"}}
        frequencies = {"scan.dat": 123220836, "b0.dat": 123220840, "reference.dat": 123220836}
        def read(path, **options):
            frequency = frequencies[path]
            header = {"MeasYaps": {"sTXSPEC": {"asNucleusInfo": [{"lFrequency": frequency}]}},
                      "Dicom": {"lFrequency": frequency}}
            return [{"hdr": header}]
        with patch("twixtools.read_twix", side_effect=read) as reader:
            result = audit_baseline_inputs(inputs)["frequency_reference"]
        self.assertTrue(result["all_three_available"])
        self.assertFalse(result["same_declared_reference"])
        self.assertEqual(result["known_B0_to_EPI_offset_Hz"], 4.)
        self.assertEqual(result["max_reference_difference_Hz"], 4.)
        self.assertEqual(reader.call_count, 3)
        for call in reader.call_args_list:
            self.assertFalse(call.kwargs["parse_data"])
            self.assertEqual(call.kwargs["include_scans"], [-1])
            self.assertFalse(call.kwargs["parse_geometry"])

    def test_inaccessible_headers_and_archives_are_nonfatal(self):
        inputs = {"paths": {"scan": "unavailable.dat", "b0": "portable.npz"}}
        with patch("twixtools.read_twix", side_effect=PermissionError("Cannot reopen file")) as reader:
            result = audit_baseline_inputs(inputs)
        self.assertEqual(reader.call_count, 1)
        self.assertFalse(result["frequency_reference"]["available"])
        self.assertIn("Cannot reopen", result["frequency_reference"]["acquisitions"]["EPI"]["reason"])
        json.dumps(result, allow_nan=False)

    def test_reports_applied_navigator_correction_separately_from_small_residual(self):
        inputs = fixture()
        report = audit_baseline_inputs(inputs)["navigators"]
        self.assertTrue(report["available"])
        self.assertEqual(report["fitted_receiver_count"], 2)
        np.testing.assert_allclose(report["already_applied_intercept_range_rad"], [.1, .12])
        np.testing.assert_allclose(report["already_applied_slope_range_rad_per_normalized_x"], [-.03, .04])
        expected = np.sqrt(np.mean(np.array([-.02, .01, -.01, .02, -.01, .01])**2))
        self.assertAlmostEqual(report["aggregate_weighted_residual_RMS_rad"], expected)
        self.assertEqual(report["recommended_residual_odd_even_phase_bounds_rad"], [-.15, .15])
        self.assertFalse(audit_baseline_inputs(inputs)["calibration_parameters_fitted"])

    def test_timing_determines_frequency_PE_displacement_and_residual_ambiguity(self):
        result = audit_baseline_inputs(fixture())["phase_encoding_degeneracy"]
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["PE_displacement_mm_per_Hz"], .224)
        self.assertAlmostEqual(result["PE_displacement_pixels_per_Hz"], .224 / 50)
        expected_rms = np.sqrt(np.mean(np.linspace(-.0003, .0003, 6)**2))
        self.assertAlmostEqual(result["nontranslation_timing_residual_RMS_s"], expected_rms)
        self.assertAlmostEqual(result["nontranslation_phase_RMS_rad_at_10Hz"], 20 * np.pi * expected_rms)
        self.assertGreater(result["timing_ky_R_squared"], .97)

    def test_multiple_RF_histories_do_not_claim_a_single_shot_degeneracy(self):
        inputs = fixture()
        inputs["acquisition"]["rf_time_s"][1:] += .01
        result = audit_baseline_inputs(inputs)["phase_encoding_degeneracy"]
        self.assertFalse(result["available"])


if __name__ == "__main__":
    unittest.main()
