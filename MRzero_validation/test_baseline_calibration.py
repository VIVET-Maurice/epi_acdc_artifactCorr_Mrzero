"""Known-parameter recovery, untouched test data and physical sample phases."""

import unittest

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from baseline_calibration import fit_baseline_calibration, phase_factor, spatial_split, _jacobian_information
from continuous_reconstruction import ContinuousReconstruction
from inplane_geometry import FrameReconstruction, inplane_matrix, rotate_acquisition_k


def fixture():
    n = 48
    x = (np.arange(n) - n // 2) * .2 / n
    X, Y = np.meshgrid(x, x)
    kx, ky = np.meshgrid((np.arange(n) - n // 2) / .2, (np.arange(n) - n // 2) / .2)
    polarity = (-1.) ** np.arange(n)
    time = .015 + np.arange(n)[:, None] * .001 + polarity[:, None] * (np.arange(n) - n // 2) * 5e-6
    acquisition = dict(kx_mInv=kx, ky_mInv=ky, time_s=time, rf_time_s=np.zeros_like(time), FOV_m=[.2, .2])
    obj = (np.exp(-((X + .02)**2 / .03**2 + (Y - .01)**2 / .05**2))
           + .6 * np.exp(-((X - .038)**2 + (Y + .035)**2) / .012**2))
    obj *= .8 + .2 * np.cos(X * 400 + Y * 150)
    obj = obj * np.exp(1j * (12 * X + 17 * Y + .8 * np.sin(Y * 100)))
    kdata = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj)))
    operator = ContinuousReconstruction(acquisition, "cpu", torch.float64)
    return acquisition, kdata, operator


class BaselineCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acquisition, cls.kdata, cls.operator = fixture()
        cls.truth = dict(frequency_offset_Hz=8., readout_shift_mm=.8, odd_even_phase_rad=.12)
        cls.measured = 4.7 * abs(cls.operator.numpy_image(cls.kdata * phase_factor(cls.acquisition, cls.truth)))
        cls.result = fit_baseline_calibration(cls.acquisition, cls.kdata, cls.measured)

    def test_recovers_injected_physical_parameters(self):
        self.assertTrue(self.result["report"]["accepted"])
        for name, value in self.truth.items():
            self.assertAlmostEqual(self.result["parameters"][name], value, places=5)
        self.assertLess(self.result["report"]["candidate"]["test"]["NRMSE_percent"], 1e-5)
        self.assertAlmostEqual(self.result["report"]["candidate_gain_from_fit_pixels"], 4.7, places=6)
        self.assertFalse(self.result["report"]["gain_applied_to_object"])

    def test_test_pixels_do_not_change_fit_gain_or_model_selection(self):
        corrupted = self.measured.copy()
        mask = self.result["test_mask"]
        # Make original predictions exactly correct ONLY on the test subset.
        corrupted[mask] = 4.7 * abs(self.operator.numpy_image(self.kdata))[mask]
        other = fit_baseline_calibration(self.acquisition, self.kdata, corrupted)
        self.assertEqual(self.result["candidate_parameters"], other["candidate_parameters"])
        self.assertEqual(self.result["report"]["candidate_models"], other["report"]["candidate_models"])
        self.assertFalse(other["report"]["accepted"])
        self.assertTrue(all(v == 0 for v in other["parameters"].values()))
        np.testing.assert_array_equal(other["calibrated_image"], other["baseline_image"])

    def test_identity_data_keeps_original_model(self):
        measured = 2.3 * abs(self.operator.numpy_image(self.kdata))
        result = fit_baseline_calibration(self.acquisition, self.kdata, measured)
        self.assertFalse(result["report"]["accepted"])
        self.assertTrue(all(v == 0 for v in result["parameters"].values()))
        np.testing.assert_allclose(result["baseline_scaled_image"], measured, atol=1e-12)

    def test_sample_phase_matches_direct_point_translation_and_frequency(self):
        a = self.acquisition
        x, y, frequency = .012, -.017, 19.
        t = a["time_s"] - a["rf_time_s"]
        p = dict(frequency_offset_Hz=7., readout_shift_mm=-.3, odd_even_phase_rad=.1)
        original = np.exp(-2j * np.pi * (a["kx_mInv"] * x + a["ky_mInv"] * y + frequency * t))
        polarity = (-1.) ** np.arange(t.shape[0])[:, None]
        direct = np.exp(-2j * np.pi * (a["kx_mInv"] * (x - .0003) + a["ky_mInv"] * y + (frequency + 7.) * t)
                        - .05j * polarity)
        np.testing.assert_allclose(original * phase_factor(a, p), direct, atol=2e-14)

    def test_spatial_guards_and_disjoint_subsets(self):
        masks = spatial_split((128, 128))
        self.assertTrue(all(mask.any() for mask in masks))
        for i, mask in enumerate(masks):
            for other in masks[i+1:]:
                self.assertFalse(np.any(binary_dilation(mask, iterations=2) & other))
        self.assertLess(sum(mask.sum() for mask in masks), 128**2)

    def test_parameters_stay_in_native_readout_frame(self):
        matrix = inplane_matrix(180.)
        framed = rotate_acquisition_k(self.acquisition, matrix=matrix)
        operator = FrameReconstruction(framed, "cpu", torch.float64, frame_matrix=matrix)
        result = fit_baseline_calibration(self.acquisition, self.kdata, self.measured, reconstruction=operator)
        for name, value in self.result["parameters"].items():
            self.assertAlmostEqual(result["parameters"][name], value, places=6)
        # A framed negative kx would invert the meaning of the readout shift.
        with self.assertRaises(ValueError):
            phase_factor(framed, self.truth)

    def test_zero_or_deficient_jacobian_is_not_identifiable(self):
        for jacobian in (np.zeros((20, 3)), np.ones((20, 3)), np.eye(3) * 1e-15):
            info = _jacobian_information(jacobian, [30., 2., .15])
            self.assertFalse(info["Jacobian_identifiable"])
            self.assertLess(info["scaled_Jacobian_rank"], 3)
        self.assertTrue(_jacobian_information(np.eye(3), [30., 2., .15])["Jacobian_identifiable"])

    def test_rejects_non_epi_and_invalid_parameters(self):
        single_polarity = dict(self.acquisition, time_s=np.sort(self.acquisition["time_s"], axis=1))
        with self.assertRaises(ValueError):
            phase_factor(single_polarity, {})
        for parameters in ({"frequency_offset_Hz": float("nan")}, {"phase_encode_shift_mm": 1.}, {"readout_shift_mm": True}):
            with self.assertRaises(ValueError):
                phase_factor(self.acquisition, parameters)
        with self.assertRaises(ValueError):
            fit_baseline_calibration(self.acquisition, self.kdata, self.measured, bounds={"frequency_offset_Hz": -1})
        with self.assertRaises(ValueError):
            fit_baseline_calibration(self.acquisition, self.kdata, self.measured, guard_pixels=4)


if __name__ == "__main__":
    unittest.main()
