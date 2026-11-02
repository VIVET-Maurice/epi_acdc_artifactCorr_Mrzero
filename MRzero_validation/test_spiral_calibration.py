"""Multi-receiver spiral phase calibration and untouched validation subsets."""

import unittest

import numpy as np
import torch

from spiral_baseline_calibration import fit_baseline_calibration, phase_factor


class ReorderedFourierReconstruction:
    """Exact small Fourier oracle for sample storage independent of image rows."""

    dtype = torch.float64

    def __init__(self, n, order):
        self.n, self.inverse_order = n, np.argsort(order)

    def numpy_image(self, samples):
        samples = np.asarray(samples)
        grid = samples[..., 0, self.inverse_order].reshape(samples.shape[:-2] + (self.n, self.n))
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid, axes=(-2, -1))), axes=(-2, -1))


def fixture():
    n = 32
    rng = np.random.default_rng(710)
    coordinates = (np.arange(n) - n // 2) * .2 / n
    X, Y = np.meshgrid(coordinates, coordinates)
    kx, ky = np.meshgrid((np.arange(n) - n // 2) / .2, (np.arange(n) - n // 2) / .2)
    order = rng.permutation(n * n)
    time = np.linspace(.003, .05, n * n)[None]
    acquisition = dict(sequence_kind="spiral", image_shape=(n, n), FOV_m=[.2, .2],
        kx_mInv=kx.ravel()[order][None], ky_mInv=ky.ravel()[order][None],
        time_s=time, rf_time_s=np.zeros_like(time))
    object_image = (np.exp(-((X + .022)**2 / .031**2 + (Y - .012)**2 / .047**2))
                    + .65 * np.exp(-((X - .041)**2 + (Y + .032)**2) / .013**2))
    object_image = object_image * np.exp(1j * (13 * X + 19 * Y + .6 * np.sin(Y * 100)))
    receivers = np.stack((1 + 2 * X + 1j * Y,
                         (.7 - Y) * np.exp(1j * 13 * X),
                         (.45 + X + 2 * Y) * np.exp(-1j * (9 * X + 6 * Y))))
    receiver_images = receivers * object_image[None]
    samples = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(receiver_images, axes=(-2, -1))), axes=(-2, -1))
    samples = samples.reshape(3, -1)[:, order][:, None]
    return acquisition, samples, ReorderedFourierReconstruction(n, order)


def rss(images):
    return np.sqrt(np.sum(abs(images)**2, axis=0))


class SpiralCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acquisition, cls.samples, cls.operator = fixture()
        cls.truth = dict(frequency_offset_Hz=7., readout_shift_mm=.7, phase_shift_mm=-.6)
        cls.measured = 3.1 * rss(cls.operator.numpy_image(
            cls.samples * phase_factor(cls.acquisition, cls.truth)))
        cls.result = fit_baseline_calibration(cls.acquisition, cls.samples,
            cls.measured, reconstruction=cls.operator)

    def test_recovers_three_parameters_from_complex_receivers_before_rss(self):
        result = self.result
        self.assertTrue(result["report"]["accepted"])
        for name, value in self.truth.items():
            self.assertAlmostEqual(result["parameters"][name], value, places=6)
        self.assertLess(result["report"]["candidate"]["test"]["NRMSE_percent"], 1e-6)
        self.assertAlmostEqual(result["report"]["candidate_gain_from_fit_pixels"], 3.1, places=7)
        self.assertEqual(result["baseline_image"].shape, (3, 32, 32))
        self.assertEqual(result["baseline_scaled_image"].shape, (32, 32))
        self.assertEqual(result["report"]["receiver_model"]["receiver_count"], 3)
        self.assertFalse(result["report"]["receiver_model"]["receiver_compression"])

    def test_unitary_receiver_mixing_preserves_rss_and_calibration(self):
        rng = np.random.default_rng(711)
        matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
        mixed = np.einsum("ab,bij->aij", matrix, self.samples)
        result = fit_baseline_calibration(self.acquisition, mixed, self.measured,
            reconstruction=self.operator)
        for name, value in self.truth.items():
            self.assertAlmostEqual(result["parameters"][name], value, places=6)
        np.testing.assert_allclose(result["candidate_scaled_image"], self.measured, atol=1e-9)

    def test_test_pixels_change_application_gate_but_not_fitting_or_selection(self):
        measured = self.measured.copy()
        measured[self.result["test_mask"]] = (
            3.1 * rss(self.operator.numpy_image(self.samples)))[self.result["test_mask"]]
        result = fit_baseline_calibration(self.acquisition, self.samples, measured,
            reconstruction=self.operator)
        self.assertEqual(result["candidate_parameters"], self.result["candidate_parameters"])
        self.assertEqual(result["report"]["candidate_models"], self.result["report"]["candidate_models"])
        self.assertFalse(result["report"]["accepted"])
        self.assertTrue(all(value == 0 for value in result["parameters"].values()))
        np.testing.assert_array_equal(result["calibrated_image"], result["baseline_image"])

    def test_single_receiver_retains_original_two_dimensional_images(self):
        samples = self.samples[0]
        measured = 2.3 * abs(self.operator.numpy_image(samples * phase_factor(self.acquisition, self.truth)))
        result = fit_baseline_calibration(self.acquisition, samples, measured,
            reconstruction=self.operator)
        self.assertTrue(result["report"]["accepted"])
        self.assertEqual(result["baseline_image"].shape, (32, 32))
        for name, value in self.truth.items():
            self.assertAlmostEqual(result["parameters"][name], value, places=6)

    def test_rejected_physical_translation_retains_eligible_frequency_model(self):
        calls = []

        def validate(parameters):
            calls.append(parameters.copy())
            accepted = abs(parameters["readout_shift_mm"]) + abs(parameters["phase_shift_mm"]) < 1e-9
            return {"accepted": accepted, "reason": "Frozen physical support check."}

        result = fit_baseline_calibration(self.acquisition, self.samples, self.measured,
            reconstruction=self.operator, candidate_validator=validate)
        self.assertEqual(len(calls), 3)
        self.assertTrue(result["report"]["accepted"])
        self.assertNotEqual(result["parameters"]["frequency_offset_Hz"], 0.)
        self.assertEqual(result["parameters"]["readout_shift_mm"], 0.)
        self.assertEqual(result["parameters"]["phase_shift_mm"], 0.)
        models = result["report"]["candidate_models"]
        self.assertTrue(models[0]["eligible"])
        self.assertTrue(all(not model["eligible"] for model in models[1:]))
        self.assertTrue(all("physical_rebuild_validation" in model for model in models))
        # Test data cannot influence either the validator parameters or selection.
        changed = self.measured.copy()
        changed[result["test_mask"]] *= 1.2
        calls.clear()
        other = fit_baseline_calibration(self.acquisition, self.samples, changed,
            reconstruction=self.operator, candidate_validator=validate)
        self.assertEqual(calls, [model["parameters"] for model in models])
        self.assertEqual(other["candidate_parameters"], result["candidate_parameters"])

    def test_bound_hits_are_ineligible_and_invalid_inputs_are_rejected(self):
        result = fit_baseline_calibration(self.acquisition, self.samples, self.measured,
            reconstruction=self.operator, bounds={"readout_shift_mm": .2, "phase_shift_mm": .2})
        self.assertTrue(result["report"]["candidate_models"][-1]["near_parameter_bound"])
        self.assertFalse(result["report"]["candidate_models"][-1]["eligible"])
        for invalid in (self.samples.real, self.samples[:, :, :-1], self.samples[None]):
            with self.assertRaises(ValueError):
                fit_baseline_calibration(self.acquisition, invalid, self.measured, reconstruction=self.operator)
        with self.assertRaises(ValueError):
            fit_baseline_calibration(self.acquisition, self.samples, self.measured,
                reconstruction=self.operator, candidate_validator=lambda parameters: {})


if __name__ == "__main__":
    unittest.main()
