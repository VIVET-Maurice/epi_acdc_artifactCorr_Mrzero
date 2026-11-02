"""Independent spiral Fourier, RF history, turn timing and current gradients."""

from copy import deepcopy
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import torch

from continuous_reconstruction import ContinuousReconstruction, compute_spiral_density_weights
from epi_twin import NativeEPITwin
from experiment_multiecho import MultiEchoVoxelACDCModel
from test_experiment_multiecho import fixture as echo_fixture
from test_experiment_voxel import quadrature_signal
from acdc_gpu import ACDCModel


def spiral_fixture(method="direct"):
    """Unequal turn lengths and an image matrix unrelated to sample storage."""
    experiment, calibration = echo_fixture(children=2, width=85.)
    time = .0003 + np.arange(23) * .00015
    angle = np.linspace(.07, 6 * np.pi - .1, time.size)
    radius = np.linspace(.2, 21., time.size)
    starts = np.array([.0002, .00135, .0027])
    ends = np.r_[starts[1:], .004]
    state = np.searchsorted(starts, time, side="right")
    experiment["acquisition"] = {
        "sequence_kind": "spiral", "image_shape": (5, 6),
        "kx_mInv": (radius * np.cos(angle))[None],
        "ky_mInv": (radius * np.sin(angle))[None],
        "time_s": time[None], "rf_time_s": np.zeros((1, time.size)),
        "line_index": state[None], "line_start_s": starts,
        "line_end_s": ends, "phase_offset_rad": np.linspace(-.1, .1, time.size)[None],
        "TE_s": .0003, "TR_s": .006, "FOV_m": np.array([.12, .075]),
        "density_compensation_weights": np.linspace(.2, 2., time.size)[None],
        "reconstruction_method": method,
    }
    return experiment, calibration


def independent_reconstruction(samples, acq):
    """Pixel-by-pixel Fourier sum with fixed real sample weights."""
    ny, nx = acq["image_shape"]
    weights = acq["density_compensation_weights"]
    image = np.zeros(samples.shape[:-2] + (ny, nx), complex)
    for row in range(ny):
        y = (row - ny // 2) * acq["FOV_m"][1] / ny
        for col in range(nx):
            x = (col - nx // 2) * acq["FOV_m"][0] / nx
            phase = np.exp(2j * np.pi * (acq["kx_mInv"] * x + acq["ky_mInv"] * y))
            image[..., row, col] = np.sum(samples * weights * phase, axis=(-2, -1)) / weights.sum()
    return image


class SpiralReconstructionTests(unittest.TestCase):
    def test_distinct_sample_image_shapes_weighted_fourier_and_batched_adjoint(self):
        acq = spiral_fixture()[0]["acquisition"]
        operator = ContinuousReconstruction(acq, "cpu", torch.float64, sample_chunk=7)
        rng = np.random.default_rng(587)
        data = rng.normal(size=(2, 3, 1, 23)) + 1j * rng.normal(size=(2, 3, 1, 23))
        image = rng.normal(size=(2, 3, 5, 6)) + 1j * rng.normal(size=(2, 3, 5, 6))
        expected = independent_reconstruction(data, acq)
        actual = operator.numpy_image(data)
        np.testing.assert_allclose(actual, expected, rtol=4e-14, atol=4e-14)
        lhs = np.vdot(actual, image)
        rhs = np.vdot(data, operator.adjoint(image).numpy())
        np.testing.assert_allclose(lhs, rhs, rtol=4e-14, atol=4e-14)
        self.assertEqual(operator.sampling_shape, (1, 23))
        self.assertEqual(operator.image_size, (5, 6))

    @unittest.skipUnless(importlib.util.find_spec("torchkbnufft"), "optional torchkbnufft is unavailable")
    def test_nufft_agrees_with_exact_nonuniform_sum_and_has_matched_adjoint(self):
        for image_shape in ((15, 16), (16, 15)):
            with self.subTest(image_shape=image_shape):
                acq = spiral_fixture("nufft")[0]["acquisition"]
                acq["image_shape"] = image_shape
                operator = ContinuousReconstruction(acq, "cpu", torch.float64)
                rng = np.random.default_rng(588)
                samples = rng.normal(size=(2, 1, 23)) + 1j * rng.normal(size=(2, 1, 23))
                image = rng.normal(size=(2,) + image_shape) + 1j * rng.normal(size=(2,) + image_shape)
                actual = operator.numpy_image(samples)
                expected = independent_reconstruction(samples, acq)
                np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=4e-4)
                np.testing.assert_allclose(np.vdot(actual, image),
                    np.vdot(samples, operator.adjoint(image).numpy()), rtol=3e-13, atol=3e-13)

    def test_density_weights_are_fixed_and_checked(self):
        acq = spiral_fixture()[0]["acquisition"]
        for bad in (np.ones((5, 6)), -np.ones((1, 23)), np.zeros((1, 23)),
                    np.full((1, 23), np.nan), np.ones((1, 23), complex)):
            acq["density_compensation_weights"] = bad
            with self.assertRaisesRegex(ValueError, "density_compensation_weights"):
                ContinuousReconstruction(acq, "cpu", torch.float64)

    @unittest.skipUnless(importlib.util.find_spec("torchkbnufft"), "optional torchkbnufft is unavailable")
    def test_pipe_menon_weights_depend_only_on_trajectory_and_preserve_adjoint(self):
        acq = spiral_fixture()[0]["acquisition"]
        weights, report = compute_spiral_density_weights(acq, iterations=3)
        self.assertEqual(weights.shape, (1, 23))
        self.assertTrue(np.isfinite(weights).all() and np.all(weights > 0))
        self.assertAlmostEqual(float(weights.mean()), 1.)
        self.assertFalse(report["depends_on_signal_or_currents"])
        repeated, _ = compute_spiral_density_weights(acq, iterations=3)
        np.testing.assert_array_equal(weights, repeated)
        acq["density_compensation_weights"] = weights
        operator = ContinuousReconstruction(acq, "cpu", torch.float64)
        rng = np.random.default_rng(593)
        samples = rng.normal(size=(1, 23)) + 1j * rng.normal(size=(1, 23))
        np.testing.assert_allclose(operator.numpy_image(samples),
            independent_reconstruction(samples, acq), rtol=3e-14, atol=3e-14)

    def test_native_baseline_twin_retains_receivers_and_distinct_sample_matrix(self):
        acq = spiral_fixture()[0]["acquisition"]
        x = np.array([[-.013, .021]])
        y = np.array([[.011, -.007]])
        amplitude = np.array([[.7, 1.1]])
        b0 = np.array([[41., -77.]])
        rate = np.array([[9., 13.]])
        anchor = .0007
        receive = np.array([[[1., .6]], [[0., .8]]], complex)
        twin = NativeEPITwin(acq, x, y, amplitude, b0, np.ones_like(x, bool),
            rate_sInv=rate, amplitude_reference_time_s=anchor,
            receive_weights=receive, sample_chunk=7)
        result = twin.simulate()
        expected = np.zeros((2, 1, 23), complex)
        for point in range(2):
            phase = (acq["kx_mInv"] * x[0, point] + acq["ky_mInv"] * y[0, point]
                     + b0[0, point] * acq["time_s"])
            signal = amplitude[0, point] * np.exp(-(acq["time_s"] - anchor) * rate[0, point])
            signal = signal * np.exp(-2j * np.pi * phase + 1j * acq["phase_offset_rad"])
            expected += receive[:, 0, point, None, None] * signal[None]
        self.assertEqual(result["kdata"].shape, (2, 1, 23))
        self.assertEqual(result["coil_images"].shape, (2, 5, 6))
        self.assertEqual(result["image"].shape, (5, 6))
        np.testing.assert_allclose(result["kdata"], expected, rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(result["coil_images"], independent_reconstruction(expected, acq),
            rtol=5e-5, atol=5e-6)


class SpiralModelTests(unittest.TestCase):
    def model(self, experiment, calibration, **options):
        return MultiEchoVoxelACDCModel(experiment, calibration, profile="cpu",
            precision="float64", sample_chunk=7, **options)

    def test_native_spiral_signal_matches_displaced_spin_quadrature(self):
        experiment, calibration = spiral_fixture()
        currents = np.random.default_rng(589).normal(size=(32, 3)) * .03
        with patch.object(ACDCModel, "_make_base", side_effect=AssertionError("native MRzero required")):
            model = self.model(experiment, calibration)
            self.assertEqual(model.nstates, 3)
            self.assertEqual(model.image_size, (5, 6))
            self.assertEqual(model.sampling_shape, (1, 23))
            elapsed = experiment["acquisition"]["time_s"]
            for response in (True, False):
                for b0 in (True, False):
                    image, signal = model.forward(currents, response_enabled=response, b0_enabled=b0)
                    expected = quadrature_signal(experiment, calibration, currents,
                        response_enabled=response, b0_enabled=b0) * np.sinc(85. * elapsed)
                    np.testing.assert_allclose(signal, expected, rtol=6e-5, atol=7e-6)
                    np.testing.assert_allclose(image, independent_reconstruction(expected,
                        experiment["acquisition"]), rtol=6e-5, atol=5e-6)

    def test_current_gradient_includes_turn_response_finite_voxel_and_density_weights(self):
        for method in ("direct", "nufft"):
            if method == "nufft" and not importlib.util.find_spec("torchkbnufft"):
                continue
            experiment, calibration = spiral_fixture(method)
            model = self.model(experiment, calibration)
            rng = np.random.default_rng(590)
            currents = rng.normal(size=(32, 3)) * .03
            direction = rng.normal(size=currents.shape)
            direction /= np.linalg.norm(direction)
            value, gradient = model.objective_and_grad(currents)
            epsilon = 1e-5
            difference = (model.objective_and_grad(currents + epsilon * direction)[0]
                          - model.objective_and_grad(currents - epsilon * direction)[0]) / (2 * epsilon)
            self.assertTrue(np.isfinite(value))
            np.testing.assert_allclose(np.sum(gradient * direction), difference, rtol=5e-6, atol=2e-9)

    def test_sample_storage_permutation_preserves_images_currents_and_turn_timing(self):
        experiment, calibration = spiral_fixture()
        original = self.model(experiment, calibration)
        shuffled = deepcopy(experiment)
        permutation = np.random.default_rng(591).permutation(23)
        for name, value in list(shuffled["acquisition"].items()):
            if isinstance(value, np.ndarray) and value.shape == (1, 23):
                shuffled["acquisition"][name] = value[:, permutation]
        permuted = self.model(shuffled, calibration)
        currents = np.random.default_rng(592).normal(size=(32, 3)) * .02
        image, signal = original.forward(currents)
        other_image, other_signal = permuted.forward(currents)
        np.testing.assert_allclose(other_signal, signal[:, permutation], rtol=8e-5, atol=1e-5)
        np.testing.assert_allclose(other_image, image, rtol=8e-5, atol=1e-5)
        np.testing.assert_allclose(permuted.H_response.numpy(),
            original.H_response.numpy()[:, permutation], rtol=0, atol=0)
        a, ga = original.objective_and_grad(currents)
        b, gb = permuted.objective_and_grad(currents)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(ga, gb, rtol=2e-4, atol=3e-7)

    def test_disabled_physical_commands_remain_zero_in_spiral_response(self):
        experiment, calibration = spiral_fixture()
        limits = np.full(32, 3.)
        limits[8] = 0.
        model = self.model(experiment, calibration, max_current_A=limits)
        commands = model.project_currents(np.ones((32, 3)), 10., 500.)
        np.testing.assert_array_equal(commands[8], 0.)
        np.testing.assert_array_equal(model.sample_waveform(commands)["actual_A"][8], 0.)
        self.assertGreater(np.max(np.abs(commands)), 0.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
