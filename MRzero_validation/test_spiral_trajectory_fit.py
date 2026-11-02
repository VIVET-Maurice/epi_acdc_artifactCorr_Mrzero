"""Independent derivatives and held-out acceptance for spiral delay fitting."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from baseline_calibration import spatial_split
from continuous_reconstruction import ContinuousReconstruction
from spiral_encoding import delayed_spiral_trajectory
from spiral_trajectory_fit import _signal_and_derivatives, fit_spiral_gradient_delays


class CachedPointModel:
    """Small analytical cache independent of the calibration implementation."""
    def __init__(self):
        self.dtype, self.device = torch.float64, torch.device("cpu")
        self.nsamples, self.sample_chunk, self.sampling_shape = 360, 47, (1, 360)
        rng = np.random.default_rng(8001)
        self._x = self.tensor(rng.uniform(-.07, .07, 37))[:, None]
        self._y = self.tensor(rng.uniform(-.07, .07, 37))[:, None]
        time = .002 + np.arange(self.nsamples) * 5e-6
        angle, radius = np.linspace(.02, 15 * np.pi, self.nsamples), np.linspace(.2, 75., self.nsamples)
        self.acquisition = dict(sequence_kind="spiral", kx_mInv=(radius * np.cos(angle))[None],
            ky_mInv=(radius * np.sin(angle))[None], time_s=time[None],
            rf_time_s=np.zeros((1, self.nsamples)), image_shape=(24, 24), FOV_m=np.array([.2, .2]))
        self._kx = self.tensor(self.acquisition["kx_mInv"].ravel())
        self._ky = self.tensor(self.acquisition["ky_mInv"].ravel())
        self._evolution = self.tensor(time)
        self._voxel_edges = self.tensor([[.0012, .0001], [-.00005, .0014]])
        self._voxel_b0_gradient = self.tensor(rng.uniform(-30., 30., (37, 2)))
        amplitude = self.tensor(rng.uniform(.2, 1., 37))[:, None]
        field = self.tensor(rng.uniform(-150., 150., 37))[:, None]
        phase = -2 * torch.pi * (self._x * self._kx + self._y * self._ky
                                 + field * self._evolution)
        self._base_cache = amplitude * torch.exp(-15 * self._evolution) * torch.exp(1j * phase)
        weights = rng.normal(size=(3, 37)) + 1j * rng.normal(size=(3, 37))
        weights /= np.sqrt(np.sum(abs(weights)**2, axis=0))[None]
        self._receive_weights = torch.as_tensor(weights)

    @staticmethod
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    @staticmethod
    def numpy(value):
        return value.detach().cpu().numpy()

    def _ensure_voxel_tensors(self):
        pass


class SpiralDelayFitTests(unittest.TestCase):
    def test_cached_signal_matches_native_mrzero_receiver_forward(self):
        from experiment_spiral import SpiralReceiverACDCModel
        from test_spiral_model import spiral_fixture
        experiment, calibration = spiral_fixture()
        shape = np.asarray(experiment["m0_proxy"]).shape
        rng = np.random.default_rng(8002)
        experiment["receive_weights"] = (rng.normal(size=(3,) + shape)
                                         + 1j * rng.normal(size=(3,) + shape))
        model = SpiralReceiverACDCModel(experiment, calibration,
            profile="cpu", precision="float64", sample_chunk=5)
        k0 = np.stack((model.numpy(model._kx), model.numpy(model._ky)))
        actual, _ = _signal_and_derivatives(model, k0)
        expected = model.receiver_forward(np.zeros((model.nc, model.nstates)))[1]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_exact_voxel_signal_and_delay_derivatives_match_independent_difference(self):
        model = CachedPointModel()
        k0 = np.stack((model.numpy(model._kx), model.numpy(model._ky)))
        step = .00001e-6
        derivative_k = []
        trajectories = []
        for axis in range(2):
            delay = np.zeros(2)
            delay[axis] = step
            plus = np.stack(delayed_spiral_trajectory(model.acquisition, *delay)).reshape(2, -1)
            minus = np.stack(delayed_spiral_trajectory(model.acquisition, *(-delay))).reshape(2, -1)
            trajectories.append((plus, minus))
            derivative_k.append((plus - minus) / (2 * step * 1e6))
        base, derivatives = _signal_and_derivatives(model, k0, np.asarray(derivative_k))
        q = [(model._voxel_edges[a, 0] * model._kx
             + model._voxel_edges[a, 1] * model._ky)[None]
             + model._voxel_b0_gradient[:, a, None] * model._evolution[None]
             for a in range(2)]
        expected = model._receive_weights @ (model._base_cache * torch.sinc(q[0]) * torch.sinc(q[1]))
        np.testing.assert_allclose(base[:, 0], model.numpy(expected), rtol=1e-13, atol=1e-13)
        for axis, (plus, minus) in enumerate(trajectories):
            positive = _signal_and_derivatives(model, plus)[0]
            negative = _signal_and_derivatives(model, minus)[0]
            finite_difference = (positive - negative) / (2 * step * 1e6)
            relative_error = np.linalg.norm(derivatives[axis] - finite_difference) / np.linalg.norm(finite_difference)
            self.assertLess(relative_error, 2e-6)

    def test_exact_candidate_improves_reserved_subsets_without_mutating_cache(self):
        model = CachedPointModel()
        original_cache = model._base_cache.clone()
        original_acquisition = deepcopy(model.acquisition)
        delayed = np.stack(delayed_spiral_trajectory(model.acquisition, 3e-6, -2e-6)).reshape(2, -1)
        samples, _ = _signal_and_derivatives(model, delayed)
        operator = ContinuousReconstruction(model.acquisition, "cpu", torch.float64)
        coils = operator.numpy_image(samples)
        measured = 2.3 * np.sqrt(np.sum(abs(coils)**2, axis=0))
        result = fit_spiral_gradient_delays(model, model.acquisition, None, measured, np.eye(2))
        self.assertTrue(result["accepted"])
        for subset in ("fit", "validation", "test"):
            self.assertLess(result["report"]["candidate"]["metrics"][subset]["NRMSE_percent"],
                            result["report"]["baseline"]["metrics"][subset]["NRMSE_percent"])
        self.assertFalse(result["report"]["bound_hit"])
        torch.testing.assert_close(original_cache, model._base_cache)
        np.testing.assert_array_equal(original_acquisition["kx_mInv"], model.acquisition["kx_mInv"])

    def test_identity_data_keep_zero_delay_and_bad_bounds_fail(self):
        model = CachedPointModel()
        k0 = np.stack((model.numpy(model._kx), model.numpy(model._ky)))
        samples, _ = _signal_and_derivatives(model, k0)
        coils = ContinuousReconstruction(model.acquisition, "cpu", torch.float64).numpy_image(samples)
        measured = np.sqrt(np.sum(abs(coils)**2, axis=0))
        result = fit_spiral_gradient_delays(model, model.acquisition, None, measured, np.eye(2))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["parameters"], dict(delay_x_s=0., delay_y_s=0.))
        for invalid in (0., -1., np.nan, True):
            with self.subTest(bound=invalid), self.assertRaises(ValueError):
                fit_spiral_gradient_delays(model, model.acquisition, None, measured,
                                           np.eye(2), max_delay_s=invalid)

    def test_iterative_proposal_selection_does_not_use_reserved_pixels(self):
        model = CachedPointModel()
        delayed = np.stack(delayed_spiral_trajectory(model.acquisition, 6e-6, -3e-6)).reshape(2, -1)
        samples, _ = _signal_and_derivatives(model, delayed)
        coils = ContinuousReconstruction(model.acquisition, "cpu", torch.float64).numpy_image(samples)
        measured = np.sqrt(np.sum(abs(coils)**2, axis=0))
        first = fit_spiral_gradient_delays(model, model.acquisition, None, measured,
                                           np.eye(2), max_proposal_steps=3)
        modified = measured.copy()
        _, validation, test = spatial_split(measured.shape)
        modified[validation | test] *= 1.7
        second = fit_spiral_gradient_delays(model, model.acquisition, None, modified,
                                            np.eye(2), max_proposal_steps=3)
        self.assertGreaterEqual(len(first["report"]["proposals"]), 2)
        self.assertEqual(first["report"]["candidate_parameters"],
                         second["report"]["candidate_parameters"])
        self.assertEqual(first["report"]["proposals"], second["report"]["proposals"])

    def test_cpu_cache_fallback_keeps_existing_block_boundaries(self):
        model = CachedPointModel()
        k0 = np.stack((model.numpy(model._kx), model.numpy(model._ky)))
        expected = _signal_and_derivatives(model, k0)[0]
        cache, calls = model._base_cache, []
        model._base_cache = None
        def get_block(selected, b0_enabled=True):
            self.assertTrue(b0_enabled)
            calls.append((selected.start, selected.stop))
            return cache[:, selected]
        model._make_base = get_block
        actual = _signal_and_derivatives(model, k0)[0]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(calls, [(first, min(first + model.sample_chunk, model.nsamples))
                                for first in range(0, model.nsamples, model.sample_chunk)])


if __name__ == "__main__":
    unittest.main()
