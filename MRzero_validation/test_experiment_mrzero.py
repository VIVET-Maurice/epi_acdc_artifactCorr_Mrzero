"""Independent signal, current-gradient and device checks for native MRzero."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Experiment"))
from test_acdc_gpu import fixture, direct_forward
from acdc_gpu import ACDCModel
from experiment_mrzero import MRzeroExperimentModel


def cpu_model(experiment, calibration, **options):
    return MRzeroExperimentModel(experiment, calibration, profile="cpu",
                                precision="float64", **options)


class NativeForwardTests(unittest.TestCase):
    def test_cancelling_native_baseline_uses_condition_aware_resummation(self):
        positions = (np.arange(32) - 16) * .2 / 32
        X, Y = np.meshgrid(positions, positions)
        support = np.ones_like(X, bool)
        kx = np.asarray([[10., 50., 100., 150.]])
        experiment = {
            "m0_proxy": np.ones_like(X), "geometry": {"X_m": X, "Y_m": Y},
            "field": {"b0Hz": np.zeros_like(X), "mask": support, "simulation_mask": support},
            "relaxation": {"T2star_s": np.inf},
            "acquisition": {
                "kx_mInv": kx, "ky_mInv": np.full_like(kx, 50.),
                "time_s": np.asarray([[.01, .0101, .0102, .0103]]), "TE_s": .01,
                "phase_offset_rad": np.asarray([[0., .3, .6, .9]]),
                "rf_time_s": np.zeros_like(kx), "sequence_kind": "epi",
                "line_index": np.ones_like(kx, dtype=int),
                "line_start_s": [0.], "line_end_s": [.02],
            },
        }
        calibration = {"acdc_maps": np.ones(X.shape + (1,)), "X_m": X, "Y_m": Y,
                       "units": "Hz/A", "registration_confirmed": True}
        model = cpu_model(experiment, calibration)
        self.assertGreater(model.runtime_info["native_resummation_relative_error"], 5e-5)
        self.assertLessEqual(model.runtime_info["native_resummation_max_tolerance_ratio"], 1.)
        self.assertLess(model.runtime_info["native_resummation_mass_relative_error"],
                        model.runtime_info["native_resummation_roundoff_tolerance"])
        self.assertTrue(np.isfinite(model.forward(np.zeros((1, 1)))[1]).all())

    def test_native_baseline_rejects_corrupted_adc_signal(self):
        import MRzeroCore as mr0

        experiment, calibration = fixture()
        execute = mr0.execute_graph

        def corrupt(*args, **kwargs):
            signal, pathways = execute(*args, **kwargs)
            signal[0, 0] += .01 * pathways[0][0][0].abs().sum()
            return signal, pathways

        with patch.object(mr0, "execute_graph", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "point resummation failed.*tolerance ratio"):
                cpu_model(experiment, calibration)

    def test_native_epi_gre_response_and_ideal_match_independent_bloch_sum(self):
        commands = np.random.default_rng(8).normal(size=(32, 3)) * .06
        cases = [("epi", False, None), ("gre", False, None),
                 ("epi", True, 133e-6 / np.log(100)),
                 ("epi", True, np.linspace(20e-6, 95e-6, 32))]
        for kind, response, tau in cases:
            with self.subTest(kind=kind, response=response, per_channel=np.ndim(tau) > 0):
                experiment, calibration = fixture(kind, response, tau)
                model = cpu_model(experiment, calibration, sample_chunk=5)
                actual = model.forward(commands)
                expected = direct_forward(experiment, calibration, commands)
                for received, reference in zip(actual, expected):
                    np.testing.assert_allclose(received, reference, rtol=2e-5, atol=2e-6)
                ideal = direct_forward(experiment, calibration, np.zeros_like(commands), ideal=True)
                np.testing.assert_allclose(model.ideal_image, ideal[0], rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(model.ideal_kdata, ideal[1], rtol=2e-5, atol=2e-6)
                self.assertEqual(model.runtime_info["simulator"], "MRzeroCore")
                self.assertLess(model.runtime_info["native_resummation_relative_error"], 5e-5)

    def test_native_baseline_does_not_use_parent_analytic_fourier_baseline(self):
        experiment, calibration = fixture()
        with patch.object(ACDCModel, "_make_base", side_effect=AssertionError("analytic fallback")):
            model = cpu_model(experiment, calibration, sample_chunk=5)
            image, _ = model.forward(np.zeros((32, 3)))
        self.assertTrue(np.isfinite(image).all())
        self.assertGreater(model.runtime_info["native_blocks_simulated"], 0)

    def test_response_override_and_infinite_t2star(self):
        experiment, calibration = fixture(response=True)
        experiment["relaxation"]["T2star_s"] = np.inf
        model = cpu_model(experiment, calibration, sample_chunk=5)
        commands = np.random.default_rng(9).normal(size=(32, 3)) * .08
        received = model.forward(commands, response_enabled=False)
        reference = direct_forward(experiment, calibration, commands, response=False)
        for actual, expected in zip(received, reference):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_current_adjoint_matches_central_finite_difference(self):
        experiment, calibration = fixture(response=True, tau=np.linspace(20e-6, 90e-6, 32))
        model = cpu_model(experiment, calibration, sample_chunk=5)
        currents = np.random.default_rng(13).normal(size=(32, 3)) * .06
        value, gradient = model.objective_and_grad(currents)
        direction = np.random.default_rng(14).normal(size=currents.shape)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-5
        plus = model.objective_and_grad(currents + epsilon * direction)[0]
        minus = model.objective_and_grad(currents - epsilon * direction)[0]
        np.testing.assert_allclose(np.sum(gradient * direction), (plus - minus) / (2 * epsilon),
                                   rtol=2e-6, atol=1e-9)
        self.assertTrue(np.isfinite(value))

    def test_cpu_cache_avoids_native_resimulation_during_iterations(self):
        experiment, calibration = fixture(response=True)
        model = cpu_model(experiment, calibration, memory_fraction=1e-9)
        self.assertIsNone(model._base_cache)
        self.assertEqual(model.runtime_info["native_baseline_cache"], "CPU")
        count = model._native_block_count
        model.forward(np.zeros((32, 3)))
        model.objective_and_grad(np.ones((32, 3)) * .01)
        model.objective_and_grad(np.ones((32, 3)) * .02)
        self.assertEqual(model._native_block_count, count)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_cpu_native_baseline_and_current_adjoint(self):
        experiment, calibration = fixture(response=True, tau=np.linspace(20e-6, 90e-6, 32))
        cpu = cpu_model(experiment, calibration, sample_chunk=5)
        gpu = MRzeroExperimentModel(deepcopy(experiment), deepcopy(calibration),
                                    profile="auto", precision="float64", sample_chunk=5)
        commands = np.random.default_rng(17).normal(size=(32, 3)) * .06
        for actual, expected in zip(gpu.forward(commands), cpu.forward(commands)):
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
        value_gpu, gradient_gpu = gpu.objective_and_grad(commands)
        value_cpu, gradient_cpu = cpu.objective_and_grad(commands)
        np.testing.assert_allclose(value_gpu, value_cpu, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(gradient_gpu, gradient_cpu, rtol=5e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
