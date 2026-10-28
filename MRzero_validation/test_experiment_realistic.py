"""Independent checks of notebook 04's complete forward/adjoint and solver."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Experiment"))
from test_acdc_gpu import fixture, direct_forward
from experiment_optimization import optimize_mrzero_image
from experiment_realistic import RealisticACDCModel


def small_model(off_grid=False, **options):
    experiment, calibration = fixture(response=True, tau=np.linspace(20e-6, 90e-6, 32))
    acq = experiment["acquisition"]
    acq.update(FOV_m=np.array([.12, .075]), TR_s=.006)
    if off_grid:
        acq["kx_mInv"] = acq["kx_mInv"] + np.array([.17, -.11, .04])[:, None]
        acq["ky_mInv"] = acq["ky_mInv"] + np.array([.03, -.05, .08])[:, None]
    model = RealisticACDCModel(
        experiment, calibration, profile="cpu", precision="float64",
        sample_chunk=5, **options)
    return experiment, calibration, model


class RealisticModelTests(unittest.TestCase):
    def test_disabled_channels_are_zero_for_every_mrzero_evaluation_and_response(self):
        for all_disabled in (False, True):
            with self.subTest(all_disabled=all_disabled):
                limits = np.full(32, 1.5)
                limits[[0, 7, 31]] = 0.
                if all_disabled:
                    limits[:] = 0.
                _, calibration, model = small_model(max_current_A=limits)
                disabled = limits == 0
                np.testing.assert_array_equal(model.maps_numpy, calibration["acdc_maps"])
                objective = model._objective_tensor
                evaluated = []

                def check_evaluation(currents, gradient=True):
                    np.testing.assert_array_equal(model.numpy(currents)[disabled], 0.)
                    evaluated.append(model.numpy(currents).copy())
                    return objective(currents, gradient=gradient)

                model._objective_tensor = check_evaluation
                requested = np.full((32, 3), .01)
                requested[disabled] = 10.
                solution = optimize_mrzero_image(
                    model, requested, total_current_limit_A=3., field_limit_Hz=100.,
                    max_iterations=3, use_cuda_graph=False)
                self.assertGreater(len(evaluated), 1)
                self.assertIsNone(solution["requested_initial_relative_mse"])
                self.assertTrue(solution["optimization_info"]["requested_initial_evaluation_skipped_disabled_channels"])
                for commands in (solution["initial_currents_A"], solution["currents_A"]):
                    np.testing.assert_array_equal(commands[disabled], 0.)
                    report = model.hardware.diagnostics(commands, 3., 100.)
                    self.assertTrue(report["all_limits_met"], report)
                    np.testing.assert_array_equal(np.asarray(report["peak_actual_channel_current_A"])[disabled], 0.)
                if not all_disabled:
                    self.assertGreater(np.abs(solution["initial_currents_A"]).max(), 0.)

    def test_native_signal_and_image_match_independent_bloch_fourier_sums(self):
        experiment, calibration, model = small_model(off_grid=True)
        currents = np.random.default_rng(20).normal(size=(32, 3)) * .04
        _, signal = direct_forward(experiment, calibration, currents)
        image, native_signal = model.forward(currents)
        np.testing.assert_allclose(native_signal, signal, rtol=2e-5, atol=3e-6)
        acq = experiment["acquisition"]
        expected = np.empty_like(image)
        for index in np.ndindex(image.shape):
            phase = (acq["kx_mInv"] * model.reconstruction.X_m[index]
                     + acq["ky_mInv"] * model.reconstruction.Y_m[index])
            expected[index] = np.sum(signal * np.exp(2j * np.pi * phase)) / signal.size
        np.testing.assert_allclose(image, expected, rtol=3e-5, atol=3e-6)

    def test_full_current_gradient_matches_directional_finite_difference(self):
        _, _, model = small_model(off_grid=True)
        rng = np.random.default_rng(21)
        currents = rng.normal(size=(32, 3)) * .04
        direction = rng.normal(size=currents.shape)
        direction /= np.linalg.norm(direction)
        value, gradient = model.objective_and_grad(currents)
        step = 1e-5
        plus = model.objective_and_grad(currents + step * direction)[0]
        minus = model.objective_and_grad(currents - step * direction)[0]
        np.testing.assert_allclose(np.sum(gradient * direction),
                                   (plus - minus) / (2 * step),
                                   rtol=3e-6, atol=1e-9)
        self.assertTrue(np.isfinite(value))

    def test_cartesian_reconstruction_keeps_the_old_objective(self):
        experiment, calibration, model = small_model()
        from experiment_mrzero import MRzeroExperimentModel
        old = MRzeroExperimentModel(deepcopy(experiment), deepcopy(calibration),
                                   profile="cpu", precision="float64", sample_chunk=5)
        currents = np.random.default_rng(22).normal(size=(32, 3)) * .04
        for actual, expected in zip(model.forward(currents), old.forward(currents)):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
        a, ga = model.objective_and_grad(currents)
        b, gb = old.objective_and_grad(currents)
        self.assertAlmostEqual(a, b, places=11)
        np.testing.assert_allclose(ga, gb, rtol=1e-10, atol=1e-11)

    def test_solver_snapshots_and_best_result_obey_new_hardware_limits(self):
        _, _, model = small_model(
            off_grid=True, hardware={"max_slew_A_per_s": 150.0,
                                     "avg_abs_current_A": .002})
        snapshots = []
        solution = optimize_mrzero_image(
            model, np.ones((32, 3)) * .2, total_current_limit_A=.5,
            field_limit_Hz=30, max_iterations=4, patience=20,
            callback=snapshots.append, use_cuda_graph=False,
        )
        for row in snapshots:
            report = model.hardware.diagnostics(row["currents_A"], .5, 30)
            self.assertTrue(report["all_limits_met"], report)
            image, _ = model.forward(row["currents_A"])
            np.testing.assert_allclose(row["image"], image)
            mse = np.sum((abs(image) - abs(model.reference))**2) / np.sum(abs(model.reference)**2)
            self.assertAlmostEqual(row["fval"], mse, places=11)
        self.assertLessEqual(solution["optimization_info"]["fval"],
                             solution["initial_relative_mse"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
