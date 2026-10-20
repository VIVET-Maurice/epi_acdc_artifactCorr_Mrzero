"""Check graph replay against the direct MRzero image, adjoint and projection."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Experiment"))
from test_acdc_gpu import fixture
from cuda_graph_helpers import GpuEvaluation
from experiment_mrzero import MRzeroExperimentModel
from experiment_optimization import optimize_mrzero_image


class DirectFallbackTests(unittest.TestCase):
    def test_cpu_preserves_direct_objective_and_projection(self):
        experiment, calibration = fixture(response=True)
        model = MRzeroExperimentModel(
            experiment, calibration, profile="cpu", precision="float64", sample_chunk=5,
        )
        evaluator = GpuEvaluation(model, 1.0, 5.0)
        self.assertFalse(evaluator.enabled)
        self.assertIn("CPU", evaluator.status)
        currents = model.tensor(np.full((model.nc, model.nstates), .06))
        for received, expected in zip(evaluator.evaluate(currents), model._objective_tensor(currents)):
            torch.testing.assert_close(received, expected, rtol=0, atol=0)
        torch.testing.assert_close(evaluator.project(currents), model.project_tensor(currents, 1.0, 5.0), rtol=0, atol=0)

    def test_optimizer_cpu_fallback_keeps_final_callback_and_numeric_history(self):
        experiment, calibration = fixture(response=True)
        model = MRzeroExperimentModel(
            experiment, calibration, profile="cpu", precision="float64", sample_chunk=5,
        )
        snapshots = []
        solution = optimize_mrzero_image(
            model, max_iterations=7, patience=20, callback_every=3,
            use_cuda_graph=True, callback=snapshots.append,
        )
        self.assertFalse(solution["optimization_info"]["cuda_graph"]["enabled"])
        self.assertIn("CPU", solution["optimization_info"]["cuda_graph"]["status"])
        self.assertEqual([row["iteration"] for row in snapshots], [0, 3, 6, 7])
        np.testing.assert_array_equal(solution["history"]["iteration"], np.arange(8))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        for precision, tau in [("float32", None), ("float64", np.linspace(20e-6, 95e-6, 32))]:
            experiment, calibration = fixture(response=True, tau=tau)
            model = MRzeroExperimentModel(
                experiment, calibration, profile="auto", precision=precision, sample_chunk=5,
            )
            evaluator = GpuEvaluation(model, 1.0, 5.0)
            cls.cases.append((model, evaluator))

    def test_replayed_different_candidates_match_direct_images_gradients_and_loss(self):
        rng = np.random.default_rng(184)
        for model, evaluator in self.cases:
            with self.subTest(dtype=model.dtype):
                self.assertTrue(evaluator.enabled, evaluator.status)
                for scale in (0.0, .06, .12, .02):
                    currents = model.tensor(rng.normal(size=(model.nc, model.nstates)) * scale)
                    received = evaluator.evaluate(currents)
                    expected = model._objective_tensor(currents)
                    for actual, reference in zip(received, expected):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_projection_is_exact_and_returned_currents_survive_later_projection(self):
        rng = np.random.default_rng(185)
        for model, evaluator in self.cases:
            with self.subTest(dtype=model.dtype):
                for scale in (.005, .2, 10.0):
                    currents = model.tensor(rng.normal(size=(model.nc, model.nstates)) * scale)
                    projected = evaluator.project(currents)
                    expected = model.project_tensor(currents, 1.0, 5.0)
                    torch.testing.assert_close(projected, expected, rtol=0, atol=0)
                    before = projected.clone()
                    # Reproduces the optimizer's stopping diagnostic, which
                    # must not overwrite its retained projected candidate.
                    evaluator.project(projected - .01)
                    torch.testing.assert_close(projected, before, rtol=0, atol=0)

    def test_evaluation_owned_copies_survive_replay(self):
        model, evaluator = self.cases[0]
        original = model.tensor(np.zeros((model.nc, model.nstates)))
        first = evaluator.evaluate(original)
        copies = tuple(value.clone() for value in first)
        expected = model._objective_tensor(original)
        evaluator.evaluate(original + .3)
        for actual, reference in zip(copies, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_uncached_gpu_baseline_uses_direct_path(self):
        experiment, calibration = fixture(response=True)
        model = MRzeroExperimentModel(
            experiment, calibration, profile="auto", precision="float32",
            memory_fraction=1e-9, sample_chunk=5,
        )
        evaluator = GpuEvaluation(model, 1.0, 5.0)
        self.assertIsNone(model._base_cache)
        self.assertFalse(evaluator.enabled)
        self.assertIn("not cached", evaluator.status)
        currents = model.tensor(np.full((model.nc, model.nstates), .03))
        for received, expected in zip(evaluator.evaluate(currents), model._objective_tensor(currents)):
            torch.testing.assert_close(received, expected, rtol=0, atol=0)

    def test_optimizer_graphs_preserve_trajectory_best_solution_and_sparse_snapshots(self):
        for model, _ in self.cases:
            with self.subTest(dtype=model.dtype):
                direct_snapshots, graph_snapshots = [], []
                options = dict(max_iterations=7, patience=20, callback_every=3)
                direct = optimize_mrzero_image(
                    model, use_cuda_graph=False, callback=direct_snapshots.append, **options,
                )
                captured = optimize_mrzero_image(
                    model, use_cuda_graph=True, callback=graph_snapshots.append, **options,
                )
                self.assertTrue(captured["optimization_info"]["cuda_graph"]["enabled"])
                self.assertEqual(captured["optimization_info"]["status"], direct["optimization_info"]["status"])
                for name in ("currents_A", "image", "kdata", "generated_field_Hz"):
                    np.testing.assert_array_equal(captured[name], direct[name])
                # Elapsed times and capture overhead naturally differ; every
                # numerical history entry and candidate must remain identical.
                for name in direct["history"]:
                    if name != "elapsed_s":
                        np.testing.assert_array_equal(captured["history"][name], direct["history"][name])
                self.assertEqual([row["iteration"] for row in graph_snapshots], [0, 3, 6, 7])
                np.testing.assert_array_equal(captured["history"]["iteration"], np.arange(8))
                for snapshot, direct_snapshot in zip(graph_snapshots, direct_snapshots):
                    for name in ("currents_A", "image"):
                        np.testing.assert_array_equal(snapshot[name], direct_snapshot[name])
                    iteration = snapshot["iteration"]
                    image, _ = model.forward(snapshot["currents_A"])
                    np.testing.assert_array_equal(snapshot["image"], image)
                    self.assertEqual(snapshot["fval"], captured["history"]["fval"][iteration])
                    np.testing.assert_array_equal(snapshot["history"]["iteration"], np.arange(iteration + 1))
                    np.testing.assert_array_equal(snapshot["history"]["fval"], captured["history"]["fval"][:iteration + 1])

    def test_optimizer_plateau_reset_and_terminal_callback_preserve_graph_trajectory(self):
        model, _ = self.cases[0]
        options = dict(max_iterations=20, patience=1,
                       min_relative_improvement=1e6, callback_every=3)
        direct = optimize_mrzero_image(model, use_cuda_graph=False, **options)
        snapshots = []
        captured = optimize_mrzero_image(
            model, use_cuda_graph=True, callback=snapshots.append, **options,
        )
        self.assertEqual(captured["optimization_info"]["status"], "plateau")
        self.assertEqual(snapshots[-1]["iteration"], captured["history"]["iteration"][-1])
        for name in ("currents_A", "image", "kdata"):
            np.testing.assert_array_equal(captured[name], direct[name])
        for name in direct["history"]:
            if name != "elapsed_s":
                np.testing.assert_array_equal(captured["history"][name], direct["history"][name])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
