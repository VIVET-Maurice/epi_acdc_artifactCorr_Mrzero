"""Check per-iteration MRzero snapshots and feasible optimization results."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Experiment"))
from test_acdc_gpu import fixture
from experiment_mrzero import MRzeroExperimentModel
from experiment_optimization import optimize_mrzero_image


class LiveSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        experiment, calibration = fixture(response=True)
        cls.model = MRzeroExperimentModel(
            experiment, calibration, profile="cpu", precision="float64",
            sample_chunk=5,
        )

    def test_each_snapshot_matches_its_evaluated_image_and_commands(self):
        snapshots = []
        solution = optimize_mrzero_image(
            self.model, max_iterations=6, patience=20, callback=snapshots.append,
        )
        iterations = [row["iteration"] for row in snapshots]
        self.assertEqual(iterations, list(range(len(snapshots))))
        self.assertEqual(len(snapshots), len(solution["history"]["iteration"]))
        reference_energy = np.abs(self.model.reference).ravel() @ np.abs(self.model.reference).ravel()
        for row in snapshots:
            image, _ = self.model.forward(row["currents_A"])
            np.testing.assert_allclose(row["image"], image, atol=1e-11, rtol=1e-11)
            mse = np.sum((np.abs(image) - np.abs(self.model.reference))**2) / reference_energy
            self.assertAlmostEqual(row["fval"], mse, places=11)
        self.assertAlmostEqual(solution["optimization_info"]["fval"], min(row["fval"] for row in snapshots), places=11)

    def test_callback_arrays_cannot_modify_solver_state(self):
        arguments = dict(max_iterations=4, patience=20)
        reference = optimize_mrzero_image(self.model, **arguments)

        def modify_snapshot(snapshot):
            snapshot["currents_A"][:] = 1000.0
            snapshot["image"][:] = 0.0

        observed = optimize_mrzero_image(self.model, callback=modify_snapshot, **arguments)
        np.testing.assert_array_equal(reference["currents_A"], observed["currents_A"])
        np.testing.assert_array_equal(reference["image"], observed["image"])

    def test_periodic_display_keeps_complete_history_and_final_state(self):
        snapshots = []
        solution = optimize_mrzero_image(
            self.model, max_iterations=7, patience=20,
            callback=snapshots.append, callback_every=3,
        )
        self.assertEqual([row["iteration"] for row in snapshots], [0, 3, 6, 7])
        np.testing.assert_array_equal(solution["history"]["iteration"], np.arange(8))
        np.testing.assert_array_equal(snapshots[-1]["history"]["iteration"], np.arange(8))
        np.testing.assert_array_equal(snapshots[-1]["history"]["fval"], solution["history"]["fval"])

    def test_infeasible_start_and_all_displayed_candidates_obey_bounds(self):
        snapshots = []
        solution = optimize_mrzero_image(
            self.model, np.full((self.model.nc, self.model.nstates), 10.0),
            total_current_limit_A=0.8, field_limit_Hz=5.0,
            max_iterations=5, patience=20, callback=snapshots.append,
        )
        self.assertTrue(solution["optimization_info"]["initial_was_projected"])
        for row in snapshots:
            currents = row["currents_A"]
            self.assertLess(np.max(np.abs(currents).sum(axis=0)), 0.8)
            self.assertLessEqual(row["max_field_Hz"], 5.0)
            self.assertTrue(np.isfinite(row["fval"]))
        self.assertTrue(solution["optimization_info"]["accepted"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
