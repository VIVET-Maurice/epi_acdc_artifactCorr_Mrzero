"""Independent child-wise Bloch sums, phase cancellation and current gradients."""

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
from experiment_realistic import RealisticACDCModel
from experiment_subspins import SubspinACDCModel


def expanded_fixture(children=2, in_plane=False):
    original, original_calibration = fixture(response=True, tau=np.linspace(20e-6, 90e-6, 32))
    original["acquisition"].update(FOV_m=np.asarray([.12, .075]), TR_s=.006)
    original["geometry"]["Z_m"] = np.full(original["m0_proxy"].shape, .03)
    parent_shape = original["m0_proxy"].shape
    parent_indices = np.flatnonzero(original["field"]["simulation_mask"].reshape(-1))
    parents = len(parent_indices)
    weights = np.full((parents, children), 1 / children)

    def expand(array):
        return np.repeat(np.asarray(array).reshape((-1,) + np.asarray(array).shape[2:])[parent_indices, None],
                         children, axis=1)

    experiment = deepcopy(original)
    experiment["m0_proxy"] = expand(original["m0_proxy"]) * weights
    experiment["geometry"] = {key: expand(original["geometry"][key]) for key in ("X_m", "Y_m", "Z_m")}
    offsets = (np.arange(children) + .5) / children - .5
    experiment["geometry"]["Z_m"] += offsets[None] * .004
    if in_plane:
        experiment["geometry"]["X_m"] += offsets[None] * .008
    experiment["field"] = {key: expand(value) for key, value in original["field"].items()}
    experiment["relaxation"]["T2star_s"] = expand(original["relaxation"]["T2star_s"])
    experiment["subspins"] = {"parent_shape": parent_shape, "parent_flat_index": parent_indices,
                              "weights": weights, "through_plane_B0": "synthetic test fixture"}
    calibration = deepcopy(original_calibration)
    calibration.update(acdc_maps=expand(original_calibration["acdc_maps"]),
                       X_m=experiment["geometry"]["X_m"].copy(),
                       Y_m=experiment["geometry"]["Y_m"].copy())
    return original, original_calibration, experiment, calibration


def model(experiment, calibration, **options):
    return SubspinACDCModel(experiment, calibration, profile="cpu", precision="float64", sample_chunk=5,
                           **options)


class SubspinModelTests(unittest.TestCase):
    def test_one_child_matches_notebook04_native_signal_image_and_gradient(self):
        original, original_calibration, expanded, calibration = expanded_fixture(children=1)
        parent = RealisticACDCModel(original, original_calibration, profile="cpu", precision="float64", sample_chunk=5)
        child = model(expanded, calibration)
        currents = np.random.default_rng(32).normal(size=(32, 3)) * .04
        for actual, expected in zip(child.forward(currents), parent.forward(currents)):
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(child.objective_and_grad(currents)[1], parent.objective_and_grad(currents)[1],
                                   rtol=2e-11, atol=2e-12)

    def test_expanded_native_signal_matches_independent_scalar_bloch_sum(self):
        _, _, expanded, calibration = expanded_fixture(children=3, in_plane=True)
        expanded["field"]["b0Hz"] += np.asarray([-47., 8., 39.])[None]
        calibration["acdc_maps"] *= np.asarray([.8, 1.1, 1.3])[None, :, None]
        currents = np.random.default_rng(33).normal(size=(32, 3)) * .04
        with patch.object(ACDCModel, "_make_base", side_effect=AssertionError("analytic baseline forbidden")):
            child = model(expanded, calibration)
            actual_image, actual_signal = child.forward(currents)
        expected_image, expected_signal = direct_forward(expanded, calibration, currents)
        np.testing.assert_allclose(actual_signal, expected_signal, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(actual_image, expected_image, rtol=3e-5, atol=3e-6)
        self.assertEqual(child.B.shape[0], int(child.support.sum()))
        self.assertGreater(child.runtime_info["native_blocks_simulated"], 0)

    def test_childwise_current_gradient_with_per_channel_response(self):
        _, _, expanded, calibration = expanded_fixture(children=3, in_plane=True)
        expanded["field"]["b0Hz"] += np.asarray([-60., 0., 60.])[None]
        calibration["acdc_maps"] *= np.asarray([.7, 1.0, 1.4])[None, :, None]
        child = model(expanded, calibration)
        rng = np.random.default_rng(34)
        currents = rng.normal(size=(32, 3)) * .04
        direction = rng.normal(size=currents.shape)
        direction /= np.linalg.norm(direction)
        _, gradient = child.objective_and_grad(currents)
        epsilon = 1e-5
        plus = child.objective_and_grad(currents + epsilon * direction)[0]
        minus = child.objective_and_grad(currents - epsilon * direction)[0]
        np.testing.assert_allclose(np.sum(gradient * direction), (plus - minus) / (2 * epsilon),
                                   rtol=4e-6, atol=2e-9)

    def test_homogeneous_zero_k_normalization_independent_of_child_count(self):
        for count in [1, 2, 5]:
            original, _, expanded, calibration = expanded_fixture(children=count)
            expanded["field"]["b0Hz"][:] = 0
            expanded["relaxation"]["T2star_s"][:] = np.inf
            for key in ("kx_mInv", "ky_mInv", "phase_offset_rad"):
                expanded["acquisition"][key][:] = 0
            child = model(expanded, calibration)
            _, signal = child.forward(np.zeros((32, 3)))
            np.testing.assert_allclose(signal, original["m0_proxy"].sum(), rtol=2e-6, atol=2e-6)

    def cancellation_model(self, in_plane=False):
        _, _, expanded, calibration = expanded_fixture(children=2, in_plane=in_plane)
        expanded["relaxation"]["T2star_s"][:] = np.inf
        offsets = expanded["geometry"]["Z_m"] - .03
        expanded["field"]["b0Hz"] = offsets * 100000
        calibration["acdc_maps"][:] = .1
        calibration["acdc_maps"][..., 0] = -expanded["field"]["b0Hz"]
        return model(expanded, calibration)

    def test_through_plane_phase_cancels_only_before_summing_children(self):
        child = self.cancellation_model()
        zeros = np.zeros((32, 3))
        currents = zeros.copy()
        currents[0] = 1
        baseline, _ = child.forward(zeros, response_enabled=False)
        corrected, _ = child.forward(currents, response_enabled=False)
        np.testing.assert_allclose(corrected, child.ideal_image, rtol=4e-5, atol=4e-6)
        self.assertGreater(np.linalg.norm(baseline - child.ideal_image), .05)

    def test_child_field_bounds_are_not_replaced_with_zero_display_average(self):
        child = self.cancellation_model()
        currents = np.zeros((32, 3))
        currents[0] = 1
        diagnostics = child.response_diagnostics(currents, field_limit_Hz=10, total_current_limit_A=100)
        self.assertFalse(diagnostics["actual_adc_field_limits_met"])
        self.assertGreater(diagnostics["maximum_actual_quality_field_Hz"], 90)
        self.assertLess(np.nanmax(np.abs(diagnostics["generated_field_at_line_center_Hz"])), 1e-10)
        self.assertEqual(diagnostics["generated_field_at_line_center_Hz"].shape, child.parent_shape + (3,))
        self.assertEqual(diagnostics["generated_field_at_line_center_subspin_Hz"].shape,
                         child.object_size + (3,))
        projected = child.project_currents(currents, total_current_limit_A=100, field_limit_Hz=10)
        self.assertLessEqual(np.max(np.abs(child.maps_numpy[child.quality_mask] @ projected)), 10)

    def test_inplane_child_b0_is_not_averaged_before_native_simulation(self):
        child = self.cancellation_model(in_plane=True)
        currents = np.zeros((32, 3))
        value, gradient = child.objective_and_grad(currents)
        self.assertGreater(value, 1e-3)
        self.assertGreater(np.linalg.norm(gradient), 1e-3)
        # Every parent's mean B0 is zero; an averaged-field model would give
        # exactly the ideal image and a zero objective at these zero currents.
        self.assertEqual(child.runtime_info["subspins"]["children_per_parent"], 2)

    def test_parent_aggregation_uses_weights_and_preserves_missing_pixels(self):
        _, _, expanded, calibration = expanded_fixture(children=2)
        child = model(expanded, calibration)
        values = np.ones(child.object_size + (3,)) * np.asarray([2, 6])[None, :, None]
        output = child.aggregate_to_parent(values)
        self.assertTrue(np.isnan(output.reshape(-1, 3)[0]).all())
        np.testing.assert_allclose(output.reshape(-1, 3)[child.parent_flat_index], 4)

    def test_chunked_child_field_maximum_matches_full_array_across_adc_blocks(self):
        rng = np.random.default_rng(36)
        maps = rng.normal(size=(101, 32))
        currents = rng.normal(size=(32, 137))
        expected = float(np.max(np.abs(maps @ currents)))
        for chunk in [1, 17, 64, 200]:
            self.assertAlmostEqual(SubspinACDCModel._maximum_child_field_hz(maps, currents, chunk),
                                   expected, places=12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_small_cuda_native_subspins_match_cpu(self):
        _, _, expanded, calibration = expanded_fixture(children=2, in_plane=True)
        cpu = model(expanded, calibration)
        gpu = SubspinACDCModel(deepcopy(expanded), deepcopy(calibration),
                              profile="auto", precision="float64", sample_chunk=5)
        currents = np.random.default_rng(35).normal(size=(32, 3)) * .04
        for actual, expected in zip(gpu.forward(currents), cpu.forward(currents)):
            np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-6)
        np.testing.assert_allclose(gpu.objective_and_grad(currents)[1], cpu.objective_and_grad(currents)[1],
                                   rtol=6e-5, atol=4e-6)


if __name__ == "__main__":
    unittest.main()
