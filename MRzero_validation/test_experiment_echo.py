"""Independent echo anchoring, full-object and frame/current-gradient checks."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from continuous_reconstruction import ContinuousReconstruction
from experiment_echo import EchoSubspinACDCModel
from experiment_subspins import SubspinACDCModel
from test_experiment_subspins import expanded_fixture
from test_acdc_gpu import direct_forward


def cpu_model(experiment, calibration, **options):
    return EchoSubspinACDCModel(experiment, calibration, profile="cpu", precision="float64",
                               sample_chunk=5, **options)


class EchoSubspinTests(unittest.TestCase):
    def test_required_gpu_is_enforced_before_native_simulation(self):
        _, _, experiment, calibration = expanded_fixture(children=2)
        experiment["amplitude_reference_time_s"] = .004
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "this run requires a GPU"):
                EchoSubspinACDCModel(experiment, calibration, profile="auto", require_cuda=True)

    def test_explicit_echo_anchor_matches_scalar_bloch_and_ideal_reference(self):
        _, _, experiment, calibration = expanded_fixture(children=3, in_plane=True)
        anchor = .0042
        experiment["amplitude_reference_time_s"] = anchor
        calibration["acdc_maps"] *= np.asarray([.8, 1.1, 1.4])[None, :, None]
        model = cpu_model(experiment, calibration)
        currents = np.random.default_rng(54).normal(size=(32, 3)) * .05
        oracle_experiment = deepcopy(experiment)
        # The independent oracle's TE field anchors amplitude; its ADC times
        # stay untouched, so this is the explicitly requested echo convention.
        oracle_experiment["acquisition"]["TE_s"] = anchor
        expected = direct_forward(oracle_experiment, calibration, currents)
        actual = model.forward(currents)
        for received, target in zip(actual, expected):
            np.testing.assert_allclose(received, target, rtol=3e-5, atol=3e-6)
        ideal = direct_forward(oracle_experiment, calibration, np.zeros_like(currents), ideal=True)
        np.testing.assert_allclose(model.ideal_image, ideal[0], rtol=3e-5, atol=3e-6)
        self.assertEqual(model.runtime_info["amplitude_reference_time_s"], anchor)

    def test_anchor_equal_to_epi_te_preserves_notebook04(self):
        _, _, experiment, calibration = expanded_fixture(children=2)
        old = SubspinACDCModel(experiment, calibration, profile="cpu", precision="float64", sample_chunk=5)
        new = cpu_model(experiment, calibration, amplitude_reference_time_s=experiment["acquisition"]["TE_s"])
        currents = np.random.default_rng(55).normal(size=(32, 3)) * .04
        for received, target in zip(new.forward(currents), old.forward(currents)):
            np.testing.assert_allclose(received, target, rtol=1e-12, atol=1e-12)

    def test_signal_domain_retains_points_outside_quality_roi(self):
        _, _, experiment, calibration = expanded_fixture(children=2)
        experiment["relaxation"]["amplitude_reference_time_s"] = .003
        base = cpu_model(experiment, calibration)
        changed = deepcopy(experiment)
        outside = ~np.asarray(changed["field"]["mask"], bool)
        self.assertTrue(outside.any())
        changed["field"]["b0Hz"][outside] += 200
        other = cpu_model(changed, calibration)
        zero = np.zeros((32, 3))
        self.assertGreater(np.linalg.norm(base.forward(zero)[1] - other.forward(zero)[1]), .01)
        self.assertGreater(base.runtime_info["simulation_children"], base.runtime_info["quality_children"])
        np.testing.assert_array_equal(base.B_quality.numpy(), other.B_quality.numpy())

    def test_factory_is_lazy_once_and_precedes_ideal_reference(self):
        _, _, experiment, calibration = expanded_fixture(children=2)
        calls = []
        def factory(acq, device, dtype):
            calls.append((device, dtype))
            return ContinuousReconstruction(acq, device, dtype)
        model = cpu_model(experiment, calibration, amplitude_reference_time_s=.004,
                          reconstruction_factory=factory)
        model.forward(np.zeros((32, 3)))
        model.objective_and_grad(np.ones((32, 3)) * .01)
        self.assertEqual(len(calls), 1)
        self.assertIs(model.reconstruction, model._echo_reconstruction)
        self.assertTrue(model.runtime_info["custom_reconstruction"])

    def test_child_hardware_field_bounds_use_actual_children(self):
        _, _, experiment, calibration = expanded_fixture(children=2)
        calibration["acdc_maps"][:] = .1
        calibration["acdc_maps"][..., 0] = np.asarray([-100., 100.])[None]
        model = cpu_model(experiment, calibration, amplitude_reference_time_s=.004)
        currents = np.zeros((32, 3))
        currents[0] = 1
        projected = model.project_currents(currents, total_current_limit_A=3, field_limit_Hz=10)
        self.assertLessEqual(np.max(np.abs(model.maps_numpy[model.quality_mask] @ projected)), 10)
        report = model.hardware.diagnostics(projected, 3, 10)
        self.assertTrue(report["all_limits_met"], report)

    def test_explicit_anchor_gradient_matches_directional_difference(self):
        _, _, experiment, calibration = expanded_fixture(children=3, in_plane=True)
        model = cpu_model(experiment, calibration, amplitude_reference_time_s=.0043)
        rng = np.random.default_rng(56)
        currents = rng.normal(size=(32, 3)) * .05
        direction = rng.normal(size=currents.shape)
        direction /= np.linalg.norm(direction)
        _, gradient = model.objective_and_grad(currents)
        epsilon = 1e-5
        plus = model.objective_and_grad(currents + epsilon * direction)[0]
        minus = model.objective_and_grad(currents - epsilon * direction)[0]
        np.testing.assert_allclose(np.sum(gradient * direction), (plus - minus) / (2 * epsilon),
                                   rtol=4e-6, atol=1e-9)

    def test_arbitrary_rotation_and_mirror_preserve_joint_frame_and_exact_adjoint(self):
        from inplane_geometry import FrameReconstruction
        _, _, experiment, calibration = expanded_fixture(children=3, in_plane=True)
        experiment["amplitude_reference_time_s"] = .0042
        original = cpu_model(experiment, calibration)
        angle = np.deg2rad(37.)
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        frame = rotation @ np.diag([-1., 1.])
        framed_experiment, framed_calibration = deepcopy(experiment), deepcopy(calibration)
        for container in (framed_experiment["geometry"], framed_calibration):
            points = frame @ np.stack((container["X_m"].reshape(-1), container["Y_m"].reshape(-1)))
            container["X_m"] = points[0].reshape(container["X_m"].shape)
            container["Y_m"] = points[1].reshape(container["Y_m"].shape)
        acq = framed_experiment["acquisition"]
        k = frame @ np.stack((acq["kx_mInv"].reshape(-1), acq["ky_mInv"].reshape(-1)))
        acq["kx_mInv"] = k[0].reshape(acq["kx_mInv"].shape)
        acq["ky_mInv"] = k[1].reshape(acq["ky_mInv"].shape)
        framed = cpu_model(framed_experiment, framed_calibration,
                           reconstruction_factory=lambda a, d, p: FrameReconstruction(a, d, p, frame_matrix=frame))
        rng = np.random.default_rng(57)
        currents = rng.normal(size=(32, 3)) * .04
        for actual, expected in zip(framed.forward(currents), original.forward(currents)):
            np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-6)
        value, gradient = framed.objective_and_grad(currents)
        original_value, original_gradient = original.objective_and_grad(currents)
        np.testing.assert_allclose(value, original_value, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(gradient, original_gradient, rtol=5e-5, atol=3e-6)
        direction = rng.normal(size=currents.shape)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-5
        plus = framed.objective_and_grad(currents + epsilon * direction)[0]
        minus = framed.objective_and_grad(currents - epsilon * direction)[0]
        np.testing.assert_allclose(np.sum(gradient * direction), (plus - minus) / (2 * epsilon),
                                   rtol=4e-6, atol=1e-9)

    def test_missing_anchor_and_reconstruction_precision_mismatch_rejected(self):
        _, _, experiment, calibration = expanded_fixture(children=1)
        with self.assertRaisesRegex(ValueError, "explicit amplitude"):
            cpu_model(experiment, calibration)
        operator = ContinuousReconstruction(experiment["acquisition"], "cpu", torch.float32)
        with self.assertRaisesRegex(ValueError, "dtype differs"):
            cpu_model(experiment, calibration, amplitude_reference_time_s=.004, reconstruction=operator)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
