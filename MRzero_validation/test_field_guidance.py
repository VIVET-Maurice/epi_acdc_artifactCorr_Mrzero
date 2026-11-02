"""Independent temporal-response, alias-branch and derivative checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Experiment"))
from acdc_gpu import ACDCModel
from test_acdc_gpu import event_response, fixture
from field_guidance import FieldGuidance


def make_model(*, response=False, tau=None, limits=3., precision="float64"):
    experiment, calibration = fixture(response=response, tau=tau)
    model = ACDCModel(experiment, calibration, limits, profile="cpu", precision=precision)
    return experiment, calibration, model


def independent_penalty(experiment, calibration, commands, centers, scale):
    """Event-ODE integral, or explicit piecewise command integration."""
    acq = experiment["acquisition"]
    if experiment["current_response"]["enabled"]:
        _, integrals = event_response(commands, acq["line_start_s"], acq["line_end_s"],
                                      centers, experiment["current_response"]["tau_s"])
    else:
        integrals = np.array([[sum(commands[channel, state] * max(0., min(t, end)-start)
            for state, (start, end) in enumerate(zip(acq["line_start_s"], acq["line_end_s"])))
            for channel in range(commands.shape[0])] for t in centers])
    actual_mean = np.diff(integrals, axis=0).T / np.diff(centers)
    mask = experiment["field"]["mask"] & experiment["field"]["simulation_mask"] & (experiment["m0_proxy"] > 0)
    residual = experiment["field"]["b0Hz"][mask, None] + calibration["acdc_maps"][mask] @ actual_mean
    return float(np.mean(residual**2) / scale**2), float(np.sqrt(np.mean(residual**2)))


class FieldGuidanceTests(unittest.TestCase):
    def test_unequal_spiral_turns_match_event_ode_with_response_and_disabled_channels(self):
        from experiment_multiecho import MultiEchoVoxelACDCModel
        from test_spiral_model import spiral_fixture

        rng = np.random.default_rng(752)
        for enabled, tau in ((True, 350e-6),
                             (True, np.linspace(100e-6, 900e-6, 32))):
            with self.subTest(enabled=enabled, channel_specific=np.ndim(tau) > 0):
                experiment, calibration = spiral_fixture()
                experiment["current_response"] = {"enabled": enabled, "tau_s": tau}
                limits = np.full(32, 3.)
                limits[8] = 0.
                model = MultiEchoVoxelACDCModel(experiment, calibration, limits,
                    profile="cpu", precision="float64", sample_chunk=7)
                guide = FieldGuidance(model, scale_Hz=75.)
                commands = model.project_currents(rng.normal(size=(32, 3)) * .12, 30., 500.)
                value, derivative, rms = guide.value_gradient(commands)
                expected = independent_penalty(experiment, calibration, commands,
                    guide.state_center_times_s, 75.)
                np.testing.assert_allclose([float(value), float(rms)], expected, rtol=3e-13)
                np.testing.assert_array_equal(commands[8], 0.)
                counts = np.bincount(model.line_index.ravel())[1:]
                self.assertGreater(np.ptp(counts), 0)
                self.assertEqual(guide.report["command_state_unit"], "spiral turn")
                self.assertEqual(guide.report["sequence_kind"], "spiral")
                direction = rng.normal(size=commands.shape)
                direction[8] = 0.
                direction /= np.linalg.norm(direction)
                step = 1e-5
                difference = (guide.value_gradient(commands + step * direction, gradient=False)[0]
                              - guide.value_gradient(commands - step * direction, gradient=False)[0]) / (2 * step)
                np.testing.assert_allclose(float((derivative * model.tensor(direction)).sum()),
                    float(difference), rtol=3e-7, atol=2e-10)
                json.dumps(guide.report, allow_nan=False)

    def test_matches_independent_response_and_preserves_chronological_order(self):
        rng = np.random.default_rng(749)
        for enabled, tau in ((False, None), (True, 350e-6), (True, np.linspace(100e-6, 900e-6, 32))):
            with self.subTest(enabled=enabled, tau=np.asarray(tau).shape):
                experiment, calibration, model = make_model(response=enabled, tau=tau)
                guide = FieldGuidance(model, scale_Hz=65.)
                commands = rng.normal(size=(32, 3)) * .18
                actual, _, rms = guide.value_gradient(model.tensor(commands))
                expected, expected_rms = independent_penalty(
                    experiment, calibration, commands, guide.line_center_times_s, 65.)
                np.testing.assert_allclose(float(actual), expected, rtol=2e-13)
                np.testing.assert_allclose(float(rms), expected_rms, rtol=2e-13)
                # The fixture stores chronological line 3 in Cartesian row 0.
                np.testing.assert_allclose(guide.line_center_times_s, [.00032, .00162, .00282])
                self.assertEqual(guide.report["interval_count"], 2)
                json.dumps(guide.report, allow_nan=False)

    def test_exact_gradient_matches_directional_finite_difference(self):
        rng = np.random.default_rng(750)
        for enabled, tau in ((False, None), (True, 450e-6), (True, np.linspace(100e-6, 900e-6, 32))):
            with self.subTest(enabled=enabled, tau=np.asarray(tau).shape):
                _, _, model = make_model(response=enabled, tau=tau)
                guide = FieldGuidance(model, scale_Hz=80.)
                commands = rng.normal(size=(32, 3)) * .1
                direction = rng.normal(size=commands.shape)
                direction /= np.linalg.norm(direction)
                _, derivative, _ = guide.value_gradient(commands)
                epsilon = 1e-5
                plus = guide.value_gradient(commands + epsilon * direction, gradient=False)[0]
                minus = guide.value_gradient(commands - epsilon * direction, gradient=False)[0]
                expected = float((plus - minus) / (2 * epsilon))
                actual = float((derivative * model.tensor(direction)).sum())
                np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-10)
                self.assertIsNone(guide.value_gradient(commands, gradient=False)[1])

    def test_delayed_windows_and_zero_command_gaps_match_event_ode(self):
        rng = np.random.default_rng(751)
        for enabled in (False, True):
            with self.subTest(response=enabled):
                experiment, calibration = fixture(response=enabled, tau=np.linspace(80e-6, 700e-6, 32))
                acquisition = experiment["acquisition"]
                acquisition["line_start_s"] = acquisition["line_start_s"] + .0001
                acquisition["line_end_s"] = np.array([.001, .00225, .0037])
                model = ACDCModel(experiment, calibration, profile="cpu", precision="float64")
                guide = FieldGuidance(model, scale_Hz=130.)
                commands = rng.normal(size=(32, 3)) * .15
                value, _, rms = guide.value_gradient(commands)
                expected = independent_penalty(experiment, calibration, commands,
                                               guide.line_center_times_s, 130.)
                np.testing.assert_allclose([float(value), float(rms)], expected, rtol=2e-13)

    def test_quality_center_domain_ignores_corner_constraint_basis(self):
        _, _, model = make_model(response=True)
        original = FieldGuidance(model)
        model.B_quality = torch.full((207, model.nc), 1e8, dtype=model.dtype)
        altered = FieldGuidance(model)
        commands = torch.ones((model.nc, model.nstates), dtype=model.dtype) * .02
        for before, after in zip(original.value_gradient(commands), altered.value_gradient(commands)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_disabled_channels_and_all_off_do_not_change_bounds_or_maps(self):
        for all_off in (False, True):
            limits = np.full(32, 3.)
            limits[[1, 8]] = 0.
            if all_off:
                limits[:] = 0.
            experiment, calibration, model = make_model(limits=limits)
            original_maps = model.maps_numpy.copy()
            guide = FieldGuidance(model)
            commands = model.project_currents(np.full((32, 3), .02), 30., 500.)
            value, derivative, rms = guide.value_gradient(commands)
            self.assertTrue(torch.isfinite(value) and torch.isfinite(derivative).all() and torch.isfinite(rms))
            np.testing.assert_array_equal(commands[limits == 0], 0.)
            np.testing.assert_array_equal(model.numpy(model.max_current_A), limits)
            np.testing.assert_array_equal(model.maps_numpy, original_maps)
            if all_off:
                expected = independent_penalty(experiment, calibration, commands, guide.line_center_times_s, 100.)
                self.assertAlmostEqual(float(value), expected[0], places=13)

    def test_full_fov_alias_has_identical_image_but_nonzero_field_penalty(self):
        # One spin, one readout sample per line: 1000 Hz adds one whole phase
        # turn each 1 ms line, shifting it by an entire 8-pixel FOV.
        ny, esp, frequency = 8, .001, 1000.
        shape = (ny, 1)
        x = np.zeros(shape)
        y = (np.arange(ny)[:, None] - ny//2) * .01
        amplitude = np.zeros(shape)
        amplitude[3, 0] = 1.
        mask = amplitude > 0
        experiment = {
            "m0_proxy": amplitude,
            "geometry": {"X_m": x, "Y_m": y},
            "field": {"b0Hz": np.full(shape, frequency), "mask": mask, "simulation_mask": mask},
            "relaxation": {"T2star_s": np.inf},
            "acquisition": {"kx_mInv": x, "ky_mInv": (np.arange(ny)[:, None]-ny//2)/.08,
                "time_s": (np.arange(ny)[:, None]+.5)*esp, "TE_s": .004,
                "line_index": np.arange(1, ny+1)[:, None], "line_start_s": np.arange(ny)*esp,
                "line_end_s": np.arange(1, ny+1)*esp, "sequence_kind": "epi"},
        }
        calibration = {"acdc_maps": np.full(shape+(1,), frequency), "X_m": x, "Y_m": y, "units": "Hz/A"}
        for precision in ("float64", "float32"):
            with self.subTest(precision=precision):
                model = ACDCModel(deepcopy(experiment), calibration, profile="cpu", precision=precision)
                guide = FieldGuidance(model)
                zero = np.zeros((1, ny))
                cancelled = -np.ones((1, ny))
                image, _ = model.forward(zero)
                np.testing.assert_allclose(abs(image), abs(model.ideal_image), atol=2e-6)
                aliased, _, rms = guide.value_gradient(zero)
                corrected, derivative, corrected_rms = guide.value_gradient(cancelled)
                self.assertAlmostEqual(float(aliased), 100., places=4)
                self.assertAlmostEqual(float(rms), frequency, places=3)
                self.assertLess(float(corrected), 1e-12)
                self.assertLess(float(corrected_rms), 1e-5)
                self.assertLess(float(derivative.abs().max()), 1e-10)

    def test_rejects_invalid_scale_non_epi_and_wrong_current_shape(self):
        _, _, model = make_model()
        for scale in (0., -1., np.inf, np.nan, True, 1j, "100", [100.]):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                FieldGuidance(model, scale_Hz=scale)
        guide = FieldGuidance(model)
        with self.assertRaises(ValueError):
            guide.value_gradient(np.zeros((32, 2)))
        with self.assertRaises(ValueError):
            guide.value_gradient(np.zeros((32, 3), complex))
        model.sequence_kind = "gre"
        with self.assertRaisesRegex(ValueError, "single-excitation EPI"):
            FieldGuidance(model)
        model.sequence_kind, model.nstates = "epi", 1
        with self.assertRaisesRegex(ValueError, "at least two"):
            FieldGuidance(model)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
