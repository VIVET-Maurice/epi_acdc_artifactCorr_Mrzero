"""Independent physics and numerical checks for the Python ACDC backend.

Run from the repository with:
    .venv/bin/python -m unittest discover -s Experiment -p test_acdc_gpu.py -v
The small fixtures run on CPU; CUDA parity runs when CUDA is available.
"""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from acdc_gpu import ACDCModel, optimize_image, project_currents_tensor, response_matrices, select_device
from acdc_gpu_field import solve_field_gpu

torch.set_num_threads(1)


def fixture(kind="epi", response=False, tau=None):
    """Reordered Cartesian rows and alternating ADC time order, with 32 coils."""
    rng = np.random.default_rng(20261014)
    x, y = np.meshgrid(np.arange(4) * .03 - .06, np.arange(3) * .025 - .025)
    m0 = rng.uniform(.3, 1.2, x.shape)
    m0[0, 0] = 0
    support = m0 > 0
    quality = support.copy()
    quality[-1, -1] = False
    b0 = rng.normal(size=x.shape) * 35
    maps = rng.normal(size=x.shape + (32,)) * 14
    row_state = np.array([2, 0, 1])
    starts = np.array([0., .0013, .0025])
    ends = np.r_[starts[1:], .0038]
    rf_by_state = np.zeros(3)
    if kind == "gre":
        rf_by_state = np.array([0., .01, .02])
        starts = rf_by_state + .0004
        ends = rf_by_state + .0015
    sample_offsets = np.array([[.0002, .00028, .00036, .00044],
                               [.00044, .00036, .00028, .0002],
                               [.0002, .00028, .00036, .00044]])
    times = starts[row_state, None] + sample_offsets
    kx, ky = np.meshgrid((np.arange(4) - 2) / .12, (np.arange(3) - 1) / .075)
    experiment = {
        "m0_proxy": m0,
        "geometry": {"X_m": x, "Y_m": y},
        "field": {"b0Hz": b0, "simulation_mask": support, "mask": quality},
        "relaxation": {"T2star_s": rng.uniform(.025, .06, x.shape)},
        "acquisition": {
            "kx_mInv": kx, "ky_mInv": ky, "time_s": times,
            "TE_s": .0018 if kind == "epi" else .0009,
            "line_index": np.repeat((row_state + 1)[:, None], 4, axis=1),
            "line_start_s": starts, "line_end_s": ends,
            "rf_time_s": np.repeat(rf_by_state[row_state, None], 4, axis=1),
            "phase_offset_rad": rng.normal(size=x.shape) * .1,
            "sequence_kind": kind,
        },
        "current_response": {
            "enabled": response,
            "tau_s": 133e-6 / np.log(100) if tau is None else tau,
        },
    }
    calibration = {"acdc_maps": maps, "X_m": x.copy(), "Y_m": y.copy(),
                   "units": "Hz/A", "registration_confirmed": True}
    return experiment, calibration


def event_response(commands, starts, ends, times, tau):
    """Integrate the ODE over command events, independently of matrix formulas."""
    tau = np.broadcast_to(np.asarray(tau, float), (commands.shape[0],))
    events = [(float(start), 1, commands[:, state])
              for state, start in enumerate(starts)]
    events += [(float(end), 0, np.zeros(commands.shape[0]))
               for end in ends if np.isfinite(end)]
    events.sort(key=lambda event: event[:2])
    actuals, integrals = [], []
    for sample in np.asarray(times).ravel():
        actual = np.zeros(commands.shape[0])
        integral = np.zeros(commands.shape[0])
        command = np.zeros(commands.shape[0])
        last = 0.
        for event_time, _, new_command in events + [(float(sample), 2, None)]:
            if event_time > sample:
                continue
            dt = event_time - last
            integral += command * dt + (actual - command) * tau * (-np.expm1(-dt / tau))
            actual = command + (actual - command) * np.exp(-dt / tau)
            last = event_time
            if new_command is None:
                break
            command = new_command
        actuals.append(actual.copy())
        integrals.append(integral.copy())
    return np.asarray(actuals), np.asarray(integrals)


def direct_forward(experiment, calibration, commands, response=None, ideal=False):
    """Scalar NumPy Bloch sum, including RF history, decay and receive phase."""
    acq = experiment["acquisition"]
    active = experiment["field"]["simulation_mask"] & (experiment["m0_proxy"] > 0)
    x = experiment["geometry"]["X_m"][active]
    y = experiment["geometry"]["Y_m"][active]
    m0 = experiment["m0_proxy"][active]
    b0 = experiment["field"]["b0Hz"][active]
    maps = calibration["acdc_maps"][active]
    t2 = np.broadcast_to(experiment["relaxation"]["T2star_s"], active.shape)[active]
    starts, ends = acq["line_start_s"], acq["line_end_s"]
    rf = acq["rf_time_s"] if acq["sequence_kind"] == "gre" else np.zeros_like(acq["time_s"])
    enabled = experiment["current_response"]["enabled"] if response is None else response
    if enabled:
        tau = experiment["current_response"]["tau_s"]
        _, integral = event_response(commands, starts, ends, acq["time_s"], tau)
        _, before_rf = event_response(commands, starts, ends, rf, tau)
        integrated = integral - before_rf
    signal = np.zeros(acq["time_s"].shape, complex)
    for sample, time in enumerate(acq["time_s"].ravel()):
        index = np.unravel_index(sample, signal.shape)
        elapsed = time - rf[index]
        if enabled:
            accumulated = integrated[sample]
        else:
            accumulated = np.zeros(commands.shape[0])
            for state, (start, end) in enumerate(zip(starts, ends)):
                duration = max(0., min(time, end) - max(rf[index], start))
                accumulated += commands[:, state] * duration
        phase = x * acq["kx_mInv"][index] + y * acq["ky_mInv"][index]
        if not ideal:
            phase += b0 * elapsed + maps @ accumulated
        decay = np.exp(-(elapsed - acq["TE_s"]) / t2)
        signal[index] = np.sum(m0 * decay * np.exp(-2j * np.pi * phase)
                               * np.exp(1j * acq["phase_offset_rad"][index]))
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(signal))), signal


def cpu_model(experiment, calibration, **options):
    return ACDCModel(experiment, calibration, profile="cpu", precision="float64", **options)


class ResponseTests(unittest.TestCase):
    def test_step_reaches_measured_99_percent_and_has_correct_integral(self):
        tau = 133e-6 / np.log(100)
        h, r = response_matrices([0.], [np.inf], [0., 133e-6], [tau])
        self.assertEqual(h.shape, (1, 2, 1))
        np.testing.assert_allclose(r[0, :, 0], [0., .99], atol=2e-15, rtol=0)
        np.testing.assert_allclose(h[0, :, 0], [0., 133e-6 - .99 * tau], atol=1e-18, rtol=0)

    def test_transitions_gaps_channel_taus_and_rf_subtraction_match_ode(self):
        starts = np.array([0., 70e-6, 150e-6])
        ends = np.array([50e-6, 120e-6, 200e-6])
        times = np.array([0., 20e-6, 50e-6, 65e-6, 70e-6, 110e-6,
                          130e-6, 150e-6, 199e-6, 250e-6])
        rf = np.maximum(times - 80e-6, 0)
        tau = np.array([20e-6, 70e-6])
        commands = np.array([[1., -.25, .4], [-.5, .8, -1.]])
        h, r = response_matrices(starts, ends, times, tau, rf_time_s=rf)
        actual, integral = event_response(commands, starts, ends, times, tau)
        _, before_rf = event_response(commands, starts, ends, rf, tau)
        np.testing.assert_allclose(np.einsum("csn,cn->sc", r, commands), actual, atol=2e-15, rtol=0)
        np.testing.assert_allclose(np.einsum("csn,cn->sc", h, commands), integral - before_rf,
                                   atol=2e-19, rtol=0)

    def test_invalid_response_windows_and_taus_are_rejected(self):
        invalid = [([0., .1], [.2, .3], [1e-4]),
                   ([0., .1], [.1, .2], [0.]),
                   ([0., .1], [.1, .2], [np.nan])]
        for starts, ends, tau in invalid:
            with self.subTest(starts=starts, ends=ends, tau=tau), self.assertRaises(ValueError):
                response_matrices(starts, ends, [.15], tau)
        with self.assertRaises(ValueError):
            response_matrices([0.], [1.], [.1], [.001], rf_time_s=[.2])


class ForwardAndGradientTests(unittest.TestCase):
    def test_epi_and_gre_match_independent_scalar_forward(self):
        commands = np.random.default_rng(8).normal(size=(32, 3)) * .06
        cases = [("epi", False, None), ("gre", False, None),
                 ("epi", True, 133e-6 / np.log(100)),
                 ("epi", True, np.linspace(20e-6, 95e-6, 32))]
        for kind, enabled, tau in cases:
            with self.subTest(kind=kind, enabled=enabled, per_channel=np.ndim(tau) > 0):
                experiment, calibration = fixture(kind, enabled, tau)
                model = cpu_model(experiment, calibration, sample_chunk=5)
                actual = model.forward(commands)
                expected = direct_forward(experiment, calibration, commands)
                np.testing.assert_allclose(actual[0], expected[0], rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(actual[1], expected[1], rtol=2e-12, atol=2e-12)

    def test_response_override_and_zero_current_preserve_baseline(self):
        experiment, calibration = fixture(response=True, tau=np.linspace(20e-6, 90e-6, 32))
        model = cpu_model(experiment, calibration)
        commands = np.random.default_rng(9).normal(size=(32, 3)) * .08
        actual = model.forward(commands, response_enabled=False)
        expected = direct_forward(experiment, calibration, commands, response=False)
        np.testing.assert_allclose(actual[1], expected[1], rtol=2e-12, atol=2e-12)
        zero_response = model.forward(np.zeros_like(commands))
        zero_instant = model.forward(np.zeros_like(commands), response_enabled=False)
        np.testing.assert_allclose(zero_response[0], zero_instant[0], rtol=0, atol=1e-14)

    def test_normalized_magnitude_objective_and_analytic_gradient(self):
        rng = np.random.default_rng(31)
        for kind, enabled in (("epi", False), ("gre", False), ("epi", True)):
            with self.subTest(kind=kind, response=enabled):
                experiment, calibration = fixture(kind, enabled, np.linspace(20e-6, 90e-6, 32))
                model = cpu_model(experiment, calibration, sample_chunk=5)
                commands = rng.normal(size=(32, 3)) * .08
                value, gradient = model.objective_and_grad(commands)
                image, _ = direct_forward(experiment, calibration, commands)
                reference, _ = direct_forward(experiment, calibration, commands, ideal=True)
                expected = np.sum((np.abs(image) - np.abs(reference)) ** 2) / np.sum(np.abs(reference) ** 2)
                self.assertAlmostEqual(value, expected, places=12)
                self.assertEqual(gradient.shape, commands.shape)
                for _ in range(3):
                    direction = rng.normal(size=commands.shape)
                    direction /= np.linalg.norm(direction)
                    step = 1e-5
                    plus = model.objective_and_grad(commands + step * direction)[0]
                    minus = model.objective_and_grad(commands - step * direction)[0]
                    difference = (plus - minus) / (2 * step)
                    analytic = np.sum(gradient * direction)
                    np.testing.assert_allclose(analytic, difference, rtol=3e-6, atol=2e-10)

    def test_chunking_does_not_change_forward_or_gradient(self):
        experiment, calibration = fixture(response=True)
        small = cpu_model(experiment, calibration, sample_chunk=2)
        large = cpu_model(experiment, calibration, sample_chunk=12)
        commands = np.random.default_rng(4).normal(size=(32, 3)) * .08
        np.testing.assert_allclose(small.forward(commands)[1], large.forward(commands)[1], atol=2e-12, rtol=2e-12)
        v1, g1 = small.objective_and_grad(commands)
        v2, g2 = large.objective_and_grad(commands)
        self.assertAlmostEqual(v1, v2, places=13)
        np.testing.assert_allclose(g1, g2, atol=2e-12, rtol=2e-12)

    def test_waveform_retains_transition_history_and_final_settling_tail(self):
        tau = np.linspace(20e-6, 90e-6, 32)
        experiment, calibration = fixture(response=True, tau=tau)
        model = cpu_model(experiment, calibration)
        commands = np.random.default_rng(17).normal(size=(32, 3)) * .2
        starts, ends = experiment["acquisition"]["line_start_s"], experiment["acquisition"]["line_end_s"]
        times = np.array([ends[-1] + 50e-6, 0., starts[1] + 20e-6, ends[-1], .0003])
        sampled = model.sample_waveform(commands, times)
        actual, _ = event_response(commands, starts, ends, times, tau)
        np.testing.assert_allclose(sampled["actual_A"], actual.T, atol=2e-15, rtol=0)
        self.assertGreater(np.linalg.norm(sampled["actual_A"][:, 0]), 0.)
        np.testing.assert_array_equal(sampled["commanded_A"][:, 0], 0.)

    def test_float32_image_and_gradient_agree_with_float64(self):
        commands = np.random.default_rng(6).normal(size=(32, 3)) * .08
        for tau in (35e-6, np.linspace(20e-6, 90e-6, 32)):
            with self.subTest(per_channel=np.ndim(tau) > 0):
                experiment, calibration = fixture(response=True, tau=tau)
                reference = cpu_model(experiment, calibration)
                fast = ACDCModel(experiment, calibration, profile="cpu", precision="float32", sample_chunk=5)
                np.testing.assert_allclose(fast.forward(commands)[1], reference.forward(commands)[1],
                                           atol=2e-5, rtol=2e-5)
                value, gradient = fast.objective_and_grad(commands)
                ref_value, ref_gradient = reference.objective_and_grad(commands)
                np.testing.assert_allclose(value, ref_value, atol=2e-6, rtol=2e-5)
                np.testing.assert_allclose(gradient, ref_gradient, atol=2e-6, rtol=2e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable in this test environment")
    def test_cuda_forward_and_gradient_match_cpu(self):
        experiment, calibration = fixture(response=True, tau=np.linspace(20e-6, 90e-6, 32))
        cpu = cpu_model(experiment, calibration, sample_chunk=5)
        gpu = ACDCModel(experiment, calibration, profile="auto", precision="float64", sample_chunk=5)
        commands = np.random.default_rng(5).normal(size=(32, 3)) * .08
        np.testing.assert_allclose(gpu.forward(commands)[1], cpu.forward(commands)[1], atol=2e-11, rtol=2e-11)
        value_cpu, grad_cpu = cpu.objective_and_grad(commands)
        value_gpu, grad_gpu = gpu.objective_and_grad(commands)
        self.assertAlmostEqual(value_cpu, value_gpu, places=11)
        np.testing.assert_allclose(grad_gpu, grad_cpu, atol=2e-11, rtol=2e-11)


class ConstraintsAndOptimizationTests(unittest.TestCase):
    def test_box_l1_projection_keeps_saturated_coordinate(self):
        experiment, calibration = fixture()
        model = cpu_model(experiment, calibration, max_current_A=2.)
        commands = np.zeros((32, 3))
        commands[:2, :2] = [[10., -10.], [1., -1.]]
        projected = model.project_currents(commands, total_current_limit_A=2.)
        self.assertGreaterEqual(projected[0, 0], 1.9998)  # Retain the documented CSV margin.
        self.assertLessEqual(projected[0, 1], -1.9998)
        np.testing.assert_allclose(projected[1, :2], 0, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(projected[2:], 0)
        exact = project_currents_tensor(torch.tensor([[10., -10.], [1., -1.]], dtype=torch.float64),
                                        2., 2., current_margin_A=0.)
        np.testing.assert_allclose(exact.numpy(), [[2., -2.], [0., 0.]], atol=1e-12, rtol=0)

    def test_projection_obeys_channel_total_and_generated_field_limits(self):
        experiment, calibration = fixture()
        limits = np.linspace(.15, .5, 32)
        model = cpu_model(experiment, calibration, max_current_A=limits)
        commands = np.random.default_rng(18).normal(size=(32, 3)) * 3
        result = model.project_currents(commands, total_current_limit_A=1.2, field_limit_Hz=2.)
        basis = calibration["acdc_maps"][experiment["field"]["mask"]]
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((np.abs(result) <= limits[:, None] + 1e-12).all())
        self.assertTrue((np.abs(result).sum(axis=0) <= 1.2 + 1e-12).all())
        self.assertLessEqual(np.abs(basis @ result).max(), 2. + 1e-10)

    def test_optimizer_reduces_independent_image_error_and_stays_feasible(self):
        experiment, calibration = fixture()
        # One measured basis channel can exactly cancel the measured B0.
        calibration["acdc_maps"][..., 0] = experiment["field"]["b0Hz"]
        model = cpu_model(experiment, calibration, max_current_A=1.5)
        initial = np.zeros((32, 3))
        initial_value = model.objective_and_grad(initial)[0]
        result = optimize_image(model, initial, total_current_limit_A=3., field_limit_Hz=100.,
                                max_iterations=80, learning_rate_A=.05, patience=30)
        commands = result["currents_A"]
        final_value = model.objective_and_grad(commands)[0]
        self.assertLess(final_value, initial_value * .5)
        image, _ = direct_forward(experiment, calibration, commands)
        reference, _ = direct_forward(experiment, calibration, commands, ideal=True)
        independent_value = np.sum((np.abs(image) - np.abs(reference)) ** 2) / np.sum(np.abs(reference) ** 2)
        self.assertAlmostEqual(final_value, independent_value, places=12)
        self.assertTrue((np.abs(commands) <= 1.5 + 1e-12).all())
        self.assertTrue((np.abs(commands).sum(axis=0) <= 3. + 1e-10).all())
        basis = calibration["acdc_maps"][experiment["field"]["mask"]]
        self.assertLessEqual(np.abs(basis @ commands).max(), 100. + 1e-9)

    def test_zero_b0_target_can_stop_without_unnecessary_optimization(self):
        experiment, calibration = fixture()
        experiment["field"]["b0Hz"][:] = 0.
        model = cpu_model(experiment, calibration)
        result = optimize_image(model, np.zeros((32, 3)), total_current_limit_A=3.,
                                field_limit_Hz=100., target_mse=1e-8, max_iterations=10)
        self.assertLess(result["optimization_info"]["fval"], 1e-8)
        self.assertTrue(result["optimization_info"]["target_reached"])
        self.assertTrue(result["optimization_info"]["accepted"])

    def test_channel_specific_response_detects_transient_field_cancellation_loss(self):
        tau = np.full(32, 50e-6)
        tau[1] = 500e-6
        experiment, calibration = fixture(response=True, tau=tau)
        calibration["acdc_maps"][..., 0] = 100.
        calibration["acdc_maps"][..., 1] = -100.
        model = cpu_model(experiment, calibration)
        commands = np.zeros((32, 3))
        commands[:2] = 1.
        command_field = calibration["acdc_maps"] @ commands
        np.testing.assert_array_equal(command_field, 0.)
        diagnostics = model.response_diagnostics(commands, field_limit_Hz=10., total_current_limit_A=3.)
        self.assertFalse(diagnostics["actual_adc_field_limits_met"])
        self.assertGreater(diagnostics["maximum_actual_quality_field_Hz"], 10.)
        self.assertTrue(diagnostics["actual_adc_channel_limits_met"])


class DeviceSelectionTests(unittest.TestCase):
    """Exercise CUDA failures without requiring a GPU on the test machine."""

    def setUp(self):
        probe = torch.ones((2, 2))
        patches = {
            "available": patch.object(torch.cuda, "is_available", return_value=True),
            "properties": patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(
                name="Test GPU", major=12, minor=0, total_memory=16 * 1024**3)),
            "architectures": patch.object(torch.cuda, "get_arch_list", return_value=["sm_70", "sm_75"]),
            "probe": patch.object(torch, "ones", return_value=probe),
            "synchronize": patch.object(torch.cuda, "synchronize"),
        }
        self.mocks = {}
        for name, patcher in patches.items():
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def test_auto_requires_working_cuda_kernels_before_selecting_gpu(self):
        device, dtype, info = select_device()
        self.assertEqual(str(device), "cuda")
        self.assertEqual(dtype, torch.float32)
        self.assertEqual(info["gpu_name"], "Test GPU")
        self.assertIsNone(info["cpu_fallback_reason"])
        self.mocks["synchronize"].assert_called_once()

    def test_auto_incompatible_build_warns_and_uses_cpu_defaults(self):
        self.mocks["probe"].side_effect = RuntimeError(
            "CUDA error: no kernel image is available for execution on the device")
        with self.assertWarnsRegex(RuntimeWarning, "Falling back to CPU"):
            device, dtype, info = select_device()
        self.assertEqual(str(device), "cpu")
        self.assertEqual(dtype, torch.float64)
        self.assertEqual(info["precision"], "float64")
        self.assertEqual(info["gpu_name"], "CPU")
        self.assertIsNone(info["memory_GiB"])
        self.assertEqual(info["detected_gpu_name"], "Test GPU")
        self.assertEqual(info["gpu_capability"], "sm_120")
        self.assertEqual(info["cuda_build_architectures"], ["sm_70", "sm_75"])
        self.assertIn("no kernel image", info["cpu_fallback_reason"])

    def test_auto_synchronization_failure_preserves_explicit_precision(self):
        self.mocks["synchronize"].side_effect = RuntimeError("CUDA error: invalid device function")
        with self.assertWarns(RuntimeWarning):
            device, dtype, info = select_device(precision="float32")
        self.assertEqual(str(device), "cpu")
        self.assertEqual(dtype, torch.float32)
        self.assertIn("invalid device function", info["cpu_fallback_reason"])

    def test_explicit_gpu_profiles_raise_with_compatibility_diagnostics(self):
        failure = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        self.mocks["probe"].side_effect = failure
        for profile in ("v100", "gtx1660ti", "blackwell"):
            with self.subTest(profile=profile), self.assertRaisesRegex(RuntimeError, "Test GPU.*sm_120") as raised:
                select_device(profile)
            self.assertIs(raised.exception.__cause__, failure)
            self.assertIn("sm_70, sm_75", str(raised.exception))
            self.assertIn("restart the kernel", str(raised.exception))

    def test_cpu_profile_never_probes_cuda(self):
        device, dtype, _ = select_device("cpu")
        self.assertEqual(str(device), "cpu")
        self.assertEqual(dtype, torch.float64)
        self.mocks["available"].assert_not_called()
        self.mocks["properties"].assert_not_called()
        self.mocks["probe"].assert_not_called()

    def test_unavailable_cuda_uses_cpu_only_for_auto(self):
        self.mocks["available"].return_value = False
        device, dtype, _ = select_device()
        self.assertEqual(str(device), "cpu")
        self.assertEqual(dtype, torch.float64)
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            select_device("v100")
        self.mocks["probe"].assert_not_called()

    def test_required_cuda_never_falls_back_after_incompatible_kernel(self):
        failure = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        self.mocks["probe"].side_effect = failure
        with self.assertRaisesRegex(RuntimeError, "Test GPU.*sm_120") as raised:
            select_device("auto", require_cuda=True)
        self.assertIs(raised.exception.__cause__, failure)

    def test_required_cuda_rejects_missing_gpu_and_explicit_cpu(self):
        self.mocks["available"].return_value = False
        with self.assertRaisesRegex(RuntimeError, "this run requires a GPU"):
            select_device("auto", require_cuda=True)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            select_device("cpu", require_cuda=True)
        with self.assertRaisesRegex(ValueError, "boolean"):
            select_device(require_cuda="yes")
        self.mocks["probe"].assert_not_called()

    def test_required_auto_accepts_each_supported_gpu_architecture(self):
        for name, major, minor in (("Tesla V100", 7, 0), ("GTX 1660 Ti", 7, 5),
                                   ("NVIDIA RTX PRO 6000 Blackwell", 12, 0)):
            with self.subTest(name=name):
                self.mocks["properties"].return_value = SimpleNamespace(
                    name=name, major=major, minor=minor, total_memory=16 * 1024**3)
                device, dtype, info = select_device("auto", require_cuda=True)
                self.assertEqual(device.type, "cuda")
                self.assertEqual(dtype, torch.float32)
                self.assertTrue(info["require_cuda"])
                self.assertEqual(info["gpu_name"], name)


class ValidationTests(unittest.TestCase):
    def test_unknown_profiles_precision_and_invalid_resource_options_fail(self):
        experiment, calibration = fixture()
        for options in ({"profile": "unknown-gpu"}, {"precision": "float16"},
                        {"memory_fraction": 0.}, {"memory_fraction": 1.1},
                        {"sample_chunk": 0}):
            settings = {"profile": "cpu", "precision": "float64", **options}
            with self.subTest(options=options), self.assertRaises(ValueError):
                ACDCModel(experiment, calibration, **settings)

    def test_missing_calibration_coverage_and_mismatched_geometry_fail(self):
        experiment, calibration = fixture()
        for mutation in ("nan", "geometry", "zero_channel"):
            changed = deepcopy(calibration)
            if mutation == "nan":
                changed["acdc_maps"][1, 1, 0] = np.nan
            elif mutation == "geometry":
                changed["X_m"][1, 1] += 1e-5
            else:
                changed["acdc_maps"][..., 0] = 0.
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cpu_model(experiment, changed)

    def test_unknown_map_values_outside_simulation_support_are_allowed(self):
        experiment, calibration = fixture()
        calibration["acdc_maps"][0, 0, :] = np.nan
        model = cpu_model(experiment, calibration)
        self.assertTrue(np.isfinite(model.forward(np.zeros((32, 3)))[0]).all())

    def test_first_order_gre_response_is_explicitly_rejected(self):
        experiment, calibration = fixture("gre", response=True)
        with self.assertRaises(ValueError):
            cpu_model(experiment, calibration)

    def test_nonfinite_or_wrong_shaped_currents_are_rejected(self):
        experiment, calibration = fixture()
        model = cpu_model(experiment, calibration)
        for commands in (np.zeros((3, 32)), np.full((32, 3), np.nan)):
            with self.subTest(shape=commands.shape), self.assertRaises(ValueError):
                model.forward(commands)


class FieldFitTests(unittest.TestCase):
    def solve(self, basis, target, **options):
        settings = dict(settings={"profile": "cpu", "precision": "float64"},
                        max_current_A=5., total_current_limit_A=10.,
                        max_iterations=1500, tolerance=1e-10, history_every=5)
        settings.update(options)
        return solve_field_gpu(basis, target, **settings)

    def test_static_fit_matches_numpy_least_squares(self):
        rng = np.random.default_rng(73)
        basis = rng.normal(size=(15, 4))
        target = basis @ np.array([.3, -.4, .2, .1]) + rng.normal(size=15) * .03
        result = self.solve(basis, target)
        expected = np.linalg.lstsq(basis, target, rcond=None)[0]
        np.testing.assert_allclose(result["currents_A"][:, 0], expected, atol=1e-9, rtol=0)
        residual = basis @ result["currents_A"][:, 0] - target
        self.assertAlmostEqual(result["loss"], np.mean(residual**2), places=12)
        self.assertAlmostEqual(result["nrmse"], np.linalg.norm(residual) / np.linalg.norm(target), places=12)

    def test_ridge_units_match_closed_form_physical_objective(self):
        rng = np.random.default_rng(19)
        basis = rng.normal(size=(11, 3)) * 7.
        target = basis @ np.array([.6, -.2, .4])
        ridge = 12.
        expected = np.linalg.solve(basis.T @ basis / 11 + ridge / 3 * np.eye(3),
                                   basis.T @ target / 11)
        result = self.solve(basis, target, ridge=ridge)
        commands = result["currents_A"][:, 0]
        np.testing.assert_allclose(commands, expected, atol=1e-9, rtol=0)
        direct = np.mean((basis @ commands - target)**2) + ridge * np.mean(commands**2)
        self.assertAlmostEqual(result["loss"], direct, places=11)

    def test_box_l1_constrained_fit_matches_independent_scipy_solution(self):
        from scipy.optimize import Bounds, LinearConstraint, minimize

        rng = np.random.default_rng(29)
        basis = rng.normal(size=(9, 3))
        target = basis @ np.array([1.5, -.8, .9])
        n = basis.shape[1]
        eye = np.eye(n)
        matrix = np.block([[eye, -eye], [-eye, -eye], [np.zeros((1, n)), np.ones((1, n))]])
        upper = np.r_[np.zeros(2 * n), 1.3]
        linear = LinearConstraint(matrix, -np.inf, upper)

        def objective(value):
            residual = basis @ value[:n] - target
            return np.mean(residual**2)

        def gradient(value):
            return np.r_[2 * basis.T @ (basis @ value[:n] - target) / target.size, np.zeros(n)]

        independent = minimize(objective, np.zeros(2 * n), jac=gradient,
                               bounds=Bounds(np.r_[np.full(n, -.7), np.zeros(n)],
                                             np.r_[np.full(n, .7), np.full(n, np.inf)]),
                               constraints=[linear], method="SLSQP",
                               options={"ftol": 1e-13, "maxiter": 300})
        self.assertTrue(independent.success, independent.message)
        result = self.solve(basis, target, max_current_A=.7, total_current_limit_A=1.3)
        np.testing.assert_allclose(result["currents_A"][:, 0], independent.x[:n], atol=3e-4, rtol=0)
        self.assertTrue((np.abs(result["currents_A"]) <= .7).all())
        self.assertLess(np.abs(result["currents_A"]).sum(), 1.3)

    def test_dynamic_response_fit_matches_direct_objective_at_every_adc(self):
        rng = np.random.default_rng(42)
        basis = rng.normal(size=(7, 2)) * 3.
        target = basis @ np.array([.3, -.2])
        starts = np.array([0., 80e-6, 160e-6])
        ends = np.r_[starts[1:], 240e-6]
        times = np.linspace(5e-6, 235e-6, 19)
        _, response = response_matrices(starts, ends, times, [25e-6, 70e-6])
        ridge = .02
        result = self.solve(basis, target, response=response, ridge=ridge,
                            initial_currents_A=np.zeros((2, 3)))
        commands = result["currents_A"]
        actual = np.einsum("cts,cs->ct", response, commands)
        residual = basis @ actual - target[:, None]
        direct = np.mean(residual**2) + ridge * np.mean(commands**2)
        self.assertAlmostEqual(result["loss"], direct, places=11)
        self.assertAlmostEqual(result["nrmse"], np.sqrt(np.mean(residual**2) / np.mean(target**2)), places=11)
        self.assertLess(result["loss"], np.mean(target**2) * .25)

    def test_zero_field_and_nominal_generated_field_bound(self):
        basis = np.array([[1., 2.], [-3., 1.], [1., -1.]])
        zero = self.solve(basis, np.zeros(3))
        np.testing.assert_array_equal(zero["currents_A"], 0.)
        self.assertIsNone(zero["nrmse"])
        self.assertEqual(zero["loss"], 0.)
        result = self.solve(basis, np.array([20., -10., 5.]), field_limit_Hz=2.,
                            max_current_A=2., total_current_limit_A=3.)
        self.assertLessEqual(np.abs(basis @ result["currents_A"]).max(), 2.)
        self.assertTrue(np.isfinite(result["loss"]))

    def test_float32_shared_and_per_channel_response_agree_with_float64(self):
        basis = np.array([[3., 1.], [-1., 2.], [2., -.5], [.5, 1.5]])
        target = basis @ np.array([.3, -.2])
        starts = np.array([0., 80e-6, 160e-6])
        ends = np.r_[starts[1:], 240e-6]
        times = np.linspace(5e-6, 235e-6, 19)
        for tau in ([35e-6, 35e-6], [25e-6, 70e-6]):
            with self.subTest(tau=tau):
                _, response = response_matrices(starts, ends, times, tau)
                reference = self.solve(basis, target, response=response, ridge=.02)
                fast = self.solve(basis, target, response=response, ridge=.02,
                                  settings={"profile": "cpu", "precision": "float32"},
                                  tolerance=1e-7)
                np.testing.assert_allclose(fast["loss"], reference["loss"], atol=2e-6, rtol=1e-5)
                actual = np.einsum("cts,cs->ct", response, fast["currents_A"])
                direct = np.mean((basis @ actual - target[:, None])**2) + .02 * np.mean(fast["currents_A"]**2)
                np.testing.assert_allclose(fast["loss"], direct, atol=2e-6, rtol=1e-5)

    def test_invalid_response_and_nonfinite_fields_are_rejected(self):
        for response in (np.eye(3), np.zeros((1, 3, 2)), np.full((2, 3, 2), np.nan)):
            with self.subTest(shape=response.shape), self.assertRaises(ValueError):
                self.solve(np.ones((4, 2)), np.ones(4), response=response)
        with self.assertRaises(ValueError):
            self.solve(np.full((4, 2), np.nan), np.ones(4))


if __name__ == "__main__":
    unittest.main()
