"""Independent waveform quadrature and continuous-time feasibility checks."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from acdc_hardware import HardwareConstraints


def tiny_model(tau=(50e-6, 150e-6), *, ends_at_cycle=False):
    """Two channels, a sign change, a zero-command gap, and final switch-off."""
    model = SimpleNamespace(
        nc=2, nstates=3, device=torch.device("cpu"), dtype=torch.float64,
        current_response={"enabled": True, "tau_s": np.asarray(tau)},
        line_start_s=np.array([0., 80e-6, 240e-6]),
        line_end_s=np.array([80e-6, 160e-6, 800e-6 if ends_at_cycle else 320e-6]),
        max_current_A=torch.tensor([3., 3.], dtype=torch.float64),
        B_quality=torch.tensor([[20., -20.], [10., 8.]], dtype=torch.float64),
    )
    model.tensor = lambda value: torch.as_tensor(value, dtype=model.dtype)
    model.numpy = lambda value: value.detach().cpu().numpy()

    def check_currents(value):
        result = model.tensor(value)
        if result.shape != (model.nc, model.nstates) or not bool(torch.isfinite(result).all()):
            raise ValueError("Bad current array.")
        return result

    model._check_currents = check_currents
    return model


def dense_waveform(model, commands, duration_s, count=160001):
    """Event recurrence independent of the production response matrices."""
    times = np.linspace(0, duration_s, count)
    actual = np.zeros((model.nc, count))
    boundaries = np.unique(np.r_[0, model.line_start_s, model.line_end_s, duration_s])
    current = np.zeros(model.nc)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        active = (model.line_start_s <= start) & (model.line_end_s > start)
        target = commands[:, active].sum(1)
        selected = (times >= start) & (times <= end)
        actual[:, selected] = (target[:, None] + (current - target)[:, None]
                               * np.exp(-(times[selected] - start) / model.current_response["tau_s"][:, None]))
        current = target + (current - target) * np.exp(-(end - start) / model.current_response["tau_s"])
    return times, actual


class PhysicalCurrentTests(unittest.TestCase):
    def test_absolute_and_square_integrals_match_dense_quadrature(self):
        model = tiny_model()
        commands = np.array([[1.2, -1.1, .7], [-.6, 1.7, -.8]])
        hardware = HardwareConstraints(model, {"cycle_duration_s": 800e-6})
        diagnostics = hardware.diagnostics(commands, 20.)
        times, actual = dense_waveform(model, commands, 800e-6)
        average_abs = np.trapezoid(np.abs(actual), times, axis=1) / times[-1]
        average_square = np.trapezoid(actual**2, times, axis=1) / times[-1]
        np.testing.assert_allclose(diagnostics["avg_abs_current_A"], average_abs, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(diagnostics["mean_square_current_A2"], average_square, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(diagnostics["end_current_A"], actual[:, -1], rtol=1e-12, atol=1e-12)
        # The zero-command gap and final decay must contribute to heating.
        occupied = np.zeros_like(times, dtype=bool)
        for start, end in zip(model.line_start_s, model.line_end_s):
            occupied |= (times >= start) & (times < end)
        self.assertTrue(np.all(np.trapezoid(np.abs(actual[:, ~occupied]), times[~occupied], axis=1) > 0))

    def test_slew_voltage_and_power_include_gaps_and_return_to_zero(self):
        model = tiny_model()
        commands = np.array([[1.2, -1.1, .7], [-.6, 1.7, -.8]])
        resistance, inductance = np.array([.3, .5]), np.array([30e-6, 80e-6])
        hardware = HardwareConstraints(model, dict(cycle_duration_s=800e-6,
                                                    coil_R_ohm=resistance, coil_L_H=inductance))
        result = hardware.diagnostics(commands, 20.)
        times, actual = dense_waveform(model, commands, 800e-6)
        # Differences approximate the physical derivative, including command edges.
        derivative = np.diff(actual, axis=1) / np.diff(times)
        midpoint_current = (actual[:, :-1] + actual[:, 1:]) / 2
        voltage = resistance[:, None] * midpoint_current + inductance[:, None] * derivative
        np.testing.assert_allclose(result["peak_slew_A_per_s"], np.abs(derivative).max(1), rtol=6e-5)
        np.testing.assert_allclose(result["peak_coil_voltage_V"], np.abs(voltage).max(1), rtol=6e-5)
        square = np.trapezoid(actual**2, times, axis=1) / times[-1]
        np.testing.assert_allclose(result["average_coil_power_W"], resistance * square, rtol=2e-8)

    def test_final_switch_off_at_TR_still_constrains_slew(self):
        model = tiny_model(tau=(100e-6, 100e-6), ends_at_cycle=True)
        # Small preceding step: its onset is mild, but switching off a near-1A
        # settled physical current at TR imposes another derivative constraint.
        commands = np.array([[.9, .9, 1.], [.9, .9, 1.]])
        hardware = HardwareConstraints(model, {"cycle_duration_s": 800e-6})
        metrics = hardware._metrics(model.tensor(commands))
        _, first, last = hardware._segments(model.tensor(commands))
        np.testing.assert_allclose(first[:, -1], last[:, -1])
        self.assertGreater(float((last[:, -1] / hardware.tau[:, 0]).max()), 9900.)
        self.assertGreater(float(metrics["peak_slew_A_per_s"].max()), 9900.)

    def test_unequal_tau_bounds_cover_dense_current_and_field_transients(self):
        model = tiny_model(tau=(15e-6, 150e-6))
        commands = np.array([[1.7, -1.5, .9], [1.7, -1.5, .9]])
        hardware = HardwareConstraints(model, {"cycle_duration_s": 800e-6})
        result = hardware.diagnostics(commands, 20., 100.)
        _, actual = dense_waveform(model, commands, 800e-6)
        self.assertGreaterEqual(result["peak_total_current_bound_A"] + 1e-12, np.abs(actual).sum(0).max())
        field = model.numpy(model.B_quality) @ actual
        self.assertGreaterEqual(result["peak_field_bound_Hz"] + 1e-12, np.abs(field).max())
        # Unequal taus can generate a field where equal commands cancel exactly.
        self.assertGreater(np.abs(field[0]).max(), 10.)

    def test_common_tau_endpoint_bounds_are_tight(self):
        model = tiny_model(tau=(60e-6, 60e-6))
        commands = np.array([[1.7, -1.5, .9], [-.8, .6, 1.2]])
        hardware = HardwareConstraints(model, {"cycle_duration_s": 800e-6})
        result = hardware.diagnostics(commands, 20., 100.)
        _, actual = dense_waveform(model, commands, 800e-6)
        self.assertAlmostEqual(result["peak_total_current_bound_A"], np.abs(actual).sum(0).max(), places=12)
        self.assertAlmostEqual(result["peak_field_bound_Hz"], np.abs(model.numpy(model.B_quality) @ actual).max(), places=12)

    def test_projection_enforces_all_optional_limits_and_preserves_commands(self):
        model = tiny_model(tau=(15e-6, 150e-6))
        config = dict(cycle_duration_s=800e-6, max_slew_A_per_s=[4000., 3000.],
                      avg_abs_current_A=[.04, .03], coil_R_ohm=[.3, .5],
                      coil_L_H=[30e-6, 80e-6], max_voltage_V=[.2, .15],
                      max_coil_power_W=[.001, .002])
        hardware = HardwareConstraints(model, config)
        original = model.tensor([[1.7, -1.5, .9], [1.7, -1.5, .9]])
        projected = hardware.additional_projection(original, .3, .5)
        result = hardware.diagnostics(projected, .3, .5)
        self.assertTrue(result["all_limits_met"], result["checks"])
        ratio = model.numpy(projected / original)
        np.testing.assert_allclose(ratio, ratio[0, 0], rtol=1e-13)
        self.assertLess(ratio[0, 0], 1.)

    def test_zero_currents_and_already_feasible_currents(self):
        model = tiny_model()
        hardware = HardwareConstraints(model, {"cycle_duration_s": 800e-6})
        for value in (0., .001):
            currents = model.tensor(np.full((2, 3), value))
            result = hardware.additional_projection(currents, 20., 100.)
            torch.testing.assert_close(result, currents, rtol=0, atol=0)
            self.assertTrue(hardware.diagnostics(result, 20., 100.)["all_limits_met"])


class HardwareInputTests(unittest.TestCase):
    def test_invalid_or_missing_hardware_parameters_are_rejected(self):
        model = tiny_model()
        for config in ({"coil_R_ohm": [-1., .3]}, {"max_slew_A_per_s": 0.},
                       {"max_voltage_V": 2.}, {"max_coil_power_W": 1.},
                       {"avg_abs_current_A": .2}, {"cycle_duration_s": 50e-6},
                       {"coil_L_H": [1e-6, 2e-6, 3e-6]}, {"dac_resolution_A": np.nan}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                HardwareConstraints(model, config)

    def test_minimum_update_interval_checks_gaps_and_adjacent_boundaries(self):
        model = tiny_model()
        # Coincident end/start events are one DAC update, not zero separation.
        hardware = HardwareConstraints(model)
        self.assertTrue(hardware.timing_report["update_intervals_met"])
        self.assertEqual(hardware.timing_report["effective_DAC_resolution_A"], .0006)
        model.line_start_s[2] = model.line_end_s[1] + 4e-6
        with self.assertRaisesRegex(ValueError, "DAC update"):
            HardwareConstraints(model)

    def test_explicit_first_order_response_required(self):
        model = tiny_model()
        model.current_response["enabled"] = False
        with self.assertRaisesRegex(ValueError, "enabled"):
            HardwareConstraints(model)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
