"""Checks for Haotian's NumPy waveform solver with channel response matrices."""

import ast
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "ACDC_shim_static_dyn_haotian"))
from ACDC_Optim import build_conv_matrix, solve_shim_waveform_fista
from ACDC_Optim.ACDC_optimization import _project_onto_box_l1ball


def response_fixture(starts, ends, samples, tau):
    """Exact step differences, used only to build small physical solver fixtures."""
    tau = np.asarray(tau)[:, None, None]
    elapsed_start = np.maximum(samples[None, :, None] - starts[None, None, :], 0)
    elapsed_end = np.maximum(samples[None, :, None] - ends[None, None, :], 0)
    return -np.expm1(-elapsed_start / tau) + np.expm1(-elapsed_end / tau)


def notebook_response_helper():
    """Load only the notebook's pure function; never execute data or export cells."""
    notebook = json.loads((Path(__file__).parent / "02_acdc_experiment_step_by_step_v3.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "acdc-dynamic-b0-target")
    definitions = ast.parse("".join(cell["source"]))
    function = next(node for node in definitions.body if isinstance(node, ast.FunctionDef)
                    and node.name == "first_order_response_matrix")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_response_helper", "exec"), namespace)
    return namespace[function.name]


class NotebookResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response = staticmethod(notebook_response_helper())

    def test_measured_step_reaches_99_percent_at_133_microseconds(self):
        tau = 133e-6 / np.log(100)
        matrix = self.response(np.array([0, 133e-6]), np.array([0]), np.array([1e-3]),
                               np.array([tau]))
        self.assertEqual(matrix.shape, (1, 2, 1))
        self.assertAlmostEqual(matrix[0, 0, 0], 0)
        self.assertAlmostEqual(matrix[0, 1, 0], 0.99)

    def test_each_channel_uses_its_own_time_constant(self):
        tau = np.array([133e-6 / np.log(100), 133e-6 / np.log(10)])
        matrix = self.response(np.array([133e-6]), np.array([0]), np.array([1e-3]), tau)
        np.testing.assert_allclose(matrix[:, 0, 0], [0.99, 0.9], atol=1e-14, rtol=0)

    def test_transition_memory_and_gaps_match_event_recurrence(self):
        starts = np.array([0, 70e-6, 150e-6])
        ends = np.array([50e-6, 120e-6, 200e-6])
        samples = np.array([0, 20e-6, 50e-6, 65e-6, 70e-6, 110e-6,
                            130e-6, 150e-6, 199e-6, 250e-6])
        tau = np.array([20e-6, 70e-6])
        commands = np.array([[1, -0.5], [-0.25, 0.8], [0.4, -1]])
        matrix = self.response(samples, starts, ends, tau)
        predicted = np.einsum("cij,jc->ic", matrix, commands)
        events = sorted([(start, 1, commands[state]) for state, start in enumerate(starts)]
                        + [(end, 0, np.zeros(2)) for end in ends], key=lambda event: event[:2])
        expected = []
        for sample in samples:
            actual, command, last = np.zeros(2), np.zeros(2), 0.0
            for event_time, _, next_command in events:
                if event_time > sample:
                    break
                actual = command + (actual - command) * np.exp(-(event_time - last) / tau)
                command, last = next_command, event_time
            expected.append(command + (actual - command) * np.exp(-(sample - last) / tau))
        np.testing.assert_allclose(predicted, expected, atol=2e-15, rtol=0)

    def test_disabled_response_is_an_instantaneous_interval_indicator(self):
        starts = np.array([0, 50e-6])
        ends = np.array([50e-6, 100e-6])
        samples = np.array([0, 25e-6, 50e-6, 100e-6])
        matrix = self.response(samples, starts, ends, np.array([20e-6, 70e-6]), enabled=False)
        expected = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
        np.testing.assert_array_equal(matrix, np.repeat(expected[None], 2, axis=0))


class DynamicB0SolverTests(unittest.TestCase):
    def setUp(self):
        self.random = np.random.default_rng(1729)

    def solve(self, target, basis, matrix=None, **options):
        settings = dict(amp_limit=2.0, l1_limit=4.0, ridge_x=1e-6,
                        max_iter=500, verbose=False, tol=1e-12)
        settings.update(options)
        return solve_shim_waveform_fista(target, basis, conv_matrix=matrix, **settings)

    def test_shared_and_repeated_channel_matrices_match(self):
        times = np.arange(7) * 50e-6
        matrix = response_fixture(times, times + 50e-6, times + 25e-6, [30e-6])[0]
        basis = self.random.normal(size=(3, 9))
        target = self.random.normal(size=(7, 9))
        shared = self.solve(target, basis, matrix)
        separate = self.solve(target, basis, np.repeat(matrix[None], 3, axis=0))
        np.testing.assert_allclose(shared[0], separate[0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(shared[1:], separate[1:], atol=1e-12, rtol=1e-12)

    def test_identity_channel_response_matches_disabled_response(self):
        basis = self.random.normal(size=(3, 8))
        target = self.random.normal(size=(5, 8))
        disabled = self.solve(target, basis)
        identity = self.solve(target, basis, np.repeat(np.eye(5)[None], 3, axis=0))
        np.testing.assert_allclose(disabled[0], identity[0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(disabled[1:], identity[1:], atol=1e-12, rtol=1e-12)

    def test_channel_response_recovers_known_field_and_currents(self):
        starts = np.array([0, 50e-6, 130e-6, 210e-6])
        ends = np.r_[starts[1:], 310e-6]
        samples = (starts + ends) / 2
        matrix = response_fixture(starts, ends, samples, [25e-6, 90e-6])
        commands = np.array([[0.3, -0.1], [-0.2, 0.4], [0.1, 0.2], [0.2, -0.3]])
        target = np.column_stack([matrix[channel] @ commands[:, channel] for channel in range(2)])
        actual, nrmse, loss = self.solve(target, np.eye(2), matrix, ridge_x=0,
                                       max_iter=2000, tol=1e-13)
        np.testing.assert_allclose(actual, commands, atol=2e-6, rtol=0)
        self.assertLess(nrmse, 1e-6)
        self.assertLess(loss, 1e-12)

    def test_rectangular_channel_response_recovers_command_states(self):
        response = notebook_response_helper()
        starts = np.array([0, 80e-6, 160e-6])
        ends = np.r_[starts[1:], 240e-6]
        samples = np.linspace(5e-6, 235e-6, 19)
        matrix = response(samples, starts, ends, np.array([25e-6, 70e-6]))
        commands = np.array([[0.3, -0.1], [-0.2, 0.4], [0.1, 0.2]])
        target = np.column_stack([matrix[channel] @ commands[:, channel] for channel in range(2)])
        fitted, nrmse, _ = self.solve(target, np.eye(2), matrix, ridge_x=0,
                                      max_iter=2000, tol=1e-13)
        self.assertEqual(fitted.shape, (3, 2))
        np.testing.assert_allclose(fitted, commands, atol=1e-7, rtol=0)
        self.assertLess(nrmse, 1e-7)

    def test_rectangular_shared_and_channel_responses_match(self):
        response = notebook_response_helper()
        starts = np.array([0, 80e-6, 160e-6])
        ends = np.r_[starts[1:], 240e-6]
        samples = np.linspace(5e-6, 235e-6, 19)
        matrix = response(samples, starts, ends, np.array([30e-6]))[0]
        basis = self.random.normal(size=(3, 9))
        target = self.random.normal(size=(len(samples), 9))
        shared = self.solve(target, basis, matrix)
        channels = self.solve(target, basis, np.repeat(matrix[None], 3, axis=0))
        self.assertEqual(shared[0].shape, (3, 3))
        np.testing.assert_allclose(shared[0], channels[0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(shared[1:], channels[1:], atol=1e-12, rtol=1e-12)

    def test_kernel_path_matches_its_square_dense_response(self):
        kernel = np.array([0.6, 0.3, 0.1])
        matrix = build_conv_matrix(kernel, 4)
        target = matrix @ np.array([[0.3, -0.1], [-0.2, 0.4], [0.1, 0.2], [0.2, -0.3]])
        dense = self.solve(target, np.eye(2), matrix, max_iter=2000)
        convolution = solve_shim_waveform_fista(
            target, np.eye(2), sirf_kernel=kernel, amp_limit=2, l1_limit=4,
            ridge_x=1e-6, max_iter=2000, tol=1e-12, verbose=False)
        np.testing.assert_allclose(dense[0], convolution[0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(dense[1:], convolution[1:], atol=1e-9, rtol=0)

    def test_box_and_l1_projection_preserves_saturated_coordinate(self):
        projected = _project_onto_box_l1ball(np.array([[10.0, 1.0], [-10.0, -1.0]]), 2, 2)
        np.testing.assert_allclose(projected, [[2, 0], [-2, 0]], atol=1e-14, rtol=0)

    def test_projection_is_feasible_and_satisfies_kkt_threshold(self):
        original = self.random.normal(size=(12, 6)) * 5
        projected = _project_onto_box_l1ball(original, 1.5, 3)
        self.assertTrue((np.abs(projected) <= 1.5).all())
        self.assertTrue((np.abs(projected).sum(axis=1) <= 3).all())
        for before, after in zip(original, projected):
            active = (np.abs(after) > 1e-10) & (np.abs(after) < 1.5 - 1e-10)
            if active.any():
                threshold = np.mean(np.abs(before[active]) - np.abs(after[active]))
                expected = np.sign(before) * np.minimum(1.5, np.maximum(np.abs(before) - threshold, 0))
                np.testing.assert_allclose(after, expected, atol=1e-12, rtol=0)

    def test_constrained_fista_and_final_loss_match_direct_objective(self):
        target = np.array([[10.0, 1.0], [-10.0, -1.0]])
        matrix = np.repeat(np.eye(2)[None], 2, axis=0)
        commands, nrmse, loss = self.solve(target, np.eye(2), matrix,
                                         amp_limit=2, l1_limit=2, ridge_x=0)
        np.testing.assert_allclose(commands, [[2, 0], [-2, 0]], atol=1e-12, rtol=0)
        residual = commands - target
        self.assertAlmostEqual(loss, 0.5 * np.sum(residual**2))
        self.assertAlmostEqual(nrmse, np.linalg.norm(residual) / np.linalg.norm(target))

    def test_response_matrices_require_finite_matching_dimensions(self):
        target, basis = np.ones((3, 4)), np.ones((2, 4))
        for invalid in (np.eye(2), np.zeros((1, 3, 3)), np.zeros((3, 0)),
                        np.zeros((2, 3, 0)), np.full((2, 3, 3), np.nan)):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    self.solve(target, basis, invalid)


if __name__ == "__main__":
    unittest.main()
