"""B0 sign, exact spatial reduction, timing and pinned upstream solvers."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from field_shim_waveforms import (UPSTREAM_COMMIT, amplifier_sirf,
                                  compress_field_target, solve_field_shim_waveforms)
from _vendor.acdc_optim import (solve_shim_static_qp,
                                solve_shim_waveform_fista_torch)
from _vendor.acdc_optim.ACDC_optimization import _sirf_conv1d_numpy


def tiny_field_model(device="cpu"):
    maps = np.array([[[20., 3.], [-5., 10.]],
                     [[15., -7.], [2., 13.]]])
    # A reachable part plus an orthogonal component tests full-field NRMSE.
    b0 = (maps @ np.array([.35, -.2])) + np.array([[.1, .2], [.3, .4]])
    return SimpleNamespace(
        nc=2, nstates=2, device=torch.device(device), sequence_kind="epi",
        quality_mask=np.ones((2, 2), bool), b0_numpy=b0, maps_numpy=maps,
        current_response={"enabled": True, "tau_s": np.array([30e-6, 30e-6])},
        max_current_A=torch.tensor([3., 3.], dtype=torch.float64),
        line_start_s=np.array([40e-6, 240e-6]),
        line_end_s=np.array([240e-6, 540e-6]),
    )


class SpatialReductionTests(unittest.TestCase):
    def test_qr_preserves_full_convolved_objective_including_orthogonal_field(self):
        rng = np.random.default_rng(5621)
        for points, channels, deficient in [(29, 4, False), (29, 4, True), (3, 7, False)]:
            with self.subTest(points=points, channels=channels, deficient=deficient):
                maps = rng.normal(size=(points, channels))
                if deficient:
                    maps[:, -1] = maps[:, 0] + maps[:, 1]
                target = rng.normal(size=points)
                currents = rng.normal(size=(9, channels))
                kernel = np.array([.5, .3, .2])
                actual = _sirf_conv1d_numpy(currents, kernel)
                reduced_maps, reduced_target, report = compress_field_target(maps, target)
                full_residual = actual @ maps.T - target
                reduced_residual = actual @ reduced_maps - reduced_target
                np.testing.assert_allclose(
                    np.square(full_residual).sum(),
                    np.square(reduced_residual).sum()
                    + len(currents) * report["orthogonal_target_squared_norm_Hz2"],
                    rtol=2e-14, atol=2e-12)
                np.testing.assert_allclose(np.linalg.norm(maps, 2), np.linalg.norm(reduced_maps, 2),
                                           rtol=2e-14)

    def test_static_qp_dense_and_compressed_solutions_match(self):
        rng = np.random.default_rng(18)
        maps = rng.normal(size=(35, 4))
        target = rng.normal(size=35)
        reduced_maps, reduced_target, _ = compress_field_target(maps, target)
        options = dict(amp_limit=.15, l1_limit=.35, ridge_x=.02)
        dense, _ = solve_shim_static_qp(target, maps.T, **options)
        compressed, _ = solve_shim_static_qp(reduced_target, reduced_maps, **options)
        np.testing.assert_allclose(compressed, dense, rtol=2e-7, atol=2e-9)

    def test_dynamic_fista_dense_and_compressed_solutions_match(self):
        rng = np.random.default_rng(21)
        maps = rng.normal(size=(31, 3))
        target = rng.normal(size=31)
        reduced_maps, reduced_target, report = compress_field_target(maps, target)
        kernel = amplifier_sirf(30e-6, 10e-6)
        options = dict(sirf_kernel=kernel, amp_limit=.2, l1_limit=.4,
                       ridge_x=.001, max_iter=30, tol=1e-15, device="cpu", verbose=False)
        dense, _, _ = solve_shim_waveform_fista_torch(np.tile(target, (12, 1)), maps.T, **options)
        compressed, _, _ = solve_shim_waveform_fista_torch(
            np.tile(reduced_target, (12, 1)), reduced_maps, **options)
        np.testing.assert_allclose(compressed, dense, rtol=3e-12, atol=3e-13)
        actual = _sirf_conv1d_numpy(compressed, kernel)
        dense_residual = actual @ maps.T - target
        reduced_residual = actual @ reduced_maps - reduced_target
        np.testing.assert_allclose(np.square(dense_residual).sum(), np.square(reduced_residual).sum()
                                   + 12 * report["orthogonal_target_squared_norm_Hz2"], rtol=1e-14)


class FieldShimTests(unittest.TestCase):
    def test_spiral_unequal_turns_keep_independent_field_raster_and_disabled_channel(self):
        model = tiny_field_model()
        model.sequence_kind, model.nstates = "spiral", 3
        model.line_start_s = np.array([40e-6, 180e-6, 370e-6])
        model.line_end_s = np.array([180e-6, 370e-6, 540e-6])
        model.max_current_A[0] = 0.
        pattern = model.maps_numpy[..., 1]
        model.b0_numpy = -.7 * pattern
        before = model.line_start_s.copy()
        result = solve_field_shim_waveforms(model, max_iterations=30)
        for name in ("static", "dynamic"):
            fit = result[name]
            np.testing.assert_array_equal(fit["currents_A"][0], 0.)
            self.assertLess(fit["report"]["full_field_nrmse"], .1)
            self.assertEqual(fit["report"]["sequence_kind"], "spiral")
            self.assertEqual(fit["report"]["image_command_state_unit"], "spiral turn")
            self.assertEqual(fit["report"]["image_command_states"], 3)
        self.assertEqual(result["dynamic"]["currents_A"].shape, (2, 50))
        self.assertFalse(result["dynamic"]["report"]["constrained_to_image_command_boundaries"])
        np.testing.assert_allclose(np.diff(result["dynamic"]["state_edges_rf_s"]), 10e-6)
        np.testing.assert_array_equal(model.line_start_s, before)

    def test_negative_B0_target_static_and_dynamic_reports_match_full_fields(self):
        model = tiny_field_model()
        original = {name: getattr(model, name).copy() for name in (
            "b0_numpy", "maps_numpy", "line_start_s", "line_end_s")}
        result = solve_field_shim_waveforms(model, max_iterations=30)
        static, dynamic = result["static"], result["dynamic"]
        self.assertEqual(static["currents_A"].shape, (2, 1))
        self.assertEqual(dynamic["currents_A"].shape, (2, 50))
        np.testing.assert_allclose(static["state_edges_rf_s"], [40e-6, 540e-6], rtol=0, atol=1e-18)
        np.testing.assert_allclose(np.diff(dynamic["state_edges_rf_s"]), 10e-6, rtol=1e-12)
        maps, target = model.maps_numpy[model.quality_mask], -model.b0_numpy[model.quality_mask]
        residual = maps @ static["currents_A"][:, 0] - target
        expected = np.linalg.norm(residual) / np.linalg.norm(target)
        np.testing.assert_allclose(static["report"]["full_field_nrmse"], expected, rtol=1e-12)
        self.assertLess(expected, .1)
        np.testing.assert_allclose(static["currents_A"][:, 0], [-.35, .2], atol=.03)
        kernel = np.asarray(dynamic["report"]["sirf_kernel"])
        actual = _sirf_conv1d_numpy(dynamic["currents_A"].T, kernel)
        residual = actual @ maps.T - target
        expected_dynamic = np.linalg.norm(residual) / (np.sqrt(actual.shape[0]) * np.linalg.norm(target))
        np.testing.assert_allclose(dynamic["report"]["full_field_nrmse"], expected_dynamic, rtol=1e-12)
        self.assertLess(expected_dynamic, .2)
        self.assertEqual(result["report"]["upstream_commit"], UPSTREAM_COMMIT)
        for name, before in original.items():
            np.testing.assert_array_equal(getattr(model, name), before)

    def test_duration_preserving_grid_never_extends_original_window(self):
        model = tiny_field_model()
        model.line_end_s[-1] = 543e-6
        result = solve_field_shim_waveforms(model, max_iterations=2)
        edges = result["dynamic"]["state_edges_rf_s"]
        self.assertEqual(edges[0], model.line_start_s[0])
        self.assertEqual(edges[-1], model.line_end_s[-1])
        self.assertEqual(len(edges), 52)
        self.assertGreaterEqual(np.diff(edges).min(), 8e-6)
        self.assertLessEqual(np.diff(edges).max(), 10e-6)

    def test_strict_total_current_budget_and_per_channel_bounds(self):
        model = tiny_field_model()
        model.b0_numpy *= 100
        result = solve_field_shim_waveforms(model, total_current_limit_A=.15, max_iterations=8)
        for name in ("static", "dynamic"):
            currents = result[name]["currents_A"]
            self.assertTrue(np.all(np.abs(currents).sum(axis=0) < .15))
            self.assertTrue(np.all(np.abs(currents) <= 3.))

    def test_disabled_strong_channel_is_excluded_and_other_channel_is_refitted(self):
        model = tiny_field_model()
        pattern = np.array([[1., 2.], [3., 4.]])
        model.maps_numpy = np.stack((20*pattern, pattern), axis=-1)
        model.b0_numpy = -.7*pattern
        unrestricted = solve_field_shim_waveforms(model, max_iterations=30)
        self.assertLess(abs(unrestricted['static']['currents_A'][1, 0]), .01)
        model.max_current_A[0] = 0.
        result = solve_field_shim_waveforms(model, max_iterations=30)
        self.assertGreater(result['static']['currents_A'][1, 0], .69)
        self.assertGreater(result['dynamic']['currents_A'][1].mean(), .69)
        for name in ('static', 'dynamic'):
            fit = result[name]
            self.assertEqual(fit['currents_A'].shape[0], 2)
            np.testing.assert_array_equal(fit['currents_A'][0], 0.)
            self.assertLess(fit['report']['full_field_nrmse'], .1)
            self.assertEqual(fit['report']['disabled_model_channels_1based'], [1])
            self.assertEqual(fit['report']['active_model_channels_1based'], [2])
            self.assertTrue(fit['report']['disabled_channels_excluded_before_fitting'])

    def test_all_disabled_and_zero_active_maps_return_baseline_without_vendor_solver(self):
        for zero_active_maps in (False, True):
            with self.subTest(zero_active_maps=zero_active_maps):
                model = tiny_field_model()
                if zero_active_maps:
                    model.max_current_A[0] = 0.
                    model.maps_numpy[..., 1] = 0.
                else:
                    model.max_current_A[:] = 0.
                with mock.patch('_vendor.acdc_optim.solve_shim_static_qp', side_effect=AssertionError('unexpected')), \
                        mock.patch('_vendor.acdc_optim.solve_shim_waveform_fista_torch', side_effect=AssertionError('unexpected')):
                    result = solve_field_shim_waveforms(model)
                for name, states in (('static', 1), ('dynamic', 50)):
                    fit = result[name]
                    self.assertEqual(fit['currents_A'].shape, (2, states))
                    np.testing.assert_array_equal(fit['currents_A'], 0.)
                    self.assertEqual(fit['report']['full_field_nrmse'], 1.)
                    self.assertTrue(fit['report']['solver_skipped'])

    def test_active_channels_alone_determine_common_amplifier_response(self):
        model = tiny_field_model()
        model.max_current_A[0] = 0.
        model.current_response['tau_s'][0] *= 2
        result = solve_field_shim_waveforms(model, max_iterations=2)
        self.assertEqual(result['dynamic']['report']['tau_s'], 30e-6)
        np.testing.assert_array_equal(result['dynamic']['currents_A'][0], 0.)

    def test_negative_channel_limits_are_rejected(self):
        model = tiny_field_model()
        model.max_current_A[0] = -1.
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            solve_field_shim_waveforms(model)

    def test_zero_B0_returns_zero_without_upstream_zero_denominator(self):
        model = tiny_field_model()
        model.b0_numpy[:] = 0
        with mock.patch("_vendor.acdc_optim.solve_shim_static_qp", side_effect=AssertionError("unexpected")), \
                mock.patch("_vendor.acdc_optim.solve_shim_waveform_fista_torch", side_effect=AssertionError("unexpected")):
            result = solve_field_shim_waveforms(model)
        for name in ("static", "dynamic"):
            np.testing.assert_array_equal(result[name]["currents_A"], 0.)
            self.assertEqual(result[name]["report"]["full_field_nrmse"], 0.)

    def test_unreachable_orthogonal_B0_returns_zero_and_unit_full_nrmse(self):
        model = tiny_field_model()
        model.maps_numpy = np.array([[[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]]])
        model.b0_numpy = np.array([[0., 0.], [1., 0.]])
        result = solve_field_shim_waveforms(model)
        for name in ("static", "dynamic"):
            np.testing.assert_array_equal(result[name]["currents_A"], 0.)
            self.assertEqual(result[name]["report"]["full_field_nrmse"], 1.)
            self.assertEqual(result[name]["report"]["solver_reduced_nrmse"], 0.)

    def test_rejects_heterogeneous_taus_and_channel_limits(self):
        model = tiny_field_model()
        model.current_response["tau_s"][1] *= 2
        with self.assertRaisesRegex(ValueError, "common amplifier time constant"):
            solve_field_shim_waveforms(model)
        model = tiny_field_model()
        model.max_current_A[1] = 2.
        with self.assertRaisesRegex(ValueError, "identical positive channel"):
            solve_field_shim_waveforms(model)

    def test_rejects_grid_below_hardware_update_interval(self):
        with self.assertRaisesRegex(ValueError, "below min_update_interval_s"):
            solve_field_shim_waveforms(tiny_field_model(), raster_s=5e-6)

    def test_rejects_invalid_scalar_settings(self):
        for options in (dict(max_iterations=True), dict(max_iterations=0),
                        dict(max_iterations=1.5), dict(max_iterations=np.nan),
                        dict(total_current_limit_A=0), dict(static_ridge=-1),
                        dict(dynamic_ridge=np.inf), dict(raster_s=False)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                solve_field_shim_waveforms(tiny_field_model(), **options)

    def test_sirf_uses_model_tau_and_unit_DC_gain(self):
        kernel = amplifier_sirf(30e-6, 10e-6)
        np.testing.assert_allclose(kernel.sum(), 1., atol=1e-15)
        np.testing.assert_allclose(kernel[1:] / kernel[:-1], np.exp(-1/3), rtol=1e-14)
        self.assertGreaterEqual(len(kernel), 30)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not visible in this process")
    def test_real_CUDA_dynamic_solver_matches_CPU(self):
        cpu = solve_field_shim_waveforms(tiny_field_model(), max_iterations=15)
        cuda = solve_field_shim_waveforms(tiny_field_model("cuda"), max_iterations=15)
        np.testing.assert_allclose(cuda["dynamic"]["currents_A"], cpu["dynamic"]["currents_A"],
                                   rtol=2e-11, atol=2e-12)
        self.assertEqual(cuda["dynamic"]["report"]["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
