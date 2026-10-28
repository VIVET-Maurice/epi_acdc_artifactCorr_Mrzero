"""Amplitude priors preserve echo nulls and must earn automatic selection."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from complex_echo_fit import fit_complex_echoes
from reference_echo_prior import (apply_reference_amplitude_prior, _envelope_kernel,
    _reference_envelope, _joint_reference_envelope, _smooth_contrast_field, _global_gain)


def fixture(noise=.01):
    times = .0021231 + np.arange(32)*.00182
    rows, cols = np.indices((4, 4))
    X, Y = np.meshgrid((np.arange(4)-2)*.003, (np.arange(4)-2)*.003)
    amplitude = 1. + .1*rows + .04*cols
    receivers = np.array([1., .4j, .2+.1j])
    receivers /= np.linalg.norm(receivers)
    mask = np.ones((4, 4), bool)
    model = dict(amplitude_reference_time_s=times[0],
        amplitude_envelope_at_reference=amplitude,
        residual_R2_sInv=np.full((4, 4), 15.), refined_b0_Hz=10.+rows-cols,
        unresolved_width_Hz=np.full((4, 4), 36.),
        dispersion_kind=np.ones((4, 4), np.int8),
        known_inplane_gradient_Hz_per_pixel=np.zeros((4, 4, 2)),
        complex_receiver_amplitude=receivers[:, None, None]*amplitude[None],
        support_mask=mask, quality_mask=mask,
        relative_complex_RMSE=np.zeros((4, 4)), reduced_chi2=np.zeros((4, 4)),
        report=dict(device="cpu", quality_reduced_chi2_limit=9.))
    kernel, _ = _envelope_kernel(model, times)
    model["predicted_RSS"] = np.abs(kernel)*amplitude[None]
    coils = kernel[:, None]*model["complex_receiver_amplitude"][None]
    if noise:
        rng = np.random.default_rng(651)
        coils += noise/np.sqrt(2)*(rng.normal(size=coils.shape)+1j*rng.normal(size=coils.shape))
    covariance = np.eye(3)*max(noise, 1e-6)**2
    fit = dict(multi_echo=model, report={}, noise_covariance=covariance)
    te_ref = .004322
    ref_kernel, _ = _envelope_kernel(model, [te_ref])
    gre = dict(magnitude=7*amplitude*np.abs(ref_kernel[0]), X_m=X, Y_m=Y,
               TE_s=te_ref, TR_s=.015, flip_deg=15.)
    inputs = dict(b0_raw=dict(coil_images=coils, X_m=X, Y_m=Y),
        echo_times_s=times, gre=gre, field=dict(b0Hz=model["refined_b0_Hz"]))
    return inputs, fit


def refitter(inputs, fit):
    return fit_complex_echoes(inputs["b0_raw"]["coil_images"], inputs["echo_times_s"],
        noise_covariance=fit["noise_covariance"],
        initial_b0_Hz=inputs["field"]["b0Hz"],
        support_mask=fit["multi_echo"]["support_mask"],
        known_inplane_gradient_Hz_per_pixel=fit["multi_echo"]["known_inplane_gradient_Hz_per_pixel"],
        device="cpu", max_iterations=40, max_width_Hz=80.)


class ReferenceEchoPriorTests(unittest.TestCase):
    def test_native_mode_is_independent_and_does_not_read_reference_or_fit(self):
        inputs, fit = fixture()
        inputs.pop("gre")
        original = deepcopy(fit)
        with patch("complex_echo_fit.fit_complex_echoes", side_effect=AssertionError("unexpected fit")):
            updated = apply_reference_amplitude_prior(inputs, fit, source="echo32")
        np.testing.assert_array_equal(updated["multi_echo"]["predicted_RSS"], original["multi_echo"]["predicted_RSS"])
        self.assertNotIn("reference_amplitude_prior", fit["report"])
        self.assertFalse(updated["report"]["reference_amplitude_prior"]["applied"])

    def test_actual_training_refit_uses_only_training_data_and_preserves_nulls(self):
        inputs, fit = fixture(.002)
        fit["multi_echo"] = refitter(inputs, fit)
        original = deepcopy(fit)
        real_fit = fit_complex_echoes
        with patch("complex_echo_fit.fit_complex_echoes", wraps=real_fit) as spy:
            updated = apply_reference_amplitude_prior(inputs, fit, source="reference", device="cpu",
                fit_options=dict(max_iterations=40, max_width_Hz=80.))
        self.assertEqual(spy.call_args.args[0].shape[0], 26)
        self.assertEqual(len(spy.call_args.args[1]), 26)
        result, report = updated["multi_echo"], updated["report"]["reference_amplitude_prior"]
        self.assertTrue(report["applied"])
        self.assertEqual(report["training_echo_indices"], list(range(26)))
        self.assertEqual(report["validation_echo_indices"], list(range(26, 32)))
        for key in ("residual_R2_sInv", "refined_b0_Hz", "unresolved_width_Hz", "dispersion_kind"):
            np.testing.assert_array_equal(result[key], original["multi_echo"][key])
        np.testing.assert_allclose(result["amplitude_envelope_at_reference"],
            fixture(0.)[1]["multi_echo"]["amplitude_envelope_at_reference"], rtol=.02)
        self.assertLess(np.max(result["predicted_RSS"][14]/result["predicted_RSS"][0]), .04)
        np.testing.assert_array_equal(fit["multi_echo"]["predicted_RSS"], original["multi_echo"]["predicted_RSS"])

    def test_auto_rejects_wrong_reference_without_hiding_disagreement(self):
        inputs, fit = fixture(0.)
        inputs["gre"]["magnitude"] *= np.linspace(.1, 4., 16).reshape(4, 4)
        # An exact training model makes the rejection deterministic and isolates
        # selection from optimization error; the real training fit is above.
        with patch("complex_echo_fit.fit_complex_echoes", return_value=deepcopy(fit["multi_echo"])):
            updated = apply_reference_amplitude_prior(inputs, fit, source="auto")
        report = updated["report"]["reference_amplitude_prior"]
        self.assertFalse(report["applied"])
        self.assertFalse(report["validation_gate_passed"])
        self.assertGreater(report["candidate_validation_whitened_complex_SSE"][-1],
                           report["candidate_validation_whitened_complex_SSE"][0])
        np.testing.assert_array_equal(updated["multi_echo"]["predicted_RSS"], fit["multi_echo"]["predicted_RSS"])

    def test_auto_accepts_reference_only_with_measured_prediction_gain(self):
        inputs, fit = fixture(0.)
        trained = deepcopy(fit["multi_echo"])
        perturb = np.linspace(.65, 1.35, 16).reshape(4, 4)
        trained["amplitude_envelope_at_reference"] *= perturb
        trained["complex_receiver_amplitude"] *= perturb[None]
        with patch("complex_echo_fit.fit_complex_echoes", return_value=trained):
            updated = apply_reference_amplitude_prior(inputs, fit, source="auto")
        report = updated["report"]["reference_amplitude_prior"]
        self.assertTrue(report["validation_gate_passed"])
        self.assertTrue(report["applied"])
        self.assertGreater(report["validation_relative_SSE_improvement"], .9)
        self.assertFalse(report["validation_uses_EPI"])

    def test_coherence_nulls_and_outside_reference_coverage_keep_native_amplitude(self):
        inputs, fit = fixture(0.)
        # First column is outside reference FOV; another pixel is at a GRE null.
        inputs["gre"]["X_m"] = inputs["gre"]["X_m"] + .0015
        inputs["gre"]["magnitude"] *= 2.
        fit["multi_echo"]["unresolved_width_Hz"][1, 1] = 1/inputs["gre"]["TE_s"]
        with patch("complex_echo_fit.fit_complex_echoes", return_value=deepcopy(fit["multi_echo"])):
            updated = apply_reference_amplitude_prior(inputs, fit, source="reference", min_common_voxels=4)
        ratio = updated["multi_echo"]["reference_amplitude_ratio"]
        np.testing.assert_array_equal(ratio[:, 0], 1.)
        self.assertEqual(ratio[1, 1], 1.)
        self.assertLess(updated["report"]["reference_amplitude_prior"]["common_valid_voxels"], 16)

    def test_reserved_echo_full_fit_parameters_cannot_change_validation_domain_or_score(self):
        inputs, fit = fixture(0.)
        trained = deepcopy(fit["multi_echo"])
        trained["amplitude_envelope_at_reference"] *= np.linspace(.65, 1.35, 16).reshape(4, 4)
        trained["complex_receiver_amplitude"] *= np.linspace(.65, 1.35, 16).reshape(1, 4, 4)
        altered = deepcopy(fit)
        altered["multi_echo"]["unresolved_width_Hz"][:, 0] = 1/inputs["gre"]["TE_s"]
        altered["multi_echo"]["refined_b0_Hz"] += 20.
        results = []
        for full_fit in (fit, altered):
            with patch("complex_echo_fit.fit_complex_echoes", return_value=deepcopy(trained)):
                results.append(apply_reference_amplitude_prior(inputs, full_fit, source="auto",
                    min_common_voxels=4)["report"]["reference_amplitude_prior"])
        self.assertEqual(results[0]["validation_common_voxels"], 16)
        self.assertEqual(results[1]["validation_common_voxels"], 16)
        np.testing.assert_array_equal(results[0]["candidate_validation_whitened_complex_SSE"],
                                      results[1]["candidate_validation_whitened_complex_SSE"])
        self.assertEqual(results[0]["common_valid_voxels"], 16)
        self.assertEqual(results[1]["common_valid_voxels"], 12)

    def test_training_reuses_early_seed_instead_of_raw_or_full_refined_B0(self):
        inputs, fit = fixture(0.)
        seed = np.full((4, 4), 7.)
        fit["multi_echo"]["initial_b0_Hz"] = seed
        trained = deepcopy(fit["multi_echo"])
        with patch("complex_echo_fit.fit_complex_echoes", return_value=trained) as spy:
            updated = apply_reference_amplitude_prior(inputs, fit, source="reference")
        np.testing.assert_array_equal(spy.call_args.kwargs["initial_b0_Hz"], seed)
        self.assertFalse(np.array_equal(spy.call_args.kwargs["initial_b0_Hz"], inputs["field"]["b0Hz"]))
        self.assertFalse(np.array_equal(spy.call_args.kwargs["initial_b0_Hz"], fit["multi_echo"]["refined_b0_Hz"]))
        self.assertIn("stored early-echo seed", updated["report"]["reference_amplitude_prior"]["validation_initial_B0_source"])

    def test_lower_resolution_reference_retains_native_temporal_grid_and_coverage(self):
        inputs, fit = fixture(0.)
        for key in ("magnitude", "X_m", "Y_m"):
            inputs["gre"][key] = inputs["gre"][key][::2, ::2]
        with patch("complex_echo_fit.fit_complex_echoes", return_value=deepcopy(fit["multi_echo"])):
            updated = apply_reference_amplitude_prior(inputs, fit, source="reference", min_common_voxels=4)
        result = updated["multi_echo"]
        report = updated["report"]["reference_amplitude_prior"]
        self.assertEqual(report["reference_shape"], [2, 2])
        self.assertEqual(report["native_echo_shape"], [4, 4])
        self.assertEqual(result["predicted_RSS"].shape, (32, 4, 4))
        self.assertEqual(report["common_valid_voxels"], 9)
        np.testing.assert_array_equal(result["reference_amplitude_ratio"][-1], 1.)
        np.testing.assert_array_equal(result["reference_amplitude_ratio"][:, -1], 1.)

    def test_joint_anatomy_earns_selection_and_preserves_temporal_coherence(self):
        inputs, fit = fixture(0.)
        trained = deepcopy(fit["multi_echo"])
        rows, columns = np.indices((4, 4))
        perturb = np.where((rows + columns) % 2, .7, 1.3)
        trained["amplitude_envelope_at_reference"] *= perturb
        trained["complex_receiver_amplitude"] *= perturb[None]
        original = deepcopy(fit)
        with patch("complex_echo_fit.fit_complex_echoes", return_value=trained):
            updated = apply_reference_amplitude_prior(inputs, fit, source="joint")
        result, report = updated["multi_echo"], updated["report"]["reference_amplitude_prior"]
        self.assertTrue(report["validation_gate_passed"])
        self.assertTrue(report["applied"])
        self.assertEqual(report["selected_source"], "joint")
        self.assertFalse(report["contrast_uses_reserved_echoes"])
        self.assertFalse(report["contrast_field_refitted_after_selection"])
        np.testing.assert_allclose(report["contrast_sigma_pixels_row_column"], [10/3, 10/3])
        self.assertTrue(np.all(result["reference_contrast_field"] > 0))
        for key in ("residual_R2_sInv", "refined_b0_Hz", "unresolved_width_Hz",
                    "dispersion_kind", "known_inplane_gradient_Hz_per_pixel"):
            np.testing.assert_array_equal(result[key], original["multi_echo"][key])
        # Amplitude may change; the signed temporal kernel and its late nulls
        # must remain exactly the same, with no artificial restoration.
        np.testing.assert_array_equal(_envelope_kernel(result, inputs["echo_times_s"])[0],
            _envelope_kernel(original["multi_echo"], inputs["echo_times_s"])[0])
        np.testing.assert_array_equal(fit["multi_echo"]["predicted_RSS"], original["multi_echo"]["predicted_RSS"])

    def test_joint_constant_spatial_gain_reduces_to_old_reference_calibration(self):
        inputs, fit = fixture(0.)
        model, gre = fit["multi_echo"], inputs["gre"]
        valid = np.ones((4, 4), bool)
        old, old_valid = _reference_envelope(gre["magnitude"], valid, model, gre["TE_s"], .2)
        joint, joint_valid = _joint_reference_envelope(gre["magnitude"], valid, model, gre["TE_s"], .2)
        contrast, report = _smooth_contrast_field(joint, joint_valid, model,
            inputs["b0_raw"]["X_m"], inputs["b0_raw"]["Y_m"], 10.)
        np.testing.assert_allclose(contrast, 1., rtol=0, atol=1e-15)
        self.assertAlmostEqual(report["contrast_field_geometric_mean"], 1.)
        whiten = np.eye(3)
        old_gain = _global_gain(old, old_valid, model, inputs["echo_times_s"], whiten)
        joint_gain = _global_gain(joint*contrast, joint_valid, model, inputs["echo_times_s"], whiten)
        np.testing.assert_allclose(old_gain*old, joint_gain*joint*contrast, rtol=1e-14)

    def test_joint_TE_transfer_does_not_invert_native_coherence_of_another_pixel_size(self):
        inputs, fit = fixture(0.)
        model, gre = fit["multi_echo"], inputs["gre"]
        model["unresolved_width_Hz"][:] = 1/gre["TE_s"]
        model["known_inplane_gradient_Hz_per_pixel"][:] = 1/gre["TE_s"]
        envelope, valid = _joint_reference_envelope(gre["magnitude"], np.ones((4, 4), bool),
            model, gre["TE_s"], .2)
        expected = gre["magnitude"]*np.exp(model["residual_R2_sInv"]
            * (gre["TE_s"]-model["amplitude_reference_time_s"]))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(envelope, expected)
        self.assertTrue(np.isfinite(envelope).all())

    def test_joint_smoothing_uses_physical_spacing_and_ignores_invalid_ratios(self):
        inputs, fit = fixture(0.)
        model, native = fit["multi_echo"], inputs["b0_raw"]
        valid = np.ones((4, 4), bool)
        valid[:, 0] = False
        reference = model["amplitude_envelope_at_reference"]/7.
        reference[:, 0] = 1e-200
        field, report = _smooth_contrast_field(reference, valid, model,
            2*native["X_m"], native["Y_m"], 6.)
        np.testing.assert_allclose(report["contrast_sigma_pixels_row_column"], [2., 1.])
        np.testing.assert_allclose(field, 1., rtol=0, atol=1e-15)

    def test_joint_lower_resolution_coverage_is_not_expanded_by_smoothing(self):
        inputs, fit = fixture(0.)
        for key in ("magnitude", "X_m", "Y_m"):
            inputs["gre"][key] = inputs["gre"][key][::2, ::2]
        trained = deepcopy(fit["multi_echo"])
        perturb = np.where(np.indices((4, 4)).sum(axis=0) % 2, .7, 1.3)
        trained["amplitude_envelope_at_reference"] *= perturb
        trained["complex_receiver_amplitude"] *= perturb[None]
        with patch("complex_echo_fit.fit_complex_echoes", return_value=trained):
            updated = apply_reference_amplitude_prior(inputs, fit, source="joint", min_common_voxels=4)
        result, report = updated["multi_echo"], updated["report"]["reference_amplitude_prior"]
        self.assertTrue(report["applied"])
        self.assertEqual(report["reference_shape"], [2, 2])
        self.assertEqual(report["contrast_training_voxels"], 9)
        self.assertEqual(report["common_valid_voxels"], 9)
        self.assertIn("no native-voxel sinc", report["resolution_limitation"])
        self.assertFalse(report["excludes_reference_coherence_nulls"])
        for array in (result["reference_contrast_field"], result["reference_amplitude_ratio"]):
            np.testing.assert_array_equal(array[-1], 1.)
            np.testing.assert_array_equal(array[:, -1], 1.)
        self.assertFalse(result["reference_contrast_coverage_mask"][-1].any())
        self.assertFalse(result["reference_contrast_coverage_mask"][:, -1].any())

    def test_joint_contrast_and_selection_do_not_leak_full_fit_parameters(self):
        inputs, fit = fixture(0.)
        trained = deepcopy(fit["multi_echo"])
        perturb = np.where(np.indices((4, 4)).sum(axis=0) % 2, .7, 1.3)
        trained["amplitude_envelope_at_reference"] *= perturb
        trained["complex_receiver_amplitude"] *= perturb[None]
        altered = deepcopy(fit)
        altered["multi_echo"]["residual_R2_sInv"] += 20.
        altered["multi_echo"]["unresolved_width_Hz"] *= 2.
        altered["multi_echo"]["amplitude_envelope_at_reference"] *= 3.
        results = []
        for full in (fit, altered):
            with patch("complex_echo_fit.fit_complex_echoes", return_value=deepcopy(trained)) as spy:
                updated = apply_reference_amplitude_prior(inputs, full, source="joint")
            np.testing.assert_array_equal(spy.call_args.args[0], inputs["b0_raw"]["coil_images"][:26])
            np.testing.assert_array_equal(spy.call_args.args[1], inputs["echo_times_s"][:26])
            results.append(updated)
        for key in ("candidate_validation_whitened_complex_SSE", "validation_best_weight"):
            np.testing.assert_array_equal(results[0]["report"]["reference_amplitude_prior"][key],
                results[1]["report"]["reference_amplitude_prior"][key])
        np.testing.assert_array_equal(results[0]["multi_echo"]["reference_contrast_field"],
            results[1]["multi_echo"]["reference_contrast_field"])

    def test_joint_reserved_observations_change_score_but_not_contrast_training(self):
        inputs, fit = fixture(0.)
        changed = deepcopy(inputs)
        changed["b0_raw"]["coil_images"][26:] *= 2.
        results = []
        for source_inputs in (inputs, changed):
            with patch("complex_echo_fit.fit_complex_echoes", return_value=deepcopy(fit["multi_echo"])) as spy:
                updated = apply_reference_amplitude_prior(source_inputs, fit, source="joint")
            np.testing.assert_array_equal(spy.call_args.args[0], inputs["b0_raw"]["coil_images"][:26])
            results.append(updated)
        np.testing.assert_array_equal(results[0]["multi_echo"]["reference_contrast_field"],
            results[1]["multi_echo"]["reference_contrast_field"])
        self.assertNotEqual(results[0]["report"]["reference_amplitude_prior"]["candidate_validation_whitened_complex_SSE"],
            results[1]["report"]["reference_amplitude_prior"]["candidate_validation_whitened_complex_SSE"])

    def test_joint_retains_native_amplitude_if_anatomy_prior_has_no_validation_gain(self):
        inputs, fit = fixture(0.)
        inputs["gre"]["magnitude"] *= np.where(np.indices((4, 4)).sum(axis=0) % 2, .1, 3.)
        with patch("complex_echo_fit.fit_complex_echoes", return_value=deepcopy(fit["multi_echo"])):
            updated = apply_reference_amplitude_prior(inputs, fit, source="joint")
        report = updated["report"]["reference_amplitude_prior"]
        self.assertFalse(report["applied"])
        self.assertEqual(report["selected_source"], "echo32")
        self.assertIn("reference_contrast_field", updated["multi_echo"])
        np.testing.assert_array_equal(updated["multi_echo"]["predicted_RSS"], fit["multi_echo"]["predicted_RSS"])

    def test_missing_te_and_temporal_extrapolation_do_not_become_PD_assumptions(self):
        inputs, fit = fixture()
        inputs["gre"].pop("TE_s")
        updated = apply_reference_amplitude_prior(inputs, fit, source="auto")
        self.assertFalse(updated["report"]["reference_amplitude_prior"]["applied"])
        with self.assertRaisesRegex(ValueError, "measured TE_s"):
            apply_reference_amplitude_prior(inputs, fit, source="reference")
        inputs["gre"]["TE_s"] = .2
        with self.assertRaisesRegex(ValueError, "temporal extrapolation"):
            apply_reference_amplitude_prior(inputs, fit, source="reference")

    def test_invalid_controls_fail_before_any_training(self):
        inputs, fit = fixture()
        for kwargs in (dict(source="PD"), dict(minimum_coherence=0),
                       dict(validation_echo_count=27), dict(candidate_weights=[1.2]),
                       dict(source="joint", contrast_sigma_mm=0),
                       dict(source="joint", contrast_sigma_mm=float("nan")),
                       dict(source="joint", contrast_sigma_mm=1j),
                       dict(source="joint", contrast_sigma_mm="10"),
                       dict(fit_options=dict(support_mask=np.ones((4, 4))))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                apply_reference_amplitude_prior(inputs, fit, **kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_training_and_validation_match_cpu(self):
        inputs, fit = fixture(.002)
        for source in ("reference", "joint"):
            with self.subTest(source=source):
                options = dict(source=source, fit_options=dict(max_iterations=40, max_width_Hz=80.))
                cpu = apply_reference_amplitude_prior(inputs, fit, device="cpu", **options)
                gpu = apply_reference_amplitude_prior(inputs, fit, device="cuda", require_cuda=True, **options)
                self.assertEqual(gpu["report"]["reference_amplitude_prior"]["train_fit_device"], "cuda")
                np.testing.assert_allclose(gpu["report"]["reference_amplitude_prior"]["candidate_validation_whitened_complex_SSE"],
                    cpu["report"]["reference_amplitude_prior"]["candidate_validation_whitened_complex_SSE"], rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(gpu["multi_echo"]["amplitude_envelope_at_reference"],
                    cpu["multi_echo"]["amplitude_envelope_at_reference"], rtol=1e-6)
                if source == "joint":
                    np.testing.assert_allclose(gpu["multi_echo"]["reference_contrast_field"],
                        cpu["multi_echo"]["reference_contrast_field"], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
