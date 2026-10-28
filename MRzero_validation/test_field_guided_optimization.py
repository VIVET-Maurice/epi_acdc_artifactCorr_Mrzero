"""Physical branch selection for an exactly aliased Fourier EPI fixture."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Experiment"))
from acdc_gpu import ACDCModel
from test_acdc_gpu import fixture
from experiment_optimization import optimize_mrzero_image


def aliased_model(*, all_disabled=False):
    """One complete FOV shift is invisible in magnitude but equals 1000 Hz.

    EPI line spacing is exactly 1 ms, with a common 0.2 ms readout time
    offset. The 1000 Hz field therefore adds only a global phase to the
    zero-field reference. One working uniform coil can cancel it; the second
    calibrated coil is deliberately disabled.
    """
    experiment, calibration = fixture(response=False)
    acquisition = experiment["acquisition"]
    states = acquisition["line_index"] - 1
    acquisition["line_start_s"] = np.arange(3) * .001
    acquisition["line_end_s"] = np.arange(1, 4) * .001
    acquisition["time_s"] = states * .001 + .0002
    experiment["field"]["b0Hz"][:] = 1000.
    calibration["acdc_maps"] = np.ones((3, 4, 2)) * np.array([1000., 100.])
    limits = [0., 0.] if all_disabled else [1.5, 0.]
    return ACDCModel(experiment, calibration, max_current_A=limits,
                     profile="cpu", precision="float64", sample_chunk=5)


class FieldGuidedOptimizationTests(unittest.TestCase):
    def test_unwrapped_field_guidance_escapes_exact_full_fov_alias(self):
        model = aliased_model()
        common = dict(total_current_limit_A=2., field_limit_Hz=1500.,
                      max_iterations=250, patience=50, learning_rate_A=.03,
                      target_mse=1e-12, use_cuda_graph=False)
        legacy = optimize_mrzero_image(model, **common)
        self.assertLess(legacy["optimization_info"]["fval"], 1e-24)
        np.testing.assert_array_equal(legacy["currents_A"], 0.)
        snapshots = []
        guided = optimize_mrzero_image(
            model, **common, field_guidance_weight=100.,
            field_guidance_scale_Hz=1000., callback=snapshots.append)
        info = guided["optimization_info"]
        self.assertGreater(info["iterations"], 0)
        self.assertNotEqual(info["status"], "target_mse")
        self.assertFalse(info["target_mse_stopping_enabled"])
        self.assertLess(info["residual_field_rms_Hz"], 10.)
        self.assertLess(info["objective"], guided["initial_objective"] * .001)
        self.assertGreater(np.max(np.abs(guided["currents_A"][0])), .8)
        for row in snapshots:
            np.testing.assert_array_equal(row["currents_A"][1], 0.)
            image_mse = model.objective_and_grad(row["currents_A"])[0]
            self.assertAlmostEqual(row["fval"], image_mse, places=12)
            self.assertAlmostEqual(row["objective"],
                                   row["fval"] + 100. * row["field_penalty"], places=10)
        history = guided["history"]
        np.testing.assert_array_equal(history["best_fval"], np.minimum.accumulate(history["fval"]))
        np.testing.assert_array_equal(history["best_objective"], np.minimum.accumulate(history["objective"]))
        self.assertAlmostEqual(info["objective"], history["objective"].min(), places=10)
        self.assertAlmostEqual(info["fval"], model.objective_and_grad(guided["currents_A"])[0], places=12)

    def test_all_disabled_channels_remain_off_with_physical_residual_reported(self):
        model = aliased_model(all_disabled=True)
        solution = optimize_mrzero_image(
            model, max_iterations=3, field_guidance_weight=.1,
            field_guidance_scale_Hz=100., field_limit_Hz=1500.)
        np.testing.assert_array_equal(solution["currents_A"], 0.)
        self.assertAlmostEqual(solution["optimization_info"]["residual_field_rms_Hz"], 1000., places=9)
        self.assertTrue(solution["optimization_info"]["accepted"])

    def test_zero_weight_preserves_legacy_waveform_and_image_metrics(self):
        model = aliased_model()
        initial = np.zeros((2, 3))
        initial[0] = -.4
        options = dict(max_iterations=5, initial_currents_A=initial,
                       total_current_limit_A=2., field_limit_Hz=1500.)
        original = optimize_mrzero_image(model, **options)
        explicit_zero = optimize_mrzero_image(
            model, **options, field_guidance_weight=0., field_guidance_scale_Hz=17.)
        np.testing.assert_array_equal(original["currents_A"], explicit_zero["currents_A"])
        np.testing.assert_array_equal(original["history"]["fval"], explicit_zero["history"]["fval"])
        self.assertNotIn("field_guidance", explicit_zero["optimization_info"])
        self.assertEqual(explicit_zero["optimization_info"]["field_guidance_weight"], 0.)
        self.assertEqual(explicit_zero["optimization_info"]["field_guidance_scale_Hz"], 17.)

    def test_invalid_guidance_configuration_is_rejected(self):
        model = aliased_model()
        for options in ({"field_guidance_weight": -1.}, {"field_guidance_weight": np.nan},
                        {"field_guidance_weight": True}, {"field_guidance_scale_Hz": 0.},
                        {"field_guidance_scale_Hz": np.inf}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                optimize_mrzero_image(model, **options)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
