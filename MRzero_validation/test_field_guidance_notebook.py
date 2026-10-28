"""Exercise the notebook's actual guided solver call and post-DAC checks."""

import ast
from contextlib import redirect_stdout
import io
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from test_disabled_channels_notebook import parameter_namespace, source_containing
from test_experiment_realistic import small_model
from experiment_optimization import optimize_mrzero_image
from export_acdc_waveform import quantize_currents
from field_guidance import FieldGuidance
from measured_image_target import relative_image_mse


class FieldGuidanceNotebookTests(unittest.TestCase):
    def test_notebook_solver_and_dac_keep_image_and_field_costs_separate(self):
        for weight in (0., .05):
            with self.subTest(weight=weight), redirect_stdout(io.StringIO()):
                namespace = parameter_namespace([2, 8])
                _, _, model = small_model(max_current_A=namespace["CHANNEL_MAX_CURRENT_A"])
                initial = np.zeros((model.nc, model.nstates))
                snapshots = []
                namespace.update(model=model, initial_currents=initial,
                    FIELD_GUIDANCE_WEIGHT=weight, MAX_ITERATIONS=4,
                    LEARNING_RATE_A=.01, update=snapshots.append,
                    optimize_mrzero_image=optimize_mrzero_image)
                tree = ast.parse(source_containing("solution=optimize_mrzero_image("))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)
                         and node.func.id == "optimize_mrzero_image"]
                self.assertEqual(len(calls), 1)
                solution = eval(compile(ast.Expression(calls[0]), "<notebook solver>", "eval"), namespace)
                info = solution["optimization_info"]
                self.assertEqual("field_guidance" in info, weight > 0)
                for snapshot in snapshots:
                    self.assertAlmostEqual(snapshot["fval"], relative_image_mse(model, snapshot["image"]))
                    np.testing.assert_array_equal(snapshot["currents_A"][[0, 7]], 0.)
                namespace.update(solution=solution, info=info,
                    currents=solution["currents_A"], history=solution["history"],
                    quantize_currents=quantize_currents, relative_image_mse=relative_image_mse,
                    FieldGuidance=FieldGuidance, table=lambda *args: None,
                    plt=plt, show=plt.close, image_physical=lambda *args, **kwargs: None,
                    optimization_reference_image=model.reference, baseline_image=model.forward(initial)[0],
                    X_image=model.reconstruction.X_m, Y_image=model.reconstruction.Y_m)
                # Editing the parameters after solving must not change the DAC objective.
                namespace["FIELD_GUIDANCE_WEIGHT"] = 999.
                namespace["FIELD_GUIDANCE_SCALE_HZ"] = .001
                archive = ast.parse(source_containing("saved_guidance_configuration="))
                config = ast.Module(body=[archive.body[0]], type_ignores=[])
                exec(compile(config, "<notebook saved config>", "exec"), namespace)
                self.assertEqual(namespace["saved_guidance_configuration"],
                                 {"weight": weight, "scale_Hz": 100.})
                exec(source_containing("quantized_objective=quantized_mse"), namespace)
                expected = namespace["quantized_mse"]
                if weight:
                    guide = FieldGuidance(model, scale_Hz=info["field_guidance_scale_Hz"])
                    penalty, _, rms = guide.value_gradient(
                        model._check_currents(namespace["quantized_currents"]), gradient=False)
                    expected += weight * float(penalty)
                    self.assertAlmostEqual(namespace["quantized_field_guidance"]["residual_field_rms_Hz"], float(rms))
                else:
                    self.assertIsNone(namespace["quantized_field_guidance"])
                self.assertAlmostEqual(namespace["quantized_objective"], expected)
                np.testing.assert_array_equal(namespace["quantized_currents"][[0, 7]], 0.)
                expected_ok = (namespace["hardware_report"]["all_limits_met"]
                    and namespace["quantized_hardware"]["all_limits_met"]
                    and info["fval"] <= solution["initial_relative_mse"] + 1e-7
                    and namespace["quantized_mse"] <= solution["initial_relative_mse"] + 1e-7
                    and expected <= solution["initial_objective"] + 1e-7)
                self.assertEqual(namespace["solver_ok"], expected_ok)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
