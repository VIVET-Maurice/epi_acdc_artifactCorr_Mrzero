"""Folder integration tests. All fixtures are synthetic and stay in /tmp.

Optional real MATLAB integration:
  ACDC_TEST_MATLAB=/path/to/matlab python -m unittest discover -s Experiment -p test_run_acdc_experiment.py
"""

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat, savemat

import run_acdc_experiment as pipeline
from test_read_acdc_sequence import make_gre_sequence


class FolderPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="acdc_folder_test_")
        self.addCleanup(self.temp.cleanup)
        self.input = Path(self.temp.name) / "input"
        self.output = Path(self.temp.name) / "output"
        pipeline.initialize(self.input)
        self.config = pipeline.template()
        self.config["files"].pop("reference_sequence")  # Legacy EPI configuration remains supported.
        self.config["files"].pop("b0_sequence")  # Already reconstructed B0 in this fixture.
        self.config["files"].update(gre="gre.npz", b0="b0.npz", scan="scan.npz",
                                    calibration="calibration.mat",
                                    sequence=str(pipeline.ROOT / "sequence" / "gre_epi_32x32_fov200mm_nav3_mintr.seq"))
        self.config["preparation"].update(geometry_confirmed=True, b0_sign_confirmed=True)
        self.config["hardware"].update(channel_map=list(range(1, 33)), channel_polarity=[1]*32,
            max_current_A=0.1, total_current_limit_A=2.0, clock_MHz=30,
            min_dac_interval_s=20e-6, min_adc_interval_s=20e-6, settling_time_s=50e-6,
            trigger_lockout_ms=1, trigger_period_s=.03, repetitions=1,
            rf_center_after_trigger_s=.0011)
        self.config["optimization"]["max_iterations"] = 5
        self.write_config()
        X, Y = np.meshgrid((np.arange(128)-64)*.2/128, (np.arange(128)-64)*.2/128)
        magnitude = np.zeros((128, 128))
        magnitude[50, 55], magnitude[65, 64], magnitude[80, 70] = 1, .7, .4
        mask = magnitude > 0
        np.savez(self.input / "gre.npz", magnitude=magnitude, X_m=X, Y_m=Y, mask=mask)
        np.savez(self.input / "b0.npz", b0Hz=np.zeros_like(X), X_m=X, Y_m=Y, mask=mask, units="Hz")
        np.savez(self.input / "scan.npz", image=np.ones((32, 32)))
        maps = np.zeros((128, 128, 32))
        maps[mask] = np.random.default_rng(27).normal(size=(3, 32)) * 100
        self.calibration = dict(acdc_maps=maps, X_m=X, Y_m=Y, units="Hz/A", registration_confirmed=True)
        savemat(self.input / "calibration.mat", self.calibration)

    def write_config(self):
        pipeline.write_json(self.input / "config.json", self.config)

    def solution(self, *, converged=True, currents=None, optimization_info=None):
        pipeline.prepare(self.input, self.output)
        report = json.loads((self.output / "prepared" / "preparation_report.json").read_text())
        folder = self.output / "optimization"
        folder.mkdir()
        savemat(folder / "solution_100Hz.mat", {
            "currents_A": np.zeros((32, 32)) if currents is None else currents,
            "line_start_s": report["state_edges_s"][:-1], "field_limit_Hz": 100,
            "calibration_mode": "registered", "registration_confirmed": True,
            "optimization_info": optimization_info if optimization_info is not None else dict(
                converged=converged, current_limits_met=True,
                field_limits_met=True, exitflag=1 if converged else 0, fval=0),
        })

    def test_prepared_contract_and_export(self):
        self.solution(currents=np.full((32, 32), .001))
        report = pipeline.export(self.output)
        self.assertEqual(report["status"], "exported")
        self.assertTrue(report["waveform_exported"])
        waveform = np.loadtxt(self.output / "waveform" / "waveform.csv", delimiter=",")
        self.assertEqual(waveform.shape, (34, 33))
        self.assertEqual([path.name for path in (self.output / "waveform").iterdir()], ["waveform.csv"])
        np.testing.assert_array_equal(waveform[[0, -1], 1:], 0)
        np.testing.assert_allclose(waveform[1:-1, 1:], .001)
        experiment = loadmat(self.output / "prepared" / "experiment_input.mat", simplify_cells=True)["experiment"]
        self.assertEqual(np.count_nonzero(experiment["m0_proxy"]), 3)
        self.assertIn("scan", experiment["images"])

    def test_export_does_not_require_fpga_controller_settings(self):
        for key in ("clock_MHz", "min_adc_interval_s", "trigger_lockout_ms", "repetitions"):
            self.config["hardware"].pop(key)
        self.write_config()
        self.solution()
        result = pipeline.export(self.output)
        self.assertEqual(result["status"], "exported")
        self.assertEqual([path.name for path in (self.output / "waveform").iterdir()], ["waveform.csv"])

    def test_run_passes_mse_target_and_delivers_only_csv_in_final_output(self):
        self.config["optimization"]["target_mse"] = .005
        self.write_config()
        workspace = []

        def synthetic_optimizer(command, check):
            request_file = Path(re.search(r"run_acdc_folder\('([^']+)'\)", command[-1]).group(1))
            request = loadmat(request_file, simplify_cells=True)
            self.assertEqual(request["target_mse"], .005)
            session = request_file.parent
            workspace.append(session)
            report = json.loads((session / "prepared/preparation_report.json").read_text())
            folder = Path(request["output_dir"])
            folder.mkdir()
            savemat(folder / "solution_100Hz.mat", {
                "currents_A": np.zeros((32, 32)),
                "line_start_s": report["state_edges_s"][:-1], "field_limit_Hz": 100,
                "calibration_mode": "registered", "registration_confirmed": True,
                "optimization_info": dict(converged=False, current_limits_met=True,
                    field_limits_met=True, exitflag=-1, fval=.004,
                    target_mse=.005, target_reached=True, accepted=True),
            })
            return subprocess.CompletedProcess(command, 0)

        with patch.object(pipeline.subprocess, "run", side_effect=synthetic_optimizer):
            result = pipeline.run(self.input, self.output)
        self.assertEqual(result["status"], "exported")
        self.assertEqual([path.name for path in self.output.iterdir()], ["waveform.csv"])
        self.assertEqual(len(workspace), 1)
        self.assertFalse(workspace[0].exists())
        data = np.loadtxt(self.output / "waveform.csv", delimiter=",")
        self.assertEqual(data.shape, (34, 33))

    def test_run_preserves_an_existing_output(self):
        self.output.mkdir()
        existing = self.output / "keep.txt"
        existing.write_text("Keep this result.")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            pipeline.run(self.input, self.output)
        self.assertEqual(existing.read_text(), "Keep this result.")

    def test_virtual_calibration_is_not_accepted(self):
        self.calibration["registration_confirmed"] = False
        savemat(self.input / "calibration.mat", self.calibration)
        with self.assertRaisesRegex(ValueError, "registration_confirmed"):
            pipeline.prepare(self.input, self.output)
        self.assertFalse((self.output / "provenance.json").exists())

    def test_incomplete_calibration_coverage_is_not_zero_filled(self):
        self.calibration["acdc_maps"][50, 55, :] = np.nan
        savemat(self.input / "calibration.mat", self.calibration)
        with self.assertRaisesRegex(ValueError, "no measured coil calibration"):
            pipeline.prepare(self.input, self.output)

    def test_nonconverged_solution_cannot_export(self):
        self.solution(converged=False)
        with self.assertRaisesRegex(ValueError, "not converged"):
            pipeline.export(self.output)
        self.assertFalse((self.output / "waveform").exists())

    def test_mse_target_stop_exports_without_solver_convergence(self):
        self.config["optimization"]["target_mse"] = .005
        self.write_config()
        self.solution(optimization_info=dict(converged=False, exitflag=-1,
            fval=.0049, current_limits_met=True, field_limits_met=True,
            target_mse=.005, target_reached=True, accepted=True))
        result = pipeline.export(self.output)
        self.assertEqual(result["optimization_info"]["exitflag"], -1)
        self.assertTrue(result["optimization_info"]["target_reached"])
        self.assertEqual(result["optimization_info"]["target_mse"], .005)
        self.assertEqual([path.name for path in (self.output / "waveform").iterdir()], ["waveform.csv"])

    def test_mse_target_stop_does_not_override_actual_current_limits(self):
        self.config["optimization"]["target_mse"] = .005
        self.write_config()
        self.calibration["acdc_maps"][:] = 1
        savemat(self.input / "calibration.mat", self.calibration)
        self.solution(currents=np.full((32, 32), .11), optimization_info=dict(
            converged=False, exitflag=-1, fval=.004, current_limits_met=True,
            field_limits_met=True, target_mse=.005, target_reached=True, accepted=True))
        with self.assertRaisesRegex(ValueError, "exceed max_current_A"):
            pipeline.export(self.output)
        self.assertFalse((self.output / "waveform").exists())

    def test_mse_target_stop_does_not_override_actual_field_limits(self):
        self.config["optimization"]["target_mse"] = .005
        self.write_config()
        self.calibration["acdc_maps"][:] = 10000
        savemat(self.input / "calibration.mat", self.calibration)
        self.solution(currents=np.full((32, 32), .001), optimization_info=dict(
            converged=False, exitflag=-1, fval=.004, current_limits_met=True,
            field_limits_met=True, target_mse=.005, target_reached=True, accepted=True))
        with self.assertRaisesRegex(ValueError, "field exceeds"):
            pipeline.export(self.output)
        self.assertFalse((self.output / "waveform").exists())

    def test_mse_target_stop_does_not_override_total_current_limit(self):
        self.config["optimization"]["target_mse"] = .005
        self.write_config()
        self.calibration["acdc_maps"][:] = 1
        savemat(self.input / "calibration.mat", self.calibration)
        self.solution(currents=np.full((32, 32), .07), optimization_info=dict(
            converged=False, exitflag=-1, fval=.004, current_limits_met=True,
            field_limits_met=True, target_mse=.005, target_reached=True, accepted=True))
        with self.assertRaisesRegex(ValueError, "total_current_limit_A"):
            pipeline.export(self.output)
        self.assertFalse((self.output / "waveform").exists())

    def test_mse_target_stop_is_strict_and_requires_consistent_solver_metadata(self):
        self.config["optimization"]["target_mse"] = .005
        good = dict(converged=False, exitflag=-1, fval=.004, current_limits_met=True,
                    field_limits_met=True, target_mse=.005, target_reached=True, accepted=True)
        self.assertTrue(pipeline.check_optimization_result(self.config, good))
        for changes in ({"fval": .005}, {"fval": .006}, {"fval": -1}, {"fval": np.nan},
                        {"target_mse": .006}, {"target_mse": []}, {"target_reached": False},
                        {"target_reached": "true"}, {"accepted": False}, {"converged": True},
                        {"exitflag": 0}, {"exitflag": -2}, {"exitflag": -1.5},
                        {"current_limits_met": False}, {"field_limits_met": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pipeline.check_optimization_result(self.config, {**good, **changes})
        for key in ("fval", "target_mse", "target_reached"):
            invalid = {name: value for name, value in good.items() if name != key}
            with self.subTest(missing=key), self.assertRaises(ValueError):
                pipeline.check_optimization_result(self.config, invalid)
        self.config["optimization"]["target_mse"] = None
        with self.assertRaises(ValueError):
            pipeline.check_optimization_result(self.config, good)

    def test_mse_target_configuration_is_optional_and_validated(self):
        pipeline.check_optimization(self.config)
        self.config["optimization"].pop("target_mse")
        pipeline.check_optimization(self.config)
        self.config["optimization"]["target_mse"] = .005
        pipeline.check_optimization(self.config)
        for value in (True, False, 0, -1, np.inf, np.nan, [.005], "0.005"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "target_mse"):
                self.config["optimization"]["target_mse"] = value
                pipeline.check_optimization(self.config)

    def test_field_limit_rechecked_after_dac_quantization(self):
        # Two almost cancelling channels: truncation can increase the net field.
        self.calibration["acdc_maps"][:] = 1
        self.calibration["acdc_maps"][:, :, 0] = 1001
        self.calibration["acdc_maps"][:, :, 1] = -2000
        savemat(self.input / "calibration.mat", self.calibration)
        currents = np.zeros((32, 32))
        currents[0], currents[1] = .1, .0001
        self.solution(currents=currents)
        with self.assertRaisesRegex(ValueError, "DAC quantization"):
            pipeline.export(self.output)
        self.assertFalse((self.output / "waveform").exists())

    def test_modified_source_blocks_export(self):
        self.solution()
        np.savez(self.input / "scan.npz", image=2*np.ones((32, 32)))
        with self.assertRaisesRegex(ValueError, "File changed since preparation"):
            pipeline.export(self.output)

    def test_modified_prepared_input_blocks_export(self):
        self.solution()
        with (self.output / "prepared" / "experiment_input.mat").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "File changed since preparation"):
            pipeline.export(self.output)

    def test_adc_settling_overlap_blocks_export(self):
        self.config["hardware"]["settling_time_s"] = .0002
        self.write_config()
        self.solution()
        with self.assertRaisesRegex(ValueError, "settling reaches the scan ADC"):
            pipeline.export(self.output)

    def test_init_does_not_overwrite_configuration(self):
        original = (self.input / "config.json").read_bytes()
        with self.assertRaises(ValueError):
            pipeline.initialize(self.input)
        self.assertEqual((self.input / "config.json").read_bytes(), original)

    @unittest.skipUnless(os.environ.get("ACDC_TEST_MATLAB"), "Set ACDC_TEST_MATLAB for the real optimizer")
    def test_real_matlab_optimizer_to_waveform_csv(self):
        self.config["optimization"]["matlab_command"] = os.environ["ACDC_TEST_MATLAB"]
        self.write_config()
        report = pipeline.run(self.input, self.output)
        self.assertEqual(report["status"], "exported")
        self.assertGreater(report["optimization_info"]["exitflag"], 0)
        self.assertTrue(report["waveform_exported"])

    @unittest.skipUnless(os.environ.get("ACDC_TEST_MATLAB"), "Set ACDC_TEST_MATLAB for the real optimizer")
    def test_real_matlab_mse_target_stop_to_waveform_csv(self):
        # Zero measured B0 already meets the target; stop before unnecessary iterations.
        self.config["optimization"].update(matlab_command=os.environ["ACDC_TEST_MATLAB"], target_mse=.005)
        self.write_config()
        report = pipeline.run(self.input, self.output)
        self.assertEqual(report["status"], "exported")
        self.assertEqual(report["optimization_info"]["exitflag"], -1)
        self.assertTrue(report["optimization_info"]["target_reached"])
        self.assertLess(report["optimization_info"]["fval"], .005)
        self.assertEqual([path.name for path in self.output.iterdir()], ["waveform.csv"])


class GreFolderPipelineTests(unittest.TestCase):
    """Two independent GRE sequences and measured-map grids through the same CLI."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="acdc_gre_folder_test_")
        self.addCleanup(temp.cleanup)
        self.input = Path(temp.name) / "input"
        self.output = Path(temp.name) / "output"
        pipeline.initialize(self.input)
        self.config = pipeline.template()
        self.config["files"].pop("b0_sequence")
        self.config["files"].update(gre="gre.npz", scan="scan.npz", b0="b0_map.npz", calibration="calibration.mat")
        self.config["preparation"].update(geometry_confirmed=True, b0_sign_confirmed=True)
        self.config["optimization"]["max_iterations"] = 5
        self.config["hardware"].update(channel_map=list(range(1, 33)), channel_polarity=[1]*32,
            max_current_A=.1, total_current_limit_A=2, clock_MHz=30,
            min_dac_interval_s=20e-6, min_adc_interval_s=20e-6, settling_time_s=50e-6,
            trigger_lockout_ms=1, trigger_period_s=.15, repetitions=1,
            rf_center_after_trigger_s=.0002)
        fov = [.24, .18]
        make_gre_sequence(self.input / "sequences/scan.seq", nx=8, ny=6, fov=fov, dummies=2,
                          phase_order=[3, 2, 4, 1, 5, 0])
        make_gre_sequence(self.input / "sequences/reference_gre.seq", nx=16, ny=12, fov=fov,
                          te=.004, tr=.012)
        X, Y = np.meshgrid((np.arange(16)-8)*fov[0]/16, (np.arange(12)-6)*fov[1]/12)
        magnitude = np.zeros((12, 16))
        magnitude[4, 6], magnitude[6, 8], magnitude[7, 10] = 1, .7, .4
        np.savez(self.input / "gre.npz", magnitude=magnitude, X_m=X, Y_m=Y, mask=magnitude>0)
        # B0 is denser than the reference, while the coil map grid is coarser.
        Bx, By = np.meshgrid((np.arange(32)-16)*fov[0]/32, (np.arange(24)-12)*fov[1]/24)
        np.savez(self.input / "b0_map.npz", b0Hz=np.zeros_like(Bx), X_m=Bx, Y_m=By,
                 mask=np.ones_like(Bx, bool), units="Hz")
        Cx, Cy = np.meshgrid((np.arange(8)-4)*fov[0]/8, (np.arange(6)-3)*fov[1]/6)
        maps = (1+Cx+2*Cy)[..., None]*np.arange(1, 33)[None, None, :]
        savemat(self.input / "calibration.mat", dict(acdc_maps=maps, X_m=Cx, Y_m=Cy,
                                                     units="Hz/A", registration_confirmed=True))
        np.savez(self.input / "scan.npz", image=np.ones((6, 8)))
        pipeline.write_json(self.input / "config.json", self.config)

    def synthetic_solution(self):
        _, report = pipeline.prepare(self.input, self.output)
        folder = self.output / "optimization"
        folder.mkdir()
        currents = np.tile(np.arange(1, 7)*.0001, (32, 1))
        savemat(folder / "solution_100Hz.mat", dict(currents_A=currents,
            line_start_s=report["state_start_s"], line_end_s=report["state_end_s"],
            field_limit_Hz=100, calibration_mode="registered", registration_confirmed=True,
            optimization_info=dict(converged=True, current_limits_met=True,
                                   field_limits_met=True, exitflag=1, fval=0)))
        return report

    def test_distinct_grids_and_two_sequences_export_with_zero_between_rfs(self):
        prepared_report = self.synthetic_solution()
        result = pipeline.export(self.output)
        self.assertEqual(result["current_states"], [32, 6])
        self.assertEqual(result["sequence_kind"], "gre")
        self.assertEqual(result["reference_sequence"]["image_shape"], [12, 16])
        self.assertEqual(prepared_report["image_shape"], [6, 8])
        self.assertEqual(prepared_report["B0_source_shape"], [24, 32])
        calibration = loadmat(self.output / "prepared/calibration.mat", simplify_cells=True)
        self.assertEqual(calibration["acdc_maps"].shape, (12, 16, 32))
        waveform = np.loadtxt(self.output / "waveform/waveform.csv", delimiter=",")
        self.assertEqual(waveform.shape, (13, 33))  # initial zero + 6*(current, zero)
        np.testing.assert_array_equal(waveform[::2, 1:], 0)
        np.testing.assert_allclose(waveform[1::2, 1], np.arange(1, 7)*.0001)
        np.testing.assert_allclose(waveform[1::2, 0], np.asarray(prepared_report["state_start_s"])+.0002)

    def test_changed_reference_sequence_blocks_export(self):
        self.synthetic_solution()
        path = self.input / "sequences/reference_gre.seq"
        path.write_text(path.read_text()+"\n# modified reference\n")
        with self.assertRaisesRegex(ValueError, "File changed"):
            pipeline.export(self.output)

    def test_gre_requires_its_reference_sequence(self):
        self.config["files"].pop("reference_sequence")
        pipeline.write_json(self.input / "config.json", self.config)
        with self.assertRaisesRegex(ValueError, "requires files.reference_sequence"):
            pipeline.prepare(self.input, self.output)

    @unittest.skipUnless(os.environ.get("ACDC_TEST_MATLAB"), "Set ACDC_TEST_MATLAB for the real optimizer")
    def test_real_gre_matlab_optimizer_to_waveform_csv(self):
        self.config["optimization"]["matlab_command"] = os.environ["ACDC_TEST_MATLAB"]
        pipeline.write_json(self.input / "config.json", self.config)
        report = pipeline.run(self.input, self.output)
        self.assertEqual(report["status"], "exported")
        self.assertEqual(report["current_states"], [32, 6])
        self.assertTrue(report["waveform_exported"])


if __name__ == "__main__":
    unittest.main()
