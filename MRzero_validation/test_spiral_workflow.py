"""Spiral calibration, physical object preparation and actual turn exports."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Experiment"))

from continuous_reconstruction import ContinuousReconstruction
from adc_waveform_export import export_adc_waveform
from echo_shim_inputs import prepare_echo_shim_inputs
from export_acdc_waveform import export_waveform
from read_spiral_sequence import read_spiral_sequence
from shim_export_timing import sequence_trigger_timing, waveform_edges_for_origin
from spiral_baseline_calibration import fit_baseline_calibration, phase_factor
from spiral_measured_comparison import _matching_trajectory
from test_echo_shim_inputs import fixture as object_fixture


SEQUENCE = ROOT / "acquisition_032_(07)/sequences/gre_spiral_128x128_fov200mm_nav3_cimax_turn_triggers_v1.seq"


class SpiralWorkflowTests(unittest.TestCase):
    def test_sample_phase_is_exact_two_dimensional_translation_and_frequency(self):
        rng = np.random.default_rng(704)
        kx, ky = rng.normal(size=(2, 1, 50)) * 100
        time = np.linspace(.002, .03, 50)[None]
        acq = dict(sequence_kind="spiral", kx_mInv=kx, ky_mInv=ky,
                   time_s=time, rf_time_s=np.zeros_like(time))
        parameters = dict(frequency_offset_Hz=7., readout_shift_mm=.4, phase_shift_mm=-.7)
        original = np.exp(-2j * np.pi * (kx * .013 + ky * -.024 + time * 35.))
        direct = np.exp(-2j * np.pi * (kx * .0134 + ky * -.0247 + time * 42.))
        np.testing.assert_allclose(original * phase_factor(acq, parameters), direct, atol=2e-14)
        with self.assertRaises(ValueError):
            phase_factor(acq, dict(odd_even_phase_rad=.1))

    def test_guarded_calibration_supports_output_image_distinct_from_sampling(self):
        # Flattened acquisition storage is unrelated to reconstruction rows.
        # Randomly ordered samples test the absence of readout-line assumptions.
        n = 32
        rng = np.random.default_rng(705)
        kx, ky = np.meshgrid((np.arange(n)-n//2)/.2, (np.arange(n)-n//2)/.2)
        order = rng.permutation(n*n)
        time = np.linspace(.003, .045, n*n)[None]
        acq = dict(sequence_kind="spiral", image_shape=(n, n), FOV_m=[.2, .2],
            kx_mInv=kx.ravel()[order][None], ky_mInv=ky.ravel()[order][None],
            time_s=time, rf_time_s=np.zeros_like(time))
        x = (np.arange(n)-n//2)*.2/n
        X, Y = np.meshgrid(x, x)
        image = (np.exp(-((X+.02)**2/.03**2+(Y-.01)**2/.05**2))
                 + .7*np.exp(-((X-.038)**2+(Y+.035)**2)/.012**2))
        image = image * np.exp(1j*(12*X+17*Y+.8*np.sin(Y*100)))
        samples = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))).ravel()[order][None]
        op = ContinuousReconstruction(acq, "cpu", torch.float64)
        truth = dict(frequency_offset_Hz=5., readout_shift_mm=.7, phase_shift_mm=-.6)
        measured = 3.1 * abs(op.numpy_image(samples * phase_factor(acq, truth)))
        result = fit_baseline_calibration(acq, samples, measured, reconstruction=op)
        self.assertTrue(result["report"]["accepted"])
        for key, value in truth.items():
            self.assertAlmostEqual(result["parameters"][key], value, places=5)
        self.assertLess(result["report"]["candidate"]["test"]["NRMSE_percent"], 1e-5)
        self.assertNotIn("phase-encode displacement", result["report"]["fixed_quantities"])

    def test_object_area_uses_output_matrix_and_translation_resamples_maps(self):
        with tempfile.TemporaryDirectory(prefix="spiral_object_") as folder:
            inputs, fit, source, path, settings = object_fixture(folder)
            original = deepcopy(inputs)
            acq = inputs["acquisition"]
            for key in ("kx_mInv", "ky_mInv", "time_s"):
                acq[key] = acq[key].reshape(1, -1)
            # More ADC samples must not change the physical pixel area.
            for key in ("kx_mInv", "ky_mInv", "time_s"):
                acq[key] = np.repeat(acq[key], 2, axis=1)
            acq.update(sequence_kind="spiral", image_shape=(4, 4))
            before = prepare_echo_shim_inputs(inputs, fit, path, settings, subspins_z=2)
            after = prepare_echo_shim_inputs(inputs, fit, path, settings, subspins_z=2,
                baseline_calibration=dict(frequency_offset_Hz=3., readout_shift_mm=.2, phase_shift_mm=-.3))
            index = before["parent_flat_index"]
            np.testing.assert_allclose(before["expanded_experiment"]["m0_proxy"].sum(axis=1),
                                       fit["first_echo_amplitude"].ravel()[index]/4.)
            np.testing.assert_array_equal(before["expanded_experiment"]["m0_proxy"], after["expanded_experiment"]["m0_proxy"])
            np.testing.assert_allclose(after["xyz_m"]-before["xyz_m"],
                np.broadcast_to([-.0003, -.0002, 0.], before["xyz_m"].shape), atol=1e-17)
            self.assertGreater(np.max(abs(after["fieldbasis"]-before["fieldbasis"])), 0)
            np.testing.assert_array_equal(inputs["b0_raw"]["X_m"], original["b0_raw"]["X_m"])

    @unittest.skipUnless(SEQUENCE.is_file(), "Actual spiral sequence is unavailable")
    def test_actual_65_turn_trigger_export_has_66_rows_and_dead_channel_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acquisition, timing = read_spiral_sequence(SEQUENCE)
            trigger = sequence_trigger_timing(SEQUENCE, timing)
        self.assertEqual(trigger["mode"], "per_turn")
        self.assertTrue(trigger["verified_per_turn_triggers"])
        self.assertEqual(trigger["trigger_count"], 65)
        edges, _ = waveform_edges_for_origin(acquisition["line_start_s"],
            acquisition["line_end_s"], trigger_rf_s=trigger["trigger_rf_s"])
        self.assertEqual(edges.size, 66)
        self.assertEqual(edges[0], 0.)
        self.assertGreater(np.ptp(np.diff(edges)), 1e-4)
        currents = .1 * np.sin(np.arange(32*65).reshape(32,65))
        currents[8] = 0.
        settings = dict(channel_map=list(range(1,33)), channel_polarity=[-1]*32,
            disabled_channels=[9], max_current_A=3., total_current_limit_A=30.,
            min_dac_interval_s=8e-6)
        with tempfile.TemporaryDirectory(prefix="spiral_csv_") as folder:
            report = export_waveform(currents, edges, settings, folder, zero_time_column=False)
            rows = np.loadtxt(report["csv_file"], delimiter=",")
            self.assertEqual(rows.shape, (66,33))
            np.testing.assert_array_equal(rows[:,0], edges)
            np.testing.assert_array_equal(rows[:,9], 0.)
            np.testing.assert_allclose(rows[:-1,1:], -currents.T)
            np.testing.assert_array_equal(rows[-1,1:], 0.)
            adc = export_adc_waveform(SEQUENCE, timing, folder, interval_s=50e-6)
            self.assertEqual(adc["trigger"]["mode"], "per_turn")
            self.assertEqual(len(adc["segments"]), 65)
            local = np.loadtxt(adc["csv_file"], delimiter=",")
            for segment in adc["segments"]:
                self.assertEqual(local[segment["csv_row_start"]], 0.)
        changed = deepcopy(acquisition)
        changed["time_s"][0,15] += 1e-5
        with self.assertRaises(ValueError):
            _matching_trajectory(acquisition, changed)


if __name__ == "__main__":
    unittest.main()
