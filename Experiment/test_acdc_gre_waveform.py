"""Offline GRE state-window export tests, independent of the Pulseq parser."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from export_acdc_waveform import export_waveform
from run_acdc_experiment import hardware_currents, hardware_edges, template


class GreWaveformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="acdc_gre_waveform_")
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "hardware"
        self.config = template()
        self.config["hardware"].update(
            channel_map=list(range(1, 33)), channel_polarity=[1] * 32,
            max_current_A=.6, total_current_limit_A=2., clock_MHz=30.,
            min_dac_interval_s=20e-6, min_adc_interval_s=20e-6,
            settling_time_s=50e-6, trigger_lockout_ms=1.,
            trigger_period_s=.13, repetitions=3, rf_center_after_trigger_s=.001,
        )
        # Two dummy excitations then four acquired lines in non-monotonic ky
        # order. Only acquired lines have optimized current states. All times
        # below are relative to the FIRST RF, including the dummy excitations.
        rf = .02 * np.arange(6)
        self.starts = rf[2:] + .0003
        self.ends = self.starts + .005
        self.report = {
            "sequence_kind": "gre", "image_shape": [4, 6], "dummy_rf_count": 2,
            "state_start_s": self.starts.tolist(), "state_end_s": self.ends.tolist(),
            "rf_start_s": (rf - .0003).tolist(), "rf_end_s": (rf + .0003).tolist(),
            "first_adc_delay_s": [.002] * 4, "rf_center_s": .0004,
            "sequence_duration_s": .12,
        }
        ky_acquisition_order = [2, 0, 3, 1]
        values_in_cartesian_order = np.array([.1, .2, .3, .4])
        self.currents = np.zeros((32, 4))
        self.currents[0] = values_in_cartesian_order[ky_acquisition_order]
        self.currents[3] = -self.currents[0] / 2

    def export(self):
        edges = hardware_edges(self.config, self.report)
        expanded = hardware_currents(self.currents, self.report)
        manifest = export_waveform(expanded, edges, self.config["hardware"], self.output)
        data = np.loadtxt(self.output / "waveform.csv", delimiter=",")
        return manifest, data

    @staticmethod
    def at(data, times):
        """Piecewise-constant command values held until the following row."""
        rows = np.searchsorted(data[:, 0], times, side="right") - 1
        if np.any(rows < 0):
            raise AssertionError("Fixture queries cannot precede the external trigger.")
        return data[rows, 1:]

    def test_chronological_states_survive_ky_order_and_dummy_excitations(self):
        manifest, data = self.export()
        offset = self.config["hardware"]["rf_center_after_trigger_s"]
        self.assertEqual(data.shape, (9, 33))  # initial zero + four start/end pairs
        self.assertEqual(manifest["states"], 7)  # four nonzero slots, three zero gaps
        np.testing.assert_array_equal(data[1::2, 1:], self.currents.T)
        np.testing.assert_array_equal(data[::2, 1:], 0)
        np.testing.assert_allclose(self.at(data, (self.starts + self.ends) / 2 + offset),
                                   self.currents.T, rtol=0, atol=1e-15)
        # No dummy line acquired any optimized current; first update is tied to
        # the third RF, not to the first image row's nominal ky position.
        np.testing.assert_array_equal(self.at(data, np.array([.001, .015, .025, .039])), 0)
        self.assertAlmostEqual(data[1, 0], self.starts[0] + offset)

    def test_every_rf_pulse_and_interline_gap_has_zero_current(self):
        _, data = self.export()
        offset = self.config["hardware"]["rf_center_after_trigger_s"]
        for start, end in zip(self.report["rf_start_s"], self.report["rf_end_s"]):
            # Stay strictly inside each RF window; its endpoint is allowed to
            # coincide with the subsequent nonzero current command.
            inside = np.linspace(start + 1e-8, end - 1e-8, 5) + offset
            np.testing.assert_array_equal(self.at(data, inside), 0)
        gaps = (self.ends[:-1] + self.starts[1:]) / 2 + offset
        np.testing.assert_array_equal(self.at(data, gaps), 0)
        np.testing.assert_array_equal(self.at(data, [self.ends[-1] + offset + .001]), 0)

    def test_export_contains_only_the_waveform_csv(self):
        result, data = self.export()
        self.assertTrue(np.all(np.diff(data[:, 0]) > 0))
        self.assertEqual([path.name for path in self.output.iterdir()], ["waveform.csv"])
        self.assertEqual(Path(result["csv_file"]), self.output / "waveform.csv")
        self.assertEqual(result["final_zero_s"], data[-1, 0])

    def test_fpga_controller_settings_are_not_required(self):
        for key in ("clock_MHz", "min_adc_interval_s", "trigger_lockout_ms", "repetitions"):
            self.config["hardware"].pop(key)
        _, data = self.export()
        self.assertEqual(data.shape, (9, 33))
        self.assertEqual([path.name for path in self.output.iterdir()], ["waveform.csv"])

    def test_optional_measured_timing_can_be_left_unset(self):
        for key in ("settling_time_s", "trigger_period_s", "min_dac_interval_s"):
            self.config["hardware"][key] = None
        _, data = self.export()
        np.testing.assert_allclose(data[1::2, 0], self.starts + .001)
        np.testing.assert_array_equal(data[::2, 1:], 0)

    def test_line_tr_is_rejected_as_whole_image_trigger_period(self):
        self.config["hardware"]["trigger_period_s"] = .02
        with self.assertRaisesRegex(ValueError, "shorter than the sequence"):
            hardware_edges(self.config, self.report)
        self.assertFalse(self.output.exists())

    def test_return_to_zero_must_settle_before_the_next_rf(self):
        # This window itself misses the RF pulse, but its measured falling
        # transition would extend into that pulse.
        next_rf = self.report["rf_start_s"][3]
        self.report["state_end_s"][0] = next_rf - 25e-6
        with self.assertRaisesRegex(ValueError, "Return to zero has not settled"):
            hardware_edges(self.config, self.report)
        self.assertFalse(self.output.exists())

    def test_nonzero_window_cannot_overlap_an_rf_pulse(self):
        self.report["state_end_s"][0] = self.report["rf_start_s"][3] + 10e-6
        with self.assertRaisesRegex(ValueError, "overlaps an RF pulse"):
            hardware_edges(self.config, self.report)

    def test_legacy_epi_currents_remain_continuous(self):
        epi = copy.deepcopy(self.report)
        epi["sequence_kind"] = "epi"
        edges = np.array([.003, .004, .005, .006, .007])
        epi["state_start_s"], epi["state_end_s"] = edges[:-1], edges[1:]
        expected = edges + self.config["hardware"]["rf_center_after_trigger_s"]
        np.testing.assert_array_equal(hardware_edges(self.config, epi), expected)
        np.testing.assert_array_equal(hardware_currents(self.currents, epi), self.currents)


if __name__ == "__main__":
    unittest.main()
