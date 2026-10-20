"""Check the waveform CSV format and rejection of invalid data."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from export_acdc_waveform import export_waveform, quantize_currents, validate_waveform_settings


class WaveformExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "output"
        self.currents = np.array([[0.2, -0.3], [-0.4, 0.5]])
        self.edges = np.array([0.002, 0.003, 0.004])
        self.settings = {
            "channel_map": [10, 2], "channel_polarity": [-1, 1],
            "max_current_A": [0.6, 0.6], "total_current_limit_A": 2.0,
        }

    def export(self):
        return export_waveform(self.currents, self.edges, self.settings, self.output)

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            self.export()
        self.assertFalse(self.output.exists())

    def test_writes_only_csv_with_mapping_and_zero_boundaries(self):
        result = self.export()
        self.assertEqual([p.name for p in self.output.iterdir()], ["waveform.csv"])
        self.assertEqual(Path(result["csv_file"]), self.output / "waveform.csv")
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (4, 17))  # time + two complete eight-output boards
        np.testing.assert_array_equal(data[:, 0], [0, *self.edges])
        np.testing.assert_array_equal(data[[0, -1], 1:], 0)
        np.testing.assert_array_equal(data[1:-1, 10], -self.currents[0])
        np.testing.assert_array_equal(data[1:-1, 2], self.currents[1])
        unused = [k + 1 for k in range(16) if k not in (1, 9)]
        np.testing.assert_array_equal(data[:, unused], 0)
        self.assertEqual(result["physical_channels"], 16)
        self.assertEqual(result["states"], 2)
        self.assertEqual(result["final_zero_s"], self.edges[-1])
        self.assertTrue(Path(result["csv_file"]).read_text().startswith("# time_s,ch0_A,ch1_A,"))

    def test_channel_64_pads_eight_boards(self):
        self.settings["channel_map"] = [1, 64]
        result = self.export()
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (4, 65))
        np.testing.assert_array_equal(data[1:-1, 1], -self.currents[0])
        np.testing.assert_array_equal(data[1:-1, 64], self.currents[1])

    def test_scalar_current_limit_is_expanded(self):
        self.settings["max_current_A"] = 0.6
        self.assertEqual(self.export()["settings"]["max_current_A"], [0.6, 0.6])

    def test_rejects_duplicate_physical_channels(self):
        self.settings["channel_map"] = [2, 2]
        self.assert_rejected("distinct integer")

    def test_rejects_invalid_channel_numbers_and_count(self):
        for channels in ([0, 2], [1, 65], [1.5, 2], [1], [1, 2, 3]):
            with self.subTest(channels=channels):
                self.settings["channel_map"] = channels
                self.assert_rejected("distinct integer")

    def test_rejects_invalid_polarity(self):
        self.settings["channel_polarity"] = [1, 0]
        self.assert_rejected("one \\+1 or -1")

    def test_rejects_missing_csv_settings(self):
        del self.settings["channel_map"]
        self.assert_rejected("Missing waveform settings")

    def test_rejects_nonfinite_or_complex_currents(self):
        self.currents[0, 0] = np.nan
        self.assert_rejected("finite")
        self.currents = np.ones((2, 2), dtype=complex) * 0.1j
        self.assert_rejected("real")

    def test_rejects_invalid_current_shape(self):
        self.currents = np.array([0.1, 0.2])
        self.assert_rejected("shape")

    def test_rejects_amplitude_without_clipping(self):
        self.currents[0, 0] = 0.60001
        self.assert_rejected("exceed max_current_A")

    def test_rejects_waveform_reader_amplitude_limit(self):
        self.settings["max_current_A"] = 3.1
        self.assert_rejected("at most 3 A")

    def test_rejects_total_current_at_limit(self):
        self.settings["total_current_limit_A"] = 0.8
        self.assert_rejected("total_current_limit_A")

    def test_rejects_nonpositive_total_current_limit(self):
        self.settings["total_current_limit_A"] = 0
        self.assert_rejected("positive scalar")

    def test_rejects_time_reset_inside_imaging(self):
        self.edges = self.edges[::-1]
        self.assert_rejected("increasing nonnegative")

    def test_state_start_at_time_origin_omits_zero_preamble(self):
        self.edges[0] = 0
        result = self.export()
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (3, 17))
        np.testing.assert_array_equal(data[:, 0], self.edges)
        np.testing.assert_array_equal(data[:-1, 10], -self.currents[0])
        np.testing.assert_array_equal(data[:-1, 2], self.currents[1])
        np.testing.assert_array_equal(data[-1, 1:], 0)

    def test_export_without_header_or_boundary_rows_uses_zero_time_column(self):
        result = export_waveform(
            self.currents, self.edges, self.settings, self.output,
            include_header=False, include_zero_preamble=False,
            include_final_zero_row=False, zero_time_column=True,
        )
        content = Path(result["csv_file"]).read_text()
        self.assertFalse(content.startswith("#"))
        self.assertEqual(len(content.splitlines()), self.currents.shape[1])
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (2, 17))
        np.testing.assert_array_equal(data[:, 0], np.zeros(2))
        np.testing.assert_array_equal(data[:, 10], -self.currents[0])
        np.testing.assert_array_equal(data[:, 2], self.currents[1])

    def test_rejects_nonfinite_or_incorrect_number_of_edges(self):
        self.edges[1] = np.inf
        self.assert_rejected("finite")
        self.edges = np.array([0.002, 0.003])
        self.assert_rejected("n_states\\+1")

    def test_optional_minimum_interval_checks_initial_and_current_states(self):
        self.settings["min_dac_interval_s"] = 10e-6
        self.edges[1] = self.edges[0] + 1e-6
        self.assert_rejected("interval below")
        self.edges = np.array([1e-6, 0.003, 0.004])
        self.assert_rejected("interval below")

    def test_rejects_invalid_optional_minimum_interval(self):
        self.settings["min_dac_interval_s"] = -1
        self.assert_rejected("positive scalar")

    def test_unknown_minimum_interval_does_not_invent_a_limit(self):
        self.settings["min_dac_interval_s"] = None
        self.edges = np.array([1e-9, 2e-9, 3e-9])
        result = self.export()
        self.assertNotIn("min_dac_interval_s", result["settings"])

    def test_csv_does_not_require_controller_programming_settings(self):
        self.assertEqual(set(validate_waveform_settings(self.settings, 2)), set(self.settings))
        self.export()
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_quantization_matches_signed_c_conversion_and_polarity(self):
        values = np.array([-3., -0.20002, -1e-6, 0, 1e-6, 0.20002, 3.])
        expected = np.array([(int(float(value) / 5.0 * 32767.0) / 32767.0) * 5.0
                             for value in values])
        np.testing.assert_array_equal(quantize_currents(values), expected)
        np.testing.assert_array_equal(quantize_currents(-values), -expected)
        self.assertLess(np.max(np.abs(expected - values)), 5 / 32767)

    def test_quantization_can_increase_generated_field(self):
        currents = np.array([1.0, 0.0001])
        response_Hz_per_A = np.array([1.0, -1.0])
        self.assertGreater(response_Hz_per_A @ quantize_currents(currents),
                           response_Hz_per_A @ currents)

    def test_refuses_overwrite_without_modifying_existing_csv(self):
        result = self.export()
        before = Path(result["csv_file"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(Path(result["csv_file"]).read_bytes(), before)
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_numbered_exports_preserve_previous_files_and_time_column_choice(self):
        original = Path(self.export()["csv_file"])
        original_bytes = original.read_bytes()
        first = Path(export_waveform(
            self.currents, self.edges, self.settings, self.output,
            numbered_filename=True, zero_time_column=True,
        )["csv_file"])
        first_bytes = first.read_bytes()
        second = Path(export_waveform(
            self.currents, self.edges, self.settings, self.output,
            numbered_filename=True,
        )["csv_file"])
        self.assertEqual(first.name, "waveform1.csv")
        self.assertEqual(second.name, "waveform2.csv")
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(first.read_bytes(), first_bytes)
        first_data = np.loadtxt(first, delimiter=",")
        second_data = np.loadtxt(second, delimiter=",")
        np.testing.assert_array_equal(first_data[:, 0], 0)
        np.testing.assert_array_equal(second_data[:, 0], [0, *self.edges])
        np.testing.assert_array_equal(first_data[:, 1:], second_data[:, 1:])

    def test_numbered_export_continues_after_largest_suffix_when_numbers_have_gaps(self):
        self.output.mkdir()
        previous = self.output / "waveform9.csv"
        previous.write_text("previous export\n")
        (self.output / "waveform2.csv").write_text("older export\n")
        (self.output / "waveform_notes.csv").write_text("notes\n")
        result = export_waveform(
            self.currents, self.edges, self.settings, self.output,
            numbered_filename=True,
        )
        self.assertEqual(Path(result["csv_file"]).name, "waveform10.csv")
        self.assertEqual(previous.read_text(), "previous export\n")


if __name__ == "__main__":
    unittest.main()
