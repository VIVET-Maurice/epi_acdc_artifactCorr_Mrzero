"""Check static CSV currents, physical output mapping and rejected exports."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from export_acdc_static import export_static_currents


class StaticCurrentExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "static"
        self.currents = np.array([0.2, -0.4])
        self.settings = {
            "channel_map": [10, 2], "channel_polarity": [-1, 1],
            "max_current_A": [0.6, 0.6], "total_current_limit_A": 2.0,
        }

    def export(self, hold_timestamp_s=1.0):
        return export_static_currents(self.currents, self.settings, self.output,
                                     hold_timestamp_s=hold_timestamp_s)

    def assert_rejected(self, pattern, hold_timestamp_s=1.0):
        with self.assertRaisesRegex(ValueError, pattern):
            self.export(hold_timestamp_s)
        self.assertFalse(self.output.exists())

    def test_two_identical_rows_keep_static_currents_and_pad_complete_boards(self):
        result = self.export()
        self.assertEqual([p.name for p in self.output.iterdir()], ["static_currents.csv"])
        data = np.loadtxt(result["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (2, 17))
        np.testing.assert_array_equal(data[:, 0], [0, 1])
        np.testing.assert_array_equal(data[0, 1:], data[1, 1:])
        np.testing.assert_array_equal(data[:, 10], [-0.2, -0.2])
        np.testing.assert_array_equal(data[:, 2], [-0.4, -0.4])
        unused_columns = [k + 1 for k in range(16) if k not in (1, 9)]
        np.testing.assert_array_equal(data[:, unused_columns], 0)
        self.assertFalse(Path(result["csv_file"]).read_text().startswith("#"))
        self.assertEqual(result["physical_channels"], 16)
        self.assertEqual(result["states"], 1)

    def test_user_example_uses_one_board_and_configurable_final_timestamp(self):
        self.currents = np.array([0.2, 0.1, -0.15])
        self.settings["channel_map"] = [1, 2, 3]
        self.settings["channel_polarity"] = [1, 1, 1]
        self.settings["max_current_A"] = 1.0
        result = self.export(5.0)
        data = np.loadtxt(result["csv_file"], delimiter=",")
        np.testing.assert_array_equal(data, [
            [0, 0.2, 0.1, -0.15, 0, 0, 0, 0, 0],
            [5, 0.2, 0.1, -0.15, 0, 0, 0, 0, 0],
        ])

    def test_channel_64_pads_eight_boards(self):
        self.settings["channel_map"] = [1, 64]
        data = np.loadtxt(self.export()["csv_file"], delimiter=",")
        self.assertEqual(data.shape, (2, 65))
        np.testing.assert_array_equal(data[:, 1], -self.currents[0])
        np.testing.assert_array_equal(data[:, 64], self.currents[1])

    def test_rejects_nonfinite_and_complex_currents(self):
        for currents in ([np.nan, 0], [np.inf, 0], [0.2j, 0]):
            with self.subTest(currents=currents):
                self.currents = currents
                self.assert_rejected("finite|real")

    def test_rejects_currents_with_multiple_states_or_empty_vector(self):
        for currents in ([[0.1, 0.2]], [], np.zeros(65)):
            with self.subTest(currents=currents):
                self.currents = currents
                self.assert_rejected("one current per channel")

    def test_rejects_amplitude_without_clipping(self):
        self.currents[0] = 0.60001
        self.assert_rejected("exceed max_current_A")
        self.settings["max_current_A"] = 3.1
        self.assert_rejected("at most 3 A")

    def test_rejects_total_current_at_limit(self):
        self.settings["total_current_limit_A"] = float(np.sum(np.abs(self.currents)))
        self.assert_rejected("total_current_limit_A")

    def test_reuses_mapping_and_polarity_validation(self):
        self.settings["channel_map"] = [2, 2]
        self.assert_rejected("distinct integer")
        self.settings["channel_map"] = [1, 2]
        self.settings["channel_polarity"] = [1, 0]
        self.assert_rejected("one \\+1 or -1")

    def test_rejects_invalid_timestamps(self):
        for hold in (0, -1, np.nan, np.inf, 1j, [1, 2]):
            with self.subTest(hold=hold):
                self.assert_rejected("positive scalar", hold)

    def test_respects_optional_minimum_dac_interval(self):
        self.settings["min_dac_interval_s"] = 1e-5
        self.assert_rejected("below min_dac_interval_s", 1e-6)

    def test_refuses_to_overwrite_existing_csv(self):
        result = self.export()
        before = Path(result["csv_file"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(Path(result["csv_file"]).read_bytes(), before)

    def test_refuses_nonempty_output_without_creating_csv(self):
        self.output.mkdir()
        unrelated_file = self.output / "existing.txt"
        unrelated_file.write_text("Keep this file.")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(unrelated_file.read_text(), "Keep this file.")
        self.assertFalse((self.output / "static_currents.csv").exists())


if __name__ == "__main__":
    unittest.main()
